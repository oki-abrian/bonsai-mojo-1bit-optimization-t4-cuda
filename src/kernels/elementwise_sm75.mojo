# ===----------------------------------------------------------------------=== #
# Module: src/kernels/elementwise_sm75.mojo
# Purpose: Kernel Device Native NVIDIA T4 (sm_75) untuk Eksekusi 100% Full GPU:
#          - RMSNorm (warp shuffle reduction di shared memory)
#          - Causal Conv1D 4-tap depthwise paralel di VRAM
#          - Head RMSNorm untuk Q-Norm & K-Norm
#          - Rekurensi Gated DeltaNet (64 block x 128 thread masif-paralel)
#          - Fused Gated RMSNorm GDN (SiLU gate + per-head RMSNorm)
#          - SwiGLU FFN (SiLU(gate) * up langsung di register GPU SFU)
#          - In-place Residual Vector Addition
#          - Parallel Argmax Reduction (248,320 logits -> 1 token id di VRAM)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, stack_allocation, AddressSpace
from math import sqrt, exp, log, sin, cos
from gpu.id import thread_idx, block_idx
from gpu import barrier
from gpu.primitives.warp import shuffle_down

# ----------------------------------------------------------------------------
# 1. RMSNorm GPU Kernel (1 block, 256 thread per baris M=1)
# ----------------------------------------------------------------------------
fn rmsnorm_sm75_gpu[
    T: DType,
    HAS_WEIGHT: Bool = True
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
    D: Int,
    eps: Float32
):
    """
    Menghitung RMSNorm di VRAM dengan reduksi intra-warp shuffle down.
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
    """
    var tid = thread_idx.x
    var lane = tid & 31
    var wid = tid >> 5

    # Alokasi 32 float SMEM untuk menampung parsial dari warp (aman untuk seluruh 32 lane)
    var smem = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    # 1. Akumulasi jumlah kuadrat x[i]^2 dengan grid stride loop
    var sum_sq: Float32 = 0.0
    var i = tid
    while i < D:
        var val = Float32(x_ptr[i])
        sum_sq += val * val
        i += 256

    # 2. Reduksi intra-warp (32 lane)
    sum_sq += shuffle_down(sum_sq, 16)
    sum_sq += shuffle_down(sum_sq, 8)
    sum_sq += shuffle_down(sum_sq, 4)
    sum_sq += shuffle_down(sum_sq, 2)
    sum_sq += shuffle_down(sum_sq, 1)

    if lane == 0:
        smem[wid] = sum_sq

    barrier()

    # 3. Warp 0 mereduksi parsial dari 8 warp
    if wid == 0:
        var warp_sum: Float32 = smem[lane] if lane < 8 else 0.0
        warp_sum += shuffle_down(warp_sum, 4)
        warp_sum += shuffle_down(warp_sum, 2)
        warp_sum += shuffle_down(warp_sum, 1)
        if lane == 0:
            smem[0] = 1.0 / sqrt(warp_sum / Float32(D) + eps)

    barrier()

    var inv_rms = smem[0]

    # 4. Normalisasi dan kalikan bobot gamma
    var j = tid
    while j < D:
        var val = Float32(x_ptr[j])
        var gamma: Float32 = 1.0
        @parameter
        if HAS_WEIGHT:
            gamma = weight_ptr[j]
        out_ptr[j] = Scalar[T](val * inv_rms * gamma)
        j += 256


# ----------------------------------------------------------------------------
# 2. SwiGLU GPU Kernel (68 block x 256 thread = 17,408 elemen)
# ----------------------------------------------------------------------------
fn swiglu_sm75_gpu[
    T: DType
](
    gate_up_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    intermediate_size: Int
):
    """
    Aktivasi SwiGLU FFN: out[i] = silu(gate[i]) * up[i]
    gate berada di [0 .. intermediate_size - 1],
    up berada di [intermediate_size .. 2*intermediate_size - 1].
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < intermediate_size:
        var g = Float32(gate_up_ptr[idx])
        var u = Float32(gate_up_ptr[intermediate_size + idx])
        var silu_g = g / (1.0 + exp(-g))
        out_ptr[idx] = Scalar[T](silu_g * u)


# ----------------------------------------------------------------------------
# 3. Residual Vector Addition GPU Kernel (20 block x 256 thread = 5,120 elemen)
# ----------------------------------------------------------------------------
fn vec_add_sm75_gpu[
    T: DType
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    res_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    D: Int
):
    """
    In-place residual addition di VRAM: x[i] += res[i]
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < D:
        x_ptr[idx] = Scalar[T](Float32(x_ptr[idx]) + Float32(res_ptr[idx]))


# ----------------------------------------------------------------------------
# 4. Causal Conv1D 4-Tap GPU Kernel (48 block x 256 thread = 12,288 elemen)
# ----------------------------------------------------------------------------
fn causal_conv1d_sm75_gpu[
    T: DType,
    HAS_WEIGHTS: Bool = True
](
    conv_buf: UnsafePointer[Float32, MutAnyOrigin],
    new_input: UnsafePointer[Scalar[T], MutAnyOrigin],
    weights: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    conv_dim: Int
):
    """
    Depthwise 1D Causal Convolution 4-tap dengan sliding buffer di VRAM:
    out[c] = silu(w0*buf[0,c] + w1*buf[1,c] + w2*buf[2,c] + w3*new_input[c])
    Geser riwayat: buf[0] = buf[1], buf[1] = buf[2], buf[2] = new_input.
    """
    var c = block_idx.x * 256 + thread_idx.x
    if c < conv_dim:
        var b0 = conv_buf[c]
        var b1 = conv_buf[conv_dim + c]
        var b2 = conv_buf[2 * conv_dim + c]
        var b3 = Float32(new_input[c])

        var acc: Float32 = 0.0
        @parameter
        if HAS_WEIGHTS:
            acc = b0 * weights[c * 4 + 0] + \
                  b1 * weights[c * 4 + 1] + \
                  b2 * weights[c * 4 + 2] + \
                  b3 * weights[c * 4 + 3]
        else:
            acc = (b0 + b1 + b2 + b3) * 0.25

        var silu_acc = acc / (1.0 + exp(-acc))
        out_ptr[c] = Scalar[T](silu_acc)

        # Update buffer sliding window kausal
        conv_buf[c]                  = b1
        conv_buf[conv_dim + c]       = b2
        conv_buf[2 * conv_dim + c]   = b3


# ----------------------------------------------------------------------------
# 5. Head RMSNorm GPU Kernel (1 warp per head, Q-Norm & K-Norm)
# ----------------------------------------------------------------------------
fn head_rmsnorm_sm75_gpu[
    T: DType,
    HAS_WEIGHT: Bool = False
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
    head_dim: Int,
    scale_factor: Float32,
    eps: Float32,
    stride: Int = 0
):
    """
    Per-head RMSNorm untuk Q dan K: 1 block per head, 32 thread (1 warp).
    Mendukung learned weight per-head opsional: out = (x * rms_scale) * weight.
    Mendukung custom stride untuk layout interleaved (misal Query Attention stride = 2 * head_dim).
    """
    var h = block_idx.x
    var tid = thread_idx.x
    var eff_stride = stride if stride > 0 else head_dim
    var head_offset = h * eff_stride

    var smem = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    var sum_sq: Float32 = 0.0
    var d = tid
    while d < head_dim:
        var val = Float32(x_ptr[head_offset + d])
        sum_sq += val * val
        d += 32

    # Reduksi intra-warp
    sum_sq += shuffle_down(sum_sq, 16)
    sum_sq += shuffle_down(sum_sq, 8)
    sum_sq += shuffle_down(sum_sq, 4)
    sum_sq += shuffle_down(sum_sq, 2)
    sum_sq += shuffle_down(sum_sq, 1)

    if tid == 0:
        smem[0] = (1.0 / sqrt(sum_sq / Float32(head_dim) + eps)) * scale_factor

    barrier()

    var rms_scale = smem[0]
    var od = tid
    while od < head_dim:
        var val = Float32(x_ptr[head_offset + od])
        var normed = val * rms_scale
        @parameter
        if HAS_WEIGHT:
            normed *= weight_ptr[od]
        out_ptr[head_offset + od] = Scalar[T](normed)
        od += 32


# ----------------------------------------------------------------------------
# 6. Gated DeltaNet Recurrence GPU Kernel (64 block x 128 thread masif-paralel)
# ----------------------------------------------------------------------------
fn gdn_recurrence_sm75_gpu[
    T: DType,
    HAS_PARAMS: Bool = True
](
    state_s: UnsafePointer[Scalar[T], MutAnyOrigin],
    q_normed: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_normed: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    repeat_factor: Int,
    D_v: Int,
    D_k: Int
):
    """
    Mengeksekusi rekurensi Gated DeltaNet 100% di GPU registers & shared memory:
    - 64 block (1 block per value head hv in 0..63).
    - 128 thread (1 thread per baris dv in 0..127).
    - k dan q distage ke shared memory (256 float = 1 KiB SMEM).
    - Setiap thread dv memperbarui row S[dv, :] dan menghasilkan out[dv] secara paralel.
    Memangkas 50.33 juta iterasi CPU (90 ms) menjadi ~0.15 ms di Tesla T4!
    """
    var hv = block_idx.x
    var dv = thread_idx.x
    var hk = hv // repeat_factor

    # Shared memory untuk k_head dan q_head (kapasitas D_k hingga 256)
    var smem_k = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var smem_q = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    # Pemuatan kooperatif — GUARD WAJIB: block = D_v thread, D_k bisa < D_v
    if dv < D_k:
        smem_k[dv] = Float32(k_normed[hk * D_k + dv])
        smem_q[dv] = Float32(q_normed[hk * D_k + dv])

    barrier()

    # Parameter decay dan beta untuk head hv
    var a_val = Float32(a_ptr[hv])
    var b_val = Float32(b_ptr[hv])

    var g_decay: Float32
    @parameter
    if HAS_PARAMS:
        var sp_in = a_val + dt_bias[hv]
        var sp = sp_in if sp_in > 20.0 else log(1.0 + exp(sp_in))
        g_decay = exp(Float32(-1.0) * exp(a_log[hv]) * sp)
    else:
        var sp_in = a_val + 1.0
        var sp = sp_in if sp_in > 20.0 else log(1.0 + exp(sp_in))
        g_decay = exp(Float32(-0.5) * sp)

    var beta = 1.0 / (1.0 + exp(-b_val))

    # Offset baris di memori S: [hv, dv, 0..D_k-1]
    var row_offset = hv * (D_v * D_k) + dv * D_k
    var row_ptr = state_s + row_offset

    # 1. kv_mem = sum(state_s * k) dengan decay — state FP16 di VRAM, math
    #    FP32 di register (akses float4, baris 16B-aligned)
    var kv_mem: Float32 = 0.0
    var dk4 = 0
    while dk4 + 3 < D_k:
        var s4 = row_ptr.load[width=4](dk4).cast[DType.float32]()
        var sd = s4 * g_decay
        row_ptr.store[width=4](dk4, sd.cast[T]())
        kv_mem += sd[0] * smem_k[dk4] + sd[1] * smem_k[dk4 + 1] + sd[2] * smem_k[dk4 + 2] + sd[3] * smem_k[dk4 + 3]
        dk4 += 4
    while dk4 < D_k:
        var s_decayed = state_s[row_offset + dk4].cast[DType.float32]() * g_decay
        state_s[row_offset + dk4] = Scalar[T](s_decayed)
        kv_mem += s_decayed * smem_k[dk4]
        dk4 += 1

    # 2. Delta error = (v - kv_mem) * beta
    var v_val = Float32(v_ptr[hv * D_v + dv])
    var delta = (v_val - kv_mem) * beta

    # 3. Update state S = S + k * delta dan hitung output S * q
    var read_out: Float32 = 0.0
    dk4 = 0
    while dk4 + 3 < D_k:
        var s4 = row_ptr.load[width=4](dk4).cast[DType.float32]()
        var n4 = s4 + SIMD[DType.float32, 4](smem_k[dk4], smem_k[dk4 + 1], smem_k[dk4 + 2], smem_k[dk4 + 3]) * delta
        row_ptr.store[width=4](dk4, n4.cast[T]())
        read_out += n4[0] * smem_q[dk4] + n4[1] * smem_q[dk4 + 1] + n4[2] * smem_q[dk4 + 2] + n4[3] * smem_q[dk4 + 3]
        dk4 += 4
    while dk4 < D_k:
        var s_new = state_s[row_offset + dk4].cast[DType.float32]() + smem_k[dk4] * delta
        state_s[row_offset + dk4] = Scalar[T](s_new)
        read_out += s_new * smem_q[dk4]
        dk4 += 1

    out_ptr[hv * D_v + dv] = Scalar[T](read_out)


# ----------------------------------------------------------------------------
# 7. Fused GDN Gating & Per-Head RMSNorm GPU Kernel
# ----------------------------------------------------------------------------
fn gdn_norm_gate_sm75_gpu[
    T: DType,
    HAS_NORM_W: Bool = True
](
    gdn_out: UnsafePointer[Scalar[T], MutAnyOrigin],
    z_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    norm_w: UnsafePointer[Float32, MutAnyOrigin],
    D_v: Int,
    eps: Float32
):
    """
    1 block per head hv (64 block total), 128 thread per block.
    1. Mengalikan out[dv] * silu(z[dv])
    2. Menghitung RMSNorm per head (D_v elemen)
    3. Mengalikan dengan bobot norm_w[dv] (jika HAS_NORM_W=True)
    """
    var hv = block_idx.x
    var dv = thread_idx.x
    var lane = dv & 31
    var wid = dv >> 5
    var idx = hv * D_v + dv

    # Alokasi 32 float SMEM aman untuk seluruh 32 lane pada reduksi inter-warp
    var smem = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    var z_val = Float32(z_ptr[idx])
    var silu_z = z_val / (1.0 + exp(-z_val))
    var gated_val = Float32(gdn_out[idx]) * silu_z

    # Hitung mean kuadrat pada 128 thread
    var sq = gated_val * gated_val
    sq += shuffle_down(sq, 16)
    sq += shuffle_down(sq, 8)
    sq += shuffle_down(sq, 4)
    sq += shuffle_down(sq, 2)
    sq += shuffle_down(sq, 1)

    if lane == 0:
        smem[wid] = sq

    barrier()

    if wid == 0:
        var wsq: Float32 = smem[lane] if lane < 4 else 0.0
        wsq += shuffle_down(wsq, 2)
        wsq += shuffle_down(wsq, 1)
        if lane == 0:
            smem[0] = 1.0 / sqrt(wsq / Float32(D_v) + eps)

    barrier()

    var inv_rms = smem[0]
    var gamma: Float32 = 1.0
    @parameter
    if HAS_NORM_W:
        gamma = norm_w[dv]
    gdn_out[idx] = Scalar[T](gated_val * inv_rms * gamma)


# ----------------------------------------------------------------------------
# 8. Parallel Argmax GPU Kernel (248,320 Logits -> 1 Token ID di VRAM)
# ----------------------------------------------------------------------------
fn argmax_sm75_stage1_gpu[
    T: DType
](
    logits: UnsafePointer[Scalar[T], MutAnyOrigin],
    block_max_vals: UnsafePointer[Float32, MutAnyOrigin],
    block_max_idxs: UnsafePointer[Int32, MutAnyOrigin],
    V: Int
):
    """
    Stage 1: 256 block x 256 thread. Masing-masing block mencari nilai maksimum
    dan indeks kandidatnya dengan grid stride loop.
    """
    var tid = thread_idx.x
    var lane = tid & 31
    var wid = tid >> 5
    var bx = block_idx.x
    var total_threads = 256 * 256

    var best_val = Float32(-3.0e38)
    var best_idx: Int = 0

    var i = bx * 256 + tid
    while i < V:
        var v = Float32(logits[i])
        if v > best_val:
            best_val = v
            best_idx = i
        i += total_threads

    # Reduksi intra-warp — leksikografis (val, idx): tie harus memilih indeks
    # TERKECIL agar paritas dengan CPU strict-> (first-index-wins), karena
    # urutan lane shuffle bukan urutan indeks global.
    var off = 16
    while off > 0:
        var other_val = shuffle_down(best_val, UInt32(off))
        var other_idx = shuffle_down(Float32(best_idx), UInt32(off))
        if (other_val > best_val) or (
            (other_val == best_val) and (Int(other_idx) < best_idx)
        ):
            best_val = other_val
            best_idx = Int(other_idx)
        off = off // 2

    var smem_v = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var smem_i = stack_allocation[
        32, Int32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    if lane == 0:
        smem_v[wid] = best_val
        smem_i[wid] = Int32(best_idx)

    barrier()

    if wid == 0:
        var v_w = smem_v[lane] if lane < 8 else Float32(-3.0e38)
        var i_w = smem_i[lane] if lane < 8 else Int32(0)
        var w_off = 4
        while w_off > 0:
            var ov = shuffle_down(v_w, UInt32(w_off))
            var oi = shuffle_down(Float32(i_w), UInt32(w_off))
            if (ov > v_w) or ((ov == v_w) and (Int(oi) < Int(i_w))):
                v_w = ov
                i_w = Int32(oi)
            w_off = w_off // 2
        if lane == 0:
            block_max_vals[bx] = v_w
            block_max_idxs[bx] = i_w


fn argmax_sm75_stage2_gpu(
    block_max_vals: UnsafePointer[Float32, MutAnyOrigin],
    block_max_idxs: UnsafePointer[Int32, MutAnyOrigin],
    out_token: UnsafePointer[Int32, MutAnyOrigin],
    num_blocks: Int
):
    """
    Stage 2: 1 block x 256 thread. Mereduksi 256 kandidat terbaik dari stage 1
    dan menulis 1 integer token ID akhir ke out_token.
    """
    var tid = thread_idx.x
    var lane = tid & 31
    var wid = tid >> 5

    var v = block_max_vals[tid] if tid < num_blocks else Float32(-3.0e38)
    var idx = block_max_idxs[tid] if tid < num_blocks else Int32(0)

    # Reduksi intra-warp (leksikografis: tie pilih indeks terkecil, paritas CPU)
    var off = 16
    while off > 0:
        var ov = shuffle_down(v, UInt32(off))
        var oi = shuffle_down(Float32(idx), UInt32(off))
        if (ov > v) or ((ov == v) and (Int(oi) < Int(idx))):
            v = ov
            idx = Int32(oi)
        off = off // 2

    var smem_v = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var smem_i = stack_allocation[
        32, Int32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    if lane == 0:
        smem_v[wid] = v
        smem_i[wid] = idx

    barrier()

    if wid == 0:
        var v_final = smem_v[lane] if lane < 8 else Float32(-3.0e38)
        var i_final = smem_i[lane] if lane < 8 else Int32(0)
        var w_off = 4
        while w_off > 0:
            var ov = shuffle_down(v_final, UInt32(w_off))
            var oi = shuffle_down(Float32(i_final), UInt32(w_off))
            if (ov > v_final) or ((ov == v_final) and (Int(oi) < Int(i_final))):
                v_final = ov
                i_final = Int32(oi)
            w_off = w_off // 2
        if lane == 0:
            out_token[0] = i_final


# ----------------------------------------------------------------------------
# 9. Partial RoPE GPU Kernel (1 block per head, 32 thread)
# ----------------------------------------------------------------------------
fn partial_rope_sm75_gpu[
    T: DType
](
    vec: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    rotary_dim: Int,
    pos: Int,
    theta_base: Float32,
    stride: Int = 0
):
    """
    Rotary Positional Embedding (RoPE) parsial (rotary_dim / head_dim) di VRAM:
    Hanya dimensi 0..rotary_dim-1 yang dirotasi, dimensi sisanya utuh linear.
    Paritas MLX rope (traditional=False / default): pasangan HALF-SPLIT —
    elemen i (i < rotary_dim/2) berpasangan dengan elemen i + rotary_dim/2,
    freq_i = theta^(-2i/rotary_dim). BUKAN pasangan bersebelahan (2i, 2i+1).
    1 block per head, 32 thread (1 warp).
    Mendukung custom stride untuk layout interleaved (misal Query Attention stride = 2 * head_dim).
    """
    var h = block_idx.x
    var tid = thread_idx.x
    var half_rotary = rotary_dim // 2
    var eff_stride = stride if stride > 0 else head_dim
    var h_offset = h * eff_stride

    if tid < half_rotary:
        var exponent = -Float32(2 * tid) / Float32(rotary_dim)
        var freq = exp(log(theta_base) * exponent)
        var m_theta = Float32(pos) * freq
        var cos_val = cos(m_theta)
        var sin_val = sin(m_theta)

        # Paritas MLX rope.cu traditional=False: index_2 = index_1 + half_rotary
        var x0 = Float32(vec[h_offset + tid])
        var x1 = Float32(vec[h_offset + tid + half_rotary])

        vec[h_offset + tid]             = Scalar[T](x0 * cos_val - x1 * sin_val)
        vec[h_offset + tid + half_rotary] = Scalar[T](x0 * sin_val + x1 * cos_val)


# ----------------------------------------------------------------------------
# 10. KV Cache Append GPU Kernel (Ring Buffer VRAM)
# ----------------------------------------------------------------------------
fn kv_cache_append_sm75_gpu[
    T: DType
](
    k_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_token: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_token: UnsafePointer[Scalar[T], MutAnyOrigin],
    pos: Int,
    kv_dim: Int
):
    """
    Menyimpan token Key dan Value ke buffer ring KV-Cache di VRAM:
    offset = pos * kv_dim.
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < kv_dim:
        var offset = pos * kv_dim + idx
        k_cache[offset] = k_token[idx]
        v_cache[offset] = v_token[idx]


# ----------------------------------------------------------------------------
# 11. Fused GQA Causal Scaled Dot-Product Attention & Sigmoid Gate GPU Kernel
# ----------------------------------------------------------------------------
fn gqa_attention_sm75_gpu[
    T: DType
](
    q_gate: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    attn_scores: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    max_seq_len: Int,
    H_q: Int,
    H_kv: Int,
    head_dim: Int,
    scale: Float32
):
    """
    Fused GQA Attention pada GPU NVIDIA T4:
    - 1 block per Query Head (H_q block).
    - 128 thread per block (1 thread per head_dim = 128).
    - Grouped Query Attention: group_size = H_q // H_kv (Bonsai-27B: 40//8 = 5).
    - Memuat query head ke Shared Memory.
    - Menghitung scaled dot-product attention scores S[t] untuk t in [0..seq_len-1].
    - Numerically stable Softmax dengan intra-warp & inter-warp reduction.
    - Akumulasi Value vector per-dimensi langsung oleh thread d = tid.
    - Fused Sigmoid Gating: out[hq * head_dim + d] = context[d] * sigmoid(gate[hq * head_dim + d]).
    """
    var hq = block_idx.x
    var tid = thread_idx.x
    var lane = tid & 31
    var wid = tid >> 5
    var group_size = H_q // H_kv
    var hkv = hq // group_size
    var kv_dim = H_kv * head_dim

    # Buffer Shared Memory Terpadu (192 Float32 = 768 B) untuk eliminasi aliasing NVPTX
    # HINDARI pointer arithmetic di shared memory (smem + offset) karena NVPTX
    # mungkin mengonversi AddressSpace.SHARED ke GENERIC → illegal memory access.
    # Gunakan direct indexing smem[offset + idx] saja.
    # Buffer Shared Memory Terpadu: head_dim (256) untuk Q + 32 reduksi warp
    # + 32 broadcast = 320 float. Model riil Bonsai: head_dim=256, H_q=24,
    # H_kv=4 (terverifikasi [GQA DIAG] Run #50).
    alias GQA_SMEM_ELEMS: Int = 320   # 256 (q) + 32 (reduce) + 32 (bcast)
    alias GQA_REDUCE_OFF: Int = 256
    alias GQA_BCAST_OFF: Int = 288
    var smem = stack_allocation[
        GQA_SMEM_ELEMS, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    # 1. Muat Query head ke Shared Memory dari layout interleaved (stride 2 * head_dim)
    smem[tid] = Float32(q_gate[hq * (2 * head_dim) + tid])
    barrier()

    # 2. Hitung Scaled Dot-Product S[t] untuk seluruh token riwayat t < seq_len
    var t = tid
    while t < seq_len:
        var k_offset = t * kv_dim + hkv * head_dim
        var dot: Float32 = 0.0
        for d in range(head_dim):
            dot += smem[d] * Float32(k_cache[k_offset + d])
        attn_scores[hq * max_seq_len + t] = dot * scale
        t += head_dim
    barrier()

    # 3. Softmax Tahap 1: Temukan max(S[t])
    var local_max: Float32 = Float32(-3.0e38)
    t = tid
    while t < seq_len:
        var s = attn_scores[hq * max_seq_len + t]
        if s > local_max:
            local_max = s
        t += head_dim

    local_max = max(local_max, shuffle_down(local_max, 16))
    local_max = max(local_max, shuffle_down(local_max, 8))
    local_max = max(local_max, shuffle_down(local_max, 4))
    local_max = max(local_max, shuffle_down(local_max, 2))
    local_max = max(local_max, shuffle_down(local_max, 1))

    if lane == 0:
        smem[GQA_REDUCE_OFF + wid] = local_max
    barrier()

    if wid == 0:
        # Block 256 thread = 8 warp; reduksi 8 nilai via shuffle 4->2->1.
        var m = smem[GQA_REDUCE_OFF + lane] if lane < 8 else Float32(-3.0e38)
        m = max(m, shuffle_down(m, 4))
        m = max(m, shuffle_down(m, 2))
        m = max(m, shuffle_down(m, 1))
        if lane == 0:
            smem[GQA_BCAST_OFF] = m
    barrier()
    var global_max = smem[GQA_BCAST_OFF]

    # 4. Softmax Tahap 2: Eksponensiasi dan akumulasi sum(exp)
    var local_sum: Float32 = 0.0
    t = tid
    while t < seq_len:
        var e = exp(attn_scores[hq * max_seq_len + t] - global_max)
        attn_scores[hq * max_seq_len + t] = e
        local_sum += e
        t += head_dim

    local_sum += shuffle_down(local_sum, 16)
    local_sum += shuffle_down(local_sum, 8)
    local_sum += shuffle_down(local_sum, 4)
    local_sum += shuffle_down(local_sum, 2)
    local_sum += shuffle_down(local_sum, 1)

    if lane == 0:
        smem[GQA_REDUCE_OFF + wid] = local_sum
    barrier()

    if wid == 0:
        # Reduksi 8 warp untuk denominator softmax (shuffle 4->2->1).
        var s = smem[GQA_REDUCE_OFF + lane] if lane < 8 else Float32(0.0)
        s += shuffle_down(s, 4)
        s += shuffle_down(s, 2)
        s += shuffle_down(s, 1)
        if lane == 0:
            smem[GQA_BCAST_OFF] = 1.0 / (s + 1e-9)
    barrier()
    var inv_sum = smem[GQA_BCAST_OFF]

    t = tid
    while t < seq_len:
        attn_scores[hq * max_seq_len + t] *= inv_sum
        t += head_dim
    barrier()

    # 5. Akumulasi Context Vector untuk dimensi tid: context[d] = sum_t (P[t] * V[t, d])
    var acc: Float32 = 0.0
    for step in range(seq_len):
        var prob = attn_scores[hq * max_seq_len + step]
        var v_val = Float32(v_cache[step * kv_dim + hkv * head_dim + tid])
        acc += prob * v_val

    # 6. Fused Sigmoid Gate: out = context * sigmoid(gate)
    # Pada layout interleaved, gate head hq berada di offset hq * 2 * head_dim + head_dim
    var gate_val = Float32(q_gate[hq * (2 * head_dim) + head_dim + tid])
    var g = 1.0 / (1.0 + exp(-gate_val))
    out_ptr[hq * head_dim + tid] = Scalar[T](acc * g)


# ----------------------------------------------------------------------------
# 12. 1-Bit Affine Embedding Lookup GPU Kernel
# ----------------------------------------------------------------------------
fn embed_lookup_1bit_sm75_gpu[
    T: DType
](
    packed: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    token_id: Int,
    D: Int
):
    """
    Dequant SATU baris embedding 1-bit affine (w = bit * scale + bias)
    langsung di VRAM ke out_ptr (hidden_states_dev).
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < D:
        var base = token_id * (D // 8)
        var srow = token_id * (D // 128)
        var byte_val = packed[base + (idx >> 3)]
        var bit = Float32((Int(byte_val) >> (idx & 7)) & 1)
        var s = scales[srow + (idx >> 7)]
        var b = biases[srow + (idx >> 7)]
        out_ptr[idx] = Scalar[T](bit * s + b)
