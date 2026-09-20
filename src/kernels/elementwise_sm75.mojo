# ===----------------------------------------------------------------------=== #
# Module: src/kernels/elementwise_sm75.mojo
# Purpose: Kernel Device Native NVIDIA T4 (sm_75) untuk Eksekusi 100% Full GPU:
#          - RMSNorm (warp shuffle reduction di shared memory)
#          - Causal Conv1D 4-tap depthwise paralel di VRAM
#          - Head RMSNorm untuk Q-Norm & K-Norm
#          - Rekurensi Gated DeltaNet (H_v block x D_v thread; Bonsai-27B: 48 x 128)
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
    # block_idx.y = indeks baris (batched prefill M-token; decode selalu 0)
    var row = block_idx.y
    var x_row = x_ptr.offset(row * D)
    var o_row = out_ptr.offset(row * D)

    var sum_sq: Float32 = 0.0
    var i = tid
    while i < D:
        var val = Float32(x_row[i])
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
        var val = Float32(x_row[j])
        var gamma: Float32 = 1.0
        @parameter
        if HAS_WEIGHT:
            gamma = weight_ptr[j]
        o_row[j] = Scalar[T](val * inv_rms * gamma)
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
    block_idx.y = indeks baris (batched prefill M-token; decode selalu 0).
    """
    var row = block_idx.y
    var gu_row = gate_up_ptr.offset(row * 2 * intermediate_size)
    var o_row = out_ptr.offset(row * intermediate_size)
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < intermediate_size:
        var g = Float32(gu_row[idx])
        var u = Float32(gu_row[intermediate_size + idx])
        var silu_g = g / (1.0 + exp(-g))
        o_row[idx] = Scalar[T](silu_g * u)


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
    block_idx.y = indeks baris (batched prefill M-token; decode selalu 0).
    """
    var row = block_idx.y
    var x_row = x_ptr.offset(row * D)
    var res_row = res_ptr.offset(row * D)
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < D:
        x_row[idx] = Scalar[T](Float32(x_row[idx]) + Float32(res_row[idx]))


fn add_rmsnorm_sm75_gpu[
    T: DType,
    HAS_WEIGHT: Bool = True
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    res_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_norm_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    weight_ptr: UnsafePointer[Float32, MutAnyOrigin],
    D: Int,
    eps: Float32
):
    """
    FUSI vec_add + rmsnorm (1 launch per situs residual; hemat launch +
    separuh trafik hidden). BIT-EXACT dgn jalur terpisah: jumlah di-round
    fp16 SEBELUM sum-of-squares (vec_add menulis fp16, rmsnorm membaca fp16),
    reduksi & iterasi identik dgn rmsnorm_sm75_gpu.
    x = hidden (in-place), res = keluaran sublayer, out_norm = x_norm.
    """
    var tid = thread_idx.x
    var lane = tid & 31
    var wid = tid >> 5
    var smem = stack_allocation[
        32, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    var row = block_idx.y
    var x_row = x_ptr.offset(row * D)
    var res_row = res_ptr.offset(row * D)
    var o_row = out_norm_ptr.offset(row * D)

    # Loop-loop di bawah dibuka 4x supaya BANYAK pemuatan melayang bersamaan.
    # Alasannya diukur, bukan dugaan: saat decode grid=(1,1,1) block=(256,1,1),
    # jadi kernel ini berjalan di SATU SM dengan cuma 8 warp. Dengan satu
    # pemuatan melayang per warp, latensi tiap pemuatan terekspos penuh dan
    # satu peluncuran memakan 21,3 us (2,723 ms/token, 128 peluncuran/token)
    # untuk cuma 20 KB trafik — itu 0,9 GB/s. URUTAN operasi sengaja TIDAK
    # diubah (akumulasi tetap 0,1,2,3 per kelompok) supaya bit-exact.
    #
    # 1. Residual in-place — persis dgn vec_add_sm75_gpu (f16(f32+f32))
    var i = tid
    while i + 768 < D:
        var a0 = Float32(x_row[i])
        var a1 = Float32(x_row[i + 256])
        var a2 = Float32(x_row[i + 512])
        var a3 = Float32(x_row[i + 768])
        var b0 = Float32(res_row[i])
        var b1 = Float32(res_row[i + 256])
        var b2 = Float32(res_row[i + 512])
        var b3 = Float32(res_row[i + 768])
        x_row[i] = Scalar[T](a0 + b0)
        x_row[i + 256] = Scalar[T](a1 + b1)
        x_row[i + 512] = Scalar[T](a2 + b2)
        x_row[i + 768] = Scalar[T](a3 + b3)
        i += 1024
    while i < D:
        x_row[i] = Scalar[T](Float32(x_row[i]) + Float32(res_row[i]))
        i += 256

    barrier()

    # 2. Sum-of-squares — persis dgn rmsnorm_sm75_gpu pada hidden baru
    var sum_sq: Float32 = 0.0
    var j = tid
    while j + 768 < D:
        var v0 = Float32(x_row[j])
        var v1 = Float32(x_row[j + 256])
        var v2 = Float32(x_row[j + 512])
        var v3 = Float32(x_row[j + 768])
        sum_sq += v0 * v0
        sum_sq += v1 * v1
        sum_sq += v2 * v2
        sum_sq += v3 * v3
        j += 1024
    while j < D:
        var val = Float32(x_row[j])
        sum_sq += val * val
        j += 256

    sum_sq += shuffle_down(sum_sq, 16)
    sum_sq += shuffle_down(sum_sq, 8)
    sum_sq += shuffle_down(sum_sq, 4)
    sum_sq += shuffle_down(sum_sq, 2)
    sum_sq += shuffle_down(sum_sq, 1)

    if lane == 0:
        smem[wid] = sum_sq

    barrier()

    if wid == 0:
        var warp_sum: Float32 = smem[lane] if lane < 8 else 0.0
        warp_sum += shuffle_down(warp_sum, 4)
        warp_sum += shuffle_down(warp_sum, 2)
        warp_sum += shuffle_down(warp_sum, 1)
        if lane == 0:
            smem[0] = 1.0 / sqrt(warp_sum / Float32(D) + eps)

    barrier()

    var inv_rms = smem[0]

    var k = tid
    while k + 768 < D:
        var v0 = Float32(x_row[k])
        var v1 = Float32(x_row[k + 256])
        var v2 = Float32(x_row[k + 512])
        var v3 = Float32(x_row[k + 768])
        var g0: Float32 = 1.0
        var g1: Float32 = 1.0
        var g2: Float32 = 1.0
        var g3: Float32 = 1.0
        @parameter
        if HAS_WEIGHT:
            g0 = weight_ptr[k]
            g1 = weight_ptr[k + 256]
            g2 = weight_ptr[k + 512]
            g3 = weight_ptr[k + 768]
        o_row[k] = Scalar[T](v0 * inv_rms * g0)
        o_row[k + 256] = Scalar[T](v1 * inv_rms * g1)
        o_row[k + 512] = Scalar[T](v2 * inv_rms * g2)
        o_row[k + 768] = Scalar[T](v3 * inv_rms * g3)
        k += 1024
    while k < D:
        var val = Float32(x_row[k])
        var gamma: Float32 = 1.0
        @parameter
        if HAS_WEIGHT:
            gamma = weight_ptr[k]
        o_row[k] = Scalar[T](val * inv_rms * gamma)
        k += 256


fn copy_vec_sm75_gpu[
    T: DType
](
    dst_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    src_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    n: Int
):
    """
    Salin vektor device-to-device: dst[i] = src[i] (untuk transplantasi
    baris terakhir prefill ke buffer decode, tanpa memicu PCIe).
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < n:
        dst_ptr[idx] = src_ptr[idx]


fn khq_ring_dual_sm75_gpu[
    T: DType
](
    dst_a: UnsafePointer[Scalar[T], MutAnyOrigin],
    dst_b: UnsafePointer[Scalar[T], MutAnyOrigin],
    src_a: UnsafePointer[Scalar[T], MutAnyOrigin],
    src_b: UnsafePointer[Scalar[T], MutAnyOrigin],
    n: Int
):
    """
    Tulis DUA vektor dalam satu launch (K roped -> ring_k, V -> ring_v).
    KHQ meluncurkan ~4 salinan kecil per layer per token; yang membebani
    bukan byte-nya (6 KB) melainkan jumlah launch kernel kecil yang
    latency-bound (grid cdiv(1024,256) = 4 block). Menggabungkan dua launch
    menjadi satu memotong separuh overhead itu tanpa mengubah data.
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx < n:
        dst_a[idx] = src_a[idx]
        dst_b[idx] = src_b[idx]


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
    stride: Int = 0,
    row_stride: Int = 0,
    out_row_stride: Int = 0
):
    """
    Per-head RMSNorm untuk Q dan K: 1 block per head, 32 thread (1 warp).
    Mendukung learned weight per-head opsional: out = (x * rms_scale) * weight.
    Mendukung custom stride untuk layout interleaved (misal Query Attention stride = 2 * head_dim).
    """
    var h = block_idx.x
    var tid = thread_idx.x
    var eff_stride = stride if stride > 0 else head_dim
    # block_idx.y = baris (batched prefill M-token; decode selalu 0).
    # row_stride = lebar baris INPUT (mis. conv_dim=10240), out_row_stride =
    # lebar baris OUTPUT (mis. qk_dim=2048) — beda bila layout x != y.
    var o_row_stride = out_row_stride if out_row_stride > 0 else row_stride
    var head_offset = block_idx.y * row_stride + h * eff_stride
    var head_offset_o = block_idx.y * o_row_stride + h * eff_stride

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
        out_ptr[head_offset_o + od] = Scalar[T](normed)
        od += 32


# ----------------------------------------------------------------------------
# 6. Gated DeltaNet Recurrence GPU Kernel (H_v block x D_v thread; 48 x 128)
# ----------------------------------------------------------------------------
fn gdn_recurrence_sm75_gpu[
    T: DType,
    HAS_PARAMS: Bool = True
](
    state_s: UnsafePointer[Float32, MutAnyOrigin],
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
    - H_v block (1 block per value head hv). Bonsai-27B: H_v = 48.
    - D_v thread (1 thread per baris dv in 0..D_v-1). Bonsai-27B: D_v = 128.
    - k dan q distage ke shared memory (2 x 256 float = 2 KiB SMEM; D_k <= 256).
    - Setiap thread dv memperbarui row S[dv, :] dan menghasilkan out[dv] secara paralel.
    Memangkas puluhan juta iterasi CPU menjadi ~0,15 ms di Tesla T4!
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

    # 1. kv_mem = sum(state_s * decay * k) — state S disimpan FP32 di
    #    VRAM (WAJIB: rounding per-langkah terakumulasi karena decay ~ 1.0;
    #    lihat config `mamba_ssm_dtype: float32` & llama.cpp GGML_TYPE_F32).
    #    Akses float4, baris 16B-aligned.
    #
    #    PENTING (diukur, bukan dugaan): fase ini DULU menulis S*decay kembali
    #    ke VRAM lalu fase 3 membacanya lagi dan menimpanya. Tulisan itu
    #    dibuang di sini — fase 3 menghitung S*decay sendiri dari S asli, dan
    #    karena `S*decay` dibulatkan ke FP32 dengan cara yang sama persis,
    #    nilainya BIT-EXACT identik. Trafik turun 12 MB -> 9 MB per layer
    #    (baca 3, baca 3, tulis 3) tanpa mengubah satu bit pun hasil akhir.
    var kv_mem: Float32 = 0.0
    var dk4 = 0
    # Loop dibuka 4x (16 float per iterasi) supaya 4 pemuatan float4 MELAYANG
    # bersamaan. Alasannya diukur, bukan dugaan: kernel ini diluncurkan
    # grid=(48,1,1) block=(128,1,1) = cuma 6144 thread untuk GPU 40 SM, dan
    # dengan satu pemuatan melayang per thread ia hanya mencapai 50,9 GB/s
    # (12 MB x 48 layer / 11,32 ms) dari puncak mesin 264 GB/s yang diukur
    # sendiri. URUTAN operasi sengaja TIDAK diubah (akumulasi tetap 0..3,
    # 4..7, 8..11, 12..15) supaya hasilnya tetap bit-exact.
    while dk4 + 15 < D_k:
        var s0 = row_ptr.load[width=4](dk4)
        var s1 = row_ptr.load[width=4](dk4 + 4)
        var s2 = row_ptr.load[width=4](dk4 + 8)
        var s3 = row_ptr.load[width=4](dk4 + 12)
        var d0 = s0 * g_decay
        var d1 = s1 * g_decay
        var d2 = s2 * g_decay
        var d3 = s3 * g_decay
        kv_mem += d0[0] * smem_k[dk4] + d0[1] * smem_k[dk4 + 1] + d0[2] * smem_k[dk4 + 2] + d0[3] * smem_k[dk4 + 3]
        kv_mem += d1[0] * smem_k[dk4 + 4] + d1[1] * smem_k[dk4 + 5] + d1[2] * smem_k[dk4 + 6] + d1[3] * smem_k[dk4 + 7]
        kv_mem += d2[0] * smem_k[dk4 + 8] + d2[1] * smem_k[dk4 + 9] + d2[2] * smem_k[dk4 + 10] + d2[3] * smem_k[dk4 + 11]
        kv_mem += d3[0] * smem_k[dk4 + 12] + d3[1] * smem_k[dk4 + 13] + d3[2] * smem_k[dk4 + 14] + d3[3] * smem_k[dk4 + 15]
        dk4 += 16
    while dk4 + 3 < D_k:
        var s4 = row_ptr.load[width=4](dk4)
        var sd = s4 * g_decay
        kv_mem += sd[0] * smem_k[dk4] + sd[1] * smem_k[dk4 + 1] + sd[2] * smem_k[dk4 + 2] + sd[3] * smem_k[dk4 + 3]
        dk4 += 4
    while dk4 < D_k:
        var s_decayed = state_s[row_offset + dk4] * g_decay
        kv_mem += s_decayed * smem_k[dk4]
        dk4 += 1

    # 2. Delta error = (v - kv_mem) * beta
    var v_val = Float32(v_ptr[hv * D_v + dv])
    var delta = (v_val - kv_mem) * beta

    # 3. Tulis state S = S * decay + k * delta dan hitung output S * q.
    #    Karena fase 1 tidak lagi menyimpan S*decay, decay DIULANG di sini
    #    dari S asli. `S * g_decay` dibulatkan ke FP32 persis seperti yang
    #    dulu dilakukan fase 1, jadi nilai yang ditulis BIT-EXACT sama dengan
    #    jalur lama; hanya tulisan perantara di VRAM yang hilang.
    var read_out: Float32 = 0.0
    dk4 = 0
    # Sama seperti fase 1: 4 pemuatan float4 melayang bersamaan, urutan
    # operasi tidak diubah (bit-exact).
    while dk4 + 15 < D_k:
        var s0 = row_ptr.load[width=4](dk4)
        var s1 = row_ptr.load[width=4](dk4 + 4)
        var s2 = row_ptr.load[width=4](dk4 + 8)
        var s3 = row_ptr.load[width=4](dk4 + 12)
        var n0 = s0 * g_decay + SIMD[DType.float32, 4](smem_k[dk4], smem_k[dk4 + 1], smem_k[dk4 + 2], smem_k[dk4 + 3]) * delta
        var n1 = s1 * g_decay + SIMD[DType.float32, 4](smem_k[dk4 + 4], smem_k[dk4 + 5], smem_k[dk4 + 6], smem_k[dk4 + 7]) * delta
        var n2 = s2 * g_decay + SIMD[DType.float32, 4](smem_k[dk4 + 8], smem_k[dk4 + 9], smem_k[dk4 + 10], smem_k[dk4 + 11]) * delta
        var n3 = s3 * g_decay + SIMD[DType.float32, 4](smem_k[dk4 + 12], smem_k[dk4 + 13], smem_k[dk4 + 14], smem_k[dk4 + 15]) * delta
        row_ptr.store[width=4](dk4, n0)
        row_ptr.store[width=4](dk4 + 4, n1)
        row_ptr.store[width=4](dk4 + 8, n2)
        row_ptr.store[width=4](dk4 + 12, n3)
        read_out += n0[0] * smem_q[dk4] + n0[1] * smem_q[dk4 + 1] + n0[2] * smem_q[dk4 + 2] + n0[3] * smem_q[dk4 + 3]
        read_out += n1[0] * smem_q[dk4 + 4] + n1[1] * smem_q[dk4 + 5] + n1[2] * smem_q[dk4 + 6] + n1[3] * smem_q[dk4 + 7]
        read_out += n2[0] * smem_q[dk4 + 8] + n2[1] * smem_q[dk4 + 9] + n2[2] * smem_q[dk4 + 10] + n2[3] * smem_q[dk4 + 11]
        read_out += n3[0] * smem_q[dk4 + 12] + n3[1] * smem_q[dk4 + 13] + n3[2] * smem_q[dk4 + 14] + n3[3] * smem_q[dk4 + 15]
        dk4 += 16
    while dk4 + 3 < D_k:
        var s4 = row_ptr.load[width=4](dk4)
        var n4 = s4 * g_decay + SIMD[DType.float32, 4](smem_k[dk4], smem_k[dk4 + 1], smem_k[dk4 + 2], smem_k[dk4 + 3]) * delta
        row_ptr.store[width=4](dk4, n4)
        read_out += n4[0] * smem_q[dk4] + n4[1] * smem_q[dk4 + 1] + n4[2] * smem_q[dk4 + 2] + n4[3] * smem_q[dk4 + 3]
        dk4 += 4
    while dk4 < D_k:
        var s_new = state_s[row_offset + dk4] * g_decay + smem_k[dk4] * delta
        state_s[row_offset + dk4] = s_new
        read_out += s_new * smem_q[dk4]
        dk4 += 1

    out_ptr[hv * D_v + dv] = Scalar[T](read_out)


# ----------------------------------------------------------------------------
# 6b. Rekurensi GDN VARIAN LEBAR: 8 thread per baris, 32 baris per blok.
#     Desain ini DIPILIH LEWAT PENGUKURAN (probe CUDA mandiri, bagian 10
#     harness uji) — BUKAN dugaan. Pada rejim nyata (48 peluncuran berurutan,
#     satu layer tiap peluncuran, persis seperti model):
#        pola lama        48 blok x 128 thread   4,037 ms/token  (1,00x)
#        32 float/thread 192 blok x 128 thread   3,115 ms/token  (1,30x)
#        16 float/thread 192 blok x 256 thread   2,671 ms/token  (1,51x) <- ini
#     Varian "1 thread = 1 baris + S di SMEM" (yang memangkas trafik jadi
#     6 MB tapi menurunkan okupansi jadi 4 blok/SM) cuma 1,04x — nyaris nol.
#     Jadi yang menang adalah MEMPERBANYAK THREAD, bukan sekadar memangkas
#     trafik. Kernel ini mendapat keduanya: thread 8x lipat DAN trafik 6 MB.
#
#     Memakai identitas aljabar (terverifikasi di probe, selisih 4,47e-08 thd
#     |out| ~1e-1, sementara gap TOP-2 = 16,25):
#        out = decay * B + delta * kq,   B = sum(S*q),  kq = sum(k*q)
#     sehingga S cukup DIBACA SEKALI dan DITULIS SEKALI.
#     TIDAK bit-exact (urutan penjumlahan berubah); diaktifkan terpisah lewat
#     BONSAI_GDN_WIDE=1, default 0 = jalur lama yang sudah terbukti.
#
#     *** HASIL UJI DI DALAM MODEL (run v36, jangan diterka dari probe!) ***
#     lama = 50,026 ms/token | lebar = 50,993 ms/token -> 0,981x, alias TIDAK
#     lebih cepat (24 token tetap identik, jadi aljabarnya benar). Jadi probe
#     yang mengukur 1,51x TIDAK mewakili keadaan nyata: di dalam model,
#     peluncuran rekurensi diselingi GEMV dan keadaannya berbeda. Kesimpulan:
#     jangan lagi menyetel kernel ini berdasar probe terisolasi.
# ----------------------------------------------------------------------------
fn gdn_recurrence_sm75_gpu_wide[
    T: DType,
    HAS_PARAMS: Bool = True
](
    state_s: UnsafePointer[Float32, MutAnyOrigin],
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
    """grid = (H_v, D_v/32, 1), block = (256, 1, 1).
    32 baris per blok, 8 thread per baris, 16 float per thread."""
    var hv = block_idx.x
    var rg = block_idx.y
    var t = thread_idx.x
    var hk = hv // repeat_factor

    var smem_k = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var smem_q = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    # 32 baris x 128 float = 4096 float = 16 KiB
    var smem_S = stack_allocation[
        4096, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var redA = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()
    var redB = stack_allocation[
        256, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    var i0 = t
    while i0 < D_k:
        smem_k[i0] = Float32(k_normed[hk * D_k + i0])
        smem_q[i0] = Float32(q_normed[hk * D_k + i0])
        i0 += 256

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

    # Tahap S ke SMEM: 1024 float4 per blok; tiap 32 thread menutup satu baris
    # penuh secara berkoalesensi (itulah sebabnya pengindeksannya "transpos").
    var Sbase = state_s + (hv * D_v + rg * 32) * D_k
    var c = t
    while c < 1024:
        var r0 = (4 * c) // D_k
        var off = (4 * c) % D_k
        var sv = Sbase.load[width=4](r0 * D_k + off)
        smem_S[r0 * D_k + off] = sv[0]
        smem_S[r0 * D_k + off + 1] = sv[1]
        smem_S[r0 * D_k + off + 2] = sv[2]
        smem_S[r0 * D_k + off + 3] = sv[3]
        c += 256

    barrier()

    var r = t >> 3
    var p = t & 7
    var base = r * D_k + p * 16

    # accA = sum(S*k) dan accB = sum(S*q), keduanya dari SMEM (cepat)
    var accA: Float32 = 0.0
    var accB: Float32 = 0.0
    var j = 0
    while j < 16:
        var s0 = smem_S[base + j]
        var s1 = smem_S[base + j + 1]
        var s2 = smem_S[base + j + 2]
        var s3 = smem_S[base + j + 3]
        accA += s0 * smem_k[p * 16 + j] + s1 * smem_k[p * 16 + j + 1] + s2 * smem_k[p * 16 + j + 2] + s3 * smem_k[p * 16 + j + 3]
        accB += s0 * smem_q[p * 16 + j] + s1 * smem_q[p * 16 + j + 1] + s2 * smem_q[p * 16 + j + 2] + s3 * smem_q[p * 16 + j + 3]
        j += 4

    redA[t] = accA
    redB[t] = accB
    barrier()

    # Reduksi 8 parsial per baris (urutan 0..7, deterministik)
    var Atot: Float32 = 0.0
    var Btot: Float32 = 0.0
    var q2 = 0
    while q2 < 8:
        Atot += redA[r * 8 + q2]
        Btot += redB[r * 8 + q2]
        q2 += 1

    var dv = rg * 32 + r
    var delta = (Float32(v_ptr[hv * D_v + dv]) - g_decay * Atot) * beta

    var kq: Float32 = 0.0
    var m = 0
    while m < D_k:
        kq += smem_k[m] * smem_q[m]
        m += 1

    if p == 0:
        out_ptr[hv * D_v + dv] = Scalar[T](g_decay * Btot + delta * kq)

    # Tulis balik S yang baru: baca dari SMEM, tulis ke VRAM (float4)
    var wr = r * D_k + p * 16
    var j2 = 0
    while j2 < 16:
        var sv = SIMD[DType.float32, 4](
            smem_S[base + j2], smem_S[base + j2 + 1],
            smem_S[base + j2 + 2], smem_S[base + j2 + 3])
        var kv = SIMD[DType.float32, 4](
            smem_k[p * 16 + j2], smem_k[p * 16 + j2 + 1],
            smem_k[p * 16 + j2 + 2], smem_k[p * 16 + j2 + 3])
        var nv = sv * g_decay + kv * delta
        Sbase.store[width=4](wr + j2, nv)
        j2 += 4


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
    eps: Float32,
    out_row_stride: Int = 0,
    z_row_stride: Int = 0,
    legacy_gate_first: Int = 0
):
    """
    1 block per head hv, D_v thread per block.
    1. Menghitung RMSNorm per head ATAS x = keluaran rekurensi MURNI
    2. Mengalikan dengan bobot norm_w[dv] (jika HAS_NORM_W=True)
    3. BARU dikalikan silu(z[dv]) — gate SETELAH normalisasi

    PARITAS WAJIB dgn Qwen3NextRMSNormGated:
      * mlx-lm `qwen3_next.py:71-78`:
            x = mx.fast.rms_norm(hidden_states, self.weight, eps)
            return silu(gate) * x
      * Prism CUDA `rmsnorm_gated.cu:80-96`:
            local_sq += xv*xv            # x MURNI, bukan x*silu(g)
            normed = xv * scale * wv
            gated  = normed * silu(gv)

    Urutan terbalik (gate dulu, lalu norm atas x*silu(z)) menghasilkan galat
    gain per-head rms(x)/rms(x*silu(z)) di SETIAP layer GDN, prefill maupun
    decode — dan tidak terdeteksi oleh tes paritas apa pun karena kembar CPU
    ikut salah.

    `legacy_gate_first` (SAKELAR A/B, default 0 = BENAR):
      * 0 -> urutan referensi (norm atas x murni, gate terakhir).
      * != 0 -> reproduksi bug LAMA (norm atas x*silu(z)) HANYA untuk mengukur
        besar dampaknya di T4. JANGAN dipakai produksi. Di-set dari env
        `BONSAI_GDN_NORM_ORDER=gate_first` oleh peluncur di ops.mojo.
    """
    var hv = block_idx.x
    var dv = thread_idx.x
    var lane = dv & 31
    var wid = dv >> 5
    var idx = hv * D_v + dv

    # SMEM reduksi. 128 float (bukan 32): JALUR UMUM di bawah menulis
    # smem[dv] untuk tiap thread dv, dan D_v bisa sebesar 128 (config.mojo
    # gdn_head_v_dim). Jalur cepat (D_v == 128) hanya menyentuh smem[0..3],
    # jadi pembesaran ini tidak mengubah hasilnya satu bit. Peluncur di
    # ops.mojo menolak D_v > 128 -> tidak ada luapan.
    var smem = stack_allocation[
        128, Float32, alignment = 16, address_space = AddressSpace.SHARED
    ]()

    var x_val = Float32(gdn_out[block_idx.y * out_row_stride + idx])
    var z_val = Float32(z_ptr[block_idx.y * z_row_stride + idx])
    var silu_z = z_val / (1.0 + exp(-z_val))

    # Nilai yang dinormalkan: x MURNI (default, paritas referensi). Mode A/B
    # menormalkan x*silu(z) — persis perilaku lama.
    var norm_in = (x_val * silu_z) if legacy_gate_first != 0 else x_val

    # Hitung mean kuadrat atas norm_in
    var sq = norm_in * norm_in

    # Reduksi sum-of-squares atas D_v thread blok. Cabang seragam (D_v bernilai
    # sama untuk seluruh thread di blok -> tidak ada barrier menyimpang).
    #
    #   * D_v == 128 (satu-satunya geometri PRODUKSI, config.mojo
    #     gdn_head_v_dim=128): ladder shuffle full-warp + reduksi antar-warp di
    #     shared memory. KODE LAMA, tidak diubah -> hasil byte-identik dengan
    #     sebelum perbaikan ini. Syaratnya bukan sekadar ">= 32": ladder shuffle
    #     hanya absah bila SETIAP warp penuh (D_v kelipatan 32), DAN tahap
    #     antar-warp menjumlahkan tepat 4 slot (smem[lane] jika lane < 4), jadi
    #     hanya D_v == 128 yang memenuhi keduanya sekaligus.
    #   * D_v lain (1..127; bentuk uji non-produksi): kedua asumsi itu jebak.
    #       - D_v < 32: blok = satu warp parsial. shuffle_down(delta >= 16)
    #         membaca lane yang tak pernah dieksekusi (sampah).
    #       - D_v = 32/64/96: warp-warp penuh, tapi jumlah warp < 4, sehingga
    #         smem[1..3] (atau smem[2..3]) dibaca namun tak pernah ditulis.
    #       - D_v bukan kelipatan 32 (mis. 48): warp terakhir parsial, sehingga
    #         ladder shuffle-nya sendiri sudah membaca lane mati.
    #     Semua itu menghasilkan RMS yang salah besar (rel ~0,6 vs referensi
    #     FP64) tanpa crash. Sebagai gantinya seluruh reduksi dikerjakan di
    #     shared memory: tiap thread menulis kuadratnya, satu thread
    #     menjumlahkan semuanya. Tidak ada asumsi geometri sama sekali.
    if D_v == 128:
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
    else:
        # D_v <= 127 -> smem[0..D_v-1] muat di 128 slot. Tanpa shuffledown:
        # shuffle pada warp parsial tidak terdefinisi. Akumulasi berurutan oleh
        # thread 0 saja; cukup untuk bentuk uji (produksi selalu lewat cabang
        # D_v == 128 di atas).
        smem[dv] = sq
        barrier()
        if dv == 0:
            var wsq: Float32 = 0.0
            var j = 0
            while j < D_v:
                wsq += smem[j]
                j += 1
            smem[0] = 1.0 / sqrt(wsq / Float32(D_v) + eps)
        barrier()

    var inv_rms = smem[0]
    var gamma: Float32 = 1.0
    @parameter
    if HAS_NORM_W:
        gamma = norm_w[dv]
    if legacy_gate_first != 0:
        # SAKELAR A/B: bug lama — gate dulu, norm atas (x*silu(z)).
        # silu(z) TIDAK dikali lagi di sini (sudah masuk ke norm_in).
        gdn_out[block_idx.y * out_row_stride + idx] = Scalar[T](
            norm_in * inv_rms * gamma
        )
    else:
        # Paritas referensi: norm atas x murni, gate silu(z) BELAKANGAN.
        gdn_out[block_idx.y * out_row_stride + idx] = Scalar[T](
            (x_val * inv_rms * gamma) * silu_z
        )


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
    - Grouped Query Attention: group_size = H_q // H_kv.
      Bonsai-27B (config.mojo): H_q=24, H_kv=4 -> group_size = 6 (bukan 40//8).
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

    # Buffer Shared Memory Terpadu: head_dim (256) untuk Q + 32 reduksi warp
    # + 32 broadcast = 320 float (1280 B). Model riil Bonsai: head_dim=256,
    # H_q=24, H_kv=4 (terverifikasi [GQA DIAG] Run #50).
    # HINDARI pointer arithmetic di shared memory (smem + offset) karena NVPTX
    # mungkin mengonversi AddressSpace.SHARED ke GENERIC → illegal memory access.
    # Gunakan direct indexing smem[offset + idx] saja.
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
