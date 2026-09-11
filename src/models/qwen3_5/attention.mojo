# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/attention.mojo
# Purpose: Gated Full Attention dengan GQA, RoPE Parsial, dan KV-Cache
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt
from gpu.host import DeviceBuffer
from .config import QwenConfig
from .norm import head_rms_norm, softmax, sigmoid
from .rope import apply_partial_rope
from .linear import QwenLinear1Bit, DeviceContextGPU
from src.ops import (
    head_rmsnorm_sm75_launch_on,
    partial_rope_sm75_launch_on,
    kv_cache_append_sm75_launch_on,
    gqa_attention_sm75_launch_on
)
from .khq_dump import khq_dump_kv, khq_dump_attn
from src.khq.runtime import (
    khq_active, khq_init_layer, khq_capture_unroped, khq_step
)

struct AttentionKVCache:
    """
    KV Cache untuk layer Full Attention (Layer 3, 7, 11, ...):
    Menyimpan token riwayat untuk 8 KV heads x 128 head_dim.
    Mendukung buffer host dan buffer VRAM (DeviceBuffer FP16).
    """
    var k_cache: UnsafePointer[Float32, MutAnyOrigin] # [max_seq_len, num_kv_heads * head_dim]
    var v_cache: UnsafePointer[Float32, MutAnyOrigin] # [max_seq_len, num_kv_heads * head_dim]
    var max_seq_len: Int
    var num_kv_heads: Int
    var head_dim: Int
    var current_len: Int

    # Buffer device di VRAM (FP16)
    var k_cache_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var v_cache_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
    var k_cache_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var v_cache_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var dev_ready: Bool

    fn __init__(
        out self,
        max_seq_len: Int = 4096,
        num_kv_heads: Int = 8,
        head_dim: Int = 128
    ):
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.current_len = 0
        var total_elems = max_seq_len * (num_kv_heads * head_dim)
        self.k_cache = alloc[Float32](total_elems)
        self.v_cache = alloc[Float32](total_elems)
        self.k_cache_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.v_cache_dev = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        self.k_cache_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.v_cache_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.dev_ready = False

    fn init_device(mut self, mut ctx: DeviceContextGPU) raises:
        """Alokasikan KV Cache langsung di VRAM GPU (FP16)."""
        if self.dev_ready:
            return
        var total_elems = self.max_seq_len * (self.num_kv_heads * self.head_dim)
        var kb_holder = alloc[DeviceBuffer[DType.float16]](1)
        kb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float16](total_elems)
        )
        self.k_cache_dev_buf = kb_holder
        self.k_cache_dev = kb_holder[].unsafe_ptr()

        var vb_holder = alloc[DeviceBuffer[DType.float16]](1)
        vb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float16](total_elems)
        )
        self.v_cache_dev_buf = vb_holder
        self.v_cache_dev = vb_holder[].unsafe_ptr()
        self.dev_ready = True

    fn append(mut self, k_token: UnsafePointer[Float32, MutAnyOrigin], v_token: UnsafePointer[Float32, MutAnyOrigin], mut pos: Int):
        var kv_dim = self.num_kv_heads * self.head_dim
        var offset = pos * kv_dim
        for i in range(kv_dim):
            self.k_cache[offset + i] = k_token[i]
            self.v_cache[offset + i] = v_token[i]

    fn free(self):
        self.k_cache.free()
        self.v_cache.free()

fn qwen3_5_gated_attention_step(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_norm: UnsafePointer[Float32, MutAnyOrigin],
    mut q_proj: QwenLinear1Bit,
    mut k_proj: QwenLinear1Bit,
    mut v_proj: QwenLinear1Bit,
    mut o_proj: QwenLinear1Bit,
    q_norm_w: UnsafePointer[Float32, MutAnyOrigin],
    k_norm_w: UnsafePointer[Float32, MutAnyOrigin],
    has_norms: Bool,
    mut kv_cache: AttentionKVCache,
    pos: Int,
    config: QwenConfig
) raises:
    """
    Forward pass 1 token penuh Gated Attention sesuai qwen3_next.py:
    1. q_proj menghasilkan Queries AND Attention Gate (32 heads x 128 x 2 = 8192)
    2. k_proj & v_proj menghasilkan Keys dan Values (8 heads x 128 = 1024)
    3. Q-Norm & K-Norm per head
    4. RoPE parsial 25% (32 dimensi pertama per head)
    5. Simpan Key dan Value ke KV Cache ring buffer
    6. Scaled Dot Product Attention dengan Grouped Query Attention (32 query : 8 KV)
    7. Attention Output digate via Sigmoid: out_ptr = context * sigmoid(gate)
    8. o_proj balik ke hidden_size (4096)
    """
    var H_q = config.num_attention_heads      # 32
    var H_kv = config.num_key_value_heads     # 8
    var D = config.head_dim                   # 128
    var rot_dim = config.rotary_dim           # 32
    var scale: Float32 = 1.0 / sqrt(Float32(D))

    var q_gate_buf = alloc[Float32](H_q * D * 2)
    var k_buf = alloc[Float32](H_kv * D)
    var v_buf = alloc[Float32](H_kv * D)
    var context = alloc[Float32](H_q * D)

    # 1. Proyeksi Linear 1-Bit
    q_proj.forward(x_norm, q_gate_buf, 1)
    k_proj.forward(x_norm, k_buf, 1)
    v_proj.forward(x_norm, v_buf, 1)

    # 2. Q-Norm & K-Norm per head (+ bobot belajar bila ada — paritas qwen3_next)
    # Layout q_gate_buf: interleaved [q_0 (D), gate_0 (D), q_1 (D), gate_1 (D), ...]
    head_rms_norm(q_gate_buf, q_gate_buf, H_q, D, 1.0, config.rms_norm_eps, 2 * D)
    head_rms_norm(k_buf, k_buf, H_kv, D, 1.0, config.rms_norm_eps, D)
    if has_norms:
        for h in range(H_q):
            var h_offset = h * (2 * D)
            for d in range(D):
                q_gate_buf[h_offset + d] *= q_norm_w[d]
        for h in range(H_kv):
            for d in range(D):
                k_buf[h * D + d] *= k_norm_w[d]

    # 3. RoPE Parsial (25% dimensi per head) dengan interleaved stride 2 * D untuk Query
    var mutable_pos = pos
    apply_partial_rope(q_gate_buf, H_q, D, rot_dim, pos, config.rope_theta, 2 * D)
    apply_partial_rope(k_buf, H_kv, D, rot_dim, pos, config.rope_theta, D)

    # 4. Simpan ke KV Cache
    kv_cache.append(k_buf, v_buf, mutable_pos)
    var seq_len = pos + 1

    # 5. Scaled Dot Product Attention dengan GQA
    var group_size = H_q // H_kv # 32 // 8 = 4
    var attn_scores = alloc[Float32](seq_len)
    var attn_probs  = alloc[Float32](seq_len)

    for hq in range(H_q):
        var hkv = hq // group_size
        var q_head = q_gate_buf.offset(hq * (2 * D))
        var gate_head = q_gate_buf.offset(hq * (2 * D) + D)
        var ctx_head = context.offset(hq * D)

        for t in range(seq_len):
            var k_cached = kv_cache.k_cache.offset(t * (H_kv * D) + hkv * D)
            var dot: Float32 = 0.0
            for d in range(D):
                dot += q_head[d] * k_cached[d]
            attn_scores[t] = dot * scale

        softmax(attn_probs, attn_scores, seq_len)

        for d in range(D):
            var v_acc: Float32 = 0.0
            for t in range(seq_len):
                var v_cached = kv_cache.v_cache.offset(t * (H_kv * D) + hkv * D)
                v_acc += attn_probs[t] * v_cached[d]
            
            # Gated Attention Output: context * sigmoid(gate)
            var g_val = sigmoid(gate_head[d])
            ctx_head[d] = v_acc * g_val

    # 7. Proyeksi Keluar Linear o_proj
    o_proj.forward(context, out_ptr, 1)

    q_gate_buf.free()
    k_buf.free()
    v_buf.free()
    context.free()
    attn_scores.free()
    attn_probs.free()


fn qwen3_5_gated_attention_step_gpu(
    out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    x_norm_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    q_gate_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    attn_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    attn_scores_dev: UnsafePointer[Float32, MutAnyOrigin],
    mut q_proj: QwenLinear1Bit,
    mut k_proj: QwenLinear1Bit,
    mut v_proj: QwenLinear1Bit,
    mut o_proj: QwenLinear1Bit,
    q_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin],
    k_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin],
    has_norms: Bool,
    mut kv_cache: AttentionKVCache,
    pos: Int,
    config: QwenConfig,
    layer_idx: Int = -1
) raises:
    """
    Forward pass 1 token penuh Gated Full Attention 100% di VRAM GPU:
    1. q_proj menghasilkan Query + Gate (40 heads x 128 x 2 = 10240) di VRAM
    2. k_proj & v_proj menghasilkan Key dan Value (8 heads x 128 = 1024) di VRAM
    3. Head RMSNorm berbobot per-head di VRAM
    4. RoPE parsial 25% (32 dimensi pertama per-head) di VRAM
    5. Simpan Key dan Value ke KV Cache ring buffer di VRAM
    6. Fused GQA (40 Q : 8 KV = rasio 5:1) + Softmax + Sigmoid Gate di VRAM
    7. o_proj balik ke hidden_size (5120) di VRAM
    """
    alias T = DType.float16
    var H_q = config.num_attention_heads      # 40 di Bonsai-27B
    var H_kv = config.num_key_value_heads     # 8 di Bonsai-27B
    var D = config.head_dim                   # 128
    var rot_dim = config.rotary_dim           # 32
    var scale: Float32 = 1.0 / sqrt(Float32(D))

    var ctx_ptr = q_proj.ctx_ptr
    if not ctx_ptr:
        raise Error("FATAL: ctx_ptr null pada qwen3_5_gated_attention_step_gpu!")
    var ctx = ctx_ptr[]

    # KV cache DI-INIT di main.mojo untuk semua layer attention sebelum loop.
    # Lazy init di sini dihapus: mutasi struct via parameter berisiko ter-copy
    # (ownership ambigu) — step ini read-only terhadap kv_cache.
    if pos == 0:
        var p_null = UnsafePointer[Scalar[DType.float16], MutAnyOrigin]()
        var f_null = UnsafePointer[Float32, MutAnyOrigin]()
        print(
            ">> [GQA DIAG] seq_len=", pos + 1,
            " max_seq=", kv_cache.max_seq_len,
            " H_q=", H_q, " H_kv=", H_kv, " D=", D, " rot=", rot_dim,
            " q_gate_null=", q_gate_dev == p_null,
            " k_null=", k_dev == p_null,
            " v_null=", v_dev == p_null,
            " kc_null=", kv_cache.k_cache_dev == p_null,
            " vc_null=", kv_cache.v_cache_dev == p_null,
            " out_null=", attn_out_dev == p_null,
            " score_null=", attn_scores_dev == f_null,
            " qn_null=", q_norm_w_dev == f_null,
            " kn_null=", k_norm_w_dev == f_null
        )

    if pos == 0:
        print(">> [ATTN STEP] pos=0: qkv_proj -> qk_norm (has_norms=", has_norms, ") -> rope -> kv_append -> gqa -> o_proj")

    # 1. Proyeksi Linear 1-Bit Masukan di VRAM (tanpa transfer host)
    q_proj.forward_device(x_norm_dev, q_gate_dev, 1)
    k_proj.forward_device(x_norm_dev, k_dev, 1)
    v_proj.forward_device(x_norm_dev, v_dev, 1)

    qwen3_5_gated_attention_step_gpu_from_proj(
        ctx_ptr, q_gate_dev, k_dev, v_dev, attn_out_dev, attn_scores_dev,
        q_norm_w_dev, k_norm_w_dev, has_norms,
        kv_cache, pos, config, layer_idx
    )

    # 6. Proyeksi Keluar Linear 1-Bit o_proj di VRAM
    o_proj.forward_device(attn_out_dev, out_dev, 1)


fn qwen3_5_gated_attention_step_gpu_from_proj(
    ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
    q_gate_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    attn_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    attn_scores_dev: UnsafePointer[Float32, MutAnyOrigin],
    q_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin],
    k_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin],
    has_norms: Bool,
    mut kv_cache: AttentionKVCache,
    pos: Int,
    config: QwenConfig,
    layer_idx: Int = -1
) raises:
    """
    Langkah attention TANPA proyeksi linear (q/k/v/o_proj di luar fungsi).
    Dipakai decode per-token DAN loop prefill batched (pointer baris M-token
    di-offset oleh pemanggil, pos absolut per token). Urutan: q/k norm ->
    RoPE parsial half-split -> append KV cache -> GQA + softmax + sigmoid gate.
    """
    alias T = DType.float16
    var H_q = config.num_attention_heads      # 48 di Bonsai-27B
    var H_kv = config.num_key_value_heads     # 8 di Bonsai-27B
    var D = config.head_dim                   # 128
    var rot_dim = config.rotary_dim           # 32
    var scale: Float32 = 1.0 / sqrt(Float32(D))

    if not ctx_ptr:
        raise Error("FATAL: ctx_ptr null pada qwen3_5_gated_attention_step_gpu_from_proj!")
    var ctx = ctx_ptr[]

    # KHQ: siapkan state layer (idempoten) bila jalur KV terkompresi aktif.
    var khq_on = khq_active()
    if khq_on:
        khq_init_layer(ctx, layer_idx, H_q, H_kv, D)

    # Q-Norm & K-Norm per-head dengan bobot belajar di VRAM (stride 2*D untuk interleaved Query)
    head_rmsnorm_sm75_launch_on[T](
        ctx, q_gate_dev, q_gate_dev, H_q, D, 1.0,
        q_norm_w_dev, has_norms, config.rms_norm_eps, 2 * D
    )
    head_rmsnorm_sm75_launch_on[T](
        ctx, k_dev, k_dev, H_kv, D, 1.0,
        k_norm_w_dev, has_norms, config.rms_norm_eps, D
    )

    # KHQ: dump K post-norm PRE-rope (unroped) + V — kalibrasi KudaHitamQuant.
    # Tanpa hook ini K harus di-unrotate ulang (buang waktu).
    khq_dump_kv(ctx, layer_idx, k_dev, v_dev)

    # KHQ: simpan K unroped utk kompresi (sebelum RoPE — tanpa unrotate).
    if khq_on:
        khq_capture_unroped(ctx, layer_idx, k_dev)

    # RoPE Parsial di VRAM (half-split MLX: pasangan (i, i+rot_dim/2) di
    # 64 dim rotary per-head; stride 2*D untuk Query interleaved)
    partial_rope_sm75_launch_on[T](
        ctx, q_gate_dev, H_q, D, rot_dim, pos, config.rope_theta, 2 * D
    )
    partial_rope_sm75_launch_on[T](
        ctx, k_dev, H_kv, D, rot_dim, pos, config.rope_theta, D
    )

    # KHQ: dump skor attention (post-RoPE, post-softmax kausal) — SmartVQ.
    khq_dump_attn(ctx, layer_idx, q_gate_dev, k_dev, pos, H_q, H_kv, D, scale)

    # KHQ aktif: attention atas KV terkompresi + jendela raw (tanpa KV cache fp16)
    if khq_on:
        if khq_step(ctx, layer_idx, q_gate_dev, k_dev, v_dev, attn_out_dev, scale):
            return
        # Jangan fallback senyap: kalau jalur kompresi aktif tapi gagal, hasil
        # fp16 akan memakai KV cache yang tidak pernah diisi (state campur).
        raise Error("KHQ aktif tapi khq_step GAGAL di layer " + String(layer_idx)
                    + " — periksa kernel/centroid, bukan fallback ke fp16")

    # Simpan Key dan Value ke KV Cache ring buffer di VRAM
    kv_cache_append_sm75_launch_on[T](
        ctx, kv_cache.k_cache_dev, kv_cache.v_cache_dev,
        k_dev, v_dev, pos, H_kv * D
    )

    # Fused GQA Attention + Softmax + Sigmoid Gate di VRAM (memproses interleaved Query & Gate)
    var seq_len = pos + 1
    gqa_attention_sm75_launch_on[T](
        ctx, q_gate_dev, kv_cache.k_cache_dev, kv_cache.v_cache_dev,
        attn_out_dev, attn_scores_dev,
        seq_len, kv_cache.max_seq_len, H_q, H_kv, D, scale
    )

