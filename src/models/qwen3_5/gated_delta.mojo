# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/gated_delta.mojo
# Purpose: Arsitektur Rekurensi Gated DeltaNet (GDN) Linear Attention O(n)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt, exp
from os import getenv
from time import monotonic
from gpu.host import DeviceBuffer, DeviceContext as DeviceContextGPU
from .config import QwenConfig
from .norm import silu, sigmoid, softplus, rms_norm, head_rms_norm, rmsnorm_gated_fused
from .conv import CausalConv1dState
from .linear import QwenLinear1Bit
from src.ops import (
    causal_conv1d_sm75_launch_on, head_rmsnorm_sm75_launch_on,
    gdn_recurrence_sm75_launch_on, gdn_norm_gate_sm75_launch_on
)

struct GatedDeltaNetState:
    """
    State Memori Kausal Rekuren Penuh:
    1. Causal Conv1D State: [3, 12288]
    2. Recurrent Memory Matrix S: [64 heads, 128, 128]
    Total memori per layer = 4.14 MiB.
    Mendukung alokasi VRAM device untuk Full GPU execution.
    """
    var conv_state: CausalConv1dState
    var state_s: UnsafePointer[Float32, MutAnyOrigin] # [H_v * D_v * D_k]
    var num_v_heads: Int
    var head_v_dim: Int
    var head_k_dim: Int
    var s_elements: Int
    var conv_dim: Int
    # Buffer device di VRAM — state S disimpan FP16 (hemat 2x trafik DRAM;
    # math di dalam kernel tetap FP32). State awal selalu nol.
    var state_s_dev: UnsafePointer[Float16, MutAnyOrigin]
    var state_s_dev_buf: UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]
    var conv_buf_dev: UnsafePointer[Float32, MutAnyOrigin]
    var conv_buf_dev_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var dev_ready: Bool

    fn __init__(
        out self,
        conv_dim: Int = 12288,
        num_v_heads: Int = 64,
        head_v_dim: Int = 128,
        head_k_dim: Int = 128
    ):
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.s_elements = num_v_heads * head_v_dim * head_k_dim
        self.conv_state = CausalConv1dState(conv_dim=conv_dim, kernel_size=4)
        self.state_s = alloc[Float32](self.s_elements)
        self.state_s_dev = UnsafePointer[Float16, MutAnyOrigin]()
        self.state_s_dev_buf = UnsafePointer[DeviceBuffer[DType.float16], MutAnyOrigin]()
        self.conv_buf_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.conv_buf_dev_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.dev_ready = False
        self.reset()

    fn init_device(mut self, mut ctx: DeviceContextGPU) raises:
        """Alokasikan state S (FP16) dan conv buffer (FP32) di VRAM."""
        if self.dev_ready:
            return
        var sb_holder = alloc[DeviceBuffer[DType.float16]](1)
        sb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float16](self.s_elements)
        )
        # State awal = nol: tulis nol FP16 dari host (state FP32 lama tidak
        # pernah berisi apa pun sebelum token pertama, jadi tidak perlu copy).
        var z16 = alloc[Float16](self.s_elements)
        for i in range(self.s_elements):
            z16[i] = Float16(0.0)
        ctx.enqueue_copy(sb_holder[], z16)
        z16.free()
        self.state_s_dev_buf = sb_holder
        self.state_s_dev = sb_holder[].unsafe_ptr()

        var cb_holder = alloc[DeviceBuffer[DType.float32]](1)
        cb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float32](3 * self.conv_dim)
        )
        ctx.enqueue_copy(cb_holder[], self.conv_state.buffer)
        self.conv_buf_dev_buf = cb_holder
        self.conv_buf_dev = cb_holder[].unsafe_ptr()
        self.dev_ready = True

    fn reset(mut self):
        self.conv_state.reset()
        for i in range(self.s_elements):
            self.state_s[i] = 0.0

    fn free(self):
        self.conv_state.free()
        self.state_s.free()

fn qwen3_5_gdn_step(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_norm: UnsafePointer[Float32, MutAnyOrigin],
    mut in_proj_all: QwenLinear1Bit,
    conv_weights: UnsafePointer[Float32, MutAnyOrigin],
    mut out_proj: QwenLinear1Bit,
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    norm_w: UnsafePointer[Float32, MutAnyOrigin],
    has_params: Bool,
    mut state: GatedDeltaNetState,
    config: QwenConfig
) raises:
    """
    Satu langkah inferensi token penuh Gated DeltaNet:
    1. in_proj_all memproyeksikan x ke buffer: [conv_dim (qkv) + z + b + a]
    2. Causal Conv1D dengan sliding window state 3 token + SiLU
    3. Ekstraksi q, k, v dari conv1d output
    4. Per-Head RMSNorm pada Q dan K dengan faktor skala inv_scale = 1 / sqrt(Dk)
    5. Rekurensi DeltaNet: S_t = S_{t-1} * g + k * delta^T
    6. RMSNormGated: RMSNorm(S * q) * silu(z)
    7. out_proj balik ke hidden_size
    """
    var H_v = config.gdn_num_v_heads
    var H_k = config.gdn_num_k_heads
    var D_k = config.gdn_head_k_dim
    var D_v = config.gdn_head_v_dim
    var conv_dim = config.gdn_conv_dim

    var total_proj_dim = in_proj_all.N
    var proj_raw = alloc[Float32](total_proj_dim)
    var conv_out = alloc[Float32](conv_dim)
    var gdn_out  = alloc[Float32](H_v * D_v)

    # 1. Proyeksi Linear 1-Bit Masukan
    in_proj_all.forward(x_norm, proj_raw, 1)

    # 2. Causal Conv1D (Kernel Size 4) pada komponen QKV
    state.conv_state.step(conv_out, proj_raw, conv_weights)

    # 3. Ekstraksi Q, K, V dari conv_out:
    var q_ptr = conv_out
    var k_ptr = conv_out.offset(H_k * D_k)
    var v_ptr = conv_out.offset(2 * H_k * D_k)

    # Z, B, A diambil dari sisa proj_raw
    var z_ptr = proj_raw.offset(conv_dim)
    var b_ptr = proj_raw.offset(conv_dim + H_v * D_v)
    var a_ptr = proj_raw.offset(conv_dim + H_v * D_v + H_v)

    # 4. Q-Norm & K-Norm per-head
    var inv_scale: Float32 = 1.0 / sqrt(Float32(D_k))
    var q_normed = alloc[Float32](H_k * D_k)
    var k_normed = alloc[Float32](H_k * D_k)
    head_rms_norm(q_normed, q_ptr, H_k, D_k, inv_scale * inv_scale, config.rms_norm_eps)
    head_rms_norm(k_normed, k_ptr, H_k, D_k, inv_scale, config.rms_norm_eps)

    # 5. Rekurensi Gated Delta Rule (sesuai mlx-lm gated_delta.py baris 392-410)
    var repeat_factor = H_v // H_k

    for hv in range(H_v):
        var hk = hv // repeat_factor
        var s_head_offset = hv * (D_v * D_k)
        var q_head = q_normed.offset(hk * D_k)
        var k_head = k_normed.offset(hk * D_k)
        var v_head = v_ptr.offset(hv * D_v)
        var out_head = gdn_out.offset(hv * D_v)

        var a_val = a_ptr[hv]
        var b_val = b_ptr[hv]
        # Paritas mlx-lm gated_delta.py: g = -exp(A_log) * softplus(a + dt_bias)
        var g_decay: Float32
        if has_params:
            g_decay = exp(Float32(-1.0) * exp(a_log[hv]) * softplus(a_val + dt_bias[hv]))
        else:
            g_decay = exp(Float32(-0.5) * softplus(a_val + 1.0))
        var beta = sigmoid(b_val)

        for dv in range(D_v):
            var row_offset = s_head_offset + dv * D_k

            # kv_mem = sum(state * k)
            var kv_mem: Float32 = 0.0
            for dk in range(D_k):
                state.state_s[row_offset + dk] *= g_decay
                kv_mem += state.state_s[row_offset + dk] * k_head[dk]

            # Delta error = (v - kv_mem) * beta
            var delta = (v_head[dv] - kv_mem) * beta

            # Update state S = S + k * delta dan baca output S * q
            var read_out: Float32 = 0.0
            for dk in range(D_k):
                state.state_s[row_offset + dk] += k_head[dk] * delta
                read_out += state.state_s[row_offset + dk] * q_head[dk]

            out_head[dv] = read_out

    # 6. Gate SiLU lalu RMSNorm PER HEAD (D_v) dengan bobot bersama [D_v] —
    #    paritas Qwen3NextRMSNormGated(head_v_dim); bobot asli [128], bukan
    #    [H_v*D_v]. Fallback: fused tanpa bobot (jalur sintetis lama).
    if has_params:
        var out_total = H_v * D_v
        for i in range(out_total):
            gdn_out[i] = gdn_out[i] * silu(z_ptr[i])
        for hv in range(H_v):
            var out_head = gdn_out.offset(hv * D_v)
            var ss: Float32 = 0.0
            for d in range(D_v):
                ss += out_head[d] * out_head[d]
            var inv = 1.0 / sqrt(ss / Float32(D_v) + config.rms_norm_eps)
            for d in range(D_v):
                out_head[d] = out_head[d] * inv * norm_w[d]
    else:
        for hv in range(H_v):
            var out_head = gdn_out.offset(hv * D_v)
            var z_head = z_ptr.offset(hv * D_v)
            rmsnorm_gated_fused(out_head, out_head, z_head, D_v, config.rms_norm_eps)

    # 7. Proyeksi Keluar Linear 1-Bit
    out_proj.forward(gdn_out, out_ptr, 1)

    proj_raw.free()
    conv_out.free()
    gdn_out.free()
    q_normed.free()
    k_normed.free()

fn qwen3_5_gdn_step_gpu(
    out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    x_norm_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    proj_raw_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    conv_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    q_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    gdn_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    mut in_proj_all: QwenLinear1Bit,
    conv_weights: UnsafePointer[Float32, MutAnyOrigin],
    mut out_proj: QwenLinear1Bit,
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    norm_w: UnsafePointer[Float32, MutAnyOrigin],
    has_params: Bool,
    mut state: GatedDeltaNetState,
    config: QwenConfig,
    pos: Int = 0
) raises:
    """
    Eksekusi 1 langkah token GDN 100% di GPU (VRAM-to-VRAM):
    1. in_proj_all.forward_device (W1A16 GPU matmul)
    2. Causal Conv1d 4-tap GPU kernel
    3. Head RMSNorm GPU kernel (Q-Norm & K-Norm)
    4. GDN Recurrence GPU kernel (64 block x 128 thread masif-paralel)
    5. Fused GDN Gating & Per-Head RMSNorm GPU kernel
    6. out_proj.forward_device (W1A16 GPU matmul)
    """
    alias T = DType.float16
    var H_v = config.gdn_num_v_heads
    var H_k = config.gdn_num_k_heads
    var D_k = config.gdn_head_k_dim
    var D_v = config.gdn_head_v_dim
    var conv_dim = config.gdn_conv_dim

    var ctx_ptr = in_proj_all.ctx_ptr
    if not ctx_ptr:
        raise Error("FATAL: ctx_ptr null pada qwen3_5_gdn_step_gpu!")
    var ctx = ctx_ptr[]

    if not state.dev_ready:
        state.init_device(ctx)

    var prof = pos == 0
    var pv = getenv("BONSAI_PROFILE")
    prof = prof and pv and pv[0] == "1"

    # 1. Proyeksi Linear 1-Bit Masukan di VRAM
    var t0 = monotonic()
    in_proj_all.forward_device(x_norm_dev, proj_raw_dev, 1)
    var t_in = 0.0
    if prof:
        ctx.synchronize()
        t_in = Float64(monotonic() - t0) / 1e3

    # 2. Causal Conv1D 4-Tap pada komponen QKV di VRAM
    var t1 = monotonic()
    causal_conv1d_sm75_launch_on[T](
        ctx,
        state.conv_buf_dev, proj_raw_dev, conv_weights,
        conv_weights != UnsafePointer[Float32, MutAnyOrigin](),
        conv_out_dev, conv_dim
    )
    var t_conv = 0.0
    if prof:
        ctx.synchronize()
        t_conv = Float64(monotonic() - t1) / 1e3

    # 3. Ekstraksi Q, K, V dan Z, B, A dari VRAM buffer
    var q_dev = conv_out_dev
    var k_dev = conv_out_dev.offset(H_k * D_k)
    var v_dev = conv_out_dev.offset(2 * H_k * D_k)

    var z_dev = proj_raw_dev.offset(conv_dim)
    var b_dev = proj_raw_dev.offset(conv_dim + H_v * D_v)
    var a_dev = proj_raw_dev.offset(conv_dim + H_v * D_v + H_v)

    # 4. Q-Norm & K-Norm per-head di VRAM
    var inv_scale_k = 1.0 / sqrt(Float32(D_k))
    var inv_scale_q = inv_scale_k * inv_scale_k
    var t2 = monotonic()
    head_rmsnorm_sm75_launch_on[T](ctx, q_dev, q_normed_dev, H_k, D_k, inv_scale_q, config.rms_norm_eps)
    head_rmsnorm_sm75_launch_on[T](ctx, k_dev, k_normed_dev, H_k, D_k, inv_scale_k, config.rms_norm_eps)
    var t_qk = 0.0
    if prof:
        ctx.synchronize()
        t_qk = Float64(monotonic() - t2) / 1e3

    # 5. Rekurensi Gated Delta Rule masif-paralel di GPU
    var repeat_factor = H_v // H_k
    var t3 = monotonic()
    gdn_recurrence_sm75_launch_on[T](
        ctx,
        state.state_s_dev, q_normed_dev, k_normed_dev,
        v_dev, a_dev, b_dev, a_log, dt_bias, has_params,
        gdn_out_dev, repeat_factor, H_v, D_v, D_k
    )
    var t_rec = 0.0
    if prof:
        ctx.synchronize()
        t_rec = Float64(monotonic() - t3) / 1e3

    # 6. Fused SiLU(z) Gate & Per-Head RMSNorm di VRAM
    var t4 = monotonic()
    gdn_norm_gate_sm75_launch_on[T](
        ctx,
        gdn_out_dev, z_dev, norm_w, norm_w != UnsafePointer[Float32, MutAnyOrigin](),
        H_v, D_v, config.rms_norm_eps
    )
    var t_gate = 0.0
    if prof:
        ctx.synchronize()
        t_gate = Float64(monotonic() - t4) / 1e3

    # 7. Proyeksi Keluar Linear 1-Bit di VRAM
    var t5 = monotonic()
    out_proj.forward_device(gdn_out_dev, out_dev, 1)
    if prof:
        ctx.synchronize()
        print(
            "[PROF-GDN] in_proj=", t_in, "us conv=", t_conv, "us qk_norm=",
            t_qk, "us recurrence=", t_rec, "us norm_gate=", t_gate,
            "us out_proj=", Float64(monotonic() - t5) / 1e3, "us"
        )

