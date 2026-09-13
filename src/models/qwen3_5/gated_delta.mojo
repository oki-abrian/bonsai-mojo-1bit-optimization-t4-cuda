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
    1. Causal Conv1D State: [3, conv_dim]  (Bonsai-27B: conv_dim = 10240)
    2. Recurrent Memory Matrix S: [H_v, D_v, D_k]  (Bonsai-27B: 48 x 128 x 128)
    Total memori per layer = 3 x 10240 x 4 B + 48 x 128 x 128 x 4 B
                           = 120 KiB + 3,0 MiB ≈ 3,1 MiB.
    (Angka lama 4,14 MiB berasal dari dimensi checkpoint lama 64x128x128 /
     conv_dim 12288 dan sudah tidak berlaku.)
    Mendukung alokasi VRAM device untuk Full GPU execution.
    """
    var conv_state: CausalConv1dState
    var state_s: UnsafePointer[Float32, MutAnyOrigin] # [H_v * D_v * D_k]
    var num_v_heads: Int
    var head_v_dim: Int
    var head_k_dim: Int
    var s_elements: Int
    var conv_dim: Int
    # Buffer device di VRAM — state S disimpan FP32 (WAJIB, bukan FP16).
    # Referensi: config model `mamba_ssm_dtype: float32`, mlx-lm (mx.float32),
    # dan llama.cpp-prism (`recurrent_type_k/v = GGML_TYPE_F32`). Karena decay
    # mendekati 1.0, error pembulatan per-langkah akan terakumulasi ratusan
    # langkah dan merusak state. State awal selalu nol.
    var state_s_dev: UnsafePointer[Float32, MutAnyOrigin]
    var state_s_dev_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var conv_buf_dev: UnsafePointer[Float32, MutAnyOrigin]
    var conv_buf_dev_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var dev_ready: Bool

    fn __init__(
        out self,
        conv_dim: Int = 10240,
        num_v_heads: Int = 48,
        head_v_dim: Int = 128,
        head_k_dim: Int = 128
    ):
        """Default = dimensi Bonsai-27B yang SEBENARNYA (config.mojo
        `qwen_27b_default`): conv_dim 10240 = 2*(16*128) + 48*128,
        H_v 48 x 128, H_k 16 x 128. Default lama (12288 / 64 head) berasal dari
        checkpoint lain dan membuat konstruksi tanpa argumen mengalokasikan
        state yang salah bentuk."""
        self.conv_dim = conv_dim
        self.num_v_heads = num_v_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.s_elements = num_v_heads * head_v_dim * head_k_dim
        self.conv_state = CausalConv1dState(conv_dim=conv_dim, kernel_size=4)
        self.state_s = alloc[Float32](self.s_elements)
        self.state_s_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.state_s_dev_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.conv_buf_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.conv_buf_dev_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.dev_ready = False
        self.reset()

    fn init_device(mut self, mut ctx: DeviceContextGPU) raises:
        """Alokasikan state S (FP32) dan conv buffer (FP32) di VRAM."""
        if self.dev_ready:
            return
        var sb_holder = alloc[DeviceBuffer[DType.float32]](1)
        sb_holder.init_pointee_move(
            ctx.enqueue_create_buffer[DType.float32](self.s_elements)
        )
        # State awal = nol: copy langsung dari host state_s (FP32, sudah nol
        # setelah reset()). Tidak ada konversi dtype lagi.
        ctx.enqueue_copy(sb_holder[], self.state_s)
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
        # PARITAS Qwen3NextRMSNormGated (mlx-lm qwen3_next.py:71-78):
        # variance dihitung atas keluaran rekurensi MURNI, dikali bobot norm,
        # lalu silu(z) DIKALIKAN TERAKHIR. Urutan terbalik menghasilkan galat
        # gain per-head rms(x)/rms(x*silu(z)) di setiap layer GDN.
        #
        # SAKELAR A/B `BONSAI_GDN_NORM_ORDER=gate_first`: reproduksi bug LAMA
        # (gate dulu, norm atas x*silu(z)) supaya dampaknya terukur di T4.
        # Disamakan dgn kernel GPU (elementwise_sm75.mojo) agar tes paritas
        # GPU-vs-CPU tetap bermakna di KEDUA mode. Default = urutan benar.
        var legacy_env = getenv("BONSAI_GDN_NORM_ORDER")
        var legacy = Bool(
            legacy_env
            and (legacy_env == "gate_first" or legacy_env == "legacy"
                 or legacy_env == "1")
        )
        for hv in range(H_v):
            var out_head = gdn_out.offset(hv * D_v)
            var z_head = z_ptr.offset(hv * D_v)
            var ss: Float32 = 0.0
            if legacy:
                for d in range(D_v):
                    var g = out_head[d] * silu(z_head[d])
                    ss += g * g
                var inv0 = 1.0 / sqrt(ss / Float32(D_v) + config.rms_norm_eps)
                for d in range(D_v):
                    out_head[d] = (out_head[d] * silu(z_head[d]) * inv0 * norm_w[d])
            else:
                for d in range(D_v):
                    ss += out_head[d] * out_head[d]
                var inv = 1.0 / sqrt(ss / Float32(D_v) + config.rms_norm_eps)
                for d in range(D_v):
                    out_head[d] = (out_head[d] * inv * norm_w[d]) * silu(z_head[d])
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
    2-6. qwen3_5_gdn_step_gpu_from_proj (conv -> norm -> recurrence -> gate)
    7. out_proj.forward_device (W1A16 GPU matmul)
    """
    var ctx_ptr = in_proj_all.ctx_ptr
    if not ctx_ptr:
        raise Error("FATAL: ctx_ptr null pada qwen3_5_gdn_step_gpu!")

    in_proj_all.forward_device(x_norm_dev, proj_raw_dev, 1)

    qwen3_5_gdn_step_gpu_from_proj(
        ctx_ptr, proj_raw_dev, conv_out_dev,
        q_normed_dev, k_normed_dev, gdn_out_dev,
        conv_weights, a_log, dt_bias, norm_w, has_params,
        state, config, pos
    )

    out_proj.forward_device(gdn_out_dev, out_dev, 1)


fn qwen3_5_gdn_step_gpu_from_proj(
    ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
    proj_raw_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    conv_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    q_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    gdn_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    conv_weights: UnsafePointer[Float32, MutAnyOrigin],
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    norm_w: UnsafePointer[Float32, MutAnyOrigin],
    has_params: Bool,
    mut state: GatedDeltaNetState,
    config: QwenConfig,
    pos: Int = 0
) raises:
    """
    Langkah GDN tanpa proyeksi linear (in_proj/out_proj di luar fungsi).
    Dipakai decode per-token DAN loop prefill batched (pointer baris M-token
    di-offset oleh pemanggil). Urutan: conv1d kausal -> q/k norm -> rekurensi
    delta rule -> gate silu(z)+rmsnorm per head. Semua update state (jendela
    conv & matriks S) berjalan sekuensial via stream yang sama.
    """
    alias T = DType.float16
    var H_v = config.gdn_num_v_heads
    var H_k = config.gdn_num_k_heads
    var D_k = config.gdn_head_k_dim
    var D_v = config.gdn_head_v_dim
    var conv_dim = config.gdn_conv_dim

    if not ctx_ptr:
        raise Error("FATAL: ctx_ptr null pada qwen3_5_gdn_step_gpu_from_proj!")
    var ctx = ctx_ptr[]

    if not state.dev_ready:
        state.init_device(ctx)

    # Causal Conv1D 4-Tap pada komponen QKV di VRAM
    causal_conv1d_sm75_launch_on[T](
        ctx,
        state.conv_buf_dev, proj_raw_dev, conv_weights,
        conv_weights != UnsafePointer[Float32, MutAnyOrigin](),
        conv_out_dev, conv_dim
    )

    # Ekstraksi Q, K, V dan Z, B, A dari VRAM buffer
    var q_dev = conv_out_dev
    var k_dev = conv_out_dev.offset(H_k * D_k)
    var v_dev = conv_out_dev.offset(2 * H_k * D_k)

    var z_dev = proj_raw_dev.offset(conv_dim)
    var b_dev = proj_raw_dev.offset(conv_dim + H_v * D_v)
    var a_dev = proj_raw_dev.offset(conv_dim + H_v * D_v + H_v)

    # Q-Norm & K-Norm per-head di VRAM
    var inv_scale_k = 1.0 / sqrt(Float32(D_k))
    var inv_scale_q = inv_scale_k * inv_scale_k
    head_rmsnorm_sm75_launch_on[T](ctx, q_dev, q_normed_dev, H_k, D_k, inv_scale_q, config.rms_norm_eps)
    head_rmsnorm_sm75_launch_on[T](ctx, k_dev, k_normed_dev, H_k, D_k, inv_scale_k, config.rms_norm_eps)

    # Rekurensi Gated Delta Rule masif-paralel di GPU
    var repeat_factor = H_v // H_k
    gdn_recurrence_sm75_launch_on[T](
        ctx,
        state.state_s_dev, q_normed_dev, k_normed_dev,
        v_dev, a_dev, b_dev, a_log, dt_bias, has_params,
        gdn_out_dev, repeat_factor, H_v, D_v, D_k
    )

    # Fused SiLU(z) Gate & Per-Head RMSNorm di VRAM
    gdn_norm_gate_sm75_launch_on[T](
        ctx,
        gdn_out_dev, z_dev, norm_w, norm_w != UnsafePointer[Float32, MutAnyOrigin](),
        H_v, D_v, config.rms_norm_eps
    )

