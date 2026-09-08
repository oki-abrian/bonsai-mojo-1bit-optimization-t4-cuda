# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/layer.mojo
# Purpose: Decoder Layer Utuh Qwen 3.5 / 3.6 (Hybrid 3:1 Interleaving)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from os import getenv
from time import monotonic
from gpu.host import DeviceBuffer
from .config import QwenConfig
from .norm import rms_norm
from .linear import QwenLinear1Bit, DeviceContextGPU
from .gated_delta import GatedDeltaNetState, qwen3_5_gdn_step, qwen3_5_gdn_step_gpu, qwen3_5_gdn_step_gpu_from_proj
from .attention import AttentionKVCache, qwen3_5_gated_attention_step, qwen3_5_gated_attention_step_gpu, qwen3_5_gated_attention_step_gpu_from_proj
from .mlp import qwen3_5_swiglu_mlp_step, qwen3_5_swiglu_mlp_step_gpu
from src.ops import rmsnorm_sm75_launch_on, vec_add_sm75_launch_on, swiglu_sm75_launch_on

struct QwenDecoderLayer:
    """
    Satu lapisan Decoder Transformer Qwen 3.5 / 3.6:
    Mengatur percabangan hybrid 3:1:
    - is_linear == True  -> Mengeksekusi Gated DeltaNet (O(n) linear attention)
    - is_linear == False -> Mengeksekusi Gated Full Attention (GQA 5:1 pada 27B)
    Dilengkapi 2 residual stream dan 2 kali normalisasi RMSNorm.
    """
    var layer_idx: Int
    var is_linear: Bool # True = Gated DeltaNet, False = Gated Attention
    var config: QwenConfig

    # Bobot Proyeksi GDN
    var gdn_in_proj_all: QwenLinear1Bit
    var gdn_conv_weights: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_out_proj: QwenLinear1Bit
    # Parameter GDN riil (A_log, dt_bias, norm berbobot)
    var gdn_a_log: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_dt_bias: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_norm_w: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_has_params: Bool

    # Bobot Proyeksi Gated Attention
    var attn_q_proj: QwenLinear1Bit
    var attn_k_proj: QwenLinear1Bit
    var attn_v_proj: QwenLinear1Bit
    var attn_o_proj: QwenLinear1Bit
    # Bobot Q-Norm & K-Norm per head
    var attn_q_norm_w: UnsafePointer[Float32, MutAnyOrigin]
    var attn_k_norm_w: UnsafePointer[Float32, MutAnyOrigin]
    var attn_has_norms: Bool

    # Bobot SwiGLU MLP
    var mlp_gate_up_proj: QwenLinear1Bit
    var mlp_down_proj: QwenLinear1Bit

    # Bobot Normalisasi RMSNorm
    var input_layernorm_w: UnsafePointer[Float32, MutAnyOrigin]
    var post_attn_layernorm_w: UnsafePointer[Float32, MutAnyOrigin]

    # Pointer & Holder Buffer Bobot di VRAM (100% GPU Resident)
    var input_layernorm_w_dev: UnsafePointer[Float32, MutAnyOrigin]
    var input_layernorm_w_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var post_attn_layernorm_w_dev: UnsafePointer[Float32, MutAnyOrigin]
    var post_attn_layernorm_w_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var gdn_conv_weights_dev: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_conv_weights_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var gdn_a_log_dev: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_a_log_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var gdn_dt_bias_dev: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_dt_bias_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var gdn_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin]
    var gdn_norm_w_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var attn_q_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin]
    var attn_q_norm_w_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var attn_k_norm_w_dev: UnsafePointer[Float32, MutAnyOrigin]
    var attn_k_norm_w_buf: UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]
    var dev_weights_ready: Bool

    fn __init__(
        out self,
        layer_idx: Int,
        is_linear: Bool,
        config: QwenConfig
    ):
        """Konstruksi kerangka layer; bobot diisi terpisah oleh loader."""
        self.layer_idx = layer_idx
        self.is_linear = is_linear
        self.config = config
        self.gdn_in_proj_all = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.gdn_conv_weights = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_out_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.attn_q_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.attn_k_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.attn_v_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.attn_o_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.mlp_gate_up_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.mlp_down_proj = QwenLinear1Bit(UnsafePointer[UInt8, MutAnyOrigin](), UnsafePointer[Float32, MutAnyOrigin](), 0, 0)
        self.input_layernorm_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.post_attn_layernorm_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_a_log = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_dt_bias = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_norm_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_has_params = False
        self.attn_q_norm_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.attn_k_norm_w = UnsafePointer[Float32, MutAnyOrigin]()
        self.attn_has_norms = False

        self.input_layernorm_w_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.input_layernorm_w_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.post_attn_layernorm_w_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.post_attn_layernorm_w_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.gdn_conv_weights_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_conv_weights_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.gdn_a_log_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_a_log_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.gdn_dt_bias_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_dt_bias_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.gdn_norm_w_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.gdn_norm_w_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.attn_q_norm_w_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.attn_q_norm_w_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.attn_k_norm_w_dev = UnsafePointer[Float32, MutAnyOrigin]()
        self.attn_k_norm_w_buf = UnsafePointer[DeviceBuffer[DType.float32], MutAnyOrigin]()
        self.dev_weights_ready = False

    fn set_ctx(
        mut self,
        ctx: UnsafePointer[DeviceContextGPU, MutAnyOrigin]
    ) raises:
        """Pasang DeviceContext bersama ke semua proyeksi linear di layer ini dan upload bobot ke VRAM."""
        if not ctx:
            return
        if self.dev_weights_ready:
            return

        if self.is_linear:
            self.gdn_in_proj_all.set_ctx(ctx)
            self.gdn_out_proj.set_ctx(ctx)
        else:
            self.attn_q_proj.set_ctx(ctx)
            self.attn_k_proj.set_ctx(ctx)
            self.attn_v_proj.set_ctx(ctx)
            self.attn_o_proj.set_ctx(ctx)
        self.mlp_gate_up_proj.set_ctx(ctx)
        self.mlp_down_proj.set_ctx(ctx)

        # Upload layer norm weights ke VRAM persisten
        if self.input_layernorm_w != UnsafePointer[Float32, MutAnyOrigin]():
            var h = alloc[DeviceBuffer[DType.float32]](1)
            h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.hidden_size))
            ctx[].enqueue_copy(h[], self.input_layernorm_w)
            self.input_layernorm_w_buf = h
            self.input_layernorm_w_dev = h[].unsafe_ptr()

        if self.post_attn_layernorm_w != UnsafePointer[Float32, MutAnyOrigin]():
            var h = alloc[DeviceBuffer[DType.float32]](1)
            h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.hidden_size))
            ctx[].enqueue_copy(h[], self.post_attn_layernorm_w)
            self.post_attn_layernorm_w_buf = h
            self.post_attn_layernorm_w_dev = h[].unsafe_ptr()

        # Upload GDN weights ke VRAM persisten
        if self.is_linear:
            if self.gdn_conv_weights != UnsafePointer[Float32, MutAnyOrigin]():
                var numel = self.config.gdn_conv_dim * self.config.gdn_conv_kernel
                var h = alloc[DeviceBuffer[DType.float32]](1)
                h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](numel))
                ctx[].enqueue_copy(h[], self.gdn_conv_weights)
                self.gdn_conv_weights_buf = h
                self.gdn_conv_weights_dev = h[].unsafe_ptr()

            if self.gdn_has_params:
                if self.gdn_a_log != UnsafePointer[Float32, MutAnyOrigin]():
                    var h = alloc[DeviceBuffer[DType.float32]](1)
                    h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.gdn_num_v_heads))
                    ctx[].enqueue_copy(h[], self.gdn_a_log)
                    self.gdn_a_log_buf = h
                    self.gdn_a_log_dev = h[].unsafe_ptr()

                if self.gdn_dt_bias != UnsafePointer[Float32, MutAnyOrigin]():
                    var h = alloc[DeviceBuffer[DType.float32]](1)
                    h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.gdn_num_v_heads))
                    ctx[].enqueue_copy(h[], self.gdn_dt_bias)
                    self.gdn_dt_bias_buf = h
                    self.gdn_dt_bias_dev = h[].unsafe_ptr()

                if self.gdn_norm_w != UnsafePointer[Float32, MutAnyOrigin]():
                    var h = alloc[DeviceBuffer[DType.float32]](1)
                    h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.gdn_head_v_dim))
                    ctx[].enqueue_copy(h[], self.gdn_norm_w)
                    self.gdn_norm_w_buf = h
                    self.gdn_norm_w_dev = h[].unsafe_ptr()
        else:
            # Upload Attention norm weights ke VRAM persisten
            if self.attn_has_norms:
                if self.attn_q_norm_w != UnsafePointer[Float32, MutAnyOrigin]():
                    var h = alloc[DeviceBuffer[DType.float32]](1)
                    h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.head_dim))
                    ctx[].enqueue_copy(h[], self.attn_q_norm_w)
                    self.attn_q_norm_w_buf = h
                    self.attn_q_norm_w_dev = h[].unsafe_ptr()

                if self.attn_k_norm_w != UnsafePointer[Float32, MutAnyOrigin]():
                    var h = alloc[DeviceBuffer[DType.float32]](1)
                    h.init_pointee_move(ctx[].enqueue_create_buffer[DType.float32](self.config.head_dim))
                    ctx[].enqueue_copy(h[], self.attn_k_norm_w)
                    self.attn_k_norm_w_buf = h
                    self.attn_k_norm_w_dev = h[].unsafe_ptr()

        self.dev_weights_ready = True

    fn forward(
        mut self,
        hidden_states: UnsafePointer[Float32, MutAnyOrigin],
        mut gdn_state: GatedDeltaNetState,
        mut kv_cache: AttentionKVCache,
        pos: Int
    ) raises:
        """
        Forward pass 1 token penuh pada decoder layer:
        1. Pre-Attention RMSNorm
        2. Gated DeltaNet (jika is_linear) ATAU Gated Attention (jika interval ke-4)
        3. Residual 1: x = x + sublayer_out
        4. Post-Attention RMSNorm
        5. SwiGLU MLP
        6. Residual 2: x = x + mlp_out
        """
        var D = self.config.hidden_size
        var x_norm = alloc[Float32](D)
        var sublayer_out = alloc[Float32](D)
        var mlp_out = alloc[Float32](D)

        # 1. Pre-Layer RMSNorm
        rms_norm(x_norm, hidden_states, self.input_layernorm_w, 1, D, self.config.rms_norm_eps)

        # 2. Hybrid Execution
        if self.is_linear:
            qwen3_5_gdn_step(
                sublayer_out, x_norm,
                self.gdn_in_proj_all, self.gdn_conv_weights, self.gdn_out_proj,
                self.gdn_a_log, self.gdn_dt_bias, self.gdn_norm_w,
                self.gdn_has_params,
                gdn_state, self.config
            )
        else:
            qwen3_5_gated_attention_step(
                sublayer_out, x_norm,
                self.attn_q_proj, self.attn_k_proj, self.attn_v_proj, self.attn_o_proj,
                self.attn_q_norm_w, self.attn_k_norm_w, self.attn_has_norms,
                kv_cache, pos, self.config
            )

        # 3. Residual 1
        for d in range(D):
            hidden_states[d] += sublayer_out[d]

        # 4. Post-Attention RMSNorm
        rms_norm(x_norm, hidden_states, self.post_attn_layernorm_w, 1, D, self.config.rms_norm_eps)

        # 5. SwiGLU MLP
        qwen3_5_swiglu_mlp_step(
            mlp_out, x_norm,
            self.mlp_gate_up_proj, self.mlp_down_proj,
            self.config.intermediate_size
        )

        # 6. Residual 2
        for d in range(D):
            hidden_states[d] += mlp_out[d]

        x_norm.free()
        sublayer_out.free()
        mlp_out.free()

    fn forward_gpu(
        mut self,
        hidden_states_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        x_norm_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        sublayer_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        mlp_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        proj_raw_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        conv_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        q_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        k_normed_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        gdn_out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        gate_up_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        swiglu_act_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        attn_scores_dev: UnsafePointer[Float32, MutAnyOrigin],
        mut gdn_state: GatedDeltaNetState,
        mut kv_cache: AttentionKVCache,
        pos: Int
    ) raises:
        """
        Forward pass 1 token penuh 100% di GPU (VRAM resident):
        1. Pre-Layer RMSNorm di GPU (rmsnorm_sm75_launch_on)
        2. GDN Step di GPU (qwen3_5_gdn_step_gpu) atau Attention Step (qwen3_5_gated_attention_step_gpu)
        3. Residual 1 di GPU (vec_add_sm75_launch_on)
        4. Post-Attention RMSNorm di GPU (rmsnorm_sm75_launch_on)
        5. SwiGLU MLP di GPU (qwen3_5_swiglu_mlp_step_gpu)
        6. Residual 2 di GPU (vec_add_sm75_launch_on)
        """
        alias T = DType.float16
        var D = self.config.hidden_size
        var ctx_ptr = self.mlp_gate_up_proj.ctx_ptr
        if not ctx_ptr:
            raise Error("FATAL: ctx_ptr null pada QwenDecoderLayer.forward_gpu!")
        var ctx = ctx_ptr[]
        var prof = pos == 0
        var pv = getenv("BONSAI_PROFILE")
        prof = prof and pv and pv[0] == "1"

        # 1. Pre-Layer RMSNorm di GPU (menggunakan pointer bobot VRAM)
        var t0 = monotonic()
        rmsnorm_sm75_launch_on[T](
            ctx, hidden_states_dev, x_norm_dev,
            self.input_layernorm_w_dev, self.input_layernorm_w_dev != UnsafePointer[Float32, MutAnyOrigin](),
            D, self.config.rms_norm_eps
        )
        var t_pren = 0.0
        if prof:
            ctx.synchronize()
            t_pren = Float64(monotonic() - t0) / 1e3

        # 2. Hybrid Execution (100% VRAM Resident)
        var t1 = monotonic()
        if self.is_linear:
            qwen3_5_gdn_step_gpu(
                sublayer_out_dev, x_norm_dev,
                proj_raw_dev, conv_out_dev, q_normed_dev, k_normed_dev, gdn_out_dev,
                self.gdn_in_proj_all, self.gdn_conv_weights_dev, self.gdn_out_proj,
                self.gdn_a_log_dev, self.gdn_dt_bias_dev, self.gdn_norm_w_dev,
                self.gdn_has_params,
                gdn_state, self.config, pos
            )
        else:
            # Gated Full Attention di VRAM (tanpa fallback CPU, tanpa PCIe copy)
            var q_gate_dev = proj_raw_dev
            var k_dev = conv_out_dev
            var v_dev = conv_out_dev.offset(self.config.num_key_value_heads * self.config.head_dim)
            var attn_out_dev = gdn_out_dev

            qwen3_5_gated_attention_step_gpu(
                sublayer_out_dev, x_norm_dev,
                q_gate_dev, k_dev, v_dev, attn_out_dev, attn_scores_dev,
                self.attn_q_proj, self.attn_k_proj, self.attn_v_proj, self.attn_o_proj,
                self.attn_q_norm_w_dev, self.attn_k_norm_w_dev, self.attn_has_norms,
                kv_cache, pos, self.config
            )
        var t_step = 0.0
        if prof:
            ctx.synchronize()
            t_step = Float64(monotonic() - t1) / 1e3

        # 3. Residual 1 di GPU
        var t2 = monotonic()
        vec_add_sm75_launch_on[T](ctx, hidden_states_dev, sublayer_out_dev, D)
        var t_r1 = 0.0
        if prof:
            ctx.synchronize()
            t_r1 = Float64(monotonic() - t2) / 1e3

        # 4. Post-Attention RMSNorm di GPU (menggunakan pointer bobot VRAM)
        var t3 = monotonic()
        rmsnorm_sm75_launch_on[T](
            ctx, hidden_states_dev, x_norm_dev,
            self.post_attn_layernorm_w_dev, self.post_attn_layernorm_w_dev != UnsafePointer[Float32, MutAnyOrigin](),
            D, self.config.rms_norm_eps
        )
        var t_post = 0.0
        if prof:
            ctx.synchronize()
            t_post = Float64(monotonic() - t3) / 1e3

        # 5. SwiGLU MLP di GPU
        var t4 = monotonic()
        qwen3_5_swiglu_mlp_step_gpu(
            mlp_out_dev, x_norm_dev, gate_up_dev, swiglu_act_dev,
            self.mlp_gate_up_proj, self.mlp_down_proj,
            self.config.intermediate_size, pos
        )
        var t_mlp = 0.0
        if prof:
            ctx.synchronize()
            t_mlp = Float64(monotonic() - t4) / 1e3

        # 6. Residual 2 di GPU
        var t5 = monotonic()
        vec_add_sm75_launch_on[T](ctx, hidden_states_dev, mlp_out_dev, D)
        if prof:
            ctx.synchronize()
            print(
                "[PROF-LAYER] prenorm=", t_pren, "us step=", t_step,
                "us resid1=", t_r1, "us postnorm=", t_post, "us mlp=",
                t_mlp, "us resid2=", Float64(monotonic() - t5) / 1e3, "us"
            )

    fn prefill_ffi_ready(self) -> Bool:
        """True bila kernel prefill batched qmm (WMMA v2) siap di layer ini."""
        if self.is_linear:
            return self.gdn_in_proj_all.prefill_ffi_ready()
        return self.attn_q_proj.prefill_ffi_ready()

    fn forward_prefill_gpu(
        mut self,
        hidden_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        x_norm_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        sublayer_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        mlp_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        proj_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        conv_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        qn_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        kn_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        gdn_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        gate_up_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        swiglu_m_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
        attn_scores_dev: UnsafePointer[Float32, MutAnyOrigin],
        mut gdn_state: GatedDeltaNetState,
        mut kv_cache: AttentionKVCache,
        pos_base: Int,
        M: Int
    ) raises:
        """
        Prefill BATCHED M-token 100% VRAM (paritas jalur MLX qmm WMMA v2):
        seluruh proyeksi berat (in/out_proj, q/k/v/o, gate_up, down) dijalankan
        sebagai GEMM M-token lewat forward_prefill_device — bobot dibaca SEKALI
        per chunk, bukan M kali. Operasi stateful per-token (conv window,
        rekurensi delta-rule, RoPE/append/GQA dengan pos absolut) tetap loop
        sekuensial di antara GEMM — komputasinya kecil, bukan bottleneck DRAM.
        Semua kernel di-enqueue asinkron; tidak ada sync host di dalam layer.
        """
        alias T = DType.float16
        var D = self.config.hidden_size
        var ctx_ptr = self.mlp_gate_up_proj.ctx_ptr
        if not ctx_ptr:
            raise Error("FATAL: ctx_ptr null pada QwenDecoderLayer.forward_prefill_gpu!")
        var ctx = ctx_ptr[]

        if self.is_linear:
            # ---------- GDN layer ----------
            var in_proj_n = self.gdn_in_proj_all.N
            var conv_dim = self.config.gdn_conv_dim
            var v_dim = self.config.gdn_num_v_heads * self.config.gdn_head_v_dim
            var qk_dim = self.config.gdn_num_k_heads * self.config.gdn_head_k_dim

            # 1. Pre-Layer RMSNorm per baris
            for t in range(M):
                rmsnorm_sm75_launch_on[T](
                    ctx, hidden_m_dev.offset(t * D), x_norm_m_dev.offset(t * D),
                    self.input_layernorm_w_dev,
                    self.input_layernorm_w_dev != UnsafePointer[Float32, MutAnyOrigin](),
                    D, self.config.rms_norm_eps
                )

            # 2. in_proj GEMM batched (bobot 16480x5120 dibaca SEKALI)
            self.gdn_in_proj_all.forward_prefill_device(x_norm_m_dev, proj_m_dev, M)

            # 3. Per-token stateful: conv -> q/k norm -> rekurensi -> gate
            for t in range(M):
                qwen3_5_gdn_step_gpu_from_proj(
                    ctx_ptr,
                    proj_m_dev.offset(t * in_proj_n),
                    conv_m_dev.offset(t * conv_dim),
                    qn_m_dev.offset(t * qk_dim),
                    kn_m_dev.offset(t * qk_dim),
                    gdn_m_dev.offset(t * v_dim),
                    self.gdn_conv_weights_dev,
                    self.gdn_a_log_dev, self.gdn_dt_bias_dev, self.gdn_norm_w_dev,
                    self.gdn_has_params,
                    gdn_state, self.config, pos_base + t
                )

            # 4. out_proj GEMM batched
            self.gdn_out_proj.forward_prefill_device(gdn_m_dev, sublayer_m_dev, M)
        else:
            # ---------- Attention layer ----------
            var H_q = self.config.num_attention_heads
            var H_kv = self.config.num_key_value_heads
            var Dh = self.config.head_dim
            var q_n = H_q * 2 * Dh         # q+gate interleaved (stride 2D per head)
            var kv_dim = H_kv * Dh
            # PENTING: attn out per token selebar H_q*Dh = 6144 (o_proj K=6144),
            # BUKAN hidden_size=5120 — stride salah membuat baris t menimpa
            # baris t+1 dan merusak hidden state (bug Run AM/AN).
            var attn_out_stride = H_q * Dh

            # 1. Pre-Layer RMSNorm per baris
            for t in range(M):
                rmsnorm_sm75_launch_on[T](
                    ctx, hidden_m_dev.offset(t * D), x_norm_m_dev.offset(t * D),
                    self.input_layernorm_w_dev,
                    self.input_layernorm_w_dev != UnsafePointer[Float32, MutAnyOrigin](),
                    D, self.config.rms_norm_eps
                )

            # 2. q/k/v GEMM batched (3 bobot masing-masing dibaca SEKALI)
            self.attn_q_proj.forward_prefill_device(x_norm_m_dev, proj_m_dev, M)
            self.attn_k_proj.forward_prefill_device(x_norm_m_dev, conv_m_dev, M)
            # v staging di kn_m_dev [M, kv_dim] (qn/kn tidak dipakai layer attention)
            self.attn_v_proj.forward_prefill_device(x_norm_m_dev, kn_m_dev, M)

            # 3. Per-token stateful: q/k norm -> RoPE(pos) -> append KV -> GQA
            #    attn out ditulis ke gdn_m_dev rows (stride H_q*Dh = 6144)
            for t in range(M):
                qwen3_5_gated_attention_step_gpu_from_proj(
                    ctx_ptr,
                    proj_m_dev.offset(t * q_n),
                    conv_m_dev.offset(t * kv_dim),
                    kn_m_dev.offset(t * kv_dim),
                    gdn_m_dev.offset(t * attn_out_stride),
                    attn_scores_dev,
                    self.attn_q_norm_w_dev, self.attn_k_norm_w_dev,
                    self.attn_has_norms,
                    kv_cache, pos_base + t, self.config
                )

            # 4. o_proj GEMM batched: attn out (gdn_m rows, [M,6144]) -> sublayer_m
            self.attn_o_proj.forward_prefill_device(gdn_m_dev, sublayer_m_dev, M)

        # 5. Residual 1 batched per baris
        for t in range(M):
            vec_add_sm75_launch_on[T](ctx, hidden_m_dev.offset(t * D), sublayer_m_dev.offset(t * D), D)

        # 6. Post-Attention RMSNorm per baris
        for t in range(M):
            rmsnorm_sm75_launch_on[T](
                ctx, hidden_m_dev.offset(t * D), x_norm_m_dev.offset(t * D),
                self.post_attn_layernorm_w_dev,
                self.post_attn_layernorm_w_dev != UnsafePointer[Float32, MutAnyOrigin](),
                D, self.config.rms_norm_eps
            )

        # 7. SwiGLU MLP batched: gate_up GEMM -> swiglu per baris -> down GEMM
        self.mlp_gate_up_proj.forward_prefill_device(x_norm_m_dev, gate_up_m_dev, M)
        for t in range(M):
            swiglu_sm75_launch_on[T](
                ctx, gate_up_m_dev.offset(t * 2 * self.config.intermediate_size),
                swiglu_m_dev.offset(t * self.config.intermediate_size),
                self.config.intermediate_size
            )
        self.mlp_down_proj.forward_prefill_device(swiglu_m_dev, mlp_m_dev, M)

        # 8. Residual 2 batched per baris
        for t in range(M):
            vec_add_sm75_launch_on[T](ctx, hidden_m_dev.offset(t * D), mlp_m_dev.offset(t * D), D)


