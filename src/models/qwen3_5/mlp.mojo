# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/mlp.mojo
# Purpose: SwiGLU Feed-Forward Network (MLP) dengan Bobot Terkuantisasi 1-Bit
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from os import getenv
from time import monotonic
from .norm import silu
from .linear import QwenLinear1Bit


fn mlp_prof_on() -> Bool:
    """Profil tahap MLP aktif bila BONSAI_PROFILE=1."""
    var v = getenv("BONSAI_PROFILE")
    return v and v[0] == "1"

fn qwen3_5_swiglu_mlp_step(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x_norm: UnsafePointer[Float32, MutAnyOrigin],
    mut gate_up_proj: QwenLinear1Bit,
    mut down_proj: QwenLinear1Bit,
    intermediate_size: Int
) raises:
    """
    SwiGLU FFN:
    1. gate_up_proj memproyeksikan x ke [2 * intermediate_size] (N=22016, K=4096)
    2. Aktivasi non-linear: swiglu_act = silu(gate) * up
    3. down_proj memproyeksikan balik ke hidden_size (N=4096, K=11008)
    """
    var gate_up_buf = alloc[Float32](2 * intermediate_size)
    var swiglu_act  = alloc[Float32](intermediate_size)

    gate_up_proj.forward(x_norm, gate_up_buf, 1)

    for i in range(intermediate_size):
        var gate = gate_up_buf[i]
        var up   = gate_up_buf[intermediate_size + i]
        swiglu_act[i] = silu(gate) * up

    down_proj.forward(swiglu_act, out_ptr, 1)

    gate_up_buf.free()
    swiglu_act.free()

fn qwen3_5_swiglu_mlp_step_gpu(
    out_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    x_norm_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    gate_up_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    swiglu_act_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    mut gate_up_proj: QwenLinear1Bit,
    mut down_proj: QwenLinear1Bit,
    intermediate_size: Int,
    pos: Int = 0
) raises:
    """
    SwiGLU FFN 100% di GPU (VRAM-to-VRAM):
    1. gate_up_proj.forward_device(x_norm_dev, gate_up_dev)
    2. swiglu_sm75_launch_on (GPU SFU di register)
    3. down_proj.forward_device(swiglu_act_dev, out_dev)
    """
    alias T = DType.float16
    from src.ops import swiglu_sm75_launch_on
    var prof = mlp_prof_on() and pos == 0
    var t0 = monotonic()
    gate_up_proj.forward_device(x_norm_dev, gate_up_dev, 1)
    var t_gu = 0.0
    if prof:
        gate_up_proj.ctx_ptr[].synchronize()
        t_gu = Float64(monotonic() - t0) / 1e3
    var t1 = monotonic()
    swiglu_sm75_launch_on[T](
        gate_up_proj.ctx_ptr[],
        gate_up_dev, swiglu_act_dev, intermediate_size
    )
    var t_sw = 0.0
    if prof:
        gate_up_proj.ctx_ptr[].synchronize()
        t_sw = Float64(monotonic() - t1) / 1e3
    var t2 = monotonic()
    down_proj.forward_device(swiglu_act_dev, out_dev, 1)
    if prof:
        gate_up_proj.ctx_ptr[].synchronize()
        print(
            "[PROF-MLP] gate_up=", t_gu, "us swiglu=", t_sw,
            "us down=", Float64(monotonic() - t2) / 1e3, "us"
        )

