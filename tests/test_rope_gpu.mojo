# ===----------------------------------------------------------------------=== #
# Module: tests/test_rope_gpu.mojo
# Purpose: Tes diferensial kernel RoPE parsial GPU (sm_75) vs referensi CPU.
#          Paritas eksak dengan MLX rope traditional=False (HALF-SPLIT):
#          pasangan (i, i + rotary_dim/2), freq_i = theta^(-2i/rotary_dim).
#          Meliputi dimensi produksi Bonsai-27B (head_dim=256, rot=64,
#          theta=1e7, Q stride 512 interleaved, K stride 256), posisi besar,
#          pos 0 (identitas), dan keutuhan region non-rotary (tail + gate).
# Jalankan: pixi run mojo run -I . tests/test_rope_gpu.mojo
# ===----------------------------------------------------------------------=== #

from math import cos, sin, exp, log
from memory import UnsafePointer, alloc
from gpu.host import DeviceContext, DeviceBuffer
from src.ops import partial_rope_sm75_launch_on


fn cpu_rope_halfsplit[T: DType](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    dst: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    rotary_dim: Int,
    pos: Int,
    theta_base: Float32,
    stride: Int
):
    """Referensi CPU: formula MLX rope traditional=False, persis per-head."""
    var half_rotary = rotary_dim // 2
    var eff_stride = stride if stride > 0 else head_dim
    for h in range(num_heads):
        var off = h * eff_stride
        for i in range(half_rotary):
            var exponent = -Float32(2 * i) / Float32(rotary_dim)
            var freq = exp(log(theta_base) * exponent)
            var m_theta = Float32(pos) * freq
            var c = cos(m_theta)
            var s = sin(m_theta)
            var x0 = Float32(x[off + i])
            var x1 = Float32(x[off + i + half_rotary])
            dst[off + i] = Scalar[T](x0 * c - x1 * s)
            dst[off + i + half_rotary] = Scalar[T](x0 * s + x1 * c)
        # Region non-rotary dibiarkan utuh (tail per-head; untuk Q interleaved
        # juga region gate [head_dim, eff_stride) per head).
        for j in range(rotary_dim, eff_stride):
            dst[off + j] = x[off + j]


fn run_rope_test[T: DType](
    mut ctx: DeviceContext,
    name: String,
    num_heads: Int,
    head_dim: Int,
    rotary_dim: Int,
    pos: Int,
    theta_base: Float32,
    stride: Int
) raises -> Bool:
    var eff_stride = stride if stride > 0 else head_dim
    var total = num_heads * eff_stride

    var h_x = alloc[Scalar[T]](total)
    var h_want = alloc[Scalar[T]](total)
    for j in range(total):
        # Deterministik pseudo-random, rentang mirip aktivasi [-2, 2)
        h_x[j] = Scalar[T](Float32((j * 7919 + 1013) % 2000) * 0.002 - 2.0)

    cpu_rope_halfsplit[T](
        h_x, h_want, num_heads, head_dim, rotary_dim, pos, theta_base, stride
    )

    var dev_x = ctx.enqueue_create_buffer[T](total)
    ctx.enqueue_copy(dev_x, h_x)

    partial_rope_sm75_launch_on[T](
        ctx, dev_x.unsafe_ptr(), num_heads, head_dim, rotary_dim,
        pos, theta_base, stride
    )

    var h_got = alloc[Scalar[T]](total)
    ctx.enqueue_copy(h_got, dev_x)
    ctx.synchronize()

    var passed = True
    var worst = Float32(0.0)
    var worst_idx = -1
    var half_rotary = rotary_dim // 2
    for h in range(num_heads):
        var off = h * eff_stride
        for j in range(eff_stride):
            var in_rotary_a = (j >= 0) and (j < half_rotary)
            var in_rotary_b = (j >= half_rotary) and (j < rotary_dim)
            var diff = Float32(h_got[off + j]) - Float32(h_want[off + j])
            var adiff = diff if diff >= 0.0 else -diff
            var rotary_elem = in_rotary_a or in_rotary_b
            if rotary_elem:
                # Toleransi FP16: nilai dirotasi ~magnitudo input, angle FP32
                # identik di kedua sisi -> selisih murni rounding penyimpanan.
                if adiff > 0.05:
                    passed = False
                    if adiff > worst:
                        worst = adiff
                        worst_idx = off + j
            else:
                # Region non-rotary WAJIB bitwise identik (tidak tersentuh).
                if h_got[off + j] != h_x[off + j]:
                    passed = False
                    if worst == 0.0:
                        worst = 1.0
                        worst_idx = off + j

    var stat = "[PASS]" if passed else "[GAGAL]"
    print(
        stat, name, "| heads=", num_heads, "hd=", head_dim, "rot=", rotary_dim,
        "pos=", pos, "stride=", stride, "| worst_err=", worst,
        "idx=", worst_idx
    )

    h_x.free()
    h_want.free()
    h_got.free()
    return passed


fn main() raises:
    print("=================================================================")
    print(">> TES DIFERENSIAL KERNEL ROPE GPU vs CPU (paritas MLX half-split)")
    print("=================================================================")
    var ctx = DeviceContext()
    var all_ok = True

    # 1. Kasus kecil head_dim=8 rot=4 — cek aritmetika half-split dasar
    all_ok = run_rope_test[DType.float16](
        ctx, "small_hd8_rot4_pos7", 2, 8, 4, 7, 10000.0, 0
    ) and all_ok

    # 2. Produksi Q: 24 head, head_dim=256, rot=64, theta=1e7, stride 512
    #    (Query + Gate interleaved; region gate wajib utuh)
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_q_h24_pos0", 24, 256, 64, 0, 10000000.0, 512
    ) and all_ok
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_q_h24_pos1", 24, 256, 64, 1, 10000000.0, 512
    ) and all_ok
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_q_h24_pos12345", 24, 256, 64, 12345, 10000000.0, 512
    ) and all_ok
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_q_h24_pos130000", 24, 256, 64, 130000, 10000000.0, 512
    ) and all_ok

    # 3. Produksi K: 4 KV head, stride 256
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_k_h4_pos777", 4, 256, 64, 777, 10000000.0, 256
    ) and all_ok

    # 4. Konteks maksimum model (max_position_embeddings = 262144)
    all_ok = run_rope_test[DType.float16](
        ctx, "prod_k_h4_pos262143", 4, 256, 64, 262143, 10000000.0, 256
    ) and all_ok

    print("-----------------------------------------------------------------")
    if all_ok:
        print(">> HASIL AKHIR: SEMUA KASUS ROPE GPU PASS (paritas MLX half-split)!")
    else:
        print(">> HASIL AKHIR: TERDAPAT KASUS GAGAL PADA ROPE GPU!")
    print("=================================================================")
