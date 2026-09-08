# ===----------------------------------------------------------------------=== #
# Module: tests/test_argmax_gpu.mojo
# Purpose: Tes diferensial kernel Argmax GPU 2-stage (sm_75) vs referensi CPU.
#          Menguji ukuran riil Vocab Bonsai-27B (V = 248,320), edge cases (peak
#          di indeks 0, V-1, acak, monotonic), dan tie-breaking.
# Jalankan: pixi run mojo run -I . tests/test_argmax_gpu.mojo
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from gpu.host import DeviceContext, DeviceBuffer
from src.ops import argmax_sm75_launch_on


fn cpu_argmax[T: DType](data: UnsafePointer[Scalar[T], MutAnyOrigin], V: Int) -> Int:
    var best_val = Float32(data[0])
    var best_idx = 0
    for i in range(1, V):
        var v = Float32(data[i])
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx


fn run_argmax_test[T: DType](
    mut ctx: DeviceContext,
    name: String,
    V: Int,
    peak_idx: Int,
    pattern: Int # 0=flat with 1 peak, 1=monotonic ascending, 2=pseudo-random
) raises -> Bool:
    var h_logits = alloc[Scalar[T]](V)
    for i in range(V):
        if pattern == 1:
            h_logits[i] = Scalar[T](Float32(i) * 0.001)
        elif pattern == 2:
            # Pseudo-random logits di rentang -10.0 s/d +10.0
            var val = Float32((i * 7919 + 1013) % 20000) * 0.001 - 10.0
            h_logits[i] = Scalar[T](val)
        else:
            h_logits[i] = Scalar[T](0.0)

    if peak_idx >= 0 and peak_idx < V:
        h_logits[peak_idx] = Scalar[T](100.0)

    var want_idx = cpu_argmax[T](h_logits, V)

    # Alokasi VRAM GPU
    var dev_logits = ctx.enqueue_create_buffer[T](V)
    var dev_bvals = ctx.enqueue_create_buffer[DType.float32](256)
    var dev_bidxs = ctx.enqueue_create_buffer[DType.int32](256)
    var dev_out = ctx.enqueue_create_buffer[DType.int32](1)

    ctx.enqueue_copy(dev_logits, h_logits)

    # Jalankan kernel reduksi 2-stage GPU
    argmax_sm75_launch_on[T](
        ctx, dev_logits.unsafe_ptr(), dev_bvals.unsafe_ptr(),
        dev_bidxs.unsafe_ptr(), dev_out.unsafe_ptr(), V
    )

    var h_out = alloc[Int32](1)
    ctx.enqueue_copy(h_out, dev_out)
    ctx.synchronize()

    var got_idx = Int(h_out[0])
    var passed = (got_idx == want_idx)

    var stat = "[PASS]" if passed else "[GAGAL]"
    print(stat, name, "| V=", V, "| got=", got_idx, "want=", want_idx)

    h_logits.free()
    h_out.free()
    return passed


fn main() raises:
    print("=================================================================")
    print(">> TES DIFERENSIAL KERNEL ARGMAX GPU 2-STAGE vs CPU")
    print("=================================================================")
    var ctx = DeviceContext()
    var all_ok = True

    # 1. Vocab kecil V=256
    all_ok = run_argmax_test[DType.float16](ctx, "small_v256_peak_first", 256, 0, 0) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "small_v256_peak_last", 256, 255, 0) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "small_v256_peak_mid", 256, 128, 0) and all_ok

    # 2. Vocab sedang V=4096
    all_ok = run_argmax_test[DType.float16](ctx, "mid_v4096_monotonic", 4096, -1, 1) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "mid_v4096_random", 4096, 2048, 2) and all_ok

    # 3. Vocab Bonsai-27B riil: V = 248,320
    all_ok = run_argmax_test[DType.float16](ctx, "bonsai_v248320_peak_0", 248320, 0, 0) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "bonsai_v248320_peak_last", 248320, 248319, 0) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "bonsai_v248320_peak_eos_248046", 248320, 248046, 0) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "bonsai_v248320_random", 248320, 151643, 2) and all_ok
    all_ok = run_argmax_test[DType.float16](ctx, "bonsai_v248320_monotonic", 248320, -1, 1) and all_ok

    print("-----------------------------------------------------------------")
    if all_ok:
        print(">> HASIL AKHIR: 10/10 KASUS ARGMAX GPU 2-STAGE PASS!")
    else:
        print(">> HASIL AKHIR: TERDAPAT KASUS GAGAL PADA ARGMAX GPU!")
    print("=================================================================")
