# ===----------------------------------------------------------------------=== #
# Module: tests/selftest_decode_gpu.mojo
# Purpose: DEBUG DIFERENSIAL kernel GPU ASLI qmv_sm75_b1_decode_gpu (T4)
#          vs Referensi FP64 CPU. Host-sim (qmv_sm75_b1_decode_body) TIDAK
#          mencakup jalur ini — selftest lama tidak pernah memvalidasi
#          kernel device. Setiap kasus mencetak PASS/FAIL + max_err +
#          koordinat mismatch pertama (n, got, want) agar lokasi bug
#          terlihat dari data, bukan tebakan.
# Jalankan: pixi run mojo run -I . tests/selftest_decode_gpu.mojo  (butuh GPU)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt
from time import monotonic
from os import getenv
from sys.ffi import OwnedDLHandle
from gpu.host import DeviceContext, DeviceBuffer
from src.common import (
    get_scale_stride, get_weight_row_bytes, decode_split_plan
)
from src.ops import (
    qmv_sm75_1bit_launch_on, CudaQmvDecodeFnFP16, try_open_cuda_lib, dummy_cuda_decode_fp16,
    CudaSyncDeviceFn, dummy_cuda_sync_device
)


@fieldwise_init
struct GpuCase(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var N: Int
    var K: Int
    var pattern: Int   # 0=random, 1=all_zeros(w=-s), 2=all_ones(w=+s)
    var bcast: Bool


fn run_case(ctx: DeviceContext, c: GpuCase) raises -> Bool:
    var M: Int = 1
    var L: Int = 1
    var groups = get_scale_stride(c.K)
    var row_bytes = get_weight_row_bytes(c.K)
    var szx = M * c.K
    var szw = c.N * row_bytes
    var szs = c.N * groups
    var szo = M * c.N

    # ---- 1. Data host deterministik ----
    var hx32 = alloc[Float32](szx)
    var hx16 = alloc[Scalar[DType.float16]](szx)
    var hw = alloc[UInt8](szw)
    var hs32 = alloc[Float32](szs)
    var hs16 = alloc[Scalar[DType.float16]](szs)
    var hy16 = alloc[Scalar[DType.float16]](szo)
    var href = alloc[Float64](szo)

    for i in range(szx):
        var v = Float32((i % 17) - 8) * 0.125
        hx32[i] = v
        hx16[i] = Scalar[DType.float16](v)
    for i in range(szs):
        var v = Float32(1.0 + Float32(i % 5) * 0.1)
        hs32[i] = v
        hs16[i] = Scalar[DType.float16](v)
    for i in range(szw):
        if c.pattern == 1:
            hw[i] = 0x00
        elif c.pattern == 2:
            hw[i] = 0xFF
        else:
            hw[i] = UInt8((i * 37 + 13) & 0xFF)
    for i in range(szo):
        hy16[i] = Scalar[DType.float16](0)
        href[i] = 0.0

    # ---- 2. Referensi FP64 CPU (kontrak: w_eff = (2*bit-1)*s, bias = -s) ----
    for n in range(c.N):
        var acc: Float64 = 0.0
        for k in range(c.K):
            var xv = hx16[k].cast[DType.float64]()
            var byte_val = hw[n * row_bytes + (k // 8)]
            var bit = Float64((Int(byte_val) >> (k % 8)) & 1)
            var sg = hs16[n * groups + (k // 128)].cast[DType.float64]()
            acc += (2.0 * bit - 1.0) * sg * xv
        href[n] = acc

    # ---- 3. Jalur GPU ASLI (paritas produksi) ----
    var bx_buf = ctx.enqueue_create_buffer[DType.float16](szx)
    var bw_buf = ctx.enqueue_create_buffer[DType.uint8](szw)
    var bs_buf = ctx.enqueue_create_buffer[DType.float16](szs)
    var by_buf = ctx.enqueue_create_buffer[DType.float16](szo)
    ctx.enqueue_copy(bx_buf, hx16)
    ctx.enqueue_copy(bw_buf, hw)
    ctx.enqueue_copy(bs_buf, hs16)
    ctx.enqueue_copy(by_buf, hy16)

    var splits = decode_split_plan(c.N, L, groups)
    var ws_buf: DeviceBuffer[DType.float32]
    var ws_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    if splits > 1:
        ws_buf = ctx.enqueue_create_buffer[DType.float32](splits * L * M * c.N)
        ws_ptr = ws_buf.unsafe_ptr()

    qmv_sm75_1bit_launch_on[DType.float16](
        ctx, bx_buf.unsafe_ptr(), bw_buf.unsafe_ptr(), bs_buf.unsafe_ptr(),
        by_buf.unsafe_ptr(), M, c.N, c.K, L, c.bcast, ws_ptr
    )
    ctx.synchronize()
    ctx.enqueue_copy(hy16, by_buf)

    # ---- 4. Bandingkan: max_err + mismatch pertama ----
    var eps: Float64 = 4.9e-4
    var atol: Float64 = 8.0 * eps * sqrt(Float64(c.K)) * 1.5 * 2.0
    var max_err: Float64 = 0.0
    var bad_n = -1
    var got_v: Float64 = 0.0
    var want_v: Float64 = 0.0
    var max_href: Float64 = 1e-9
    for i in range(szo):
        var got = hy16[i].cast[DType.float64]()
        var diff = abs(got - href[i])
        if abs(href[i]) > max_href:
            max_href = abs(href[i])
        if diff > max_err:
            max_err = diff
        if diff > atol and bad_n == -1:
            bad_n = i
            got_v = got
            want_v = href[i]

    var passed = max_err <= atol
    var stat = "[PASS]" if passed else "[GAGAL]"
    print(stat, c.name, "| N=", c.N, "K=", c.K, "splits=", splits,
          "bcast=", c.bcast, "| max_err=", max_err, "atol=", atol,
          "| rel_err=", max_err / max_href)
    if bad_n != -1:
        print("   >> MISMATCH PERTAMA n=", bad_n, "got=", got_v,
              "want=", want_v, "selisih=", got_v - want_v)

    hx32.free()
    hx16.free()
    hw.free()
    hs32.free()
    hs16.free()
    hy16.free()
    href.free()
    return passed


fn bench_case(ctx: DeviceContext, name: String, N: Int, K: Int, iters: Int) raises:
    """Bench kernel decode ASLI dalam isolasi (tanpa pipeline model):
    ukuran produksi, warmup, lalu rata-rata per iterasi -> GB/s & GMAC/s.
    Ini penentu: kalau di isolasi ~60 GB/s (seperti referensi), masalahnya
    di level pipeline/launch; kalau tetap ~30 GB/s, masalahnya di kernel."""
    var M: Int = 1
    var L: Int = 1
    var groups = get_scale_stride(K)
    var row_bytes = get_weight_row_bytes(K)
    var szx = M * K
    var szw = N * row_bytes
    var szs = N * groups
    var szo = M * N

    var hx16 = alloc[Scalar[DType.float16]](szx)
    var hw = alloc[UInt8](szw)
    var hs16 = alloc[Scalar[DType.float16]](szs)
    var hy16 = alloc[Scalar[DType.float16]](szo)
    for i in range(szx):
        hx16[i] = Scalar[DType.float16](Float32((i % 17) - 8) * 0.125)
    for i in range(szs):
        hs16[i] = Scalar[DType.float16](Float32(1.0 + Float32(i % 5) * 0.1))
    for i in range(szw):
        hw[i] = UInt8((i * 37 + 13) & 0xFF)

    var bx = ctx.enqueue_create_buffer[DType.float16](szx)
    var bw = ctx.enqueue_create_buffer[DType.uint8](szw)
    var bs = ctx.enqueue_create_buffer[DType.float16](szs)
    var by = ctx.enqueue_create_buffer[DType.float16](szo)
    ctx.enqueue_copy(bx, hx16)
    ctx.enqueue_copy(bw, hw)
    ctx.enqueue_copy(bs, hs16)
    ctx.synchronize()

    var splits = decode_split_plan(N, L, groups)
    var ws_ptr = UnsafePointer[Float32, MutAnyOrigin]()
    var ws_buf: DeviceBuffer[DType.float32]
    if splits > 1:
        ws_buf = ctx.enqueue_create_buffer[DType.float32](splits * L * M * N)
        ws_ptr = ws_buf.unsafe_ptr()

    var ffi_ready = False
    var h_buf = alloc[OwnedDLHandle](1)
    var ffi_fn = dummy_cuda_decode_fp16
    var ffi_sync_fn = dummy_cuda_sync_device
    var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
    if not (disable and (disable == "1" or disable == "true")):
        try:
            h_buf.init_pointee_move(try_open_cuda_lib())
            ffi_fn = h_buf[].get_function[CudaQmvDecodeFnFP16]("launch_qmv_sm75_b1_decode_fp16")
            try:
                ffi_sync_fn = h_buf[].get_function[CudaSyncDeviceFn]("qmv_sm75_device_synchronize")
            except:
                pass
            ffi_ready = True
        except:
            ffi_ready = False

    for _ in range(5):
        if ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            _ = ffi_fn(
                bx.unsafe_ptr(), bw.unsafe_ptr(), bs.unsafe_ptr(),
                by.unsafe_ptr(), ws_ptr,
                Int32(M), Int32(N), Int32(K), Int32(L),
                Int32(0), Int32(splits), cuda_stream
            )
        else:
            qmv_sm75_1bit_launch_on[DType.float16](
                ctx, bx.unsafe_ptr(), bw.unsafe_ptr(), bs.unsafe_ptr(),
                by.unsafe_ptr(), M, N, K, L, False, ws_ptr
            )
    if ffi_ready:
        _ = ffi_sync_fn()
    else:
        ctx.synchronize()

    var t0 = monotonic()
    for _ in range(iters):
        if ffi_ready:
            var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
            _ = ffi_fn(
                bx.unsafe_ptr(), bw.unsafe_ptr(), bs.unsafe_ptr(),
                by.unsafe_ptr(), ws_ptr,
                Int32(M), Int32(N), Int32(K), Int32(L),
                Int32(0), Int32(splits), cuda_stream
            )
        else:
            qmv_sm75_1bit_launch_on[DType.float16](
                ctx, bx.unsafe_ptr(), bw.unsafe_ptr(), bs.unsafe_ptr(),
                by.unsafe_ptr(), M, N, K, L, False, ws_ptr
            )
    if ffi_ready:
        _ = ffi_sync_fn()
    else:
        ctx.synchronize()
    var ms = Float64(monotonic() - t0) / 1e6 / Float64(iters)

    if ffi_ready:
        h_buf.destroy_pointee()
        h_buf.free()

    var bytes = Float64(szw) + Float64(szx) * 2.0 + Float64(szs) * 2.0 + Float64(szo) * 2.0
    var gbps = bytes / (ms * 1e-3) / 1e9
    var gmacs = Float64(N) * Float64(K) / (ms * 1e-3) / 1e9
    var target_badge = " [TARGET >= 60 GB/s TERCAPAI!]" if gbps >= 60.0 else " [DI BAWAH 60 GB/s]"
    print("[BENCH]", name, "| N=", N, "K=", K, "splits=", splits,
          "|", ms, "ms/iter |", gbps, "GB/s", target_badge, "|", gmacs, "GMAC/s")

    hx16.free()
    hw.free()
    hs16.free()
    hy16.free()


fn main() raises:
    print("=================================================================")
    print(">> DEBUG DIFERENSIAL KERNEL GPU DECODE (bukan host-sim)")
    print("=================================================================")
    var ctx = DeviceContext()

    var cases = List[GpuCase]()
    # 1 tile, 1 block: staging + build + dot + reduce minimal
    cases.append(GpuCase("n32_k1024_1blk_1tile", 32, 1024, 0, False))
    # Multi-tile K (decode riil K=5120 = 5 tile)
    cases.append(GpuCase("n32_k5120_multitile", 32, 5120, 0, False))
    cases.append(GpuCase("n32_k6144_multitile", 32, 6144, 0, False))
    # Multi-block
    cases.append(GpuCase("n64_k1024_2blk", 64, 1024, 0, False))
    # Split-K workspace + reduce kernel
    cases.append(GpuCase("n160_k5120_splitk", 160, 5120, 0, False))
    cases.append(GpuCase("n1024_k6144_splitk", 1024, 6144, 0, False))
    # Ukuran produksi gate_up (tanpa split)
    cases.append(GpuCase("n34816_k5120_gateup", 34816, 5120, 0, False))
    # Broadcast weight
    cases.append(GpuCase("n64_k1024_bcast", 64, 1024, 0, True))
    # Pola bit ekstrem (deteksi salah staging nol/satu)
    cases.append(GpuCase("n32_k1024_allzeros", 32, 1024, 1, False))
    cases.append(GpuCase("n32_k1024_allones", 32, 1024, 2, False))

    var passed = 0
    for i in range(len(cases)):
        if run_case(ctx, cases[i]):
            passed += 1
    print("=================================================================")
    print(">> HASIL: ", passed, "/", len(cases), " kasus PASS")
    print("=================================================================")

    # ---- Bench kernel-only ukuran produksi (isolasi dari pipeline) ----
    print("=================================================================")
    print(">> BENCH KERNEL-ONLY (ukuran produksi, decode L=1)")
    print("=================================================================")
    bench_case(ctx, "gate_up_n34816_k5120", 34816, 5120, 100)
    bench_case(ctx, "in_proj_n16480_k5120", 16480, 5120, 100)
    bench_case(ctx, "down_n5120_k17408", 5120, 17408, 100)
    bench_case(ctx, "out_proj_n5120_k6144", 5120, 6144, 100)
    bench_case(ctx, "q_proj_n12288_k5120", 12288, 5120, 100)
    bench_case(ctx, "lm_head_n248320_k5120", 248320, 5120, 20)
    print("=================================================================")
