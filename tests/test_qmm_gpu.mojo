# ===----------------------------------------------------------------------=== #
# tests/test_qmm_gpu.mojo
# Purpose: Tes diferensial kernel PREFILL BATCHED qmm WMMA v2 (nvcc FFI)
#          y[M,N] = x[M,K] . w[N,K/8]^T vs referensi FP64 CPU.
#          Kasus M melewati batas dispatch tile (m<=32 vs m>32) dan N tak
#          kelipatan 64 untuk menguji guard tepi.
# Run: mojo build -I . tests/test_qmm_gpu.mojo && ./test_qmm_gpu
# ===----------------------------------------------------------------------=== #

from gpu.host import DeviceContext, DeviceBuffer
from memory import UnsafePointer, alloc
from math import sqrt
from sys.ffi import OwnedDLHandle
from src.ops import (
    CudaQmmPrefillFnFP16, CudaSyncDeviceFn, dummy_cuda_qmm_prefill_fp16,
    try_open_cuda_lib, dummy_cuda_sync_device,
    CudaQmvDecodeFnFP16, dummy_cuda_decode_fp16
)
from src.common import get_scale_stride, get_weight_row_bytes, decode_split_plan


struct QmmCase(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var M: Int
    var N: Int
    var K: Int
    var pattern: Int   # 0=random, 1=all_zeros(w=-s), 2=all_ones(w=+s)

    fn __init__(out self, name: String, M: Int, N: Int, K: Int, pattern: Int):
        self.name = name
        self.M = M
        self.N = N
        self.K = K
        self.pattern = pattern


fn run_qmm_case(ctx: DeviceContext, c: QmmCase) raises -> Bool:
    var L: Int = 1
    var groups = get_scale_stride(c.K)
    var row_bytes = get_weight_row_bytes(c.K)
    var szx = c.M * c.K
    var szw = c.N * row_bytes
    var szs = c.N * groups
    var szo = c.M * c.N

    # ---- 1. Data host deterministik ----
    var hx16 = alloc[Scalar[DType.float16]](szx)
    var hw = alloc[UInt8](szw)
    var hs16 = alloc[Scalar[DType.float16]](szs)
    var hy16 = alloc[Scalar[DType.float16]](szo)
    var href = alloc[Float64](szo)

    for i in range(szx):
        var v = Float32((i % 17) - 8) * 0.125
        hx16[i] = Scalar[DType.float16](v)
    for i in range(szs):
        var v = Float32(1.0 + Float32(i % 5) * 0.1)
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
    for m in range(c.M):
        for n in range(c.N):
            var acc: Float64 = 0.0
            for k in range(c.K):
                var xv = hx16[m * c.K + k].cast[DType.float64]()
                var byte_val = hw[n * row_bytes + (k // 8)]
                var bit = Float64((Int(byte_val) >> (k % 8)) & 1)
                var sg = hs16[n * groups + (k // 128)].cast[DType.float64]()
                acc += (2.0 * bit - 1.0) * sg * xv
            href[m * c.N + n] = acc

    # ---- 3. Jalur GPU FFI qmm (paritas produksi prefill) ----
    var bx = ctx.enqueue_create_buffer[DType.float16](szx)
    var bw = ctx.enqueue_create_buffer[DType.uint8](szw)
    var bs = ctx.enqueue_create_buffer[DType.float16](szs)
    var by = ctx.enqueue_create_buffer[DType.float16](szo)
    ctx.enqueue_copy(bx, hx16)
    ctx.enqueue_copy(bw, hw)
    ctx.enqueue_copy(bs, hs16)
    ctx.enqueue_copy(by, hy16)
    ctx.synchronize()

    var qmm_fn = dummy_cuda_qmm_prefill_fp16
    var sync_fn = dummy_cuda_sync_device
    var h_buf = alloc[OwnedDLHandle](1)
    h_buf.init_pointee_move(try_open_cuda_lib())
    qmm_fn = h_buf[].get_function[CudaQmmPrefillFnFP16]("launch_qmm_sm75_b1_prefill_fp16")
    sync_fn = h_buf[].get_function[CudaSyncDeviceFn]("qmv_sm75_device_synchronize")
    var gemv_fn = h_buf[].get_function[CudaQmvDecodeFnFP16]("launch_qmv_sm75_b1_decode_fp16")

    var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
    var ret = qmm_fn(
        bx.unsafe_ptr(), bw.unsafe_ptr(), bs.unsafe_ptr(),
        by.unsafe_ptr(),
        Int32(c.M), Int32(c.N), Int32(c.K), Int32(L),
        Int32(0), cuda_stream
    )
    if ret != 0:
        print("[GAGAL]", c.name, "| FFI qmm return code:", ret)
        return False
    _ = sync_fn()
    ctx.enqueue_copy(hy16, by)

    # ---- 3b. Jalur GEMV decode per-baris pada DATA SAMA -> out pembanding ----
    var bgv = ctx.enqueue_create_buffer[DType.float16](szo)
    ctx.enqueue_copy(bgv, hy16)  # isi dulu agar residu tak mengganggu
    var splits = decode_split_plan(c.N, L, groups)
    var ws_buf = ctx.enqueue_create_buffer[DType.float32](splits * L * c.N)
    for t in range(c.M):
        var r2 = gemv_fn(
            bx.unsafe_ptr().offset(t * c.K), bw.unsafe_ptr(), bs.unsafe_ptr(),
            bgv.unsafe_ptr().offset(t * c.N), ws_buf.unsafe_ptr(),
            Int32(1), Int32(c.N), Int32(c.K), Int32(L),
            Int32(0), Int32(splits), cuda_stream
        )
        if r2 != 0:
            print("[GAGAL]", c.name, "| FFI gemv return code:", r2)
            return False
    _ = sync_fn()
    ctx.enqueue_copy(hy16, by)
    var hgv16 = alloc[Scalar[DType.float16]](szo)
    ctx.enqueue_copy(hgv16, bgv)
    ctx.synchronize()

    # ---- 4. Bandingkan: max_err + mismatch pertama ----
    var eps: Float64 = 4.9e-4
    var atol: Float64 = 8.0 * eps * sqrt(Float64(c.K)) * 1.5 * 2.0 * 4.0
    var max_err: Float64 = 0.0
    var bad_i = -1
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
        if diff > atol and bad_i == -1:
            bad_i = i
            got_v = got
            want_v = href[i]

    var passed = max_err <= atol
    var stat = "[PASS]" if passed else "[GAGAL]"
    print(stat, c.name, "| M=", c.M, "N=", c.N, "K=", c.K,
          "| max_err=", max_err, "atol=", atol,
          "| rel_err=", max_err / max_href)
    if bad_i != -1:
        var m_i = bad_i // c.N
        var n_i = bad_i % c.N
        print("   >> MISMATCH PERTAMA (m,n)=(", m_i, ",", n_i, ") got=", got_v,
              "want=", want_v, "selisih=", got_v - want_v)

    # ---- 5. QMM vs GEMV (data sama): detektor bug vs beda-akumulasi ----
    # Keduanya akumulasi FP32 di atas produk fp16 yang eksak; satu-satunya
    # perbedaan seharusnya URUTAN penjumlahan + rounding output fp16 ->
    # max|qmm-gemv| wajar <= beberapa ulp fp16 dari skala output. Selisih
    # besar = ada bug indexing/layout, bukan numerik.
    var max_qg: Float64 = 0.0
    var qg_i = -1
    var rel_qg: Float64 = 0.0
    for i in range(szo):
        var a = hy16[i].cast[DType.float64]()
        var b = hgv16[i].cast[DType.float64]()
        var d = abs(a - b)
        if d > max_qg:
            max_qg = d
            qg_i = i
    if qg_i != -1 and abs(hy16[qg_i].cast[DType.float64]()) > 0:
        rel_qg = max_qg / abs(hy16[qg_i].cast[DType.float64]())
    print("   [QMM-vs-GEMV]", c.name, "| max|diff|=", max_qg,
          "rel=", rel_qg, "pada idx=", qg_i)

    hx16.free()
    hw.free()
    hs16.free()
    hy16.free()
    href.free()
    hgv16.free()
    h_buf.destroy_pointee()
    h_buf.free()
    return passed


fn main() raises:
    print("=================================================================")
    print(">> TES DIFERENSIAL PREFILL BATCHED QMM WMMA v2 (nvcc FFI) vs FP64 CPU")
    print("=================================================================")
    var ctx = DeviceContext()

    var cases = List[QmmCase]()
    # Sanity M=1 (tile 32)
    cases.append(QmmCase("m1_n4096_k5120", 1, 4096, 5120, 0))
    # Produksi: M=9 (prompt ChatML "Hello"), N produksi
    cases.append(QmmCase("m9_n5120_k5120_outproj", 9, 5120, 5120, 0))
    cases.append(QmmCase("m9_n16480_k5120_inproj", 9, 16480, 5120, 0))
    cases.append(QmmCase("m9_n34816_k5120_gateup", 9, 34816, 5120, 0))
    cases.append(QmmCase("m9_n1024_k5120_kvproj", 9, 1024, 5120, 0))
    cases.append(QmmCase("m9_n5120_k17408_down", 9, 5120, 17408, 0))
    # Melewati dispatch tile: M=33 -> tile 64x64x64
    cases.append(QmmCase("m33_n5120_k5120", 33, 5120, 5120, 0))
    # N tidak kelipatan 64 (guard tepi epilog) + pattern ekstrem
    cases.append(QmmCase("m9_n5150_k5120_edge", 9, 5150, 5120, 0))
    cases.append(QmmCase("m9_n5120_k5120_allzeros", 9, 5120, 5120, 1))
    cases.append(QmmCase("m9_n5120_k5120_allones", 9, 5120, 5120, 2))

    var passed = 0
    var total = len(cases)
    for c in cases:
        if run_qmm_case(ctx, c):
            passed += 1

    print(">> HASIL:", passed, "/", total, "kasus QMM PREFILL PASS")
    if passed != total:
        print(">> HASIL AKHIR: TERDAPAT KASUS GAGAL PADA QMM PREFILL!")
        raise Error("qmm prefill differential FAILED")
    print(">> HASIL AKHIR: SEMUA KASUS QMM PREFILL PASS!")
