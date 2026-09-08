# ===----------------------------------------------------------------------=== #
# Module: tests/test_cuda_ffi_smoketest.mojo
# Purpose: Gerbang verifikasi pertama interop Mojo MAX Runtime <-> CUDA FFI .so:
#          1. Memastikan libbonsai_qmv_sm75.so berhasil di-load via OwnedDLHandle
#          2. Memastikan DeviceBuffer Mojo (VRAM) dapat diakses & dimutasi oleh kernel CUDA .so
#             tanpa konflik Primary Context
#          3. Memvalidasi kontrak numerik Q1O: w_eff = (2*bit - 1)*s
#             (Pola 0xFF -> +K*s, Pola 0x00 -> -K*s)
# ===----------------------------------------------------------------------=== #

from sys.ffi import OwnedDLHandle
from memory import UnsafePointer, alloc
from os import getenv
from gpu.host import DeviceContext as DeviceContextGPU, DeviceBuffer

alias CudaInitFn = fn() -> Int32

alias CudaQmvDecodeFnFP16 = fn(
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # x
    UnsafePointer[UInt8, MutAnyOrigin],                 # w
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # scales
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # out_ptr
    UnsafePointer[Float32, MutAnyOrigin],              # ws
    Int32, Int32, Int32, Int32,                       # m, n, k, l
    Int32,                                            # broadcast_w
    Int32,                                            # splits
    UnsafePointer[Float32, MutAnyOrigin]              # stream
) -> Int32


fn try_load_lib() raises -> OwnedDLHandle:
    var env_path = getenv("BONSAI_CUDA_LIB")
    if env_path:
        try:
            var h = OwnedDLHandle(env_path)
            print("[SMOKE-TEST] Menggunakan BONSAI_CUDA_LIB:", env_path)
            return h^
        except:
            print("[SMOKE-TEST] Gagal membuka BONSAI_CUDA_LIB:", env_path)

    try:
        var h = OwnedDLHandle("libbonsai_qmv_sm75.so")
        print("[SMOKE-TEST] Berhasil memuat: libbonsai_qmv_sm75.so")
        return h^
    except:
        pass

    try:
        var h = OwnedDLHandle("/kaggle/working/libbonsai_qmv_sm75.so")
        print("[SMOKE-TEST] Berhasil memuat: /kaggle/working/libbonsai_qmv_sm75.so")
        return h^
    except:
        pass

    try:
        var h = OwnedDLHandle("build/libbonsai_qmv_sm75.so")
        print("[SMOKE-TEST] Berhasil memuat: build/libbonsai_qmv_sm75.so")
        return h^
    except:
        pass

    try:
        var h = OwnedDLHandle("/kaggle/working/build/libbonsai_qmv_sm75.so")
        print("[SMOKE-TEST] Berhasil memuat: /kaggle/working/build/libbonsai_qmv_sm75.so")
        return h^
    except:
        pass

    raise Error("libbonsai_qmv_sm75.so tidak ditemukan")


fn main() raises:
    print("=========================================================")
    print(" SMOKE-TEST: FFI CUDA SM75 & MOJO PRIMARY CONTEXT")
    print("=========================================================")

    var handle: OwnedDLHandle
    try:
        handle = try_load_lib()
    except:
        print("[SMOKE-TEST SKIP] libbonsai_qmv_sm75.so tidak ditemukan.")
        print("[INFO] Ini normal di lingkungan Mac lokal tanpa GPU/nvcc.")
        print("[INFO] Kompilasi .so akan berjalan otomatis di container GPU Kaggle.")
        return

    # 1. Cek simbol init
    try:
        var init_fn = handle.get_function[CudaInitFn]("bonsai_cuda_sm75_init")
        var init_code = init_fn()
        if init_code != 0:
            print("[ERROR] bonsai_cuda_sm75_init mengembalikan kode:", init_code)
            return
        print("[OK] Simbol C ABI berhasil di-resolve: bonsai_cuda_sm75_init() == 0")
    except e:
        print("[ERROR] Gagal memanggil bonsai_cuda_sm75_init:", e)
        return

    var decode_fn = handle.get_function[CudaQmvDecodeFnFP16]("launch_qmv_sm75_b1_decode_fp16")
    print("[OK] Simbol C ABI berhasil di-resolve: launch_qmv_sm75_b1_decode_fp16")

    # 2. Inisialisasi MAX GPU DeviceContext & Alokasi VRAM Mojo
    var ctx = DeviceContextGPU()
    var M: Int = 1
    var N: Int = 32      # 1 block SM75 = 32 baris
    var K: Int = 128     # 1 grup g128 = 128 aktivasi
    var L: Int = 1
    var groups_per_row = 1
    var row_bytes = 16   # 128 / 8 = 16 byte per baris

    var hx = alloc[Scalar[DType.float16]](K)
    var hw = alloc[UInt8](N * row_bytes)
    var hs = alloc[Scalar[DType.float16]](N * groups_per_row)
    var hy = alloc[Scalar[DType.float16]](N)

    # Inisialisasi: x = 1.0, s = 0.5
    for i in range(K):
        hx[i] = Scalar[DType.float16](1.0)
    for i in range(N):
        hs[i] = Scalar[DType.float16](0.5)
        hy[i] = Scalar[DType.float16](0.0)

    # Setengah baris pertama w = 0xFF (+K*s = +64.0), setengah baris kedua w = 0x00 (-K*s = -64.0)
    for row in range(N):
        var byte_val: UInt8 = 0xFF if row < 16 else 0x00
        for b in range(row_bytes):
            hw[row * row_bytes + b] = byte_val

    var bx_buf = ctx.enqueue_create_buffer[DType.float16](K)
    var bw_buf = ctx.enqueue_create_buffer[DType.uint8](N * row_bytes)
    var bs_buf = ctx.enqueue_create_buffer[DType.float16](N * groups_per_row)
    var by_buf = ctx.enqueue_create_buffer[DType.float16](N)

    ctx.enqueue_copy(bx_buf, hx)
    ctx.enqueue_copy(bw_buf, hw)
    ctx.enqueue_copy(bs_buf, hs)
    ctx.enqueue_copy(by_buf, hy)
    ctx.synchronize()

    # 3. Panggil kernel CUDA via FFI pada device buffer Mojo
    print("[RUN] Mengeksekusi launch_qmv_sm75_b1_decode_fp16 pada buffer VRAM Mojo...")
    var null_stream = UnsafePointer[Float32, MutAnyOrigin]()
    var null_ws = UnsafePointer[Float32, MutAnyOrigin]()

    var ret = decode_fn(
        bx_buf.unsafe_ptr(),
        bw_buf.unsafe_ptr(),
        bs_buf.unsafe_ptr(),
        by_buf.unsafe_ptr(),
        null_ws,
        Int32(M), Int32(N), Int32(K), Int32(L),
        Int32(1), # broadcast_w
        Int32(1), # splits
        null_stream
    )

    if ret != 0:
        print("[ERROR] launch_qmv_sm75_b1_decode_fp16 gagal dengan CUDA error code:", ret)
        return

    ctx.synchronize()
    ctx.enqueue_copy(hy, by_buf)
    ctx.synchronize()

    # 4. Validasi Numerik
    var all_ok = True
    for row in range(N):
        var got = Float32(hy[row])
        var want: Float32 = Float32(64.0) if row < 16 else Float32(-64.0)
        var diff = abs(got - want)
        if diff > 0.1:
            print("[GAGAL] Baris", row, "got=", got, "want=", want, "diff=", diff)
            all_ok = False
            break

    if all_ok:
        print("[PASS] 16 baris 0xFF bernilai +64.0, 16 baris 0x00 bernilai -64.0!")
        print("[SUCCESS] Primary Context sharing Mojo <-> CUDA FFI BERFUNGSI SEMPURNA!")
    else:
        print("[ERROR] Hasil numerik tidak sesuai target.")

    hx.free()
    hw.free()
    hs.free()
    hy.free()
