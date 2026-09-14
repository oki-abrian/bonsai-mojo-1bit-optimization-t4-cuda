# ===----------------------------------------------------------------------=== #
# Module: ops.mojo
# Purpose: High-level Host Dispatching API untuk W1A16 g128 pada NVIDIA T4.
#          Mengotomasi routing cerdas: Decode (M <= 8) -> qmv_sm75_1bit
#                                      Prefill (M > 8)  -> qmm_sm75_1bit
#          Decode memakai geometri paritas qmv_vec_splitk_kernel: M di dalam
#          block (bucket MPAD), split-K deterministik fill-320, dan reduksi
#          ascending tanpa atomicAdd.
# ===----------------------------------------------------------------------=== #

from os import getenv
from sys.ffi import OwnedDLHandle
from memory import UnsafePointer, alloc
from .common import (
    cdiv, PREFILL_BM, PREFILL_BN, PREFILL_BK, PREFILL_THREADS, PREFILL_PAD,
    DECODE_ROWS_PER_BLOCK, DECODE_THREADS, DECODE_K_TILE, DECODE_GRP_PAD,
    DECODE_GS,
    DECODE_MPAD_MAX, decode_mpad_bucket, decode_split_plan, decode_group_range,
    DIRECT_ROWS_PER_BLOCK, DIRECT_SMEM_ELEMS, direct_pad_bucket,
    direct_split_plan, get_scale_stride, get_weight_row_bytes
)
from .kernels.prefill_sm75 import qmm_sm75_b1_kernel_body, qmm_sm75_b1_prefill_gpu
from .kernels.decode_sm75 import (
    qmv_sm75_b1_decode_body, qmv_sm75_b1_decode_gpu, qmv_split_reduce_gpu
)
from .kernels.direct_smallm import qmv_direct_smallm_body, qmv_direct_smallm_gpu
from .kernels.prefill_wmma import (
    WMMA_BM, WMMA_BN, WMMA_BK, WMMA_PAD, WMMA_SMEM_ELEMS, WMMA_THREADS,
    qmm_wmma_b1_body, qmm_wmma_b1_gpu
)
from .kernels.elementwise_sm75 import (
    rmsnorm_sm75_gpu, add_rmsnorm_sm75_gpu, swiglu_sm75_gpu, vec_add_sm75_gpu, copy_vec_sm75_gpu,
    causal_conv1d_sm75_gpu, head_rmsnorm_sm75_gpu,
    gdn_recurrence_sm75_gpu, gdn_norm_gate_sm75_gpu,
    argmax_sm75_stage1_gpu, argmax_sm75_stage2_gpu,
    partial_rope_sm75_gpu, kv_cache_append_sm75_gpu,
    gqa_attention_sm75_gpu, embed_lookup_1bit_sm75_gpu
)
from gpu.host import DeviceContext as DeviceContextGPU

# ----------------------------------------------------------------------------
# Geometri Grid/Block & Abstraksi DeviceContext untuk GPU NVIDIA Tesla T4
# ----------------------------------------------------------------------------
struct Dim3(Copyable, Movable, ImplicitlyCopyable):
    var x: Int
    var y: Int
    var z: Int

    fn __init__(out self, x: Int = 1, y: Int = 1, z: Int = 1):
        self.x = x
        self.y = y
        self.z = z

struct DeviceContext:
    var device_id: Int
    var is_active: Bool

    fn __init__(out self, device_id: Int = 0):
        self.device_id = device_id
        self.is_active = True

    fn synchronize(self):
        """Placeholder sinkronisasi. Pada port GPU nyata (modul `gpu.host`)
        dipetakan ke DeviceContext.synchronize() hardware; saat ini no-op
        karena eksekusi masih simulasi sekuensial di host CPU."""
        pass

# ----------------------------------------------------------------------------
# 1. GPU Dispatcher: Prefill GEMM (M > 8)
# ----------------------------------------------------------------------------
fn qmm_sm75_1bit[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
):
    """
    Mengeksekusi kernel prefill W1A16 g128 (M > 8).
    Mode eksekusi saat ini: simulasi sekuensial per-thread di host CPU —
    DeviceContext/Dim3 di bawah adalah abstraksi geometri placeholder untuk
    port GPU nyata (max.gpu.host.DeviceContext + enqueue_function).
    Grid dimension:
      - grid.x = ceil(N / BN)
      - grid.y = ceil(M / BM)
      - grid.z = L (batch)
    Block dimension: 128 thread (4 warp).
    """
    var grid = Dim3(cdiv(N, PREFILL_BN), cdiv(M, PREFILL_BM), L)
    var block = Dim3(PREFILL_THREADS, 1, 1)
    var ctx = DeviceContext(0)

    # Alokasi Shared Memory As dengan padding anti-bank-conflict
    var smem_size = PREFILL_BM * (PREFILL_BK + PREFILL_PAD)
    var smem_a = alloc[Scalar[T]](smem_size)

    # Eksekusi grid CTA — simulasi host sekuensial per-thread
    for bz in range(grid.z):
        for by in range(grid.y):
            for bx in range(grid.x):
                for tid in range(block.x):
                    qmm_sm75_b1_kernel_body[T](
                        x, w, scales, y,
                        M, N, K, L, broadcast_w,
                        bx, by, bz, tid,
                        smem_a
                    )

    ctx.synchronize()
    smem_a.free()

# ----------------------------------------------------------------------------
# 2. GPU Dispatcher: Decode GEMV Split-K (M <= 8, Optimized untuk M=1)
# ----------------------------------------------------------------------------
fn qmv_sm75_1bit[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
):
    """
    Mengeksekusi kernel decode W1A16 g128 (M <= 8, token generation).
    Mode eksekusi saat ini: simulasi sekuensial per-thread di host CPU.
    Geometri paritas qmv_vec_splitk_kernel:
      - grid.x = ceil(N / 32), grid.y = 1 (M di dalam block via bucket MPAD
        {1,2,4,8} — bobot hanya di-stream 1x per block), grid.z = L.
      - Split-K deterministik: bila grid kurang dari setengah fill target
        (320 blok, kalibrasi T4), grup g128 dipecah kontigu maks 16 slice;
        partial FP32 ditulis ke workspace lalu direduksi ASCENDING per slice
        (tanpa atomicAdd -> deterministik bitwise run-to-run).
      - Skala baris milik block di-stage ke SMEM per K-tile.
    Block dimension: 256 thread (8 warp).
    Shared Memory: MPAD_MAX x (8 grup x 132 float) + s_s 256 + reduksi 256
    = 8960 float (35,8 KiB FP32).
    """
    var mpad = decode_mpad_bucket(M)
    var groups_per_row = get_scale_stride(K)
    var blocks_x = cdiv(N, DECODE_ROWS_PER_BLOCK)
    var splits = decode_split_plan(N, L, groups_per_row)
    var use_ws = splits > 1

    var grid = Dim3(blocks_x, 1, L)
    var block = Dim3(DECODE_THREADS, 1, 1)
    var ctx = DeviceContext(0)

    # Alokasi SMEM host-sim: staging x + s_s + nib_s + xg_s + scratchpad reduksi.
    var smem_size = 32768
    var smem_x = alloc[Float32](smem_size)

    # Workspace partial split-K FP32 [splits][L * M * N] (kosong bila splits=1)
    var ws = UnsafePointer[Float32, MutAnyOrigin]()
    if use_ws:
        ws = alloc[Float32](splits * L * M * N)

    # Eksekusi grid CTA per slice — simulasi host sekuensial per-thread
    for s in range(splits):
        var gr = decode_group_range(groups_per_row, splits, s)
        for bz in range(grid.z):
            for bx in range(grid.x):
                for tid in range(block.x):
                    qmv_sm75_b1_decode_body[T](
                        x, w, scales, y, ws, use_ws, s,
                        gr.g_begin, gr.g_count,
                        M, N, K, L, broadcast_w, mpad,
                        bx, bz, tid,
                        smem_x
                    )

    # Reduksi split-K deterministik (ascending per slice) di host
    if use_ws:
        var total = L * M * N
        for e in range(total):
            var acc: Float32 = 0.0
            for s in range(splits):
                acc += ws[s * total + e]
            y[e] = Scalar[T](acc)
        ws.free()

    ctx.synchronize()
    smem_x.free()

# ----------------------------------------------------------------------------
# 2b. GPU Dispatcher: Direct Small-M Prefill (8 < M <= 64)
# ----------------------------------------------------------------------------
fn qmv_direct_smallm_1bit[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
):
    """
    Mengeksekusi kernel direct small-m (8 < M <= 64) — paritas
    qmv_direct_smallm_kernel: grid (ceil(N/8), 1, L) dengan 1 warp per baris
    output, akumulator register per-mi (bucket PAD {16,32,64}), bobot
    di-stream 1x per block, split-K deterministik gate 40 / target 80,
    reduksi ascending tanpa atomicAdd.
    Mode eksekusi saat ini: simulasi sekuensial per-thread di host CPU.
    Shared Memory: staging DIRECT_SMEM_ELEMS elemen T (17408 = kasus terburuk
    PAD 64/GS 2 atau PAD 32/GS 4) + scratchpad reduksi 256 float.
    """
    var pad = direct_pad_bucket(M)
    var groups_per_row = get_scale_stride(K)
    var blocks_x = cdiv(N, DIRECT_ROWS_PER_BLOCK)
    var splits = direct_split_plan(N, L, groups_per_row)
    var use_ws = splits > 1

    var ctx = DeviceContext(0)
    var smem_x = alloc[Scalar[T]](DIRECT_SMEM_ELEMS)
    var smem_red = alloc[Float32](DECODE_THREADS)

    # Workspace partial split-K FP32 [splits][L * M * N] (kosong bila splits=1)
    var ws = UnsafePointer[Float32, MutAnyOrigin]()
    if use_ws:
        ws = alloc[Float32](splits * L * M * N)

    # Eksekusi grid CTA per slice — simulasi host sekuensial per-thread
    for s in range(splits):
        var gr = decode_group_range(groups_per_row, splits, s)
        for bz in range(L):
            for bx in range(blocks_x):
                for tid in range(DECODE_THREADS):
                    qmv_direct_smallm_body[T](
                        x, w, scales, y, ws, use_ws, s,
                        gr.g_begin, gr.g_count,
                        M, N, K, L, broadcast_w, pad,
                        bx, bz, tid,
                        smem_x, smem_red
                    )

    # Reduksi split-K deterministik (ascending per slice) di host
    if use_ws:
        var total = L * M * N
        for e in range(total):
            var acc: Float32 = 0.0
            for s in range(splits):
                acc += ws[s * total + e]
            y[e] = Scalar[T](acc)
        ws.free()

    ctx.synchronize()
    smem_x.free()
    smem_red.free()

# ----------------------------------------------------------------------------
# 2c. GPU Dispatcher: Prefill Produksi WMMA v2 (M >= 8) — paritas
#     qmm_impl_sm75_wmma.cuh, kernel yang dieksekusi wheel referensi
# ----------------------------------------------------------------------------
fn qmm_wmma_b1_1bit[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
):
    """
    Mengeksekusi kernel prefill produksi W1A16 g128 (M >= 8) — port
    qmm_impl_sm75_wmma.cuh (WMMA v2): tiling BM=64/BN=64/BK=64, 128 thread
    (4 warp x sub-tile 32x32), As[64][72] + Wt[64][72] transposed PAD=8,
    dequant uint32 coalesced -> (2q-1)*s.
    Grid: (ceil(N/64), ceil(M/64), L). Block: 128 thread.
    Mode eksekusi saat ini: simulasi sekuensial per-thread di host CPU.
    Shared Memory: 9216 elemen T (~18 KiB fp16).
    """
    var grid = Dim3(cdiv(N, WMMA_BN), cdiv(M, WMMA_BM), L)
    var block = Dim3(WMMA_THREADS, 1, 1)
    var ctx = DeviceContext(0)

    var smem_as = alloc[Scalar[T]](WMMA_SMEM_ELEMS)
    var smem_wt = smem_as + WMMA_BM * (WMMA_BK + 8)

    # Eksekusi grid CTA — simulasi host sekuensial per-thread
    for bz in range(grid.z):
        for by in range(grid.y):
            for bx in range(grid.x):
                for tid in range(block.x):
                    qmm_wmma_b1_body[T](
                        x, w, scales, y,
                        M, N, K, L, broadcast_w,
                        bx, by, bz, tid,
                        smem_as, smem_wt
                    )

    ctx.synchronize()
    smem_as.free()

# ----------------------------------------------------------------------------
# 3. Automatic Unified Router (Sesuai quantized.cpp MLX)
# ----------------------------------------------------------------------------
fn quantized_matmul_1bit[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
):
    """
    Router produksi 1-bit g128 Affine — STRICT, paritas quantized.cpp
    referensi ("No Silent Fallbacks"):
    - M < 8 (fase decode token) -> qmv_sm75_1bit (vec split-K, MPAD 1/2/4/8).
    - M >= 8 (fase prefill)     -> qmm_wmma_b1_1bit (tiled WMMA v2 BM/BN/BK
      64/64/64 — kernel yang dieksekusi wheel referensi).
    Jalur direct small-m (8 < M <= 64) tetap tersedia sebagai fungsi
    eksplisit (qmv_direct_smallm_1bit) namun tidak dipakai router produksi —
    sama seperti referensi (hanya dicover self-test).
    """
    if M < 8:
        qmv_sm75_1bit[T](x, w, scales, y, M, N, K, L, broadcast_w)
    else:
        qmm_wmma_b1_1bit[T](x, w, scales, y, M, N, K, L, broadcast_w)

# ----------------------------------------------------------------------------
# 3b. Router NATIVE GPU (kernel device asli, bukan simulasi host)
# ----------------------------------------------------------------------------
fn quantized_matmul_1bit_gpu[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
) raises:
    """
    Router produksi yang benar-benar meluncurkan kernel ke hardware NVIDIA
    T4 (max.gpu.host.DeviceContext + enqueue_function). Berbeda dengan
    quantized_matmul_1bit (simulasi host sekuensial per-thread yang dipakai
    untuk validasi numerik), fungsi ini MENGHASILKAN eksekusi GPU sungguhan.
    Routing mengikuti referensi: decode M < 8 -> qmv_sm75 decode split-K,
    prefill M >= 8 -> prefill tiled sm_75.
    """
    if M < 8:
        qmv_sm75_1bit_gpu_launch[T](x, w, scales, y, M, N, K, L, broadcast_w)
    else:
        qmm_sm75_1bit_gpu_launch[T](x, w, scales, y, M, N, K, L, broadcast_w)

# ----------------------------------------------------------------------------
# 4. Native GPU Launchers Resmi (Modular MAX GPU SDK)
# ----------------------------------------------------------------------------
fn qmv_sm75_1bit_gpu_launch[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
) raises:
    """
    Meluncurkan kernel decode W1A16 g128 langsung ke hardware NVIDIA Tesla T4
    menggunakan API resmi Modular MAX GPU (max.gpu.host.DeviceContext).
    Grid = (ceil(N/32), 1, L): dimensi M diproses di dalam block via bucket
    MPAD {1,2,4,8}. Shared Memory dialokasi di dalam kernel device
    (AddressSpace.SHARED). Split-K: partial FP32 ditulis ke buffer device,
    lalu kernel reduce ascending menjumlahkannya ke y — tanpa atomicAdd.
    """
    from gpu.host import DeviceContext

    var mpad = decode_mpad_bucket(M)
    var groups_per_row = get_scale_stride(K)
    var grid_x = cdiv(N, DECODE_ROWS_PER_BLOCK)
    var splits = decode_split_plan(N, L, groups_per_row)
    var use_ws = splits > 1

    var ctx = DeviceContext()
    var ws_empty = UnsafePointer[Float32, MutAnyOrigin]()

    # Split-K: buffer device hidup sampai setelah synchronize (buffer device
    # TIDAK BOLEH terbebaskan selagi kernel masih mengantre di stream).
    if use_ws:
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](splits * L * M * N)
        var ws = ws_buf.unsafe_ptr()
        for s in range(splits):
            var gr = decode_group_range(groups_per_row, splits, s)
            ctx.enqueue_function[qmv_sm75_b1_decode_gpu[T]](
                x, w, scales, y, ws, use_ws, s,
                gr.g_begin, gr.g_count,
                M, N, K, L, broadcast_w, mpad,
                grid_dim=(grid_x, 1, L),
                block_dim=(DECODE_THREADS, 1, 1)
            )
        var total = L * M * N
        ctx.enqueue_function[qmv_split_reduce_gpu[T]](
            ws, y, splits, total,
            grid_dim=(cdiv(total, DECODE_THREADS), 1, 1),
            block_dim=(DECODE_THREADS, 1, 1)
        )
        ctx.synchronize()
    else:
        var gr = decode_group_range(groups_per_row, 1, 0)
        ctx.enqueue_function[qmv_sm75_b1_decode_gpu[T]](
            x, w, scales, y, ws_empty, use_ws, 0,
            gr.g_begin, gr.g_count,
            M, N, K, L, broadcast_w, mpad,
            grid_dim=(grid_x, 1, L),
            block_dim=(DECODE_THREADS, 1, 1)
        )

    ctx.synchronize()

# ----------------------------------------------------------------------------
# Integrasi CUDA FFI (nvcc) untuk Decode GEMV SM75 W1A16 Q1O
# ----------------------------------------------------------------------------
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


# Prefill batched W1A16 (WMMA v2 tensor core, tanpa workspace split-K).
alias CudaQmmPrefillFnFP16 = fn(
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # x [L,M,K]
    UnsafePointer[UInt8, MutAnyOrigin],                 # w [N,K/8]
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # scales [N,K/128]
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin], # out [L,M,N]
    Int32, Int32, Int32, Int32,                       # m, n, k, l
    Int32,                                            # broadcast_w
    UnsafePointer[Float32, MutAnyOrigin]              # stream
) -> Int32


fn dummy_cuda_qmm_prefill_fp16(
    x: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    m: Int32, n: Int32, k: Int32, l: Int32,
    broadcast_w: Int32,
    stream: UnsafePointer[Float32, MutAnyOrigin]
) -> Int32:
    return -1


fn dummy_cuda_decode_fp16(
    x: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    ws: UnsafePointer[Float32, MutAnyOrigin],
    m: Int32, n: Int32, k: Int32, l: Int32,
    broadcast_w: Int32,
    splits: Int32,
    stream: UnsafePointer[Float32, MutAnyOrigin]
) -> Int32:
    return -1


alias CudaSyncDeviceFn = fn() -> Int32

fn dummy_cuda_sync_device() -> Int32:
    return -1


fn try_open_cuda_lib() raises -> OwnedDLHandle:
    """Mencari dan membuka libbonsai_qmv_sm75.so dari kandidat lokasi."""
    var env_path = getenv("BONSAI_CUDA_LIB")
    if env_path:
        return OwnedDLHandle(env_path)
    try:
        return OwnedDLHandle("libbonsai_qmv_sm75.so")
    except:
        pass
    try:
        return OwnedDLHandle("/kaggle/working/libbonsai_qmv_sm75.so")
    except:
        pass
    try:
        return OwnedDLHandle("build/libbonsai_qmv_sm75.so")
    except:
        pass
    return OwnedDLHandle("/kaggle/working/build/libbonsai_qmv_sm75.so")


fn try_cuda_ffi_decode_fp16(
    x: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    ws: UnsafePointer[Float32, MutAnyOrigin],
    m: Int, n: Int, k: Int, l: Int,
    broadcast_w: Bool,
    splits: Int,
    stream: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin]()
) -> Bool:
    """Mencoba mengeksekusi kernel CUDA nvcc via FFI jika libbonsai_qmv_sm75.so tersedia."""
    var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
    if disable and (disable == "1" or disable == "true"):
        return False

    try:
        var h = try_open_cuda_lib()
        var f = h.get_function[CudaQmvDecodeFnFP16]("launch_qmv_sm75_b1_decode_fp16")
        var ret = f(
            x, w, scales, out_ptr, ws,
            Int32(m), Int32(n), Int32(k), Int32(l),
            Int32(1 if broadcast_w else 0),
            Int32(splits),
            stream
        )
        return ret == 0
    except:
        return False


fn qmv_sm75_1bit_launch_on[
    T: DType
](
    ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True,
    ws: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer[Float32, MutAnyOrigin]()
) raises:
    """
    Meluncurkan kernel decode W1A16 g128 pada DeviceContext yang SUDAH ADA.
    Workspace split-K harus dipasok pemanggil (buffer device persisten):
    membuatnya di sini berarti cudaMalloc per matmul — sinkron se-device,
    ~10 ms x 48-64 proyeksi attention per token.
    Prioritas 1: FFI CUDA Shared Library (nvcc SASS optimal, zero-sync stream sharing).
    Prioritas 2: Native Mojo GPU Kernel (fallback jika FFI tidak tersedia).
    """
    var mpad = decode_mpad_bucket(M)
    var groups_per_row = get_scale_stride(K)
    var splits = decode_split_plan(N, L, groups_per_row)

    @parameter
    if T == DType.float16:
        var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()

        if try_cuda_ffi_decode_fp16(
            x.bitcast[Scalar[DType.float16]](),
            w,
            scales.bitcast[Scalar[DType.float16]](),
            y.bitcast[Scalar[DType.float16]](),
            ws,
            M, N, K, L, broadcast_w, splits,
            cuda_stream
        ):
            return

    var grid_x = cdiv(N, DECODE_ROWS_PER_BLOCK)
    var use_ws = splits > 1

    if use_ws:
        for s in range(splits):
            var gr = decode_group_range(groups_per_row, splits, s)
            ctx.enqueue_function[qmv_sm75_b1_decode_gpu[T, 1]](
                x, w, scales, y, ws, use_ws, s,
                gr.g_begin, gr.g_count,
                M, N, K, L, broadcast_w, 1,
                grid_dim=(grid_x, 1, L),
                block_dim=(DECODE_THREADS, 1, 1)
            )
        var total = L * M * N
        ctx.enqueue_function[qmv_split_reduce_gpu[T]](
            ws, y, splits, total,
            grid_dim=(cdiv(total, DECODE_THREADS), 1, 1),
            block_dim=(DECODE_THREADS, 1, 1)
        )
    else:
        var gr = decode_group_range(groups_per_row, 1, 0)
        ctx.enqueue_function[qmv_sm75_b1_decode_gpu[T, 1]](
            x, w, scales, y, ws, use_ws, 0,
            gr.g_begin, gr.g_count,
            M, N, K, L, broadcast_w, 1,
            grid_dim=(grid_x, 1, L),
            block_dim=(DECODE_THREADS, 1, 1)
        )


fn qmv_direct_smallm_gpu_launch[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
) raises:
    """
    Meluncurkan kernel direct small-m (8 < M <= 64) ke hardware NVIDIA Tesla
    T4 via max.gpu.host.DeviceContext. Grid = (ceil(N/8), 1, L), 1 warp per
    baris output, bucket PAD {16,32,64}. Shared Memory dialokasi di dalam
    kernel device (AddressSpace.SHARED, kontrak T fp16/bf16). Split-K:
    partial FP32 ke buffer device + kernel reduce ascending.
    """
    from gpu.host import DeviceContext

    var pad = direct_pad_bucket(M)
    var groups_per_row = get_scale_stride(K)
    var grid_x = cdiv(N, DIRECT_ROWS_PER_BLOCK)
    var splits = direct_split_plan(N, L, groups_per_row)
    var use_ws = splits > 1

    var ctx = DeviceContext()
    var ws_empty = UnsafePointer[Float32, MutAnyOrigin]()

    if use_ws:
        var ws_buf = ctx.enqueue_create_buffer[DType.float32](splits * L * M * N)
        var ws = ws_buf.unsafe_ptr()
        for s in range(splits):
            var gr = decode_group_range(groups_per_row, splits, s)
            ctx.enqueue_function[qmv_direct_smallm_gpu[T]](
                x, w, scales, y, ws, use_ws, s,
                gr.g_begin, gr.g_count,
                M, N, K, L, broadcast_w, pad,
                grid_dim=(grid_x, 1, L),
                block_dim=(DECODE_THREADS, 1, 1)
            )
        var total = L * M * N
        ctx.enqueue_function[qmv_split_reduce_gpu[T]](
            ws, y, splits, total,
            grid_dim=(cdiv(total, DECODE_THREADS), 1, 1),
            block_dim=(DECODE_THREADS, 1, 1)
        )
        ctx.synchronize()
    else:
        var gr = decode_group_range(groups_per_row, 1, 0)
        ctx.enqueue_function[qmv_direct_smallm_gpu[T]](
            x, w, scales, y, ws_empty, use_ws, 0,
            gr.g_begin, gr.g_count,
            M, N, K, L, broadcast_w, pad,
            grid_dim=(grid_x, 1, L),
            block_dim=(DECODE_THREADS, 1, 1)
        )

    ctx.synchronize()

fn qmm_wmma_b1_gpu_launch[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
) raises:
    """
    Meluncurkan kernel prefill produksi WMMA v2 (M >= 8) ke hardware NVIDIA
    Tesla T4 via max.gpu.host.DeviceContext. Grid = (ceil(N/64), ceil(M/64),
    L), block 128 thread. Shared Memory dialokasi di dalam kernel device
    (AddressSpace.SHARED).
    """
    from gpu.host import DeviceContext

    var grid_x = cdiv(N, WMMA_BN)
    var grid_y = cdiv(M, WMMA_BM)
    var grid_z = L

    var ctx = DeviceContext()
    ctx.enqueue_function[qmm_wmma_b1_gpu[T]](
        x, w, scales, y,
        M, N, K, L, broadcast_w,
        grid_dim=(grid_x, grid_y, grid_z),
        block_dim=(WMMA_THREADS, 1, 1)
    )
    ctx.synchronize()

fn qmm_sm75_1bit_gpu_launch[
    T: DType
](
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    w: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Scalar[T], MutAnyOrigin],
    y: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int = 1,
    broadcast_w: Bool = True
) raises:
    """
    Meluncurkan kernel prefill W1A16 g128 langsung ke hardware NVIDIA Tesla T4
    menggunakan API resmi Modular MAX GPU (max.gpu.host.DeviceContext).
    Shared Memory dialokasi di dalam kernel device pada address space SHARED
    (stack_allocation) — pointer heap host tidak pernah dipass ke device.
    """
    from gpu.host import DeviceContext

    var grid_x = cdiv(N, PREFILL_BN)
    var grid_y = cdiv(M, PREFILL_BM)
    var grid_z = L

    var ctx = DeviceContext()
    ctx.enqueue_function[qmm_sm75_b1_prefill_gpu[T]](
        x, w, scales, y,
        M, N, K, L, broadcast_w,
        grid_dim=(grid_x, grid_y, grid_z),
        block_dim=(PREFILL_THREADS, 1, 1)
    )
    ctx.synchronize()

# ----------------------------------------------------------------------------
# 4. Peluncur Kernel Elementwise & Non-Matmul untuk Full GPU Pipeline
# ----------------------------------------------------------------------------
fn rmsnorm_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    has_weight: Bool,
    D: Int,
    eps: Float32 = 1e-6,
    rows: Int = 1
) raises:
    """Meluncurkan RMSNorm di VRAM (1 block x 256 thread per baris).
    rows > 1 = batched prefill M-token (grid.y = baris)."""
    if has_weight and weight != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[rmsnorm_sm75_gpu[T, True]](
            x, out_ptr, weight, D, eps,
            grid_dim=(1, rows, 1),
            block_dim=(256, 1, 1)
        )
    else:
        var null_w = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[rmsnorm_sm75_gpu[T, False]](
            x, out_ptr, null_w, D, eps,
            grid_dim=(1, rows, 1),
            block_dim=(256, 1, 1)
        )

fn add_rmsnorm_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    res: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_norm: UnsafePointer[Scalar[T], MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    has_weight: Bool,
    D: Int,
    eps: Float32 = 1e-6,
    rows: Int = 1
) raises:
    """FUSI residual-add + RMSNorm dalam 1 launch (bit-exact dgn jalur
    vec_add_sm75_launch_on + rmsnorm_sm75_launch_on berurutan)."""
    if has_weight and weight != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[add_rmsnorm_sm75_gpu[T, True]](
            x, res, out_norm, weight, D, eps,
            grid_dim=(1, rows, 1),
            block_dim=(256, 1, 1)
        )
    else:
        var null_w = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[add_rmsnorm_sm75_gpu[T, False]](
            x, res, out_norm, null_w, D, eps,
            grid_dim=(1, rows, 1),
            block_dim=(256, 1, 1)
        )

fn swiglu_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    gate_up: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    intermediate_size: Int,
    rows: Int = 1
) raises:
    """Meluncurkan aktivasi SwiGLU di VRAM (cdiv(inter,256) x rows block)."""
    var grid_x = cdiv(intermediate_size, 256)
    ctx.enqueue_function[swiglu_sm75_gpu[T]](
        gate_up, out_ptr, intermediate_size,
        grid_dim=(grid_x, rows, 1),
        block_dim=(256, 1, 1)
    )

fn vec_add_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    res: UnsafePointer[Scalar[T], MutAnyOrigin],
    D: Int,
    rows: Int = 1
) raises:
    """Meluncurkan in-place residual addition di VRAM: x += res.
    rows > 1 = batched prefill M-token (grid.y = baris)."""
    var grid_x = cdiv(D, 256)
    ctx.enqueue_function[vec_add_sm75_gpu[T]](
        x, res, D,
        grid_dim=(grid_x, rows, 1),
        block_dim=(256, 1, 1)
    )

fn copy_vec_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    dst: UnsafePointer[Scalar[T], MutAnyOrigin],
    src: UnsafePointer[Scalar[T], MutAnyOrigin],
    n: Int
) raises:
    """Salin vektor device-to-device di VRAM: dst = src."""
    var grid_x = cdiv(n, 256)
    ctx.enqueue_function[copy_vec_sm75_gpu[T]](
        dst, src, n,
        grid_dim=(grid_x, 1, 1),
        block_dim=(256, 1, 1)
    )


# GDN sequence fused (tiru gdn_step_kernel MLX fork): 1 launch per layer
# untuk seluruh chunk T token, state di register.
alias CudaGdnSeqFnFP16 = fn(
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # q
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # k
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # v
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # a
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # b
    UnsafePointer[Float32, MutAnyOrigin],                # a_log
    UnsafePointer[Float32, MutAnyOrigin],                # dt_bias
    UnsafePointer[Float32, MutAnyOrigin],                # state fp32 in/out
    UnsafePointer[Scalar[DType.float16], MutAnyOrigin],  # y [T, y_stride]
    Int32, Int32, Int32, Int32, Int32,                   # T, Hv, Hk, Dk, Dv
    Int32, Int32, Int32, Int32,                          # qk,v,ab,y strides
    Int32, Int32,                                        # a_off, b_off
    UnsafePointer[Float32, MutAnyOrigin]                 # stream
) -> Int32


fn gdn_seq_sm75_try_launch(
    ctx: DeviceContextGPU,
    q: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    k: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    v: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    a: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    b: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    state: UnsafePointer[Float32, MutAnyOrigin],
    y: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    T: Int, Hv: Int, Hk: Int, Dk: Int, Dv: Int,
    qk_stride: Int, v_stride: Int, ab_stride: Int, y_stride: Int,
    a_off: Int, b_off: Int
) -> Bool:
    """Jalankan kernel GDN sequence fused via FFI. False = .so tidak tersedia
    (pemanggil fallback ke loop per-token)."""
    var disable = getenv("BONSAI_DISABLE_CUDA_FFI")
    if disable and (disable == "1" or disable == "true"):
        return False
    try:
        var h = try_open_cuda_lib()
        var f = h.get_function[CudaGdnSeqFnFP16]("launch_gdn_seq_sm75_fp16")
        var cuda_stream = UnsafePointer[Float32, MutAnyOrigin]()
        var ret = f(
            q, k, v, a, b, a_log, dt_bias, state, y,
            Int32(T), Int32(Hv), Int32(Hk), Int32(Dk), Int32(Dv),
            Int32(qk_stride), Int32(v_stride), Int32(ab_stride), Int32(y_stride),
            Int32(a_off), Int32(b_off), cuda_stream
        )
        return ret == 0
    except:
        return False


# ============================================================================ #
fn causal_conv1d_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    conv_buf: UnsafePointer[Float32, MutAnyOrigin],
    new_input: UnsafePointer[Scalar[T], MutAnyOrigin],
    weights: UnsafePointer[Float32, MutAnyOrigin],
    has_weights: Bool,
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    conv_dim: Int
) raises:
    """Meluncurkan Causal Conv1D 4-tap di VRAM (48 block x 256 thread)."""
    var grid_x = cdiv(conv_dim, 256)
    if has_weights and weights != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[causal_conv1d_sm75_gpu[T, True]](
            conv_buf, new_input, weights, out_ptr, conv_dim,
            grid_dim=(grid_x, 1, 1),
            block_dim=(256, 1, 1)
        )
    else:
        var null_w = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[causal_conv1d_sm75_gpu[T, False]](
            conv_buf, new_input, null_w, out_ptr, conv_dim,
            grid_dim=(grid_x, 1, 1),
            block_dim=(256, 1, 1)
        )

fn head_rmsnorm_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    scale_factor: Float32,
    eps: Float32 = 1e-6,
    stride: Int = 0,
    rows: Int = 1,
    row_stride: Int = 0,
    out_row_stride: Int = 0
) raises:
    """Meluncurkan Head RMSNorm (Q-Norm & K-Norm) tanpa bobot di VRAM (untuk GDN).
    rows > 1 = batched prefill (grid.y = baris token, row_stride = lebar baris)."""
    var null_w = UnsafePointer[Float32, MutAnyOrigin]()
    ctx.enqueue_function[head_rmsnorm_sm75_gpu[T, False]](
        x, out_ptr, null_w, head_dim, scale_factor, eps, stride, row_stride,
        out_row_stride,
        grid_dim=(num_heads, rows, 1),
        block_dim=(32, 1, 1)
    )

fn head_rmsnorm_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    x: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    scale_factor: Float32,
    weight: UnsafePointer[Float32, MutAnyOrigin],
    has_weight: Bool,
    eps: Float32 = 1e-6,
    stride: Int = 0,
    rows: Int = 1,
    row_stride: Int = 0,
    out_row_stride: Int = 0
) raises:
    """Meluncurkan Head RMSNorm berbobot di VRAM (untuk Gated Attention).
    CATATAN: out_row_stride WAJIB diteruskan eksplisit ke kernel —
    enqueue_function tidak mengisi param kernel yang di-default
    (bug Run AU/AV: CUDA_ERROR_INVALID_VALUE, 8 arg vs kernel 9 param)."""
    if has_weight and weight != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[head_rmsnorm_sm75_gpu[T, True]](
            x, out_ptr, weight, head_dim, scale_factor, eps, stride, row_stride,
            out_row_stride,
            grid_dim=(num_heads, rows, 1),
            block_dim=(32, 1, 1)
        )
    else:
        var null_w = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[head_rmsnorm_sm75_gpu[T, False]](
            x, out_ptr, null_w, head_dim, scale_factor, eps, stride, row_stride,
            out_row_stride,
            grid_dim=(num_heads, rows, 1),
            block_dim=(32, 1, 1)
        )

fn gdn_recurrence_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    state_s: UnsafePointer[Float32, MutAnyOrigin],
    q_normed: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_normed: UnsafePointer[Scalar[T], MutAnyOrigin],
    v: UnsafePointer[Scalar[T], MutAnyOrigin],
    a: UnsafePointer[Scalar[T], MutAnyOrigin],
    b: UnsafePointer[Scalar[T], MutAnyOrigin],
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    has_params: Bool,
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    repeat_factor: Int,
    H_v: Int,
    D_v: Int,
    D_k: Int
) raises:
    """Meluncurkan rekurensi Gated DeltaNet di VRAM (H_v block x D_v thread).
    WAJIB dari config runtime — JANGAN hardcode. Dimensi Bonsai-27B yang BENAR
    (config.mojo `qwen_27b_default`, diverifikasi thd config.json checkpoint):
    H_v = 48, D_v = D_k = 128, H_k = 16 -> grid (48,1,1) x block (128,1,1).
    Angka lama yang pernah ditulis di sini (64x128, lalu "H_v=32 D_v=256")
    keduanya SALAH untuk model ini; memakai salah satunya meluap melewati
    buffer state atau mematikan separuh dimensi."""
    if has_params and a_log != UnsafePointer[Float32, MutAnyOrigin]() and dt_bias != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[gdn_recurrence_sm75_gpu[T, True]](
            state_s, q_normed, k_normed, v, a, b, a_log, dt_bias,
            out_ptr, repeat_factor, D_v, D_k,
            grid_dim=(H_v, 1, 1),
            block_dim=(D_v, 1, 1)
        )
    else:
        var null_f = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[gdn_recurrence_sm75_gpu[T, False]](
            state_s, q_normed, k_normed, v, a, b, null_f, null_f,
            out_ptr, repeat_factor, D_v, D_k,
            grid_dim=(H_v, 1, 1),
            block_dim=(D_v, 1, 1)
        )

fn gdn_norm_gate_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    gdn_out: UnsafePointer[Scalar[T], MutAnyOrigin],
    z: UnsafePointer[Scalar[T], MutAnyOrigin],
    norm_w: UnsafePointer[Float32, MutAnyOrigin],
    has_norm_w: Bool,
    H_v: Int,
    D_v: Int,
    eps: Float32 = 1e-6,
    rows: Int = 1,
    out_row_stride: Int = 0,
    z_row_stride: Int = 0,
    legacy_override: Int = -1
) raises:
    """Meluncurkan Fused GDN Gating & Per-Head RMSNorm di VRAM.
    rows > 1 = batched prefill (grid.y = baris; stride per baris eksplisit).

    SAKELAR A/B `BONSAI_GDN_NORM_ORDER`: default (tidak diset) = urutan BENAR
    (norm atas x murni, gate silu(z) terakhir — paritas Qwen3NextRMSNormGated).
    Set `gate_first` (atau `legacy`/`1`) untuk mereproduksi bug LAMA supaya
    besar dampaknya bisa diukur di T4. JANGAN dipakai produksi.

    `legacy_override`: -1 = ikuti env (perilaku normal). 0/1 = PAKSA mode,
    dipakai tes diferensial supaya bisa menguji kedua urutan tanpa menyentuh
    environment proses.
    """
    var legacy_flag = 0
    if legacy_override >= 0:
        legacy_flag = legacy_override
    else:
        var legacy = getenv("BONSAI_GDN_NORM_ORDER")
        if legacy and (legacy == "gate_first" or legacy == "legacy" or legacy == "1"):
            legacy_flag = 1
            print(">> [GDN-AB] BONSAI_GDN_NORM_ORDER=gate_first -> memakai urutan "
                  "norm/gate LAMA (bug). Hanya untuk A/B, bukan produksi.")
    if has_norm_w and norm_w != UnsafePointer[Float32, MutAnyOrigin]():
        ctx.enqueue_function[gdn_norm_gate_sm75_gpu[T, True]](
            gdn_out, z, norm_w, D_v, eps, out_row_stride, z_row_stride,
            legacy_flag,
            grid_dim=(H_v, rows, 1),
            block_dim=(D_v, 1, 1)
        )
    else:
        var null_w = UnsafePointer[Float32, MutAnyOrigin]()
        ctx.enqueue_function[gdn_norm_gate_sm75_gpu[T, False]](
            gdn_out, z, null_w, D_v, eps, out_row_stride, z_row_stride,
            legacy_flag,
            grid_dim=(H_v, rows, 1),
            block_dim=(D_v, 1, 1)
        )

fn argmax_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    logits: UnsafePointer[Scalar[T], MutAnyOrigin],
    block_vals: UnsafePointer[Float32, MutAnyOrigin],
    block_idxs: UnsafePointer[Int32, MutAnyOrigin],
    out_token: UnsafePointer[Int32, MutAnyOrigin],
    V: Int
) raises:
    """Meluncurkan reduksi 2-stage Argmax di VRAM (248,320 -> 1 token id)."""
    ctx.enqueue_function[argmax_sm75_stage1_gpu[T]](
        logits, block_vals, block_idxs, V,
        grid_dim=(256, 1, 1),
        block_dim=(256, 1, 1)
    )
    ctx.enqueue_function[argmax_sm75_stage2_gpu](
        block_vals, block_idxs, out_token, 256,
        grid_dim=(1, 1, 1),
        block_dim=(256, 1, 1)
    )


fn partial_rope_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    vec: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    rotary_dim: Int,
    pos: Int,
    theta_base: Float32 = 100000.0,
    stride: Int = 0
) raises:
    """Meluncurkan Rotary Positional Embedding (RoPE) parsial di VRAM."""
    ctx.enqueue_function[partial_rope_sm75_gpu[T]](
        vec, num_heads, head_dim, rotary_dim, pos, theta_base, stride,
        grid_dim=(num_heads, 1, 1),
        block_dim=(32, 1, 1)
    )


fn kv_cache_append_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    k_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_token: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_token: UnsafePointer[Scalar[T], MutAnyOrigin],
    pos: Int,
    kv_dim: Int
) raises:
    """Menyimpan token Key dan Value ke buffer KV Cache ring di VRAM."""
    var grid_x = cdiv(kv_dim, 256)
    ctx.enqueue_function[kv_cache_append_sm75_gpu[T]](
        k_cache, v_cache, k_token, v_token, pos, kv_dim,
        grid_dim=(grid_x, 1, 1),
        block_dim=(256, 1, 1)
    )


fn gqa_attention_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    q_gate: UnsafePointer[Scalar[T], MutAnyOrigin],
    k_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    v_cache: UnsafePointer[Scalar[T], MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    attn_scores: UnsafePointer[Float32, MutAnyOrigin],
    seq_len: Int,
    max_seq_len: Int,
    H_q: Int,
    H_kv: Int,
    head_dim: Int,
    scale: Float32
) raises:
    """Meluncurkan Fused GQA Scaled Dot-Product Attention & Sigmoid Gate di VRAM."""
    # GUARD: kernel GQA mengasumsikan head_dim == 256 PERSIS. Dua batas:
    #   (a) shared memory berukuran TETAP — GQA_SMEM_ELEMS = 320 float =
    #       head_dim(<=256) untuk Q + 32 reduksi warp + 32 broadcast
    #       (elementwise_sm75.mojo:807). head_dim > 256 -> penulisan smem[tid]
    #       MELUAP ke region reduksi/broadcast.
    #   (b) reduksi tahap-2 di-hardcode untuk 8 WARP: `lane < 8`
    #       (elementwise_sm75.mojo:868 dan :898, komentar "Block 256 thread =
    #       8 warp"). Dengan head_dim < 256 jumlah warp < 8, sehingga lane
    #       4..7 membaca slot smem yang TIDAK PERNAH DITULIS — nilainya
    #       sampah. Akibatnya bisa salah apa saja: max softmax bisa jadi
    #       raksasa -> seluruh exp(s-max) mendekati 0 -> denominator ~0 ->
    #       konteks jadi nol. Ini kerusakan SENYAP, bukan crash.
    # Karena itu hanya head_dim == 256 yang diterima. Generalisasi reduksi
    # (mis. jumlah warp dinamis + pola shuffle menyesuaikan) adalah pekerjaan
    # lanjutan, bukan perbaikan bug ini.
    if head_dim != 256:
        raise Error(
            "FATAL: gqa_attention_sm75_gpu hanya mendukung head_dim == 256 "
            "(SMEM tetap 320 float DAN reduksi tahap-2 di-hardcode utk 8 warp). "
            "head_dim=" + String(head_dim)
            + " akan meluap atau membaca shared memory tak-terinisialisasi "
            + "dan menghasilkan nilai salah tanpa error."
        )
    ctx.enqueue_function[gqa_attention_sm75_gpu[T]](
        q_gate, k_cache, v_cache, out_ptr, attn_scores,
        seq_len, max_seq_len, H_q, H_kv, head_dim, scale,
        grid_dim=(H_q, 1, 1),
        block_dim=(head_dim, 1, 1)
    )


fn embed_lookup_1bit_sm75_launch_on[
    T: DType
](
    mut ctx: DeviceContextGPU,
    packed: UnsafePointer[UInt8, MutAnyOrigin],
    scales: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    out_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    token_id: Int,
    D: Int
) raises:
    """Meluncurkan dekuantisasi 1-bit affine embedding langsung di VRAM."""
    var grid_x = cdiv(D, 256)
    ctx.enqueue_function[embed_lookup_1bit_sm75_gpu[T]](
        packed, scales, biases, out_ptr, token_id, D,
        grid_dim=(grid_x, 1, 1),
        block_dim=(256, 1, 1)
    )

