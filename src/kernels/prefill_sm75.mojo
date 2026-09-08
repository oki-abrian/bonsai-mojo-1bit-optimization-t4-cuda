# ===----------------------------------------------------------------------=== #
# Module: kernels/prefill_sm75.mojo
# Purpose: GPU Kernel Prefill W1A16 g128 untuk NVIDIA T4 (sm_75)
#          Tiled CTA (BM=64, BN=32, BK=64) dengan Shared Memory staging,
#          dekuantisasi branchless on-the-fly, dan FP32 accumulator.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, stack_allocation, AddressSpace
from sys import size_of
from ..common import (
    PREFILL_BM, PREFILL_BN, PREFILL_BK, PREFILL_PAD, PREFILL_THREADS,
    GROUP_SIZE, WEIGHT_PACK_FACTOR, get_scale_stride, block_barrier
)
from ..dequant import extract_bit_lsb, dequant_affine_bonsai

# ----------------------------------------------------------------------------
# 1. Prefill Device Kernel: Tiled Direct-Dot GEMM (M > 1)
# ----------------------------------------------------------------------------
fn qmm_sm75_b1_kernel_body[
    T: DType
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    w_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    scales_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int,
    broadcast_w: Bool,
    block_idx_x: Int, block_idx_y: Int, block_idx_z: Int,
    thread_idx_x: Int,
    shared_mem_a: UnsafePointer[Scalar[T], MutAnyOrigin]
):
    """
    Logika device kernel prefill yang mengeksekusi satu thread CTA.
    Menggunakan tile BM=64 baris x BN=32 kolom, di-loop sepanjang K dengan step BK=64.
    Aktivasi x ditahapkan dalam Shared Memory dengan padding anti-bank-conflict.
    """
    var block_m = block_idx_y * PREFILL_BM
    var block_n = block_idx_x * PREFILL_BN
    var ly = block_idx_z

    var tid = thread_idx_x
    var nthreads = PREFILL_THREADS

    var scale_stride = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    # Offset batch bobot dan skala (mendukung mode broadcast_w)
    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * scale_stride

    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset

    # Stride baris untuk shared memory As dengan padding anti-bank-conflict
    var smem_stride_k = PREFILL_BK + PREFILL_PAD

    # Setiap thread bertanggung jawab atas satu atau lebih pasangan sel (r, c)
    var idx = tid
    while idx < PREFILL_BM * PREFILL_BN:
        var r = idx // PREFILL_BN
        var c = idx % PREFILL_BN
        var gm = block_m + r
        var gn = block_n + c

        # PENTING: Thread non-valid tetap wajib melewati loop K dan seluruh barrier
        # agar tidak terjadi barrier deadlock pada arsitektur SIMT T4!
        var valid = (gm < M) and (gn < N)

        var acc: Float32 = 0.0
        var wrow = wbase + (gn * weight_row_bytes) if valid else UnsafePointer[UInt8, MutAnyOrigin]()

        var k_base = 0
        while k_base < K:
            # ---- 1. Pemuatan Kooperatif Aktivasi x ke Shared Memory ----
            var sidx2 = tid
            while sidx2 < PREFILL_BM * PREFILL_BK:
                var sr = sidx2 // PREFILL_BK
                var sc = sidx2 % PREFILL_BK
                var gmk = block_m + sr
                var gkk = k_base + sc

                var val = x_ptr[(ly * M + gmk) * K + gkk] if (gmk < M and gkk < K) else Scalar[T](0)
                shared_mem_a[sr * smem_stride_k + sc] = val
                sidx2 += nthreads

            # Inisialisasi deterministik seluruh tile SMEM — WAJIB untuk mode
            # simulasi host sekuensial: TANPA syarat tid==0, karena pada
            # eksekusi sekuensial thread>0 membaca SMEM basi milik thread
            # sebelumnya (tile K-pass terakhirnya) bila init bersyarat.
            # Setiap thread menahapkan ulang tile penuh untuk dirinya sendiri.
            # (Varian GPU native memakai staging kooperatif + barrier hardware.)
            for s in range(PREFILL_BM * PREFILL_BK):
                var sr = s // PREFILL_BK
                var sc = s % PREFILL_BK
                var gmk = block_m + sr
                var gkk = k_base + sc
                var val = x_ptr[(ly * M + gmk) * K + gkk] if (gmk < M and gkk < K) else Scalar[T](0)
                shared_mem_a[sr * smem_stride_k + sc] = val

            # SINKRONISASI 1: Pastikan seluruh tile x telah termuat di SMEM
            block_barrier()
            
            # ---- 2. Loop Kontraksi K (Unrolled Step) ----
            for kk in range(PREFILL_BK):
                var gk = k_base + kk
                if valid and (gk < K):
                    # Ambil skala grup: group_size = 128 (gk >> 7)
                    var group_idx = gk >> 7
                    var sg = Float32(sbase[gn * scale_stride + group_idx])

                    # Ekstraksi byte bobot dan bit LSB-first
                    var byte_offset = gk >> 3
                    var byte_val = wrow[byte_offset]
                    var bit_val = extract_bit_lsb(byte_val, gk)

                    # Ambil aktivasi dari shared memory
                    var xv = Float32(shared_mem_a[r * smem_stride_k + kk])

                    # FMA Affine: w_eff = (2 * bit - 1) * sg
                    var w_eff = dequant_affine_bonsai(bit_val, sg)
                    acc += w_eff * xv

            # SINKRONISASI 2: Tunggu semua thread selesai membaca As sebelum
            # tile berikutnya menimpa SMEM. No-op pada host CPU;
            # barrier hardware saat eksekusi GPU device.
            block_barrier()

            k_base += PREFILL_BK

        # Simpan hasil akumulasi FP32 ke memori global y (dikonversi ke tipe T)
        if valid:
            var out_idx = (ly * M + gm) * N + gn
            y_ptr[out_idx] = Scalar[T](acc)

        idx += nthreads

# ----------------------------------------------------------------------------
# 2. Native GPU Device Kernel: Tiled Direct-Dot GEMM (Tesla T4 SM75 Hardware)
# ----------------------------------------------------------------------------
fn qmm_sm75_b1_prefill_gpu[
    T: DType
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    w_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    scales_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    M: Int, N: Int, K: Int, L: Int,
    broadcast_w: Bool
):
    """
    Device kernel native prefill untuk eksekusi langsung di hardware NVIDIA Tesla T4.
    Mengambil thread_idx dan block_idx langsung dari hardware register GPU.
    Sinkronisasi menggunakan hardware barrier CUDA.
    Shared Memory dialokasikan DI DALAM kernel pada address space SHARED
    (stack_allocation) — pointer heap host tidak pernah dipass ke device.
    """
    from gpu.id import thread_idx, block_idx
    from gpu import barrier

    # Alokasi SMEM device internal: tile As BM x (BK + PAD) elemen Scalar[T]
    # dengan padding anti-bank-conflict (64 x 72 = 4608 elemen).
    var shared_mem_a = stack_allocation[
        PREFILL_BM * (PREFILL_BK + PREFILL_PAD),
        Scalar[T],
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var bx = block_idx.x
    var by = block_idx.y
    var bz = block_idx.z

    var block_m = by * PREFILL_BM
    var block_n = bx * PREFILL_BN
    var ly = bz

    var scale_stride = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * scale_stride

    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset
    var smem_stride_k = PREFILL_BK + PREFILL_PAD

    var idx = tid
    while idx < PREFILL_BM * PREFILL_BN:
        var r = idx // PREFILL_BN
        var c = idx % PREFILL_BN
        var gm = block_m + r
        var gn = block_n + c

        var valid = (gm < M) and (gn < N)
        var acc: Float32 = 0.0
        var wrow = wbase + (gn * weight_row_bytes) if valid else UnsafePointer[UInt8, MutAnyOrigin]()

        var k_base = 0
        while k_base < K:
            # Pemuatan kooperatif SIMT murni paralel
            var sidx2 = tid
            while sidx2 < PREFILL_BM * PREFILL_BK:
                var sr = sidx2 // PREFILL_BK
                var sc = sidx2 % PREFILL_BK
                var gmk = block_m + sr
                var gkk = k_base + sc

                var val = x_ptr[(ly * M + gmk) * K + gkk] if (gmk < M and gkk < K) else Scalar[T](0)
                shared_mem_a[sr * smem_stride_k + sc] = val
                sidx2 += PREFILL_THREADS

            barrier()

            for kk in range(PREFILL_BK):
                var gk = k_base + kk
                if valid and (gk < K):
                    var group_idx = gk >> 7
                    var sg = Float32(sbase[gn * scale_stride + group_idx])
                    var byte_offset = gk >> 3
                    var byte_val = wrow[byte_offset]
                    var bit_val = extract_bit_lsb(byte_val, gk)
                    var xv = Float32(shared_mem_a[r * smem_stride_k + kk])
                    var w_eff = dequant_affine_bonsai(bit_val, sg)
                    acc += w_eff * xv

            barrier()
            k_base += PREFILL_BK

        if valid:
            var out_idx = (ly * M + gm) * N + gn
            y_ptr[out_idx] = Scalar[T](acc)

        idx += PREFILL_THREADS
