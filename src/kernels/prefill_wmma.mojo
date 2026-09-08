# ===----------------------------------------------------------------------=== #
# Module: kernels/prefill_wmma.mojo
# Purpose: PREFILL PRODUKSI W1A16 g128 (M >= 8) — port qmm_impl_sm75_wmma.cuh
#          (WMMA v2 Optimized), kernel yang benar-benar dieksekusi wheel MLX
#          CUDA referensi (qmm_impl_sm75_b1.cu:5 meng-include file ini).
#          Tiling BM=64 / BN=64 / BK=64, 128 thread (4 warp x sub-tile 32x32),
#          As[64][72] + Wt[64][72] TRANSPOSED ([k][n]) dengan PAD=8 anti
#          bank-conflict, dequant uint32 coalesced -> (2q-1)*s (kontrak Bonsai
#          b=-s), epilog ter-guard.
#          Catatan port: eksekusi sub-tile memakai FMA skalar per-lane-1-baris
#          (deterministik, API terverifikasi); upgrade tensor-core via
#          gpu.compute.mma adalah langkah lanjut pasca-kompilasi pertama.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, stack_allocation, AddressSpace
from ..common import (
    GROUP_SIZE, WEIGHT_PACK_FACTOR, get_scale_stride, block_barrier
)
from ..dequant import dequant_affine_bonsai

# Geometri WMMA v2 (paritas konstanta wmma_b1)
alias WMMA_BM: Int = 64
alias WMMA_BN: Int = 64
alias WMMA_BK: Int = 64
alias WMMA_PAD: Int = 8
alias WMMA_THREADS: Int = 128
alias WMMA_SMEM_ELEMS: Int = WMMA_BM * (WMMA_BK + WMMA_PAD) + WMMA_BK * (WMMA_BN + WMMA_PAD)

# ----------------------------------------------------------------------------
# 1. Prefill WMMA v2 Body (Host-Sim Sequential per-tid)
# ----------------------------------------------------------------------------
fn qmm_wmma_b1_body[
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
    smem_as: UnsafePointer[Scalar[T], MutAnyOrigin],
    smem_wt: UnsafePointer[Scalar[T], MutAnyOrigin]
):
    """
    Body host-sim prefill produksi (dieksekusi sekuensial per-tid):
    - Staging As[BM][BK+PAD] (aktivasi, chunk 8 elemen, zero-fill penuh) dan
      Wt[BK][BN+PAD] TRANSPOSED hasil dequant uint32 coalesced (2 bobot-bit
      per ekstraksi, w_eff = (2q-1)*s).
    - Inisialisasi deterministik tid==0 menutup seluruh As + Wt (WAJIB mode
      host sekuensial; dihapus saat kompilasi GPU device).
    - Komputasi: warp wid memiliki sub-tile 32x32 (wm, wn); tiap lane
      menghitung 1 baris output penuh 32 kolom dengan akumulasi FP32.
    """
    var tid = thread_idx_x
    var ly = block_idx_z
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    var block_n = block_idx_x * WMMA_BN
    var block_m = block_idx_y * WMMA_BM

    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row
    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset

    # ---- Loop K per tile BK=64 ----
    # Akumulator per-lane persisten lintas K-pass (1 lane = 1 baris output,
    # 32 kolom per sub-tile warp 32x32). Store ke y HANYA setelah loop K
    # selesai — store di dalam loop menimpa y dengan partial satu pass.
    var warp_id = tid // 32
    var lane_id = tid % 32
    var wm = (warp_id // 2) * 32
    var wn = (warp_id % 2) * 32
    var lane_gm = block_m + wm + lane_id
    var acc_row = InlineArray[Float32, 32](0.0)
    var k_base = 0
    while k_base < K:
        # 1a. Staging kooperatif As
        var total_vec_a = (WMMA_BM * WMMA_BK) // 8
        var idx = tid
        while idx < total_vec_a:
            var r = idx // (WMMA_BK // 8)
            var c = (idx % (WMMA_BK // 8)) * 8
            var gm = block_m + r
            var gk = k_base + c
            var dst = r * (WMMA_BK + WMMA_PAD) + c
            if (gm < M) and (gk + 8 <= K):
                var v = x_ptr.load[width=8]((ly * M + gm) * K + gk)
                smem_as.store[width=8](dst, v)
            else:
                for j in range(8):
                    var gk_j = gk + j
                    smem_as[dst + j] = x_ptr[(ly * M + gm) * K + gk_j] if ((gm < M) and (gk_j < K)) else Scalar[T](0)
            idx += WMMA_THREADS

        # 1b. Inisialisasi deterministik As penuh (mode host sekuensial)
# (init SMEM penuh dijalankan TANPA syarat tid — host-sim sekuensial)
        for i in range(WMMA_BM * WMMA_BK):
            var r = i // WMMA_BK
            var c = i % WMMA_BK
            var gm = block_m + r
            var gk = k_base + c
            smem_as[r * (WMMA_BK + WMMA_PAD) + c] = x_ptr[(ly * M + gm) * K + gk] if ((gm < M) and (gk < K)) else Scalar[T](0)

        # 1c. Staging kooperatif Wt[BK][BN+PAD] transposed: 128 thread x 1 uint32
        #     (r = baris N lokal, wc = sub-kolom K 32 bobot)
        var r_w = tid // 2
        var wc_u32 = tid % 2
        var gn_w = block_n + r_w
        var gk_w = k_base + wc_u32 * 32
        var raw_u32: UInt32 = 0
        var s_val: Scalar[T] = Scalar[T](0)
        if (gn_w < N) and (gk_w < K):
            var byte_offset = gn_w * weight_row_bytes + gk_w // 8
            if (byte_offset % 4 == 0) and (gk_w + 32 <= K):
                var wv = wbase.load[width=4](byte_offset)
                raw_u32 = UInt32(wv[0]) | (UInt32(wv[1]) << 8) | (UInt32(wv[2]) << 16) | (UInt32(wv[3]) << 24)
            else:
                for b in range(4):
                    if gk_w + b * 8 < K:
                        raw_u32 |= UInt32(wbase[byte_offset + b]) << (b * 8)
            s_val = sbase[gn_w * groups_per_row + gk_w // GROUP_SIZE]

        for p in range(16):
            var k_offset = wc_u32 * 32 + p * 2
            var gk_cur = k_base + k_offset
            var pairbits = (Int(raw_u32) >> (p * 2)) & 3
            var w_lo = dequant_affine_bonsai(Float32(pairbits & 1), Float32(s_val))
            var w_hi = dequant_affine_bonsai(Float32((pairbits >> 1) & 1), Float32(s_val))
            var gk_ok = (gn_w < N) and (gk_cur < K)
            smem_wt[k_offset * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](w_lo) if gk_ok else Scalar[T](0)
            smem_wt[(k_offset + 1) * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](w_hi) if gk_ok else Scalar[T](0)

        # 1d. Inisialisasi deterministik Wt penuh (mode host sekuensial) —
        #     TANPA syarat tid: setiap thread menahapkan Wt penuh untuk dirinya.
        for t in range(WMMA_THREADS):
            var r0 = t // 2
            var wc0 = t % 2
            var gn0 = block_n + r0
            var gk0 = k_base + wc0 * 32
            var raw0: UInt32 = 0
            var s0: Scalar[T] = Scalar[T](0)
            if (gn0 < N) and (gk0 < K):
                var boff0 = gn0 * weight_row_bytes + gk0 // 8
                for b in range(4):
                    if gk0 + b * 8 < K:
                        raw0 |= UInt32(wbase[boff0 + b]) << (b * 8)
                s0 = sbase[gn0 * groups_per_row + gk0 // GROUP_SIZE]
            for p in range(16):
                var k_off0 = wc0 * 32 + p * 2
                var gk_ok0 = (gn0 < N) and (k_base + k_off0 < K)
                var pb0 = (Int(raw0) >> (p * 2)) & 3
                var wlo0 = dequant_affine_bonsai(Float32(pb0 & 1), Float32(s0))
                var whi0 = dequant_affine_bonsai(Float32((pb0 >> 1) & 1), Float32(s0))
                smem_wt[k_off0 * (WMMA_BN + WMMA_PAD) + r0] = Scalar[T](wlo0) if gk_ok0 else Scalar[T](0)
                smem_wt[(k_off0 + 1) * (WMMA_BN + WMMA_PAD) + r0] = Scalar[T](whi0) if gk_ok0 else Scalar[T](0)

        # SINKRONISASI 1: As + Wt siap untuk seluruh thread block.
        # No-op pada host CPU; barrier hardware saat eksekusi GPU device.
        block_barrier()

        # ---- 2. Komputasi sub-tile warp 32x32: akumulasi lintas K-pass ----
        if lane_gm < M:
            var arow = (wm + lane_id) * (WMMA_BK + WMMA_PAD)
            for j in range(32):
                var gn = block_n + wn + j
                if gn < N:
                    for kk in range(WMMA_BK):
                        var gk = k_base + kk
                        if gk < K:
                            var xv = Float32(smem_as[arow + kk])
                            var wv = Float32(smem_wt[kk * (WMMA_BN + WMMA_PAD) + (wn + j)])
                            acc_row[j] += xv * wv

        # SINKRONISASI 2: semua pembacaan selesai sebelum tile berikutnya.
        block_barrier()
        k_base += WMMA_BK

    # Store hasil akumulasi penuh (sekali, setelah seluruh K-pass)
    if lane_gm < M:
        for j in range(32):
            var gn = block_n + wn + j
            if gn < N:
                y_ptr[(ly * M + lane_gm) * N + gn] = Scalar[T](acc_row[j])

# ----------------------------------------------------------------------------
# 2. Native GPU Device Kernel: Prefill WMMA v2 (Tesla T4 SM75 Hardware)
# ----------------------------------------------------------------------------
fn qmm_wmma_b1_gpu[
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
    Device kernel native prefill produksi (paritas qmm_impl_sm75_wmma.cuh)
    untuk hardware NVIDIA Tesla T4. thread_idx/block_idx dari register
    hardware GPU; barrier hardware CUDA; Shared Memory dialokasi DI DALAM
    kernel (AddressSpace.SHARED, 9216 elemen T ~= 18 KiB fp16). Struktur
    tiling/staging identik referensi; eksekusi sub-tile 32x32 per warp
    memakai 1 lane = 1 baris (FMA skalar FP32) — upgrade tensor-core
    gpu.compute.mma adalah langkah lanjut pasca-kompilasi pertama.
    """
    from gpu.id import thread_idx, block_idx
    from gpu import barrier

    var smem_as = stack_allocation[
        WMMA_BM * (WMMA_BK + WMMA_PAD),
        Scalar[T],
        address_space = AddressSpace.SHARED
    ]()
    var smem_wt = stack_allocation[
        WMMA_BK * (WMMA_BN + WMMA_PAD),
        Scalar[T],
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var ly = block_idx.z
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    var block_n = block_idx.x * WMMA_BN
    var block_m = block_idx.y * WMMA_BM

    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row
    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset

    # Akumulator per-lane persisten lintas K-pass (store ke y setelah loop K)
    var warp_id = tid // 32
    var lane_id = tid % 32
    var wm = (warp_id // 2) * 32
    var wn = (warp_id % 2) * 32
    var lane_gm = block_m + wm + lane_id
    var acc_row = InlineArray[Float32, 32](0.0)

    var k_base = 0
    while k_base < K:
        # 1a. Staging kooperatif As (chunk 8 elemen, zero-fill)
        var total_vec_a = (WMMA_BM * WMMA_BK) // 8
        var idx = tid
        while idx < total_vec_a:
            var r = idx // (WMMA_BK // 8)
            var c = (idx % (WMMA_BK // 8)) * 8
            var gm = block_m + r
            var gk = k_base + c
            var dst = r * (WMMA_BK + WMMA_PAD) + c
            if (gm < M) and (gk + 8 <= K):
                var v = x_ptr.load[width=8]((ly * M + gm) * K + gk)
                smem_as.store[width=8](dst, v)
            else:
                for j in range(8):
                    var gk_j = gk + j
                    smem_as[dst + j] = x_ptr[(ly * M + gm) * K + gk_j] if ((gm < M) and (gk_j < K)) else Scalar[T](0)
            idx += WMMA_THREADS

        # 1b. Staging kooperatif Wt transposed (128 thread x 1 uint32)
        var r_w = tid // 2
        var wc_u32 = tid % 2
        var gn_w = block_n + r_w
        var gk_w = k_base + wc_u32 * 32
        var raw_u32: UInt32 = 0
        var s_val: Scalar[T] = Scalar[T](0)
        if (gn_w < N) and (gk_w < K):
            var byte_offset = gn_w * weight_row_bytes + gk_w // 8
            if (byte_offset % 4 == 0) and (gk_w + 32 <= K):
                var wv = wbase.load[width=4](byte_offset)
                raw_u32 = UInt32(wv[0]) | (UInt32(wv[1]) << 8) | (UInt32(wv[2]) << 16) | (UInt32(wv[3]) << 24)
            else:
                for b in range(4):
                    if gk_w + b * 8 < K:
                        raw_u32 |= UInt32(wbase[byte_offset + b]) << (b * 8)
            s_val = sbase[gn_w * groups_per_row + gk_w // GROUP_SIZE]

        for p in range(16):
            var k_offset = wc_u32 * 32 + p * 2
            var gk_cur = k_base + k_offset
            var gk_ok = (gn_w < N) and (gk_cur < K)
            if gk_ok:
                var pairbits = (Int(raw_u32) >> (p * 2)) & 3
                var s_f = Float32(s_val)
                smem_wt[k_offset * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](dequant_affine_bonsai(Float32(pairbits & 1), s_f))
                smem_wt[(k_offset + 1) * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](dequant_affine_bonsai(Float32((pairbits >> 1) & 1), s_f))
            else:
                smem_wt[k_offset * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](0)
                smem_wt[(k_offset + 1) * (WMMA_BN + WMMA_PAD) + r_w] = Scalar[T](0)

        # Barrier hardware GPU T4 (__syncthreads): As + Wt visible
        barrier()

        # 2. Komputasi sub-tile warp 32x32: akumulasi lintas K-pass
        if lane_gm < M:
            var arow = (wm + lane_id) * (WMMA_BK + WMMA_PAD)
            for j in range(32):
                var gn = block_n + wn + j
                if gn < N:
                    for kk in range(WMMA_BK):
                        var gk = k_base + kk
                        if gk < K:
                            acc_row[j] += Float32(smem_as[arow + kk]) * Float32(smem_wt[kk * (WMMA_BN + WMMA_PAD) + (wn + j)])

        # Semua pembacaan selesai sebelum tile berikutnya me-restage
        barrier()
        k_base += WMMA_BK

    # Store hasil akumulasi penuh (sekali, setelah seluruh K-pass)
    if lane_gm < M:
        for j in range(32):
            var gn = block_n + wn + j
            if gn < N:
                y_ptr[(ly * M + lane_gm) * N + gn] = Scalar[T](acc_row[j])
