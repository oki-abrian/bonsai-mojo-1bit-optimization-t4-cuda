# ===----------------------------------------------------------------------=== #
# Module: kernels/direct_smallm.mojo
# Purpose: Prefill Register-M Direct-Dot (8 < M <= 64) paritas
#          qmv_direct_smallm_kernel referensi:
#          grid (ceil(N/8), 1, L) dengan 1 warp <-> 1 baris output N,
#          akumulator register per-mi, staging x SMEM per K-tile
#          (GS 4/2, stride grup 128 -> 136), segmen bobot WPL per lane
#          (LSB-first), split-K deterministik gate 40 / target 80, dan
#          reduksi deterministik tanpa atomicAdd.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, stack_allocation, AddressSpace
from collections import InlineArray
from sys import size_of
from ..common import (
    GROUP_SIZE, WEIGHT_PACK_FACTOR, get_scale_stride,
    DECODE_THREADS, DIRECT_ROWS_PER_BLOCK, DIRECT_MAX_M, DIRECT_GRP_PAD,
    DIRECT_SMEM_ELEMS, direct_gs, block_barrier
)
from ..dequant import dequant_affine_bonsai

# ----------------------------------------------------------------------------
# 1. Direct Small-M Body (Host-Sim Sequential per-tid)
# ----------------------------------------------------------------------------
fn qmv_direct_smallm_body[
    T: DType
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    w_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    scales_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    ws_ptr: UnsafePointer[Float32, MutAnyOrigin],
    use_ws: Bool,
    slice_idx: Int,
    g_begin: Int,
    g_count: Int,
    M: Int, N: Int, K: Int, L: Int,
    broadcast_w: Bool,
    PAD: Int,
    block_idx_x: Int, block_idx_z: Int,
    thread_idx_x: Int,
    smem_x: UnsafePointer[Scalar[T], MutAnyOrigin],
    smem_red: UnsafePointer[Float32, MutAnyOrigin]
):
    """
    Body host-sim jalur direct (dieksekusi sekuensial per-tid oleh ops.mojo):
    - 1 warp menghitung 1 baris output N; 32 lane memegang segmen bobot grup.
    - Staging x: PAD x K_TILE elemen T per K-tile dalam chunk 8 elemen
      (e = tid*8, stride 2048), zero-fill penuh untuk mi >= M / gk >= K.
    - Dot: tiap lane mengakumulasi WPL bobot segmennya ke seluruh PAD baris m
      via dequant (2q-1)*s — kontrak Bonsai b = -s terserap tanpa bias tensor.
    - Reduksi host: scratchpad SMEM + block_barrier (padanan hardware dari
      shuffle full-warp yang dipakai kernel GPU).
    """
    var tid = thread_idx_x
    var ly = block_idx_z
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR
    var gs = direct_gs(PAD)            # 4 (PAD <= 32) atau 2 (PAD = 64)
    var k_tile = gs * GROUP_SIZE
    var lpg = 32 // gs                 # lanes per group: 8 atau 16
    var wpl = GROUP_SIZE // lpg        # weights per lane: 16 atau 8
    var wbytes = wpl // 8              # 2 atau 1

    var warp_id = tid // 32
    var lane_id = tid % 32
    var seg = lane_id // lpg           # grup tile milik segmen lane ini
    var sub = lane_id % lpg            # posisi bobot dalam grup

    var row = block_idx_x * DIRECT_ROWS_PER_BLOCK + warp_id
    var valid_row = (row < N)

    # Offset batch (supports broadcast_w)
    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row
    var wrow = w_ptr + w_layer_offset + (row * weight_row_bytes if valid_row else 0)
    var srow = scales_ptr + s_layer_offset + (row * groups_per_row if valid_row else 0)

    # Akumulator register per-baris-m (hanya indeks 0..PAD-1 yang dipakai)
    var acc = InlineArray[Float32, DIRECT_MAX_M](0.0)

    var g_end = g_begin + g_count
    var g0 = g_begin
    while g0 < g_end:
        var kc = g0 * GROUP_SIZE
        # ---- 1. Staging x: PAD x K_TILE elemen, chunk 8 elemen per iterasi ----
        # Seluruh slot kolom valid ditulis setiap iterasi (zero-fill saat out
        # of range); gap padding 128..135 tidak pernah ditulis maupun dibaca.
        # Host-sim sekuensial: staging MULAI dari 0 (penuh per thread) — bukan
        # kooperatif — agar tidak ada thread yang membaca SMEM basi lintas tid.
        var e = 0
        while e < PAD * k_tile:
            var mi = e // k_tile
            var lc = e % k_tile
            var gi = lc // GROUP_SIZE
            var off = lc % GROUP_SIZE
            var dst = mi * (gs * DIRECT_GRP_PAD) + gi * DIRECT_GRP_PAD + off
            if (mi < M) and (kc + lc < K):
                var v = x_ptr.load[width=8]((ly * M + mi) * K + kc + lc)
                smem_x.store[width=8](dst, v)
            else:
                for j in range(8):
                    smem_x[dst + j] = Scalar[T](0)
            e += DECODE_THREADS * 8

        # SINKRONISASI 1: x_s visible untuk seluruh thread block.
        # No-op pada host CPU; barrier hardware saat eksekusi GPU device.
        block_barrier()

        # ---- 2. Dot segmen: 1 grup g128 milik segmen, WPL bobot per lane ----
        var g = g0 + seg
        var g_ok = valid_row and (g < groups_per_row) and (g < g_end) and (g >= g_begin)
        var wbits: UInt32 = 0
        var sf: Float32 = 0.0
        if g_ok:
            var boff = g * (GROUP_SIZE // WEIGHT_PACK_FACTOR) + sub * wbytes
            # WBYTES = 2 (PAD 16/32) atau 1 (PAD 64); LSB-first per byte.
            # Grup penuh g128 menjamin boff + wbytes tidak melewati baris.
            if wbytes == 2:
                var hw = wrow.load[width=2](boff)
                wbits = UInt32(hw[0]) | (UInt32(hw[1]) << 8)
            else:
                wbits = UInt32(wrow[boff])
            sf = Float32(srow[g])
        var col0 = seg * DIRECT_GRP_PAD + sub * wpl
        for mi in range(PAD):
            var xbase = mi * (gs * DIRECT_GRP_PAD) + col0
            var a: Float32 = acc[mi]
            for i in range(wpl):
                var xv = Float32(smem_x[xbase + i])
                var bit = (Int(wbits) >> i) & 1
                a += xv * dequant_affine_bonsai(Float32(bit), sf)
            acc[mi] = a

        # SINKRONISASI 2: semua pembacaan selesai sebelum tile berikutnya
        # me-restage x_s. No-op pada host CPU; barrier hardware di GPU.
        block_barrier()
        g0 += gs

    # ---- 3. Reduksi full-warp deterministik (paritas shfl 16..1, lane 0) ----
    for mi in range(PAD):
        smem_red[tid] = acc[mi]

        # SINKRONISASI 3: tulisan seluruh lane terlihat sebelum lane 0 menjumlah.
        # No-op pada host CPU; barrier hardware saat eksekusi GPU device.
        block_barrier()

        if valid_row and (lane_id == 0) and (mi < M):
            var total_row_acc: Float32 = 0.0
            for t in range(32):
                total_row_acc += smem_red[tid + t]
            if use_ws:
                ws_ptr[slice_idx * L * M * N + ly * M * N + mi * N + row] = total_row_acc
            else:
                var out_idx = (ly * M + mi) * N + row
                y_ptr[out_idx] = Scalar[T](total_row_acc)

# ----------------------------------------------------------------------------
# 2. Native GPU Device Kernel: Direct Small-M (Tesla T4 SM75 Hardware)
# ----------------------------------------------------------------------------
fn qmv_direct_smallm_gpu[
    T: DType
](
    x_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    w_ptr: UnsafePointer[UInt8, MutAnyOrigin],
    scales_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    ws_ptr: UnsafePointer[Float32, MutAnyOrigin],
    use_ws: Bool,
    slice_idx: Int,
    g_begin: Int,
    g_count: Int,
    M: Int, N: Int, K: Int, L: Int,
    broadcast_w: Bool,
    PAD: Int
):
    """
    Device kernel native direct small-m untuk hardware NVIDIA Tesla T4.
    thread_idx/block_idx dari register hardware GPU; barrier hardware CUDA;
    Shared Memory dialokasi DI DALAM kernel pada address space SHARED
    (stack_allocation). Reduksi memakai shuffle_down full-warp hardware
    (offset 16..1, lane 0 penulis tunggal) — tanpa atomicAdd.
    Kontrak referensi: T harus fp16/bf16 (staging SMEM 17408 x 2 byte).
    """
    from gpu.id import thread_idx, block_idx
    from gpu import barrier
    from gpu.primitives.warp import shuffle_down

    constrained[size_of[T]() == 2, "direct_smallm GPU: T harus fp16/bf16 (paritas referensi sm75)"]()

    var smem_x = stack_allocation[
        DIRECT_SMEM_ELEMS,
        Scalar[T],
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var ly = block_idx.z
    var bx = block_idx.x
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR
    var gs = direct_gs(PAD)
    var k_tile = gs * GROUP_SIZE
    var lpg = 32 // gs
    var wpl = GROUP_SIZE // lpg
    var wbytes = wpl // 8

    var warp_id = tid // 32
    var lane_id = tid % 32
    var seg = lane_id // lpg
    var sub = lane_id % lpg

    var row = bx * DIRECT_ROWS_PER_BLOCK + warp_id
    var valid_row = (row < N)

    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row
    var wrow = w_ptr + w_layer_offset + (row * weight_row_bytes if valid_row else 0)
    var srow = scales_ptr + s_layer_offset + (row * groups_per_row if valid_row else 0)

    var acc = InlineArray[Float32, DIRECT_MAX_M](0.0)

    var g_end = g_begin + g_count
    var g0 = g_begin
    while g0 < g_end:
        var kc = g0 * GROUP_SIZE
        var e = tid * 8
        while e < PAD * k_tile:
            var mi = e // k_tile
            var lc = e % k_tile
            var gi = lc // GROUP_SIZE
            var off = lc % GROUP_SIZE
            var dst = mi * (gs * DIRECT_GRP_PAD) + gi * DIRECT_GRP_PAD + off
            if (mi < M) and (kc + lc < K):
                var v = x_ptr.load[width=8]((ly * M + mi) * K + kc + lc)
                smem_x.store[width=8](dst, v)
            else:
                for j in range(8):
                    smem_x[dst + j] = Scalar[T](0)
            e += DECODE_THREADS * 8

        # Barrier hardware GPU T4 (__syncthreads): x_s visible
        barrier()

        var g = g0 + seg
        var g_ok = valid_row and (g < groups_per_row) and (g < g_end) and (g >= g_begin)
        var wbits: UInt32 = 0
        var sf: Float32 = 0.0
        if g_ok:
            var boff = g * (GROUP_SIZE // WEIGHT_PACK_FACTOR) + sub * wbytes
            if wbytes == 2:
                var hw = wrow.load[width=2](boff)
                wbits = UInt32(hw[0]) | (UInt32(hw[1]) << 8)
            else:
                wbits = UInt32(wrow[boff])
            sf = Float32(srow[g])
        var col0 = seg * DIRECT_GRP_PAD + sub * wpl
        for mi in range(PAD):
            var xbase = mi * (gs * DIRECT_GRP_PAD) + col0
            var a: Float32 = acc[mi]
            for i in range(wpl):
                var xv = Float32(smem_x[xbase + i])
                var bit = (Int(wbits) >> i) & 1
                a += xv * dequant_affine_bonsai(Float32(bit), sf)
            acc[mi] = a

        # Semua pembacaan selesai sebelum tile berikutnya me-restage x_s
        barrier()
        g0 += gs

    # Reduksi full-warp hardware shuffle (offset 16..1, lane 0 penulis)
    for mi in range(PAD):
        var v: Float32 = acc[mi]
        var off = 16
        while off > 0:
            v += shuffle_down(v, UInt32(off))
            off = off // 2
        if valid_row and (lane_id == 0) and (mi < M):
            if use_ws:
                ws_ptr[slice_idx * L * M * N + ly * M * N + mi * N + row] = v
            else:
                var out_idx = (ly * M + mi) * N + row
                y_ptr[out_idx] = Scalar[T](v)
