# ===----------------------------------------------------------------------=== #
# Module: kernels/decode_sm75.mojo
# Purpose: GPU Kernel Decode W1A16 g128 untuk NVIDIA T4 (sm_75)
#          Paritas penuh qmv_vec_splitk_kernel: dimensi M di dalam block
#          (bucket MPAD 1/2/4/8), split-K deterministik fill-320, staging
#          skala ke SMEM, pemuatan bobot SIMD 16-byte (1 grup g128 per lane
#          per tile), SMEM pad 128->132 anti bank-conflict, dan reduksi
#          kooperatif deterministik tanpa atomicAdd.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, stack_allocation, AddressSpace, bitcast
from collections import InlineArray
from sys import size_of
from ..common import (
    GROUP_SIZE, WEIGHT_PACK_FACTOR, get_scale_stride,
    DECODE_THREADS, DECODE_ROWS_PER_BLOCK,
    DECODE_LANES_PER_ROW, DECODE_GS, DECODE_K_TILE, DECODE_GRP_PAD,
    DECODE_MPAD_MAX, block_barrier,
    QMV_NIB_BITS, QMV_NIB_PER_GRP, QMV_NIB_ENT, QMV_NIB_ENT_PAD,
    QMV_NIB_ELEMS, DECODE_NIB_SMEM_ELEMS
)
from ..dequant import extract_bit_lsb, dequant_affine_bonsai, unpack_byte_to_simd8

# ----------------------------------------------------------------------------
# 1. Decode Device Kernel: Vectorized GEMV Split-K (M = 1, Nibble LUT)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 1. Decode Device Kernel: Vectorized GEMV Split-K (M <= 8)
# ----------------------------------------------------------------------------
fn qmv_sm75_b1_decode_body[
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
    MPAD: Int,
    block_idx_x: Int, block_idx_z: Int,
    thread_idx_x: Int,
    smem_x: UnsafePointer[Float32, MutAnyOrigin]
):
    """
    Logika device kernel decode token-generation (paritas qmv_vec_splitk_kernel):
    - 1 block memproses 32 baris output N (8 warp x 4 baris/warp).
    - Dimensi M berada DI DALAM block: seluruh MPAD baris m diakumulasi
      per-thread, sehingga bobot hanya di-stream 1x per block.
    - Slice split-K: block hanya menyapu grup g128 [g_begin, g_begin + g_count);
      hasil parsial FP32 ditulis ke workspace (use_ws) atau langsung ke y.
    - Skala baris milik block di-stage ke SMEM (s_s 32x8 FP32) per K-tile.
    - Bobot dimuat SIMD 16-byte (128 bobot = 1 grup g128 penuh per lane).
    - Aktivasi x ditahapkan SMEM FP32 per-mi dengan padding 128 -> 132 float.
    - Reduksi kooperatif intra-warp deterministik (zero atomicAdd).
    """
    var tid = thread_idx_x
    var ly = block_idx_z
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    # Mapping thread ke baris output lokal dan grup
    var warp_id = tid // 32
    var lane_id = tid % 32
    var sub_row = lane_id // DECODE_LANES_PER_ROW      # 0..3
    var local_row = warp_id * 4 + sub_row              # 0..31
    var lane_in_row = lane_id % DECODE_LANES_PER_ROW   # 0..7

    var global_n = block_idx_x * DECODE_ROWS_PER_BLOCK + local_row
    var valid_row = (global_n < N)

    # Offset batch
    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row

    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset
    var wrow = wbase + (global_n * weight_row_bytes) if valid_row else UnsafePointer[UInt8, MutAnyOrigin]()

    # Akumulator FP32 per-baris-m dalam array inline tetap; hanya indeks
    # 0..MPAD-1 yang dipakai (MPAD <= DECODE_MPAD_MAX selalu).
    var thread_acc = InlineArray[Float32, DECODE_MPAD_MAX](0.0)

    # Layout SMEM: staging MPAD_MAX x (8 grup x 132 float) | s_s 256 float
    # (skala 32 baris x 8 grup tile) | nib_s 4352 float | xg_s 8 float | scratchpad reduksi.
    var s_s = smem_x + (DECODE_MPAD_MAX * DECODE_GS * DECODE_GRP_PAD)
    var nib_s = s_s + DECODE_THREADS
    var xg_s = nib_s + QMV_NIB_ELEMS
    var smem_red = xg_s + DECODE_GS

    # Loop K-tile dalam slice grup [g_begin, g_begin + g_count), langkah GS=8
    var g_end = g_begin + g_count
    var g0 = g_begin
    while g0 < g_end:
        # ---- 1. Pemuatan Aktivasi x ke Shared Memory (FP32 on-the-fly) ----
        # Staging kooperatif menutup MPAD x K_TILE elemen; setiap slot ditulis
        # setiap iterasi (zero-fill saat out of range) sehingga fase dot tidak
        # pernah membaca SMEM basi.
        var kc = g0 * GROUP_SIZE
        var idx_stage = tid
        while idx_stage < MPAD * DECODE_K_TILE:
            var mi = idx_stage // DECODE_K_TILE
            var lc = idx_stage % DECODE_K_TILE
            var gk = kc + lc
            var g_idx = lc // GROUP_SIZE
            var elem_in_grp = lc % GROUP_SIZE
            var smem_idx = mi * (DECODE_GS * DECODE_GRP_PAD) + g_idx * DECODE_GRP_PAD + elem_in_grp
            var val: Float32 = Float32(x_ptr[(ly * M + mi) * K + gk]) if ((mi < M) and (gk < K)) else 0.0
            smem_x[smem_idx] = val
            idx_stage += DECODE_THREADS

        # ---- 1b. Staging skala baris milik block x grup tile (FP32) ----
        # 32 baris x 8 grup = 256 slot = tepat 1 slot per thread.
        var sr_s = tid // DECODE_GS
        var sg_s = tid % DECODE_GS
        var grow_s = block_idx_x * DECODE_ROWS_PER_BLOCK + sr_s
        var g_s = g0 + sg_s
        var ok_s = (grow_s < N) and (g_s < g_end) and (g_s < groups_per_row)
        s_s[sr_s * DECODE_GS + sg_s] = Float32(sbase[grow_s * groups_per_row + g_s]) if ok_s else 0.0

        # Inisialisasi deterministik seluruh tile SMEM — WAJIB untuk mode
        # simulasi host sekuensial (thread dieksekusi per-tid; thread 0
        # mengisi tile penuh sebelum thread lain membacanya).
        if tid == 0:
            for i in range(DECODE_K_TILE):
                var gk = kc + i
                var g_idx = i // GROUP_SIZE
                var elem_in_grp = i % GROUP_SIZE
                var smem_idx = g_idx * DECODE_GRP_PAD + elem_in_grp
                smem_x[smem_idx] = Float32(x_ptr[ly * K + gk]) if (gk < K) else 0.0

            for sr in range(DECODE_ROWS_PER_BLOCK):
                var grow = block_idx_x * DECODE_ROWS_PER_BLOCK + sr
                for sg in range(DECODE_GS):
                    var g_s = g0 + sg
                    var ok = (grow < N) and (g_s < g_end) and (g_s < groups_per_row)
                    s_s[sr * DECODE_GS + sg] = Float32(sbase[grow * groups_per_row + g_s]) if ok else 0.0

            for grp in range(DECODE_GS):
                var xg = smem_x + grp * DECODE_GRP_PAD
                var ntg = nib_s + grp * (QMV_NIB_PER_GRP * QMV_NIB_ENT_PAD)
                var xsum: Float32 = 0.0
                for nib in range(QMV_NIB_PER_GRP):
                    var x0 = xg[nib * 4 + 0]
                    var x1 = xg[nib * 4 + 1]
                    var x2 = xg[nib * 4 + 2]
                    var x3 = xg[nib * 4 + 3]
                    var ntl = ntg + (nib ^ grp) * QMV_NIB_ENT_PAD
                    ntl[0] = 0.0
                    ntl[1] = x0
                    ntl[2] = x1
                    ntl[3] = x0 + x1
                    ntl[4] = x2
                    ntl[5] = x0 + x2
                    ntl[6] = x1 + x2
                    ntl[7] = x0 + x1 + x2
                    ntl[8] = x3
                    ntl[9] = x0 + x3
                    ntl[10] = x1 + x3
                    ntl[11] = x0 + x1 + x3
                    ntl[12] = x2 + x3
                    ntl[13] = x0 + x2 + x3
                    ntl[14] = x1 + x2 + x3
                    ntl[15] = x0 + x1 + x2 + x3
                    xsum += (x0 + x1 + x2 + x3)
                xg_s[grp] = xsum

        # SINKRONISASI 1: x_s + s_s (+ nib_s + xg_s) siap untuk seluruh thread block.
        # No-op pada host CPU; barrier hardware saat eksekusi GPU device.
        block_barrier()

        # ---- 2. Dot Product: Nibble LUT (32 table lookups per grup g128) ----
        var g = g0 + lane_in_row
        var g_ok = valid_row and (g < g_end) and (g < groups_per_row)
        var sg = s_s[local_row * DECODE_GS + lane_in_row]
        if g_ok:
            var byte_group_offset = g * (GROUP_SIZE // WEIGHT_PACK_FACTOR) # g * 16
            var wv = wrow.load[width=16](byte_group_offset)
            var ntg = nib_s + lane_in_row * (QMV_NIB_PER_GRP * QMV_NIB_ENT_PAD)
            var p: Float32 = 0.0

            # Unroll compile-time (@parameter): wv[k] dengan k konstan diekstrak
            # dari register SIMD. Tanpa unroll, LLVM NVPTX meng-spill wv ke
            # local memory (.local) dan membacanya balik 16x (~20-30 cycle/load).
            @parameter
            for k in range(16):
                var b_int = Int(wv[k])
                var nib0 = b_int & 15
                var nib1 = (b_int >> 4) & 15
                var nidx0 = (k * 2) ^ lane_in_row
                var nidx1 = (k * 2 + 1) ^ lane_in_row
                p += ntg[nidx0 * QMV_NIB_ENT_PAD + nib0] + ntg[nidx1 * QMV_NIB_ENT_PAD + nib1]

            # Identitas Affine Bonsai: w_eff = (2*bit - 1)*s => s*(2*p - x_sum)
            thread_acc[0] += sg * (2.0 * p - xg_s[lane_in_row])

        # SINKRONISASI 2: Tunggu sebelum menimpa SMEM pada K-tile berikutnya.
        # No-op pada host CPU; barrier hardware saat eksekusi GPU device.
        block_barrier()

        g0 += DECODE_GS

    # ---- 3. Reduksi Intra-Warp Deterministik per-mi (8 Lane -> 1 Baris) ----
    # Pola deterministik lama di-loop per mi; hasil ditulis ke workspace FP32
    # (split-K) atau langsung ke y.
    for mi in range(MPAD):
        # Scratchpad PER-MI: pada host-sim sekuensial, loop mi berada di dalam
        # panggilan per-thread — buffer tunggal akan tertimpa mi berikutnya
        # sebelum reduksi mi ini membaca (penyebab m>1 salah; m1 kebetulan
        # benar). GPU variant tetap memakai buffer tunggal + barrier hardware.
        smem_red[mi * DECODE_THREADS + tid] = thread_acc[mi]

        # SINKRONISASI 3: Tulisan scratchpad seluruh lane harus terlihat sebelum
        # reduksi dibaca. No-op pada host CPU; barrier hardware saat eksekusi
        # GPU device. Traversal seragam: MPAD seragam untuk seluruh thread.
        block_barrier()

        if valid_row and (lane_in_row == 7) and (mi < M):
            var base_tid = tid - 7
            var total_row_acc: Float32 = 0.0
            for lane in range(DECODE_LANES_PER_ROW):
                total_row_acc += smem_red[mi * DECODE_THREADS + base_tid + lane]
            if use_ws:
                # Partial split-K: [slices][L * M * N], FP32 murni
                ws_ptr[slice_idx * L * M * N + ly * M * N + mi * N + global_n] = total_row_acc
            else:
                var out_idx = (ly * M + mi) * N + global_n
                y_ptr[out_idx] = Scalar[T](total_row_acc)

# ----------------------------------------------------------------------------
# 2. Native GPU Device Kernel: Vectorized GEMV Split-K (Tesla T4 SM75)
# ----------------------------------------------------------------------------
fn qmv_sm75_b1_decode_gpu[
    T: DType,
    MPAD_STATIC: Int = 1
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
    MPAD: Int
):
    """
    Device kernel native decode (paritas qmv_vec_splitk_kernel) untuk hardware
    NVIDIA Tesla T4. thread_idx/block_idx dari register hardware GPU; barrier
    hardware CUDA; Shared Memory dialokasi DI DALAM kernel pada address space
    SHARED (stack_allocation) — pointer heap host tidak pernah dipass ke
    device. Grid = (ceil(N/32), 1, L), M di dalam block via bucket MPAD,
    split-K grup kontigu [g_begin, g_begin + g_count).
    """
    from gpu.id import thread_idx, block_idx
    from gpu import barrier
    from gpu.primitives.warp import shuffle_down

    # Alokasi SMEM device internal:
    # Untuk MPAD_STATIC=1 (decode): staging x (1056) + s_s (256) + nib_s (4352) + xg_s (8) = 5672 float (22.15 KiB).
    # Untuk MPAD_STATIC>1: staging MPAD_STATIC x (8 grup x 132 float) + s_s 256 float.
    # Seluruhnya aman di bawah limit 48 KiB Tesla T4 (sm_75).
    alias SMEM_ALLOC_ELEMS = DECODE_NIB_SMEM_ELEMS
    var smem_x = stack_allocation[
        SMEM_ALLOC_ELEMS,
        Float32,
        alignment = 16,
        address_space = AddressSpace.SHARED
    ]()

    var tid = thread_idx.x
    var ly = block_idx.z
    var bx = block_idx.x
    var groups_per_row = get_scale_stride(K)
    var weight_row_bytes = K // WEIGHT_PACK_FACTOR

    var warp_id = tid // 32
    var lane_id = tid % 32
    var sub_row = lane_id // DECODE_LANES_PER_ROW
    var local_row = warp_id * 4 + sub_row
    var lane_in_row = lane_id % DECODE_LANES_PER_ROW

    var global_n = bx * DECODE_ROWS_PER_BLOCK + local_row
    var valid_row = (global_n < N)

    var w_layer_offset = 0 if broadcast_w else ly * N * weight_row_bytes
    var s_layer_offset = 0 if broadcast_w else ly * N * groups_per_row

    var wbase = w_ptr + w_layer_offset
    var sbase = scales_ptr + s_layer_offset
    var wrow = wbase + (global_n * weight_row_bytes) if valid_row else UnsafePointer[UInt8, MutAnyOrigin]()

    var thread_acc: Float32 = 0.0
    var s_s = smem_x + (DECODE_GS * DECODE_GRP_PAD)
    var nib_s = s_s + DECODE_THREADS
    var xg_s = nib_s + QMV_NIB_ELEMS

    var g_end = g_begin + g_count
    # Konstanta staging per-thread (invarian lintas tile)
    var stage_off = Int(tid) * 8
    var stage_in_x = stage_off < DECODE_K_TILE
    var stage_dst = (stage_off // GROUP_SIZE) * DECODE_GRP_PAD + stage_off % GROUP_SIZE
    var sr_s = tid // DECODE_GS
    var sg_s = tid % DECODE_GS
    var grow_s = bx * DECODE_ROWS_PER_BLOCK + sr_s

    # ---- PROLOGUE tile g_begin: stage x (vektor) + skala dari global ----
    # GUARD WAJIB menutup LOAD dan STORE: tile x hanya 1024 elemen — thread
    # tid>=128 tidak boleh menyentuh smem_x (store liar menimpa region s_s).
    var kc = g_begin * GROUP_SIZE
    if stage_in_x:
        var xv0 = SIMD[DType.float32, 8](0)
        if (kc + stage_off) < K:
            xv0 = x_ptr.load[width=8](ly * K + kc + stage_off).cast[DType.float32]()
        @parameter
        for j in range(8):
            smem_x[stage_dst + j] = xv0[j]
    var g_s0 = g_begin + sg_s
    var ok_s0 = (grow_s < N) and (g_s0 < g_end) and (g_s0 < groups_per_row)
    s_s[sr_s * DECODE_GS + sg_s] = Float32(sbase[grow_s * groups_per_row + g_s0]) if ok_s0 else 0.0

    barrier()

    # Prefetch bobot uint4 tile pertama (sekali-satunya load terekspos;
    # selanjutnya selalu di-prefetch selama build tile sebelumnya).
    var g_wv = g_begin + lane_in_row
    var wv = SIMD[DType.uint8, 16](0)
    if valid_row and (g_wv < g_end) and (g_wv < groups_per_row):
        wv = wrow.load[width=16](g_wv * (GROUP_SIZE // WEIGHT_PACK_FACTOR))

    var g0 = g_begin
    while g0 < g_end:

        # ---- 2. Pembangunan Tabel Subset-Sum Nibble (Parallel Cooperative Warp Build) ----
        # Warp warp_id (0..7) membangun tabel untuk grup warp_id
        # Lane lane_id (0..31) membangun 16 entri tabel untuk nibble lane_id (4 aktivasi)
        var xg = smem_x + (warp_id * DECODE_GRP_PAD + lane_id * QMV_NIB_BITS)
        var x0 = xg[0]
        var x1 = xg[1]
        var x2 = xg[2]
        var x3 = xg[3]

        var t0: Float32 = 0.0
        var t1 = x0
        var t2 = x1
        var t3 = x0 + x1
        var t4 = x2
        var t5 = t1 + x2
        var t6 = t2 + x2
        var t7 = t3 + x2
        var t8 = x3
        var t9 = t1 + x3
        var t10 = t2 + x3
        var t11 = t3 + x3
        var t12 = t4 + x3
        var t13 = t5 + x3
        var t14 = t6 + x3
        var t15 = t7 + x3

        # Swizzle (lane_id ^ warp_id) dengan stride 17 untuk zero bank-conflict pada STS & LDS
        var ntl = nib_s + (warp_id * (QMV_NIB_PER_GRP * QMV_NIB_ENT_PAD) + (lane_id ^ warp_id) * QMV_NIB_ENT_PAD)
        ntl[0] = t0
        ntl[1] = t1
        ntl[2] = t2
        ntl[3] = t3
        ntl[4] = t4
        ntl[5] = t5
        ntl[6] = t6
        ntl[7] = t7
        ntl[8] = t8
        ntl[9] = t9
        ntl[10] = t10
        ntl[11] = t11
        ntl[12] = t12
        ntl[13] = t13
        ntl[14] = t14
        ntl[15] = t15

        # Reduksi sum seluruh 128 aktivasi dalam grup (t15) menggunakan hardware shuffle intra-warp
        var gs = t15
        gs += shuffle_down(gs, 16)
        gs += shuffle_down(gs, 8)
        gs += shuffle_down(gs, 4)
        gs += shuffle_down(gs, 2)
        gs += shuffle_down(gs, 1)
        if lane_id == 0:
            xg_s[warp_id] = gs

        # ---- Prefetch penuh tile gn: x-chunk + bobot uint4 ke register ----
        # Diterbitkan SEBELUM barrier agar ~6 KB/SM load tetap in-flight
        # selama fase dot + barrier (kunci saturasi DRAM, paritas referensi
        # qmv_vec_nib_kernel: prefetch sebelum barrier konsumen).
        var gn = g0 + DECODE_GS
        var have_next = gn < g_end
        var xv_next = SIMD[DType.float32, 8](0)
        var wv_next = SIMD[DType.uint8, 16](0)
        if have_next:
            if stage_in_x:
                if (gn * GROUP_SIZE + stage_off) < K:
                    xv_next = x_ptr.load[width=8](ly * K + gn * GROUP_SIZE + stage_off).cast[DType.float32]()
            var gw = gn + lane_in_row
            if valid_row and (gw < g_end) and (gw < groups_per_row):
                wv_next = wrow.load[width=16](gw * (GROUP_SIZE // WEIGHT_PACK_FACTOR))

        # SINKRONISASI 2: nib_s dan xg_s siap dibaca seluruh warp dalam block
        barrier()

        # Restage x tile gn dari REGISTER (bukan global) — x_s tidak dibaca
        # lagi setelah barrier ini (dot hanya membaca nib_s/s_s/xg_s).
        if have_next and stage_in_x:
            @parameter
            for j in range(8):
                smem_x[stage_dst + j] = xv_next[j]

        # ---- 3. Dot Product Nibble LUT (32 table lookups per grup g128) ----
        var g = g0 + lane_in_row
        var g_ok = valid_row and (g < g_end) and (g < groups_per_row)
        var sg = s_s[local_row * DECODE_GS + lane_in_row]
        if g_ok:
            # wv sudah di-prefetch ke register setelah SINKRONISASI 1.
            var ntg = nib_s + lane_in_row * (QMV_NIB_PER_GRP * QMV_NIB_ENT_PAD)
            var p: Float32 = 0.0

            # Ekstraksi NIBBLE-PER-KATA 32-bit (paritas dot referensi):
            # 1 lookup + 1 FADD per 4 bobot (~1.6 ops/bobot termasuk rakit
            # kata). Versi per-byte lama = ~6 ops/bobot (ekstrak byte + 2
            # mask + shift + 2 XOR + 2 IMAD + 2 LDS per 2 bobot) -> kernel
            # issue-bound ~30 GB/s. Bit mapping identik: LSB-first per byte,
            # word little-endian.
            @parameter
            for wi in range(4):
                var bits = (
                    Int(wv[4 * wi])
                    | (Int(wv[4 * wi + 1]) << 8)
                    | (Int(wv[4 * wi + 2]) << 16)
                    | (Int(wv[4 * wi + 3]) << 24)
                )
                @parameter
                for h in range(8):
                    var nidx = (wi * 8 + h) ^ lane_in_row
                    p += ntg[nidx * QMV_NIB_ENT_PAD + ((bits >> (4 * h)) & 15)]

            # Identitas Affine Bonsai: w_eff = (2*bit - 1)*s => s*(2*p - x_sum)
            thread_acc += sg * (2.0 * p - xg_s[lane_in_row])

        wv = wv_next
        g0 = gn

        # SINKRONISASI 3: restage x_s selesai & seluruh bacaan dot selesai
        barrier()

        # Stage skala tile gn dari global (512 B/block — kecil, tertutup
        # prefetch in-flight warp lain; terlihat setelah barrier berikutnya).
        if have_next:
            var g_sn = gn + sg_s
            var ok_sn = (grow_s < N) and (g_sn < g_end) and (g_sn < groups_per_row)
            s_s[sr_s * DECODE_GS + sg_s] = Float32(sbase[grow_s * groups_per_row + g_sn]) if ok_sn else 0.0

    # Reduksi deterministik per-warp memakai hardware shuffle width-8
    # (paritas __shfl_down_sync(0xffffffff, v, off, 8) referensi). Shuffle
    # Mojo full-warp tanpa parameter width, sehingga segmen 8-lane diemulasi
    # dengan select konvergen: lane di luar segmen menambah 0. Hasil untuk
    # lane penulis (lane_in_row == 0, paritas grp==0 referensi) identik dan
    # bebas atomicAdd.
    var v: Float32 = thread_acc
    var off = DECODE_LANES_PER_ROW // 2
    while off > 0:
        var other = shuffle_down(v, UInt32(off))
        var take = (lane_in_row + off) < DECODE_LANES_PER_ROW
        v += other if take else 0.0
        off = off // 2

    if valid_row and (lane_in_row == 0):
        if use_ws:
            ws_ptr[slice_idx * L * M * N + ly * M * N + global_n] = v
        else:
            var out_idx = ly * N + global_n
            y_ptr[out_idx] = Scalar[T](v)

# ----------------------------------------------------------------------------
# 3. Kernel Reduce Split-K (Deterministik, Ascending, Tanpa atomicAdd)
# ----------------------------------------------------------------------------
fn qmv_split_reduce_gpu[
    T: DType
](
    ws_ptr: UnsafePointer[Float32, MutAnyOrigin],
    y_ptr: UnsafePointer[Scalar[T], MutAnyOrigin],
    splits: Int,
    total: Int
):
    """
    Reduce workspace split-K [splits][total] ke output y: setiap thread
    menjumlahkan slice secara ASCENDING (urutan tetap -> deterministik
    bitwise run-to-run, paritas qmv_split_reduce_kernel).
    """
    from gpu.id import thread_idx, block_idx

    var e = block_idx.x * DECODE_THREADS + thread_idx.x
    if e < total:
        var acc: Float32 = 0.0
        for s in range(splits):
            acc += ws_ptr[s * total + e]
        y_ptr[e] = Scalar[T](acc)
