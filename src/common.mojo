# ===----------------------------------------------------------------------=== #
# Module: common.mojo
# Purpose: Definisi konstanta arsitektur hardware T4 (sm_75), geometri tiling,
#          stride memori, dan helper kalkulasi dimensi untuk W1A16 g128.
# ===----------------------------------------------------------------------=== #

from sys import size_of

# ----------------------------------------------------------------------------
# 1. Konstanta Arsitektur Kuantisasi Bonsai 1-Bit
# ----------------------------------------------------------------------------
alias GROUP_SIZE: Int = 128            # Ukuran grup kuantisasi (g128)
alias BITS: Int = 1                    # Format bobot 1-bit biner (±scale)
alias WEIGHT_PACK_FACTOR: Int = 8      # 8 bobot biner per byte (uint8)

# ----------------------------------------------------------------------------
# 2. Geometri Tiling Prefill Kernel (qmm_sm75_b1: GEMM untuk M > 1)
# ----------------------------------------------------------------------------
alias PREFILL_BM: Int = 64             # Baris output M per block CTA
alias PREFILL_BN: Int = 32             # Kolom output N per block CTA
alias PREFILL_BK: Int = 64             # Kedalaman kontraksi K per tile loop
alias PREFILL_PAD: Int = 8             # Padding SMEM untuk eliminasi LDS bank conflict
alias PREFILL_THREADS: Int = 128       # 4 warp per block CTA (128 threads)

# ----------------------------------------------------------------------------
# 3. Geometri Decode Kernel (qmv_sm75_b1: GEMV untuk M <= 8, token generation)
# ----------------------------------------------------------------------------
alias DECODE_WARPS: Int = 8            # 8 warp per block = 256 threads
alias DECODE_THREADS: Int = 256        # Total thread per block CTA decode
alias DECODE_ROWS_PER_BLOCK: Int = 32  # 32 baris output N per block (4 baris/warp)
alias DECODE_LANES_PER_ROW: Int = 8    # 8 lane bekerja sama pada 1 baris
alias DECODE_GS: Int = 8               # 8 grup g128 (1024 aktivasi K) per tile
alias DECODE_K_TILE: Int = DECODE_GS * GROUP_SIZE # 1024 elemen K per staging tile
alias DECODE_GRP_PAD: Int = GROUP_SIZE + 4        # 128 -> 132 float (132 % 32 == 4, zero bank conflict)
alias DECODE_MPAD_MAX: Int = 8         # Bucket pad M maksimum (paritas MPAD qmv_vec_splitk)

# Konstanta Arsitektur Nibble LUT 4-Bit (Paritas MLX qmv_vec_nib_kernel)
alias QMV_NIB_BITS: Int = 4                         # 4 bit per nibble
alias QMV_NIB_PER_GRP: Int = GROUP_SIZE // QMV_NIB_BITS # 32 nibble per grup g128
alias QMV_NIB_ENT: Int = 16                         # 16 entri tabel per nibble
alias QMV_NIB_ENT_PAD: Int = QMV_NIB_ENT + 1        # 17 float (pad 1 anti-bank conflict STS/LDS)
alias QMV_NIB_ELEMS: Int = DECODE_GS * QMV_NIB_PER_GRP * QMV_NIB_ENT_PAD # 4352 float
alias DECODE_NIB_SMEM_ELEMS: Int = (DECODE_GS * DECODE_GRP_PAD) + DECODE_THREADS + QMV_NIB_ELEMS + DECODE_GS # 5672 float

@always_inline
fn decode_mpad_bucket(m: Int) -> Int:
    """
    Memetakan jumlah baris M fase decode ke bucket MPAD gaya referensi
    qmv_vec_splitk_kernel: bucket terkecil dari {1, 2, 4, 8} yang >= m.
    Nilai di atas 8 di-clamp ke 8 (decode hanya dipakai untuk m <= 8).
    """
    if m <= 1:
        return 1
    elif m <= 2:
        return 2
    elif m <= 4:
        return 4
    return 8

# ----------------------------------------------------------------------------
# 3b. Split-K Deterministik Decode (paritas QMV_VEC_BLOCK_FILL_TARGET)
# ----------------------------------------------------------------------------
alias DECODE_BLOCK_FILL_TARGET: Int = 320 # Kalibrasi T4: 40 SM x 8 blok/SM
alias DECODE_MAX_SPLITS: Int = 3

@fieldwise_init
struct DecodeSplitRange(Copyable, Movable, ImplicitlyCopyable):
    """Rentang grup g128 kontigu milik satu slice split-K."""
    var g_begin: Int
    var g_count: Int

@always_inline
fn decode_split_plan(n: Int, l: Int, groups_per_row: Int) -> Int:
    """
    Menentukan jumlah slice split-K decode (paritas qmv_lut_make_plan jalur
    vec): split HANYA bila grid kurang dari setengah penuh (butuh >= 2x blok
    untuk mencapai fill target), n > 128, dan ada >= 2 grup per baris.
    Deterministik murni dari bentuk masalah.
    """
    var total_blocks = cdiv(n, DECODE_ROWS_PER_BLOCK) * l
    var splits = 1
    if (n > 128) and (total_blocks * 2 < DECODE_BLOCK_FILL_TARGET) and (groups_per_row >= 2):
        var want = cdiv(DECODE_BLOCK_FILL_TARGET, total_blocks)
        if want > groups_per_row:
            want = groups_per_row
        if want > DECODE_MAX_SPLITS:
            want = DECODE_MAX_SPLITS
        splits = want
    return splits

@always_inline
fn decode_group_range(groups_per_row: Int, splits: Int, slice_idx: Int) -> DecodeSplitRange:
    """
    Partisi kontigu grup g128 per slice: slice awal menerima sisa pembagian.
    Tanpa atomicAdd — urutan reduce ascending menjamin determinisme bitwise.
    """
    var base = groups_per_row // splits
    var rem = groups_per_row % splits
    var begin = slice_idx * base + min(slice_idx, rem)
    var count = base + (1 if slice_idx < rem else 0)
    return DecodeSplitRange(begin, count)

# ----------------------------------------------------------------------------
# 3c. Direct Small-M Prefill (8 < M <= 64, paritas qmv_direct_smallm_kernel)
# ----------------------------------------------------------------------------
alias DIRECT_ROWS_PER_BLOCK: Int = 8  # 1 warp <-> 1 baris output (paritas QMV_LUT_ROWS_PER_BLOCK)
alias DIRECT_MAX_M: Int = 64
alias DIRECT_SPLIT_GATE: Int = 40     # 1 wave T4: split bila total blok < 40
alias DIRECT_FILL_TARGET: Int = 80    # 2 wave: want = ceil(80 / total_blok)
alias DIRECT_MAX_SPLITS: Int = 16
alias DIRECT_GRP_PAD: Int = GROUP_SIZE + 8 # 128 -> 136 (stride 68 word, konflik bank <= 4-way)
# Ukuran staging SMEM terburuk: PAD=64 x GS=2 x 136 = PAD=32 x GS=4 x 136 = 17408 elemen T
alias DIRECT_SMEM_ELEMS: Int = DIRECT_MAX_M * 2 * DIRECT_GRP_PAD

@always_inline
fn direct_pad_bucket(m: Int) -> Int:
    """Bucket register-M jalur direct untuk 8 < m <= 64: {16, 32, 64}."""
    if m <= 16:
        return 16
    elif m <= 32:
        return 32
    return 64

@always_inline
fn direct_gs(pad: Int) -> Int:
    """Grup g128 yang di-stage per K-tile jalur direct (paritas qmv_direct_gs)."""
    if pad <= 8:
        return 8
    elif pad <= 32:
        return 4
    return 2

@always_inline
fn direct_split_plan(n: Int, l: Int, groups_per_row: Int) -> Int:
    """
    Plan split-K jalur direct (paritas make_plan direct): split hanya bila
    total blok kurang dari 1 wave (40 blok T4) dan >= 2 grup per baris;
    target pengisian 2 wave (80 blok), maks 16 slice. Deterministik.
    """
    var total_blocks = cdiv(n, DIRECT_ROWS_PER_BLOCK) * l
    var splits = 1
    if (total_blocks < DIRECT_SPLIT_GATE) and (groups_per_row >= 2):
        var want = cdiv(DIRECT_FILL_TARGET, total_blocks)
        if want > groups_per_row:
            want = groups_per_row
        if want > DIRECT_MAX_SPLITS:
            want = DIRECT_MAX_SPLITS
        splits = want
    return splits

# ----------------------------------------------------------------------------
# 4. Invarian Barrier Blok (Host CPU Sequential vs GPU Device)
# ----------------------------------------------------------------------------
@always_inline
fn block_barrier():
    """
    Titik sinkronisasi blok eksplisit, padanan `barrier()` / `__syncthreads()`.
    Dispatcher host (ops.mojo) mengeksekusi body kernel secara sekuensial
    per-tid, sehingga setiap thread menyelesaikan seluruh body sebelum thread
    berikutnya dimulai dan fungsi ini menjadi no-op. Saat body kernel
    dikompilasi untuk GPU device, titik panggilan ini wajib dipetakan ke
    barrier hardware agar pemuatan kooperatif SMEM tidak berlomba dengan
    pembacaannya.
    """
    pass

# ----------------------------------------------------------------------------
# 5. Helper Kalkulasi Dimensi & Keselarasan Memori
# ----------------------------------------------------------------------------
@always_inline
fn cdiv(a: Int, b: Int) -> Int:
    """Kalkulasi pembagian pembulatan ke atas (ceiling division)."""
    return (a + b - 1) // b

@always_inline
fn get_scale_stride(k: Int) -> Int:
    """Menghitung jumlah grup skala per baris bobot (group_size = 128)."""
    return cdiv(k, GROUP_SIZE)

@always_inline
fn get_weight_row_bytes(k: Int) -> Int:
    """Menghitung ukuran baris bobot packed dalam byte (k / 8)."""
    return k // WEIGHT_PACK_FACTOR

@always_inline
fn is_aligned_16_bytes(addr: Int) -> Bool:
    """Memeriksa apakah pointer memori selaras dengan transaksi 128-bit (uint4)."""
    return (addr % 16) == 0

@always_inline
fn is_aligned_4_bytes(addr: Int) -> Bool:
    """Memeriksa apakah pointer memori selaras dengan transaksi 32-bit (uint32)."""
    return (addr % 4) == 0

# ----------------------------------------------------------------------------
# 6. Struct Shape untuk Tensor 3D (Batch L, Rows M, Cols N/K)
# ----------------------------------------------------------------------------
@fieldwise_init
struct TensorShape3D(Copyable, Movable, ImplicitlyCopyable):
    var l: Int  # Dimensi Batch
    var m: Int  # Dimensi Baris (Sequence Length)
    var k: Int  # Dimensi Kolom (Hidden Size / Channel)

    @always_inline
    fn num_elements(self) -> Int:
        return self.l * self.m * self.k
