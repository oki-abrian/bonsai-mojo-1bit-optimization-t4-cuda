# ===----------------------------------------------------------------------=== #
# Script: main.mojo
# Purpose: CLI INFERENSI NATIVE MOJO (tanpa Python) untuk Bonsai-27B-mlx-1bit:
#          muat config.json + safetensors via loader Mojo, bangun 64 layer
#          hybrid Qwen3.5 (3 GDN : 1 Gated Attention), generasi greedy.
#          Prompt diberikan sebagai token id (v1); tokenisasi BPE menyusul.
# Catatan: ini harness korektess FP32 host-sim — decode 27B di CPU lambat
#          (~detik per token); performa datang dari jalur GPU MAX.
# Risiko kompilasi terflag: String(StringSpan), Span[UInt8, MutAnyOrigin](ptr=ptr, length=n),
#          FileHandle.seek(offset, whence), `from io.file import open`,
#          assignment elemen UnsafePointer[Struct, MutAnyOrigin] (lihat NOTES_API_*.md).
# ===----------------------------------------------------------------------=== #

from sys import argv
from os import setenv, getenv
from memory import UnsafePointer, alloc
from src import (
    QwenConfig, QwenDecoderLayer, QwenLinear1Bit, GatedDeltaNetState,
    AttentionKVCache, qwen3_5_model_forward,
    khq_dump_configure, khq_dump_flush,
    khq_active, khq_activate, khq_prof_report
)
from time import monotonic
from math import exp, sqrt
from src.jsonlite import JsonDoc
from src.safetensors import SafeTensorsIndex, fuse_u8, fuse_f32, load_qlinear
from src.models.qwen3_5.linear import use_gpu_matmul, DeviceContextGPU
from src.models.qwen3_5.gpu_ctx import gpu_ctx_new
from gpu.host import DeviceBuffer
from src.ops import (
    rmsnorm_sm75_launch_on, argmax_sm75_launch_on,
    embed_lookup_1bit_sm75_launch_on, copy_vec_sm75_launch_on
)

fn dump_top2_prefill(
    gpu_ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
    logits_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    V: Int
) raises:
    """Diagnostik near-tie: top-2 logit + gap di batas prefill. Gap kecil
    (<~0.05) + flip token vs jalur per-token = wajar (beda urutan akumulasi
    fp16 antara WMMA vs GEMV); gap besar = indikasi bug layout."""
    var ctx = gpu_ctx_ptr[]
    var tmp = ctx.enqueue_create_buffer[DType.float16](V)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, tmp.unsafe_ptr(), logits_dev, V
    )
    ctx.synchronize()
    var hlog = alloc[Scalar[DType.float16]](V)
    ctx.enqueue_copy(hlog, tmp)
    ctx.synchronize()
    var b1 = 0
    var b2 = 0
    var v1: Float64 = -1e30
    var v2: Float64 = -1e30
    for i in range(V):
        var lv = hlog[i].cast[DType.float64]()
        if lv > v1:
            v2 = v1
            b2 = b1
            v1 = lv
            b1 = i
        elif lv > v2:
            v2 = lv
            b2 = i
    print(">> [TOP2] 1st:", b1, "=", v1, "| 2nd:", b2, "=", v2,
          "| gap:", v1 - v2)
    hlog.free()


fn find_flex(index: SafeTensorsIndex, name: String) -> Int:
    """Cari tensor: nama apa adanya, lalu fallback prefix 'language_model.'
    (repo Bonsai-27B-mlx-1bit dikemas ala VLM Qwen3.5)."""
    var e = index.find(name)
    if e != -1:
        return e
    return index.find("language_model." + name)

fn read_scales_eff(index: SafeTensorsIndex, e_s: Int, numel: Int, bits: Int = 1) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    """Scales efektif untuk kernel.

    bits=1: s_eff = s_checkpoint / 2 (kernel menghitung (2q-1)*s_eff).
    bits=2: scales dipakai MENTAH — kontrak (q-1)*s (biases == -s penuh).

    GAGAL KERAS bila tensor scales tidak ada. Sebelumnya `e_s == -1` hanya
    menghasilkan buffer `alloc` yang TAK-TERINISIALISASI: kernel QMV/QMM lalu
    membaca skala sampah dan mengeluarkan hasil salah TANPA satu pun pesan
    error. Sama kelasnya dengan bug "hasil read dibuang" (FIX-5), tetapi di
    jalur skala — yang justru mengalikan seluruh kontribusi bobot.
    """
    if e_s == -1 and numel > 0:
        raise Error(
            "FATAL: tensor scales tidak ditemukan (numel=" + String(numel)
            + "). Tanpa skala, kernel membaca memori tak-terinisialisasi "
            + "dan hasilnya salah tanpa error."
        )
    var p = alloc[Float32](numel if numel > 0 else 1)
    if e_s != -1 and numel > 0:
        if not index.read_f32(e_s, p, numel):
            raise Error(
                "FATAL: gagal baca scales (numel=" + String(numel)
                + ") — dtype/jumlah elemen tidak cocok dgn header safetensors."
            )
        if bits == 1:
            for i in range(numel):
                p[i] *= Float32(0.5)
    return p

fn read_biases_f32(index: SafeTensorsIndex, e_b: Int, numel: Int) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    """Biases checkpoint mentah (w = q*s_ckpt + b, affine eksak).

    Bila tensor biases TIDAK ADA, buffer dikembalikan dalam keadaan NOL (bukan
    tak-terinisialisasi). Ini aman karena kernel QMV/QMM MENDERIVASI bias
    sebagai `-s_eff` dan tidak pernah membaca tensor biases; nol membuat
    perilaku itu eksplisit, bukan kebetulan. Bila biases ADA tetapi gagal
    dibaca, kita tetap gagal keras.
    """
    var p = alloc[Float32](numel if numel > 0 else 1)
    for i in range(numel if numel > 0 else 1):
        p[i] = 0.0
    if e_b != -1 and numel > 0:
        if not index.read_f32(e_b, p, numel):
            raise Error(
                "FATAL: gagal baca biases (numel=" + String(numel)
                + ") — dtype/jumlah elemen tidak cocok dgn header safetensors."
            )
    return p

fn must_read_f32(
    index: SafeTensorsIndex,
    e: Int,
    dst: UnsafePointer[Float32, MutAnyOrigin],
    numel: Int,
    what: String
) raises:
    """Baca FP32 dan GAGAL KERAS bila tidak cocok.

    Sebelumnya semua pemanggil memakai `var _ = index.read_f32(...)` — nilai
    balik dibuang. Bila dtype/jumlah elemen tidak cocok, read_f32 mengembalikan
    False dan buffer hasil `alloc` tetap TAK-TERINISIALISASI, lalu dipakai
    sebagai A_log / dt_bias / bobot norm tanpa satu pun pesan error.
    """
    if not index.read_f32(e, dst, numel):
        raise Error(
            "FATAL: gagal baca " + what + " (numel=" + String(numel)
            + ") — dtype/jumlah elemen tidak cocok dgn header safetensors."
        )

fn must_read_raw(
    index: SafeTensorsIndex,
    e: Int,
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    what: String
) raises:
    """Baca blob mentah (bobot 1-bit terpak) dan gagal keras bila tidak cocok."""
    if not index.read_raw(e, dst, nbytes):
        raise Error(
            "FATAL: gagal baca " + what + " (nbytes=" + String(nbytes) + ")."
        )


fn verify_affine_zero_bias(
    s: UnsafePointer[Float32, MutAnyOrigin],
    biases: UnsafePointer[Float32, MutAnyOrigin],
    numel: Int,
    s_is_eff: Bool,
    what: String,
    bits: Int = 1
) raises:
    """Verifikasi kontrak affine checkpoint.

    bits=1: `biases == -scales_ckpt / 2`. Dua konvensi pemakaian skala hidup
    berdampingan, dan KEDUANYA bergantung pada kontrak yang sama:
      * `s_is_eff=True`  -> `s` sudah s_eff = s_ckpt/2 (loader membagi 2).
        Dipakai QwenLinear1Bit -> kernel QMV/QMM yang MENDERIVASI bias sebagai
        `-s_eff` dan tidak pernah membaca tensor biases. Harapan: b == -s.
      * `s_is_eff=False` -> `s` masih s_ckpt mentah. Dipakai kernel embed
        (`bit*s + b`, affine penuh). Harapan: b == -s/2.

    bits=2 (Bonsai-2): kontrak berbeda — `biases == -scales` penuh pada skala
    mentah, sehingga w = q*s + b = (q-1)*s. k=1.0 dipakai pada s mentah.

    INI ADALAH SATU-SATUNYA alasan `GDN_AFFINE_ZERO_CORRECTION=True` boleh
    melewati loop koreksi affine per grup (src/models/qwen3_5/linear.mojo:40-45).
    Sebelum ini kontrak tersebut hanya "diukur sekali dgn tangan" dan tidak
    pernah diverifikasi per tensor — bila satu grup menyimpang, hasilnya salah
    TANPA satu pun pesan error.

    Sampel merata maks 4096 titik; set BONSAI_VERIFY_AFFINE=full utk menyisir
    seluruh elemen.
    """
    if numel <= 0:
        return
    var full = getenv("BONSAI_VERIFY_AFFINE")
    var semua = full and (full == "full" or full == "1")
    var n_chk = numel if semua else min(numel, 4096)
    var step = numel // n_chk
    if step < 1:
        step = 1
    # Harapan bias utk skala s:
    #   bits=2 atau s_is_eff -> b == -s      (k = 1.0)
    #   bits=1 s mentah       -> b == -s/2   (k = 0.5)
    var k = Float32(1.0) if (s_is_eff or bits == 2) else Float32(0.5)
    var worst: Float32 = 0.0
    var worst_i = 0
    var i = 0
    while i < numel:
        var sv = s[i]
        var bv = biases[i]
        # Relatif thd |s| supaya skala besar/kecil diperlakukan sama.
        var rel = abs(bv + sv * k) / (abs(sv) * k + Float32(1e-12))
        if rel > worst:
            worst = rel
            worst_i = i
        i += step
    if worst > Float32(1e-3):
        raise Error(
            "FATAL: kontrak affine dilanggar pada " + what + " (indeks "
            + String(worst_i) + "): |b + s*" + String(k) + "| / |s*" + String(k)
            + "| = " + String(worst) + ". GDN_AFFINE_ZERO_CORRECTION=True "
            + "TIDAK valid — loop koreksi affine per grup WAJIB dijalankan, "
            + "kalau tidak hasilnya salah tanpa error apa pun."
        )

fn parse_int_list(spec: String) -> UnsafePointer[Int, MutAnyOrigin]:
    """Parse "1,2,3" -> array Int; slot 0 menyimpan jumlah elemen,
    elemen token mulai slot 1."""
    var bytes = spec.as_bytes()
    var n = len(bytes)
    var count = 1
    for i in range(n):
        if bytes[i] == UInt8(ord(",")):
            count += 1
    var out = alloc[Int](count + 1)
    var v = 0
    var idx = 1 # slot 0 dicadangkan untuk jumlah
    var has_digits = False
    for i in range(n):
        var c = bytes[i]
        if c == UInt8(ord(",")):
            out[idx] = v
            idx += 1
            v = 0
            has_digits = False
        elif c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            v = v * 10 + Int(c) - Int(ord("0"))
            has_digits = True
    if has_digits:
        out[idx] = v
        idx += 1
    out[0] = idx - 1 # jumlah elemen efektif
    return out


fn is_stop_token(id: Int, stops: UnsafePointer[Int, MutAnyOrigin]) -> Bool:
    """True bila `id` termasuk daftar token henti (stops[0] = jumlah elemen).

    Dipakai loop decode untuk berhenti di EOS. Sebelumnya loop selalu
    menjalankan `max_tokens - 1` langkah tanpa syarat, sehingga token yang
    dihasilkan SETELAH model mengakhiri gilirannya tetap dipaksa keluar —
    keluarannya di luar distribusi dan mudah runtuh jadi pengulangan.
    """
    if not stops:
        return False
    for i in range(1, stops[0] + 1):
        if stops[i] == id:
            return True
    return False

fn read_small_file(path: String, out_buf: UnsafePointer[UInt8, MutAnyOrigin], cap: Int) raises -> Int:
    """Baca file teks kecil (config.json) — kembalikan jumlah byte terbaca."""
    var f = open(path, "r")
    var n = f.read(Span[UInt8, MutAnyOrigin](ptr=out_buf, length=cap))
    f.close()
    return n


fn env_int(name: String, dflt: Int) -> Int:
    """Baca env var integer; kembalikan dflt bila tidak diset / tidak valid."""
    var v = getenv(name)
    if not v:
        return dflt
    try:
        return Int(v)
    except:
        return dflt


fn bonsai_bits() -> Int:
    """Bit-width bobot pack: 1 (Bonsai-27B lama, default) atau
    2 (Bonsai-2 / Ternary-Bonsai-2-27B, prism_hadamard_qwen35)."""
    return env_int("BONSAI_BITS", 1)


# =============================================================================
# Bonsai-2: kontrak Hadamard pack (prism_hadamard_qwen35)
#
# runtime/runtime.py referensi memetakan SIGNS BUKAN per modul, melainkan
# per LEBAR INPUT (K): signs[width] untuk seluruh modul dengan K = width.
# hadamard.json berisi sign_widths [5120, 6144, 17408] + sign_values
# (28672 nilai ±1, concat sesuai urutan widths). Karena qkv/z dan
# gate/up semuanya K=5120, fusion baris tetap EKSAK (sama signs).
# =============================================================================

struct HadamardSigns:
    var block: Int
    var buf: UnsafePointer[Float32, MutAnyOrigin]      # [total] ±1
    var widths: UnsafePointer[Int, MutAnyOrigin]       # [n_widths]
    var n_widths: Int
    var total: Int

    # Mojo 25.x tidak membuatkan konstruktor anggota otomatis — harus ditulis.
    fn __init__(
        out self,
        block: Int,
        buf: UnsafePointer[Float32, MutAnyOrigin],
        widths: UnsafePointer[Int, MutAnyOrigin],
        n_widths: Int,
        total: Int
    ):
        self.block = block
        self.buf = buf
        self.widths = widths
        self.n_widths = n_widths
        self.total = total


fn hadamard_signs_none() -> HadamardSigns:
    """Pack 1-bit: tidak ada transformasi — for_k mengembalikan pointer nol."""
    return HadamardSigns(
        1024,
        UnsafePointer[Float32, MutAnyOrigin](),
        UnsafePointer[Int, MutAnyOrigin](),
        0,
        0,
    )


fn _hd_find_value(
    buf: UnsafePointer[UInt8, MutAnyOrigin], n: Int, key: String
) -> Int:
    """Cari `"key"` dalam JSON datar; kembalikan offset nilai setelah ':'
    (spasi dilewati), atau -1 bila tidak ketemu."""
    var kb = key.as_bytes()
    var klen = len(kb)
    if klen < 2:
        return -1
    var i = 0
    var found = -1
    while i + klen <= n:
        var ok = True
        for j in range(klen):
            if buf[i + j] != kb[j]:
                ok = False
                break
        if ok:
            found = i
            break
        i += 1
    if found < 0:
        return -1
    var p = found + klen
    while p < n and (
        buf[p] == UInt8(ord(" ")) or buf[p] == UInt8(ord("\n"))
        or buf[p] == UInt8(ord("\t")) or buf[p] == UInt8(ord("\r"))
    ):
        p += 1
    if p >= n:
        return -1
    if buf[p] != UInt8(ord(":")):
        return -1
    p += 1
    while p < n and (
        buf[p] == UInt8(ord(" ")) or buf[p] == UInt8(ord("\n"))
        or buf[p] == UInt8(ord("\t")) or buf[p] == UInt8(ord("\r"))
    ):
        p += 1
    return p


fn _hd_skip_ws(buf: UnsafePointer[UInt8, MutAnyOrigin], p: Int, n: Int) -> Int:
    var q = p
    while q < n and (
        buf[q] == UInt8(ord(" ")) or buf[q] == UInt8(ord("\n"))
        or buf[q] == UInt8(ord("\t")) or buf[q] == UInt8(ord("\r"))
    ):
        q += 1
    return q


# Mojo (rilis 25.x yang dipakai proyek ini) tidak mendukung kembalian tuple,
# jadi offset akhir dikembalikan lewat pointer `end` (1 elemen).
fn _hd_scan_int(
    buf: UnsafePointer[UInt8, MutAnyOrigin], p: Int, n: Int,
    end: UnsafePointer[Int, MutAnyOrigin]
) -> Int:
    """Pindai integer bertanda di p. Mengembalikan nilainya; offset setelah
    angka ditulis ke end[]. Gagal -> end[] == p dan hasil 0."""
    var q = p
    var neg = False
    if q < n and (buf[q] == UInt8(ord("-")) or buf[q] == UInt8(ord("+"))):
        neg = (buf[q] == UInt8(ord("-")))
        q += 1
    var v = 0
    var any = False
    while q < n and buf[q] >= UInt8(ord("0")) and buf[q] <= UInt8(ord("9")):
        v = v * 10 + (Int(buf[q]) - Int(ord("0")))
        any = True
        q += 1
    if not any:
        end[] = p
        return 0
    end[] = q
    return Int(-1) * v if neg else v


fn _hd_scan_sign(
    buf: UnsafePointer[UInt8, MutAnyOrigin], p: Int, n: Int,
    end: UnsafePointer[Int, MutAnyOrigin]
) -> Float32:
    """Pindai satu elemen sign_values (hanya ±1 diperbolehkan: format
    "-1.0" / "1.0"). Mengembalikan nilainya; offset setelahnya ke end[].
    Gagal -> end[] == p dan hasil 0.0."""
    var q = p
    var neg = False
    if q < n and buf[q] == UInt8(ord("-")):
        neg = True
        q += 1
    else:
        if q < n and buf[q] == UInt8(ord("+")):
            q += 1
    # Harus dimulai dengan '1' (satu-satunya magnitudo yang valid).
    if q >= n or buf[q] != UInt8(ord("1")):
        end[] = p
        return Float32(0.0)
    q += 1
    # Lewati bagian pecahan/eksponen (.0, e0, dsb.) sampai ',' atau ']'.
    while q < n and buf[q] != UInt8(ord(",")) and buf[q] != UInt8(ord("]")):
        q += 1
    end[] = q
    return Float32(-1.0) if neg else Float32(1.0)


fn load_hadamard_signs(model_dir: String) raises -> HadamardSigns:
    """Parse hadamard.json pack Bonsai-2 -> vektor signs per lebar input K.

    Kontrak (prism.hadamard.*):
      block_size     : ukuran blok Hadamard (512/1024/2048/4096)
      sign_widths    : [w0, w1, ...]
      sign_values    : concat dari vektor signs [w0]+[w1]+..., semua ±1
    """
    var path = model_dir + "/hadamard.json"
    var cap = 1 << 20
    var buf = alloc[UInt8](cap)
    var n = read_small_file(path, buf, cap)
    # Offset akhir pemindaian dikembalikan lewat pointer (Mojo 25.x tanpa tuple).
    var hd_end = alloc[Int](1)
    if n <= 0:
        buf.free()
        hd_end.free()
        raise Error("FATAL: tidak dapat membaca hadamard.json di " + path)
    var p_block = _hd_find_value(buf, n, "\"prism.hadamard.block_size\"")
    if p_block < 0:
        buf.free()
        hd_end.free()
        raise Error("FATAL: hadamard.json: prism.hadamard.block_size tidak ada")
    var block = _hd_scan_int(buf, p_block, n, hd_end)
    if block != 512 and block != 1024 and block != 2048 and block != 4096:
        buf.free()
        hd_end.free()
        raise Error(
            "FATAL: hadamard.json: block_size=" + String(block)
            + " tidak divalidasi (harus 512/1024/2048/4096)"
        )
    var p_w = _hd_find_value(buf, n, "\"prism.hadamard.sign_widths\"")
    if p_w < 0 or p_w >= n or buf[p_w] != UInt8(ord("[")):
        buf.free()
        hd_end.free()
        raise Error("FATAL: hadamard.json: sign_widths tidak ada/bukan array")
    var widths = alloc[Int](16)
    var n_widths = 0
    var q = p_w + 1
    q = _hd_skip_ws(buf, q, n)
    while q < n and buf[q] != UInt8(ord("]")):
        var w = _hd_scan_int(buf, q, n, hd_end)
        var q2 = hd_end[]
        if q2 == q:
            buf.free()
            widths.free()
            raise Error("FATAL: hadamard.json: sign_widths elemen tidak valid")
        if n_widths >= 16:
            buf.free()
            widths.free()
            raise Error("FATAL: hadamard.json: terlalu banyak sign_widths (>16)")
        widths[n_widths] = w
        n_widths += 1
        q = _hd_skip_ws(buf, q2, n)
        if q < n and buf[q] == UInt8(ord(",")):
            q += 1
            q = _hd_skip_ws(buf, q, n)
    q += 1  # lewati ']'
    # Vektor signs.
    var p_v = _hd_find_value(buf, n, "\"prism.hadamard.sign_values\"")
    if p_v < 0 or p_v >= n or buf[p_v] != UInt8(ord("[")):
        buf.free()
        widths.free()
        hd_end.free()
        raise Error("FATAL: hadamard.json: sign_values tidak ada/bukan array")
    var total = 0
    for i in range(n_widths):
        total += widths[i]
    var sbuf = alloc[Float32](total if total > 0 else 1)
    q = p_v + 1
    q = _hd_skip_ws(buf, q, n)
    var idx = 0
    while q < n and buf[q] != UInt8(ord("]")):
        var sv = _hd_scan_sign(buf, q, n, hd_end)
        var q2 = hd_end[]
        if q2 == q:
            buf.free()
            widths.free()
            sbuf.free()
            raise Error("FATAL: hadamard.json: sign_values elemen tidak valid")
        if idx >= total:
            buf.free()
            widths.free()
            sbuf.free()
            raise Error(
                "FATAL: hadamard.json: sign_values lebih panjang dari "
                + "jumlah sign_widths (" + String(total) + ")"
            )
        if sv != Float32(1.0) and sv != Float32(-1.0):
            buf.free()
            widths.free()
            sbuf.free()
            raise Error("FATAL: hadamard.json: sign_values bukan ±1")
        sbuf[idx] = sv
        idx += 1
        q = _hd_skip_ws(buf, q2, n)
        if q < n and buf[q] == UInt8(ord(",")):
            q += 1
            q = _hd_skip_ws(buf, q, n)
    if idx != total:
        buf.free()
        widths.free()
        sbuf.free()
        raise Error(
            "FATAL: hadamard.json: sign_values [" + String(idx)
            + "] != jumlah sign_widths [" + String(total) + "]"
        )
    buf.free()
    hd_end.free()
    print(">> [HADAMARD] block=", block, " widths=",
          widths[0] if n_widths > 0 else 0,
          widths[1] if n_widths > 1 else 0,
          widths[2] if n_widths > 2 else 0,
          " total_signs=", total)
    # Konstruktor struct Mojo bersifat posisional (tanpa argumen kata kunci).
    return HadamardSigns(block, sbuf, widths, n_widths, total)


fn hadamard_signs_for_k(
    hd: HadamardSigns, k: Int
) -> UnsafePointer[Float32, MutAnyOrigin]:
    """Vektor signs untuk modul dengan lebar input k (pointer nol = modul
    tidak terfold). Semua modul dengan K yang sama berbagi vektor yang
    sama (kontrak runtime.py)."""
    var off = 0
    for i in range(hd.n_widths):
        if hd.widths[i] == k:
            return hd.buf + off
        off += hd.widths[i]
    return UnsafePointer[Float32, MutAnyOrigin]()


fn fwht_blocks_host(
    x: UnsafePointer[Float32, MutAnyOrigin],
    signs: UnsafePointer[Float32, MutAnyOrigin],
    k: Int, block: Int, inverse: Bool
) raises:
    """FWHT blok-blok Sylvester ternormalisasi √B pada host; paritas eksak
    runtime/runtime.py:fwht — signs dikalikan SEBELUM butterfly (forward)
    atau SESUDAHNYA (inverse). In-place; paritas numerik butterfly
    "rendah=x+y, tinggi=x-y" == H_B/√B."""
    if k % block != 0:
        raise Error(
            "FWHT host: k=" + String(k) + " tidak habis dibagi block="
            + String(block)
        )
    # Uji null pointer memakai perbandingan eksplisit (idiom codebase);
    # `if ptr:` saja tidak lazim dipakai di Mojo 25.x.
    if signs != UnsafePointer[Float32, MutAnyOrigin]():
        if not inverse:
            for i in range(k):
                x[i] = x[i] * signs[i]
    var inv_sqrt = Float32(1.0) / sqrt(Float32(block))
    var nblocks = k // block
    for b in range(nblocks):
        var base = b * block
        var h = 1
        while h < block:
            var i = 0
            while i < block:
                var j = i
                while j < i + h:
                    var a = x[base + j]
                    var bb = x[base + j + h]
                    x[base + j] = a + bb
                    x[base + j + h] = a - bb
                    j += 1
                i += 2 * h
            h = h * 2
        for i2 in range(block):
            x[base + i2] = x[base + i2] * inv_sqrt
    if signs != UnsafePointer[Float32, MutAnyOrigin]():
        if inverse:
            for i in range(k):
                x[i] = x[i] * signs[i]


fn embed_lookup_2bit_host(
    embed_w: UnsafePointer[UInt8, MutAnyOrigin],   # U32 [V, K/16] (LE, 16 bobot/word)
    embed_s: UnsafePointer[Float32, MutAnyOrigin], # [V, K/128]
    embed_b: UnsafePointer[Float32, MutAnyOrigin], # [V, K/128]
    signs: UnsafePointer[Float32, MutAnyOrigin],   # [K] ±1 (inverse) atau null
    tok: Int, k: Int, block: Int,
    row32: UnsafePointer[Float32, MutAnyOrigin],   # scratch [K]
    row16: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]  # scratch [K]
) raises:
    """Lookup embedding 2-bit + inverse FWHT (model.embed_tokens adalah
    satu-satunya modul inverse_weight_names). w = q*s + b (affine penuh,
    b == -s untuk pack ini -> ekuivalen (q-1)*s)."""
    var words_per_row = k // 16
    var groups_per_row = k // 128
    var off = tok * words_per_row * 4
    for g in range(groups_per_row):
        var s_val = embed_s[tok * groups_per_row + g]
        var b_val = embed_b[tok * groups_per_row + g]
        for j in range(8):   # 8 word U32 per grup 128 bobot
            var wb = off + (g * 8 + j) * 4
            var w = UInt32(embed_w[wb]) | (UInt32(embed_w[wb + 1]) << 8) | (
                UInt32(embed_w[wb + 2]) << 16
            ) | (UInt32(embed_w[wb + 3]) << 24)
            for lane in range(16):   # lane i di bit 2i
                var q = Float32(Int((w >> UInt32(lane * 2)) & UInt32(3)))
                row32[g * 128 + j * 16 + lane] = q * s_val + b_val
    if signs != UnsafePointer[Float32, MutAnyOrigin]():
        fwht_blocks_host(row32, signs, k, block, True)
    for i in range(k):
        row16[i] = row32[i].cast[DType.float16]()


fn embed_lookup_2bit_to_dev(
    gpu_ctx_ptr: UnsafePointer[DeviceContextGPU, MutAnyOrigin],
    embed_w: UnsafePointer[UInt8, MutAnyOrigin],
    embed_s: UnsafePointer[Float32, MutAnyOrigin],
    embed_b: UnsafePointer[Float32, MutAnyOrigin],
    signs: UnsafePointer[Float32, MutAnyOrigin],
    tok: Int, k: Int, block: Int,
    dst_dev: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    scratch_buf: DeviceBuffer[DType.float16],
    row32: UnsafePointer[Float32, MutAnyOrigin],
    row16: UnsafePointer[Scalar[DType.float16], MutAnyOrigin]
) raises:
    """Lookup embedding 2-bit di host lalu salin ke device (H2D + D2D kecil;
    embed hanya 1 baris K elemen per token)."""
    embed_lookup_2bit_host(
        embed_w, embed_s, embed_b, signs, tok, k, block, row32, row16
    )
    var ctx = gpu_ctx_ptr[]
    ctx.enqueue_copy(scratch_buf, row16)
    copy_vec_sm75_launch_on[DType.float16](
        ctx, dst_dev, scratch_buf.unsafe_ptr(), k
    )


fn rng_next(state: UnsafePointer[UInt64, MutAnyOrigin]) -> UInt64:
    """Xorshift64 — cukup untuk sampling, tidak untuk kriptografi."""
    var x = state[0]
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    state[0] = x
    return x


fn sample_from_logits(
    h_logits: UnsafePointer[Scalar[DType.float16], MutAnyOrigin],
    V: Int,
    cand_val: UnsafePointer[Float32, MutAnyOrigin],
    topk_val: UnsafePointer[Float32, MutAnyOrigin],
    topk_p: UnsafePointer[Float32, MutAnyOrigin],
    topk_idx: UnsafePointer[Int32, MutAnyOrigin],
    rep_flag: UnsafePointer[UInt8, MutAnyOrigin],
    temp_x100: Int,
    top_k_in: Int,
    top_p_x1000: Int,
    min_p_x1000: Int,
    rep_x100: Int,
    rep_win: Int,
    rng_state: UnsafePointer[UInt64, MutAnyOrigin],
    generated: UnsafePointer[Int, MutAnyOrigin],
    n_generated: Int,
) -> Int:
    """Sampling sesuai resep resmi Bonsai: top-k -> top-p -> min-p -> temperature.

    Parameter env diskalakan ke integer (temp_x100, top_p_x1000, min_p_x1000,
    rep_x100) supaya tidak perlu parsing float. Hanya dipakai bila temp_x100 > 0
    — jalur greedy lama sama sekali tidak tersentuh.

    Urutan mengikuti rantai sampler llama.cpp: top-k, top-p, dan min-p bekerja
    pada logits MENTAH; temperature diterapkan paling akhir, tepat sebelum
    pengundian. Repetition penalty tidak ada di spesifikasi resmi Bonsai, jadi
    default-nya mati (rep_x100 = 100).

    Buffer topk_val/topk_p/topk_idx disediakan pemanggil, ukurannya K_MAX = 256.
    """
    # top-k dibatasi supaya buffer tetap kecil; top-k > 256 tidak realistis.
    var K = top_k_in
    if K < 1:
        K = 1
    if K > 256:
        K = 256
    var rep_pen = Float32(rep_x100) / 100.0

    # 1) Tandai token yang sudah muncul di jendela terakhir (rep_win token).
    var lo = n_generated - rep_win
    if lo < 0:
        lo = 0
    var use_rep = rep_pen != 1.0 and n_generated > 0
    if use_rep:
        for j in range(lo, n_generated):
            rep_flag[generated[j]] = 1

    # 2) Logits mentah (+ penalti opsional) ke cand_val. TIDAK diskala suhu dulu:
    #    top-k/top-p/min-p resmi bekerja pada logits mentah.
    for i in range(V):
        var x = Float32(h_logits[i])
        if use_rep and rep_flag[i] == 1:
            x = x / rep_pen if x > 0.0 else x * rep_pen
        cand_val[i] = x

    if use_rep:
        for j in range(lo, n_generated):
            rep_flag[generated[j]] = 0

    # 3) top-k: sisipkan ke daftar kecil yang selalu urut menurun.
    #    O(V) dengan konstanta kecil — hanya geser bila kandidat lolos ambang.
    var cnt = 0
    for i in range(V):
        var x = cand_val[i]
        if cnt < K:
            var p = cnt
            while p > 0 and topk_val[p - 1] < x:
                topk_val[p] = topk_val[p - 1]
                topk_idx[p] = topk_idx[p - 1]
                p -= 1
            topk_val[p] = x
            topk_idx[p] = Int32(i)
            cnt += 1
        elif x > topk_val[K - 1]:
            var p = K - 1
            while p > 0 and topk_val[p - 1] < x:
                topk_val[p] = topk_val[p - 1]
                topk_idx[p] = topk_idx[p - 1]
                p -= 1
            topk_val[p] = x
            topk_idx[p] = Int32(i)

    if cnt == 0:
        return 0

    # 4) Softmax atas kandidat top-k (stabil: kurangi nilai maksimum).
    var m = topk_val[0]
    var total = Float32(0.0)
    for j in range(cnt):
        var e = exp(topk_val[j] - m)
        topk_p[j] = e
        total += e
    if total <= Float32(0.0):
        return Int(topk_idx[0])
    for j in range(cnt):
        topk_p[j] = topk_p[j] / total

    # 5) top-p (nucleus): ambil prefix terkecil dengan kumulatif >= p.
    var p_target = Float32(top_p_x1000) / 1000.0
    var keep = 1
    var acc = Float32(0.0)
    for j in range(cnt):
        acc += topk_p[j]
        keep = j + 1
        if acc >= p_target:
            break

    # 6) min-p: buang token dengan peluang < min_p x peluang maksimum.
    #    Karena sudah urut menurun, yang lolos selalu berupa prefix.
    var keep2 = keep
    if min_p_x1000 > 0:
        var thr = (Float32(min_p_x1000) / 1000.0) * topk_p[0]
        keep2 = 1
        for j in range(keep):
            if topk_p[j] >= thr:
                keep2 = j + 1

    # 7) Temperature diterapkan paling akhir, lalu pengundian kumulatif.
    var inv_temp = 100.0 / Float32(temp_x100)
    var tot = Float32(0.0)
    for j in range(keep2):
        var w = exp((topk_val[j] - topk_val[0]) * inv_temp)
        topk_p[j] = w
        tot += w
    if tot <= Float32(0.0):
        return Int(topk_idx[0])

    var u = Float32(Float64(rng_next(rng_state) >> UInt64(11)) / 9007199254740992.0)
    var target = u * tot
    var acc2 = Float32(0.0)
    var pick = Int(topk_idx[0])
    for j in range(keep2):
        acc2 += topk_p[j]
        if acc2 >= target:
            pick = Int(topk_idx[j])
            break
    return pick


fn main() raises:
    # ---------------- 0. Argumen CLI ----------------
    var args = argv()
    var model_dir = String()
    var prompt_spec = String()
    var max_tokens = 64
    var have_model_dir = False
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--model-dir" and i + 1 < len(args):
            i += 1
            model_dir = String(args[i])
            have_model_dir = True
        elif a == "--prompt-tokens" and i + 1 < len(args):
            i += 1
            prompt_spec = String(args[i])
        elif a == "--max-tokens" and i + 1 < len(args):
            i += 1
            max_tokens = Int(args[i])
        elif a == "--gpu":
            _ = setenv("BONSAI_USE_GPU", "1", 1)
        i += 1

    if not have_model_dir or len(prompt_spec) == 0:
        print("Pemakaian: mojo run main.mojo -- --model-dir <dir> --prompt-tokens <id,id,...> --max-tokens <n> --gpu")
        return

    # GAGAL CEPAT bila jalur GPU tidak aktif. Tanpa ini program memuat SELURUH
    # bobot 1-bit (~2,8 GB dari shard safetensors) dulu, baru berhenti di akhir
    # main() — membuang waktu menit-an dan mengubur salah-set env di balik log
    # pemuatan yang panjang. CPU fallback sudah dihapus, jadi tidak ada alasan
    # melanjutkan tanpa GPU.
    if not use_gpu_matmul():
        raise Error(
            "Jalur GPU wajib: set BONSAI_USE_GPU=1 atau jalankan dengan flag "
            "--gpu. CPU fallback (host-sim) telah dihapus — program berhenti di "
            "sini, SEBELUM memuat bobot."
        )
    print(">> [MOJO-NATIVE] Inferensi Bonsai-27B-mlx-1bit (JALUR GPU T4)")

    # ---------------- 1. Config ----------------
    var cfg = QwenConfig.qwen_27b_default()
    var cfg_path = model_dir + "/" + "config.json"
    var cfg_buf = alloc[UInt8](1 << 20)
    var cfg_len = read_small_file(cfg_path, cfg_buf, 1 << 20)
    if cfg_len > 0:
        var cdoc = JsonDoc()
        if cdoc.parse_bytes(cfg_buf, cfg_len):
            var r = cdoc.root()
            # Repo asli = kemasan VLM: parameter teks bersarang di 'text_config'
            var tcfg = cdoc.obj_get(r, "text_config")
            if tcfg != -1:
                r = tcfg
            var nd = cdoc.obj_get(r, "hidden_size")
            if nd != -1:
                cfg.hidden_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "intermediate_size")
            if nd != -1:
                cfg.intermediate_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_hidden_layers")
            if nd != -1:
                cfg.num_hidden_layers = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_attention_heads")
            if nd != -1:
                cfg.num_attention_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "num_key_value_heads")
            if nd != -1:
                cfg.num_key_value_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "head_dim")
            if nd != -1:
                cfg.head_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "vocab_size")
            if nd != -1:
                cfg.vocab_size = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "rms_norm_eps")
            if nd != -1:
                cfg.rms_norm_eps = Float32(cdoc.as_f64(nd))
            nd = cdoc.obj_get(r, "rope_theta")
            if nd != -1:
                cfg.rope_theta = Float32(cdoc.as_f64(nd))
            nd = cdoc.obj_get(r, "partial_rotary_factor")
            if nd != -1:
                cfg.partial_rotary_factor = Float32(cdoc.as_f64(nd))
                cfg.rotary_dim = Int(Float32(cfg.head_dim) * cfg.partial_rotary_factor)
            # rope_theta ASLI ada di objek bersarang "rope_parameters"
            # (qwen3_5: 10000000) — key flat "rope_theta" TIDAK ADA di
            # text_config; tanpa ini RoPE memakai default 100000 (salah).
            # partial_rotary_factor JUGA bersarang di sana (lihat definisi
            # arsitektur mlx-lm qwen3_5.py:55-84) — sebelumnya hanya dibaca
            # dari level atas, sehingga rotary_dim diam-diam tertinggal di
            # nilai default. rotary_dim dihitung ULANG setelah override ini.
            var rp = cdoc.obj_get(r, "rope_parameters")
            if rp != -1:
                var rt = cdoc.obj_get(rp, "rope_theta")
                if rt != -1:
                    cfg.rope_theta = Float32(cdoc.as_f64(rt))
                var prf = cdoc.obj_get(rp, "partial_rotary_factor")
                if prf != -1:
                    cfg.partial_rotary_factor = Float32(cdoc.as_f64(prf))
            cfg.rotary_dim = Int(
                Float32(cfg.head_dim) * cfg.partial_rotary_factor
            )
            print(">> Config RoPE: theta=", cfg.rope_theta,
                  " partial_rotary_factor=", cfg.partial_rotary_factor,
                  " rotary_dim=", cfg.rotary_dim, " head_dim=", cfg.head_dim)
            nd = cdoc.obj_get(r, "linear_num_value_heads")
            if nd != -1:
                cfg.gdn_num_v_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_num_key_heads")
            if nd != -1:
                cfg.gdn_num_k_heads = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_key_head_dim")
            if nd != -1:
                cfg.gdn_head_k_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_value_head_dim")
            if nd != -1:
                cfg.gdn_head_v_dim = cdoc.as_int(nd)
            nd = cdoc.obj_get(r, "linear_conv_kernel_dim")
            if nd != -1:
                cfg.gdn_conv_kernel = cdoc.as_int(nd)
            cfg.gdn_conv_dim = 2 * (cfg.gdn_num_k_heads * cfg.gdn_head_k_dim) + (cfg.gdn_num_v_heads * cfg.gdn_head_v_dim)
        cdoc.free()
    cfg_buf.free()

    var D = cfg.hidden_size
    var V = cfg.vocab_size
    var bits = bonsai_bits()
    print(">> Config: layers=", cfg.num_hidden_layers, " hidden=", D, " vocab=", V,
          " bits=", bits)

    # ---------------- 1b. Hadamard manifest (Bonsai-2) ----------------
    # signs ±1 untuk modul terfold diambil dari hadamard.json, BUKAN dari
    # tensor .signs di safetensors (kontrak runtime/runtime.py: signs dipetakan
    # per LEBAR INPUT K, dibagikan semua modul dgn K sama).
    var hd = hadamard_signs_none()
    if bits == 2:
        hd = load_hadamard_signs(model_dir)

    # ---------------- 2. Buka safetensors ----------------
    print(">> [STEP] open_dir...")
    var index = SafeTensorsIndex()
    index.open_dir(model_dir)
    if not index.ok:
        raise Error("FATAL: tidak ada tensor safetensors ditemukan di " + model_dir)
    print(">> [STEP] indeks:", index.n_entries, "tensor /", index.n_shards, "shard")

    # ---------------- 3. Muat bobot global ----------------
    # 1-bit: bobot U8 terpaket 8/byte (Q1_0_g128). 2-bit: U32 [N, K/16],
    # 16 bobot/word LE, lane i di bit 2i (prism_hadamard_qwen35).
    print(">> [STEP] cari embed...")
    var e_emb = find_flex(index, "model.embed_tokens.weight")
    var e_emb_s = find_flex(index, "model.embed_tokens.scales")
    # Gagal KERAS (raise), bukan print+return: `return` dari main() keluar
    # dengan status 0, sehingga gerbang CI/deploy membaca "sukses" padahal
    # inferensi tidak pernah berjalan.
    if e_emb == -1 or e_emb_s == -1:
        raise Error("FATAL: embed_tokens/scales tidak ditemukan (kedua prefix).")
    var V_emb = index.entries[e_emb].d0
    # 1-bit: 32 bobot/kata U32; 2-bit: 16 bobot/kata U32.
    var D_real = index.entries[e_emb].d1 * (32 // bits)
    var embed_w = alloc[UInt8](index.entries[e_emb].nbytes)
    must_read_raw(index, e_emb, embed_w, index.entries[e_emb].nbytes, "model.embed_tokens.weight")
    var embed_s = alloc[Float32](V_emb * (D_real // 128))
    must_read_f32(index, e_emb_s, embed_s, V_emb * (D_real // 128), "model.embed_tokens.scales")
    var e_emb_b = find_flex(index, "model.embed_tokens.biases")
    var embed_b = alloc[Float32](V_emb * (D_real // 128))
    # Skala embed selalu MENTAH (kernel embed affine penuh q*s+b untuk 1-bit;
    # lookup 2-bit juga q*s+b dgn b == -s). Bagi 2 hanya untuk path 1-bit
    # proyeksi — tidak berlaku embed.
    if e_emb_b != -1:
        must_read_f32(index, e_emb_b, embed_b, V_emb * (D_real // 128), "embed_tokens.biases")
        verify_affine_zero_bias(
            embed_s, embed_b, V_emb * (D_real // 128), False, "model.embed_tokens", bits
        )
    else:
        for i in range(V_emb * (D_real // 128)):
            embed_b[i] = 0.0
    # Bonsai-2: embed adalah satu-satunya modul inverse_weight_names —
    # keluarannya harus di-inverse-FWHT. signs[K] wajib ada.
    var embed_signs = UnsafePointer[Float32, MutAnyOrigin]()
    if bits == 2:
        embed_signs = hadamard_signs_for_k(hd, D_real)
        if embed_signs == UnsafePointer[Float32, MutAnyOrigin]():
            raise Error(
                "FATAL: pack Bonsai-2 tidak memiliki vektor signs untuk K="
                + String(D_real) + " (hadamard.json) — modul embed terfold."
            )
    D = D_real
    V = V_emb
    cfg.hidden_size = D
    cfg.vocab_size = V

    var final_norm_w = alloc[Float32](D)
    var e_fnorm = find_flex(index, "model.norm.weight")
    if e_fnorm != -1:
        must_read_f32(index, e_fnorm, final_norm_w, D, "model.norm.weight")

    var e_lm = find_flex(index, "lm_head.weight")
    var e_lm_s = find_flex(index, "lm_head.scales")
    if e_lm == -1 or e_lm_s == -1:
        raise Error("FATAL: lm_head.weight / lm_head.scales tidak ditemukan.")
    var V_lm = index.entries[e_lm].d0
    var lm_w = alloc[UInt8](index.entries[e_lm].nbytes)
    must_read_raw(index, e_lm, lm_w, index.entries[e_lm].nbytes, "lm_head.weight")
    var lm_s = alloc[Float32](V_lm * (D // 128))
    must_read_f32(index, e_lm_s, lm_s, V_lm * (D // 128), "lm_head.scales")
    var e_lm_b = find_flex(index, "lm_head.biases")
    var lm_b = alloc[Float32](V_lm * (D // 128))
    if e_lm_b != -1:
        must_read_f32(index, e_lm_b, lm_b, V_lm * (D // 128), "lm_head.biases")
    # bits=1: scales_eff = s_ckpt/2 (kernel (2q-1)*s_eff).
    # bits=2: scales mentah (kernel (q-1)*s); lm_head terfold -> FWHT forward.
    if bits == 1:
        for i in range(V_lm * (D // 128)):
            lm_s[i] = lm_s[i] * 0.5
    if e_lm_b != -1:
        # bits=1: skala sudah s_eff -> kernel MENDERIVASI bias = -s_eff.
        # bits=2: skala mentah, harapan b == -s.
        verify_affine_zero_bias(
            lm_s, lm_b, V_lm * (D // 128), bits == 1, "lm_head", bits
        )
    else:
        print(">> [AFFINE-WARN] lm_head.biases tidak ada — kontrak affine "
              "TIDAK dapat diverifikasi.")
    var lm_signs = hadamard_signs_for_k(hd, D)
    var lm_proj = QwenLinear1Bit(lm_w, lm_s, lm_b, V_lm, D, bits, lm_signs)
    V = V_lm
    print(">> [STEP] bobot global siap (packed", bits, "bit, V=", V, " D=", D,
          " lm_signs=", lm_signs != UnsafePointer[Float32, MutAnyOrigin](), ")")

    # ---------------- 4. Muat bobot per layer (dengan fusion Paket C) ----------------
    var n_layers = cfg.num_hidden_layers

    # KHQ: dump K/V untuk kalibrasi (env BONSAI_DUMP_KV_DIR) — K unroped.
    var khq_dir = getenv("BONSAI_DUMP_KV_DIR")
    if khq_dir:
        khq_dump_configure(
            khq_dir, 2048, cfg.num_key_value_heads * cfg.head_dim
        )
        print(">> [KHQ-DUMP] aktif ->", khq_dir)

    # KHQ: aktivasi dipindah ke setelah DeviceContext terpasang (lihat bawah).

    var layers = alloc[QwenDecoderLayer](n_layers)
    var gdn_states = alloc[GatedDeltaNetState](n_layers)
    var kv_caches = alloc[AttentionKVCache](n_layers)
    # pemetaan layer -> indeks state (GDN) / cache (attention)
    var gdn_idx = alloc[Int](n_layers)
    var kv_idx = alloc[Int](n_layers)
    var n_gdn = 0
    var n_kv = 0
    var max_seq = 4096

    for li in range(n_layers):
        var is_linear = (li % cfg.full_attention_interval) != (cfg.full_attention_interval - 1)
        layers[li] = QwenDecoderLayer(layer_idx=li, is_linear=is_linear, config=cfg)
        var prefix = "model.layers." + String(li) + "."
        # norm
        var ln1 = alloc[Float32](D)
        var e_ln1 = find_flex(index, prefix + "input_layernorm.weight")
        if e_ln1 != -1:
            must_read_f32(index, e_ln1, ln1, D, prefix + "input_layernorm.weight")
        var ln2 = alloc[Float32](D)
        var e_ln2 = find_flex(index, prefix + "post_attention_layernorm.weight")
        if e_ln2 != -1:
            must_read_f32(index, e_ln2, ln2, D, prefix + "post_attention_layernorm.weight")
        layers[li].input_layernorm_w = ln1
        layers[li].post_attn_layernorm_w = ln2

        if is_linear:
            # GDN: in_proj_all (fused) atau fusion manual qkv+z+b+a
            # Bonsai-2: in_proj_b / in_proj_a adalah F32 [H_v, D] TIDAK
            # terkuantisasi dan TIDAK terfold (satu-satunya modul LM dense di
            # pack ini). Mereka menjadi EKOR DENSE [n_tail, K] pada
            # QwenLinear1Bit — layout output GDN tetap [qkv | z | b | a].
            var w_all = UnsafePointer[UInt8, MutAnyOrigin]()
            var s_all = UnsafePointer[Float32, MutAnyOrigin]()
            var b_all = UnsafePointer[Float32, MutAnyOrigin]()
            var n_all = 0
            var k_all = 0
            var tail_w = UnsafePointer[Float32, MutAnyOrigin]()
            var n_tail = 0
            # b/a dense (F32) — ada di pack Bonsai-2 maupun 1-bit lama TIDAK
            # memilikinya (semua 4 sub-modul terkuantisasi).
            var e_ib = find_flex(index, prefix + "linear_attn.in_proj_b.weight")
            var e_ia = find_flex(index, prefix + "linear_attn.in_proj_a.weight")
            if bits == 2 and e_ib != -1 and e_ia != -1:
                var n_b = index.dim0(e_ib)
                var k_b = index.entries[e_ib].d1
                var n_a = index.dim0(e_ia)
                var k_a = index.entries[e_ia].d1
                if k_b != k_a or n_b != n_a:
                    raise Error(
                        "FATAL: in_proj_b/a shape tidak konsisten pada layer "
                        + String(li)
                    )
                var bb = alloc[Float32](n_b * k_b)
                must_read_f32(index, e_ib, bb, n_b * k_b, prefix + "in_proj_b")
                var aa = alloc[Float32](n_a * k_a)
                must_read_f32(index, e_ia, aa, n_a * k_a, prefix + "in_proj_a")
                tail_w = fuse_f32(bb, n_b * k_b, aa, n_a * k_a)
                n_tail = n_b + n_a
                bb.free()
                aa.free()
            elif bits == 2 and (e_ib != -1 or e_ia != -1):
                raise Error(
                    "FATAL: hanya salah satu in_proj_b/a ditemukan layer "
                    + String(li)
                )
            var e_fused_w = find_flex(index, prefix + "linear_attn.in_proj_all.weight")
            if e_fused_w != -1:
                var nb = index.entries[e_fused_w].nbytes
                w_all = alloc[UInt8](nb)
                must_read_raw(index, e_fused_w, w_all, nb, prefix + "linear_attn.in_proj_all.weight")
                n_all = index.dim0(e_fused_w)
                k_all = nb * 8 // bits // n_all
                var e_fused_s = find_flex(index, prefix + "linear_attn.in_proj_all.scales")
                s_all = read_scales_eff(index, e_fused_s, n_all * (k_all // 128), bits)
                var e_fused_b = find_flex(index, prefix + "linear_attn.in_proj_all.biases")
                b_all = read_biases_f32(index, e_fused_b, n_all * (k_all // 128))
                if e_fused_b != -1:
                    verify_affine_zero_bias(
                        s_all, b_all, n_all * (k_all // 128), bits == 1,
                        prefix + "linear_attn.in_proj_all", bits
                    )
            else:
                # fusion manual (paritas loader.py).
                # 1-bit: qkv+z+b+a semua terkuantisasi -> fuse 4.
                # 2-bit : qkv+z terkuantisasi -> fuse 2 (b/a jadi ekor dense).
                var pq = load_qlinear(index, prefix + "linear_attn.in_proj_qkv.weight", prefix + "linear_attn.in_proj_qkv.scales", prefix + "linear_attn.in_proj_qkv.biases", bits)
                var pz = load_qlinear(index, prefix + "linear_attn.in_proj_z.weight", prefix + "linear_attn.in_proj_z.scales", prefix + "linear_attn.in_proj_z.biases", bits)
                if not (pq.ok and pz.ok):
                    raise Error(
                        "FATAL: bobot GDN tidak lengkap pada layer " + String(li)
                        + " (in_proj_qkv/z)."
                    )
                var w01 = fuse_u8(pq.w, pq.nbytes, pz.w, pz.nbytes)
                var s01 = fuse_f32(pq.scales, pq.n_rows * (pq.k_dim // 128), pz.scales, pz.n_rows * (pz.k_dim // 128))
                var b01 = fuse_f32(pq.biases, pq.n_rows * (pq.k_dim // 128), pz.biases, pz.n_rows * (pz.k_dim // 128))
                if n_tail == 0:
                    # 1-bit: b/a juga terkuantisasi -> masukkan ke bagian packed.
                    var pb = load_qlinear(index, prefix + "linear_attn.in_proj_b.weight", prefix + "linear_attn.in_proj_b.scales", prefix + "linear_attn.in_proj_b.biases", bits)
                    var pa = load_qlinear(index, prefix + "linear_attn.in_proj_a.weight", prefix + "linear_attn.in_proj_a.scales", prefix + "linear_attn.in_proj_a.biases", bits)
                    if not (pb.ok and pa.ok):
                        raise Error(
                            "FATAL: bobot GDN tidak lengkap pada layer "
                            + String(li) + " (in_proj_b/a 1-bit)."
                        )
                    var w23 = fuse_u8(pb.w, pb.nbytes, pa.w, pa.nbytes)
                    w_all = fuse_u8(w01, pq.nbytes + pz.nbytes, w23, pb.nbytes + pa.nbytes)
                    var s23 = fuse_f32(pb.scales, pb.n_rows * (pb.k_dim // 128), pa.scales, pa.n_rows * (pa.k_dim // 128))
                    s_all = fuse_f32(s01, (pq.n_rows + pz.n_rows) * (pq.k_dim // 128), s23, (pb.n_rows + pa.n_rows) * (pb.k_dim // 128))
                    var b23 = fuse_f32(pb.biases, pb.n_rows * (pb.k_dim // 128), pa.biases, pa.n_rows * (pa.k_dim // 128))
                    b_all = fuse_f32(b01, (pq.n_rows + pz.n_rows) * (pq.k_dim // 128), b23, (pb.n_rows + pa.n_rows) * (pb.k_dim // 128))
                    n_all = pq.n_rows + pz.n_rows + pb.n_rows + pa.n_rows
                    w01.free()
                    s01.free()
                    b01.free()
                    w23.free()
                    s23.free()
                    b23.free()
                else:
                    # 2-bit: qkv+z saja yang packed; b/a sudah jadi ekor dense.
                    # w01/s01/b01 diambil alih (jangan di-free — ownership
                    # berpindah ke w_all/s_all/b_all, bukan salinan).
                    w_all = w01
                    s_all = s01
                    b_all = b01
                    n_all = pq.n_rows + pz.n_rows
                k_all = pq.k_dim
            layers[li].gdn_in_proj_all = QwenLinear1Bit(
                w_all, s_all, b_all, n_all + n_tail, k_all, bits,
                hadamard_signs_for_k(hd, k_all), tail_w, n_tail
            )

            # conv1d weight -> FP32
            var cw = alloc[Float32](cfg.gdn_conv_dim * cfg.gdn_conv_kernel)
            var e_cw = find_flex(index, prefix + "linear_attn.conv1d.weight")
            if e_cw != -1:
                must_read_f32(
                    index, e_cw, cw, cfg.gdn_conv_dim * cfg.gdn_conv_kernel,
                    prefix + "linear_attn.conv1d.weight"
                )
            layers[li].gdn_conv_weights = cw

            # out_proj
            var e_ow = find_flex(index, prefix + "linear_attn.out_proj.weight")
            var ow = UnsafePointer[UInt8, MutAnyOrigin]()
            var os_ = UnsafePointer[Float32, MutAnyOrigin]()
            var ob_ = UnsafePointer[Float32, MutAnyOrigin]()
            var on = 0
            var ok_ = 0
            if e_ow != -1:
                var nb = index.entries[e_ow].nbytes
                ow = alloc[UInt8](nb)
                must_read_raw(index, e_ow, ow, nb, prefix + "linear_attn.out_proj.weight")
                on = index.dim0(e_ow)
                ok_ = nb * 8 // bits // on
                var e_os = find_flex(index, prefix + "linear_attn.out_proj.scales")
                os_ = read_scales_eff(index, e_os, on * (ok_ // 128), bits)
                var e_ob = find_flex(index, prefix + "linear_attn.out_proj.biases")
                ob_ = read_biases_f32(index, e_ob, on * (ok_ // 128))
                if e_ob != -1:
                    verify_affine_zero_bias(
                        os_, ob_, on * (ok_ // 128), bits == 1,
                        prefix + "linear_attn.out_proj", bits
                    )
            layers[li].gdn_out_proj = QwenLinear1Bit(
                ow, os_, ob_, on, ok_, bits, hadamard_signs_for_k(hd, ok_)
            )

            # Parameter riil Qwen3-Next: A_log, dt_bias, norm GDN
            var e_al = find_flex(index, prefix + "linear_attn.A_log")
            if e_al != -1:
                var nn_al = index.entries[e_al].d0 * max(index.entries[e_al].d1, 1)
                var a_log = alloc[Float32](nn_al)
                must_read_f32(index, e_al, a_log, nn_al, prefix + "linear_attn.A_log")
                layers[li].gdn_a_log = a_log
            var e_dt = find_flex(index, prefix + "linear_attn.dt_bias")
            if e_dt != -1:
                var nn_dt = index.entries[e_dt].d0 * max(index.entries[e_dt].d1, 1)
                var dtb = alloc[Float32](nn_dt)
                must_read_f32(index, e_dt, dtb, nn_dt, prefix + "linear_attn.dt_bias")
                layers[li].gdn_dt_bias = dtb
            var e_gn = find_flex(index, prefix + "linear_attn.norm.weight")
            if e_gn != -1:
                var nn_gn = index.entries[e_gn].d0 * max(index.entries[e_gn].d1, 1)
                var gnw = alloc[Float32](nn_gn)
                must_read_f32(index, e_gn, gnw, nn_gn, prefix + "linear_attn.norm.weight")
                layers[li].gdn_norm_w = gnw
            if e_al != -1 and e_dt != -1 and e_gn != -1:
                layers[li].gdn_has_params = True

            # state GDN + pemetaan
            gdn_states[n_gdn] = GatedDeltaNetState(
                cfg.gdn_conv_dim, cfg.gdn_num_v_heads,
                cfg.gdn_head_v_dim, cfg.gdn_head_k_dim
            )
            gdn_idx[li] = n_gdn
            n_gdn += 1
            kv_idx[li] = -1
        else:
            # Gated Full Attention: q/k/v/o (eksplisit, paritas loader.py)
            var pq = load_qlinear(index, prefix + "self_attn.q_proj.weight", prefix + "self_attn.q_proj.scales", prefix + "self_attn.q_proj.biases", bits)
            var pk = load_qlinear(index, prefix + "self_attn.k_proj.weight", prefix + "self_attn.k_proj.scales", prefix + "self_attn.k_proj.biases", bits)
            var pv = load_qlinear(index, prefix + "self_attn.v_proj.weight", prefix + "self_attn.v_proj.scales", prefix + "self_attn.v_proj.biases", bits)
            var po = load_qlinear(index, prefix + "self_attn.o_proj.weight", prefix + "self_attn.o_proj.scales", prefix + "self_attn.o_proj.biases", bits)
            if not (pq.ok and pk.ok and pv.ok and po.ok):
                raise Error(
                    "FATAL: bobot attention tidak lengkap pada layer " + String(li)
                    + " (q/k/v/o_proj)."
                )
            layers[li].attn_q_proj = QwenLinear1Bit(pq.w, pq.scales, pq.biases, pq.n_rows, pq.k_dim, bits, hadamard_signs_for_k(hd, pq.k_dim))
            layers[li].attn_k_proj = QwenLinear1Bit(pk.w, pk.scales, pk.biases, pk.n_rows, pk.k_dim, bits, hadamard_signs_for_k(hd, pk.k_dim))
            layers[li].attn_v_proj = QwenLinear1Bit(pv.w, pv.scales, pv.biases, pv.n_rows, pv.k_dim, bits, hadamard_signs_for_k(hd, pv.k_dim))
            layers[li].attn_o_proj = QwenLinear1Bit(po.w, po.scales, po.biases, po.n_rows, po.k_dim, bits, hadamard_signs_for_k(hd, po.k_dim))

            # Bobot Q-Norm & K-Norm per head (paritas mlx-lm qwen3_next)
            var e_qn = find_flex(index, prefix + "self_attn.q_norm.weight")
            var nn_q = 0
            if e_qn != -1:
                nn_q = index.entries[e_qn].d0 * max(index.entries[e_qn].d1, 1)
                var qnw = alloc[Float32](nn_q)
                must_read_f32(index, e_qn, qnw, nn_q, prefix + "self_attn.q_norm.weight")
                layers[li].attn_q_norm_w = qnw
            var e_kn = find_flex(index, prefix + "self_attn.k_norm.weight")
            var nn_k = 0
            if e_kn != -1:
                nn_k = index.entries[e_kn].d0 * max(index.entries[e_kn].d1, 1)
                var knw = alloc[Float32](nn_k)
                must_read_f32(index, e_kn, knw, nn_k, prefix + "self_attn.k_norm.weight")
                layers[li].attn_k_norm_w = knw
            if e_qn != -1 and e_kn != -1:
                layers[li].attn_has_norms = True
            if li == 3:
                print(">> [ATTN LOAD] Layer", li, "attn_has_norms:", layers[li].attn_has_norms, "nn_q:", nn_q, "nn_k:", nn_k)
            # KV cache + pemetaan
            kv_caches[n_kv] = AttentionKVCache(
                max_seq, cfg.num_key_value_heads, cfg.head_dim
            )
            kv_idx[li] = n_kv
            n_kv += 1
            gdn_idx[li] = -1

        # MLP SwiGLU: gate_up fused atau gate+up
        var e_guw = find_flex(index, prefix + "mlp.gate_up_proj.weight")
        if e_guw != -1:
            var nb = index.entries[e_guw].nbytes
            var wb = alloc[UInt8](nb)
            must_read_raw(index, e_guw, wb, nb, prefix + "mlp.gate_up_proj.weight")
            var nn = index.dim0(e_guw)
            var kk = nb * 8 // bits // nn
            var e_gus = find_flex(index, prefix + "mlp.gate_up_proj.scales")
            var sb = read_scales_eff(index, e_gus, nn * (kk // 128), bits)
            var e_gub = find_flex(index, prefix + "mlp.gate_up_proj.biases")
            var gb = read_biases_f32(index, e_gub, nn * (kk // 128))
            if e_gub != -1:
                verify_affine_zero_bias(
                    sb, gb, nn * (kk // 128), bits == 1, prefix + "mlp.gate_up_proj", bits
                )
            layers[li].mlp_gate_up_proj = QwenLinear1Bit(
                wb, sb, gb, nn, kk, bits, hadamard_signs_for_k(hd, kk)
            )
        else:
            # MLP: fusion gate+up (paritas loader.py). gate & up sama-sama
            # K=D -> berbagi vektor signs yang sama -> fusion EKSAK.
            var pg = load_qlinear(index, prefix + "mlp.gate_proj.weight", prefix + "mlp.gate_proj.scales", prefix + "mlp.gate_proj.biases", bits)
            var pu = load_qlinear(index, prefix + "mlp.up_proj.weight", prefix + "mlp.up_proj.scales", prefix + "mlp.up_proj.biases", bits)
            if not (pg.ok and pu.ok):
                raise Error(
                    "FATAL: bobot MLP tidak lengkap pada layer " + String(li)
                    + " (gate_proj/up_proj)."
                )
            layers[li].mlp_gate_up_proj = QwenLinear1Bit(
                fuse_u8(pg.w, pg.nbytes, pu.w, pu.nbytes),
                fuse_f32(pg.scales, pg.n_rows * (pg.k_dim // 128), pu.scales, pu.n_rows * (pu.k_dim // 128)),
                fuse_f32(pg.biases, pg.n_rows * (pg.k_dim // 128), pu.biases, pu.n_rows * (pu.k_dim // 128)),
                pg.n_rows + pu.n_rows, pg.k_dim, bits,
                hadamard_signs_for_k(hd, pg.k_dim)
            )
        var e_dw = find_flex(index, prefix + "mlp.down_proj.weight")
        var e_ds = find_flex(index, prefix + "mlp.down_proj.scales")
        var e_db = find_flex(index, prefix + "mlp.down_proj.biases")
        if e_dw != -1 and e_ds != -1:
            var nb = index.entries[e_dw].nbytes
            var wb = alloc[UInt8](nb)
            must_read_raw(index, e_dw, wb, nb, prefix + "mlp.down_proj.weight")
            var nn = index.dim0(e_dw)
            var kk = nb * 8 // bits // nn
            var sb = read_scales_eff(index, e_ds, nn * (kk // 128), bits)
            var db = read_biases_f32(index, e_db, nn * (kk // 128))
            if e_db != -1:
                verify_affine_zero_bias(
                    sb, db, nn * (kk // 128), bits == 1, prefix + "mlp.down_proj", bits
                )
            layers[li].mlp_down_proj = QwenLinear1Bit(
                wb, sb, db, nn, kk, bits, hadamard_signs_for_k(hd, kk)
            )

    print(">> Bobot termuat:", n_layers, "layer (", n_gdn, "GDN,", n_kv, "attention )")
    print(">> [AFFINE] kontrak biases == -scales_ckpt/2 terverifikasi utk "
          "setiap tensor 1-bit yang dimuat (sampel; BONSAI_VERIFY_AFFINE=full "
          "utk menyisir semua elemen).")

    # DeviceContext tunggal: bobot diunggah sekali ke global memory T4, bukan
    # per token. Tanpa ini tiap proyeksi membuat-buang context (release race).
    alias T = DType.float16
    var gpu_ctx_ptr = UnsafePointer[DeviceContextGPU, MutAnyOrigin]()

    # Pointer activation workspace VRAM persisten (Full GPU resident)
    var act_hidden_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_x_norm_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_sublayer_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_mlp_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_proj_raw_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_conv_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_q_normed_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_k_normed_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_gdn_out_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_gate_up_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_swiglu_act_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_logits_dev = UnsafePointer[Scalar[T], MutAnyOrigin]()
    var act_token_out_dev = UnsafePointer[Int32, MutAnyOrigin]()
    var act_stage1_vals_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var act_stage1_idxs_dev = UnsafePointer[Int32, MutAnyOrigin]()
    var fnorm_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var act_attn_scores_dev = UnsafePointer[Float32, MutAnyOrigin]()
    var h_hidden_holder = alloc[DeviceBuffer[T]](1)
    var h_token_out_holder = alloc[DeviceBuffer[DType.int32]](1)
    # Buffer logits device di-hoist ke scope fungsi: blok setup buffer dan blok
    # loop decode adalah DUA `if use_gpu_matmul():` yang terpisah, jadi `h_log`
    # yang lokal di blok pertama tidak terlihat dari loop decode.
    var logits_buf = UnsafePointer[DeviceBuffer[T], MutAnyOrigin]()

    if use_gpu_matmul():
        gpu_ctx_ptr = gpu_ctx_new()
        for li in range(n_layers):
            layers[li].set_ctx(gpu_ctx_ptr)
        lm_proj.set_ctx(gpu_ctx_ptr)
        print(">> [GPU] DeviceContext tunggal terpasang ke", n_layers, "layer + lm_head")

        # KHQ: jalur KV terkompresi (env BONSAI_KHQ_PATH=<file centroid>).
        # Tanpa fallback: gagal aktivasi = berhenti, bukan diam-diam fp16.
        var khq_path = getenv("BONSAI_KHQ_PATH")
        if khq_path:
            if not khq_activate(gpu_ctx_ptr[], khq_path, 4096):
                raise Error("KHQ gagal aktif (centroid tidak valid): " + khq_path)

        # Inisialisasi VRAM KV Cache untuk seluruh layer Attention (Layer 3, 7, 11, ...)
        for ki in range(n_kv):
            kv_caches[ki].init_device(gpu_ctx_ptr[])

        # Inisialisasi VRAM state untuk seluruh layer GDN (48 layer)
        for gi in range(n_gdn):
            gdn_states[gi].init_device(gpu_ctx_ptr[])

        # Alokasi buffer aktivasi VRAM persisten (seumur proses, < 1 MB VRAM)
        h_hidden_holder.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_hidden_dev = h_hidden_holder[].unsafe_ptr()

        var h_x_norm = alloc[DeviceBuffer[T]](1)
        h_x_norm.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_x_norm_dev = h_x_norm[].unsafe_ptr()

        var h_sub = alloc[DeviceBuffer[T]](1)
        h_sub.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_sublayer_out_dev = h_sub[].unsafe_ptr()

        var h_mlp = alloc[DeviceBuffer[T]](1)
        h_mlp.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](D))
        act_mlp_out_dev = h_mlp[].unsafe_ptr()

        var h_proj = alloc[DeviceBuffer[T]](1)
        h_proj.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](17408))
        act_proj_raw_dev = h_proj[].unsafe_ptr()

        var h_conv = alloc[DeviceBuffer[T]](1)
        h_conv.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_conv_dim))
        act_conv_out_dev = h_conv[].unsafe_ptr()

        var h_qn = alloc[DeviceBuffer[T]](1)
        h_qn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_k_heads * cfg.gdn_head_k_dim))
        act_q_normed_dev = h_qn[].unsafe_ptr()

        var h_kn = alloc[DeviceBuffer[T]](1)
        h_kn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_k_heads * cfg.gdn_head_k_dim))
        act_k_normed_dev = h_kn[].unsafe_ptr()

        var h_gdn = alloc[DeviceBuffer[T]](1)
        h_gdn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.gdn_num_v_heads * cfg.gdn_head_v_dim))
        act_gdn_out_dev = h_gdn[].unsafe_ptr()

        var h_gu = alloc[DeviceBuffer[T]](1)
        h_gu.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](2 * cfg.intermediate_size))
        act_gate_up_dev = h_gu[].unsafe_ptr()

        var h_sw = alloc[DeviceBuffer[T]](1)
        h_sw.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](cfg.intermediate_size))
        act_swiglu_act_dev = h_sw[].unsafe_ptr()

        var h_log = alloc[DeviceBuffer[T]](1)
        h_log.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](V))
        act_logits_dev = h_log[].unsafe_ptr()
        logits_buf = h_log

        h_token_out_holder.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.int32](1))
        act_token_out_dev = h_token_out_holder[].unsafe_ptr()

        var h_s1v = alloc[DeviceBuffer[DType.float32]](1)
        h_s1v.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](256))
        act_stage1_vals_dev = h_s1v[].unsafe_ptr()

        var h_s1i = alloc[DeviceBuffer[DType.int32]](1)
        h_s1i.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.int32](256))
        act_stage1_idxs_dev = h_s1i[].unsafe_ptr()

        var h_fnorm = alloc[DeviceBuffer[DType.float32]](1)
        h_fnorm.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](D))
        gpu_ctx_ptr[].enqueue_copy(h_fnorm[], final_norm_w)
        fnorm_dev = h_fnorm[].unsafe_ptr()

        # Workspace attention scores (H_q * max_seq_len Float32)
        var h_as = alloc[DeviceBuffer[DType.float32]](1)
        h_as.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](cfg.num_attention_heads * max_seq))
        act_attn_scores_dev = h_as[].unsafe_ptr()

        print(">> [GPU] VRAM resident activation buffers siap.")

    # ---------------- 5. Prompt tokens ----------------
    var ptoks = parse_int_list(prompt_spec)
    var prompt_len = ptoks[0] # slot 0 = jumlah elemen
    print(">> Prompt tokens:", prompt_len, "| max_tokens:", max_tokens)

    # Penjaga panjang konteks. KV cache (`AttentionKVCache(max_seq, ...)`) dan
    # workspace skor attention (`H_q * max_seq` Float32) dialokasikan SEKALI
    # dengan ukuran `max_seq`; tidak ada satu pun batas di dalam kernel. Prompt
    # yang lebih panjang (atau prompt + generasi yang melewati `max_seq`) akan
    # membuat `kv_cache_append` menulis di luar buffer dan `gqa_attention`
    # mengindeks `attn_scores[hq*max_seq_len + t]` di luar workspace — kerusakan
    # memori SENYAP di GPU (bisa salah token, bisa crash, tanpa pesan).
    # `--prompt-tokens` datang dari CLI, jadi ini benar-benar bisa dipicu.
    if prompt_len <= 0:
        raise Error("FATAL: prompt kosong (0 token).")
    if prompt_len > max_seq:
        raise Error(
            "FATAL: panjang prompt " + String(prompt_len) + " > max_seq "
            + String(max_seq) + " — KV cache & workspace attention akan "
            + "meluap. Perpendek prompt (atau naikkan max_seq di main.mojo)."
        )
    if prompt_len + max_tokens > max_seq:
        var boleh = max_seq - prompt_len
        raise Error(
            "FATAL: prompt " + String(prompt_len) + " + max_tokens "
            + String(max_tokens) + " = " + String(prompt_len + max_tokens)
            + " > max_seq " + String(max_seq) + " — decode akan meluap. "
            + "Maksimum --max-tokens yang aman: " + String(boleh) + "."
        )

    # ---------------- 6. Generasi greedy ----------------
    # Warmup clock GPU (metodologi benchmark MLX): beberapa iterasi GEMM
    # terbesar (lm_head) menaikkan clock ke keadaan sustain SEBELUM timer —
    # menghapus bias cold-clock pada angka prefill. BONSAI_WARMUP=0 mematikan.
    var warmup_env = getenv("BONSAI_WARMUP")
    if use_gpu_matmul() and not (warmup_env and warmup_env == "0"):
        for _ in range(12):
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)
        gpu_ctx_ptr[].synchronize()

    var t_all = monotonic()
    var generated = alloc[Int](max_tokens + 1)
    var n_generated = 0
    var pos = 0
    var next_tok = 0

    if use_gpu_matmul():
        var next_tok_host = alloc[Int32](1)

        # ---------------- Sampling opsional (default: greedy) ----------------
        # Semua default membuat jalur greedy tetap utuh, jadi kontrak bit-exact
        # (gate KHQ di deploy_on_kaggle.sh) tidak berubah sama sekali.
        #   BONSAI_TEMP_X100=70     -> temperature 0.70 (0 = mati/greedy)
        #   BONSAI_TOP_K=20         -> top-k    (resep resmi Bonsai: 20)
        #   BONSAI_TOP_P_X1000=950  -> top-p    (resep resmi: 850-950)
        #   BONSAI_MIN_P_X1000=0    -> min-p    (resep resmi: 0)
        #   BONSAI_REP_PENALTY_X100 -> repetition penalty (100 = mati). TIDAK ada
        #                              di spesifikasi resmi -> default MATI.
        #   BONSAI_REP_WINDOW=256   -> jendela token yang dikenai penalti
        #   BONSAI_SEED=1234        -> seed RNG
        var samp_temp_x100 = env_int("BONSAI_TEMP_X100", 0)
        var samp_top_k = env_int("BONSAI_TOP_K", 20)
        var samp_top_p_x1000 = env_int("BONSAI_TOP_P_X1000", 950)
        var samp_min_p_x1000 = env_int("BONSAI_MIN_P_X1000", 0)
        var samp_rep_x100 = env_int("BONSAI_REP_PENALTY_X100", 100)
        var samp_rep_win = env_int("BONSAI_REP_WINDOW", 256)
        var samp_seed = env_int("BONSAI_SEED", 1234)
        var sampling_on = samp_temp_x100 > 0
        var h_logits = alloc[Scalar[T]](V)
        var samp_cand = alloc[Float32](V)
        var samp_flag = alloc[UInt8](V)
        for i in range(V):
            samp_flag[i] = 0
        # Buffer top-k tetap kecil; K dibatasi 256 di dalam sample_from_logits.
        var samp_topk_val = alloc[Float32](256)
        var samp_topk_p = alloc[Float32](256)
        var samp_topk_idx = alloc[Int32](256)
        var h_rng = alloc[UInt64](1)
        h_rng[0] = UInt64(samp_seed) * 2654435761 + 12345
        if sampling_on:
            print(">> [SAMPLE] temperature =", Float32(samp_temp_x100) / 100.0,
                  "| top_k =", samp_top_k,
                  "| top_p =", Float32(samp_top_p_x1000) / 1000.0,
                  "| min_p =", Float32(samp_min_p_x1000) / 1000.0,
                  "| rep_penalty =", Float32(samp_rep_x100) / 100.0,
                  "| seed =", samp_seed)
        else:
            print(">> [SAMPLE] mati — decoding greedy (argmax)")

        # Upload embedding ke VRAM SEKALI (packed + skala + bias):
        # lookup per token jalan di GPU (embed_lookup_1bit_sm75) — menghapus
        # dequant CPU + cast skalar + H2D PCIe pada SETIAP token.
        # Bonsai-2 (bits=2): embed U32 [V, K/16] + inverse FWHT — lookup
        # dikerjakan di HOST (1 baris K elemen per token; biaya ~puluhan µs)
        # sehingga upload 318 MB ke VRAM TIDAK diperlukan; hanya buffer
        # scratch F16 [K] + 2 buffer host [K] yang dibuat sekali.
        var embed_w_dev = UnsafePointer[UInt8, MutAnyOrigin]()
        var embed_s_dev = UnsafePointer[Float32, MutAnyOrigin]()
        var embed_b_dev = UnsafePointer[Float32, MutAnyOrigin]()
        # Slot DeviceBuffer di scope luar agar buffer bertahan seumur hidup
        # loop decode (RAII di dalam if-branch akan membebaskannya terlalu
        # dini sedangkan raw pointer masih dipakai).
        var embed_w_dev_buf = alloc[DeviceBuffer[DType.uint8]](1)
        var embed_s_dev_buf = alloc[DeviceBuffer[DType.float32]](1)
        var embed_b_dev_buf = alloc[DeviceBuffer[DType.float32]](1)
        var embed_b2_scratch = alloc[DeviceBuffer[T]](1)
        var embed_row32 = UnsafePointer[Float32, MutAnyOrigin]()
        var embed_row16 = UnsafePointer[Scalar[T], MutAnyOrigin]()
        if bits == 1:
            embed_w_dev_buf.init_pointee_move(
                gpu_ctx_ptr[].enqueue_create_buffer[DType.uint8](index.entries[e_emb].nbytes)
            )
            gpu_ctx_ptr[].enqueue_copy(embed_w_dev_buf[], embed_w)
            embed_s_dev_buf.init_pointee_move(
                gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](V_emb * (D_real // 128))
            )
            gpu_ctx_ptr[].enqueue_copy(embed_s_dev_buf[], embed_s)
            embed_b_dev_buf.init_pointee_move(
                gpu_ctx_ptr[].enqueue_create_buffer[DType.float32](V_emb * (D_real // 128))
            )
            gpu_ctx_ptr[].enqueue_copy(embed_b_dev_buf[], embed_b)
            embed_w_dev = embed_w_dev_buf[].unsafe_ptr()
            embed_s_dev = embed_s_dev_buf[].unsafe_ptr()
            embed_b_dev = embed_b_dev_buf[].unsafe_ptr()
        else:
            embed_b2_scratch.init_pointee_move(
                gpu_ctx_ptr[].enqueue_create_buffer[T](D_real)
            )
            embed_row32 = alloc[Float32](D_real)
            embed_row16 = alloc[Scalar[T]](D_real)

        # Timer prefill mulai SETELAH upload embedding (~238 MB via PCIe) —
        # itu biaya setup one-time, bukan beban latensi prompt.
        var t_prefill_start = monotonic()

        # prefill (Full GPU resident)
        # Jalur BATCHED (default): prompt diproses per chunk M-token via kernel
        # qmm WMMA v2 (tensor core) — bobot 2.8 GB dibaca SEKALI per chunk,
        # bukan M kali seperti GEMV per-token. Fallback per-token otomatis bila
        # .so tidak memiliki simbol qmm atau env BONSAI_PREFILL_PER_TOKEN=1.
        var force_per_token = getenv("BONSAI_PREFILL_PER_TOKEN")
        var batched_prefill = layers[0].prefill_ffi_ready() and not (
            force_per_token and force_per_token[0] == "1"
        )
        if batched_prefill:
            alias PF_CHUNK = 256
            var chunk_len = min(PF_CHUNK, prompt_len)

            # Buffer prefill M-token (fp16, RAII hidup selama blok ini)
            var h_pf_hidden = alloc[DeviceBuffer[T]](1)
            h_pf_hidden.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_hidden = h_pf_hidden[].unsafe_ptr()
            var h_pf_xn = alloc[DeviceBuffer[T]](1)
            h_pf_xn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_xn = h_pf_xn[].unsafe_ptr()
            var h_pf_sub = alloc[DeviceBuffer[T]](1)
            h_pf_sub.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_sub = h_pf_sub[].unsafe_ptr()
            var h_pf_mlp = alloc[DeviceBuffer[T]](1)
            h_pf_mlp.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * D))
            var pf_mlp = h_pf_mlp[].unsafe_ptr()
            var h_pf_proj = alloc[DeviceBuffer[T]](1)
            h_pf_proj.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * 17408))
            var pf_proj = h_pf_proj[].unsafe_ptr()
            var h_pf_conv = alloc[DeviceBuffer[T]](1)
            h_pf_conv.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * cfg.gdn_conv_dim))
            var pf_conv = h_pf_conv[].unsafe_ptr()
            var pf_qk_dim = cfg.gdn_num_k_heads * cfg.gdn_head_k_dim
            var h_pf_qn = alloc[DeviceBuffer[T]](1)
            h_pf_qn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_qk_dim))
            var pf_qn = h_pf_qn[].unsafe_ptr()
            var h_pf_kn = alloc[DeviceBuffer[T]](1)
            h_pf_kn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_qk_dim))
            var pf_kn = h_pf_kn[].unsafe_ptr()
            var pf_v_dim = cfg.gdn_num_v_heads * cfg.gdn_head_v_dim
            var h_pf_gdn = alloc[DeviceBuffer[T]](1)
            h_pf_gdn.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * pf_v_dim))
            var pf_gdn = h_pf_gdn[].unsafe_ptr()
            var h_pf_gu = alloc[DeviceBuffer[T]](1)
            h_pf_gu.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * 2 * cfg.intermediate_size))
            var pf_gu = h_pf_gu[].unsafe_ptr()
            var h_pf_sw = alloc[DeviceBuffer[T]](1)
            h_pf_sw.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * cfg.intermediate_size))
            var pf_sw = h_pf_sw[].unsafe_ptr()
            # Scratch FWHT 2-bit: SATU buffer bersama untuk seluruh modul.
            # Ukurannya mengikuti K terbesar (down_proj = cfg.intermediate_size).
            # Aman dipakai bergantian karena setiap modul meng-enqueue
            # FWHT -> GEMM secara berurutan pada stream yang sama.
            var h_pf_fwht = alloc[DeviceBuffer[T]](1)
            h_pf_fwht.init_pointee_move(gpu_ctx_ptr[].enqueue_create_buffer[T](chunk_len * cfg.intermediate_size))
            var pf_fwht = h_pf_fwht[].unsafe_ptr()

            var pos_pf = 0
            while pos_pf < prompt_len:
                var m = min(PF_CHUNK, prompt_len - pos_pf)
                # Embed lookup per baris (M row di pf_hidden)
                for t in range(m):
                    if bits == 1:
                        embed_lookup_1bit_sm75_launch_on[T](
                            gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                            pf_hidden.offset(t * D), ptoks[pos_pf + t + 1], D
                        )
                    else:
                        embed_lookup_2bit_to_dev(
                            gpu_ctx_ptr, embed_w, embed_s, embed_b, embed_signs,
                            ptoks[pos_pf + t + 1], D, hd.block,
                            pf_hidden.offset(t * D), embed_b2_scratch[],
                            embed_row32, embed_row16
                        )
                for li in range(n_layers):
                    if layers[li].is_linear:
                        layers[li].forward_prefill_gpu(
                            pf_hidden, pf_xn, pf_sub, pf_mlp,
                            pf_proj, pf_conv, pf_qn, pf_kn, pf_gdn,
                            pf_gu, pf_sw, act_attn_scores_dev,
                            gdn_states[gdn_idx[li]], kv_caches[0], pos_pf, m,
                            pf_fwht
                        )
                    else:
                        layers[li].forward_prefill_gpu(
                            pf_hidden, pf_xn, pf_sub, pf_mlp,
                            pf_proj, pf_conv, pf_qn, pf_kn, pf_gdn,
                            pf_gu, pf_sw, act_attn_scores_dev,
                            gdn_states[0], kv_caches[kv_idx[li]], pos_pf, m,
                            pf_fwht
                        )
                pos_pf += m
            pos = pos_pf

            # Transplantasi baris terakhir prefill -> buffer decode (D2D)
            copy_vec_sm75_launch_on[T](
                gpu_ctx_ptr[], act_hidden_dev,
                pf_hidden.offset((prompt_len - 1) % chunk_len * D), D
            )
            gpu_ctx_ptr[].synchronize()

            # Norm final + lm_head + argmax hanya pada baris terakhir
            rmsnorm_sm75_launch_on[T](
                gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                fnorm_dev, True, D, cfg.rms_norm_eps
            )
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)

            # Diagnostik near-tie di batas prefill (BONSAI_DUMP_TOP2=1)
            var dumpv = getenv("BONSAI_DUMP_TOP2")
            if dumpv and dumpv[0] == "1":
                dump_top2_prefill(gpu_ctx_ptr, act_logits_dev, V)

            argmax_sm75_launch_on[T](
                gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                act_token_out_dev, V
            )
            gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
            gpu_ctx_ptr[].synchronize()
            next_tok = Int(next_tok_host[0])
            # Buffer pf_* dibebaskan otomatis oleh RAII di akhir blok.
        else:
            # Fallback per-token (GEMV decode per baris; hasil identik)
            for t in range(prompt_len):
                var cur_tok = ptoks[t + 1]
                if bits == 1:
                    embed_lookup_1bit_sm75_launch_on[T](
                        gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                        h_hidden_holder[].unsafe_ptr(), cur_tok, D
                    )
                else:
                    embed_lookup_2bit_to_dev(
                        gpu_ctx_ptr, embed_w, embed_s, embed_b, embed_signs,
                        cur_tok, D, hd.block,
                        h_hidden_holder[].unsafe_ptr(), embed_b2_scratch[],
                        embed_row32, embed_row16
                    )

                for li in range(n_layers):
                    if layers[li].is_linear:
                        var gi = gdn_idx[li]
                        layers[li].forward_gpu(
                            act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                            act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                            act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                            act_attn_scores_dev,
                            gdn_states[gi], kv_caches[0], pos
                        )
                    else:
                        var ki = kv_idx[li]
                        layers[li].forward_gpu(
                            act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                            act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                            act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                            act_attn_scores_dev,
                            gdn_states[0], kv_caches[ki], pos
                        )

                # LM head + argmax hanya untuk token TERAKHIR prompt (yang
                # menghasilkan prediksi token generasi pertama). Untuk token
                # intermediate next_tok dibuang — cur_tok diambil dari prompt —
                # jadi lewati proyeksi V=248320 + reduksi argmax + sync host.
                # State GDN/KV tetap ter-update asinkron via stream yang sama.
                if t == prompt_len - 1:
                    rmsnorm_sm75_launch_on[T](
                        gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                        fnorm_dev, True, D, cfg.rms_norm_eps
                    )
                    lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)

                    # Diagnostik near-tie di batas prefill (BONSAI_DUMP_TOP2=1)
                    var dumpv2 = getenv("BONSAI_DUMP_TOP2")
                    if dumpv2 and dumpv2[0] == "1":
                        dump_top2_prefill(gpu_ctx_ptr, act_logits_dev, V)

                    argmax_sm75_launch_on[T](
                        gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                        act_token_out_dev, V
                    )
                    gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
                    gpu_ctx_ptr[].synchronize()
                    next_tok = Int(next_tok_host[0])
                pos += 1
        generated[n_generated] = next_tok
        n_generated += 1
        print(">> [GEN] token id:", next_tok)
        var t_decode = monotonic()

        # decode (Full GPU resident)
        # Profil per-subsystem (BONSAI_PROFILE=1): sinkronisasi per layer agar
        # waktu GPU ter-atribusi akurat ke GDN / Attention / LM head.
        var prof_v = getenv("BONSAI_PROFILE")
        var prof = prof_v and prof_v[0] == "1"
        var acc_gdn: Int = 0
        var acc_attn: Int = 0
        var acc_lm: Int = 0
        # Pecahan acc_lm (hanya bermakna saat prof aktif — butuh sync pemisah).
        var acc_lm_gemm: Int = 0
        var acc_lm_am: Int = 0
        var acc_lm_d2h: Int = 0
        # Token henti (EOS): default <|im_end|> = 248046 (ChatML Qwen3).
        # Override dgn BONSAI_STOP_IDS="id1,id2,..."; set "none" untuk mematikan
        # (berguna saat ingin mengukur perilaku SETELAH giliran berakhir).
        var stop_env = getenv("BONSAI_STOP_IDS")
        var stop_spec = stop_env if len(stop_env) > 0 else String("248046")
        var stop_off = stop_spec == "none" or stop_spec == "off"
        var stop_ids = parse_int_list(String("") if stop_off else stop_spec)
        if stop_ids[0] > 0:
            print(">> [STOP] henti-di-EOS aktif:", stop_ids[0], "token id")

        for step in range(max_tokens - 1):
            var t0 = monotonic()
            if bits == 1:
                embed_lookup_1bit_sm75_launch_on[T](
                    gpu_ctx_ptr[], embed_w_dev, embed_s_dev, embed_b_dev,
                    h_hidden_holder[].unsafe_ptr(), next_tok, D
                )
            else:
                embed_lookup_2bit_to_dev(
                    gpu_ctx_ptr, embed_w, embed_s, embed_b, embed_signs,
                    next_tok, D, hd.block,
                    h_hidden_holder[].unsafe_ptr(), embed_b2_scratch[],
                    embed_row32, embed_row16
                )

            for li in range(n_layers):
                var tl = monotonic()
                # Fusi residual2+pre-norm: weight norm layer berikutnya
                # (layer terakhir: final norm sebelum lm_head).
                var nnw = (
                    layers[li + 1].input_layernorm_w_dev
                    if li + 1 < n_layers else fnorm_dev
                )
                if layers[li].is_linear:
                    var gi = gdn_idx[li]
                    layers[li].forward_gpu(
                        act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                        act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                        act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                        act_attn_scores_dev,
                        gdn_states[gi], kv_caches[0], pos,
                        nnw, li > 0
                    )
                    if prof:
                        gpu_ctx_ptr[].synchronize()
                    acc_gdn += monotonic() - tl
                else:
                    var ki = kv_idx[li]
                    layers[li].forward_gpu(
                        act_hidden_dev, act_x_norm_dev, act_sublayer_out_dev, act_mlp_out_dev,
                        act_proj_raw_dev, act_conv_out_dev, act_q_normed_dev, act_k_normed_dev,
                        act_gdn_out_dev, act_gate_up_dev, act_swiglu_act_dev,
                        act_attn_scores_dev,
                        gdn_states[0], kv_caches[ki], pos,
                        nnw, li > 0
                    )
                    if prof:
                        gpu_ctx_ptr[].synchronize()
                    acc_attn += monotonic() - tl

            if prof:
                gpu_ctx_ptr[].synchronize()
            var tlm = monotonic()

            # Final norm dilewati bila fusi aktif: residual-2 layer terakhir
            # sudah menulis act_x_norm_dev dgn fnorm_dev (bit-exact identik).
            var no_fuse = getenv("BONSAI_NO_FUSE")
            if not (no_fuse and no_fuse == "1"):
                pass
            else:
                rmsnorm_sm75_launch_on[T](
                    gpu_ctx_ptr[], act_hidden_dev, act_x_norm_dev,
                    fnorm_dev, True, D, cfg.rms_norm_eps
                )
            lm_proj.forward_device(act_x_norm_dev, act_logits_dev, 1)
            if prof:
                gpu_ctx_ptr[].synchronize()
            var t_lm_gemm = monotonic()
            acc_lm_gemm += t_lm_gemm - tlm
            if sampling_on:
                # Logits fp16 (V=248320 ≈ 0,5 MB) ditarik ke host lalu disampling
                # di CPU. Biaya D2H + sampling kecil dibanding ~54 ms/token.
                gpu_ctx_ptr[].enqueue_copy(h_logits, logits_buf[])
                gpu_ctx_ptr[].synchronize()
                var t_samp = monotonic()
                next_tok = sample_from_logits(
                    h_logits, V, samp_cand,
                    samp_topk_val, samp_topk_p, samp_topk_idx, samp_flag,
                    samp_temp_x100, samp_top_k, samp_top_p_x1000, samp_min_p_x1000,
                    samp_rep_x100, samp_rep_win,
                    h_rng, generated, n_generated
                )
                var t_samp_end = monotonic()
                acc_lm_am += t_samp - t_lm_gemm
                acc_lm_d2h += t_samp_end - t_samp
                acc_lm += t_samp_end - tlm
            else:
                argmax_sm75_launch_on[T](
                    gpu_ctx_ptr[], act_logits_dev, act_stage1_vals_dev, act_stage1_idxs_dev,
                    act_token_out_dev, V
                )
                if prof:
                    gpu_ctx_ptr[].synchronize()
                var t_lm_am = monotonic()
                gpu_ctx_ptr[].enqueue_copy(next_tok_host, h_token_out_holder[])
                gpu_ctx_ptr[].synchronize()
                var t_lm_end = monotonic()
                acc_lm_am += t_lm_am - t_lm_gemm
                acc_lm_d2h += t_lm_end - t_lm_am
                acc_lm += t_lm_end - tlm
                next_tok = Int(next_tok_host[0])
            pos += 1
            generated[n_generated] = next_tok
            n_generated += 1
            var dt = monotonic() - t0
            var ms = Float64(dt) / 1e6
            var tps = 1000.0 / ms if ms > 0.0 else 0.0
            print(">> [GEN] token id:", next_tok, "|", ms, "ms |", tps, "tok/s")
            if is_stop_token(next_tok, stop_ids):
                print(">> [STOP] token henti", next_tok, "pada langkah", step + 1,
                      "— generasi dihentikan")
                break

        next_tok_host.free()

        var total_ms = Float64(monotonic() - t_all) / 1e6
        var dec_ms = Float64(monotonic() - t_decode) / 1e6
        var prefill_ms = Float64(t_decode - t_prefill_start) / 1e6
        # Jumlah langkah decode yang BENAR-BENAR dijalankan (bisa < max_tokens-1
        # bila berhenti di EOS). n_generated sudah termasuk token hasil prefill.
        var n_dec = n_generated - 1
        if n_dec > 0:
            print(">> [PERF] prefill", prompt_len, "token |", prefill_ms, "ms |",
                  prefill_ms / Float64(prompt_len), "ms/token |",
                  Float64(prompt_len) * 1000.0 / prefill_ms if prefill_ms > 0.0 else 0.0,
                  "tok/s | decode", n_dec, "token")
            print(">> [PERF] rata-rata decode:", dec_ms / n_dec, "ms/token |",
                  Float64(n_dec) * 1000.0 / dec_ms if dec_ms > 0.0 else 0.0, "tok/s")
            # Tanpa BONSAI_PROFILE tidak ada sync per-layer, jadi acc_gdn/acc_attn
            # hanyalah waktu SUBMIT CPU dan acc_lm menyerap seluruh kerja GPU token
            # yang terkuras di sync terakhir. Menampilkannya sebagai "ms per
            # subsistem" pernah menyesatkan; sekarang dinyatakan terang-terangan.
            if prof:
                print(">> [PROF/SPLIT] per token -> GDN:",
                      Float64(acc_gdn) / 1e6 / n_dec, "ms | ATTN:",
                      Float64(acc_attn) / 1e6 / n_dec, "ms | LM_HEAD+argmax:",
                      Float64(acc_lm) / 1e6 / n_dec, "ms")
                print(">> [PROF/LM] gemm:", Float64(acc_lm_gemm) / 1e6 / n_dec,
                      "ms | argmax:", Float64(acc_lm_am) / 1e6 / n_dec,
                      "ms | D2H+sync:", Float64(acc_lm_d2h) / 1e6 / n_dec, "ms")
            else:
                print(">> [PROF/SPLIT] BONSAI_PROFILE tidak aktif:",
                      Float64(acc_gdn + acc_attn) / 1e6 / n_dec,
                      "ms/token submit CPU (bukan GPU) |",
                      Float64(acc_lm) / 1e6 / n_dec,
                      "ms/token = kerja GPU SELURUH token, bukan LM head")
        print(">> [PERF] total:", total_ms, "ms")
        khq_prof_report()
        # Sebelumnya baris ini selalu mencetak "(greedy)" walau sampling aktif —
        # menyesatkan saat membaca log gerbang koherensi.
        if sampling_on:
            print(">> Selesai:", n_generated, "token di-generate (sampling).")
        else:
            print(">> Selesai:", n_generated, "token di-generate (greedy).")
        if khq_dir:
            khq_dump_flush()
    else:
        # FALLBACK DIHAPUS: jalur host-sim CPU tidak lagi dipakai produksi.
        # GPU wajib — tanpa BONSAI_USE_GPU program error, bukan diam-diam CPU.
        raise Error(
            "CPU fallback dilarang: jalur host-sim telah dihapus. "
            "Set BONSAI_USE_GPU=1 agar inferensi berjalan di NVIDIA T4."
        )

    generated.free()
