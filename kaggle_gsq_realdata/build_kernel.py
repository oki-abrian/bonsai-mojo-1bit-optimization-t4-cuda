#!/usr/bin/env python3
# ==============================================================================
# GENERATOR (jalankan sekali di mesin lokal) — membangun run_realdata.py
# ------------------------------------------------------------------------------
# Kernel skrip Kaggle HANYA menyalin `code_file` (run_realdata.py) ke
# /kaggle/src; berkas pendamping tidak ikut (terbukti di kernel guru sebelumnya:
# "gguf_py.tar.gz tidak ditemukan"). Karena itu semua modul yang dibutuhkan
# disematkan sebagai base64 di dalam run_realdata.py:
#
#   gguf_py.tar.gz   -> modul gguf (dekuantisasi guru)
#   teacher_gguf.py  -> jembatan GGUF -> NumPy per-layer
#   gsq_driver.py    -> driver refine (yang SEDANG diuji)
#   gsq_ternary.py   -> inti GSQ
#   ternary_sim.py   -> kontrak pack Bonsai-2
#
# Tidak ada file terbukti yang disentuh; ini folder baru (aditif).
# ==============================================================================

import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CONV = os.path.join(ROOT, "references", "converter-mxfp4-backfree-sparse-test-full-graph")

SOURCES = [
    ("_GGUF_PY_TAR_GZ_B64", os.path.join(HERE, "gguf_py.tar.gz")),
    ("_TEACHER_GGUF_PY_B64", os.path.join(CONV, "teacher_gguf.py")),
    ("_GSQ_DRIVER_PY_B64", os.path.join(CONV, "gsq_driver.py")),
    ("_GSQ_TERNARY_PY_B64", os.path.join(CONV, "gsq_ternary.py")),
    ("_TERNARY_SIM_PY_B64", os.path.join(CONV, "ternary_sim.py")),
]

# ==============================================================================
# BODY kernel — logika utama (ditempel setelah blob base64)
# ==============================================================================
BODY = r'''# ==============================================================================
# UJI DATA NYATA UNTUK gsq_driver.py  (CPU, tanpa internet)
# ------------------------------------------------------------------------------
# Menutup celah: driver GSQ belum pernah dijalankan terhadap data nyata (semua
# uji 27/27 + 19/19 adalah dunia sintetis). Kernel ini membuktikan tiga hal
# terhadap ARTIFAK ASLI (bukan tiruan):
#
#   [A] Cakupan kunci guru: untuk SEMUA 64 layer, setiap sufiks yang dipetakan
#       gsq_driver.py benar-benar ada di GGUF (cek level metadata, murah).
#   [B] Resolusi kunci + fusi concat: gsq_driver.py TeacherSource.get()
#       menemukan & menggabungkan tensor guru dari .npz hasil ekstraksi nyata.
#   [C] Kontrak pack & refine end-to-end: read_quant_tensor pada byte safetensors
#       ASLI (validasi biases==-scales, bentuk, q di {0,1,2}), refine sungguhan,
#       tulis shard keluaran, lalu verifikasi ulang kontraknya.
#
# PENTING — batasan yang DISENGAJA dan harus dilaporkan apa adanya:
#   * Kalibrasi (aktivasi x) adalah SINTETIS (seed tetap), BUKAN dump aktivitas
#     BONSAI_BITS=2. Sebab itu angka qad loss di sini TIDAK bermakna sebagai
#     ukuran akurasi; yang diuji adalah PIPA data nyata (apakah driver bisa
#     membaca artefak asli tanpa kunci hilang / kontrak dilanggar / crash),
#     bukan kualitas kuantisasi. Dump aktivasi nyata adalah langkah berikutnya
#     dalam rantai yang sudah direncanakan.
#   * Guru diekstrak hanya 1 layer (batas RAM/disk).
#   * Pack 8,6 GB TIDAK pernah dimuat utuh; hanya header + byte range tensor
#     yang dipakai.
#
# Dataset yang dipasang (sudah ada, lihat kernel-metadata.json):
#   okiabrian/qwen38-27b-gsq-rco-iq3s    -> guru GGUF IQ3_S (11,8 GB)
#   okiabrian/bonsai-2bit-weights        -> pack Bonsai-2 (model.safetensors
#                                           8,6 GB + hadamard.json + config)
# ==============================================================================

import base64
import gc
import json
import os
import subprocess
import sys
import tarfile
import time

import numpy as np

TMP = "/kaggle/tmp/gsq_realdata"
os.makedirs(TMP, exist_ok=True)
OUT = "/kaggle/working"
os.makedirs(OUT, exist_ok=True)

T0 = time.time()


def log(*a):
    print(">>", *a, flush=True)


def stage(n, total, msg):
    log("[%d/%d] %s" % (n, total, msg))


def rss_hwm_mb():
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


def mem_total_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


VERDICT = []  # (label, ok, detail)


def record(label, ok, detail=""):
    VERDICT.append((label, bool(ok), detail))
    log("[VOK]" if ok else "[VGAGAL]", label, ("| " + detail) if detail else "")


# ---------------------------------------------------------------------------
# [1/8] Buka bundel yang disematkan
# ---------------------------------------------------------------------------
TOTAL = 8
stage(1, TOTAL, "Membuka bundel gguf-py + modul konverter ...")

for _name, _b64 in (
    ("gguf_py.tar.gz", _GGUF_PY_TAR_GZ_B64),
    ("teacher_gguf.py", _TEACHER_GGUF_PY_B64),
    ("gsq_driver.py", _GSQ_DRIVER_PY_B64),
    ("gsq_ternary.py", _GSQ_TERNARY_PY_B64),
    ("ternary_sim.py", _TERNARY_SIM_PY_B64),
):
    with open(os.path.join(TMP, _name), "wb") as fh:
        fh.write(base64.b64decode(_b64))
    log("   tulis %s (%d byte)" % (_name, os.path.getsize(os.path.join(TMP, _name))))

with tarfile.open(os.path.join(TMP, "gguf_py.tar.gz")) as tf:
    tf.extractall(os.path.join(TMP, "unpacked"))

_gguf_root = None
for _root, _dirs, _files in os.walk(os.path.join(TMP, "unpacked")):
    if "gguf" in _dirs and os.path.isfile(os.path.join(_root, "gguf", "__init__.py")):
        _gguf_root = _root
        break
if not _gguf_root:
    log("   [FATAL] paket gguf tidak ditemukan setelah ekstraksi")
    sys.exit(1)
os.environ["GGUF_PY_PATH"] = _gguf_root
sys.path.insert(0, TMP)
log("   [OK] GGUF_PY_PATH =", _gguf_root)

# ---------------------------------------------------------------------------
# [2/8] Temukan artefak masukan di mount dataset
# ---------------------------------------------------------------------------
stage(2, TOTAL, "Mencari gguf guru + pack Bonsai-2 ...")
GGUF = None
PACK_DIR = None
HADAMARD = None
for _root, _dirs, _files in os.walk("/kaggle/input"):
    for _f in _files:
        if _f.endswith(".gguf") and GGUF is None:
            GGUF = os.path.join(_root, _f)
        if _f == "model.safetensors" and PACK_DIR is None:
            PACK_DIR = _root
        if _f == "hadamard.json" and HADAMARD is None:
            HADAMARD = os.path.join(_root, _f)
if not GGUF:
    log("   [FATAL] tidak ada .gguf di /kaggle/input")
    sys.exit(1)
if not PACK_DIR or not HADAMARD:
    log("   [FATAL] pack (model.safetensors/hadamard.json) tidak ditemukan")
    log("   isi /kaggle/input:", sorted(os.listdir("/kaggle/input")))
    sys.exit(1)
PACK_FILE = os.path.join(PACK_DIR, "model.safetensors")
log("   guru    :", GGUF, os.path.getsize(GGUF), "byte")
log("   pack    :", PACK_FILE, os.path.getsize(PACK_FILE), "byte")
log("   hadamard:", HADAMARD, os.path.getsize(HADAMARD), "byte")

# ---------------------------------------------------------------------------
# [3/8] hadamard.json asli (modul yang sama dipakai driver)
# ---------------------------------------------------------------------------
stage(3, TOTAL, "Membaca hadamard.json asli ...")
from ternary_sim import load_hadamard_signs, signs_for_k  # noqa: E402

hd = load_hadamard_signs(HADAMARD)
log("   block=%d widths=%s gdn_v_grouped=%s inverse=%s"
    % (hd["block"], list(hd["widths"]), hd["gdn_v_grouped"],
       list(hd["inverse_weight_names"])))
for _w in (5120, 6144, 17408):
    _sg = signs_for_k(hd, _w)
    record("signs hadamard.json punya lebar K=%d" % _w, _sg is not None,
           "bentuk %s" % ((_sg.shape,) if _sg is not None else "tidak ada"))
    if _sg is not None:
        _ok = bool(np.isin(np.asarray(_sg), (-1, 1)).all())
        record("signs K=%d semuanya +-1" % _w, _ok)

# ---------------------------------------------------------------------------
# [4/8] Header safetensors saja (JANGAN muat 8,6 GB) + cek nama & dtype
# ---------------------------------------------------------------------------
stage(4, TOTAL, "Membaca header safetensors (hanya metadata) ...")


def read_st_header(path):
    with open(path, "rb") as fh:
        _n = int(np.fromfile(fh, dtype="<u8", count=1)[0])
        raw = fh.read(_n)
    return json.loads(raw)


hdr = read_st_header(PACK_FILE)
names = [k for k in hdr if k != "__metadata__"]
log("   jumlah tensor pack: %d" % len(names))
_meta = hdr.get("__metadata__")
if _meta:
    log("   metadata pack:", {k: str(v)[:80] for k, v in list(_meta.items())[:4]})

# inventaris dtype per kategori akhiran
_dtype_by_cat = {}
for _n in names:
    _cat = _n.rsplit(".", 1)[-1] if "." in _n else "?"
    _dt = str(hdr[_n].get("dtype"))
    _dtype_by_cat.setdefault(_cat, {}).setdefault(_dt, 0)
    _dtype_by_cat[_cat][_dt] += 1
for _cat in sorted(_dtype_by_cat):
    log("   kategori .%s : %s" % (_cat, _dtype_by_cat[_cat]))

# Kritis: driver memakai read_f16_scales (menginterpretasi byte sebagai F16).
# Mojo read_f32 menerima F16/F32/BF16, jadi pack boleh menyimpan salah satunya.
# UKUR, jangan asumsikan.
_scales_dt = set()
for _n in names:
    if _n.endswith(".scales"):
        _scales_dt.add(str(hdr[_n].get("dtype")))
log("   [Scales-DTYPE] dtype .scales di pack: %s" % sorted(_scales_dt))
if _scales_dt == {"F16"}:
    record("dtype .scales adalah F16 (asumsi read_f16_scales)", True, "F16")
else:
    # read_f16_scales akan salah menafsirkan byte F32/BF16 sebagai F16.
    # BUKAN kesalahan pack — ini perlu perhatian driver; laporkan apa adanya.
    record("dtype .scales adalah F16 (asumsi read_f16_scales)", False,
           "ditemukan %s" % sorted(_scales_dt))

# ---------------------------------------------------------------------------
# Deteksi PREFIX nama layer di pack (UKUR, jangan asumsikan).
# main.mojo find_flex() mencoba nama apa adanya lalu fallback prefix
# "language_model.". Pack MLX prism-ml menyimpan "language_model.model.layers.N."
# sedangkan gsq_driver memakai "model.layers.N.".
# Deteksi dari nama tensor APA PUN yang punya pola "*.layers.0.".
# ---------------------------------------------------------------------------
import re  # noqa: E402

_pack_pref = "model.layers."  # default = konvensi driver
for _n in names:
    _m = re.search(r"^(.*\.layers\.)0\.", _n)
    if _m:
        _pack_pref = _m.group(1)
        break
log("   [PACK-PREFIX] prefix layer di pack: %r" % _pack_pref)
if _pack_pref != "model.layers.":
    log("   [TEMUAN] pack memakai prefix %r, bukan 'model.layers.' —" % _pack_pref)
    log("           gsq_driver.py BELUM menangani fallback ini (main.mojo")
    log("           memakai find_flex). Mini-pack dinamai ulang ke konvensi")
    log("           driver agar logikanya tetap diuji di atas byte asli.")

# Tampilkan SEMUA nama tensor pack untuk layer 0 + 3 (hanya nama, murah)
for _li in (0, 3):
    _lnames = sorted(_n for _n in names if (".layers.%d." % _li) in _n)
    log("   [PACK-NAMES-L%02d] %d tensor:" % (_li, len(_lnames)))
    for _n in _lnames:
        log("     %s  %s %s" % (_n, hdr[_n].get("dtype"), hdr[_n].get("shape")))

# Apakah pack memakai gate_up FUSED atau gate/up SPLIT?
_gu_fused = ("%s0.mlp.gate_up_proj.weight" % _pack_pref) in hdr
_ga_split = ("%s0.mlp.gate_proj.weight" % _pack_pref) in hdr
# in_proj_all: fused, atau split in_proj_qkv + in_proj_z (main.mojo:1168-1176)
_ipa_fused = ("%s0.linear_attn.in_proj_all.weight" % _pack_pref) in hdr
_ipa_split = ("%s0.linear_attn.in_proj_qkv.weight" % _pack_pref) in hdr
log("   [MLP-FORM] gate_up fused=%s | gate/up split=%s" % (_gu_fused, _ga_split))
log("   [GDN-FORM] in_proj_all fused=%s | qkv+z split=%s"
    % (_ipa_fused, _ipa_split))
record("bentuk tensor MLP pack dikenali", _gu_fused or _ga_split,
       "fused=%s split=%s" % (_gu_fused, _ga_split))
record("bentuk tensor in_proj_all pack dikenali", _ipa_fused or _ipa_split,
       "fused=%s split=%s" % (_ipa_fused, _ipa_split))
if _ga_split and not _gu_fused:
    log("   [TEMUAN] pack menyimpan mlp.gate_proj + mlp.up_proj TERPISAH;")
    log("           main.mojo:1342-1358 men-fuse-nya saat muat. Mini-pack")
    log("           di bawah juga men-fuse agar kontrak driver (gate_up)")
    log("           teruji di atas byte asli.")
if _ipa_split and not _ipa_fused:
    log("   [TEMUAN] pack menyimpan linear_attn.in_proj_qkv + in_proj_z")
    log("           TERPISAH; main.mojo:1168-1176 men-fuse-nya saat muat.")
    log("           Mini-pack di bawah juga men-fuse.")

# cek nama tensor pack untuk 2 jenis layer (GDN=0, attn penuh=3)
_expect = {
    0: ["mlp.gate_up_proj", "mlp.down_proj",
        "linear_attn.in_proj_all", "linear_attn.out_proj"],
    3: ["mlp.gate_up_proj", "mlp.down_proj",
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj"],
}


def _tri_ok(pref, li, suf):
    """Cek triptych ada; untuk sufiks bentuk-fused, terima juga bentuk SPLIT
    (gate/up, qkv/z) karena loader Mojo men-fuse keduanya."""
    _tri = [("%s%d.%s.%s" % (pref, li, suf, _e)) for _e in
            ("weight", "scales", "biases")]
    _miss = [t for t in _tri if t not in hdr]
    if _miss:
        _splits = {
            "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
            "linear_attn.in_proj_all": ["linear_attn.in_proj_qkv",
                                        "linear_attn.in_proj_z"],
        }
        if suf in _splits:
            _alt = [("%s%d.%s.%s" % (pref, li, _s, _e)) for _s in
                    _splits[suf] for _e in ("weight", "scales", "biases")]
            _miss = [t for t in _alt if t not in hdr]
            return (not _miss, "%s (bentuk split, %d tensor)"
                    % ("hilang %s" % _miss if _miss else "OK", len(_alt)))
    return (not _miss, "hilang %s" % _miss if _miss else "%d tensor" % len(_tri))


for _li, _sufs in _expect.items():
    for _suf in _sufs:
        _ok, _det = _tri_ok(_pack_pref, _li, _suf)
        record("pack L%02d %-24s triptych ada" % (_li, _suf), _ok, _det)

# ---------------------------------------------------------------------------
# [5/8] Ekstrak 1 layer guru (GDN layer 0) ke .npz — data nyata pertama
# ---------------------------------------------------------------------------
stage(5, TOTAL, "Mengekstrak guru layer 0 (dekuantisasi nyata) ...")
import teacher_gguf as tg  # noqa: E402

_t1 = time.time()
_man = tg.extract_layer(GGUF, 0, TMP, dtype="float16")
_t_ex = time.time() - _t1
TEACHER_NPZ = _man["path"]
log("   ekstrak %.1f dtk -> %s" % (_t_ex, TEACHER_NPZ))
log("   %d tensor guru:" % len(_man["tensors"]))
for _k, _sh in sorted(_man["tensors"].items()):
    log("     %-32s %s" % (_k, _sh))
record("ekstrak guru layer 0 berhasil", os.path.getsize(TEACHER_NPZ) > 0,
       "%d tensor, %.1f dtk" % (len(_man["tensors"]), _t_ex))

# ---------------------------------------------------------------------------
# [6/8] Kalibrasi SINTETIS (lihat batasan di docstring bagian atas)
# ---------------------------------------------------------------------------
stage(6, TOTAL, "Membangun kalibrasi SINTETIS (seed tetap) ...")
_rng = np.random.default_rng(12345)
_M = 8  # token
# lebar harus sama persis dgn K tensor tujuan, jika tidak driver men-skip
calib = {
    "mlp_in_L00": (_rng.standard_normal((_M, 5120)) * 0.10).astype(np.float32),
    "ssm_in_L00": (_rng.standard_normal((_M, 6144)) * 0.05).astype(np.float32),
}
CALIB_NPZ = os.path.join(TMP, "calib_sintetis.npz")
np.savez(CALIB_NPZ, **calib)
log("   %s (SINTETIS — loss TIDAK bermakna sbg ukuran akurasi)" % CALIB_NPZ)
for _k, _v in calib.items():
    log("     %-14s %s" % (_k, _v.shape))

# ---------------------------------------------------------------------------
# [7/8] [A] Cakupan kunci guru untuk SEMUA 64 layer (metadata only)
#       [B] dry-run driver asli terhadap .npz nyata (layer 0)
# ---------------------------------------------------------------------------
stage(7, TOTAL, "[A] Cakupan kunci guru, 64 layer (metadata) ...")
_rep = tg.inspect(GGUF)
_have = {t["name"] for t in _rep["tensors"]}
_miss_total = 0
_cov_total = 0
for _li in range(64):
    _linear = (_li % 4) != 3  # interval=4; attn penuh saat li%4==3
    _sufs = [
        ("mlp.gate_up_proj", ["ffn_gate", "ffn_up"]),
        ("mlp.down_proj", ["ffn_down"]),
    ]
    if _linear:
        _sufs += [
            ("linear_attn.in_proj_all", ["attn_qkv", "attn_gate"]),
            ("linear_attn.in_proj_qkv", ["attn_qkv"]),
            ("linear_attn.in_proj_z", ["attn_gate"]),
            ("linear_attn.out_proj", ["ssm_out"]),
        ]
    else:
        _sufs += [
            ("self_attn.q_proj", ["attn_q"]),
            ("self_attn.k_proj", ["attn_k"]),
            ("self_attn.v_proj", ["attn_v"]),
            ("self_attn.o_proj", ["attn_output"]),
        ]
    for _suf, _subs in _sufs:
        for _sub in _subs:
            _cov_total += 1
            if "blk.%d.%s.weight" % (_li, _sub) not in _have:
                _miss_total += 1
                log("   [HILANG] blk.%d.%s.weight" % (_li, _sub))
record("[A] cakupan kunci guru 64 layer (%d sufiks)" % _cov_total,
       _miss_total == 0, "%d hilang" % _miss_total)

log(">> [7/8] [B] gsq_driver.py refine --dry-run (layer 0, data nyata) ...")
_drv = os.path.join(TMP, "gsq_driver.py")
_r = subprocess.run(
    [sys.executable, _drv, "refine",
     "--pack", PACK_FILE, "--teacher", TEACHER_NPZ,
     "--calib", CALIB_NPZ, "--hadamard", HADAMARD,
     "--out", os.path.join(TMP, "dry_out"),
     "--layers", "0", "--dry-run"],
    capture_output=True, text=True)
log("   rc=%d" % _r.returncode)
for _line in (_r.stdout + _r.stderr).splitlines():
    log("   | %s" % _line)
_avail = 0
_missing = 0
for _line in _r.stdout.splitlines():
    if _line.startswith("guru tersedia"):
        _avail = int(_line.split(":")[-1])
    if _line.startswith("guru hilang"):
        _missing = int(_line.split(":")[-1])
record("[B] dry-run layer 0: guru tersedia %d / hilang %d" % (_avail, _missing),
       _r.returncode == 0 and _missing == 0,
       "rc=%d" % _r.returncode)

# ---------------------------------------------------------------------------
# [8/8] [C] Refine sungguhan pada byte pack ASLI (mini-pack layer 0)
# ---------------------------------------------------------------------------
stage(8, TOTAL, "[C] Membangun mini-pack dari byte pack asli ...")


def read_selected(path, header, want_names):
    """Baca HANYA byte range tensor yang dibutuhkan (tidak muat 8,6 GB)."""
    _ST = {"BOOL": "?", "U8": "<u1", "I8": "<i1", "U16": "<u2", "I16": "<i2",
           "U32": "<u4", "I32": "<i4", "U64": "<u8", "I64": "<i8",
           "F16": "<f2", "F32": "<f4", "F64": "<f8"}
    out = {}
    with open(path, "rb") as fh:
        _n = int(np.fromfile(fh, dtype="<u8", count=1)[0])
        base = 8 + _n
        for _nm in want_names:
            if _nm not in header:
                continue
            a, b = header[_nm]["data_offsets"]
            fh.seek(base + a)
            buf = fh.read(b - a)
            dt = np.dtype(_ST[str(header[_nm]["dtype"])])
            out[_nm] = np.frombuffer(buf, dtype=dt).reshape(header[_nm]["shape"])
    return out


# Bangun mini-pack dari byte pack asli. Untuk MLP: bila pack memakai
# gate/up SPLIT, muat keduanya dan fuse tepat seperti main.mojo:1352-1357
# (concat sepanjang N: bobot U32, scales, biases). Kode driver (kontrak
# gate_up fused) lalu diuji di atas byte asli.
def load_tri(src_hdr, src_path, pref, suf, n_layers=0):
    """Muat satu triptych [weight U32, scales F16, biases F16] dari pack."""
    _want = ["%s%s.%s" % (pref, suf, _e) for _e in ("weight", "scales", "biases")]
    got = read_selected(src_path, src_hdr, _want)
    got = {"model.layers.%d.%s.%s" % (n_layers, suf, _e):
           got["%s%s.%s" % (pref, suf, _e)]
           for _e in ("weight", "scales", "biases")
           if "%s%s.%s" % (pref, suf, _e) in got}
    return got


mini = {}
if _gu_fused:
    mini.update(load_tri(hdr, PACK_FILE, "%s0." % _pack_pref,
                         "mlp.gate_up_proj", 0))
else:
    _g = load_tri(hdr, PACK_FILE, "%s0." % _pack_pref, "mlp.gate_proj", 0)
    _u = load_tri(hdr, PACK_FILE, "%s0." % _pack_pref, "mlp.up_proj", 0)
    if _g and _u:
        # fuse sepanjang N (urutan: gate lalu up — sama dgn loader Mojo)
        mini["model.layers.0.mlp.gate_up_proj.weight"] = np.concatenate(
            [_g["model.layers.0.mlp.gate_proj.weight"],
             _u["model.layers.0.mlp.up_proj.weight"]], axis=0)
        mini["model.layers.0.mlp.gate_up_proj.scales"] = np.concatenate(
            [_g["model.layers.0.mlp.gate_proj.scales"],
             _u["model.layers.0.mlp.up_proj.scales"]], axis=0)
        mini["model.layers.0.mlp.gate_up_proj.biases"] = np.concatenate(
            [_g["model.layers.0.mlp.gate_proj.biases"],
             _u["model.layers.0.mlp.up_proj.biases"]], axis=0)
        log("   [FUSE] gate+up digabung -> gate_up %s"
            % (mini["model.layers.0.mlp.gate_up_proj.weight"].shape,))
mini.update(load_tri(hdr, PACK_FILE, "%s0." % _pack_pref, "mlp.down_proj", 0))
if _ipa_fused:
    mini.update(load_tri(hdr, PACK_FILE, "%s0." % _pack_pref,
                         "linear_attn.in_proj_all", 0))
elif _ipa_split:
    _q = load_tri(hdr, PACK_FILE, "%s0." % _pack_pref,
                  "linear_attn.in_proj_qkv", 0)
    _z = load_tri(hdr, PACK_FILE, "%s0." % _pack_pref,
                  "linear_attn.in_proj_z", 0)
    if _q and _z:
        for _e in ("weight", "scales", "biases"):
            mini["model.layers.0.linear_attn.in_proj_all.%s" % _e] = (
                np.concatenate([
                    _q["model.layers.0.linear_attn.in_proj_qkv.%s" % _e],
                    _z["model.layers.0.linear_attn.in_proj_z.%s" % _e]],
                    axis=0))
        log("   [FUSE] in_proj_qkv+in_proj_z digabung -> in_proj_all %s"
            % (mini["model.layers.0.linear_attn.in_proj_all.weight"].shape,))
mini.update(load_tri(hdr, PACK_FILE, "%s0." % _pack_pref,
                     "linear_attn.out_proj", 0))

log("   mini-pack: %d tensor dari pack asli" % len(mini))
for _nm, _arr in sorted(mini.items()):
    log("     %-52s %s %s" % (_nm, _arr.dtype, _arr.shape))

MINI_DIR = os.path.join(TMP, "minipack")
os.makedirs(MINI_DIR, exist_ok=True)
MINI_FILE = os.path.join(MINI_DIR, "model-mini-L00.safetensors")
from gsq_driver import write_safetensors  # noqa: E402

write_safetensors(mini, MINI_FILE, metadata={"src": "bonsai-2bit-weights slice"})
# catat dimensi sebelum mini dibebaskan
_NFULL = mini["model.layers.0.mlp.gate_up_proj.weight"].shape[0]
_KFULL = mini["model.layers.0.mlp.gate_up_proj.weight"].shape[1] * 16
del mini
gc.collect()
log("   [OK] mini-pack ditulis: %s (%.1f MB) | gate_up N=%d K=%d"
    % (MINI_FILE, os.path.getsize(MINI_FILE) / 1e6, _NFULL, _KFULL))

# --- jalankan driver ASLI pada mini-pack (byte asli, bukan sintetis) --------
log(">> [8/8] [C1] refine nyata: linear_attn.out_proj (kecil) ...")
_rc = subprocess.run(
    [sys.executable, _drv, "refine",
     "--pack", MINI_DIR, "--teacher", TEACHER_NPZ,
     "--calib", CALIB_NPZ, "--hadamard", HADAMARD,
     "--out", os.path.join(TMP, "out_small"),
     "--layers", "0", "--suffixes", "linear_attn.out_proj",
     "--iters", "10", "--lr", "5e-2"],
    capture_output=True, text=True)
log("   rc=%d" % _rc.returncode)
for _line in (_rc.stdout + _rc.stderr).splitlines():
    log("   | %s" % _line)
_qad_line = [l for l in _rc.stdout.splitlines() if l.startswith("[GSQ]")]
record("[C1] refine out_proj atas byte pack asli jalan",
       _rc.returncode == 0 and bool(_qad_line),
       ("rc=%d" % _rc.returncode) + (" | %s" % _qad_line[0] if _qad_line else ""))

# verifikasi ulang kontrak pada KELUARAN driver
OUT_SMALL = os.path.join(TMP, "out_small", "model-mini-L00.safetensors")
if os.path.isfile(OUT_SMALL):
    try:
        from gsq_driver import read_quant_tensor  # noqa: E402

        _o_hdr = read_st_header(OUT_SMALL)
        _o_names = [k for k in _o_hdr if k != "__metadata__"]
        log("   keluaran: %d tensor" % len(_o_names))
        _q, _s = read_quant_tensor(
            read_selected(OUT_SMALL, _o_hdr, _o_names),
            "model.layers.0.", "linear_attn.out_proj")
        _uniq = sorted(set(np.unique(_q).tolist()))
        _qok = set(_uniq) <= {0, 1, 2}
        log("   q unik: %s | bentuk q %s s %s" % (_uniq, _q.shape, _s.shape))
        record("[C1] keluaran refine: q dalam codebook {0,1,2}", _qok,
               "unik=%s" % _uniq)
        record("[C1] keluaran refine: bentuk scales [N, K/128]",
               _s.shape == (_q.shape[0], _q.shape[1] // 128), "%s" % (_s.shape,))
    except Exception as _e:
        log("   [ERROR] verifikasi keluaran: %r" % (_e,))
        record("[C1] keluaran refine tertulis", False,
               "%s: %s" % (type(_e).__name__, _e))
else:
    record("[C1] keluaran refine tertulis", False, "berkas tak ada")

# --- [C2] tensor BESAR (gate_up) — RAM diukur, BUKAN asumsi ---------------
# v4 Kernel: rc=-9 (SIGKILL) pada gate_up penuh [34816,5120] walau MemTotal
# 32100 MB -> kernel Kaggle membatasi memori LEBIH RENDAH dari MemTotal.
# Estimasi kebutuhan gsq_refine untuk [N,K]:
#   logits [N,K,3] f32  = 4*N*K*3
#   Adam m + v          = 2x logits
#   w_teacher f32       = 4*N*K
#   -> kasar 17 * N * K byte  (ditambah transient gumbel/grad)
_mt = mem_total_mb()
_cg = -1
for _p in ("/sys/fs/cgroup/memory.max",
           "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
    try:
        with open(_p) as fh:
            _cg = int(fh.read().strip()) // (1024 * 1024)
        break
    except Exception:
        pass
log(">> [8/8] [C2] gate_up (besar) — MemTotal=%d MB cgroup=%d MB"
    % (_mt, _cg))
# pakai batas terkecil yg terukur (cgroup lebih otoritatif bila ada)
_eff = min([x for x in (_mt, _cg) if x > 0], default=0)


def _est_mb(n, k):
    return (17 * n * k) // (1024 * 1024)


# cari skala baris terbesar yg muat: N penuh dulu, lalu pangkas 50% bertahap
_KFULL = 5120
log("   gate_up penuh N=%d K=%d -> est %.0f MB (batas efektif %d MB)"
    % (_NFULL, _KFULL, _est_mb(_NFULL, _KFULL), _eff))

_run_big = False
if _eff > 0 and _est_mb(_NFULL, _KFULL) <= _eff * 0.75:
    _run_big = True
    _N = _NFULL
else:
    # pangkas baris sampai muat dgn margin 25%; tetap data nyata, hanya
    # subset baris (bukan tensor sintetis)
    _N = _NFULL
    while _N > 512 and _est_mb(_N, _KFULL) > _eff * 0.75:
        _N //= 2
    log("   [PANGKAS] gate_up dipangkas ke N=%d (est %d MB) — N penuh "
        "melebihi batas memori kernel (terbukti rc=-9 di kernel sebelumnya)"
        % (_N, _est_mb(_N, _KFULL)))

if _run_big:
    _rc2 = subprocess.run(
        [sys.executable, _drv, "refine",
         "--pack", MINI_DIR, "--teacher", TEACHER_NPZ,
         "--calib", CALIB_NPZ, "--hadamard", HADAMARD,
         "--out", os.path.join(TMP, "out_big"),
         "--layers", "0",
         "--suffixes", "mlp.gate_up_proj,mlp.down_proj",
         "--iters", "5", "--lr", "5e-2"],
        capture_output=True, text=True)
    log("   rc=%d" % _rc2.returncode)
    for _line in (_rc2.stdout + _rc2.stderr).splitlines():
        log("   | %s" % _line)
    _gsq2 = [l for l in _rc2.stdout.splitlines() if l.startswith("[GSQ]")]
    _has_down = any("mlp.down_proj" in l for l in _gsq2)
    record("[C2] refine gate_up (N penuh) + down_proj (turunan) jalan",
           _rc2.returncode == 0 and len(_gsq2) >= 1 and _has_down,
           "rc=%d | %d baris [GSQ], down_proj=%s"
           % (_rc2.returncode, len(_gsq2), _has_down))
else:
    # bangun mini-pack terpangkas dengan membaca ulang dari MINI_FILE (mini
    # sudah di-del). Potong baris weight (U32 [N,K/16]) dan scales/biases
    # ([N,K/128]) sepanjang sumbu 0; tensor lain disalin utuh.
    _mini_hdr = read_st_header(MINI_FILE)
    _mini_names = [k for k in _mini_hdr if k != "__metadata__"]
    _mini_all = read_selected(MINI_FILE, _mini_hdr, _mini_names)
    _mp_small = {}
    for _nm, _arr in _mini_all.items():
        if _nm.endswith("mlp.gate_up_proj.weight") or \
           _nm.endswith("mlp.gate_up_proj.scales") or \
           _nm.endswith("mlp.gate_up_proj.biases"):
            _mp_small[_nm] = _arr[:_N]
        else:
            _mp_small[_nm] = _arr
    del _mini_all
    gc.collect()
    SMALL_DIR = os.path.join(TMP, "minipack_small")
    os.makedirs(SMALL_DIR, exist_ok=True)
    write_safetensors(_mp_small,
                      os.path.join(SMALL_DIR, "model-mini-small.safetensors"),
                      metadata={"src": "slice N=%d" % _N})
    del _mp_small
    gc.collect()
    log("   mini-pack terpangkas N=%d ditulis" % _N)
    _rc2 = subprocess.run(
        [sys.executable, _drv, "refine",
         "--pack", SMALL_DIR, "--teacher", TEACHER_NPZ,
         "--calib", CALIB_NPZ, "--hadamard", HADAMARD,
         "--out", os.path.join(TMP, "out_big"),
         "--layers", "0",
         "--suffixes", "mlp.gate_up_proj,mlp.down_proj",
         "--iters", "5", "--lr", "5e-2"],
        capture_output=True, text=True)
    log("   rc=%d" % _rc2.returncode)
    for _line in (_rc2.stdout + _rc2.stderr).splitlines():
        log("   | %s" % _line)
    _gsq2 = [l for l in _rc2.stdout.splitlines() if l.startswith("[GSQ]")]
    _has_down = any("mlp.down_proj" in l for l in _gsq2)
    record("[C2] refine gate_up (N=%d, pangkas RAM) + down_proj jalan" % _N,
           _rc2.returncode == 0 and len(_gsq2) >= 1 and _has_down,
           "rc=%d | %d baris [GSQ], down_proj=%s"
           % (_rc2.returncode, len(_gsq2), _has_down))
    record("[C2] gate_up N penuh %d MUAT di RAM kernel" % _NFULL, False,
           "est %.0f MB > batas %d MB (rc=-9 terbukti sebelumnya)"
           % (_est_mb(_NFULL, _KFULL), _eff))

log(">> [RAM] VmHWM kernel induk = %d MB" % rss_hwm_mb())

# ---------------------------------------------------------------------------
# Ringkasan akhir
# ---------------------------------------------------------------------------
log("")
log("=========================================================")
log(" RINGKASAN VERDICT")
log("=========================================================")
_n_ok = 0
for _lab, _ok, _det in VERDICT:
    log("  [%s] %s%s" % ("OK " if _ok else "GAGAL", _lab,
                         ("  (" + _det + ")") if _det else ""))
    _n_ok += 1 if _ok else 0
log("  --- %d/%d lulus ---" % (_n_ok, len(VERDICT)))
log("  total %.1f menit" % ((time.time() - T0) / 60.0))

_summary = {
    "lulus": _n_ok,
    "total": len(VERDICT),
    "durasi_menit": round((time.time() - T0) / 60.0, 1),
    "ram_hwm_mb": rss_hwm_mb(),
    "mem_total_mb": _mt,
    "cgroup_limit_mb": _cg,
    "memori_efektif_mb": _eff,
    "gate_up_n_penuh": _NFULL,
    "gate_up_est_mb_penuh": _est_mb(_NFULL, _KFULL) if _NFULL else None,
    "gate_up_n_dipakai": _N if _NFULL else None,
    "scales_dtype": sorted(_scales_dt),
    "pack_layer_prefix": _pack_pref,
    "pack_n_tensors": len(names),
    "mlp_fused": _gu_fused,
    "in_proj_all_fused": _ipa_fused,
    "guru_hilang_64layer": _miss_total,
    "calib": "SINTETIS (seed 12345) - loss BUKAN ukuran akurasi",
    "verdict": [
        {"label": _l, "ok": bool(_o), "detail": _d} for _l, _o, _d in VERDICT
    ],
}
with open(os.path.join(OUT, "realdata_verdict.json"), "w") as fh:
    json.dump(_summary, fh, indent=2)
log("  verdict tertulis: %s/realdata_verdict.json" % OUT)

sys.exit(0 if _n_ok == len(VERDICT) else 1)
'''

# ---------------------------------------------------------------------------
# Tulis kernel akhir
# ---------------------------------------------------------------------------
BLOBS = []
for var, path in SOURCES:
    if not os.path.isfile(path):
        raise SystemExit("sumber hilang: %s" % path)
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    BLOBS.append((var, b64))
    print("[gen] %-24s %8d byte -> %8d b64" % (var, os.path.getsize(path), len(b64)))

_dst = os.path.join(HERE, "run_realdata.py")
with open(_dst, "w", encoding="utf-8") as out:
    out.write('#!/usr/bin/env python3\n')
    out.write('# DIBANGKITKAN OLEH build_kernel.py — jangan disunting tangan.\n')
    out.write('# Jalankan: python3 build_kernel.py\n\n')
    for var, b64 in BLOBS:
        out.write('%s = "%s"\n\n' % (var, b64))
    out.write(BODY)

print("[gen] selesai : %s" % _dst)
print("[gen] ukuran  : %d byte (%.1f KB)" % (os.path.getsize(_dst),
                                             os.path.getsize(_dst) / 1024.0))
