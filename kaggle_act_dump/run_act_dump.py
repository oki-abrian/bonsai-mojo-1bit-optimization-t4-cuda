#!/usr/bin/env python3
# ==============================================================================
# DUMP AKTIVASI KALIBRASI NYATA DARI MODEL 2-BIT (GPU Kaggle T4)
# ------------------------------------------------------------------------------
# Berdiri sendiri — TIDAK menyentuh deploy_on_kaggle.sh / run_deploy.py /
# push_to_kaggle.sh / kaggle_cpu_build/push_cpu_build.sh, dan TIDAK mengubah
# satu baris pun src/** atau main.mojo (perbaikan ada di tingkat *driver*).
#
# Tujuan (rantai akurasi): mengganti kalibrasi SINTETIS (seed 12345) di kernel
# gsq-realdata-e2e dgn aktivasi NYATA yang dilihat model 2-bit saat decode,
# agar gsq_refine meminimalkan kesalahan keluaran pada distribusi sungguhan.
#
# Spesifikasi kalibrasi diambil dari converter referensi
# (convert_mxfp4_optimized.py:2025-2073, extract_scpfe_micro_patch):
#   20 task universal (5 coding + 5 math + 5 logic + 5 chat), sampler per
#   kategori (temp 0.8/1.0, top_p 0.95, top_k 20), aktivasi per task per
#   layer di input MLP (site 0) & input attention (site 1); Mojo menambah
#   site 2 (input out_proj GDN).
#
# Sifat act_dump_flush() (act_dump.mojo:188): open(path,"w") MEMOTONG, dan
# dump hanya menyala untuk generate() pertama satu proses. Karena itu setiap
# task dijalankan sebagai invokasi bonsai_infer SENDIRI dgn dir dump sendiri,
# lalu format ACTD di-merge (concat per-layer token block + perbarui
# n_tokens). Hasil merge kompatibel dgn act_dump_reader.py (asertif
# off == len(data)).
#
# -----------------------------------------------------------------------------
# PERBAIKAN CRASH rc=-11 (kernel v1..v3, temuan forensik biner 2026-09-28):
# -----------------------------------------------------------------------------
# `src::ops::khq_state_slot_cell` di bonsai_infer (build CPU) terkodekan sebagai
#     dlopen(so) -> dlsym("khq_state_slot") -> dlclose(so) -> call *fn
# yaitu dlclose DULU, baru memanggil simbol. Ketika dlclose itu menjatuhkan
# refcount dlopen ke nol, glibc MELEPAS PEMETAAN .so, dan call tak-langsung
# melompat ke alamat yang tak lagi terpetakan -> SIGSEGV. Ke-7 titik FFI lain
# di biner yang sama memanggil simbol DULU, lalu dlclose (benar); hanya
# fungsi pengakses slot ini yang kompilernya menggeser destruktor handle ke
# depan pemanggilan ekor. Konsekuensinya persis cocok dgn gejala: crash hanya
# saat BONSAI_DUMP_ACT_DIR diset, di dalam act_dump_configure -> _ad() ->
# khq_state_slot_cell(2), sebelum cetakan "[ACT-DUMP] aktif", bahkan saat
# cap=0. Jalur 2-bit sendiri selamat karena tiap titik FFI lain punya
# fallback Mojo native; pengakses slot TIDAK punya fallback (except: NULL).
#
# Perbaikan (aditif, level driver, tanpa build ulang): LD_PRELOAD .so sehingga
# ia masuk peta tautan awal proses dan tak pernah bisa dilepas oleh dlclose
# tersebut. Diverifikasi oleh probe terisolasi tahap 2c yang mereproduksi
# persis pola kodegen itu di dalam proses Python anak.
#
# Alur:
#   1. Ambil wheel hasil build CPU (okiabrian/bonsai-build-cpu), ekstrak,
#      verifikasi simbol kernel 2-bit (tolak biner basi).
#   2. Siapkan runtime Mojo (libKGENCompilerRTShared.so; pasang pixi bila
#      pustaka belum ada di container).
#  2c. PROBE FFI: reproduksi dlopen->dlsym->dlclose->call (tanpa dan dgn
#      LD_PRELOAD) dalam proses anak terisolasi.
#   3. Pakai pack 2-bit dari dataset okiabrian/bonsai-2bit-weights (sudah
#      ter-mount; TIDAK ada unduhan 8,6 GB).
#   4. Tokenisasi 20 prompt kalibrasi (bungkus ChatML manual + validasi id).
#  4b. Matriks diagnostik 6 run pendek (isolasi penyebab crash kernel v1).
#   5. Per task: bonsai_infer --gpu BONSAI_BITS=2 + env sampling + env dump.
#   6. Merge seluruh dump per-task -> act_*.bin tunggal di /kaggle/working.
#   7. Validasi (parse ulang + statistik non-degenerasi) + verdict JSON.
# ==============================================================================

import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# Parameter (env hanya untuk penyetelan dari luar; default = spesifikasi
# converter referensi)
# ---------------------------------------------------------------------------
TOKENS_PER_TASK = int(os.environ.get("BONSAI_AD_TOKENS", "256"))
# converter referensi memakai 512; 256 sudah memberi 5120 token decode total
# (20 task), ~40x dari konvensi 128 sampel kalibrasi GPTQ/GSQ biasa.

EXPECTED_PACK_BYTES = 8595477990

# 20 prompt — disalin persis dari convert_mxfp4_optimized.py:2025-2053
# (kalibrasi milik proyek ini, bukan teks umum).
PROMPTS = {
    "coding": [
        "Tuliskan kode Python lengkap untuk implementasi algoritma A* Search pada grid 2D, sertakan penjelasan kompleksitas waktunya.",
        "Buatlah script Python untuk melakukan web scraping data cuaca menggunakan BeautifulSoup, lalu olah datanya dengan Pandas.",
        "Tuliskan fungsi dalam C++ untuk membalik sebuah Binary Tree. Jelaskan time complexity dan space complexity-nya.",
        "Rancang sebuah arsitektur microservices untuk aplikasi e-commerce menggunakan Docker dan Kubernetes. Sebutkan komponen utamanya.",
        "Berikan contoh kode SQL untuk melakukan query JOIN pada tiga tabel yang memiliki relasi many-to-many, dan urutkan hasilnya.",
    ],
    "math": [
        "Selesaikan persamaan diferensial orde dua berikut: d^2y/dx^2 - 3(dy/dx) + 2y = e^x. Tunjukkan langkah-langkah pembuktiannya secara rinci.",
        "Buktikan bahwa akar kuadrat dari 2 adalah bilangan irasional menggunakan kontradiksi.",
        "Hitung integral tentu dari x^2 * sin(x) dx mulai dari 0 hingga pi menggunakan metode integrasi parsial.",
        "Jika peluang turun hujan besok adalah 0.3 dan lusa adalah 0.6, berapakah probabilitas setidaknya ada satu hari hujan?",
        "Tentukan nilai eigen dan vektor eigen dari matriks 2x2: [[4, 1], [2, 3]].",
    ],
    "logic": [
        "Terdapat 3 kotak. Kotak A berisi apel, Kotak B berisi jeruk, Kotak C berisi campuran. Semua label di luar kotak salah. Jika kamu hanya boleh mengambil 1 buah dari 1 kotak, bagaimana caramu melabeli ulang semuanya dengan benar? Jelaskan alur logikanya.",
        "Tiga orang pengembara harus menyeberangi sungai dengan sebuah perahu yang hanya muat 2 orang. Dua di antaranya memiliki beban berat yang tak bisa ditinggalkan. Bagaimana cara mereka menyeberang?",
        "Dalam sebuah balapan, kamu baru saja menyalip orang di posisi kedua. Sekarang kamu berada di posisi berapa?",
        "Jika 5 mesin butuh 5 menit untuk membuat 5 alat, berapa lama waktu yang dibutuhkan 100 mesin untuk membuat 100 alat?",
        "Sebuah kereta listrik melaju ke arah selatan dengan kecepatan 100 km/jam. Angin bertiup ke timur. Ke arah mana asap kereta akan terbang?",
    ],
    "chat": [
        "Jelaskan sejarah revolusi industri secara panjang lebar dan bagaimana dampaknya terhadap kehidupan sosial ekonomi masyarakat modern saat ini.",
        "Buatlah draf email profesional untuk menolak tawaran pekerjaan secara halus karena sudah menerima tawaran dari perusahaan lain.",
        "Tuliskan sebuah esai persuasif tentang pentingnya menjaga kesehatan mental di lingkungan kerja yang serba cepat.",
        "Ceritakan kisah fiksi pendek tentang seorang astronot yang tersesat di planet tanpa nama dan menemukan sisa-sisa peradaban kuno.",
        "Rangkumkan perbedaan utama antara filsafat Stoikisme dan Eksistensialisme dalam bahasa yang mudah dipahami.",
    ],
}

# (kategori, prompt, temp, top_p, top_k, presence, rep) — sama persis dgn
# cabang "universal" converter referensi. Catatan: mesin Mojo TIDAK punya
# presence penalty (hanya rep penalty) -> presence diabaikan (dilaporkan);
# rep 1.0 = BONSAI_REP_PENALTY_X100=100 = mati (default).
TASKS = []
for _cat in ("coding", "math", "logic", "chat"):
    for _i, _p in enumerate(PROMPTS[_cat]):
        if _cat == "coding":
            TASKS.append(("%s_%d" % (_cat, _i), _p, 0.8, 0.95, 20, 0.0, 1.0))
        else:
            TASKS.append(("%s_%d" % (_cat, _i), _p, 1.0, 0.95, 20, 1.5, 1.0))

NEW_SYMBOLS = [
    "launch_qmv_sm75_b2_decode_fp16",
    "launch_fwht_sm75_fp16",
    "launch_qmv_sm75_dense_fp16",
    "launch_qmm_sm75_b2_prefill_fp16",
    "launch_qmm_sm75_b2_prefill_int8",
    "launch_qmv_sm75_b2_decode_h2",
    "launch_qmv_sm75_b2_decode_h2b",
]

PIXI_TOML = """[project]
name = "bonsai-mojo-runtime"
version = "0.1.0"
channels = ["https://conda.modular.com/max",
            "https://repo.prefix.dev/modular-community",
            "conda-forge"]
platforms = ["linux-64"]

[dependencies]
# Pin seri 25.x — sama persis dengan pixi.toml proyek.
max = ">=25.0,<26.0"
"""

RUNTIME_NAMES = ("libKGENCompilerRTShared.so", "libMSupportGlobals.so",
                 "libAsyncRTRuntimeGlobals.so", "libNVPTX.so")

ACT_MAGIC = 0x44544341  # 'ACTD'
SITE_FILES = {0: "act_mlp_in.bin", 1: "act_attn_in.bin", 2: "act_ssm_in.bin"}
SITE_NAMES = {0: "mlp_in", 1: "attn_in", 2: "ssm_in"}
# Lebar site yang diharapkan (diconfig; dicek ulang dari berkas dump).
SITE_WIDTH_EXPECT = {0: 5120, 1: 5120, 2: 6144}

# Layer-type rule (main.mojo): is_linear = li % 4 != 3 -> 48 GDN + 16 attn.
FULL_ATTN_INTERVAL = 4

# Daftar verdict global (probe + diagnostik + merge sama-sama dilaporkan).
VERDICT = []


def log(*a):
    print(">>", *a, flush=True)


def fatal(*a):
    print(">> [FATAL]", *a, flush=True)
    sys.exit(1)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def record(label, ok, detail=""):
    VERDICT.append({"label": label, "ok": bool(ok), "detail": str(detail)})
    tag = "OK " if ok else "GAGAL"
    log("   [%s] %s  (%s)" % (tag, label, detail))


# ---------------------------------------------------------------------------
# 1. Ambil biner hasil build CPU
# ---------------------------------------------------------------------------
log("1. MENGAMBIL BINER HASIL BUILD CPU")
wheels = sorted(glob.glob("/kaggle/input/**/bonsai_1bit_t4-*.whl", recursive=True))
if not wheels:
    fatal("wheel hasil build CPU tidak ditemukan di /kaggle/input")
whl = wheels[-1]
log("   wheel:", whl, "(%d byte)" % os.path.getsize(whl))

base = "/kaggle/working"
build_dir = os.path.join(base, "build2")
if os.path.isdir(build_dir):
    shutil.rmtree(build_dir)
os.makedirs(build_dir, exist_ok=True)
with zipfile.ZipFile(whl) as z:
    z.extractall(build_dir)

infer = os.path.join(build_dir, "bonsai_1bit_t4", "bin", "bonsai_infer")
so = os.path.join(build_dir, "bonsai_1bit_t4", "libbonsai_qmv_sm75.so")
for p in (infer, so):
    if not os.path.exists(p):
        fatal("artefak tidak ada:", p)
os.chmod(infer, 0o755)
log("   bonsai_infer:", os.path.getsize(infer), "byte")
log("   .so         :", os.path.getsize(so), "byte")

# ---------------------------------------------------------------------------
# 2. Tolak biner basi
# ---------------------------------------------------------------------------
log("2. VERIFIKASI .so BUKAN BINER BASI")
syms = sh(["nm", "-D", so])
missing = [s for s in NEW_SYMBOLS if s not in syms]
if missing:
    fatal("simbol 2-bit hilang dari .so (biner basi?):", missing)
log("   [OK] semua kernel 2-bit ada:", ", ".join(NEW_SYMBOLS))
if "khq_state_slot" not in syms:
    fatal("khq_state_slot hilang dari .so — slot state tak bisa diakses")
log("   [OK] khq_state_slot ada (dipakai act_dump slot 2)")

# ---------------------------------------------------------------------------
# 2b. Sediakan runtime Mojo (sama persis dgn kaggle_2bit_test/run_infer_2bit.py)
# ---------------------------------------------------------------------------
log("2b. MENYIAPKAN RUNTIME MOJO")


def find_kgen(root="/"):
    pat = " -o ".join("-name '" + n + "'" for n in RUNTIME_NAMES)
    out = sh(["bash", "-lc",
              "find " + root + " \\( " + pat + " \\) -not -path '/proc/*' "
              "2>/dev/null | head -20"])
    dirs = []
    for l in out.splitlines():
        l = l.strip()
        if not l:
            continue
        d = os.path.dirname(l)
        if d not in dirs:
            dirs.append(d)
    return dirs


def ldd_missing(binary, extra_dirs):
    ld = os.pathsep.join(list(extra_dirs) + [os.environ.get("LD_LIBRARY_PATH", "")])
    out = sh(["bash", "-c",
              "LD_LIBRARY_PATH='" + ld + "' ldd " + binary
              + " 2>&1 | grep 'not found' || true"])
    return sorted({l.split("=>")[0].strip() for l in out.splitlines()
                   if "not found" in l})


rt_libs = []
hits = find_kgen()
if hits:
    log("   pustaka runtime ditemukan di:", ", ".join(hits))
    rt_libs = list(hits)

missing = ldd_missing(infer, rt_libs)
log("   pustaka yang belum terpenuhi:", missing if missing else "tidak ada")

if missing:
    log("   -> memasang pixi + max 25.x")
    env_dir = os.path.join(base, "mojoenv")
    if os.path.isdir(env_dir):
        shutil.rmtree(env_dir)
    os.makedirs(env_dir, exist_ok=True)
    with open(os.path.join(env_dir, "pixi.toml"), "w", encoding="utf-8") as f:
        f.write(PIXI_TOML)
    subprocess.run(["bash", "-c", "curl -fsSL https://pixi.sh/install.sh | bash"])
    pixi = os.path.expanduser("~/.pixi/bin/pixi")
    if not os.path.exists(pixi):
        pixi = sh(["bash", "-lc", "command -v pixi"]).strip()
    if not pixi or not os.path.exists(pixi):
        fatal("pixi gagal terpasang — runtime Mojo tidak bisa disiapkan")
    log("   pixi:", sh([pixi, "--version"]).strip())
    log("   ruang disk sebelum install:", sh(["df", "-h", base]).splitlines()[-1])
    log("   menjalankan pixi install (mengunduh toolchain Mojo 25.x) ...")
    r = subprocess.run([pixi, "install"], cwd=env_dir)
    log("   pixi install exit:", r.returncode)
    log("   ruang disk setelah install:", sh(["df", "-h", base]).splitlines()[-1])
    for d in find_kgen(env_dir):
        if d not in rt_libs:
            rt_libs.insert(0, d)
    default_lib = os.path.join(env_dir, ".pixi", "envs", "default", "lib")
    if os.path.isdir(default_lib) and default_lib not in rt_libs:
        rt_libs.insert(0, default_lib)
    log("   direktori runtime:", ", ".join(rt_libs))
    missing = ldd_missing(infer, rt_libs)
    log("   pustaka yang belum terpenuhi setelah install:",
        missing if missing else "tidak ada")
    if missing:
        fatal("runtime Mojo tidak lengkap meski pixi sudah terpasang: "
              + ", ".join(missing))

ld_path = os.pathsep.join(rt_libs + [build_dir,
                                    os.environ.get("LD_LIBRARY_PATH", "")])

# ---------------------------------------------------------------------------
# 2c. PROBE FFI — reproduksi pola kodegen khq_state_slot_cell di proses anak.
#     Biner memanggil: dlopen -> dlsym -> dlclose -> call. Jika dlclose
#     melepas pemetaan .so, pemanggilan melompat ke alamat mati -> SIGSEGV
#     (rc=-11). Bandingkan tanpa dan dgn LD_PRELOAD (perbaikan driver).
#     Dijalankan dalam proses anak terisolasi agar SIGSEGV-nya tak membunuh
#     kernel ini.
# ---------------------------------------------------------------------------
log("2c. PROBE FFI dlopen->dlsym->dlclose->call (reproduksi kodegen slot)")

PROBE_SRC = r"""
import ctypes, sys
so_path, mode = sys.argv[1], sys.argv[2]
lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
fn = lib.khq_state_slot
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_int32]
if mode == "kodegen":
    # PERSIS seperti khq_state_slot_cell pada bonsai_infer build CPU:
    # dlsym dulu, dlclose, BARU memanggil simbolnya.
    lib.close()
    print("dlclose selesai; memanggil simbol sekarang ...", flush=True)
cell = fn(2)
print("OK cell(2) = 0x%x" % int(cell or 0), flush=True)
"""


def run_probe(mode, preload):
    """Jalankan probe FFI di proses anak. mode: 'aman' (panggil lalu tutup)
    atau 'kodegen' (tutup lalu panggil — persis kodegen bonsai_infer)."""
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = ld_path
    if preload:
        e["LD_PRELOAD"] = so
    else:
        e.pop("LD_PRELOAD", None)
    try:
        r = subprocess.run([sys.executable, "-c", PROBE_SRC, so, mode],
                           env=e, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"rc": "timeout", "stdout": "", "stderr": ""}
    return {"rc": r.returncode, "stdout": r.stdout or "",
            "stderr": r.stderr or ""}


_probe_results = {}
for _mode in ("aman", "kodegen"):
    for _pre in (False, True):
        _r = run_probe(_mode, _pre)
        _key = "%s/preload=%s" % (_mode, _pre)
        _probe_results[_key] = _r
        _ok = (_r["rc"] == 0)
        # 'aman' harus lulus dua-duanya; 'kodegen' tanpa preload DIHARAPKAN
        # crash (rc=-11) karena itulah bug-nya, dan 'kodegen' dgn preload
        # harus lulus (bukti perbaikan).
        log("   [%s] rc=%s %s"
            % (_key, _r["rc"], "(diharapkan: crash)" if (_mode == "kodegen"
               and not _pre and not _ok) else ""))
        for _ln in (_r["stdout"] + _r["stderr"]).strip().splitlines()[-4:]:
            if _ln.strip():
                log("        | " + _ln[:140])

record("probe FFI pola aman (panggil->tutup), tanpa preload",
       _probe_results["aman/preload=False"]["rc"] == 0,
       "rc=%s" % _probe_results["aman/preload=False"]["rc"])
_bug_rep = _probe_results["kodegen/preload=False"]["rc"]
record("probe FFI kodegen (tutup->panggil) TERPRODUKSI crash-nya",
       _bug_rep != 0, "rc=%s (SIGSEGV = -11)" % _bug_rep)
_fix_ok = _probe_results["kodegen/preload=True"]["rc"] == 0
record("probe FFI kodegen + LD_PRELOAD lulus (perbaikan)",
       _fix_ok, "rc=%s" % _probe_results["kodegen/preload=True"]["rc"])

if not _fix_ok:
    log("   [FATAL] LD_PRELOAD tak memperbaiki panggilan pasca-dlclose —")
    log("   kemungkinan .so ini juga bermasalah; periksa output probe di atas.")
    fatal("perbaikan driver tidak efektif; hentikan sebelum membuang GPU time")

# ---------------------------------------------------------------------------
# 3. Siapkan pack 2-bit dari dataset Kaggle
# ---------------------------------------------------------------------------
log("3. MENYIAPKAN BOBOT 2-BIT")
model_dir = None
for cand in sorted(glob.glob("/kaggle/input/**/model.safetensors",
                             recursive=True)):
    if os.path.getsize(cand) > 1024:
        model_dir = os.path.dirname(cand)
        break
if model_dir is None:
    fatal("model.safetensors tidak ter-mount dari okiabrian/bonsai-2bit-weights")
log("   model_dir:", model_dir)
log("   isi:", sorted(os.listdir(model_dir)))

st = os.path.join(model_dir, "model.safetensors")
got = os.path.getsize(st)
log("   model.safetensors: %d byte (diharapkan %d)" % (got, EXPECTED_PACK_BYTES))
if got < 1024:
    fatal("model.safetensors kosong/terpotong")
if got != EXPECTED_PACK_BYTES:
    log("   [WARN] ukuran berbeda dari harapan — lanjut, tapi waspada")

hj = os.path.join(model_dir, "hadamard.json")
if not os.path.exists(hj):
    fatal("hadamard.json tidak ada")
raw = open(hj, encoding="utf-8").read()
log("   hadamard.json: %d byte" % len(raw))
for key in ("prism.hadamard.block_size", "prism.hadamard.sign_widths",
            "prism.hadamard.sign_values"):
    log("   %s: %s" % (key, "ADA" if key in raw else "TIDAK ADA"))

# ---------------------------------------------------------------------------
# 3b. Environment dasar bonsai_infer (dipakai diagnostik maupun run task)
# ---------------------------------------------------------------------------
# PERBAIKAN SUMBER (2026-09-28, src/ops.mojo): khq_state_slot_cell kini
# mematikan hasilnya dulu (`var ret = f(...); return ret`) seperti 8 titik FFI
# lain — dlclose tak lagi digeser ke depan pemanggilan. Karena itu LD_PRELOAD
# (perbaikan driver versi v4) sekarang DEFAULT MATI; nyalakan hanya bila
# biner di wheel ini ternyata belum memuat perbaikan (env BONSAI_AD_PRELOAD=1).
# A/B: v3 (biner lama, tanpa preload) = crash; v5 (biner baru, tanpa preload)
# harus lulus. Itulah bukti perbaikan sumber mempan.
_PRELOAD = os.environ.get("BONSAI_AD_PRELOAD", "0") in ("1", "true", "yes")
env_base = dict(os.environ)
env_base["BONSAI_BITS"] = "2"
env_base["BONSAI_CUDA_LIB"] = so
env_base["BONSAI_USE_GPU"] = "1"
env_base["LD_LIBRARY_PATH"] = ld_path
if _PRELOAD:
    env_base["LD_PRELOAD"] = so
    log("   LD_PRELOAD DIPASANG (BONSAI_AD_PRELOAD=1):", so)
else:
    env_base.pop("LD_PRELOAD", None)
    log("   LD_PRELOAD MATI — menguji perbaikan sumber apa adanya")
# Sampling default (mati) — BONSAI_DUMP_TOP2 tidak diset (biaya turunan).
env_base["BONSAI_TOP_K"] = "20"
env_base["BONSAI_TOP_P_X1000"] = "950"
env_base["BONSAI_MIN_P_X1000"] = "0"
env_base["BONSAI_REP_PENALTY_X100"] = "100"  # 100 = mati (rep 1.0)


def run_infer_once(ids, max_tokens, dump_dir=None, temp_x100=None, seed=None,
                   cap=None, unbuffered=False, kv_dir=None):
    """Jalankan bonsai_infer sekali. Kembalikan dict hasil (rc, stdout, ...).

    `cap`    : override BONSAI_DUMP_ACT_TOKENS.
    `kv_dir` : set BONSAI_DUMP_KV_DIR (probe silang slot 1, khq_dump).
    `unbuffered` : bungkus dgn `stdbuf -o0 -e0` supaya cetakan langsung
               keluar ke pipa — SIGSEGV tak bisa menelan baris terakhir
               (stdout pipe defaultnya di-buffer; buffer hilang saat crash).
    """
    e = dict(env_base)
    if temp_x100 is not None:
        e["BONSAI_TEMP_X100"] = str(temp_x100)
    if seed is not None:
        e["BONSAI_SEED"] = str(seed)
    if dump_dir is not None:
        os.makedirs(dump_dir, exist_ok=True)
        e["BONSAI_DUMP_ACT_DIR"] = dump_dir
        e["BONSAI_DUMP_ACT_TOKENS"] = str(cap if cap is not None else max_tokens)
    if kv_dir is not None:
        os.makedirs(kv_dir, exist_ok=True)
        e["BONSAI_DUMP_KV_DIR"] = kv_dir
    pt = ",".join(str(i) for i in ids)
    cmd = [infer, "--model-dir", model_dir, "--prompt-tokens", pt,
           "--max-tokens", str(max_tokens), "--gpu"]
    if unbuffered:
        _sb = sh(["which", "stdbuf"]).strip()
        if _sb:
            cmd = [_sb, "-o0", "-e0"] + cmd
        else:
            log("   [WARN] stdbuf tak ada — output tetap ter-buffer")
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    out = p.stdout or ""
    dump_live = ">> [ACT-DUMP] aktif" in out
    kv_live = ">> [KHQ-DUMP] aktif" in out
    ngen = -1
    for ln in out.splitlines():
        if "Selesai:" in ln and "token di-generate" in ln:
            try:
                ngen = int(ln.split("Selesai:")[1].split("token")[0].strip())
            except Exception:
                pass
    return {"rc": p.returncode, "stdout": out, "stderr": p.stderr or "",
            "dump_live": dump_live, "kv_live": kv_live, "ngen": ngen,
            "ok": p.returncode == 0 and ngen > 0}


# ---------------------------------------------------------------------------
# 4. Tokenisasi prompt kalibrasi
# ---------------------------------------------------------------------------
log("4. TOKENISASI %d PROMPT KALIBRASI" % len(TASKS))
tok = None
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
except Exception as e:
    log("   [WARN] transformers gagal (%s) -> coba pip install" % str(e)[:100])
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "transformers", "tokenizers"])
    log("   pip install exit:", r.returncode)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception as e2:
        log("   [FATAL] tokenizer tetap tidak tersedia:", str(e2)[:200])

if tok is None:
    fatal("tidak bisa menokenisasi prompt — tokenizer tidak tersedia")

# Vocab dipakai memvalidasi setiap id token (config.json -> vocab_size).
VOCAB = 248320
try:
    _cfg = json.load(open(os.path.join(model_dir, "config.json")))
    VOCAB = int(_cfg.get("vocab_size") or VOCAB)
except Exception:
    pass
log("   vocab_size (validasi rentang id):", VOCAB)


def encode_prompt(prompt):
    """Tokenisasi satu prompt ke id ChatML.

    PAKAI encode manual dgn bungkus ChatML — bukan apply_chat_template.
    Alasan (kejadian kernel v1): apply_chat_template(..., tokenize=True)
    mengembalikan BatchEncoding, dan list(BatchEncoding) adalah
    ['input_ids','attention_mask'] (2 STRING), bukan id. Prompt jadi
    "input_ids,attention_mask" dan mesin menerima id yang bukan id -> 2
    "token" untuk prompt 100+ kata, lalu crash. Cara manual ini persis yang
    dipakai kaggle_2bit_test/run_infer_2bit.py dan terbukti jalan.
    """
    text = ("<|im_start|>user\n" + prompt
            + "<|im_end|>\n<|im_start|>assistant\n")
    ids = tok.encode(text, add_special_tokens=False)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(i) for i in ids]


task_ids = []
for name, prompt, temp, top_p, top_k, pres, rep in TASKS:
    ids = encode_prompt(prompt)
    if len(ids) < 4:
        fatal("prompt %s hanya %d token — tokenizer rusak" % (name, len(ids)))
    bad = [i for i in ids if i < 0 or i >= VOCAB]
    if bad:
        fatal("prompt %s punya id di luar vocab: %s" % (name, bad[:8]))
    task_ids.append(ids)
    log("   %-10s prompt %4d token | cek: %s"
        % (name, len(ids), tok.decode(ids[:6]).replace("\n", " | ")[:60]))

# Referensi silang: apply_chat_template harus memberi id yang sama untuk
# prompt pertama (bukti bungkus ChatML manual setara template resmi).
try:
    _enc = tok.apply_chat_template(
        [{"role": "user", "content": TASKS[0][1]}],
        add_generation_prompt=True, tokenize=True)
    _ref = [int(i) for i in _enc["input_ids"]]
    log("   referensi apply_chat_template: %s (%d id)"
        % ("IDENTIK" if _ref == task_ids[0] else "BERBEDA", len(_ref)))
except Exception as e:
    log("   referensi apply_chat_template gagal (info saja): %s" % str(e)[:80])

max_prompt = max(len(x) for x in task_ids)
if max_prompt + TOKENS_PER_TASK > 4096:
    fatal("prompt %d + %d token > max_seq 4096 (lihat main.mojo:1528)"
          % (max_prompt, TOKENS_PER_TASK))
log("   [OK] prompt terpanjang %d + %d token <= max_seq 4096"
    % (max_prompt, TOKENS_PER_TASK))

# ---------------------------------------------------------------------------
# 4b. MATRIKS DIAGNOSTIK — run pendek (max-tokens 4) untuk mengisolasi
#     penyebab crash kernel v1 (rc=-11 sebelum dump hidup). Karena stdout
#     pipa di-buffer, baris terakhir yang terlihat BUKAN penanda pasti titik
#     crash; diagnosis diambil dari kombinasi lulus/gagal ke-6 run.
#
#     A: prompt dikenal-baik, TANPA env dump
#     B: prompt dikenal-baik, DENGAN env dump act  -> target utama
#     C: prompt nyata (tokenizer sudah diperbaiki), TANPA env dump
#     D: prompt nyata, DENGAN env dump act         -> target sebenarnya
#     E: prompt nyata, dump act cap=0              -> discriminator: jika
#        lulus, crash bukan di act_dump_configure
#     F: prompt dikenal-baik, DENGAN env dump KV   -> probe silang slot 1:
#        apakah pola slot (yg sama persis di khq_dump) juga lumpuh
# ---------------------------------------------------------------------------
log("4b. MATRIKS DIAGNOSTIK (6 run pendek, stdout tak-terbuffer)")

# Id ChatML "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
# (sama dgn kaggle_2bit_test/run_infer_2bit.py — terbukti jalan).
KNOWN_GOOD = [248045, 872, 198, 9419, 248046, 198, 248045, 74455, 198]

DUMP_ROOT = os.path.join(base, "actd")
if os.path.isdir(DUMP_ROOT):
    shutil.rmtree(DUMP_ROOT)
os.makedirs(DUMP_ROOT, exist_ok=True)

# E = dump dgn cap=0: `act_dump_site` pulang SEGERA (n[idx] >= cap) sebelum
#    alokasi staging/salin GPU. Bila E LULUS tapi D GAGAL -> crash ada di
#    badan act_dump_site, BUKAN di act_dump_configure.
# Semua run memakai stdbuf -o0 agar baris terakhir sebelum SIGSEGV terlihat.
diag = []
for _lab, _ids, _dump, _cap, _kv in (
    ("A_baik_nodump", KNOWN_GOOD, False, None, False),
    ("B_baik_dump", KNOWN_GOOD, True, None, False),
    ("C_real_nodump", task_ids[0], False, None, False),
    ("D_real_dump", task_ids[0], True, None, False),
    ("E_real_dump_cap0", task_ids[0], True, 0, False),
    ("F_baik_kvdump", KNOWN_GOOD, False, None, True),
):
    _d = os.path.join(DUMP_ROOT, "diag_%s" % _lab) if _dump else None
    _kv = os.path.join(DUMP_ROOT, "diag_%s" % _lab) if _kv else None
    r = run_infer_once(_ids, 4, dump_dir=_d, cap=_cap, unbuffered=True,
                       kv_dir=_kv)
    diag.append({"label": _lab, **{k: r[k] for k in ("rc", "dump_live",
                                                    "kv_live", "ngen", "ok")}})
    _cap_note = "" if _cap is None else " cap=%s" % _cap
    log("   [%s] rc=%s dump_live=%s ngen=%s%s"
        % (_lab, r["rc"], r["dump_live"], r["ngen"], _cap_note))
    _lines = [l for l in r["stdout"].splitlines() if l.strip()]
    if r["ok"]:
        for ln in _lines:
            log("        | " + ln[:150])
    else:
        # Run GAGAL: cetak 12 baris terakhir — itulah titik henti (stdbuf
        # menjamin tak ada yang tertahan di buffer saat SIGSEGV).
        log("        | ... 12 baris terakhir sebelum henti ...")
        for ln in _lines[-12:]:
            log("        | " + ln[:150])
    if r["stderr"].strip():
        for ln in r["stderr"].strip().splitlines()[-4:]:
            log("        ! " + ln[:150])
    record("diag %s" % _lab, r["ok"],
           "rc=%s ngen=%s%s" % (r["rc"], r["ngen"], _cap_note))

diag_ok = all(d["ok"] for d in diag if d["label"] != "E_real_dump_cap0")
# E sendiri tak harus lulus utk lanjut; ia discriminator, bukan syarat.
log("   ringkasan diagnostik: %d/%d lulus (E adalah discriminator)"
    % (sum(1 for d in diag if d["ok"]), len(diag)))
for d in diag:
    log("     %-18s rc=%-4s ngen=%-4s ok=%s"
        % (d["label"], d["rc"], d["ngen"], d["ok"]))

_dump_aborted = not diag_ok
if _dump_aborted:
    log("   [STOP] diagnostik gagal — dump batal; periksa log di atas.")

# ---------------------------------------------------------------------------
# 5. Jalankan inferensi 2-bit per task, dump aktivasi decode
# ---------------------------------------------------------------------------
task_results = []
if _dump_aborted:
    log("5. DILEWATI (diagnostik gagal)")
else:
    log("5. MENJALANKAN INFERENSI 2-BIT PER TASK (BONSAI_BITS=2, dump aktif)")
    log("   GPU:", sh(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader"]).strip())
    for ti, (name, prompt, temp, top_p, top_k, pres, rep) in enumerate(TASKS):
        tdir = os.path.join(DUMP_ROOT, "task%02d" % ti)
        r = run_infer_once(task_ids[ti], TOKENS_PER_TASK, dump_dir=tdir,
                           temp_x100=int(round(temp * 100)), seed=1234 + ti)
        task_results.append({
            "name": name, "rc": r["rc"], "dump_live": r["dump_live"],
            "tokens": r["ngen"], "ok": r["ok"],
        })
        if not r["ok"]:
            log("   [WARN] %s rc=%s dump_live=%s ngen=%s"
                % (name, r["rc"], r["dump_live"], r["ngen"]))
            for ln in (r["stdout"] or r["stderr"]).splitlines()[-8:]:
                if ln.strip():
                    log("        | " + ln[:150])
        else:
            log("   [OK] %s | %d token | dump hidup" % (name, r["ngen"]))

n_ok = sum(1 for t in task_results if t["ok"])
log("   ringkasan task: %d/%d sehat" % (n_ok, len(task_results)))

# ---------------------------------------------------------------------------
# 6. Merge dump per-task -> act_*.bin tunggal
# ---------------------------------------------------------------------------
log("6. MERGE DUMP PER-TASK")


def read_actd(path):
    """Baca satu site file -> (site, width, {layer_id: arr [n,width] f16})."""
    with open(path, "rb") as f:
        data = f.read()
    magic, site, n_layers, width = struct.unpack_from("<IIII", data, 0)
    assert magic == ACT_MAGIC, "magic salah di %s: 0x%08X" % (path, magic)
    off = 16
    blocks = []
    for _ in range(n_layers):
        layer_id, n_tokens = struct.unpack_from("<II", data, off)
        off += 8
        blocks.append((layer_id, n_tokens))
    out = {}
    for layer_id, n_tokens in blocks:
        cnt = n_tokens * width
        arr = np.frombuffer(data, dtype="<f2", count=cnt, offset=off)
        off += cnt * 2
        out[layer_id] = arr.reshape(n_tokens, width)
    assert off == len(data), "ukuran tidak pas di %s: %d vs %d" % (path, off, len(data))
    return site, width, out


def write_actd(path, site, width, data):
    """Tulis site file merged. data: {layer_id: arr [n,width] f16}."""
    lids = sorted(data)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", ACT_MAGIC, site, len(lids), width))
        for lid in lids:
            f.write(struct.pack("<II", lid, int(data[lid].shape[0])))
        for lid in lids:
            arr = np.ascontiguousarray(data[lid], dtype="<f2")
            f.write(arr.tobytes())


merged = {}          # site -> {layer_id: [arr, ...]}
site_width = {}      # site -> width
for ti, tr in enumerate(task_results):
    if not tr["ok"]:
        continue
    tdir = os.path.join(DUMP_ROOT, "task%02d" % ti)
    for site in SITE_FILES:
        fp = os.path.join(tdir, SITE_FILES[site])
        if not os.path.exists(fp):
            continue
        try:
            s, w, layers = read_actd(fp)
        except Exception as e:
            log("   [WARN] gagal baca %s: %s" % (fp, e))
            continue
        if s != site:
            log("   [WARN] site tidak cocok di %s: %d" % (fp, s))
            continue
        if site in site_width and site_width[site] != w:
            log("   [WARN] lebar site %d berubah: %d vs %d"
                % (site, w, site_width[site]))
            continue
        site_width[site] = w
        m = merged.setdefault(site, {})
        for lid, arr in layers.items():
            m.setdefault(lid, []).append(arr)

# Validasi keterbagaian layer + tulis merged.
for site in sorted(SITE_FILES):
    name = SITE_NAMES[site]
    if site not in merged or not merged[site]:
        record("site %d (%s) ada hasil merge" % (site, name), False,
               "tidak ada data")
        continue
    data = {lid: np.concatenate(parts, axis=0)
            for lid, parts in merged[site].items()}
    w = site_width[site]
    lids = sorted(data)
    n_tok_total = int(sum(data[l].shape[0] for l in lids))
    out_path = os.path.join(base, SITE_FILES[site])
    write_actd(out_path, site, w, data)
    sz = os.path.getsize(out_path)

    # Lapisan mana yang seharusnya menyala sesuai aturan tipe layer
    # (main.mojo: is_linear = li % 4 != 3 -> 48 GDN + 16 attn).
    # site 0 menyala di SEMUA layer; site 1 hanya layer attention (li%4==3);
    # site 2 hanya layer GDN (li%4!=3).
    if site == 0:
        want = list(range(64))
    elif site == 1:
        want = [l for l in range(64) if l % FULL_ATTN_INTERVAL == 3]
    else:
        want = [l for l in range(64) if l % FULL_ATTN_INTERVAL != 3]

    n_attn = len([l for l in want if l % FULL_ATTN_INTERVAL == 3])
    n_gdn = len(want) - n_attn
    cover = all(l in data for l in want)
    record("site %d (%s) tertulis" % (site, name), os.path.exists(out_path),
           "%s | %d layer | lebar %d | %d token | %.1f MB"
           % (SITE_FILES[site], len(lids), w, n_tok_total, sz / 1e6))
    record("site %d (%s) lebar sesuai harapan" % (site, name),
           w == SITE_WIDTH_EXPECT[site],
           "%d (harap %d)" % (w, SITE_WIDTH_EXPECT[site]))
    record("site %d (%s) cakupan layer %d (attn=%d gdn=%d)"
           % (site, name, len(want), n_attn, n_gdn), cover,
           "lengkap" if cover else "hilang: %s"
           % sorted(set(want) - set(lids))[:8])

    # Statistik non-degenerasi (dump kosong/nol = hook tidak menyala).
    sub = data[lids[0]][:64].astype(np.float32)
    amax = float(np.max(np.abs(sub)))
    amin = float(np.min(np.abs(sub)))
    nz = float(np.mean(np.abs(sub) > 0))
    record("site %d (%s) tidak degenerate" % (site, name),
           amax > 0 and nz > 0.5,
           "L%02d max|a|=%.4e min|a|=%.4e frac!=0=%.3f"
           % (lids[0], amax, amin, nz))

# Rata-rata token per task per site = total token / jumlah layer (semua layer
# di sebuah site melihat token task yang sama, jadi ini = rata-rata per task).
tok_per_task = {}
for site, parts in merged.items():
    tot = sum(sum(a.shape[0] for a in arrs) for arrs in parts.values())
    if parts:
        tok_per_task[site] = tot / len(parts)

# ---------------------------------------------------------------------------
# 7. Bersih-bersih + verdict JSON
# ---------------------------------------------------------------------------
log("7. PEMBERSIHAN & VERDICT")
# Buang dump per-task (sudah di-merge) + build dir agar output kernel ringkas.
for d in (DUMP_ROOT, build_dir):
    if d and os.path.isdir(d):
        shutil.rmtree(d)
        log("   dihapus:", d)

n_pass = sum(1 for v in VERDICT if v["ok"])
log("   --- %d/%d lulus ---" % (n_pass, len(VERDICT)))

out_json = {
    "lulus": n_pass,
    "total": len(VERDICT),
    "n_tasks": len(TASKS),
    "n_tasks_ok": n_ok,
    "tokens_per_task_budget": TOKENS_PER_TASK,
    "rata_token_per_task": {SITE_NAMES[s]: round(v, 1)
                            for s, v in tok_per_task.items()},
    "task_detail": task_results,
    "diagnostik": diag,
    "probe_ffi": {k: {"rc": v["rc"],
                      "tail": (v["stdout"] + v["stderr"]).strip()[-200:]}
                  for k, v in _probe_results.items()},
    "sites": {SITE_NAMES[s]: {"file": SITE_FILES[s], "width": site_width.get(s)}
              for s in sorted(SITE_FILES)},
    "verdict": VERDICT,
    "catatan": "sampling: temp 0.8/1.0, top_p 0.95, top_k 20, seed 1234+i. "
               "Presence penalty converter referensi TIDAK ada di mesin Mojo "
               "(diabaikan). rep penalty 1.0 = mati. Max-tokens referensi 512; "
               "di sini %d (total %d token decode). Perbaikan crash rc=-11: "
               "LD_PRELOAD libbonsai_qmv_sm75.so (lihat probe_ffi)." %
               (TOKENS_PER_TASK, TOKENS_PER_TASK * len(TASKS)),
}
vp = os.path.join(base, "actdump_verdict.json")
with open(vp, "w", encoding="utf-8") as f:
    json.dump(out_json, f, indent=1, ensure_ascii=False)
log("   verdict tertulis:", vp)
for ln in sh(["ls", "-la", base]).splitlines():
    if any(k in ln for k in ("act_", "verdict")):
        log("   ", ln)
