#!/usr/bin/env python3
# ==============================================================================
# UJI INFERENSI BONSAI-2 (2-bit ternary) DI GPU KAGGLE (T4)
# ------------------------------------------------------------------------------
# Berdiri sendiri — TIDAK menyentuh deploy_on_kaggle.sh / run_deploy.py.
# Alasan: skrip proven itu mencari bobot bernama 'bonsai-27b-mlx-1bit' di
# /kaggle/input, sedangkan pack 2-bit (8,6 GB) belum ada sebagai dataset Kaggle.
# Karena itu bobot diunduh LANGSUNG DI DALAM CONTAINER dari HuggingFace
# (internet aktif) — bukan di komputer lokal, sesuai aturan proyek.
#
# Alur:
#   1. Ambil biner hasil build CPU dari wheel di output kernel bonsai-build-cpu.
#   2. Verifikasi .so BUKAN biner basi (wajib memuat 3 kernel 2-bit).
#   3. Unduh prism-ml/Ternary-Bonsai-2-27B-mlx-2bit ke /kaggle/temp.
#   4. Jalankan bonsai_infer dengan BONSAI_BITS=2.
# ==============================================================================

import glob
import json
import os
import shutil
import subprocess
import sys
import zipfile

REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
FILES = [
    "config.json",
    "hadamard.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
]
NEW_SYMBOLS = [
    "launch_qmv_sm75_b2_decode_fp16",
    "launch_fwht_sm75_fp16",
    "launch_qmv_sm75_dense_fp16",
    # Prefill batched 2-bit (WMMA v2). Tanpa simbol ini prefill mundur ke
    # loop per-token dan angkanya tidak bisa dibandingkan dengan jalur 1-bit.
    "launch_qmm_sm75_b2_prefill_fp16",
    # Prefill batched 2-bit INT8 (tensor core int8 Turing). Jalur tambahan;
    # tanpa simbol ini BONSAI_PREFILL_INT8=1 tanpa suara jatuh ke WMMA fp16.
    "launch_qmm_sm75_b2_prefill_int8",
    # Decode 2-bit jalur cepat (PRMT + HFMA2 half2). Tanpa simbol ini
    # BONSAI_DECODE_H2=1 tanpa suara jatuh ke jalur skalar lama, dan run
    # tampak "lulus" tanpa menguji apa pun.
    "launch_qmv_sm75_b2_decode_h2",
    # Varian lebar (2 grup per lane, 4x uint4 in flight). Dipakai run H
    # untuk A/B melawan varian di atas dalam satu build yang sama.
    "launch_qmv_sm75_b2_decode_h2b",
]


def log(*a):
    print(">>", *a, flush=True)


def fatal(*a):
    print(">> [FATAL]", *a, flush=True)
    sys.exit(1)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def pick_dir():
    """Pilih lokasi unduh: /kaggle/temp bila ada (kuota lebih longgar)."""
    for cand in ("/kaggle/temp", "/kaggle/working"):
        if os.path.isdir(cand):
            return cand
    return "/tmp"


# --------------------------------------------------------------------------
# 1. Ambil artefak hasil build CPU dari wheel
# --------------------------------------------------------------------------
log("1. MENGAMBIL BINER HASIL BUILD CPU")
wheels = sorted(glob.glob("/kaggle/input/**/bonsai_1bit_t4-*.whl", recursive=True))
if not wheels:
    fatal("wheel hasil build CPU tidak ditemukan di /kaggle/input")
whl = wheels[-1]
log("   wheel:", whl, f"({os.path.getsize(whl)} byte)")

base = pick_dir()
log("   base:", base)
build_dir = os.path.join(base, "build2")
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

# --------------------------------------------------------------------------
# 2. Tolak biner basi — jangan tertipu artefak lama (pernah terjadi di v9)
# --------------------------------------------------------------------------
log("2. VERIFIKASI .so BUKAN BINER BASI")
syms = sh(["nm", "-D", so])
missing = [s for s in NEW_SYMBOLS if s not in syms]
if missing:
    fatal("simbol 2-bit hilang dari .so (biner basi?):", missing)
log("   [OK] semua kernel 2-bit ada:", ", ".join(NEW_SYMBOLS))

# --------------------------------------------------------------------------
# 2b. Sediakan runtime Mojo.
#     bonsai_infer adalah biner Mojo: ia butuh libKGENCompilerRTShared.so.
#     Di skrip proven (deploy_on_kaggle.sh baris 769) pustaka itu ikut lewat
#     LD_LIBRARY_PATH=$REPO_DIR/.pixi/envs/default/lib — yaitu environment
#     pixi tempat biner itu dibangun. Kernel ini TIDAK membangun dari sumber,
#     jadi environment itu harus dipasang dulu, bila belum ada.
# --------------------------------------------------------------------------
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


def find_kgen(root="/"):
    """Cari pustaka runtime Mojo, kembalikan direktori yang memuatnya.

    Pakai `find`, bukan glob: environment pixi berada di direktori tersembunyi
    `.pixi/`, dan glob di Python tidak melintasi direktori berawal titik.
    """
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
    """Daftar pustaka dinamis yang belum terpenuhi untuk `binary`."""
    ld = os.pathsep.join(list(extra_dirs) + [os.environ.get("LD_LIBRARY_PATH", "")])
    out = sh(["bash", "-c",
              "LD_LIBRARY_PATH='" + ld + "' ldd " + binary
              + " 2>&1 | grep 'not found' || true"])
    return sorted({l.split("=>")[0].strip() for l in out.splitlines()
                   if "not found" in l})


log("2b. MENYIAPKAN RUNTIME MOJO")
rt_libs = []
env_dir = None
hits = find_kgen()
if hits:
    log("   pustaka runtime ditemukan di:", ", ".join(hits))
    rt_libs = list(hits)

missing = ldd_missing(infer, rt_libs)
log("   pustaka yang belum terpenuhi:", missing if missing else "tidak ada")

if missing:
    # Pustaka yang ada di container ternyata tidak lengkap (hanya satu berkas
    # di artifacts_modal). Pasang environment pixi seperti saat build.
    log("   -> memasang pixi + max 25.x")
    env_dir = os.path.join(base, "mojoenv")
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

# --------------------------------------------------------------------------
# 3. Unduh bobot 2-bit dari HuggingFace
# --------------------------------------------------------------------------
log("3. MENYIAPKAN BOBOT 2-BIT")
# Bobot kini tersedia sebagai dataset Kaggle (okiabrian/bonsai-2bit-weights),
# dibuat oleh kaggle_2bit_weights/make_weights_dataset.py. Karena ter-mount
# dari /kaggle/input, tidak ada lagi unduhan 8,6 GB dari HuggingFace tiap run
# (~180 s). Bila dataset belum ter-mount, mundur ke unduhan seperti lama.
from_dataset = False
model_dir = os.path.join(base, "bonsai2")
for cand in sorted(glob.glob("/kaggle/input/**/model.safetensors",
                             recursive=True)):
    if os.path.getsize(cand) > 1024:
        model_dir = os.path.dirname(cand)
        from_dataset = True
        break
if from_dataset:
    log("   [OK] bobot dibaca dari dataset Kaggle:", model_dir)
else:
    os.makedirs(model_dir, exist_ok=True)
    log("   dataset tidak ter-mount -> unduh HuggingFace ke:", model_dir)
log("   target:", model_dir)
log("   ruang disk:", sh(["df", "-h", base]).splitlines()[-1] if sh(["df", "-h", base]) else "?")

log("   df -h /kaggle:")
for ln in sh(["df", "-h"]).splitlines():
    if "/kaggle" in ln or ln.startswith("Filesystem"):
        log("     ", ln)


def curl_get(url, dst):
    """Unduh via curl: ikuti pengalihan, coba ulang, dan lanjutkan (-C -)
    bila terputus. Lebih dapat diprediksi daripada hf_hub_download, yang
    pada run pertama meninggalkan model.safetensors berukuran 0 byte."""
    import time
    for attempt in range(1, 4):
        size_before = os.path.getsize(dst) if os.path.exists(dst) else 0
        cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "5",
               "--retry-all-errors", "-C", "-", "-sS",
               "-o", dst, url]
        log(f"     [percobaan {attempt}] ukuran awal {size_before} -> {' '.join(cmd[:6])} ...")
        r = subprocess.run(cmd)
        size_after = os.path.getsize(dst) if os.path.exists(dst) else 0
        log(f"     exit={r.returncode} ukuran akhir={size_after}")
        if r.returncode == 0 and size_after > 0:
            return size_after
        time.sleep(5)
    return 0


for f in FILES:
    dst = os.path.join(model_dir, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"   sudah ada {f} ({os.path.getsize(dst)} byte)")
        continue
    if from_dataset:
        # Mount dataset bersifat baca-saja: jangan mencoba mengunduh ke sana.
        log(f"   [WARN] {f} tidak ada di dataset — dilewati")
        continue
    log(f"   mengunduh {f} ...")
    got = curl_get(f"https://huggingface.co/{REPO}/resolve/main/{f}", dst)
    if got == 0:
        if f in ("chat_template.jinja", "generation_config.json"):
            log(f"   [LEWATI] {f} tidak wajib")
            continue
        fatal("gagal unduh", f, "(0 byte setelah 3 percobaan)")
    log(f"   -> {f} OK ({got} byte)")

log("   isi model_dir:", sorted(os.listdir(model_dir)))

# Tolak berkas bobot yang kosong/terpotong — jangan sampai inferensi jalan
# di atas berkas 0 byte (pernah terjadi: unduhan putus tanpa error).
st = os.path.join(model_dir, "model.safetensors")
EXPECTED = 8595477990
if os.path.exists(st):
    got = os.path.getsize(st)
    log(f"   model.safetensors: {got} byte (diharapkan {EXPECTED})")
    if got < 1024:
        fatal("model.safetensors kosong/terpotong — unduhan gagal")
    if got != EXPECTED:
        log("   [WARN] ukuran berbeda dari harapan — lanjut, tapi waspada")
else:
    fatal("model.safetensors tidak ada")

# --------------------------------------------------------------------------
# 4. Sanity hadamard.json (jalur parser yang sama dipakai main.mojo)
# --------------------------------------------------------------------------
log("4. PEMERIKSAAN hadamard.json")
hj = os.path.join(model_dir, "hadamard.json")
if not os.path.exists(hj):
    fatal("hadamard.json tidak ada")
raw = open(hj, encoding="utf-8").read()
log(f"   ukuran {len(raw)} byte")
for key in ("prism.hadamard.block_size", "prism.hadamard.sign_widths",
            "prism.hadamard.sign_values"):
    log(f"   {key}: {'ADA' if key in raw else 'TIDAK ADA'}")
i = raw.find('"prism.hadamard.sign_widths"')
if i >= 0:
    log("   potongan sign_widths:", raw[i:i + 90].replace("\n", " "))

# --------------------------------------------------------------------------
# 5. Tokenisasi prompt
# --------------------------------------------------------------------------
log("5. TOKENISASI PROMPT")
# Prompt default sengaja PANJANG (~110 token): prefill 9 token terlalu pendek
# untuk mengukur throughput, sebab waktunya didominasi overhead tetap.
prompt = os.environ.get("BONSAI_PROMPT", "")
if not prompt:
    prompt = (
        "The capital of France is Paris, and the river Seine flows through it. "
    ) * 8
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    ids = tok.encode(text, add_special_tokens=False)
except Exception as e:
    log("   [WARN] transformers gagal (" + str(e)[:80] + ") — pakai id cadangan")
    # "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" (ChatML Qwen3)
    ids = [248045, 872, 198, 9419, 248046, 198, 248045, 74455, 198]
prompt_tokens = ",".join(str(i) for i in ids)
log("   token ids:", prompt_tokens)

# --------------------------------------------------------------------------
# 6. Jalankan inferensi 2-bit
#    6a. prefill BATCHED (kernel qmm WMMA v2) — jalur cepat.
#    6b. prefill PER-TOKEN (BONSAI_PREFILL_PER_TOKEN=1) — jalur referensi
#        yang loop decode per token. Numeriknya harus cocok; selisih kecil
#        wajar karena urutan akumulasi fp16 berbeda (WMMA vs GEMV).
#    6c. Verifikasi: TOP2 logit + token generasi + timing.
# --------------------------------------------------------------------------
log("6. MENJALANKAN INFERENSI BONSAI-2 (BONSAI_BITS=2)")
log("   GPU:", sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]).strip())

env = dict(os.environ)
env["BONSAI_BITS"] = "2"
env["BONSAI_CUDA_LIB"] = so
env["BONSAI_USE_GPU"] = "1"
# Top-2 logit di batas prefill dicetak untuk kedua run — ini yang dibandingkan
# secara kuantitatif (selisih logit + gap). Tanpa ini kita cuma punya
# kecocokan token, bukan angka.
env["BONSAI_DUMP_TOP2"] = "1"
ld_path = os.pathsep.join(rt_libs + [build_dir, env.get("LD_LIBRARY_PATH", "")])
env["LD_LIBRARY_PATH"] = ld_path

# Pastikan semua pustaka dinamis terpenuhi SEBELUM inferensi berjalan.
# (v2 gagal di sini: exit 127, libKGENCompilerRTShared.so tidak ditemukan.)
log("   LD_LIBRARY_PATH:", ld_path)
subprocess.run(["bash", "-c",
                "LD_LIBRARY_PATH='" + ld_path + "' ldd " + infer
                + " | grep 'not found' || echo '   ldd: semua pustaka terpenuhi'"])

# Kedua run harus memakai prompt + max-tokens yang SAMA, kalau tidak
# perbandingannya tidak bermakna.
MAX_TOKENS = os.environ.get("BONSAI_MAX_TOKENS", "24")

import re  # noqa: E402  (impor di sini supaya langkah 1-5 tetap ringkas)


def run_infer(label, extra_env):
    """Jalankan bonsai_infer sekali, kembalikan ringkasan yang bisa dibandingkan."""
    e = dict(env)
    e.update(extra_env)
    cmd = [
        infer,
        "--model-dir", model_dir,
        "--prompt-tokens", prompt_tokens,
        "--max-tokens", MAX_TOKENS,
        "--gpu",
    ]
    log(f"6x. RUN [{label}] env={extra_env}")
    log("   perintah:", " ".join(cmd))
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    out = p.stdout or ""
    # Cetak baris-baris substantif saja, bukan seluruh log ratusan KB.
    keep = ("[HADAMARD]", "[STEP]", "[GPU]", "[TOP2]", "[GEN]", "[PERF]",
            "[FATAL]", "[ERROR]", "Selesai", "Config:", "[PROF/")
    # [PROF-LAYER] tercetak 64x (sekali per layer) HANYA di run kelima.
    # Dicapkan semuanya tapi diringkas di bagian 6f supaya log tidak meledak.
    if extra_env.get("BONSAI_PROFILE") == "1" and \
            extra_env.get("BONSAI_PREFILL_PER_TOKEN") == "1":
        keep = keep + ("[PROF-LAYER]",)
    # Baris [INT8] dicetak tiap modul (~ratusan) — cukup cetak yang pertama
    # sebagai bukti jalur int8 benar-benar diambil, bukan jatuh ke fp16.
    int8_seen = False
    h2_seen = False
    for line in out.splitlines():
        if "[H2]" in line:
            if h2_seen:
                continue
            h2_seen = True
            print("   " + line + "   (baris ini muncul per-modul, dicetak sekali)")
            continue
        if "[INT8]" in line:
            if int8_seen:
                continue
            int8_seen = True
            print("   " + line + "   (baris ini muncul per-modul, dicetak sekali)")
            continue
        if any(k in line for k in keep):
            print("   " + line)
    if p.returncode != 0:
        log(f"   [RUN {label}] EXIT {p.returncode}")
        # Cetak ekor stdout MENTAH juga: penyaring `keep` di atas bisa
        # menyembunyikan pesan Error yang tidak diawali token yang dikenal,
        # sehingga kegagalan tampak sebagai "EXIT 1" tanpa sebab. Ini pernah
        # terjadi — run gagal di [STEP] cari embed tanpa satu pun baris sebab.
        for line in (p.stdout or "").strip().splitlines()[-25:]:
            print("   stdout:", line)
        tail = (p.stderr or "").strip().splitlines()[-15:]
        if tail:
            for line in tail:
                print("   stderr:", line)
        else:
            print("   stderr: (kosong)")
    return {"label": label, "rc": p.returncode, "raw": out}


def parse_top2(raw):
    """Ambil baris [TOP2] pertama: (id1, v1, id2, v2, gap)."""
    for line in raw.splitlines():
        if "[TOP2]" not in line:
            continue
        m = re.search(
            r"1st:\s*(-?\d+)\s*=\s*(-?[\d.eE+-]+)\s*\|\s*"
            r"2nd:\s*(-?\d+)\s*=\s*(-?[\d.eE+-]+)\s*\|\s*"
            r"gap:\s*(-?[\d.eE+-]+)",
            line,
        )
        if m:
            return tuple(m.groups())
    return None


def parse_gens(raw):
    out = []
    for l in raw.splitlines():
        if "[GEN] token id:" not in l:
            continue
        # Formatnya dua macam:
        #   ">> [GEN] token id: 248068"                       (batas prefill)
        #   ">> [GEN] token id: 198 | 123.920605 ms | ..."    (langkah decode)
        body = l.split("token id:", 1)[1]
        out.append(int(body.split("|", 1)[0].strip()))
    return out


def parse_perf(raw):
    pf = next((l.strip() for l in raw.splitlines() if "[PERF] prefill" in l), None)
    dc = next((l.strip() for l in raw.splitlines()
               if "[PERF] rata-rata decode" in l), None)
    return pf, dc


res_a = run_infer("prefill batched (qmm WMMA v2)", {})
res_b = run_infer("prefill per-token (referensi)",
                  {"BONSAI_PREFILL_PER_TOKEN": "1"})
res_c = run_infer("prefill int8 (tensor core int8 W2A8)",
                  {"BONSAI_PREFILL_INT8": "1"})
# Run keempat: HANYA untuk mengukur. BONSAI_PROFILE=1 menyalakan sinkronisasi
# per layer (65x/token) supaya waktu GPU benar-benar ter-atribusi ke GDN /
# ATTN / LM head — tanpa itu acc_gdn/acc_attn cuma waktu submit CPU dan
# seluruh kerja GPU token terserap di acc_lm (lihat komentar main.mojo:1970).
# Konsekuensi: angka tok/s run ini TIDAK bisa dibandingkan dengan A/B/C,
# karena pipa asinkron sengaja dimatikan. Yang dipakai hanya baris [PROF/].
res_d = run_infer("profil per-subsistem (BONSAI_PROFILE=1)",
                  {"BONSAI_PROFILE": "1"})
# Run kelima: memancing baris [PROF-LAYER] yang sudah ada di layer.mojo:413.
# Baris itu hanya tercetak saat `pos == 0 and BONSAI_PROFILE=1` — yaitu token
# PERTAMA pada jalur prefill per-token. Isinya pecahan per fase dalam satu
# layer (prenorm / step / postnorm / mlp / resid2), yang TIDAK ada di
# [PROF/SPLIT]: tanpa ini kita cuma tahu "layer GDN 52 ms" tanpa tahu apakah
# itu bobot proyeksi, MLP, atau kernel elementwise GDN.
res_e = run_infer("profil per-fase layer (per-token + BONSAI_PROFILE=1)",
                  {"BONSAI_PREFILL_PER_TOKEN": "1", "BONSAI_PROFILE": "1"})
# Run keenam: decode memakai jalur cepat PRMT+HFMA2 (BONSAI_DECODE_H2=1).
# GEMV decode menyerap ~82% waktu decode menurut nvprof, jadi inilah yang
# diserang. Jalur ini mengubah urutan penjumlahan internal, karena itu ia
# diverifikasi dua kali: token stream harus identik dengan run A (jalur
# proven) DAN dengan BASELINE yang direkam sebelum perubahan ada.
res_f = run_infer("decode jalur cepat half2 (BONSAI_DECODE_H2=1)",
                  {"BONSAI_DECODE_H2": "1"})
# Run ketujuh: jalur cepat + profil per-subsistem. Tujuannya menjawab
# "setelah GEMV dipercepat 1,38x, ke mana sisa ~50 ms/token pergi?" —
# tanpa ini kita cuma tahu totalnya turun, bukan subsistem mana yang masih
# menahan. Angka tok/s run ini juga TIDAK bisa dibandingkan (pipa mati).
res_g = run_infer("profil per-subsistem + jalur cepat half2",
                  {"BONSAI_DECODE_H2": "1", "BONSAI_PROFILE": "1"})
# Run kedelapan: varian LEBAR (2 grup per lane, 4x uint4 diterbitkan sebelum
# dihitung). Ini eksperimen yang menjawab "GEMV decode ~145 GB/s itu sudah
# mentok DRAM, atau kekurangan byte yang melayang?" — metrik perangkat keras
# tidak tersedia di container (ncu exit 1, nvprof --metrics menolak CC>=7.5),
# jadi A/B ujung-ke-ujung satu-satunya jalan.
res_h = run_infer("decode jalur cepat half2b (BONSAI_DECODE_H2=2)",
                  {"BONSAI_DECODE_H2": "2"})

# Rekurensi GDN varian lebar (8 thread/baris). Desainnya dipilih oleh probe
# CUDA di bagian 10 (1,51x lebih cepat pada rejim 48 luncuran berurutan).
# TIDAK bit-exact, maka ia punya run sendiri dan diadu langsung ke run F.
res_i = run_infer("rekurensi GDN varian lebar (BONSAI_DECODE_H2=1, BONSAI_GDN_WIDE=1)",
                  {"BONSAI_DECODE_H2": "1", "BONSAI_GDN_WIDE": "1"})

# --------------------------------------------------------------------------
# 6c. Verifikasi: bandingkan prefill batched vs per-token
# --------------------------------------------------------------------------
log("6c. VERIFIKASI prefill batched vs per-token (bits=2)")
ok = True
if res_a["rc"] != 0 or res_b["rc"] != 0:
    log("   [GAGAL] salah satu run non-zero exit:",
        res_a["rc"], res_b["rc"])
    ok = False

t2a = parse_top2(res_a["raw"])
t2b = parse_top2(res_b["raw"])
ga = parse_gens(res_a["raw"])
gb = parse_gens(res_b["raw"])
pa, da = parse_perf(res_a["raw"])
pb, db = parse_perf(res_b["raw"])

log("   [A batched ] TOP2:", t2a)
log("   [B per-token] TOP2:", t2b)
log("   [A batched ] prefill:", pa)
log("   [B per-token] prefill:", pb)
log("   [A batched ] decode :", da)
log("   [B per-token] decode :", db)
log(f"   [A batched ] {len(ga)} token generasi:", ga)
log(f"   [B per-token] {len(gb)} token generasi:", gb)

if t2a is None or t2b is None:
    # TOP2 adalah sinyal kuantitatif, tapi kecocokan token sudah menjadi
    # bukti utama. Jangan gagalkan seluruh uji hanya karena baris ini hilang.
    log("   [WARN] baris [TOP2] tidak ditemukan di salah satu run — "
        "verifikasi mengandalkan kecocokan token generasi")
else:
    id1a, v1a, id2a, v2a, ga_p = t2a
    id1b, v1b, id2b, v2b, gb_p = t2b
    v1a, v1b = float(v1a), float(v1b)
    gapa, gapb = float(ga_p), float(gb_p)
    d_top1 = abs(v1a - v1b)
    same_token = id1a == id1b
    # Aturan (sama dengan jalur 1-bit di deploy_on_kaggle.sh):
    #   - token top-1 sama + selisih logit kecil  -> cocok
    #   - token flip tetapi KEDUA gap kecil (<0,05)-> wajar (urutan
    #     akumulasi fp16 WMMA vs GEMV berbeda)
    #   - gap besar atau selisih logit besar        -> bug layout dicurigai
    log(f"   token top-1: A={id1a} B={id1b} -> "
        + ("SAMA" if same_token else "BEDA"))
    log(f"   selisih |logit top-1| = {d_top1:.6f}")
    log(f"   gap A = {gapa:.6f} | gap B = {gapb:.6f}")
    if not same_token:
        if gapa < 0.05 and gapb < 0.05:
            log("   [Wajar] token flip tapi kedua gap < 0,05 — urutan "
                "akumulasi fp16 WMMA vs GEMV berbeda")
        else:
            log("   [GAGAL] token flip dengan gap besar — indikasi bug "
                "layout di kernel qmm 2-bit")
            ok = False
    if d_top1 > 0.05:
        log("   [GAGAL] selisih logit top-1 > 0,05 — numerik tidak cocok")
        ok = False

# Token generasi: identik idealnya. Setiap divergensi berarti logit prefill
# berbeda cukup untuk mengubah argmax di suatu tempat — anggap gagal dan
# selidiki. (Kedua jalur memakai fp16, jadi tidak ada alasan kuantisasi
# tambahan yang membolehkan perbedaan.)
common = min(len(ga), len(gb))
first_diff = next((i for i in range(common) if ga[i] != gb[i]), None)
if first_diff is None:
    log(f"   [OK] seluruh {common} token generasi IDENTIK")
else:
    log(f"   [GAGAL] token generasi mulai berbeda di indeks {first_diff}: "
        f"A={ga[first_diff]} vs B={gb[first_diff]}")
    ok = False

log("   VERDIK:", "LULUS — prefill batched setara numerik dgn per-token"
    if ok else "TIDAK LULUS — selidiki kernel qmm 2-bit")

# --------------------------------------------------------------------------
# 6d. Verifikasi jalur INT8 (tensor core int8, W2A8) vs referensi per-token.
#     Berbeda dari 6c: jalur ini MENGGANTI basis numerik (aktivasi dikuantisasi
#     ke int8), jadi selisih logit top-1 BUKAN nol secara harapan. Aturannya:
#       - token top-1 harus sama (syarat mutlak)
#       - selisih logit harus jauh lebih kecil dari gap runner-up, agar
#         jelas keputusan argmax tidak koin-balik
#       - token generasi harus identik (decode sama, hanya prefill beda)
#     Lulus 6d tidak menggantikan 6c — 6c wajib lulus untuk fp16 path.
# --------------------------------------------------------------------------
ok_int8 = True
log("6d. VERIFIKASI prefill int8 (W2A8) vs per-token (bits=2)")
if res_c["rc"] != 0:
    log("   [GAGAL] run int8 non-zero exit:", res_c["rc"])
    ok_int8 = False

t2c = parse_top2(res_c["raw"])
gc = parse_gens(res_c["raw"])
pc, dc = parse_perf(res_c["raw"])
log("   [C int8    ] TOP2:", t2c)
log("   [C int8    ] prefill:", pc)
log("   [C int8    ] decode :", dc)
log(f"   [C int8    ] {len(gc)} token generasi:", gc)

if t2c is None or t2b is None:
    log("   [WARN] baris [TOP2] tidak ditemukan — verifikasi int8 mengandalkan "
        "kecocokan token generasi")
else:
    id1c, v1c, id2c, v2c, gc_p = t2c
    v1c = float(v1c)
    gapc = float(gc_p)
    d_top1_c = abs(v1c - v1b)
    same_token_c = id1c == id1b
    log(f"   token top-1: C={id1c} B={id1b} -> "
        + ("SAMA" if same_token_c else "BEDA"))
    log(f"   selisih |logit top-1| C vs B = {d_top1_c:.6f}")
    log(f"   gap C = {gapc:.6f} | gap B = {gapb:.6f}")
    if not same_token_c:
        log("   [GAGAL] token top-1 int8 berbeda dari referensi per-token")
        ok_int8 = False
    else:
        # Syarat: selisih logit harus < 25% gap runner-up. Gap di batas
        # prefill biasanya besar (puluhan), jadi ini longgar untuk selisih
        # kuantisasi int8 (~1% relatif) tapi tetap menangkap bug layout
        # yang menghasilkan selisih sebanding dengan logit itu sendiri.
        if d_top1_c > 0.25 * max(gapc, 1.0):
            log(f"   [GAGAL] selisih logit {d_top1_c:.6f} > 25% gap "
                f"({0.25 * max(gapc, 1.0):.6f}) — terlalu besar untuk "
                "kuantisasi int8, selidiki kernel int8")
            ok_int8 = False
        else:
            log("   [OK] selisih logit dalam batas kuantisasi int8")

common_c = min(len(gc), len(gb))
first_diff_c = next((i for i in range(common_c) if gc[i] != gb[i]), None)
if first_diff_c is None:
    log(f"   [OK] seluruh {common_c} token generasi IDENTIK dengan referensi")
else:
    log(f"   [GAGAL] token generasi mulai berbeda di indeks {first_diff_c}: "
        f"C={gc[first_diff_c]} vs B={gb[first_diff_c]}")
    ok_int8 = False

log("   VERDIK int8:", "LULUS — jalur int8 setara dalam batas kuantisasinya"
    if ok_int8 else "TIDAK LULUS — selidiki kernel int8")

# Perbandingan throughput: ini alasan utama jalur int8 dibuat.
def _tok_s(perf_line):
    m = re.search(r"([\d.eE+-]+)\s*tok/s", perf_line or "")
    return float(m.group(1)) if m else None


ta, tc_t = _tok_s(pa), _tok_s(pc)
if ta and tc_t:
    log(f"   throughput prefill: WMMA fp16 = {ta:.2f} tok/s | "
        f"int8 = {tc_t:.2f} tok/s | rasio = {tc_t / ta:.3f}x")
    if tc_t <= ta * 1.02:
        log("   [PERHATIAN] int8 tidak lebih cepat dari fp16 — kemungkinan "
            "jalur int8 tidak diambil (cek baris [INT8] di atas) atau "
            "dekuantisasi/kuantisasi menjadi kendala")
else:
    log("   [PERHATIAN] tok/s tidak terurai dari salah satu run")

# --------------------------------------------------------------------------
# 6e. Pembagian waktu decode per subsistem (BONSAI_PROFILE=1).
#     Ini TUJUAN utama run keempat: sebelum menyentuh kernel apa pun, harus
#     diketahui dulu apakah ~15 ms non-GEMV per token itu benar-benar ada dan
#     di subsistem mana ia berada. Mengoptimasi tanpa ini = menebak.
# --------------------------------------------------------------------------
log("6e. PROFIL DECODE PER SUBSISTEM (BONSAI_PROFILE=1)")
if res_d["rc"] != 0:
    log("   [GAGAL] run profil non-zero exit:", res_d["rc"])
else:
    prof_lines = [l.strip() for l in res_d["raw"].splitlines()
                  if "[PROF/" in l]
    if not prof_lines:
        log("   [WARN] tidak ada baris [PROF/] — BONSAI_PROFILE tidak terbaca")
    for l in prof_lines:
        log("   " + l)
    log("   CATATAN: angka di atas diukur DENGAN sinkronisasi per layer, "
        "jadi jumlahnya lebih besar dari ms/token jalur normal "
        "(pipa asinkron dimatikan). Yang bermakna adalah PEMBAGIANnya, "
        "bukan nilai mutlaknya.")

# --------------------------------------------------------------------------
# 6f. Pecahan per FASE di dalam satu layer (baris [PROF-LAYER]).
#     Ini yang menjawab pertanyaan sebenarnya: dari ~1,09 ms per layer GDN,
#     berapa yang proyeksi bobot, berapa yang MLP, berapa yang kernel
#     elementwise GDN (konv/norm/rekurensi/gate).
# --------------------------------------------------------------------------
log("6f. PROFIL PER FASE DALAM LAYER ([PROF-LAYER], pos=0)")
if res_e["rc"] != 0:
    log("   [GAGAL] run per-fase non-zero exit:", res_e["rc"])
else:
    lay = []
    for l in res_e["raw"].splitlines():
        if "[PROF-LAYER]" not in l:
            continue
        m = re.search(
            r"prenorm=\s*([\d.eE+-]+)\s*us\s+step=\s*([\d.eE+-]+)\s*us\s+"
            r"postnorm\(fused\)=\s*([\d.eE+-]+)\s*us\s+mlp=\s*([\d.eE+-]+)\s*us\s+"
            r"resid2=\s*([\d.eE+-]+)\s*us", l)
        if m:
            lay.append([float(g) for g in m.groups()])
    log("   baris [PROF-LAYER] terbaca:", len(lay))
    if len(lay) < 8:
        log("   [WARN] terlalu sedikit — kemungkinan `pos` tidak pernah 0 "
            "di jalur prefill per-token")
    else:
        # layer.mojo:226 "Gated Attention (jika interval ke-4)" dan dok
        # AttentionKVCache "Layer 3, 7, 11, ..." -> indeks i%4==3 = attention.
        gdn = [r for i, r in enumerate(lay) if i % 4 != 3]
        attn = [r for i, r in enumerate(lay) if i % 4 == 3]
        names = ["prenorm", "step", "postnorm", "mlp", "resid2"]

        def ringkas(rows, label):
            if not rows:
                return
            tot = [0.0] * 5
            for r in rows:
                for j in range(5):
                    tot[j] += r[j]
            n = len(rows)
            log(f"   [{label}] {n} layer | rata-rata per layer (us):")
            for j, nm in enumerate(names):
                log(f"      {nm:9s} {tot[j] / n:10.3f} us   "
                    f"(total {tot[j] / 1000.0:8.3f} ms)")
            log(f"      {'JUMLAH':9s} {sum(tot) / n:10.3f} us   "
                f"(total {sum(tot) / 1000.0:8.3f} ms)")

        ringkas(gdn, "GDN ")
        ringkas(attn, "ATTN")
        log("   CATATAN: 'step' mencakup SEMUA proyeksi sublayer "
            "(in_proj+out_proj utk GDN; q/k/v/o_proj utk attention) "
            "BESERTA kernel elementwise GDN. 'mlp' = gate_up+swiglu+down. "
            "Satuan Mikrodetik.")

# --------------------------------------------------------------------------
# 6g. Verifikasi decode jalur cepat PRMT+HFMA2 (BONSAI_DECODE_H2=1).
# --------------------------------------------------------------------------
# Aliran token yang DIREKAM sebelum jalur half2 ada (run v15-v22, prompt
# sama, greedy). Dipakai sebagai patokan mutlak: jalur baru boleh lebih
# cepat, tapi tidak boleh menghasilkan token yang berbeda.
BASELINE = [248068, 198, 760, 1156, 682, 11173, 279, 1788, 11316, 5081,
            2942, 25, 328, 760, 6511, 314, 9338, 369, 11751, 11, 321, 279,
            14367, 181474]


def _ms_pt(line):
    """Ambil ms/token dari baris '[PERF] rata-rata decode: X ms/token'."""
    m = re.search(r"rata-rata decode:\s*([\d.eE+-]+)\s*ms/token", line or "")
    return float(m.group(1)) if m else None


ok_h2 = True
log("6g. VERIFIKASI decode jalur cepat PRMT+HFMA2 (BONSAI_DECODE_H2=1)")
if res_f["rc"] != 0:
    log("   [GAGAL] run h2 non-zero exit:", res_f["rc"])
    ok_h2 = False

gf = parse_gens(res_f["raw"])
_, df = parse_perf(res_f["raw"])
_, da = parse_perf(res_a["raw"])
log("   [F h2      ] decode:", df)
log("   [A batched ] decode:", da)
log(f"   [F h2      ] {len(gf)} token generasi:", gf)

mf, ma = _ms_pt(df), _ms_pt(da)
if mf and ma:
    log(f"   ms/token: jalur lama = {ma:.3f} | half2 = {mf:.3f} | "
        f" percepatan = {ma / mf:.3f}x")
    if mf >= ma:
        log("   [CATATAN] jalur half2 TIDAK lebih cepat — berarti GEMV "
            "decode bukan terhambat instruksi, melainkan DRAM")
else:
    log("   [PERHATIAN] ms/token tidak terurai")

if gf == ga:
    log(f"   [OK] ke-{len(gf)} token identik dengan jalur proven (run A)")
else:
    d = next((i for i in range(min(len(gf), len(ga))) if gf[i] != ga[i]), None)
    log(f"   [GAGAL] token generasi berbeda dari jalur proven di indeks {d}")
    ok_h2 = False

if gf == BASELINE:
    log("   [OK] token identik dengan BASELINE pra-perubahan")
else:
    d = next((i for i in range(min(len(gf), len(BASELINE)))
              if gf[i] != BASELINE[i]), None)
    log(f"   [GAGAL] token menyimpang dari BASELINE di indeks {d}: "
        f"{gf[d] if d is not None and d < len(gf) else '?'} vs "
        f"{BASELINE[d] if d is not None else '?'}")
    ok_h2 = False

log("   VERDIK h2:", "LULUS" if ok_h2 else "TIDAK LULUS")

# CATATAN: `ok = ok and ok_int8 and ok_h2` dan VERDIK TOTAL sengaja
# DITARUH di akhir (setelah 6i), karena 6i ikut menguji varian h2b dan
# dapat mengubah ok_h2. Kalau verdi total dicetak sebelum 6i, run yang
# h2b-nya menghasilkan token salah tetap tercetak LULUS.

# --------------------------------------------------------------------------
# 6h. Ke mana sisa waktu pergi SETELAH jalur cepat aktif.
#     Membandingkan [PROF/SPLIT] jalur lama (run D) dengan jalur half2
#     (run G). Keduanya memakai sinkronisasi per layer yang sama, jadi
#     selisihnya murni efek kernel GEMV.
# --------------------------------------------------------------------------
log("6h. PEMBAGIAN WAKTU SETELAH JALUR CEPAT (run D vs run G)")


def parse_split(raw):
    """Ambil (gdn, attn, lm) dari baris [PROF/SPLIT] per token."""
    for l in raw.splitlines():
        if "[PROF/SPLIT] per token" in l:
            m = re.search(r"GDN:\s*([\d.]+).*ATTN:\s*([\d.]+).*LM_HEAD\+argmax:\s*([\d.]+)", l)
            if m:
                return (float(m.group(1)), float(m.group(2)), float(m.group(3)))
    return None


sd, sg = parse_split(res_d["raw"]), parse_split(res_g["raw"])
if sd and sg:
    for nama, i in (("GDN  (48 layer, +MLP)", 0), ("ATTN (16 layer, +MLP)", 1),
                    ("LM_HEAD+argmax", 2)):
        log(f"   {nama:24s} lama = {sd[i]:7.3f} ms | half2 = {sg[i]:7.3f} ms"
            f" | selisih = {sd[i] - sg[i]:+7.3f} ms "
            f"({sd[i] / sg[i] if sg[i] else 0:.3f}x)")
    log(f"   {'JUMLAH':24s} lama = {sum(sd):7.3f} ms | half2 = {sum(sg):7.3f} ms"
        f" | selisih = {sum(sd) - sum(sg):+7.3f} ms")
    log("   CATATAN: diukur DENGAN sinkronisasi per layer (pipa mati), jadi "
        "nilai mutlaknya lebih besar dari ms/token jalur normal; yang "
        "bermakna adalah perbandingan D vs G.")
else:
    log("   [PERHATIAN] [PROF/SPLIT] tidak terbaca dari run D atau run G")

# --------------------------------------------------------------------------
# 6i. A/B varian h2 (1 grup/lane) vs h2b (2 grup/lane, 4x uint4 in flight).
#     Menjawab: apakah GEMV decode masih bisa ditarik lebih jauh dengan
#     memperbanyak byte yang melayang, atau sudah mentok DRAM?
# --------------------------------------------------------------------------
log("6i. A/B VARIAN LEBAR: h2 (run F) vs h2b (run H)")
gh = parse_gens(res_h["raw"])
_, dh = parse_perf(res_h["raw"])
log("   [F h2  1 grup ] decode:", df)
log("   [H h2b 2 grup ] decode:", dh)
mf2, mh2 = _ms_pt(df), _ms_pt(dh)
if mf2 and mh2:
    log(f"   ms/token: h2 = {mf2:.3f} | h2b = {mh2:.3f} | "
        f"rasio h2/h2b = {mf2 / mh2:.3f}x "
        f"({'h2b LEBIH CEPAT' if mh2 < mf2 else 'h2b TIDAK lebih cepat'})")
else:
    log("   [PERHATIAN] ms/token tidak terurai")
if gh == gf:
    log(f"   [OK] ke-{len(gh)} token identik dengan varian h2")
else:
    d = next((i for i in range(min(len(gh), len(gf))) if gh[i] != gf[i]), None)
    log(f"   [GAGAL] token h2b menyimpang dari h2 di indeks {d}")
    ok_h2 = False
if gh == BASELINE:
    log("   [OK] token h2b identik dengan BASELINE pra-perubahan")
else:
    d = next((i for i in range(min(len(gh), len(BASELINE)))
              if gh[i] != BASELINE[i]), None)
    log(f"   [GAGAL] token h2b menyimpang dari BASELINE di indeks {d}")
    ok_h2 = False

# --------------------------------------------------------------------------
# 6j. A/B REKURENSI GDN: jalur lama (run F) vs varian lebar (run I)
#     Varian lebar memakai identitas aljabar out = decay*B + delta*kq, jadi
#     TIDAK bit-exact. Yang diuji: apakah 24 tokennya masih sama persis.
# --------------------------------------------------------------------------
log("6j. A/B REKURENSI GDN: lama (run F) vs varian lebar (run I)")
gi = parse_gens(res_i["raw"])
_, di = parse_perf(res_i["raw"])
log("   [F rekurensi lama   ] decode:", df)
log("   [I rekurensi lebar  ] decode:", di)
mf3, mi3 = _ms_pt(df), _ms_pt(di)
ok_wide = True
if mf3 and mi3:
    log(f"   ms/token: lama = {mf3:.3f} | lebar = {mi3:.3f} | "
        f"percepatan = {mf3 / mi3:.3f}x "
        f"({'LEBIH CEPAT' if mi3 < mf3 else 'TIDAK lebih cepat'})")
else:
    log("   [PERHATIAN] ms/token tidak terurai")
    ok_wide = False
if gi == gf:
    log(f"   [OK] ke-{len(gi)} token identik dengan rekurensi lama")
else:
    d = next((i for i in range(min(len(gi), len(gf))) if gi[i] != gf[i]), None)
    log(f"   [GAGAL] token varian lebar menyimpang di indeks {d}")
    ok_wide = False
if gi == BASELINE:
    log("   [OK] token varian lebar identik dengan BASELINE pra-perubahan")
else:
    d = next((i for i in range(min(len(gi), len(BASELINE)))
              if gi[i] != BASELINE[i]), None)
    log(f"   [GAGAL] token varian lebar menyimpang dari BASELINE di indeks {d}")
    ok_wide = False

ok = ok and ok_int8 and ok_h2 and ok_wide
log("   VERDIK TOTAL:", "LULUS" if ok else "TIDAK LULUS")

# --------------------------------------------------------------------------
# 7. PROFIL PER-KERNEL (nvprof / nsys).
#    [PROF-LAYER] menggabungkan semua kernel jadi 5 fase dan TIAP fase
#    diakhiri ctx.synchronize() (~31-80 us), jadi tidak bisa memisahkan
#    kernel elementwise GDN yang kecil dari overhead itu. Profiler CUDA
#    mengukur TIAP kernel langsung di GPU — tanpa menyentuh kode Mojo.
#    Cukup 3 token: 3 x ~1200 kernel sudah mewakili, dan outputnya
#    tetap terbaca.
# --------------------------------------------------------------------------
log("7. PROFIL PER-KERNEL (nvprof / nsys)")
prof_tool = ""
for cand in ("nsys", "nvprof", "ncu"):
    p = sh(["bash", "-lc", "command -v " + cand]).strip()
    log(f"   {cand}: {p if p else 'TIDAK ADA'}")
    if p and not prof_tool:
        prof_tool = cand

if not prof_tool:
    log("   [WARN] tidak ada profiler CUDA di container — pecahan per-kernel "
        "tidak bisa diukur dari luar; butuh instrumentasi di kode Mojo")
else:
    log("   memakai:", prof_tool)
    def prof_run(env_extra, label):
        log("   === PROFIL:", label, "===")
        e = dict(env)
        e.update(env_extra)
        # WAJIB: prompt 1 token membuat model langsung mengeluarkan EOS, dan
        # main.mojo menghentikan generasi di situ — run ini pernah hanya
        # menjalankan 2 langkah decode sementara pembagi tetap 11, sehingga
        # semua angka "per token" salah 5,5x. Matikan henti-di-EOS.
        e["BONSAI_STOP_IDS"] = "none"
        # PROMPT 1 TOKEN — ini kunci agar agregasi murni decode. Jalur prefill
        # BATCHED tetap mengulang kernel stateful GDN (conv/qnorm/knorm/
        # rekurensi/gate) sekali PER TOKEN dalam loop sekuensial, jadi dengan
        # prompt 129 token hasilnya didominasi prefill: pernah menghasilkan
        # 35.185 baris yang ~90%-nya prefill. Dengan 1 token, jumlah launch
        # per nama = 11 langkah decode (+1 langkah prefill).
        cmd = [infer, "--model-dir", model_dir,
               "--prompt-tokens", "248045", "--max-tokens", "12", "--gpu"]
        if prof_tool == "nsys":
            pc = ["nsys", "profile", "--stats=true", "--force-overwrite", "true",
                  "-o", os.path.join(base, "bonsai_prof")] + cmd
        elif prof_tool == "nvprof":
            pc = ["nvprof", "--print-gpu-trace", "--csv"] + cmd
        else:
            pc = ["ncu", "--target-processes", "all", "--csv",
                  "--launch-count", "400"] + cmd
        log("   perintah:", " ".join(pc[:6]), "...")
        t0 = __import__("time").time()
        pr = subprocess.run(pc, env=e, capture_output=True, text=True)
        log(f"   exit={pr.returncode} ({__import__('time').time() - t0:.1f} s)")
        blob = (pr.stdout or "") + "\n" + (pr.stderr or "")
        # Ambil jumlah langkah decode yang BENAR-BENAR dijalankan dari baris
        # [PERF] — jangan mengasumsikan max_tokens-1 (generasi bisa berhenti
        # di EOS, dan itu pernah membuat pembagi salah 5,5x).
        m_dec = re.search(r"decode\s+(\d+)\s+token", blob)
        n_dec = int(m_dec.group(1)) if m_dec else 0
        if n_dec <= 0:
            n_dec = 11
            log("   [WARN] jumlah langkah decode tidak terbaca, pakai 11")
        log("   langkah decode yang benar-benar dijalankan:", n_dec)
        if prof_tool == "nsys":
            # Ambil bagian "CUDA Kernel Statistics" saja (tabel per-kernel).
            lines = blob.splitlines()
            i0 = next((i for i, l in enumerate(lines)
                       if "CUDA Kernel Statistics" in l), None)
            if i0 is None:
                i0 = next((i for i, l in enumerate(lines)
                           if "Kernel Name" in l or "Kernel" in l), None)
            if i0 is None:
                log("   [WARN] tabel kernel tidak ditemukan; ekor output:")
                for l in lines[-25:]:
                    print("   " + l)
            else:
                for l in lines[i0:i0 + 40]:
                    print("   " + l)
        else:
            # nvprof --csv. Baris kepala diawali banner "==PID== NVPROF ...",
            # jadi header CSV HARUS dicari, bukan diambil dari baris pertama
            # (kepala[0] pernah berisi perintah lengkap yang ikut punya koma
            # karena --prompt-tokens dipisah koma).
            import csv as _csv
            lines = blob.splitlines()
            hdr_i = next((i for i, l in enumerate(lines)
                          if "Start" in l and "Duration" in l
                          and l.count(",") >= 5), None)
            log("   indeks header CSV:", hdr_i)
            if hdr_i is None:
                log("   [WARN] header CSV tidak ketemu; 25 baris pertama:")
                for l in lines[:25]:
                    print("   " + l[:150])
            else:
                log("   header:", lines[hdr_i][:200])
                for l in lines[hdr_i + 1:hdr_i + 4]:
                    print("   contoh:", l[:200])
                rows = [r for r in _csv.reader(lines[hdr_i:]) if len(r) > 3]
                log("   baris data:", len(rows))
                # PENTING (pernah salah dua kali):
                #  * baris ke-(hdr_i+1) adalah baris SATUAN ("s,ms,...,KB,GB/s"),
                #    bukan data — harus dilewati;
                #  * Duration ada di kolom 1 dan TANPA sufiks; satuannya
                #    milidetik (dari baris satuan), jadi x1000 -> mikrodetik;
                #  * nama kernel ada di kolom 19, BUKAN kolom terakhir (kolom
                #    terakhir berisi id numerik yang tidak berkepentingan).
                agg = {}
                for r in rows[2:]:
                    if len(r) < 19:
                        continue
                    # Kolom 18 = "Name". Kolom 19 BUKAN nama: itu id numerik
                    # per-launch (terlihat sebagai "2644", "30", ...). Memakai
                    # kolom 19 pernah membuat agregasi menghasilkan 35.183 "nama"
                    # unik yang semuanya angka — setiap launch jadi nama sendiri.
                    name = r[18].strip()
                    if not name or name.lower() == "name":
                        continue
                    try:
                        dur = float(r[1].strip()) * 1000.0   # ms -> us
                    except ValueError:
                        continue
                    # Kolom 2..7 = Grid X/Y/Z, Block X/Y/Z — sidik jari untuk
                    # mengenali kernel Mojo yang namanya cuma hash.
                    grid = "/".join(r[2:5]).strip(",") or "-"
                    blk = "/".join(r[5:8]).strip(",") or "-"
                    a = agg.setdefault(name, [0, 0.0, "", ""])
                    a[0] += 1
                    a[1] += dur
                    a[2] = (grid + " blk " + blk)[:34]
                    a[3] = r[8].strip() if len(r) > 8 else ""   # registers
                log("   nama kernel berbeda:", len(agg))
                if not agg:
                    log("   [WARN] agregasi kosong — format tidak dikenali")
                    for l in lines[hdr_i + 1:hdr_i + 6]:
                        print("   mentah:", l[:200])
                top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:30]
                log("   30 kernel termahal. 'per token' = total/11. "
                    "PERHATIAN: nvprof menambah overhead per launch, jadi kernel "
                    "kecil terukur jauh lebih lambat dari aslinya — jumlah TOTAL "
                    "yang dipakai untuk menilai, bukan nilai mutlaknya.")
                for nm, (cnt, tot, gb, regs) in top:
                    print(f"      {tot / n_dec:9.1f} us/token {tot:11.1f} us "
                          f"x{cnt:6d} ~{cnt / n_dec:6.1f}/token "
                          f"grid {gb:34s} reg {regs:>4s}  {nm[:52]}")

                # Tabel kedua: agregasi per (nama, grid, block). SEMUA GEMV
                # memakai kernel yang sama (qmv_vec_q2t_kernel) dan hanya bisa
                # dibedakan dari grid-nya — tanpa ini kita cuma tahu "GEMV 43 ms"
                # tanpa tahu GEMV MANA yang lambat (N=5120 vs N=34816).
                agg2 = {}
                for r in rows[2:]:
                    if len(r) < 19:
                        continue
                    name = r[18].strip()
                    if not name or name.lower() == "name":
                        continue
                    try:
                        dur = float(r[1].strip()) * 1000.0
                    except ValueError:
                        continue
                    grid = "/".join(x for x in r[2:5]).strip(",") or "-"
                    blk = "/".join(x for x in r[5:8]).strip(",") or "-"
                    key = (name[:44], grid, blk)
                    a = agg2.setdefault(key, [0, 0.0])
                    a[0] += 1
                    a[1] += dur
                top2 = sorted(agg2.items(), key=lambda kv: -kv[1][1])[:30]
                log("   30 (nama,grid,block) termahal — untuk membedakan GEMV "
                    "per bentuk:")
                for (nm, g, b), (cnt, tot) in top2:
                    print(f"      {tot / n_dec:9.1f} us/token {tot:11.1f} us "
                          f"x{cnt:6d} ~{cnt / n_dec:6.1f}/token "
                          f"grid {g:14s} blk {b:10s}  {nm}")

                # ---- CELAH ANTAR-KERNEL: berapa lama GPU menganggur? ----
                # nvprof mencatat waktu MULAI tiap kernel (kolom 0, detik) dan
                # lamanya (kolom 1, ms). Selisih antara saat kernel ke-i
                # selesai dan kernel ke-i+1 mulai pada stream yang sama itu
                # waktu GPU menganggur: kirim dari CPU, sinkronisasi, atau
                # tunggu. Ini menjawab LANGSUNG apakah CUDA Graphs (yang
                # menghapus biaya kirim per peluncuran) akan menolong. Kalau
                # celahnya mendekati nol, GPU sudah padat dan CUDA Graphs
                # tidak akan memberi apa-apa.
                ev = []
                for r in rows[2:]:
                    if len(r) < 19:
                        continue
                    try:
                        st = float(r[0].strip())
                        du = float(r[1].strip())
                    except ValueError:
                        continue
                    nm = r[18].strip()
                    if not nm:
                        continue
                    ev.append((st, du, nm, r[17].strip() if len(r) > 17 else ""))
                ev.sort(key=lambda x: x[0])
                # PENTING: pemuatan bobot di awal (1369 salinan HtoD) ikut
                # terprofil dan celah-celahnya MENDOMINASI — itu biaya CPU
                # sekali jalan, sama sekali bukan perilaku decode. Buang
                # semua kejadian sampai salinan HtoD terakhir; sisanya baru
                # keadaan mapan yang mau diukur.
                i0 = 0
                for i, x in enumerate(ev):
                    if "memcpy HtoD" in x[2]:
                        i0 = i + 1
                ev = ev[i0:]
                tot_du = sum(x[1] for x in ev) * 1000.0     # ms -> us
                gaps = []
                for i in range(len(ev) - 1):
                    st, du, nm, sd = ev[i]
                    st2, _du2, _nm2, sd2 = ev[i + 1]
                    if sd != sd2:
                        continue
                    g = (st2 - (st + du * 1e-3)) * 1e6      # detik -> us
                    if g > 0:
                        gaps.append((g, nm))
                tot_gap = sum(g for g, _ in gaps)
                if tot_du > 0:
                    log("   [%s] MAPAN (sesudah pemuatan bobot): %d kejadian | "
                        "waktu kernel %.1f us/token | celah %.1f us/token "
                        "(%.2f%% dari waktu kernel)"
                        % (label, len(ev), tot_du / n_dec, tot_gap / n_dec,
                           100.0 * tot_gap / tot_du))
                    nbig = sum(1 for g, _ in gaps if g > 20.0)
                    log("      celah >20 us: %d buah (%.1f/token) | "
                        "terbesar %.1f us | rata-rata %.2f us"
                        % (nbig, nbig / n_dec,
                           max((g for g, _ in gaps), default=0.0),
                           tot_gap / max(len(gaps), 1)))
                    gs = sorted(gaps, reverse=True)[:8]
                    log("      8 celah terbesar — kernel SEBELUM celah itu:")
                    for g, nm in gs:
                        print(f"         {g:9.1f} us  sesudah {nm[:56]}")

    prof_run({}, "jalur lama (tanpa BONSAI_DECODE_H2)")
    prof_run({"BONSAI_DECODE_H2": "1"},
             "jalur cepat half2 (BONSAI_DECODE_H2=1)")

# --------------------------------------------------------------------------
# 7b. METRIK nvprof — berapa persen dari puncak DRAM yang benar-benar
#     tercapai GEMV decode. ncu gagal mengumpulkan metrik di container ini
#     (exit 1), tapi nvprof adalah alat yang BERBEDA; penghitung
#     kelas-nvprof kadang masih boleh diakses walau ncu tidak. Bila ini
#     pun gagal, baris "exit" dan "kepala" di bawah akan menunjukkan
#     alasannya — jangan diulang buta.
#     Pertanyaan yang dijawab: laju ~142 GB/s itu sudah mentok DRAM
#     (320 GB/s puncak T4), atau masih ada ruang?
# --------------------------------------------------------------------------
log("7b. METRIK nvprof (utilisasi DRAM GEMV decode)")
METRIK_NV = ("dram_read_throughput,dram_write_throughput,"
             "gld_throughput,shared_load_throughput,"
             "achieved_occupancy,sm_efficiency")
# BONSAI_DECODE_H2=1 supaya yang diukur benar-benar kernel half2, bukan
# kernel skalar lama. BONSAI_STOP_IDS=none agar generasi tidak berhenti
# di EOS pada langkah kedua (sudah pernah membuat pembagi salah 5,5x).
e_m = dict(env)
e_m["BONSAI_DECODE_H2"] = "1"
e_m["BONSAI_STOP_IDS"] = "none"
# Percobaan pertama memakai "--kernels regex:NAMA" dan DITOLAK:
#   "Invalid kernel path syntax" / "Encountered invalid option : regex:..."
# Jadi nvprof di sini tidak menerima awalan regex:; pakai nama polos.
# --max-tokens 3 (2 langkah decode) karena --metrics mengulang-ulang jalan
# tiap kernel (kernel replay) — dengan 24 token ia akan meng replay
# ~350 launch/token berkali-kali dan memakan waktu berlebih.
cmd_m = ["nvprof", "--csv", "--kernels", "qmv_vec_q2t_h2_kernel",
         "--metrics", METRIK_NV,
         infer, "--model-dir", model_dir,
         "--prompt-tokens", "248045", "--max-tokens", "3", "--gpu"]
try:
    pm = subprocess.run(cmd_m, capture_output=True, text=True, timeout=900,
                        env=e_m)
    log("   exit:", pm.returncode)
    for l in (pm.stdout or "").splitlines()[:40]:
        if l.strip():
            print("   " + l)
    for l in (pm.stderr or "").splitlines()[:8]:
        if l.strip():
            print("   [stderr] " + l)
except Exception as e:
    log("   [GAGAL] menjalankan nvprof --metrics:", e)

# --------------------------------------------------------------------------
# 8. METRIK ncu — penentu: apakah GEMV decode itu memory-bound atau
#    compute-bound. Tanpa ini, mengoptimasi kernel adalah menebak: laju
#    ~136 GB/s bisa berarti "DRAM sudah mentok" (turun-instruksi sia-sia)
#    atau "kernel kehabisan instruksi" (turun-instruksi besar untungnya).
#    Dua kernel diukur: GEMV decode dan kernel elementwise Mojo (yang
#    termahal di situ adalah rekurensi GDN).
# --------------------------------------------------------------------------
log("8. METRIK ncu (memory-bound vs compute-bound)")
METRIK = ("gpu__time_duration.sum,dram__bytes.sum,"
          "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
          "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
          "launch__grid_size,launch__block_size")
for target, tag in (("regex:qmv_vec_q2t_kernel", "GEMV decode"),
                    ("regex:elementwise", "elementwise Mojo")):
    pc = ["ncu", "--csv", "--target-processes", "all",
          "--kernel-name", target, "--launch-count", "16",
          "--metrics", METRIK,
          infer, "--model-dir", model_dir,
          "--prompt-tokens", "248045", "--max-tokens", "4", "--gpu"]
    e = dict(env)
    e["BONSAI_STOP_IDS"] = "none"
    log(f"   --- {tag} ---")
    pr = subprocess.run(pc, env=e, capture_output=True, text=True)
    log("   exit:", pr.returncode)
    outp = (pr.stdout or "")
    if not outp.strip():
        log("   [WARN] stdout kosong; ekor stderr:")
        for l in (pr.stderr or "").splitlines()[-12:]:
            print("   " + l[:160])
        continue
    # ncu --csv: baris data dimulai setelah baris kepala berisi "ID".
    # Cetak hanya kolom yang bermakna supaya log tidak meledak.
    ls = outp.splitlines()
    i0 = next((i for i, l in enumerate(ls) if l.startswith('"ID"')), 0)
    log("   kepala:", ls[i0][:160] if i0 < len(ls) else "(tidak ada)")
    n = 0
    for l in ls[i0:]:
        if "qmv_vec_q2t" in l or "elementwise" in l or "Duration" in l \
                or "dram" in l or "sm__throughput" in l or "launch__" in l:
            print("   " + l[:200])
            n += 1
        if n >= 40:
            break

# --------------------------------------------------------------------------
# 9. PUNCAK DRAM mesin ini, diukur sendiri.
#    Penghitung perangkat keras tidak tersedia (ncu exit 1; nvprof --metrics
#    menolak CC>=7.5), jadi "320 GB/s" selama ini hanya angka lembar data.
#    Tanpa ini kita tidak bisa membedakan dua penjelasan untuk laju GEMV
#    ~150 GB/s: (a) mesin memang hanya sanggup segitu, (b) pola akses kita
#    yang boros. Program kecil di bawah mengukur keduanya dalam satu jalan:
#      k_coalesced    = aliran berkoalesensi penuh (kasus terbaik)
#      k_gemv_pattern = pola instruksi GEMV decode: 8 lane menutup 256 B
#                       berurutan, tapi SATU instruksi hanya mengambil 16 B
#                       dari tiap sektor 32 B.
#    Bila k_gemv_pattern jauh di bawah k_coalesced -> ada amplifikasi baca
#    dan pola akses layak dirombak. Bila keduanya sama -> dinding memori
#    memang ada di situ, dan sisa optimasi harus ke non-GEMV.
# --------------------------------------------------------------------------
log("9. PUNCAK DRAM MESIN (diukur sendiri, bukan lembar data)")
BW_SRC = r"""
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %d at line %d\n", (int)e, __LINE__); return 1; } } while (0)

__global__ void k_coalesced(const uint4* __restrict__ p, long long n,
                            unsigned* __restrict__ out) {
    unsigned a = 0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        uint4 v = p[i];
        a += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a) out[blockIdx.x] = a;
}

__global__ void k_gemv_pattern(const uint4* __restrict__ p, long long rows,
                               long long row_u4, unsigned* __restrict__ out) {
    unsigned a = 0;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long lane_row = tid / 8;
    long long lane_grp = tid % 8;
    long long rows_per_sweep = ((long long)gridDim.x * blockDim.x) / 8;
    for (long long r = lane_row; r < rows; r += rows_per_sweep) {
        const uint4* base = p + r * row_u4;
        for (long long g = lane_grp * 2; g < row_u4; g += 16) {
            uint4 v0 = base[g];
            uint4 v1 = base[g + 1];
            a += (v0.x ^ v0.y ^ v0.z ^ v0.w) ^ (v1.x ^ v1.y ^ v1.z ^ v1.w);
        }
    }
    if (a) out[blockIdx.x] = a;
}

// Sapuan OKUPANSI: kernel yang sama, tapi dipaksa menempati N blok per SM
// lewat __launch_bounds__. Ini menjawab "apakah ~150 GB/s pada GEMV itu
// kekurangan byte melayang karena okupansi rendah?" — kalau GB/s masih
// naik terus sampai okupansi penuh, berarti ya, dan menaikkan okupansi
// kernel GEMV akan menolong. Kalau sudah datar di okupansi 4 (okupansi
// GEMV kita sekarang: 64 register -> 4 blok/SM), berarti bukan itu sebabnya.
template <int OCC>
__global__ void __launch_bounds__(256, OCC) k_occ(
    const uint4* __restrict__ p, long long n, unsigned* __restrict__ out) {
    unsigned a = 0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        uint4 v = p[i];
        a += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a) out[blockIdx.x] = a;
}

// SATU BLOK = SATU ALIRAN: blok ke-b membaca satu rentang berurutan
// [b*chunk, (b+1)*chunk) lalu selesai. Persis seperti GEMV decode: tiap
// blok membaca 32 baris yang BERURUTAN di memori (= satu aliran), jadi
// jumlah aliran = jumlah blok. Grid-stride di atas sebaliknya memberi
// ratusan aliran. Ini menguji hipotesis "laju GEMV rendah karena aliran
// memori terlalu sedikit", yang menjelaskan mengapa lm_head (7760 blok)
// jauh lebih cepat per byte daripada proyeksi layer (160-1088 blok).
__global__ void k_chunk(const uint4* __restrict__ p, long long chunk_u4,
                        unsigned* __restrict__ out) {
    unsigned a = 0;
    const uint4* base = p + (long long)blockIdx.x * chunk_u4;
    for (long long i = threadIdx.x; i < chunk_u4; i += blockDim.x) {
        uint4 v = base[i];
        a += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (a) out[blockIdx.x] = a;
}

int main() {
    size_t bytes = 512ull * 1024ull * 1024ull;
    uint4* d = 0; unsigned* o = 0;
    CK(cudaMalloc((void**)&d, bytes));
    CK(cudaMalloc((void**)&o, 65536));
    CK(cudaMemset(d, 1, bytes));
    long long nu4 = (long long)(bytes / 16);
    long long row_u4 = 2048;              // 32 KB per "baris"
    long long rows = nu4 / row_u4;
    int threads = 256, blocks = 640;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int rep = 0; rep < 3; ++rep) {
        float ms;
        cudaEventRecord(s);
        k_coalesced<<<blocks, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("k_coalesced    %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_gemv_pattern<<<blocks, threads>>>(d, rows, row_u4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("k_gemv_pattern %8.3f ms   %7.1f GB/s\n", ms,
               (double)(rows * row_u4 * 16) / (ms * 1e-3) / 1e9);
    }
    printf("--- sapuan okupansi (blok per SM x 256 thread) ---\n");
    for (int rep = 0; rep < 2; ++rep) {
        float ms;
        cudaEventRecord(s);
        k_occ<1><<<40, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ= 1  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_occ<2><<<80, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ= 2  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_occ<4><<<160, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ= 4  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_occ<6><<<240, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ= 6  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_occ<8><<<320, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ= 8  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
        cudaEventRecord(s);
        k_occ<12><<<480, threads>>>(d, nu4, o);
        cudaEventRecord(e); cudaEventSynchronize(e);
        cudaEventElapsedTime(&ms, s, e);
        printf("occ=12  %8.3f ms   %7.1f GB/s\n", ms,
               (double)bytes / (ms * 1e-3) / 1e9);
    }
    printf("--- sapuan JUMLAH BLOK (= jumlah aliran memori) ---\n");
    const int grids[] = {160, 320, 640, 1280, 2560, 7760};
    for (int rep = 0; rep < 2; ++rep) {
        for (int gi = 0; gi < 6; ++gi) {
            int g = grids[gi];
            long long chunk = nu4 / g;
            float ms;
            cudaEventRecord(s);
            k_chunk<<<g, threads>>>(d, chunk, o);
            cudaEventRecord(e); cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms, s, e);
            printf("blok=%5d  %8.3f ms   %7.1f GB/s\n", g, ms,
                   (double)(chunk * g * 16) / (ms * 1e-3) / 1e9);
        }
    }
    cudaFree(d); cudaFree(o);
    return 0;
}
"""
try:
    bw_dir = os.path.join(base, "bwtest")
    os.makedirs(bw_dir, exist_ok=True)
    bw_cu = os.path.join(bw_dir, "bw.cu")
    with open(bw_cu, "w") as f:
        f.write(BW_SRC)
    nvcc = sh(["bash", "-lc", "command -v nvcc"]).strip() or \
        "/usr/local/cuda/bin/nvcc"
    cb = subprocess.run([nvcc, "-O3", "-arch=sm_75", "-o",
                         os.path.join(bw_dir, "bw"), bw_cu],
                        capture_output=True, text=True)
    log("   nvcc:", nvcc, "| exit:", cb.returncode)
    if cb.returncode != 0:
        for l in (cb.stderr or "").splitlines()[:15]:
            print("   [nvcc] " + l[:200])
    else:
        rb = subprocess.run([os.path.join(bw_dir, "bw")],
                            capture_output=True, text=True, timeout=600)
        log("   jalankan exit:", rb.returncode)
        for l in (rb.stdout or "").splitlines()[:40]:
            if l.strip():
                print("   " + l)
        for l in (rb.stderr or "").splitlines()[:8]:
            if l.strip():
                print("   [stderr] " + l[:200])
        shutil.rmtree(bw_dir, ignore_errors=True)
except Exception as e:
    log("   [GAGAL] pengukur bandwidth:", e)

# --------------------------------------------------------------------------
# 10. PROBE REKURENSI GDN — ukur DULU desain mana yang menang, sebelum
#     menulis ulang kernel Mojo-nya.
#     Alasan: dua prediksi struktural di sesi ini sudah difalsifikasi
#     pengukuran (varian GEMV "lebar" 3,4% lebih lambat; unroll 4x di
#     rekurensi 4,6% lebih lambat). Jadi jangan menulis ~120 baris Mojo baru
#     berdasarkan dugaan. Probe ini memakai pola akses yang PERSIS sama
#     (state [48,128,128] FP32, 1 thread = 1 baris 128 float) dan menjalankan
#     4 desain pada beban yang identik:
#       V0 = pola sekarang   : baca, baca+tulis            = 9 MB/layer
#       V1 = SMEM 32 baris   : baca sekali, tulis sekali    = 6 MB/layer
#       V2 = 4 thread/baris  : idem + 4x lipat thread       = 6 MB/layer
#       V3 = 8 thread/baris  : idem + 8x lipat thread       = 6 MB/layer
#     V1 itu penting: ia menurunkan trafik TETAPI menurunkan okupansi jadi
#     4 blok/SM. Kalau kernel ini terhambat latensi (ia jalan di 53 GB/s =
#     20% puncak, jadi bukan bandwidth), V1 bisa lebih LAMBAT — itu yang
#     ingin diketahui sebelum mengerjakannya.
#     Satu peluncuran = 48 layer sekaligus (grid.z = LAY), supaya waktu
#     luncuran tidak ikut terukur.
# --------------------------------------------------------------------------
log("10. PROBE REKURENSI GDN (desain mana yang menang — diukur)")
GDN_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %d at line %d\n", (int)e, __LINE__); return 1; } } while (0)

#define HV   48
#define DV   128
#define DK   128
#define LAY  48
#define SZ   ((size_t)HV * DV * DK)

__device__ __forceinline__ float dot16(float4 a, float4 b, float4 c, float4 d,
                                       const float* y) {
    return a.x*y[0]  + a.y*y[1]  + a.z*y[2]  + a.w*y[3]
         + b.x*y[4]  + b.y*y[5]  + b.z*y[6]  + b.w*y[7]
         + c.x*y[8]  + c.y*y[9]  + c.z*y[10] + c.w*y[11]
         + d.x*y[12] + d.y*y[13] + d.z*y[14] + d.w*y[15];
}

__device__ __forceinline__ void upd16(float4 s0, float4 s1, float4 s2, float4 s3,
                                      const float* kk, float decay, float delta,
                                      float4& n0, float4& n1, float4& n2, float4& n3) {
    n0 = make_float4(s0.x*decay + kk[0]*delta,  s0.y*decay + kk[1]*delta,
                     s0.z*decay + kk[2]*delta,  s0.w*decay + kk[3]*delta);
    n1 = make_float4(s1.x*decay + kk[4]*delta,  s1.y*decay + kk[5]*delta,
                     s1.z*decay + kk[6]*delta,  s1.w*decay + kk[7]*delta);
    n2 = make_float4(s2.x*decay + kk[8]*delta,  s2.y*decay + kk[9]*delta,
                     s2.z*decay + kk[10]*delta, s2.w*decay + kk[11]*delta);
    n3 = make_float4(s3.x*decay + kk[12]*delta, s3.y*decay + kk[13]*delta,
                     s3.z*decay + kk[14]*delta, s3.w*decay + kk[15]*delta);
}

// ---------------- V0: pola SEKARANG (baca, lalu baca+tulis) ---------------
__global__ void v0_cur(float* __restrict__ S, const float* __restrict__ vv,
                       float* __restrict__ out, const float* __restrict__ kq,
                       float decay, float beta) {
    __shared__ float sk[DK], sq[DK];
    int hv = blockIdx.x;
    int dv = blockIdx.y * blockDim.x + threadIdx.x;
    int L  = blockIdx.z;
    for (int i = threadIdx.x; i < DK; i += blockDim.x) {
        sk[i] = kq[i]; sq[i] = kq[DK + i];
    }
    __syncthreads();
    float* row = S + L * SZ + (size_t)(hv * DV + dv) * DK;
    float kv = 0.f;
    for (int i = 0; i < DK; i += 16) {
        float4 s0 = *(const float4*)(row + i);
        float4 s1 = *(const float4*)(row + i + 4);
        float4 s2 = *(const float4*)(row + i + 8);
        float4 s3 = *(const float4*)(row + i + 12);
        float4 d0 = make_float4(s0.x*decay, s0.y*decay, s0.z*decay, s0.w*decay);
        float4 d1 = make_float4(s1.x*decay, s1.y*decay, s1.z*decay, s1.w*decay);
        float4 d2 = make_float4(s2.x*decay, s2.y*decay, s2.z*decay, s2.w*decay);
        float4 d3 = make_float4(s3.x*decay, s3.y*decay, s3.z*decay, s3.w*decay);
        kv += dot16(d0, d1, d2, d3, sk + i);
    }
    float delta = (vv[hv * DV + dv] - kv) * beta;
    float ro = 0.f;
    for (int i = 0; i < DK; i += 16) {
        float4 s0 = *(const float4*)(row + i);
        float4 s1 = *(const float4*)(row + i + 4);
        float4 s2 = *(const float4*)(row + i + 8);
        float4 s3 = *(const float4*)(row + i + 12);
        float4 n0, n1, n2, n3;
        upd16(s0, s1, s2, s3, sk + i, decay, delta, n0, n1, n2, n3);
        *(float4*)(row + i)      = n0;
        *(float4*)(row + i + 4)  = n1;
        *(float4*)(row + i + 8)  = n2;
        *(float4*)(row + i + 12) = n3;
        ro += dot16(n0, n1, n2, n3, sq + i);
    }
    out[L * HV * DV + hv * DV + dv] = ro;
}

// ---------------- V1: 32 baris/blok, S ditahan di SMEM 16 KB --------------
__global__ void v1_smem32(float* __restrict__ S, const float* __restrict__ vv,
                          float* __restrict__ out, const float* __restrict__ kq,
                          float decay, float beta) {
    extern __shared__ float sh[];
    __shared__ float sk[DK], sq[DK];
    int hv = blockIdx.x, rg = blockIdx.y, tl = threadIdx.x, L = blockIdx.z;
    for (int i = tl; i < DK; i += blockDim.x) { sk[i] = kq[i]; sq[i] = kq[DK + i]; }
    int dv = rg * blockDim.x + tl;
    float* row = S + L * SZ + (size_t)(hv * DV + dv) * DK;
    for (int i = 0; i < DK; i += 16) {
        float4 s0 = *(const float4*)(row + i);
        float4 s1 = *(const float4*)(row + i + 4);
        float4 s2 = *(const float4*)(row + i + 8);
        float4 s3 = *(const float4*)(row + i + 12);
        *(float4*)&sh[tl * DK + i]      = s0;
        *(float4*)&sh[tl * DK + i + 4]  = s1;
        *(float4*)&sh[tl * DK + i + 8]  = s2;
        *(float4*)&sh[tl * DK + i + 12] = s3;
    }
    __syncthreads();
    float kv = 0.f;
    for (int i = 0; i < DK; i += 16) {
        float4 s0 = *(const float4*)&sh[tl * DK + i];
        float4 s1 = *(const float4*)&sh[tl * DK + i + 4];
        float4 s2 = *(const float4*)&sh[tl * DK + i + 8];
        float4 s3 = *(const float4*)&sh[tl * DK + i + 12];
        float4 d0 = make_float4(s0.x*decay, s0.y*decay, s0.z*decay, s0.w*decay);
        float4 d1 = make_float4(s1.x*decay, s1.y*decay, s1.z*decay, s1.w*decay);
        float4 d2 = make_float4(s2.x*decay, s2.y*decay, s2.z*decay, s2.w*decay);
        float4 d3 = make_float4(s3.x*decay, s3.y*decay, s3.z*decay, s3.w*decay);
        kv += dot16(d0, d1, d2, d3, sk + i);
    }
    float delta = (vv[hv * DV + dv] - kv) * beta;
    float ro = 0.f;
    for (int i = 0; i < DK; i += 16) {
        float4 s0 = *(const float4*)&sh[tl * DK + i];
        float4 s1 = *(const float4*)&sh[tl * DK + i + 4];
        float4 s2 = *(const float4*)&sh[tl * DK + i + 8];
        float4 s3 = *(const float4*)&sh[tl * DK + i + 12];
        float4 n0, n1, n2, n3;
        upd16(s0, s1, s2, s3, sk + i, decay, delta, n0, n1, n2, n3);
        *(float4*)(row + i)      = n0;
        *(float4*)(row + i + 4)  = n1;
        *(float4*)(row + i + 8)  = n2;
        *(float4*)(row + i + 12) = n3;
        ro += dot16(n0, n1, n2, n3, sq + i);
    }
    out[L * HV * DV + hv * DV + dv] = ro;
}

// ---------------- V2: 4 thread per baris (32 float/thread) ----------------
__global__ void v2_tpr4(float* __restrict__ S, const float* __restrict__ vv,
                        float* __restrict__ out, const float* __restrict__ kq,
                        float decay, float beta) {
    __shared__ float sh[32 * DK];
    __shared__ float sk[DK], sq[DK];
    __shared__ float redA[128], redB[128];
    int hv = blockIdx.x, rg = blockIdx.y, t = threadIdx.x, L = blockIdx.z;
    for (int i = t; i < DK; i += blockDim.x) { sk[i] = kq[i]; sq[i] = kq[DK + i]; }
    float* Sbase = S + L * SZ + (size_t)(hv * DV + rg * 32) * DK;
    for (int c = t; c < 32 * DK / 4; c += blockDim.x) {
        int r = (4 * c) / DK;
        int off = (4 * c) % DK;
        *(float4*)&sh[r * DK + off] = *(const float4*)(Sbase + r * DK + off);
    }
    __syncthreads();
    int r = t >> 2, p = t & 3;
    const float* sr = &sh[r * DK + p * 32];
    float A = 0.f, B = 0.f;
    for (int i = 0; i < 32; i += 16) {
        float4 s0 = *(const float4*)(sr + i);
        float4 s1 = *(const float4*)(sr + i + 4);
        float4 s2 = *(const float4*)(sr + i + 8);
        float4 s3 = *(const float4*)(sr + i + 12);
        A += dot16(s0, s1, s2, s3, sk + p * 32 + i);
        B += dot16(s0, s1, s2, s3, sq + p * 32 + i);
    }
    redA[t] = A; redB[t] = B;
    __syncthreads();
    float At = 0.f, Bt = 0.f;
    for (int q = 0; q < 4; ++q) { At += redA[r * 4 + q]; Bt += redB[r * 4 + q]; }
    float delta = (vv[hv * DV + rg * 32 + r] - decay * At) * beta;
    float kqsum = 0.f;
    for (int i = 0; i < DK; ++i) kqsum += sk[i] * sq[i];
    float* wr = Sbase + r * DK + p * 32;
    for (int i = 0; i < 32; i += 16) {
        float4 s0 = *(const float4*)(sr + i);
        float4 s1 = *(const float4*)(sr + i + 4);
        float4 s2 = *(const float4*)(sr + i + 8);
        float4 s3 = *(const float4*)(sr + i + 12);
        float4 n0, n1, n2, n3;
        upd16(s0, s1, s2, s3, sk + p * 32 + i, decay, delta, n0, n1, n2, n3);
        *(float4*)(wr + i)      = n0;
        *(float4*)(wr + i + 4)  = n1;
        *(float4*)(wr + i + 8)  = n2;
        *(float4*)(wr + i + 12) = n3;
    }
    if (p == 0) out[L * HV * DV + hv * DV + rg * 32 + r] = decay * Bt + delta * kqsum;
}

// ---------------- V3: 8 thread per baris (16 float/thread) ----------------
__global__ void v3_tpr8(float* __restrict__ S, const float* __restrict__ vv,
                        float* __restrict__ out, const float* __restrict__ kq,
                        float decay, float beta) {
    __shared__ float sh[32 * DK];
    __shared__ float sk[DK], sq[DK];
    __shared__ float redA[256], redB[256];
    int hv = blockIdx.x, rg = blockIdx.y, t = threadIdx.x, L = blockIdx.z;
    for (int i = t; i < DK; i += blockDim.x) { sk[i] = kq[i]; sq[i] = kq[DK + i]; }
    float* Sbase = S + L * SZ + (size_t)(hv * DV + rg * 32) * DK;
    for (int c = t; c < 32 * DK / 4; c += blockDim.x) {
        int r = (4 * c) / DK;
        int off = (4 * c) % DK;
        *(float4*)&sh[r * DK + off] = *(const float4*)(Sbase + r * DK + off);
    }
    __syncthreads();
    int r = t >> 3, p = t & 7;
    const float* sr = &sh[r * DK + p * 16];
    float A = 0.f, B = 0.f;
    for (int i = 0; i < 16; i += 16) {
        float4 s0 = *(const float4*)(sr + i);
        float4 s1 = *(const float4*)(sr + i + 4);
        float4 s2 = *(const float4*)(sr + i + 8);
        float4 s3 = *(const float4*)(sr + i + 12);
        A += dot16(s0, s1, s2, s3, sk + p * 16 + i);
        B += dot16(s0, s1, s2, s3, sq + p * 16 + i);
    }
    redA[t] = A; redB[t] = B;
    __syncthreads();
    float At = 0.f, Bt = 0.f;
    for (int q = 0; q < 8; ++q) { At += redA[r * 8 + q]; Bt += redB[r * 8 + q]; }
    float delta = (vv[hv * DV + rg * 32 + r] - decay * At) * beta;
    float kqsum = 0.f;
    for (int i = 0; i < DK; ++i) kqsum += sk[i] * sq[i];
    float* wr = Sbase + r * DK + p * 16;
    for (int i = 0; i < 16; i += 16) {
        float4 s0 = *(const float4*)(sr + i);
        float4 s1 = *(const float4*)(sr + i + 4);
        float4 s2 = *(const float4*)(sr + i + 8);
        float4 s3 = *(const float4*)(sr + i + 12);
        float4 n0, n1, n2, n3;
        upd16(s0, s1, s2, s3, sk + p * 16 + i, decay, delta, n0, n1, n2, n3);
        *(float4*)(wr + i)      = n0;
        *(float4*)(wr + i + 4)  = n1;
        *(float4*)(wr + i + 8)  = n2;
        *(float4*)(wr + i + 12) = n3;
    }
    if (p == 0) out[L * HV * DV + hv * DV + rg * 32 + r] = decay * Bt + delta * kqsum;
}

// =========================================================================
// POLA BACA GEMV DECODE (kernel h2): 8 lane per baris, 4 baris per warp,
// 32 baris per blok, satu grup 128 bobot = 32 B per lane per iterasi.
// Yang diuji: apakah BACAAN SKALA (2 byte per grup, terpisah dari kode)
// memakan biaya bandwidth. Kalau ya, skala harus ditahankan ke SMEM.
//   GA = cuma kode (batas atas)
//   GB = kode + skala dibaca dari global (pola sekarang)
//   GC = kode + skala ditahankan ke SMEM lewat tahapan berkoalesensi
// =========================================================================
#define LPR 8
#define RPW 4
#define RPB 32

__global__ void k_gemv_codes(const uint4* __restrict__ w, int n, int k,
                             int gpr, unsigned* __restrict__ out) {
    const int tid = threadIdx.x, lane = tid & 31;
    const int r = lane / LPR, grp = lane % LPR;
    const int row = blockIdx.x * RPB + (tid >> 5) * RPW + r;
    if (row >= n) return;
    const uint4* w_row = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(w) + (size_t)row * (size_t)(k / 4));
    unsigned a = 0;
    for (int g0 = 0; g0 < gpr; g0 += 8) {
        int gg = g0 + grp;
        if (gg < gpr) {
            uint4 c0 = w_row[gg * 2];
            uint4 c1 = w_row[gg * 2 + 1];
            a ^= c0.x ^ c0.y ^ c0.z ^ c0.w ^ c1.x ^ c1.y ^ c1.z ^ c1.w;
        }
    }
    if (a) out[row] = a;
}

__global__ void k_gemv_scales(const uint4* __restrict__ w,
                              const unsigned short* __restrict__ sc,
                              int n, int k, int gpr, unsigned* __restrict__ out) {
    const int tid = threadIdx.x, lane = tid & 31;
    const int r = lane / LPR, grp = lane % LPR;
    const int row = blockIdx.x * RPB + (tid >> 5) * RPW + r;
    if (row >= n) return;
    const uint4* w_row = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(w) + (size_t)row * (size_t)(k / 4));
    unsigned a = 0;
    for (int g0 = 0; g0 < gpr; g0 += 8) {
        int gg = g0 + grp;
        if (gg < gpr) {
            uint4 c0 = w_row[gg * 2];
            uint4 c1 = w_row[gg * 2 + 1];
            a ^= c0.x ^ c0.y ^ c0.z ^ c0.w ^ c1.x ^ c1.y ^ c1.z ^ c1.w;
            a += (unsigned)sc[(size_t)row * gpr + gg];
        }
    }
    if (a) out[row] = a;
}

__global__ void k_gemv_smem(const uint4* __restrict__ w,
                            const unsigned short* __restrict__ sc,
                            int n, int k, int gpr, unsigned* __restrict__ out) {
    __shared__ unsigned short ssm[RPB * 64];
    const int tid = threadIdx.x, lane = tid & 31;
    // Tahapan skala: baris-block x gpr itu KONTIGU di memori (baris berurutan),
    // jadi 256 thread bisa menariknya berkoalesensi penuh.
    for (int i = tid; i < RPB * gpr; i += 256) {
        int rr = i / gpr, gg2 = i % gpr;
        ssm[rr * 64 + gg2] = sc[(size_t)(blockIdx.x * RPB + rr) * gpr + gg2];
    }
    __syncthreads();
    const int r = lane / LPR, grp = lane % LPR;
    const int row = blockIdx.x * RPB + (tid >> 5) * RPW + r;
    if (row >= n) return;
    const uint4* w_row = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(w) + (size_t)row * (size_t)(k / 4));
    const int roff = (tid >> 5) * RPW + r;
    unsigned a = 0;
    for (int g0 = 0; g0 < gpr; g0 += 8) {
        int gg = g0 + grp;
        if (gg < gpr) {
            uint4 c0 = w_row[gg * 2];
            uint4 c1 = w_row[gg * 2 + 1];
            a ^= c0.x ^ c0.y ^ c0.z ^ c0.w ^ c1.x ^ c1.y ^ c1.z ^ c1.w;
            a += (unsigned)ssm[roff * 64 + gg];
        }
    }
    if (a) out[row] = a;
}

// =========================================================================
// SALINAN SETIA kernel h2 yang benar-benar dipakai model
// (qmv_vec_q2t_h2_kernel di src/csrc/qmv_sm75_kernel.cu).
//
// Mengapa ini perlu: GA/GB/GC di atas mengukur kecepatan BACA saja — kata
// bobot cukup di-XOR, tanpa dekode PRMT + HFMA2 yang sesungguhnya. Jadi
// 256 GB/s pada GA adalah batas ATAS memori, bukan laju kernel nyata. Di
// dalam model, GEMV menyerap sekitar 35 ms dari 49 ms/token untuk kira-kira
// 6,4 GB bobot = sekitar 190 GB/s. Selisih 256 -> 190 GB/s itu bisa berasal
// dari (a) bacaan skala, (b) tahapan x per tile berikut barrier-nya, atau
// (c) kerja PRMT + HFMA2 itu sendiri. Ketiganya dimatikan satu-satu lewat
// parameter templat USE_SCALE / USE_PRMF / X_ONCE sehingga biayanya terukur,
// bukan diterka.
// =========================================================================
#define GS      8          // grup per tile K
#define LPR8    8          // lane per baris keluaran
#define RPW4    4          // baris per warp
#define RPB32   32         // baris per blok
#define GRP_PAD 136        // half per grup di SMEM (lihat kernel asli)
#define K_TILE  1024       // GS * 128
#define BLK     256

// Tabel byte-tinggi half(q-1), diindeks langsung oleh q. Disalin VERBATIM
// dari q2t_expand_word_h2 agar beban instruksinya sama persis.
__device__ __forceinline__ void q2t_expand_word_h2(unsigned word, unsigned out[8]) {
    const unsigned TAB_H2 = 0x403C00BCu;
    const unsigned pe0 = __byte_perm(TAB_H2, TAB_H2,  word        & 0x33333333u);
    const unsigned po0 = __byte_perm(TAB_H2, TAB_H2, (word >>  2) & 0x33333333u);
    const unsigned pe1 = __byte_perm(TAB_H2, TAB_H2, (word >> 16) & 0x33333333u);
    const unsigned po1 = __byte_perm(TAB_H2, TAB_H2, (word >> 18) & 0x33333333u);
    const unsigned d0 = __byte_perm(pe0, po0, 0x5140u);
    const unsigned d1 = __byte_perm(pe0, po0, 0x7362u);
    const unsigned d2 = __byte_perm(pe1, po1, 0x5140u);
    const unsigned d3 = __byte_perm(pe1, po1, 0x7362u);
    const unsigned z = 0u;
    out[0] = __byte_perm(d0, z, 0x1404u);
    out[1] = __byte_perm(d0, z, 0x3424u);
    out[2] = __byte_perm(d1, z, 0x1404u);
    out[3] = __byte_perm(d1, z, 0x3424u);
    out[4] = __byte_perm(d2, z, 0x1404u);
    out[5] = __byte_perm(d2, z, 0x3424u);
    out[6] = __byte_perm(d3, z, 0x1404u);
    out[7] = __byte_perm(d3, z, 0x3424u);
}

template <int USE_SCALE, int USE_PRMF, int X_ONCE>
__global__ void k_h2(const __half* __restrict__ x,
                     const uint8_t* __restrict__ w,
                     const __half* __restrict__ scales,
                     float* __restrict__ out,
                     int n, int k) {
    // 5440 half = 10,9 KB: cukup untuk menahapkan SELURUH x (K=5120) pada
    // varian X_ONCE, dan untuk satu tile (indeks maks 1079) pada varian
    // bertile. 4 blok/SM x 10,9 KB = 43,5 KB masih di bawah 64 KB/SM, jadi
    // okupansi tidak berubah antar varian.
    __shared__ __align__(16) __half x_s[5440];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int r    = lane / LPR8;
    const int grp  = lane % LPR8;
    const int row  = blockIdx.x * RPB32 + (tid >> 5) * RPW4 + r;
    const bool row_valid = row < n;
    const uint8_t* w_row = w + (size_t)row * (size_t)(k / 4);
    const int gpr = k / 128;
    const __half hz = __float2half(0.0f);
    float acc = 0.0f;

    if (X_ONCE) {
        for (int e = tid * 8; e < k; e += BLK * 8) {
            uint4 v = *reinterpret_cast<const uint4*>(x + e);
            *reinterpret_cast<uint4*>(&x_s[(e / 128) * GRP_PAD + (e % 128)]) = v;
        }
        __syncthreads();
    }

    for (int g0 = 0; g0 < gpr; g0 += GS) {
        if (!X_ONCE) {
            const int kc = g0 * 128;
            for (int e = tid * 8; e < K_TILE; e += BLK * 8) {
                uint4 v = *reinterpret_cast<const uint4*>(x + kc + e);
                *reinterpret_cast<uint4*>(&x_s[(e / 128) * GRP_PAD + (e % 128)]) = v;
            }
            __syncthreads();
        }
        const int gg = g0 + grp;
        if (row_valid && gg < gpr) {
            float s_val = 1.0f;
            if (USE_SCALE) s_val = __half2float(scales[(size_t)row * gpr + gg]);
            const __half2* xg2 =
                reinterpret_cast<const __half2*>(&x_s[grp * GRP_PAD]);
            const uint4* wptr =
                reinterpret_cast<const uint4*>(w_row + (size_t)gg * 32);
            const uint4 wvec[2] = { wptr[0], wptr[1] };
            float local = 0.0f;
            unsigned ax = 0u;
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const unsigned* ww = reinterpret_cast<const unsigned*>(&wvec[half]);
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    if (USE_PRMF) {
                        unsigned h2[8];
                        q2t_expand_word_h2(ww[wi], h2);
                        __half2 a2 = __halves2half2(hz, hz);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            a2 = __hfma2(
                                *reinterpret_cast<const __half2*>(&h2[j]),
                                xg2[half * 32 + wi * 8 + j], a2);
                            if ((j & 3) == 3) {
                                local += __low2float(a2) + __high2float(a2);
                                a2 = __halves2half2(hz, hz);
                            }
                        }
                    } else {
                        // BACA SAJA: beban memori identik, tanpa PRMT/HFMA2.
                        // Hasil tetap dipakai di bawah supaya kompilator
                        // tidak membuang pemuatannya.
                        ax ^= ww[wi];
                    }
                }
            }
            acc += s_val * (USE_PRMF ? local : (float)(ax & 0xFFFFu));
        }
        if (!X_ONCE) __syncthreads();
    }

    float v = acc;
    #pragma unroll
    for (int off = LPR8 / 2; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffffu, v, off, LPR8);
    if (row_valid && grp == 0) out[row] = v;
}

template <typename F>
static double bench(F launch, int R) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    launch();
    cudaDeviceSynchronize();
    cudaEventRecord(s, 0);
    for (int i = 0; i < R; ++i) launch();
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return (double)ms / (double)R;
}

// Rejim NYATA: satu token = 48 peluncuran BERURUTAN, satu layer tiap
// luncuran (layer L harus menunggu layer L-1). Di sinilah 48 blok x 128
// thread harus bekerja sendirian tanpa lapisan lain yang menemani.
template <typename F>
static double bench_seq(F launch_at, int n, int R) {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < n; ++i) launch_at(i);
    cudaDeviceSynchronize();
    cudaEventRecord(s, 0);
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < n; ++i) launch_at(i);
    cudaEventRecord(e, 0);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return (double)ms / (double)R;
}

int main() {
    const float decay = 0.99f, beta = 0.5f;
    const int R = 20;
    const size_t nf = SZ * (size_t)LAY;
    float *dS = 0, *dS2 = 0, *dout = 0, *dout2 = 0, *dvv = 0, *dkq = 0;
    CK(cudaMalloc((void**)&dS,  nf * sizeof(float)));
    CK(cudaMalloc((void**)&dS2, nf * sizeof(float)));
    CK(cudaMalloc((void**)&dout,  (size_t)LAY * HV * DV * sizeof(float)));
    CK(cudaMalloc((void**)&dout2, (size_t)LAY * HV * DV * sizeof(float)));
    CK(cudaMalloc((void**)&dvv, (size_t)HV * DV * sizeof(float)));
    CK(cudaMalloc((void**)&dkq, 2 * DK * sizeof(float)));

    float* hb = (float*)malloc(nf * sizeof(float));
    for (size_t i = 0; i < nf; ++i) hb[i] = (float)(i % 31) * 0.01f;
    CK(cudaMemcpy(dS,  hb, nf * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS2, hb, nf * sizeof(float), cudaMemcpyHostToDevice));
    free(hb);

    float* hvv = (float*)malloc((size_t)HV * DV * sizeof(float));
    float* hkq = (float*)malloc(2 * DK * sizeof(float));
    for (int i = 0; i < HV * DV; ++i) hvv[i] = (float)(i % 17) * 0.01f;
    for (int i = 0; i < 2 * DK; ++i) hkq[i] = (float)(i % 13) * 0.01f + 0.01f;
    CK(cudaMemcpy(dvv, hvv, (size_t)HV * DV * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkq, hkq, 2 * DK * sizeof(float), cudaMemcpyHostToDevice));
    free(hvv); free(hkq);

    dim3 g0(HV, 1, LAY);
    dim3 g4(HV, DV / 32, LAY);
    size_t smem1 = (size_t)(32 * DK) * sizeof(float);

    printf("trafik: V0 = 432 MB/token (9 MB x 48) | V1..V3 = 288 MB/token (6 MB x 48)\n");

    // ---- REJIM A: seperti model nyata, 48 luncuran berurutan ----
    printf("--- REJIM A: 48 luncuran BERURUTAN, 1 layer/luncuran (model nyata) ---\n");
    double a0 = bench_seq([&](int i){ v0_cur<<<dim3(HV,1,1), 128>>>(dS + (size_t)i*SZ, dvv, dout, dkq, decay, beta); }, LAY, R);
    printf("  V0 sekarang  48 blok x128 thr  %8.3f ms/token  %7.1f GB/s\n", a0, 432.0 / a0);
    double a1 = bench_seq([&](int i){ v1_smem32<<<dim3(HV,DV/32,1), 32, smem1>>>(dS + (size_t)i*SZ, dvv, dout, dkq, decay, beta); }, LAY, R);
    printf("  V1 smem32   192 blok x 32 thr  %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", a1, 288.0 / a1, a0 / a1);
    double a2 = bench_seq([&](int i){ v2_tpr4<<<dim3(HV,DV/32,1), 128>>>(dS + (size_t)i*SZ, dvv, dout, dkq, decay, beta); }, LAY, R);
    printf("  V2 tpr4     192 blok x128 thr  %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", a2, 288.0 / a2, a0 / a2);
    double a3 = bench_seq([&](int i){ v3_tpr8<<<dim3(HV,DV/32,1), 256>>>(dS + (size_t)i*SZ, dvv, dout, dkq, decay, beta); }, LAY, R);
    printf("  V3 tpr8     192 blok x256 thr  %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", a3, 288.0 / a3, a0 / a3);

    // ---- REJIM B: 48 layer sekaligus dalam 1 luncuran (batas atas) ----
    printf("--- REJIM B: 1 luncuran, 48 layer SERENTAK (batas atas, bukan model nyata) ---\n");
    double m0 = bench([&](){ v0_cur<<<g0, 128>>>(dS, dvv, dout, dkq, decay, beta); }, R);
    printf("  V0 sekarang            %8.3f ms/token  %7.1f GB/s\n", m0, 432.0 / m0);
    double m1 = bench([&](){ v1_smem32<<<g4, 32, smem1>>>(dS, dvv, dout, dkq, decay, beta); }, R);
    printf("  V1 smem32              %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", m1, 288.0 / m1, m0 / m1);
    double m2 = bench([&](){ v2_tpr4<<<g4, 128>>>(dS, dvv, dout, dkq, decay, beta); }, R);
    printf("  V2 tpr4                %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", m2, 288.0 / m2, m0 / m2);
    double m3 = bench([&](){ v3_tpr8<<<g4, 256>>>(dS, dvv, dout, dkq, decay, beta); }, R);
    printf("  V3 tpr8                %8.3f ms/token  %7.1f GB/s  (%.2fx vs V0)\n", m3, 288.0 / m3, m0 / m3);
    printf("CATATAN: yang dipakai model adalah REJIM A. Selisih A vs B = harga\n");
    printf("         peluncuran 48 blok yang berdiri sendiri (okupansi 7,5%% GPU).\n");

    // UJI KESEPADANAN ALJABAR: V0 memakai out = sum(S_baru * q); V2/V3 memakai
    // identitas out = decay*B + delta*kq. Keduanya harus nyaris sama.
    CK(cudaMemcpy(dS2, dS, nf * sizeof(float), cudaMemcpyDeviceToDevice));
    v0_cur<<<g0, 128>>>(dS2, dvv, dout, dkq, decay, beta);
    v2_tpr4<<<g4, 128>>>(dS, dvv, dout2, dkq, decay, beta);
    cudaDeviceSynchronize();
    size_t no = (size_t)LAY * HV * DV;
    float* h1 = (float*)malloc(no * sizeof(float));
    float* h2 = (float*)malloc(no * sizeof(float));
    CK(cudaMemcpy(h1, dout,  no * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h2, dout2, no * sizeof(float), cudaMemcpyDeviceToHost));
    double mx = 0.0, sa = 0.0;
    for (size_t i = 0; i < no; ++i) {
        double d = (double)h1[i] - (double)h2[i];
        if (d < 0) d = -d;
        if (d > mx) mx = d;
        sa += ((double)h1[i] < 0 ? -(double)h1[i] : (double)h1[i]);
    }
    printf("kesepadanan V0 vs V2: selisih maks = %.3e | |out| rata2 = %.3e | contoh V0[0]=%.6f V2[0]=%.6f\n",
           mx, sa / (double)no, h1[0], h2[0]);
    free(h1); free(h2);

    // ---- POLA BACA GEMV: apakah bacaan SKALA memakan bandwidth? ----
    // Byte yang dihitung = kode SAJA untuk ketiga varian, jadi kalau skala
    // memakan biaya, angka GB/s-nya langsung terlihat turun.
    {
        const int GN = 34816, GK = 5120, GPR = GK / 128;
        const size_t cb = (size_t)GN * (size_t)(GK / 4);
        uint4* gw = 0; unsigned short* gs = 0; unsigned* go = 0;
        CK(cudaMalloc((void**)&gw, cb));
        CK(cudaMalloc((void**)&gs, (size_t)GN * GPR * sizeof(unsigned short)));
        CK(cudaMalloc((void**)&go, (size_t)GN * sizeof(unsigned)));
        cudaMemset(gw, 0x5A, cb);
        cudaMemset(gs, 0x11, (size_t)GN * GPR * sizeof(unsigned short));
        const int gb = GN / RPB;
        const double bytes = (double)cb;
        printf("--- POLA BACA GEMV (gate_up %dx%d, %d blok x 256 thr) ---\n", GN, GK, gb);
        double ta = bench([&](){ k_gemv_codes<<<gb, 256>>>(gw, GN, GK, GPR, go); }, 20);
        printf("  GA cuma kode             %8.3f ms  %7.1f GB/s\n",
               ta, bytes / (ta * 1e-3) / 1e9);
        double tb = bench([&](){ k_gemv_scales<<<gb, 256>>>(gw, gs, GN, GK, GPR, go); }, 20);
        printf("  GB kode + skala global   %8.3f ms  %7.1f GB/s  (%.2fx vs GA)\n",
               tb, bytes / (tb * 1e-3) / 1e9, ta / tb);
        double tc = bench([&](){ k_gemv_smem<<<gb, 256>>>(gw, gs, GN, GK, GPR, go); }, 20);
        printf("  GC kode + skala di SMEM  %8.3f ms  %7.1f GB/s  (%.2fx vs GA)\n",
               tc, bytes / (tc * 1e-3) / 1e9, ta / tc);
        cudaFree(gw); cudaFree(gs); cudaFree(go);
    }

    // ---- KERNEL h2 SEBENARNYA: batas atas memori vs laju nyata ----
    // GB/s dihitung dari byte KODE saja supaya langsung sebanding dengan
    // GA/GB/GC di atas (yang juga hanya menghitung byte kode).
    {
        const int GN = 34816, GK = 5120, GPR = GK / 128;
        const size_t cb = (size_t)GN * (size_t)(GK / 4);
        uint8_t* gw = 0; __half* gs = 0; __half* gx = 0; float* go = 0;
        CK(cudaMalloc((void**)&gw, cb));
        CK(cudaMalloc((void**)&gs, (size_t)GN * GPR * sizeof(__half)));
        CK(cudaMalloc((void**)&gx, (size_t)GK * sizeof(__half)));
        CK(cudaMalloc((void**)&go, (size_t)GN * sizeof(float)));
        cudaMemset(gw, 0x5A, cb);
        cudaMemset(gs, 0x11, (size_t)GN * GPR * sizeof(__half));
        cudaMemset(gx, 0x22, (size_t)GK * sizeof(__half));
        const int gb = GN / RPB32;
        const double bytes = (double)cb;
        printf("--- KERNEL h2 SEBENARNYA (gate_up %dx%d, %d blok x %d thr) ---\n",
               GN, GK, gb, BLK);
        printf("    GB/s dihitung dari byte KODE saja (sebanding dgn GA/GB/GC)\n");
        double t1 = bench([&](){ k_h2<1,1,0><<<gb, BLK>>>(gx, gw, gs, go, GN, GK); }, 20);
        printf("  H1 setia (skala+PRMT+x per tile) %8.3f ms  %7.1f GB/s\n",
               t1, bytes / (t1 * 1e-3) / 1e9);
        double t2 = bench([&](){ k_h2<0,1,0><<<gb, BLK>>>(gx, gw, gs, go, GN, GK); }, 20);
        printf("  H2 tanpa baca skala             %8.3f ms  %7.1f GB/s  (%.2fx vs H1)\n",
               t2, bytes / (t2 * 1e-3) / 1e9, t1 / t2);
        double t3 = bench([&](){ k_h2<1,0,0><<<gb, BLK>>>(gx, gw, gs, go, GN, GK); }, 20);
        printf("  H3 tanpa PRMT+HFMA2 (baca saja) %8.3f ms  %7.1f GB/s  (%.2fx vs H1)\n",
               t3, bytes / (t3 * 1e-3) / 1e9, t1 / t3);
        double t4 = bench([&](){ k_h2<1,1,1><<<gb, BLK>>>(gx, gw, gs, go, GN, GK); }, 20);
        printf("  H4 x ditahapkan sekali          %8.3f ms  %7.1f GB/s  (%.2fx vs H1)\n",
               t4, bytes / (t4 * 1e-3) / 1e9, t1 / t4);
        cudaFree(gw); cudaFree(gs); cudaFree(gx); cudaFree(go);
    }

    cudaFree(dS); cudaFree(dS2); cudaFree(dout); cudaFree(dout2); cudaFree(dvv); cudaFree(dkq);
    return 0;
}
"""
try:
    gd_dir = os.path.join(base, "gdnprobe")
    os.makedirs(gd_dir, exist_ok=True)
    gd_cu = os.path.join(gd_dir, "gdn.cu")
    with open(gd_cu, "w") as f:
        f.write(GDN_SRC)
    nvcc = sh(["bash", "-lc", "command -v nvcc"]).strip() or \
        "/usr/local/cuda/bin/nvcc"
    cg = subprocess.run([nvcc, "-O3", "-arch=sm_75", "-o",
                         os.path.join(gd_dir, "gdn"), gd_cu],
                        capture_output=True, text=True)
    log("   nvcc:", nvcc, "| exit:", cg.returncode)
    if cg.returncode != 0:
        for l in (cg.stderr or "").splitlines()[:20]:
            print("   [nvcc] " + l[:200])
    else:
        rg = subprocess.run([os.path.join(gd_dir, "gdn")],
                            capture_output=True, text=True, timeout=900)
        log("   jalankan exit:", rg.returncode)
        for l in (rg.stdout or "").splitlines()[:45]:
            if l.strip():
                print("   " + l)
        for l in (rg.stderr or "").splitlines()[:8]:
            if l.strip():
                print("   [stderr] " + l[:200])
        shutil.rmtree(gd_dir, ignore_errors=True)
except Exception as e:
    log("   [GAGAL] probe rekurensi GDN:", e)

# Bobot 8,6 GB dan environment Mojo JANGAN ikut ter-commit ke output
# kernel — selain memakan kuota, itu membuat unduhan log jadi raksasa.
if from_dataset:
    # Jangan hapus: ini mount baca-saja milik dataset, bukan salinan unduhan.
    log("pembersihan dilewati (bobot dari mount dataset):", model_dir)
else:
    log("membersihkan:", model_dir)
    shutil.rmtree(model_dir, ignore_errors=True)
if env_dir:
    log("membersihkan:", env_dir)
    shutil.rmtree(env_dir, ignore_errors=True)
sys.exit(0 if ok else 1)
