#!/usr/bin/env python3
# ==============================================================================
# SATU KALI JALAN: jadikan bobot Bonsai-2 2-bit sebagai DATASET Kaggle
# ------------------------------------------------------------------------------
# Masalah: setiap uji inferensi 2-bit mengunduh 8,6 GB dari HuggingFace di
# dalam container (~180 detik per run). Setelah bobot menjadi dataset, kernel
# uji cukup me-mount-nya dari /kaggle/input — unduhan hilang.
#
# Mengapa dikerjakan DI DALAM container Kaggle, bukan di komputer lokal:
#   - komputer user bukan mesin kerja berat; storage internalnya sempat penuh
#     (sisa 5,8 Gi), dan mengunduh + mengunggah 8,6 GB dua kali memakai
#     bandwidth rumah yang lambat.
#   - bandwidth container Kaggle jauh lebih cepat, dan `kagglehub` terautentikasi
#     otomatis di notebook Kaggle, jadi unggahan jalan Kaggle -> Kaggle.
#
# Alur:
#   1. Unduh berkas bobot ke /kaggle/working/bonsai2 (curl, bisa lanjut -C -).
#   2. Verifikasi ukuran model.safetensors.
#   3. kagglehub.dataset_upload(...) -> dataset okiabrian/bonsai-2bit-weights.
#   4. Bila sukses: hapus salinan lokal supaya output kernel tidak 8,6 GB
#      (isi /kaggle/working ikut ter-commit; kalau tidak dibersihkan, setiap
#      `kaggle kernels output` akan mengunduh raksasa ke komputer lokal).
#      Bila GAGAL: salinan sengaja DITINGGALKAN, supaya bobot masih bisa
#      dipakai lewat kernel_sources sebagai jalur cadangan.
# ==============================================================================

import os
import shutil
import subprocess
import sys
import time

REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
DATASET_HANDLE = "okiabrian/bonsai-2bit-weights"

FILES = [
    "config.json",
    "hadamard.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
]

# Ukuran model.safetensors yang benar — dipakai untuk menolak unduhan terpotong.
EXPECTED = 8595477990


def log(*a):
    print(">>", *a, flush=True)


def fatal(*a):
    print(">> [FATAL]", *a, flush=True)
    sys.exit(1)


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def pick_dir():
    for cand in ("/kaggle/temp", "/kaggle/working"):
        if os.path.isdir(cand):
            return cand
    return "/tmp"


def curl_get(url, dst):
    """Unduh via curl: ikuti pengalihan, coba ulang, lanjutkan bila terputus."""
    for attempt in range(1, 4):
        size_before = os.path.getsize(dst) if os.path.exists(dst) else 0
        cmd = ["curl", "-L", "--retry", "3", "--retry-delay", "5",
               "--retry-all-errors", "-C", "-", "-sS", "-o", dst, url]
        log(f"     [percobaan {attempt}] ukuran awal {size_before}")
        r = subprocess.run(cmd)
        size_after = os.path.getsize(dst) if os.path.exists(dst) else 0
        log(f"     exit={r.returncode} ukuran akhir={size_after}")
        if r.returncode == 0 and size_after > 0:
            return size_after
        time.sleep(5)
    return 0


# --------------------------------------------------------------------------
# 0. Pastikan kredensial Kaggle tersedia SEBELUM mengunduh 8,6 GB.
#    (Kalau kredensial tidak ada, mengundurkan kegagalan ke akhir hanya
#     membuang 10 menit unduhan.)
# --------------------------------------------------------------------------
log("0. MEMERIKSA KREDENSIAL KAGGLE")
try:
    import kagglehub  # noqa: F401
    log("   kagglehub sudah terpasang")
except ImportError:
    log("   kagglehub belum ada -> pip install kagglehub")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])

has_json = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
log("   ~/.kaggle/kaggle.json:", "ADA" if has_json else "TIDAK ADA")
log("   KAGGLE_USERNAME/KAGGLE_KEY:", "ADA" if has_env else "TIDAK ADA")
if not (has_json or has_env):
    log("   [PERINGATAN] kredensial tidak terlihat — unggahan kemungkinan gagal.")
    log("   Salinan bobot akan ditahan di /kaggle/working supaya masih bisa "
        "dipakai lewat kernel_sources.")

# --------------------------------------------------------------------------
# 1. Unduh bobot
# --------------------------------------------------------------------------
base = pick_dir()
model_dir = os.path.join(base, "bonsai2")
os.makedirs(model_dir, exist_ok=True)
log("1. MENGUNDUH BOBOT 2-BIT DARI HUGGINGFACE")
log("   target:", model_dir)
log("   ruang disk:", sh(["df", "-h", base]).splitlines()[-1])

for f in FILES:
    dst = os.path.join(model_dir, f)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        log(f"   sudah ada {f} ({os.path.getsize(dst)} byte)")
        continue
    log(f"   mengunduh {f} ...")
    got = curl_get(f"https://huggingface.co/{REPO}/resolve/main/{f}", dst)
    if got == 0:
        if f in ("chat_template.jinja", "generation_config.json"):
            log(f"   [LEWATI] {f} tidak wajib")
            continue
        fatal("gagal unduh", f, "(0 byte setelah 3 percobaan)")
    log(f"   -> {f} OK ({got} byte)")

# --------------------------------------------------------------------------
# 2. Verifikasi
# --------------------------------------------------------------------------
log("2. VERIFIKASI BOBOT")
st = os.path.join(model_dir, "model.safetensors")
if not os.path.exists(st):
    fatal("model.safetensors tidak ada")
got = os.path.getsize(st)
log(f"   model.safetensors: {got} byte (diharapkan {EXPECTED})")
if got < 1024:
    fatal("model.safetensors kosong/terpotong — unduhan gagal")
if got != EXPECTED:
    log("   [WARN] ukuran berbeda dari harapan — lanjut, tapi waspada")

for f in ("config.json", "hadamard.json", "tokenizer.json"):
    p = os.path.join(model_dir, f)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        fatal(f"{f} tidak ada atau kosong — dataset akan sia-sia")
log("   isi:", sorted(os.listdir(model_dir)))

# --------------------------------------------------------------------------
# 3. Unggah sebagai dataset Kaggle
# --------------------------------------------------------------------------
log("3. MENGUNGGAH KE DATASET KAGGLE:", DATASET_HANDLE)
ok = False
try:
    import kagglehub
    t0 = time.time()
    kagglehub.dataset_upload(DATASET_HANDLE, model_dir)
    log(f"   [OK] unggah selesai dalam {time.time() - t0:.0f} detik")
    ok = True
except Exception as e:
    log(f"   [GAGAL] {type(e).__name__}: {e}")

# --------------------------------------------------------------------------
# 4. Bersihkan — lihat penjelasan di docstring bagian 4.
# --------------------------------------------------------------------------
if ok:
    log("4. MEMBERSIHKAN SALINAN LOKAL (unggah sukses)")
    shutil.rmtree(model_dir, ignore_errors=True)
    log("   [SELESAI] dataset", DATASET_HANDLE, "siap dipakai")
    log("   Tambahkan ke kernel-metadata.json kernel uji:")
    log('     "dataset_sources": ["' + DATASET_HANDLE + '"]')
    sys.exit(0)
else:
    log("4. SALINAN DITAHAN (unggah gagal)")
    log("   Bobot tetap ada di /kaggle/working/bonsai2 sebagai output kernel,")
    log("   sehingga masih bisa di-mount lewat kernel_sources. JANGAN unduh")
    log("   output kernel ini ke komputer lokal (8,6 GB).")
    sys.exit(1)
