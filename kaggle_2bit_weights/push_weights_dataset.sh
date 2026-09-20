#!/usr/bin/env bash
# ==============================================================================
# JALANKAN KERNEL PENGUNGGAH BOBOT  ->  okiabrian/bonsai-2bit-weights-fetch
#
# Satu kali jalan saja. Setelah dataset okiabrian/bonsai-2bit-weights ada,
# tambahkan ke kernel-metadata.json kernel uji:
#     "dataset_sources": ["okiabrian/bonsai-2bit-weights"]
# dan kernel uji tidak perlu mengunduh 8,6 GB dari HuggingFace lagi.
#
# Pemakaian:
#   ./kaggle_2bit_weights/push_weights_dataset.sh
# ==============================================================================
set -e

export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

KERNEL_SLUG="okiabrian/bonsai-2bit-weights-fetch"
OUTPUT_DIR="/Users/macmini/.mounty/SSD_External/fix/dist_kaggle_weights"

echo "========================================================="
echo " 1. MEMERIKSA KAGGLE CLI & METADATA"
echo "========================================================="
if ! command -v kaggle >/dev/null 2>&1; then
    echo ">> [ERROR] Perintah 'kaggle' tidak ditemukan di PATH." >&2
    exit 1
fi
for f in "$SCRIPT_DIR/kernel-metadata.json" "$SCRIPT_DIR/make_weights_dataset.py"; do
    if [ ! -f "$f" ]; then
        echo ">> [ERROR] File wajib tidak ditemukan: $f" >&2
        exit 1
    fi
done
echo ">> [OK] Metadata terverifikasi. Kernel: $KERNEL_SLUG (CPU, internet aktif)"

echo ""
echo "========================================================="
echo " 2. PUSH & JALANKAN KERNEL"
echo "========================================================="
cd "$SCRIPT_DIR"
python3 - <<'PY'
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

try:
    api = KaggleApi()
    api.authenticate()
    print(">> [OK] Kernel ter-trigger:", api.kernels_push("."), flush=True)
except Exception as e:
    print(">> [ERROR] %s: %s" % (type(e).__name__, e), flush=True)
    sys.exit(1)
PY
cd "$REPO_DIR"

echo ""
echo "========================================================="
echo " 3. PEMANTAUAN STATUS"
echo "========================================================="
mkdir -p "$OUTPUT_DIR"

# Pemantauan lewat API Python, bukan CLI `kaggle` (binary CLI pernah menggantung
# di mesin ini). Status enum dibandingkan lewat rsplit('.') — pernah putus
# lebih awal karena dibandingkan langsung sebagai string berawalan nama enum.
KERNEL_SLUG="$KERNEL_SLUG" OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PY'
import json, os, sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

slug = os.environ["KERNEL_SLUG"]
out_dir = os.environ["OUTPUT_DIR"]
api = KaggleApi(); api.authenticate()

status = "UNKNOWN"
for _ in range(60):
    raw = str(getattr(api.kernels_status(slug), "status", "UNKNOWN"))
    status = raw.rsplit(".", 1)[-1] if "." in raw else raw
    print("[%s] Status: %s" % (time.strftime("%H:%M:%S"), status), flush=True)
    if status.lower() not in ("running", "queued", "pending", "starting"):
        break
    time.sleep(30)

print(">> [INFO] Status akhir:", status, flush=True)

# Hanya log stdout yang diambil — BUKAN output kernel. Kalau unggahan gagal,
# skrip sengaja menahan bobot 8,6 GB di output; mengunduhnya akan membanjiri
# disk komputer ini.
try:
    body = str(api.kernels_logs(slug))
    parts = []
    for line in body.splitlines():
        line = line.lstrip(",").strip()
        if not line.startswith("{"):
            continue
        try:
            parts.append(json.loads(line).get("data", ""))
        except Exception:
            pass
    txt = "".join(parts)
    dst = os.path.join(out_dir, "weights_fetch_log.txt")
    open(dst, "w", encoding="utf-8").write(txt)
    print(">> [OK] Log tersimpan:", dst, "(%d byte)" % len(txt), flush=True)
except Exception as e:
    print(">> [WARN] unduh log gagal:", type(e).__name__, e, flush=True)

print(">> ---- 40 baris terakhir ----", flush=True)
for l in txt.splitlines()[-40:]:
    print(l, flush=True)

sys.exit(0 if status.lower() == "complete" else 1)
PY
echo ">> [SELESAI]"
