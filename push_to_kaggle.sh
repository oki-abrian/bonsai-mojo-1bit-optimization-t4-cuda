#!/usr/bin/env bash
# ==============================================================================
# Script Otomasi Sinkronisasi & Deployment Kaggle (CLI) untuk Mojo T4
# 1. Update/Create Private Dataset Source Code di Kaggle (okiabrian/bonsai-mojo-source)
# 2. Push & Trigger Kernel Build di CPU Kaggle (okiabrian/bonsai-mojo-t4-build)
# 3. Live Polling Status & Auto-Download Hasil Artefak Build
# ==============================================================================
set -e

# Pastikan PATH menyertakan biner conda kaggle
export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_SLUG="okiabrian/bonsai-mojo-source"
KERNEL_SLUG="okiabrian/bonsai-mojo-t4-build"
OUTPUT_DIR="/Users/macmini/.mounty/SSD_External/fix/dist_kaggle_mojo"

echo "========================================================="
echo " 1. MEMERIKSA KAGGLE CLI & METADATA"
echo "========================================================="
if ! command -v kaggle >/dev/null 2>&1; then
    echo ">> [ERROR] Perintah 'kaggle' tidak ditemukan di PATH." >&2
    exit 1
fi

if [ ! -f "dataset-metadata.json" ] || [ ! -f "kernel-metadata.json" ]; then
    echo ">> [ERROR] dataset-metadata.json atau kernel-metadata.json tidak ditemukan." >&2
    exit 1
fi

echo ">> [OK] Kaggle CLI dan metadata terverifikasi."

echo ""
echo "========================================================="
echo " 2. SINKRONISASI SOURCE CODE KE KAGGLE DATASET"
echo "========================================================="
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

PKG_DIR="/tmp/mojo_kaggle_pkg"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR"

# Salin metadata dataset dan file pemicu
cp dataset-metadata.json "$PKG_DIR/"
cp deploy_on_kaggle.sh "$PKG_DIR/"
cp main.mojo "$PKG_DIR/"
cp run_deploy.py "$PKG_DIR/"
cp pixi.toml mojoproject.toml "$PKG_DIR/"
cp README.md "$PKG_DIR/" 2>/dev/null || touch "$PKG_DIR/README.md"
cp -r src "$PKG_DIR/"
cp -r tests "$PKG_DIR/"
cp -r benchmarks "$PKG_DIR/"
cp -r scripts "$PKG_DIR/" 2>/dev/null || true

echo ">> Mengompres source code bersih (tanpa ._* dan file metadata macOS)..."
export COPYFILE_DISABLE=1
tar --exclude="._*" \
    --exclude=".DS_Store" \
    --exclude=".git" \
    --exclude="__pycache__" \
    --exclude="build" \
    --exclude="dist" \
    --exclude=".magic" \
    --exclude="references" \
    --exclude=".pixi" \
    --exclude="dist_kaggle_mojo" \
    -czf "$PKG_DIR/mojo_source.tar.gz" .

# Coba perbarui versi dataset yang sudah ada (-r tar untuk direktori src & tests, -d untuk hapus snapshot usang)
if kaggle datasets status "$DATASET_SLUG" >/dev/null 2>&1; then
    echo ">> Memperbarui versi dataset ($DATASET_SLUG)..."
    kaggle datasets version -p "$PKG_DIR" -r tar -m "Auto sync: $TIMESTAMP" -d
else
    echo ">> Dataset belum ada, membuat dataset baru ($DATASET_SLUG)..."
    kaggle datasets create -p "$PKG_DIR" -r tar
fi

rm -rf "$PKG_DIR"
echo ">> [OK] Source code bersih berhasil diunggah."
echo ">> Memberikan jeda awal 45 detik agar server Kaggle mencatat versi dataset baru..."
sleep 45

echo ""
echo "========================================================="
echo " 3. PUSH & JALANKAN KERNEL BUILD DI KAGGLE (CPU)"
echo "========================================================="
echo ">> Menunggu server Kaggle memproses dataset (status ready)..."
for i in {1..30}; do
    DS_STATUS=$(kaggle datasets status "$DATASET_SLUG" 2>/dev/null || echo "")
    if echo "$DS_STATUS" | grep -qi "ready"; then
        echo ">> [OK] Dataset siap: $DS_STATUS"
        echo ">> Memberikan jeda propagasi mount backend Kaggle (90 detik)..."
        sleep 90
        break
    fi
    echo ">> Status dataset saat ini: ${DS_STATUS:-processing}... menunggu 10 detik..."
    sleep 10
done

echo ">> Mengirim kernel ke Kaggle..."
python - <<'PY'
import sys
import json
import time

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    with open("kernel-metadata.json") as f:
        meta = json.load(f)
    kernel_id = meta.get("id")
    print(f">> [INFO] Menyiapkan kernel: {kernel_id}", flush=True)

    try:
        res = api.kernels_push(".", None, "NvidiaTeslaT4")
        print(f">> [OK] Kernel berhasil di-trigger: {res}", flush=True)
    except Exception as push_err:
        print(f">> [WARN] Push awal mengalami konflik ({push_err}).", flush=True)
        print(f">> [CLEANUP] Menghapus kernel lama '{kernel_id}' di server Kaggle...", flush=True)
        try:
            api.kernel_delete(kernel_id)
            print(f">> [OK] Kernel lama berhasil dihapus. Menunggu 5 detik...", flush=True)
            time.sleep(5)
        except Exception as del_err:
            print(f">> [INFO] Status delete: {del_err}", flush=True)

        print(f">> [RETRY] Mengirim ulang kernel '{kernel_id}'...", flush=True)
        res = api.kernels_push(".", None, "NvidiaTeslaT4")
        print(f">> [OK] Kernel berhasil di-trigger: {res}", flush=True)

except Exception as e:
    print("\n" + "=" * 65, flush=True)
    print(">> [DETAIL ERROR DARI SERVER KAGGLE]:", flush=True)
    print(f"   Tipe Error : {type(e).__name__}", flush=True)
    print(f"   Pesan      : {e}", flush=True)
    print("=" * 65 + "\n", flush=True)
    sys.exit(1)
PY

if [ $? -ne 0 ]; then
    echo ">> [STOP] Eksekusi dihentikan karena kernel push gagal." >&2
    exit 1
fi

echo ""
echo "========================================================="
echo " 4. LIVE MONITORING STATUS EKSEKUSI (CPU KAGGLE)"
echo "========================================================="
echo ">> Menunggu Kaggle mengalokasikan instance CPU..."
sleep 15

echo ">> Memulai pemantauan status build..."
echo ">> Tekan Ctrl+C kapan saja jika ingin membiarkan berjalan di background."

mkdir -p "$OUTPUT_DIR"

while true; do
    STATUS_OUT=$(kaggle kernels status "$KERNEL_SLUG" 2>/dev/null || echo "Unknown")
    CURRENT_TIME=$(date +"%H:%M:%S")
    echo "[$CURRENT_TIME] Status: $STATUS_OUT"

    if echo "$STATUS_OUT" | grep -qi "complete"; then
        echo ""
        echo "========================================================="
        echo " 5. BUILD SELESAI - MENGUNDUH ARTEFAK & LOG"
        echo "========================================================="
        mkdir -p "$OUTPUT_DIR"
        echo ">> Mengunduh paket hasil build via endpoint legacy (hanya .whl + log)..."
        # Beri jeda finalisasi output oleh Kaggle
        sleep 45
        export KAGGLE_KERNEL_SLUG="$KERNEL_SLUG"
        export KAGGLE_OUT_DIR="$OUTPUT_DIR"
        DOWNLOAD_SUCCESS=false
        # Endpoint RPC baru (ListKernelSessionOutput) 429 menetap; endpoint
        # legacy /api/v1/kernels/output berfungsi (Bearer token) dan memberi
        # URL ter-signed kaggleusercontent.com tanpa rate limit API.
        if python3 - <<'PYDL'
import json, os, sys, time, requests
user, slug = os.environ["KAGGLE_KERNEL_SLUG"].split("/", 1)
outdir = os.environ["KAGGLE_OUT_DIR"]
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
h = {"Authorization": f"Bearer {tok}"}
url = "https://www.kaggle.com/api/v1/kernels/output"
os.makedirs(outdir, exist_ok=True)
whl_url, log_obj = None, None
for attempt in range(5):
    try:
        page = None
        for _ in range(40):
            params = {"userName": user, "kernelSlug": slug}
            if page:
                params["pageToken"] = page
            r = requests.get(url, params=params, headers=h, timeout=90)
            if r.status_code == 429:
                raise RuntimeError("429 rate limit")
            r.raise_for_status()
            d = r.json()
            if log_obj is None:
                try:
                    log_obj = json.loads(d.get("log") or "[]")
                except Exception:
                    log_obj = []
            for f in (d.get("files") or []):
                p = f.get("path") or f.get("pathNullable") or f.get("fileName") or ""
                if "/" not in p and p.endswith(".whl"):
                    whl_url = f.get("urlNullable") or f.get("url")
            page = d.get("nextPageToken") if (d.get("hasNextPageToken") or d.get("hasNextPage")) else None
            if not page or whl_url:
                break
        break
    except Exception as e:
        w = 60 + attempt * 30
        print(f">> [WARN] {e} — tunggu {w} detik (percobaan {attempt+1}/5)", flush=True)
        time.sleep(w)
if log_obj is not None:
    with open(os.path.join(outdir, "build_log.json"), "w") as f:
        json.dump(log_obj, f)
    print(">> [OK] Log build disimpan: dist_kaggle_mojo/build_log.json", flush=True)
if whl_url:
    r = requests.get(whl_url, timeout=180)
    p = os.path.join(outdir, "bonsai_1bit_t4-0.1.0-py3-none-any.whl")
    open(p, "wb").write(r.content)
    print(f">> [OK] .whl diunduh: {p} ({len(r.content)} bytes)", flush=True)
else:
    print(">> [WARN] .whl tidak ditemukan di output kernel", flush=True)
    sys.exit(1)
PYDL
then
    DOWNLOAD_SUCCESS=true
fi

        if [ "$DOWNLOAD_SUCCESS" = true ]; then
            echo ""
            echo ">> [SUKSES] Artefak berhasil diunduh ke: $OUTPUT_DIR"
            ls -lh "$OUTPUT_DIR"
        else
            echo ">> [WARN] Unduh otomatis terkena rate-limit. Anda dapat mengunduh langsung dari web Kaggle atau menjalankan ulang skrip beberapa menit lagi."
        fi
        break
    elif echo "$STATUS_OUT" | grep -qi "error"; then
        echo ""
        echo "========================================================="
        echo ">> [GAGAL] Build menghasilkan error di server Kaggle."
        echo "========================================================="
        mkdir -p "$OUTPUT_DIR"
        kaggle kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR" --file-pattern ".*\.log" --page-size 200 2>/dev/null || true
        exit 1
    elif echo "$STATUS_OUT" | grep -qi "cancel"; then
        echo ">> [CANCEL] Kernel dibatalkan di server Kaggle."
        exit 1
    fi

    sleep 60
done
