#!/usr/bin/env bash
# ==============================================================================
# PUSH KERNEL BUILD CPU-ONLY  ->  okiabrian/bonsai-build-cpu
#
# MEMISAHKAN BUILD (CPU) DARI UJI INFERENSI (GPU)
# ------------------------------------------------------------------------------
# Sebelumnya satu kernel mengerjakan semuanya di T4: kompilasi Mojo (berat) DAN
# uji inferensi. Kompilasi sama sekali tidak butuh GPU:
#   - `nvcc -arch=sm_75` adalah CROSS-COMPILER (offline compilation NVIDIA):
#     butuh host compiler, bukan device.
#   - `mojo build` adalah compiler; `gpu-arch = "sm_75"` di mojoproject.toml
#     hanya deklarasi target, tidak membuka device.
# Jadi kuota GPU terpakai untuk pekerjaan yang tidak memerlukannya.
#
# SEKARANG:
#   [1] KERNEL INI (CPU-only, kuota GPU = 0)
#         menjalankan run_deploy.py + deploy_on_kaggle.sh yang SAMA PERSIS
#         dengan kernel GPU, lalu menulis
#         /kaggle/working/mojo_build_cache.tar.gz
#   [2] KERNEL GPU  okiabrian/bonsai-mojo-t4-build   (lewat ./push_to_kaggle.sh)
#         kernel_sources = ["okiabrian/bonsai-build-cpu"]
#         -> me-restore cache dari [1] (pixi_cache + cache Mojo + ccache),
#            sehingga build menjadi hampir instan, lalu §5 menjalankan uji
#            inferensi T4 yang sesungguhnya.
#
# Isi kernel IDENTIK dengan kernel GPU. Yang berbeda HANYA metadata:
#   enable_gpu: false, tanpa machine_shape, tanpa dataset bobot 4,5 GB
#   (uji inferensi §5 memang di-skip di CPU karena tidak ada nvidia-smi).
# run_deploy.py dan deploy_on_kaggle.sh TIDAK diubah satu baris pun.
#
# PENTING — push TANPA argumen `acc`:
#   Docstring Kaggle CLI 2.2.4: "acc ... overrides boolean settings for
#   GPU/TPU found in the metadata file." Jadi `acc="NvidiaTeslaT4"` (yang
#   dipakai push_to_kaggle.sh untuk kernel GPU) akan MEMAKSA instance GPU dan
#   membatalkan `enable_gpu: false` di kernel ini.
#
# Pemakaian:
#   ./kaggle_cpu_build/push_cpu_build.sh
#   BONSAI_SKIP_DATASET_SYNC=1 ./kaggle_cpu_build/push_cpu_build.sh  # dataset sudah sinkron
#   BONSAI_CPU_SKIP_DOWNLOAD=1 ./kaggle_cpu_build/push_cpu_build.sh  # tak perlu unduh cache
# ==============================================================================
set -e

# Pastikan PATH menyertakan biner conda kaggle
export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

DATASET_SLUG="okiabrian/bonsai-mojo-source"
CPU_KERNEL_SLUG="okiabrian/bonsai-build-cpu"
GPU_KERNEL_SLUG="okiabrian/bonsai-mojo-t4-build"
OUTPUT_DIR="/Users/macmini/.mounty/SSD_External/fix/dist_kaggle_cpu"
STAGE_DIR="/tmp/bonsai_cpu_kernel"

echo "========================================================="
echo " 1. MEMERIKSA KAGGLE CLI & METADATA"
echo "========================================================="
if ! command -v kaggle >/dev/null 2>&1; then
    echo ">> [ERROR] Perintah 'kaggle' tidak ditemukan di PATH." >&2
    exit 1
fi

for f in "dataset-metadata.json" "$SCRIPT_DIR/kernel-metadata.json" "run_deploy.py"; do
    if [ ! -f "$f" ]; then
        echo ">> [ERROR] File wajib tidak ditemukan: $f" >&2
        exit 1
    fi
done

echo ">> [OK] Kaggle CLI dan metadata terverifikasi."
echo ">> [INFO] Kernel CPU   : $CPU_KERNEL_SLUG (enable_gpu=false)"
echo ">> [INFO] Kernel GPU   : $GPU_KERNEL_SLUG"
echo ">> [INFO] Output lokal : $OUTPUT_DIR"

# ------------------------------------------------------------------------------
# 2. SINKRONISASI SOURCE CODE KE KAGGLE DATASET
#    Blok ini SAMA dengan push_to_kaggle.sh §2 supaya kedua kernel selalu
#    melihat source yang identik. Bisa dilewati bila dataset baru saja disinkron.
# ------------------------------------------------------------------------------
if [ "${BONSAI_SKIP_DATASET_SYNC:-0}" = "1" ]; then
    echo ""
    echo ">> [SKIP] BONSAI_SKIP_DATASET_SYNC=1 -> sinkronisasi dataset dilewati."
else
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

    # --- Teruskan variabel lingkungan BONSAI_* ke container -------------------
    # PENTING: variabel yang diset di shell LOKAL tidak otomatis sampai ke
    # kernel Kaggle — env shell lokal tidak ikut terkirim, dan run_deploy.py
    # tidak menyuntikkan env apa pun. Tanpa blok ini, perintah seperti
    # `BONSAI_COH_THINK=both ./push_cpu_build.sh` akan DIAM-DIAM mengabaikan
    # variabelnya dan build berjalan dengan default (hasilnya negatif palsu).
    # Solusi: bekukan di sini, lalu run_deploy.py men-source-nya di container.
    # Catatan: `case` TIDAK boleh ditaruh di dalam $( ... ) di sini — bash 3.2
    # (bawaan macOS) salah-parse `)` dari pola `BONSAI_*)` sebagai penutup $( ).
    ENV_SNAPSHOT="/tmp/bonsai_env_snapshot.$$"
    env > "$ENV_SNAPSHOT"

    ENV_FILE="$PKG_DIR/bonsai_env.sh"
    {
        echo "# Dibuat otomatis oleh kaggle_cpu_build/push_cpu_build.sh pada $TIMESTAMP"
        echo "# Di-source oleh run_deploy.py di dalam container, sebelum deploy_on_kaggle.sh."
        echo "# Jangan disunting tangan - isinya mengikuti environment shell saat push."
    } > "$ENV_FILE"

    while IFS='=' read -r _k _v; do
        case "$_k" in
            BONSAI_*)
                printf 'export %s=%q\n' "$_k" "$_v" >> "$ENV_FILE"
                ;;
        esac
    done < "$ENV_SNAPSHOT"

    echo ">> [ENV] Variabel BONSAI_* yang diteruskan ke container (bonsai_env.sh):"
    N_BONSAI=0
    while IFS='=' read -r _k _v; do
        case "$_k" in
            BONSAI_*)
                printf '>>        %s=%s\n' "$_k" "$_v"
                N_BONSAI=$((N_BONSAI + 1))
                ;;
        esac
    done < "$ENV_SNAPSHOT"
    rm -f "$ENV_SNAPSHOT"

    if [ "$N_BONSAI" -eq 0 ]; then
        echo ">>        (tidak ada) -> container memakai default."
        echo ">>        Contoh: BONSAI_COH_THINK=both ./kaggle_cpu_build/push_cpu_build.sh"
    fi

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
        kaggle datasets version -p "$PKG_DIR" -r tar -m "Auto sync (cpu build): $TIMESTAMP" -d
    else
        echo ">> Dataset belum ada, membuat dataset baru ($DATASET_SLUG)..."
        kaggle datasets create -p "$PKG_DIR" -r tar
    fi

    rm -rf "$PKG_DIR"
    echo ">> [OK] Source code bersih berhasil diunggah."
    echo ">> Memberikan jeda awal 45 detik agar server Kaggle mencatat versi dataset baru..."
    sleep 45

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
fi

echo ""
echo "========================================================="
echo " 3. PUSH & JALANKAN KERNEL BUILD DI KAGGLE (CPU-ONLY)"
echo "========================================================="
# Kernel ini memakai run_deploy.py dari root repo TANPA modifikasi. Salin ke
# folder staging bersama metadata-nya, supaya `run_deploy.py` tetap satu sumber
# kebenaran (tidak ada salinan yang bisa basi di dalam repo).
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$SCRIPT_DIR/kernel-metadata.json" "$STAGE_DIR/kernel-metadata.json"
cp "$REPO_DIR/run_deploy.py" "$STAGE_DIR/run_deploy.py"
echo ">> [OK] Staging: $STAGE_DIR (kernel-metadata.json + run_deploy.py)"

cd "$STAGE_DIR"
python3 - <<'PY'
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
    print(f">> [INFO] enable_gpu={meta.get('enable_gpu')} "
          f"machine_shape={meta.get('machine_shape')} "
          f"kernel_sources={meta.get('kernel_sources')}", flush=True)

    def _push():
        # PENTING: argumen ke-3 (`acc`) SENGAJA TIDAK DIPAKAI di sini.
        # Docstring Kaggle CLI 2.2.4: "acc ... overrides boolean settings for
        # GPU/TPU found in the metadata file." acc="NvidiaTeslaT4" akan
        # MEMAKSA instance GPU dan membatalkan enable_gpu=false kernel ini.
        res = api.kernels_push(".")
        for attr in ("invalidTags", "invalidDatasetSources",
                     "invalidCompetitionSources", "invalidKernelSources"):
            bad = getattr(res, attr, None)
            if bad:
                print(f">> [WARN] Kaggle menolak {attr}: {bad}", flush=True)
        return res

    try:
        res = _push()
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
        res = _push()
        print(f">> [OK] Kernel berhasil di-trigger: {res}", flush=True)

except Exception as e:
    print("\n" + "=" * 65, flush=True)
    print(">> [DETAIL ERROR DARI SERVER KAGGLE]:", flush=True)
    print(f"   Tipe Error : {type(e).__name__}", flush=True)
    print(f"   Pesan      : {e}", flush=True)
    print("=" * 65 + "\n", flush=True)
    sys.exit(1)
PY
cd "$REPO_DIR"

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
    STATUS_OUT=$(kaggle kernels status "$CPU_KERNEL_SLUG" 2>/dev/null || echo "Unknown")
    CURRENT_TIME=$(date +"%H:%M:%S")
    echo "[$CURRENT_TIME] Status: $STATUS_OUT"

    if echo "$STATUS_OUT" | grep -qi "complete"; then
        echo ""
        echo "========================================================="
        echo " 5. BUILD CPU SELESAI - MENGUNDUH CACHE & ARTEFAK"
        echo "========================================================="
        if [ "${BONSAI_CPU_SKIP_DOWNLOAD:-0}" = "1" ]; then
            echo ">> [SKIP] BONSAI_CPU_SKIP_DOWNLOAD=1 -> unduh dilewati."
            echo ">>        Kernel GPU akan membaca cache langsung dari mount"
            echo ">>        kernel_sources ['$CPU_KERNEL_SLUG'], jadi unduh ini opsional."
            break
        fi
        mkdir -p "$OUTPUT_DIR"
        echo ">> Mengunduh artefak via endpoint legacy..."
        # Beri jeda finalisasi output oleh Kaggle
        sleep 45
        export KAGGLE_KERNEL_SLUG="$CPU_KERNEL_SLUG"
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

# Yang paling penting adalah mojo_build_cache.tar.gz: itulah yang membuat kernel
# GPU cukup me-restore cache, bukan mengompilasi Mojo dari nol.
# bonsai_infer + libbonsai_qmv_sm75.so berguna untuk jalur Modal (unduh -> unggah).
WANTED = ("mojo_build_cache.tar.gz", "bonsai_infer", "libbonsai_qmv_sm75.so")
targets = {}

for attempt in range(5):
    try:
        page = None
        for _ in range(12):
            params = {"userName": user, "kernelSlug": slug}
            if page:
                params["pageToken"] = page
            r = requests.get(url, params=params, headers=h, timeout=90)
            if r.status_code == 429:
                raise RuntimeError("429 rate limit")
            r.raise_for_status()
            d = r.json()
            for f in (d.get("files") or []):
                p = f.get("path") or f.get("pathNullable") or f.get("fileName") or ""
                u = f.get("urlNullable") or f.get("url")
                if not p or not u:
                    continue
                base = os.path.basename(p)
                if not (base.endswith(".whl") or base in WANTED):
                    continue
                # Utamakan entri tingkat-atas (tanpa "/") bila ada duplikat.
                if base not in targets or ("/" not in p and "/" in targets[base][1]):
                    targets[base] = (u, p)
            page = d.get("nextPageToken") if (d.get("hasNextPageToken") or d.get("hasNextPage")) else None
            if not page:
                break
            if set(WANTED).issubset(targets) and any(b.endswith(".whl") for b in targets):
                break
        break
    except Exception as e:
        w = 60 + attempt * 30
        print(f">> [WARN] {e} — tunggu {w} detik (percobaan {attempt+1}/5)", flush=True)
        time.sleep(w)

if not targets:
    print(">> [WARN] Tidak ada file yang bisa diunduh dari output kernel", flush=True)
    sys.exit(1)

for name in sorted(targets):
    u, _p = targets[name]
    dest = os.path.join(outdir, name)
    r = requests.get(u, timeout=900)
    r.raise_for_status()
    with open(dest, "wb") as fh:
        fh.write(r.content)
    print(f">> [OK] {name} -> {dest} ({len(r.content)} bytes)", flush=True)

missing = [n for n in WANTED if n not in targets]
if missing:
    print(f">> [WARN] Tidak ada di output kernel: {', '.join(missing)}", flush=True)
    print(">>        (normal bila nvcc tidak tersedia di instance CPU — "
          "libbonsai_qmv_sm75.so akan dikompilasi oleh kernel GPU)", flush=True)
PYDL
        then
            DOWNLOAD_SUCCESS=true
        fi

        if [ "$DOWNLOAD_SUCCESS" = true ]; then
            echo ""
            echo ">> [SUKSES] Artefak berhasil diunduh ke: $OUTPUT_DIR"
            ls -lh "$OUTPUT_DIR"
        else
            echo ">> [WARN] Unduh otomatis terkena rate-limit. Anda dapat mengunduh langsung dari web Kaggle."
            echo ">>        Tidak fatal: kernel GPU membaca cache dari mount kernel_sources."
        fi
        break
    elif echo "$STATUS_OUT" | grep -qi "error"; then
        echo ""
        echo "========================================================="
        echo ">> [GAGAL] Build CPU menghasilkan error di server Kaggle."
        echo "========================================================="
        mkdir -p "$OUTPUT_DIR"
        # Ambil log lewat endpoint legacy (Bearer token). `kaggle kernels output`
        # berulang kali timeout / Exit 137 untuk kernel yang error, sehingga
        # akar masalahnya tidak pernah terbaca — jalur ini menghindari itu.
        export KAGGLE_KERNEL_SLUG="$CPU_KERNEL_SLUG"
        export KAGGLE_OUT_DIR="$OUTPUT_DIR"
        python3 - <<'PYLOG' || true
import json, os, requests

user, slug = os.environ["KAGGLE_KERNEL_SLUG"].split("/", 1)
outdir = os.environ["KAGGLE_OUT_DIR"]
os.makedirs(outdir, exist_ok=True)
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
h = {"Authorization": f"Bearer {tok}"}
r = requests.get("https://www.kaggle.com/api/v1/kernels/output",
                 params={"userName": user, "kernelSlug": slug}, headers=h, timeout=90)
r.raise_for_status()
d = r.json()
try:
    log = json.loads(d.get("log") or "[]")
except Exception:
    log = []
path = os.path.join(outdir, "cpu_build_log.json")
with open(path, "w") as fh:
    json.dump(log, fh)
lines = [l.get("data", "") if isinstance(l, dict) else str(l) for l in log]
print(f">> [LOG] {len(lines)} baris disimpan ke {path}", flush=True)
print(">> ---- 60 baris terakhir ----", flush=True)
for l in lines[-60:]:
    print(l, end="" if l.endswith("\n") else "\n")
PYLOG
        exit 1
    elif echo "$STATUS_OUT" | grep -qi "cancel"; then
        echo ">> [CANCEL] Kernel dibatalkan di server Kaggle."
        exit 1
    fi

    sleep 60
done

echo ""
echo "========================================================="
echo ">> [SELESAI] BUILD CPU SELESAI"
echo "========================================================="
echo ">> Langkah berikutnya — jalankan uji inferensi di GPU:"
echo ">>     ./push_to_kaggle.sh"
echo ">> Kernel GPU ($GPU_KERNEL_SLUG) memakai"
echo ">>     kernel_sources = ['$CPU_KERNEL_SLUG']"
echo ">> sehingga ia me-restore cache di atas, bukan mengompilasi dari nol."
