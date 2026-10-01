#!/usr/bin/env bash
# ==============================================================================
# PUSH KERNEL DUMP AKTIVASI 2-BIT  ->  okiabrian/bonsai-act-dump
#
# Berkas baru (aditif) — tidak mengubah deploy_on_kaggle.sh / run_deploy.py /
# push_to_kaggle.sh / kaggle_cpu_build/push_cpu_build.sh.
#
# Bobot 2-bit dari dataset okiabrian/bonsai-2bit-weights (ter-mount, tanpa
# unduhan 8,6 GB). Biner dari output kernel CPU okiabrian/bonsai-build-cpu.
#
# Pemakaian:
#   ./kaggle_act_dump/push_act_dump.sh
# ==============================================================================
set -e

export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="/tmp/bonsai_act_dump_kernel"
KERNEL_SLUG="okiabrian/bonsai-act-dump"
OUTPUT_DIR="${SCRIPT_DIR}/out"

if ! command -v kaggle >/dev/null 2>&1; then
    echo ">> [ERROR] Perintah 'kaggle' tidak ditemukan di PATH." >&2
    exit 1
fi

echo ">> Menyiapkan staging: $STAGE_DIR"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
cp "$SCRIPT_DIR/kernel-metadata.json" "$STAGE_DIR/"
cp "$SCRIPT_DIR/run_act_dump.py" "$STAGE_DIR/"

cd "$STAGE_DIR"
python3 - <<'PY'
import json, sys, time
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    meta = json.load(open("kernel-metadata.json"))
    kid = meta.get("id")
    print(f">> [INFO] Menyiapkan kernel: {kid}", flush=True)
    print(f">> [INFO] enable_gpu={meta.get('enable_gpu')} "
          f"machine_shape={meta.get('machine_shape')} "
          f"internet={meta.get('enable_internet')}", flush=True)
    print(f">> [INFO] dataset_sources={meta.get('dataset_sources')}", flush=True)
    print(f">> [INFO] kernel_sources={meta.get('kernel_sources')}", flush=True)

    def _push():
        res = api.kernels_push(".")
        for attr in ("invalidTags", "invalidDatasetSources",
                     "invalidKernelSources", "invalidModelSources"):
            bad = getattr(res, attr, None)
            if bad:
                print(f">> [WARN] Kaggle menolak {attr}: {bad}", flush=True)
        return res

    try:
        res = _push()
    except Exception as e:
        print(f">> [WARN] Push awal konflik ({e}). Menghapus kernel lama...", flush=True)
        try:
            api.kernel_delete(kid)
            time.sleep(5)
        except Exception as de:
            print(f">> [INFO] Status delete: {de}", flush=True)
        res = _push()
    print(f">> [OK] Kernel ter-trigger: {res}", flush=True)
except Exception as e:
    print(">> [ERROR]", type(e).__name__, e, flush=True)
    sys.exit(1)
PY

echo ""
echo ">> Memantau status (Ctrl+C untuk membiarkan berjalan di background)..."
sleep 20
for i in $(seq 1 150); do
    S=$(kaggle kernels status "$KERNEL_SLUG" 2>/dev/null || echo "Unknown")
    echo "[$(date +%H:%M:%S)] $S"
    case "$S" in
        *COMPLETE*|*ERROR*|*CANCEL*)
            echo "========================================================="
            echo " SELESAI - mengunduh log + output"
            echo "========================================================="
            mkdir -p "$OUTPUT_DIR"
            kaggle kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR" || \
                echo ">> [WARN] unduh output gagal"
            break
            ;;
    esac
    sleep 60
done
