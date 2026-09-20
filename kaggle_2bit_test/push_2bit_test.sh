#!/usr/bin/env bash
# ==============================================================================
# KIRIM & JALANKAN UJI INFERENSI 2-BIT DI GPU KAGGLE (T4)
#
# Kernel: okiabrian/bonsai-2bit-infer
#   - Mengambil biner hasil build CPU dari output kernel bonsai-build-cpu.
#   - Mengunduh bobot 2-bit dari HuggingFace DI DALAM container (bukan lokal).
#   - Menjalankan bonsai_infer dengan BONSAI_BITS=2.
#
# Berbeda dari push_to_kaggle.sh: skrip ini TIDAK menyentuh source dataset dan
# TIDAK menjalankan deploy_on_kaggle.sh — murni uji jalur 2-bit.
#
# Pemakaian:
#   ./kaggle_2bit_test/push_2bit_test.sh
# ==============================================================================
set -e

export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

KERNEL_SLUG="okiabrian/bonsai-2bit-infer"
OUTPUT_DIR="/Users/macmini/.mounty/SSD_External/fix/dist_kaggle_2bit"

echo "========================================================="
echo " 1. MEMERIKSA KAGGLE CLI & METADATA"
echo "========================================================="
if ! command -v kaggle >/dev/null 2>&1; then
    echo ">> [ERROR] Perintah 'kaggle' tidak ditemukan di PATH." >&2
    exit 1
fi
for f in "$SCRIPT_DIR/kernel-metadata.json" "$SCRIPT_DIR/run_infer_2bit.py"; do
    if [ ! -f "$f" ]; then
        echo ">> [ERROR] File wajib tidak ditemukan: $f" >&2
        exit 1
    fi
done
# CATATAN: memanggil binary `kaggle` untuk sekadar mencetak versi pernah
# menggantung di mesin ini; versi dibaca lewat modul Python-nya saja.
KAGGLE_VER=$(python3 -c "import kaggle,sys;print(getattr(kaggle,'__version__','tidak diketahui'))" 2>/dev/null || echo "tidak diketahui")
echo ">> [OK] Modul kaggle versi $KAGGLE_VER"
echo ">> [INFO] Kernel: $KERNEL_SLUG (GPU T4, internet aktif)"

echo ""
echo "========================================================="
echo " 2. PUSH & JALANKAN KERNEL"
echo "========================================================="
cd "$SCRIPT_DIR"
python3 - <<'PY'
import json
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    meta = json.load(open("kernel-metadata.json"))
    print(">> [INFO] Menyiapkan kernel:", meta.get("id"), flush=True)
    print(">> [INFO] enable_gpu=%s machine_shape=%s kernel_sources=%s" % (
        meta.get("enable_gpu"), meta.get("machine_shape"),
        meta.get("kernel_sources")), flush=True)

    def _push():
        res = api.kernels_push(".")
        for attr in ("invalidTags", "invalidDatasetSources",
                     "invalidCompetitionSources", "invalidKernelSources"):
            bad = getattr(res, attr, None)
            if bad:
                print(">> [WARN] Kaggle menolak %s: %s" % (attr, bad), flush=True)
        return res

    try:
        print(">> [OK] Kernel ter-trigger:", _push(), flush=True)
    except Exception as push_err:
        print(">> [WARN] Push konflik (%s). Menghapus kernel lama..." % push_err, flush=True)
        try:
            api.kernel_delete(meta.get("id"))
            import time
            time.sleep(5)
        except Exception as del_err:
            print(">> [INFO] Status delete:", del_err, flush=True)
        print(">> [RETRY] Mengirim ulang...", flush=True)
        print(">> [OK] Kernel ter-trigger:", _push(), flush=True)

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

# Pemantauan juga lewat API Python, bukan CLI `kaggle`, karena pemanggilan
# binary CLI pernah menggantung di mesin ini.
KERNEL_SLUG="$KERNEL_SLUG" OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PY'
import json
import os
import time

from kaggle.api.kaggle_api_extended import KaggleApi

slug = os.environ["KERNEL_SLUG"]
out_dir = os.environ["OUTPUT_DIR"]
api = KaggleApi()
api.authenticate()

status = "UNKNOWN"
for _ in range(120):
    raw = str(getattr(api.kernels_status(slug), "status", "UNKNOWN"))
    status = raw.rsplit(".", 1)[-1] if "." in raw else raw
    print("[%s] Status: %s" % (time.strftime("%H:%M:%S"), status), flush=True)
    if status.lower() not in ("running", "queued", "pending", "starting"):
        break
    time.sleep(60)

print(">> [INFO] Status akhir:", status, flush=True)

# Hanya log stdout yang diambil — bukan seluruh output kernel. Isi output bisa
# memuat bobot 8,6 GB bila pembersihan di akhir skrip gagal, dan mengunduhnya
# akan memenuhi disk komputer ini.
try:
    body = api.kernels_logs(slug)
    parts = []
    for line in str(body).splitlines():
        line = line.lstrip(",").strip()
        if not line.startswith("{"):
            continue
        try:
            parts.append(json.loads(line).get("data", ""))
        except Exception:
            pass
    txt = "".join(parts)
    dst = os.path.join(out_dir, "2bit_log.txt")
    open(dst, "w", encoding="utf-8").write(txt)
    print(">> [OK] Log tersimpan:", dst, "(%d byte)" % len(txt), flush=True)
except Exception as e:
    print(">> [WARN] unduh log gagal:", type(e).__name__, e, flush=True)

sys_code = 0 if status.lower() == "complete" else 1
raise SystemExit(sys_code)
PY
echo ">> [SELESAI] UJI 2-BIT"
