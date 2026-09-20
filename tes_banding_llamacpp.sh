#!/usr/bin/env bash
# ==============================================================================
# tes_banding_llamacpp.sh — jalankan pembanding llama.cpp di Kaggle T4
#
# Kernel MANDIRI (slug sendiri, tidak memakai / menyentuh kernel produksi
# bonsai-mojo-t4-build, tidak menyinkron dataset bonsai-mojo-source).
# Semua unduhan (GGUF + llama.cpp) dilakukan DI DALAM Kaggle.
#
# Pemakaian:
#   ./tes_banding_llamacpp.sh            # push + tunggu + unduh log
#   ./tes_banding_llamacpp.sh --status   # cek status kernel saja
#   ./tes_banding_llamacpp.sh --no-push  # jangan push; hanya unduh log
#   ./tes_banding_llamacpp.sh --tail     # tampilkan ekor log tanpa menunggu
# ==============================================================================
set -e

export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDIR="$ROOT/banding_llamacpp"
OUTDIR="$ROOT/dist_kaggle_mojo/banding_llamacpp"
SLUG="okiabrian/bonsai-banding-llamacpp"

DO_PUSH=1
ONLY_STATUS=0
ONLY_TAIL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --no-push) DO_PUSH=0; shift ;;
        --status)  ONLY_STATUS=1; shift ;;
        --tail)    ONLY_TAIL=1; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo ">> [ERROR] argumen tidak dikenal: $1" >&2; exit 2 ;;
    esac
done

command -v kaggle >/dev/null 2>&1 || { echo ">> [ERROR] kaggle CLI tidak ada di PATH" >&2; exit 1; }
[ -f ~/.kaggle/access_token ] || { echo ">> [ERROR] ~/.kaggle/access_token tidak ada" >&2; exit 1; }

if [ "$ONLY_STATUS" = "1" ]; then
    kaggle kernels status "$SLUG"
    exit 0
fi

unduh_log() {
    mkdir -p "$OUTDIR"
    python3 - "$SLUG" "$OUTDIR" <<'PYDL'
import json, os, sys, time, urllib.parse, urllib.request
user, slug = sys.argv[1].split("/", 1)
outdir = sys.argv[2]
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
url = "https://www.kaggle.com/api/v1/kernels/output?" + urllib.parse.urlencode(
    {"userName": user, "kernelSlug": slug})
obj = None
for attempt in range(5):
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read().decode())
        try:
            obj = json.loads(d.get("log") or "[]")
        except Exception:
            obj = []
        break
    except Exception as e:
        w = 30 + attempt * 30
        print(f">> [WARN] {e} — tunggu {w}s (percobaan {attempt+1}/5)", flush=True)
        time.sleep(w)
if obj is None:
    print(">> [ERROR] gagal mengunduh log"); sys.exit(1)
json.dump(obj, open(os.path.join(outdir, "banding_log.json"), "w"))
txt = "".join(e.get("data", "") or "" for e in obj)
open(os.path.join(outdir, "banding.log"), "w").write(txt)
print(f">> [OK] log disimpan: {outdir}/banding.log ({len(txt)} char)")
PYDL
}

siapkan_cache_binari() {
    # Build llama.cpp CUDA memakan ~28 menit dari jatah sesi dan tiap run Kaggle
    # mulai dari container kosong. Jadi hasil build diambil dari keluaran kernel,
    # dijadikan dataset `llama-cuda-bin`, lalu ditambahkan ke dataset_sources —
    # run berikutnya langsung memakai binari itu tanpa build lagi.
    local KOUT="$OUTDIR/kernel_output"
    local DDIR="$OUTDIR/dataset_cache"
    rm -rf "$KOUT" "$DDIR"
    mkdir -p "$KOUT"
    if ! kaggle kernels output "$SLUG" -p "$KOUT" >/dev/null 2>&1; then
        echo ">> [INFO] keluaran kernel tidak terunduh — cache binari dilewati"
        return 0
    fi
    if [ ! -f "$KOUT/llama-cuda-bin/bin/llama-cli" ]; then
        echo ">> [INFO] keluaran tidak memuat llama-cuda-bin/bin/llama-cli — cache dilewati"
        return 0
    fi
    # 'bin' ditaruh di akar dataset supaya terpasang sebagai
    # /kaggle/input/llama-cuda-bin/bin/llama-cli
    mkdir -p "$DDIR"
    cp -a "$KOUT/llama-cuda-bin/bin" "$DDIR/bin"
    python3 - "$DDIR" <<'PYDS'
import json, os, sys
d = sys.argv[1]
json.dump({"title": "llama cuda bin",
           "id": "okiabrian/llama-cuda-bin",
           "licenses": [{"name": "other"}]},
          open(os.path.join(d, "dataset-metadata.json"), "w"))
PYDS
    if kaggle datasets create -p "$DDIR" >/dev/null 2>&1; then
        echo ">> [OK] dataset 'okiabrian/llama-cuda-bin' dibuat"
    elif kaggle datasets version -p "$DDIR" -m "perbarui binari llama.cpp CUDA" >/dev/null 2>&1; then
        echo ">> [OK] dataset 'okiabrian/llama-cuda-bin' diperbarui"
    else
        echo ">> [WARN] gagal membuat/memperbarui dataset cache (butuh waktu siap?)"
    fi
    python3 - "$KDIR/kernel-metadata.json" <<'PYMS'
import json, sys
p = sys.argv[1]
m = json.load(open(p))
src = m.setdefault("dataset_sources", [])
if "okiabrian/llama-cuda-bin" in src:
    print(">> [INFO] dataset_sources sudah memuat cache")
else:
    src.append("okiabrian/llama-cuda-bin")
    json.dump(m, open(p, "w"), indent=2)
    print(">> [OK] dataset_sources ditambah: okiabrian/llama-cuda-bin")
PYMS
}

if [ "$ONLY_TAIL" = "1" ]; then
    unduh_log
    tail -60 "$OUTDIR/banding.log"
    exit 0
fi

if [ "$DO_PUSH" = "1" ]; then
    echo ""
    echo "========================================================="
    echo ">> Mengirim kernel mandiri '$SLUG' ke Kaggle..."
    echo "========================================================="
    if ! kaggle kernels push -p "$KDIR"; then
        echo ">> [WARN] push gagal — coba hapus & kirim ulang..."
        kaggle kernels delete "$SLUG" 2>/dev/null || true
        sleep 5
        kaggle kernels push -p "$KDIR"
    fi
    echo ">> [OK] kernel terkirim. Menunggu alokasi instance T4..."
    sleep 20
fi

echo ""
echo "========================================================="
echo ">> Memantau status (Ctrl+C boleh; kernel tetap jalan di Kaggle)"
echo "========================================================="
while true; do
    ST="$(kaggle kernels status "$SLUG" 2>/dev/null || echo Unknown)"
    echo "[$(date +%H:%M:%S)] $ST"
    if echo "$ST" | grep -qi "complete"; then break; fi
    if echo "$ST" | grep -qi "error"; then
        echo ">> [GAGAL] kernel error. Log mentah tetap diunduh di bawah."
        break
    fi
    if echo "$ST" | grep -qi "cancel"; then
        echo ">> [CANCEL] kernel dibatalkan."; exit 1
    fi
    sleep 45
done

echo ""
echo ">> Mengunduh log kernel (jeda finalisasi 30 detik)..."
sleep 30
unduh_log

echo ""
echo ">> Menyiapkan cache binari (build sekali, dipakai berulang)..."
siapkan_cache_binari

echo ""
echo "========================================================="
echo ">> JEJAK PENTING (ringkasan)"
echo "========================================================="
grep -E "^>> \[(OK|FAIL|WARN|CMD)\]" "$OUTDIR/banding.log" | tail -25 || true
echo ""
echo ">> Log lengkap: $OUTDIR/banding.log"
