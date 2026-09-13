#!/usr/bin/env bash
# ==============================================================================
# cek_infer.sh — Jalankan inferensi Kaggle lalu tampilkan hasilnya sebagai TEKS
#
# Pemakaian:
#   ./cek_infer.sh                 # baca log terbaru, detokenisasi -> teks
#   ./cek_infer.sh --push          # push & jalankan kernel Kaggle dulu, lalu detok
#   ./cek_infer.sh --raw           # tampilkan juga tag <|im_end|> / <think>
#   ./cek_infer.sh --log <path>    # detok log tertentu
#   ./cek_infer.sh --run 2         # hanya run ke-2
#
# Opsi setelah '--' diteruskan apa adanya ke detok_infer.py.
# ==============================================================================
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV="/Users/macmini/.workbuddy-ai/binaries/python/envs/default"
MANAGED_PY="/Users/macmini/.workbuddy-ai/binaries/python/versions/3.13.12/bin/python3"

# --- 1. Siapkan interpreter yang punya modul 'tokenizers' ---
PY=""
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import tokenizers" >/dev/null 2>&1; then
    PY="$VENV/bin/python"
elif python3 -c "import tokenizers" >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo ">> [SETUP] Menyiapkan lingkungan Python + modul 'tokenizers'..."
    BASE_PY="$MANAGED_PY"
    [ -x "$BASE_PY" ] || BASE_PY="$(command -v python3)"
    "$BASE_PY" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip >/dev/null 2>&1 || true
    "$VENV/bin/pip" install -q tokenizers
    PY="$VENV/bin/python"
fi
echo ">> [PY] $PY"

# --- 2. Opsional: push & jalankan kernel Kaggle dulu ---
DO_PUSH=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--push" ]; then DO_PUSH=1; else ARGS+=("$a"); fi
done

if [ "$DO_PUSH" = "1" ]; then
    echo ""
    echo "========================================================="
    echo ">> Menjalankan push_to_kaggle.sh (build + inferensi T4)"
    echo ">> Ini memakan waktu ~15-20 menit dan kuota GPU Kaggle."
    echo "========================================================="
    bash "$ROOT/push_to_kaggle.sh"
    echo ""
    echo ">> Kernel selesai. Melanjutkan ke detokenisasi log hasilnya..."
fi

# --- 3. Detokenisasi ---
exec "$PY" "$ROOT/detok_infer.py" "${ARGS[@]}"
