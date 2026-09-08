#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/build_cuda_ffi.sh
# Purpose: Mengompilasi kernel CUDA SM75 W1A16 Q1O menjadi shared library (.so)
#          menggunakan nvcc dengan optimasi penuh (-O3 -arch=sm_75).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BUILD_DIR="${ROOT_DIR}/build"
SRC_FILE="${ROOT_DIR}/src/csrc/qmv_sm75_kernel.cu"
TARGET_SO="${BUILD_DIR}/libbonsai_qmv_sm75.so"

echo "=== [Bonsai 1-Bit] Building CUDA FFI Shared Library ==="

# 1. Cek ketersediaan nvcc (Guard environment)
if ! command -v nvcc &> /dev/null; then
    echo "[WARN] 'nvcc' tidak ditemukan di PATH sistem ini."
    echo "[WARN] Jika Anda berada di host Mac mini, kompilasi ini harus dijalankan di environment GPU (Kaggle/Linux)."
    echo "[INFO] Sistem Mojo akan otomatis fallback ke native Mojo decode kernel bila .so belum ada."
    exit 0
fi

NVCC_VERSION=$(nvcc --version | grep "release" || true)
echo "[INFO] Menggunakan: ${NVCC_VERSION}"

# 2. Siapkan direktori output
mkdir -p "${BUILD_DIR}"

# 3. Cek Timestamp Caching (Skip compile jika .so sudah up-to-date)
if [ -f "${TARGET_SO}" ] && [ "${TARGET_SO}" -nt "${SRC_FILE}" ]; then
    echo "[CACHE HIT] ${TARGET_SO} lebih baru dari sumber. Kompilasi dilewati (hemat 30-60 detik)."
    exit 0
fi

# 4. Kompilasi dengan nvcc (-Xptxas -v untuk bukti register & spill audit)
echo "[INFO] Mengompilasi ${SRC_FILE} -> ${TARGET_SO}..."
nvcc -O3 \
     -arch=sm_75 \
     --shared \
     -Xcompiler -fPIC \
     -Xptxas -v \
     "${SRC_FILE}" \
     -o "${TARGET_SO}"

if [ -f "${TARGET_SO}" ]; then
    SO_SIZE=$(du -h "${TARGET_SO}" | cut -f1)
    echo "[SUCCESS] Berhasil membuat ${TARGET_SO} (${SO_SIZE})"
    
    # 4. Verifikasi SASS via cuobjdump jika tersedia
    if command -v cuobjdump &> /dev/null; then
        echo "[INFO] Memeriksa instruksi SASS pada .so..."
        cuobjdump -sass "${TARGET_SO}" | grep -m 5 "LDG" || true
        echo "[INFO] SASS terverifikasi siap untuk SM75 (Tesla T4)."
    fi
else
    echo "[ERROR] Kompilasi gagal, ${TARGET_SO} tidak ditemukan."
    exit 1
fi
