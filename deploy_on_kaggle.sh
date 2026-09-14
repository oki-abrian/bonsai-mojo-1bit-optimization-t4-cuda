#!/usr/bin/env bash
# ==============================================================================
# Script Build & Verifikasi Bonsai-1Bit-T4 di Kaggle (CPU Environment)
# - Menghemat kuota GPU Kaggle (berjalan di instance CPU gratis)
# - Melakukan cross-compilation target T4 (sm_75)
# - Menggunakan persistent caching di /kaggle/working untuk recompile kilat
# ==============================================================================
set -e

echo "========================================================="
echo " 1. INISIALISASI DIRECTORY & CACHE PERSISTEN"
echo "========================================================="
WORKING="/kaggle/working"
REPO_DIR="$WORKING/bonsai-1bit-t4-mojo"
CACHE_DIR="$WORKING/.cache_t4_build"
DIST_DIR="$WORKING/dist"

# Deteksi lingkungan: di Mac lokal /kaggle tidak ada (read-only) ->
# pakai folder kerja lokal; repo = direktori tempat script dijalankan.
if [ ! -d "/kaggle" ]; then
    WORKING="$(pwd)/.kaggle_local_working"
    CACHE_DIR="$WORKING/.cache_t4_build"
    DIST_DIR="$WORKING/dist"
    echo ">> [INFO] Lingkungan LOKAL terdeteksi (bukan container Kaggle)."
    echo ">> [INFO] Working diarahkan ke: $WORKING"
    if [ -f "$(pwd)/pixi.toml" ]; then
        REPO_DIR="$(pwd)"
    fi
fi

mkdir -p "$CACHE_DIR" "$DIST_DIR"

# Persistent caching sederhana: archive cache hidup di /kaggle/working.
# Awal run: decompress + hapus kompresinya (dipakai langsung sebagai folder).
CACHE_READY=0      # 1 = isi CACHE_DIR layak diarsipkan (restore sukses / tak ada arsip)
CACHE_ARCHIVED=0   # 1 = arsip sudah pernah ditulis pada run ini

if [ -f "$WORKING/mojo_build_cache.tar.gz" ]; then
    echo ">> [CACHE] Me-restore cache dari $WORKING/mojo_build_cache.tar.gz..."
    if tar -xzf "$WORKING/mojo_build_cache.tar.gz" -C "$WORKING"; then
        rm -f "$WORKING/mojo_build_cache.tar.gz"
        CACHE_READY=1
        echo ">> [CACHE] Restore selesai, archive terkompresi dihapus."
    else
        # Jangan arsipkan hasil ekstraksi yang setengah jadi: itu akan menimpa
        # arsip lama dengan arsip rusak. Buang sisa ekstraksi lalu mulai bersih,
        # supaya run ini membuat arsip BARU yang sehat (sembuh sendiri) alih-alih
        # gagal restore terus-menerus di setiap run berikutnya.
        echo ">> [WARN] Restore cache GAGAL — sisa ekstraksi dibuang, arsip akan dibuat ulang."
        rm -rf "$CACHE_DIR"
        mkdir -p "$CACHE_DIR"
        CACHE_READY=1
    fi
else
    CACHE_READY=1
    echo ">> [CACHE] Tidak ada arsip cache sebelumnya — mulai dari kosong."
fi

# -------------------------------------------------------------------------
# PENYIMPANAN CACHE — dulu hanya di paling akhir skrip. Karena skrip pakai
# `set -e`, satu kegagalan di tengah (mis. langkah uji yang rapuh) membuat arsip
# TIDAK pernah ditulis, sehingga run berikutnya selalu memakai arsip lama dan
# cache "terasa tidak menolong". Sekarang arsip ditulis:
#   (a) segera setelah build selesai (pemanggilan save_build_cache di §4), dan
#   (b) lewat trap EXIT sebagai jaring pengaman apa pun yang terjadi.
# -------------------------------------------------------------------------
save_build_cache() {
    # $1 = "1" untuk memaksa tulis ulang (refresh); selain itu hanya bila belum.
    local force="${1:-0}"
    if [ "$CACHE_READY" != "1" ]; then return 0; fi
    if [ ! -d "$CACHE_DIR" ]; then return 0; fi
    if [ "$CACHE_ARCHIVED" = "1" ] && [ "$force" != "1" ]; then return 0; fi
    echo ">> Menyimpan archive cache kompilasi inkremental ke $WORKING/mojo_build_cache.tar.gz..."
    # Tulis ke nama sementara lalu pindahkan: kalau tar gagal di tengah (mis.
    # disk penuh) kita TIDAK meninggalkan arsip parsial yang akan membuat
    # restore gagal terus di run berikutnya.
    local tmp_ar="$WORKING/.mojo_build_cache.tmp.tar.gz"
    rm -f "$tmp_ar"
    if tar -czf "$tmp_ar" -C "$WORKING" ".cache_t4_build" 2>/dev/null; then
        mv -f "$tmp_ar" "$WORKING/mojo_build_cache.tar.gz"
        CACHE_ARCHIVED=1
        echo ">> [OK] Ukuran cache tersimpan: $(du -sh "$WORKING/mojo_build_cache.tar.gz" 2>/dev/null | cut -f1)"
    else
        rm -f "$tmp_ar"
        echo ">> [WARN] Gagal menulis arsip cache — dilewati (run tetap lanjut)."
    fi
    return 0
}

cleanup_cache_dir() {
    # Output kernel harus RINGAN: hapus cache ter-uncompress + env pixi.
    # Ribuan file di output adalah penyebab listing output macet/429.
    # Aman di sini karena seluruh pemakaian pixi sudah selesai.
    rm -rf "$CACHE_DIR" "$REPO_DIR/.pixi" 2>/dev/null || true
}

# Trap EXIT: arsipkan bila belum pernah, lalu bersihkan. Kode keluar ASLI
# dipertahankan supaya kegagalan tetap terbaca sebagai kegagalan.
on_exit_archive_cache() {
    local rc=$?
    save_build_cache 0 || true
    cleanup_cache_dir || true
    exit $rc
}
trap on_exit_archive_cache EXIT

export PIXI_HOME="$CACHE_DIR/pixi"
export PIXI_CACHE_DIR="$CACHE_DIR/pixi_cache"
export MAGIC_HOME="$CACHE_DIR/magic"
export MODULAR_HOME="$CACHE_DIR/modular"
export MOJO_CACHE_DIR="$CACHE_DIR/mojo"
export CCACHE_DIR="$CACHE_DIR/ccache"
export CCACHE_MAXSIZE="5G"
export PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$MAGIC_HOME/bin:$PATH"

# -------------------------------------------------------------------------
# ccache untuk nvcc.
# DULU: CCACHE_DIR di-export tetapi nvcc dipanggil LANGSUNG tanpa prefix
# `ccache`, jadi ccache tidak pernah dipakai sama sekali (konfigurasi hampa).
# SEKARANG: ccache dideteksi, dipasang bila perlu (binernya disimpan ke dalam
# cache supaya tidak diulang tiap run), dan benar-benar dipakai di langkah nvcc.
# -------------------------------------------------------------------------
CCACHE_BIN=""
if command -v ccache >/dev/null 2>&1; then
    CCACHE_BIN="$(command -v ccache)"
elif [ -x "$CACHE_DIR/bin/ccache" ]; then
    CCACHE_BIN="$CACHE_DIR/bin/ccache"
else
    echo ">> [CCACHE] Belum tersedia — mencoba memasang sekali (biner disimpan ke cache)..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get install -y -qq ccache >/dev/null 2>&1 || true
    fi
    if command -v ccache >/dev/null 2>&1; then
        CCACHE_BIN="$(command -v ccache)"
        mkdir -p "$CACHE_DIR/bin"
        cp "$CCACHE_BIN" "$CACHE_DIR/bin/ccache" 2>/dev/null || true
        echo ">> [CCACHE] Terpasang dan disimpan ke cache: $CCACHE_BIN"
    fi
fi

if [ -n "$CCACHE_BIN" ]; then
    # Hash ISI compiler, bukan mtime+size: mtime biner toolchain berubah tiap
    # sesi Kaggle, sehingga mode default membuat cache selalu dianggap invalid.
    export CCACHE_COMPILERCHECK="content"
    export CCACHE_NLEVELS="4"
    mkdir -p "$CCACHE_DIR"
    echo ">> [CCACHE] Aktif: $CCACHE_BIN ($("$CCACHE_BIN" --version 2>/dev/null | head -1))"
    "$CCACHE_BIN" -s 2>/dev/null | head -5 || true
else
    echo ">> [CCACHE] Tidak tersedia — nvcc dipanggil langsung (tanpa cache)."
fi

echo ">> [OK] Direktori kerja: $REPO_DIR"
echo ">> [OK] Direktori cache persisten: $CACHE_DIR"
echo ">> [OK] Direktori output: $DIST_DIR"

echo ""
echo "========================================================="
echo " 2. CEK & INSTALASI TOOLCHAIN MODULAR / PIXI"
echo "========================================================="
if command -v pixi >/dev/null 2>&1; then
    echo ">> [CACHE HIT] Pixi CLI siap di PATH: $(pixi --version 2>/dev/null || echo 'tersedia')"
elif [ -f "$PIXI_HOME/bin/pixi" ]; then
    echo ">> [CACHE HIT] Pixi ditemukan di $PIXI_HOME/bin/pixi"
    export PATH="$PIXI_HOME/bin:$PATH"
elif [ -f "$HOME/.pixi/bin/pixi" ]; then
    echo ">> [CACHE HIT] Pixi ditemukan di $HOME/.pixi/bin/pixi"
    export PATH="$HOME/.pixi/bin:$PATH"
else
    echo ">> [CACHE MISS] Mengunduh dan menginstal Pixi ke $PIXI_HOME..."
    mkdir -p "$PIXI_HOME/bin"
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
    if [ -f "$HOME/.pixi/bin/pixi" ]; then
        cp "$HOME/.pixi/bin/pixi" "$PIXI_HOME/bin/" 2>/dev/null || true
    fi
fi

if command -v pixi >/dev/null 2>&1; then
    echo ">> [OK] Pixi CLI siap: $(pixi --version 2>/dev/null || true)"
else
    echo ">> [WARN] Binary pixi belum terdeteksi langsung di PATH, memeriksa status modular/magic..."
fi

echo ""
echo "========================================================="
echo " 3. VERIFIKASI REFERENSI MATEMATIKA BONSAI-27B"
echo "========================================================="
cd "$REPO_DIR"
python3 tests/verify_differential.py
python3 benchmarks/bench_bonsai_layer.py

echo ""
echo "========================================================="
echo " 3b. KOMPILASI KERNEL CUDA SM75 SHARED LIBRARY (NVCC)"
echo "========================================================="
export PATH="/usr/local/cuda/bin:$PATH"
export BONSAI_CUDA_LIB="$WORKING/libbonsai_qmv_sm75.so"
export LD_LIBRARY_PATH="$WORKING:$REPO_DIR:$REPO_DIR/build:$WORKING/build:${LD_LIBRARY_PATH:-}"

if command -v nvcc >/dev/null 2>&1 && [ -f "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" ]; then
    echo ">> [NVCC] Ditemukan: $(nvcc --version | grep 'release' || true)"
    mkdir -p "$WORKING/build" "$REPO_DIR/build"
    
    # Recompile shared library dengan constructor RTLD_NODELETE
    echo ">> [NVCC] Mengompilasi $REPO_DIR/src/csrc/qmv_sm75_kernel.cu -> $WORKING/libbonsai_qmv_sm75.so..."
    # Lewat ccache bila tersedia (lihat blok [CCACHE] di §1).
    NVCC_CMD=(nvcc)
    if [ -n "$CCACHE_BIN" ]; then
        NVCC_CMD=("$CCACHE_BIN" nvcc)
        echo ">> [NVCC] Dijalankan lewat ccache: ${CCACHE_BIN} nvcc"
    fi
    "${NVCC_CMD[@]}" -O3 -arch=sm_75 --shared -Xcompiler -fPIC -Xptxas -v \
        "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" \
        -o "$WORKING/libbonsai_qmv_sm75.so"
    echo ">> [OK] libbonsai_qmv_sm75.so berhasil dibuat ($(du -h "$WORKING/libbonsai_qmv_sm75.so" | cut -f1))"
    if [ -n "$CCACHE_BIN" ]; then
        # Statistik ini yang membuktikan cache benar-benar HIT/MISS — jangan
        # diasumsikan dari konfigurasi saja.
        echo ">> [CCACHE] Statistik setelah kompilasi nvcc:"
        "$CCACHE_BIN" -s 2>/dev/null | head -6 || true
    fi
    cp "$WORKING/libbonsai_qmv_sm75.so" "$REPO_DIR/libbonsai_qmv_sm75.so" 2>/dev/null || true
    cp "$WORKING/libbonsai_qmv_sm75.so" "$REPO_DIR/build/libbonsai_qmv_sm75.so" 2>/dev/null || true
    cp "$WORKING/libbonsai_qmv_sm75.so" "$DIST_DIR/libbonsai_qmv_sm75.so" 2>/dev/null || true

    if command -v cuobjdump >/dev/null 2>&1; then
        echo ">> [SASS] Verifikasi instruksi SM75 LDG pada libbonsai_qmv_sm75.so:"
        cuobjdump -sass "$WORKING/libbonsai_qmv_sm75.so" | grep -m 5 "LDG" || true
    fi

    # Gerbang verifikasi: smoke-test FFI & primary context sharing
    if command -v pixi >/dev/null 2>&1 && [ -f "$REPO_DIR/tests/test_cuda_ffi_smoketest.mojo" ]; then
        echo ">> [SMOKE-TEST] Menjalankan smoke test primary context & FFI..."
        pixi run mojo run -I . "$REPO_DIR/tests/test_cuda_ffi_smoketest.mojo" || echo ">> [WARN] Smoke test gagal — fallback aktif"
    fi
else
    echo ">> [WARN] nvcc tidak tersedia atau qmv_sm75_kernel.cu tidak ditemukan — fallback ke Mojo native"
fi

echo ""
echo "========================================================="
echo " 4. CROSS-COMPILATION TARGET T4 (sm_75) DI CPU"
echo "========================================================="
# `mojo package src` dan `mojo build src/__init__.mojo` sengaja TIDAK dijalankan:
# keduanya mengompilasi ulang seluruh paket dari nol padahal artefaknya tidak
# dipakai binary inferensi. Cukup satu kompilasi di langkah 4a (main.mojo).
if command -v pixi >/dev/null 2>&1; then
    echo ">> [BUILD] Versi compiler yang dipakai: $(pixi run mojo --version 2>/dev/null || echo 'tidak diketahui')"
elif command -v magic >/dev/null 2>&1; then
    echo ">> [BUILD] Versi compiler (magic): $(magic run mojo --version 2>/dev/null || echo 'tidak diketahui')"
fi

# -------------------------------------------------------------------------
# 4a. Kompilasi SATU-SATUNYA: entry CLI inferensi native (main.mojo).
#     `mojo package src` dihapus karena hasilnya tidak dipakai binary dan
#     menambah satu kompilasi penuh lagi.
# -------------------------------------------------------------------------
BONSAI_BIN=""
if command -v pixi >/dev/null 2>&1; then
    echo ">> [BUILD] Kompilasi main.mojo (CLI inferensi native)..."
    rm -f "$WORKING/bonsai_infer"
    if pixi run mojo build -I . main.mojo -o "$WORKING/bonsai_infer"; then
        echo ">> [OK] main.mojo terkompilasi: $WORKING/bonsai_infer"
        BONSAI_BIN="$WORKING/bonsai_infer"
    else
        echo ">> [WARN] main.mojo gagal dikompilasi — wheel terbit tanpa binary"
    fi
fi

# -------------------------------------------------------------------------
# Build Python Wheel (.whl) langsung di container Kaggle
# -------------------------------------------------------------------------
echo ">> [WHL] Membuat paket Python Wheel (.whl) di $WORKING..."
python3 - <<'EOF'
import os, zipfile

pkg_name = "bonsai_1bit_t4"
version = "0.1.0"
whl_name = f"{pkg_name}-{version}-py3-none-any.whl"
whl_path = os.path.join("/kaggle/working", whl_name)
dist_info = f"{pkg_name}-{version}.dist-info"

metadata = f"""Metadata-Version: 2.1
Name: {pkg_name}
Version: {version}
Summary: High-Performance W1A16 Quantized Matmul & Decode for NVIDIA T4 in Mojo
Author: PrismML Eng & Antigravity
"""

wheel = """Wheel-Version: 1.0
Generator: inline_builder
Root-Is-Purelib: true
Tag: py3-none-any
"""

with zipfile.ZipFile(whl_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Wheel = paket uji mandiri: src + tests + manifest toolchain (untuk Modal)
    names = []
    for top in ("src", "tests"):
        for root, dirs, files in os.walk(top):
            for f in files:
                p = os.path.join(root, f)
                arc = os.path.join(pkg_name, p)
                zf.write(p, arc)
                names.append(arc)
    for extra in ("main.mojo", "pixi.toml", "mojoproject.toml"):
        if os.path.exists(extra):
            arc = os.path.join(pkg_name, extra)
            zf.write(extra, arc)
            names.append(arc)
    # Binary inferensi hasil kompilasi Kaggle (Linux x86_64) — Modal hanya
    # mengekstrak dan menjalankannya (tanpa kompilasi).
    bin_path = os.path.join("/kaggle/working", "bonsai_infer")
    if os.path.exists(bin_path):
        arc = os.path.join(pkg_name, "bin", "bonsai_infer")
        zf.write(bin_path, arc)
        names.append(arc)
        print(f">> [OK] Binary bonsai_infer masuk wheel ({os.path.getsize(bin_path)} bytes)")
    else:
        print(">> [WARN] $WORKING/bonsai_infer tidak ditemukan — wheel TANPA binary")
    so_path = os.path.join("/kaggle/working", "libbonsai_qmv_sm75.so")
    if os.path.exists(so_path):
        arc = os.path.join(pkg_name, "libbonsai_qmv_sm75.so")
        zf.write(so_path, arc)
        names.append(arc)
        print(f">> [OK] Shared lib libbonsai_qmv_sm75.so masuk wheel ({os.path.getsize(so_path)} bytes)")
    dist_files = [
        f"{dist_info}/METADATA", f"{dist_info}/WHEEL", f"{dist_info}/top_level.txt",
    ]
    for d in dist_files:
        names.append(d)
    # RECORD (hash kosong = valid untuk pip; wajib agar pip install tidak error)
    record = "\n".join(n + ",," for n in names) + f"\n{dist_info}/RECORD,,\n"
    zf.writestr(f"{dist_info}/RECORD", record)
    zf.writestr(f"{dist_info}/METADATA", metadata)
    zf.writestr(f"{dist_info}/WHEEL", wheel)
    zf.writestr(f"{dist_info}/top_level.txt", f"{pkg_name}\n")

print(f">> [OK] Wheel berhasil dibuat: {whl_path} ({os.path.getsize(whl_path)} bytes)")
EOF

# Salin sumber dan metadata ke dist folder sebagai paket distribusi
tar -czf "$DIST_DIR/bonsai_1bit_t4_src.tar.gz" src/ main.mojo pixi.toml mojoproject.toml README.md scripts/

# -------------------------------------------------------------------------
# 4b. Salin binary hasil build ke dist (arsip hasil build Kaggle)
# -------------------------------------------------------------------------
if [ -f "$WORKING/bonsai_infer" ]; then
    cp "$WORKING/bonsai_infer" "$DIST_DIR/bonsai_infer"
    echo ">> [OK] Binary tersalin ke $DIST_DIR/bonsai_infer"
fi

# -------------------------------------------------------------------------
# 4c. Menjalankan self-test Mojo (non-fatal; hasil tampil di log kernel)
# -------------------------------------------------------------------------
# Audit codegen GPU (hipotesis "codegen Mojo 2x lebih lambat dari CUDA"):
# dump resource-usage per fungsi device — register & stack. Stack > 0 =
# spill local-memory = bukti keras codegen; REG jangan > 255.
if command -v cuobjdump >/dev/null 2>&1 && [ -f "$WORKING/bonsai_infer" ]; then
    echo ">> [SASS] Audit resource usage device code (register/spill)..."
    if cuobjdump --dump-resource-usage --elf "$WORKING/bonsai_infer" > "$DIST_DIR/sass_resource.txt" 2>&1; then
        grep -E "Function.*REG:|REG:" "$DIST_DIR/sass_resource.txt" | head -12 || true
        n_spill=$(grep -cE "STACK: (0x[1-9a-fA-F]|[1-9])" "$DIST_DIR/sass_resource.txt" || true)
        echo ">> [SASS] Fungsi dengan STACK > 0 (spill): $n_spill (detail: $DIST_DIR/sass_resource.txt)"
    else
        echo ">> [SASS] cuobjdump gagal membaca binary Mojo — perlu jalur verifikasi lain"
        head -3 "$DIST_DIR/sass_resource.txt" || true
    fi
else
    echo ">> [SASS] cuobjdump tidak tersedia di container ini — audit dilewati"
fi
# Debug diferensial kernel GPU decode SELALU jalan (gerbang korektness):
# kernel device tidak pernah divalidasi selftest lama (yang hanya host-sim).
echo ">> [TEST] Differential kernel GPU decode vs FP64 CPU..."
export BONSAI_CUDA_LIB="$WORKING/libbonsai_qmv_sm75.so"
export LD_LIBRARY_PATH="$WORKING:$REPO_DIR:$REPO_DIR/build:$WORKING/build:$DIST_DIR:$REPO_DIR/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
pixi run mojo run -I . tests/selftest_decode_gpu.mojo || echo ">> [WARN] selftest_decode_gpu GAGAL — LIHAT HASIL DI ATAS"
pixi run mojo run -I . tests/test_qmm_gpu.mojo || echo ">> [WARN] test_qmm_gpu GAGAL — LIHAT HASIL DI ATAS"
# Gerbang korektness RoPE (paritas MLX traditional=False half-split) + Argmax
# GPU: RoPE salah pairing dulu lolos semua tes matmul tanpa pernah terdeteksi.
pixi run mojo run -I . tests/test_rope_gpu.mojo || echo ">> [WARN] test_rope_gpu GAGAL — LIHAT HASIL DI ATAS"
pixi run mojo run -I . tests/test_argmax_gpu.mojo || echo ">> [WARN] test_argmax_gpu GAGAL — LIHAT HASIL DI ATAS"

# Gerbang korektness PRESISI STATE GDN (fix fp16 -> fp32).
# Kernel delta-rule sebelumnya sama sekali tidak punya test numerik. Test ini
# membandingkan state rekuren FP32 di GPU vs referensi FP64 selama 1/64/512
# langkah. State FP32 harus ~1e-6; kalau kembali ke FP16 melonjak ~1000x.
# Ini penjaga langsung terhadap regresi yang kita perbaiki.
echo ">> [TEST] Presisi state rekurensi GDN (fp32) vs FP64..."
if ! pixi run mojo run -I . tests/test_gdn_state_precision.mojo; then
    echo ">> [WARN] test_gdn_state_precision GAGAL — state GDN mungkin kembali FP16!"
    if [ "$BONSAI_GDN_STATE_FATAL" = "1" ]; then
        echo ">> [FATAL] BONSAI_GDN_STATE_FATAL=1 -> hentikan deploy."
        exit 1
    fi
fi

# Diagnostik harness di T4: apakah `ctx.enqueue_function` menghormati offset pada
# argumen pointer device? Kalau TIDAK, maka prefill M>1 memakai data sampah untuk
# t>=1 (layer.mojo baris 484 & 523) -> kandidat kuat akar degenerasi.
# Non-fatal; tujuannya MELAPORKAN verdict di T4. Ini penentu, bukan opsional.
echo ">> [TEST] Diagnostik offset pointer device (enqueue_function)..."
pixi run mojo run -I . tests/test_enqueue_offset.mojo || echo ">> [WARN] test_enqueue_offset gagal — lihat verdict di atas"

# Self-test lainnya dimatikan secara default di Kaggle: setiap `mojo run`
# mengompilasi ulang seluruh paket dari nol (3 file = 3 kompilasi penuh).
if [ "$BONSAI_RUN_SELFTEST" = "1" ] && command -v pixi >/dev/null 2>&1; then
    echo ">> [TEST] Menjalankan self-test Mojo (import absolut src.*)..."
    for t in tests/selftest_sm75.mojo tests/test_bonsai_shapes.mojo tests/test_qwen3_5_architecture.mojo; do
        echo ">> [TEST] $t"
        pixi run mojo run -I . "$t" || echo ">> [WARN] $t gagal — periksa output di atas"
    done
else
    echo ">> [SKIP] Self-test lainnya dilewati (set BONSAI_RUN_SELFTEST=1)."
fi

echo ">> [OK] Hasil build di $DIST_DIR:"
ls -lh "$DIST_DIR"

# -------------------------------------------------------------------------
# SIMPAN CACHE DI SINI — build sudah selesai, sedangkan §5 (uji inferensi,
# profiling, gerbang) panjang dan rapuh. Kalau arsip baru ditulis di akhir,
# satu kegagalan di §5 menghapus seluruh manfaat cache untuk run berikutnya.
# Trap EXIT tetap menjadi jaring pengaman bila bagian ini tidak tercapai.
# -------------------------------------------------------------------------
save_build_cache 0

# -------------------------------------------------------------------------
# 5. UJI INFERENSI LANGSUNG DI GPU KAGGLE (T4) — bukan di Modal.
#    Binary hasil build dijalankan terhadap bobot asli prism-ml/Bonsai-27B.
# -------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && [ -f "$WORKING/bonsai_infer" ]; then
    echo ""
    echo "========================================================="
    echo " 5. UJI INFERENSI DI GPU KAGGLE ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))"
    echo "========================================================="
    # Bobot diasumsikan terpasang dari Kaggle Dataset (okiabrian/bonsai-27b-mlx-1bit)
    # di /kaggle/input — TANPA mengunduh ulang 4.9GB setiap run.
    KMODEL=""
    for cand in /kaggle/input/*/bonsai-27b-mlx-1bit /kaggle/input/bonsai-27b-mlx-1bit /kaggle/input/datasets/okiabrian/bonsai-27b-mlx-1bit; do
        if [ -f "$cand/model.safetensors.index.json" ] || [ -f "$cand/model.safetensors" ]; then
            KMODEL="$cand"; break
        fi
    done
    if [ -z "$KMODEL" ]; then
        echo ">> [WARN] Dataset bobot tidak ditemukan di /kaggle/input — uji inferensi dilewati."
        echo ">> [INFO] Daftarkan 'okiabrian/bonsai-27b-mlx-1bit' di dataset_sources kernel-metadata.json."
        KMODEL="/tmp/bonsai_model"
    else
        echo ">> [T4] Bobot ditemukan (tanpa unduh): $KMODEL"
    fi

    # 5.0 TES TOKENIZER — verifikasi ID khusus thd checkpoint ITU SENDIRI.
    # Latar: src/tokenizer.py dulu memakai 151643/151644/151645, yaitu rentang
    # Qwen2.5/Qwen3.0 (vocab 151936). Model ini vocab 248320 sehingga
    # is_stop_token() tidak pernah menyala di EOS asli. Tes ini menahan
    # regresi itu: ID kelas harus ikut config.json, bukan ingatan generasi lama.
    echo ">> [TOK] 5.0 tes tokenizer vs config checkpoint"
    if python3 - "$KMODEL" "$REPO_DIR" <<'PYEOF'
import json, os, sys
mdir, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo)
fail = []

cfg = json.load(open(os.path.join(mdir, "config.json")))
tcfg = cfg.get("text_config", cfg)
vocab = tcfg["vocab_size"]
eos = tcfg.get("eos_token_id")
tc = json.load(open(os.path.join(mdir, "tokenizer_config.json")))
print("   checkpoint: vocab=%s eos=%s eos_token=%r" % (vocab, eos, tc.get("eos_token")))

from src.tokenizer import QwenTokenizer
t = QwenTokenizer(mdir)
tok = t.hf_tokenizer
print("   kelas: eos=%s im_start=%s im_end=%s" % (t.eos_token_id, t.im_start_token_id, t.im_end_token_id))

# 1. ID kelas harus mengikuti checkpoint, dan bukan rentang generasi lama.
if t.eos_token_id != eos:
    fail.append("eos_token_id kelas=%s != checkpoint=%s" % (t.eos_token_id, eos))
if t.im_end_token_id != eos:
    fail.append("im_end_token_id kelas=%s != eos checkpoint=%s" % (t.im_end_token_id, eos))
if (t.eos_token_id, t.im_start_token_id, t.im_end_token_id) == (151643, 151644, 151645):
    fail.append("ID masih rentang Qwen2.5/Qwen3.0 (151xxx)")

# 2. Token khusus harus benar-benar ada di vocab dan memetakan ke ID yang sama.
def tok_id(s):
    if hasattr(tok, "convert_tokens_to_ids"):
        return tok.convert_tokens_to_ids(s)
    if hasattr(tok, "token_to_id"):
        return tok.token_to_id(s)
    return None

for text, want, label in (("<|im_start|>", t.im_start_token_id, "im_start"),
                          ("<|im_end|>",   t.im_end_token_id,   "im_end")):
    got = tok_id(text)
    if got != want:
        fail.append("%s -> vocab id %s, diharapkan %s (%s)" % (text, got, want, label))
    if not isinstance(want, int) or not (0 <= want < vocab):
        fail.append("%s id %s di luar vocab %s" % (label, want, vocab))

# 3. Stop token harus menyala di EOS asli, dan encode/decode harus bolak-balik
#    (bukti fallback byte-level sudah benar-benar hilang).
if not t.is_stop_token(eos):
    fail.append("is_stop_token(%s) = False" % eos)
if t.is_stop_token(151643):
    fail.append("is_stop_token(151643) = True (id lama, seharusnya bukan stop)")

probe = "Halo, apa kabar? 123"
if t.decode(t.encode(probe)) != probe:
    fail.append("round-trip gagal utk %r" % probe)

# 4. Prompt ChatML nyata (yang dipakai run inferensi) harus di dalam vocab.
ids = t.encode("<|im_start|>user\n" + probe + "<|im_end|>\n<|im_start|>assistant\n")
bad = [i for i in ids if not (0 <= i < vocab)]
if bad:
    fail.append("id di luar vocab: %s" % bad)
else:
    print("   prompt ChatML %d token, semua di dalam vocab" % len(ids))

if fail:
    for f in fail:
        print("   - " + f)
    sys.exit(1)
PYEOF
    then
        echo ">> [TOK-OK] 5.0 ID khusus cocok dgn checkpoint & vocab; fallback byte-level tidak ada"
    else
        echo ">> [TOK-FAIL] tes tokenizer GAGAL — run dihentikan, jangan lanjut."
        exit 1
    fi
    export BONSAI_CUDA_LIB="$WORKING/libbonsai_qmv_sm75.so"
    export LD_LIBRARY_PATH="$WORKING:$REPO_DIR:$REPO_DIR/build:$WORKING/build:$DIST_DIR:$REPO_DIR/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
    export BONSAI_USE_GPU=1
    # BONSAI_PROFILE=1 DI-MATIKAN: sync 65x/token di main loop mematikan
    # pipeline async (bukti: regresi 8.00 -> 7.54 tok/s saat profil aktif).
    # Aktifkan hanya untuk sesi profiling terpisah.
    # export BONSAI_PROFILE=1
    export MOJO_ENABLE_STACK_TRACE_ON_ERROR=1

    # DIAGNOSTIK CLOCK: kalau SM clock throttled (jauh di bawah 1590 MHz),
    # SEMUA kernel 2x lebih lambat dan itu menjelaskan gap 30 vs 62 GB/s.
    nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,pstate --format=csv,noheader > "$DIST_DIR/gpu_clock_before.txt" 2>/dev/null || true
    cat "$DIST_DIR/gpu_clock_before.txt" || true
    rm -f "$DIST_DIR/gpu_clock_trace.txt"
    ( while true; do
        nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw --format=csv,noheader >> "$DIST_DIR/gpu_clock_trace.txt" 2>/dev/null || break
        sleep 2
      done ) &
    CLOCK_PID=$!

    # Encode prompt ChatML asli dengan tokenizer MODEL itu sendiri (bukan
    # hardcode ID): prompt teks dari BONSAI_PROMPT (default "Hello") dibungkus
    # template <|im_start|>user ... <|im_start|>assistant. TANPA FALLBACK —
    # kalau tokenizer gagal, run langsung berhenti (hasil membingungkan lebih
    # buruk daripada gagal jelas).
    BONSAI_PROMPT="${BONSAI_PROMPT:-Hello}"
    PROMPT_TOKENS="$(python3 - "$KMODEL" "$BONSAI_PROMPT" <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
text = "<|im_start|>user\n" + sys.argv[2] + "<|im_end|>\n<|im_start|>assistant\n"
ids = tok.encode(text, add_special_tokens=False)
print(",".join(str(i) for i in ids))
PYEOF
)" || { echo ">> [ERROR] Encode tokenizer GAGAL — run dihentikan, tidak ada fallback."; exit 1; }
    echo ">> [PROMPT] BONSAI_PROMPT=\"$BONSAI_PROMPT\" -> token ids: $PROMPT_TOKENS"

    # Run 1: prefill BATCHED (qmm WMMA) + dump top-2 logit di batas prefill
    BONSAI_DUMP_TOP2=1 "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$PROMPT_TOKENS" --max-tokens 24 --gpu 2>&1 | tee "$DIST_DIR/infer_t4.log" \
        || echo ">> [WARN] inferensi GPU gagal — periksa log di atas"

    # Run 2: benchmark adil gaya MLX — prompt panjang (~40 token), warmup aktif.
    # Angka 120 tok/s MLX diukur pada prompt ChatML penuh + warmup eksplisit,
    # bukan 9 token dingin. Ini pembanding apples-to-apples.
    PROMPT_LONG="Jelaskan secara singkat apa itu kompresi kuantisasi biner, mengapa model bahasa besar tetap bisa menghasilkan keluaran yang baik dengan bobot 1-bit, serta apa keuntungan dan kerugiannya dibanding bobot presisi penuh."
    PROMPT_TOKENS_LONG="$(python3 - "$KMODEL" "$PROMPT_LONG" <<'PYEOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
msgs = [{"role": "user", "content": sys.argv[2]}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)
print(",".join(str(i) for i in ids))
PYEOF
)" || { echo ">> [ERROR] Encode prompt panjang GAGAL."; exit 1; }
    echo ">> [PROMPT-LONG] ${#PROMPT_TOKENS_LONG} token | $PROMPT_LONG"
    BONSAI_DUMP_TOP2=1 "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$PROMPT_TOKENS_LONG" --max-tokens 8 --gpu 2>&1 | tee "$DIST_DIR/infer_t4_long.log" \
        || echo ">> [WARN] run prompt panjang gagal — periksa log"

    # Run 2b: A/B fusi di node sama — fusion dimatikan utk pembanding.
    BONSAI_NO_FUSE=1 "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$PROMPT_TOKENS" --max-tokens 24 --gpu 2>&1 | tee "$DIST_DIR/infer_t4_nofuse.log" \
        || echo ">> [WARN] run no-fuse gagal — periksa log"

    # Run 3: jalur per-token (fallback) — pembanding stream & TOP2 di node sama
    echo ">> [AB] BONSAI_PREFILL_PER_TOKEN=1 (pembanding stream)..."
    BONSAI_DUMP_TOP2=1 BONSAI_PREFILL_PER_TOKEN=1 "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$PROMPT_TOKENS" --max-tokens 24 --gpu 2>&1 | tee "$DIST_DIR/infer_t4_pertoken.log" \
        || echo ">> [WARN] inferensi per-token gagal — periksa log di atas"

    # ---------------------------------------------------------------
    # Run 4: GERBANG REGRESI — KOHERENSI GENERASI PANJANG (state GDN fp32)
    #
    # Bug yang dijaga gerbang ini: state rekuren Gated DeltaNet dulu disimpan
    # fp16 di VRAM dan kernel membulatkannya 2x tiap langkah decode. Karena
    # decay ~ 1.0 galat itu tidak teredam -> output runtuh jadi siklus setelah
    # ~50-100 token. Perbaikannya state disimpan fp32 (lihat
    # LAPORAN_FIX_STATE_FP32.md).
    #
    # PERBAIKAN GERBANG (dulu verdict-nya tidak bisa ditafsirkan):
    #   1. Dulu ambang disalin dari baseline yang diukur dgn SAMPLING T=0.70
    #      seed 1234, sementara run-nya GREEDY — dua hal yang tidak sebanding.
    #      Angka baseline sekarang hanya dicetak sebagai KONTEKS, bukan ambang.
    #   2. Dulu SEMUA metrik dihitung atas 96 token terakhir dari seluruh
    #      stream. Dua akibat buruk:
    #        a) siklus yang terjadi LEBIH AWAL lalu ditinggalkan model tidak
    #           pernah terlihat;
    #        b) blok thinking yang wajar (verbositas terstruktur, kosakata
    #           sempit) ikut dihukum.
    #      Sekarang: gerbang A memindai SELURUH stream, sedangkan gerbang B/C
    #      dinilai atas region JAWABAN (sesudah penanda tutup thinking).
    #   3. Ditambah gerbang C yang SELF-CALIBRATING: ekor-50 dibanding awal-50
    #      pada RUN YANG SAMA. Tidak bergantung baseline eksternal maupun mode
    #      decoding, jadi tetap bermakna walau ambang absolut meleset.
    #   4. Mode thinking kini bisa diuji langsung: BONSAI_COH_THINK=both
    #      menjalankan tiap prompt dgn thinking MATI dan HIDUP, sehingga
    #      hipotesis "runtuh = loop di dalam mode thinking" teruji tanpa
    #      mengubah gerbangnya.
    #
    # GERBANG (semua objektif):
    #   A) TIDAK ada rentetan periodik-persis >= 64 token dgn periode p <= 48
    #      DI MANA PUN dalam stream (menangkap loop di dalam blok thinking).
    #   B) 50 token terakhir region jawaban >= 12 token unik
    #      (baseline rusak: 0.16 x 50 = 8)
    #   C) ekor-50 >= 50% keragaman awal-50 region jawaban (bila region >= 100)
    # Default NON-FATAL supaya sisa pipeline tetap jalan; set
    # BONSAI_COH_FATAL=1 untuk menjadikannya gerbang keras (exit 1).
    # ---------------------------------------------------------------
    COH_DIR="$DIST_DIR/coherence"
    mkdir -p "$COH_DIR"
    COH_TOKENS="${BONSAI_COH_TOKENS:-512}"
    # Mode thinking: "0" (default) = MATI -> gerbang mengukur koherensi JAWABAN.
    # "1" = hidup. "both" = keduanya (uji hipotesis mode thinking).
    # Template Qwen3 memaksa <think> bila enable_thinking tidak dimatikan
    # (temuan #1 di infer_susah/HASIL_TES_PEMISAH.md).
    COH_THINK="${BONSAI_COH_THINK:-0}"
    case "$COH_THINK" in
        both) COH_MODES=("0" "1") ;;
        *)    COH_MODES=("$COH_THINK") ;;
    esac
    echo ""
    echo ">> [COH] Run 4: gerbang koherensi $COH_TOKENS token (GREEDY) — regresi state GDN fp32"
    echo ">> [COH] mode thinking diuji: ${COH_MODES[*]}"

    # Token penanda blok thinking — diambil sekali, dipakai gerbang untuk
    # memisahkan region JAWABAN dari region thinking.
    # Cara mengambilnya MENIRU llama.cpp (common/chat.cpp):
    #   * tag dideteksi dari SOURCE template, bukan dari nama/versi model
    #     (llama.cpp: supports_reasoning = tmpl.source().find("<think>"));
    #   * id diresolusi lewat encode() dan wajib TEPAT SATU token;
    #   * tanpa default hardcode — resolusi gagal harus terlihat (-1).
    # Qwen3.5 memakai <think>/</think>; Qwen3 memakai ' thinking'/'\u200b'.
    # Keduanya BEDA dan tidak boleh disamakan.
    if ! python3 - "$KMODEL" "$COH_DIR/meta.json" <<'PYEOF'
import json, os, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)


def one_id(s):
    """id token HANYA bila `s` ter-encode jadi TEPAT SATU token.

    Meniru llama.cpp (common/chat.cpp): tag reasoning diambil dari artefak
    model itu sendiri, bukan konstanta yang di-hardcode. llama.cpp mendeteksi
    dukungan reasoning dgn mencari tag di SOURCE template; hal yang sama
    dilakukan di sini, lalu id-nya diresolusi lewat encode().
    encode() melewati pipeline yang sama dgn inferensi, jadi inilah yang
    benar-benar terbukti -- dan aman kalau `s` terpecah jadi beberapa token.
    """
    if not s:
        return -1
    try:
        ids = tok.encode(s, add_special_tokens=False)
    except Exception:
        return -1
    return ids[0] if len(ids) == 1 else -1


# Pasangan tag reasoning yang dikenal, urut prioritas.
# PENTING: Qwen3.5 memakai <think>/</think>.
#          Qwen3  memakai ' thinking'/'\u200b'  (BEDA, jangan disamakan).
KNOWN_PAIRS = (
    ("<think>", "</think>"),
    (" thinking", "\u200b"),
    ("[THINK]", "[/THINK]"),
    ("<|channel|>analysis<|message|>", "<|end|>"),
)

# 1) Baca SOURCE template -- llama.cpp mendeteksi dari template, bukan nama model.
tpl = ""
try:
    tpl = tok.chat_template or ""
except Exception:
    tpl = ""
if not tpl:
    for fn in ("chat_template.jinja", "chat_template.json", "tokenizer_config.json"):
        try:
            with open(os.path.join(sys.argv[1], fn)) as fh:
                tpl += fh.read()
        except Exception:
            pass

open_tag = close_tag = ""
for o, c in KNOWN_PAIRS:
    if o and o in tpl:
        open_tag, close_tag = o, c
        break

# 2) Sengaja TIDAK ada tebak-tebakan bila template tak memberi petunjuk.
#    encode(" thinking") pada vocab Qwen3.5 menghasilkan 7047 = kata biasa
#    'Ġthinking', BUKAN penanda. Jadi menebak lewat "token-nya ada" bisa
#    menyuntik penanda PALSU -- persis jenis kegagalan senyap yang dulu
#    membuat think_close = -1 tanpa ketahuan. Lebih baik -1 yang kelihatan
#    daripada id yang salah tapi tampak wajar.

# Tanpa default hardcode: resolusi yang gagal harus TERLIHAT sebagai -1.
meta = {
    "think_open": one_id(open_tag),
    "think_close": one_id(close_tag),
    "think_open_str": open_tag,
    "think_close_str": close_tag,
    "im_start": one_id("<|im_start|>"),
    "im_end": one_id("<|im_end|>"),
    "template_has_think": "<think>" in tpl,
}
with open(sys.argv[2], "w") as fh:
    json.dump(meta, fh)
print("   token penanda:", meta)
PYEOF
    then
        echo ">> [COH-WARN] meta token penanda gagal — region thinking dilewati"
    fi

    COH_PROMPTS=(
      "cerita|Tulis sebuah cerita pendek sekitar 250 kata tentang seorang nelayan tua yang menemukan botol berisi peta harta karun di tepi pantai."
      "penalaran|Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu pipa B ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. Berapa jam total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? Tunjukkan langkah perhitungannya."
    )
    for coh_entry in "${COH_PROMPTS[@]}"; do
        coh_name="${coh_entry%%|*}"
        coh_text="${coh_entry#*|}"
        for coh_think in "${COH_MODES[@]}"; do
            coh_tag="$coh_name.think$coh_think"
            coh_tok="$(python3 - "$KMODEL" "$coh_text" "$coh_think" \
                             "$COH_DIR/meta.json" "$COH_DIR/$coh_tag.prompt.json" <<'PYEOF'
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
msgs = [{"role": "user", "content": sys.argv[2]}]
try:
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=(sys.argv[3] == "1"))
except TypeError:
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)
try:
    tc = json.load(open(sys.argv[4])).get("think_close", -1)
except Exception:
    tc = -1
json.dump({"prompt_len": len(ids), "think_close": tc,
           "prompt_has_think_close": bool(isinstance(tc, int) and tc >= 0
                                          and tc in ids)},
          open(sys.argv[5], "w"))
print(",".join(str(i) for i in ids))
PYEOF
)" || { echo ">> [COH-WARN] encode prompt '$coh_tag' GAGAL — dilewati"; continue; }
            echo ">> [COH] --- $coh_tag (greedy, $COH_TOKENS token) ---"
            BONSAI_TEMP_X100=0 "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
                --prompt-tokens "$coh_tok" --max-tokens "$COH_TOKENS" --gpu \
                2>&1 | tee "$COH_DIR/$coh_tag.log" \
                | grep -E "\[GEN\] token id|\[PERF\] rata-rata" || true
        done
    done

    COH_RC=0
    if python3 - "$COH_DIR" <<'PYEOF'
import glob, json, os, re, sys

d = sys.argv[1]
# Gerbang A: rentetan periodik-persis minimal sepanjang ini -> dianggap loop.
MIN_LOOP = 64
# Gerbang A: periode terbesar yang masih dianggap "loop", bukan teks wajar.
P_MAX = 48
# Gerbang B: baseline rusak = 8 token unik dari 50 (0.16 x 50).
MIN_UNIK_EKOR = 12
# Gerbang C: ekor-50 minimal sekian dari awal-50 (self-calibrating).
RASIO_EKOR_MIN = 0.5
# Catatan saja: jawaban dianggap "sangat pendek" bila < 25% anggaran token.
RASIO_JAWABAN_MIN = 0.25
# HANYA KONTEKS — diukur dgn SAMPLING T=0.70 seed 1234, bukan GREEDY. TIDAK
# dipakai sebagai ambang (lihat catatan 1 di blok komentar Run 4).
BASE = {"cerita": (0.28, 0.42), "penalaran": (0.21, 0.16)}

try:
    META = json.load(open(os.path.join(d, "meta.json")))
except Exception:
    META = {}
THINK_CLOSE = META.get("think_close", -1)
if not isinstance(THINK_CLOSE, int):
    THINK_CLOSE = -1
# Tanpa default hardcode: id milik keluarga model lain bisa menyesatkan gerbang.
IM_END = META.get("im_end", -1)
if not isinstance(IM_END, int):
    IM_END = -1


def toks(p):
    try:
        t = open(p).read()
    except FileNotFoundError:
        return []
    return [int(m) for m in re.findall(r"\[GEN\] token id:\s*(\d+)", t)]


def longest_periodic_run(seq, pmax, min_len):
    """Rentetan periodik-persis terpanjang DI MANA PUN.

    Untuk tiap periode p, hitung panjang rentetan terpanjang dgn
    seq[i] == seq[i-p] berturut-turut. Mengembalikan (p, panjang, indeks_awal)
    atau None. Berbeda dari memeriksa EKOR saja: siklus yang terjadi lebih awal
    lalu ditinggalkan model tetap tertangkap.
    """
    n = len(seq)
    best = None
    for p in range(1, min(pmax, n // 2) + 1):
        run = best_len = best_end = 0
        for i in range(p, n):
            if seq[i] == seq[i - p]:
                run += 1
                if run > best_len:
                    best_len, best_end = run, i
            else:
                run = 0
        if not best_len:
            continue
        panjang = best_len + p
        if panjang >= min_len and (best is None or panjang > best[1]):
            best = (p, panjang, best_end - best_len - p + 1)
    return best


def region_of(tag, g):
    """Region yang DINILAI + labelnya.

    Jawaban = token sesudah penanda tutup thinking yang terakhir. Bila prompt
    SUDAH menutup thinking (enable_thinking=False -> template menaruh blok
    kosong di prompt), seluruh token hasil generate memang sudah jawaban.
    """
    try:
        pi = json.load(open(os.path.join(d, tag + ".prompt.json")))
    except Exception:
        pi = {}
    if pi.get("prompt_has_think_close"):
        return g, "jawaban", 0
    if THINK_CLOSE >= 0 and THINK_CLOSE in g:
        i = len(g) - 1 - g[::-1].index(THINK_CLOSE)
        return g[i + 1:], "jawaban", i + 1
    return g, "seluruh stream (thinking tak ditutup)", 0


logs = sorted(glob.glob(os.path.join(d, "*.log")))
if not logs:
    print(">> [COH-FAIL] tidak ada log generasi di " + d)
    sys.exit(1)

gagal, catatan, sebaran = [], [], {}
print("   --- gerbang koherensi generasi panjang (GREEDY) ---")
print(f"   {'run':<20} {'gen':>4} {'rgn':>4} {'unik':>5} {'rasio':>6} "
      f"{'awal50':>6} {'ekor50':>6} {'loop@':>8}  region")
for lg in logs:
    tag = os.path.basename(lg)[:-4]
    nama = tag.split(".")[0]
    g = toks(lg)
    if not g:
        gagal.append(f"{tag}: token tidak terekam di log")
        print(f"   {tag:<20} {'—':>4}  (tidak ada token di log)")
        continue
    rgn, rlabel, rgn_off = region_of(tag, g)
    # --- Gerbang A: loop di mana pun pada SELURUH stream ---
    loop = longest_periodic_run(g, P_MAX, MIN_LOOP)
    if not rgn:
        gagal.append(f"{tag}: region jawaban KOSONG — model menutup thinking "
                     f"tepat di akhir sehingga tidak ada jawaban")
        print(f"   {tag:<20} {len(g):>4} {0:>4}  (region jawaban kosong)")
        continue
    uniq = len(set(rgn))
    u_awal = len(set(rgn[:50]))
    u_ekor = len(set(rgn[-50:]))
    loop_txt = f"p{loop[0]}@{loop[2]}" if loop else "—"
    b0, b1 = BASE.get(nama, (float("nan"), float("nan")))
    print(f"   {tag:<20} {len(g):>4} {len(rgn):>4} {uniq:>5} "
          f"{uniq / len(rgn):>6.2f} {u_awal:>6} {u_ekor:>6} "
          f"{loop_txt:>8}  {rlabel} [konteks {b0:.2f}/{b1:.2f}]")
    # Konvensi jendela-50 sejajar-depan: HANYA untuk melihat tren.
    sebaran[tag] = [len(set(rgn[i:i + 50])) / 50.0
                    for i in range(0, len(rgn) - 49, 50)]
    if loop:
        gagal.append(f"{tag}: loop periodik p={loop[0]} sepanjang {loop[1]} token "
                     f"mulai token {loop[2]} (di mana pun dalam stream)")
    # --- Gerbang B: keragaman ekor absolut pada region jawaban ---
    if u_ekor < MIN_UNIK_EKOR:
        gagal.append(f"{tag}: 50 token terakhir jawaban hanya {u_ekor} token unik "
                     f"(<{MIN_UNIK_EKOR})")
    # --- Gerbang C: self-calibrating (ekor vs AWAL region jawaban) ---
    if len(rgn) >= 100 and u_awal > 0 and u_ekor < RASIO_EKOR_MIN * u_awal:
        gagal.append(f"{tag}: ekor50 ({u_ekor}) < {RASIO_EKOR_MIN:.0%} dari "
                     f"awal50 ({u_awal}) — runtuh relatif, tanpa baseline eksternal")
    # --- Catatan (bukan gerbang) ---
    if rlabel.startswith("seluruh stream"):
        catatan.append(f"{tag}: penanda tutup thinking tidak muncul dalam "
                       f"{len(g)} token — model tidak pernah keluar dari blok thinking")
    elif len(g) and len(rgn) < RASIO_JAWABAN_MIN * len(g):
        catatan.append(f"{tag}: jawaban hanya {len(rgn)}/{len(g)} token "
                       f"(<{RASIO_JAWABAN_MIN:.0%}) — anggaran habis di blok thinking")
    if IM_END in g:
        print(f"   {tag:<20} catatan: EOS muncul di posisi {g.index(IM_END)}")

print("   sebaran rasio id unik per jendela 50 token REGION JAWABAN "
      "(1.00 = tanpa pengulangan):")
for tag, ser in sebaran.items():
    print(f"     {tag:<20}: " + " ".join(f"{v:.2f}" for v in ser))
if catatan:
    print("   catatan:")
    for c in catatan:
        print("     - " + c)
print("   'konteks a/b' = baseline SAMPLING T=0.70 seed 1234, BUKAN ambang gerbang.")

if gagal:
    print(">> [COH-FAIL] " + "; ".join(gagal))
    sys.exit(1)
print(">> [COH-OK] tidak ada loop & ekor jawaban masih beragam "
      "-> regresi state GDN fp16 tidak kembali")
PYEOF
    then
        echo ">> [COH] gerbang LULUS"
    else
        COH_RC=1
        echo ">> [COH] gerbang GAGAL — lihat baris [COH-FAIL] di atas"
        if [ "$BONSAI_COH_FATAL" = "1" ]; then
            echo ">> [COH] BONSAI_COH_FATAL=1 -> pipeline dihentikan"
            exit 1
        fi
    fi
    echo ">> [COH] status gerbang koherensi: rc=$COH_RC (0=lulus, 1=gagal)"

    kill $CLOCK_PID 2>/dev/null || true
    nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader > "$DIST_DIR/gpu_clock_after.txt" 2>/dev/null || true
    echo ">> [CLOCK] Sampel selama inferensi (min/max SM clock):"
    sort -t, -k2 -n "$DIST_DIR/gpu_clock_trace.txt" 2>/dev/null | sed -n '1p;$p' || true
elif [ ! -f "$WORKING/bonsai_infer" ]; then
    echo ">> [SKIP] binary bonsai_infer tidak ada — uji inferensi dilewati"
else
    echo ">> [SKIP] Tidak ada GPU (nvidia-smi) — uji inferensi T4 dilewati"
fi

echo ""
echo "========================================================="
echo " 6. MENYEGARKAN ARSIP CACHE UNTUK RUN BERIKUTNYA"
echo "========================================================="
# Arsip pertama sudah ditulis tepat setelah build selesai (§4), jadi cache tidak
# lagi bergantung pada tercapainya bagian akhir skrip. Di sini arsip ditulis
# ULANG untuk menangkap cache yang terbentuk selama §5 (mis. cache Mojo dari
# menjalankan tes). Penghapusan folder tak-terkompres + .pixi dilakukan trap EXIT.
save_build_cache 1

echo ""
echo "========================================================="
echo ">> [SELESAI] BUILD + UJI INFERENSI T4 DI KAGGLE!"
echo "========================================================="
