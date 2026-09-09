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
if [ -f "$WORKING/mojo_build_cache.tar.gz" ]; then
    echo ">> [CACHE] Me-restore cache dari $WORKING/mojo_build_cache.tar.gz..."
    tar -xzf "$WORKING/mojo_build_cache.tar.gz" -C "$WORKING" && rm -f "$WORKING/mojo_build_cache.tar.gz"
    echo ">> [CACHE] Restore selesai, archive terkompresi dihapus."
fi

export PIXI_HOME="$CACHE_DIR/pixi"
export PIXI_CACHE_DIR="$CACHE_DIR/pixi_cache"
export MAGIC_HOME="$CACHE_DIR/magic"
export MODULAR_HOME="$CACHE_DIR/modular"
export MOJO_CACHE_DIR="$CACHE_DIR/mojo"
export CCACHE_DIR="$CACHE_DIR/ccache"
export CCACHE_MAXSIZE="5G"
export PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$MAGIC_HOME/bin:$PATH"

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
    nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC -Xptxas -v \
        "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" \
        -o "$WORKING/libbonsai_qmv_sm75.so"
    echo ">> [OK] libbonsai_qmv_sm75.so berhasil dibuat ($(du -h "$WORKING/libbonsai_qmv_sm75.so" | cut -f1))"
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
echo " 5. MENGARSIPKAN CACHE UNTUK RUN BERIKUTNYA"
echo "========================================================="
# Simpan archive cache ke /kaggle/working agar sesi berikutnya langsung hit
if [ -d "$CACHE_DIR" ]; then
    echo ">> Menyimpan archive cache kompilasi inkremental ke $WORKING/mojo_build_cache.tar.gz..."
    tar -czf "$WORKING/mojo_build_cache.tar.gz" -C "$WORKING" ".cache_t4_build" 2>/dev/null || true
    echo ">> [OK] Ukuran cache tersimpan: $(du -sh "$WORKING/mojo_build_cache.tar.gz" 2>/dev/null | cut -f1)"
    # Output kernel harus RINGAN: hapus cache ter-uncompress + env pixi.
    # Ribuan file di output adalah penyebab listing output macet/429.
    rm -rf "$CACHE_DIR" "$REPO_DIR/.pixi"
    echo ">> [OK] Cache ter-uncompress & .pixi dihapus dari output (arsip .tar.gz tetap)."
fi

echo ""
echo "========================================================="
echo ">> [SELESAI] BUILD + UJI INFERENSI T4 DI KAGGLE!"
echo "========================================================="
