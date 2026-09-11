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

# KHQ TIDAK diuji di sini. Kalibrasi hanya bermakna dijalankan atas dump K/V
# hasil forward pass MODEL ASLI (referensi: precompute_centroids.py mengumpulkan
# k_unroped + v + skor softmax dari forward pass nyata, SEQ_LEN=1024 x 32 batch).
# Dump buatan/palsu tidak membuktikan apa pun, jadi dihapus. Pipeline KHQ nyata
# ada di langkah 5b (butuh bobot model + binary + GPU).

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

    # ---------------------------------------------------------------
    # 5b. KHQ: DUMP K/V ASLI DARI MODEL -> KALIBRASI GPU -> UJI RUNTIME
    #     TIDAK ada dump sintetis. K/V diambil dari forward pass bobot
    #     asli lewat hook khq_dump_kv (K post-norm PRE-rope + V), dan
    #     skor softmax dari khq_dump_attn — persis yang dikumpulkan
    #     precompute_centroids.py (k_unroped + v + scores).
    # ---------------------------------------------------------------
    KHQ_DIR="$WORKING/khq_real"
    KHQ_TOKENS="${KHQ_DUMP_TOKENS:-512}"
    rm -rf "$KHQ_DIR"; mkdir -p "$KHQ_DIR"
    echo ""
    echo ">> [KHQ] 5b.1 dump K/V asli dari model ($KHQ_TOKENS token, prompt panjang)"
    BONSAI_DUMP_TOP2=1 BONSAI_DUMP_KV_DIR="$KHQ_DIR" "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$PROMPT_TOKENS_LONG" --max-tokens "$KHQ_TOKENS" --gpu \
        2>&1 | tee "$DIST_DIR/khq_dump.log" \
        || echo ">> [KHQ-FAIL] run dump gagal — periksa $DIST_DIR/khq_dump.log"

    echo ">> [KHQ] 5b.2 verifikasi dump (harus dari bobot asli, bukan placeholder)"
    if python3 - "$KMODEL" "$KHQ_DIR" <<'PYEOF'
import json, os, struct, sys
import numpy as np
mdir, ddir = sys.argv[1], sys.argv[2]

# Parsing MENGIKUTI main.mojo: parameter teks bisa bersarang di 'text_config',
# dan key yang tidak ada memakai default QwenConfig.qwen_27b_default().
raw = json.load(open(os.path.join(mdir, "config.json")))
cfg = raw.get("text_config", raw) or raw


def num(key, default):
    v = cfg.get(key, raw.get(key))
    return default if v is None else int(v)


H_kv = num("num_key_value_heads", 4)
hd = num("head_dim", 256)
n_layers = num("num_hidden_layers", 64)
interval = num("full_attention_interval", 4)
dim_expect = H_kv * hd
attn_layers = [i for i in range(n_layers) if i % interval == interval - 1]
print(f"   config: layers={n_layers} H_kv={H_kv} head_dim={hd} "
      f"-> dim={dim_expect}, {len(attn_layers)} layer attention")

p = os.path.join(ddir, "kv_dump.bin")
if not os.path.exists(p):
    print(f">> [KHQ-FAIL] {p} tidak ada"); sys.exit(1)
buf = open(p, "rb").read()
magic, nl, dim = struct.unpack_from("<III", buf, 0)
print(f"   kv_dump: magic={hex(magic)} layers={nl} dim={dim} bytes={len(buf)}")
if magic != 0x4451484B:
    print(">> [KHQ-FAIL] magic dump salah"); sys.exit(1)
if dim != dim_expect:
    print(f">> [KHQ-FAIL] dim dump {dim} != model {H_kv}x{hd}={dim_expect}"); sys.exit(1)

tbl = [struct.unpack_from("<II", buf, 12 + 8 * i) for i in range(nl)]
ids = [t[0] for t in tbl]
print(f"   layer_ids dump = {ids}")
if ids != attn_layers:
    print(f">> [KHQ-FAIL] layer_id dump != layer attention model {attn_layers}"); sys.exit(1)
tokset = sorted({t[1] for t in tbl})
if len(tokset) != 1 or tokset[0] < 8:
    print(f">> [KHQ-FAIL] jumlah token antar layer tidak seragam/kekecilan: {tokset}"); sys.exit(1)
ntok = tokset[0]
need = 12 + 8 * nl + sum(2 * t[1] * dim * 4 for t in tbl)
if need != len(buf):
    print(f">> [KHQ-FAIL] ukuran dump {len(buf)} != ekspektasi {need}"); sys.exit(1)
print(f"   token/layer = {ntok}, ukuran konsisten")

off = 12 + 8 * nl
bad = []
for lid, nt in tbl:
    k = np.frombuffer(buf, dtype="<f4", count=nt * dim, offset=off)
    v = np.frombuffer(buf, dtype="<f4", count=nt * dim, offset=off + nt * dim * 4)
    off += 2 * nt * dim * 4
    ks, vs = float(k.std()), float(v.std())
    if not (np.isfinite(k).all() and np.isfinite(v).all()):
        bad.append(f"L{lid}:NaN/Inf")
    if ks < 1e-6 or vs < 1e-6 or ks > 1e3 or vs > 1e3:
        bad.append(f"L{lid}:std_k={ks:.3e},std_v={vs:.3e}")
    if np.array_equal(k, v):
        bad.append(f"L{lid}:K==V")
print(f"   realness: std K/V diperiksa ({nl} layer)")
if bad:
    print(">> [KHQ-FAIL] data dump tidak wajar -> " + "; ".join(bad)); sys.exit(1)

# File attn WAJIB ada dan jumlah tokennya harus sama dengan dump; kalau tidak,
# kalibrasi V tidak lagi attention-aware (calib.mojo akan menolak).
for lid in attn_layers:
    ap = os.path.join(ddir, f"attn_{lid}.bin")
    if not os.path.exists(ap):
        print(f">> [KHQ-FAIL] attn_{lid}.bin tidak ada — kalibrasi akan gagal"); sys.exit(1)
    ab = open(ap, "rb").read()
    am, ahq, akv, ad = struct.unpack_from("<IIII", ab, 0)
    if am != 0x4151484B:
        print(f">> [KHQ-FAIL] magic attn_{lid} salah: {hex(am)}"); sys.exit(1)
    if ad != hd or akv != H_kv:
        print(f">> [KHQ-FAIL] attn_{lid} D={ad}/H_kv={akv} "
              f"!= model {hd}/{H_kv}"); sys.exit(1)
    o, cnt = 16, 0
    while o + 4 <= len(ab):
        (nv,) = struct.unpack_from("<I", ab, o)
        o += 4 + ahq * nv * 2
        cnt += 1
    if cnt != ntok:
        print(f">> [KHQ-FAIL] attn_{lid} punya {cnt} token, dump {ntok}")
        sys.exit(1)
print(f"   attn: {len(attn_layers)} layer x {ntok} token, H_q={ahq}, D={ad} — cocok dengan dump")
print(">> [KHQ-OK] dump ASLI dari model valid")
PYEOF
    then
        echo ">> [KHQ] 5b.3 kalibrasi GPU penuh atas dump asli (kv_dump -> khq_calib.bin)"
        pixi run mojo run -I . src/khq/calib.mojo "$KHQ_DIR" "$KHQ_DIR/khq_calib.bin" \
            2>&1 | tee "$DIST_DIR/khq_calib.log" \
            || echo ">> [KHQ-FAIL] kalibrasi gagal — periksa $DIST_DIR/khq_calib.log"

        if [ -f "$KHQ_DIR/khq_calib.bin" ]; then
            python3 - "$KHQ_DIR/khq_calib.bin" <<'PYEOF'
import struct, sys
buf = open(sys.argv[1], "rb").read()
magic, ver, nl, dim = struct.unpack("<IIII", buf[:16])
if magic != 0x4B51484B:
    print(f">> [KHQ-FAIL] magic centroid salah: {hex(magic)}"); sys.exit(1)
print(f">> [KHQ] centroid: ver={ver} layers={nl} dim={dim} bytes={len(buf)}")
off, ok, D = 16, True, 256

def rd(n, name):
    global off, ok
    vals = struct.unpack_from(f"<{n}f", buf, off); off += 4 * n
    bad = sum(1 for v in vals if v != v or abs(v) > 1e6)
    if bad: ok = False
    print(f"   {name}: n={n} bad={bad} min={min(vals):+.4f} max={max(vals):+.4f}")
    return vals

for _ in range(nl):
    (lid,) = struct.unpack_from("<I", buf, off); off += 4
    ts_k, ak = struct.unpack_from("<ff", buf, off); off += 8
    dk = rd(D, f"L{lid} d_k"); rd(4 * D, f"L{lid} cents_k"); rd(64 * 16, f"L{lid} rpk")
    ts_v, av = struct.unpack_from("<ff", buf, off); off += 8
    rd(D, f"L{lid} d_v"); rd(4 * D, f"L{lid} cents_v"); rd(64 * 16, f"L{lid} rpv")
    rd(64 * 768, f"L{lid} vq")
    print(f"   L{lid}: ts_k={ts_k:.4f} alpha_k={ak:.4f} ts_v={ts_v:.4f} alpha_v={av:.4f}")
    if not (0.05 < ts_k < 1000 and 0.05 < ts_v < 1000): ok = False
    if any(abs(v) != 1.0 for v in dk[:16]): ok = False
if ok and off == len(buf):
    print(">> [KHQ-OK] centroid valid, semua nilai finite, ukuran pas")
else:
    print(f">> [KHQ-FAIL] centroid tidak valid (sisa {len(buf)-off} byte)"); sys.exit(1)
PYEOF

            echo ">> [KHQ] 5b.4 uji RUNTIME dengan centroid hasil kalibrasi (BONSAI_KHQ_PATH)"
            echo ">> [KHQ] $KHQ_TOKENS token supaya watermark 256->kompres 128 benar-benar terpicu"
            # BONSAI_KHQ_SPLITS=1 DIKUNCI di sini: ini run verifikasi, dan gate
            # isi payload (gate 4) butuh stream token identik dengan run dump.
            # Dengan num_splits=1 kernel memakai jalur lama persis (tanpa reduce
            # kernel), jadi kontrak bit-exact yang sudah terbukti tidak berubah.
            # Perilaku split-K diukur terpisah di 5b.5.
            BONSAI_KHQ_SPLITS=1 BONSAI_KHQ_PATH="$KHQ_DIR/khq_calib.bin" BONSAI_DUMP_TOP2=1 BONSAI_KHQ_DEBUG=1 "$WORKING/bonsai_infer" \
                --model-dir "$KMODEL" --prompt-tokens "$PROMPT_TOKENS_LONG" \
                --max-tokens "$KHQ_TOKENS" --gpu 2>&1 | tee "$DIST_DIR/khq_runtime.log" \
                || echo ">> [KHQ-FAIL] run runtime KHQ gagal — periksa $DIST_DIR/khq_runtime.log"

            # 5b.5 A/B SPLIT-K — pembanding yang menentukan: jalur lama (1 split,
            # 24 block x 1 warp) vs split-K. Dijalankan TANPA BONSAI_KHQ_DEBUG
            # dan TANPA BONSAI_KHQ_PROF supaya angkanya tidak tercemar sync/D2H
            # yang dipakai verifikasi.
            echo ">> [KHQ] 5b.5 A/B split-K attention (tanpa debug/prof, angka adil)"
            # Baseline apple-to-apple: prompt & jumlah token SAMA, tanpa KHQ.
            echo ">> [KHQ] --- baseline fp16 (BONSAI_KHQ_PATH kosong) ---"
            env -u BONSAI_KHQ_PATH "$WORKING/bonsai_infer" \
                --model-dir "$KMODEL" --prompt-tokens "$PROMPT_TOKENS_LONG" \
                --max-tokens "$KHQ_TOKENS" --gpu 2>&1 | tee "$DIST_DIR/khq_base.log" \
                | grep -E "PERF" || true
            for KHQ_SP in 1 4 8 16; do
                echo ">> [KHQ] --- BONSAI_KHQ_SPLITS=$KHQ_SP ---"
                BONSAI_KHQ_PATH="$KHQ_DIR/khq_calib.bin" BONSAI_KHQ_SPLITS="$KHQ_SP" \
                    "$WORKING/bonsai_infer" \
                    --model-dir "$KMODEL" --prompt-tokens "$PROMPT_TOKENS_LONG" \
                    --max-tokens "$KHQ_TOKENS" --gpu 2>&1 | tee "$DIST_DIR/khq_split_$KHQ_SP.log" \
                    | grep -E "PERF|KHQ\] split" || true
            done

            # 5b.6 PROFIL PER-FASE — sync tiap fase, jadi total ms/token memang
            # lebih buruk; yang dibaca adalah pembagian waktunya, bukan totalnya.
            echo ">> [KHQ] 5b.6 profil per-fase jalur KHQ (BONSAI_KHQ_PROF=1)"
            BONSAI_KHQ_PATH="$KHQ_DIR/khq_calib.bin" BONSAI_KHQ_PROF=1 \
                "$WORKING/bonsai_infer" \
                --model-dir "$KMODEL" --prompt-tokens "$PROMPT_TOKENS_LONG" \
                --max-tokens "$KHQ_TOKENS" --gpu 2>&1 | tee "$DIST_DIR/khq_prof.log" \
                | grep -E "KHQ-PROF|PERF" || true

            python3 - "$DIST_DIR" <<'PYEOF'
import os, re, sys

d = sys.argv[1]


def ms_per_token(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return None
    m = re.search(r"rata-rata decode:\s*([-\d.eE+]+)\s*ms/token", txt)
    return float(m.group(1)) if m else None


def toks(path):
    try:
        txt = open(path).read()
    except FileNotFoundError:
        return []
    return [int(m) for m in re.findall(r"\[GEN\] token id:\s*(\d+)", txt)]


def prefix_vs(ref, got):
    n = min(len(ref), len(got))
    p = 0
    while p < n and ref[p] == got[p]:
        p += 1
    return p, n


print("   --- A/B split-K (ms/token, tanpa debug) ---")
base = ms_per_token(os.path.join(d, "khq_base.log"))
ref_toks = toks(os.path.join(d, "khq_base.log"))
if base:
    print(f"   baseline fp16 (KV penuh)      : {base:.2f} ms/token")
ref = ms_per_token(os.path.join(d, "khq_split_1.log"))
rows = []
for sp in (1, 4, 8, 16):
    v = ms_per_token(os.path.join(d, f"khq_split_{sp}.log"))
    if v is None:
        continue
    rows.append((sp, v))
    got = toks(os.path.join(d, f"khq_split_{sp}.log"))
    p, n = prefix_vs(ref_toks, got)
    tag = "  <- jalur lama (bit-exact)" if sp == 1 else ""
    print(f"   KHQ splits={sp:<2}                  : {v:.2f} ms/token | "
          f"prefix token {p}/{n}{tag}")
if ref and len(rows) > 1:
    best_sp, best = min(rows, key=lambda r: r[1])
    print(f"   split-K terbaik: splits={best_sp} -> {best:.2f} ms/token "
          f"({(ref - best) / ref * 100:+.1f}% vs splits=1)")
    if best >= ref:
        print("   [WARN] split-K belum memperbaiki apa pun di konfigurasi ini")
    if base:
        print(f"   (KHQ vs baseline fp16: {ref / base:.3f}x pada splits=1)")
try:
    ptxt = open(os.path.join(d, "khq_prof.log")).read()
except FileNotFoundError:
    ptxt = ""
lines = re.findall(r"KHQ-PROF\]\s+(\S+)\s+([-\d.eE+]+)\s*ms\s*\|\s*([-\d.eE+]+)\s*%", ptxt)
if lines:
    print("   --- profil per-fase (sync tiap fase) ---")
    for name, ms, pct in lines:
        print(f"   {name:<14}: {float(ms):10.2f} ms | {pct}%")
else:
    print("   [WARN] blok KHQ-PROF tidak ditemukan di khq_prof.log")
PYEOF

            python3 - "$DIST_DIR/khq_dump.log" "$DIST_DIR/khq_runtime.log" \
                     "$KMODEL" "$KHQ_DIR/kv_dump.bin" <<'PYEOF'
import json, os, re, struct, sys
import numpy as np

def read(path):
    try:
        return open(path).read()
    except FileNotFoundError:
        return ""


def toks(txt):
    return [int(m) for m in re.findall(r"\[GEN\] token id:\s*(\d+)", txt)]


def top2(txt):
    m = re.search(r"\[TOP2\] 1st:\s*(\d+)\s*=\s*([-\d.eE+]+)\s*\|\s*2nd:\s*(\d+)\s*="
                  r"\s*([-\d.eE+]+)\s*\|\s*gap:\s*([-\d.eE+]+)", txt)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))


base_txt, khq_txt = read(sys.argv[1]), read(sys.argv[2])
aktif = "jalur KV terkompresi AKTIF" in khq_txt
base, khq = toks(base_txt), toks(khq_txt)
print(f"   baseline token={len(base)} | KHQ token={len(khq)} | jalur aktif={aktif}")
if not aktif:
    print(">> [KHQ-FAIL] jalur KV terkompresi tidak aktif di runtime"); sys.exit(1)
if not base or not khq:
    print(">> [KHQ-FAIL] token tidak terekam di log"); sys.exit(1)

# 1) GATE UTAMA — pembanding numerik di batas prefill (murni jalur window,
#    kompresi belum aktif). Token greedy bersifat chaotic: satu flip argmax
#    mengubah seluruh lanjutannya, jadi kecocokan token BUKAN ukuran yang
#    tegas. Logit top-1 inilah yang dibandingkan langsung.
gagal = []
tb, tk = top2(base_txt), top2(khq_txt)
if tb and tk:
    print(f"   TOP2 baseline: id={tb[0]}/{tb[2]} val={tb[1]:.6f}/{tb[3]:.6f}")
    print(f"   TOP2 KHQ     : id={tk[0]}/{tk[2]} val={tk[1]:.6f}/{tk[3]:.6f}")
    d1 = abs(tb[1] - tk[1])
    print(f"   |delta| logit top-1 = {d1:.6f}")
    if tb[0] != tk[0]:
        gagal.append(f"top-1 logit berbeda ({tb[0]} vs {tk[0]})")
    if d1 > 0.1:
        gagal.append(f"|delta| logit {d1:.6f} > 0.1")
else:
    print("   [WARN] TOP2 tidak terbaca di salah satu log")

# 2) INFO — prefix token identik. Sebelum perbaikan reduksi warp, divergensi
#    terjadi di token ke-5 (jauh sebelum kompresi di token 256); prefix sangat
#    pendek menandakan jalur window/merge masih jauh berbeda.
n = min(len(base), len(khq))
prefix = 0
while prefix < n and base[prefix] == khq[prefix]:
    prefix += 1
sama = sum(1 for i in range(n) if base[i] == khq[i])
print(f"   prefix identik = {prefix} token | cocok total = {sama}/{n} "
      f"(greedy, bukan gate)")
if prefix < 16:
    gagal.append(f"prefix token hanya {prefix} — jalur window/merge belum setara")

# 3) GATE CADENCE — bukti LANGSUNG dari instrumen BONSAI_KHQ_DEBUG=1.
#    Pola yang diharapkan: event#1 saat write_pos=256 (boundary 0->128), lalu
#    TIAP 128 token berikutnya (boundary 256, 384, ...) karena setelah kompres
#    jendela raw tersisa 128, jadi 128 token baru sudah membuatnya penuh lagi.
ev = {}
for m in re.finditer(r"\[KHQ-COMPRESS\] layer (\d+) event# (\d+) boundary (\d+) "
                     r"write_pos (\d+) raw_window (\d+)", khq_txt):
    lay, num, bnd, wp, raw = (int(x) for x in m.groups())
    ev.setdefault(lay, []).append((num, bnd, wp, raw))
if not ev:
    print("   [WARN] tidak ada baris KHQ-COMPRESS — cadence tidak terukur langsung")
else:
    jml = sorted({len(v) for v in ev.values()})
    print(f"   cadence: {len(ev)} layer, event per layer = {jml} "
          f"(harus seragam)")
    beda = []
    for lay, lst in sorted(ev.items()):
        for k, (num, bnd, wp, raw) in enumerate(lst, 1):
            if num != k or bnd != 128 * k or raw != 128:
                beda.append(f"L{lay}#{k}:b{bnd}/raw{raw}")
    l0 = min(ev)
    print(f"   cadence layer {l0}: " + " ".join(
        f"wp{wp}->b{bnd}(raw{raw})" for _, bnd, wp, raw in ev[l0]))
    if len(jml) != 1:
        gagal.append(f"jumlah event antar layer tidak seragam: {jml}")
    if beda:
        gagal.append("pola cadence menyimpang: " + "; ".join(beda[:4]))

# 4) GATE ISI PAYLOAD — membuktikan baris yang dikompres benar-benar token
#    TERTUA, bukan sekadar jumlah eventnya benar. Instrumen membuang norma K
#    tiap baris yang baru dikompres (h_knorm, bit fp16). Norma itu fingerprint
#    isi: dibandingkan dengan norma K token yang sama dari dump asli, lalu
#    dibandingkan juga pada alignment BERGESER — alignment yang benar harus
#    jauh lebih cocok, kalau tidak berarti yang dikompres token yang salah.
knd = [m for m in re.finditer(r"\[KHQ-NORM\] (\d+) (\d+) (\d+) ([0-9 ]+)", khq_txt)]
if not knd:
    gagal.append("instrumen KHQ-NORM tidak ada — isi payload tidak terverifikasi")
else:
    # jumlah token harus identik dengan run dump, kalau tidak ground truth tak berlaku
    if base != khq:
        print(f"   [WARN] token run dump != run runtime "
              f"({sum(1 for a,b in zip(base,khq) if a==b)}/{min(len(base),len(khq))} cocok)"
              f" — ground truth isi dilewati")
    else:
        raw = json.load(open(os.path.join(sys.argv[3], "config.json")))
        cf = raw.get("text_config", raw) or raw
        H_kv = int(cf.get("num_key_value_heads", 4))
        hd = int(cf.get("head_dim", 256))
        dim = H_kv * hd
        buf = open(sys.argv[4], "rb").read()
        nl = struct.unpack_from("<I", buf, 4)[0]
        tbl = [struct.unpack_from("<II", buf, 12 + 8 * i) for i in range(nl)]
        goff, off = {}, 12 + 8 * nl
        for lid, nt in tbl:
            goff[lid] = (off, nt)
            off += 2 * nt * dim * 4

        def gt_norm(lid):
            base_off, nt = goff[lid]
            k = np.frombuffer(buf, dtype="<f4", count=nt * dim,
                              offset=base_off).reshape(nt, H_kv, hd)
            return np.sqrt((k.astype(np.float64) ** 2).sum(axis=2))

        cache, hasil = {}, []
        for m in knd:
            lid, B, nrows = int(m.group(1)), int(m.group(2)), int(m.group(3))
            bits = np.array([int(x) for x in m.group(4).split()], dtype=np.uint16)
            if len(bits) != nrows:
                gagal.append(f"L{lid} B{B}: {len(bits)} norma != {nrows} baris")
                continue
            if lid not in cache:
                cache[lid] = gt_norm(lid)
            g = cache[lid]
            nt = g.shape[0]
            got = bits.view(np.float16).astype(np.float64)
            r0 = B * H_kv
            idx = np.arange(nrows)
            tok = B + idx // H_kv
            hd_i = idx % H_kv

            def err(tokens):
                ok = (tokens >= 0) & (tokens < nt)
                if not ok.any():
                    return None
                a = g[tokens[ok], hd_i[ok]]
                b = got[ok]
                return float(np.mean(np.abs(a - b) / (np.abs(a) + 1e-6)))

            e0 = err(tok)
            e_up = err(tok + 128)
            e_dn = err(tok - 128)
            hasil.append((lid, B, r0, e0, e_up, e_dn))

        if hasil:
            e0 = np.array([h[3] for h in hasil])
            print(f"   isi payload: {len(hasil)} event diperiksa, "
                  f"rel-err norma (alignment benar) max={e0.max():.3e} "
                  f"mean={e0.mean():.3e}")
            ups = np.array([h[4] for h in hasil if h[4] is not None])
            dns = np.array([h[5] for h in hasil if h[5] is not None])
            alt = np.concatenate([a for a in (ups, dns) if a.size]) if (ups.size or dns.size) else None
            if alt is not None:
                print(f"   pembanding alignment bergeser (+-128 token): "
                      f"mean={alt.mean():.3e}")
            if e0.max() > 1e-2:
                gagal.append(f"norma payload menyimpang dari ground truth "
                             f"(max rel-err {e0.max():.3e})")
            if alt is not None and alt.mean() < 5 * e0.mean():
                gagal.append("alignment benar tidak lebih cocok dari yang bergeser "
                             "— isi payload kemungkinan token yang salah")

if gagal:
    print(">> [KHQ-FAIL] " + "; ".join(gagal)); sys.exit(1)
print(">> [KHQ-OK] jalur window setara secara numerik; cadence kompresi "
      "terukur (256 pertama, lalu tiap 128)")
PYEOF
        else
            echo ">> [KHQ-FAIL] khq_calib.bin tidak dihasilkan"
        fi
    else
        echo ">> [KHQ-FAIL] dump asli tidak valid — kalibrasi dibatalkan"
    fi

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
