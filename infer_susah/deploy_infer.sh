#!/usr/bin/env bash
# ==============================================================================
# deploy_infer.sh — TES INFERENSI SOAL SUSAH di Kaggle T4
#
# Hanya: siapkan toolchain -> kompilasi kernel CUDA sm_75 -> build main.mojo
#        -> encode prompt dari infer_config.json -> jalankan inferensi.
#
# Dukung DUA bentuk config:
#   * "prompt": "..."                 -> satu prompt
#   * "prompts": [ {prompt, max_tokens}, ... ]  -> banyak prompt (tes pemisah)
# Tiap prompt dijalankan di proses bonsai_infer terpisah (cache KV bersih).
#
# TIDAK menjalankan self-test, kalibrasi KHQ, atau A/B. Pipeline lengkap tetap
# di deploy_on_kaggle.sh. Jalankan dari direktori repo (di-set oleh run_infer.py).
# ==============================================================================
set -e

echo "========================================================="
echo " TES INFERENSI SOAL SUSAH — Bonsai-27B 1-bit (T4)"
echo "========================================================="

WORKING="/kaggle/working"
REPO_DIR="$WORKING/bonsai-1bit-t4-mojo"
CACHE_DIR="$WORKING/.cache_t4_build"
DIST_DIR="$WORKING/dist_infer_susah"
CFG="${SUSAH_CONFIG:-$WORKING/infer_config.json}"

mkdir -p "$CACHE_DIR" "$DIST_DIR"

if [ -f "$WORKING/mojo_build_cache.tar.gz" ]; then
    echo ">> [CACHE] restore cache build..."
    tar -xzf "$WORKING/mojo_build_cache.tar.gz" -C "$WORKING" && rm -f "$WORKING/mojo_build_cache.tar.gz"
fi

export PIXI_HOME="$CACHE_DIR/pixi"
export PIXI_CACHE_DIR="$CACHE_DIR/pixi_cache"
export MAGIC_HOME="$CACHE_DIR/magic"
export MODULAR_HOME="$CACHE_DIR/modular"
export MOJO_CACHE_DIR="$CACHE_DIR/mojo"
export CCACHE_DIR="$CACHE_DIR/ccache"
export CCACHE_MAXSIZE="5G"
export PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$MAGIC_HOME/bin:$PATH"

# ---------------------------------------------------------------- toolchain
if ! command -v pixi >/dev/null 2>&1; then
    echo ">> [TOOL] memasang pixi..."
    mkdir -p "$PIXI_HOME/bin"
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi
command -v pixi >/dev/null 2>&1 \
    && echo ">> [OK] pixi: $(pixi --version 2>/dev/null || echo tersedia)" \
    || echo ">> [WARN] pixi belum ada di PATH"

cd "$REPO_DIR"

# ------------------------------------------------------- kernel CUDA sm_75
export PATH="/usr/local/cuda/bin:$PATH"
export BONSAI_CUDA_LIB="$WORKING/libbonsai_qmv_sm75.so"
export LD_LIBRARY_PATH="$WORKING:$REPO_DIR:$REPO_DIR/build:$WORKING/build:$DIST_DIR:${LD_LIBRARY_PATH:-}"

if command -v nvcc >/dev/null 2>&1 && [ -f "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" ]; then
    echo ">> [NVCC] kompilasi kernel sm_75 -> libbonsai_qmv_sm75.so"
    nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC -Xptxas -v \
        "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" \
        -o "$WORKING/libbonsai_qmv_sm75.so"
    cp "$WORKING/libbonsai_qmv_sm75.so" "$DIST_DIR/" 2>/dev/null || true
    echo ">> [OK] .so dibuat ($(du -h "$WORKING/libbonsai_qmv_sm75.so" | cut -f1))"
else
    echo ">> [WARN] nvcc tidak tersedia — jalur CUDA FFI tidak akan aktif"
fi

# --------------------------------------------------------- build main.mojo
if command -v pixi >/dev/null 2>&1; then
    echo ">> [BUILD] kompilasi main.mojo (CLI inferensi)..."
    rm -f "$WORKING/bonsai_infer"
    pixi run mojo build -I . main.mojo -o "$WORKING/bonsai_infer"
    echo ">> [OK] binary: $WORKING/bonsai_infer ($(du -h "$WORKING/bonsai_infer" | cut -f1))"
else
    echo ">> [ERROR] pixi tidak tersedia — tidak bisa build"
    exit 1
fi

# ------------------------------------------------------------ direktori bobot
KMODEL=""
for cand in /kaggle/input/*/bonsai-27b-mlx-1bit /kaggle/input/bonsai-27b-mlx-1bit \
            /kaggle/input/datasets/okiabrian/bonsai-27b-mlx-1bit; do
    if [ -f "$cand/config.json" ]; then KMODEL="$cand"; break; fi
done
if [ -z "$KMODEL" ]; then
    echo ">> [ERROR] bobot tidak ditemukan di /kaggle/input"
    echo ">> [DEBUG] isi /kaggle/input: $(ls /kaggle/input 2>/dev/null | tr '\n' ' ')"
    exit 1
fi
echo ">> [OK] bobot: $KMODEL"

# ------------------------------------------------------- encode prompt sulit
if [ ! -f "$CFG" ]; then
    echo ">> [ERROR] config prompt tidak ada: $CFG"
    exit 1
fi
# Normalisasi daftar prompt: dukung "prompt" tunggal ATAU "prompts" (list).
# Tiap entri: string, atau {"prompt": ..., "max_tokens": N, "label": "...",
# "env": {"BONSAI_NO_FUSE": "1"}}. "label"/"env" dipertahankan apa adanya.
PLIST="$WORKING/prompt_list.json"
python3 - "$CFG" "$PLIST" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
items = cfg.get("prompts")
if not items:
    items = [{"prompt": cfg.get("prompt", ""), "max_tokens": cfg.get("max_tokens", 512)}]
norm = []
for it in items:
    if isinstance(it, str):
        norm.append({"prompt": it, "max_tokens": int(cfg.get("max_tokens", 512))})
    else:
        e = {"prompt": it.get("prompt", ""),
             "max_tokens": int(it.get("max_tokens", cfg.get("max_tokens", 512)))}
        if it.get("label"):
            e["label"] = str(it["label"])
        if it.get("env"):
            e["env"] = {str(k): str(v) for k, v in dict(it["env"]).items()}
        norm.append(e)
json.dump(norm, open(sys.argv[2], "w"), ensure_ascii=False)
print(f">> [PROMPT] {len(norm)} prompt akan dijalankan")
for i, it in enumerate(norm, 1):
    extra = ""
    if it.get("label"):
        extra += f" label={it['label']!r}"
    if it.get("env"):
        extra += " env=" + ",".join(f"{k}={v}" for k, v in it["env"].items())
    print(f">> [PROMPT] {i}. max_tokens={it['max_tokens']}{extra} | {it['prompt'][:160]}")
PY
NPROMPT="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$PLIST")"

# ------------------------------------------------------------- inferensi T4
export BONSAI_USE_GPU=1
export BONSAI_DUMP_TOP2=1
export MOJO_ENABLE_STACK_TRACE_ON_ERROR=1

# ------------------------------------------------- parameter sampling (opsional)
# Diteruskan ke main.mojo lewat env. Default temperature_x100=0 -> greedy/argmax,
# jadi jalur greedy (dan kontrak bit-exact gate KHQ) tidak berubah.
# Resep resmi Bonsai (mode thinking): temp 0.7 | top-k 20 | top-p 0.95 | min-p 0.
# rep_penalty TIDAK ada di spesifikasi resmi -> default 100 = MATI.
eval "$(python3 - "$CFG" <<'PYSAMP'
import json, sys
c = json.load(open(sys.argv[1]))
def g(k, d):
    try:
        return int(c.get(k, d))
    except Exception:
        return d
print(f'export BONSAI_TEMP_X100={g("temperature_x100", 0)}')
print(f'export BONSAI_TOP_K={g("top_k", 20)}')
print(f'export BONSAI_TOP_P_X1000={g("top_p_x1000", 950)}')
print(f'export BONSAI_MIN_P_X1000={g("min_p_x1000", 0)}')
print(f'export BONSAI_REP_PENALTY_X100={g("rep_penalty_x100", 100)}')
print(f'export BONSAI_REP_WINDOW={g("rep_window", 256)}')
print(f'export BONSAI_SEED={g("seed", 1234)}')
PYSAMP
)"
if [ "$BONSAI_TEMP_X100" = "0" ]; then
    echo ">> [SAMPLE] temperature_x100=0 -> GREEDY (argmax)"
else
    echo ">> [SAMPLE] temperature=$((BONSAI_TEMP_X100 / 100)).$((BONSAI_TEMP_X100 % 100))" \
         "| top_k=$BONSAI_TOP_K" \
         "| top_p=$((BONSAI_TOP_P_X1000 / 1000)).$((BONSAI_TOP_P_X1000 % 1000))" \
         "| min_p=$((BONSAI_MIN_P_X1000 / 1000)).$((BONSAI_MIN_P_X1000 % 1000))" \
         "| rep_penalty=$((BONSAI_REP_PENALTY_X100 / 100)).$((BONSAI_REP_PENALTY_X100 % 100))" \
         "| seed=$BONSAI_SEED"
fi

nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader 2>/dev/null || true

LOG_ALL="$DIST_DIR/hard_infer.log"
: > "$LOG_ALL"

echo ""
echo "========================================================="
echo " MULAI GENERASI — $NPROMPT prompt"
echo "========================================================="

# Satu proses bonsai_infer PER prompt -> tiap prompt mulai dari cache kosong
# (tidak ada kontaminasi KV antar prompt). Bobot dimuat ulang tiap iterasi.
i=0
while [ "$i" -lt "$NPROMPT" ]; do
    i=$((i + 1))

    # Penanda run + prompt. Prompt ditulis sebagai JSON supaya aman untuk teks
    # multi-baris / bertanda kutip (detok_infer.py membacanya kembali).
    python3 - "$PLIST" "$i" <<'PY' | tee -a "$LOG_ALL"
import json, sys
it = json.load(open(sys.argv[1]))[int(sys.argv[2]) - 1]
print("")
print("=" * 60)
extra = f" label={json.dumps(it['label'], ensure_ascii=False)}" if it.get("label") else ""
print(f">> [RUN] idx={sys.argv[2]} max_tokens={it['max_tokens']}{extra}")
print(">> [PROMPT-JSON] " + json.dumps(it["prompt"], ensure_ascii=False))
print(">> [PROMPT] " + it["prompt"].replace("\n", " "))
if it.get("env"):
    print(">> [ENV] " + " ".join(f"{k}={v}" for k, v in it["env"].items()))
PY

    P_MAX="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])-1]['max_tokens'])" "$PLIST" "$i")"

    # Env per-prompt (opsional): dipakai untuk A/B jalur kernel pada prompt yang
    # sama, mis. BONSAI_NO_FUSE=1 atau BONSAI_PREFILL_PER_TOKEN=1.
    P_ENV=()
    while IFS= read -r kv; do
        [ -n "$kv" ] && P_ENV+=("$kv")
    done < <(python3 -c "
import json,sys
e=json.load(open(sys.argv[1]))[int(sys.argv[2])-1]
for k,v in (e.get('env') or {}).items(): print(f'{k}={v}')
" "$PLIST" "$i")

    # Encode prompt ke-i dengan chat template model (sama seperti deploy_on_kaggle.sh).
    P_TOKENS="$(python3 - "$KMODEL" "$PLIST" "$i" <<'PY'
import json, sys
from transformers import AutoTokenizer
mdir, plist, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompt = json.load(open(plist))[idx - 1]["prompt"]
tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)
print(",".join(str(i) for i in ids))
PY
)" || { echo ">> [ERROR] encode prompt $i GAGAL — dilewati" | tee -a "$LOG_ALL"; continue; }

    NTOK="$(printf '%s' "$P_TOKENS" | tr ',' '\n' | grep -c . || true)"
    echo ">> [OK] prompt $i ter-encode: $NTOK token" | tee -a "$LOG_ALL"

    # `env` tanpa penetapan apa pun hanya menjalankan perintah -> aman saat P_ENV kosong.
    env "${P_ENV[@]}" "$WORKING/bonsai_infer" --model-dir "$KMODEL" \
        --prompt-tokens "$P_TOKENS" --max-tokens "$P_MAX" --gpu \
        2>&1 | tee -a "$LOG_ALL" \
        || echo ">> [WARN] inferensi prompt $i gagal — periksa log di atas" | tee -a "$LOG_ALL"
done

echo ""
echo ">> [SELESAI] log gabungan: $LOG_ALL"
