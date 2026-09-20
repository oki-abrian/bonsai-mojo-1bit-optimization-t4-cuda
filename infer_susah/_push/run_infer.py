#!/usr/bin/env python3
# ==============================================================================
# Runner kernel Kaggle: TES INFERENSI SOAL SUSAH (Bonsai-27B 1-bit, T4)
#
# FILE INI ADALAH TEMPLATE. Blok di antara penanda ==BEGIN/==END== diisi ulang
# oleh tes_infer_susah.sh, lalu hasil render-nya (infer_susah/_push/run_infer.py)
# yang dikirim ke Kaggle.
#
# Kenapa mandiri (self-contained)? Kernel "script" Kaggle hanya membawa code_file;
# file pendamping (deploy_infer.sh / infer_config.json) TIDAK tersedia saat runtime
# — sudah terbukti: "[ERROR] deploy_infer.sh tidak ditemukan". Jadi skrip shell
# dan konfigurasi prompt ditanam langsung di sini.
#
# Hanya menjalankan SATU hal: build binary inferensi lalu jalankan satu prompt
# sulit. TIDAK ada self-test / kalibrasi KHQ / A/B.
# ==============================================================================

import json
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "bonsai-1bit-t4-mojo")

# ==BEGIN PROMPT==
HARD_PROMPT = 'Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu pipa B ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. Berapa jam total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? Tunjukkan langkah perhitungannya.'
# ==END PROMPT==
# ==BEGIN PROMPTSJSON==
PROMPTS_JSON = None
# ==END PROMPTSJSON==
# ==BEGIN MAXTOK==
MAX_TOKENS = 2048
# ==END MAXTOK==
# ==BEGIN DEPLOY==
DEPLOY_SH = '#!/usr/bin/env bash\n# ==============================================================================\n# deploy_infer.sh — TES INFERENSI SOAL SUSAH di Kaggle T4\n#\n# Hanya: siapkan toolchain -> kompilasi kernel CUDA sm_75 -> build main.mojo\n#        -> encode prompt dari infer_config.json -> jalankan inferensi.\n#\n# Dukung DUA bentuk config:\n#   * "prompt": "..."                 -> satu prompt\n#   * "prompts": [ {prompt, max_tokens}, ... ]  -> banyak prompt (tes pemisah)\n# Tiap prompt dijalankan di proses bonsai_infer terpisah (cache KV bersih).\n#\n# TIDAK menjalankan self-test, kalibrasi KHQ, atau A/B. Pipeline lengkap tetap\n# di deploy_on_kaggle.sh. Jalankan dari direktori repo (di-set oleh run_infer.py).\n# ==============================================================================\nset -e\n\necho "========================================================="\necho " TES INFERENSI SOAL SUSAH — Bonsai-27B 1-bit (T4)"\necho "========================================================="\n\nWORKING="/kaggle/working"\nREPO_DIR="$WORKING/bonsai-1bit-t4-mojo"\nCACHE_DIR="$WORKING/.cache_t4_build"\nDIST_DIR="$WORKING/dist_infer_susah"\nCFG="${SUSAH_CONFIG:-$WORKING/infer_config.json}"\n\nmkdir -p "$CACHE_DIR" "$DIST_DIR"\n\nif [ -f "$WORKING/mojo_build_cache.tar.gz" ]; then\n    echo ">> [CACHE] restore cache build..."\n    tar -xzf "$WORKING/mojo_build_cache.tar.gz" -C "$WORKING" && rm -f "$WORKING/mojo_build_cache.tar.gz"\nfi\n\nexport PIXI_HOME="$CACHE_DIR/pixi"\nexport PIXI_CACHE_DIR="$CACHE_DIR/pixi_cache"\nexport MAGIC_HOME="$CACHE_DIR/magic"\nexport MODULAR_HOME="$CACHE_DIR/modular"\nexport MOJO_CACHE_DIR="$CACHE_DIR/mojo"\nexport CCACHE_DIR="$CACHE_DIR/ccache"\nexport CCACHE_MAXSIZE="5G"\nexport PATH="$PIXI_HOME/bin:$HOME/.pixi/bin:$MAGIC_HOME/bin:$PATH"\n\n# ---------------------------------------------------------------- toolchain\nif ! command -v pixi >/dev/null 2>&1; then\n    echo ">> [TOOL] memasang pixi..."\n    mkdir -p "$PIXI_HOME/bin"\n    curl -fsSL https://pixi.sh/install.sh | bash\n    export PATH="$HOME/.pixi/bin:$PATH"\nfi\ncommand -v pixi >/dev/null 2>&1 \\\n    && echo ">> [OK] pixi: $(pixi --version 2>/dev/null || echo tersedia)" \\\n    || echo ">> [WARN] pixi belum ada di PATH"\n\ncd "$REPO_DIR"\n\n# ------------------------------------------------------- kernel CUDA sm_75\nexport PATH="/usr/local/cuda/bin:$PATH"\nexport BONSAI_CUDA_LIB="$WORKING/libbonsai_qmv_sm75.so"\nexport LD_LIBRARY_PATH="$WORKING:$REPO_DIR:$REPO_DIR/build:$WORKING/build:$DIST_DIR:${LD_LIBRARY_PATH:-}"\n\nif command -v nvcc >/dev/null 2>&1 && [ -f "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" ]; then\n    echo ">> [NVCC] kompilasi kernel sm_75 -> libbonsai_qmv_sm75.so"\n    nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC -Xptxas -v \\\n        "$REPO_DIR/src/csrc/qmv_sm75_kernel.cu" \\\n        -o "$WORKING/libbonsai_qmv_sm75.so"\n    cp "$WORKING/libbonsai_qmv_sm75.so" "$DIST_DIR/" 2>/dev/null || true\n    echo ">> [OK] .so dibuat ($(du -h "$WORKING/libbonsai_qmv_sm75.so" | cut -f1))"\nelse\n    echo ">> [WARN] nvcc tidak tersedia — jalur CUDA FFI tidak akan aktif"\nfi\n\n# --------------------------------------------------------- build main.mojo\nif command -v pixi >/dev/null 2>&1; then\n    echo ">> [BUILD] kompilasi main.mojo (CLI inferensi)..."\n    rm -f "$WORKING/bonsai_infer"\n    pixi run mojo build -I . main.mojo -o "$WORKING/bonsai_infer"\n    echo ">> [OK] binary: $WORKING/bonsai_infer ($(du -h "$WORKING/bonsai_infer" | cut -f1))"\nelse\n    echo ">> [ERROR] pixi tidak tersedia — tidak bisa build"\n    exit 1\nfi\n\n# ------------------------------------------------------------ direktori bobot\nKMODEL=""\nfor cand in /kaggle/input/*/bonsai-27b-mlx-1bit /kaggle/input/bonsai-27b-mlx-1bit \\\n            /kaggle/input/datasets/okiabrian/bonsai-27b-mlx-1bit; do\n    if [ -f "$cand/config.json" ]; then KMODEL="$cand"; break; fi\ndone\nif [ -z "$KMODEL" ]; then\n    echo ">> [ERROR] bobot tidak ditemukan di /kaggle/input"\n    echo ">> [DEBUG] isi /kaggle/input: $(ls /kaggle/input 2>/dev/null | tr \'\\n\' \' \')"\n    exit 1\nfi\necho ">> [OK] bobot: $KMODEL"\n\n# ------------------------------------------------------- encode prompt sulit\nif [ ! -f "$CFG" ]; then\n    echo ">> [ERROR] config prompt tidak ada: $CFG"\n    exit 1\nfi\n# Normalisasi daftar prompt: dukung "prompt" tunggal ATAU "prompts" (list).\n# Tiap entri: string, atau {"prompt": ..., "max_tokens": N, "label": "...",\n# "env": {"BONSAI_NO_FUSE": "1"}}. "label"/"env" dipertahankan apa adanya.\nPLIST="$WORKING/prompt_list.json"\npython3 - "$CFG" "$PLIST" <<\'PY\'\nimport json, sys\ncfg = json.load(open(sys.argv[1]))\nitems = cfg.get("prompts")\nif not items:\n    items = [{"prompt": cfg.get("prompt", ""), "max_tokens": cfg.get("max_tokens", 512)}]\nnorm = []\nfor it in items:\n    if isinstance(it, str):\n        norm.append({"prompt": it, "max_tokens": int(cfg.get("max_tokens", 512))})\n    else:\n        e = {"prompt": it.get("prompt", ""),\n             "max_tokens": int(it.get("max_tokens", cfg.get("max_tokens", 512)))}\n        if it.get("label"):\n            e["label"] = str(it["label"])\n        if it.get("env"):\n            e["env"] = {str(k): str(v) for k, v in dict(it["env"]).items()}\n        norm.append(e)\njson.dump(norm, open(sys.argv[2], "w"), ensure_ascii=False)\nprint(f">> [PROMPT] {len(norm)} prompt akan dijalankan")\nfor i, it in enumerate(norm, 1):\n    extra = ""\n    if it.get("label"):\n        extra += f" label={it[\'label\']!r}"\n    if it.get("env"):\n        extra += " env=" + ",".join(f"{k}={v}" for k, v in it["env"].items())\n    print(f">> [PROMPT] {i}. max_tokens={it[\'max_tokens\']}{extra} | {it[\'prompt\'][:160]}")\nPY\nNPROMPT="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$PLIST")"\n\n# ------------------------------------------------------------- inferensi T4\nexport BONSAI_USE_GPU=1\nexport BONSAI_DUMP_TOP2=1\nexport MOJO_ENABLE_STACK_TRACE_ON_ERROR=1\n\n# ------------------------------------------------- parameter sampling (opsional)\n# Diteruskan ke main.mojo lewat env. Default temperature_x100=0 -> greedy/argmax,\n# jadi jalur greedy (dan kontrak bit-exact gate KHQ) tidak berubah.\n# Resep resmi Bonsai (mode thinking): temp 0.7 | top-k 20 | top-p 0.95 | min-p 0.\n# rep_penalty TIDAK ada di spesifikasi resmi -> default 100 = MATI.\neval "$(python3 - "$CFG" <<\'PYSAMP\'\nimport json, sys\nc = json.load(open(sys.argv[1]))\ndef g(k, d):\n    try:\n        return int(c.get(k, d))\n    except Exception:\n        return d\nprint(f\'export BONSAI_TEMP_X100={g("temperature_x100", 0)}\')\nprint(f\'export BONSAI_TOP_K={g("top_k", 20)}\')\nprint(f\'export BONSAI_TOP_P_X1000={g("top_p_x1000", 950)}\')\nprint(f\'export BONSAI_MIN_P_X1000={g("min_p_x1000", 0)}\')\nprint(f\'export BONSAI_REP_PENALTY_X100={g("rep_penalty_x100", 100)}\')\nprint(f\'export BONSAI_REP_WINDOW={g("rep_window", 256)}\')\nprint(f\'export BONSAI_SEED={g("seed", 1234)}\')\nPYSAMP\n)"\nif [ "$BONSAI_TEMP_X100" = "0" ]; then\n    echo ">> [SAMPLE] temperature_x100=0 -> GREEDY (argmax)"\nelse\n    echo ">> [SAMPLE] temperature=$((BONSAI_TEMP_X100 / 100)).$((BONSAI_TEMP_X100 % 100))" \\\n         "| top_k=$BONSAI_TOP_K" \\\n         "| top_p=$((BONSAI_TOP_P_X1000 / 1000)).$((BONSAI_TOP_P_X1000 % 1000))" \\\n         "| min_p=$((BONSAI_MIN_P_X1000 / 1000)).$((BONSAI_MIN_P_X1000 % 1000))" \\\n         "| rep_penalty=$((BONSAI_REP_PENALTY_X100 / 100)).$((BONSAI_REP_PENALTY_X100 % 100))" \\\n         "| seed=$BONSAI_SEED"\nfi\n\nnvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw --format=csv,noheader 2>/dev/null || true\n\nLOG_ALL="$DIST_DIR/hard_infer.log"\n: > "$LOG_ALL"\n\necho ""\necho "========================================================="\necho " MULAI GENERASI — $NPROMPT prompt"\necho "========================================================="\n\n# Satu proses bonsai_infer PER prompt -> tiap prompt mulai dari cache kosong\n# (tidak ada kontaminasi KV antar prompt). Bobot dimuat ulang tiap iterasi.\ni=0\nwhile [ "$i" -lt "$NPROMPT" ]; do\n    i=$((i + 1))\n\n    # Penanda run + prompt. Prompt ditulis sebagai JSON supaya aman untuk teks\n    # multi-baris / bertanda kutip (detok_infer.py membacanya kembali).\n    python3 - "$PLIST" "$i" <<\'PY\' | tee -a "$LOG_ALL"\nimport json, sys\nit = json.load(open(sys.argv[1]))[int(sys.argv[2]) - 1]\nprint("")\nprint("=" * 60)\nextra = f" label={json.dumps(it[\'label\'], ensure_ascii=False)}" if it.get("label") else ""\nprint(f">> [RUN] idx={sys.argv[2]} max_tokens={it[\'max_tokens\']}{extra}")\nprint(">> [PROMPT-JSON] " + json.dumps(it["prompt"], ensure_ascii=False))\nprint(">> [PROMPT] " + it["prompt"].replace("\\n", " "))\nif it.get("env"):\n    print(">> [ENV] " + " ".join(f"{k}={v}" for k, v in it["env"].items()))\nPY\n\n    P_MAX="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])-1][\'max_tokens\'])" "$PLIST" "$i")"\n\n    # Env per-prompt (opsional): dipakai untuk A/B jalur kernel pada prompt yang\n    # sama, mis. BONSAI_NO_FUSE=1 atau BONSAI_PREFILL_PER_TOKEN=1.\n    P_ENV=()\n    while IFS= read -r kv; do\n        [ -n "$kv" ] && P_ENV+=("$kv")\n    done < <(python3 -c "\nimport json,sys\ne=json.load(open(sys.argv[1]))[int(sys.argv[2])-1]\nfor k,v in (e.get(\'env\') or {}).items(): print(f\'{k}={v}\')\n" "$PLIST" "$i")\n\n    # Encode prompt ke-i dengan chat template model (sama seperti deploy_on_kaggle.sh).\n    P_TOKENS="$(python3 - "$KMODEL" "$PLIST" "$i" <<\'PY\'\nimport json, sys\nfrom transformers import AutoTokenizer\nmdir, plist, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])\nprompt = json.load(open(plist))[idx - 1]["prompt"]\ntok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)\nmsgs = [{"role": "user", "content": prompt}]\ntext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)\nids = tok.encode(text, add_special_tokens=False)\nprint(",".join(str(i) for i in ids))\nPY\n)" || { echo ">> [ERROR] encode prompt $i GAGAL — dilewati" | tee -a "$LOG_ALL"; continue; }\n\n    NTOK="$(printf \'%s\' "$P_TOKENS" | tr \',\' \'\\n\' | grep -c . || true)"\n    echo ">> [OK] prompt $i ter-encode: $NTOK token" | tee -a "$LOG_ALL"\n\n    # `env` tanpa penetapan apa pun hanya menjalankan perintah -> aman saat P_ENV kosong.\n    env "${P_ENV[@]}" "$WORKING/bonsai_infer" --model-dir "$KMODEL" \\\n        --prompt-tokens "$P_TOKENS" --max-tokens "$P_MAX" --gpu \\\n        2>&1 | tee -a "$LOG_ALL" \\\n        || echo ">> [WARN] inferensi prompt $i gagal — periksa log di atas" | tee -a "$LOG_ALL"\ndone\n\necho ""\necho ">> [SELESAI] log gabungan: $LOG_ALL"\n'
# ==END DEPLOY==
# ==BEGIN CFGJSON==
CFG_JSON = '{"prompt": "Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu pipa B ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. Berapa jam total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? Tunjukkan langkah perhitungannya.", "max_tokens": 2048, "temperature_x100": 70, "top_k": 20, "top_p_x1000": 950, "min_p_x1000": 0, "rep_penalty_x100": 100, "rep_window": 256, "seed": 1234, "catatan": "RUN TUNGGAL untuk melihat kualitas output (14 Sep 2026). Mode prompt tunggal, max_tokens 2048, prompt = soal penalaran (pipa A/B). Resep resmi Bonsai (mode thinking): temperature 0.70, top-k 20, top-p 0.95, min-p 0. rep_penalty_x100=100 artinya MATI. temperature_x100=0 -> greedy (argmax). Parameter diteruskan ke main.mojo lewat env BONSAI_TEMP_X100 / BONSAI_TOP_K / BONSAI_TOP_P_X1000 / BONSAI_MIN_P_X1000 dst."}'
# ==END CFGJSON==


def log(msg):
    print(f">> [SUSAH] {msg}", flush=True)


def find_source_dir():
    """Direktori sumber segar di /kaggle/input.

    Syarat sama seperti run_deploy.py yang sudah terbukti: berisi
    deploy_on_kaggle.sh DAN main.mojo. Direktori dataset diutamakan supaya
    tidak memakai salinan basi dari output kernel lama.
    """
    cands = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "deploy_on_kaggle.sh" in files and "main.mojo" in files:
            cands.append(root)
    for c in cands:
        if "/datasets/" in c:
            return c
    return cands[0] if cands else None


def copy_source():
    src = find_source_dir()
    if not src:
        log("[ERROR] sumber tidak ditemukan di /kaggle/input")
        dump_listings()
        sys.exit(1)
    log(f"sumber segar: {src}")
    os.makedirs(REPO, exist_ok=True)
    for item in os.listdir(src):
        s, d = os.path.join(src, item), os.path.join(REPO, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)
    for f in os.listdir(src):
        if f.endswith(".tar.gz") or f.endswith(".tar"):
            log(f"ekstrak {f}")
            flag = "-xzf" if f.endswith(".tar.gz") else "-xf"
            subprocess.run(["tar", flag, os.path.join(src, f), "-C", REPO], check=True)
    log(f"sumber tersalin ke {REPO}")


def copy_cache():
    """Cache build dari output kernel sebelumnya (kernel_sources)."""
    for root, dirs, files in os.walk("/kaggle/input"):
        if "/datasets/" in root:
            continue
        if "mojo_build_cache.tar.gz" in files:
            dst = os.path.join(WORK, "mojo_build_cache.tar.gz")
            if not os.path.exists(dst):
                shutil.copyfile(os.path.join(root, "mojo_build_cache.tar.gz"), dst)
                log(f"cache build diambil dari {root}")
            return
    log("[WARN] cache build tidak ditemukan — kompilasi dari nol (lebih lama)")


def dump_listings():
    for d in ("/kaggle", "/kaggle/working", os.getcwd(), "/kaggle/input"):
        try:
            items = sorted(os.listdir(d))
            log(f"[DEBUG] isi {d} ({len(items)}): {items[:40]}")
        except Exception as e:
            log(f"[DEBUG] {d}: {e}")


def write_deploy_sh():
    """Tanam skrip shell ke /kaggle/working lalu jalankan dari sana."""
    if not DEPLOY_SH or DEPLOY_SH.startswith("PLACEHOLDER"):
        log("[ERROR] DEPLOY_SH kosong — template belum di-render oleh tes_infer_susah.sh")
        sys.exit(1)
    p = os.path.join(WORK, "deploy_infer.sh")
    with open(p, "w") as f:
        f.write(DEPLOY_SH)
    os.chmod(p, 0o755)
    log(f"skrip shell ditanam: {p} ({len(DEPLOY_SH)} byte)")
    return p


def write_config():
    """Tulis config LENGKAP (termasuk parameter sampling) ke /kaggle/working.

    Parameter sampling HARUS ikut, kalau tidak deploy_infer.sh membaca default 0
    dan inferensi jatuh kembali ke greedy — inilah bug yang membuat run pertama
    dengan sampling menghasilkan output identik dengan greedy.

    Dua mode:
      * PROMPTS_JSON diisi  -> tulis cfg["prompts"] (banyak prompt sekaligus)
      * PROMPTS_JSON kosong -> tulis cfg["prompt"]  (satu prompt, jalur lama)
    """
    try:
        cfg = json.loads(CFG_JSON)
    except Exception as e:
        log(f"[WARN] CFG_JSON tidak terbaca ({e}) — pakai default")
        cfg = {}

    prompts = None
    if PROMPTS_JSON:
        try:
            prompts = json.loads(PROMPTS_JSON)
        except Exception as e:
            log(f"[WARN] PROMPTS_JSON tidak terbaca ({e}) — fallback prompt tunggal")
            prompts = None

    if prompts:
        cfg["prompts"] = prompts
        cfg.pop("prompt", None)
        log(f"mode MULTI-PROMPT: {len(prompts)} prompt")
        for i, it in enumerate(prompts, 1):
            txt = it if isinstance(it, str) else it.get("prompt", "")
            mt = "" if isinstance(it, str) else f" [max={it.get('max_tokens')}]"
            log(f"  {i}.{mt} {txt[:120]}")
    else:
        cfg["prompt"] = HARD_PROMPT
        cfg["max_tokens"] = MAX_TOKENS
        cfg.pop("prompts", None)
        log(f"mode PROMPT TUNGGAL ({len(HARD_PROMPT)} char)")

    p = os.path.join(WORK, "infer_config.json")
    with open(p, "w") as f:
        json.dump(cfg, f, ensure_ascii=False)
    t = int(cfg.get("temperature_x100", 0))
    log(f"max_tokens default = {cfg.get('max_tokens')} | temperature_x100 = {t}"
        + (" (GREEDY)" if t == 0 else " (SAMPLING)"))
    log(f"kunci config yang diteruskan: {sorted(cfg.keys())}")
    return p


def main():
    log("mulai")
    copy_source()
    copy_cache()
    cfg = write_config()
    deploy = write_deploy_sh()

    env = dict(os.environ)
    env["SUSAH_CONFIG"] = cfg
    log(f"menjalankan {deploy}")
    res = subprocess.run(f"cd {REPO} && bash {deploy}", shell=True, env=env)
    log(f"selesai, status code: {res.returncode}")
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
