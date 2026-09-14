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
# sulit. TIDAK ada self-test / kalibrasi / A/B.
# ==============================================================================

import json
import os
import shutil
import subprocess
import sys

WORK = "/kaggle/working"
REPO = os.path.join(WORK, "bonsai-1bit-t4-mojo")

# ==BEGIN PROMPT==
HARD_PROMPT = "PLACEHOLDER_PROMPT"
# ==END PROMPT==
# ==BEGIN PROMPTSJSON==
PROMPTS_JSON = None
# ==END PROMPTSJSON==
# ==BEGIN MAXTOK==
MAX_TOKENS = 512
# ==END MAXTOK==
# ==BEGIN DEPLOY==
DEPLOY_SH = "PLACEHOLDER_DEPLOY"
# ==END DEPLOY==
# ==BEGIN CFGJSON==
CFG_JSON = "PLACEHOLDER_CFG"
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
