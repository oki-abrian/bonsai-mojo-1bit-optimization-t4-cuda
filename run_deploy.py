#!/usr/bin/env python3
# ==============================================================================
# Runner Eksekusi Kaggle Container
# - Sumber kode segar diambil dari mount DATASET (bukan dari output kernel
#   sebelumnya — itu basi).
# - Cache build diambil dari output kernel sebelumnya (kernel_sources
#   self-reference) ke /kaggle/working, lalu di-restore oleh deploy script.
# ==============================================================================

import os
import shutil
import subprocess
import sys


def _find_fresh_source():
    """Direktori sumber segar: mount yang berisi deploy_on_kaggle.sh + main.mojo.
    Utamakan path dataset (menghindari salinan lama di output kernel)."""
    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "deploy_on_kaggle.sh" in files and "main.mojo" in files:
            candidates.append(root)
    for c in candidates:
        if "/datasets/" in c:
            return c
    return candidates[0] if candidates else None


def main():
    print(">> [RUNNER] Mempersiapkan source code Mojo...", flush=True)
    work_dir = "/kaggle/working/bonsai-1bit-t4-mojo"
    os.makedirs(work_dir, exist_ok=True)

    src_dir = _find_fresh_source()
    if not src_dir:
        print(">> [ERROR] Source tidak ditemukan di /kaggle/input!", flush=True)
        sys.exit(1)
    print(f">> [RUNNER] Sumber segar: {src_dir}", flush=True)

    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(work_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copyfile(s, d)
    print(f">> [RUNNER] Source tersalin ke {work_dir}", flush=True)

    # Ekstrak seluruh arsip tar / tar.gz sumber segar jika ada
    print(f">> [RUNNER] File dalam src_dir: {os.listdir(src_dir)}", flush=True)
    for f in os.listdir(src_dir):
        fp = os.path.join(src_dir, f)
        if f.endswith(".tar.gz"):
            print(f">> [RUNNER] Mengekstrak {f} ke {work_dir}...", flush=True)
            subprocess.run(["tar", "-xzf", fp, "-C", work_dir], check=True)
        elif f.endswith(".tar"):
            print(f">> [RUNNER] Mengekstrak {f} ke {work_dir}...", flush=True)
            subprocess.run(["tar", "-xf", fp, "-C", work_dir], check=True)

    # Cache build dari output run sebelumnya (jangan dari mount dataset).
    copied = False
    for root, dirs, files in os.walk("/kaggle/input"):
        if copied:
            break
        if "/datasets/" in root:
            continue
        for f in files:
            if f == "mojo_build_cache.tar.gz":
                src = os.path.join(root, f)
                dst = "/kaggle/working/mojo_build_cache.tar.gz"
                if not os.path.exists(dst):
                    print(f">> [RUNNER] Cache build: {src} -> {dst}", flush=True)
                    shutil.copyfile(src, dst)
                copied = True
                break

    script_path = os.path.join(work_dir, "deploy_on_kaggle.sh")
    if not os.path.exists(script_path):
        print(f">> [ERROR] File deploy_on_kaggle.sh tidak ditemukan di {work_dir}!", flush=True)
        print(f">> [DEBUG] Isi {work_dir}: {os.listdir(work_dir)}", flush=True)
        sys.exit(1)

    os.chmod(script_path, 0o755)

    # Variabel BONSAI_* dibekukan oleh push_to_kaggle.sh ke bonsai_env.sh, karena
    # environment shell LOKAL tidak ikut terkirim ke container Kaggle. Tanpa ini,
    # `BONSAI_COH_THINK=both ./push_to_kaggle.sh` diabaikan tanpa peringatan dan
    # eksperimen berjalan dengan default (negatif palsu).
    env_file = os.path.join(work_dir, "bonsai_env.sh")
    if os.path.exists(env_file):
        print(f">> [RUNNER] Menerapkan variabel dari {env_file}:", flush=True)
        try:
            with open(env_file) as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("export ") and "=" in line:
                        print(f">> [ENV]   {line[len('export '):]}", flush=True)
        except Exception as e:  # pragma: no cover - diagnostik saja
            print(f">> [WARN] Gagal membaca {env_file}: {e}", flush=True)
        # `set -a` -> semua variabel yang di-source otomatis ter-export, sehingga
        # diwarisi oleh deploy_on_kaggle.sh DAN biner mojo yang dijalankannya.
        cmd = (
            f"cd {work_dir} && set -a && . ./bonsai_env.sh && set +a "
            f"&& bash {script_path}"
        )
    else:
        print(">> [RUNNER] bonsai_env.sh tidak ada -> memakai environment default.", flush=True)
        cmd = f"cd {work_dir} && bash {script_path}"

    print(f">> [RUNNER] Menjalankan: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True)
    print(f">> [RUNNER] Eksekusi selesai dengan status code: {res.returncode}", flush=True)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
