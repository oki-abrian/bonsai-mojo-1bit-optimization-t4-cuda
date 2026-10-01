#!/usr/bin/env python3
# ==============================================================================
# Kernel Kaggle: unduh guru GGUF dari HuggingFace lalu terbitkan sebagai
# dataset Kaggle okiabrian/qwen38-27b-gsq-rco-iq3s.
# SELURUH proses berat (unduh 11,8 GB + unggah ke dataset) berjalan di sisi
# Kaggle — mesin lokal hanya mengirim skrip ini (beberapa KB).
# ==============================================================================
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ID = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
FNAME = "Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf"
DS_ID = "okiabrian/qwen38-27b-gsq-rco-iq3s"
MIN_BYTES = 10 * 1024**3  # sanity: berkas asli 11.771.546.784 byte (~11,0 GiB);
# ambang 11 GiB pernah salah membatalkan unduhan yang sebenarnya utuh (v1).


def log(*a):
    print(*a, flush=True)


def run(cmd, **kw):
    log(">> [RUN]", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def main():
    # 0. Gagal-cepat: pastikan auth Kaggle & huggingface_hub tersedia SEBELUM
    #    menghabiskan waktu unduh.
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        KaggleApi().authenticate()
        log(">> [OK] autentikasi Kaggle di dalam kernel")
    except Exception as e:
        raise SystemExit("FATAL: auth Kaggle dalam kernel gagal: %r" % (e,))
    try:
        import huggingface_hub

        log(">> [OK] huggingface_hub", huggingface_hub.__version__)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"],
            check=True)

    from huggingface_hub import hf_hub_download

    # 1. Unduh ke /kaggle/tmp (ephemeral — TIDAK menjadi output kernel).
    dst_dir = "/kaggle/tmp/hf"
    log(">> [1/4] Mengunduh", REPO_ID + "/" + FNAME, "...")
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO_ID, filename=FNAME, local_dir=dst_dir)
    size = os.path.getsize(p)
    log(">> [OK] unduhan selesai: %s (%d bytes, %.1f menit)"
        % (p, size, (time.time() - t0) / 60.0))
    if size < MIN_BYTES:
        raise SystemExit(
            "FATAL: ukuran %d < %d — kemungkinan LFS pointer / unduhan rusak"
            % (size, MIN_BYTES))
    with open(p, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise SystemExit("FATAL: magic berkas %r bukan GGUF" % (magic,))

    # 2. Staging folder dataset (rename, bukan copy — hemat disk).
    ds_dir = "/kaggle/tmp/ds"
    shutil.rmtree(ds_dir, ignore_errors=True)
    os.makedirs(ds_dir)
    meta = {
        "title": DS_ID.split("/", 1)[1],
        "id": DS_ID,
        "licenses": [{"name": "other"}],
    }
    with open(os.path.join(ds_dir, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    os.rename(p, os.path.join(ds_dir, FNAME))
    log(">> [2/4] staging siap:", sorted(os.listdir(ds_dir)))

    # 3. Buat dataset (kaggle CLI di dalam kernel ter-auth sebagai pemilik).
    log(">> [3/4] kaggle datasets create ...")
    t1 = time.time()
    r = run(["kaggle", "datasets", "create", "-p", ds_dir, "-r", "skip"],
            check=False)
    if r.returncode != 0:
        log(">> [WARN] create gagal (mungkin dataset sudah ada) — coba version")
        r = run(["kaggle", "datasets", "version", "-p", ds_dir, "-r", "skip",
                 "-m", "re-sync gguf"], check=False)
        if r.returncode != 0:
            raise SystemExit("FATAL: gagal membuat/memperbarui dataset")
    log(">> [OK] unggah dataset selesai (%.1f menit)"
        % ((time.time() - t1) / 60.0))

    # 4. Tunggu dataset ready lalu tampilkan isinya.
    #    CATATAN v2: endpoint GetDatasetStatus membalas 403 Forbidden selama
    #    ~2 menit pertama setelah create (propagasi) — itu BUKAN kegagalan.
    #    Jangan FATAL hanya karena teks "Error" muncul di output CLI (v2
    #    salah mati di sini walau unggah 11,8 GB sudah sukses). Cukup tunggu
    #    "ready"; kalau habis batas, keluar NON-fatal — unggahan sudah aman.
    log(">> [4/4] menunggu dataset ready ...")
    ready = False
    for i in range(60):
        s = subprocess.run(["kaggle", "datasets", "status", DS_ID],
                           capture_output=True, text=True)
        out = (s.stdout + s.stderr).strip()
        if "ready" in out.lower():
            ready = True
            log("[%d] status: ready" % i)
            break
        log("[%d] belum ready (propagasi 403 = normal): %s"
            % (i, out[-120:]))
        time.sleep(30)
    run(["kaggle", "datasets", "files", DS_ID], check=False)
    # /kaggle/tmp dibiarkan — ephemeral, tidak jadi output; /kaggle/working
    # tetap kosong agar output kernel hanya berisi log.
    log(">> [SELESAI] dataset:", DS_ID, "| ready:", ready)


main()
