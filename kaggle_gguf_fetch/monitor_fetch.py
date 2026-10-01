#!/usr/bin/env python3
# Pemantau kernel gguf-fetch-qwen38-iq3s: polling status sampai tuntas,
# lalu verifikasi dataset dan tulis ringkasan ke fetch_result.txt.
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL = "okiabrian/gguf-fetch-qwen38-iq3s"
DS_ID = "okiabrian/qwen38-27b-gsq-rco-iq3s"
LOG = os.path.join(HERE, "monitor.log")
RESULT = os.path.join(HERE, "fetch_result.txt")
MAX_SEC = 5 * 3600


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    start = time.time()
    final = None
    failmsg = ""
    while time.time() - start < MAX_SEC:
        try:
            s = api.kernels_status(KERNEL)
        except Exception as e:
            log("API error (diulang): %r" % (e,))
            time.sleep(60)
            continue
        # Versi kaggle baru mengembalikan objek bertipe
        # (ApiGetKernelSessionStatusResponse), versi lama dict — dukung dua-duanya.
        if isinstance(s, dict):
            st = s.get("status", "?")
            failmsg = s.get("failureMessage") or ""
        else:
            st = getattr(s, "status", "?")
            failmsg = getattr(s, "failureMessage", None) or ""
        # Status bisa "RUNNING" (dict lama) atau "KernelWorkerStatus.ERROR"
        # (enum bertipe baru) — potong prefix sebelum titik terakhir.
        tail = str(st).upper().rsplit(".", 1)[-1]
        log("status: %s" % tail)
        if tail.startswith(("COMPLETE", "ERROR", "CANCEL")):
            final = tail
            break
        time.sleep(60)

    if final is None:
        log("LEWAT BATAS waktu pantau — kernel masih berjalan?")
        return 2

    if final != "COMPLETE":
        log("KERNEL GAGAL: %s | failureMessage: %s" % (final, failmsg))
        # tarik log kernel untuk diagnosis
        try:
            out = api.kernels_output(KERNEL, path=HERE)
            log("log kernel diunduh ke %s: %r" % (HERE, out))
        except Exception as e:
            log("gagal mengunduh log: %r" % (e,))
        with open(RESULT, "w") as f:
            f.write("GAGAL: %s\n%s\n" % (final, failmsg))
        return 1

    log("kernel COMPLETE — verifikasi dataset ...")
    time.sleep(30)
    s = subprocess.run(["kaggle", "datasets", "status", DS_ID],
                       capture_output=True, text=True)
    ds_status = (s.stdout + s.stderr).strip()
    log("dataset status: %s" % ds_status)
    f = subprocess.run(["kaggle", "datasets", "files", DS_ID],
                       capture_output=True, text=True)
    ds_files = (f.stdout + f.stderr).strip()
    log("dataset files:\n%s" % ds_files)
    ok = "ready" in ds_status.lower() and ".gguf" in ds_files.lower()
    with open(RESULT, "w") as out:
        out.write("kernel: %s (%s)\n" % (KERNEL, final))
        out.write("dataset: %s\nstatus: %s\n\nfiles:\n%s\n"
                  % (DS_ID, ds_status, ds_files))
        out.write("VERDICT: %s\n" % ("OK" if ok else "PERIKSA MANUAL"))
    log("selesai — ringkasan di %s" % RESULT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
