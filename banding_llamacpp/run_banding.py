# ==============================================================================
# run_banding.py — Kernel Kaggle MANDIRI: jalankan llama.cpp CUDA pada model yang
#                  SAMA dengan port Mojo, lalu cetak teksnya untuk dibandingkan.
#
# Kenapa terpisah: tahap ini butuh unduh + build llama.cpp (~10-20 menit) dan
# tidak ada hubungannya dengan pipeline build/UJI T4 yang sudah tervalidasi.
# Kernel ini TIDAK menyentuh dataset bonsai-mojo-source maupun kernel produksi.
#
# Tujuan pembandingan: memisahkan "bug di port Mojo" dari "memang begitu
# perilaku model 1-bit". Prompt dan template IDENTIK dengan gerbang koherensi
# di deploy_on_kaggle.sh (cerita + penalaran), greedy 512 token.
#
# Catatan model: dataset Kaggle berformat MLX safetensors sehingga TIDAK bisa
# dibaca llama.cpp. Dipakai GGUF Q1_0 (bobot sama, wadah berbeda).
# ==============================================================================

import os
import subprocess
import sys
import threading

WORK = "/kaggle/working"
GGUF_DIR = os.path.join(WORK, "gguf")
LCPP = os.path.join(WORK, "llama.cpp")
LCPP_BUILD = os.path.join(LCPP, "build")
OUT_DIR = os.path.join(WORK, "banding")
MAX_TOKENS = 512

PROMPTS = [
    ("cerita",
     "Tulis sebuah cerita pendek sekitar 250 kata tentang seorang nelayan tua "
     "yang menemukan botol berisi peta harta karun di tepi pantai."),
    ("penalaran",
     "Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B "
     "dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu pipa B "
     "ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. Berapa jam "
     "total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? Tunjukkan "
     "langkah perhitungannya."),
]

# Kandidat GGUF. Yang pertama adalah yang didokumentasikan di
# NOTES_MODEL_BONSAI_SETTINGS.md; sisanya hanya variasi nama repo yang lazim.
GGUF_REPOS = ["prism-ml/Bonsai-27B-gguf", "prism-ml/Bonsai-27B-GGUF"]


def log(msg):
    print(msg, flush=True)


def run(cmd, cwd=None, timeout=None, check=False):
    log(">> [CMD] " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, timeout=timeout, check=False,
                       stdin=subprocess.DEVNULL,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = p.stdout or ""
    # Batas ini harus cukup besar untuk memuat SELURUH jawaban llama.cpp
    # (512 token ~ 3 KB); kalau kekecilan, teks yang mau dibandingkan malah
    # yang terpotong. Batas tetap ada supaya log build cmake tidak membanjiri.
    if len(tail) > 24000:
        tail = tail[:2000] + "\n...[dipotong]...\n" + tail[-22000:]
    log(tail)
    if check and p.returncode != 0:
        log(">> [FAIL] rc=%s: %s" % (p.returncode, " ".join(cmd)))
        sys.exit(1)
    return p.returncode


def run_stream(cmd, tee_path=None, timeout=None):
    """
    Jalankan sambil mencetak tiap baris keluaran APA ADANYA.

    `run()` menampung keluaran di pipe dan baru mencetaknya setelah proses
    selesai. Kalau sesi Kaggle diakhiri di tengah generasi, isi pipe ikut hilang
    bersama prosesnya — dan itulah yang membuat run-run sebelumnya berakhir tanpa
    satu token pun teks dari llama.cpp. Di sini tiap baris langsung dicetak
    (log() memakai flush=True) sehingga masuk ke log kernel saat itu juga, dan
    sekaligus ditulis ke berkas di /kaggle/working sebagai cadangan.

    `timeout` wajib diisi untuk generasi: tanpa itu, generasi yang tersangkut
    tidak dihentikan oleh apa pun selain batas sesi Kaggle — dan itu sudah
    terbukti memakan satu sesi penuh (6 jam) tanpa menghasilkan apa-apa.
    """
    log(">> [CMD] " + " ".join(cmd))
    tee = open(tee_path, "w") if tee_path else None
    pembunuh = None
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        if timeout:
            def _bunuh():
                log(">> [TIMEOUT] %d detik terlampaui — proses dihentikan; "
                    "teks yang sudah keluar tetap tersimpan" % timeout)
                try:
                    p.kill()
                except Exception:
                    pass
            pembunuh = threading.Timer(timeout, _bunuh)
            pembunuh.start()
        for line in p.stdout:
            log(line.rstrip("\n"))
            if tee:
                tee.write(line)
                tee.flush()
        return p.wait()
    finally:
        if pembunuh:
            pembunuh.cancel()
        if tee:
            tee.close()


def find_model_dir():
    import glob
    for pat in ("/kaggle/input/*/bonsai-27b-mlx-1bit",
                "/kaggle/input/bonsai-27b-mlx-1bit",
                "/kaggle/input/datasets/*/bonsai-27b-mlx-1bit"):
        for c in glob.glob(pat):
            if os.path.exists(os.path.join(c, "tokenizer.json")):
                return c
    return None


def unduh_gguf():
    os.makedirs(GGUF_DIR, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except Exception as e:
        log(">> [FAIL] huggingface_hub tidak tersedia: %s" % e)
        return None

    for repo in GGUF_REPOS:
        try:
            files = list_repo_files(repo)
        except Exception as e:
            log("   [INFO] %s tidak terbaca: %s" % (repo, e))
            continue
        gg = [f for f in files if f.lower().endswith(".gguf")]
        log("   [INFO] %s memuat %d berkas .gguf: %s" % (repo, len(gg), gg[:5]))
        if not gg:
            continue
        # utamakan Q1_0 (1-bit); kalau tidak ada, pakai yang pertama
        pick = sorted(gg, key=lambda f: ("q1_0" not in f.lower(), len(f)))[0]
        log("   [INFO] dipilih: %s" % pick)
        try:
            return hf_hub_download(repo_id=repo, filename=pick, local_dir=GGUF_DIR)
        except Exception as e:
            log("   [WARN] unduh gagal: %s" % e)
    return None


def _cari_libcuda():
    """Kembalikan path libcuda.so (dev) atau libcuda.so.1 yang benar-benar ada."""
    dirs = [
        "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/nvidia/lib64",
        "/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
        "/usr/local/cuda/compat",
    ]
    for d in dirs:
        p = os.path.join(d, "libcuda.so")
        if os.path.exists(p):
            return p, True          # sudah versi dev, tidak perlu symlink
    # ldconfig = sumber paling otoritatif kalau libnya di luar tempat biasa
    kode, out = subprocess.getstatusoutput(
        "ldconfig -p 2>/dev/null | grep 'libcuda\\.so' | head -1")
    if kode == 0 and "=>" in out:
        p = out.split("=>")[-1].strip()
        if os.path.exists(p):
            log("   [INFO] ldconfig menunjuk: %s" % p)
            return p, False
    import glob as _glob
    for d in dirs:
        for h in sorted(_glob.glob(os.path.join(d, "libcuda.so*"))):
            if os.path.isfile(h) and not os.path.islink(h):
                return h, False
    return None, False


def siapkan_libcuda():
    """
    FindCUDAToolkit CMAKE mencari file `libcuda.so` (symlink dev). Image Kaggle
    T4 ternyata tidak menyediakannya di mana pun — hasil ukur: baik
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 maupun .../cuda/lib64/stubs/libcuda.so
    tidak ada. Akibatnya target `CUDA::cuda_driver` tidak terbentuk dan cmake
    gagal di ggml/src/ggml-cuda/CMakeLists.txt:171.

    Di sini libcuda-nya DICARI (bukan ditebak), lalu dibuatkan symlink
    `libcuda.so` di sampingnya dan di direktori lib CUDA agar cmake menemukannya.
    Kalau memang tidak ada sama sekali, paket stub resmi NVIDIA dipasang.
    """
    p, sudah_dev = _cari_libcuda()
    if not p:
        log("   [INFO] libcuda tidak ada — coba pasang stub resmi cuda-driver-dev")
        subprocess.getstatusoutput("apt-get install -y -q cuda-driver-dev-12-8")
        p, sudah_dev = _cari_libcuda()

    if not p:
        log("   [WARN] libcuda tetap tidak ditemukan — mengandalkan GGML_CUDA_NO_VMM")
        return
    log("   [OK] libcuda ditemukan: %s%s" % (p, " (versi dev)" if sudah_dev else ""))
    if sudah_dev:
        return

    tujuan = [os.path.dirname(p)] + [
        "/usr/local/cuda/lib64/stubs",
        "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
        "/usr/local/cuda/lib64",
    ]
    for t in tujuan:
        q = os.path.join(t, "libcuda.so")
        if os.path.exists(q):
            continue
        try:
            os.makedirs(t, exist_ok=True)
            os.symlink(p, q)
            log("   [OK] symlink %s -> %s" % (q, p))
        except Exception as e:
            log("   [INFO] lewati %s: %s" % (q, e))


BIN_CACHE_OUT = os.path.join(WORK, "llama-cuda-bin", "bin")


def cari_bin_cache():
    """Binari dari build sebelumnya, kalau dataset cache-nya sudah terpasang."""
    import glob
    for pat in ("/kaggle/input/llama-cuda-bin/bin/llama-cli",
                "/kaggle/input/*/llama-cuda-bin/bin/llama-cli"):
        for c in glob.glob(pat):
            return c
    return None


def build_llama_cpp():
    # Build memakan ~28 menit dari jatah sesi, dan tiap run Kaggle memulai
    # container dari nol sehingga tanpa cache ongkos itu dibayar berulang kali.
    # Karena itu hasil build disimpan ke /kaggle/working untuk dijadikan dataset
    # `llama-cuda-bin`; kalau dataset itu sudah terpasang, langkah build dilewati.
    cache = cari_bin_cache()
    if cache:
        log(">> [OK] binari cache terpakai — build DILEWATI: %s" % cache)
        os.chmod(cache, 0o755)
        return cache

    if not os.path.isdir(LCPP):
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp", LCPP], timeout=1200, check=True)
    siapkan_libcuda()
    # GGML_CUDA_NO_VMM: T4 (sm_75) tidak punya VMM, jadi mematikannya tidak
    # merugikan. Opsi ini juga membuat ggml-cuda tidak wajib menautkan
    # CUDA::cuda_driver. Kalau versi llama.cpp tidak mengenalnya, CMake hanya
    # mengabaikannya sebagai variable tak terpakai — jadi aman dikirim selalu.
    run(["cmake", "-S", LCPP, "-B", LCPP_BUILD,
         "-DGGML_CUDA=ON", "-DGGML_CUDA_NO_VMM=ON",
         "-DCMAKE_CUDA_ARCHITECTURES=75",
         "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], timeout=1800, check=True)
    run(["cmake", "--build", LCPP_BUILD, "--config", "Release",
         "-j%d" % os.cpu_count(), "--target", "llama-cli"], timeout=3600, check=True)
    cli = os.path.join(LCPP_BUILD, "bin", "llama-cli")
    if not os.path.exists(cli):
        log(">> [FAIL] llama-cli tidak ada di %s" % cli)
        return None

    # seluruh isi bin/ ikut disimpan: llama-cli membutuhkan libllama.so dkk
    # yang duduk di direktori yang sama.
    try:
        os.makedirs(os.path.dirname(BIN_CACHE_OUT), exist_ok=True)
        run(["cp", "-a", os.path.join(LCPP_BUILD, "bin"), BIN_CACHE_OUT], check=True)
        log(">> [OK] hasil build disimpan: %s" % BIN_CACHE_OUT)
        log(">> [INFO] jadikan dataset 'llama-cuda-bin' dari keluaran kernel ini,")
        log(">> [INFO] maka run berikutnya tidak perlu build lagi (~28 menit hemat).")
    except Exception as e:
        log(">> [WARN] gagal menyimpan cache: %s" % e)
    return cli


def render_prompt(model_dir, text, out_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    msgs = [{"role": "user", "content": text}]
    try:
        rendered = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
    except TypeError:
        rendered = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    with open(out_path, "w") as f:
        f.write(rendered)
    ids = tok.encode(rendered, add_special_tokens=False)
    log("   [TOK] %d token prompt (sama dgn jalur Mojo: template + tokenizer model ini)"
        % len(ids))
    return len(ids)


def main():
    log("=" * 70)
    log(">> BANDING llama.cpp CUDA — model sama, prompt sama, greedy %d token" % MAX_TOKENS)
    log("=" * 70)
    run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

    model_dir = find_model_dir()
    if not model_dir:
        log(">> [FAIL] dataset model tidak terpasang — tokenizer tidak bisa dirender")
        sys.exit(1)
    log(">> [OK] dataset model: %s" % model_dir)

    log("")
    log(">> [1/4] unduh GGUF Q1_0")
    gguf = unduh_gguf()
    if not gguf or not os.path.exists(gguf):
        log(">> [FAIL] tidak ada GGUF yang berhasil diunduh")
        sys.exit(1)
    log(">> [OK] GGUF: %s (%.2f GiB)" % (gguf, os.path.getsize(gguf) / 2**30))

    log("")
    log(">> [2/4] build llama.cpp CUDA (sm_75)")
    cli = build_llama_cpp()
    if not cli:
        sys.exit(1)
    log(">> [OK] %s" % cli)

    log("")
    log(">> [3/4] render prompt (template & tokenizer model itu sendiri)")
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = []
    for name, text in PROMPTS:
        p = os.path.join(OUT_DIR, name + ".prompt.txt")
        n = render_prompt(model_dir, text, p)
        jobs.append((name, p, n))

    log("")
    log(">> [4/4] jalankan llama-cli (greedy, -ngl 99)")
    for name, p, n in jobs:
        log("")
        log("=" * 70)
        log(">> llama.cpp: %s | prompt %d token | greedy %d token" % (name, n, MAX_TOKENS))
        log("=" * 70)
        # LD_LIBRARY_PATH: libllama.so dkk ada di direktori yang sama dengan
        # llama-cli — baik saat masih di build/ maupun saat dari dataset cache.
        rc = run_stream(["env", "LD_LIBRARY_PATH=" + os.path.dirname(cli),
                         cli, "-m", gguf, "-f", p, "-c", "4096",
                         "-n", str(MAX_TOKENS), "-ngl", "99", "--temp", "0"],
                        tee_path=os.path.join(OUT_DIR, "%s.llama.txt" % name),
                        timeout=1800)
        log(">> [rc=%d] selesai: %s" % (rc, name))

    log("")
    log("=" * 70)
    log(">> SELESAI. Bandingkan teks di atas dengan hasil port Mojo")
    log(">> (gerbang koherensi cerita.think0 / penalaran.think0 di log deploy).")
    log("=" * 70)


if __name__ == "__main__":
    main()
