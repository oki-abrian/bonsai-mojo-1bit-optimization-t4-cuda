#!/usr/bin/env python3
"""
Uji INFERENSI native Mojo di Modal.com (T4).

ALUR YANG DISARANKAN (hemat GPU — build di CPU, jalan di GPU)
------------------------------------------------------------
 1) Build artefak di Kaggle CPU-ONLY (tanpa GPU, tanpa kuota GPU):
        ./build_cpu/push_build_cpu.sh
    -> menghasilkan artifacts_modal/{bonsai_infer, libbonsai_qmv_sm75.so,
                                    libKGENCompilerRTShared.so,
                                    libAsyncRTMojoBindings.so, MANIFEST.txt}
 2) Jalankan inferensi di Modal T4 (TIDAK build apa pun):
        python modal_t4_mojo.py kaggle

Bila artifacts_modal/ belum ada, skrip otomatis jatuh ke jalur lama: image
CUDA *devel* + pixi, dan build di dalam container GPU (lambat & memakan GPU).

Langkah lain:
  probe   : cek GPU + (nvcc / artefak) — jalur mana yang aktif.
  infer   : binary SIAP PAKAI dari wheel dengan token dummy (uji cepat FFI).
  kaggle  : replika pipeline Kaggle: encode prompt chat template -> inferensi
            2048 token dengan resep sampling resmi.
  all     : probe -> kaggle

Env:
  BONSAI_MODEL   default prism-ml/Bonsai-27B-mlx-1bit   (publik, tidak gated)
  MOJO_WHEEL     override lokasi wheel
  MOJO_BUILD     1 (default) = build di container; 0 = pakai binary wheel
                 (hanya berlaku bila artifacts_modal/ belum ada)
  MOJO_T4_GPU    default T4
  MAX_NEW_TOKENS default 2048 (mode kaggle) / 3 (mode infer)
  BONSAI_TEMP_X100, BONSAI_TOP_K, BONSAI_TOP_P_X1000, BONSAI_MIN_P_X1000,
  BONSAI_REP_PENALTY_X100, BONSAI_REP_WINDOW, BONSAI_SEED  (parameter sampling)

Catatan: HF_TOKEN tidak diperlukan — prism-ml/Bonsai-27B-mlx-1bit publik
(`gated: false`). Export HF_TOKEN hanya bila model di-gate di masa depan.
"""

import base64
import glob
import json
import os
import sys
import zipfile
from contextlib import contextmanager

import modal

APP_NAME = "bonsai-mojo-infer"
REPO = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("MOJO_T4_GPU", "T4")
PY_VER = os.environ.get("MOJO_PY", "3.12")
MODEL_ID = os.environ.get("BONSAI_MODEL", "prism-ml/Bonsai-27B-mlx-1bit")

# ── Artefak hasil build Kaggle CPU-only ─────────────────────────────────────
ARTIFACTS_DIR = os.path.join(REPO, "artifacts_modal")
ARTIFACT_FILES = [
    "bonsai_infer",                 # hasil `mojo build main.mojo`
    "libbonsai_qmv_sm75.so",        # kernel CUDA sm_75 hasil nvcc
    "libKGENCompilerRTShared.so",   # runtime Mojo (DT_NEEDED bonsai_infer)
    "libAsyncRTMojoBindings.so",    # runtime Mojo (DT_NEEDED bonsai_infer)
]


def _artifacts_missing() -> list:
    return [n for n in ARTIFACT_FILES
            if not os.path.isfile(os.path.join(ARTIFACTS_DIR, n))]


_MISSING = _artifacts_missing()
USE_ARTIFACTS = not _MISSING

# ── Parameter tes "kaggle" — HARUS sama dengan infer_susah/infer_config.json ──
HARD_PROMPT = (
    "Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B "
    "dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu "
    "pipa B ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. "
    "Berapa jam total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? "
    "Tunjukkan langkah perhitungannya."
)
MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
SAMPLING = {
    "BONSAI_TEMP_X100": os.environ.get("BONSAI_TEMP_X100", "70"),
    "BONSAI_TOP_K": os.environ.get("BONSAI_TOP_K", "20"),
    "BONSAI_TOP_P_X1000": os.environ.get("BONSAI_TOP_P_X1000", "950"),
    "BONSAI_MIN_P_X1000": os.environ.get("BONSAI_MIN_P_X1000", "0"),
    "BONSAI_REP_PENALTY_X100": os.environ.get("BONSAI_REP_PENALTY_X100", "100"),
    "BONSAI_REP_WINDOW": os.environ.get("BONSAI_REP_WINDOW", "256"),
    "BONSAI_SEED": os.environ.get("BONSAI_SEED", "1234"),
}
BUILD_FROM_SOURCE = os.environ.get("MOJO_BUILD", "1") not in ("0", "false", "no")
PROMPT = os.environ.get("BONSAI_PROMPT", HARD_PROMPT)

# ── Sumber wheel (hanya dipakai bila artifacts_modal/ belum ada) ────────────
WHEEL_GLOBS = [
    os.environ.get("MOJO_WHEEL", ""),
    os.path.join(REPO, "..", "dist_kaggle_mojo", "bonsai_1bit_t4-*.whl"),
    os.path.join(REPO, "dist_kaggle_mojo", "bonsai_1bit_t4-*.whl"),
]


def _wheel_is_valid(path: str) -> bool:
    """Wheel ini paket transport sumber — wajib berisi bonsai_1bit_t4/src.

    Guard penting: `dist_kaggle_mojo/` di dalam repo pernah tertimpa wheel pip
    (berisi `pip/`, bukan sumber proyek). Tanpa cek ini, `mojo build` gagal
    dengan pesan membingungkan setelah image dibangun.
    """
    try:
        with zipfile.ZipFile(path) as z:
            return any(n.startswith("bonsai_1bit_t4/src/") for n in z.namelist())
    except Exception:
        return False


def _find_wheel():
    for g in WHEEL_GLOBS:
        if not g:
            continue
        hits = sorted(glob.glob(g), key=lambda p: os.path.getmtime(p))
        good = [h for h in hits if _wheel_is_valid(h)]
        if good:
            return good[-1]
        if hits:
            print(f"[WARN] {hits[-1]} bukan wheel sumber proyek — dilewati", flush=True)
    return None


# ── Image ───────────────────────────────────────────────────────────────────
def _pip_tail(img):
    return img.pip_install("huggingface_hub", "requests", "transformers", "jinja2")


if USE_ARTIFACTS:
    # Jalur CEPAT: CUDA *runtime* (bukan devel) — tanpa nvcc, tanpa pixi.
    # Artefak sudah dibangun di Kaggle CPU-only dan ditanam ke image.
    _img = _pip_tail(
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python=PY_VER
        ).apt_install(["curl", "ca-certificates", "bash"])
    )
    for _n in ARTIFACT_FILES:
        _img = _img.add_local_file(
            os.path.join(ARTIFACTS_DIR, _n), f"/artifacts/{_n}"
        )
    _built = _img
else:
    # Jalur LAMA: CUDA devel + pixi, build di dalam container (lambat).
    _base = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python=PY_VER
        )
        .apt_install(["git", "curl", "ca-certificates", "bash"])
        .run_commands(
            "curl -fsSL https://pixi.sh/install.sh | bash -s -- --yes",
            "ln -sf /root/.pixi/bin/pixi /usr/local/bin/pixi",
        )
    )
    _built = _pip_tail(_base)

    # Modal me-import ulang file ini di dalam container — kode level modul yang
    # menyentuh filesystem LOKAL membuat import crash + retry.
    _IS_CONTAINER_IMPORT = __name__ != "__main__"

    def _make_image():
        w = _find_wheel()
        if not w:
            raise SystemExit(
                "[ERROR] artifacts_modal/ kosong DAN wheel sumber tidak ditemukan.\n"
                "  Bangun artefak dulu:  ./build_cpu/push_build_cpu.sh\n"
                "  Dicari di: " + "\n            ".join(g for g in WHEEL_GLOBS if g)
            )
        print(f">> Wheel: {w}  ({os.path.getsize(w)/1e6:.2f} MB)", flush=True)
        return _built.add_local_file(w, f"/wheels/{os.path.basename(w)}")

    if not _IS_CONTAINER_IMPORT:
        _built = _make_image()

# ── Storage & secrets ───────────────────────────────────────────────────────
_cache_vol = modal.Volume.from_name("bonsai-pixi-cache", create_if_missing=True)
_model_vol = modal.Volume.from_name("bonsai-model-27b", create_if_missing=True)


def _hf_secrets():
    tok = os.environ.get("HF_TOKEN")
    return [modal.Secret.from_dict({"HF_TOKEN": tok})] if tok else []


_HF_SECRETS = _hf_secrets()

app = modal.App(APP_NAME)

_COMMON = dict(
    image=_built,
    volumes={"/root/.cache": _cache_vol, "/volumes/model": _model_vol},
    env={
        "PIXI_HOME": "/root/.cache/pixi_home",
        "PIXI_CACHE_DIR": "/root/.cache/pixi_cache",
        "MODULAR_HOME": "/root/.cache/modular",
        "MOJO_CACHE_DIR": "/root/.cache/mojo",
        "BONSAI_MODEL": MODEL_ID,
        "PYTHONUNBUFFERED": "1",
    },
    # CATATAN: `ephemeral_disk` di Modal minimal 524288 MiB (512 GiB) — default
    # 512 GiB sudah cukup, jadi parameternya TIDAK diisi.
    timeout=60 * 90,
)


@app.function(gpu=GPU, cpu=8, memory=32768, secrets=_HF_SECRETS, **_COMMON)
def run_t4(cmd: str) -> str:
    return _shell(cmd)


_WHEEL_READY = False


def _prepare_container() -> str:
    """Jalur build: ekstrak wheel (paket transport sumber). No-op bila artefak."""
    if USE_ARTIFACTS:
        return ""
    global _WHEEL_READY
    if _WHEEL_READY:
        return ""
    import subprocess as sp

    found = sorted(glob.glob("/wheels/bonsai_1bit_t4-*.whl"))
    if not found:
        return "wheel tidak ada di /wheels"
    r = sp.run(
        ["python3", "-m", "zipfile", "-e", found[-1], "/tmp/wheelroot"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return f"WHEEL EXTRACT exit={r.returncode}\n{(r.stdout or '') + (r.stderr or '')}"
    _WHEEL_READY = True
    return ""


def _shell(cmd: str) -> str:
    import subprocess as sp

    err = _prepare_container()
    if err:
        print(err, flush=True)
        return err
    print(f"$ {cmd[:400]}...", flush=True)
    p = sp.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print("---- HEAD ----", flush=True)
    print(out[:6000], flush=True)
    print("---- TAIL ----", flush=True)
    print(out[-14000:], flush=True)
    print(f"[exit={p.returncode}]", flush=True)
    sys.stdout.flush()
    return f"exit={p.returncode}\n{out}"


# ── Perintah remote ─────────────────────────────────────────────────────────
_RETRY_FN = (
    # retry(): crash runtime KGEN bersifat probabilistik (ASLR entropy kernel
    # baru mematahkan init libAsyncRTRuntimeGlobals) — ulang sampai sukses.
    "retry() { local i; for i in $(seq 1 12); do \"$@\" && return 0; "
    "echo \"[RETRY $i] $1\"; sleep 1; done; echo '[GAGAL 12x]'; return 1; }; "
)

_PKG_TOOLCHAIN = (
    _RETRY_FN +
    "set +e; "
    "PKG=/tmp/wheelroot/bonsai_1bit_t4; "
    "echo \"PKG=$PKG\"; [ -d \"$PKG/src\" ] || { echo 'src tidak ada di wheel'; exit 1; }; "
    "cd \"$PKG\" && "
    "export PATH=/usr/local/cuda/bin:$PIXI_HOME/bin:/root/.pixi/bin:$PATH && "
    "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-} && "
    "export MODULAR_HOME=/tmp/modular_home MOJO_CACHE_DIR=/tmp/mojo_cache && "
    "pixi install 2>&1 | tail -5 && "
    "echo '--- mojo --version ---' && "
    "retry pixi run mojo --version"
)

_DOWNLOAD_WEIGHTS = (
    "echo '--- bobot model (cache Volume) ---' && "
    "python3 -c \""
    "import os; from huggingface_hub import snapshot_download; "
    "p = snapshot_download(os.environ['BONSAI_MODEL'], local_dir='/volumes/model'); "
    "print('MODEL_DIR=', p)\" && "
    "mkdir -p /tmp/model_local && cp -rL /volumes/model/. /tmp/model_local/ && "
    "du -sh /tmp/model_local && "
)

if USE_ARTIFACTS:
    _PROBE = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version "
        "--format=csv,noheader; "
        "echo '--- artefak yang ditanam ke image ---'; ls -lh /artifacts; "
        "echo '--- ldd bonsai_infer ---'; "
        "export LD_LIBRARY_PATH=/artifacts:/usr/local/cuda/lib64:$LD_LIBRARY_PATH; "
        "ldd /artifacts/bonsai_infer"
    )
else:
    _PROBE = (
        "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version "
        "--format=csv,noheader; echo '--- toolchain dari wheel ---'; " + _PKG_TOOLCHAIN
    )

_INFER = (
    "set +e; " + _PKG_TOOLCHAIN + " && " + _DOWNLOAD_WEIGHTS +
    "echo '--- binary dari wheel (dikonpilasi di Kaggle) ---' && "
    "[ -f $PKG/bin/bonsai_infer ] || { echo '[ERROR] binary tidak ada di wheel'; exit 1; }; "
    "chmod +x $PKG/bin/bonsai_infer && cp $PKG/bin/bonsai_infer /tmp/bonsai_infer && "
    "export BONSAI_CUDA_LIB=$PKG/libbonsai_qmv_sm75.so && "
    "export LD_LIBRARY_PATH=$PKG:$PKG/.pixi/envs/default/lib:$LD_LIBRARY_PATH && "
    "echo '--- INFERENSI GREEDY di T4 ---' && "
    "retry /tmp/bonsai_infer --model-dir /tmp/model_local "
    "--prompt-tokens 1,2,3 --max-tokens " + os.environ.get("MAX_NEW_TOKENS", "3")
)


# ── Encode prompt (chat template) ───────────────────────────────────────────
_ENCODE_PY = '''\
import os
import transformers
from transformers import AutoTokenizer

MDIR = "/tmp/model_local"
prompt = open("/tmp/prompt.txt", encoding="utf-8").read()
print(">> [ENC] transformers", transformers.__version__, flush=True)

tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
msgs = [{"role": "user", "content": prompt}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids = tok.encode(text, add_special_tokens=False)
open("/tmp/ptokens.txt", "w").write(",".join(str(i) for i in ids))
print(">> [ENC] n_token =", len(ids), flush=True)
print(">> [ENC] templated (400 char):", repr(text[:400]), flush=True)
'''


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _run_inference_tail(bin_dir: str, lib_dir: str) -> str:
    """Bagian akhir yang sama untuk kedua jalur: encode + jalankan binary."""
    env_exports = " ".join(f"{k}={v}" for k, v in SAMPLING.items())
    return (
        "echo '--- encode prompt (chat template) ---' && "
        "python3 /tmp/encode.py && "
        "P_TOKENS=$(cat /tmp/ptokens.txt) && "
        "echo \"tokens: $(printf '%s' \"$P_TOKENS\" | tr ',' '\\n' | grep -c . || true) token\" && "
        "nvidia-smi --query-gpu=name,clocks.sm,power.draw --format=csv,noheader && "
        f"export BONSAI_CUDA_LIB={lib_dir}/libbonsai_qmv_sm75.so && "
        f"export LD_LIBRARY_PATH={bin_dir}:{lib_dir}:/usr/local/cuda/lib64:$LD_LIBRARY_PATH && "
        "export BONSAI_USE_GPU=1 BONSAI_DUMP_TOP2=1 "
        "MOJO_ENABLE_STACK_TRACE_ON_ERROR=1 && "
        f"echo 'sampling: {env_exports}' && "
        "echo '=========================================================' && "
        f"echo ' MULAI GENERASI — max_tokens={MAX_TOKENS}' && "
        "echo '=========================================================' && "
        f"retry env {env_exports} {bin_dir}/bonsai_infer --model-dir /tmp/model_local "
        f"--prompt-tokens \"$P_TOKENS\" --max-tokens {MAX_TOKENS} --gpu"
    )


def _payload_prefix() -> str:
    return (
        _RETRY_FN + "set +e; "
        # Payload base64: prompt multi-baris & tanda kutip jadi aman.
        f"echo '{_b64(PROMPT)}' | base64 -d > /tmp/prompt.txt && "
        f"echo '{_b64(_ENCODE_PY)}' | base64 -d > /tmp/encode.py && "
        "echo \"prompt: $(wc -c < /tmp/prompt.txt) byte\" && "
    )


def _kaggle_cmd_artifacts() -> str:
    """Jalur CEPAT: tidak ada build. GPU hanya memuat bobot + inferensi."""
    return (
        _payload_prefix() +
        "echo '--- [1/3] artefak (sudah dibangun di Kaggle CPU-only) ---' && "
        "ls -lh /artifacts && "
        "mkdir -p /tmp/run && cp /artifacts/* /tmp/run/ && "
        "chmod +x /tmp/run/bonsai_infer && "
        "echo '--- [2/3] bobot model ---' && " +
        _DOWNLOAD_WEIGHTS +
        "echo '--- [3/3] encode + INFERENSI ---' && " +
        _run_inference_tail("/tmp/run", "/tmp/run")
    )


def _kaggle_cmd_build() -> str:
    """Jalur LAMA: build di dalam container GPU (nvcc + mojo build)."""
    if BUILD_FROM_SOURCE:
        build = (
            "echo '--- [1/4] nvcc sm_75 -> libbonsai_qmv_sm75.so ---' && "
            "mkdir -p build && "
            "nvcc -O3 -arch=sm_75 --shared -Xcompiler -fPIC "
            "src/csrc/qmv_sm75_kernel.cu -o build/libbonsai_qmv_sm75.so && "
            "ls -lh build/libbonsai_qmv_sm75.so && "
            "echo '--- [2/4] mojo build main.mojo -> bonsai_infer ---' && "
            "retry pixi run mojo build -I . main.mojo -o /tmp/bonsai_infer && "
            "ls -lh /tmp/bonsai_infer && "
        )
    else:
        build = (
            "echo '--- [1/4] SKIP build (MOJO_BUILD=0) — pakai binary wheel ---' && "
            "mkdir -p build && cp $PKG/libbonsai_qmv_sm75.so build/ && "
            "cp $PKG/bin/bonsai_infer /tmp/bonsai_infer && chmod +x /tmp/bonsai_infer && "
        )
    return (
        _payload_prefix() + _PKG_TOOLCHAIN + " && " + build +
        "echo '--- [3/4] bobot model ---' && " + _DOWNLOAD_WEIGHTS +
        "echo '--- [4/4] encode + INFERENSI ---' && " +
        _run_inference_tail("/tmp", "$PKG/build")
    )


def _kaggle_cmd() -> str:
    return _kaggle_cmd_artifacts() if USE_ARTIFACTS else _kaggle_cmd_build()


# ── Orkestrasi lokal ────────────────────────────────────────────────────────
def _show(title: str, res: str, keep: int = 20000) -> None:
    print("\n" + "=" * 92)
    print(f" {title}")
    print("=" * 92)
    print(res[-keep:])


@contextmanager
def _session():
    with modal.enable_output():
        with app.run():
            yield


def probe() -> None:
    with _session():
        _show(f"PROBE @{GPU}", run_t4.remote(_PROBE))


def infer() -> None:
    if USE_ARTIFACTS:
        print("[INFO] artifacts_modal/ terisi — mode `infer` (jalur wheel) dilewati;"
              " pakai `kaggle`.")
        return
    with _session():
        _show(f"INFERENSI (binary wheel) @{GPU}", run_t4.remote(_INFER))


def kaggle() -> None:
    if USE_ARTIFACTS:
        print(">> Jalur  : ARTEFAK (build di Kaggle CPU-only) — container GPU TIDAK build")
        for n in ARTIFACT_FILES:
            p = os.path.join(ARTIFACTS_DIR, n)
            print(f"   {n:32s} {os.path.getsize(p)/1e6:8.2f} MB")
    else:
        print(f">> Jalur  : BUILD DI CONTAINER GPU (artifacts_modal/ belum ada)")
        print(f"   hilang: {_MISSING}")
        print(f"   build dari sumber = {BUILD_FROM_SOURCE}")
        print("   Hemat GPU: jalankan ./build_cpu/push_build_cpu.sh lebih dulu.")
    print(f">> Prompt  : {len(PROMPT)} char | max_tokens = {MAX_TOKENS}")
    print(f">> Sampling: {json.dumps(SAMPLING)}")
    with _session():
        _show(f"TES KAGGLE-EQUIVALENT @{GPU}", run_t4.remote(_kaggle_cmd()), 24000)


def run_all() -> None:
    # SATU sesi Modal untuk seluruh rangkaian (app.run tidak boleh nested)
    with _session():
        _show(f"PROBE @{GPU}", run_t4.remote(_PROBE))
        _show(f"TES KAGGLE-EQUIVALENT @{GPU}", run_t4.remote(_kaggle_cmd()), 24000)


_STEPS = {"probe": probe, "infer": infer, "kaggle": kaggle, "all": run_all}


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in _STEPS:
        _STEPS[step]()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
