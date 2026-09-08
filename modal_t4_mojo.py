#!/usr/bin/env python3
"""
Uji INFERENSI native Mojo di Modal.com (T4) memakai wheel hasil build Kaggle.

Alur (wheel = sumber kebenaran, tanpa kompilasi library):
  1. Wheel `dist_kaggle_mojo/bonsai_1bit_t4-*.whl` di-mount + di-pip-install.
  2. Bobot model prism-ml/Bonsai-27B-mlx-1bit diunduh ke Modal Volume (cache).
  3. Entry CLI `main.mojo` dikompilasi sekali di container (satu file, detik;
     library tetap dari wheel).
  4. Inferensi greedy: embed -> 64 layer hybrid -> norm -> argmax, di T4.

Pemakaian (dari root repo):
  pip install modal && modal setup        # sekali
  export HF_TOKEN=hf_xxx                  # bila model gated
  python modal_t4_mojo.py probe           # cek GPU + toolchain + model
  python modal_t4_mojo.py infer           # unduh model + jalankan inferensi
  python modal_t4_mojo.py all             # probe -> infer

Env: BONSAI_MODEL (default prism-ml/Bonsai-27B-mlx-1bit),
     PROMPT_TOKENS (default "1,2,3,4,5"), MAX_NEW_TOKENS (default 8),
     MOJO_WHEEL (override lokasi wheel).
"""

import os
import sys
from contextlib import contextmanager

import modal

APP_NAME = "bonsai-mojo-infer"
REPO = os.path.dirname(os.path.abspath(__file__))
GPU = os.environ.get("MOJO_T4_GPU", "T4")
PY_VER = os.environ.get("MOJO_PY", "3.12")
MODEL_ID = os.environ.get("BONSAI_MODEL", "prism-ml/Bonsai-27B-mlx-1bit")
PROMPT = os.environ.get("PROMPT_TOKENS", "1,2,3")
NEW_TOKENS = os.environ.get("MAX_NEW_TOKENS", "3")

# ── Sumber wheel ────────────────────────────────────────────────────────────
WHEEL_GLOBS = [
    os.environ.get("MOJO_WHEEL", ""),
    os.path.join(REPO, "..", "dist_kaggle_mojo", "bonsai_1bit_t4-*.whl"),
    os.path.join(REPO, "dist_kaggle_mojo", "bonsai_1bit_t4-*.whl"),
]


def _find_wheel():
    import glob

    for g in WHEEL_GLOBS:
        hits = sorted(glob.glob(g), key=lambda p: os.path.getmtime(p)) if g else []
        if hits:
            return hits[-1]
    return None


# ── Image: CUDA base + pixi + huggingface_hub (sekali build, cache Modal) ──
_base = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python=PY_VER
).apt_install(["git", "curl", "ca-certificates", "bash"]).run_commands(
    "curl -fsSL https://pixi.sh/install.sh | bash -s -- --yes",
    "ln -sf /root/.pixi/bin/pixi /usr/local/bin/pixi",
).pip_install("huggingface_hub", "requests")

# Pelajaran dari modal_a10_audit.py: Modal me-import ulang file ini di dalam
# container — kode level modul yang menyentuh filesystem LOKAL membuat import
# crash + retry. Image dibuat MALAS di sisi container.
_IS_CONTAINER_IMPORT = __name__ != "__main__"


def _make_image():
    w = _find_wheel()
    if not w:
        raise SystemExit(
            "[ERROR] wheel tidak ditemukan. Jalankan ./push_to_kaggle.sh dulu "
            "atau set MOJO_WHEEL=/path/bonsai_1bit_t4-*.whl"
        )
    wn = os.path.basename(w)
    print(f">> Wheel: {w}", flush=True)
    # Binary bonsai_infer (dikompilasi di Kaggle) ikut di dalam wheel —
    # Modal TIDAK mengompilasi apa pun, hanya menjalankan.
    return _base.add_local_file(w, f"/wheels/{wn}")


_built = _base if _IS_CONTAINER_IMPORT else _make_image()

# ── Storage & secrets ───────────────────────────────────────────────────────
_cache_vol = modal.Volume.from_name("bonsai-pixi-cache", create_if_missing=True)
_model_vol = modal.Volume.from_name("bonsai-model-27b", create_if_missing=True)


def _hf_secrets():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return [modal.Secret.from_dict({"HF_TOKEN": tok})]
    return []


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
    timeout=60 * 90,
)


@app.function(gpu=GPU, cpu=8, memory=32768, secrets=_HF_SECRETS, **_COMMON)
def run_t4(cmd: str) -> str:
    return _shell(cmd)


_WHEEL_READY = False


def _prepare_container() -> str:
    """Ekstrak wheel (paket transport sumber) + siapkan path PKG."""
    global _WHEEL_READY
    if _WHEEL_READY:
        return ""
    import glob
    import subprocess as sp

    found = sorted(glob.glob("/wheels/bonsai_1bit_t4-*.whl"))
    if not found:
        return "wheel tidak ada di /wheels"
    # pip menuntut metadata RECORD penuh — wheel ini murni paket transport
    # sumber (.mojo), jadi cukup diekstrak langsung tanpa pip.
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
    print(f"$ {cmd}", flush=True)
    p = sp.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print("---- HEAD ----", flush=True)
    print(out[:6000], flush=True)
    print("---- TAIL ----", flush=True)
    print(out[-8000:], flush=True)
    print(f"[exit={p.returncode}]", flush=True)
    sys.stdout.flush()
    return f"exit={p.returncode}\n{out}"


# ── Perintah remote ─────────────────────────────────────────────────────────
# 1. Wheel terpasang di site-packages/bonsai_1bit_t4 (src + tests + pixi.toml)
# 2. Toolchain MAX 25.x dari manifest wheel (cache Volume)
# 3. Model HF di /volumes/model (cache Volume, unduh sekali)
_PKG_TOOLCHAIN = (
    # retry(): crash runtime KGEN bersifat probabilistik (ASLR entropy kernel
    # baru mematahkan init libAsyncRTRuntimeGlobals) — ulang sampai sukses.
    "retry() { local i; for i in $(seq 1 12); do \"$@\" && return 0; "
    "echo \"[RETRY $i] $1\"; sleep 1; done; echo '[GAGAL 12x]'; return 1; }; "
    "set +e; "
    "PKG=/tmp/wheelroot/bonsai_1bit_t4; "
    "echo \"PKG=$PKG\"; [ -d \"$PKG/src\" ] || { echo 'src tidak ada di wheel'; exit 1; }; "
    "cd \"$PKG\" && "
    "export PATH=$PIXI_HOME/bin:/root/.pixi/bin:$PATH && "
    # Cache runtime Mojo di /tmp segar per run: mengisolasi dari state Volume
    # yang mungkin korup lintas run gagal (abort libAsyncRTRuntimeGlobals).
    "export MODULAR_HOME=/tmp/modular_home MOJO_CACHE_DIR=/tmp/mojo_cache && "
    "pixi install 2>&1 | tail -5 && "
    "echo '--- mojo --version ---' && "
    "retry pixi run mojo --version"
)

_PROBE = (
    "nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version "
    "--format=csv,noheader; echo '--- toolchain dari wheel ---'; " + _PKG_TOOLCHAIN
)

_INFER = (
    "set +e; " + _PKG_TOOLCHAIN + " && "
    "echo '--- unduh model (cache Volume) ---' && "
    "python3 -c \""
    "import os; from huggingface_hub import snapshot_download; "
    "p = snapshot_download(os.environ['BONSAI_MODEL'], local_dir='/volumes/model'); "
    "print('MODEL_DIR=', p)\" && "
    "ls /tmp/model_local | wc -l && ls -lh /tmp/model_local | head -14 && "
    "echo '--- salin model ke disk lokal (io.file butuh FS lokal) ---' && "
    "mkdir -p /tmp/model_local && cp -rL /volumes/model/. /tmp/model_local/ && du -sh /tmp/model_local && "
    "echo '--- binary dari wheel (dikompilasi di Kaggle; Modal tidak compile) ---' && "
    "[ -f $PKG/bin/bonsai_infer ] || { echo '[ERROR] binary bonsai_infer tidak ada di wheel — jalankan build Kaggle terbaru'; exit 1; }; "
    "chmod +x $PKG/bin/bonsai_infer && cp $PKG/bin/bonsai_infer /tmp/bonsai_infer && "
    "export LD_LIBRARY_PATH=$PKG/.pixi/envs/default/lib:$LD_LIBRARY_PATH && "
    "echo '--- INFERENSI GREEDY di T4 ---' && "
    "retry /tmp/bonsai_infer --model-dir /tmp/model_local "
    "--prompt-tokens " + PROMPT + " --max-tokens " + NEW_TOKENS
)


# ── Orkestrasi lokal ────────────────────────────────────────────────────────
def _show(title: str, res: str, keep: int = 9000) -> None:
    print("\n" + "=" * 92)
    print(f" {title}")
    print("=" * 92)
    print(res[-keep:])


@contextmanager
def _session():
    with modal.enable_output():
        with app.run():
            yield


def _run_in_session() -> None:
    print(">> [1/2] PROBE", flush=True)
    _show(f"PROBE @{GPU}", run_t4.remote(_PROBE))
    print(">> [2/2] INFERENSI", flush=True)
    if not _HF_SECRETS:
        print("[INFO] HF_TOKEN tidak diset — bila model gated, export HF_TOKEN dulu.")
    _show(f"INFERENSI @{GPU}", run_t4.remote(_INFER), 12000)


def probe() -> None:
    with _session():
        _show(f"PROBE @{GPU}", run_t4.remote(_PROBE))


def infer() -> None:
    with _session():
        if not _HF_SECRETS:
            print("[INFO] HF_TOKEN tidak diset — bila model gated, export HF_TOKEN dulu.")
        _show(f"INFERENSI @{GPU}", run_t4.remote(_INFER), 12000)


def run_all() -> None:
    # SATU sesi Modal untuk seluruh rangkaian (app.run tidak boleh nested)
    with _session():
        _run_in_session()


_STEPS = {"probe": probe, "infer": infer, "all": run_all}


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in _STEPS:
        _STEPS[step]()
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
