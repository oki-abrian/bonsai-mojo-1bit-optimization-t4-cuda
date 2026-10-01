"""
teacher_gguf.py — Jembatan akses guru: GGUF -> bobot per-layer terdekuantisasi.

Guru yang dikunci (keputusan user 2026-09-26): **Qwen3.8-27B GSQ-RCO GGUF
IQ3_S (11,8 GB, 3,5 bpw)** — bukan BF16. Alasan user: hasilnya setara FP16
pada skor tugas (AIME25 100,00; LiveCodeBench v6 85,71).

Peran modul ini dalam rantai perbaikan akurasi Bonsai-2:
  GGUF IQ3_S --(gguf-py, references/llama.cpp-prism)--> NumPy per tensor
  --(per layer)--> .npz --(mx.array)--> MLX untuk kalibrasi/fitting.

Prinsip desain:
  * Memori ringan: mendekuantisasi SATU tensor per langkah, tanpa pernah
    memuat seluruh model ke RAM (dekuantisasi gguf-py murni NumPy, lambat
    tetapi cukup dijalankan sekali lalu di-cache ke .npz).
  * Cakupan tipe: setiap tensor dicek terhadap registry dekuantisasi gguf-py;
    tipe yang tidak didukung dilaporkan, bukan diam-diam dilewati.
  * Penamaan layer mengikuti konvensi GGUF llama.cpp (`blk.{i}.*`), dengan
    pola HF (`layers.{i}.*`) sebagai cadangan; sisanya = "global"
    (token_embd/output_norm/output).

Pemakaian CLI:
  python3 teacher_gguf.py inspect   MODEL.gguf
  python3 teacher_gguf.py layers    MODEL.gguf
  python3 teacher_gguf.py extract   MODEL.gguf --layer 3 --out DIR [--dtype f16]
  python3 teacher_gguf.py extract-all MODEL.gguf --out DIR [--dtype f16]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Any, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Pemuat gguf-py (dari references/llama.cpp-prism; bukan pip)
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_GGUF_PY_CANDIDATES = [
    os.environ.get("GGUF_PY_PATH", ""),
    os.path.join(_HERE, "..", "llama.cpp-prism", "gguf-py"),
    os.path.join(_HERE, "..", "..", "llama.cpp-prism", "gguf-py"),
]


def load_gguf():
    """Mengimpor paket `gguf` dari pohon llama.cpp-prism. Mengembalikan modul."""
    import importlib

    tried = []
    for cand in _GGUF_PY_CANDIDATES:
        if not cand:
            continue
        cand = os.path.abspath(cand)
        tried.append(cand)
        if os.path.isdir(os.path.join(cand, "gguf")) and cand not in sys.path:
            sys.path.insert(0, cand)
            break
    try:
        mod = importlib.import_module("gguf")
    except ImportError as e:
        raise ImportError(
            "paket `gguf` tidak ditemukan. Cari di: " + ", ".join(tried)
        ) from e
    if not hasattr(mod, "GGUFReader") or not hasattr(mod, "GGUFWriter"):
        raise ImportError("modul `gguf` yang terpasang bukan gguf-py lengkap")
    return mod


def is_supported(qtype: Any, gguf_mod=None) -> bool:
    """Apakah qtype punya jalur dekuantisasi di gguf-py? (F32/F16 selalu ada.)"""
    g = gguf_mod or load_gguf()
    from gguf import quants  # noqa: PLC0415 — setelah sys.path disiapkan

    if qtype in (g.GGMLQuantizationType.F32, g.GGMLQuantizationType.F16):
        return True
    return qtype in quants._type_traits  # registry kelas kuantisasi


# ---------------------------------------------------------------------------
# Inventaris & pengelompokan layer
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"(?:^|\.)(?:blk|layers)\.(\d+)\.")


def layer_of(name: str) -> int | None:
    """Indeks layer dari nama tensor GGUF; None = global (embed/norm/head)."""
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def dequantize_tensor(t: Any, gguf_mod=None) -> np.ndarray:
    """ReaderTensor -> array float32 berbentuk logis [numpy order]."""
    g = gguf_mod or load_gguf()
    return g.quants.dequantize(t.data, t.tensor_type)


def inspect(path: str) -> dict[str, Any]:
    """
    Inventaris file GGUF TANPA mendekuantisasi apa pun:
      {tensors: [...], qtypes: {nama_tipe: {count, bytes, supported}},
       unsupported: [nama tensor], layers: {indeks: count}, n_global: int}
    Inilah jawaban atas pertanyaan "cakupan tipe gguf-py untuk file campuran".
    """
    g = load_gguf()
    reader = g.GGUFReader(path)
    tensors: list[dict[str, Any]] = []
    qtypes: dict[str, dict[str, Any]] = {}
    unsupported: list[str] = []
    layers: dict[str, int] = {}
    n_global = 0
    for t in reader.tensors:
        qname = t.tensor_type.name
        ok = is_supported(t.tensor_type, g)
        tensors.append(
            {
                "name": t.name,
                "qtype": qname,
                # Catatan: t.shape dari GGUFReader berurutan GGML [K, N]
                # (terbalik). Kita laporkan urutan NumPy [N, K] — sama dengan
                # keluaran dequantize().
                "shape": tuple(int(d) for d in t.shape[::-1]),
                "n_elements": int(t.n_elements),
                "n_bytes": int(t.n_bytes),
                "supported": ok,
                "layer": layer_of(t.name),
            }
        )
        slot = qtypes.setdefault(qname, {"count": 0, "bytes": 0, "supported": ok})
        slot["count"] += 1
        slot["bytes"] += int(t.n_bytes)
        if not ok:
            unsupported.append(t.name)
        li = layer_of(t.name)
        if li is None:
            n_global += 1
        else:
            layers[str(li)] = layers.get(str(li), 0) + 1
    return {
        "path": path,
        "n_tensors": len(tensors),
        "tensors": tensors,
        "qtypes": qtypes,
        "unsupported": unsupported,
        "layers": dict(sorted(layers.items(), key=lambda kv: int(kv[0]))),
        "n_global": n_global,
    }


def iter_layer_tensors(path: str, layer: int | None) -> Iterator[tuple[str, np.ndarray]]:
    """Mendekuantisasi tensor milik satu layer (atau global) satu per satu."""
    g = load_gguf()
    reader = g.GGUFReader(path)
    for t in reader.tensors:
        if layer_of(t.name) != layer:
            continue
        if not is_supported(t.tensor_type, g):
            raise NotImplementedError(
                f"tensor {t.name} bertipe {t.tensor_type.name} tidak didukung gguf-py"
            )
        yield t.name, dequantize_tensor(t, g)


def extract_layer(
    path: str, layer: int | None, out_dir: str, dtype: str = "float32"
) -> dict[str, Any]:
    """
    Menulis bobot satu layer (atau global) ke `out_dir/<label>.npz`.
    dtype: "float32" (bawaan, presisi penuh) atau "float16" (hemat disk).
    Mengembalikan manifes {nama: bentuk}.
    """
    if dtype not in ("float32", "float16"):
        raise ValueError("dtype harus float32 atau float16")
    os.makedirs(out_dir, exist_ok=True)
    label = "global" if layer is None else f"layer_{layer:02d}"
    out_path = os.path.join(out_dir, f"{label}.npz")
    arrays: dict[str, np.ndarray] = {}
    for name, arr in iter_layer_tensors(path, layer):
        arrays[name] = arr.astype(np.float16 if dtype == "float16" else np.float32)
    if not arrays:
        raise ValueError(f"tidak ada tensor untuk layer={layer}")
    np.savez(out_path, **arrays)
    return {"path": out_path, "dtype": dtype, "tensors": {k: tuple(v.shape) for k, v in arrays.items()}}


def extract_all(path: str, out_dir: str, dtype: str = "float32") -> list[dict[str, Any]]:
    """Mengekstrak semua layer + global. Urut: global dulu, lalu layer 0..N."""
    rep = inspect(path)
    wanted: list[int | None] = [None] + [int(k) for k in rep["layers"]]
    return [extract_layer(path, li, out_dir, dtype=dtype) for li in wanted]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} GB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="inventaris tipe & layer (tanpa dequant)")
    p_ins.add_argument("model")

    p_lay = sub.add_parser("layers", help="daftar indeks layer")
    p_lay.add_argument("model")

    def add_out(p):
        p.add_argument("--out", required=True)
        p.add_argument("--dtype", choices=("float32", "float16"), default="float32")

    p_ex = sub.add_parser("extract", help="ekstrak satu layer (atau global)")
    p_ex.add_argument("model")
    p_ex.add_argument("--layer", type=int, default=0)
    p_ex.add_argument("--global_", action="store_true", help="ekstrak tensor global")
    add_out(p_ex)

    p_all = sub.add_parser("extract-all", help="ekstrak semua layer + global")
    p_all.add_argument("model")
    add_out(p_all)

    args = ap.parse_args()

    if args.cmd == "inspect":
        rep = inspect(args.model)
        print(f"tensor: {rep['n_tensors']} | layer: {len(rep['layers'])} | global: {rep['n_global']}")
        print(f"{'tipe':<14} {'jumlah':>6} {'ukuran':>10}  didukung")
        for qname, slot in rep["qtypes"].items():
            print(f"{qname:<14} {slot['count']:>6} {_fmt_bytes(slot['bytes']):>10}  "
                  f"{'ya' if slot['supported'] else 'TIDAK'}")
        if rep["unsupported"]:
            print(f"\n[PERINGATAN] {len(rep['unsupported'])} tensor tak didukung:")
            for n in rep["unsupported"][:20]:
                print("  -", n)
        else:
            print("\nSemua tipe tensor didukung dekuantisasi gguf-py.")
    elif args.cmd == "layers":
        rep = inspect(args.model)
        print("layer:", " ".join(rep["layers"].keys()) or "(tidak ada)")
        print("global tensors:", rep["n_global"])
    elif args.cmd == "extract":
        layer = None if args.global_ else args.layer
        man = extract_layer(args.model, layer, args.out, dtype=args.dtype)
        print(f"ditulis: {man['path']} ({len(man['tensors'])} tensor, {man['dtype']})")
    elif args.cmd == "extract-all":
        for man in extract_all(args.model, args.out, dtype=args.dtype):
            print(f"ditulis: {man['path']} ({len(man['tensors'])} tensor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
