#!/usr/bin/env python3
# ==============================================================================
# detok_infer.py — Baca log inferensi Kaggle, ubah ID token jadi TEKS
#
# Latar belakang: main.mojo hanya mencetak ">> [GEN] token id: N | X ms".
# Tanpa detokenisasi, output tampak seperti angka acak (mis. 248068, 271, ...)
# padahal 248068 = <think>, 271 = "\n".
#
# Pemakaian:
#   python3 detok_infer.py                       # pakai log terbaru otomatis
#   python3 detok_infer.py --log <path>          # log tertentu (.json / .log)
#   python3 detok_infer.py --tokenizer <path>    # tokenizer.json tertentu
#   python3 detok_infer.py --raw                 # tampilkan juga tag <|im_end|> dll
#   python3 detok_infer.py --run 2               # hanya run ke-2
# ==============================================================================

import argparse
import json
import os
import re
import subprocess
import sys

# Token khusus yang WAJIB ditandai saat decode (model ini, bukan Qwen standar)
EOS_ID = 248046          # <|im_end|>
IM_START_ID = 248045     # <|im_start|>
SPECIAL_IDS = {
    248044: "<|endoftext|>",
    248045: "<|im_start|>",
    248046: "<|im_end|>",
    248047: "<|object_ref_start|>",
    248048: "<|object_ref_end|>",
    248049: "<|box_start|>",
    248050: "<|box_end|>",
    248051: "<|quad_start|>",
    248052: "<|quad_end|>",
    248053: "<|vision_start|>",
    248054: "<|vision_end|>",
    248055: "<|vision_pad|>",
    248056: "<|image_pad|>",
    248057: "<|video_pad|>",
    248058: "<tool_call>",
    248059: "</tool_call>",
    248060: "<|fim_prefix|>",
    248061: "<|fim_middle|>",
    248062: "<|fim_suffix|>",
    248063: "<|fim_pad|>",
    248064: "<|repo_name|>",
    248065: "<|file_sep|>",
    248066: "<tool_response>",
    248067: "</tool_response>",
    248068: "<think>",
    248069: "</think>",
    248070: "<|audio_start|>",
    248071: "<|audio_end|>",
    248072: "<tts_pad>",
    248073: "<tts_text_bos>",
    248074: "<tts_text_eod>",
    248075: "<tts_text_bos_single>",
    248076: "<|audio_pad|>",
}

DEFAULT_TOKENIZER_CANDIDATES = [
    os.path.expanduser("~/.cache/bonsai_tok/tokenizer.json"),
    "/tmp/bonsai_tok/tokenizer.json",
]
TOKENIZER_DATASET = "okiabrian/bonsai-27b-mlx-1bit"
KAGGLE_BIN_CANDIDATES = [
    "/Users/macmini/.mounty/SSD_External/miniconda3/bin",
]


def find_log(explicit):
    """Cari log inferensi terbaru kalau --log tidak diberikan."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f">> [ERROR] log tidak ditemukan: {explicit}")
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    cands = [
        os.path.join(parent, "dist_kaggle_mojo", "build_log.json"),
        os.path.join(parent, "dist_kaggle_mojo", "bonsai-mojo-t4-build.log"),
        os.path.join(here, "dist_kaggle_mojo", "build_log.json"),
        os.path.join(here, "dist_kaggle_mojo", "bonsai-mojo-t4-build.log"),
    ]
    cands = [c for c in cands if os.path.exists(c)]
    if not cands:
        sys.exit(">> [ERROR] tidak ada log ditemukan. Berikan path lewat --log.")
    newest = max(cands, key=os.path.getmtime)
    print(f">> [LOG] memakai log terbaru: {newest}")
    return newest


def load_log_text(path):
    """Log Kaggle bisa berupa array JSON {stream_name,data} atau teks biasa."""
    raw = open(path, "r", errors="replace").read()
    stripped = raw.lstrip()
    if stripped.startswith("["):
        try:
            entries = json.loads(raw)
            if isinstance(entries, list) and entries and isinstance(entries[0], dict):
                return "".join(e.get("data", "") or "" for e in entries)
        except Exception:
            pass
    if stripped.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "logNullable" in obj:
                entries = json.loads(obj["logNullable"])
                return "".join(e.get("data", "") or "" for e in entries)
        except Exception:
            pass
    return raw


def ensure_tokenizer(explicit):
    """Pastikan tokenizer.json tersedia; unduh dari Kaggle bila belum ada."""
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f">> [ERROR] tokenizer tidak ditemukan: {explicit}")
        return explicit
    env = os.environ.get("BONSAI_TOKENIZER")
    if env and os.path.exists(env):
        return env
    for c in DEFAULT_TOKENIZER_CANDIDATES:
        if os.path.exists(c):
            return c

    cache = os.path.expanduser("~/.cache/bonsai_tok")
    os.makedirs(cache, exist_ok=True)
    target = os.path.join(cache, "tokenizer.json")
    print(f">> [TOK] tokenizer belum ada — mengunduh dari Kaggle "
          f"({TOKENIZER_DATASET}) ...")
    env2 = dict(os.environ)
    for d in KAGGLE_BIN_CANDIDATES:
        if os.path.isdir(d):
            env2["PATH"] = d + os.pathsep + env2.get("PATH", "")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", TOKENIZER_DATASET,
             "-f", "tokenizer.json", "-p", cache, "--force"],
            check=True, env=env2, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        sys.exit(f">> [ERROR] gagal mengunduh tokenizer ({e}).\n"
                 f"   Unduh manual lalu jalankan: --tokenizer <path tokenizer.json>")
    if not os.path.exists(target):
        sys.exit(">> [ERROR] tokenizer.json tidak muncul setelah unduh.")
    print(f">> [TOK] tersimpan: {target}")
    return target


def split_runs(text):
    """Pecah log jadi blok per-run (tiap run dimulai dengan >> [MOJO-NATIVE]).

    Penanda run/prompt dicetak SEBELUM >> [MOJO-NATIVE] run tsb, jadi berada di
    ujung blok sebelumnya (`prev`).
    """
    parts = text.split(">> [MOJO-NATIVE]")
    runs = []
    for i, p in enumerate(parts[1:], 1):
        ids = [int(m) for m in re.findall(r">> \[GEN\] token id:\s*(\d+)", p)]
        if not ids:
            continue
        prev = parts[i - 1]

        # Prompt: utamakan penanda JSON (aman untuk teks multi-baris/berkutip),
        # lalu jatuh ke format lama BONSAI_PROMPT="...".
        prompt_text = None
        pj = re.findall(r">> \[PROMPT-JSON\]\s*(\{.*\})", prev)
        if pj:
            try:
                prompt_text = json.loads(pj[-1])
            except Exception:
                prompt_text = None
        if prompt_text is None:
            old = re.search(r">> \[PROMPT\] BONSAI_PROMPT=\"([^\"]*)\"", prev) or \
                re.search(r">> \[PROMPT\] BONSAI_PROMPT=\"([^\"]*)\"", p)
            prompt_text = old.group(1) if old else None

        runmark = re.findall(r">> \[RUN\] idx=(\d+)\s+max_tokens=(\d+)(?:\s+label=(.+))?", prev)
        envmark = re.findall(r">> \[ENV\] (.+)", prev)
        ptoks = re.search(r">> Prompt tokens:\s*(\d+) \| max_tokens:\s*(\d+)", p)
        top2 = re.search(r">> \[TOP2\] 1st:\s*(\d+)\s*=\s*([-\d.eE+]+)\s*\|"
                         r"\s*2nd:\s*(\d+)\s*=\s*([-\d.eE+]+)", p)
        perf = re.search(r">> \[PERF\] rata-rata decode:\s*([-\d.eE+]+)\s*ms/token"
                         r"\s*\|\s*([-\d.eE+]+)\s*tok/s", p)

        max_tokens = None
        if ptoks:
            max_tokens = int(ptoks.group(2))
        elif runmark:
            max_tokens = int(runmark[-1][1])

        label = None
        if runmark and runmark[-1][2]:
            label = runmark[-1][2].strip()
            if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
                try:
                    label = json.loads(label)
                except Exception:
                    label = label[1:-1]

        runs.append({
            "idx": i,
            "run_no": int(runmark[-1][0]) if runmark else None,
            "label": label,
            "env": envmark[-1].strip() if envmark else None,
            "prompt_text": prompt_text,
            "prompt_len": int(ptoks.group(1)) if ptoks else None,
            "max_tokens": max_tokens,
            "top2": top2.groups() if top2 else None,
            "ms_per_tok": float(perf.group(1)) if perf else None,
            "ids": ids,
        })
    return runs


def decode_ids(tok, ids, skip_special):
    """Decode dengan penjagaan id di luar vocab (mis. padding model)."""
    vocab_size = tok.get_vocab_size()
    out = []
    pending = []
    for i in ids:
        if i < vocab_size:
            pending.append(i)
        else:
            if pending:
                out.append(tok.decode(pending, skip_special_tokens=skip_special))
                pending = []
            out.append(f"<id-luar-vocab:{i}>")
    if pending:
        out.append(tok.decode(pending, skip_special_tokens=skip_special))
    return "".join(out)


def run_name(r):
    """Nama ringkas sebuah run: RUN n (prompt #k) [label] env=..."""
    parts = [f"RUN {r['idx']}"]
    if r.get("run_no"):
        parts.append(f"(prompt #{r['run_no']})")
    if r.get("label"):
        parts.append(f"[{r['label']}]")
    if r.get("env"):
        parts.append("env=" + r["env"])
    return " ".join(parts)


def diff_runs(runs, tok):
    """Bandingkan urutan token antar run terhadap run pertama.

    Dipakai untuk A/B jalur kernel: kalau dua konfigurasi menghasilkan urutan
    token yang sama persis, berarti sakelar itu tidak mengubah apa pun; kalau
    berbeda, indeks divergensi pertama menunjukkan di mana jalurnya berpisah.
    """
    base = runs[0]
    print("\n" + "=" * 72)
    print("MODE BANDING (--diff) — urutan token antar run")
    print("=" * 72)
    print("acuan: " + run_name(base))
    for r in runs:
        extra = []
        if r["prompt_len"] is not None:
            extra.append(f"prompt {r['prompt_len']}")
        extra.append(f"gen {len(r['ids'])}")
        if r["ms_per_tok"]:
            extra.append(f"{r['ms_per_tok']:.1f} ms/tok")
        print("  " + run_name(r) + "  | " + " | ".join(extra))

    for r in runs[1:]:
        a, b = base["ids"], r["ids"]
        n = min(len(a), len(b))
        first = next((k for k in range(n) if a[k] != b[k]), None)
        print("\n" + "-" * 72)
        print(f"{run_name(base)}  vs  {run_name(r)}")
        if first is None:
            tail = "" if len(a) == len(b) else f" (panjang beda: {len(a)} vs {len(b)})"
            print(f"  IDENTIK untuk {n} token pertama{tail}")
        else:
            lo = max(0, first - 5)
            ca = tok.decode(a[lo:first + 5], skip_special_tokens=False)
            cb = tok.decode(b[lo:first + 5], skip_special_tokens=False)
            print(f"  cocok {first} token, lalu BEDA di index {first} (token ke-{first + 1})")
            print(f"    acuan  : id {a[first]:>7}  konteks {ca!r}")
            print(f"    ini    : id {b[first]:>7}  konteks {cb!r}")
        print(f"  id unik: acuan {len(set(a))}/{len(a)}"
              f" | ini {len(set(b))}/{len(b)}")

    print("\n" + "=" * 72)
    print("Catatan: jalankan dengan greedy (BONSAI_TEMP_X100=0) supaya perbedaan")
    print("token benar-benar berasal dari jalur kernel, bukan dari RNG sampling.")


def main():
    ap = argparse.ArgumentParser(description="Detokenisasi log inferensi Bonsai-27B")
    ap.add_argument("--log", help="path log (.json atau .log); default: terbaru")
    ap.add_argument("--tokenizer", help="path tokenizer.json")
    ap.add_argument("--run", type=int, help="hanya tampilkan run ke-N")
    ap.add_argument("--raw", action="store_true",
                    help="tampilkan juga tag khusus (<|im_end|> dll) apa adanya")
    ap.add_argument("--chars", type=int, default=2000,
                    help="maks karakter teks yang dicetak per run (default 2000)")
    ap.add_argument("--diff", action="store_true",
                    help="bandingkan urutan token antar run (A/B jalur kernel)")
    args = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit(">> [ERROR] modul 'tokenizers' belum terpasang.\n"
                 "   pip install tokenizers   (atau jalankan lewat ./cek_infer.sh)")

    log_path = find_log(args.log)
    tok_path = ensure_tokenizer(args.tokenizer)
    tok = Tokenizer.from_file(tok_path)

    text = load_log_text(log_path)
    runs = split_runs(text)
    if not runs:
        sys.exit(">> [ERROR] tidak ada baris '>> [GEN] token id' di log ini.")

    print(f">> [TOK] vocab={tok.get_vocab_size()} | {tok_path}")
    print(f">> [LOG] {len(runs)} run berisi token ter-generate")

    if args.diff:
        diff_runs(runs, tok)
        return

    for r in runs:
        if args.run and r["idx"] != args.run:
            continue
        ids = r["ids"]
        clean = decode_ids(tok, ids, skip_special=True)
        full = decode_ids(tok, ids, skip_special=False)

        # Potong di EOS pertama: itulah jawaban "sesungguhnya"
        cut = ids.index(EOS_ID) if EOS_ID in ids else len(ids)
        answer = decode_ids(tok, ids[:cut], skip_special=False)
        has_eos = EOS_ID in ids

        print("\n" + "=" * 72)
        print(run_name(r))
        if r["prompt_text"]:
            print("  prompt: " + clip(r["prompt_text"], 300))
        meta = []
        if r["prompt_len"] is not None:
            meta.append(f"prompt {r['prompt_len']} token")
        if r["max_tokens"] is not None:
            meta.append(f"max {r['max_tokens']}")
        meta.append(f"ter-generate {len(ids)} token")
        if r["ms_per_tok"]:
            meta.append(f"{r['ms_per_tok']:.1f} ms/token")
        print("  " + " | ".join(meta))
        if r["top2"]:
            a, av, b, bv = r["top2"]
            print(f"  TOP2 di batas prefill: {a} ({av}) | {b} ({bv})")
        print(f"  EOS (<|im_end|>={EOS_ID}) muncul di token ke-"
              f"{cut + 1}? {'YA' if has_eos else 'TIDAK'}")

        show = answer if answer.strip() else full
        print("\n  --- TEKS (sampai EOS pertama) ---")
        print(indent(clip(show, args.chars)))
        if args.raw:
            print("\n  --- TEKS MENTAH (tag khusus ditampilkan) ---")
            print(indent(clip(full, args.chars)))
            print(f"\n  ids = {ids[:64]}{' ...' if len(ids) > 64 else ''}")

    print("\n" + "=" * 72)
    print("Catatan: main.mojo BERHENTI di token henti (lihat '[STOP] token henti ...'")
    print("di log mentah), jadi teks di atas berakhir tepat di EOS pertama. Kalau EOS")
    print("tidak pernah muncul, generasi berakhir karena menyentuh batas max_tokens.")
    print("Pemotongan di atas hanya untuk tampilan; isi lengkap ada di log mentah.")


def clip(s, n):
    return s if len(s) <= n else s[:n] + f"\n  ...[dipotong, total {len(s)} karakter]"


def indent(s):
    return "\n".join("  | " + line for line in s.splitlines()) or "  | (kosong)"


if __name__ == "__main__":
    main()
