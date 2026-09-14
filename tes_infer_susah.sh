#!/usr/bin/env bash
# ==============================================================================
# tes_infer_susah.sh — TES INFERENSI SOAL SUSAH di Kaggle T4, hasilnya jadi TEKS
#
# Alur: tulis prompt -> push kernel khusus -> tunggu -> unduh log -> detokenisasi
#
# Pemakaian:
#   ./tes_infer_susah.sh                          # pakai prompt di infer_susah/infer_config.json
#   ./tes_infer_susah.sh --prompt "soal baru..."  # ganti prompt
#   ./tes_infer_susah.sh --prompt-file soal.txt   # prompt dari file
#   ./tes_infer_susah.sh --max-tokens 768
#   ./tes_infer_susah.sh --temp-x100 70           # temperature 0.70 (sampling)
#   ./tes_infer_susah.sh --greedy                 # matikan sampling (argmax)
#   ./tes_infer_susah.sh --separator-test         # 3 prompt: fakta / cerita / penalaran
#   ./tes_infer_susah.sh --kernel-ab              # 4 varian jalur kernel, greedy, prompt sama
#   ./tes_infer_susah.sh --no-sync                # lewati sinkronisasi sumber (kode tak berubah)
#   ./tes_infer_susah.sh --no-push                # jangan push; hanya unduh log + detok
#   ./tes_infer_susah.sh --status                 # cek status kernel saja
#
# PENTING: kernel mengambil main.mojo/src dari dataset Kaggle, bukan dari direktori
# ini. Karena itu sumber disinkronkan dulu (default). Pakai --no-sync kalau kode
# tidak berubah, supaya tidak menunggu ~2 menit ekstra.
#
# Catatan: kernel ini HANYA menjalankan inferensi (build + 1 prompt). Tidak ada
# self-test / kalibrasi / A/B, jadi jauh lebih singkat dari push_to_kaggle.sh.
# ==============================================================================
set -e

export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRCDIR="$ROOT/infer_susah"          # sumber yang Anda edit
KDIR="$SRCDIR/_push"                # hasil render -> yang dikirim ke Kaggle
CFG="$SRCDIR/infer_config.json"
OUTDIR="$(dirname "$ROOT")/dist_kaggle_mojo/infer_susah"
SLUG="okiabrian/bonsai-infer-susah"
SRC_SLUG="okiabrian/bonsai-mojo-source"
PY="/Users/macmini/.workbuddy-ai/binaries/python/envs/default/bin/python"

PROMPT=""
PROMPT_FILE=""
MAX_TOKENS=""
TEMP_X100=""
DO_PUSH=1
SYNC_SOURCE=1
ONLY_STATUS=0
SEP_TEST=0
KERNEL_AB=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prompt)      PROMPT="$2"; shift 2 ;;
        --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
        --max-tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --temp-x100)   TEMP_X100="$2"; shift 2 ;;
        --greedy)      TEMP_X100="0"; shift ;;
        --separator-test) SEP_TEST=1; shift ;;
        --kernel-ab)   KERNEL_AB=1; shift ;;
        --no-sync)     SYNC_SOURCE=0; shift ;;
        --no-push)     DO_PUSH=0; shift ;;
        --status)      ONLY_STATUS=1; shift ;;
        -h|--help)     sed -n '2,24p' "$0"; exit 0 ;;
        *) echo ">> [ERROR] argumen tidak dikenal: $1" >&2; exit 2 ;;
    esac
done

command -v kaggle >/dev/null 2>&1 || { echo ">> [ERROR] kaggle CLI tidak ada di PATH" >&2; exit 1; }
[ -f ~/.kaggle/access_token ] || { echo ">> [ERROR] ~/.kaggle/access_token tidak ada" >&2; exit 1; }

if [ "$ONLY_STATUS" = "1" ]; then
    kaggle kernels status "$SLUG"
    exit 0
fi

# ---------------------------------------------------------------- 1. prompt
if [ -n "$PROMPT_FILE" ]; then
    [ -f "$PROMPT_FILE" ] || { echo ">> [ERROR] file prompt tidak ada: $PROMPT_FILE" >&2; exit 1; }
    PROMPT="$(cat "$PROMPT_FILE")"
fi

if [ -n "$PROMPT" ] || [ -n "$MAX_TOKENS" ] || [ -n "$TEMP_X100" ]; then
    "$PY" - "$CFG" "$PROMPT" "$MAX_TOKENS" "$TEMP_X100" <<'PY'
import json, sys
cfg_path, prompt, max_tokens, temp = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = json.load(open(cfg_path))
if prompt:
    cfg["prompt"] = prompt
    cfg.pop("prompts", None)   # prompt tunggal -> batalkan mode multi-prompt
if max_tokens:
    cfg["max_tokens"] = int(max_tokens)
if temp != "":
    cfg["temperature_x100"] = int(temp)
json.dump(cfg, open(cfg_path, "w"), ensure_ascii=False, indent=2)
print(">> [CFG] diperbarui")
PY
fi

# --------------------------------------------------------- mode tes pemisah
# Tiga prompt yang memisahkan "penalaran rusak" dari "konteks panjang rusak":
#   1. fakta pendek   -> sanity check: bisakah model menghasilkan teks koheren?
#   2. cerita panjang -> output panjang TANPA penalaran
#   3. penalaran      -> soal cerita matematika (butuh langkah beruntun)
# Kalau #1/#2 bagus tapi #3 rusak  -> masalah di penalaran.
# Kalau #2 juga rusak              -> masalah di generasi panjang / konteks.
if [ "$SEP_TEST" = "1" ]; then
    "$PY" - "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
mt = int(cfg.get("max_tokens", 512))
cfg.pop("prompt", None)
cfg["prompts"] = [
    {"prompt": "Sebutkan lima nama buah tropis yang umum dijumpai di Indonesia. "
               "Cukup tuliskan daftar namanya, tanpa penjelasan.",
     "max_tokens": 96},
    {"prompt": "Tulis sebuah cerita pendek sekitar 250 kata tentang seorang nelayan tua "
               "yang menemukan botol berisi peta harta karun di tepi pantai.",
     "max_tokens": mt},
    {"prompt": "Sebuah kolam dapat diisi penuh oleh pipa A dalam 6 jam dan oleh pipa B "
               "dalam 4 jam. Mula-mula kedua pipa dibuka bersamaan selama 2 jam, lalu pipa B "
               "ditutup dan hanya pipa A yang terus mengalir sampai kolam penuh. Berapa jam "
               "total waktu yang dibutuhkan untuk mengisi kolam sampai penuh? Tunjukkan "
               "langkah perhitungannya.",
     "max_tokens": mt},
]
json.dump(cfg, open(sys.argv[1], "w"), ensure_ascii=False, indent=2)
print(">> [CFG] mode TES PEMISAH aktif: 3 prompt "
      "(fakta pendek / cerita panjang / penalaran)")
PY
fi

# --------------------------------------------------- mode A/B jalur kernel
# Prompt SAMA, budget SAMA, GREEDY, hanya jalur kernel yang berbeda. Kalau
# urutan tokennya berbeda antar varian, sakelar itu benar-benar mengubah
# perhitungan -> sumber divergensi ada di jalur tersebut. Kalau identik,
# masalahnya bukan di sana.
#   BONSAI_NO_FUSE=1            -> jalankan ulang RMS-norm final yang di-skip fusi
#   BONSAI_PREFILL_PER_TOKEN=1  -> ganti prefill WMMA batched jadi GEMV per token
if [ "$KERNEL_AB" = "1" ]; then
    "$PY" - "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["temperature_x100"] = 0          # greedy: perbedaan token = perbedaan jalur
mt = int(cfg.get("max_tokens", 256))
cfg.pop("prompt", None)
P = ("Tulis sebuah cerita pendek sekitar 250 kata tentang seorang nelayan tua "
     "yang menemukan botol berisi peta harta karun di tepi pantai.")
cfg["prompts"] = [
    {"label": "baseline (fusi + prefill batched)", "prompt": P, "max_tokens": mt},
    {"label": "no-fuse", "prompt": P, "max_tokens": mt,
     "env": {"BONSAI_NO_FUSE": "1"}},
    {"label": "prefill per-token", "prompt": P, "max_tokens": mt,
     "env": {"BONSAI_PREFILL_PER_TOKEN": "1"}},
    {"label": "no-fuse + per-token", "prompt": P, "max_tokens": mt,
     "env": {"BONSAI_NO_FUSE": "1", "BONSAI_PREFILL_PER_TOKEN": "1"}},
]
json.dump(cfg, open(sys.argv[1], "w"), ensure_ascii=False, indent=2)
print(f">> [CFG] mode A/B JALUR KERNEL aktif: 4 varian, greedy, {mt} token")
PY
fi

"$PY" - "$CFG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
t = int(cfg.get("temperature_x100", 0))
print(">> [CFG] max_tokens =", cfg.get("max_tokens"))
if t == 0:
    print(">> [CFG] sampling   = GREEDY (temperature_x100=0)")
else:
    print(f">> [CFG] sampling   = temperature={t/100:.2f}"
          f" | top_k={cfg.get('top_k',20)}"
          f" | top_p={int(cfg.get('top_p_x1000',950))/1000:.3f}"
          f" | min_p={int(cfg.get('min_p_x1000',0))/1000:.3f}"
          f" | rep_penalty={int(cfg.get('rep_penalty_x100',100))/100:.2f}"
          f" | seed={cfg.get('seed',1234)}")
prompts = cfg.get("prompts")
if prompts:
    print(f">> [CFG] prompts    = {len(prompts)} buah (mode multi-prompt):")
    for i, it in enumerate(prompts, 1):
        if isinstance(it, str):
            print(f">> [CFG]   {i}. {it[:110]}")
            continue
        extra = ""
        if it.get("label"):
            extra += f" [{it['label']}]"
        if it.get("env"):
            extra += " env=" + ",".join(f"{k}={v}" for k, v in it["env"].items())
        print(f">> [CFG]   {i}. [max={it.get('max_tokens')}]{extra} {it.get('prompt','')[:100]}")
else:
    print(">> [CFG] prompt     =", cfg.get("prompt"))
PY

# ------------------------------------------- 2. sinkron sumber ke dataset Kaggle
# Kernel mengambil main.mojo/src dari dataset, BUKAN dari direktori ini. Tanpa
# langkah ini, perubahan kode lokal tidak akan ikut teruji (kode basi).
if [ "$SYNC_SOURCE" = "1" ] && [ "$DO_PUSH" = "1" ]; then
    echo ""
    echo "========================================================="
    echo ">> Sinkronisasi sumber ke dataset ($SRC_SLUG)..."
    echo "========================================================="
    PKG="$(mktemp -d)"
    export COPYFILE_DISABLE=1
    cp "$ROOT/dataset-metadata.json" "$PKG/"
    cp "$ROOT/deploy_on_kaggle.sh" "$PKG/"
    cp "$ROOT/main.mojo" "$PKG/"
    cp "$ROOT/run_deploy.py" "$PKG/"
    cp "$ROOT/pixi.toml" "$ROOT/mojoproject.toml" "$PKG/"
    cp "$ROOT/README.md" "$PKG/" 2>/dev/null || touch "$PKG/README.md"
    cp -r "$ROOT/src" "$ROOT/tests" "$ROOT/benchmarks" "$PKG/"
    cp -r "$ROOT/scripts" "$PKG/" 2>/dev/null || true
    ( cd "$ROOT" && tar --exclude="._*" --exclude=".DS_Store" --exclude=".git" \
        --exclude="__pycache__" --exclude="build" --exclude="dist" \
        --exclude=".magic" --exclude="references" --exclude=".pixi" \
        --exclude="dist_kaggle_mojo" --exclude="infer_susah" \
        -czf "$PKG/mojo_source.tar.gz" . )
    kaggle datasets version -p "$PKG" -r tar -m "sync: $(date +'%Y-%m-%d %H:%M:%S')" -d
    rm -rf "$PKG"
    echo ">> Menunggu dataset siap..."
    for _ in $(seq 1 30); do
        DS_ST="$(kaggle datasets status "$SRC_SLUG" 2>/dev/null || echo '')"
        if echo "$DS_ST" | grep -qi ready; then echo ">> [OK] $DS_ST"; break; fi
        sleep 10
    done
    sleep 20
fi

# ------------------------------------------------- 3. render kernel mandiri
echo ""
echo "========================================================="
echo ">> Merender kernel mandiri (prompt + skrip shell ditanam)..."
echo "========================================================="
"$PY" - "$SRCDIR" "$KDIR" "$CFG" <<'PY'
import json, os, re, shutil, sys

srcdir, kdir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(cfg_path))
prompts = cfg.get("prompts")
prompt = cfg.get("prompt", "")
max_tokens = int(cfg.get("max_tokens", 512))
shell = open(os.path.join(srcdir, "deploy_infer.sh")).read()
tpl = open(os.path.join(srcdir, "run_infer.py")).read()

def sub_block(text, tag, body):
    pat = re.compile(r"(?s)# ==BEGIN " + tag + r"==\n.*?\n# ==END " + tag + r"==")
    if not pat.search(text):
        sys.exit(f">> [ERROR] penanda ==BEGIN {tag}== tidak ada di template")
    return pat.sub(lambda m: f"# ==BEGIN {tag}==\n{body}\n# ==END {tag}==", text, count=1)

# PROMPTS_JSON: None kalau mode prompt tunggal, else JSON list.
if prompts:
    prompts_repr = repr(json.dumps(prompts, ensure_ascii=False))
else:
    prompts_repr = "None"

out = sub_block(tpl, "PROMPT", f"HARD_PROMPT = {prompt!r}")
out = sub_block(out, "PROMPTSJSON", f"PROMPTS_JSON = {prompts_repr}")
out = sub_block(out, "MAXTOK", f"MAX_TOKENS = {max_tokens}")
out = sub_block(out, "DEPLOY", f"DEPLOY_SH = {shell!r}")
out = sub_block(out, "CFGJSON", f"CFG_JSON = {json.dumps(cfg, ensure_ascii=False)!r}")

os.makedirs(kdir, exist_ok=True)
with open(os.path.join(kdir, "run_infer.py"), "w") as f:
    f.write(out)
shutil.copyfile(os.path.join(srcdir, "kernel-metadata.json"),
                os.path.join(kdir, "kernel-metadata.json"))
mode = f"MULTI ({len(prompts)} prompt)" if prompts else f"TUNGGAL ({len(prompt)} char)"
print(f">> [RENDER] {kdir}/run_infer.py  ({len(out)} byte)")
print(f">> [RENDER] mode {mode} | max_tokens {max_tokens} | shell {len(shell)} byte")
PY
# Cek cepat: hasil render harus bisa di-parse Python
"$PY" -m py_compile "$KDIR/run_infer.py" || { echo ">> [ERROR] hasil render tidak valid" >&2; exit 1; }
rm -rf "$KDIR/__pycache__"

# ----------------------------------------------------------------- 4. push
if [ "$DO_PUSH" = "1" ]; then
    echo ""
    echo "========================================================="
    echo ">> Mengirim kernel '$SLUG' ke Kaggle..."
    echo "========================================================="
    if ! kaggle kernels push -p "$KDIR"; then
        echo ">> [WARN] push gagal (mungkin konflik kernel lama). Coba hapus & kirim ulang..."
        kaggle kernels delete "$SLUG" 2>/dev/null || true
        sleep 5
        kaggle kernels push -p "$KDIR"
    fi
    echo ">> [OK] kernel terkirim. Menunggu alokasi instance T4..."
    sleep 20
fi

# ---------------------------------------------------------------- 5. polling
echo ""
echo "========================================================="
echo ">> Memantau status (Ctrl+C boleh; kernel tetap jalan di Kaggle)"
echo "========================================================="
while true; do
    ST="$(kaggle kernels status "$SLUG" 2>/dev/null || echo Unknown)"
    echo "[$(date +%H:%M:%S)] $ST"
    if echo "$ST" | grep -qi "complete"; then break; fi
    if echo "$ST" | grep -qi "error"; then
        echo ">> [GAGAL] kernel error. Log mentah tetap diunduh di bawah."
        break
    fi
    if echo "$ST" | grep -qi "cancel"; then
        echo ">> [CANCEL] kernel dibatalkan."; exit 1
    fi
    sleep 45
done

# ------------------------------------------------------------- 6. unduh log
echo ""
echo ">> Mengunduh log kernel (beri jeda finalisasi 30 detik)..."
sleep 30
mkdir -p "$OUTDIR"

KAGGLE_KERNEL_SLUG="$SLUG" KAGGLE_OUT_DIR="$OUTDIR" "$PY" <<'PYDL'
import json, os, sys, time, urllib.parse, urllib.request

user, slug = os.environ["KAGGLE_KERNEL_SLUG"].split("/", 1)
outdir = os.environ["KAGGLE_OUT_DIR"]
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
url = "https://www.kaggle.com/api/v1/kernels/output?" + urllib.parse.urlencode(
    {"userName": user, "kernelSlug": slug})

log_obj = None
for attempt in range(5):
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=90) as r:
            d = json.loads(r.read().decode())
        try:
            log_obj = json.loads(d.get("log") or "[]")
        except Exception:
            log_obj = []
        break
    except Exception as e:
        w = 30 + attempt * 30
        print(f">> [WARN] {e} — tunggu {w}s (percobaan {attempt+1}/5)", flush=True)
        time.sleep(w)

if log_obj is None:
    print(">> [ERROR] gagal mengunduh log"); sys.exit(1)

p = os.path.join(outdir, "infer_susah_log.json")
with open(p, "w") as f:
    json.dump(log_obj, f)
txt = "".join(e.get("data", "") or "" for e in log_obj)
open(os.path.join(outdir, "infer_susah.log"), "w").write(txt)
print(f">> [OK] log disimpan: {p} ({len(txt)} char)")
PYDL

# --------------------------------------------------------- 7. detokenisasi
echo ""
echo "========================================================="
echo ">> DETOKENISASI -> TEKS"
echo "========================================================="
if [ "$KERNEL_AB" = "1" ]; then
    # Mode A/B: yang utama adalah PERBANDINGAN urutan token antar varian.
    "$ROOT/cek_infer.sh" --log "$OUTDIR/infer_susah_log.json" --diff
    echo ""
    echo ">> Teks tiap varian (dipotong 500 karakter):"
    "$ROOT/cek_infer.sh" --log "$OUTDIR/infer_susah_log.json" --chars 500
else
    "$ROOT/cek_infer.sh" --log "$OUTDIR/infer_susah_log.json" --raw
fi

echo ""
echo ">> Log mentah: $OUTDIR/infer_susah.log"
