# Bonsai-27B 1-Bit Inference + KHQ KV-Cache Compression untuk NVIDIA T4

Inferensi **Bonsai-27B** (kuantisasi 1-bit biner, W1A16, `group_size 128`, Affine Bonsai) di GPU **NVIDIA T4** (Turing, `sm_75`), ditulis dalam **Mojo** dengan kernel CUDA `sm_75` lewat FFI.

Repositori ini berisi dua bagian:

1. **Kernel matmul & decode 1-bit** — GEMV decode (`M ≤ 8`), prefill WMMA (`M > 8`), plus rangkaian kernel elementwise/GDN untuk arsitektur hybrid Bonsai-27B.
2. **KHQ — kompresi KV-cache** — kalibrasi penuh di GPU dan runtime kompresi KV untuk layer attention, menggantikan KV cache fp16 dengan representasi terkompresi. Seluruh komputasi kalibrasi berjalan di CUDA; Mojo hanya menangani IO/FFI (baca dump, marshal buffer, tulis file). **Tidak ada fallback CPU.**

---

## Hasil Terukur (Kaggle, Tesla T4)

**Korektnes runtime KHQ** — dibandingkan dengan baseline KV cache fp16 pada prompt & bobot yang sama:

| Metrik | Hasil |
|---|---|
| Token greedy identik | **512 / 512** (prefix identik 512) |
| Selisih logit top-1 di batas prefill | **0.000000** |
| Layer attention terkalibrasi | **16** (id 3, 7, 11, …, 63) |
| Event kompresi | **3 per layer** (token 256, 384, 512) |

**Cadence kompresi** (terukur via `BONSAI_KHQ_DEBUG=1`):

```
token 256 -> kompres 128 tertua -> boundary 128, jendela raw 128
token 384 -> kompres 128 tertua -> boundary 256, jendela raw 128
token 512 -> kompres 128 tertua -> boundary 384, jendela raw 128
```

Setelah event pertama, kompresi terjadi **tiap 128 token**; jendela raw berayun 128 ↔ 256 dan region terkompresi tumbuh monoton.

**Performa** — KHQ **belum** lebih cepat pada konteks pendek; ia menukar memori dengan waktu:

| Konfigurasi | ms/token | tok/s |
|---|---|---|
| Baseline KV fp16 | 53,2 | 18,8 |
| KHQ aktif | 75,0 | 13,3 |

Ukuran KV terkompresi turun dari **1024 B → 220 B** per (token, head) — sekitar **4,7×** lebih kecil. Karena biaya attention tumbuh linear terhadap panjang sekuens, keuntungan ini baru terasa pada konteks panjang; pada 512 token KHQ masih berupa biaya tambahan.

---

## Arsitektur Bonsai-27B (Qwen 3.5 / 3.8 Hybrid)

- **64 layer**: 48 *Gated DeltaNet* (linear, stateless, **tanpa** KV cache) + 16 *full attention*.
- Layer attention adalah `li ≡ 3 (mod 4)` → **3, 7, 11, …, 63**. Layer non-attention tidak menyimpan KV.
- Konfigurasi: `hidden 5120`, `vocab 248320`, `H_q 24`, `H_kv 4` (GQA), `head_dim 256`, `rotary_dim 64`, `rope_theta 1e7`.

---

## KHQ: Kompresi KV-Cache

### Cara kerja

- **Ring buffer** 256 slot menyimpan K/V mentah (fp16). `watermark = 256`, `chunk = 128`.
- Saat jendela raw mencapai 256, **128 token tertua** dikompres; `boundary` maju 128 sehingga jendela kembali 128. Kompresi berikutnya terjadi tiap 128 token.
- Setiap langkah attention menghitung **dua region** lalu menggabungkannya dengan *log-sum-exp*:
  1. region terkompresi (dekompresi payload K/V),
  2. jendela raw di ring (K sudah di-RoPE pada posisi absolutnya).
- Terakhir, sigmoid gate diterapkan setelah merge.

### Skema kompresi per (token, head)

| Komponen | K | V |
|---|---|---|
| Mask 2-bit (16 × u32) | 64 B | — |
| Payload | 40 B (outlier 4-bit + tanda) | 104 B (VQ 7-bit + dim3 4-bit) |
| Norma / residual | 2 B + 2 B | 2 B + 2 B |
| Shared meta | 4 B | (pakai bersama) |

K: normalisasi → FWHT → rotor 4D → kuantisasi mask 2-bit dengan maksimum 50 outlier. V: normalisasi → FWHT → rotor → VQ (codebook 256 × 3 per patch) + komponen dimensi ke-3.

### Kalibrasi

Driver `src/khq/calib.mojo` menerima `kv_dump.bin` (hasil forward pass model asli) dan menghasilkan `khq_calib.bin`. Semua operasi berat di GPU via `launch_cal_op` (10 op) dan `launch_cal_svq_train`:

FWHT base → K-static → rotate → PCA → kmeans 4/dim → kmeans VQ → Procrustes → SmartK → SmartV → head-major, ditambah turnamen **SmartVQ** (seeds 42/137/271, 2 ronde 30+70 iterasi, Adam, temperature `1.5·(0.1/1.5)^(i/max)`, seleksi winner dari bobot *best*).

---

## Pemakaian

### Prasyarat

Kompilasi kernel CUDA (butuh `nvcc`) menjadi `libbonsai_qmv_sm75.so`:

```bash
bash scripts/build_cuda_ffi.sh
export BONSAI_CUDA_LIB=$PWD/build/libbonsai_qmv_sm75.so
```

### Menjalankan inferensi

```bash
pixi run mojo build -I . main.mojo -o bonsai_infer
BONSAI_USE_GPU=1 ./bonsai_infer --model-dir <dir_model> \
    --prompt-tokens 248045,846,198 --max-tokens 24 --gpu
```

Opsi: `--model-dir`, `--prompt-tokens` (id dipisah koma), `--max-tokens`, `--gpu`.

### Menghasilkan dump K/V untuk kalibrasi

```bash
BONSAI_DUMP_KV_DIR=/tmp/khq ./bonsai_infer --model-dir <dir_model> \
    --prompt-tokens <ids> --max-tokens 512 --gpu
# -> /tmp/khq/kv_dump.bin + attn_<layer>.bin
```

K/V diambil **dari forward pass model asli** (K post-norm pre-RoPE + V + skor softmax kausal). File `attn_<layer>.bin` wajib ada dan jumlah tokennya harus sama dengan dump, kalau tidak kalibrasi menolak berjalan (bukan diam-diam turun ke mode parseval).

### Menjalankan kalibrasi

```bash
pixi run mojo run -I . src/khq/calib.mojo /tmp/khq /tmp/khq/khq_calib.bin
```

### Mengaktifkan jalur kompresi

```bash
BONSAI_KHQ_PATH=/tmp/khq/khq_calib.bin ./bonsai_infer --model-dir <dir_model> \
    --prompt-tokens <ids> --max-tokens 512 --gpu
```

### Variabel lingkungan

| Variabel | Fungsi |
|---|---|
| `BONSAI_CUDA_LIB` | Path `libbonsai_qmv_sm75.so` |
| `BONSAI_USE_GPU` | Wajib; tanpa ini program berhenti (tanpa fallback CPU) |
| `BONSAI_KHQ_PATH` | Aktifkan KV terkompresi (file centroid) |
| `BONSAI_DUMP_KV_DIR` | Dump K/V model asli untuk kalibrasi |
| `BONSAI_KHQ_DEBUG` | Cetak event kompresi + norma payload tiap event |
| `BONSAI_DUMP_TOP2` | Cetak logit top-2 (pembanding numerik) |
| `BONSAI_PROFILE` | Sync per tahap (profil akurat, ~6% lebih lambat) |
| `BONSAI_NO_FUSE` | Matikan fusi (pembanding A/B) |
| `BONSAI_PREFILL_PER_TOKEN` | Paksa prefill per-token (pembanding) |
| `BONSAI_DISABLE_CUDA_FFI` | Matikan jalur CUDA FFI |

### Deploy & uji otomatis di Kaggle

```bash
bash push_to_kaggle.sh
```

`deploy_on_kaggle.sh` secara berurutan: kompilasi CUDA → smoke test FFI → uji kernel → build `main.mojo` → buat wheel → jalankan inferensi di T4 → **dump K/V asli → kalibrasi → uji runtime KHQ**, dengan gate otomatis (verifikasi dump, TOP2, cadence, dan isi payload).

---

## Kernel Matmul 1-Bit

### Jalur decode (`M = 1` dan batch kecil `M ≤ 8`)

- **Vectorized memory coalescing** — pemuatan bobot 32-bit hingga 128-bit (`uint4` = 1 grup penuh `g128` per lane).
- **Eliminasi LDS bank conflict** — padding `128 → 132` float (`132 % 32 == 4`), 8 kolom grup mengakses 8 bank berbeda.
- **Akumulator register-M** — bobot dibaca **1 kali** dari DRAM per token.
- **Reduksi warp deterministik** — tanpa `atomicAdd`, determinisme bitwise 100%.

### Jalur prefill (`M > 8`)

- **Dua varian tiling** — `prefill_sm75` (`BM=64, BN=32, BK=64`) dan `prefill_wmma` (`BM=64, BN=64, BK=64`, WMMA).
- **Eliminasi bank conflict** — `PAD = 8` pada matriks shared memory.
- **Dekuantisasi branchless** — `w_eff = s · (2·bit − 1)` dan trik tanda IEEE FP16 (`0xBC00 ^ (bit << 15)`), tanpa percabangan per-bit.
- **Uniform barrier** — seluruh thread CTA memanggil barrier bersama walau sel di luar batas matriks, mencegah deadlock warp pada batas ragged.

---

## Kontrak Matematika Kuantisasi

- **Bobot** `uint8`, shape `[N, K/8]`, 8 bobot per byte urutan **LSB-first**:
  `bit_i = (byte >> (k mod 8)) & 1`
- **Skala** `Float16`/`Float32`, shape `[N, (K + 127) / 128]`, `group_size = 128`.
- **Kontrak Affine Bonsai** (`b = −s`):
  `w_eff = (2·bit − 1) · s` → `+s` bila bit `1`, `−s` bila bit `0`.
  Bias `−s` terserap ke perkalian, tanpa alokasi bias terpisah maupun FMA tambahan.

---

## Struktur Repositori

```text
bonsai-1bit-t4-mojo/
├── main.mojo                     # CLI inferensi native
├── deploy_on_kaggle.sh           # Build + uji + kalibrasi + runtime KHQ di Kaggle
├── push_to_kaggle.sh             # Push dataset & kernel, lalu unduh artefak
├── src/
│   ├── common.mojo               # Konstanta arsitektur, geometri tile, stride
│   ├── dequant.mojo              # Ekstraksi bit LSB-first & branchless sign-flip
│   ├── ops.mojo                  # Dispatcher host + FFI CUDA (termasuk cal_op)
│   ├── csrc/qmv_sm75_kernel.cu   # Kernel CUDA sm_75 (matmul, elementwise, KHQ, calib)
│   ├── khq/
│   │   ├── calib.mojo            # Driver kalibrasi (dump -> khq_calib.bin)
│   │   └── runtime.mojo          # Runtime kompresi KV (ring, kompresi, attention)
│   ├── kernels/                  # Kernel Mojo: decode, prefill, elementwise, direct_smallm
│   ├── models/qwen3_5/           # Layer, attention, GDN, MLP, RoPE, norm, khq_dump
│   └── safetensors.mojo          # Loader safetensors + jsonlite
├── tests/                        # Selftest kernel, FFI, rope, argmax, arsitektur
├── benchmarks/                   # Micro-benchmark T4 + profil layer
└── scripts/build_cuda_ffi.sh     # Kompilasi kernel CUDA -> .so
```

---

## Menjalankan Uji

```bash
# Selftest kernel W1A16: kasus ekstrem (K mini, ragged N/M, broadcast, pola bit)
pixi run mojo run tests/selftest_sm75.mojo

# Validasi dimensi layer Bonsai-27B
pixi run mojo run tests/test_bonsai_shapes.mojo

# Smoke test FFI & primary context
pixi run mojo run tests/test_cuda_ffi_smoketest.mojo

# Verifikasi silang vs referensi FP64 (Python + NumPy)
python3 tests/verify_differential.py

# Benchmark throughput & bandwidth
pixi run mojo run benchmarks/bench_t4.mojo
```

---

## Keterbatasan yang Diketahui

Bagian ini jujur soal apa yang **belum** setara dengan implementasi referensi (Python/MLX):

**Kalibrasi**
- Jumlah token kalibrasi di pipeline deploy 567 token, sedangkan referensi memakai 32.768 (`SEQ_LEN=1024 × 32 batch`). Keterwakilan distribusi lebih rendah.
- Vektor `d` memakai seed turunan `layer_id` dengan xorshift, referensi memakai `layer_id // 4` via `np.random.RandomState`. Konsisten internal (kalibrasi ↔ runtime), tapi tidak bit-parity dengan referensi.
- RNG inisialisasi/noise SmartVQ memakai xorshift per-thread, referensi memakai MLX. Algoritma & distribusi sama, lintasan acak berbeda.
- Skor attention disimpan fp16; referensi bf16.

**Runtime**
- `win_len` belum dijaga terhadap `KHQ_RING` (256). Pada `max_seq = 4096` aman karena kompresi selalu mengejar, tetapi bila `max_seq` dinaikkan melewati ~4200 jendela raw dapat melebihi ring dan kernel membaca slot melingkar tanpa peringatan.
- Belum ada API reset state antar-sekuens; `khq_init_layer` idempoten.
- State per-layer saat ini tidak dipersistensikan antar-proses.

**Performa**
- `khq_attn_kernel` di-launch `<<<H_q = 24, 32>>>` — hanya 24 block × 32 thread (≈1,9% okupansi T4), dan loop token di dalam block berjalan serial. Ini kandidat utama optimasi (mis. split-K + merge logsumexp).
- Profiling per-tahap (`BONSAI_PROFILE=1`) dimatikan secara bawaan pada deploy, sehingga pembagian waktu GPU vs submit CPU belum terukur akurat.

---

## Lisensi

Apache-2.0 License.
