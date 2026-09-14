# Bonsai-27B 1-Bit Inference untuk NVIDIA T4

Inferensi **Bonsai-27B** (kuantisasi 1-bit biner, W1A16, `group_size 128`, Affine Bonsai) di GPU **NVIDIA T4** (Turing, `sm_75`), ditulis dalam **Mojo** dengan kernel CUDA `sm_75` lewat FFI.

Repositori ini berisi:

1. **Kernel matmul & decode 1-bit** — GEMV decode (`M ≤ 8`), prefill WMMA (`M > 8`), plus rangkaian kernel elementwise/GDN untuk arsitektur hybrid Bonsai-27B.

---

## Hasil Terukur (Kaggle, Tesla T4)


**Catatan soal profil per-subsistem.** Baris `[PROF/SPLIT]` **tidak** memecah waktu GPU per subsistem pada konfigurasi default. Di `main.mojo`, `acc_gdn +=` dan `acc_attn +=` berada **di luar** `if prof:` — hanya `synchronize()`-nya yang di dalam `if prof:`. Karena `BONSAI_PROFILE` dimatikan di `deploy_on_kaggle.sh` (sengaja: sync 65×/token mematikan pipeline async), `prof` bernilai false, sehingga `acc_gdn`/`acc_attn` hanyalah waktu *submit CPU* dan `acc_lm` adalah waktu GPU **seluruh token** yang terkuras pada sync terakhir.

Konsekuensinya, angka `LM_HEAD+argmax` pada baris itu (~47 ms) **bukan** biaya LM head — itu waktu forward satu token penuh. Kalau angka itu benar-benar LM head, 64 layer harus membaca 4,9 GB dataset bobot dalam ~3 ms = ~1.500 GB/s, jauh di atas puncak bandwidth T4 (320 GB/s).

Yang bisa disimpulkan: decode benar-benar bandwidth-bound, membaca ~4,9 GB bobot per token dalam ~47 ms ≈ **104 GB/s** (≈33% puncak T4). Pemecahan per-subsistem yang sah hanya didapat dengan `BONSAI_PROFILE=1` di sesi profiling terpisah.

---

## Arsitektur Bonsai-27B (Qwen 3.5 / 3.8 Hybrid)

- **64 layer**: 48 *Gated DeltaNet* (linear, stateless, **tanpa** KV cache) + 16 *full attention*.
- Layer attention adalah `li ≡ 3 (mod 4)` → **3, 7, 11, …, 63**. Layer non-attention tidak menyimpan KV.
- Konfigurasi: `hidden 5120`, `vocab 248320`, `H_q 24`, `H_kv 4` (GQA), `head_dim 256`, `rotary_dim 64`, `rope_theta 1e7`.

---

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


### Variabel lingkungan

| Variabel | Fungsi |
|---|---|
| `BONSAI_CUDA_LIB` | Path `libbonsai_qmv_sm75.so` |
| `BONSAI_USE_GPU` | Wajib; tanpa ini program berhenti (tanpa fallback CPU) |
| `BONSAI_DUMP_TOP2` | Cetak logit top-2 (pembanding numerik) |
| `BONSAI_PROFILE` | Sync per tahap (profil akurat, ~6% lebih lambat) |
| `BONSAI_NO_FUSE` | Matikan fusi (pembanding A/B) |
| `BONSAI_PREFILL_PER_TOKEN` | Paksa prefill per-token (pembanding) |
| `BONSAI_DISABLE_CUDA_FFI` | Matikan jalur CUDA FFI |

### Deploy & uji otomatis di Kaggle

```bash
bash push_to_kaggle.sh
```

`deploy_on_kaggle.sh` secara berurutan: kompilasi CUDA → smoke test FFI → uji kernel → build `main.mojo` → buat wheel → jalankan inferensi di T4 → gerbang koherensi, lalu **profil per-fase**.

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
├── deploy_on_kaggle.sh           # Build + uji + inferensi di Kaggle
├── push_to_kaggle.sh             # Push dataset & kernel, lalu unduh artefak
├── src/
│   ├── common.mojo               # Konstanta arsitektur, geometri tile, stride
│   ├── dequant.mojo              # Ekstraksi bit LSB-first & branchless sign-flip
│   ├── ops.mojo                  # Dispatcher host + FFI CUDA
│   ├── csrc/qmv_sm75_kernel.cu   # Kernel CUDA sm_75 (matmul, elementwise, GDN)
│   ├── kernels/                  # Kernel Mojo: decode, prefill, elementwise, direct_smallm
│   ├── models/qwen3_5/           # Layer, attention, GDN, MLP, RoPE, norm
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


**Runtime**
- State per-layer saat ini tidak dipersistensikan antar-proses.

**Performa**
- `BONSAI_PROFILE=1` (per-tahap GDN/ATTN) dimatikan secara bawaan karena sync 65×/token mematikan pipeline async.

---

## Lisensi

Apache-2.0 License.
