# CATATAN IMPLEMENTASI BONSAI-2 (2-bit Ternary) — Qwen3.8-27B

Dukungan inferensi 2-bit ternary untuk pack MLX `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`.
Dirakit sebagai **jalur paralel yang aditif** terhadap jalur 1-bit yang sudah ada:
semua parameter baru punya default yang mereproduksi perilaku 1-bit secara
byte-identik. Tidak ada satu pun fungsi 1-bit yang diubah isinya.

Aktifkan dengan variabel lingkungan `BONSAI_BITS=2` (default `1`).

---

## 1. Sumber kebenaran yang dipakai

Runtutan ini diturunkan dari tiga sumber yang saling diperiksa:

| Sumber | Isi yang diambil |
|---|---|
| `runtime/runtime.py` (pack MLX) | Kontrak dequant, kontrak signs, urutan FWHT, layout GDN, validasi `load()` |
| `hadamard.json` (297.903 byte, diunduh) | `block_size`, `sign_widths`, `sign_values`, `inverse_weight_names`, `gdn_v_grouped` |
| Header `model.safetensors` (303.102 byte JSON, 2390 tensor) | Bentuk & dtype tiap modul nyata |

`model_type` pack = `"prism_hadamard_qwen35"`, `schema_version: 2`,
`quantization: {bits: 2, group_size: 128, mode: "affine"}`.
Base model: **Qwen3.8-27B** (`qwen3_5_text`, arsitektur tidak berubah),
64 layer = 48 linear-attention + 16 full-attention (`full_attention_interval: 4`).

---

## 2. Kontrak dequant 2-bit (diverifikasi numerik)

```
w = q*s + b,  dengan b = -s  secara eksak
  => w = (q - 1) * s   untuk q ∈ {0, 1, 2}  ->  {-s, 0, +s}
```

**Tidak ada pembagian dengan 2.** Ini berbeda fundamental dari jalur 1-bit,
di mana `b = -s_ckpt/2` dan kernel memakai `(2q - 1) * (s_ckpt/2)`.

Konsekuensi pada loader:
- `read_scales_eff(index, e_s, numel, bits)` — loop `p[i] *= 0.5` hanya jalan
  saat `bits == 1`. Saat `bits == 2` scales dipakai mentah.
- `verify_affine_zero_bias(..., bits)` — faktor ekspektasi `1.0` saat
  `bits == 2` (atau saat `s_is_eff`), `0.5` saat `bits == 1`.

Layout bit: bobot tersimpan sebagai **U32 little-endian, 16 bobot per word,
lane `i` berada di bit `2i`** (mask `(word >> (2*i)) & 3`).
Scales/biases: **F16 [N, K/128]**, dibaca sebagai FP32 oleh `read_f32`.

---

## 3. Kontrak signs: dikunci oleh lebar masukan K, bukan per-modul

Ini temuan paling penting dan mengubah desain awal.

`runtime.py:92-100` membaca `prism.hadamard.sign_widths` = `[5120, 6144, 17408]`
dan `prism.hadamard.sign_values` (28.672 nilai ±1, konkatenasi berurutan sesuai
urutan widths), lalu membangun kamus `signs[width]`. Pada `runtime.py:233-240`
sebuah modul diaktifkan dengan `signs[shape[1]]` — yakni **signs dipilih
berdasarkan lebar masukan K saja**, bukan nama modul.

`hadamard.json` memang memiliki 401 `weight_names` yang terfold, tetapi hanya
**3** `sign_widths`. Dan meskipun ada 402 tensor `.signs` di `model.safetensors`
(F32, bentuk 5120/6144/17408), **runtime referensi tidak memakainya sama sekali**
— signs dibaca dari metadata hadamard, bukan dari safetensors.

Konsekuensi yang sudah diverifikasi dan diterapkan:

1. `in_proj_qkv` (K=5120) dan `in_proj_z` (K=5120) **berbagi signs yang sama**
   → fusi baris qkv+z adalah **eksak**.
2. `mlp.gate_proj` dan `mlp.up_proj` (keduanya K=5120) berbagi signs yang sama
   → fusi gate+up adalah **eksak**.
3. `in_proj_b` / `in_proj_a` adalah **F32 [48, 5120] — tidak terkuantisasi dan
   tidak terfold** (kota `"Unimplemented transformed float matrix"` di
   `runtime.py:254-255`). Keduanya harus menjadi **ekor dense** yang
   dikalikan dengan **aktivasi asli yang belum ditransformasi**.

Implementasi: `hadamard_signs_for_k(hd, k)` berjalan melewati `sign_widths`
dan mengembalikan vektor shared untuk lebar `k`, atau null bila tidak terfold.
Validasi: blok ∈ {512, 1024, 2048, 4096}, panjang `sign_values` == jumlah widths.

---

## 4. Konvensi FWHT (diverifikasi numerik)

Butterfly di mana **elemen rendah dari sepasang mengambil `x+y` dan elemen
tinggi mengambil `x-y`** ekuivalen dengan Sylvester `H_B / sqrt(B)`.

- **Forward**: `H_B @ (signs * x) / sqrt(B)`
- **Inverse**: `signs * (H_B @ x) / sqrt(B)`

Identitas jalur terfold: `(H @ (s*x))^T @ (H @ W) == x^T @ (s*W)` karena
`H^T H = I`. Inilah sebabnya bobot terfold dapat dikalikan dengan aktivasi
terotasi tanpa pernah menyimpan `H @ W` secara eksplisit.

`hadamard.json`:
- `block_size: 1024`
- `transform: "normalized-sylvester-walsh-hadamard"`
- `axis: "input-last-dimension"`
- `sign_mode: "explicit"` (semua nilai ±1, divalidasi)
- `inverse_weight_names`: **tepat 1** — `language_model.model.embed_tokens.weight`
- `gdn_v_grouped: True`

---

## 5. Layout modul nyata (diverifikasi dari header safetensors)

Layer-0 pack ini **tidak memiliki** `in_proj_all` maupun `gate_up_proj` yang
sudah difused — keduanya harus difuse di loader.

| Modul | Dtype | Bentuk |
|---|---|---|
| `input_layernorm.weight` | F32 | [5120] |
| `linear_attn.conv1d.weight` | F32 | [10240, 4, 1] |
| `linear_attn.in_proj_a.weight` | **F32** | **[48, 5120]** (dense, tak terfold) |
| `linear_attn.in_proj_b.weight` | **F32** | **[48, 5120]** (dense, tak terfold) |
| `linear_attn.in_proj_qkv.weight` | U32 | [10240, 320] |
| `linear_attn.in_proj_z.weight` | U32 | [6144, 320] |
| `linear_attn.norm.weight` | F32 | [128] |
| `linear_attn.out_proj.weight` | U32 | [5120, 384] |
| `mlp.down_proj.weight` | U32 | [5120, 1088] |
| `mlp.gate_proj.weight` | U32 | [17408, 320] |
| `mlp.up_proj.weight` | U32 | [17408, 320] |
| `post_attention_layernorm.weight` | F32 | [5120] |
| `self_attn.{q,k,v,o}_proj.weight` | U32 | (16 layer full-attention) |
| `self_attn.{q,k}_norm.weight` | F32 | (16 layer full-attention) |

Global: `lm_head.weight` U32 [248320, 320]; `model.embed_tokens.weight`
U32 [248320, 320] (satu-satunya inverse); `model.norm.weight` F32 [5120].
Vision tower (`vision_tower.*`, 333 tensor) F16 tanpa signs — tidak tersentuh.

Aturan derive bentuk: `K = nbytes * 8 // bits // n_rows`.

---

## 6. Numerik ekor dense (diverifikasi secara analitik)

Karena modul GDN punya campuran baris terkuantisasi (terhadap aktivasi
terotasi `H x`) dan baris dense (terhadap aktivasi asli `x`), kita andalkan:

```
y_tail = W_tail @ x_original = W @ (H^T H) x = (W H^T) @ (H x)
```

Sehingga **satu buffer aktivasi dapat memberi kedua bagian**: FWHT forward
ditulis ke buffer layer-private `self.x_dev` (sekali, menghindari transformasi
ganda saat q/k/v berbagi `x_norm_dev`), bagian packed memakai `self.x_dev`,
dan bagian dense memakai `x_dev` yang asli.

`N = N_packed + n_tail`: bagian packed menempati baris output `[0, N_packed)`,
ekor dense F32 `[n_tail, K]` menempati `[N_packed, N)`.

---

## 7. Berkas yang diubah (semuanya aditif)

### `src/csrc/qmv_sm75_kernel.cu`
- `namespace bonsai::sm75::q2t {`
  - `qmv_vec_q2t_kernel<T>` — 256 thread, 32 baris/blok, 8 lane/baris,
    K-tile 1024 = 8 grup, `GRP_PAD = 132`, membaca `w_row + gg*32` sebagai
    dua `uint4` 16-byte berurutan, decode `(word >> (2h)) & 3`,
    akumulasi `(float)(q-1) * xg[...]` lalu `acc += s_val * local`,
    split-K via `ws`/`slice_idx`.
  - `fwht_sm75_kernel<B, T>` — satu blok CUDA per (baris, B-blok),
    `extern __shared__ float smem[]`, butterfly
    `s[tid] = ((tid & h) == 0) ? (a + b) : (b - a)` dengan `b = s[tid ^ h]`,
    skala `1/sqrtf(B)`, signs dikali sebelum (forward) / sesudah (inverse).
    Aman in-place.
  - `qmv_dense_kernel<T>` — GEMV dense FP16 untuk ekor tak terkuantisasi.
    **Memperhatikan stride**: baris output distride oleh `N_total`, bukan
    `n_tail`, jadi indeks `out[row * n_total + n_packed + on]`.
- Launcher `extern "C"`:
  - `launch_qmv_sm75_b2_decode_fp16` — error `-100` bila `k % 128 != 0`,
    `-101` bila `(k/4) % 16 != 0`.
  - `launch_fwht_sm75_fp16` — `constexpr int B = 1024`, error `-102` bila
    `k % B != 0`.
  - `launch_qmv_sm75_dense_fp16` — `grid(n_tail, l, 1)`, `block(256,1,1)`.

Build tidak berubah: `deploy_on_kaggle.sh:435-437` mengompilasi satu berkas
ini menjadi `libbonsai_qmv_sm75.so`.

### `src/dequant.mojo`
`extract_2bit_lane`, `dequant_ternary_bonsai(q, s) = (q - 1.0) * s`,
`fma_2bit`, `unpack_byte_to_simd4_ternary`. Fungsi 1-bit tidak disentuh.

### `src/safetensors.mojo`
- `load_qlinear(st, w_name, s_name, b_name, bits: Int = 1)` —
  `var kk = nb * 8 // bits // nn`; loop `sb[i] *= 0.5` dijaga oleh
  `if bits == 1:`.
- `fn load_signs(st, name, k_dim)` — null bila tidak ada.

### `src/ops.mojo`
Alias FFI `CudaFwhtFnFP16`, `CudaQmvDenseFnFP16` + `dummy_cuda_fwht_fp16`,
`dummy_cuda_qmv_dense_fp16` (mengembalikan `-1`).

### `src/models/qwen3_5/linear.mojo`
Field baru: `bits`, `signs`, `signs_dev`, `signs_dev_buf`, `has_signs`,
`ffi_fwht_ready`, `ffi_fwht_fn`, serta blok ekor dense
(`N_packed`, `n_tail`, `tail_w`, `tail_dev`, `tail_dev_buf`,
`ffi_dense_ready`, `ffi_dense_fn`).

- `w_nbytes = (N_packed * K) // (8 // bits)` (bukan `N * K`).
- `ensure_dev_ready`: buffer scales, loop `s16h`, dan workspace split-K
  semua diubah ukurannya ke `N_packed`; ekor diunggah bila `n_tail > 0`;
  simbol dipilih menurut lebar bit
  (`launch_qmv_sm75_b1_decode_fp16` vs `launch_qmv_sm75_b2_decode_fp16`);
  `ffi_qmm_ready = False` saat `bits != 1`; `launch_fwht_sm75_fp16` diresolve
  saat `has_signs`; `launch_qmv_sm75_dense_fp16` saat `n_tail > 0`.
- `forward_device`: FWHT **forward** dari `x_dev` ke `self.x_dev` (menghindari
  transformasi ganda saat q/k/v berbagi `x_norm_dev`), launch packed dengan
  `N_packed`, lalu ekor dense dengan **`x_dev` asli**.
- `forward_gpu_on`: FWHT forward in-place pada `self.x_dev`, launch packed
  `N_packed`, dan ekor dense dihitung di **host** dari `x` host asli
  **setelah** copy-back ( agar tidak tertimpa).

### `main.mojo`
Helper baru: `bonsai_bits()` (env `BONSAI_BITS`), `struct HadamardSigns`,
`load_hadamard_signs(model_dir)` (pemindai JSON linear — menghindari
`arr_len`/`child_at` O(n²) atas 28.672 elemen), `hadamard_signs_for_k(hd, k)`,
`fwht_blocks_host(x, signs, k, block, inverse)`, `embed_lookup_2bit_host(...)`,
`embed_lookup_2bit_to_dev(...)`.

Parameter `bits` ditambahkan ke `verify_affine_zero_bias` dan `read_scales_eff`
(default `1` mempertahankan perilaku 1-bit).

Tubuh program: cabang muat berdasarkan `bits`; embed mendapat inverse FWHT
(satu-satunya `inverse_weight_names`); GDN `in_proj` membangun ekor dense dari
`in_proj_b`+`in_proj_a`; semua konstruksi `QwenLinear1Bit` meneruskan `bits` +
signs; **tiga** situs pemanggilan embedding lookup (batched prefill, prefill
per-token, decode loop) bercabang `if bits == 1: … else: embed_lookup_2bit_to_dev`.

---

## 8. Ukuran dan tekanan memori

| Format | Ukuran | bpw |
|---|---|---|
| Ternary g128 ideal | 5,8 GB | 1,72 |
| PQ2_0 (llama.cpp) | 7,21 GB | 2,13 |
| **MLX 2-bit (pack ini)** | **7,67 GB** | **2,25** |
| Total di disk (termasuk vision tower) | 8,60 GB | — |

Embedding 2-bit **tidak** diunggah ke VRAM (318 MB) — lookup dijalankan di
host (satu baris K elemen per token, ~puluhan µs) lalu disalin via buffer
scratch F16 [K]. Hanya `embed_b2_scratch` + dua buffer host [K] yang dibuat.

---

## 9. Sampling mode thinking Bonsai-2

`temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`;
reasoning effort default `xhigh`.

---

## 10. Status verifikasi — 2026-09-19 (Kaggle CPU build + GPU T4)

**Build CPU (kernel `okiabrian/bonsai-build-cpu` v11): BERHASIL.**
Perlu tiga percobaan; v9 tampak sukses padahal gagal total.

| | `.so` | `bonsai_infer` | wheel |
|---|---|---|---|
| v9 (palsu) | 2.011.848 B — **biner basi** | 859.592 B — **biner basi** | 225.199 B "TANPA binary" |
| v10 | 2.061.576 B — nvcc OK | masih gagal | 934.919 B |
| **v11** | **2.061.576 B — 6 launcher** | **885.760 B** | **1.182.749 B** |

`nm -D libbonsai_qmv_sm75.so` pada v11 memuat **6 launcher**:
`b1_decode_fp16`, `b1_decode_bf16`, `qmm_b1_prefill_fp16` (jalur 1-bit utuh,
tanpa regresi) + **`b2_decode_fp16`, `fwht_sm75_fp16`, `dense_fp16`** (baru).
`bonsai_infer` mengandung `BONSAI_BITS`, `hadamard.json` (13×), ketiga launcher
2-bit, dan penanda `[HADAMARD]`.

**Trap bahasa Mojo 25.x yang menyebabkan 51 error** (lihat MEMORY.md):
tanpa interpolasi `\(`, tanpa kembalian/destructuring tuple, `out` kata kunci
cadangan, parameter `len` menutupi bawaan, konstruktor struct posisional
**dan tidak dibuatkan otomatis** (wajib tulis `fn __init__(out self, ...)`).
Trap CUDA: `qmv_dense_kernel` harus berada di dalam `namespace bonsai::sm75::q2t`
karena `QmvTraits` hidup di `bonsai::sm75`.

**Uji inferensi GPU (`okiabrian/bonsai-mojo-t4-build` v125): LULUS** —
exit code 0, `>> [SELESAI] BUILD + UJI INFERENSI T4 DI KAGGLE!` (1970 s).
Bobot 1-bit, `Config: ... bits= 1`. Prefill 56 token ≈ 451 ms (**123 tok/s**);
decode rata-rata **≈ 68 ms/token (≈ 14,6 tok/s)**; run 511 token stabil.
`ptxas` mengompilasi `q2t::fwht_sm75_kernel<1024, __half>` untuk sm_75.
**Tidak ada regresi pada jalur 1-bit.**

### 10b. Uji jalur 2-bit itu sendiri — BERHASIL (kernel `okiabrian/bonsai-2bit-infer` v5)

Kernel terpisah `kaggle_2bit_test/` (berdiri sendiri; tidak menyentuh
`deploy_on_kaggle.sh` / `run_deploy.py`): mengambil biner dari wheel output
`bonsai-build-cpu`, mengunduh bobot 2-bit di dalam container, lalu menjalankan
`bonsai_infer` dengan `BONSAI_BITS=2`. Status `COMPLETE`, exit code 0.

```
>> Config: layers= 64  hidden= 5120  vocab= 248320  bits= 2
>> [HADAMARD] block= 1024  widths= 5120 6144 17408  total_signs= 28672
>> [STEP] indeks: 2391 tensor / 1 shard
>> [STEP] bobot global siap (packed 2 bit, V= 248320  D= 5120  lm_signs= True )
>> Bobot termuat: 64 layer ( 48 GDN, 16 attention )
>> [GPU] DeviceContext tunggal terpasang ke 64 layer + lm_head
>> [PERF] prefill 9 token | 1162.649313 ms | 129.183257 ms/token
>> [PERF] rata-rata decode: 64.33618953333333 ms/token | 15.543351374297483 tok/s
>> Selesai: 16 token di-generate (greedy).
```

Keluaran 16 token didekode memakai `tokenizer.json` pack tersebut:

```
<think>
The user said "Hello" - a simple greeting. I should respond
```

Koheren. `bits= 2` terbaca, signs hadamard terbaca (28.672 nilai, 3 lebar),
64 layer terbagi 48 GDN + 16 attention sesuai `layer_types` di `config.json`,
dan `lm_signs= True` (embedding memakai inverse FWHT).

Prefill 9 token memakan **129 ms/token** — itu jalur fallback per-token,
konsisten dengan `ffi_qmm_ready` yang dipaksa `False` untuk `bits != 1`.
Decode **64,3 ms/token (15,5 tok/s)**, sekelas dengan jalur 1-bit (14,6 tok/s).

### 10c. Prefill batched 2-bit (kernel qmm WMMA v2) — hasil pengukuran

Setelah 10b, jalur prefill batched diimplementasikan:
`launch_qmm_sm75_b2_prefill_fp16` + `qmm_sm75_b2_kernel` di namespace
`bonsai::sm75::wmma_b2`, dipanggil dari `forward_prefill_device` saat
`bits == 2`. Tiga hal yang berbeda dari versi 1-bit:

1. **2 bit per bobot**: 16 bobot per word U32 (lane i di bit 2i), baris bobot
   K/4 byte. Nilai bobot `{-1,0,+1}` dibentuk lewat tabel 4 entri berbasis
   PRMT (`qmm_b2_half_bits`).
2. **Stride baris output dipisah**: parameter `n_total` boleh lebih besar dari
   `n`, karena ekor dense (`in_proj_b/a`, 96 baris) menempati kolom
   `[n_packed, n_total)` dan diisi `launch_qmv_sm75_dense_fp16`.
3. **FWHT sebelum GEMM**: aktivasi M baris ditransformasi ke buffer scratch
   bersama (`pf_fwht`, `chunk_len × cfg.intermediate_size`) karena
   `x_norm_m_dev` dipakai bersama q/k/v — transformasi in-place akan
   menggandakan transformasi.

Ukuran pengukuran (prompt 129 token, T4, greedy, `BONSAI_BITS=2`):

| tahap | prefill | tok/s |
|---|---|---|
| fallback per-token (10b) | 129 ms/token | 7,7 |
| qmm 2-bit, BM=64 | 12,32 ms/token | 81,2 |
| + dekuantisasi PRMT | 12,28 ms/token | 81,4 (tak berpengaruh) |
| + BM=128 & lompat ubin bantalan | **11,03 ms/token** | **90,6** |

Dua catatan penting dari pengukuran ini:

- **Dekuantisasi PRMT tidak mengubah apa pun** (81,2 → 81,4 tok/s, dan ukuran
  `.so` identik ke byte). Kesimpulannya: ptxas sudah menerjemahkan rangkaian
  banding/select yang lama menjadi PRMT dengan sendirinya — jadi biayanya
  bukan operasi skalar per bobot.
- Yang berpengaruh adalah **jumlah pass dekuantisasi bobot**, yaitu
  `ceil(M/BM)`: BM=64 untuk M=129 butuh 3 pass, BM=128 cukup 2 pass
  (penghematan 33%) → +11,4%.

**Belum mencapai 100+ tok/s.** Perbandingan dengan jalur 1-bit (124 tok/s)
**tidak setara**: angka itu diukur pada prompt 56 token, sedangkan angka 2-bit
di atas pada prompt 129 token. Untuk menutup sisa gap diperlukan profil aktual
(`BONSAI_PROFILE=1` atau ncu), bukan perkiraan — perkiraan kasar dari dua
titik ukur memperkirakan MMA ≈ 34%, dekuantisasi ≈ 23%, dan sisanya di luar
GEMM, tetapi itu **inferensi**, bukan hasil ukur.

Empat kegagalan sebelum v5 berhasil — semuanya di luar kode kernel 2-bit:

| versi | penyebab | perbaikan |
|---|---|---|
| v1 | `hf_hub_download` meninggalkan `model.safetensors` 0 byte | `curl -L --retry 3 -C -` + penjaga ukuran 8.595.477.990 B |
| v2 | `libKGENCompilerRTShared.so: cannot open shared object file` (exit 127) | pasang runtime Mojo |
| v3 | `artifacts_modal/` hanya punya 1 dari 4 pustaka runtime (kurang `libMSupportGlobals.so`, `libAsyncRTRuntimeGlobals.so`, `libNVPTX.so`) | periksa `ldd` dulu, baru pasang pixi bila kurang |
| v4 | `glob` Python tidak melintasi `.pixi/` (direktori berawal titik) | pakai `find` + kandidat eksplisit `<env>/.pixi/envs/default/lib` |

## 11. Yang masih belum selesai / risiko terbuka

1. ~~**Jalur 2-bit belum pernah dijalankan.**~~ **SELESAI** — lihat 10b.
   ~~Yang tersisa dari pokok ini hanya kenyamanan: bobot 2-bit belum
   menjadi dataset Kaggle, sehingga setiap uji mengunduh 8,6 GB dari
   HuggingFace di dalam container (≈ 180 s).~~ **SELESAI** — bobot kini
   menjadi dataset `okiabrian/bonsai-2bit-weights`; lihat §13b.
2. ~~**Belum dikompilasi atau diuji.**~~ **SELESAI** — dikompilasi di
   `bonsai-build-cpu` v11 (`nvcc` 0 error, `mojo build` menghasilkan
   `bonsai_infer` 885.760 B) dan diuji di T4 (10b).
3. ~~**Jalur prefill batched 2-bit.**~~ **SELESAI** — lihat 10c: kernel
   `launch_qmm_sm75_b2_prefill_fp16` sudah terpasang dan diukur 90,6 tok/s
   pada prompt 129 token. **Kesetaraan numeriknya dengan jalur per-token
   juga sudah diverifikasi (§12g, VERDIK: LULUS)** — token top-1, nilai
   logit, dan 24 token generasi identik.
   ~~Yang tersisa: target 100+ tok/s belum tercapai (86,7 tok/s pada run
   verifikasi).~~ **TERCAPAI** lewat jalur int8 opt-in: 89,09 → **101,58
   tok/s** (rasio 1,140×) dengan verifikasi 6d LULUS — lihat §12h. Untuk
   melampaui angka ini butuh profil aktual, bukan perkiraan: ~98% waktu
   prefill habis di kuantisasi/dekuantisasi pada CUDA core, bukan di MMA.
   Dua pengungkit yang teridentifikasi dari referensi ada di §12f.
4. **GDN v-head permutation.** `hadamard.json` punya `gdn_v_grouped: True`,
   dan runtime referensi memperingatkan: *"GDN activations are already
   grouped in the bundled runtime; do not permute them again"*. Jalur GDN
   project ini harus dipastikan tidak menerapkan vperm tambahan untuk pack ini.
5. **Modal T4** masih terhalang (`modal.exception.InvalidError: Please add a
   payment method to use T4 GPU functions`).
6. `deploy_on_kaggle.sh` §4c: `--target-accelerator=sm_75` belum ditambahkan
   ke enam panggilan `pixi run mojo run tests/*_gpu.mojo`.
7. `modal_t4_mojo.py` baris 170/460 masih merujuk `./build_cpu/push_build_cpu.sh`
   yang sudah kedaluwarsa.

## 12. Audit referensi llama.cpp-prism — apakah ada pendekatan yang lebih baik?

Audit dijalankan atas permintaan ("cek llama.cpp, apakah di sana ada yang
lebih baik"). **Jawabannya: ya, materiil.** Lima temuan, semua diverifikasi
dengan membaca sumber referensi (hanya-baca, `references/llama.cpp-prism/`).

### 12a. Referensi mempunyai tipe yang PERSIS sama dengan format kita

`GGML_TYPE_PQ2_0` (`ggml-common.h`):

```c
#define QK_PQ2_0 128
typedef struct {
    ggml_half d;                // delta (scale)
    uint8_t qs[QK_PQ2_0 / 4];   // 2 bits per element
} block_pq2_0;
```

Komentar referensi: *"Prism-private Q2_0 at group size 128. Same 2-bit codec
as Q2_0 (group 64) but one fp16 scale per 128 weights."* — itu **persis**
kontrak Bonsai-2 kita (2 bit, grup 128, satu skala fp16 per 128 bobot).
Kode 2-bit → int8 lewat tabel PRMT:

```c
const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
```

Tabel `{0xFF, 0x00, 0x01, 0x02}` = **{-1, 0, +1, +2}** untuk kode {0,1,2,3}.
Kontrak kita `(q-1)*s` untuk `q ∈ {0,1,2}` → `{-1,0,+1}` adalah **himpunan
bagian** dari codec itu (kode 3 tak terpakai). Jadi kernel referensi PQ2_0
sudah memecahkan masalah yang sama dengan nilai yang sama.

### 12b. Referensi memakai INT8 tensor core, bukan FP16

`mma.cuh` (sm_75 / Turing):

```c
// On Turing m16n8k16 mma is not available, use 2x m8n8k16 mma instead:
asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 ...");
```

Decode memakai `ggml_cuda_dp4a` (produk titik int8 per warp), prefill
memakai `mma.sync` int8 dengan akumulasi **int32**. Konsekuensinya pada T4
(dikonfirmasi dari halaman spesifikasi resmi NVIDIA T4):

| presisi | throughput T4 |
|---|---|
| FP16 (tensor core) | 65 TFLOPS |
| **INT8 (tensor core)** | **130 TOPS** |

**2× throughput tensor core** dibanding kernel kita yang memakai
`nvcuda::wmma` fragment FP16.

### 12c. Skala diterapkan SEKALI di akhir, bukan per bobot

Kita: dekuantisasi mengembangkan kode ke FP16 `{-1,0,+1}` lalu mengalikan
dengan skala fp16 (`__hmul2`) **saat dekuantisasi** — satu perkalian fp16
per 2 bobot.

Referensi: bobot tetap sebagai **byte int8 mentah** (hasil PRMT), skala
fp16 baru dikalikan setelah akumulasi int32 selesai:

```c
sum[...] += C.x[l]*dA*dB;   // dA = skala bobot, dB = skala aktivasi
```

Satu perkalian per elemen output, bukan per bobot. Untuk grup 128 itu
menggantikan ~64 perkalian `half2` dengan 1 perkalian `float`.

### 12d. Aktivasi dikuantisasi ke Q8_1 SEKALI, dipakai ulang oleh seluruh ubin

- Prefill: `quantize_mmq_q8_1_cuda()` (mmq.cu:195) mengonversi blok M×K
  aktivasi ke int8 **sekali**, sebelum kernel MMQ berjalan.
- Decode: `quantize_row_q8_1_cuda()` (mmvq.cu:1442).

Kernel kita membaca aktivasi sebagai **FP16 per potongan-K** dari memori
global (2 byte/elemen + pembagian transposisi). Referensi membaca int8
(1 byte/elemen + skala per 32 yang teramortisasi) — **separuh bandwidth
aktivasi**.

### 12e. Konfigurasi ubin PQ2_0 di Turing

`mmq-config-ampere.cuh` (T4 memakai konfigurasi Ampere karena
`highest_compiled_arch >= VOLTA`):

```
CASE(GGML_TYPE_PQ2_0, 256, 1, 128, 128, ..., MMQ_ITER_K=256, stream_k=true)
         nthreads=256, occ=1, I=128, J=128, K_vram=256
```

- `I = 128` (baris bobot per blok) vs `BN = 64` kita → separuh jumlah blok.
- `J = 128` (baris batch per blok) vs `BM = 128` kita → sama.
- `stream_k = true`: dekomposisi stream-K
  (https://arxiv.org/abs/2301.03598) untuk menyeimbangkan beban antar SM.
- SMEM ≈ 56,5 KB (ubin bobot 38.912 B + ubin aktivasi 18.432 B + ids),
  masih di dalam jatah 64 KB T4. Kernel kita memakai 31.744 B untuk BM=128
  (`As[128][72]` 18.432 B + `Wt[64][72]` 9.216 B + `Csc[4][16][16]` 4.096 B),
  jadi `BN` naik ke 128 (total ≈ 39 KB) masih muat di 64 KB T4.

### 12f. Implikasi untuk target 100+ tok/s

Dua yang paling berdampak, berdasarkan temuan di atas:

1. **Beralih ke int8 MMA/DP4A** — 2× throughput tensor core di T4. Ini
   adalah port dari rancangan yang sudah terbukti di referensi, bukan
   penemuan baru, tetapi pekerjaannya bukan sepele: butuh kernel kecil
   untuk menguantisasi blok aktivasi M×K ke int8 (dengan skala per 32),
   mengubah kernel qmm agar bobot tetap int8 mentah (PRMT saja, tanpa
   perkalian skala), dan menerapkan skala fp16 di epilog.
2. **`BN` 64 → 128** — separuh jumlah blok, amortisasi pemuatan ubin
   aktivasi lebih baik. Mandiri dari (1), murah.

**Catatan presisi yang jujur:** menguantisasi aktivasi ke int8
menambahkan galat. Referensi menerima ini (itulah cara kerja llama.cpp),
dan bahkan secara eksplisit memilih MMQ di setiap batch untuk PTQ1_0
karena alasan ini — lihat komentar di `ggml_cuda_should_use_mmq`:
*"the fp16 dequantize + cuBLAS fallback is the source of PTQ1_0's extra
error on CUDA, so the MMQ tile path runs at every batch by default"*.
Bila jalur ini diadopsi, verifikasi numerik prefill-vs-decode (§12g)
menjadi makin penting karena kedua jalur akan memakai basis int8 yang
berbeda.

### 12g. Verifikasi numerik prefill vs decode (bits=2)

Mekanisme yang sudah ada: `BONSAI_DUMP_TOP2=1` mencetak top-2 logit di
batas prefill (lihat `dump_top2_prefill` di `main.mojo`). Jalur
pembandingnya adalah `BONSAI_PREFILL_PER_TOKEN=1` yang memaksa prefill
loop decode per-token (benar, lambat). Aturan yang sama dengan jalur
1-bit di `deploy_on_kaggle.sh`: token top-1 sama + selisih logit kecil =
cocok; token flip dengan kedua gap < 0,05 = wajar (urutan akumulasi fp16
WMMA vs GEMV berbeda); gap besar = indikasi bug layout.

Sebelum audit ini, perbandingan tersebut **belum pernah dijalankan untuk
bits=2** — kebenaran jalur prefill batched hanya disimpulkan dari
koherensi generasi (10b) dan waktu. Bagian 6 skrip `run_infer_2bit.py`
sekarang menjalankan A/B tersebut secara otomatis dan mencetak VERDIK.

**Hasil yang terukur (Kaggle T4, kernel `okiabrian/bonsai-2bit-infer`,
prompt 129 token, 24 token generasi, greedy):**

```
>> 6c. VERIFIKASI prefill batched vs per-token (bits=2)
>>    [A batched ] TOP2: ('248068', '30.8125', '248046', '14.5625', '16.25')
>>    [B per-token] TOP2: ('248068', '30.8125', '9175',   '14.5625', '16.25')
>>    token top-1: A=248068 B=248068 -> SAMA
>>    selisih |logit top-1| = 0.000000
>>    gap A = 16.250000 | gap B = 16.250000
>>    [OK] seluruh 24 token generasi IDENTIK
>>    VERDIK: LULUS — prefill batched setara numerik dgn per-token
```

| Kanal | A (prefill batched, qmm WMMA v2) | B (prefill per-token) |
|---|---|---|
| Prefill | 1487,77 ms → **86,71 tok/s** | 8873,69 ms → 14,54 tok/s |
| Decode | 73,86 ms/token (13,54 tok/s) | 73,02 ms/token (13,69 tok/s) |

Poin-poin yang penting dari angka di atas:

- **Token top-1 identik (248068) DAN nilai logitnya identik (30,8125),
  bukan sekadar id yang sama.** `selisih |logit top-1| = 0.000000` dan
  `gap = 16.25` di kedua sisi. Ini bukti numerik, bukan hanya kebetulan
  argmax yang sama.
- **24 token generasi identik byte-demi-byte** di kedua jalur.
- **Token peringkat-2 berbeda** (`248046` vs `9175`) tetapi **nilainya
  sama persis (14,5625)** di kedua run. Ini berarti kedua logit itu
  benar-benar imbang di posisi kedua (near-tie) — salah satunya menang
  hanya karena urutan akumulasi berbeda antara WMMA ubin batched dan
  GEMV per-token. Karena gap-nya 16,25 (jauh di atas ambang 0,05), ini
  **bukan** indikasi bug layout; hanya pemecahan seri yang arbitrer.
  `248046` = `<|im_end|>` dan `9175` keduanya tidak terpilih, jadi
  dampaknya nol pada generasi.
- **Decode praktis tidak berubah** (73,86 vs 73,02 ms/token) — sesuai
  harapan, karena env hanya mengubah prefill, bukan kernel decode.

**Kesimpulan 12g:** jalur prefill batched 2-bit terbukti setara secara
numerik dengan jalur per-token. Kecocokan berlaku untuk fp16 WMMA
(sekarang); bila int8 MMA (§12f) diadopsi, uji ini harus diulang karena
kedua jalur akan memakai basis int8 yang berbeda dan galat kuantisasi
aktivasi masuk.

---

### 12h. INT8 MMA (W2A8) — diadopsi, terukur, LULUS

Pengungkit terbesar dari audit §12b diwujudkan: Turing (T4) menjalankan
`mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32`, yakni **130 TOPS INT8
vs 65 TFLOPS FP16**. Implementasinya **aditif dan opt-in**:

- Kernel baru `bonsai::sm75::imma_b2::qmm_sm75_b2_int8_kernel` + launcher
  `launch_qmm_sm75_b2_prefill_int8`, dengan konfigurasi ubin identik
  (BM=BN=BK=64, 128 thread, 4 warp disusun 2×2).
- Diaktifkan lewat env `BONSAI_PREFILL_INT8=1`. **Default tetap fp16**, dan
  jalur fp16 WMMA tidak disentuh sama sekali. Bila simbol tidak ada di
  `.so` atau launcher mengembalikan bukan nol, eksekusi jatuh ke fp16,
  lalu ke loop per-token — jadi 2-bit tidak pernah kehilangan fallback.
- Penanda `>> [INT8] jalur prefill int8 (W2A8) aktif, N=... K=...` dicetak
  sekali, supaya "lulus" tidak bisa terjadi karena jalur int8 diam-diam
  tidak pernah dipakai.

**Skema W2A8.** Bobot 2-bit didekuantisasi ke int8 {-1,0,+1} lewat kodek
`__byte_perm`, aktivasi dikuantisasi **per baris per chunk-64** secara
simetris (`inv = 127/maxabs`), lalu MMA mengalikan keduanya. Skala bobot
`sw` dan skala aktivasi `sx = maxabs/127` diterapkan **sekali** setelah
akumulasi. Perhatikan `sx` adalah *kebalikan* dari `inv` — menyimpan `inv`
dan mengalikannya ke akumulator adalah kesalahan yang pernah terjadi di
sini (galatnya `(127/maxabs)²`).

**Pemetaan register** (dikonfirmasi dari `mma.cuh:239–271` referensi,
`tile<8,4,int>` untuk A dan `tile<16,8,int>` untuk D): A `.row` → baris
`lane/4`, kolom `4*(lane%4)`; B `.col` → kolom `lane/4`, baris `4*(lane%4)`;
D → `d[0]=D[g][2c]`, `d[1]=D[g][2c+1]`, `d[2]=D[8+g][2c]`, `d[3]=D[8+g][2c+1]`.
Karena A = bobot dan B = aktivasi, SMEM disusun `[n][k]` dan `[m][k]`
sehingga kedua operan dibaca dengan `int*` 4-byte sejajar — tanpa
`ldmatrix`, tanpa transpose, tanpa konflik bank.

**Dua bug yang tertangkap sebelum rilis** (keduanya karena tidak ada nvcc
lokal, jadi logika murni diuji host-side):

1. **Pengepakan aktivasi kehilangan separuh nilai.** Satu thread memegang
   **32** nilai, tetapi kode memakai `unsigned packed[4]` (= 16 byte),
   memadatkan **8** nilai ke satu `unsigned` 4 byte (geseran `e*8` untuk
   `e>=4` adalah perilaku tak terdefinisi), dan hanya satu simpanan
   `uint4`. Kolom `c0+16..c0+31` tidak pernah tertulis. Diperbaiki jadi
   `packed[8]` dengan indeks byte `j = g*8+e` → `packed[j/4]`, posisi
   `(j%4)*8`, dan dua simpanan `uint4`.
   *Gejala yang menjebak:* kernel tampak **lebih cepat** (114,58 tok/s)
   justru karena separuh simpanan dilewati, sementara logit runtuh
   (gap top-2 menyusut 19×, token generasi menjadi sampah berulang).
   Angka 114,58 tok/s itu **palsu** — jangan dipakai sebagai acuan.
2. Versi awal tabel kodek memetakan `q=3 → 0`, sedangkan kernel fp16
   (yang sudah terbukti bit-exact) memetakan `q=3 → +2` sesuai rumus
   `(q-1)`. `q=3` memang tidak pernah muncul, tetapi tabel tetap
   disamakan menjadi `0x020100FF` agar identik dengan sumber yang proven.

**Hasil terukur** (Kaggle T4, `okiabrian/bonsai-2bit-infer` v15, bobot
dari dataset, prompt 129 token, 24 token generasi, greedy):

```
>>    [C int8    ] TOP2: ('248068','30.734375','248046','14.5546875','16.1796875')
>>    token top-1: C=248068 B=248068 -> SAMA
>>    selisih |logit top-1| C vs B = 0.078125
>>    [OK] seluruh 24 token generasi IDENTIK dengan referensi
>>    VERDIK int8: LULUS
>>    throughput prefill: WMMA fp16 = 89.09 tok/s | int8 = 101.58 tok/s | rasio = 1.140x
```

| Kanal | A (fp16, proven) | B (per-token) | C (int8, baru) |
|---|---|---|---|
| Prefill | 1447,95 ms → 89,09 tok/s | 8721,28 ms → 14,79 tok/s | 1269,96 ms → **101,58 tok/s** |
| Decode | 72,38 ms/token | 70,92 ms/token | 70,79 ms/token |

Selisih logit int8 terhadap referensi hanya **0,078 pada logit 30,81
(≈0,25%)**, gap 16,180 vs 16,250, dan **24/24 token identik** — jauh di
dalam ambang kelonggaran (25% gap) dan cukup kecil untuk membuktikan
keputusan argmax tidak koin-balik.

**Mengapa "hanya" 1,14×, bukan 2×.** Penggandaan throughput tensor core
hanya berlaku pada MMA itu sendiri. Dari analisis FLOP, plafon tensor
core untuk bentuk ini ≈ 27,8 µs/lapisan, sedangkan yang teramati
~0,13 ms/lapisan — jadi **~98% waktu prefill habis di kuantisasi,
dekuantisasi, dan penskalaan pada CUDA core**, bukan di MMA. Itulah
alasan batas atas realistisnya ~1,4–1,8× (dan 1,14× yang tercapai masih
di bawahnya karena jalur ini juga menambah kerja kuantisasi aktivasi).
Peningkatan berikutnya harus menyasar kuantisasi/dekuantisasi, bukan MMA.

**Kesimpulan 12h:** int8 MMA diadopsi sebagai jalur opt-in, lolos
verifikasi 6d, dan membawa prefill melampaui target 100 tok/s
(89,09 → 101,58 tok/s) tanpa menyentuh jalur fp16 yang sudah proven.

---

## 13. Dua perbaikan penunjang

### 13a. Bug lama: penutup NUL path shard tidak pernah ditulis

`SafeTensorsIndex._add_shard_path()` (`src/safetensors.mojo`) mengalokasikan
`n_shard_bytes + n + 1` byte dengan komentar "+1: byte nul terminator
(wajib)" — **tetapi byte itu tidak pernah ditulis**. `read_raw()` membangun
path lewat `String(unsafe_from_utf8_ptr=self.shard_paths + off)` yang
berhenti di byte nol, sehingga tanpa penutup path terbaca melewati batas
dan menyambung isi memori tetangga.

Bug ini tertutup selama ini karena dengan path pendek
(`/kaggle/working/bonsai2/model.safetensors`) byte tersebut kebetulan nol
(`alloc` tidak menjamin memori nol — ia hanya kebetulan berasal dari
halaman segar). Begitu bobot dipindah ke mount dataset yang path-nya lebih
panjang, alokasi jatuh di kelas ukuran lain dan path terbaca sebagai:

```
file path '/kaggle/input/bonsai-2bit-weights/model.safetensorsjson' not found for read
```

Perbaikan: **satu baris**, `fresh[self.n_shard_bytes + n] = UInt8(0)`,
ditulis *sebelum* `n_shard_bytes += n`.

### 13b. Bobot 2-bit menjadi dataset Kaggle

Setiap uji sebelumnya mengunduh 8,6 GB dari HuggingFace di dalam container
(~180 s per run). Kini bobot berada di dataset
`okiabrian/bonsai-2bit-weights` dan cukup di-mount dari `/kaggle/input`.

Pembuatannya dikerjakan **di dalam container Kaggle**
(`kaggle_2bit_weights/make_weights_dataset.py`), bukan di komputer lokal:
unduh dari HuggingFace pakai bandwidth Kaggle, lalu
`kagglehub.dataset_upload()` unggah Kaggle→Kaggle. Ini menghindari
menaruh 8,6 GB di komputer lokal (disk internal hanya sisa 5,8 Gi).
Bila unggahan gagal, skrip **sengaja menahan** salinan di output kernel
supaya masih bisa dipakai lewat `kernel_sources`.

`run_infer_2bit.py` kini mencari `model.safetensors` di bawah
`/kaggle/input` dan memakainya bila ada; kalau tidak ada, mundur ke
unduhan HuggingFace seperti sebelumnya. Pembersihan akhir dilewati untuk
mount dataset (baca-saja).
