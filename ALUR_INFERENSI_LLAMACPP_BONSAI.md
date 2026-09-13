# Alur Inferensi llama.cpp-prism — Bonsai-27B 1-bit (Qwen3.5) dari Inferensi sampai CUDA

Disusun dari pembacaan penuh berkas rujukan `references/llama.cpp-prism/` (baca-utuh, bukan
pencarian potongan). Semua angka/klaim di bawah berasal dari berkas yang dibaca; yang berupa
turunan ditandai **[turunan]**.

Berkas rujukan utama:

| Lapis | Berkas | Isi |
|---|---|---|
| Host / loop | `src/llama-context.cpp` | `llama_decode` → `process_ubatch` → `graph_compute` |
| Model | `src/llama-model.cpp` | `build_graph`, muat `prism.hadamard.*`, peta RoPE per-arsitektur |
| Graf arsitektur | `src/models/qwen35.cpp` | graf Qwen3.5 (64 layer hibrida) |
| Graf GDN | `src/models/delta-net-base.cpp` | pembangun graf Gated-DeltaNet |
| Graf umum | `src/llama-graph.cpp` | `build_lora_mm`, `build_norm`, `build_ffn`, `build_attn_mha`, Hadamard |
| Op GGML | `ggml/src/ggml.c` | `ggml_swiglu_split` → `GGML_OP_GLU` |
| Dispatch CUDA | `ggml/src/ggml-cuda/ggml-cuda.cu` | `ggml_cuda_compute_forward` + tabel fusi |
| Kernel GDN | `ggml/src/ggml-cuda/gated_delta_net.cu` | aturan delta di CUDA |
| Kernel conv | `ggml/src/ggml-cuda/ssm-conv.cu` | short conv (k=4) + fusi bias/SiLU |
| Kernel norm | `ggml/src/ggml-cuda/norm.cu` | `rms_norm_f32`, `l2_norm_f32` |
| Kernel RoPE | `ggml/src/ggml-cuda/rope.cu` | `rope_multi` (IMROPE) |
| Kernel GLU | `ggml/src/ggml-cuda/unary.cu` | `swiglu` (gate = SiLU) |
| Kuantisasi | `ggml/src/ggml-cuda/{quantize,dequantize,vecdotq,mmvq}.cu` | Q8_1 aktivasi, dot Q1_0 |
| Format blok | `ggml/src/ggml-common.h` | `block_q1_0` |
| Arsitektur | `/tmp/qwen35_tok/config.json` | dimensi + `quantization{bits:1,group_size:128}` |

---

## 1. Rantai pemanggilan (host → device)

```
llama_decode(batch)                                  src/llama-context.cpp:1739
 └─ balloc->init(...)                                bagi batch → ubatch
 └─ memory->init_batch(...)                          alokasi slot KV / state rekuren
 └─ do { process_ubatch(ubatch, gtype, mctx, st) }    :1920
     └─ graph_params(res, ubatch, mctx, gtype)        :2562  (membawa hadamard_rotations!)
     └─ model.build_graph(gparams)                    llama-model.cpp:2902
         └─ build_arch_graph(params)                  → llm_graph_context_qwen35 (qwen35.cpp)
         └─ build_pooling / build_sampling / set_outputs
     └─ ggml_backend_sched_alloc_graph(sched, gf)
     └─ res->set_inputs(&ubatch)                      isi token/pos/embd ke tensor input
     └─ graph_compute(gf, batched)                    :2588
         └─ ggml_backend_sched_graph_compute_async(sched, gf)
             └─ [backend CUDA] ggml_cuda_compute_forward(ctx, dst)   ggml-cuda.cu:2068
                 └─ switch(dst->op) → kernel spesifik (lihat §6)
```

Poin penting:
- **Graf di-cache/reuse.** `res->can_reuse(gparams)` (`:1443`) memakai ulang graf bila topologi
  sama; hanya bila berubah graf dibangun ulang + dialokasikan.
- **Hadamard diangkut di parameter graf.** `graph_params` menyertakan
  `&model.hadamard_rotations` dan `&model.hadamard_inverses` (`:2579-2580`) — jadi transformasi
  Hadamard adalah fitur kelas satu di level model, bukan detail kernel.
- **Tidak ada jalur khusus 1-bit di host.** Host tak tahu bobotnya 1-bit; ia hanya membangun
  `GGML_OP_MUL_MAT`. Jenis bobot (Q1_0) hanya muncul saat dispatch CUDA memilih kernel.

---

## 2. Graf per-layer Qwen3.5 (`src/models/qwen35.cpp`)

64 layer, hibrida 3:1. Pola rekuren ditetapkan di `load_arch_hparams`:

```cpp
uint32_t full_attn_interval = 4;
for (uint32_t i = 0; i < hparams.n_layer_all; ++i)
    hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
```

→ layer `i` **rekuren (GDN)** bila `(i+1) % 4 != 0`; **full-attention** bila kelipatan 4.
`n_layer == 64` → `LLM_TYPE_27B`.

Badan loop layer (`graph::graph`):

```
attn_norm (RMSNorm)
 ├─ jika rekuren : build_layer_attn_linear(...)   // GDN
 └─ jika full    : build_layer_attn(...)          // GQA + RoPE + gate
(+) residual
attn_post_norm (RMSNorm)
build_layer_ffn(...)                              // SwiGLU, intermediate 17408
(+) residual
build_cvec(...)
... (64×) ...
output_norm (RMSNorm) → LM head (build_lora_mm)
```

### 2.1 Titik FIX-1 — urutan norm vs gate (`build_norm_gated`, qwen35.cpp:272-281)

```cpp
ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
// silu(gate) * normalized sebagai satu op GLU
return ggml_swiglu_split(ctx0, gate, normalized);
```

**Terkonfirmasi di tiga lapis** (bukan asumsi):

1. **Graf model** — `gate` argumen **pertama**, `normalized` kedua.
2. **Konstruksi op** — `ggml/src/ggml.c:3078` `ggml_swiglu_split(ctx,a,b)` →
   `ggml_glu_impl(ctx,a,b,GGML_GLU_OP_SWIGLU,false)` → `GGML_OP_GLU`, `src[0]=a=gate`,
   `src[1]=b=normalized` (`ggml.c:2943-2945`).
3. **Kernel CUDA** — `ggml/src/ggml-cuda/unary.cu:276`
   ```cpp
   dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);   // x=src0=gate, g=src1=normalized
   ```
   via `ggml_cuda_op_swiglu` → `ggml_cuda_op_unary_gated<op_silu>` (`unary.cu:348-350`), dan
   `op_silu(x) = x / (1.0f + expf(-x))` (`unary.cuh:96-98`).

**Kesimpulan pasti:** keluaran GDN = **`silu(gate) · rms_norm(output)`** — gate dikali
**terakhir**, dan variance RMS dihitung atas keluaran rekurensi **murni**. Ini menegaskan
perbaikan BUG-1 di sisi Mojo (`elementwise_sm75.mojo`), dan selaras dengan
`output_gate_type: "swish"` (swish ≡ SiLU) pada `config.json`.

### 2.2 Jalur full-attention (`build_layer_attn`, qwen35.cpp:283-362)

- Proyeksi Q memancarkan **Q dan gate berselang-seling**: `Qcur_full = wq(x)`, lalu `view` Q di
  offset 0 dan `view` gate di offset `n_embd_head`.
- Q dinorm (`build_norm` RMS per-head), K dari `wk` di-reshape + `k_norm`, V dari `wv`.
- `ggml_rope_multi` pada Q dan K.
- `build_attn` (GQA, flash-attn) → `cur`.
- **Gate:** `cur = ggml_mul(cur, ggml_sigmoid(gate))` → baru `wo`.
  Jadi keluaran attention dikali `sigmoid(gate)` sebelum proyeksi output.
- `build_attn` juga menerapkan rotasi k/v Hadamard (`self_k_rot`/`self_v_rot`).

### 2.3 Jalur rekuren GDN (`build_layer_attn_linear`, qwen35.cpp:364-544)

```
build_qkvz          → in_proj_qkv (flat) + z + b + a
beta  = sigmoid(beta_raw)                       (beta_raw disimpan utk jalur raw-gates)
alpha → softplus(alpha + dt_bias) * ssm_a       ATAU jalur raw (gdn_raw_*)
build_conv_state    → concat(state, transpose(qkv))
ggml_ssm_conv       → silu
views q/k/v
l2_norm bersama pada view gabungan qk_conv      (satu op menorm q DAN k)
build_recurrent_attn → aturan delta
z_2d
build_norm_gated(output, ssm_norm, z_2d, il)    ← FIX-1 di sini
ssm_out
```

Detail yang mudah salah:
- **q dan k dinorm bersama.** q dan k adalah dua `view` berdampingan dari keluaran conv yang
  sama; satu `ggml_l2_norm` atas view gabungan `qk_conv` menormalkan keduanya sekaligus.
- **State conv**: `build_conv_state` (`delta-net-base.cpp:454-530`) me-reshape state
  `(conv_kernel_size-1, conv_channels, n_seqs)`, concat dengan qkv ter-transpose, lalu menulis
  balik `conv_kernel_size-1` kolom terakhir (satu slot bila `n_rs_seq==0`, selain itu loop
  rollback K-slot).
- **Raw-gates** (`delta-net-base.cpp:373-428`):
  ```cpp
  const bool raw = gdn_raw_beta && gdn_raw_alpha && gdn_raw_dt_bias && gdn_raw_a;
  ggml_tensor * result = ggml_gated_delta_net(ctx0, q,k,v, raw?gdn_raw_alpha:g, raw?gdn_raw_beta:b, s, 1);
  if (raw) ggml_gated_delta_net_set_raw_gates(result, gdn_raw_dt_bias, gdn_raw_a);
  ```
  Artinya aktivasi gate bisa diserahkan ke kernel (lihat §6.3).

### 2.4 MTP / NextN (`graph_mtp`, qwen35.cpp:562-718)

Satu blok decoder dense tambahan (`mtp_num_hidden_layers: 1`). Punya `eh_proj`, `enorm`,
`hnorm`, opsional embedding/shared-head khusus. **Dimuat tetapi tidak dieksekusi di jalur utama.**

---

## 3. Dimensi (dari `config.json` asli)

| Simbol | Nilai | Catatan |
|---|---|---|
| `hidden_size` | 5120 | |
| `num_hidden_layers` | 64 | `LLM_TYPE_27B` |
| `head_dim` | 256 | |
| `num_attention_heads` | 24 | Q |
| `num_key_value_heads` | 4 | KV (GQA 6:1) |
| `intermediate_size` | 17408 | FFN |
| `linear_conv_kernel_dim` | 4 | short conv |
| `linear_key_head_dim` | 128 | GDN k-head |
| `linear_num_key_heads` | 16 | |
| `linear_num_value_heads` | 48 | |
| `linear_value_head_dim` | 128 | |
| `partial_rotary_factor` | 0.25 | → `n_rot = 64` dari 256 |
| `mrope_section` | [11,11,10] | |
| `mrope_interleaved` | true | → IMROPE |
| `rope_theta` | 10000000 | |
| `mamba_ssm_dtype` | float32 | state GDN fp32 |
| `vocab_size` | 248320 | |
| `attn_output_gate` | true | |
| `output_gate_type` | "swish" | SiLU |
| `mtp_num_hidden_layers` | 1 | |
| `quantization` | `{bits: 1, group_size: 128}` | → Q1_0 |

---

## 4. RoPE: IMROPE, half-split (BUG-7 terjawab)

`llama-model.cpp:3192` → `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE;`

→ mode `GGML_ROPE_TYPE_IMROPE` → kernel **`rope_multi`** (`rope.cu:200-291`) dengan
`is_imrope=true`.

Dua hal yang menentukan hasil:

1. **Pairing half-split (NeoX), BUKAN interleaved.** Di dalam `rope_multi`:
   ```cpp
   const float x0 = x[ix + n_offs/2 + 0];
   const float x1 = x[ix + n_offs/2 + n_dims/2];
   dst[...+0]        = x0*cos - x1*sin;
   dst[...+n_dims/2] = x0*sin + x1*cos;
   ```
   Pasangan yang diputar adalah `(j, j + n_dims/2)`, dengan `n_dims = 64` (karena
   `partial_rotary_factor = 0.25` × `head_dim 256`). Kanal 64..255 **tidak diputar**.
   Ini **selaras** dengan kernel GPU Mojo (half-split) dan **bertentangan** dengan
   `rope.mojo:36-41` (interleaved) yang sudah ditandai sebagai jalur mati/ranjau di BUG-7.

2. **Sektor IMROPE** (`rope.cu:256-265`):
   ```cpp
   if (sector % 3 == 1 && sector < 3*sections.v[1]) theta = pos[i2+ne02*1]*...;  // h
   else if (sector % 3 == 2 && ...)               theta = pos[i2+ne02*2]*...;     // w
   else if (sector % 3 == 0 && ...)               theta = pos[i2]*...;           // t
   else                                            theta = pos[i2+ne02*3]*...;    // extra
   ```
   `sector = (iw/2) % sect_dims`, `sect_dims = 11+11+10 = 32`.

   **Implikasi penting untuk Bonsai (teks murni):** untuk teks tanpa dimensi spasial,
   posisi t/h/w identik (`pos[i2] == pos[i2+ne02*1] == pos[i2+ne02*2]`), sehingga pemilihan
   sektor **degenerasi menjadi RoPE NeoX biasa** dengan `n_rot=64`. Jadi meniru IMROPE
   secara penuh tidak wajib untuk teks, **tetapi pairing half-split wajib**.

Catatan: ada kernel gabungan `rms_norm_mul_rope_f32` (`rope.cu:710-788`) untuk
RMS_NORM + MUL + ROPE (+VIEW+SET_ROWS); kernel ini hanya mendukung `is_neox` — konsisten
dengan fakta bahwa jalur 1-bit memakai half-split.

---

## 5. FFN dan head

- `build_ffn` (`llama-graph.cpp`): `LLM_FFN_SILU` + `LLM_FFN_PAR` →
  `ggml_swiglu_split(ctx0, cur, tmp)`. Argumen **pertama** (`cur`, proyeksi w1) yang
  mendapat SiLU — konsisten dengan `silu(a)·b` yang dikonfirmasi di §2.1.
- LM head dibangun lewat `build_lora_mm` → `GGML_OP_MUL_MAT` → kernel GEMV/GEMM terkuantisasi.
- **Hadamard**: `build_lora_mm` (`llama-graph.cpp:1506-1563`) dapat **melipat** transformasi
  Hadamard ke aktivasi sebelum `ggml_mul_mat` (plus `w_s` scale + LoRA). `build_inp_embd`
  memulihkan basis primal setelah `ggml_get_rows` (`h = s·(H z)`); `build_attn` menerapkan
  rotasi k/v. Ini ciri khas skema 1-bit dan **bukan** sekadar detail kosmetik.

---

## 6. Lapis CUDA

### 6.1 Dispatch (`ggml-cuda.cu:2068` `ggml_cuda_compute_forward`)

Op yang relevan untuk Qwen3.5 1-bit:

| `GGML_OP_*` | Fungsi CUDA | Peran di Qwen3.5 |
|---|---|---|
| `MUL_MAT` | `ggml_cuda_mul_mat` → mmvq/mmq | semua proyeksi (Q1_0) |
| `GLU` + `SWIGLU` | `ggml_cuda_op_swiglu` | gate GDN (FIX-1) & FFN |
| `RMS_NORM` | `ggml_cuda_op_rms_norm` | attn_norm, post_norm, ssm_norm, output_norm |
| `L2_NORM` | `ggml_cuda_op_l2_norm` | norm bersama q/k |
| `ROPE` | `ggml_cuda_op_rope` | Q/K (IMROPE) |
| `SSM_CONV` | `ggml_cuda_op_ssm_conv` | short conv k=4 |
| `GATED_DELTA_NET` | `ggml_cuda_op_gated_delta_net` | rekurensi delta |
| `SET_ROWS` | `ggml_cuda_op_set_rows` | tulis snapshot state |
| `CONCAT` | `ggml_cuda_op_concat` | concat state+qkv |
| `CPY` / `DUP` | `ggml_cuda_cpy` / `ggml_cuda_dup` | salin state |
| `FLASH_ATTN_EXT` | `ggml_cuda_flash_attn_ext` | full-attention |
| `SOFT_MAX` | `ggml_cuda_op_soft_max` | bila bukan flash-attn |

### 6.2 Tabel fusi (`ggml-cuda.cu`, `ggml_cuda_can_fuse` / `ggml_cuda_op_*_fused`)

Fusi inilah yang menjelaskan mengapa satu "op" di graf bisa menjadi satu kernel khusus:

| Pola subgraf | Kernel terpadu | Baris |
|---|---|---|
| `{UNARY(SILU), MUL}` | `ggml_cuda_op_unary_mul` (`silu(x)*g`) | :4119-4124 |
| `{ROPE, VIEW, SET_ROWS}` | `ggml_cuda_op_rope_fused` | :3504-3510 |
| `{RMS_NORM, MUL, ROPE, VIEW, SET_ROWS}` | `ggml_cuda_op_rms_norm_mul_rope_fused` | :4089-4092 |
| `{RMS_NORM, MUL, ROPE}` | idem (tanpa set_rows) | :4094-4097 |
| `{RMS_NORM, MUL, ADD}` | `ggml_cuda_op_rms_norm_fused_add` | :4099-4102 |
| `{RMS_NORM, MUL}` | `ggml_cuda_op_rms_norm_fused` | :4104-4107 |
| `{SSM_CONV, ADD, UNARY(SILU)}` | `ggml_cuda_op_ssm_conv` (fusi bias+SiLU) | :4109-4112 |
| `{SSM_CONV, UNARY(SILU)}` | idem (tanpa bias) | :4114-4117 |
| `{UNARY, SQR}` + RELU | `ggml_cuda_op_relu_sqr` | :4126-4129 |
| `{MUL, RESHAPE, MUL_MAT}` + hint Hadamard | `ggml_cuda_op_fwht_signed` | :3480-3501 |
| `MUL_MAT` + bias | `ggml_cuda_mul_mat_vec_{f,q}` | :4070-4082 |

> Perhatikan: `GGML_GLU_OP_SWIGLU` **dengan dua sumber** (kasus `ggml_swiglu_split`) masuk
> langsung ke `ggml_cuda_op_swiglu` (bukan fusi `{UNARY,MUL}`); fusi `{UNARY,MUL}` adalah
> jalur bagi graf yang menuliskan `mul(silu(a),b)` eksplisit. **Keduanya memberi
> `silu(operand-pertama) · operand-kedua`.**

### 6.3 Kernel GDN (`gated_delta_net.cu`)

Aturan delta inti (`:107-140`):

```cpp
const float g_val = G_PRECOMPUTED ? g0 : expf(g0);
// ...
float kv_col = warp_reduce_sum<warp_size>(kv_shard);
float delta_col = (v_t[col + c] - g_val * kv_col) * beta_val;
s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
// ...
if (lane == 0) attn_data[col + c] = attn_col * scale;   // scale = 1/sqrt(S_v)
```

Aktivasi raw di dalam kernel (`:93-113`):

```cpp
beta_val = 1/(1+expf(-beta_val));                                  // sigmoid
g0 = raw_a[h_idx] * ((x > 20.0f) ? x : logf(1.0f + expf(x)));      // softplus
```

- `launch_gated_delta_net` memilih spesialisasi `S_v ∈ {16,32,64,128}`; `scale = 1/√S_v`.
- `K` = jumlah slot snapshot; `state_slot_stride` mengatur penulisan snapshot.
- Jalur precompute `exp(g)` hanya untuk `cc == GGML_CUDA_CC_DGX_SPARK && S_v == 128 &&
  n_tokens >= 32` (GB10, prompt panjang).
- Layout state **terbalik**: `M[col][i] = S[i][col]`, baris `col` kontigu. Dtype state fp32
  (`mamba_ssm_dtype: float32`).

### 6.4 Kernel short conv (`ssm-conv.cu`)

- Template `d_conv`; ukuran yang didukung: **{3, 4, 5, 9, 15}**. Bonsai: **4**.
- `sumf += x[(i+j) % d_conv] * w[j]; sumf += b;` → `apply_silu ? silu(sumf) : sumf`.
- Fusi `ADD(bias)` + `SILU` (bias selalu datang bersama SiLU — `GGML_ASSERT(!fuse_bias || fuse_silu)`).
- Dua kernel: `ssm_conv_f32` (n_t ≤ 32) dan `ssm_conv_long_token_f32` (smem, split_n_t=32).

### 6.5 RMSNorm & L2 (`norm.cu`)

```cpp
const float mean  = tmp / ncols;
const float scale = rsqrtf(mean + eps);
dst[col] = scale * x[col] * mul[mul_col] + add[add_col];   // varian terfusi
```

Varian: `rms_norm_f32`, `norm_f32`, `group_norm_f32`, `l2_norm_f32`, `add_rms_norm_f32`, plus
jalur cepat GB10 `RMS128` untuk `ncols <= 128`.

---

## 7. Format 1-bit (Q1_0)

Struktur blok (`ggml-common.h`):

```c
#define QK1_0 128
typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;   // 2 + 16 = 18 B → 1.125 bpw
```

Dequant (`dequantize.cuh`):

```cpp
const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
v.x = (2*bit_0 - 1) * d;      // bit 1 → +d ; bit 0 → −d
```

GEMV (mmvq) — aktivasi di-kuantisasi dulu ke **Q8_1**:

- `quantize_q8_1` (`quantize.cu:55`): per blok 32, `amax = warp_reduce_max`, `d = amax/127`,
  `q = roundf(x/d)`; menyimpan `ds = make_half2(d, sum)`.
- `vec_dot_q1_0_q8_1` (`vecdotq.cuh`): bit disebar menjadi byte **±1** lewat
  `__byte_perm(0x01FF, 0x01FF, …)` (0xFF = −1, 0x01 = +1), lalu diakumulasi dengan
  **DP4A** (`ggml_cuda_dp4a`, int8 dot-product).
- Hasil: `d1 * d8 * sumi` — **satu skala fp16 per 128 bobot**, satu skala fp16 per 32 aktivasi.

Inilah sebabnya kernel QMV proyek (`src/csrc/qmv_sm75_kernel.cu`) harus persis meniru:
bit → ±1, satu skala per 128, akumulasi integer.

Catatan numerik penting (`quantize.cu:509-511`):

```cpp
// Preserve the operation order of the unfused RMS_NORM -> MUL graph:
// (scale * x) * weight.  Reassociating this as x * (scale * weight)
// can change the last bit and, in turn, a greedy decoding trajectory.
```

→ jangan mereasosiasi `(scale*x)*weight` menjadi `x*(scale*weight)`.

---

## 8. Hadamard (prism) — konfigurasi di GGUF

`llama-model.cpp:1236-1332` memuat `prism.hadamard.*` dan **menolak** memuat bobot terlipat
untuk arsitektur yang belum diverifikasi. `LLM_ARCH_QWEN35` termasuk yang diizinkan.

Jenis bobot yang boleh dilipat (`is_foldable_weight`):

```
attn_q, attn_k, attn_v, attn_qkv, attn_gate, attn_output,
ffn_gate, ffn_up, ffn_down, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
ffn_gate_up_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, ssm_out,
output.weight
```

Tabel yang di-inverse **setelah lookup** hanya `token_embd.weight`
(`prism.hadamard.inverse_weight_names`). Ada juga `prism.hadamard.sign_mode` /
`sign_widths` / `sign_values` (nilai wajib ±1) dan `prism.hadamard.gdn_v_grouped`.

---

## 9. Implikasi langsung untuk proyek Mojo (checklist)

1. **FIX-1 sudah benar arahnya** — pertahankan `silu(gate) · rms_norm(output)` (variance atas
   keluaran murni). Sudah dikonfirmasi di graf model, konstruksi op, dan kernel CUDA.
2. **RoPE wajib half-split** dengan `n_rot = 64`; pairing `(j, j+32)`; kanal 64..255 disalin
   apa adanya. IMROPE-sektor boleh diabaikan untuk teks (posisi t/h/w sama) — tetapi jangan
   mengubah pairing. Jalur `rope.mojo:36-41` (interleaved) harus dianggap ranjau.
3. **Q1_0**: pastikan bit 1 → `+d`, bit 0 → `−d`; satu skala fp16 per 128; akumulasi int8/DP4A.
4. **GDN**: state terbalik & fp32; `scale = 1/√S_v`; aktivasi gate (`sigmoid`, `softplus`) bisa
   di dalam kernel — cocokkan dengan referensi raw-gates.
5. **Short conv** `d_conv = 4`, dengan `silu` setelah bias.
6. **Hadamard** bukan opsional bila checkpoint membawa `prism.hadamard.*` — cek GGUF.
7. **Jangan reasosiasi** `(scale*x)*weight`.

---

## 10. Yang belum dibaca penuh (batas kejujuran)

Berkas besar yang **belum** dibaca utuh di sesi ini: `src/llama.cpp`, `src/llama-vocab.cpp`,
`src/llama-sampler.cpp`, `ggml/src/ggml-cuda/mmq.cu`, `ggml/src/ggml-cuda/cpy-utils.cuh`,
`ggml/src/ggml-cuda/fattn*.cu`, `ggml/src/ggml-cuda/set-rows.cu`, `ggml/src/ggml-cuda/getrows.cu`.
Rantai **inferensi → CUDA untuk jalur produksi Bonsai** (graf, GDN, conv, norm, RoPE, GLU,
Q1_0 GEMV, dispatch, fusi) sudah tertutup penuh.
