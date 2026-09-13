# Arsitektur Model Qwen 3.5 / 3.6 / 3.8 Hybrid di Mojo (T4 1-Bit W1A16 g128)

Dokumen ini ditujukan sebagai **panduan teknis komprehensif bagi AI Agent, LLM, dan Pengembang** untuk memahami struktur internal, persamaan matematika, kontrak tensor, dan alur eksekusi dari paket modul `src/models/qwen3_5/`.

> **⚠️ PERINGATAN DIMENSI — BACA DULU.**
> Dokumen ini ditulis untuk konfigurasi generik lama dan **angka-angkanya masih
> memakai konfigurasi itu** (hidden 4096, intermediate 11008, 32 query head,
> 8 KV head, GDN 64 V-head, rotary_dim 32, vocab 152064). Model yang benar-benar
> dijalankan adalah **Bonsai-27B-mlx-1bit** dengan dimensi di
> `config.mojo::qwen_27b_default()` — **angka itulah yang otoritatif**:
>
> | Parameter | Nilai Bonsai-27B (otoritatif) | Angka lama di dokumen ini |
> | :--- | :--- | :--- |
> | `hidden_size` | **5120** | 4096 |
> | `intermediate_size` | **17408** | 11008 |
> | `num_attention_heads` (H_q) | **24** | 32 |
> | `num_key_value_heads` (H_kv) | **4** | 8 |
> | `head_dim` | **256** | 128 |
> | `rotary_dim` (partial 0.25) | **64** dari 256 | 32 dari 128 |
> | `vocab_size` | **248320** | 152064 |
> | `rope_theta` | **1e7** | 100000 |
> | GDN H_v / H_k / D_v=D_k | **48 / 16 / 128** | 64 / 16 / 128 |
> | `gdn_conv_dim` | **10240** | 12288 |
> | `gate_up_proj` | **[34816, 5120]** | [22016, 4096] |
> | `down_proj` | **[5120, 17408]** | [4096, 11008] |
>
> GQA `group_size` = 24 // 4 = **6** (bukan 4 atau 5). Setiap angka pada tabel
> ini terverifikasi thd `config.json` checkpoint dan dipakai oleh jalur GPU
> produksi; angka di badan dokumen hanya relevan sebagai latar historis.

---

## 1. Ringkasan Arsitektur

Paket ini mengimplementasikan model bahasa **Qwen 3.5 / 3.6 / 3.8 Hybrid (27B & Base Variants)** secara penuh tanpa penyederhanaan di atas kernel komputasi kuantisasi biner 1-bit affine ($b = -s$, group size 128) NVIDIA T4 di Mojo.

### Karakteristik Utama:
* **Hybrid Interleaving 3:1**: 
  - **75% Layer (Lapisan $0, 1, 2 \pmod 4$)**: **Gated DeltaNet (GDN)** — Linear Attention rekuren berorde linier $O(n)$ terhadap panjang konteks.
  - **25% Layer (Lapisan $3 \pmod 4$)**: **Gated Full Attention** — Multi-Head/Grouped Query Attention (GQA) untuk penarikan asosiatif memori panjang (*exact recall*).
* **SwiGLU FFN**: Di seluruh lapisan menggunakan proyeksi fused `gate_up_proj` $[N=22016, K=4096]$ dan `down_proj` $[N=4096, K=11008]$.
* **Akselerasi Biner T4**: Seluruh matriks proyeksi linear besar dihitung menggunakan kernel biner hardware NVIDIA T4 (`quantized_matmul_1bit`).

---

## 2. Peta File & Tanggung Jawab Modul

Direktori ini dipisahkan secara modular ke dalam 10 file dengan tanggung jawab tunggal:

| Nama File | Komponen / Simbol Utama | Deskripsi & Persamaan Matematika |
| :--- | :--- | :--- |
| [`config.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/config.mojo) | `QwenConfig` | Hyperparameter arsitektur model (Hidden: 4096, Intermediate: 11008, 64 Layer, 32 Query Heads, 8 KV Heads, GDN: 64 V-heads, 16 K-heads). |
| [`norm.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/norm.mojo) | `silu`, `sigmoid`, `softplus`, `rms_norm`, `head_rms_norm`, `softmax` | Operator aktivasi dan normalisasi numerik stabil. Termasuk per-head RMSNorm untuk $Q$-Norm & $K$-Norm. |
| [`rope.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/rope.mojo) | `apply_partial_rope` | Rotary Positional Embedding parsial: hanya 25% dimensi per head (32 dim) yang dirotasi, 96 dim sisanya dibiarkan linear. |
| [`linear.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/linear.mojo) | `QwenLinear1Bit` | Wrapper layer proyeksi matriks dengan bobot biner W1A16 g128 yang memanggil kernel T4 `quantized_matmul_1bit`. |
| [`conv.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/conv.mojo) | `CausalConv1dState` | Depthwise Causal 1D Convolution dengan sliding window buffer 3 token sebelumnya (`kernel_size = 4`) + aktivasi SiLU. |
| [`gated_delta.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/gated_delta.mojo) | `GatedDeltaNetState`, `qwen3_5_gdn_step` | Rekurensi state matrix $S \in \mathbb{R}^{64 \times 128 \times 128}$ (4 MiB/layer) via Delta Rule: $S_t = S_{t-1} g + k \delta^T$, diikuti RMSNormGated. |
| [`attention.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/attention.mojo) | `AttentionKVCache`, `qwen3_5_gated_attention_step` | Gated Full Attention dengan GQA (32:8), KV-Cache ring buffer, Softmax, dan output gating: $\text{out} = (\text{Softmax} \cdot V) \odot \text{Sigmoid}(\text{gate})$. |
| [`mlp.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/mlp.mojo) | `qwen3_5_swiglu_mlp_step` | SwiGLU MLP: $\text{out} = \text{down\_proj}(\text{SiLU}(\text{gate}) \odot \text{up})$. |
| [`layer.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/layer.mojo) | `QwenDecoderLayer` | Forward pass 1 blok decoder utuh: Hybrid router (GDN vs Attention) + Dual Residual Stream + Dual RMSNorm. |
| [`model.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/model.mojo) | `embed_tokens_step`, `qwen3_5_model_forward`, `lm_head_argmax_step` | Transformasi token embedding lookup, normalisasi final RMSNorm, dan seleksi greedy logits token generasi (Argmax). |
| [`__init__.mojo`](file:///Users/macmini/.mounty/SSD_External/fix/bonsai-1bit-t4-mojo/src/models/qwen3_5/__init__.mojo) | - | Re-ekspor seluruh API publik modul untuk integrasi mulus ke `src/__init__.mojo`. |

---

## 3. Diagram Alur Komputasi (Dataflow Pipeline)

Setiap token masukan $x_t \in \mathbb{R}^{4096}$ diproses melalui pipeline berulang:

```
Masukan Token ID
       │
       ▼
[embed_tokens_step] (Tabel Embedding)
       │
       ▼  x_t [dim: 4096]
┌──────────────────────────────────────────────────────────────┐
│ Lapisan i (QwenDecoderLayer)                                 │
│                                                              │
│  x_norm = rms_norm(x_t)                                      │
│                                                              │
│  ┌─────────────────────────┬──────────────────────────────┐  │
│  │ Jika (i + 1) % 4 != 0:  │ Jika (i + 1) % 4 == 0:       │  │
│  │ [Gated DeltaNet (GDN)]  │ [Gated Full Attention (GQA)] │  │
│  │ - in_proj_all (1-bit)   │ - q_proj, k_proj, v_proj     │  │
│  │ - Causal Conv1D (k=4)   │ - Q-Norm & K-Norm per-head   │  │
│  │ - Head RMSNorm (Q, K)   │ - RoPE Parsial (25% dim)     │  │
│  │ - Delta Recurrence:     │ - KV Cache Ring Buffer       │  │
│  │   kv_mem = S * k        │ - Softmax(Q * K^T / sqrt(D)) │  │
│  │   delta  = (v-kv_mem)*b │ - Context * Sigmoid(gate)    │  │
│  │   S = S * g + k * delta │ - o_proj (1-bit)             │  │
│  │ - RMSNormGated(S*q, z)  │                              │  │
│  │ - out_proj (1-bit)      │                              │  │
│  └─────────────────────────┴──────────────────────────────┘  │
│                           │                                  │
│                           ▼                                  │
│              x = x + sublayer_output (Residual 1)            │
│                           │                                  │
│              x_norm = rms_norm(x)                            │
│                           │                                  │
│              [SwiGLU Feed-Forward Network]                   │
│              - gate_up_proj (1-bit, N=22016, K=4096)         │
│              - act = silu(gate) * up                         │
│              - down_proj (1-bit, N=4096, K=11008)            │
│                           │                                  │
│              x = x + mlp_output (Residual 2)                 │
└──────────────────────────────────────────────────────────────┘
       │  (Diupload melalui 64 Lapisan)
       ▼
[qwen3_5_model_forward] (Final RMSNorm)
       │
       ▼
[lm_head_argmax_step] (Proyeksi Kosakata & Seleksi Token)
       │
       ▼
Token Berikutnya
```

---

## 4. Spesifikasi Matematis Presisi

### A. Causal Conv1D Gated DeltaNet
Konvolusi kedalaman 1D (*depthwise*) dijalankan secara autoregresif:
$$\text{conv\_out}[c] = \text{SiLU}\left(\sum_{k=0}^{3} w[c, k] \cdot \text{buffer}[k, c]\right)$$
di mana $\text{buffer}$ adalah riwayat kausal 3 token sebelumnya.

### B. Normalisasi Per-Head ($Q$-Norm & $K$-Norm)
Sebelum perkalian state atau atensi, Query dan Key dinormalisasi per head:
$$Q_h = \frac{1}{D_k} \cdot \frac{Q_h}{\sqrt{\frac{1}{D_k}\sum Q_{h,d}^2 + \epsilon}}, \quad K_h = \frac{1}{\sqrt{D_k}} \cdot \frac{K_h}{\sqrt{\frac{1}{D_k}\sum K_{h,d}^2 + \epsilon}}$$

### C. RoPE Parsial (25% Rotary Factor)
Hanya $d \in [0, 32)$ yang dirotasikan, sementara $d \in [32, 128)$ dibiarkan linear:
$$\theta_i = \text{rope\_theta}^{-2i / 32}$$
$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

### D. Rekurensi State Delta Rule
$$\text{kv\_mem} = S_{t-1} \cdot k_t$$
$$\delta_t = (v_t - \text{kv\_mem}) \cdot \text{sigmoid}(b_t)$$
$$S_t = S_{t-1} \cdot \exp(-\exp(0.5) \cdot \text{softplus}(a_t + 1.0)) + k_t \cdot \delta_t^T$$
$$\text{readout} = S_t \cdot q_t$$
$$\text{output} = \text{RMSNorm}(\text{readout}) \odot \text{SiLU}(z_t)$$

---

## 5. Cara Penggunaan untuk AI / Pengembang

Contoh instansiasi dan forward pass:

```mojo
from models.qwen3_5 import (
    QwenConfig, GatedDeltaNetState, AttentionKVCache, QwenDecoderLayer
)

fn run_layer_step():
    let config = QwenConfig.qwen_27b_default()
    
    # Alokasi state rekuren GDN dan KV-Cache
    var gdn_state = GatedDeltaNetState(
        conv_dim=config.gdn_conv_dim,
        num_v_heads=config.gdn_num_v_heads,
        head_v_dim=config.gdn_head_v_dim,
        head_k_dim=config.gdn_head_k_dim
    )
    var kv_cache = AttentionKVCache(max_seq_len=4096, num_kv_heads=8, head_dim=128)
    
    # Bebaskan memori setelah generasi selesai
    gdn_state.free()
    kv_cache.free()
```
