# Bonsai-27B 1-Bit Inference + KHQ KV-Cache Compression for NVIDIA T4

Inference for **Bonsai-27B** (binary 1-bit quantization, W1A16, `group_size 128`, Affine Bonsai) on an **NVIDIA T4** GPU (Turing, `sm_75`), written in **Mojo** with `sm_75` CUDA kernels via FFI.

This repository contains two parts:

1. **1-bit matmul & decode kernels** — GEMV decode (`M ≤ 8`), WMMA prefill (`M > 8`), plus the elementwise/GDN kernel set for the hybrid Bonsai-27B architecture.
2. **KHQ — KV-cache compression** — full GPU-side calibration and a KV compression runtime for the attention layers, replacing the fp16 KV cache with a compressed representation. All calibration compute runs in CUDA; Mojo only handles IO/FFI (read dumps, marshal buffers, write files). **There is no CPU fallback.**

---

## Measured Results (Kaggle, Tesla T4)

**Throughput** (Bonsai-27B 1-bit, greedy, from a T4 inference log):

| Phase | Speed |
|---|---|
| Prefill (81 tokens) | **120.4 tokens/s** (8.30 ms/token) |
| Decode (511 tokens) | **18.2 tokens/s** (54.95 ms/token) |

**KHQ runtime correctness** — compared against the fp16 KV-cache baseline with the same prompt and weights:

| Metric | Result |
|---|---|
| Identical greedy tokens | **512 / 512** (identical 512-token prefix) |
| Top-1 logit gap at the prefill boundary | **0.000000** |
| Calibrated attention layers | **16** (ids 3, 7, 11, …, 63) |
| Compression events | **3 per layer** (tokens 256, 384, 512) |

**Compression cadence** (measured via `BONSAI_KHQ_DEBUG=1`):

```
token 256 -> compress 128 oldest -> boundary 128, raw window 128
token 384 -> compress 128 oldest -> boundary 256, raw window 128
token 512 -> compress 128 oldest -> boundary 384, raw window 128
```

After the first event, compression happens **every 128 tokens**; the raw window swings 128 ↔ 256 and the compressed region grows monotonically.

**Performance** — the numbers below come from **one controlled T4 run**: 56-token prompt, 512-token decode, identical binary and conditions, only `BONSAI_KHQ_SPLITS` changed (A/B from `deploy_on_kaggle.sh`):

| Configuration | ms/token | Overhead vs baseline |
|---|---|---|
| Baseline KV fp16 | 57.17 | — |
| KHQ `splits=1` (legacy path) | 66.67 | +9.50 (+16.6%) |
| KHQ `splits=4` | 62.01 | +4.84 (+8.5%) |
| KHQ `splits=8` | 59.98 | +2.81 (+4.9%) |

Split-K cuts the **KHQ overhead** from 9.50 → 2.81 ms/token (−70%), which means a **6.69 ms/token (−10.0%)** gain in decode speed. Every splits value still produced an identical 512/512 token stream with top-1 logit `|delta| = 0.000000`.

Note: the previously reported "53.2 → 75.0 ms/token (+22 ms)" was **not reproduced** in this controlled measurement; the measured KHQ overhead is +9.50 ms, not +22 ms. The table above supersedes it.

Compressed KV size drops from **1024 B → 220 B** per (token, head) — roughly **4.7×** smaller. Because attention cost grows linearly with sequence length, this memory win only pays off at long context.

**A note on the per-subsystem profile.** The `[PROF/SPLIT]` line does **not** break GPU time down per subsystem under the default configuration. In `main.mojo`, `acc_gdn +=` and `acc_attn +=` sit **outside** `if prof:` — only their `synchronize()` is inside. Since `BONSAI_PROFILE` is off in `deploy_on_kaggle.sh` (deliberately: syncing 65×/token kills the async pipeline), `prof` is false, so `acc_gdn`/`acc_attn` are only *CPU submit* times, and `acc_lm` is the GPU time of **all tokens** drained at the last sync.

As a consequence, the `LM_HEAD+argmax` figure on that line (~47 ms) is **not** the LM-head cost — it is the full forward time of one token. If it really were the LM head, 64 layers would have to read the 4.9 GB weight set in ~3 ms ≈ 1500 GB/s, far above the T4 peak bandwidth (320 GB/s).

What can be concluded: decode is genuinely bandwidth-bound, reading ~4.9 GB of weights per token in ~47 ms ≈ **104 GB/s** (~33% of the T4 peak). A real per-subsystem breakdown is only available with `BONSAI_PROFILE=1`.

---

## Bonsai-27B Architecture (Qwen 3.5 Hybrid)

- **64 layers**: 48 *Gated DeltaNet* (linear, stateless, **no** KV cache) + 16 *full attention*.
- Attention layers are `li ≡ 3 (mod 4)` → **3, 7, 11, …, 63**. Non-attention layers keep no KV.
- Configuration: `hidden 5120`, `vocab 248320`, `H_q 24`, `H_kv 4` (GQA), `head_dim 256`, `rotary_dim 64`, `rope_theta 1e7`.

---

## KHQ: KV-Cache Compression

### How it works

- A **ring buffer** of 256 slots holds raw K/V (fp16). `watermark = 256`, `chunk = 128`.
- When the raw window reaches 256, the **128 oldest tokens** are compressed; `boundary` advances by 128 so the window returns to 128. The next compression happens 128 tokens later.
- Each attention step computes **two regions** and merges them with a *log-sum-exp*:
  1. the compressed region (decompressed K/V payloads),
  2. the raw window in the ring (K already RoPE'd at its absolute positions).
- A sigmoid gate is applied after the merge.
- **Split-K** on the compressed region: the token range is split into `BONSAI_KHQ_SPLITS` parts, each part becomes its own block, and the partials (acc, `max_s`, `sum_exp`) are merged by a reduce kernel via logsumexp. This raises parallelism from `H_q × 1 warp` (24 warps) to `H_q × splits`. With `splits = 1` the kernel takes the exact legacy path, so the bit-exact contract stays intact.

### Compression scheme per (token, head)

| Component | K | V |
|---|---|---|
| 2-bit mask (16 × u32) | 64 B | — |
| Payload | 40 B (4-bit outliers + sign) | 104 B (VQ 7-bit + dim3 4-bit) |
| Norm / residual | 2 B + 2 B | 2 B + 2 B |
| Shared meta | 4 B | (shared) |

K: normalize → FWHT → 4D rotor → 2-bit mask quantization with at most 50 outliers. V: normalize → FWHT → rotor → VQ (codebook 256 × 3 per patch) + a third-dimension component.

### Calibration

The driver `src/khq/calib.mojo` takes `kv_dump.bin` (from a forward pass of the original model) and produces `khq_calib.bin`. All heavy operations run on the GPU via `launch_cal_op` (10 ops) and `launch_cal_svq_train`:

FWHT base → K-static → rotate → PCA → kmeans 4/dim → kmeans VQ → Procrustes → SmartK → SmartV → head-major, plus the **SmartVQ** tournament (seeds 42/137/271, 2 rounds of 30+70 iterations, Adam, temperature `1.5·(0.1/1.5)^(i/max)`, winner selection from the *best* weights).

---

## Usage

### Prerequisites

Compile the CUDA kernels (requires `nvcc`) into `libbonsai_qmv_sm75.so`:

```bash
bash scripts/build_cuda_ffi.sh
export BONSAI_CUDA_LIB=$PWD/build/libbonsai_qmv_sm75.so
```

### Running inference

```bash
pixi run mojo build -I . main.mojo -o bonsai_infer
BONSAI_USE_GPU=1 ./bonsai_infer --model-dir <model_dir> \
    --prompt-tokens 248045,846,198 --max-tokens 24 --gpu
```

Options: `--model-dir`, `--prompt-tokens` (comma-separated ids), `--max-tokens`, `--gpu`.

### Generating the K/V dump for calibration

```bash
BONSAI_DUMP_KV_DIR=/tmp/khq ./bonsai_infer --model-dir <model_dir> \
    --prompt-tokens <ids> --max-tokens 512 --gpu
# -> /tmp/khq/kv_dump.bin + attn_<layer>.bin
```

K/V is captured **from a forward pass of the original model** (post-norm pre-RoPE K + V + causal softmax scores). The `attn_<layer>.bin` files are mandatory and their token count must match the dump, otherwise calibration refuses to run (it never silently falls back to Parseval mode).

### Running calibration

```bash
pixi run mojo run -I . src/khq/calib.mojo /tmp/khq /tmp/khq/khq_calib.bin
```

### Enabling the compression path

```bash
BONSAI_KHQ_PATH=/tmp/khq/khq_calib.bin ./bonsai_infer --model-dir <model_dir> \
    --prompt-tokens <ids> --max-tokens 512 --gpu
```

### Environment variables

| Variable | Purpose |
|---|---|
| `BONSAI_CUDA_LIB` | Path to `libbonsai_qmv_sm75.so` |
| `BONSAI_USE_GPU` | Required; without it the program stops (no CPU fallback) |
| `BONSAI_KHQ_PATH` | Enable compressed KV (centroid file) |
| `BONSAI_DUMP_KV_DIR` | Dump the original model's K/V for calibration |
| `BONSAI_KHQ_DEBUG` | Print compression events + payload norms per event |
| `BONSAI_KHQ_SPLITS` | Number of split-K parts for compressed attention (default 8, max 16) |
| `BONSAI_KHQ_PROF` | Profile the KHQ path per phase (syncs each phase, slower overall) |
| `BONSAI_DUMP_TOP2` | Print top-2 logits (numeric comparison) |
| `BONSAI_PROFILE` | Sync per stage (accurate profile, ~6% slower) |
| `BONSAI_NO_FUSE` | Disable fusion (A/B comparison) |
| `BONSAI_PREFILL_PER_TOKEN` | Force per-token prefill (comparison) |
| `BONSAI_DISABLE_CUDA_FFI` | Disable the CUDA FFI path |

### Automated deploy & test on Kaggle

```bash
bash push_to_kaggle.sh
```

In order, `deploy_on_kaggle.sh`: compiles CUDA → FFI smoke test → kernel tests → builds `main.mojo` → builds the wheel → runs inference on the T4 → **original K/V dump → calibration → KHQ runtime test**, with automatic gates (dump verification, TOP2, cadence, payload content), then the **split-K A/B** (splits 1/4/8/16 + fp16 baseline) and the **per-phase profile**.

---

## 1-Bit Matmul Kernels

### Decode path (`M = 1` and small batches `M ≤ 8`)

- **Vectorized memory coalescing** — 32-bit weight loads up to 128-bit (`uint4` = one full `g128` group per lane).
- **LDS bank-conflict elimination** — padding `128 → 132` floats (`132 % 32 == 4`), so the 8 group columns hit 8 different banks.
- **Register-tiled accumulators** — weights are read **once** from DRAM per token.
- **Deterministic warp reduction** — no `atomicAdd`, 100% bitwise determinism.

### Prefill path (`M > 8`)

- **Two tiling variants** — `prefill_sm75` (`BM=64, BN=32, BK=64`) and `prefill_wmma` (`BM=64, BN=64, BK=64`, WMMA).
- **Bank-conflict elimination** — `PAD = 8` on the shared-memory matrices.
- **Branchless dequantization** — `w_eff = s · (2·bit − 1)` and the IEEE FP16 sign trick (`0xBC00 ^ (bit << 15)`), no per-bit branches.
- **Uniform barrier** — all CTA threads call the barrier together even for out-of-bounds matrix cells, preventing warp deadlock on ragged boundaries.

---

## Quantization Math Contract

- **Weights** `uint8`, shape `[N, K/8]`, 8 weights per byte, **LSB-first** order:
  `bit_i = (byte >> (k mod 8)) & 1`
- **Scales** `Float16`/`Float32`, shape `[N, (K + 127) / 128]`, `group_size = 128`.
- **Affine Bonsai contract** (`b = −s`):
  `w_eff = (2·bit − 1) · s` → `+s` when the bit is `1`, `−s` when the bit is `0`.
  The `−s` bias is absorbed into the multiply — no separate bias allocation, no extra FMA.

---

## Repository Structure

```text
bonsai-1bit-t4-mojo/
├── main.mojo                     # Native inference CLI
├── deploy_on_kaggle.sh           # Build + test + calibration + KHQ runtime on Kaggle
├── push_to_kaggle.sh             # Push dataset & kernel, then download artifacts
├── src/
│   ├── common.mojo               # Architecture constants, tile geometry, strides
│   ├── dequant.mojo              # LSB-first bit extraction & branchless sign-flip
│   ├── ops.mojo                  # Host dispatcher + CUDA FFI (including cal_op)
│   ├── csrc/qmv_sm75_kernel.cu   # sm_75 CUDA kernels (matmul, elementwise, KHQ, calib)
│   ├── khq/
│   │   ├── calib.mojo            # Calibration driver (dump -> khq_calib.bin)
│   │   └── runtime.mojo          # KV compression runtime (ring, compress, attention)
│   ├── kernels/                  # Mojo kernels: decode, prefill, elementwise, direct_smallm
│   ├── models/qwen3_5/           # Layer, attention, GDN, MLP, RoPE, norm, khq_dump
│   └── safetensors.mojo          # Safetensors loader + jsonlite
├── tests/                        # Kernel selftests, FFI, rope, argmax, architecture
├── benchmarks/                   # T4 micro-benchmarks + layer profiles
└── scripts/build_cuda_ffi.sh     # Compile CUDA kernels -> .so
```

---

## Running the Tests

```bash
# W1A16 kernel selftest: edge cases (tiny K, ragged N/M, broadcast, bit patterns)
pixi run mojo run tests/selftest_sm75.mojo

# Validate Bonsai-27B layer dimensions
pixi run mojo run tests/test_bonsai_shapes.mojo

# FFI & primary context smoke test
pixi run mojo run tests/test_cuda_ffi_smoketest.mojo

# Cross-check against an FP64 reference (Python + NumPy)
python3 tests/verify_differential.py

# Throughput & bandwidth benchmark
pixi run mojo run benchmarks/bench_t4.mojo
```

---

## Known Limitations

This section is honest about what is **not yet** on par with the reference implementation (Python/MLX):

**Calibration**
- The deploy pipeline calibrates on 567 tokens, while the reference uses 32,768 (`SEQ_LEN=1024 × 32 batch`). Distribution coverage is lower.
- The `d` vectors use a `layer_id`-derived seed via xorshift; the reference uses `layer_id // 4` via `np.random.RandomState`. Internally consistent (calibration ↔ runtime), but not bit-parity with the reference.
- The SmartVQ init/noise RNG uses per-thread xorshift; the reference uses MLX. Same algorithm and distribution, different random trajectory.
- Attention scores are stored fp16; the reference uses bf16.

**Runtime**
- `win_len` is not yet guarded against `KHQ_RING` (256). At `max_seq = 4096` this is safe because compression always catches up, but if `max_seq` is raised past ~4200 the raw window can exceed the ring and the kernel reads ring slots without warning.
- No state-reset API between sequences yet; `khq_init_layer` is idempotent.
- Per-layer state is currently not persisted across processes.

**Performance**
- `khq_attn_kernel` originally launched as `<<<H_q = 24, 32>>>` — only 24 blocks × 32 threads (≈1.9% T4 occupancy) with a serial token loop inside the block. Split-K (`BONSAI_KHQ_SPLITS`, default 8) is implemented to address this; **its improvement is not yet measured** on the T4 — measurement lives in the A/B section of `deploy_on_kaggle.sh`. Since this kernel is latency-bound (not throughput-bound), a device with more compute does not automatically help.
- Split-K changes the floating-point accumulation order. It is mathematically identical (logsumexp merge), but the final result can differ at rounding level. The verification runs in deploy are therefore pinned to `BONSAI_KHQ_SPLITS=1` so the bit-exact contract and payload-content gates hold; the token stream for each splits value is compared separately in the A/B.
- `BONSAI_PROFILE=1` (per-stage GDN/ATTN) is off by default because syncing 65×/token kills the async pipeline. For the KHQ path, `BONSAI_KHQ_PROF=1` measures the 7 internal KHQ phases (ring-write, compress-K/V, gather-q, compressed attention, raw-window attention, merge+gate).

---

## License

Apache-2.0
