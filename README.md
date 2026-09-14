# Bonsai-27B 1-Bit Inference for NVIDIA T4

Inference engine for **Bonsai-27B** (binary 1-bit, W1A16, `group_size 128`, Affine Bonsai) on an **NVIDIA T4** (Turing, `sm_75`), written in **Mojo** with hand-tuned `sm_75` CUDA kernels through FFI.

## Highlights

- **Prefill 120.4 tok/s · decode 18.2 tok/s** on a single Kaggle T4 (measured, greedy)
- Bitwise-deterministic decode — identical token streams, no `atomicAdd`
- Hybrid design: 48 Gated DeltaNet layers + 16 full-attention layers
- GPU required — no CPU fallback

## What's inside

- **1-bit matmul kernels** — GEMV decode (`M ≤ 8`), WMMA prefill (`M > 8`)
- **Elementwise / GDN kernel set** for the hybrid Bonsai-27B architecture
- **Kaggle pipeline** — build, test, and measure end to end on T4

---

## Results (Kaggle, Tesla T4)

### Throughput

| Phase | Speed | Latency |
|---|---|---|
| Prefill (81 tokens) | **120.4 tok/s** | 8.30 ms/token |
| Decode (511 tokens) | **18.2 tok/s** | 54.95 ms/token |

*Greedy decoding, from a T4 inference log.*

### About the `[PROF/SPLIT]` line

> - It does **not** split GPU time per subsystem in the default configuration.
> - In `main.mojo`, `acc_gdn +=` and `acc_attn +=` sit outside `if prof:` — only their `synchronize()` is inside. With `BONSAI_PROFILE` off (deliberately: syncing 65×/token kills the async pipeline), they measure **CPU submit time**, while `acc_lm` is the GPU drain of **all** tokens at the last sync.
> - So `LM_HEAD+argmax` (~47 ms) is **one full token forward**, not the LM head — otherwise 64 layers would read the 4.9 GB weight set in ~3 ms ≈ 1500 GB/s, far above the T4's 320 GB/s peak.
> - The real picture: decode is bandwidth-bound — ~4.9 GB of weights per token in ~47 ms ≈ **104 GB/s** (~33% of peak). For a genuine per-subsystem breakdown, run with `BONSAI_PROFILE=1`.

---

## Architecture (Qwen 3.5 / 3.8 hybrid)

| Parameter | Value |
|---|---|
| Layers | 64 — 48 *Gated DeltaNet* (linear, stateless, no KV cache) + 16 *full attention* |
| Attention layers | `li ≡ 3 (mod 4)` → 3, 7, 11, …, 63 |
| Hidden / vocab | 5120 / 248320 |
| Heads | 24 query · 4 KV (GQA) · head_dim 256 |
| RoPE | rotary_dim 64 · theta 1e7 |

---

## Usage

**1 — Build the CUDA library** (requires `nvcc`):

```bash
bash scripts/build_cuda_ffi.sh
export BONSAI_CUDA_LIB=$PWD/build/libbonsai_qmv_sm75.so
```

**2 — Build and run inference:**

```bash
pixi run mojo build -I . main.mojo -o bonsai_infer
BONSAI_USE_GPU=1 ./bonsai_infer --model-dir <model_dir> \
    --prompt-tokens 248045,846,198 --max-tokens 24 --gpu
```

Options: `--model-dir` · `--prompt-tokens` (comma-separated ids) · `--max-tokens` · `--gpu`

**3 — Environment variables:**

| Variable | Purpose |
|---|---|
| `BONSAI_CUDA_LIB` | Path to `libbonsai_qmv_sm75.so` |
| `BONSAI_USE_GPU` | **Required** — without it the program stops (no CPU fallback) |
| `BONSAI_DUMP_TOP2` | Print top-2 logits (numeric comparison) |
| `BONSAI_PROFILE` | Sync per stage (accurate profile, ~6% slower) |
| `BONSAI_NO_FUSE` | Disable fusion (A/B comparison) |
| `BONSAI_PREFILL_PER_TOKEN` | Force per-token prefill (comparison) |
| `BONSAI_DISABLE_CUDA_FFI` | Disable the CUDA FFI path |

**4 — Automated deploy & test on Kaggle:**

```bash
bash push_to_kaggle.sh
```

Pipeline order in `deploy_on_kaggle.sh`: compile CUDA → FFI smoke test → kernel tests → build `main.mojo` → build wheel → inference on T4 → coherence gate → per-phase profile.

---

## 1-bit matmul kernels

**Decode path (`M = 1`, batch `M ≤ 8`)**

- Vectorized memory coalescing — weight loads up to 128-bit (`uint4` = one full `g128` group per lane)
- LDS bank-conflict elimination — shared memory padded `128 → 132` floats (`132 % 32 == 4`), 8 group columns hit 8 different banks
- Register-tiled accumulators — weights read **once** from DRAM per token
- Deterministic warp reduction — no `atomicAdd`, 100% bitwise reproducible

**Prefill path (`M > 8`)**

- Two tiling variants — `prefill_sm75` (BM=64, BN=32, BK=64) and `prefill_wmma` (BM=64, BN=64, BK=64, WMMA)
- Bank-conflict elimination — `PAD = 8` on the shared-memory matrices
- Branchless dequantization — `w_eff = s · (2·bit − 1)` via the IEEE FP16 sign trick (`0xBC00 ^ (bit << 15)`)
- Uniform barrier — all CTA threads call the barrier together even for out-of-bounds cells, preventing warp deadlock on ragged boundaries

---

## Quantization contract

| Item | Value |
|---|---|
| Weights | `uint8`, shape `[N, K/8]`, 8 weights per byte, **LSB-first**: `bit_i = (byte >> (k mod 8)) & 1` |
| Scales | `Float16` / `Float32`, shape `[N, (K+127)/128]`, `group_size = 128` |
| Affine Bonsai | `w_eff = (2·bit − 1) · s` → `+s` for bit `1`, `−s` for bit `0`; the `−s` bias is absorbed into the multiply — no separate bias allocation, no extra FMA |

---

## Repository structure

```text
bonsai-1bit-t4-mojo/
├── main.mojo                     # Native inference CLI
├── deploy_on_kaggle.sh           # Build + test + inference on Kaggle
├── push_to_kaggle.sh             # Push dataset & kernel, then download artifacts
├── src/
│   ├── common.mojo               # Architecture constants, tile geometry, strides
│   ├── dequant.mojo              # LSB-first bit extraction & branchless sign-flip
│   ├── ops.mojo                  # Host dispatcher + CUDA FFI
│   ├── csrc/qmv_sm75_kernel.cu   # sm_75 CUDA kernels (matmul, elementwise, GDN)
│   ├── kernels/                  # Mojo kernels: decode, prefill, elementwise, direct_smallm
│   ├── models/qwen3_5/           # Layer, attention, GDN, MLP, RoPE, norm
│   └── safetensors.mojo          # Safetensors loader + jsonlite
├── tests/                        # Kernel selftests, FFI, rope, argmax, architecture
├── benchmarks/                   # T4 micro-benchmarks + layer profiles
└── scripts/build_cuda_ffi.sh     # Compile CUDA kernels -> .so
```

---

## Running the tests

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

## Known limitations

Honest list of what is **not yet** on par with the reference implementation (Python/MLX):

- **Runtime state** — per-layer state is not persisted across processes.
- **Profiling** — `BONSAI_PROFILE=1` (per-stage GDN/ATTN) is off by default: syncing 65×/token kills the async pipeline.

---

## License

Apache-2.0
