# Bonsai-27B 1-Bit Inference for NVIDIA T4

Inference for **Bonsai-27B** (binary 1-bit quantization, W1A16, `group_size 128`, Affine Bonsai) on an **NVIDIA T4** GPU (Turing, `sm_75`), written in **Mojo** with `sm_75` CUDA kernels via FFI.

This repository contains:

1. **1-bit matmul & decode kernels** — GEMV decode (`M ≤ 8`), WMMA prefill (`M > 8`), plus the elementwise/GDN kernel set for the hybrid Bonsai-27B architecture.

---

## Measured Results (Kaggle, Tesla T4)

**Throughput** (Bonsai-27B 1-bit, greedy, from a T4 inference log):

| Phase | Speed |
|---|---|
| Prefill (81 tokens) | **120.4 tokens/s** (8.30 ms/token) |
| Decode (511 tokens) | **18.2 tokens/s** (54.95 ms/token) |

**A note on the per-subsystem profile.** The `[PROF/SPLIT]` line does **not** break GPU time down per subsystem under the default configuration. In `main.mojo`, `acc_gdn +=` and `acc_attn +=` sit **outside** `if prof:` — only their `synchronize()` is inside. Since `BONSAI_PROFILE` is off in `deploy_on_kaggle.sh` (deliberately: syncing 65×/token kills the async pipeline), `prof` is false, so `acc_gdn`/`acc_attn` are only *CPU submit* times, and `acc_lm` is the GPU time of **all tokens** drained at the last sync.

As a consequence, the `LM_HEAD+argmax` figure on that line (~47 ms) is **not** the LM-head cost — it is the full forward time of one token. If it really were the LM head, 64 layers would have to read the 4.9 GB weight set in ~3 ms ≈ 1500 GB/s, far above the T4 peak bandwidth (320 GB/s).

What can be concluded: decode is genuinely bandwidth-bound, reading ~4.9 GB of weights per token in ~47 ms ≈ **104 GB/s** (~33% of the T4 peak). A real per-subsystem breakdown is only available with `BONSAI_PROFILE=1` in a separate profiling session.

---

## Bonsai-27B Architecture (Qwen 3.5 / 3.8 Hybrid)

- **64 layers**: 48 *Gated DeltaNet* (linear, stateless, **no** KV cache) + 16 *full attention*.
- Attention layers are `li ≡ 3 (mod 4)` → **3, 7, 11, …, 63**. Non-attention layers keep no KV.
- Configuration: `hidden 5120`, `vocab 248320`, `H_q 24`, `H_kv 4` (GQA), `head_dim 256`, `rotary_dim 64`, `rope_theta 1e7`.

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

### Environment variables

| Variable | Purpose |
|---|---|
| `BONSAI_CUDA_LIB` | Path to `libbonsai_qmv_sm75.so` |
| `BONSAI_USE_GPU` | Required; without it the program stops (no CPU fallback) |
| `BONSAI_DUMP_TOP2` | Print top-2 logits (numeric comparison) |
| `BONSAI_PROFILE` | Sync per stage (accurate profile, ~6% slower) |
| `BONSAI_NO_FUSE` | Disable fusion (A/B comparison) |
| `BONSAI_PREFILL_PER_TOKEN` | Force per-token prefill (comparison) |
| `BONSAI_DISABLE_CUDA_FFI` | Disable the CUDA FFI path |

### Automated deploy & test on Kaggle

```bash
bash push_to_kaggle.sh
```

In order, `deploy_on_kaggle.sh`: compiles CUDA → FFI smoke test → kernel tests → builds `main.mojo` → builds the wheel → runs inference on the T4 → coherence gate, then the **per-phase profile**.

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

**Runtime**
- Per-layer state is currently not persisted across processes.

**Performance**
- `BONSAI_PROFILE=1` (per-stage GDN/ATTN) is off by default because syncing 65×/token kills the async pipeline.

---

## License

Apache-2.0
