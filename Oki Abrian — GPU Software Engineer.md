# Oki Abrian

## Software Engineer — GPU & AI Inference

✉ oki.abrian@gmail.com ⌂ Indonesia (UTC+7) ◉ github.com/oki-abrian

## ABOUT

Software engineer focused on GPU programming for AI/LLM inference. I write and optimize CUDA kernels by hand (Turing `sm_75`: GEMV decode, WMMA tensor-core prefill, shared-memory tiling, deterministic warp reductions) and drive them from higher-level runtimes — Mojo via FFI, Rust, and Apple MLX. I care about measurable performance (tokens/s, bandwidth utilization, bitwise reproducibility) and numerical correctness. Background in Rust/C systems programming (Linux kernel modules, async network services), which shapes how I profile and reason about hardware.

## TECHNICAL SKILLS

- **GPU / Acceleration:** CUDA (`sm_75`/Turing kernel authoring — GEMV, WMMA Tensor Cores, shared-memory tiling, LDS bank-conflict elimination, branchless dequantization, deterministic warp reduction), cuobjdump kernel inspection, custom GPU profiling harnesses, Apple MLX
- **Languages:** Mojo, Rust (no_std, Tokio, Axum), C, C++ (host-side/FFI), Python, TypeScript
- **AI / Inference:** 1-bit weight quantization (W1A16, group-wise scales), KV-cache compression, Gated DeltaNet + hybrid attention architectures, OpenAI-compatible inference APIs
- **Systems:** Linux kernel modules, Traffic Control (TC/Qdisc), multi-queue NIC, Docker, GitHub Actions
- **Backend / Frontend:** Axum, Actix, Node.js, PostgreSQL, Redis/Valkey; React, Three.js

## SELECTED PROJECTS

### Bonsai 27B 1-bit inference engine for NVIDIA T4 — github.com/oki-abrian/bonsai-mojo-1bit-optimization-t4-cuda
*Mojo, CUDA sm_75 (C++), Python — W1A16 quantization, Affine Bonsai*

- Wrote hand-tuned Turing CUDA kernels end to end: GEMV decode (`M ≤ 8`) with 128-bit vectorized weight loads (`uint4` = one full `g128` group per lane), register-tiled accumulators, and conflict-free shared-memory padding (`128 → 132` floats) — weights are read from DRAM exactly once per token.
- Implemented WMMA tensor-core prefill (two tiling variants) with branchless 1-bit dequantization (`w_eff = s·(2·bit − 1)` via the IEEE FP16 sign trick) and uniform barriers to prevent warp deadlock on ragged boundaries.
- Achieved bitwise-deterministic decode — no `atomicAdd`, deterministic warp reductions, identical token streams across runs.
- Measured on Kaggle T4: **120.4 tok/s prefill, 18.2 tok/s decode** for a 27B model (~4.6 GB VRAM); profiled with custom harnesses and cuobjdump, showing a bandwidth-bound decode (~104 GB/s, ~33% of T4 peak).

### KHQ — KV-cache compression for hybrid LLM inference — github.com/oki-abrian/bonsai-mojo-1bit-optimization-t4-cuda-khq
*Mojo, CUDA — GPU-side calibration + compression runtime*

- Built a full GPU calibration pipeline (FWHT, PCA, k-means, Procrustes, SmartVQ tournament with Adam) running entirely in CUDA; the host language only handles IO/FFI.
- Runtime compresses KV per (token, head) from **1024 B → 220 B (~4.7× smaller)** using 2-bit masks, 4-bit outliers, and VQ payloads; attention runs over compressed + raw regions merged via log-sum-exp.
- Added split-K attention to raise parallelism from 24 warps to `H_q × splits`, cutting measured overhead from **+9.50 → +2.81 ms/token (−70%)** while preserving bit-exact token streams.
- Verified correctness against the fp16 baseline: **512/512 identical greedy tokens, top-1 logit gap 0.000000**.

### Kudahitam MLX LLM Gateway — github.com/oki-abrian/kudahitam-mlx-llm-gateway
*Rust (Axum, Tokio, Tower), Apple MLX, Redis/Valkey*

- OpenAI-compatible LLM inference gateway for Apple Silicon; multi-crate Rust workspace separating gateway, auth, quota, inference workers, and storage.
- RAII drop-guard pattern with atomics to reconcile token quotas when SSE streams drop; HTTP security middleware.

### Adaptive Multi-Queue Cake SQM — github.com/oki-abrian/adaptive-cake-sqm
*Rust (no_std), C (Linux kernel), Python*

- Hybrid smart queue management: Cobalt AQM (CoDel + BLUE), 8-way set-associative fair queuing, DRR++ shaper spanning 64 bps–100 Gbps in fixed-point arithmetic.
- 43 unit and stress tests covering boundary calculations, u32 saturation, and min-heap invariants.

## TECHNICAL FOCUS & INTERESTS

GPU architecture and kernel performance optimization for LLM inference — memory-bound vs compute-bound analysis, quantization numerics, and deterministic GPU pipelines. Also interested in high-performance networking (congestion control, smart queue management) and memory-efficient AI systems.

Last updated: September 2026
