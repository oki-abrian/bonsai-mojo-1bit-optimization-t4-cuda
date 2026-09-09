// Prefill W1A16 g128 — Langkah 3 design-brief (v2 Optimized). Ditulis untuk
// sm_75 (T4) tapi portabel: hanya wmma 16x16x16 fp16 + bfe.u32 (universal,
// PTX ISA 2.0), tanpa cp.async/ldmatrix -> aktif di sm_75..sm_120.
//
// GEMM: y[L,M,N] = x[L,M,K] · w[N,K/8]^T ; bobot biner packed uint8,
// mode Affine dengan b=-s sehingga w_eff = s·(2q-1) (bias terserap).
//
// Optimasi v2:
//   * PTX Bit-Field Extract (`bfe.u32`) untuk dekuantisasi SIMD branchless di ALU register.
//   * SMEM Bank-Conflict Elimination (PAD=8) pada As dan Wt.
//   * Vectorized `__half2` Epilog Store dengan proteksi keselarasan alamat 4-byte.
//   * Akumulator FP32 WMMA (16x16x16) lintas iterasi-K.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <mma.h>

namespace mlx::core::cu::wmma_b1 {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;

using namespace nvcuda;

// Ekstraksi 2-bit biner dari 1 byte via instruksi hardware PTX Bit-Field Extract
__device__ inline unsigned extract_2bits(unsigned raw_byte, int pair_idx) {
  unsigned pair;
  asm("bfe.u32 %0, %1, %2, 2;" : "=r"(pair) : "r"(raw_byte), "r"(pair_idx * 2));
  return pair;
}

__device__ inline __half2 expand2(unsigned bits, __half2 s2) {
  // w_eff = s*(2q-1): basis -1 (0xBC00), bit=1 -> flip tanda menjadi +1 (0x3C00).
  unsigned lo = 0xBC00u ^ ((bits & 1u) << 15);
  unsigned hi = 0xBC00u ^ (((bits >> 1) & 1u) << 15);
  return __hmul2(__halves2half2(__ushort_as_half(lo), __ushort_as_half(hi)), s2);
}

template <typename T, int BM_VAL = 64, int BN_VAL = 64, int BK_VAL = 64, int LA = 0, int LB = 0>
__global__ void qmm_sm75_b1_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const T* __restrict__ scales,
    const T* __restrict__ /*biases*/,
    T* __restrict__ y,
    int M, int N, int K, int L,
    bool broadcast_w) {
  static_assert(sizeof(T) == 2, "fp16/bf16 saja");

  const int block_n = blockIdx.x * BN_VAL;
  const int block_m = blockIdx.y * BM_VAL;
  const int ly = blockIdx.z;

  // Wt disimpan TRANSPOSED ([k][n]) supaya matrix_b row_major
  // (elemen(k,n)=ptr[k*ldm+n]) langsung cocok tanpa shuffle.
  constexpr int PAD = 8;
  __shared__ __half As[BM_VAL][BK_VAL + PAD];
  __shared__ __half Wt[BK_VAL][BN_VAL + PAD];
  __shared__ float Csc[4][16][16];

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const uint8_t* wbase =
      w + (broadcast_w ? 0 : (size_t)ly * N * (K / 8));
  const size_t scale_stride = (K + 127) / 128;
  const T* sbase =
      scales + (broadcast_w ? 0 : (size_t)ly * N * scale_stride);

  // akumulator per warp
  const int wid = tid / 32;
  constexpr int WARP_M_TILES = (BM_VAL == 32) ? 1 : 2;
  constexpr int WARP_N_TILES = 2;
  const int wm = (BM_VAL == 32) ? (wid / 2) * 16 : (wid / 2) * 32;
  const int wn = (wid % 2) * 32;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WARP_M_TILES][WARP_N_TILES];
#pragma unroll
  for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j)
      wmma::fill_fragment(acc[i][j], 0.0f);

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[WARP_M_TILES];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fb[WARP_N_TILES];

  for (int k_base = 0; k_base < K; k_base += BK_VAL) {
    // ---- 1. Muat Aktivasi A (Vectorized 128-bit uint4 / 8 half) ----
    constexpr int TOTAL_VEC_A = (BM_VAL * BK_VAL) / 8;
    for (int idx = tid; idx < TOTAL_VEC_A; idx += nthreads) {
      int r = idx / (BK_VAL / 8);
      int c = (idx % (BK_VAL / 8)) * 8;
      int gm = block_m + r;
      int gk = k_base + c;

      uint4 u = make_uint4(0, 0, 0, 0);
      if (gm < M && gk < K) {
        if (gk + 8 <= K && (reinterpret_cast<uintptr_t>(&x[((size_t)ly * M + gm) * K + gk]) % 16 == 0)) {
          u = *reinterpret_cast<const uint4*>(&x[((size_t)ly * M + gm) * K + gk]);
        } else {
          __half tmp[8] = {__ushort_as_half(0), __ushort_as_half(0),
                           __ushort_as_half(0), __ushort_as_half(0),
                           __ushort_as_half(0), __ushort_as_half(0),
                           __ushort_as_half(0), __ushort_as_half(0)};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            if (gk + e < K) {
              tmp[e] = x[((size_t)ly * M + gm) * K + gk + e];
            }
          }
          u = *reinterpret_cast<const uint4*>(tmp);
        }
      }
      *reinterpret_cast<uint4*>(&As[r][c]) = u;
    }

    // ---- 2. Dequant Bobot W -> Wt (Vectorized 32-bit Coalesced Load + PTX BFE) ----
    {
      // 128 thread CTA memuat 128 word uint32_t (total 512 byte = tile BKxBN)
      int r = tid / 2;               // baris N lokal (0..63)
      int wc_u32 = tid % 2;          // sub-kolom K lokal (0..1, masing-masing 32 bobot)
      int gn = block_n + r;
      int gk = k_base + wc_u32 * 32;

      uint32_t raw_u32 = 0;
      if (gn < N && gk < K) {
        size_t byte_offset = (size_t)gn * (K / 8) + gk / 8;
        // Muat 4-byte sekaligus jika aligned, fallback skalar aman jika unaligned / boundary K
        if (byte_offset % 4 == 0 && (gk + 32 <= K)) {
          raw_u32 = *reinterpret_cast<const uint32_t*>(&wbase[byte_offset]);
        } else {
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            if (gk + b * 8 < K) {
              raw_u32 |= (static_cast<uint32_t>(wbase[byte_offset + b]) << (b * 8));
            }
          }
        }
      }

      // Pre-load scale 1 kali di luar unroll loop (g128: 32 elemen K berada dalam group yang sama)
      __half2 s2 = __half2half2(__ushort_as_half(0));
      if (gn < N && gk < K) {
        __half s_val = *reinterpret_cast<const __half*>(
            &sbase[(size_t)gn * scale_stride + gk / 128]);
        s2 = __half2half2(s_val);
      }

      // Dekuantisasi 16 pasangan bit dari register raw_u32 ke Wt
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        int k_offset = wc_u32 * 32 + 2 * p;
        int gk_cur = k_base + k_offset;
        __half2 h = __half2half2(__ushort_as_half(0));

        if (gn < N && gk_cur < K) {
          unsigned pairbits;
          asm("bfe.u32 %0, %1, %2, 2;" : "=r"(pairbits) : "r"(raw_u32), "r"(p * 2));
          h = expand2(pairbits, s2);
        }

        if (LB == 0) {                             // Wt[k][n]
          Wt[k_offset][r] = h.x;
          Wt[k_offset + 1][r] = h.y;
        } else {                                   // Wf[n][k] (flat)
          __half* Wf = &Wt[0][0];
          Wf[(size_t)r * (BN_VAL + PAD) + k_offset] = h.x;
          Wf[(size_t)r * (BN_VAL + PAD) + k_offset + 1] = h.y;
        }
      }
    }
    __syncthreads();

    // ---- 3. Komputasi Sub-Tile Warp WMMA ----
#pragma unroll
    for (int sl = 0; sl < BK_VAL / 16; ++sl) {
      const int kk = sl * 16;
#pragma unroll
      for (int i = 0; i < WARP_M_TILES; ++i) {
        if (LA == 0)
          wmma::load_matrix_sync(fa[i], &As[wm + 16 * i][kk], BK_VAL + PAD);
        else
          wmma::load_matrix_sync(fa[i], &As[0][wm + 16 * i], BM_VAL);
      }
#pragma unroll
      for (int j = 0; j < WARP_N_TILES; ++j) {
        if (LB == 0)
          wmma::load_matrix_sync(fb[j], &Wt[kk][wn + 16 * j], BN_VAL + PAD);
        else
          wmma::load_matrix_sync(fb[j], &Wt[0][wn + 16 * j], BK_VAL + PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j)
          wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }

    __syncthreads();
  }

  // ---- 4. Epilog Vectorized __half2 Store (Aligned Safe) ----
  const int lane = tid % 32;
#pragma unroll
  for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j) {
      wmma::store_matrix_sync(&Csc[wid][0][0], acc[i][j], 16, wmma::mem_row_major);
      for (int idx = lane; idx < 128; idx += 32) { // 128 pasang = 256 elemen
        int r = idx / 8;
        int c = (idx % 8) * 2;
        int gm = block_m + wm + 16 * i + r;
        int gn = block_n + wn + 16 * j + c;
        if (gm < M && gn < N) {
          size_t out_idx = ((size_t)ly * M + gm) * N + gn;
          if (gn + 1 < N && (out_idx % 2 == 0)) {
            __half2 v = __halves2half2(
                static_cast<__half>(Csc[wid][r][c]),
                static_cast<__half>(Csc[wid][r][c + 1]));
            *reinterpret_cast<__half2*>(&y[out_idx]) = v;
          } else {
            y[out_idx] = static_cast<T>(Csc[wid][r][c]);
            if (gn + 1 < N) {
              y[out_idx + 1] = static_cast<T>(Csc[wid][r][c + 1]);
            }
          }
        }
      }
    }
}

} // namespace mlx::core::cu::wmma_b1

