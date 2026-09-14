// ===----------------------------------------------------------------------=== //
// Module: src/csrc/qmv_sm75_kernel.cu
// Purpose: Implementasi CUDA kernel SM75 W1A16 g128 Nibble-LUT dengan mode Q1O
//          (s' = 2s, b = -s) untuk inferensi Bonsai-27B di NVIDIA Tesla T4.
//          Didesain untuk diekspor via C ABI shared library (libbonsai_qmv_sm75.so)
//          dan dipanggil dari Mojo via Foreign Function Interface (FFI).
// ===----------------------------------------------------------------------=== //

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <dlfcn.h>

__attribute__((constructor))
static void __bonsai_cuda_pin_library() {
    // Pin libbonsai_qmv_sm75.so secara permanen di address space proses
    // sehingga dlclose() tidak pernah men-detach library dari memori virtual.
    dlopen("libbonsai_qmv_sm75.so", RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE);
}

namespace bonsai::sm75 {

// Geometri Hardware SM75 (Tesla T4)
constexpr int QMV_LUT_BLOCK_THREADS = 256;  // 8 warps = 256 threads
constexpr int QMV_LUT_GROUP_SIZE    = 128;  // Kuantisasi g128
constexpr int QMV_VEC_ROWS_PER_BLOCK = 32;  // 32 baris output per block
constexpr int QMV_VEC_LANES_PER_ROW  = 8;   // 8 lanes per baris output
constexpr int QMV_VEC_GS             = 8;   // 8 grup (1024 elemen K) per staging tile
constexpr int QMV_VEC_K_TILE         = QMV_VEC_GS * QMV_LUT_GROUP_SIZE; // 1024
constexpr int QMV_VEC_GRP_PAD        = QMV_LUT_GROUP_SIZE + 4;          // 132 (anti-bank conflict)

constexpr int QMV_NIB_BITS           = 4;   // 4 bit per nibble
constexpr int QMV_NIB_PER_GRP        = QMV_LUT_GROUP_SIZE / QMV_NIB_BITS; // 32 nibble
constexpr int QMV_NIB_ENT            = 16;  // 16 entri tabel per nibble
constexpr int QMV_NIB_ENT_PAD        = QMV_NIB_ENT + 1;                  // 17 float (anti-bank conflict)

// Traits konversi tipe floating point
template <typename T>
struct QmvTraits;

template <>
struct QmvTraits<__half> {
    static __host__ __device__ __forceinline__ float to_float(__half v) {
        return __half2float(v);
    }
    static __host__ __device__ __forceinline__ __half from_float(float v) {
        return __float2half(v);
    }
};

template <>
struct QmvTraits<__nv_bfloat16> {
    static __host__ __device__ __forceinline__ float to_float(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __host__ __device__ __forceinline__ __nv_bfloat16 from_float(float v) {
        return __float2bfloat16(v);
    }
};

// Unpack 16-byte uint4 (8 elemen T) ke 8 elemen FP32
template <typename T>
__device__ __forceinline__ void qmv_vec_unpack8(const uint4& v, float* dst) {
    const unsigned short* hw = reinterpret_cast<const unsigned short*>(&v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        dst[j] = QmvTraits<T>::to_float(*reinterpret_cast<const T*>(&hw[j]));
    }
}

// ----------------------------------------------------------------------------
// Kernel 1: Nibble-LUT W1A16 Decode dengan Mode Q1O (s' = 2s, b = -s)
// ----------------------------------------------------------------------------
template <typename T>
__global__ void qmv_vec_nib_q1o_kernel(
    const T* __restrict__ x,
    const uint8_t* __restrict__ w,
    const T* __restrict__ scales,
    T* __restrict__ out,
    float* __restrict__ ws, // nullptr => single-split, tulis langsung ke out
    int slice_idx,
    int m,
    int n,
    int k,
    int l,
    bool broadcast_w,
    int g_begin,
    int g_count
) {
    constexpr int GS  = QMV_VEC_GS;
    constexpr int LPR = QMV_VEC_LANES_PER_ROW;
    constexpr int RPW = 32 / LPR;              // 4 baris per warp
    constexpr int RPB = QMV_VEC_ROWS_PER_BLOCK;// 32 baris per block
    constexpr int GRP_PAD = QMV_VEC_GRP_PAD;   // 132 float

    // Alokasi Shared Memory (Total ~25 KiB, aman di bawah limit 48 KiB T4)
    __shared__ __align__(16) float x_s[GS * GRP_PAD];
    __shared__ __align__(16) float nib[GS][QMV_NIB_PER_GRP][QMV_NIB_ENT_PAD];
    __shared__ float xg_s[GS];
    // Double-buffered scales & biases untuk latency hiding tanpa barrier tambahan
    __shared__ float s_s[2][RPB][GS];
    __shared__ float b_s[2][RPB][GS];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r    = lane / LPR;   // Baris output warp (0..3)
    const int grp  = lane % LPR;   // Grup g128 dalam K-tile (0..7)

    const int row = static_cast<int>(blockIdx.x) * RPB + warp * RPW + r;
    const int l_idx = static_cast<int>(blockIdx.z);
    const int w_batch = broadcast_w ? 0 : l_idx;

    const int64_t row_base = static_cast<int64_t>(row) + static_cast<int64_t>(n) * w_batch;
    const uint8_t* w_row   = w + row_base * (k / 8);

    const int groups_per_row = k / QMV_LUT_GROUP_SIZE;
    const bool row_valid     = row < n;

    const int64_t out_batch = static_cast<int64_t>(l_idx) * m * n;
    const int64_t ws_batch  = (ws != nullptr)
        ? static_cast<int64_t>(slice_idx) * l * m * n + static_cast<int64_t>(l_idx) * m * n
        : 0;

    float acc = 0.0f;
    const int g_end = g_begin + g_count;

    // ---- Prologue: Stage tile 0 & prefetch bobot uint4 ----
    int g0 = g_begin;
    int buf = 0;
    {
        const int kc = g0 * QMV_LUT_GROUP_SIZE;
        for (int e = tid * 8; e < QMV_VEC_K_TILE; e += QMV_LUT_BLOCK_THREADS * 8) {
            const int gi  = e / QMV_LUT_GROUP_SIZE;
            const int off = e % QMV_LUT_GROUP_SIZE;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (e + kc < k) {
                v = *reinterpret_cast<const uint4*>(
                    x + static_cast<int64_t>(l_idx) * k + kc + e
                );
            }
            qmv_vec_unpack8<T>(v, &x_s[gi * GRP_PAD + off]);
        }

        // Mode Q1O: s' = 2*scales, b = -scales (langsung dihitung saat staging, zero extra memory)
        for (int idx = tid; idx < RPB * GS; idx += QMV_LUT_BLOCK_THREADS) {
            const int sr = idx / GS;
            const int sg = idx % GS;
            const int grow = static_cast<int>(blockIdx.x) * RPB + sr;
            const int g = g0 + sg;
            const bool ok = (grow < n) && (g < g_end) && (g < groups_per_row);
            const int64_t grb = static_cast<int64_t>(grow) + static_cast<int64_t>(n) * w_batch;
            float raw_s = ok ? QmvTraits<T>::to_float(scales[grb * groups_per_row + g]) : 0.0f;
            s_s[buf][sr][sg] = 2.0f * raw_s;
            b_s[buf][sr][sg] = -raw_s;
        }
    }

    uint4 wv = make_uint4(0u, 0u, 0u, 0u);
    if (row_valid && (g0 + grp) < g_end && (g0 + grp) < groups_per_row) {
        wv = *reinterpret_cast<const uint4*>(w_row + static_cast<int64_t>(g0 + grp) * 16);
    }

    // ---- Software Pipelined Loop (2 Barriers per Tile) ----
    while (g0 < g_end) {
        __syncthreads(); // Barrier 1: x_s dan s_s[buf] siap dibaca

        // 1. Bangun tabel subset-sum nibble (warp w menangani grup w, lane l menangani nibble l)
        const float* xg = &x_s[warp * GRP_PAD + lane * QMV_NIB_BITS];
        float t[QMV_NIB_ENT];
        t[0]  = 0.0f;
        t[1]  = xg[0];
        t[2]  = xg[1];
        t[3]  = xg[0] + xg[1];
        t[4]  = xg[2];
        t[5]  = t[1] + xg[2];
        t[6]  = t[2] + xg[2];
        t[7]  = t[3] + xg[2];
        t[8]  = xg[3];
        t[9]  = t[1] + xg[3];
        t[10] = t[2] + xg[3];
        t[11] = t[3] + xg[3];
        t[12] = t[4] + xg[3];
        t[13] = t[5] + xg[3];
        t[14] = t[6] + xg[3];
        t[15] = t[7] + xg[3];

        float* ntl = &nib[warp][lane ^ warp][0];
#pragma unroll
        for (int e = 0; e < QMV_NIB_ENT; ++e) {
            ntl[e] = t[e];
        }

        // Reduksi jumlah seluruh x dalam grup dengan butterfly shuffle
        float gs = t[QMV_NIB_ENT - 1];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            gs += __shfl_xor_sync(0xffffffffu, gs, off);
        }
        if (lane == 0) {
            xg_s[warp] = gs;
        }

        // 2. Prefetch tile berikutnya gn = g0 + GS ke register
        const int gn = g0 + GS;
        uint4 vn = make_uint4(0u, 0u, 0u, 0u);
        uint4 wv_next = make_uint4(0u, 0u, 0u, 0u);
        if (gn < g_end) {
            const int kcn = gn * QMV_LUT_GROUP_SIZE;
            const int en = tid * 8;
            if (en < QMV_VEC_K_TILE && en + kcn < k) {
                vn = *reinterpret_cast<const uint4*>(
                    x + static_cast<int64_t>(l_idx) * k + kcn + en
                );
            }
            if (row_valid && (gn + grp) < g_end && (gn + grp) < groups_per_row) {
                wv_next = *reinterpret_cast<const uint4*>(
                    w_row + static_cast<int64_t>(gn + grp) * 16
                );
            }
        }

        __syncthreads(); // Barrier 2: nib dan xg_s siap dibaca untuk dot product

        // 3. Dot Product Nibble LUT (1 lookup per 4 bobot)
        const float* ntg = &nib[grp][0][0];
        const unsigned* ww = reinterpret_cast<const unsigned*>(&wv);
        float p = 0.0f;
#pragma unroll
        for (int wi = 0; wi < 4; ++wi) {
            const unsigned bits = ww[wi];
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                const int nidx = wi * 8 + h;
                p += ntg[(nidx ^ grp) * QMV_NIB_ENT_PAD + ((bits >> (4 * h)) & 15u)];
            }
        }
        const int rb = warp * RPW + r;
        // Akumulasi Q1O: (2s)*p + (-s)*xg_s = s*(2p - xg_s) == (2q-1)*s
        acc += s_s[buf][rb][grp] * p + b_s[buf][rb][grp] * xg_s[grp];

        // 4. Restage tile berikutnya gn ke SMEM (x_s dan buf^1)
        if (gn < g_end) {
            const int en = tid * 8;
            if (en < QMV_VEC_K_TILE) {
                const int gi = en / QMV_LUT_GROUP_SIZE;
                const int off = en % QMV_LUT_GROUP_SIZE;
                qmv_vec_unpack8<T>(vn, &x_s[gi * GRP_PAD + off]);
            }
            const int nbuf = buf ^ 1;
            for (int idx = tid; idx < RPB * GS; idx += QMV_LUT_BLOCK_THREADS) {
                const int sr = idx / GS;
                const int sg = idx % GS;
                const int grow = static_cast<int>(blockIdx.x) * RPB + sr;
                const int g = gn + sg;
                const bool ok = (grow < n) && (g < g_end) && (g < groups_per_row);
                const int64_t grb = static_cast<int64_t>(grow) + static_cast<int64_t>(n) * w_batch;
                float raw_s = ok ? QmvTraits<T>::to_float(scales[grb * groups_per_row + g]) : 0.0f;
                s_s[nbuf][sr][sg] = 2.0f * raw_s;
                b_s[nbuf][sr][sg] = -raw_s;
            }
        }

        wv = wv_next;
        g0 = gn;
        buf ^= 1;
    }

    // ---- Reduksi Deterministik Per-Warp (8 Lanes) ----
    float v = acc;
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off, LPR);
    }

    const int64_t flat_base = (ws != nullptr ? ws_batch : out_batch);
    if (row_valid && grp == 0) {
        if (ws != nullptr) {
            ws[flat_base + static_cast<int64_t>(row)] = v;
        } else {
            out[flat_base + static_cast<int64_t>(row)] = QmvTraits<T>::from_float(v);
        }
    }
}

// ----------------------------------------------------------------------------
// Kernel 2: Reduksi Split-K Deterministik (Ascending order, tanpa atomics)
// ----------------------------------------------------------------------------
template <typename T>
__global__ void qmv_split_reduce_kernel(
    const float* __restrict__ ws,
    T* __restrict__ out,
    long long total,
    int splits
) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;

    float a = 0.0f;
    for (int s = 0; s < splits; ++s) {
        a += ws[static_cast<int64_t>(s) * total + i];
    }
    out[i] = QmvTraits<T>::from_float(a);
}

} // namespace bonsai::sm75

// ============================================================================
// Prefill GEMM W1A16 g128 — port qmm_impl_sm75_wmma.cuh (WMMA v2, jalur
// prefill aktif referensi prism-mlx-cuda1bit, ~120 tok/s di T4).
// y[L,M,N] = x[L,M,K] · w[N,K/8]^T ; w_eff = s*(2q-1) (bias terserap).
// Tile BMxBNxBK 64/64/64 (atau 32/64/64 untuk M kecil), WMMA 16x16x16 fp16
// dengan akumulasi FP32, dekuant via PTX bfe.u32, SMEM pad 8 anti bank-conflict.
// ============================================================================
#include <mma.h>

namespace bonsai::sm75::wmma_b1 {

constexpr int QMM_BN = 64;

__device__ inline unsigned qmm_extract_2bits(unsigned raw_byte, int pair_idx) {
    unsigned pair;
    asm("bfe.u32 %0, %1, %2, 2;" : "=r"(pair) : "r"(raw_byte), "r"(pair_idx * 2));
    return pair;
}

__device__ inline __half2 qmm_expand2(unsigned bits, __half2 s2) {
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

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[WARP_M_TILES][WARP_N_TILES];
#pragma unroll
  for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j)
      nvcuda::wmma::fill_fragment(acc[i][j], 0.0f);

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> fa[WARP_M_TILES];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> fb[WARP_N_TILES];

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

      // Pre-load scale 1 kali di luar unroll loop (g128: 32 elemen K berada dalam grup yang sama)
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
          h = qmm_expand2(pairbits, s2);
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
          nvcuda::wmma::load_matrix_sync(fa[i], &As[wm + 16 * i][kk], BK_VAL + PAD);
        else
          nvcuda::wmma::load_matrix_sync(fa[i], &As[0][wm + 16 * i], BM_VAL);
      }
#pragma unroll
      for (int j = 0; j < WARP_N_TILES; ++j) {
        if (LB == 0)
          nvcuda::wmma::load_matrix_sync(fb[j], &Wt[kk][wn + 16 * j], BN_VAL + PAD);
        else
          nvcuda::wmma::load_matrix_sync(fb[j], &Wt[0][wn + 16 * j], BK_VAL + PAD);
      }
#pragma unroll
      for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
        for (int j = 0; j < WARP_N_TILES; ++j)
          nvcuda::wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }

    __syncthreads();
  }

  // ---- 4. Epilog Vectorized __half2 Store (Aligned Safe) ----
  const int lane = tid % 32;
#pragma unroll
  for (int i = 0; i < WARP_M_TILES; ++i)
#pragma unroll
    for (int j = 0; j < WARP_N_TILES; ++j) {
      nvcuda::wmma::store_matrix_sync(&Csc[wid][0][0], acc[i][j], 16, nvcuda::wmma::mem_row_major);
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

} // namespace bonsai::sm75::wmma_b1

// ============================================================================
// GDN SEQUENCE FUSED (tiru struktur prefill MLX fork gdn_step_kernel.cuh):
// SELURUH sekuens T token diproses dalam SATU launch, state S hidup di
// register (nol round-trip VRAM per token). Gating math replika 1:1 dari
// kernel rekurensi Mojo kita (a_log/dt_bias di dalam loop). State dibulatkan
// ke fp16 tiap token agar bit-exact dengan jalur per-token decode.
// ============================================================================

__device__ inline float gdn_seq_warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

template <bool HAS_PARAMS, int NPT>
__global__ void gdn_seq_sm75_kernel(
    const __half* __restrict__ q,   // [T, Hk*Dk] (stride qk_stride)
    const __half* __restrict__ k,   // [T, Hk*Dk]
    const __half* __restrict__ v,   // [T, v_stride] (v head hv di hv*Dv)
    const __half* __restrict__ a,   // [T, ab_stride] (a di +a_off)
    const __half* __restrict__ b,   // [T, ab_stride] (b di +b_off)
    const float* __restrict__ a_log,   // [Hv]
    const float* __restrict__ dt_bias, // [Hv]
    float* __restrict__ state,         // [Hv, Dv, Dk] fp32 in/out
    __half* __restrict__ y,            // [T, y_stride] (head hv di hv*Dv)
    int T, int Hv, int Hk, int Dk, int Dv,
    int qk_stride, int v_stride, int ab_stride, int y_stride,
    int a_off, int b_off) {
    const int n = blockIdx.z;          // hv
    const int hv = n;
    const int hk = hv / (Hv / Hk);
    const int dv = blockIdx.y * blockDim.y + threadIdx.y;
    if (dv >= Dv) return;

    const int lane = threadIdx.x;

    const __half* qp = q + hk * Dk;
    const __half* kp = k + hk * Dk;
    const __half* vp = v + hv * Dv + dv;
    __half* yp = y + hv * Dv + dv;

    // State ke register (fp32) — baris S[hv, dv, :]. State disimpan FP32 di
    // VRAM (WAJIB: config `mamba_ssm_dtype: float32`; llama.cpp GGML_TYPE_F32).
    float* sp = state + ((size_t)n * Dv + dv) * Dk;
    float st[NPT];
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        int sidx = NPT * lane + i;
        st[i] = (sidx < Dk) ? sp[sidx] : 0.0f;
    }

    const float al = a_log[hv];
    const float dtb = dt_bias[hv];

    for (int t = 0; t < T; ++t) {
        // Gating math replika kernel Mojo (HAS_PARAMS=True)
        float a_val = __half2float(a[(size_t)t * ab_stride + a_off + hv]);
        float b_val = __half2float(b[(size_t)t * ab_stride + b_off + hv]);
        float sp_in = a_val + dtb;
        float spf = (sp_in > 20.0f) ? sp_in : logf(1.0f + expf(sp_in));
        float dec = expf(-expf(al) * spf);
        float beta = 1.0f / (1.0f + expf(-b_val));

        float kt[NPT], qt[NPT];
        float kv_mem = 0.f;
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            int sidx = NPT * lane + i;
            float kv = (sidx < Dk) ? __half2float(kp[(size_t)t * qk_stride + sidx]) : 0.0f;
            kt[i] = kv;
            qt[i] = (sidx < Dk) ? __half2float(qp[(size_t)t * qk_stride + sidx]) : 0.0f;
            // State tetap FP32 penuh — TIDAK ada round fp16 per langkah.
            // Rounding per-langkah + decay ~1.0 = drift terakumulasi.
            float sdec = st[i] * dec;
            st[i] = sdec;
            kv_mem += sdec * kt[i];
        }
        kv_mem = gdn_seq_warp_sum(kv_mem);

        float delta = (__half2float(vp[(size_t)t * v_stride]) - kv_mem) * beta;

        float out = 0.f;
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            // State tetap FP32 penuh — TIDAK ada round fp16 per langkah.
            float snew = st[i] + kt[i] * delta;
            out += snew * qt[i];
            st[i] = snew;
        }
        out = gdn_seq_warp_sum(out);
        if (lane == 0) yp[(size_t)t * y_stride] = __float2half(out);
    }

    // Tulis balik state akhir (fp32) — SEKALI untuk seluruh sekuens
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        int sidx = NPT * lane + i;
        if (sidx < Dk) sp[sidx] = st[i];
    }
}

// ============================================================================
// C ABI Export Functions (Dipanggil via Mojo FFI)
// ============================================================================
extern "C" {

int bonsai_cuda_sm75_init() {
    return 0; // Berhasil
}

int launch_qmv_sm75_b1_decode_fp16(
    const void* x,          // [L, M, K] __half
    const void* w,          // [N, K/8] uint8
    const void* scales,     // [N, K/128] __half (s_eff)
    void* out,              // [L, M, N] __half
    float* ws,              // [splits, L*M*N] float workspace (or nullptr)
    int m, int n, int k, int l,
    int broadcast_w,
    int splits,
    cudaStream_t stream
) {
    using namespace bonsai::sm75;

    const __half* xp = reinterpret_cast<const __half*>(x);
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w);
    const __half* sp = reinterpret_cast<const __half*>(scales);
    __half* op = reinterpret_cast<__half*>(out);

    const int groups_per_row = k / QMV_LUT_GROUP_SIZE;
    const int blocks_x = (n + QMV_VEC_ROWS_PER_BLOCK - 1) / QMV_VEC_ROWS_PER_BLOCK;

    dim3 grid(blocks_x, 1, l);
    dim3 block(QMV_LUT_BLOCK_THREADS, 1, 1);

    if (splits <= 1) {
        qmv_vec_nib_q1o_kernel<__half><<<grid, block, 0, stream>>>(
            xp, wp, sp, op, nullptr, 0,
            m, n, k, l, broadcast_w != 0,
            0, groups_per_row
        );
    } else {
        const int base = groups_per_row / splits;
        const int rem  = groups_per_row % splits;

        for (int s = 0; s < splits; ++s) {
            int g_begin = s * base + (s < rem ? s : rem);
            int g_count = base + (s < rem ? 1 : 0);

            qmv_vec_nib_q1o_kernel<__half><<<grid, block, 0, stream>>>(
                xp, wp, sp, op, ws, s,
                m, n, k, l, broadcast_w != 0,
                g_begin, g_count
            );
        }

        const long long total = static_cast<long long>(l) * m * n;
        const unsigned reduce_blocks = (total + QMV_LUT_BLOCK_THREADS - 1) / QMV_LUT_BLOCK_THREADS;
        qmv_split_reduce_kernel<__half><<<dim3(reduce_blocks, 1, 1), block, 0, stream>>>(
            ws, op, total, splits
        );
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int launch_qmv_sm75_b1_decode_bf16(
    const void* x,
    const void* w,
    const void* scales,
    void* out,
    float* ws,
    int m, int n, int k, int l,
    int broadcast_w,
    int splits,
    cudaStream_t stream
) {
    using namespace bonsai::sm75;

    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x);
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w);
    const __nv_bfloat16* sp = reinterpret_cast<const __nv_bfloat16*>(scales);
    __nv_bfloat16* op = reinterpret_cast<__nv_bfloat16*>(out);

    const int groups_per_row = k / QMV_LUT_GROUP_SIZE;
    const int blocks_x = (n + QMV_VEC_ROWS_PER_BLOCK - 1) / QMV_VEC_ROWS_PER_BLOCK;

    dim3 grid(blocks_x, 1, l);
    dim3 block(QMV_LUT_BLOCK_THREADS, 1, 1);

    if (splits <= 1) {
        qmv_vec_nib_q1o_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
            xp, wp, sp, op, nullptr, 0,
            m, n, k, l, broadcast_w != 0,
            0, groups_per_row
        );
    } else {
        const int base = groups_per_row / splits;
        const int rem  = groups_per_row % splits;

        for (int s = 0; s < splits; ++s) {
            int g_begin = s * base + (s < rem ? s : rem);
            int g_count = base + (s < rem ? 1 : 0);

            qmv_vec_nib_q1o_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
                xp, wp, sp, op, ws, s,
                m, n, k, l, broadcast_w != 0,
                g_begin, g_count
            );
        }

        const long long total = static_cast<long long>(l) * m * n;
        const unsigned reduce_blocks = (total + QMV_LUT_BLOCK_THREADS - 1) / QMV_LUT_BLOCK_THREADS;
        qmv_split_reduce_kernel<__nv_bfloat16><<<dim3(reduce_blocks, 1, 1), block, 0, stream>>>(
            ws, op, total, splits
        );
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

int qmv_sm75_device_synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Prefill batched W1A16 (WMMA v2 tensor core). x fp16 [L,M,K] row-major,
// y fp16 [L,M,N]; syarat K % 128 == 0 (tile BK=64 tak boleh menyilang grup).
int launch_qmm_sm75_b1_prefill_fp16(
    const void* x,          // [L, M, K] __half
    const void* w,          // [N, K/8] uint8
    const void* scales,     // [N, K/128] __half (s_eff)
    void* out,              // [L, M, N] __half
    int m, int n, int k, int l,
    int broadcast_w,
    cudaStream_t stream
) {
    using namespace bonsai::sm75::wmma_b1;

    if (k % 128 != 0) return -100; // kontrak g128 dilanggar

    const __half* xp = reinterpret_cast<const __half*>(x);
    const uint8_t* wp = reinterpret_cast<const uint8_t*>(w);
    const __half* sp = reinterpret_cast<const __half*>(scales);
    __half* op = reinterpret_cast<__half*>(out);
    const __half* bp = nullptr;  // biases terserap (b=-s), tak pernah deref
    bool bw_arg = broadcast_w != 0;

    if (m <= 32) {
        qmm_sm75_b1_kernel<__half, 32, 64, 64>
            <<<dim3((n + QMM_BN - 1) / QMM_BN, (m + 31) / 32, l),
               dim3(128), 0, stream>>>(
                xp, wp, sp, bp, op, m, n, k, l, bw_arg);
    } else {
        qmm_sm75_b1_kernel<__half, 64, 64, 64>
            <<<dim3((n + QMM_BN - 1) / QMM_BN, (m + 63) / 64, l),
               dim3(128), 0, stream>>>(
                xp, wp, sp, bp, op, m, n, k, l, bw_arg);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// GDN sequence fused: seluruh chunk T token dalam 1 launch, state register.
// a_off/b_off = offset a & b di dalam baris proj (ab_stride); v head hv di
// hv*Dv dalam baris v_stride; y head hv di hv*Dv dalam baris y_stride.
int launch_gdn_seq_sm75_fp16(
    const void* q, const void* k, const void* v,
    const void* a, const void* b,
    const void* a_log, const void* dt_bias,
    void* state, void* y,
    int t_len, int hv, int hk, int dk, int dv,
    int qk_stride, int v_stride, int ab_stride, int y_stride,
    int a_off, int b_off,
    cudaStream_t stream
) {
    const __half* qp = reinterpret_cast<const __half*>(q);
    const __half* kp = reinterpret_cast<const __half*>(k);
    const __half* vp = reinterpret_cast<const __half*>(v);
    const __half* ap = reinterpret_cast<const __half*>(a);
    const __half* bp = reinterpret_cast<const __half*>(b);
    const float* alp = reinterpret_cast<const float*>(a_log);
    const float* dbp = reinterpret_cast<const float*>(dt_bias);
    float* sp = reinterpret_cast<float*>(state);
    __half* yp = reinterpret_cast<__half*>(y);

    // 1 warp per baris dv: block (32, 4), grid (1, ceil(Dv/4), Hv)
    dim3 block(32, 4, 1);
    dim3 grid(1, (dv + 3) / 4, hv);

    if (dk == 128) {
        gdn_seq_sm75_kernel<true, 4><<<grid, block, 0, stream>>>(
            qp, kp, vp, ap, bp, alp, dbp, sp, yp,
            t_len, hv, hk, dk, dv,
            qk_stride, v_stride, ab_stride, y_stride, a_off, b_off);
    } else if (dk == 256) {
        gdn_seq_sm75_kernel<true, 8><<<grid, block, 0, stream>>>(
            qp, kp, vp, ap, bp, alp, dbp, sp, yp,
            t_len, hv, hk, dk, dv,
            qk_stride, v_stride, ab_stride, y_stride, a_off, b_off);
    } else if (dk == 64) {
        gdn_seq_sm75_kernel<true, 2><<<grid, block, 0, stream>>>(
            qp, kp, vp, ap, bp, alp, dbp, sp, yp,
            t_len, hv, hk, dk, dv,
            qk_stride, v_stride, ab_stride, y_stride, a_off, b_off);
    } else {
        return -101; // Dk tidak didukung instansiasi NPT
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

} // extern "C"
