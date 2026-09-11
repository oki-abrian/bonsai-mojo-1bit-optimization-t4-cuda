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
    __half* __restrict__ state,        // [Hv, Dv, Dk] fp16 in/out
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

    // State ke register (fp32) — baris S[hv, dv, :]
    __half* sp = state + ((size_t)n * Dv + dv) * Dk;
    float st[NPT];
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        int sidx = NPT * lane + i;
        st[i] = (sidx < Dk) ? __half2float(sp[sidx]) : 0.0f;
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
            // Replika persis kernel per-token: decay lalu ROUND fp16 (state
            // disimpan fp16), sedangkan kv_mem memakai nilai decay BELUM round.
            float sdec = st[i] * dec;
            st[i] = __half2float(__float2half(sdec));
            kv_mem += sdec * kt[i];
        }
        kv_mem = gdn_seq_warp_sum(kv_mem);

        float delta = (__half2float(vp[(size_t)t * v_stride]) - kv_mem) * beta;

        float out = 0.f;
#pragma unroll
        for (int i = 0; i < NPT; ++i) {
            // out memakai state baru BELUM round; state disimpan round fp16.
            float snew = st[i] + kt[i] * delta;
            out += snew * qt[i];
            st[i] = __half2float(__float2half(snew));
        }
        out = gdn_seq_warp_sum(out);
        if (lane == 0) yp[(size_t)t * y_stride] = __float2half(out);
    }

    // Tulis balik state akhir (fp16) — SEKALI untuk seluruh sekuens
#pragma unroll
    for (int i = 0; i < NPT; ++i) {
        int sidx = NPT * lane + i;
        if (sidx < Dk) sp[sidx] = __float2half(st[i]);
    }
}

// ============================================================================ //
// KHQ (KudaHitamQuant) — port kernel referensi KudaHitamMLX.py:
//   CUDA_SOURCE      -> khq_compress_kernel   (baris 576-835)
//   CUDA_ATTN_SOURCE -> khq_attn_kernel       (baris 924-1311)
// Layout payload mengikuti referensi apa adanya (40 B K / 104 B V).
// ============================================================================ //
__global__ void khq_compress_kernel(
    const __half* __restrict__ x,             // (N, D) fp16
    const __half* __restrict__ d_vec,         // (D) fp16
    const __half* __restrict__ rotor_params,  // (NP*16) fp16
    const __half* __restrict__ centroids,     // (4, D) fp16
    const __half* __restrict__ vq_centroids,  // (NP*256*3) fp16
    unsigned int* __restrict__ out_mask,      // (N, D/16) u32
    __half* __restrict__ out_norms,           // (N)
    __half* __restrict__ out_r_norms,         // (N) residual_scale (K) / energy (V)
    unsigned char* __restrict__ out_payload,  // (N, 40|104)
    unsigned char* __restrict__ out_shared_meta,  // (N, 4) float
    int D_val, int N, float threshold_scale, float alpha_param, int is_v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const float f_s = rsqrtf((float)D_val);

    for (int row_id = tid; row_id < N; row_id += stride) {
        float sum_sq = 0.0f;
        for (int i = 0; i < D_val; ++i) {
            float v = __half2float(x[row_id * D_val + i]);
            sum_sq += v * v;
        }
        float norm = sqrtf(sum_sq + 1e-8f);
        out_norms[row_id] = __float2half(norm);

        float r[256];
        for (int i = 0; i < D_val; i++)
            r[i] = (__half2float(x[row_id * D_val + i]) * __half2float(d_vec[i])) / (norm + 1e-8f);
        for (int step = 1; step < D_val; step <<= 1) {
            for (int i = 0; i < D_val; i += 2 * step) {
                for (int j = 0; j < step; j++) {
                    float a = r[i + j], b = r[i + j + step];
                    r[i + j] = a + b; r[i + j + step] = a - b;
                }
            }
        }

        float rot[256];
        for (int patch_id = 0; patch_id < D_val / 4; patch_id++) {
            for (int comp_id = 0; comp_id < 4; comp_id++) {
                float rot_val = 0.0f;
                for (int k = 0; k < 4; k++)
                    rot_val += r[patch_id * 4 + k] *
                        __half2float(rotor_params[patch_id * 16 + k * 4 + comp_id]);
                rot[patch_id * 4 + comp_id] = rot_val;
            }
        }

        int payload_stride = (is_v == 1) ? 104 : 40;
        unsigned int b_plan[16] = {0};
        float s_outlier_proj_abs[50];
        unsigned int s_outlier_dim_idx[50];
        unsigned int s_outlier_signs[50];
        unsigned int s_total_outliers = 0;
        float sum_normals = 0.0f;
        float sum_cbase = 0.0f;

        for (int i = 0; i < D_val; i++) {
            float proj = rot[i] * f_s;
            unsigned int s_bit;
            if (is_v == 1) {
                float th_high = threshold_scale * f_s;
                if (proj > th_high) s_bit = 3;
                else if (proj > 0.0f) s_bit = 2;
                else if (proj > -th_high) s_bit = 1;
                else s_bit = 0;
            } else {
                float proj_scaled = proj * threshold_scale;
                float abs_proj = fabsf(proj);
                if (abs_proj > 1.60f * f_s) {
                    if (s_total_outliers < 50) {
                        s_bit = 3;
                        s_outlier_proj_abs[s_total_outliers] = fabsf(proj_scaled);
                        s_outlier_dim_idx[s_total_outliers] = i;
                        s_outlier_signs[s_total_outliers] = (proj > 0.0f) ? 1 : 0;
                        s_total_outliers++;
                    } else {
                        s_bit = (proj > 0.0f) ? 2 : 1;
                    }
                } else {
                    if (proj > 0.47f * f_s) s_bit = 2;
                    else if (proj < -0.47f * f_s) s_bit = 1;
                    else s_bit = 0;
                    if (s_bit == 1 || s_bit == 2) {
                        sum_normals += abs_proj;
                        sum_cbase += __half2float(centroids[2 * D_val + i]);
                    }
                }
            }
            b_plan[i / 16] |= (s_bit << ((i % 16) * 2));
        }
        // Hanya cabang K yang memiliki mask; cabang V tidak boleh menimpanya
        // (V memakai skema VQ tanpa k_mask).
        if (is_v == 0)
            for (int k = 0; k < 16; k++) out_mask[row_id * 16 + k] = b_plan[k];

        if (is_v == 1) {
            unsigned int s_vq_codes[64];
            unsigned int s_dim3_codes[64];
            unsigned int s_dim3_dumps[64];
            float s_patch_res[64];
            float s_true_energy[64];
            float s_vq_energy[64];
            float max_res = 0.0f;
            for (int patch_id = 0; patch_id < D_val / 4; patch_id++) {
                float v0 = rot[patch_id * 4 + 0] * f_s;
                float v1 = rot[patch_id * 4 + 1] * f_s;
                float v2 = rot[patch_id * 4 + 2] * f_s;
                const __half* vq_codebook = vq_centroids + patch_id * 256 * 3;
                float min_dist_vq = 1e20f;
                unsigned int best_vq = 0;
                for (unsigned int i = 0; i < 256; i++) {
                    float c0 = __half2float(vq_codebook[i * 3 + 0]);
                    float c1 = __half2float(vq_codebook[i * 3 + 1]);
                    float c2 = __half2float(vq_codebook[i * 3 + 2]);
                    float d0 = v0 - c0 * alpha_param;
                    float d1 = v1 - c1 * alpha_param;
                    float d2 = v2 - c2 * alpha_param;
                    float dist = d0 * d0 + d1 * d1 + d2 * d2;
                    if (dist < min_dist_vq) { min_dist_vq = dist; best_vq = i; }
                }
                s_vq_codes[patch_id] = best_vq;
                float c0_b = __half2float(vq_codebook[best_vq * 3 + 0]) * alpha_param;
                float c1_b = __half2float(vq_codebook[best_vq * 3 + 1]) * alpha_param;
                float c2_b = __half2float(vq_codebook[best_vq * 3 + 2]) * alpha_param;
                s_true_energy[patch_id] = v0 * v0 + v1 * v1 + v2 * v2;
                s_vq_energy[patch_id] = c0_b * c0_b + c1_b * c1_b + c2_b * c2_b;

                float v3 = rot[patch_id * 4 + 3] * f_s;
                float min_dist_s = 1e20f;
                unsigned int best_s = 0;
                for (unsigned int i = 0; i < 2; i++) {
                    float c_val = __half2float(centroids[i * D_val + patch_id * 4 + 3]);
                    float dist = fabsf(v3 - c_val * alpha_param);
                    if (dist < min_dist_s) { min_dist_s = dist; best_s = i; }
                }
                s_dim3_codes[patch_id] = best_s;
                float c_val_best = __half2float(centroids[best_s * D_val + patch_id * 4 + 3]);
                float residual = v3 - c_val_best * alpha_param;
                s_patch_res[patch_id] = residual;
                if (fabsf(residual) > max_res) max_res = fabsf(residual);
            }
            float* sm_scale = (float*)(out_shared_meta + row_id * 4);
            sm_scale[0] = max_res;
            float res_scale = fmaxf(max_res, 1e-4f);
            for (int patch_id = 0; patch_id < D_val / 4; patch_id++) {
                float scaled = (s_patch_res[patch_id] / res_scale) * 7.0f + 8.0f;
                s_dim3_dumps[patch_id] =
                    (unsigned int)fmaxf(0.0f, fminf(15.0f, roundf(scaled)));
            }
            unsigned int* p_payload =
                (unsigned int*)(out_payload + row_id * payload_stride);
            for (int group = 0; group < 8; group++) {
                unsigned int p[8];
                for (int i = 0; i < 8; i++) {
                    int pid = group * 8 + i;
                    p[i] = (s_vq_codes[pid] & 0x7F) | ((s_dim3_codes[pid] & 0x1) << 7)
                        | ((s_dim3_dumps[pid] & 0xF) << 8);
                }
                p_payload[group * 3 + 0] = p[0] | (p[1] << 12) | ((p[2] & 0xFF) << 24);
                p_payload[group * 3 + 1] = (p[2] >> 8) | (p[3] << 4) | (p[4] << 16)
                    | ((p[5] & 0xF) << 28);
                p_payload[group * 3 + 2] = (p[5] >> 4) | (p[6] << 8) | (p[7] << 20);
            }
            unsigned int vq_high_lo = 0, vq_high_hi = 0;
            for (int i = 0; i < 32; i++) vq_high_lo |= ((s_vq_codes[i] >> 7) & 1) << i;
            for (int i = 0; i < 32; i++) vq_high_hi |= ((s_vq_codes[32 + i] >> 7) & 1) << i;
            p_payload[24] = vq_high_lo;
            p_payload[25] = vq_high_hi;
            float sum_true = 0.0f, sum_vq = 0.0f;
            for (int i = 0; i < 64; i++) { sum_true += s_true_energy[i]; sum_vq += s_vq_energy[i]; }
            out_r_norms[row_id] = __float2half(sqrtf(sum_true / (sum_vq + 1e-8f)));
        } else {
            unsigned int total_outliers = s_total_outliers;
            if (total_outliers > 50) total_outliers = 50;
            float c_base_mult = (sum_cbase > 1e-5f) ? (sum_normals / sum_cbase) : 1.0f;
            if (c_base_mult < 0.1f || c_base_mult > 10.0f || isnan(c_base_mult)) c_base_mult = 1.0f;
            float max_residual = 0.0f;
            float residuals[50];
            for (unsigned int j = 0; j < total_outliers; j++) {
                unsigned int dim_i = s_outlier_dim_idx[j];
                float c_base_i = __half2float(centroids[2 * D_val + dim_i]) * c_base_mult;
                float res_j = fmaxf(0.0f, s_outlier_proj_abs[j] - c_base_i);
                residuals[j] = res_j;
                if (res_j > max_residual) max_residual = res_j;
            }
            float residual_scale = (total_outliers > 0)
                ? (fmaxf(max_residual, 1e-4f) / 15.0f) : 0.0f;
            out_r_norms[row_id] = __float2half(residual_scale);
            unsigned char* p_bytes = out_payload + row_id * payload_stride;
            unsigned int c_mult_uint = (unsigned int)(c_base_mult * 1000.0f);
            *(unsigned int*)(p_bytes + 0) = c_mult_uint;
            for (int k = 0; k < 25; k++) p_bytes[4 + k] = 0;
            for (unsigned int j = 0; j < total_outliers; j++) {
                if (residual_scale > 0.0f) {
                    float r_4 = roundf(residuals[j] / residual_scale);
                    unsigned char r_val = (unsigned char)fmaxf(0.0f, fminf(15.0f, r_4));
                    p_bytes[4 + (j / 2)] |= (r_val << ((j % 2) * 4));
                }
            }
            for (int k = 0; k < 7; k++) p_bytes[29 + k] = 0;
            for (unsigned int j = 0; j < total_outliers; j++) {
                if (s_outlier_signs[j]) p_bytes[29 + (j / 8)] |= (1 << (j % 8));
            }
            p_bytes[36] = 0x02;
            p_bytes[37] = 0; p_bytes[38] = 0; p_bytes[39] = 0;
            unsigned int* p_sm = (unsigned int*)(out_shared_meta + row_id * 4);
            p_sm[0] = 0;
        }
    }
}

// Fused attention atas KV terkompresi (CUDA_ATTN_SOURCE). 32 thread/block,
// grid = num_queries (B*H*Lq). Output (N, D+2): acc unnormalized + max_s + sum_exp.
__global__ void khq_attn_kernel(
    const __half* __restrict__ q_p,
    const unsigned char* __restrict__ k_payload,
    const unsigned char* __restrict__ v_payload,
    const unsigned char* __restrict__ v_shared_meta,
    const unsigned int* __restrict__ k_mask,
    const __half* __restrict__ k_norms,
    const __half* __restrict__ k_r_norms,
    const __half* __restrict__ v_norms,
    const __half* __restrict__ v_r_norms,
    const __half* __restrict__ centroids,
    const __half* __restrict__ v_centroids,
    const __half* __restrict__ vq_centroids,
    const __half* __restrict__ k_rotor_params,
    const __half* __restrict__ v_rotor_params,
    const __half* __restrict__ d_vec_k,
    const __half* __restrict__ d_vec_v,
    float* __restrict__ out_attn,
    int D_v, int c_len, int h_kv, float sc,
    float a_k_param, float a_v_param, int n_rep,
    int num_queries, int num_heads, int L_q, float rope_base,
    int start_pos)
{
    const uint total_id = blockIdx.x;
    if (total_id >= (uint)num_queries) return;
    const uint head_idx = total_id / (uint)L_q;
    const uint l_idx = total_id % (uint)L_q;
    const uint tid = threadIdx.x;
    if (head_idx >= (uint)num_heads) return;
    const uint head_idx_kv = head_idx / (uint)n_rep;

    const __half* q_ptr = q_p + (total_id * D_v);

    float freqs[8];
    for (int j = 0; j < 8; j++) freqs[j] = 1.0f;
    if (tid < 8) {
        for (int j = 0; j < 8; j++) {
            uint m = (tid & 3) * 8 + j;
            freqs[j] = powf(rope_base, -2.0f * (float)m / 64.0f);
        }
    }

    float q_val[8];
    for (int j = 0; j < 8; j++) q_val[j] = __half2float(q_ptr[tid * 8 + j]);

    float sum_exp = 0.0f, max_s = -1e20f;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float k_val_fwht[8];

    uint group = tid / 4;
    uint p_v_offset = group * 3;

    float c_val_0[2], c_val_1[2];
    const __half* thread_vq_codebook[2];
    for (int p = 0; p < 2; p++) {
        uint patch_id = tid * 2 + p;
        thread_vq_codebook[p] = vq_centroids + patch_id * 256 * 3;
        c_val_0[p] = __half2float(v_centroids[0 * D_v + patch_id * 4 + 3]) * a_v_param;
        c_val_1[p] = __half2float(v_centroids[1 * D_v + patch_id * 4 + 3]) * a_v_param;
    }

    int s_limit = (L_q > 1) ? min(c_len, (int)l_idx + 1) : c_len;
    int start_s = 0;

    float local_c_base[8];
    float local_d_vec_k[8];
    float f_s_inv_k = 1.0f / 256.0f;
    for (int j = 0; j < 8; j++) {
        uint bit_idx = tid * 8 + j;
        local_c_base[j] = __half2float(centroids[2 * 256 + bit_idx]);
        local_d_vec_k[j] = f_s_inv_k / (__half2float(d_vec_k[bit_idx]) + 1e-8f);
    }

    float local_r_mat[2][16];
    for (int p = 0; p < 2; p++) {
        uint patch_id = tid * 2 + p;
        const __half* r_mat = k_rotor_params + patch_id * 16;
        for (int i = 0; i < 16; i++) local_r_mat[p][i] = __half2float(r_mat[i]);
    }

    float current_cos[8], current_sin[8], step_cos[8], step_sin[8];
    for (int j = 0; j < 8; j++) {
        float start_theta = ((float)(start_pos + start_s)) * freqs[j];
        current_cos[j] = cosf(start_theta);
        current_sin[j] = sinf(start_theta);
        step_cos[j] = cosf(freqs[j]);
        step_sin[j] = sinf(freqs[j]);
    }

    uint g_kv = start_s * h_kv + head_idx_kv;
    float norm_val_k = 0.0f, norm_val_v = 0.0f, energy_correction = 0.0f, sm_scale_val = 0.0f;
    uint word0 = 0, word1 = 0, word2 = 0;
    uint vq_high_lo = 0, vq_high_hi = 0;
    if (s_limit > start_s) {
        norm_val_k = __half2float(k_norms[g_kv]);
        norm_val_v = __half2float(v_norms[g_kv]);
        energy_correction = __half2float(v_r_norms[g_kv]);
        const float* sm_scale_ptr = (const float*)(v_shared_meta + g_kv * 4);
        sm_scale_val = sm_scale_ptr[0];
        const uint32_t* p_v = (const uint32_t*)(v_payload + g_kv * 104);
        word0 = p_v[p_v_offset + 0];
        word1 = p_v[p_v_offset + 1];
        word2 = p_v[p_v_offset + 2];
        vq_high_lo = p_v[24];
        vq_high_hi = p_v[25];
    }

    for (int s = start_s; s < s_limit; s++) {
        if ((s - start_s) > 0 && (s - start_s) % 256 == 0) {
            for (int j = 0; j < 8; j++) {
                float true_theta = ((float)(start_pos + s)) * freqs[j];
                current_cos[j] = cosf(true_theta);
                current_sin[j] = sinf(true_theta);
            }
        }

        float residual_scale = __half2float(k_r_norms[g_kv]);
        const unsigned char* p_k_bytes = (const unsigned char*)(k_payload + g_kv * 40);
        float c_base_mult = (float)(*(const uint32_t*)(p_k_bytes + 0)) / 1000.0f;
        if (c_base_mult < 0.1f || c_base_mult > 10.0f || isnan(c_base_mult)) c_base_mult = 1.0f;
        const unsigned char* residual_arr = p_k_bytes + 4;
        const unsigned char* signs_bits = p_k_bytes + 29;

        uint base_outlier_idx = 0;
        for (uint w = 0; w < tid / 2; w++) {
            uint32_t mw = k_mask[g_kv * 16 + w];
            base_outlier_idx += __popc((mw & (mw >> 1)) & 0x55555555);
        }
        if ((tid % 2) == 1) {
            uint32_t my_word = k_mask[g_kv * 16 + tid / 2];
            base_outlier_idx += __popc((my_word & (my_word >> 1)) & 0x00005555);
        }
        uint local_outlier = base_outlier_idx;

        for (int j = 0; j < 8; j++) {
            uint bit_idx = tid * 8 + j;
            uint k_bit = (k_mask[g_kv * 16 + bit_idx / 16] >> ((bit_idx % 16) * 2)) & 3;
            float c_base = local_c_base[j] * c_base_mult;
            float k_val = 0.0f;
            if (k_bit == 3) {
                uint j_out = local_outlier;
                float sign = (j_out < 50 && ((signs_bits[j_out / 8] >> (j_out % 8)) & 1)) ? 1.0f : -1.0f;
                float res_4bit = (j_out < 50)
                    ? (float)((residual_arr[j_out / 2] >> ((j_out % 2) * 4)) & 0x0F) : 0.0f;
                k_val = sign * (c_base + res_4bit * residual_scale);
                local_outlier++;
            } else if (k_bit == 2) {
                k_val = c_base;
            } else if (k_bit == 1) {
                k_val = -c_base;
            }
            k_val_fwht[j] = k_val * a_k_param * norm_val_k;
        }

        float new_k[8];
        for (int p = 0; p < 2; p++) {
            for (int c = 0; c < 4; c++) {
                float inv_rot_k = 0.0f;
                for (uint k = 0; k < 4; k++)
                    inv_rot_k += k_val_fwht[p * 4 + k] * local_r_mat[p][c * 4 + k];
                new_k[p * 4 + c] = inv_rot_k;
            }
        }
        for (int j = 0; j < 8; j++) k_val_fwht[j] = new_k[j];

        for (int j = 0; j < 8; j += 2) {
            float a = k_val_fwht[j], b = k_val_fwht[j + 1];
            k_val_fwht[j] = a + b; k_val_fwht[j + 1] = a - b;
        }
        for (int base4 = 0; base4 < 8; base4 += 4) {
            for (int k = 0; k < 2; k++) {
                float a = k_val_fwht[base4 + k], b = k_val_fwht[base4 + k + 2];
                k_val_fwht[base4 + k] = a + b; k_val_fwht[base4 + k + 2] = a - b;
            }
        }
        for (int j = 0; j < 4; j++) {
            float a = k_val_fwht[j], b = k_val_fwht[j + 4];
            k_val_fwht[j] = a + b; k_val_fwht[j + 4] = a - b;
        }

        #define BUTTERFLY_K(lane_mask, bit_check)                                   \
        {                                                                           \
            bool is_lower = (tid & (bit_check)) == 0;                               \
            for (int j = 0; j < 8; j++) {                                           \
                float pt = __shfl_xor_sync(0xFFFFFFFFu, k_val_fwht[j], lane_mask, 32); \
                k_val_fwht[j] = is_lower ? (k_val_fwht[j] + pt) : (pt - k_val_fwht[j]); \
            }                                                                       \
        }
        BUTTERFLY_K(1, 1)
        BUTTERFLY_K(2, 2)
        BUTTERFLY_K(4, 4)
        BUTTERFLY_K(8, 8)
        BUTTERFLY_K(16, 16)
        #undef BUTTERFLY_K

        for (int j = 0; j < 8; j++) k_val_fwht[j] = k_val_fwht[j] * local_d_vec_k[j];

        for (int j = 0; j < 8; j++) {
            float x_other = __shfl_xor_sync(0xFFFFFFFFu, k_val_fwht[j], 4, 32);
            if (tid < 8) {
                float cos_val = current_cos[j];
                float sin_val = current_sin[j];
                if (tid < 4) k_val_fwht[j] = k_val_fwht[j] * cos_val - x_other * sin_val;
                else k_val_fwht[j] = x_other * sin_val + k_val_fwht[j] * cos_val;
                current_cos[j] = cos_val * step_cos[j] - sin_val * step_sin[j];
                current_sin[j] = sin_val * step_cos[j] + cos_val * step_sin[j];
            }
        }

        float p_sum = 0.0f;
        for (int j = 0; j < 8; j++) p_sum += q_val[j] * k_val_fwht[j];
        float current_qk = p_sum;
        for (int offset = 16; offset > 0; offset /= 2)
            current_qk += __shfl_down_sync(0xFFFFFFFFu, current_qk, offset);
        current_qk = __shfl_sync(0xFFFFFFFFu, current_qk, 0);

        uint next_s = s + 1;
        float next_norm_val_k = 0, next_norm_val_v = 0, next_energy_correction = 0, next_sm_scale_val = 0;
        uint next_word0 = 0, next_word1 = 0, next_word2 = 0;
        uint next_vq_high_lo = 0, next_vq_high_hi = 0;
        uint next_g_kv = next_s * h_kv + head_idx_kv;
        if ((int)next_s < s_limit) {
            next_norm_val_k = __half2float(k_norms[next_g_kv]);
            next_norm_val_v = __half2float(v_norms[next_g_kv]);
            next_energy_correction = __half2float(v_r_norms[next_g_kv]);
            const float* next_sm = (const float*)(v_shared_meta + next_g_kv * 4);
            next_sm_scale_val = next_sm[0];
            const uint32_t* next_p_v = (const uint32_t*)(v_payload + next_g_kv * 104);
            next_word0 = next_p_v[p_v_offset + 0];
            next_word1 = next_p_v[p_v_offset + 1];
            next_word2 = next_p_v[p_v_offset + 2];
            next_vq_high_lo = next_p_v[24];
            next_vq_high_hi = next_p_v[25];
        }

        float s_val = current_qk * sc;
        float score = 0.0f;
        if (s_val > max_s) {
            float fac = expf(max_s - s_val);
            sum_exp = sum_exp * fac + 1.0f;
            for (int j = 0; j < 8; j++) acc[j] *= fac;
            max_s = s_val; score = 1.0f;
        } else {
            float e = expf(s_val - max_s);
            sum_exp += e; score = e;
        }

        float v_val[8];
        float pre_v_scale = a_v_param * norm_val_v * energy_correction;
        float pre_res_scale = sm_scale_val * norm_val_v;
        for (int p = 0; p < 2; p++) {
            uint patch_id = tid * 2 + p;
            uint sub_id = patch_id % 8;
            uint64_t low64 = ((uint64_t)word1 << 32) | word0;
            uint64_t high64 = ((uint64_t)word2 << 32) | word1;
            uint p_val = (sub_id < 4)
                ? ((low64 >> (sub_id * 12)) & 0xFFF)
                : ((high64 >> ((sub_id - 4) * 12 + 16)) & 0xFFF);
            uint high_bit = (patch_id < 32)
                ? ((vq_high_lo >> patch_id) & 1)
                : ((vq_high_hi >> (patch_id - 32)) & 1);
            uint vq_code = (p_val & 0x7F) | (high_bit << 7);
            uint dim3_code = (p_val >> 7) & 0x1;
            uint dim3_dump = (p_val >> 8) & 0xF;
            for (int k = 0; k < 3; k++)
                v_val[p * 4 + k] = __half2float(thread_vq_codebook[p][vq_code * 3 + k]) * pre_v_scale;
            float c_val = (dim3_code == 0) ? c_val_0[p] : c_val_1[p];
            v_val[p * 4 + 3] = (c_val * norm_val_v)
                + (((float)dim3_dump - 8.0f) / 7.0f * pre_res_scale);
        }
        for (int j = 0; j < 8; j++) acc[j] += score * v_val[j];

        if ((int)next_s < s_limit) {
            g_kv = next_g_kv;
            norm_val_k = next_norm_val_k;
            norm_val_v = next_norm_val_v;
            energy_correction = next_energy_correction;
            sm_scale_val = next_sm_scale_val;
            word0 = next_word0; word1 = next_word1; word2 = next_word2;
            vq_high_lo = next_vq_high_lo;
            vq_high_hi = next_vq_high_hi;
        }
    }

    float new_x[8];
    for (int p = 0; p < 2; p++) {
        uint patch_id = tid * 2 + p;
        const __half* r_mat = v_rotor_params + patch_id * 16;
        for (int c = 0; c < 4; c++) {
            float inv_rot_v = 0.0f;
            for (uint k = 0; k < 4; k++)
                inv_rot_v += acc[p * 4 + k] * __half2float(r_mat[c * 4 + k]);
            new_x[p * 4 + c] = inv_rot_v;
        }
    }
    for (int j = 0; j < 8; j++) acc[j] = new_x[j];

    for (int j = 0; j < 8; j += 2) {
        float a = acc[j], b = acc[j + 1];
        acc[j] = a + b; acc[j + 1] = a - b;
    }
    for (int base4 = 0; base4 < 8; base4 += 4) {
        for (int k = 0; k < 2; k++) {
            float a = acc[base4 + k], b = acc[base4 + k + 2];
            acc[base4 + k] = a + b; acc[base4 + k + 2] = a - b;
        }
    }
    for (int j = 0; j < 4; j++) {
        float a = acc[j], b = acc[j + 4];
        acc[j] = a + b; acc[j + 4] = a - b;
    }

    #define BUTTERFLY_SIMD_INV(lane_mask, bit_check)                            \
    {                                                                           \
        bool is_lower = (tid & (bit_check)) == 0;                               \
        for (int j = 0; j < 8; j++) {                                           \
            float pt = __shfl_xor_sync(0xFFFFFFFFu, acc[j], lane_mask, 32);     \
            acc[j] = is_lower ? (acc[j] + pt) : (pt - acc[j]);                  \
        }                                                                       \
    }
    BUTTERFLY_SIMD_INV(1, 1)
    BUTTERFLY_SIMD_INV(2, 2)
    BUTTERFLY_SIMD_INV(4, 4)
    BUTTERFLY_SIMD_INV(8, 8)
    BUTTERFLY_SIMD_INV(16, 16)
    #undef BUTTERFLY_SIMD_INV

    float f_s_inv = 1.0f / 16.0f;
    for (int j = 0; j < 8; j++) {
        out_attn[total_id * (D_v + 2) + tid * 8 + j] =
            (acc[j] * f_s_inv) / (__half2float(d_vec_v[tid * 8 + j]) + 1e-8f);
    }
    if (tid == 0) {
        out_attn[total_id * (D_v + 2) + D_v] = max_s;
        out_attn[total_id * (D_v + 2) + D_v + 1] = sum_exp;
    }
}

// Attention jendela raw (fp16 ring) — output UNNORMALIZED (acc + max_s + sum_exp)
// supaya bisa digabung dengan region terkompresi via logsumexp (V21.23).
// Satu block per (query head); K/V sudah roped di ring, jadi tanpa RoPE di sini.
__global__ void khq_window_attn_kernel(
    const __half* __restrict__ q,        // (H_q, D) — q_buf tanpa gate (stride D)
    const __half* __restrict__ ring_k,   // (H_kv, RING, D)
    const __half* __restrict__ ring_v,   // (H_kv, RING, D)
    float* __restrict__ out_attn,        // (H_q, D+2) float
    int win_start, int win_len,          // slot awal + panjang jendela
    int ring_cap, int H_q, int H_kv, int D_v, float sc)
{
    const uint h = blockIdx.x;
    if (h >= (uint)H_q) return;
    const uint tid = threadIdx.x;
    const int n_rep = H_q / H_kv;
    const uint hkv = h / n_rep;
    const int per = (D_v + blockDim.x - 1) / blockDim.x;
    // Reduksi QK harus mencakup SELURUH D_v. Shuffle hanya menjumlah dalam satu
    // warp (32 lane), jadi dengan blockDim=256 (8 warp) hasilnya cuma 32 dari
    // 256 dim dan tiap warp memakai skor berbeda. Gabungkan antar-warp di SMEM.
    __shared__ float s_warp[32];
    __shared__ float s_qk;
    const int nwarps = blockDim.x >> 5;
    const uint lane = tid & 31u;

    float sum_exp = 0.0f, max_s = -1e20f;
    float acc[8];
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    float qv[8];
    for (int i = 0; i < 8; i++) {
        int idx = tid + i * blockDim.x;
        qv[i] = (idx < D_v) ? __half2float(q[h * D_v + idx]) : 0.0f;
    }

    for (int s = 0; s < win_len; s++) {
        int slot = (win_start + s) % ring_cap;
        // ring ditulis token-major kontigu (lihat khq_step): slot token menempati
        // [slot*H_kv, (slot+1)*H_kv) baris, head ke-h pada offset (slot*H_kv+h)*D.
        const __half* krow = ring_k + ((size_t)slot * H_kv + hkv) * D_v;
        const __half* vrow = ring_v + ((size_t)slot * H_kv + hkv) * D_v;
        float p_sum = 0.0f;
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < D_v) p_sum += qv[i] * __half2float(krow[idx]);
        }
        float qk = p_sum;
        for (int off = 16; off > 0; off /= 2)
            qk += __shfl_down_sync(0xFFFFFFFFu, qk, off);
        if (lane == 0) s_warp[tid >> 5] = qk;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < nwarps; w++) tot += s_warp[w];
            s_qk = tot;
        }
        __syncthreads();
        qk = s_qk;

        float s_val = qk * sc;
        float score;
        if (s_val > max_s) {
            float fac = expf(max_s - s_val);
            sum_exp = sum_exp * fac + 1.0f;
            for (int i = 0; i < 8; i++) acc[i] *= fac;
            max_s = s_val;
            score = 1.0f;
        } else {
            score = expf(s_val - max_s);
            sum_exp += score;
        }
        for (int i = 0; i < 8; i++) {
            int idx = tid + i * blockDim.x;
            if (idx < D_v) acc[i] += score * __half2float(vrow[idx]);
        }
    }

    for (int i = 0; i < 8; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < D_v) out_attn[h * (D_v + 2) + idx] = acc[i];
    }
    if (tid == 0) {
        out_attn[h * (D_v + 2) + D_v] = max_s;
        out_attn[h * (D_v + 2) + D_v + 1] = sum_exp;
    }
}

// Pisahkan Q dari buffer interleaved [H_q, 2D] (q di 0..D, gate di D..2D)
// menjadi Q kontigu [H_q, D] yang dibutuhkan kernel KHQ.
__global__ void khq_gather_q_kernel(
    const __half* __restrict__ q_gate,   // (H_q, 2D)
    __half* __restrict__ q_out,          // (H_q, D)
    int H_q, int D_v)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_q * D_v) return;
    int h = i / D_v;
    int d = i % D_v;
    q_out[i] = q_gate[h * 2 * D_v + d];
}

// Merge dua region attention (terkompresi + jendela raw) via logsumexp,
// lalu terapkan sigmoid gate dan tulis fp16. Bila has_w=0 hanya region kompresi.
__global__ void khq_merge_gate_kernel(
    const float* __restrict__ attn_c,   // (H_q, D+2)
    const float* __restrict__ attn_w,   // (H_q, D+2) atau null
    const __half* __restrict__ q_gate,  // (H_q, 2D): gate di D..2D per head
    __half* __restrict__ out,           // (H_q, D)
    int H_q, int D_v, int has_w)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_q * D_v) return;
    int h = i / D_v;
    int d = i % D_v;
    const float* cc = attn_c + h * (D_v + 2);
    float num = cc[d];
    float den = cc[D_v + 1];
    if (has_w) {
        const float* cw = attn_w + h * (D_v + 2);
        float mc = cc[D_v], mw = cw[D_v];
        float M = fmaxf(mc, mw);
        float wc = expf(mc - M);
        float ww = expf(mw - M);
        num = num * wc + cw[d] * ww;
        den = den * wc + cw[D_v + 1] * ww;
    }
    float val = num / (den + 1e-8f);
    float g = __half2float(q_gate[h * 2 * D_v + D_v + d]);
    val = val * (1.0f / (1.0f + expf(-g)));
    out[i] = __float2half(val);
}

// ============================================================================
// CALIB GPU — kalibrasi KHQ full-GPU (pengganti loop host calib.mojo).
// Pola FWHT: 1 warp per baris (in-register 8 elem + 5 butterfly shfl) —
// urutan step sama dengan fwht_rows host (bit 0..7), output natural order.
// ============================================================================

// base = fwht(x*d/norm) * (1/sqrt(256)); norm keluar (f32). 1 warp per baris.
__global__ void cal_fwht_base_kernel(
    const float* __restrict__ x, const float* __restrict__ d,
    float* __restrict__ out_norm, float* __restrict__ base, int rows)
{
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;
    for (int r = warp; r < rows; r += total_warps) {
        float q[8];
        float sq = 0.0f;
        for (int j = 0; j < 8; j++) {
            int i = r * 256 + lane * 8 + j;
            float v = x[i];
            q[j] = v;
            sq += v * v;
        }
        for (int off = 16; off > 0; off /= 2)
            sq += __shfl_down_sync(0xFFFFFFFFu, sq, off);
        float norm = __shfl_sync(0xFFFFFFFFu, sq, 0);
        norm = sqrtf(norm + 1e-8f);
        if (lane == 0) out_norm[r] = norm;
        for (int j = 0; j < 8; j++) {
            int i = r * 256 + lane * 8 + j;
            q[j] = (q[j] * __half2float(__float2half(d[lane * 8 + j]))) / (norm + 1e-8f);
        }
        for (int j = 0; j < 8; j += 2) {
            float a = q[j], b = q[j + 1];
            q[j] = a + b; q[j + 1] = a - b;
        }
        for (int j = 0; j < 8; j += 4) {
            float a = q[j], b = q[j + 2];
            q[j] = a + b; q[j + 2] = a - b;
        }
        for (int j = 0; j < 4; j++) {
            float a = q[j], b = q[j + 4];
            q[j] = a + b; q[j + 4] = a - b;
        }
        #define CAL_BFLY(mask)                                                  \
        {                                                                       \
            bool lo = (lane & (mask)) == 0;                                     \
            for (int j = 0; j < 8; j++) {                                       \
                float pt = __shfl_xor_sync(0xFFFFFFFFu, q[j], (mask), 32);      \
                q[j] = lo ? (q[j] + pt) : (pt - q[j]);                          \
            }                                                                   \
        }
        CAL_BFLY(1) CAL_BFLY(2) CAL_BFLY(4) CAL_BFLY(8) CAL_BFLY(16)
        #undef CAL_BFLY
        const float FS = rsqrtf(256.0f);
        for (int j = 0; j < 8; j++)
            base[r * 256 + lane * 8 + j] = q[j] * FS;
    }
}

// rot[n,p,c] = Σ_k in[n,p*4+k] * rp[p*16+k*4+c]
__global__ void cal_rotate_kernel(
    const float* __restrict__ in, const float* __restrict__ rp,
    float* __restrict__ out, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * 256;
    if (i >= total) return;
    int n = i >> 8, p = (i >> 2) & 63, c = i & 3;
    float acc = 0.0f;
    for (int k = 0; k < 4; k++)
        acc += in[n * 256 + p * 4 + k] * rp[p * 16 + k * 4 + c];
    out[i] = acc;
}

// VQ STE per (n,p): dist 256, argmin hard, softmax(-dist/temp), rec hard,
// idx simpan, dictionary loss atomik. wbuf[(n*64+p)*256+k].
__global__ void cal_vq_ste_kernel(
    const float* __restrict__ rot, const float* __restrict__ cb,
    float* __restrict__ rec, float* __restrict__ wbuf,
    int* __restrict__ idx_all, float* __restrict__ loss_dict,
    int rows, float temp, int need_grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * 64;
    if (i >= total) return;
    int n = i >> 6, p = i & 63;
    const float* rp3 = rot + n * 256 + p * 4;
    const float* cbase = cb + p * 768;
    float v0 = rp3[0], v1 = rp3[1], v2 = rp3[2];
    float best = 1e30f;
    int bk = 0;
    float zmax = -1e30f;
    for (int k = 0; k < 256; k++) {
        float d0 = v0 - cbase[k * 3 + 0];
        float d1 = v1 - cbase[k * 3 + 1];
        float d2 = v2 - cbase[k * 3 + 2];
        float dd = d0 * d0 + d1 * d1 + d2 * d2;
        if (dd < best) { best = dd; bk = k; }
        float z = -dd / temp;
        if (z > zmax) zmax = z;
    }
    idx_all[n * 64 + p] = bk;
    float dict = 0.0f;
    for (int d = 0; d < 3; d++) {
        float diff = cbase[bk * 3 + d] - rp3[d];
        dict += diff * diff;
        rec[n * 256 + p * 4 + d] = cbase[bk * 3 + d];
    }
    if (need_grad) {
        float zsum = 0.0f;
        float* wb = wbuf + (n * 64 + p) * 256;
        for (int k = 0; k < 256; k++) {
            float d0 = v0 - cbase[k * 3 + 0];
            float d1 = v1 - cbase[k * 3 + 1];
            float d2 = v2 - cbase[k * 3 + 2];
            float e = expf(-(d0 * d0 + d1 * d1 + d2 * d2) / temp - zmax);
            wb[k] = e;
            zsum += e;
        }
        float winv = 1.0f / zsum;
        for (int k = 0; k < 256; k++) wb[k] *= winv;
    }
    atomicAdd(loss_dict, dict / (float)(rows * 64 * 3));
}

// dim3 STE: rec[n,p,3] = round_half_even(rot*16)/16
__global__ void cal_dim3_kernel(
    const float* __restrict__ rot, float* __restrict__ rec, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int n = i >> 6, p = i & 63;
    float x = rot[n * 256 + p * 4 + 3] * 16.0f;
    float f = floorf(x);
    float diff = x - f;
    float r;
    if (diff > 0.5f) r = f + 1.0f;
    else if (diff < 0.5f) r = f;
    else {
        float half = f * 0.5f;
        r = (half == floorf(half)) ? f : f + 1.0f;
    }
    rec[n * 256 + p * 4 + 3] = r / 16.0f;
}

// inverse-rotate per patch (referensi __call__ step 6: v_fwht_rec = rec_rot @ Rt):
// dst[n,p,c] = Σ_k src[n,p,k] * rp[p*16 + c*4 + k]
__global__ void cal_invrot_kernel(
    const float* __restrict__ src, const float* __restrict__ rp,
    float* __restrict__ dst, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int n = i >> 6, p = i & 63;
    float q[4];
    for (int c = 0; c < 4; c++) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++)
            acc += src[n * 256 + p * 4 + k] * rp[p * 16 + c * 4 + k];
        q[c] = acc;
    }
    for (int c = 0; c < 4; c++) dst[n * 256 + p * 4 + c] = q[c];
}

// W_trm maju: dlt = rec + (rec@A)@B − v
__global__ void cal_wtrm_fwd_kernel(
    const float* __restrict__ rec, const float* __restrict__ wa,
    const float* __restrict__ wb, const float* __restrict__ v,
    float* __restrict__ dlt, float* __restrict__ va, int rows)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rows) return;
    for (int r = 0; r < 8; r++) {
        float acc = 0.0f;
        for (int j = 0; j < 256; j++)
            acc += rec[n * 256 + j] * wa[j * 8 + r];
        va[n * 8 + r] = acc;
    }
    for (int d = 0; d < 256; d++) {
        float acc = rec[n * 256 + d];
        for (int r = 0; r < 8; r++)
            acc += va[n * 8 + r] * wb[r * 256 + d];
        dlt[n * 256 + d] = acc - v[n * 256 + d];
    }
}

// y[h,t,d] = Σ_s A[h,t,s] * dlt[(kv(h)*T+s)*256+d]; loss atomik
__global__ void cal_attn_fwd_kernel(
    const float* __restrict__ attn, const float* __restrict__ dlt,
    float* __restrict__ y, float* __restrict__ loss,
    int T, int H_q, int n_rep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H_q * T * 256;
    if (i >= total) return;
    int d = i & 255;
    int t = (i >> 8) % T;
    int h = i / (T * 256);
    int kv = h / n_rep;
    const float* ah = attn + (long)h * T * T + (long)t * T;
    float acc = 0.0f;
    for (int s = 0; s < T; s++)
        acc += ah[s] * dlt[((long)kv * T + s) * 256 + d];
    y[((long)h * T + t) * 256 + d] = acc;
    atomicAdd(loss, acc * acc / (float)(H_q * T * 256));
}

// gVc[(kv*T+t)*256+d] += 2/(H_q*T*256) * Σ_s A[h,s,t] * y[(h*T+s)*256+d]
__global__ void cal_attn_bwd_kernel(
    const float* __restrict__ attn, const float* __restrict__ y,
    float* __restrict__ gvc, int T, int H_q, int n_rep, float gnorm)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)H_q * T * 256;
    if (i >= total) return;
    int d = i & 255;
    int t = (i >> 8) % T;
    int h = i / (T * 256);
    int kv = h / n_rep;
    float acc = 0.0f;
    for (int s = 0; s < T; s++)
        acc += attn[((long)h * T + s) * T + t] * y[((long)h * T + s) * 256 + d];
    atomicAdd(&gvc[((long)kv * T + t) * 256 + d], acc * gnorm);
}

// grec = gvc + (gvc@B)@Aᵀ ; gW = recᵀ@gvc (thread per (j,d))
__global__ void cal_wtrm_bwd_kernel(
    const float* __restrict__ gvc, const float* __restrict__ rec,
    const float* __restrict__ wa, const float* __restrict__ wb,
    float* __restrict__ grec, float* __restrict__ gvab,
    float* __restrict__ gw, int rows)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < rows) {
        for (int r = 0; r < 8; r++) {
            float acc = 0.0f;
            for (int d = 0; d < 256; d++)
                acc += gvc[n * 256 + d] * wb[r * 256 + d];
            gvab[n * 8 + r] = acc;
        }
        for (int j = 0; j < 256; j++) {
            float acc = gvc[n * 256 + j];
            for (int r = 0; r < 8; r++)
                acc += gvab[n * 8 + r] * wa[j * 8 + r];
            grec[n * 256 + j] = acc;
        }
    }
    // gw bagian: thread (j,d) grid kedua dipisah kernel agar sederhana
}

__global__ void cal_gw_kernel(
    const float* __restrict__ rec, const float* __restrict__ gvc,
    float* __restrict__ gw, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256 * 256) return;
    int j = i >> 8, d = i & 255;
    float acc = 0.0f;
    for (int n = 0; n < rows; n++)
        acc += rec[n * 256 + j] * gvc[n * 256 + d];
    gw[i] = acc;
}

// gwa = gW@wbᵀ (256x8), gwb = waᵀ@gW (8x256)
__global__ void cal_gwab_kernel(
    const float* __restrict__ gw, const float* __restrict__ wa,
    const float* __restrict__ wb,
    float* __restrict__ gwa, float* __restrict__ gwb)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256 * 8) {
        int j = i >> 3, r = i & 7;
        float acc = 0.0f;
        for (int d = 0; d < 256; d++)
            acc += gw[j * 256 + d] * wb[r * 256 + d];
        gwa[i] = acc;
    } else if (i < 256 * 8 + 8 * 256) {
        int i2 = i - 256 * 8;
        int r = i2 >> 8, d = i2 & 255;
        float acc = 0.0f;
        for (int j = 0; j < 256; j++)
            acc += wa[j * 8 + r] * gw[j * 256 + d];
        gwb[i2] = acc;
    }
}

// gqs[n,p,c] = Σ_d grec[n,p,d] * rp[p*16+d*4+c], c=0..2 (dim3 dibuang)
__global__ void cal_rotbwd_kernel(
    const float* __restrict__ grec, const float* __restrict__ rp,
    float* __restrict__ gqs, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int n = i >> 6, p = i & 63;
    for (int c = 0; c < 3; c++) {
        float acc = 0.0f;
        for (int d = 0; d < 3; d++)
            acc += grec[n * 256 + p * 4 + d] * rp[p * 16 + d * 4 + c];
        gqs[(n * 64 + p) * 3 + c] = acc;
    }
}

// swg[n] = Σ_k w[n,k]*gWk[n,k]; gWk disimpan di buffer dist (reuse)
__global__ void cal_swg_kernel(
    const float* __restrict__ gqs, const float* __restrict__ cb,
    const float* __restrict__ wbuf, float* __restrict__ gwk,
    float* __restrict__ swg, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int n = i >> 6, p = i & 63;
    const float* gq = gqs + (n * 64 + p) * 3;
    const float* cbase = cb + p * 768;
    const float* wb = wbuf + (n * 64 + p) * 256;
    float* gk = gwk + (n * 64 + p) * 256;
    float acc = 0.0f;
    for (int k = 0; k < 256; k++) {
        float v = gq[0] * cbase[k * 3 + 0] + gq[1] * cbase[k * 3 + 1]
            + gq[2] * cbase[k * 3 + 2];
        gk[k] = v;
        acc += wb[k] * v;
    }
    swg[n * 64 + p] = acc;
}

// gC[k,0..2] += w*gq + (−w*(gWk−swg)/temp)*2*(C−rot); dict: idx cluster
__global__ void cal_cgrad_kernel(
    const float* __restrict__ wbuf, const float* __restrict__ gwk,
    const float* __restrict__ swg, const float* __restrict__ gqs,
    const float* __restrict__ rot, const float* __restrict__ cb,
    const int* __restrict__ idx_all, float* __restrict__ gcb,
    int rows, float temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64 * 256) return;
    int k = i & 255;
    int np = i >> 8;
    int n = np >> 6, p = np & 63;
    const float* cbase = cb + p * 768 + k * 3;
    float w = wbuf[i];
    float gq0 = gqs[np * 3 + 0], gq1 = gqs[np * 3 + 1], gq2 = gqs[np * 3 + 2];
    float gdist = -(w * (gwk[i] - swg[np])) / temp;
    float c0 = cbase[0], c1 = cbase[1], c2 = cbase[2];
    const float* r3 = rot + n * 256 + p * 4;
    atomicAdd(&gcb[p * 768 + k * 3 + 0], w * gq0 + gdist * 2.0f * (c0 - r3[0]));
    atomicAdd(&gcb[p * 768 + k * 3 + 1], w * gq1 + gdist * 2.0f * (c1 - r3[1]));
    atomicAdd(&gcb[p * 768 + k * 3 + 2], w * gq2 + gdist * 2.0f * (c2 - r3[2]));
    if (k == 0) {
        int bk = idx_all[np];
        float dn = 2.0f / (float)(rows * 64 * 3);
        atomicAdd(&gcb[p * 768 + bk * 3 + 0],
                  dn * (cb[p * 768 + bk * 3 + 0] - r3[0]));
        atomicAdd(&gcb[p * 768 + bk * 3 + 1],
                  dn * (cb[p * 768 + bk * 3 + 1] - r3[1]));
        atomicAdd(&gcb[p * 768 + bk * 3 + 2],
                  dn * (cb[p * 768 + bk * 3 + 2] - r3[2]));
    }
}

// Adam elemenwise (MLX: m,v bias-corrected)
__global__ void cal_adam_kernel(
    float* __restrict__ p, const float* __restrict__ g,
    float* __restrict__ m, float* __restrict__ v,
    float lr, float bc1, float bc2, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    m[i] = 0.9f * m[i] + 0.1f * gi;
    v[i] = 0.999f * v[i] + 0.001f * gi * gi;
    p[i] -= lr * (m[i] / bc1) / (sqrtf(v[i] / bc2) + 1e-8f);
}

// noise warm-restart: cb = best + randn*scale (xorshift per elemen)
__global__ void cal_noise_kernel(
    const float* __restrict__ best, float* __restrict__ p,
    unsigned int seed, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int s = seed ^ (unsigned int)i * 747796405u + 2891336453u;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float u1 = (float)(s & 0xFFFFFF) / 16777216.0f + 1e-7f;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float u2 = (float)(s & 0xFFFFFF) / 16777216.0f;
    float g = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    p[i] = best[i] + g * scale;
}

// mode parseval (tanpa attn): grad + loss rekonstruksi langsung
__global__ void cal_parseval_grad_kernel(
    const float* __restrict__ rec, const float* __restrict__ v,
    float* __restrict__ grec, float* __restrict__ loss, int rows, int with_grad)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 256) return;
    float diff = rec[i] - v[i];
    if (with_grad) grec[i] = 2.0f * diff / (float)(rows * 256);
    atomicAdd(loss, diff * diff / (float)(rows * 256));
}

// W_trm = I + A@B (256x256) — dipakai sekali di akhir utk output
__global__ void cal_wtrm_compose_kernel(
    const float* __restrict__ wa, const float* __restrict__ wb,
    float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256 * 256) return;
    int j = i >> 8, d = i & 255;
    float acc = (j == d) ? 1.0f : 0.0f;
    for (int r = 0; r < 8; r++)
        acc += wa[j * 8 + r] * wb[r * 256 + d];
    out[i] = acc;
}

#define CDIV(a, b) (((long)(a) + (b) - 1) / (b))

namespace calsvq {

struct SeedState {
    float *cb, *wa, *wb;
    float *mcb, *vcb, *mwa, *vwa, *mwb, *vwb;
    float *best_cb, *best_wa, *best_wb;
    float best_loss;
    int stagnant;
    float lr_mult;
    int step_cb, step_trm;
    unsigned int seed;
};

// forward + loss (recon + dict_w*dict) tanpa gradien
static float forward_only(
    const float* v, const float* attn, const float* rp, SeedState& st,
    float* rot, float* rec, float* wbuf, int* idx_all, float* dlt,
    float* va, float* y, float* loss_buf,
    int N, int T, int H_q, int H_kv, float temp, float dict_w,
    cudaStream_t s)
{
    int n_rep = H_q / H_kv;
    cudaMemsetAsync(loss_buf, 0, 8, s);
    cal_rotate_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(v, rp, rot, N);
    cal_vq_ste_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(
        rot, st.cb, rec, wbuf, idx_all, loss_buf + 1, N, temp, 0);
    cal_dim3_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(rot, rec, N);
    // inverse-rotate ke domain FWHT (v_fwht_rec referensi) sebelum W_trm/loss
    cal_invrot_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(rec, rp, dlt, N);
    if (attn) {
        cal_wtrm_fwd_kernel<<<CDIV(N, 256), 256, 0, s>>>(
            dlt, st.wa, st.wb, v, rec, va, N);
        cal_attn_fwd_kernel<<<CDIV((long)H_q * T * 256, 256), 256, 0, s>>>(
            attn, rec, y, loss_buf, T, H_q, n_rep);
    } else {
        cal_parseval_grad_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(
            dlt, v, nullptr, loss_buf, N, 0);
    }
    float h[2];
    cudaMemcpyAsync(h, loss_buf, 8, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    return h[0] + dict_w * h[1];
}

// satu ronde training (temp/lr schedule + Adam + patience + warm restart)
static void run_seed(
    SeedState& st, const float* v, const float* attn, const float* rp,
    float* rot, float* rec, float* wbuf, int* idx_all, float* dlt,
    float* va, float* gvab, float* y, float* gvc, float* grec, float* gw,
    float* gqs, float* swg, float* gwk, float* gcb, float* gwa, float* gwb,
    float* loss_buf,
    int N, int T, int H_q, int H_kv, int iters, int start_iter, float lr,
    cudaStream_t s)
{
    int n_rep = H_q / H_kv;
    int total = start_iter + iters;
    int max_iters = total - 1;
    if (max_iters < 1) max_iters = 1;
    for (int i = start_iter; i < total; i++) {
        float temp = 1.5f * expf((logf(0.1f) - logf(1.5f))
            * (float)i / (float)max_iters);
        float sched = lr * fmaxf(0.1f, 1.0f - (float)i / (float)total);
        float cur = sched * st.lr_mult;
        float lr_trm = cur * 0.05f;

        cudaMemsetAsync(gcb, 0, 49152 * 4, s);
        cudaMemsetAsync(gwa, 0, 2048 * 4, s);
        cudaMemsetAsync(gwb, 0, 2048 * 4, s);
        if (attn) cudaMemsetAsync(gvc, 0, (size_t)N * 256 * 4, s);
        cudaMemsetAsync(loss_buf, 0, 8, s);

        cal_rotate_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(v, rp, rot, N);
        cal_vq_ste_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(
            rot, st.cb, rec, wbuf, idx_all, loss_buf + 1, N, temp, 1);
        cal_dim3_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(rot, rec, N);
        // inverse-rotate ke domain FWHT (v_fwht_rec referensi); gwk dipakai
        // sebagai rec_fwht sampai cal_swg menimpanya
        cal_invrot_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(rec, rp, gwk, N);
        if (attn) {
            cal_wtrm_fwd_kernel<<<CDIV(N, 256), 256, 0, s>>>(
                gwk, st.wa, st.wb, v, dlt, va, N);
            cal_attn_fwd_kernel<<<CDIV((long)H_q * T * 256, 256), 256, 0, s>>>(
                attn, dlt, y, loss_buf, T, H_q, n_rep);
        } else {
            cal_parseval_grad_kernel<<<CDIV(N * 256, 256), 256, 0, s>>>(
                gwk, v, grec, loss_buf, N, 1);
        }

        if (attn) {
            cal_attn_bwd_kernel<<<CDIV((long)H_q * T * 256, 256), 256, 0, s>>>(
                attn, y, gvc, T, H_q, n_rep, 2.0f / (float)(H_q * T * 256));
            cal_wtrm_bwd_kernel<<<CDIV(N, 256), 256, 0, s>>>(
                gvc, gwk, st.wa, st.wb, grec, gvab, gw, N);
            cal_gw_kernel<<<CDIV(256 * 256, 256), 256, 0, s>>>(
                gwk, gvc, gw, N);
            cal_gwab_kernel<<<CDIV(256 * 8 + 8 * 256, 256), 256, 0, s>>>(
                gw, st.wa, st.wb, gwa, gwb);
        }
        cal_rotbwd_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(grec, rp, gqs, N);
        cal_swg_kernel<<<CDIV(N * 64, 256), 256, 0, s>>>(
            gqs, st.cb, wbuf, gwk, swg, N);
        cal_cgrad_kernel<<<CDIV((long)N * 64 * 256, 256), 256, 0, s>>>(
            wbuf, gwk, swg, gqs, rot, st.cb, idx_all, gcb, N, temp);

        st.step_cb++; st.step_trm++;
        float bc1 = 1.0f - powf(0.9f, (float)st.step_cb);
        float bc2 = 1.0f - powf(0.999f, (float)st.step_cb);
        float bc1t = 1.0f - powf(0.9f, (float)st.step_trm);
        float bc2t = 1.0f - powf(0.999f, (float)st.step_trm);
        cal_adam_kernel<<<CDIV(49152, 256), 256, 0, s>>>(
            st.cb, gcb, st.mcb, st.vcb, cur, bc1, bc2, 49152);
        cal_adam_kernel<<<CDIV(2048, 256), 256, 0, s>>>(
            st.wa, gwa, st.mwa, st.vwa, lr_trm, bc1t, bc2t, 2048);
        cal_adam_kernel<<<CDIV(2048, 256), 256, 0, s>>>(
            st.wb, gwb, st.mwb, st.vwb, lr_trm, bc1t, bc2t, 2048);

        float h[2];
        cudaMemcpyAsync(h, loss_buf, 8, cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        float loss = h[0] + h[1];

        if (loss < st.best_loss - 1e-7f) {
            st.best_loss = loss;
            cudaMemcpyAsync(st.best_cb, st.cb, 49152 * 4,
                            cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(st.best_wa, st.wa, 2048 * 4,
                            cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(st.best_wb, st.wb, 2048 * 4,
                            cudaMemcpyDeviceToDevice, s);
            st.stagnant = 0;
        } else {
            st.stagnant++;
        }
        if (st.stagnant >= 15) {
            if (st.lr_mult > 0.1f) {
                st.lr_mult *= 0.5f;
            } else {
                st.lr_mult = 3.0f;
                float progress = (float)i / (float)max_iters;
                float ns = 1e-3f * expf(logf(0.01f) * progress);
                cal_noise_kernel<<<CDIV(49152, 256), 256, 0, s>>>(
                    st.best_cb, st.cb, st.seed ^ (unsigned int)i, ns, 49152);
                cudaMemcpyAsync(st.wa, st.best_wa, 2048 * 4,
                                cudaMemcpyDeviceToDevice, s);
                cudaMemcpyAsync(st.wb, st.best_wb, 2048 * 4,
                                cudaMemcpyDeviceToDevice, s);
            }
            st.stagnant = 0;
        }
        if ((i + 1) % 50 == 0 || i == start_iter)
            printf("      SmartVQ iter %d | loss %.6f | temp %.4f\n",
                   i + 1, loss, temp);
    }
}

} // namespace calsvq


// ============================================================================
// CALIB GPU fase 2 — K static, SmartK, PCA, kmeans, Procrustes, SmartV.
// Semua putaran berat dari calib.mojo dipindah ke T4; Mojo tinggal IO.
// ============================================================================

#define CAL_FS 0.0625f
#define CAL_OUT_MAX 50

__device__ __forceinline__ unsigned int cal_upk32(
    const unsigned char* p, int off)
{
    return (unsigned int)p[off] | ((unsigned int)p[off + 1] << 8)
        | ((unsigned int)p[off + 2] << 16) | ((unsigned int)p[off + 3] << 24);
}

__device__ __forceinline__ void cal_pk32(
    unsigned char* p, int off, unsigned int v)
{
    p[off] = (unsigned char)(v & 0xFF);
    p[off + 1] = (unsigned char)((v >> 8) & 0xFF);
    p[off + 2] = (unsigned char)((v >> 16) & 0xFF);
    p[off + 3] = (unsigned char)((v >> 24) & 0xFF);
}

__device__ __forceinline__ float cal_fp16(float x)
{
    return __half2float(__float2half(x));
}

__global__ void cal_fp16_inplace_kernel(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = cal_fp16(p[i]);
}

__global__ void cal_fp16_round_kernel(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __half2float(__float2half(p[i]));
}

__global__ void cal_div_scalar_kernel(
    const float* __restrict__ in, float* __restrict__ out, float s, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / s;
}

// cents = fp16(base * ts) — kandidat SmartK
__global__ void cal_cents_scale_kernel(
    const float* __restrict__ base, float* __restrict__ out, float ts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 1024) out[i] = cal_fp16(base[i] * ts);
}

// ---- K static: sigma per dim + partial sum_exp; finish: ts + cents ----
__global__ void cal_kstatic_p_kernel(
    const float* __restrict__ rot, float* __restrict__ sigma,
    float* __restrict__ psum, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;
    float m = 0.0f;
    for (int r = 0; r < rows; r++) m += rot[r * 256 + i];
    m /= (float)rows;
    float v = 0.0f;
    for (int r = 0; r < rows; r++) {
        float dv = rot[r * 256 + i] - m;
        v += dv * dv;
    }
    float sg = sqrtf(v / (float)rows);
    sigma[i] = sg;
    float cb = 0.9350f * sg;
    psum[i] = 0.528756f * cb * cb + 0.464545f * sg * sg;
}

__global__ void cal_kstatic_f_kernel(
    const float* __restrict__ sigma, const float* __restrict__ psum,
    float* __restrict__ cents, float* __restrict__ out_ts)
{
    float se = 0.0f;
    for (int i = 0; i < 256; i++) se += psum[i];
    float ts = 16.0f / sqrtf(se);
    for (int i = 0; i < 256; i++) {
        float cb = 0.9350f * sigma[i] * ts;
        cents[0 * 256 + i] = 0.0f;
        cents[1 * 256 + i] = -cb;
        cents[2 * 256 + i] = cb;
        cents[3 * 256 + i] = cb;
    }
    out_ts[0] = ts;
}

// ---- dimensi stat: mag/var (sebelum normalisasi) ----
__global__ void cal_dimstats_kernel(
    const float* __restrict__ x, float* __restrict__ dim_mag,
    float* __restrict__ dim_var, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;
    float m = 0.0f, ma = 0.0f;
    for (int r = 0; r < rows; r++) {
        float v = x[r * 256 + i];
        m += v;
        ma += fabsf(v);
    }
    m /= (float)rows;
    dim_mag[i] = ma / (float)rows;
    float vv = 0.0f;
    for (int r = 0; r < rows; r++) {
        float dv = x[r * 256 + i] - m;
        vv += dv * dv;
    }
    dim_var[i] = vv / (float)rows;
}

__global__ void cal_dimstats_norm_kernel(
    float* __restrict__ dim_mag, float* __restrict__ dim_var)
{
    float sm = 0.0f, sv = 0.0f;
    for (int i = 0; i < 256; i++) {
        sm += dim_mag[i];
        sv += dim_var[i];
    }
    sm += 1e-8f;
    sv += 1e-8f;
    for (int i = 0; i < 256; i++) {
        dim_mag[i] /= sm;
        dim_var[i] /= sv;
    }
}

// ---- compress K (persis compress_k host; thread per baris) ----
__global__ void cal_compress_k_kernel(
    const float* __restrict__ proj, const float* __restrict__ cents,
    float ts, const float* __restrict__ norms,
    unsigned int* __restrict__ mask, float* __restrict__ r_norms,
    unsigned char* __restrict__ payload, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* pr = proj + (size_t)row * 256;
    const float* cb = cents + 2 * 256;
    unsigned int word = 0;
    float sum_n = 0.0f, sum_c = 0.0f;
    int out_dim[CAL_OUT_MAX];
    float out_abs[CAL_OUT_MAX];
    int out_sign[CAL_OUT_MAX];
    float res[CAL_OUT_MAX];
    int total = 0;
    for (int i = 0; i < 256; i++) {
        float proj_v = pr[i];
        float p_scaled = proj_v * ts;
        float abs_proj = fabsf(proj_v);
        unsigned int b = 0;
        if (abs_proj > 1.60f * CAL_FS) {
            if (total < CAL_OUT_MAX) {
                out_dim[total] = i;
                out_abs[total] = fabsf(p_scaled);
                out_sign[total] = (proj_v > 0.0f) ? 1 : 0;
                total++;
            }
            b = (proj_v > 0.0f) ? 2u : 1u;
        } else if (proj_v > 0.47f * CAL_FS) {
            b = 2u;
        } else if (proj_v < -0.47f * CAL_FS) {
            b = 1u;
        }
        if (b == 1u || b == 2u) {
            sum_n += abs_proj;
            sum_c += cal_fp16(cb[i]);
        }
        word |= b << ((i % 16) * 2);
        if (i % 16 == 15) {
            mask[row * 16 + i / 16] = word;
            word = 0u;
        }
    }
    float res_scale = 0.0f;
    float mult = 1.0f;
    if (sum_c > 1e-5f) mult = sum_n / sum_c;
    if (mult < 0.1f || mult > 10.0f || mult != mult) mult = 1.0f;
    float max_res = 0.0f;
    for (int j = 0; j < total; j++) {
        float cbi = cal_fp16(cb[out_dim[j]]) * mult;
        float rj = out_abs[j] - cbi;
        if (rj < 0.0f) rj = 0.0f;
        res[j] = rj;
        if (rj > max_res) max_res = rj;
    }
    if (total > 0) res_scale = fmaxf(max_res, 1e-4f) / 15.0f;
    r_norms[row] = res_scale;
    unsigned char* pb = payload + (size_t)row * 40;
    for (int k = 0; k < 40; k++) pb[k] = 0;
    cal_pk32(pb, 0, (unsigned int)(mult * 1000.0f));
    for (int j = 0; j < total; j++) {
        if (res_scale > 0.0f) {
            int r4 = (int)roundf(res[j] / res_scale);
            if (r4 < 0) r4 = 0;
            if (r4 > 15) r4 = 15;
            pb[4 + j / 2] |= (unsigned char)r4 << ((j % 2) * 4);
        }
    }
    for (int j = 0; j < total; j++) {
        if (out_sign[j] == 1)
            pb[29 + j / 8] |= (unsigned char)1 << (j % 8);
    }
    pb[36] = 0x02;
}

// ---- decompress K (persis decompress_k host; thread per baris) ----
__global__ void cal_decompress_k_kernel(
    const float* __restrict__ cents, const float* __restrict__ rp,
    const float* __restrict__ d, float alpha,
    const float* __restrict__ norms, const unsigned int* __restrict__ mask,
    const float* __restrict__ r_norms,
    const unsigned char* __restrict__ payload,
    float* __restrict__ out, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned char* pb = payload + (size_t)row * 40;
    float mult = (float)cal_upk32(pb, 0) / 1000.0f;
    if (mult < 0.1f || mult > 10.0f) mult = 1.0f;
    float kv[256];
    float tmp[256];
    int seen = 0;
    for (int i = 0; i < 256; i++) {
        unsigned int word = mask[row * 16 + i / 16];
        unsigned int bit = (word >> ((i % 16) * 2)) & 3u;
        float c_base = cents[2 * 256 + i] * mult;
        float v = 0.0f;
        if (bit == 3u) {
            int j = seen;
            float sign = -1.0f;
            if (j < CAL_OUT_MAX
                && ((pb[29 + j / 8] >> (j % 8)) & 1) == 1)
                sign = 1.0f;
            float res4 = 0.0f;
            if (j < CAL_OUT_MAX)
                res4 = (float)((pb[4 + j / 2] >> ((j % 2) * 4)) & 0x0F);
            v = sign * (c_base + res4 * r_norms[row]);
            seen++;
        } else if (bit == 2u) {
            v = c_base;
        } else if (bit == 1u) {
            v = -c_base;
        }
        kv[i] = v * alpha * norms[row];
    }
    // rotor inverse: dst[p*4+c] = Σ_k kv[p*4+k]·rp[p*16 + c*4 + k]
    for (int p = 0; p < 64; p++) {
        float q[4];
        for (int c = 0; c < 4; c++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++)
                acc += kv[p * 4 + k] * rp[p * 16 + c * 4 + k];
            q[c] = acc;
        }
        for (int c = 0; c < 4; c++) tmp[p * 4 + c] = q[c];
    }
    // FWHT 256 (urutan step 1..128, natural order)
    for (int step = 1; step < 256; step *= 2) {
        for (int i = 0; i < 256; i += step * 2) {
            float a = tmp[i], b = tmp[i + step];
            tmp[i] = a + b;
            tmp[i + step] = a - b;
        }
    }
    for (int i = 0; i < 256; i++)
        out[(size_t)row * 256 + i] = tmp[i] / (256.0f * d[i]);
}

// ---- compress V (persis compress_v host; thread per baris) ----
__global__ void cal_compress_v_kernel(
    const float* __restrict__ proj, const float* __restrict__ cents,
    const float* __restrict__ vq, float ts, float alpha,
    const float* __restrict__ norms, float* __restrict__ r_norms,
    unsigned char* __restrict__ payload, float* __restrict__ smeta,
    int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* pr = proj + (size_t)row * 256;
    float th = ts * CAL_FS;
    unsigned int vq_codes[64];
    unsigned int d3_codes[64];
    unsigned int d3_dumps[64];
    float pres[64];
    float te[64], ve[64];
    for (int p = 0; p < 64; p++) {
        float v0 = pr[p * 4 + 0];
        float v1 = pr[p * 4 + 1];
        float v2 = pr[p * 4 + 2];
        int bki = 0;
        float bd = 1e30f;
        for (int k = 0; k < 256; k++) {
            float c0 = cal_fp16(vq[p * 768 + k * 3 + 0]) * alpha;
            float c1 = cal_fp16(vq[p * 768 + k * 3 + 1]) * alpha;
            float c2 = cal_fp16(vq[p * 768 + k * 3 + 2]) * alpha;
            float d0 = v0 - c0, d1 = v1 - c1, d2 = v2 - c2;
            float dd = d0 * d0 + d1 * d1 + d2 * d2;
            if (dd < bd) {
                bd = dd;
                bki = k;
            }
        }
        vq_codes[p] = (unsigned int)bki;
        float c0b = cal_fp16(vq[p * 768 + bki * 3 + 0]) * alpha;
        float c1b = cal_fp16(vq[p * 768 + bki * 3 + 1]) * alpha;
        float c2b = cal_fp16(vq[p * 768 + bki * 3 + 2]) * alpha;
        te[p] = v0 * v0 + v1 * v1 + v2 * v2;
        ve[p] = c0b * c0b + c1b * c1b + c2b * c2b;
        float v3 = pr[p * 4 + 3];
        float cv0 = cal_fp16(cents[0 * 256 + p * 4 + 3]) * alpha;
        float cv1 = cal_fp16(cents[1 * 256 + p * 4 + 3]) * alpha;
        float d0 = fabsf(v3 - cv0);
        float d1 = fabsf(v3 - cv1);
        if (d0 < d1) {
            d3_codes[p] = 0;
            pres[p] = v3 - cv0;
        } else {
            d3_codes[p] = 1;
            pres[p] = v3 - cv1;
        }
    }
    float g_max = 0.0f;
    for (int p = 0; p < 64; p++) {
        float a = fabsf(pres[p]);
        if (a > g_max) g_max = a;
    }
    smeta[row] = g_max;
    float rscale = fmaxf(g_max, 1e-4f);
    for (int p = 0; p < 64; p++) {
        int rv = (int)roundf(pres[p] / rscale * 7.0f + 8.0f);
        if (rv < 0) rv = 0;
        if (rv > 15) rv = 15;
        d3_dumps[p] = (unsigned int)rv;
    }
    unsigned char* pb = payload + (size_t)row * 104;
    for (int k = 0; k < 104; k++) pb[k] = 0;
    for (int group = 0; group < 8; group++) {
        unsigned int pv[8];
        for (int i = 0; i < 8; i++) {
            int pid = group * 8 + i;
            pv[i] = (vq_codes[pid] & 0x7Fu)
                | ((d3_codes[pid] & 0x1u) << 7)
                | ((d3_dumps[pid] & 0xFu) << 8);
        }
        cal_pk32(pb, (group * 3 + 0) * 4,
                 pv[0] | (pv[1] << 12) | ((pv[2] & 0xFFu) << 24));
        cal_pk32(pb, (group * 3 + 1) * 4,
                 (pv[2] >> 8) | (pv[3] << 4) | (pv[4] << 16)
                     | ((pv[5] & 0xFu) << 28));
        cal_pk32(pb, (group * 3 + 2) * 4,
                 (pv[5] >> 4) | (pv[6] << 8) | (pv[7] << 20));
    }
    unsigned int hi_lo = 0, hi_hi = 0;
    for (int i = 0; i < 32; i++) {
        hi_lo |= ((vq_codes[i] >> 7) & 1u) << i;
        hi_hi |= ((vq_codes[32 + i] >> 7) & 1u) << i;
    }
    cal_pk32(pb, 24 * 4, hi_lo);
    cal_pk32(pb, 25 * 4, hi_hi);
    float st = 0.0f, sv = 0.0f;
    for (int p = 0; p < 64; p++) {
        st += te[p];
        sv += ve[p];
    }
    r_norms[row] = sqrtf(st / (sv + 1e-8f));
}

// ---- decompress V (persis decompress_v host; thread per baris) ----
__global__ void cal_decompress_v_kernel(
    const float* __restrict__ cents, const float* __restrict__ vq,
    const float* __restrict__ rp, const float* __restrict__ d, float alpha,
    const float* __restrict__ norms, const float* __restrict__ r_norms,
    const unsigned char* __restrict__ payload,
    const float* __restrict__ smeta,
    float* __restrict__ out, int rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const unsigned char* pb = payload + (size_t)row * 104;
    unsigned int hi_lo = cal_upk32(pb, 24 * 4);
    unsigned int hi_hi = cal_upk32(pb, 25 * 4);
    unsigned int pv[64];
    for (int group = 0; group < 8; group++) {
        unsigned int w0 = cal_upk32(pb, (group * 3 + 0) * 4);
        unsigned int w1 = cal_upk32(pb, (group * 3 + 1) * 4);
        unsigned int w2 = cal_upk32(pb, (group * 3 + 2) * 4);
        pv[group * 8 + 0] = w0 & 0xFFFu;
        pv[group * 8 + 1] = (w0 >> 12) & 0xFFFu;
        pv[group * 8 + 2] = (w0 >> 24) | ((w1 & 0xFu) << 8);
        pv[group * 8 + 3] = (w1 >> 4) & 0xFFFu;
        pv[group * 8 + 4] = (w1 >> 16) & 0xFFFu;
        pv[group * 8 + 5] = (w1 >> 28) | ((w2 & 0xFFu) << 4);
        pv[group * 8 + 6] = (w2 >> 8) & 0xFFFu;
        pv[group * 8 + 7] = (w2 >> 20) & 0xFFFu;
    }
    float vrot[256];
    float tmp[256];
    for (int p = 0; p < 64; p++) {
        unsigned int code_lo = pv[p] & 0x7Fu;
        unsigned int hb = (p < 32) ? ((hi_lo >> p) & 1u)
                                   : ((hi_hi >> (p - 32)) & 1u);
        int code = (int)(code_lo | (hb << 7));
        int d3c = (int)((pv[p] >> 7) & 1u);
        float dump = (float)((pv[p] >> 8) & 0xFu);
        for (int c = 0; c < 3; c++)
            vrot[p * 4 + c] = vq[p * 768 + code * 3 + c] * alpha
                * norms[row] * r_norms[row];
        float cv = cents[d3c * 256 + p * 4 + 3] * alpha;
        vrot[p * 4 + 3] = cv * norms[row]
            + ((dump - 8.0f) / 7.0f) * smeta[row] * norms[row];
    }
    for (int p = 0; p < 64; p++) {
        float q[4];
        for (int c = 0; c < 4; c++) {
            float acc = 0.0f;
            for (int k = 0; k < 4; k++)
                acc += vrot[p * 4 + k] * rp[p * 16 + c * 4 + k];
            q[c] = acc;
        }
        for (int c = 0; c < 4; c++) tmp[p * 4 + c] = q[c];
    }
    for (int step = 1; step < 256; step *= 2) {
        for (int i = 0; i < 256; i += step * 2) {
            float a = tmp[i], b = tmp[i + step];
            tmp[i] = a + b;
            tmp[i + step] = a - b;
        }
    }
    for (int i = 0; i < 256; i++)
        out[(size_t)row * 256 + i] = tmp[i] / (16.0f * d[i]);
}

// ---- MSE 3 jalur + maxerr (block-reduce + atomik) ----
__global__ void cal_mse_kernel(
    const float* __restrict__ x32, const float* __restrict__ rec,
    const float* __restrict__ dim_mag, const float* __restrict__ dim_var,
    float* __restrict__ out4, long elems)
{
    __shared__ float sh[4][256];
    int tid = threadIdx.x;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, mx = 0.0f;
    for (long i = (long)blockIdx.x * blockDim.x + tid; i < elems;
         i += (long)gridDim.x * blockDim.x)
    {
        int di = (int)(i & 255);
        float diff = x32[i] - rec[i];
        float sq = diff * diff;
        s0 += sq;
        s1 += dim_mag[di] * sq;
        s2 += dim_var[di] * sq;
        float ad = fabsf(diff);
        if (ad > mx) mx = ad;
    }
    sh[0][tid] = s0;
    sh[1][tid] = s1;
    sh[2][tid] = s2;
    sh[3][tid] = mx;
    __syncthreads();
    for (int off = 128; off > 0; off /= 2) {
        if (tid < off) {
            sh[0][tid] += sh[0][tid + off];
            sh[1][tid] += sh[1][tid + off];
            sh[2][tid] += sh[2][tid + off];
            if (sh[3][tid + off] > sh[3][tid])
                sh[3][tid] = sh[3][tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&out4[0], sh[0][0]);
        atomicAdd(&out4[1], sh[1][0]);
        atomicAdd(&out4[2], sh[2][0]);
        atomicMax((int*)&out4[3], __float_as_int(sh[3][0]));
    }
}

// ---- PCA per patch (cov + 5 sweep Jacobi + sel kolom [1,2,3,0]) ----
__global__ void cal_pca_kernel(
    const float* __restrict__ base, float* __restrict__ out_rp, int rows)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= 64) return;
    float cov[16];
    float v[16];
    float npc[4], nqc[4];
    for (int i = 0; i < 16; i++) cov[i] = 0.0f;
    for (int r = 0; r < rows; r++) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cov[i * 4 + j] += base[(size_t)r * 256 + p * 4 + i]
                    * base[(size_t)r * 256 + p * 4 + j];
    }
    for (int i = 0; i < 4; i++) cov[i * 4 + i] += 1e-3f;
    for (int i = 0; i < 16; i++) v[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    for (int sweep = 0; sweep < 5; sweep++) {
        for (int pair = 0; pair < 6; pair++) {
            int pi = pair < 3 ? 0 : (pair < 5 ? 1 : 2);
            int qi;
            if (pair == 0) qi = 1;
            else if (pair == 1) qi = 2;
            else if (pair == 2) qi = 3;
            else if (pair == 3) qi = 2;
            else if (pair == 4) qi = 3;
            else qi = 3;
            float apq = cov[pi * 4 + qi];
            float app = cov[pi * 4 + pi];
            float aqq = cov[qi * 4 + qi];
            float phi = 0.5f * atan2f(2.0f * apq, aqq - app);
            float c = cosf(phi), s = sinf(phi);
            for (int i = 0; i < 4; i++) {
                npc[i] = c * cov[i * 4 + pi] - s * cov[i * 4 + qi];
                nqc[i] = s * cov[i * 4 + pi] + c * cov[i * 4 + qi];
            }
            for (int i = 0; i < 4; i++) {
                cov[i * 4 + pi] = npc[i];
                cov[i * 4 + qi] = nqc[i];
            }
            for (int j = 0; j < 4; j++) {
                cov[pi * 4 + j] = npc[j];
                cov[qi * 4 + j] = nqc[j];
            }
            cov[pi * 4 + pi] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
            cov[qi * 4 + qi] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
            cov[pi * 4 + qi] = 0.0f;
            cov[qi * 4 + pi] = 0.0f;
            for (int i = 0; i < 4; i++) {
                float vp = v[i * 4 + pi];
                float vq2 = v[i * 4 + qi];
                v[i * 4 + pi] = c * vp - s * vq2;
                v[i * 4 + qi] = s * vp + c * vq2;
            }
        }
    }
    float eig[4];
    int order[4];
    int sel[4] = {1, 2, 3, 0};
    for (int i = 0; i < 4; i++) {
        eig[i] = cov[i * 4 + i];
        order[i] = i;
    }
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++)
            if (eig[order[j]] > eig[order[i]]) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
    for (int cix = 0; cix < 4; cix++)
        for (int i = 0; i < 4; i++)
            out_rp[p * 16 + i * 4 + cix] = v[i * 4 + order[sel[cix]]];
}

// ---- kmeans 4 per dim (thread per dim, 20 iter Lloyd) ----
__global__ void cal_kmeans4_kernel(
    const float* __restrict__ data, float* __restrict__ out_cents, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256) return;
    float mn = data[i], mx = data[i];
    for (int r = 0; r < rows; r++) {
        float v = data[r * 256 + i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    float c[4];
    for (int k = 0; k < 4; k++)
        c[k] = (mn + (mx - mn) * ((float)k / 3.0f)) * 0.8f;
    for (int it = 0; it < 20; it++) {
        float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float cnts[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int r = 0; r < rows; r++) {
            float v = data[r * 256 + i];
            int bk = 0;
            float bd = (v - c[0]) * (v - c[0]);
            for (int k = 1; k < 4; k++) {
                float dd = (v - c[k]) * (v - c[k]);
                if (dd < bd) {
                    bd = dd;
                    bk = k;
                }
            }
            sums[bk] += v;
            cnts[bk] += 1.0f;
        }
        for (int k = 0; k < 4; k++)
            c[k] = sums[k] / fmaxf(cnts[k], 1.0f);
    }
    for (int k = 0; k < 4; k++) out_cents[k * 256 + i] = c[k];
}

// ---- kmeans VQ 256x3 per patch (atomik) ----
__global__ void cal_kmeans_vq_init_kernel(
    const float* __restrict__ data, float* __restrict__ C,
    unsigned int seed, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 64 * 256) return;
    int p = i >> 8, k = i & 255;
    unsigned int s = seed ^ ((unsigned int)i * 2654435761u + 1u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    int idx = (int)((float)(s & 0xFFFFFF) / 16777216.0f * (float)rows);
    if (idx >= rows) idx = rows - 1;
    for (int c = 0; c < 3; c++)
        C[p * 768 + k * 3 + c] = data[(size_t)idx * 256 + p * 4 + c];
}

__global__ void cal_kmeans_vq_zero_kernel(
    float* __restrict__ sums, float* __restrict__ cnts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 64 * 768) sums[i] = 0.0f;
    if (i < 64 * 256) cnts[i] = 0.0f;
}

__global__ void cal_kmeans_vq_assign_kernel(
    const float* __restrict__ data, const float* __restrict__ C,
    float* __restrict__ sums, float* __restrict__ cnts, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int r = i >> 6, p = i & 63;
    float v0 = data[(size_t)r * 256 + p * 4 + 0];
    float v1 = data[(size_t)r * 256 + p * 4 + 1];
    float v2 = data[(size_t)r * 256 + p * 4 + 2];
    int bk = 0;
    float bd = 1e30f;
    for (int k = 0; k < 256; k++) {
        float d0 = v0 - C[p * 768 + k * 3 + 0];
        float d1 = v1 - C[p * 768 + k * 3 + 1];
        float d2 = v2 - C[p * 768 + k * 3 + 2];
        float dd = d0 * d0 + d1 * d1 + d2 * d2;
        if (dd < bd) {
            bd = dd;
            bk = k;
        }
    }
    atomicAdd(&sums[p * 768 + bk * 3 + 0], v0);
    atomicAdd(&sums[p * 768 + bk * 3 + 1], v1);
    atomicAdd(&sums[p * 768 + bk * 3 + 2], v2);
    atomicAdd(&cnts[p * 256 + bk], 1.0f);
}

__global__ void cal_kmeans_vq_update_kernel(
    float* __restrict__ C, const float* __restrict__ sums,
    const float* __restrict__ cnts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 64 * 256) return;
    int p = i >> 8, k = i & 255;
    float cnt = cnts[i];
    if (cnt > 0.0f) {
        for (int c = 0; c < 3; c++)
            C[p * 768 + k * 3 + c] = sums[p * 768 + k * 3 + c] / cnt;
    }
}

// reorder head-major utk SmartVQ: out[(h*n+t)*256+i] = in[(t*heads+h)*256+i]
__global__ void cal_headmajor_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int n_tokens, int heads)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)n_tokens * heads * 256;
    if (i >= total) return;
    int r = (int)(i >> 8);
    int di = (int)(i & 255);
    int h = r / n_tokens;
    int t = r % n_tokens;
    out[i] = in[((long)t * heads + h) * 256 + di];
}

// ---- Procrustes-Lloyd: assign target + M + NS 12x ----
__global__ void cal_procrustes_assign_kernel(
    const float* __restrict__ rot, const float* __restrict__ vq,
    float* __restrict__ tgt, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * 64) return;
    int r = i >> 6, p = i & 63;
    float v0 = rot[(size_t)r * 256 + p * 4 + 0];
    float v1 = rot[(size_t)r * 256 + p * 4 + 1];
    float v2 = rot[(size_t)r * 256 + p * 4 + 2];
    int bk = 0;
    float bd = 1e30f;
    for (int k = 0; k < 256; k++) {
        float d0 = v0 - vq[p * 768 + k * 3 + 0];
        float d1 = v1 - vq[p * 768 + k * 3 + 1];
        float d2 = v2 - vq[p * 768 + k * 3 + 2];
        float dd = d0 * d0 + d1 * d1 + d2 * d2;
        if (dd < bd) {
            bd = dd;
            bk = k;
        }
    }
    tgt[(size_t)r * 256 + p * 4 + 0] = vq[p * 768 + bk * 3 + 0];
    tgt[(size_t)r * 256 + p * 4 + 1] = vq[p * 768 + bk * 3 + 1];
    tgt[(size_t)r * 256 + p * 4 + 2] = vq[p * 768 + bk * 3 + 2];
    tgt[(size_t)r * 256 + p * 4 + 3] = rot[(size_t)r * 256 + p * 4 + 3];
}

__global__ void cal_procrustes_m_kernel(
    const float* __restrict__ base, const float* __restrict__ tgt,
    float* __restrict__ M, int rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 64 * 16) return;
    int p = i >> 4, ij = i & 15;
    float acc = 0.0f;
    for (int r = 0; r < rows; r++)
        acc += base[(size_t)r * 256 + p * 4 + (ij >> 2)]
            * tgt[(size_t)r * 256 + p * 4 + (ij & 3)];
    M[i] = acc;
}

__global__ void cal_procrustes_ns_kernel(
    const float* __restrict__ M, float* __restrict__ rp)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= 64) return;
    float X[16], XtX[16], term[16], Xo[16];
    float frob = 0.0f;
    for (int i = 0; i < 16; i++) {
        X[i] = M[p * 16 + i];
        frob += X[i] * X[i];
    }
    frob = sqrtf(frob + 1e-8f);
    for (int i = 0; i < 16; i++) X[i] /= frob;
    for (int ns = 0; ns < 12; ns++) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 4; k++)
                    acc += X[k * 4 + i] * X[k * 4 + j];
                XtX[i * 4 + j] = acc;
            }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                term[i * 4 + j] = ((i == j) ? 3.0f : 0.0f) - XtX[i * 4 + j];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                float acc = 0.0f;
                for (int k = 0; k < 4; k++)
                    acc += X[i * 4 + k] * term[k * 4 + j];
                Xo[i * 4 + j] = 0.5f * acc;
            }
        for (int i = 0; i < 16; i++) X[i] = Xo[i];
    }
    bool nan = false;
    for (int i = 0; i < 16; i++)
        if (X[i] != X[i]) nan = true;
    if (!nan)
        for (int i = 0; i < 16; i++) rp[p * 16 + i] = X[i];
}


// ============================================================================
// C ABI Export Functions (Dipanggil via Mojo FFI)
// ============================================================================
extern "C" {

// slot statik utk state modul Mojo (runtime=0, dump=1) — Mojo 25.x tanpa
// global var; pointer struct Mojo diparkir di memori statik lib ini.
static void* g_khq_slots[2] = {nullptr, nullptr};
void* khq_state_slot(int which) {
    if (which < 0 || which > 1) which = 0;
    return (void*)&g_khq_slots[which];
}

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
    __half* sp = reinterpret_cast<__half*>(state);
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

// Kompresi K/V KHQ: N baris (token x head) -> mask/norms/r_norms/payload.
// is_v=0: payload 40 B (K), is_v=1: payload 104 B (V, butuh vq_centroids).
int launch_khq_compress_fp16(
    const void* x,             // (N, D) __half
    const void* d_vec,         // (D) __half
    const void* rotor,         // (NP*16) __half
    const void* centroids,     // (4, D) __half
    const void* vq_centroids,  // (NP*256*3) __half (boleh null utk K)
    void* out_mask,            // (N, 16) uint32
    void* out_norms,           // (N) __half
    void* out_r_norms,         // (N) __half
    void* out_payload,         // (N, 40|104) uint8
    void* out_shared_meta,     // (N, 4) float
    int n, int d, float threshold_scale, float alpha, int is_v,
    cudaStream_t stream)
{
    if (n <= 0 || d != 256) return -100;
    int threads = 256;
    int blocks = min((n + threads - 1) / threads, 4096);
    khq_compress_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(x),
        reinterpret_cast<const __half*>(d_vec),
        reinterpret_cast<const __half*>(rotor),
        reinterpret_cast<const __half*>(centroids),
        reinterpret_cast<const __half*>(vq_centroids),
        reinterpret_cast<unsigned int*>(out_mask),
        reinterpret_cast<__half*>(out_norms),
        reinterpret_cast<__half*>(out_r_norms),
        reinterpret_cast<unsigned char*>(out_payload),
        reinterpret_cast<unsigned char*>(out_shared_meta),
        d, n, threshold_scale, alpha, is_v);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Attention terfusi atas KV terkompresi KHQ. Output (num_queries, D+2) float:
// acc unnormalized + max_s + sum_exp (merge logsumexp dilakukan di Mojo).
int launch_khq_attn_fp16(
    const void* q,               // (num_queries, D) __half
    const void* k_payload,       // (N_kv, 40) uint8
    const void* v_payload,       // (N_kv, 104) uint8
    const void* v_shared_meta,   // (N_kv, 4) float
    const void* k_mask,          // (N_kv, 16) uint32
    const void* k_norms,         // (N_kv) __half
    const void* k_r_norms,       // (N_kv) __half
    const void* v_norms,         // (N_kv) __half
    const void* v_r_norms,       // (N_kv) __half
    const void* centroids_k,     // (4, D) __half
    const void* centroids_v,     // (4, D) __half
    const void* vq_centroids,    // (NP*256*3) __half
    const void* k_rotor,         // (NP*16) __half
    const void* v_rotor,         // (NP*16) __half
    const void* d_vec_k,         // (D) __half
    const void* d_vec_v,         // (D) __half
    void* out_attn,              // (num_queries, D+2) float
    int num_queries, int num_heads, int L_q, int c_len, int stride_s,
    int n_rep, float scale, float alpha_k, float alpha_v,
    float rope_base, int start_pos,
    cudaStream_t stream)
{
    if (num_queries <= 0 || c_len <= 0) return -100;
    khq_attn_kernel<<<dim3(num_queries), dim3(32), 0, stream>>>(
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const unsigned char*>(k_payload),
        reinterpret_cast<const unsigned char*>(v_payload),
        reinterpret_cast<const unsigned char*>(v_shared_meta),
        reinterpret_cast<const unsigned int*>(k_mask),
        reinterpret_cast<const __half*>(k_norms),
        reinterpret_cast<const __half*>(k_r_norms),
        reinterpret_cast<const __half*>(v_norms),
        reinterpret_cast<const __half*>(v_r_norms),
        reinterpret_cast<const __half*>(centroids_k),
        reinterpret_cast<const __half*>(centroids_v),
        reinterpret_cast<const __half*>(vq_centroids),
        reinterpret_cast<const __half*>(k_rotor),
        reinterpret_cast<const __half*>(v_rotor),
        reinterpret_cast<const __half*>(d_vec_k),
        reinterpret_cast<const __half*>(d_vec_v),
        reinterpret_cast<float*>(out_attn),
        256, c_len, (n_rep > 0 ? num_heads / n_rep : 1), scale,
        alpha_k, alpha_v, n_rep,
        num_queries, num_heads, L_q, rope_base, start_pos);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Attention jendela raw fp16 (unnormalized) untuk merge logsumexp.
int launch_khq_window_attn_fp16(
    const void* q,          // (H_q, D) __half (q_buf tanpa gate, stride D)
    const void* ring_k,     // (H_kv, RING, D) __half (sudah RoPE)
    const void* ring_v,     // (H_kv, RING, D) __half
    void* out_attn,         // (H_q, D+2) float
    int win_start, int win_len, int ring_cap,
    int H_q, int H_kv, int d, float scale,
    cudaStream_t stream)
{
    if (H_q <= 0 || win_len <= 0) return -100;
    khq_window_attn_kernel<<<dim3(H_q), dim3(256), 0, stream>>>(
        reinterpret_cast<const __half*>(q),
        reinterpret_cast<const __half*>(ring_k),
        reinterpret_cast<const __half*>(ring_v),
        reinterpret_cast<float*>(out_attn),
        win_start, win_len, ring_cap, H_q, H_kv, d, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Pisah Q dari interleaved [H_q, 2D] -> kontigu [H_q, D].
int launch_khq_gather_q_fp16(
    const void* q_gate, void* q_out, int H_q, int d, cudaStream_t stream)
{
    if (H_q <= 0 || d <= 0) return -100;
    int total = H_q * d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    khq_gather_q_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(q_gate),
        reinterpret_cast<__half*>(q_out), H_q, d);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Merge region terkompresi + jendela raw (logsumexp) + sigmoid gate -> fp16.
int launch_khq_merge_gate_fp16(
    const void* attn_c, const void* attn_w, const void* q_gate, void* out,
    int H_q, int d, int has_w, cudaStream_t stream)
{
    if (H_q <= 0 || d <= 0) return -100;
    int total = H_q * d;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    khq_merge_gate_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const float*>(attn_c),
        reinterpret_cast<const float*>(attn_w),
        reinterpret_cast<const __half*>(q_gate),
        reinterpret_cast<__half*>(out), H_q, d, has_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : static_cast<int>(err);
}

// Turnamen SmartVQ full-GPU (train_smartvq.py): seed [42,137,271],
// ronde1 30 iter + eliminasi top-3 (attn-MSE proxy temp=0.05) + ronde2.
// out_cb = codebook winner (best), out_wtrm = I + A@B (boleh null).
int launch_cal_svq_train(
    const void* v_, const void* attn_, const void* init_cb_, const void* rp_,
    void* out_cb_, void* out_wtrm_,
    int N, int T, int H_q, int H_kv, int iterations, int n_seeds, float lr,
    cudaStream_t stream)
{
    const float* v = reinterpret_cast<const float*>(v_);
    const float* attn = reinterpret_cast<const float*>(attn_);
    const float* init_cb = reinterpret_cast<const float*>(init_cb_);
    const float* rp = reinterpret_cast<const float*>(rp_);
    float* out_cb = reinterpret_cast<float*>(out_cb_);
    float* out_wtrm = reinterpret_cast<float*>(out_wtrm_);
    if (n_seeds > 3) n_seeds = 3;
    if (n_seeds < 1) n_seeds = 1;

    // scratch bersama
    float *rot, *rec, *wbuf, *dlt, *va, *gvab, *y, *gvc, *grec, *gw, *gqs;
    float *swg, *gwk, *gcb, *gwa, *gwb, *loss_buf, *wtrm;
    int* idx_all;
    cudaMalloc(&rot, (size_t)N * 256 * 4);
    cudaMalloc(&rec, (size_t)N * 256 * 4);
    cudaMalloc(&wbuf, (size_t)N * 64 * 256 * 4);
    cudaMalloc(&idx_all, (size_t)N * 64 * 4);
    cudaMalloc(&dlt, (size_t)N * 256 * 4);
    cudaMalloc(&va, (size_t)N * 8 * 4);
    cudaMalloc(&gvab, (size_t)N * 8 * 4);
    cudaMalloc(&y, (size_t)H_q * T * 256 * 4);
    cudaMalloc(&gvc, (size_t)N * 256 * 4);
    cudaMalloc(&grec, (size_t)N * 256 * 4);
    cudaMalloc(&gw, 256 * 256 * 4);
    cudaMalloc(&gqs, (size_t)N * 64 * 3 * 4);
    cudaMalloc(&swg, (size_t)N * 64 * 4);
    cudaMalloc(&gwk, (size_t)N * 64 * 256 * 4);
    cudaMalloc(&gcb, 49152 * 4);
    cudaMalloc(&gwa, 2048 * 4);
    cudaMalloc(&gwb, 2048 * 4);
    cudaMalloc(&loss_buf, 8);
    cudaMalloc(&wtrm, 256 * 256 * 4);

    calsvq::SeedState st[3];
    int seeds[3] = {42, 137, 271};
    for (int s = 0; s < n_seeds; s++) {
        calsvq::SeedState& t = st[s];
        cudaMalloc(&t.cb, 49152 * 4);
        cudaMalloc(&t.wa, 2048 * 4);
        cudaMalloc(&t.wb, 2048 * 4);
        cudaMalloc(&t.mcb, 49152 * 4);
        cudaMalloc(&t.vcb, 49152 * 4);
        cudaMalloc(&t.mwa, 2048 * 4);
        cudaMalloc(&t.vwa, 2048 * 4);
        cudaMalloc(&t.mwb, 2048 * 4);
        cudaMalloc(&t.vwb, 2048 * 4);
        cudaMalloc(&t.best_cb, 49152 * 4);
        cudaMalloc(&t.best_wa, 2048 * 4);
        cudaMalloc(&t.best_wb, 2048 * 4);
        cudaMemsetAsync(t.mcb, 0, 49152 * 4, stream);
        cudaMemsetAsync(t.vcb, 0, 49152 * 4, stream);
        cudaMemsetAsync(t.mwa, 0, 2048 * 4, stream);
        cudaMemsetAsync(t.vwa, 0, 2048 * 4, stream);
        cudaMemsetAsync(t.mwb, 0, 2048 * 4, stream);
        cudaMemsetAsync(t.vwb, 0, 2048 * 4, stream);
        cudaMemsetAsync(t.wa, 0, 2048 * 4, stream);
        cudaMemsetAsync(t.wb, 0, 2048 * 4, stream);
        // W_trm_A ~ randn/sqrt(256); B = 0
        cal_noise_kernel<<<CDIV(2048, 256), 256, 0, stream>>>(
            t.wa, t.wa, seeds[s] * 2654435761u + 1u, 1.0f / 16.0f, 2048);
        cudaMemcpyAsync(t.cb, init_cb, 49152 * 4, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t.best_cb, t.cb, 49152 * 4, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t.best_wa, t.wa, 2048 * 4, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(t.best_wb, t.wb, 2048 * 4, cudaMemcpyDeviceToDevice, stream);
        t.best_loss = calsvq::forward_only(
            v, attn, rp, t, rot, rec, wbuf, idx_all, dlt, va, y, loss_buf,
            N, T, H_q, H_kv, 1.5f, 1.0f, stream);
        t.stagnant = 0;
        t.lr_mult = 1.0f;
        t.step_cb = t.step_trm = 0;
        t.seed = (unsigned int)seeds[s];
    }

    int round1 = (iterations < 30) ? iterations : 30;
    int round2 = iterations - round1;
    for (int s = 0; s < n_seeds; s++) {
        printf("   [SmartVQ] Round 1 seed %d\n", seeds[s]);
        calsvq::run_seed(st[s], v, attn, rp, rot, rec, wbuf, idx_all, dlt,
                         va, gvab, y, gvc, grec, gw, gqs, swg, gwk, gcb,
                         gwa, gwb, loss_buf, N, T, H_q, H_kv,
                         round1, 0, lr, stream);
    }

    // Paritas referensi (train_smartvq.py:399-403): seleksi proxy harus dinilai
    // atas bobot BEST, bukan bobot iterasi terakhir. Optimizer state tetap
    // dibawa ke round 2 (referensi memakai cstate['opt_*'] yang sama).
    for (int s = 0; s < n_seeds; s++) {
        cudaMemcpyAsync(st[s].cb, st[s].best_cb, 49152 * 4,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(st[s].wa, st[s].best_wa, 2048 * 4,
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(st[s].wb, st[s].best_wb, 2048 * 4,
                        cudaMemcpyDeviceToDevice, stream);
    }

    float proxy[3];
    for (int s = 0; s < n_seeds; s++) {
        proxy[s] = calsvq::forward_only(
            v, attn, rp, st[s], rot, rec, wbuf, idx_all, dlt, va, y,
            loss_buf, N, T, H_q, H_kv, 0.05f, 0.0f, stream);
        printf("   [SmartVQ] seed %d attn-MSE proxy: %.8f\n", seeds[s], proxy[s]);
    }
    int order[3] = {0, 1, 2};
    for (int i = 0; i < n_seeds; i++)
        for (int j = i + 1; j < n_seeds; j++)
            if (proxy[order[j]] < proxy[order[i]]) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }
    int top = (n_seeds < 3) ? n_seeds : 3;
    if (round2 > 0) {
        for (int rank = 0; rank < top; rank++) {
            int m = order[rank];
            printf("   [SmartVQ] Round 2 seed %d (proxy %.8f)\n",
                   seeds[m], proxy[m]);
            calsvq::run_seed(st[m], v, attn, rp, rot, rec, wbuf, idx_all,
                             dlt, va, gvab, y, gvc, grec, gw, gqs, swg, gwk,
                             gcb, gwa, gwb, loss_buf, N, T, H_q, H_kv,
                             round2, round1, lr, stream);
            proxy[m] = calsvq::forward_only(
                v, attn, rp, st[m], rot, rec, wbuf, idx_all, dlt, va, y,
                loss_buf, N, T, H_q, H_kv, 0.05f, 0.0f, stream);
            printf("   [SmartVQ] seed %d final proxy: %.8f\n",
                   seeds[m], proxy[m]);
        }
    }
    int winner = order[0];
    for (int rank = 0; rank < top; rank++)
        if (proxy[order[rank]] < proxy[winner]) winner = order[rank];
    printf("   [SmartVQ] winner seed %d proxy %.8f\n",
           seeds[winner], proxy[winner]);

    cudaMemcpyAsync(out_cb, st[winner].best_cb, 49152 * 4,
                    cudaMemcpyDeviceToDevice, stream);
    if (out_wtrm) {
        cal_wtrm_compose_kernel<<<CDIV(256 * 256, 256), 256, 0, stream>>>(
            st[winner].best_wa, st[winner].best_wb, out_wtrm);
    }
    cudaStreamSynchronize(stream);

    for (int s = 0; s < n_seeds; s++) {
        cudaFree(st[s].cb); cudaFree(st[s].wa); cudaFree(st[s].wb);
        cudaFree(st[s].mcb); cudaFree(st[s].vcb);
        cudaFree(st[s].mwa); cudaFree(st[s].vwa);
        cudaFree(st[s].mwb); cudaFree(st[s].vwb);
        cudaFree(st[s].best_cb); cudaFree(st[s].best_wa);
        cudaFree(st[s].best_wb);
    }
    cudaFree(rot); cudaFree(rec); cudaFree(wbuf); cudaFree(idx_all);
    cudaFree(dlt); cudaFree(va); cudaFree(gvab); cudaFree(y);
    cudaFree(gvc); cudaFree(grec); cudaFree(gw); cudaFree(gqs);
    cudaFree(swg); cudaFree(gwk); cudaFree(gcb); cudaFree(gwa);
    cudaFree(gwb); cudaFree(loss_buf); cudaFree(wtrm);
    return 0;
}

// Turnamen SmartK/SmartV + semua langkah kalibrasi non-SmartVQ.
// op: 1=FWHT-base 2=K-static 3=rotate 4=PCA 5=kmeans4 6=kmeans-VQ
//     7=Procrustes 8=SmartK 9=SmartV
int launch_cal_op(
    int op,
    const void* a_, const void* b_, const void* c_, const void* d_,
    void* e_, void* f_, void* g_,
    float f1, float f2,
    int i1, int i2, int i3, int i4,
    cudaStream_t stream)
{
    const float* a = reinterpret_cast<const float*>(a_);
    const float* b = reinterpret_cast<const float*>(b_);
    const float* c = reinterpret_cast<const float*>(c_);
    const float* d = reinterpret_cast<const float*>(d_);
    float* e = reinterpret_cast<float*>(e_);
    float* f = reinterpret_cast<float*>(f_);
    float* g = reinterpret_cast<float*>(g_);
    int rows = i1;

    if (op == 1) { // FWHT base: base = fwht(x*d/norm)*fs, norm keluar
        cal_fwht_base_kernel<<<CDIV((long)rows * 64, 256), 256, 0, stream>>>(
            a, b, e, f, rows);
        return 0;
    }
    if (op == 3) { // rotate
        cal_rotate_kernel<<<CDIV((long)rows * 256, 256), 256, 0, stream>>>(
            a, b, e, rows);
        return 0;
    }
    if (op == 4) { // PCA per patch
        cal_pca_kernel<<<CDIV(64, 64), 64, 0, stream>>>(a, e, rows);
        return 0;
    }
    if (op == 5) { // kmeans4 per dim
        cal_kmeans4_kernel<<<CDIV(256, 64), 64, 0, stream>>>(a, e, rows);
        return 0;
    }
    if (op == 2) { // K static: sigma/ts/cents
        float *sigma, *psum;
        cudaMalloc(&sigma, 256 * 4);
        cudaMalloc(&psum, 256 * 4);
        cal_kstatic_p_kernel<<<CDIV(256, 64), 64, 0, stream>>>(a, sigma, psum, rows);
        cal_kstatic_f_kernel<<<1, 1, 0, stream>>>(sigma, psum, f, g);
        cudaStreamSynchronize(stream);
        cudaFree(sigma);
        cudaFree(psum);
        return 0;
    }
    if (op == 6) { // kmeans VQ 256x3 (30 iter)
        float *sums, *cnts;
        cudaMalloc(&sums, (size_t)64 * 768 * 4);
        cudaMalloc(&cnts, (size_t)64 * 256 * 4);
        cal_kmeans_vq_init_kernel<<<CDIV(64 * 256, 256), 256, 0, stream>>>(
            a, e, (unsigned int)i2, rows);
        for (int it = 0; it < 30; it++) {
            cal_kmeans_vq_zero_kernel<<<CDIV(64 * 768, 256), 256, 0, stream>>>(
                sums, cnts);
            cal_kmeans_vq_assign_kernel<<<CDIV((long)rows * 64, 256), 256, 0, stream>>>(
                a, e, sums, cnts, rows);
            cal_kmeans_vq_update_kernel<<<CDIV(64 * 256, 256), 256, 0, stream>>>(
                e, sums, cnts);
        }
        cudaStreamSynchronize(stream);
        cudaFree(sums);
        cudaFree(cnts);
        return 0;
    }
    if (op == 7) { // Procrustes-Lloyd i2 iterasi
        float *rot, *tgt, *M;
        cudaMalloc(&rot, (size_t)rows * 256 * 4);
        cudaMalloc(&tgt, (size_t)rows * 256 * 4);
        cudaMalloc(&M, (size_t)64 * 16 * 4);
        for (int it = 0; it < i2; it++) {
            cal_rotate_kernel<<<CDIV((long)rows * 256, 256), 256, 0, stream>>>(
                a, e, rot, rows);
            cal_procrustes_assign_kernel<<<CDIV((long)rows * 64, 256), 256, 0, stream>>>(
                rot, b, tgt, rows);
            cal_procrustes_m_kernel<<<CDIV(64 * 16, 256), 256, 0, stream>>>(
                a, tgt, M, rows);
            cal_procrustes_ns_kernel<<<CDIV(64, 64), 64, 0, stream>>>(M, e);
        }
        cudaStreamSynchronize(stream);
        cudaFree(rot);
        cudaFree(tgt);
        cudaFree(M);
        return 0;
    }
    if (op == 10) { // head-major reorder (in=a, out=e, n=i1, heads=i2)
        cal_headmajor_kernel<<<CDIV((long)i1 * i2 * 256, 256), 256, 0, stream>>>(
            a, e, i1, i2);
        return 0;
    }
    if (op == 8 || op == 9) { // turnamen SmartK / SmartV
        const float* x32 = a;
        const float* x16 = b;
        const float* dv = c;
        const float* rp = d;
        const float* cents_in = e;
        const float* vq_in = f;
        bool is_k = (op == 8);
        int sub = rows;
        float *dim_mag, *dim_var, *proj, *rec, *norms, *rnorms, *smeta;
        float *cents_cur, *base_c, *out4;
        unsigned int* mask;
        unsigned char* payload;
        cudaMalloc(&dim_mag, 256 * 4);
        cudaMalloc(&dim_var, 256 * 4);
        cudaMalloc(&proj, (size_t)sub * 256 * 4);
        cudaMalloc(&rec, (size_t)sub * 256 * 4);
        cudaMalloc(&norms, (size_t)sub * 4);
        cudaMalloc(&rnorms, (size_t)sub * 4);
        cudaMalloc(&smeta, (size_t)sub * 4);
        cudaMalloc(&cents_cur, 1024 * 4);
        cudaMalloc(&out4, 16);
        if (is_k) cudaMalloc(&base_c, 1024 * 4);
        if (is_k) cudaMalloc(&mask, (size_t)sub * 16 * 4);
        else mask = nullptr;
        cudaMalloc(&payload, (size_t)sub * (is_k ? 40 : 104));

        cal_dimstats_kernel<<<CDIV(256, 64), 64, 0, stream>>>(x32, dim_mag, dim_var, sub);
        cal_dimstats_norm_kernel<<<1, 1, 0, stream>>>(dim_mag, dim_var);
        if (is_k) {
            cal_div_scalar_kernel<<<CDIV(1024, 256), 256, 0, stream>>>(
                cents_in, base_c, f1 /*orig_ts*/, 1024);
        }

        struct Best { float a, ts, e, me; };
        Best bf = {1.0f, f1, 1e30f, 1e30f};
        Best bm = bf, bv = bf;
        float deltas[8] = {-0.03f, -0.02f, -0.01f, 0.0f, 0.01f, 0.02f, 0.03f, 0.04f};
        for (int phase = 0; phase < 4; phase++) {
            float a_center = 1.0f, m_center = 1.0f;
            if (phase == 1) { a_center = bf.a; m_center = bf.ts / f1; }
            else if (phase == 2) { a_center = bm.a; m_center = bm.ts / f1; }
            else if (phase == 3) { a_center = bv.a; m_center = bv.ts / f1; }
            for (int ia = 0; ia < 8; ia++) {
                for (int itx = 0; itx < 8; itx++) {
                    float a_val, mult;
                    if (phase == 0) {
                        a_val = 0.80f + 0.05f * (float)ia;
                        mult = 0.80f + 0.05f * (float)itx;
                    } else {
                        a_val = a_center + deltas[ia];
                        mult = m_center + deltas[itx];
                    }
                    float cur_ts = f1 * mult;
                    if (is_k) {
                        cal_cents_scale_kernel<<<CDIV(1024, 256), 256, 0, stream>>>(
                            base_c, cents_cur, cur_ts);
                        cal_fwht_base_kernel<<<CDIV((long)sub * 64, 256), 256, 0, stream>>>(
                            x16, dv, norms, proj, sub);
                        cal_fp16_inplace_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            norms, sub);
                        cal_compress_k_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            proj, cents_cur, cur_ts, norms, mask, rnorms, payload, sub);
                        cal_decompress_k_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            cents_cur, rp, dv, a_val, norms, mask, rnorms,
                            payload, rec, sub);
                    } else {
                        cal_fwht_base_kernel<<<CDIV((long)sub * 64, 256), 256, 0, stream>>>(
                            x16, dv, norms, proj, sub);
                        cal_fp16_inplace_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            norms, sub);
                        cal_compress_v_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            proj, cents_in, vq_in, cur_ts, a_val, norms,
                            rnorms, payload, smeta, sub);
                        cal_fp16_inplace_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            rnorms, sub);
                        cal_decompress_v_kernel<<<CDIV(sub, 256), 256, 0, stream>>>(
                            cents_in, vq_in, rp, dv, a_val, norms, rnorms,
                            payload, smeta, rec, sub);
                    }
                    cudaMemsetAsync(out4, 0, 16, stream);
                    cal_mse_kernel<<<256, 256, 0, stream>>>(
                        x32, rec, dim_mag, dim_var, out4, (long)sub * 256);
                    float h4[4];
                    cudaMemcpyAsync(h4, out4, 16, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    float mse_flat = h4[0] / (float)(sub * 256);
                    float mse_mag = h4[1] / (float)sub;
                    float mse_var = h4[2] / (float)sub;
                    float maxerr = h4[3];
                    if (mse_flat < bf.e) { bf.e = mse_flat; bf.a = a_val; bf.ts = cur_ts; bf.me = maxerr; }
                    if (mse_mag < bm.e) { bm.e = mse_mag; bm.a = a_val; bm.ts = cur_ts; bm.me = maxerr; }
                    if (mse_var < bv.e) { bv.e = mse_var; bv.a = a_val; bv.ts = cur_ts; bv.me = maxerr; }
                }
            }
        }
        float w_a = bf.a, w_ts = bf.ts, w_e = bf.me;
        if (bm.me < w_e) { w_e = bm.me; w_a = bm.a; w_ts = bm.ts; }
        if (bv.me < w_e) { w_e = bv.me; w_a = bv.a; w_ts = bv.ts; }
        printf("    [%s] winner alpha= %g ts= %g maxerr= %g\n",
               is_k ? "SmartK" : "SmartV", w_a, w_ts, w_e);
        if (is_k) {
            cal_cents_scale_kernel<<<CDIV(1024, 256), 256, 0, stream>>>(
                base_c, f, w_ts);
        }
        float hout[2] = {w_ts, w_a};
        cudaMemcpyAsync(g, hout, 8, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaFree(dim_mag);
        cudaFree(dim_var);
        cudaFree(proj);
        cudaFree(rec);
        cudaFree(norms);
        cudaFree(rnorms);
        cudaFree(smeta);
        cudaFree(cents_cur);
        cudaFree(out4);
        cudaFree(payload);
        if (is_k) { cudaFree(base_c); cudaFree(mask); }
        return 0;
    }
    return -1;
}

} // extern "C"
