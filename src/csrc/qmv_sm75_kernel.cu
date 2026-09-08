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

} // extern "C"
