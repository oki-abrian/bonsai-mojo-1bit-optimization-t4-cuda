// ============================================================================
// bench_ref_qmm.cu — Benchmark kernel qmm WMMA v2 ASLI dari referensi MLX
// fork (qmm_impl_sm75_wmma.cuh), dikompilasi langsung dengan nvcc TANPA
// Mojo/FFI/porting. Tujuan: memastikan angka throughput kernel referensi
// pada node yang sama dengan port kita, sehingga selisih (bila ada) pasti
// berasal dari port/FFI, bukan dari kernelnya.
//
// Build: nvcc -O3 -arch=sm_75 -o bench_ref_qmm bench_ref_qmm.cu
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "qmm_impl_sm75_wmma.cuh"  // namespace mlx::core::cu::wmma_b1

using mlx::core::cu::wmma_b1::qmm_sm75_b1_kernel;

static void launch_ref(const __half* x, const uint8_t* w, const __half* s,
                       __half* y, int m, int n, int k, cudaStream_t stream) {
    dim3 grid, block(128);
    if (m <= 32) {
        grid = dim3((n + 63) / 64, (m + 31) / 32, 1);
        qmm_sm75_b1_kernel<__half, 32, 64, 64>
            <<<grid, block, 0, stream>>>(x, w, s, nullptr, y, m, n, k, 1, false);
    } else {
        grid = dim3((n + 63) / 64, (m + 63) / 64, 1);
        qmm_sm75_b1_kernel<__half, 64, 64, 64>
            <<<grid, block, 0, stream>>>(x, w, s, nullptr, y, m, n, k, 1, false);
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s | SM: %d.%d | SMs: %d | clock: %d MHz\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.clockRate / 1000);

    struct Shape { const char* name; int M, N, K; };
    Shape shapes[] = {
        {"gateup_m9",  9, 34816, 5120},
        {"inproj_m9",  9, 16480, 5120},
        {"down_m9",    9, 5120, 17408},
        {"outproj_m9", 9, 5120, 6144},
        {"qproj_m9",   9, 12288, 5120},
        {"gateup_m32", 32, 34816, 5120},
        {"gateup_m9_repeat", 9, 34816, 5120},  // ulang di akhir: cek efek clock ramp
    };

    const int MAXN = 34816, MAXK = 17408, MAXM = 32;
    __half* x;  cudaMalloc(&x,  MAXM * MAXK * 2);
    uint8_t* w; cudaMalloc(&w, (size_t)MAXN * MAXK / 8);
    __half* s;  cudaMalloc(&s, (size_t)MAXN * (MAXK / 128) * 2);
    __half* y;  cudaMalloc(&y, (size_t)MAXM * MAXN * 2);

    // isi data sederhana
    __half* hx = (__half*)malloc(MAXM * MAXK * 2);
    for (int i = 0; i < MAXM * MAXK; ++i) hx[i] = __float2half(((i % 17) - 8) * 0.125f);
    uint8_t* hw = (uint8_t*)malloc((size_t)MAXN * MAXK / 8);
    for (size_t i = 0; i < (size_t)MAXN * MAXK / 8; ++i) hw[i] = (uint8_t)((i * 37 + 13) & 0xFF);
    __half* hs = (__half*)malloc((size_t)MAXN * (MAXK / 128) * 2);
    for (size_t i = 0; i < (size_t)MAXN * (MAXK / 128); ++i) hs[i] = __float2half(1.0f);
    cudaMemcpy(x, hx, MAXM * MAXK * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw, (size_t)MAXN * MAXK / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(s, hs, (size_t)MAXN * (MAXK / 128) * 2, cudaMemcpyHostToDevice);
    free(hx); free(hw); free(hs);

    cudaStream_t st; cudaStreamCreate(&st);

    // Warmup panjang agar clock GPU benar-benar ramping (pelajaran Run AP:
    // bench pertama diukur pada clock rendah -> 2x lebih lambat).
    for (int i = 0; i < 100; ++i) launch_ref(x, w, s, y, 9, 34816, 5120, st);
    cudaStreamSynchronize(st);

    for (auto& sh : shapes) {
        const int WARM = 20, ITERS = 200;
        for (int i = 0; i < WARM; ++i) launch_ref(x, w, s, y, sh.M, sh.N, sh.K, st);
        cudaStreamSynchronize(st);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, st);
        for (int i = 0; i < ITERS; ++i) launch_ref(x, w, s, y, sh.M, sh.N, sh.K, st);
        cudaEventRecord(e1, st);
        cudaStreamSynchronize(st);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= ITERS;
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        double bytes = (double)sh.N * sh.K / 8 + (double)sh.M * sh.K * 2 +
                       (double)sh.N * (sh.K / 128) * 2 + (double)sh.M * sh.N * 2;
        double gbps = bytes / (ms * 1e-3) / 1e9;
        double gmacs = (double)sh.N * sh.K * sh.M / (ms * 1e-3) / 1e9;
        printf("[REF-QMM] %s | M=%d N=%d K=%d | %.4f ms/iter | %.1f GB/s | %.0f GMAC/s\n",
               sh.name, sh.M, sh.N, sh.K, ms, gbps, gmacs);
    }

    // Clock saat ini (verifikasi tidak throttling saat bench terakhir)
    int sm_clock = 0;
    // cudaDeviceGetAttribute tidak memberi clock live; pakai nvidia-smi dari luar.
    printf("done\n");
    return 0;
}
