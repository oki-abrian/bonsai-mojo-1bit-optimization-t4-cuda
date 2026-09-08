# ===----------------------------------------------------------------------=== #
# Module: benchmarks/bench_t4.mojo
# Purpose: Micro-benchmark Throughput (TFLOPS) dan Bandwidth Memori (GB/s)
#          kernel W1A16 g128 pada NVIDIA T4 (sm_75) di Mojo.
# ===----------------------------------------------------------------------=== #

from time import now
from memory import UnsafePointer
from ..common import get_scale_stride, get_weight_row_bytes
from ..ops import quantized_matmul_1bit

fn benchmark_shape(name: String, M: Int, N: Int, K: Int, iters: Int = 50):
    let scale_groups = get_scale_stride(K)
    let weight_row_bytes = get_weight_row_bytes(K)

    let szx = M * K
    let szw = N * weight_row_bytes
    let szs = N * scale_groups
    let out_sz = M * N

    let hx = UnsafePointer[Float32].alloc(szx)
    let hw = UnsafePointer[UInt8].alloc(szw)
    let hs = UnsafePointer[Float32].alloc(szs)
    let hy = UnsafePointer[Float32].alloc(out_sz)

    # Inisialisasi
    for i in range(szx): hx[i] = 0.5
    for i in range(szs): hs[i] = 1.0
    for i in range(szw): hw[i] = 0xAA
    for i in range(out_sz): hy[i] = 0.0

    # Warmup
    for _ in range(5):
        quantized_matmul_1bit[DType.float32](hx, hw, hs, hy, M, N, K, 1, True)

    # Pengukuran Waktu
    let t_start = now()
    for _ in range(iters):
        quantized_matmul_1bit[DType.float32](hx, hw, hs, hy, M, N, K, 1, True)
    let t_end = now()

    let total_sec = Float64(t_end - t_start) / 1e9
    let avg_sec = total_sec / Float64(iters)
    let avg_ms = avg_sec * 1000.0

    # FLOPs = 2 * M * N * K
    let flops = 2.0 * Float64(M) * Float64(N) * Float64(K)
    let tflops = (flops / avg_sec) / 1e12

    # Weight Bytes Read = N * (K / 8) + N * (K / 128) * 2 + M * K * 2
    let bytes_read = Float64(szw + szs * 2 + szx * 2)
    let gb_per_sec = (bytes_read / avg_sec) / 1e9

    print("-----------------------------------------------------------------")
    print(">>", name)
    print("   Dimensi       : M=", M, "N=", N, "K=", K)
    print("   Latensi Rata2 :", avg_ms, "ms per eksekusi")
    if M > 1:
        print("   Komputasi     :", tflops, "TFLOPS")
    else:
        print("   Throughput BW :", gb_per_sec, "GB/s")

    hx.free()
    hw.free()
    hs.free()
    hy.free()

fn main():
    print("=================================================================")
    print(">> BENCHMARK THROUGHPUT W1A16 g128 DI MOJO (NVIDIA T4)")
    print("=================================================================")

    # 1. Decode Token Benchmarks (Memory-Bound)
    benchmark_shape("Bonsai-27B MLP Decode (M=1)", M=1, N=11008, K=4096, iters=20)
    benchmark_shape("Bonsai-27B Fused MLP Decode (M=1)", M=1, N=22016, K=4096, iters=20)

    # 2. Prefill Prompt Benchmarks (Compute-Bound)
    benchmark_shape("Bonsai-27B Short Prefill (M=25)", M=25, N=11008, K=4096, iters=20)
    benchmark_shape("Bonsai-27B Medium Prefill (M=128)", M=128, N=11008, K=4096, iters=10)
