# ===----------------------------------------------------------------------=== #
# Module: tests/test_bonsai_shapes.mojo
# Purpose: Pengujian bentuk layer riil model Bonsai-27B-mlx-1bit
#          (MLP intermediate N=11008, Fused gate_up N=22016, GDN in_proj_all)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt
from src.common import get_scale_stride, get_weight_row_bytes
from src.ops import quantized_matmul_1bit

fn test_bonsai_layer(layer_name: String, M: Int, N: Int, K: Int) -> Bool:
    print(">> Menguji Layer Bonsai-27B:", layer_name, "(M=", M, "N=", N, "K=", K, ")")
    
    var scale_groups = get_scale_stride(K)
    var weight_row_bytes = get_weight_row_bytes(K)
    
    var szx = M * K
    var szw = N * weight_row_bytes
    var szs = N * scale_groups
    var out_sz = M * N

    var hx = alloc[Float32](szx)
    var hw = alloc[UInt8](szw)
    var hs = alloc[Float32](szs)
    var hy = alloc[Float32](out_sz)
    var href = alloc[Float64](out_sz)

    for i in range(szx):
        hx[i] = Float32((i % 11) - 5) * 0.1
    for i in range(szs):
        hs[i] = 1.0 + Float32(i % 7) * 0.05
    for i in range(szw):
        hw[i] = UInt8((i * 19 + 7) & 0xFF)
    for i in range(out_sz):
        hy[i] = 0.0
        href[i] = 0.0

    # Referensi FP64
    for m in range(M):
        for n in range(N):
            var acc: Float64 = 0.0
            for k in range(K):
                var xv = Float64(hx[m * K + k])
                var byte_offset = n * weight_row_bytes + (k // 8)
                var byte_val = hw[byte_offset]
                var bit = Float64((Int(byte_val) >> (k % 8)) & 1)
                var sg = Float64(hs[n * scale_groups + (k // 128)])
                acc += (2.0 * bit - 1.0) * sg * xv
            href[m * N + n] = acc

    # Eksekusi Kernel Mojo
    quantized_matmul_1bit[DType.float32](
        hx, hw, hs, hy,
        M, N, K, 1, True
    )

    var max_err: Float64 = 0.0
    for i in range(out_sz):
        var diff = abs(Float64(hy[i]) - href[i])
        if diff > max_err:
            max_err = diff

    var eps: Float64 = 4.9e-4
    var atol: Float64 = 8.0 * eps * sqrt(Float64(K)) * 1.5 * 2.0
    var passed = max_err <= atol

    var status = "[PASS]" if passed else "[GAGAL]"
    print("   ", status, layer_name, "| max_err=", max_err, "atol=", atol)

    hx.free()
    hw.free()
    hs.free()
    hy.free()
    href.free()

    return passed

fn main():
    print("=================================================================")
    print(">> Validasi Khusus Dimensi Layer Model Bonsai-27B (NVIDIA T4)")
    print("=================================================================")

    var all_ok = True

    # 1. Decode Single Token (M = 1)
    all_ok = test_bonsai_layer("Bonsai-27B MLP Down (Decode)", M=1, N=4096, K=11008) and all_ok
    all_ok = test_bonsai_layer("Bonsai-27B MLP Gate/Up (Decode)", M=1, N=11008, K=4096) and all_ok
    all_ok = test_bonsai_layer("Bonsai-27B Fused MLP (Decode)", M=1, N=22016, K=4096) and all_ok
    all_ok = test_bonsai_layer("Bonsai-27B GDN in_proj_all (Decode)", M=1, N=7168, K=4096) and all_ok

    # 2. Prefill Small Prompt (M = 25)
    all_ok = test_bonsai_layer("Bonsai-27B MLP Down (Prefill M=25)", M=25, N=4096, K=11008) and all_ok
    all_ok = test_bonsai_layer("Bonsai-27B Fused MLP (Prefill M=25)", M=25, N=22016, K=4096) and all_ok

    # 3. Prefill Long Prompt (M = 128)
    all_ok = test_bonsai_layer("Bonsai-27B MLP Down (Prefill M=128)", M=128, N=4096, K=11008) and all_ok
    all_ok = test_bonsai_layer("Bonsai-27B Fused MLP (Prefill M=128)", M=128, N=22016, K=4096) and all_ok

    print("\n>> Status Akhir Validasi Layer Bonsai-27B:", "SEMUA PASS" if all_ok else "ADA YANG GAGAL")
