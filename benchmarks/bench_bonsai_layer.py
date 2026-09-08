#!/usr/bin/env python3
"""
Benchmark Layer Bonsai-27B (Python Simulation & Profile).
Mengukur waktu komputasi teoritis dan bandwidth pada bentuk layer riil Bonsai-27B.
"""

import time

def simulate_layer_profile():
    print("=================================================================")
    print(">> PROFILING SIMULASI LAYER BONSAI-27B (NVIDIA T4)")
    print("=================================================================")
    
    layers = [
        ("GDN in_proj_all (Decode)", 1, 7168, 4096),
        ("GDN out_proj (Decode)",    1, 4096, 4096),
        ("MLP Fused gate_up (Decode)", 1, 22016, 4096),
        ("MLP down_proj (Decode)",     1, 4096, 11008),
        ("MLP Fused gate_up (Prefill M=128)", 128, 22016, 4096),
        ("MLP down_proj (Prefill M=128)",     128, 4096, 11008),
    ]
    
    for name, M, N, K in layers:
        weight_bytes = N * (K // 8)
        scale_bytes = N * ((K + 127) // 128) * 2
        act_bytes = M * K * 2
        out_bytes = M * N * 2
        total_traffic = weight_bytes + scale_bytes + act_bytes + out_bytes
        flops = 2.0 * M * N * K
        
        # Plafon bandwidth GDDR6 T4: 320 GB/s (realistis ~240 GB/s)
        # Plafon FP32 ALU T4: 8.1 TFLOPS
        t_bw_ms = (total_traffic / 240e9) * 1000.0
        t_compute_ms = (flops / 8.1e12) * 1000.0
        roofline_ms = max(t_bw_ms, t_compute_ms)
        
        bound_type = "BW-BOUND" if t_bw_ms > t_compute_ms else "COMPUTE-BOUND"
        
        print(f"[{bound_type}] {name:<35}: M={M:<3} N={N:<5} K={K:<5}")
        print(f"   * Bobot Packed  : {weight_bytes / 1024 / 1024:.2f} MiB")
        print(f"   * Roofline Time : {roofline_ms:.3f} ms (Batas Fisik T4)")
        print("-" * 65)

if __name__ == "__main__":
    simulate_layer_profile()
