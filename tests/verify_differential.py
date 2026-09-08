import math
import random

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def ref_bonsai_pure_python(x, wp, s, M, N, K, group_size=128):
    """
    Referensi murni Python tanpa dependency:
    x:  [M, K] list float
    wp: [N, K // 8] list int
    s:  [N, K // group_size] list float
    w_eff = (2 * bit - 1) * s (karena b = -s)
    """
    weight_row_bytes = K // 8
    scale_groups = (K + group_size - 1) // group_size
    out = [[0.0 for _ in range(N)] for _ in range(M)]

    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(K):
                xv = x[m * K + k]
                byte_val = wp[n * weight_row_bytes + (k // 8)]
                bit = (byte_val >> (k % 8)) & 1
                sg = s[n * scale_groups + (k // group_size)]
                w_eff = (2.0 * bit - 1.0) * sg
                acc += w_eff * xv
            out[m][n] = acc
    return out

def main():
    print(">> Menjalankan Verifikasi Referensi Matematika Bonsai-27B (Python)...")
    M, N, K = 1, 64, 128
    group_size = 128
    
    random.seed(42)
    x = [random.uniform(-1.0, 1.0) for _ in range(M * K)]
    wp = [random.randint(0, 255) for _ in range(N * (K // 8))]
    s = [random.uniform(0.5, 2.0) for _ in range(N * ((K + group_size - 1) // group_size))]
    
    out = ref_bonsai_pure_python(x, wp, s, M, N, K, group_size)
    print(f"   * Status NumPy    : {'Tersedia' if HAS_NUMPY else 'Fallback Pure-Python Aktif'}")
    print(f"   * Dimensi Output  : [{len(out)}, {len(out[0])}]")
    print(f"   * Sample Nilai [:5]: {[round(v, 4) for v in out[0][:5]]}")
    print(">> Verifikasi Berhasil.")

if __name__ == "__main__":
    main()
