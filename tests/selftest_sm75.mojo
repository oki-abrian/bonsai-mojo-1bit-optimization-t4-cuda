# ===----------------------------------------------------------------------=== #
# Module: tests/selftest_sm75.mojo
# Purpose: Uji mandiri komprehensif 43 kasus ekstrem W1A16 g128 untuk T4 (sm_75)
#          Verifikasi diferensial vs Referensi FP64 CPU + Uji Determinisme Bitwise.
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from math import sqrt
from src.common import get_scale_stride, get_weight_row_bytes
from src.ops import qmm_sm75_1bit, qmv_sm75_1bit, quantized_matmul_1bit

@fieldwise_init
struct TestCase(Copyable, Movable, ImplicitlyCopyable):
    var name: String
    var M: Int
    var N: Int
    var K: Int
    var L: Int
    var broadcast_w: Bool
    var pattern_mode: Int # 0=pseudo-random, 1=all_zeros (w=-s), 2=all_ones (w=+s), 3=checkerboard (0xAA)

fn run_case(c: TestCase) -> Bool:
    var szx = c.L * c.M * c.K
    var scale_groups = get_scale_stride(c.K)
    var weight_row_bytes = get_weight_row_bytes(c.K)
    
    var w_layers = 1 if c.broadcast_w else c.L
    var szw = w_layers * c.N * weight_row_bytes
    var szs = w_layers * c.N * scale_groups
    var out_sz = c.L * c.M * c.N

    # Alokasi buffer memori
    var hx = alloc[Float32](szx)
    var hw = alloc[UInt8](szw)
    var hs = alloc[Float32](szs)
    var hy1 = alloc[Float32](out_sz)
    var hy2 = alloc[Float32](out_sz)
    var href = alloc[Float64](out_sz)

    # Inisialisasi data deterministik
    for i in range(szx):
        hx[i] = Float32((i % 17) - 8) * 0.125
    for i in range(szs):
        hs[i] = Float32(1.0 + Float32(i % 5) * 0.1)

    for i in range(szw):
        if c.pattern_mode == 1:
            hw[i] = 0x00 # All zeros -> w = -s
        elif c.pattern_mode == 2:
            hw[i] = 0xFF # All ones  -> w = +s
        elif c.pattern_mode == 3:
            hw[i] = 0xAA # Checkerboard 10101010
        else:
            hw[i] = UInt8((i * 37 + 13) & 0xFF)

    # Inisialisasi output
    for i in range(out_sz):
        hy1[i] = 0.0
        hy2[i] = 0.0
        href[i] = 0.0

    # ---- 1. Referensi FP64 di CPU ----
    for l in range(c.L):
        var w_layer = 0 if c.broadcast_w else l
        for m in range(c.M):
            for n in range(c.N):
                var acc: Float64 = 0.0
                for k in range(c.K):
                    var xv = Float64(hx[(l * c.M + m) * c.K + k])
                    var byte_offset = (w_layer * c.N + n) * weight_row_bytes + (k // 8)
                    var byte_val = hw[byte_offset]
                    var bit = Float64((Int(byte_val) >> (k % 8)) & 1)
                    var sg = Float64(hs[(w_layer * c.N + n) * scale_groups + (k // 128)])
                    
                    var w_eff = (2.0 * bit - 1.0) * sg
                    acc += w_eff * xv
                href[(l * c.M + m) * c.N + n] = acc

    # ---- 2. Eksekusi Run 1 (Kernel Mojo) ----
    quantized_matmul_1bit[DType.float32](
        hx, hw, hs, hy1,
        c.M, c.N, c.K, c.L, c.broadcast_w
    )

    # ---- 3. Eksekusi Run 2 (Determinisme Bitwise) ----
    quantized_matmul_1bit[DType.float32](
        hx, hw, hs, hy2,
        c.M, c.N, c.K, c.L, c.broadcast_w
    )

    # ---- 4. Validasi Numerik & Determinisme ----
    var is_deterministic = True
    var max_err: Float64 = 0.0

    for i in range(out_sz):
        if hy1[i] != hy2[i]:
            is_deterministic = False
        var diff = abs(Float64(hy1[i]) - href[i])
        if diff > max_err:
            max_err = diff

    # Toleransi presisi: atol = 8 * eps * sqrt(K) * max|s| * max|x|
    var eps: Float64 = 4.9e-4
    var atol: Float64 = 8.0 * eps * sqrt(Float64(c.K)) * 1.5 * 2.0
    var pass_accuracy = (max_err <= atol) or (atol < 1e-5 and max_err < 1e-4)
    var passed = is_deterministic and pass_accuracy

    var status_str = "[PASS]" if passed else "[GAGAL]"
    print(status_str, c.name, "M=", c.M, "N=", c.N, "K=", c.K, "L=", c.L,
          "| max_err=", max_err, "atol=", atol, "det=", is_deterministic)

    # Bersihkan memori
    hx.free()
    hw.free()
    hs.free()
    hy1.free()
    hy2.free()
    href.free()

    return passed

fn main():
    print("=================================================================")
    print(">> Menjalankan 43 Kasus Uji Ekstrem W1A16 g128 untuk T4 di Mojo")
    print("=================================================================")

    var kasus = List[TestCase]()

    # 1. Ekstrim K Mini
    kasus.append(TestCase("k16_mini",            16,  64,   16, 1, False, 0))
    kasus.append(TestCase("k32_mini",            16,  64,   32, 1, False, 0))
    kasus.append(TestCase("k48_mini",            32,  64,   48, 1, False, 0))
    kasus.append(TestCase("k64_single_tile",     64,  64,   64, 1, False, 0))

    # 2. Ekstrim N Mini & Ragged N
    kasus.append(TestCase("n8_extreme_ragged",   16,   8,  128, 1, False, 0))
    kasus.append(TestCase("n16_sub_tile",        16,  16,  128, 1, False, 0))
    kasus.append(TestCase("n27_odd_prime",       32,  27,  128, 1, False, 0))
    kasus.append(TestCase("n33_cross_half",      32,  33,  128, 1, False, 0))
    kasus.append(TestCase("n65_cross_cta",       64,  65,  256, 1, False, 0))
    kasus.append(TestCase("n100_ragged",         96, 100,  128, 1, False, 0))
    kasus.append(TestCase("n127_odd_bound",      64, 127,  128, 1, False, 0))

    # 3. Token Generation & Ragged M
    kasus.append(TestCase("m1_decode_token",      1,  64,  128, 1, False, 0))
    kasus.append(TestCase("m2_small_batch",       2, 128,  256, 1, False, 0))
    kasus.append(TestCase("m7_prime_rows",        7,  64,  128, 1, False, 0))
    kasus.append(TestCase("m8_wmma_threshold",    8,  64,  128, 1, False, 0))
    kasus.append(TestCase("m11_bonsai_prefill",  11, 128,  256, 1, False, 0))
    kasus.append(TestCase("m11_bonsai_hidden",   11, 256,  256, 1, False, 0))
    kasus.append(TestCase("m11_bonsai_bcast",    11, 128,  256, 2, True,  0))
    kasus.append(TestCase("m15_sub_frag",        15,  64,  128, 1, False, 0))
    kasus.append(TestCase("m25_ragged",          25, 128,  256, 1, False, 0))
    kasus.append(TestCase("m33_cross_warp",      33, 128,  256, 1, False, 0))
    kasus.append(TestCase("m65_cross_block",     65,  64,  256, 1, False, 0))
    kasus.append(TestCase("m130_multi_block",   130,  64,  256, 1, False, 0))

    # 4. Kasus Ekstrim Ganda
    kasus.append(TestCase("m13_n29_ragged",      13,  29,  128, 1, False, 0))
    kasus.append(TestCase("m47_n83_ragged",      47,  83,  256, 1, False, 0))
    kasus.append(TestCase("m111_n159_ragged",   111, 159,  128, 1, False, 0))

    # 5. Multi-Tile K & Scale Groups
    kasus.append(TestCase("k192_multi_tile",     64, 128,  192, 1, False, 0))
    kasus.append(TestCase("k256_two_groups",     64, 128,  256, 1, False, 0))
    kasus.append(TestCase("k384_three_groups",   64, 128,  384, 1, False, 0))
    kasus.append(TestCase("k512_four_groups",    64, 128,  512, 1, False, 0))
    kasus.append(TestCase("k1024_deep",          64, 128, 1024, 1, False, 0))
    kasus.append(TestCase("k2048_stress",       128, 256, 2048, 1, False, 0))

    # 6. Batched GEMM (L > 1) & Weight Broadcast
    kasus.append(TestCase("batch2_standard",     17, 192,  128, 2, False, 0))
    kasus.append(TestCase("batch3_odd",          32,  64,  128, 3, False, 0))
    kasus.append(TestCase("batch5_prime",        16, 128,  128, 5, False, 0))
    kasus.append(TestCase("batch8_deep",         64,  64,  256, 8, False, 0))
    kasus.append(TestCase("bcast_l2",            32, 128,  256, 2, True,  0))
    kasus.append(TestCase("bcast_l4_ragged",     33,  65,  128, 4, True,  0))

    # 7. Pola Bit Ekstrim
    kasus.append(TestCase("pat_all_zeros_minus", 64,  64,  256, 1, False, 1))
    kasus.append(TestCase("pat_all_ones_plus",   64,  64,  256, 1, False, 2))
    kasus.append(TestCase("pat_checkerboard",    64,  64,  256, 1, False, 3))

    var passed_count = 0
    for i in range(len(kasus)):
        var ok = run_case(kasus[i])
        if ok:
            passed_count += 1

    print("\n=================================================================")
    print(">> HASIL SELFTEST: ", passed_count, "/", len(kasus), " Kasus Berhasil PASS")
    print("=================================================================")
