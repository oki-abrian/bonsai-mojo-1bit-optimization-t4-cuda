# ===----------------------------------------------------------------------=== #
# Module: dequant.mojo
# Purpose: Logika ekstraksi bit LSB-first dan dekuantisasi affine Bonsai (b = -s)
#          dengan manipulasi bitwise branchless dan akselerasi SIMD.
# ===----------------------------------------------------------------------=== #

from sys import size_of

# ----------------------------------------------------------------------------
# 1. Ekstraksi Bit Tunggal (LSB-First dalam uint8)
# ----------------------------------------------------------------------------
@always_inline
fn extract_bit_lsb(byte_val: UInt8, bit_idx: Int) -> Float32:
    """
    Mengekstrak bit ke-i (0..7) dari 1 byte packed bobot.
    Format Bonsai menggunakan konvensi LSB-first:
    bit 0 mewakili elemen K pertama, bit 7 mewakili elemen K terakhir dalam byte.
    """
    var bit = (Int(byte_val) >> (bit_idx & 7)) & 1
    return Float32(bit)

# ----------------------------------------------------------------------------
# 2. Dekuantisasi Affine Bonsai (b = -s)
# ----------------------------------------------------------------------------
@always_inline
fn dequant_affine_bonsai(bit: Float32, scale: Float32) -> Float32:
    """
    Kontrak Affine Bonsai:
    w = q * s + b, di mana b = -s dan q in {0, 1}.
    Maka:
      w = q * s - s = s * (2 * q - 1)
      - jika bit == 1 -> w = +s
      - jika bit == 0 -> w = -s
    Bias terserap sempurna ke dalam perkalian tanpa operasi penjumlahan terpisah.
    """
    return (2.0 * bit - 1.0) * scale

# ----------------------------------------------------------------------------
# 3. Fused Multiply-Accumulate (FMA) 1-Bit
# ----------------------------------------------------------------------------
@always_inline
fn fma_1bit(acc: Float32, byte_val: UInt8, bit_idx: Int, scale: Float32, x_val: Float32) -> Float32:
    """
    Menjalankan ekstraksi bit, dekuantisasi affine, dan FMA akumulasi dalam satu langkah:
    acc = acc + w_eff * x_val
    """
    var bit = extract_bit_lsb(byte_val, bit_idx)
    var w_eff = dequant_affine_bonsai(bit, scale)
    return acc + w_eff * x_val

# ----------------------------------------------------------------------------
# 4. Unpack 8 Bobot dari 1 Byte ke SIMD[Float32, 8]
# ----------------------------------------------------------------------------
@always_inline
fn unpack_byte_to_simd8(byte_val: UInt8, scale: Float32) -> SIMD[DType.float32, 8]:
    """
    Mendekode 1 byte uint8 (8 bobot biner) langsung ke SIMD vector 8 elemen FP32
    dengan perkalian skala grup scale.
    """
    var res = SIMD[DType.float32, 8](0.0)
    var b_int = Int(byte_val)
    
    # Branchless compile-time unrolled SIMD packing
    for i in range(8):
        var bit = Float32((b_int >> i) & 1)
        res[i] = (2.0 * bit - 1.0) * scale
    return res

# ----------------------------------------------------------------------------
# 5. Branchless IEEE-754 Half Sign-Flip (Trik Optimasi T4 sm_75)
# ----------------------------------------------------------------------------
@always_inline
fn bit_to_half_sign_flip(bit: UInt32) -> UInt16:
    """
    Representasi IEEE-754 FP16:
    0xBC00 = -1.0 (sign=1, exp=01111, mantissa=0)
    0x3C00 = +1.0 (sign=0, exp=01111, mantissa=0)
    
    Jika bit == 1 -> kita ingin +1.0 (flip bit sign dari 1 menjadi 0 via XOR 0x8000)
    Jika bit == 0 -> kita ingin -1.0 (bit sign tetap 1 via XOR 0x0000)
    Formula: 0xBC00 ^ ((bit & 1) << 15)
    Menghilangkan 100% instruksi branch divergen pada register ALU T4.
    """
    var mask: UInt16 = 0xBC00
    var flip: UInt16 = UInt16((bit & 1) << 15)
    return mask ^ flip

# ============================================================================
# 6. Ekstraksi 2-Bit Ternary (Bonsai-2 / Ternary-Bonsai-2-27B, Qwen3.8)
# ============================================================================
# Kontrak pack MLX "prism_hadamard_qwen35" (diverifikasi numerik terhadap
# runtime/codec.py referensi, lihat CATATAN_IMPLEMENTASI_BONSAI2.md):
#   - bobot U32 [N, K/16], 16 bobot per word little-endian, lane i di bit 2i
#   - 1 byte = 4 bobot, sub-lane j (0..3) di bit 2j
#   - scales F16 [N, K/128] = s_ckpt MENTAH; biases == -scales
#   - dequant: w = q*s + b = (q-1)*s, q di {0,1,2} -> w di {-s, 0, +s}
#   - TIDAK ada pembagian skala (berbeda dari 1-bit yg memakai (2q-1)*(s/2)).
# ----------------------------------------------------------------------------

@always_inline
fn extract_2bit_lane(byte_val: UInt8, lane_idx: Int) -> Float32:
    """
    Mengekstrak kode 2-bit ke-j (0..3) dari 1 byte bobot terpaket.
    Konvensi LSB-first: sub-lane 0 = bit 0-1 (elemen ke-0), sub-lane 3 = bit 6-7.
    Mengembalikan kode mentah q di {0, 1, 2} (BUKAN nilai ternary).
    """
    var q = (Int(byte_val) >> ((lane_idx & 3) * 2)) & 3
    return Float32(q)

@always_inline
fn dequant_ternary_bonsai(q: Float32, scale: Float32) -> Float32:
    """
    Kontrak Affine Ternary Bonsai-2:
    w = q * s + b, dengan b = -s dan q di {0,1,2}.
    Maka w = q*s - s = s * (q - 1):
      - q == 0 -> w = -s
      - q == 1 -> w =  0
      - q == 2 -> w = +s
    Bias terserap sempurna tanpa operasi penjumlahan terpisah.
    """
    return (q - 1.0) * scale

@always_inline
fn fma_2bit(
    acc: Float32, byte_val: UInt8, lane_idx: Int, scale: Float32, x_val: Float32
) -> Float32:
    """
    Ekstraksi 2-bit + dekuantisasi affine ternary + FMA akumulasi dalam satu
    langkah: acc = acc + w_eff * x_val, w_eff = (q-1)*s.
    """
    var q = extract_2bit_lane(byte_val, lane_idx)
    var w_eff = dequant_ternary_bonsai(q, scale)
    return acc + w_eff * x_val

@always_inline
fn unpack_byte_to_simd4_ternary(
    byte_val: UInt8, scale: Float32
) -> SIMD[DType.float32, 4]:
    """
    Mendekode 1 byte uint8 (4 bobot ternary 2-bit) langsung ke SIMD[Float32, 4].
    Sub-lane j menempati bit 2j..2j+1 (LSB-first), persis seperti layout U32
    pack MLX saat diperlakukan sebagai deretan byte.
    """
    var res = SIMD[DType.float32, 4](0.0)
    var b_int = Int(byte_val)
    for j in range(4):
        var q = Float32((b_int >> (j * 2)) & 3)
        res[j] = (q - 1.0) * scale
    return res
