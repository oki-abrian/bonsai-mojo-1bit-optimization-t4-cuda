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
