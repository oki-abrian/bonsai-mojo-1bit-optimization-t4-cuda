# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/rope.mojo
# Purpose: Rotary Positional Embeddings (RoPE) Parsial (25% Factor)
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from math import exp, log, cos, sin

fn apply_partial_rope(
    vec: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    rotary_dim: Int,
    pos: Int,
    theta_base: Float32 = 100000.0,
    stride: Int = 0
):
    """
    Menerapkan Rotary Positional Embedding (RoPE) parsial:
    Hanya dimensi 0..rotary_dim-1 yang dirotasi, dimensi sisanya utuh linear.
    Frekuensi: theta_i = base ** (-2i / rotary_dim).

    PASANGAN HALF-SPLIT — paritas MLX rope (traditional=False, default) dan
    kernel GPU `partial_rope_sm75_gpu` (elementwise_sm75.mojo:731-736):
    elemen i (i < rotary_dim/2) berpasangan dengan elemen i + rotary_dim/2.
    BUKAN pasangan bersebelahan (2i, 2i+1).

    Sebelumnya fungsi ini memakai pasangan BERSEBELAHAN (2i, 2i+1) sehingga
    TIDAK sepakat dgn kernel GPU. Jalur ini memang tidak dipakai oleh
    main.mojo (yang memakai kernel GPU), tetapi perbedaannya adalah jebakan:
    setiap tes paritas CPU-vs-GPU akan gagal/atau "membuktikan" hal yang salah.

    Mendukung custom stride untuk layout interleaved (misal Query Attention
    stride = 2 * head_dim).
    """
    var half_rotary = rotary_dim // 2
    var eff_stride = stride if stride > 0 else head_dim
    for h in range(num_heads):
        var h_offset = h * eff_stride
        for i in range(half_rotary):
            var exponent = -Float32(2 * i) / Float32(rotary_dim)
            var freq = exp(log(theta_base) * exponent)
            var m_theta = Float32(pos) * freq
            var cos_val = cos(m_theta)
            var sin_val = sin(m_theta)

            # Half-split: index_2 = index_1 + half_rotary (paritas MLX rope.cu).
            var x0 = vec[h_offset + i]
            var x1 = vec[h_offset + i + half_rotary]

            # Rotasi 2D standar bidang kompleks
            vec[h_offset + i]              = x0 * cos_val - x1 * sin_val
            vec[h_offset + i + half_rotary] = x0 * sin_val + x1 * cos_val
