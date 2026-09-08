# ===----------------------------------------------------------------------=== #
# Module: src/models/qwen3_5/norm.mojo
# Purpose: Operator matematika, aktivasi non-linear, dan normalisasi layer
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer
from math import sqrt, exp, log

@always_inline
fn silu(x: Float32) -> Float32:
    """Aktivasi SiLU (Swish-1): x * sigmoid(x)."""
    return x / (1.0 + exp(-x))

@always_inline
fn sigmoid(x: Float32) -> Float32:
    """Aktivasi Sigmoid numerik standar: 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + exp(-x))

@always_inline
fn softplus(x: Float32) -> Float32:
    """Aktivasi Softplus stabil: log(1 + exp(x))."""
    if x > 20.0:
        return x
    return log(1.0 + exp(x))

fn rms_norm(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    weight: UnsafePointer[Float32, MutAnyOrigin],
    M: Int, D: Int, eps: Float32 = 1e-6
):
    """
    Root Mean Square Normalization presisi FP32:
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * gamma
    """
    for m in range(M):
        var row = m * D
        var sum_sq: Float32 = 0.0
        for d in range(D):
            var val = x[row + d]
            sum_sq += val * val
        var inv_rms = 1.0 / sqrt(sum_sq / Float32(D) + eps)
        for d in range(D):
            var gamma = weight[d] if weight else 1.0
            out_ptr[row + d] = x[row + d] * inv_rms * gamma

fn head_rms_norm(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    num_heads: Int,
    head_dim: Int,
    scale_factor: Float32 = 1.0,
    eps: Float32 = 1e-6,
    stride: Int = 0
):
    """
    Normalisasi RMSNorm independen per head untuk Query dan Key (Q-Norm & K-Norm)
    sesuai baris 180-183 di qwen3_5.py dan baris 137-139 di qwen3_next.py.
    Mendukung custom stride untuk layout interleaved (misal Query Attention stride = 2 * head_dim).
    """
    var eff_stride = stride if stride > 0 else head_dim
    for h in range(num_heads):
        var offset = h * eff_stride
        var sum_sq: Float32 = 0.0
        for d in range(head_dim):
            var val = x[offset + d]
            sum_sq += val * val
        var inv_rms = (1.0 / sqrt(sum_sq / Float32(head_dim) + eps)) * scale_factor
        for d in range(head_dim):
            out_ptr[offset + d] = x[offset + d] * inv_rms

fn softmax(out_ptr: UnsafePointer[Float32, MutAnyOrigin], x: UnsafePointer[Float32, MutAnyOrigin], N: Int):
    """Softmax numerik stabil dengan penarikan nilai maksimum."""
    var max_val: Float32 = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
    var sum_exp: Float32 = 0.0
    for i in range(N):
        var e = exp(x[i] - max_val)
        out_ptr[i] = e
        sum_exp += e
    var inv_sum = 1.0 / sum_exp
    for i in range(N):
        out_ptr[i] *= inv_sum

fn rmsnorm_gated_fused(
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],
    x: UnsafePointer[Float32, MutAnyOrigin],
    gate: UnsafePointer[Float32, MutAnyOrigin],
    D: Int,
    eps: Float32 = 1e-6
):
    """
    Fused Single-Pass RMSNorm Gated:
    Menghitung inv_rms = 1 / sqrt(mean(x^2) + eps),
    lalu dalam 1 pass register mengalikan x[d] * inv_rms * silu(gate[d]).
    Memangkas bolak-balik akses VRAM.
    """
    var sum_sq: Float32 = 0.0
    for d in range(D):
        var val = x[d]
        sum_sq += val * val
    var inv_rms = 1.0 / sqrt(sum_sq / Float32(D) + eps)
    for d in range(D):
        out_ptr[d] = (x[d] * inv_rms) * silu(gate[d])

