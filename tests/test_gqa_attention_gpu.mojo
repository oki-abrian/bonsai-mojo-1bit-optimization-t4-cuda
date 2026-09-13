# ===----------------------------------------------------------------------=== #
# Module: tests/test_gqa_attention_gpu.mojo
# Purpose: Tes diferensial kernel Fused GQA Attention + Sigmoid Gate (sm_75)
#          vs referensi FP64, PLUS tes penjaga `head_dim == 256`.
#
# Kenapa tes ini ada:
#   `gqa_attention_sm75_gpu` adalah satu-satunya kernel besar di jalur GPU yang
#   TIDAK punya tes numerik sama sekali (QMV/QMM/RoPE/Argmax/GDN sudah ada).
#   Ia menggabungkan 4 hal yang mudah salah: softmax stabil, reduksi lintas
#   warp, pembobotan V, dan gerbang sigmoid fused.
#
# BATAS KERAS YANG DIKUNCI TES INI (temuan 2026-09-13):
#   Kernel hanya benar untuk `head_dim == 256` PERSIS:
#     (a) SMEM tetap 320 float (256 Q + 32 reduksi + 32 broadcast) ->
#         head_dim > 256 meluap.
#     (b) reduksi tahap-2 di-hardcode untuk 8 WARP (`lane < 8`, komentar
#         "Block 256 thread = 8 warp") -> head_dim < 256 membuat lane 4..7
#         membaca SMEM yang TIDAK PERNAH DITULIS (sampah) -> hasil bisa
#         sepenuhnya salah tanpa error.
#   Penjaga di `gqa_attention_sm75_launch_on` (src/ops.mojo) menolak
#   head_dim != 256. Kasus 3 di bawah menguji penjaga itu.
#
# Jalankan: pixi run mojo run -I . tests/test_gqa_attention_gpu.mojo
# ===----------------------------------------------------------------------=== #

from math import exp, sqrt
from memory import UnsafePointer, alloc
from gpu.host import DeviceContext, DeviceBuffer
from src.ops import gqa_attention_sm75_launch_on


fn sigmoid64(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))


fn ref_gqa_fp64(
    dst: UnsafePointer[Float64, MutAnyOrigin],
    q_gate: UnsafePointer[Float64, MutAnyOrigin],
    k_cache: UnsafePointer[Float64, MutAnyOrigin],
    v_cache: UnsafePointer[Float64, MutAnyOrigin],
    scores: UnsafePointer[Float64, MutAnyOrigin],
    seq_len: Int,
    max_seq_len: Int,
    H_q: Int,
    H_kv: Int,
    head_dim: Int,
    scale: Float64
):
    """Referensi FP64 dari kernel Fused GQA + gerbang sigmoid.

    Meniru urutan operasi kernel, termasuk `1/(Z + 1e-9)` sebagai penyebut
    softmax (bukan 1/Z), supaya perbandingan apple-to-apple.
    """
    var group = H_q // H_kv
    var kv_dim = H_kv * head_dim
    for hq in range(H_q):
        var hkv = hq // group
        # skor
        for t in range(seq_len):
            var dot: Float64 = 0.0
            for d in range(head_dim):
                dot += q_gate[hq * (2 * head_dim) + d] * k_cache[t * kv_dim + hkv * head_dim + d]
            scores[hq * max_seq_len + t] = dot * scale
        # max
        var m: Float64 = -3.0e38
        for t in range(seq_len):
            if scores[hq * max_seq_len + t] > m:
                m = scores[hq * max_seq_len + t]
        # exp + sum
        var z: Float64 = 0.0
        for t in range(seq_len):
            var e = exp(scores[hq * max_seq_len + t] - m)
            scores[hq * max_seq_len + t] = e
            z += e
        var inv_z = 1.0 / (z + 1e-9)
        # konteks + gerbang sigmoid
        for d in range(head_dim):
            var ctx: Float64 = 0.0
            for t in range(seq_len):
                var p = scores[hq * max_seq_len + t] * inv_z
                ctx += p * v_cache[t * kv_dim + hkv * head_dim + d]
            var g = sigmoid64(q_gate[hq * (2 * head_dim) + head_dim + d])
            dst[hq * head_dim + d] = ctx * g


fn run_gqa_case(
    mut ctx: DeviceContext,
    name: String,
    H_q: Int,
    H_kv: Int,
    head_dim: Int,
    seq_len: Int,
    max_seq_len: Int,
    tol: Float32
) raises -> Bool:
    var kv_dim = H_kv * head_dim
    var q_len = H_q * 2 * head_dim
    var score_len = H_q * max_seq_len
    var scale = Float64(1.0) / sqrt(Float64(head_dim))

    # ---- host: q_gate (interleaved q|gate), k_cache, v_cache ----
    var h_qg = alloc[Float32](q_len)
    var h_k = alloc[Float32](max_seq_len * kv_dim)
    var h_v = alloc[Float32](max_seq_len * kv_dim)
    for i in range(q_len):
        h_qg[i] = Float32((i * 7919 + 1013) % 2000) * 0.001 - 1.0
    for i in range(max_seq_len * kv_dim):
        h_k[i] = Float32((i * 6271 + 5003) % 2000) * 0.001 - 1.0
        h_v[i] = Float32((i * 4931 + 7919) % 2000) * 0.002 - 2.0

    # ---- referensi FP64 memakai nilai FP32 yang SAMA ----
    var r_qg = alloc[Float64](q_len)
    var r_k = alloc[Float64](max_seq_len * kv_dim)
    var r_v = alloc[Float64](max_seq_len * kv_dim)
    for i in range(q_len):
        r_qg[i] = Float64(h_qg[i])
    for i in range(max_seq_len * kv_dim):
        r_k[i] = Float64(h_k[i])
        r_v[i] = Float64(h_v[i])
    var r_scores = alloc[Float64](score_len)
    var r_out = alloc[Float64](H_q * head_dim)
    ref_gqa_fp64(
        r_out, r_qg, r_k, r_v, r_scores,
        seq_len, max_seq_len, H_q, H_kv, head_dim, scale
    )

    # ---- sisi GPU (buffer dasar, offset 0) ----
    var dev_qg = ctx.enqueue_create_buffer[DType.float32](q_len)
    var dev_k = ctx.enqueue_create_buffer[DType.float32](max_seq_len * kv_dim)
    var dev_v = ctx.enqueue_create_buffer[DType.float32](max_seq_len * kv_dim)
    var dev_out = ctx.enqueue_create_buffer[DType.float32](H_q * head_dim)
    var dev_scores = ctx.enqueue_create_buffer[DType.float32](score_len)
    ctx.enqueue_copy(dev_qg, h_qg)
    ctx.enqueue_copy(dev_k, h_k)
    ctx.enqueue_copy(dev_v, h_v)
    ctx.synchronize()

    gqa_attention_sm75_launch_on[DType.float32](
        ctx,
        dev_qg.unsafe_ptr(), dev_k.unsafe_ptr(), dev_v.unsafe_ptr(),
        dev_out.unsafe_ptr(), dev_scores.unsafe_ptr(),
        seq_len, max_seq_len, H_q, H_kv, head_dim, Float32(scale)
    )
    ctx.synchronize()

    var h_out = alloc[Float32](H_q * head_dim)
    ctx.enqueue_copy(h_out, dev_out)
    ctx.synchronize()

    var max_abs: Float64 = 0.0
    var max_ref: Float64 = 0.0
    for i in range(H_q * head_dim):
        var d = Float64(h_out[i]) - r_out[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_abs:
            max_abs = ad
        var ar = r_out[i] if r_out[i] >= 0.0 else -r_out[i]
        if ar > max_ref:
            max_ref = ar
    var rel = Float32(max_abs / (max_ref + 1e-12))
    var passed = rel <= tol
    var stat = "[PASS]" if passed else "[GAGAL]"
    print(
        stat, name, "| H_q=", H_q, "H_kv=", H_kv, "D=", head_dim,
        "seq=", seq_len, "| rel=", rel, "(tol", tol, ")",
        "| abs=", Float32(max_abs), "ref_maks=", Float32(max_ref)
    )

    h_qg.free(); h_k.free(); h_v.free()
    r_qg.free(); r_k.free(); r_v.free(); r_scores.free(); r_out.free()
    h_out.free()
    return passed


fn main() raises:
    print("=================================================================")
    print(">> TES DIFFERENSIAL KERNEL FUSED GQA ATTENTION + SIGMOID GATE")
    print(">> Referensi: FP64, softmax stabil + pembobotan V + sigmoid(gate).")
    print("=================================================================")
    var ctx = DeviceContext()
    var all_ok = True

    # Toleransi longgar: output FP32, akumulasi GPU FP32 vs referensi FP64.
    var tol = Float32(0.002)

    # 1. Bentuk produksi Bonsai: head_dim=256, H_q=24, H_kv=4 (group=6).
    #    seq_len kecil supaya cepat; kernel tidak bergantung panjang riwayat.
    all_ok = run_gqa_case(ctx, "gqa_prod_h24_kv4_d256_seq8", 24, 4, 256, 8, 16, tol) and all_ok

    # 2. seq_len = max_seq_len (riwayat penuh) + H_q lebih sedikit.
    all_ok = run_gqa_case(ctx, "gqa_h4_kv2_d256_seq16", 4, 2, 256, 16, 16, tol) and all_ok

    # 3. seq_len = 1 (kasus decode token pertama; softmax trivial).
    all_ok = run_gqa_case(ctx, "gqa_h4_kv2_d256_seq1", 4, 2, 256, 1, 16, tol) and all_ok

    # 4. PENJAGA: head_dim != 256 harus DITOLAK, bukan diam-diam salah.
    print("-----------------------------------------------------------------")
    var ditolak = False
    try:
        run_gqa_case(ctx, "gqa_guard_d128_harus_tolak", 4, 2, 128, 4, 8, tol)
    except:
        ditolak = True
    if ditolak:
        print(">> [OK] penjaga menolak head_dim=128 (bukan 256) — mencegah "
              "pembacaan shared memory tak-terinisialisasi")
    else:
        print(">> [FAIL] head_dim=128 DITERIMA — penjaga tidak bekerja!")
        all_ok = False

    print("-----------------------------------------------------------------")
    if all_ok:
        print(">> HASIL AKHIR: SEMUA KASUS GQA PASS!")
    else:
        print(">> HASIL AKHIR: TERDAPAT KASUS GAGAL — PERIKSA rel DI ATAS!")
    print("=================================================================")
