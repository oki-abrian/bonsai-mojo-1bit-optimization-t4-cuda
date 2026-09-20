# ===----------------------------------------------------------------------=== #
# Module: tests/test_gdn_state_precision.mojo
# Purpose: Tes diferensial kernel rekurensi Gated DeltaNet (sm_75) vs referensi
#          FP64 CPU, dengan fokus pada PRESISI STATE.
#
# Latar: state rekuren S=[H_v, D_v, D_k] dulu disimpan fp16 dan dibulatkan 2x
#        per langkah. Karena decay ~ 1.0, galat pembulatan TIDAK teredam dan
#        menumpuk. Kernel ini sebelumnya sama sekali tidak punya test numerik
#        (QMV/QMM/RoPE/Argmax ada; delta-rule tidak).
#
# Yang diuji: dengan state fp32, galat relatif terhadap referensi FP64 harus
#             tetap kecil setelah banyak langkah. Kalau state kembali ke fp16,
#             galat melonjak ~100x dan test GAGAL.
#
# CATATAN HARNESS (penting, hasil investigasi 2026-09-12):
#   `ctx.enqueue_function` TIDAK menangani argumen pointer device yang sudah
#   di-`.offset(n)` (n>0). Kernel membaca sampah. Terbukti dengan kernel probe
#   minimal: `buf.unsafe_ptr()` benar, `buf.unsafe_ptr().offset(n)` -> sampah,
#   bahkan pada launch PERTAMA di konteks segar. Buffer TERPISAH (offset 0)
#   terbukti benar.
#   Karena itu test ini TIDAK memakai offset pada pointer device: tiap langkah
#   datanya disalin dari host ke buffer "scratch" (offset 0) lalu di-launch.
#   Ini juga sebabnya test multi-langkah versi lama gagal padahal kernelnya benar.
#   Lihat juga: tests/test_enqueue_offset.mojo (regresi khusus masalah ini).
#
# Catatan desain:
#   - Input q/k/v/a/b dibulatkan ke FP16 DULU, lalu referensi memakai nilai
#     FP16 itu juga. Jadi selisih yang terukur murni dari akumulasi state.
#   - Parameter dipilih supaya decay ~ 0.99 (a_log = -4.6, softplus ~ 1.0),
#     yaitu rezim "nyaris tidak melupakan" — rezim yang sama dengan model.
#   - Aturan delta stabil hanya bila beta*|k|^2 < 2. Dengan k ~ [-0.5,0.5) dan
#     beta = 0.1 (b = -2.197), beta*|k|^2 ~ 0.13 << 1. Kalau tidak stabil,
#     referensi FP64 sendiri meledak dan test mengukur kekacauan, bukan presisi.
#
# Jalankan: pixi run mojo run -I . tests/test_gdn_state_precision.mojo
# ===----------------------------------------------------------------------=== #

from math import exp, log, sqrt
from memory import UnsafePointer, alloc
from gpu.host import DeviceContext, DeviceBuffer
from src.ops import (
    gdn_recurrence_sm75_launch_on, gdn_norm_gate_sm75_launch_on
)


# ---------------------------------------------------------------------------
# Referensi FP64 — meniru persis urutan operasi kernel:
#   1. sdec = S*decay ; S = sdec ; kv_mem += sdec*k
#   2. delta = (v - kv_mem) * beta
#   3. S = S + k*delta ; out += S*q
# `round_fp16` mensimulasikan bug lama (state disimpan fp16) untuk mengukur
# daya-beda test ini.
# ---------------------------------------------------------------------------
fn gdn_ref_step(
    s: UnsafePointer[Float64, MutAnyOrigin],
    q: UnsafePointer[Float16, MutAnyOrigin],
    k: UnsafePointer[Float16, MutAnyOrigin],
    v: UnsafePointer[Float16, MutAnyOrigin],
    a: UnsafePointer[Float16, MutAnyOrigin],
    b: UnsafePointer[Float16, MutAnyOrigin],
    a_log: UnsafePointer[Float32, MutAnyOrigin],
    dt_bias: UnsafePointer[Float32, MutAnyOrigin],
    out_ref: UnsafePointer[Float64, MutAnyOrigin],
    has_params: Bool,
    repeat_factor: Int,
    round_fp16: Bool,
    H_v: Int,
    D_v: Int,
    D_k: Int
):
    for hv in range(H_v):
        var hk = hv // repeat_factor
        var a_val = Float64(Float32(a[hv]))
        var b_val = Float64(Float32(b[hv]))
        var g_decay: Float64
        if has_params:
            var sp_in = a_val + Float64(dt_bias[hv])
            var sp = sp_in if sp_in > 20.0 else log(1.0 + exp(sp_in))
            g_decay = exp(-exp(Float64(a_log[hv])) * sp)
        else:
            var sp_in = a_val + 1.0
            var sp = sp_in if sp_in > 20.0 else log(1.0 + exp(sp_in))
            g_decay = exp(-0.5 * sp)
        var beta = 1.0 / (1.0 + exp(-b_val))

        for dv in range(D_v):
            var row = (hv * D_v + dv) * D_k
            var kv_mem: Float64 = 0.0
            for dk in range(D_k):
                var sdec = s[row + dk] * g_decay
                s[row + dk] = sdec
                kv_mem += sdec * Float64(Float32(k[hk * D_k + dk]))
            var delta = (Float64(Float32(v[hv * D_v + dv])) - kv_mem) * beta
            var acc: Float64 = 0.0
            for dk in range(D_k):
                var snew = s[row + dk] + Float64(Float32(k[hk * D_k + dk])) * delta
                if round_fp16:
                    snew = Float64(Float32(Float16(Float32(snew))))
                s[row + dk] = snew
                acc += snew * Float64(Float32(q[hk * D_k + dk]))
            out_ref[hv * D_v + dv] = acc


fn run_gdn_test(
    mut ctx: DeviceContext,
    name: String,
    H_v: Int,
    H_k: Int,
    D_v: Int,
    D_k: Int,
    T_steps: Int,
    has_params: Bool,
    tol_state: Float32
) raises -> Bool:
    var repeat_factor = H_v // H_k
    var qk_len = H_k * D_k
    var v_len = H_v * D_v
    var s_len = H_v * D_v * D_k

    # ---------------- host: input, dibulatkan ke FP16 ----------------
    var h_q = alloc[Float16](T_steps * qk_len)
    var h_k = alloc[Float16](T_steps * qk_len)
    var h_v = alloc[Float16](T_steps * v_len)
    var h_a = alloc[Float16](T_steps * H_v)
    var h_b = alloc[Float16](T_steps * H_v)
    var h_alog = alloc[Float32](H_v)
    var h_dtb = alloc[Float32](H_v)

    for t in range(T_steps):
        for j in range(qk_len):
            h_q[t * qk_len + j] = Float16(
                Float32((j * 7919 + t * 104729 + 1013) % 2000) * 0.0005 - 0.5
            )
            h_k[t * qk_len + j] = Float16(
                Float32((j * 6271 + t * 15485863 + 5003) % 2000) * 0.0005 - 0.5
            )
        for j in range(v_len):
            h_v[t * v_len + j] = Float16(
                Float32((j * 4931 + t * 32452843 + 7919) % 2000) * 0.001 - 1.0
            )
        for j in range(H_v):
            h_a[t * H_v + j] = Float16(0.5413)   # softplus(a+0) ~ 1.0 -> decay ~0.99
            h_b[t * H_v + j] = Float16(-2.197)   # sigmoid(b) = 0.1
    for j in range(H_v):
        h_alog[j] = -4.6
        h_dtb[j] = 0.0

    # ---------------- referensi FP64 (state fp32) ----------------
    var ref_s = alloc[Float64](s_len)
    var ref_out = alloc[Float64](T_steps * v_len)
    for i in range(s_len):
        ref_s[i] = 0.0
    for i in range(T_steps * v_len):
        ref_out[i] = 0.0
    for t in range(T_steps):
        gdn_ref_step(
            ref_s, h_q.offset(t * qk_len), h_k.offset(t * qk_len),
            h_v.offset(t * v_len), h_a.offset(t * H_v), h_b.offset(t * H_v),
            h_alog, h_dtb, ref_out.offset(t * v_len),
            has_params, repeat_factor, False, H_v, D_v, D_k
        )

    # ---------------- referensi pembanding: state FP16 (bug lama) ----------------
    var ref16_s = alloc[Float64](s_len)
    for i in range(s_len):
        ref16_s[i] = 0.0
    for t in range(T_steps):
        gdn_ref_step(
            ref16_s, h_q.offset(t * qk_len), h_k.offset(t * qk_len),
            h_v.offset(t * v_len), h_a.offset(t * H_v), h_b.offset(t * H_v),
            h_alog, h_dtb, ref_out.offset(t * v_len),
            has_params, repeat_factor, True, H_v, D_v, D_k
        )

    # ---------------- sisi GPU ----------------
    # Buffer state TUNGGAL (pointer dasar, tidak pernah di-offset).
    var dev_s = ctx.enqueue_create_buffer[DType.float32](s_len)
    var h_zero = alloc[Float32](s_len)
    for i in range(s_len):
        h_zero[i] = 0.0
    ctx.enqueue_copy(dev_s, h_zero)

    # Buffer scratch PER-LANGKAH (selalu offset 0) — menghindari bug offset
    # pointer device pada `enqueue_function` (lihat catatan harness di atas).
    var dev_q = ctx.enqueue_create_buffer[DType.float16](qk_len)
    var dev_k = ctx.enqueue_create_buffer[DType.float16](qk_len)
    var dev_v = ctx.enqueue_create_buffer[DType.float16](v_len)
    var dev_a = ctx.enqueue_create_buffer[DType.float16](H_v)
    var dev_b = ctx.enqueue_create_buffer[DType.float16](H_v)
    var dev_alog = ctx.enqueue_create_buffer[DType.float32](H_v)
    var dev_dtb = ctx.enqueue_create_buffer[DType.float32](H_v)
    var dev_out = ctx.enqueue_create_buffer[DType.float16](v_len)

    var hs_q = alloc[Float16](qk_len)
    var hs_k = alloc[Float16](qk_len)
    var hs_v = alloc[Float16](v_len)
    var hs_a = alloc[Float16](H_v)
    var hs_b = alloc[Float16](H_v)
    var hs_out = alloc[Float16](v_len)

    ctx.enqueue_copy(dev_alog, h_alog)
    ctx.enqueue_copy(dev_dtb, h_dtb)
    ctx.synchronize()

    var got_out = alloc[Float16](T_steps * v_len)

    for t in range(T_steps):
        for j in range(qk_len):
            hs_q[j] = h_q[t * qk_len + j]
            hs_k[j] = h_k[t * qk_len + j]
        for j in range(v_len):
            hs_v[j] = h_v[t * v_len + j]
        for j in range(H_v):
            hs_a[j] = h_a[t * H_v + j]
            hs_b[j] = h_b[t * H_v + j]

        ctx.enqueue_copy(dev_q, hs_q)
        ctx.enqueue_copy(dev_k, hs_k)
        ctx.enqueue_copy(dev_v, hs_v)
        ctx.enqueue_copy(dev_a, hs_a)
        ctx.enqueue_copy(dev_b, hs_b)

        gdn_recurrence_sm75_launch_on[DType.float16](
            ctx,
            dev_s.unsafe_ptr(),
            dev_q.unsafe_ptr(),
            dev_k.unsafe_ptr(),
            dev_v.unsafe_ptr(),
            dev_a.unsafe_ptr(),
            dev_b.unsafe_ptr(),
            dev_alog.unsafe_ptr(),
            dev_dtb.unsafe_ptr(),
            has_params,
            dev_out.unsafe_ptr(),
            repeat_factor, H_v, D_v, D_k
        )
        ctx.synchronize()
        ctx.enqueue_copy(hs_out, dev_out)
        ctx.synchronize()
        for j in range(v_len):
            got_out[t * v_len + j] = hs_out[j]

    var got_s = alloc[Float32](s_len)
    ctx.enqueue_copy(got_s, dev_s)
    ctx.synchronize()

    # ---------------- bandingkan STATE (fp32 vs FP64) ----------------
    var max_abs: Float64 = 0.0
    var max_ref: Float64 = 0.0
    var max_abs16: Float64 = 0.0
    var worst_idx = -1
    for i in range(s_len):
        var d = Float64(got_s[i]) - ref_s[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_abs:
            max_abs = ad
            worst_idx = i
        var d16 = ref16_s[i] - ref_s[i]
        var ad16 = d16 if d16 >= 0.0 else -d16
        if ad16 > max_abs16:
            max_abs16 = ad16
        var ar = ref_s[i] if ref_s[i] >= 0.0 else -ref_s[i]
        if ar > max_ref:
            max_ref = ar
    var rel_state = Float32(max_abs / (max_ref + 1e-12))
    var rel_state16 = Float32(max_abs16 / (max_ref + 1e-12))

    # ---------------- bandingkan OUT (fp16) ----------------
    var max_out: Float64 = 0.0
    var max_out_ref: Float64 = 0.0
    for i in range(T_steps * v_len):
        var d = Float64(Float32(got_out[i])) - ref_out[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_out:
            max_out = ad
        var ar = ref_out[i] if ref_out[i] >= 0.0 else -ref_out[i]
        if ar > max_out_ref:
            max_out_ref = ar
    var rel_out = Float32(max_out / (max_out_ref + 1e-12))

    var passed = rel_state <= tol_state
    if max_ref > 1e6:
        passed = False
        print(
            "        [TEST-TIDAK-VALID] referensi FP64 meledak ke", max_ref,
            "-> aturan delta tidak stabil; perkecil beta atau |k|."
        )
    var stat = "[PASS]" if passed else "[GAGAL]"
    print(
        stat, name,
        "| H_v=", H_v, "D_v=", D_v, "D_k=", D_k, "T=", T_steps,
        "params=", has_params,
        "| rel_state=", rel_state, "(tol", tol_state, ")",
        "| rel_out=", rel_out,
        "| worst_idx=", worst_idx
    )
    print(
        "        daya-beda: rel_state bila state FP16 =", rel_state16,
        "->", Float32(rel_state16 / (rel_state + 1e-12)), "x lebih buruk"
    )

    h_q.free(); h_k.free(); h_v.free(); h_a.free(); h_b.free()
    h_alog.free(); h_dtb.free(); h_zero.free()
    hs_q.free(); hs_k.free(); hs_v.free(); hs_a.free(); hs_b.free(); hs_out.free()
    ref_s.free(); ref16_s.free(); ref_out.free(); got_s.free(); got_out.free()
    return passed


# ---------------------------------------------------------------------------
# Tes diferensial KEDUA: kernel norm+gate GDN (`gdn_norm_gate_sm75_gpu`).
#
# Regresi yang dijaga: BUG-1 — URUTAN RMSNorm vs gate. Kernel WAJIB
# menormalkan `x` (keluaran rekurensi) MURNI, lalu mengalikan `silu(z)`
# PALING AKHIR, paritas Qwen3NextRMSNormGated (qwen3_next.py:71-78,
# rmsnorm_gated.cu:80-96). Urutan terbalik (norm atas x*silu(z)) menghasilkan
# galat gain per-head rms(x)/rms(x*silu(z)) di SETIAP layer GDN.
#
# DAYA-BEDA: tes ini menjalankan kernel dgn KEDUA urutan (lewat
# `legacy_override`) dan menuntut urutan BENAR cocok dgn FP64 SEDANGKAN urutan
# LAMA menyimpang. Kalau keduanya lolos, tes tidak punya daya-beda.
#
# Semantik yang sama juga dikunci di CPU (tanpa GPU) oleh
# tests/test_gdn_norm_gate_order.mojo.
# ---------------------------------------------------------------------------
fn silu_f64(x: Float64) -> Float64:
    return x / (1.0 + exp(-x))


fn ref_norm_gate_fp64(
    dst: UnsafePointer[Float64, MutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    z: UnsafePointer[Float64, MutAnyOrigin],
    gamma: UnsafePointer[Float64, MutAnyOrigin],
    D: Int,
    eps: Float64
):
    """Qwen3NextRMSNormGated, FP64: rms_norm atas x murni, lalu dikali silu(z)."""
    var ss: Float64 = 0.0
    for d in range(D):
        ss += x[d] * x[d]
    var inv = 1.0 / sqrt(ss / Float64(D) + eps)
    for d in range(D):
        dst[d] = (x[d] * inv * gamma[d]) * silu_f64(z[d])


fn run_norm_gate_test(
    mut ctx: DeviceContext,
    name: String,
    H_v: Int,
    D_v: Int,
    legacy: Int,
    tol: Float32
) raises -> Bool:
    var n = H_v * D_v
    var eps = Float64(1e-6)

    # Input x (keluaran rekurensi) & z (gate), dibulatkan FP16 seperti kernel.
    var h_x = alloc[Float16](n)
    var h_z = alloc[Float16](n)
    var h_g = alloc[Float32](D_v)
    for i in range(n):
        h_x[i] = Float16(Float32((i * 7919 + 1013) % 2000) * 0.001 - 1.0)
    for i in range(n):
        # z sengaja mencakup rentang sangat negatif (silu -> 0) dan positif;
        # di situlah kedua urutan paling berbeda.
        h_z[i] = Float16(Float32((i * 6271 + 5003) % 2000) * 0.008 - 8.0)
    for d in range(D_v):
        h_g[d] = Float32(0.5) + Float32((d * 37) % 1000) * 0.001

    # Referensi FP64 memakai nilai FP16 yang SAMA, jadi selisih yang terukur
    # murni berasal dari urutan operasi + akumulasi FP32 vs FP64.
    var rx = alloc[Float64](n)
    var rz = alloc[Float64](n)
    var rg = alloc[Float64](D_v)
    for i in range(n):
        rx[i] = Float64(Float32(h_x[i]))
        rz[i] = Float64(Float32(h_z[i]))
    for d in range(D_v):
        rg[d] = Float64(h_g[d])
    var refv = alloc[Float64](n)
    for hv in range(H_v):
        ref_norm_gate_fp64(
            refv.offset(hv * D_v), rx.offset(hv * D_v), rz.offset(hv * D_v),
            rg, D_v, eps
        )

    # ---- sisi GPU (buffer dasar, offset 0 — lihat catatan harness di atas) ----
    var dev_x = ctx.enqueue_create_buffer[DType.float16](n)
    var dev_z = ctx.enqueue_create_buffer[DType.float16](n)
    var dev_g = ctx.enqueue_create_buffer[DType.float32](D_v)
    ctx.enqueue_copy(dev_x, h_x)
    ctx.enqueue_copy(dev_z, h_z)
    ctx.enqueue_copy(dev_g, h_g)
    ctx.synchronize()

    gdn_norm_gate_sm75_launch_on[DType.float16](
        ctx, dev_x.unsafe_ptr(), dev_z.unsafe_ptr(), dev_g.unsafe_ptr(),
        True, H_v, D_v, Float32(eps), 1, 0, 0, legacy
    )
    ctx.synchronize()

    var got = alloc[Float16](n)
    ctx.enqueue_copy(got, dev_x)
    ctx.synchronize()

    var max_abs: Float64 = 0.0
    var max_ref: Float64 = 0.0
    for i in range(n):
        var d = Float64(Float32(got[i])) - refv[i]
        var ad = d if d >= 0.0 else -d
        if ad > max_abs:
            max_abs = ad
        var ar = refv[i] if refv[i] >= 0.0 else -refv[i]
        if ar > max_ref:
            max_ref = ar
    var rel = Float32(max_abs / (max_ref + 1e-12))
    var passed = rel <= tol
    var stat = "[PASS]" if passed else "[GAGAL]"
    print(
        stat, name, "| H_v=", H_v, "D_v=", D_v, "legacy=", legacy,
        "| rel=", rel, "(tol", tol, ")",
        "| abs=", Float32(max_abs), "ref_maks=", Float32(max_ref)
    )

    h_x.free(); h_z.free(); h_g.free()
    rx.free(); rz.free(); rg.free(); refv.free(); got.free()
    return passed


fn main() raises:
    print("=================================================================")
    print(">> TES PRESISI STATE REKURENSI GDN (sm_75) vs REFERENSI FP64")
    print(">> Regresi yang dijaga: state S harus FP32, bukan FP16.")
    print("=================================================================")
    var ctx = DeviceContext()
    var all_ok = True

    # Toleransi dipilih di TENGAH antara dua rezim terukur:
    #   state FP32 -> rel_state ~1.5e-6 (margin ~66x di bawah toleransi)
    #   state FP16 -> rel_state ~1.3e-3 (margin ~13x di atas toleransi)
    # Jadi test ini punya daya-beda ~1000x, bukan sekadar lulus asal-asalan.
    var tol = Float32(0.0001)

    # 1. Satu langkah — kontrol dasar.
    all_ok = run_gdn_test(ctx, "step1_h2_dv8_dk8", 2, 2, 8, 8, 1, True, tol) and all_ok

    # 2. 64 langkah, decay ~0.99 (rezim "nyaris tidak melupakan").
    all_ok = run_gdn_test(ctx, "step64_h2_dv8_dk8", 2, 2, 8, 8, 64, True, tol) and all_ok

    # 3. 512 langkah — sepanjang budget generasi yang runtuh di T4.
    all_ok = run_gdn_test(ctx, "step512_h2_dv8_dk8", 2, 2, 8, 8, 512, True, tol) and all_ok

    # 4. Tanpa parameter (HAS_PARAMS=False) — decay = exp(-0.5*softplus(a+1)).
    all_ok = run_gdn_test(ctx, "step64_noparams", 2, 2, 8, 8, 64, False, tol) and all_ok

    # 5. Bentuk produksi Bonsai: 4 head-K -> 8 head-V (repeat_factor=2), D=16.
    all_ok = run_gdn_test(ctx, "step64_rep2_dv16_dk16", 8, 4, 16, 16, 64, True, tol) and all_ok

    # -----------------------------------------------------------------
    # 6. Kernel NORM+GATE (BUG-1): urutan RMSNorm vs gate.
    #    Urutan BENAR harus cocok; urutan LAMA WAJIB menyimpang.
    # -----------------------------------------------------------------
    print("-----------------------------------------------------------------")
    print(">> TES DIFFERENSIAL KERNEL NORM+GATE GDN (BUG-1)")
    print(">> Kernel harus: norm atas x MURNI, lalu silu(z) TERAKHIR.")
    print("-----------------------------------------------------------------")
    var tol_ng = Float32(0.002)
    # Bentuk produksi: D_v = 128 (gdn_head_v_dim), beberapa head.
    all_ok = run_norm_gate_test(ctx, "normgate_h4_dv128_ok", 4, 128, 0, tol_ng) and all_ok
    # Bentuk kecil tambahan (latensi tes rendah, cakupan head > 1).
    all_ok = run_norm_gate_test(ctx, "normgate_h2_dv16_ok", 2, 16, 0, tol_ng) and all_ok
    # Reduksi warp non-trivial (regresi 2026-09-20): jalur umum harus benar untuk
    # D_v yang BUKAN 128. D_v=64/96 = warp penuh tapi < 4 warp (tahap antar-warp
    # lama membaca smem yang tak pernah ditulis); D_v=48 = warp terakhir parsial
    # (ladder shuffle lama membaca lane mati). Sebelum perbaikan ketiganya
    # menghasilkan RMS yang salah besar tanpa crash.
    all_ok = run_norm_gate_test(ctx, "normgate_h2_dv48_ok", 2, 48, 0, tol_ng) and all_ok
    all_ok = run_norm_gate_test(ctx, "normgate_h2_dv64_ok", 2, 64, 0, tol_ng) and all_ok
    all_ok = run_norm_gate_test(ctx, "normgate_h2_dv96_ok", 2, 96, 0, tol_ng) and all_ok

    # Urutan LAMA: diharapkan GAGAL. Kalau justru lolos, tes ini tidak punya
    # daya-beda dan regresi BUG-1 tidak akan pernah tertangkap.
    var legacy_lolos = run_norm_gate_test(ctx, "normgate_h4_dv128_legacy", 4, 128, 1, tol_ng)
    if legacy_lolos:
        print(">> [FAIL] urutan LAMA LOLOS toleransi — tes TIDAK punya daya-beda "
              "untuk regresi BUG-1!")
        all_ok = False
    else:
        print(">> [OK] urutan LAMA menyimpang dari FP64 -> tes punya daya-beda "
              "untuk regresi BUG-1")

    print("-----------------------------------------------------------------")
    if all_ok:
        print(">> HASIL AKHIR: SEMUA KASUS PRESISI STATE GDN PASS!")
    else:
        print(">> HASIL AKHIR: TERDAPAT KASUS GAGAL — PERIKSA rel_state DI ATAS!")
    print("=================================================================")
