# ===----------------------------------------------------------------------=== #
# Module: tests/test_gdn_norm_gate_order.mojo
# Purpose: Spesifikasi-eksekusi + pembuktian daya-beda untuk URUTAN
#          RMSNorm vs gate pada keluaran Gated DeltaNet (BUG-1).
#
# LATAR (BUG-1, lihat LAPORAN_PERBAIKAN_BUG_JALUR_GPU.md §3.1):
#   `gdn_norm_gate_sm75_gpu` DULU menghitung variance atas `x*silu(z)`, lalu
#   mengalikan bobot norm. Referensi menghitung variance atas `x` MURNI, baru
#   mengalikan `silu(z)` PALING AKHIR:
#     * mlx-lm qwen3_next.py:71-78
#           x = mx.fast.rms_norm(hidden_states, weight, eps)
#           return silu(gate) * x
#     * prism CUDA rmsnorm_gated.cu:80-96
#           local_sq += xv*xv        # x MURNI
#           normed = xv * scale * wv
#           gated  = normed * silu(gv)
#   Karena kembar CPU (gated_delta.mojo) ikut salah, SEMUA tes paritas
#   GPU-vs-CPU lulus bersama-sama dan bug ini tidak pernah terdeteksi.
#
# APA YANG DIUJI FILE INI (dan apa yang TIDAK):
#   DIUJI: kontrak matematisnya. Kedua rumus (BENAR dan LAMA) dihitung di CPU
#          dan dibandingkan terhadap referensi FP64 independen. Ini mengunci
#          semantik + mengukur BESAR galat yang ditimbulkan bug lama.
#   TIDAK: kode kernel GPU-nya sendiri (`gdn_norm_gate_sm75_gpu`). Itu butuh
#          T4 — lihat tes diferensial kernel di tests/test_gdn_state_precision.mojo
#          dan jalur `gdn_norm_gate_sm75_launch_on`.
#   Jadi file ini adalah JARING PENGAMAN SEMANTIK, bukan bukti numerik GPU.
#
# MENGAPA INI BERGUNA WALAU BUKAN TES KERNEL:
#   Kalau seseorang membalik lagi urutannya, tes ini tetap lulus (rumusnya
#   masih "benar" di atas kertas) — tetapi ia mendokumentasikan kontrak secara
#   EKSEKUTABEL dan mengukur daya-beda tes: berapa besar galat yang HARUS
#   terlihat oleh tes kernel apa pun. Angka itu dipakai sebagai ambang di tes T4.
#
# Jalankan (CPU saja, tidak butuh GPU):
#   pixi run mojo run tests/test_gdn_norm_gate_order.mojo
# atau tanpa pixi:
#   MODULAR_HOME=<env>/share/max <env>/bin/mojo run tests/test_gdn_norm_gate_order.mojo
# ===----------------------------------------------------------------------=== #

from math import exp, sqrt
from memory import UnsafePointer, alloc


fn silu(x: Float64) -> Float64:
    """SiLU/Swish: x * sigmoid(x)."""
    return x / (1.0 + exp(-x))


fn ref_norm_gated(
    dst: UnsafePointer[Float64, MutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    z: UnsafePointer[Float64, MutAnyOrigin],
    gamma: UnsafePointer[Float64, MutAnyOrigin],
    D: Int,
    eps: Float64
):
    """Referensi Qwen3NextRMSNormGated — FP64, urutan BENAR.

    Persis `qwen3_next.py:71-78`: rms_norm atas x murni, lalu dikali silu(gate).
    """
    var ss: Float64 = 0.0
    for d in range(D):
        ss += x[d] * x[d]
    var inv = 1.0 / sqrt(ss / Float64(D) + eps)
    for d in range(D):
        dst[d] = (x[d] * inv * gamma[d]) * silu(z[d])


fn kernel_formula(
    dst: UnsafePointer[Float64, MutAnyOrigin],
    x: UnsafePointer[Float64, MutAnyOrigin],
    z: UnsafePointer[Float64, MutAnyOrigin],
    gamma: UnsafePointer[Float64, MutAnyOrigin],
    D: Int,
    eps: Float64,
    legacy_gate_first: Bool
):
    """Rumus kernel `gdn_norm_gate_sm75_gpu`, disalin apa adanya.

    legacy_gate_first=False -> urutan BENAR (norm atas x murni, gate terakhir).
    legacy_gate_first=True  -> bug LAMA (norm atas x*silu(z); silu(z) TIDAK
                               dikali lagi di akhir).
    """
    var norm_in = alloc[Float64](D)
    for d in range(D):
        norm_in[d] = (x[d] * silu(z[d])) if legacy_gate_first else x[d]
    var ss: Float64 = 0.0
    for d in range(D):
        ss += norm_in[d] * norm_in[d]
    var inv = 1.0 / sqrt(ss / Float64(D) + eps)
    for d in range(D):
        if legacy_gate_first:
            dst[d] = norm_in[d] * inv * gamma[d]
        else:
            dst[d] = (x[d] * inv * gamma[d]) * silu(z[d])
    norm_in.free()


fn max_abs_diff(
    a: UnsafePointer[Float64, MutAnyOrigin],
    b: UnsafePointer[Float64, MutAnyOrigin],
    n: Int
) -> Float64:
    var m: Float64 = 0.0
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


fn rms(v: UnsafePointer[Float64, MutAnyOrigin], n: Int) -> Float64:
    var ss: Float64 = 0.0
    for i in range(n):
        ss += v[i] * v[i]
    return sqrt(ss / Float64(n))


fn main() raises:
    # D_v = 128 seperti model (gdn_head_v_dim), eps seperti config.
    alias D = 128
    var eps = Float64(1e-6)
    # 3 seed deterministik: satu "jinak", satu dgn z sangat negatif (silu ~ 0,
    # pembeda terkuat), satu dgn z besar positif.
    var seeds = List[Int](101, 202, 303)

    print("=== tes urutan RMSNorm vs gate GDN (BUG-1) ===")
    print("D_v =", D, " eps =", eps)
    print("")

    var worst_correct: Float64 = 0.0
    var worst_legacy: Float64 = 0.0
    var worst_gain_ratio: Float64 = 0.0
    var min_gain_ratio: Float64 = 1.0e30

    for si in range(len(seeds)):
        var seed = seeds[si]
        var x = alloc[Float64](D)
        var z = alloc[Float64](D)
        var gamma = alloc[Float64](D)
        # PRNG LCG sederhana, deterministik.
        var state: Int = seed
        for d in range(D):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            x[d] = (Float64(state % 20000) / 10000.0) - 1.0     # x ~ [-1, 1)
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            var zr = Float64(state % 20000) / 10000.0 - 1.0
            # Seed ke-2: dorong z sangat negatif -> silu(z) ~ 0
            if si == 1:
                zr = zr * 8.0 - 6.0
            # Seed ke-3: z besar positif -> silu(z) ~ z
            if si == 2:
                zr = zr * 6.0 + 4.0
            z[d] = zr
            gamma[d] = 0.5 + Float64((state % 1000)) / 1000.0    # ~ [0.5, 1.5)

        var refv = alloc[Float64](D)
        var got_ok = alloc[Float64](D)
        var got_legacy = alloc[Float64](D)
        ref_norm_gated(refv, x, z, gamma, D, eps)
        kernel_formula(got_ok, x, z, gamma, D, eps, False)
        kernel_formula(got_legacy, x, z, gamma, D, eps, True)

        var d_ok = max_abs_diff(refv, got_ok, D)
        var d_legacy = max_abs_diff(refv, got_legacy, D)

        # Gain per-head yang ditimbulkan bug lama: rms(x)/rms(x*silu(z)).
        var xs = alloc[Float64](D)
        for d in range(D):
            xs[d] = x[d] * silu(z[d])
        var rx = rms(x, D)
        var rxs = rms(xs, D)
        var gain = rx / rxs if rxs > 0.0 else 0.0
        xs.free()

        print("seed", seed,
              "| BENAR vs FP64 max|Δ| =", d_ok,
              "| LAMA vs FP64 max|Δ| =", d_legacy,
              "| gain rms(x)/rms(x*silu(z)) =", gain)

        if d_ok > worst_correct:
            worst_correct = d_ok
        if d_legacy > worst_legacy:
            worst_legacy = d_legacy
        if gain > worst_gain_ratio:
            worst_gain_ratio = gain
        if gain < min_gain_ratio:
            min_gain_ratio = gain

        x.free()
        z.free()
        gamma.free()
        refv.free()
        got_ok.free()
        got_legacy.free()

    print("")
    print("ringkasan:")
    print("  galat urutan BENAR  (maks) :", worst_correct)
    print("  galat urutan LAMA   (maks) :", worst_legacy)
    print("  rentang gain bug lama      :", min_gain_ratio, "..", worst_gain_ratio)
    print("")

    # --- Assertion 1: urutan BENAR harus sama persis dgn referensi FP64. ---
    # Keduanya memakai operasi yang sama dalam orde yang sama, jadi selisih
    # seharusnya nol sampai pembulatan; ambang longgar 1e-9 sudah cukup.
    var ok = True
    if worst_correct > 1.0e-9:
        print(">> [FAIL] urutan BENAR menyimpang dari referensi FP64:",
              worst_correct)
        ok = False
    else:
        print(">> [OK] urutan BENAR == referensi FP64 (Qwen3NextRMSNormGated)")

    # --- Assertion 2: urutan LAMA harus JELAS berbeda -> tes ini punya daya. ---
    # Kalau ini tidak terpenuhi, artinya urutan tidak berpengaruh dan bug-nya
    # tidak akan pernah terdeteksi oleh tes apa pun (situasi sebelum perbaikan).
    if worst_legacy <= 1.0e-3:
        print(">> [FAIL] urutan LAMA tidak terbedakan — tes ini TIDAK punya "
              "daya-beda, sehingga regresi BUG-1 tidak akan tertangkap.")
        ok = False
    else:
        print(">> [OK] urutan LAMA terbedakan (galat s/d", worst_legacy,
              ") -> tes punya daya-beda")

    if ok:
        print(">> [PASS] kontrak norm/gate terkunci & bug lama terbukti terbedakan")
    else:
        print(">> [FAIL] lihat pesan di atas")
        raise Error("tes urutan norm/gate GDN GAGAL")
