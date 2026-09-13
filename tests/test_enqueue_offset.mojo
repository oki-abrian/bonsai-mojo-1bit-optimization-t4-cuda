# ===----------------------------------------------------------------------=== #
# Module: tests/test_enqueue_offset.mojo
# Purpose: Regresi HALUS untuk satu bug harness yang ditemukan 2026-09-12:
#
#   `ctx.enqueue_function` MENGABAIKAN offset pada argumen pointer device.
#   - `buf.unsafe_ptr()`                 -> kernel membaca data BENAR.
#   - `buf.unsafe_ptr().offset(n)` (n>0) -> kernel membaca SAMPAH.
#
#   Terbukti dengan kernel probe minimal (hanya menyalin nilai yang dibacanya):
#     launch 1 (offset 0)  -> q=0.11 k=0.21 v=0.31   (benar)
#     launch 2 (offset n)  -> q=-4.8e-07 ...          (sampah)
#   Juga terjadi pada launch PERTAMA di konteks segar -> bukan soal urutan.
#   Buffer TERPISAH (selalu offset 0) terbukti BENAR.
#
# KENAPA PENTING (TARGET: T4 / CUDA):
#   `src/models/qwen3_5/layer.mojo` memakai pola offset device ini:
#     - baris 484-493: `causal_conv1d_sm75_launch_on(..., proj_m_dev.offset(t*in_proj_n), ...)`
#       di dalam `for t in range(M)` -> TANPA syarat, selalu jalan saat prefill.
#     - baris 523-537: loop fallback `gdn_recurrence_sm75_launch_on(... .offset(t*qk_dim) ...)`
#   Kalau offset tidak dihormati di T4, maka prefill (M>1) memakai data SAMPAH
#   untuk t>=1 -> kandidat kuat penyebab degenerasi generasi panjang.
#
# CATATAN METODOLOGI: pengamatan awal ini muncul saat iterasi di mesin
# pengembang (backend non-CUDA). Hasil itu TIDAK otoritatif untuk T4 dan
# sengaja tidak dijadikan dasar kesimpulan. Test ini adalah satu-satunya cara
# menjawabnya: jalankan di T4 dan baca VERDICT-nya.
#
# Test ini SENGAJA tidak fatal secara default, supaya bisa dijalankan di mana
# saja dan MELAPORKAN verdictnya. Set `BONSAI_OFFSET_STRICT=1` untuk menjadikan
# kegagalan offset sebagai exit code != 0.
#
# Jalankan: pixi run mojo run -I . tests/test_enqueue_offset.mojo
# ===----------------------------------------------------------------------=== #

from memory import UnsafePointer, alloc
from os import getenv
from gpu.host import DeviceContext, DeviceBuffer
from gpu import block_idx, thread_idx


fn probe_read(
    q: UnsafePointer[Float16, MutAnyOrigin],
    dump: UnsafePointer[Float32, MutAnyOrigin],
    n: Int
):
    """Kernel minimal: salin n nilai pertama yang DILIHAT di `q` ke `dump`."""
    if block_idx.x == 0 and thread_idx.x == 0:
        for i in range(n):
            dump[i] = Float32(q[i])


fn main() raises:
    print("=================================================================")
    print(">> REGRESI: OFFSET POINTER DEVICE PADA ctx.enqueue_function")
    print(">> Kalau bagian A benar dan bagian B salah -> bug offset KONFIRMASI.")
    print("=================================================================")
    var ctx = DeviceContext()
    var n = 4

    var h_q = alloc[Float16](2 * n)
    for i in range(n):
        h_q[i] = Float16(0.11)      # set A
        h_q[n + i] = Float16(0.99)  # set B

    var d_q = ctx.enqueue_create_buffer[DType.float16](2 * n)
    var d_dump = ctx.enqueue_create_buffer[DType.float32](n)
    ctx.enqueue_copy(d_q, h_q)
    ctx.synchronize()

    var h_dump = alloc[Float32](n)

    # ---- A: pointer DASAR (offset 0) -> harus membaca 0.11 ----
    ctx.enqueue_function[probe_read](
        d_q.unsafe_ptr(), d_dump.unsafe_ptr(), n,
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
    )
    ctx.synchronize()
    ctx.enqueue_copy(h_dump, d_dump)
    ctx.synchronize()
    var got_base = h_dump[0]
    var base_ok = abs(Float64(got_base) - 0.11) < 0.01
    print("A. pointer dasar   : dibaca =", got_base, " harap 0.11 ->",
          "[BENAR]" if base_ok else "[SALAH]")

    # ---- B: pointer DI-OFFSET n -> harus membaca 0.99 ----
    ctx.enqueue_function[probe_read](
        d_q.unsafe_ptr().offset(n), d_dump.unsafe_ptr(), n,
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
    )
    ctx.synchronize()
    ctx.enqueue_copy(h_dump, d_dump)
    ctx.synchronize()
    var got_off = h_dump[0]
    var off_ok = abs(Float64(got_off) - 0.99) < 0.01
    print("B. pointer offset(", n, "): dibaca =", got_off, " harap 0.99 ->",
          "[BENAR]" if off_ok else "[SALAH]")

    # ---- C: kontrol — buffer TERPISAH (offset 0) harus benar ----
    var d_q2 = ctx.enqueue_create_buffer[DType.float16](n)
    var h_b = alloc[Float16](n)
    for i in range(n):
        h_b[i] = Float16(0.99)
    ctx.enqueue_copy(d_q2, h_b)
    ctx.synchronize()
    ctx.enqueue_function[probe_read](
        d_q2.unsafe_ptr(), d_dump.unsafe_ptr(), n,
        grid_dim=(1, 1, 1), block_dim=(1, 1, 1)
    )
    ctx.synchronize()
    ctx.enqueue_copy(h_dump, d_dump)
    ctx.synchronize()
    var got_sep = h_dump[0]
    var sep_ok = abs(Float64(got_sep) - 0.99) < 0.01
    print("C. buffer terpisah : dibaca =", got_sep, " harap 0.99 ->",
          "[BENAR]" if sep_ok else "[SALAH]")

    print("-----------------------------------------------------------------")
    if base_ok and off_ok and sep_ok:
        print(">> VERDICT: offset pointer device BERFUNGSI.")
        print(">> Produksi prefill AMAN dari masalah ini. Cari akar lain.")
    elif base_ok and not off_ok:
        print(">> VERDICT: OFFSET POINTER DEVICE TIDAK DIHORMATI.")
        print(">> prefill (M>1) memakai data sampah untuk t>=1.")
        print(">> layer.mojo baris 484-493 & 523-537 terdampak -> perbaiki.")
    else:
        print(">> VERDICT: tidak konklusif — bahkan pointer dasar/terpisah salah.")
        print(">> Ada masalah lain di harness/backend; selidiki terpisah.")
    print(">> (Hasil otoritatif = run di T4 via deploy_on_kaggle.sh.)")

    h_q.free(); h_b.free(); h_dump.free()

    var strict = getenv("BONSAI_OFFSET_STRICT")
    if strict and (strict == "1" or strict == "true") and not off_ok:
        print(">> BONSAI_OFFSET_STRICT=1 -> keluar dengan status GAGAL.")
        raise Error("offset pointer device salah (lihat verdict di atas)")
    print("=================================================================")
