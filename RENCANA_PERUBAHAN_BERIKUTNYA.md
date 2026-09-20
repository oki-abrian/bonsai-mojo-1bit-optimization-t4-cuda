# RENCANA PERUBAHAN BERIKUTNYA — tinjauan ulang, 2026-09-20 (revisi 2)

## 0. Status berkas ini (baca dulu)

- Ini RENCANA, bukan laporan pelaksanaan. **Tidak ada satu baris kode yang
  sudah diubah untuk membuat berkas ini.**
- Tidak ada asumsi bahwa pemilik proyek setuju atau tidak setuju atas
  perubahan mana pun. Setiap butir menunggu jawaban eksplisit.
- Semua angka bersumber dari pengukuran (run v36–v42, nvprof CSV, probe CUDA
  mandiri), kecuali yang secara tersurat ditandai **[INFERENSI]**.
- Satu perubahan saja yang sampai saat ini saya setujui (bagian 3). Sisanya
  ditolak dengan alasan terukur, atau memang bukan wewenang saya (bagian 4).
- **Revisi 2** (menyusul permintaan kedua untuk meninjau ulang diri):
  tinjauan pertama ternyata hanya meninjau sisi DECODE. Proyek ini punya dua
  kanal terukur — decode DAN prefill — jadi kandidat prefill (bagian 2I–2K)
  kini ditinjau juga. Hasilnya: satu celah cakupan ditemukan dan ditutup,
  tetapi **kesimpulan tidak berubah** — hanya §4c yang saya setujui.
- **Revisi 3** (20-09-2026 sore): §4c sudah dijalankan dan diverifikasi (bagian
  6–7). Terpisah dari rencana ini, satu kegagalan yang ditemukan di §7 item 1
  (`normgate_h2_dv16_ok`) telah diperbaiki atas instruksi "perbaiki" —
  rinciannya di bagian 8. Item 2 (`[COH-FAIL]`) diperiksa dan terbukti tidak
  ada hubungannya dengan perubahan mana pun. **Bagian 2–5 tetap berlaku
  utuh: belum ada satu pun kandidat di sana yang saya eksekusi.**

## 1. Dasar pengukuran yang dipegang

Anggaran decode jalur h2, run v42, dinding **47,363 ms/token** (jalur lama
67,488 ms/token; percepatan 1,425×; 24 token identik; VERDIK TOTAL: LULUS).

Dua koreksi WAJIB sebelum membaca tabel (tanpa ini angka melenceng jauh):

1. `qmm_sm75_b2_kernel` 24,62 ms/token itu **prefill** (288 = 48 layer × 6
   peluncuran, sekali saja), bukan decode — dibuang.
2. 12 peluncuran pemanasan LM head (`main.mojo:1533`, grid 7760, 1.832 µs)
   ikut terprofil tetapi tidak masuk penghitung waktu: 1.998 ms/token
   dikurangkan dari GEMV.

| Bagian | ms/token | porsi |
|---|---|---|
| GEMV `qmv_vec_q2t_h2_kernel` (354/token) | 32,77 | 69,2% |
| Rekurensi GDN (48/token) | 8,24 | 17,7% |
| FWHT (316/token) | 1,83 | 3,9% |
| `add_rmsnorm` (128/token) | 1,21 | 2,6% |
| `qmv_dense` (52/token) | 0,72 | 1,5% |
| `gdn_seq` (4,4/token) | 0,14 | 0,3% |
| `qmv_split_reduce` + 13 kernel kecil | 1,75 | 3,7% |
| **Jumlah** | **46,66** | (dinding 47,363; selisih 1,5% = inflasi profiler) |

Fakta pendukung yang sudah terukur tuntas:

- Puncak DRAM T4 yang diukur sendiri = **264 GB/s** (bukan 320 lembar data),
  datar terhadap okupansi (1–12 blok/SM) dan jumlah aliran.
- GEMV h2 (probe salinan setia kernel, gate_up 34816×5120): setia
  **220,6 GB/s** | tanpa baca skala 235,7 | tanpa PRMT+HFMA2 223,7 |
  x ditahapkan sekali 220,4 | baca kode murni 256,3.
  ⇒ dekode PRMT+HFMA2 = 1%, tahapan x per tile = 0%, skala = 7%.
- GEMV di dalam model = ~212 GB/s = **96% dari kemampuan kernelnya berdiri
  sendiri**, 80% dari puncak DRAM.
- Rekurensi GDN di dalam model: 171,7 µs/peluncuran; probe terisolasi untuk
  kode yang sama: 84 µs (2,0×). Tiga intervensi berbeda (trafik 12→9 MB,
  okupansi 8× via varian lebar, unroll) semuanya **nol atau negatif** di
  dalam model: 0,981× (v36), 0,976× (v40), 0,986× (v42).
- Celah antar-kernel: rata-rata **0,99 µs**, terbesar 10,3 µs, **nol** di
  atas 20 µs ⇒ total waktu menganggur ≤ ~1,5 ms/token (≤3%).

## 2. Tinjauan ulang yang diminta: kandidat demi kandidat

Saya meninjaunya ulang dari awal, bukan sekadar mengulang kesimpulan lama.

### A. `deploy_on_kaggle.sh` §4c — `--target-accelerator=sm_75` → **SETUJU**

Enam pemanggilan `pixi run mojo run -I .` (baris 614, 615, 618, 619, 627,
640) belum memakai flag yang sudah dipakai `mojo build` di baris 499.
Peringatan yang muncul sekarang: `constraint failed: the target architecture
'' is invalid` — Mojo mencoba menebak target akselerator, dan di instance
build yang tidak memasang GPU hasil tebakannya kosong.

Yang jujur harus dikatakan: nilai perubahan ini adalah **kebersihan dan
konsistensi, bukan kinerja, dan bukan perbaikan kebenaran**. Biner yang
dikirim dikirimkan oleh `mojo build` yang sudah benar; keenam tes tetap
berjalan hari ini. Yang diperoleh: (a) peringatan hilang, (b) kode GPU yang
diuji dan kode GPU yang dikirim ditempa dengan target yang sama persis,
sehingga teoretis tidak ada celah "tes lulus tetapi biner berbeda".

Risiko: rendah — lima dari enam pemanggilan memakai `|| echo WARN`, dan
baris 627 hanya menghentikan deploy bila `BONSAI_GDN_STATE_FATAL=1`.
**[INFERENSI]** bahwa `mojo run` menerima flag yang sama seperti
`mojo build` — belum pernah diverifikasi (Mojo hanya ada di Kaggle).
Mitigasi ada di bagian 5.

### B. Mengubah apa pun di dalam kernel GEMV h2 → **TIDAK SETUJU**

Probe H1–H4 menutupnya: beban komputasi dekode 1%, tahapan x 0%, skala 7%
(dari itu pun SMEM hanya mengembalikan ~2%). Kernel murni terhambat memori
di 84% puncak terukur, dan di dalam model sudah 96% dari kemampuannya
sendiri. Ini bukan "tidak ada ide", ini "tidak ada byte yang bisa direbut".

### C. Rekurensi GDN, termasuk varian lebar / identitas aljabar → **TIDAK SETUJU**

`out = decay·B + delta·kq` sudah TERPASANG sebagai
`gdn_recurrence_sm75_gpu_wide` (aktif hanya bila `BONSAI_GDN_WIDE=1`,
bawaan mati). Diukur ujung-ke-ujung tiga kali: 0,981×, 0,976×, 0,986× —
tiga kali tidak lebih cepat, padahal probe terisolasi menjanjikan 1,51×.
Perkiraan "11,3 ms → ~2 ms" pada laporan lama terbantahkan oleh A/B di
dalam model. Ditambah ia tidak bit-exact. Tidak ada alasan menghidupkannya.

### D. CUDA Graphs → **TIDAK SETUJU**

Dijawab langsung oleh ukuran celah: rata-rata 0,99 µs, terbesar 10–14 µs,
nol di atas 20 µs. Langit-langit ~3%, biaya implementasi besar, dan di
jalur ber-env-switch seperti ini malah berpotensi mengunci struktur
peluncuran.

### E. Fusi kernel-kernel kecil (FWHT + norm + elementwise, total 5,65 ms) → **TIDAK SETUJU sebagai langkah berikutnya**

Masing-masing 5–7 µs per peluncuran, alias biaya kirim + latensi. Potensi
hemat teoretis 2–3% (sejalan dengan langit-langit celah di atas), tetapi:
setiap perubahan Mojo berisiko gagal build (tidak ada fallback native untuk
jalur 2-bit), biaya satu siklus build+uji 17+ menit, dan tiga intervensi
kernel kecil sebelumnya semuanya null/negatif. Biaya-kesempatan buruk.
Kalau kelak dikejar, yang sah adalah A/B ujung-ke-ujung satu sesi, bukan
probe terisolasi (pelajaran terukur dari varian lebar).

### F. Menjadikan `BONSAI_DECODE_H2=1` bawaan → **BUKAN KEPUTUSAN SAYA**

Jalurnya sudah terukur 1,425× dan terverifikasi identik berulang kali, jadi
secara teknis layak. Tetapi desain yang disepakati proyek ini adalah
"env off-switch per fusi" — membalik bawaan adalah keputusan kebijakan
pemilik proyek, bukan keputusan teknis saya. Dicatat di sini supaya tidak
hilang, bukan diusulkan.

### G. Menghapus varian mati (`BONSAI_GDN_WIDE`, h2b) → **BUKAN KEPUTUSAN SAYA**

Keduanya bawaan-mati dan tidak menbiayai apa pun saat inferensi. Nilainya
sekarang justru sebagai bukti terdokumentasi atas hasil negatif. Menghapus
atau mempertahankan adalah pilihan pemilik proyek.

### H. Uji akurasi ratusan token untuk varian tak-bit-exact → **TIDAK PENTING SAAT INI**

Pertanyaan "mengubah akurasi?" sudah terjawab tiga tingkat (angka berubah
~4e-7 relatif; penyimpanan FP16 praktis tidak berubah ~99,92%; keputusan
token tidak terbalik, rasio ~1,6e8). Uji ratusan token hanya menjadi wajib
**bila** varian itu mau dipakai — dan ia tidak akan dipakai karena lebih
lambat (butir C). Jadi tidak perlu dibelanjakan sekarang.

### I. Mengejar prefill di atas 101,58 tok/s → **TIDAK DIUSULKAN** (kandidat sisi prefill)

Kanal prefill punya status terukur sendiri (CATATAN §12g/§12h): jalur fp16
WMMA 89,09 tok/s, jalur int8 W2A8 opt-in (`BONSAI_PREFILL_INT8=1`)
101,58 tok/s, keduanya LULUS verifikasi numerik 24/24 token. Target 100+
tok/s yang dulu terbuka **sudah tercapai**. Analisis FLOP menunjukkan ~98%
waktu prefill habis di kuantisasi/dekuantisasi pada CUDA core, bukan di MMA,
jadi sisa ruangnya ada di sana — tetapi: prefill adalah biaya sekali per
prompt (129 token ≈ 1,27 s), bukan biaya per token; tidak ada permintaan
yang berdiri untuk melampaui 101,58; dan tiap perubahan menuntut siklus
build 17+ menit plus verifikasi §12g diulang. Biaya-kesempatan buruk.

### J. `BN` 64 → 128 pada kernel qmm → **TIDAK DIUSULKAN sebagai langkah berikut**

Ini satu-satunya pengungkit dari audit referensi (§12f.2: "mandiri dari
int8, murah") yang belum pernah dicoba. Dicatat di sini supaya tidak
hilang. Alasan tidak mengusulkannya: ia hanya menyentuh prefill, yang
targetnya sudah tercapai (butir I), dan analisis FLOP menunjukkan prefill
terhambat kuantisasi/dekuantisasi (98%), bukan tiling MMA — jadi dampaknya
diperkirakan kecil **[INFERENSI]**. Kalau kelak prefill mau didorong lebih
jauh, pengungkit yang benar adalah kuantisasi/dekuantisasinya (butir I),
bukan ubinnya, dan ukuran yang sah tetap A/B ujung-ke-ujung satu sesi.

### K. Mempercepat pemuatan bobot → **TIDAK DIUSULKAN**

Terukur di log run v42: 1.376 salinan `[CUDA memcpy HtoD]` total **1,45 s**
saat model dimuat. Bila hampir semuanya adalah bobot 8,6 GB, lajunya
~5,9 GB/s — khas memori pageable tanpa staging pinned **[INFERENSI]**.
Ini biaya sekali saat startup, tidak menyentuh satu token pun, dan tidak
ada permintaan yang berdiri. Angka dicatat supaya tidak hilang.

## 3. Perubahan yang saya setujui, rinciannya

**Berkas**: `deploy_on_kaggle.sh` (berkas terbukti — menunggu izin eksplisit).
**Bentuk**: menyisipkan `--target-accelerator=sm_75` setelah `mojo run`,
menjadi `pixi run mojo run --target-accelerator=sm_75 -I . tests/….mojo`.
**Baris**: 614, 615, 618, 619, 627, 640 — hanya keenam ini, tidak ada
baris lain yang disentuh (baris 457 adalah smoketest jalur lain; baris 648
loop self-test yang mati secara bawaan; baris 1308 kalibrasi khq).

| baris | isi sekarang |
|---|---|
| 614 | `pixi run mojo run -I . tests/selftest_decode_gpu.mojo \|\| echo …` |
| 615 | `pixi run mojo run -I . tests/test_qmm_gpu.mojo \|\| echo …` |
| 618 | `pixi run mojo run -I . tests/test_rope_gpu.mojo \|\| echo …` |
| 619 | `pixi run mojo run -I . tests/test_argmax_gpu.mojo \|\| echo …` |
| 627 | `if ! pixi run mojo run -I . tests/test_gdn_state_precision.mojo; then` |
| 640 | `pixi run mojo run -I . tests/test_enqueue_offset.mojo \|\| echo …` |

**Efek yang diharapkan**: peringatan `constraint failed: the target
architecture '' is invalid` hilang dari log build; keenam tes mengompilasi
dengan target sm_75 eksplisit, sama seperti biner yang dikirim.
**Efek yang TIDAK diharapkan**: tidak ada perubahan kinerja, tidak ada
perubahan pada biner `bonsai_infer`.

## 4. Di mana penalaran saya bisa salah (tinjauan-diri, terbuka)

1. **`mojo run` menerima flag itu** — [INFERENSI], belum diverifikasi karena
   Mojo hanya ada di Kaggle. Kalau flag ditolak, keenam tes akan mencetak
   peringatan (baris 627 hanya fatal bila `BONSAI_GDN_STATE_FATAL=1`),
   dan urutan tindakannya adalah MEMBUKAI enam baris itu kembali.
2. **Peringatan itu memang hanya kosmetik** — dasarnya log: build tetap
   selesai, keenam tes tetap menghasilkan verdict. Saya tidak pernah
   membandingkan isi PTX tes-tanpa-flag vs biner-ber-flag; kalau ingin
   benar-benar yakin, bandingkan di Kaggle. [INFERENSI]
3. **Probe rekurensi mungkin menyanjungi diri** — [HIPOTESIS BELUM TERUJI]:
   probe menjalankan 48 peluncuran berurutan tanpa GEMV di antaranya, jadi
   keadaan L2-nya berbeda dari model. Ini TIDAK mengubah keputusan apa pun,
   karena keputusan C bertumpu pada A/B di dalam model, bukan pada probe.
   Dicatat supaya tidak ada yang kelak memakai angka probe itu sebagai janji.
4. **nvprof menggelembungkan kernel kecil** — dicatat tersurat oleh log
   sendiri. Karena itu pada anggaran hanya JUMLAH TOTAL yang dipakai untuk
   menyimpulkan, bukan nilai mutlak per kernel.
5. **Drift Kaggle 5–7% antar sesi** — semua putusan di atas tidak bergantung
   pada selisih <8%; kandidat yang saya tolak ditolak dengan selisih terukur
   atau dengan argumen byte, bukan dengan selisih tipis.
6. **Celah cakupan tinjauan pertama** — revisi pertama berkas ini hanya
   meninjau sisi decode; kandidat sisi prefill (I–K) baru ditambahkan di
   revisi kedua setelah permintaan tinjauan ulang. Ini kesalahan nyata pada
   tinjauan pertama saya, dan sengaja ditulis apa adanya di sini sebagai
   bagian dari review-diri.

## 5. Urutan eksekusi KALAU disetujui (belum dijalankan)

1. Edit keenam baris di `deploy_on_kaggle.sh` (rincian di bagian 3).
2. Push build CPU ke Kaggle seperti biasa (`kaggle_cpu_build/`).
3. Cek log: peringatan `the target architecture '' is invalid` hilang;
   keenam tes tetap mencetak verdict yang sama seperti sebelumnya.
4. Kalau `mojo run` menolak flag (tes mencetak error flag): buka kembali
   keenam baris (kembalikan seperti tabel di bagian 3), laporkan, selesai.
5. Tidak menyentuh berkas lain sama sekali. `run_infer_2bit.py` perlu
   dijalankan ulang bila ingin memastikan jalur 2-bit tidak terpengaruh —
   itu langkah terpisah, hanya kalau diminta.

## 6. Hasil verifikasi (build CPU, kernel v24, 20-09-2026 17:00)

Enam baris sudah dijalankan di Kaggle. Yang terbukti:

1. **`mojo run` MENERIMA flag itu** — keenam tes berhasil dikompilasi, tanpa
   satu pun "unrecognized option" dan tanpa `constraint failed: the target
   architecture '' is invalid`. Ini menuntaskan [INFERENSI] nomor 1 di
   bagian 4 secara positif.
2. **Keenam tes kini gagal pada saat JALAN, bukan saat kompilasi**:
   `Unhandled exception … Failed to open library "libcuda.so.1"`. Itu
   keadaan bawaan instance CPU-only (tidak ada driver CUDA), jadi keenam
   tes GPU memang tidak bisa lulus di kernel build CPU. Sebelum perubahan,
   bukti tak langsungnya kuat bahwa mereka gagal pada saat KOMPILASI:
   smoketest di baris 457 — yang polanya sama dan sengaja tidak diubah —
   masih gagal dengan `error: failed to run the pass manager` dan
   constraint yang sama. Jadi perubahan ini memindahkan keenam tes dari
   "tidak bisa dikompilasi" menjadi "dikompilasi, menunggu GPU".
3. **Peringatan itu belum hilang seluruhnya**: masih muncul SATU kali, dari
   smoketest baris 457 (`test_cuda_ffi_smoketest.mojo`), yang berada di
   luar cakupan yang diizinkan dan sengaja tidak saya sentuh.
4. **Build sendiri tidak terpengaruh**: `main.mojo` terkompilasi,
   `bonsai_infer` 949.792 B masuk wheel, `libbonsai_qmv_sm75.so` 2.398.784 B,
   berkas-berkasnya terunduh, keluar dengan kode 0.

Yang BELUM terbukti dan tidak bisa dibuktikan di kernel CPU: keenam tes
mencetak verdict LULUS. Satu-satunya tempat hal itu mungkin adalah kernel
GPU (`okiabrian/bonsai-mojo-t4-build` lewat `./push_to_kaggle.sh`), karena
di sanalah `libcuda.so.1` ada. Itu memerlukan kuota GPU, jadi menunggu
izin tersendiri.

## 7. Hasil verifikasi kernel GPU (20-09-2026, 39 menit)

Izin diberikan, termasuk menambahkan flag yang sama ke baris 457 sehingga
total pemanggilan `mojo run` ber-flag menjadi 7. Log:
`dist_kaggle_mojo/gpu_build_log.txt` (700 KB).

**Tujuan tercapai:**

- Peringatan `constraint failed: the target architecture '' is invalid`
  **tidak muncul satu kali pun** di seluruh log GPU.
- Smoketest baris 457, yang dulu gagal KOMPILASI, kini lulus:
  `[PASS] 16 baris 0xFF bernilai +64.0, 16 baris 0x00 bernilai -64.0`.
- `selftest_decode_gpu`: 10/10 kasus PASS (termasuk `n34816_k5120_gateup`).
- `test_qmm_gpu`: "SEMUA KASUS QMM PREFILL PASS!" (10/10).
- `test_rope_gpu` 7/7 PASS, `test_argmax_gpu` 7/7 PASS.
- `test_gdn_state_precision`: 5/5 kasus presisi state PASS (dan kontrol
  FP16-nya menunjukkan 840–5511× lebih buruk, jadi tes punya daya-beda).
- `test_enqueue_offset`: tanpa baris GAGAL.

**Dua kegagalan yang ditemukan, keduanya BUKAN akibat flag ini:**

1. `[GAGAL] normgate_h2_dv16_ok` — `H_v=2 D_v=16`, rel = 0,593 (tol 0,002).
   Bentuk produksi `normgate_h4_dv128_ok` (D_v=128) justru **PASS**, dan
   `normgate_h4_dv128_legacy` gagal sebagaimana yang memang diharapkan
   (itulah daya-beda tesnya). Jadi ada satu bentuk non-produksi — D_v=16 —
   tempat kernel `gdn_norm_gate_sm75_gpu` menyimpang dari referensi FP64.
   Alasan kuat bahwa ini bukan akibat perubahan saya: di instance GPU,
   target sudah terselesaikan ke sm_75 lewat deteksi otomatis, jadi flag itu
   tidak mengubah kode yang dihasilkan **[INFERENSI]** — bukti langsungnya
   belum ada, karena tidak ada log GPU sebelum perubahan untuk dibandingkan.
2. Gerbang koherensi `[COH-FAIL]`: `cerita.think0` mengulang dengan periode
   48 selama 97 token mulai token 176 (rc=1). Ini mutu generasi, sama
   sekali tidak bersinggungan dengan target kompilasi.

Catatan sampingan yang ikut terbaca: decode jalur 1-bit di GPU 56,61 ms/token
(17,67 tok/s), build + uji §5 selesai dengan keluar kode 0..

## 8. Tindak lanjut: normgate_h2_dv16_ok diperbaiki (20-09-2026, "perbaiki")

Item 1 bagian 7 di atas sudah diselesaikan. Bukan akibat flag §4c — memang cacat
kernel yang sudah ada sebelumnya.

**Akar masalah** (diverifikasi dari peluncur + kernel + geometri, bukan asumsi):
`gdn_norm_gate_sm75_launch_on` selalu memakai `block_dim=(D_v,1,1)`. Pada D_v=16
blok hanya 16 thread = satu warp parsial, sehingga dua hal yang diasumsikan
kernel menjadi tidak benar:

1. `sq += shuffle_down(sq, 16)` membaca lane 16–31 yang tidak pernah dieksekusi
   (nilai tak terdefinisi).
2. Tingkat antar-warp membaca `smem[1..3]` yang tidak pernah ditulis (hanya
   `wid==0` yang ada) — shared memory yang tidak diinisialisasi.

Produksi selalu `D_v=128` (tepat 4 warp penuh), tempat kedua asumsi itu benar;
itulah sebabnya `normgate_h4_dv128_ok` selalu lulus dan cacat ini tidak terlihat.

**Perbaikan** (`src/kernels/elementwise_sm75.mojo`, satu blok): cabang runtime
seragam `if D_v >= 32`. Cabang `>= 32` adalah kode lama persis tanpa perubahan;
cabang `< 32` mereduksi sepenuhnya lewat shared memory. Kontrak `smem[0]`
tetap, jalur produksi byte-identik.

**Verifikasi ulang** (CPU build 20 m 49 s + kernel GPU 39 m 40 s, keduanya
keluar kode 0; log `dist_kaggle_mojo/gpu_build_log_fixdv16.txt`):

- `[PASS] normgate_h2_dv16_ok` rel **0,000169** (sebelumnya 0,593).
- `[PASS] normgate_h4_dv128_ok` rel **0,00026833234**, abs 0,0038836887,
  ref_maks 14,473428 — **sama persis sampai digit terakhir** dengan run
  pra-perbaikan ⇒ jalur produksi terbukti byte-identik.
- `normgate_h4_dv128_legacy` **masih GAGAL** (rel 0,691) ⇒ daya-beda tes untuk
  regresi BUG-1 tetap utuh.
- Semua tes lain masih lulus: smoketest, decode 10/10, qmm 10/10, rope 7/7,
  argmax 10/10, presisi state 5/5, enqueue offset.

**Item 2 bagian 7 (`[COH-FAIL]`) diperiksa, BUKAN akibat perubahan apa pun.**
Dua log GPU dibandingkan langsung (sebelum dan sesudah perbaikan D_v=16):

| | sebelum | sesudah |
|---|---|---|
| `cerita.think0` (gen/unik/rasio/loop) | 274/132/0,48/p48@176 | 274/132/0,48/p48@176 |
| EOS cerita | posisi 273 | posisi 273 |
| `penalaran.think0` | 512/136/0,27/tanpa loop | 512/136/0,27/tanpa loop |
| sebaran unik per jendela 50 | 0,86 0,86 0,86 0,76 0,72 | 0,86 0,86 0,86 0,76 0,72 |

Semua angka identik sampai digit terakhir — keluaran generasi byte-identik, jadi
perbaikan terbukti tidak mengubah produksi sama sekali. Gerbang ini gagal jauh
sebelumnya dan sudah didokumentasikan sejak kernel v119 (12-09-2026,
`LAPORAN_HASIL_GERBANG_KOHERENSI.md`). Dibanding v119, kondisinya jauh membaik
(cerita rasio unik 0,16 → 0,48; penalaran cycle@48 → tanpa loop; EOS kini
muncul). Hipotesis yang sudah dicatat di v119 bagian 6: degenerasi adalah loop
greedy di dalam mode thinking, bukan akumulator galat numerik.

