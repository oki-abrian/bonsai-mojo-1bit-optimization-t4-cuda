# MEMORY.md — bonsai-1bit-t4-mojo (catatan jangka panjang)

## Aturan kerja proyek

- Proyek meniru llama.cpp (`references/llama.cpp-prism/` = referensi baca-saja).
  Metodologi: `LAPORAN_AUDIT_PARITAS_LLAMACPP_PRISM.md`.
- **Mesin lokal BUKAN mesin build.** Build & toolchain di Kaggle/T4 (sm_75).
  Lokal tak bisa kompiler Mojo. Verifikasi: CPU build (compile gate ~20 mnt)
  lalu GPU build (~40 mnt).
- **Diff harus aditif**: parameter baru punya default yang mereproduksi perilaku
  lama byte-identik; jangan mengubah isi fungsi yang ada.
- Berkas terbukti, **tidak disentuh tanpa izin eksplisit**: `deploy_on_kaggle.sh`,
  `run_deploy.py`, `push_to_kaggle.sh`, `kaggle_cpu_build/push_cpu_build.sh`.
  (MENJALANKAN-nya boleh, mengeditnya tidak.)

## Kontrak Bonsai-2 (2-bit ternary, pack MLX prism_hadamard_qwen35)

- Dequant `w = (q-1)*s`, q ∈ {0,1,2}, **q=0 = −1 kode sah**. Tidak ada /2
  (1-bit: `(2q-1)*(s_ckpt/2)`).
- Bit: U32 LE, 16 bobot/word, lane `i` di bit `2i`. Scales/bias F16 [N,K/128].
- Signs dikunci lebar masuk K: `sign_widths=[5120,6144,17408]`, 28.672 nilai ±1.
  Tensor `.signs` ADA tapi TIDAK dipakai runtime referensi.
- FWHT butterfly `rendah=x+y, tinggi=x-y` = Sylvester `H_B/sqrt(B)`.
  `embed_tokens.weight` satu-satunya `inverse_weight_names`.
- `gdn_v_grouped: True` — aktivasi GDN sudah terkelompok, JANGAN permutasi.
- `BONSAI_BITS=2` (default 1). Geometri GDN: `config.mojo:62`
  `gdn_head_v_dim=128`, H_v=48, D_k=128.

## Trap bahasa Mojo 25.x (semua pernah bikin build Kaggle gagal)

- **Tidak ada interpolasi string `\(`** (itu Swift): `+ String(x)`.
- **Tidak ada kembalian tuple**: nilai kedua lewat `UnsafePointer[Int]`.
- **`out` kata kunci cadangan** — parameter pakai `dst`.
- **Konstruktor posisional**: `HadamardSigns(block=1,...)` error; tulis
  `HadamardSigns(1, buf, widths, 0, 0)`.
- **Tidak ada konstruktor anggota otomatis** — tulis `__init__` eksplisit.
- **Jangan uji pointer `if ptr:`** — pakai
  `if p != UnsafePointer[Float32, MutAnyOrigin]():`.
- `sqrt` impor dari `math`, dipanggil `sqrt(Float32(x))`.
- **`shuffle_down` butuh delta konstan waktu-kompilasi**, jadi ladder tak bisa
  menyesuaikan jumlah warp runtime; untuk geometri runtime tak terduga pakai
  reduksi murni di shared memory.
- **Jangan namai parameter `len`** (menutupi `len()` bawaan);
  **`ord(...)` selalu dibungkus** `UInt8(ord(","))`.

## Trap CUDA

- `QmvTraits<T>` hidup di `namespace bonsai::sm75` (qmv_sm75_kernel.cu:42-65);
  kernel baru HARUS di namespace turunan (`bonsai::sm75::q2t`).
- `fuse_u8`/`fuse_f32` **mengalokasi & menyalin** (bukan view): setelah fusi,
  free hanya di cabang yang masih pegang ownership asli.
- `DeviceBuffer` RAII: jangan deklarasi di dalam `if` kalau raw pointer dipakai
  di scope luar — hoist. Output GEMV dense terstrided `N_total`, bukan `n_tail`.
- **Kelas defek "reduksi warp bergantung geometri" (diperbaiki 2026-09-20):**
  `sq += shuffle_down(sq,16/8/4/2/1)` + `smem[lane] if lane<NWARP else 0` hanya
  absah bila SETIAP warp penuh DAN jumlah warp tepat = NWARP; kasus lain
  menghasilkan nilai salah TANPA crash. Peluruhan: `gdn_norm_gate` kini
  `if D_v == 128` (kode lama byte-identik) + cabang SMEM murni + guard
  `D_v > 128 → Error` (ops.mojo). Kernel block_size TETAP (rmsnorm 256,
  head_rmsnorm 32, gqa 256, argmax 256, gdn_recurrence) AMAN karena
  geometrinya tak bergantung dimensi runtime.

## Hasil ukur T4 (fondasi keputusan optimasi)

- **Puncak DRAM terukur 264 GB/s**, datar vs okupansi & aliran.
- **Decode 2-bit h2 (v40, 48,97 ms/token)**: GEMV 32,50 (66%) | rekurensi GDN
  8,75 (18%) | FWHT 1,89 | add_rmsnorm 1,26 | qmv_dense 0,73 | kecil sisanya.
- **GEMV 2-bit h2 sudah mentok**: ~214 GB/s di model = 96% kemampuan kernelnya
  (probe: baca kode murni 256 GB/s; bacaan skala = 7% sisanya).
- **PENTING — "GEMV habis" HANYA berlaku untuk 2-bit.** Kernel produksi 1-bit
  (`qmv_vec_nib_q1o_kernel`, qmv_sm75_kernel.cu:78) memakai teknik lain:
  stages x sebagai float + LUT nibble di SMEM + 32 *indexed SMEM gather* per
  128 bobot; 2-bit h2 memakai register expand + `__hfma2`. 1-bit membaca 0,55×
  byte 2-bit tetapi 1,196× lebih lambat ⇒ ~2,2× efisiensi lebih rendah
  (~67 vs ~147 GB/s, inferensi silang-run BUKAN ukuran langsung). Estimasi
  headroom ~2,35×; bila ditutup, decode 1-bit 56,61 → ~32,5 ms/token.
  **Belum pernah diuji langsung** (butuh port teknik h2 ke 1-bit + A/B).
- **Presisi aritmetika BEDA dari presisi penyimpanan.** Bobot disimpan 2-bit
  tetapi DIHITUNG dalam FP16. Bukti (dibaca dari kode): kernel 2-bit SKALAR
  men-stage x sebagai `float x_s[]` (qmv_sm75_kernel.cu:3400) => jalur FP32,
  tanpa half2 sama sekali; kernel h2 men-stage x sebagai `__half x_s[]`
  (:3587), bobot diperluas ke register half2 (`q2t_expand_word_h2`:3542),
  lalu `__hfma2` = 2 hasil kali per instruksi, akumulasi dibuang ke FP32 tiap
  8 bobot. Kernel 1-bit juga `float x_s[]` (:100) + LUT nibble float.
  Jadi h2 menang karena PINDAH KE FP16 BERKEMASAN, bukan sekadar
  "lebih sedikit instruksi". IMPLIKASI: format 4-bit (NVFP4, E2M1) mengubah
  PENYIMPANAN (lebih banyak byte dibaca) sementara aritmetika tetap FP16 —
  tidak ada percepatan 8x. Lebar kemasan register hanya menolong SELAMA
  kernel terhambat instruksi; h2 sudah di 96% dinding bandwidth.
- **Rekurensi GDN tidak bereaksi terhadap apa pun di dalam kernelnya** — jangan
  setel lagi. Memperbanyak byte/instruksi melayang TIDAK pernah menolong
  (h2b 3,4% lebih lambat; unroll 4x rekurensi 4,6% lebih lambat).
- **CUDA Graphs tidak ada gunanya**: celah antar-kernel rata-rata 0,99 us.
  Unroll obat setempat: −56% di `add_rmsnorm`, nol di rekurensi GDN.
- **PROBE TERISOLASI MENIPU**: rekurensi 1,51x lebih cepat terisolasi, 0,981x
  di dalam model. Probe yang cuma membaca mengukur pola akses, BUKAN laju.
  Ukuran sah = A/B ujung-ke-ujung di model.
- **Drift mesin Kaggle 5–7% antar sesi**; perubahan <8% tak bisa dinilai
  lintas sesi. Yang sah: A/B satu sesi, atau selisih >25% per kernel nvprof.
- **JANGAN UKUR DENGAN RUN PENDEK.** Run 7/13/24 token meremehkan biaya 4–18%
  karena state GDN & konteks tumbuh monoton. Bukti 1-bit: 13 token = 58,53
  ms/token, 1739 token = 60,66. **Steady-state 2-bit h2 belum pernah diukur.**

## Status pengujian (2026-09-21)

- **Jalur 2-bit BELUM PERNAH diuji akurasinya.** Semua run 2-bit adalah smoke
  test (3/12/16/24 token, prompt pengisi). Yang terverifikasi hanya struktur:
  `bits= 2`, signs 28.672 nilai, 64 layer, exit 0. Angka "512/512 token
  identik" adalah uji kompresi KV KHQ, BUKAN akurasi kuantisasi 2-bit.
- **`BONSAI_DECODE_H2` OPT-IN** (linear.mojo:409) dan TIDAK ADA di
  `deploy_on_kaggle.sh`. Saat 2-bit masuk produksi, jika tak diset eksplisit
  akan diam-diam pakai kernel skalar (72,7 vs 47,4 ms/token = 1,53× lebih
  lambat, tanpa pesan error).

## Cara membaca profil nvprof

- Hardware counter TIDAK tersedia (cc 7.5): `ncu` exit 1. Hanya
  `--print-gpu-trace --csv`; `--kernels regex:` ditolak, nama polos OK.
- CSV: `Start` kolom 0, `Duration` kolom 1 (ms), nama kernel **kolom 18**
  (baris `[CUDA memcpy HtoD]` namanya bergeser ke kolom 19). Baris setelah
  header adalah baris SATUAN, bukan data. nvprof menambah overhead per launch
  jadi kernel kecil terukur jauh lebih lambat — nilai TOTAL yang dipakai.
- Untuk celah antar-kernel: potong semua kejadian sampai `memcpy HtoD`
  terakhir (fase muat bobot mendominasi).

## Trap Kaggle

- Isi `/kaggle/working` ikut ter-commit jadi OUTPUT → bersihkan di `finally`;
  ambil log lewat `api.kernels_logs()`, bukan `kaggle kernels output`.
- Binary CLI `kaggle` bisa menggantung — pakai `KaggleApi()` dari Python.
- **Tidak ada cara membatalkan kernel yang sedang jalan.** Endpoint
  cancel-session 404; `KaggleApi` tak punya metode cancel. Menghentikan skrip
  lokal TIDAK menghentikan kernel Kaggle (tetap makan kuota sampai COMPLETE).
- `kernels_status()` mengembalikan enum: bandingkan `str(x).rsplit('.',1)[-1]`.
  `COMPLETE` bukan bukti sukses; baca log.
- Sebelum tiap API: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  all_proxy ALL_PROXY` + `export PATH=".../miniconda3/bin:$PATH"`.
- **Kernel build CPU TIDAK bisa meluluskan tes GPU**: `mojo run tests/*_gpu.mojo`
  gagal `libcuda.so.1` — BUKAN bug kode, CPU build hanya compile gate.
  `mojo run` menerima `--target-accelerator=sm_75` (tanpa ini gagal kompilasi:
  `the target architecture '' is invalid`).
- Biner `bonsai_infer` di luar env build butuh `libKGENCompilerRTShared.so`
  dsb — sumbernya hanya env pixi; paket pip `max`/`modular` tidak memuatnya.
  glob Python tak menembus direktori berawal titik — pakai `find`.
