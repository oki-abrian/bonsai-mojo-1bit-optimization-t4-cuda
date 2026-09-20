# MEMORY.md — bonsai-1bit-t4-mojo (catatan jangka panjang)

## Aturan kerja proyek

- Proyek meniru llama.cpp (`references/llama.cpp-prism/` = referensi baca-saja).
  Metodologi: `LAPORAN_AUDIT_PARITAS_LLAMACPP_PRISM.md`.
- **Mesin lokal BUKAN mesin build.** Build & toolchain besar di Kaggle/T4.
  Mojo/pixi hanya ada di Kaggle (sm_75). Verifikasi: CPU build (compile gate,
  ~20 mnt) lalu GPU build (~40 mnt). Lokal tak bisa kompiler Mojo.
- **Diff harus aditif**: parameter baru punya default yang mereproduksi perilaku
  lama byte-identik; jangan mengubah isi fungsi yang ada.
- Berkas terbukti, **tidak disentuh tanpa izin eksplisit**: `deploy_on_kaggle.sh`,
  `run_deploy.py`, `push_to_kaggle.sh`, `kaggle_cpu_build/push_cpu_build.sh`.
  (MENJALANKAN-nya boleh, mengeditnya tidak.)

## Kontrak Bonsai-2 (2-bit ternary, pack MLX prism_hadamard_qwen35)

- Dequant `w = (q-1)*s`, q ∈ {0,1,2}, **q=0 = −1 adalah kode sah**.
  **Tidak ada /2** (1-bit memakai `(2q-1)*(s_ckpt/2)`).
- Bit: U32 LE, 16 bobot/word, lane `i` di bit `2i`. Scales/bias F16 [N,K/128]→FP32.
- **Signs dikunci lebar masuk K**, bukan per-modul: `sign_widths=[5120,6144,17408]`,
  28.672 nilai ±1 di hadamard.json. Tensor `.signs` ADA tapi TIDAK dipakai runtime
  referensi. Akibat: fusi qkv+z dan gate+up eksak; `in_proj_a/b` dense F32
  [48,5120] tak terfold → ekor dense.
- FWHT butterfly `rendah=x+y, tinggi=x-y` = Sylvester `H_B/sqrt(B)`.
  Forward `H@(s*x)/sqrt(B)`, inverse `s*(H@x)/sqrt(B)`.
  `embed_tokens.weight` satu-satunya `inverse_weight_names`.
- `gdn_v_grouped: True` — aktivasi GDN sudah terkelompok, JANGAN permutasi lagi.
  Rantai sumber: qwen.py:545-562 → base.py:752 → llama-model.cpp:2080 →
  llama-graph.cpp:1521/1581 → kita `hk = hv // 3`.
- Aktifkan `BONSAI_BITS=2` (default 1). Rincian: `CATATAN_IMPLEMENTASI_BONSAI2.md`.
- Geometri produksi GDN: `config.mojo:62` `gdn_head_v_dim=128`, H_v=48, D_k=128.

## Trap bahasa Mojo 25.x (semuanya pernah bikin build Kaggle gagal)

- **TIDAK ada interpolasi string `\(`** (itu Swift). Pakai `+ String(x)`.
- **TIDAK ada kembalian tuple**. Nilai kedua lewat `UnsafePointer[Int]`.
- **`out` kata kunci cadangan** — nama parameter pakai `dst`.
- **Konstruktor posisional**: `HadamardSigns(block=1,...)` error; tulis
  `HadamardSigns(1, buf, widths, 0, 0)`.
- **Mojo 25.x TIDAK membuatkan konstruktor anggota otomatis** — selalu tulis
  `__init__` eksplisit.
- **Jangan uji pointer dengan `if ptr:`** — pakai
  `if p != UnsafePointer[Float32, MutAnyOrigin]():`.
- `sqrt`: `from math import sqrt`, dipanggil `sqrt(Float32(x))`.
- **`shuffle_down` butuh delta KONSTAN WAKTU-KOMPILASI** (param `@parameter`),
  jadi ladder tak bisa menyesuaikan jumlah warp runtime. Untuk geometri runtime
  tak-terduga, gunakan reduksi murni di shared memory.
- **Jangan menamai parameter `len`** — menutupi `len()` bawaan.
- **`ord(...)` selalu dibungkus**: `buf[i] == UInt8(ord(","))`.

## Trap CUDA / bug yang sudah dikenal

- `QmvTraits<T>` hidup di `namespace bonsai::sm75` (qmv_sm75_kernel.cu:42-65).
  Kernel baru HARUS di namespace turunan (`bonsai::sm75::q2t`), kalau tidak →
  "identifier QmvTraits is undefined".
- `fuse_u8`/`fuse_f32` **mengalokasi & menyalin** (bukan view): setelah fusi,
  free hanya di cabang yang masih pegang ownership asli.
- `DeviceBuffer` RAII: jangan deklarasi di dalam `if` kalau raw pointer-nya
  dipakai di scope luar — hoist.
- Output GEMV dense terstrided oleh `N_total`, bukan `n_tail`.
- `arr_len`/`child_at` pada JSON besar (28K elemen) O(n²) — pakai pemindai linear.
- **`<current_time>` di context BISA STALE** — cek `date` sebelum menamai log harian.
- **`grep`/`rg` lewat Bash mengembalikan KOSONG meski pola pasti ada** — pakai
  Grep tool bawaan.
- **Kelas defek "reduksi warp bergantung geometri" (diperbaiki 2026-09-20):**
  pola `sq += shuffle_down(sq,16/8/4/2/1)` lalu `smem[lane] if lane<NWARP else 0`
  hanya benar bila (a) SETIAP warp penuh (D kelipatan 32) DAN (b) jumlah warp
  tepat = NWARP. Jika D<32: shuffle baca lane mati. Jika D kelipatan 32 tapi
  warp<NWARP: slot smem dibaca tak pernah ditulis. Jika D bukan kelipatan 32
  (mis. 48): warp terakhir parsial, ladder shuffle-nya sendiri salah. Semua
  menghasilkan nilai salah TANPA crash.
  - `gdn_norm_gate_sm75_gpu` (block=(D_v,1,1)): SEPENUHNYA diperbaiki.
    Cabang `if D_v == 128` (satu-satunya geometri produksi) = kode lama
    byte-identik; cabang lain = reduksi murni di SMEM 128 float tanpa asumsi
    geometri. Peluncur ops.mojo menolak D_v>128 → Error. SMEM sengaja
    diperbesar 32→128 float (grid cuma 48 blok, tak ada efek okupansi).
  - Kernel dengan block_size TETAP (rmsnorm/add_rmsnorm 256=8 warp,
    head_rmsnorm 32 serial grid-stride, gqa 256=8 warp, argmax 256) AMAN:
    geometrinya tak bergantung dimensi runtime. `gqa` sudah punya guard
    `head_dim != 256 → Error` (pola yang dicontoh perbaikan norm-gate).
  - `gdn_recurrence_sm75_gpu` (block=(D_v,1,1)) AMAN: tak ada reduksi
    antar-thread sama sekali — tiap thread dv mengurus baris dv sendiri,
    satu-satunya sinkronisasi adalah `barrier()` yang dicapai semua thread.

## Hasil ukur T4 (fondasi semua keputusan optimasi)

- **Puncak DRAM terukur = 264 GB/s**, datar vs okupansi (1–12 blok/SM) & aliran.
- **Decode jalur h2 (v40, 48,97 ms/token)**: GEMV 32,50 (66%) | rekurensi GDN
  8,75 (18%) | FWHT 1,89 | add_rmsnorm 1,26 | qmv_dense 0,73 | gdn_seq 0,13 |
  ~13 kernel kecil 1,71.
- **GEMV h2 tuntas diukur** (v39, gate_up 34816×5120): setia 220,6 GB/s |
  tanpa skala 235,7 | tanpa PRMT+HFMA2 223,7 | x distage sekali 220,4 | baca
  kode murni 256,3 ⇒ dekode PRMT+HFMA2 = 1%, staging x = 0%, skala = 7%.
  Di model ~214 GB/s = 96% kemampuan kernelnya. **Celah GEMV habis.**
- **Rekurensi GDN tidak bereaksi terhadap apa pun di dalam kernelnya**
  (trafik 9→6 MB nol, okupansi nol, unroll negatif). Jangan setel lagi.
- **CUDA Graphs tidak ada gunanya (v42)**: celah antar-kernel rata-rata
  0,99 us, terbesar 10–14 us ⇒ total ≤1,5 ms/token (3%).
- **Yang menolong: kurangi trafik & instruksi.** PRMT+HFMA2 (6→~1,25
  instr/bobot) = 1,37x. Buang tulisan `S*decay` (12→9 MB) = −28%.
  **Memperbanyak byte/instruksi melayang TIDAK pernah menolong**
  (h2b 3,4% lebih lambat; unroll 4x rekurensi 4,6% lebih lambat).
- **Unroll obat setempat**: −56% di `add_rmsnorm`, nol di rekurensi GDN.
- **PROBE TERISOLASI MENIPU**: rekurensi lebar 1,51x lebih cepat terisolasi,
  0,981x di dalam model. Kernel kecil yang terjepit GEMV: satu-satunya ukuran
  sah = A/B ujung-ke-ujung di model.
- **PROBE YANG CUMA MEMBACA mengukur pola akses, BUKAN laju kernel**.
- **Drift mesin Kaggle 5–7% antar sesi**; perubahan <8% tak bisa dinilai
  lintas sesi. Yang sah: A/B satu sesi, atau selisih >25% per kernel nvprof.
- Probe CUDA mandiri MURAH: `.cu` di `run_infer_2bit.py`, `nvcc -O3 -arch=sm_75`.

## Cara membaca profil nvprof

- Hardware counter TIDAK tersedia (cc 7.5): `ncu` exit 1, `--metrics` ditolak.
  Hanya `--print-gpu-trace --csv`; `--kernels regex:` ditolak, nama polos OK.
- CSV: `Start` kolom 0 (detik), `Duration` kolom 1 (ms), nama kernel **kolom 18**
  — tapi baris `[CUDA memcpy HtoD]` namanya bergeser ke **kolom 19**.
- Kurangi 12 peluncuran warmup LM head (`main.mojo:1533`) yang ikut terprofil
  tapi tak masuk timer. Untuk celah antar-kernel: potong semua kejadian sampai
  `memcpy HtoD` terakhir (fase muat bobot 1369 salinan mendominasi).

## Trap Kaggle (kernel/CLI)

- Isi `/kaggle/working` ikut ter-commit jadi OUTPUT → **bersihkan di `finally`**;
  ambil log lewat `api.kernels_logs()`, bukan `kaggle kernels output`.
- `/kaggle/temp` tidak selalu ada; yang pasti `/kaggle/src` & `/kaggle/working`.
- Binary CLI `kaggle` bisa menggantung — pakai `KaggleApi()` dari Python.
- **TIDAK ada cara membatalkan kernel yang sedang jalan.** Menghentikan skrip
  lokal tidak menghentikan kernel Kaggle (kernel terus sampai COMPLETE dan
  tetap memakan kuota). Endpoint `POST /api/v1/kernels/cancel-session/{id}`
  yang disebut issue kaggle-cli#1172 mengembalikan **404** di akun ini, dan
  `KaggleApi` terpasang tidak punya metode cancel (satu-satunya kecocokan
  "cancel" di kaggle_api_extended.py = "Deletion cancelled", konfirmasi hapus).
- `api.kernels_status()` mengembalikan **enum**; bandingkan dengan
  `str(x).rsplit('.',1)[-1]`. Status `COMPLETE` **bukan** bukti sukses; baca log.
- Sebelum tiap API: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  all_proxy ALL_PROXY` + `export PATH="/Users/macmini/.mounty/SSD_External/miniconda3/bin:$PATH"`.
- **Kernel build CPU TIDAK bisa meluluskan tes GPU**: keenam
  `mojo run tests/*_gpu.mojo` gagal `Failed to open library "libcuda.so.1"`.
  Itu BUKAN bug kode — CPU build hanya **compile gate**. Verdict LULUS hanya
  mungkin di kernel GPU (`push_to_kaggle.sh`).
- `mojo run` **menerima** `--target-accelerator=sm_75`. Tanpa flag itu tes GPU
  gagal KOMPILASI (`the target architecture '' is invalid`).
- Menjalankan biner `bonsai_infer` di luar env build butuh
  `libKGENCompilerRTShared.so`, `libMSupportGlobals.so`,
  `libAsyncRTRuntimeGlobals.so`, `libNVPTX.so` — sumbernya hanya env pixi
  (`.pixi/envs/default/lib`, lihat `deploy_on_kaggle.sh:769`). Paket pip
  `max`/`modular` TIDAK memuat `libKGENCompilerRTShared.so`. glob Python tak
  menembus direktori berawal titik — pakai `find`.
