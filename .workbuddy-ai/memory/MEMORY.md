# MEMORY.md — bonsai-1bit-t4-mojo

## Aturan kerja proyek

- Meniru llama.cpp (`references/llama.cpp-prism/`, baca-saja). Metodologi:
  `LAPORAN_AUDIT_PARITAS_LLAMACPP_PRISM.md`.
- **Mesin lokal BUKAN mesin build.** Build di Kaggle/T4 (sm_75). Verifikasi:
  CPU build (compile gate ~20 mnt) lalu GPU build (~40 mnt).
- **Diff harus aditif**: parameter baru punya default yang mereproduksi perilaku
  lama byte-identik; jangan mengubah isi fungsi yang ada.
- Berkas terbukti, **tidak disentuh tanpa izin eksplisit**: `deploy_on_kaggle.sh`,
  `run_deploy.py`, `push_to_kaggle.sh`, `kaggle_cpu_build/push_cpu_build.sh`.
  (MENJALANKAN-nya boleh, mengeditnya tidak.)
- **Jangan menyajikan 1-bit vs 2-bit sbg pilihan A/B** (teguran 2026-09-22:
  "keduanya penting"). Keduanya didukung bersamaan, dipilih via `BONSAI_BITS`.

## Kontrak Bonsai-2 (2-bit ternary, pack MLX prism_hadamard_qwen35)

- Dequant `w = (q-1)*s`, q ∈ {0,1,2}, **q=0 = −1 kode sah**. Tidak ada /2
  (1-bit: `(2q-1)*(s_ckpt/2)`).
- Bit: U32 LE, 16 bobot/word, lane `i` di bit `2i`. Scales/bias F16 [N,K/128].
- Signs dikunci lebar masuk K: `sign_widths=[5120,6144,17408]`, 28.672 nilai ±1.
  Tensor `.signs` ADA tapi TIDAK dipakai runtime referensi.
- FWHT butterfly `rendah=x+y, tinggi=x-y` = Sylvester `H_B/sqrt(B)`.
  `embed_tokens.weight` satu-satunya `inverse_weight_names`.
- `gdn_v_grouped: True` — aktivasi GDN sudah terkelompok, JANGAN permutasi.
- `BONSAI_BITS=2` (default 1). Geometri GDN `config.mojo:62`: `gdn_head_v_dim=128`,
  H_v=48, D_k=128.

## Trap bahasa Mojo 25.x (semua pernah bikin build Kaggle gagal)

- **Tidak ada interpolasi string `\(`** (itu Swift): `+ String(x)`.
- **Tidak ada kembalian tuple**: nilai kedua lewat `UnsafePointer[Int]`.
- **`out` kata kunci cadangan** — parameter pakai `dst`.
- **Konstruktor posisional**: `HadamardSigns(block=1,...)` error; tulis
  `HadamardSigns(1, buf, widths, 0, 0)`.
- **Tidak ada konstruktor anggota otomatis** — tulis `__init__` eksplisit.
- **Jangan uji pointer `if ptr:`** — pakai
  `if p != UnsafePointer[Float32, MutAnyOrigin]():`.
- `sqrt` dari `math`, dipanggil `sqrt(Float32(x))`.
- **`shuffle_down` butuh delta konstan waktu-kompilasi**; untuk geometri runtime
  tak terduga pakai reduksi murni di shared memory.
- **Jangan namai parameter `len`**; **`ord(...)` selalu dibungkus** `UInt8(ord(","))`.

## Trap CUDA

- `QmvTraits<T>` di `namespace bonsai::sm75` (qmv_sm75_kernel.cu:42-65); kernel
  baru HARUS di namespace turunan (`bonsai::sm75::q2t`).
- `fuse_u8`/`fuse_f32` **mengalokasi & menyalin** (bukan view): setelah fusi,
  free hanya di cabang yang pegang ownership asli.
- `DeviceBuffer` RAII: jangan deklarasi di dalam `if` kalau raw pointer dipakai
  di scope luar — hoist. Output GEMV dense terstrided `N_total`, bukan `n_tail`.
- **Kelas defek "reduksi warp bergantung geometri" (diperbaiki 2026-09-20):**
  `sq += shuffle_down(sq,16/8/4/2/1)` + `smem[lane] if lane<NWARP else 0` hanya
  absah bila SETIAP warp penuh DAN jumlah warp tepat = NWARP; kasus lain
  menghasilkan nilai salah TANPA crash. `gdn_norm_gate` kini `if D_v == 128`
  (kode lama identik) + cabang SMEM murni + guard `D_v > 128 → Error` (ops.mojo).
  block_size kernel lain TETAP AMAN (geometri tak bergantung dimensi runtime).

## Hasil ukur T4 (fondasi keputusan optimasi)

- **Puncak DRAM terukur 264 GB/s**, datar vs okupansi & aliran.
- **Decode 2-bit h2 (v40, 48,97 ms/token)**: GEMV 32,50 (66%) | rekurensi GDN
  8,75 (18%) | FWHT 1,89 | add_rmsnorm 1,26 | qmv_dense 0,73 | kecil sisanya.
- **GEMV 2-bit h2 sudah mentok**: ~214 GB/s = 96% kemampuan kernelnya.
  Rumus berlaku: `ms/token ≈ byte dibaca ÷ bandwidth tercapai`
  (6,95 GB ÷ 214 = 32,5 ms ↔ GEMV terukur 32,50 ms).
- **"GEMV habis" HANYA untuk 2-bit.** Kernel produksi 1-bit
  (`qmv_vec_nib_q1o_kernel`, qmv_sm75_kernel.cu:78) memakai teknik lain:
  stage x sebagai float + LUT nibble di SMEM + 32 indexed SMEM gather per 128
  bobot; 2-bit h2 memakai register expand + `__hfma2`. 1-bit membaca 0,55×
  byte 2-bit tapi 1,196× lebih lambat ⇒ ~2,2× efisiensi lebih rendah (~67 vs
  ~147 GB/s; inferensi silang-run BUKAN ukuran langsung). Headroom ~2,35×;
  bila ditutup, decode 1-bit 56,61 → ~32,5 ms/token. **Belum pernah diuji
  langsung** (butuh port teknik h2 ke 1-bit + A/B).
- **Presisi aritmetika BEDA dari presisi penyimpanan.** Bobot disimpan 2-bit
  tapi DIHITUNG dalam FP16. Kernel 2-bit skalar stage `float x_s[]` (:3400) ⇒
  jalur FP32; kernel h2 stage `__half x_s[]` (:3587), bobot diperluas ke
  register half2 (`q2t_expand_word_h2`:3542), `__hfma2` = 2 hasil kali per
  instruksi, flush ke FP32 tiap 8 bobot. Kernel 1-bit juga `float x_s[]` (:100).
  IMPLIKASI: format 4-bit (NVFP4/E2M1) mengubah PENYIMPANAN (lebih banyak byte)
  sementara aritmetika tetap FP16 — bukan 8× lebih cepat, malah ~2× lebih
  lambat (13,9 GB ÷ 214 ≈ 65 ms). T4 = Turing sm_75, tidak punya tensor core FP4
  (Blackwell-saja).
- **Rekurensi GDN tidak bereaksi terhadap apa pun di dalam kernelnya** — jangan
  setel lagi (h2b 3,4% lebih lambat; unroll 4x 4,6% lebih lambat).
- **CUDA Graphs tidak ada gunanya**: celah antar-kernel 0,99 us.
- **PROBE TERISOLASI MENIPU**: rekurensi 1,51× lebih cepat terisolasi, 0,981×
  di dalam model. Ukuran sah = A/B ujung-ke-ujung di model.
- **Drift mesin Kaggle 5–7% antar sesi**; selisih <8% tak bisa dinilai lintas
  sesi. Yang sah: A/B satu sesi, atau selisih >25% per kernel nvprof.
- **Produksi TIDAK mencetak teks hasil inferensi** — hanya `>> [GEN] token id: N`
  (grep di deploy_on_kaggle.sh:1144,1653,1734 memang hanya ekstrak id).
  Jadi untuk melihat jawaban sbg teks, token id HARUS didecode lokal dgn
  `tokenizer.json` model yg relevan (`tokenizers` lib). Bukan pemborosan:
  memang tidak ada sumber teksnya di log.
- **JANGAN UKUR DENGAN RUN PENDEK.** Terbukti: 2-bit h2 47,36 ms (24 token) →
  **58,48 ms steady-state (766 token)** = +23%; skalar 72,7 → 75,25 = +3,5%.
  1-bit: 58,53 (13 token) → 60,66 (1739 token).

## Status 2-bit (diuji 2026-09-21/22)

- **Uji akurasi pertama**, kernel `okiabrian/bonsai-2bit-infer` v44, soal pipa
  A/B (`infer_susah/infer_config.json`), max 2048 token, sampling temp 0,70.
  Skalar 781 token / **75,25 ms**, h2 766 token / **58,48 ms** ⇒ h2 1,286× lebih
  cepat. **Keduanya menjawab 3 jam — benar**, penalaran lengkap (1/6+1/4=5/12),
  ada LaTeX + tabel verifikasi, **nol kebocoran CJK** (referensi 1-bit bocor
  `第一阶段`). Semua verdict struktural LULUS (6c prefill, 6d int8 W2A8, 6g h2).
- **731 token A/B beda, pertama di indeks 21 BUKAN bug**: sampling aktif, drift
  logit kecil membalik pilihan token; pada 24 token *greedy* h2 == skalar ==
  BASELINE persis. Uji greedy ~2000 token penentu ekuivalensi belum dijalankan.
- **`BONSAI_DECODE_H2` OPT-IN** (linear.mojo:409 mensyaratkan `self.bits == 2`,
  jadi aman diekspor global). Tanpa itu 2-bit diam-diam pakai kernel skalar:
  75,25 vs 58,48 ms/token.

## Dukungan 1-bit + 2-bit di produksi (diedit 2026-09-22, atas izin "iya")

- `deploy_on_kaggle.sh` §5 memilih direktori bobot dari `BONSAI_BITS`:
  `2` → `/kaggle/input/*/bonsai-2bit-weights`, selainnya → `bonsai-27b-mlx-1bit`
  (default 1 = perilaku lama identik).
- `export BONSAI_DECODE_H2=1` ditambahkan di blok export (~baris 831).
- Nama cache KHQ kini per-model: `kv_dump_b${BONSAI_BITS}.bin` /
  `khq_dump_b${BONSAI_BITS}.log` (nama kerja `$KHQ_DIR/kv_dump.bin` dan
  `$DIST_DIR/khq_dump.log` sengaja dibiarkan agar pembaca 5b.2/5b.9 tak berubah).
- **Paket 2-bit adalah MODEL BERBEDA, bukan bit berbeda.** 1-bit =
  `prism-ml/Bonsai-27B-mlx-1bit` (`model_type: qwen3_5`, config 3.790 B,
  tokenizer 19,99 MB, bobot 5,13 GB terpecah). 2-bit =
  `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (`prism_hadamard_qwen35`,
  `schema_version: 2`, config 58.145 B, tokenizer 12,81 MB, bobot 8,60 GB satu
  berkas + `hadamard.json` 297 KB). Searsitektur (27B, hibrida Qwen3.5 3:1,
  64 layer, hidden 5120, intermed 17408, vocab 248320), dua rilis berbeda.
- Bobot 2-bit sudah ada sbg dataset Kaggle `okiabrian/bonsai-2bit-weights`
  (8.595.477.990 B). `enable_internet: true` di kernel produksi.
- **Gerbang produksi sebagian besar MODEL-AGNOSTIK** (saya pernah keliru
  menyangka sebaliknya): gerbang koherensi = deteksi loop + keragaman ekor +
  gerbang-C swakalibrasi; nilai `BASE` yg dicetak HANYA konteks, bukan ambang
  (deploy_on_kaggle.sh:1227). Uji tokenizer 5.0 memakai config.json checkpoint
  itu sendiri. Yang terikat bobot: cache KHQ (deploy_on_kaggle.sh:1266
  "BOBOT TETAP").
- **Sakelar KHQ: `BONSAI_KHQ_ENABLE`** (ditambahkan 2026-09-22 atas izin
  "dimatikan dulu gpp, kan ini dicoba dulu tanpa khq"). Default tidak diset =
  cabang lama jalan persis. `BONSAI_KHQ_ENABLE=0` → seluruh blok 5b (dump, uji
  runtime kompresi KV, kalibrasi) dilewati; dicetak `>> [KHQ] 5b DILEWATI`.
  Karena `push_to_kaggle.sh` hanya meneruskan `BONSAI_*`, nama HARUS pakai
  prefiks itu (sebelumnya `KHQ_DUMP_FORCE` TIDAK ikut ke container).
  Bukti run 2-bit KHQ=0: 16 menit (vs 39 menit KHQ aktif), exit 0, semua
  gerbang lulus.
- **Sidik jari build-cache** (§4a): butuh sha256 `main.mojo` + `src/**/*.mojo`
  sama dengan `$CACHE_DIR/bonsai_infer.fp`, kalau beda FATAL tanpa fallback.
  Jadi setiap kali .mojo berubah (atau cache basi) → jalankan CPU build dulu.

## Cara membaca profil nvprof

- Hardware counter TIDAK tersedia (cc 7.5): `ncu` exit 1. Hanya
  `--print-gpu-trace --csv`; `--kernels regex:` ditolak, nama polos OK.
- CSV: `Start` kolom 0, `Duration` kolom 1 (ms), nama kernel **kolom 18**
  (baris `[CUDA memcpy HtoD]` namanya bergeser ke kolom 19). Baris setelah
  header adalah baris SATUAN. nvprof menambah overhead per launch — pakai TOTAL.
- Celah antar-kernel: potong semua kejadian sampai `memcpy HtoD` terakhir.

## Trap Kaggle

- Isi `/kaggle/working` ikut ter-commit jadi OUTPUT → bersihkan di `finally`;
  ambil log lewat `api.kernels_logs()`, bukan `kaggle kernels output`.
- CLI `kaggle` bisa menggantung — pakai `KaggleApi()` dari Python.
- **Tidak ada cara membatalkan kernel yang sedang jalan.** Menghentikan skrip
  lokal TIDAK menghentikan kernel Kaggle.
- `kernels_status()` mengembalikan enum: bandingkan `str(x).rsplit('.',1)[-1]`.
  `COMPLETE` bukan bukti sukses; baca log.
- Sebelum tiap API: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  all_proxy ALL_PROXY` + `export PATH=".../miniconda3/bin:$PATH"`.
- **Kernel build CPU TIDAK bisa meluluskan tes GPU**: `mojo run tests/*_gpu.mojo`
  gagal `libcuda.so.1` — BUKAN bug kode. `mojo run` butuh
  `--target-accelerator=sm_75`.
- Biner `bonsai_infer` butuh `libKGENCompilerRTShared.so` dsb dari env pixi;
  paket pip `max`/`modular` tidak memuatnya. glob Python tak menembus direktori
  berawal titik — pakai `find`.
