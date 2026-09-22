# MEMORY.md — bonsai-1bit-t4-mojo

## Aturan kerja

- Repo `oki-abrian/bonsai-mojo-1bit-optimization-t4-cuda`, cabang `main`. Catatan
  harian ikut ter-commit. Meniru llama.cpp (`references/llama.cpp-prism/`).
- **Mesin lokal BUKAN mesin build.** Build/uji di Kaggle T4 (sm_75). Alur verifikasi:
  CPU build (~20 mnt) lalu GPU build (~40 mnt).
- **Diff harus aditif**: parameter baru punya default yang mereproduksi perilaku
  lama byte-identik; jangan mengubah isi fungsi yang ada.
- Berkas terbukti, **tidak disentuh tanpa izin eksplisit**: `deploy_on_kaggle.sh`,
  `run_deploy.py`, `push_to_kaggle.sh`, `kaggle_cpu_build/push_cpu_build.sh`.
- **Jangan menyajikan 1-bit vs 2-bit sebagai pilihan A/B** (teguran: "keduanya
  penting"). Keduanya didukung bersamaan, dipilih via `BONSAI_BITS`.
- `KHQ_DUMP_FORCE`, dan apa pun yang tidak berprefiks `BONSAI_`, TIDAK ikut ke
  container — `push_to_kaggle.sh` hanya meneruskan `BONSAI_*`.

## Kontrak model

- **1-bit** = `prism-ml/Bonsai-27B-mlx-1bit` (`model_type: qwen3_5`). Dequant
  `(2q-1)*(s_ckpt/2)`.
- **2-bit** = `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`
  (`prism_hadamard_qwen35`, `schema_version: 2`, butuh `hadamard.json`).
  Dequant `w=(q-1)*s`, q∈{0,1,2}, **q=0 = −1 sah**, tanpa /2. U32 LE, 16
  bobot/word, lane `i` di bit `2i`. Scales/bias F16 `[N,K/128]`.
- Signs dikunci lebar masuk K: `sign_widths=[5120,6144,17408]` (28.672 nilai ±1).
  Tensor `.signs` ada tapi TIDAK dipakai runtime referensi.
- FWHT butterfly `rendah=x+y, tinggi=x-y` (= Sylvester `H_B/sqrt(B)`).
  `embed_tokens.weight` satu-satunya `inverse_weight_names`.
- `gdn_v_grouped: True` — aktivasi GDN sudah terkelompok, JANGAN permutasi.
  Geometri GDN `config.mojo:62`: `gdn_head_v_dim=128`, H_v=48, D_k=128.
- Searsitektur: 27B hibrida Qwen3.5 3:1, 64 layer, hidden 5120, intermed 17408,
  vocab 248320. `BONSAI_BITS=2` (default 1).
- `BONSAI_DECODE_H2=1` OPT-IN, aman global (linear.mojo:409 mensyaratkan
  `bits==2`). Tanpa itu 2-bit diam-diam pakai kernel skalar: 75,25 vs 58,48 ms/token.

## Trap bahasa Mojo 25.x (semua pernah menggagalkan build Kaggle)

- Tidak ada interpolasi `\(` (itu Swift): pakai `+ String(x)`.
- Tidak ada kembalian tuple: nilai kedua lewat `UnsafePointer[Int]`.
- `out` kata kunci cadangan → parameter pakai `dst`. Jangan namai parameter `len`.
- Konstruktor posisional saja: `HadamardSigns(1, buf, widths, 0, 0)`, bukan
  `HadamardSigns(block=1,...)`. Tidak ada konstruktor anggota otomatis — tulis
  `__init__` eksplisit.
- Jangan uji pointer `if ptr:` → `if p != UnsafePointer[Float32, MutAnyOrigin]():`.
- `sqrt` dari `math`, dipanggil `sqrt(Float32(x))`. `ord()` selalu dibungkus
  `UInt8(ord(","))`.
- `shuffle_down` butuh delta konstan waktu-kompilasi; geometri runtime tak
  terduga → pakai reduksi murni di shared memory.

## Trap CUDA

- `QmvTraits<T>` di `namespace bonsai::sm75` (qmv_sm75_kernel.cu:42-65); kernel
  baru HARUS di namespace turunan (`bonsai::sm75::q2t`).
- `fuse_u8`/`fuse_f32` **mengalokasi & menyalin** (bukan view): setelah fusi,
  free hanya di cabang yang pegang ownership asli.
- `DeviceBuffer` RAII: jangan deklarasi di dalam `if` kalau raw pointer dipakai
  di scope luar — hoist. Output GEMV dense terstrided `N_total`, bukan `n_tail`.
- **Kelas defek "reduksi warp bergantung geometri"** (diperbaiki): `sq +=
  shuffle_down(sq,16/8/4/2/1)` + `smem[lane] if lane<NWARP else 0` hanya absah
  bila SETIAP warp penuh DAN jumlah warp = NWARP; kasus lain salah TANPA crash.
  `gdn_norm_gate` kini `if D_v == 128` + cabang SMEM murni + guard `D_v>128 →
  Error` (ops.mojo). block_size kernel lain tetap aman.

## Hasil ukur T4 & disiplin ukur

- Puncak DRAM terukur **264 GB/s**, datar vs okupansi & aliran.
- Decode 2-bit h2 (v40, 48,97 ms/token): GEMV 32,50 (66%) | rekurensi GDN 8,75
  (18%) | FWHT 1,89 | add_rmsnorm 1,26 | qmv_dense 0,73.
- GEMV 2-bit h2 **mentok** ~214 GB/s = 96% kemampuan kernelnya. Rumus:
  `ms/token ≈ byte dibaca ÷ bandwidth tercapai`.
- "GEMV habis" HANYA untuk 2-bit. Kernel 1-bit (`qmv_vec_nib_q1o_kernel`, :78)
  beda teknik (stage float + LUT nibble SMEM + 32 indexed gather/128 bobot);
  2-bit h2 pakai register expand + `__hfma2`. 1-bit baca 0,55× byte tapi 1,196×
  lebih lambat ⇒ headroom ~2,35× (56,61 → ~32,5 ms/token). **Belum diuji
  langsung** (butuh port teknik h2 + A/B).
- **Presisi aritmetika ≠ presisi penyimpanan**: bobot 2-bit DIHITUNG dalam FP16.
  Kernel 2-bit skalar stage `float x_s[]` (:3400) ⇒ jalur FP32; h2 stage
  `__half x_s[]` (:3587) + `q2t_expand_word_h2` (:3542). Implikasi: format 4-bit
  (NVFP4) menambah byte, aritmetika tetap FP16 ⇒ ~2× lebih lambat, dan T4
  (Turing sm_75) tidak punya tensor core FP4 (Blackwell-saja).
- Rekurensi GDN tak bereaksi terhadap apa pun di dalam kernelnya — jangan setel
  lagi (h2b +3,4%, unroll 4x +4,6% lebih lambat). CUDA Graphs tak berguna
  (celah antar-kernel 0,99 us).
- **PROBE TERISOLASI MENIPU**: rekurensi 1,51× lebih cepat terisolasi, 0,981× di
  dalam model. Ukuran sah = A/B ujung-ke-ujung.
- **Drift mesin 5–7% antar sesi**; selisih <8% tak bisa dinilai lintas sesi.
- **JANGAN ukur dengan run pendek**: 2-bit h2 47,36 ms (24 token) → 58,48 ms
  steady-state (766 token) = +23%. 1-bit: 58,53 → 60,66.

## Gerbang koherensi BUKAN uji kebenaran

- deploy_on_kaggle.sh:961-1276. Yang diperiksa hanya degenerasi: siklus
  periodik (`loop@`), keragaman 50 token ekor (ambang `<12` unik = gagal), rasio
  id unik per jendela. **LULUS ≠ jawaban benar** (v133 penalaran.think0 LULUS
  tapi terpotong di 512 token tepat sebelum angka akhir).
- `BONSAI_COH_TOKENS` (bawaan **512**, :961), `BONSAI_COH_THINK` (0/1/both,
  :966), GREEDY (`BONSAI_TEMP_X100=0`, :1100).
- **Plafon keras 4096**: `main.mojo:1080` `var max_seq = 4096` (konstanta, tanpa
  env override) + penjaga :1518 `if prompt_len+max_tokens > max_seq: raise FATAL`.
  Prompt terukur: cerita 27, penalaran 71 (+template ≈ 86) ⇒ aman maks ≈ 4010.
  Dipakai **3900** dan lolos.
- **Jeblakan**: pipeline COH (:1101-1103) disaring
  `grep -E "\[GEN\] token id|\[PERF\] rata-rata"`, jadi pesan FATAL biner hilang.
  Gejala: "token tidak terekam di log" + run cepat (26 s vs 200 s). Curigai
  penjaga panjang konteks, BUKAN modelnya.
- Dua prompt tetap (:1065-1068): `cerita` (nelayan + peta) dan `penalaran`
  (pipa A 6 jam / B 4 jam, jawaban benar = **3 jam**).
- **Produksi TIDAK mencetak teks inferensi** — hanya `>> [GEN] token id: N`
  (grep di :1144,1653,1734 hanya ekstrak id). Teks harus didecode lokal dengan
  `tokenizer.json` model relevan (lib `tokenizers`). Bukan pemborosan: memang
  tidak ada sumber teksnya di log.
- Uji kebenaran soal kompleks = anggaran ≥2048 token + decode lokal + bandingkan
  dgn jawaban diketahui. Kernel khusus: `infer_susah/` →
  `okiabrian/bonsai-2bit-infer` (max 2048, T=0,70); v44 menjawab 3 jam lengkap.
- 731 token A/B beda (pertama di indeks 21) BUKAN bug: sampling aktif. Pada 24
  token *greedy*, h2 == skalar == BASELINE persis.

## KHQ (§5b: dump → verifikasi → kalibrasi → uji runtime)

- Sakelar **`BONSAI_KHQ_ENABLE=0`** melewati seluruh 5b (default tidak diset =
  cabang lama jalan). Knob: `KHQ_DUMP_TOKENS` (512), `KHQ_DUMP_FORCE=1` (tidak
  bisa dikirim dari lokal).
- Hasil uji 5b.4 memakai `BONSAI_KHQ_PATH` dengan prompt `PROMPT_TOKENS_LONG`
  (508–512 token); 5b.5/5b.8 ikut menulis `[GEN]`, jadi ekstraksi HARUS dibatasi
  sampai baris `>> [PERF] rata-rata` berikutnya.
- **[DIPERBAIKI] Cache KHQ tidak lengkap**: cache hanya simpan `kv_dump.bin` +
  `khq_dump.log`, padahal 5b.2 mewajibkan `attn_<lid>.bin` (16 × 7,3 MB). Tiap
  run pakai-ulang gagal verifikasi (`[KHQ-FAIL] attn_3.bin tidak ada`) →
  kalibrasi batal, tapi pipeline tetap exit 0 & gerbang LULUS — hijau palsu.
  Perbaikan (:1313,1351-1358,1376-1380): `ATTN_CACHE_DIR=$CACHE_DIR/attn_b${BONSAI_BITS}`,
  syarat pakai-ulang mensyaratkan attn ada, kalau tidak → dump segar.
- **Mekanisme lestari**: yang terarsip ke `mojo_build_cache.tar.gz` HANYA
  `.cache_t4_build` (= `CACHE_DIR`, deploy_on_kaggle.sh:79). `KHQ_DIR`
  (`$WORKING/khq_real`) TIDAK pernah lestari antar run — tetapi arsip yang
  direstorasi berasal dari kernel **`bonsai-build-cpu`**, bukan output run GPU,
  jadi isi `CACHE_DIR` juga tidak kembali. Yang benar-benar lestari adalah
  mount OUTPUT `khq_real/` (terbukti: `kv_dump_b2.bin diimpor dari output GPU
  run sebelumnya`).
- **TERBUKTI v138 (41 mnt) & v139 (36 mnt), exit 0**: dump segar 13,8 mnt →
  `[KHQ-OK] dump ASLI dari model valid` → kalibrasi 4,8 mnt →
  `centroid: ver=2 layers=16 dim=256 bytes=3440976` → 5b.4 `jalur KV terkompresi
  AKTIF (watermark 256/128)`, `[KHQ-COMPRESS]` di 16 layer → `[KHQ-OK] jalur
  window setara secara numerik`. splits=16 = 51,50 ms/token; beban konteks
  pendek +0,1%; regime 512 token splits=1 = 1,209× baseline fp16.
- Run 2-bit dengan KHQ=0: 16 menit (vs 39 menit KHQ aktif), exit 0.
- **Trap `set -e` pada penugasan**: `VAR="$(perintah)"` mewarisi status keluar;
  `find` ke direktori tak ada → 1 → seluruh run mati (v137 exit 1 tepat setelah
  gerbang koherensi). `2>/dev/null` tidak menolong — SELALU `|| true`.

## Cache build & mekanika Kaggle

- Sidik jari build-cache (§4a): sha256 `main.mojo` + `src/**/*.mojo` harus sama
  dengan `$CACHE_DIR/bonsai_infer.fp`, kalau beda FATAL tanpa fallback → setiap
  kali .mojo berubah (atau cache basi) jalankan CPU build dulu.
- Isi `/kaggle/working` ikut ter-commit jadi OUTPUT → bersihkan di `finally`;
  ambil log lewat `api.kernels_logs()`, bukan `kaggle kernels output`.
- CLI `kaggle` bisa menggantung — pakai `KaggleApi()` dari Python. Sebelum tiap
  API: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY`
  + `export PATH=".../miniconda3/bin:$PATH"`.
- **Tidak ada cara membatalkan kernel yang sedang jalan.** `kernels_status()`
  mengembalikan enum: bandingkan `str(x).rsplit('.',1)[-1]`. `COMPLETE` bukan
  bukti sukses; baca log.
- Kernel build CPU TIDAK bisa meluluskan tes GPU (`mojo run tests/*_gpu.mojo`
  gagal `libcuda.so.1`, bukan bug kode); `mojo run` butuh
  `--target-accelerator=sm_75`. Biner butuh `libKGENCompilerRTShared.so` dari
  env pixi. glob Python tak menembus direktori berawal titik — pakai `find`.

## Cara membaca profil nvprof

- Hardware counter TIDAK tersedia (cc 7.5): `ncu` exit 1. Hanya
  `--print-gpu-trace --csv`; `--kernels regex:` ditolak, nama polos OK.
- CSV: `Start` kolom 0, `Duration` kolom 1 (ms), nama kernel **kolom 18**
  (baris `[CUDA memcpy HtoD]` namanya bergeser ke kolom 19). Baris setelah
  header adalah baris SATUAN. nvprof menambah overhead per launch — pakai TOTAL.
- Celah antar-kernel: potong semua kejadian sampai `memcpy HtoD` terakhir.
