# MEMORY.md — bonsai-1bit-t4-mojo

## Aturan kerja
- **Mesin lokal BUKAN mesin build** — Kaggle T4 (sm_75): CPU ~20 mnt, GPU ~40 mnt.
- **Diff aditif**; berkas terbukti (`deploy_on_kaggle.sh`, `run_deploy.py`, `push_to_kaggle.sh`, `kaggle_cpu_build/push_cpu_build.sh`) tak disentuh tanpa izin.
- **JANGAN pakai git sebagai acuan** — tertinggal ~2 minggu; working tree = keadaan mutakhir. Hanya env `BONSAI_*` masuk container; 1&2-bit via `BONSAI_BITS` (bukan A/B).
- **JANGAN beberapa Edit paralel pada 1 berkas** (write kedua menimpa pertama diam-diam).

## Kontrak model (27B Qwen3.5 3:1, 64 layer, hidden 5120, intermed 17408, vocab 248320)
- 1-bit `prism-ml/Bonsai-27B-mlx-1bit`: `(2q-1)*(s/2)`. 2-bit `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` (schema 2, butuh `hadamard.json`): `w=(q-1)*s`, q∈{0,1,2}, tanpa /2; U32 LE 16 bobot/word, lane i di bit 2i; scales F16 `[N,K/128]`.
- Signs dikunci K `[5120,6144,17408]`; `.signs` tak dipakai runtime. `gdn_v_grouped:True`; H_v=48, head_v_dim=D_k=128. `BONSAI_DECODE_H2=1` OPT-IN.

## Trap Mojo 25.x
- Tak ada `\(`, tak ada tuple return (ke-2 via `UnsafePointer[Int]`); `out`/`len` terlarang → `dst`. **`out` adalah kata kunci terpasang** (dipakai `__init__(out self)`): dipakai sebagai nama parameter/variabel fn biasa → galat parse "expected argument name" (terbukti build CPU v36, 2026-09-28, membuang satu putaran build ~24 mnt).
- **`UnsafePointer.address_of` TIDAK ADA di 25.x** (deprecated 25.3, dihapus 25.5). Pengganti resmi: `UnsafePointer[T, MutAnyOrigin](to=var_lokal)` — ambil alamat variabel lokal tanpa malloc (tanda tangan diverifikasi di sumber stdlib v25.6.0/v25.7.0; pola terbukti di uji resmi v25.7.0). Compiler build Kaggle = 0.25.7.0; gaya lama `UnsafePointer[T, MutAnyOrigin]` masih sah di env build itu (build sukses 2026-09-20 sudah 0.25.7.0).
- **`Scalar[dtype, origin]` TIDAK sah** — `Scalar`=`SIMD[dtype,1]`. Yang sah: `UnsafePointer[Scalar[DType.float16], MutAnyOrigin]`. `bitcast` 1 param: `x.bitcast[UnsafePointer[X, MutAnyOrigin]]()`.

## Trap CUDA
- Kernel baru WAJIB namespace `bonsai::sm75::` (`qmv_sm75_kernel.cu:42-65`). Reduksi warp hanya absah jika warp penuh & #warp=NWARP.

## Disiplin ukur
- DRAM 264 GB/s; GEMV h2 mentok 214 GB/s. Drift antar-sesi 5–7%; **run pendek tidak sah** (47,36@24→58,48@766 tok). Probe terisolasi menipu → A/B e2e. T4 tak ada FP4 tensor core.

## Gerbang koherensi (deploy_on_kaggle.sh:961-1276) BUKAN uji kebenaran
- Cek degenerasi saja. Gagal BUKAN bug instrumentasi (v143 vs v145: kontrol tanpa dump = token IDENTIK). Knob `BONSAI_COH_TOKENS`(512), `BONSAI_TEMP_X100=0`. Plafon 4096; pakai 3900. Uji benar: ≥2048 tok + `infer_susah/`.

## KHQ (§5b) — default `BONSAI_KHQ_ENABLE=1`
- **TERBUKTI v138/v139/v146 exit 0**: dump ~13 mnt → kalibrasi ~4,8 mnt → `[KHQ-COMPRESS]` 16 layer → numerik setara. splits=16 terbaik.
- **Slot state WAJIB cocok di sisi C**: `khq_state_slot(n)` MENJEPIT n luar jangkauan ke 0. Dulu [2] → act-dump pakai slot 2 belum ada → timpa `KhqGlobals` → crash "khq_step GAGAL di layer 3" padahal KHQ_ENABLE=0. Kini [3]+clamp>2 (0=runtime,1=dump,2=act-dump).
- Jebakan `set -e`: `VAR="$(find ...)"` status 1 membunuh run → SELALU `|| true`.

## Dump aktivasi (v143)
- `BONSAI_DUMP_ACT_DIR` → `act_dump.mojo` slot 2 (3 site; decode saja); gate=`getenv`. Isi = run TERAKHIR → **ukur berkas, jangan baca log**.
- **TRAP FFI build CPU (2026-09-28)**: `khq_state_slot_cell` terkodekan `dlopen→dlsym→dlclose→call` (dlclose DULU). Saat refcount dlopen jatuh ke 0, glibc melepas `.so` → SIGSEGV. Hanya terjadi di build CPU (wheel); build GPU (`--target-accelerator=sm_75`) codegen-nya benar (7 titik FFI lain panggil-dulu-baru-dlclose). Semua konsumen slot (`_ad`, `_dg`) lumpuh di build CPU.
- **PERBAIKAN POLA SUMBER GAGAL (terbukti act-dump v5, 2026-09-30)**: bentuk out-pointer `dst[] = f(...)` di `khq_state_slot_cell` pun masih ditaruh dlclose SEBELUM call oleh kompilator CPU → biner v38 tetap SIGSEGV tanpa LD_PRELOAD (diag B/D/E/F rc=-11, 0/20 task; A/C lulus). Bentuk `return f(...)` (tail call) juga buruk (v35). **Kesimpulan: urutan kodegen dlclose-vs-call TAK BISA diandalkan pada build CPU, bentuk sumber apa pun.**
- **PERBAIKAN DETERMINISTIK (2026-09-30, src/ops.mojo)**: `try_open_cuda_lib()` membuka .so dengan `RTLD.NOW | RTLD.GLOBAL | RTLD.NODELETE` (alias `_RTLD_FLAGS`). RTLD_NODELETE (glibc 4096; stdlib Mojo 25.7.0 punya `RTLD.NODELETE`) = dlclose tak PERNAH melepas pemetaan → urutan kodegen tak relevan; semua titik FFI kebal kelas bug ini. Verifikasi stdlib: tag `modular/v25.7.0`, `mojo/stdlib/stdlib/sys/ffi.mojo` (`OwnedDLHandle(path, flags: Int = DEFAULT_RTLD)`). Menunggu verifikasi e2e (build CPU v39 + act-dump v6 tanpa preload).
- **Perbaikan driver (cadangan terbukti, tanpa sentuh src/)**: `LD_PRELOAD=<dir>/libbonsai_qmv_sm75.so` di env `bonsai_infer`. Catatan: probe ctypes "kodegen" TIDAK stabil sebagai kontrol (v5 rc=0 padahal biner nyata crash) — probe .so-level jangan dijadikan acuan; hanya run biner sungguhan yang sah.
- Forensik: wheel terekstrak lokal di `/tmp/whl2/bonsai_1bit_t4/`; `objdump -p` (macOS Xcode) bisa baca ELF; NEEDED `.so` hanya libc/libm/ld.

## Kaggle
- Sidik jari sha256 `main.mojo`+`src/**/*.mojo` vs `$CACHE_DIR/bonsai_infer.fp`; berubah → CPU build dulu. Jebakan dua mount (v141, diperbaiki): `run_deploy.py` kini prioritaskan `/kaggle/input/bonsai-build-cpu/mojo_build_cache.tar.gz`.
- `/kaggle/working`=OUTPUT. CLI menggantung → `KaggleApi()`+unset proxy/PATH miniconda. `COMPLETE`≠sukses. `mojo run --target-accelerator=sm_75`; butuh `libKGENCompilerRTShared.so`.
- **Jebakan disk lokal (2026-09-30)**: `kaggle kernels output` mengunduh SEMUA `/kaggle/working` termasuk env pixi Linux ~618 MB (`mojoenv`, sampah di macOS) dan dump besar (act_*.bin ~833 MB). Tar source (`push_cpu_build.sh` memaketkan seluruh repo tanpa exclude folder output) membengkak ~741 MB. Disk internal sering <1 GiB bebas → tar WAJIB lewat `BONSAI_PKG_DIR` ke SSD eksternal. Baris 92 `push_cpu_build.sh` kini `PKG_DIR="${BONSAI_PKG_DIR:-/tmp/mojo_kaggle_pkg}"` (diubah DENGAN IZIN user 2026-09-30; default tak berubah).

## Rencana akurasi 2-bit (2026-09-26)
- **Guru** `okiabrian/qwen38-27b-gsq-rco-iq3s` (`Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` 11.771.546.784 B) masuk sebagai **BOBOT**, bukan model dijalankan. Rantai & rincian tiap tahap di log harian.
- **MLX 2-bit di T4 wajib build sumber** + `MLX_CUDA_ARCHITECTURES=75` (wheel resmi hanya sm_100a+). Mojo hanya inferensi. Kernel skrip: hanya `code_file` ke /kaggle/src → pendamping WAJIB base64/dataset.
- Terbuka: repack ternary lossless (signs, `mode: affine`); 8,6 GB ≈ 2,53 bpw.
- **Nama tensor guru TERVERIFIKASI LITERAL** (kernel v3 `okiabrian/teacher-gguf-inspect`, MATCH): GGUF arch qwen35 → `blk.%d.<tensor>.<weight|bias>`. 16 layer attn penuh (indeks 3,7,…,63) punya kunci `attn_q/attn_k/attn_v/attn_output` TERPISAH; 48 layer GDN (0..62) punya `attn_qkv`+`attn_gate`+`ssm_out`; global `token_embd`/`output_norm`/`output`. Semua sufiks `DEFAULT_TEACHER_SUFFIXES` ADA di berkas (0 hilang). `gsq_driver.py` 27/27, `gsq_ternary.py` 19/19. Catatan: `ssm_a` di layer GDN punya nama tanpa akhiran `.weight` (kunci `''`) — tak dipakai driver, tak berdampak.
