# Analisis Akibat: Skema KV Bertingkat (lebar dimensi 64 / 256)

Konteks: Bonsai-27B (Qwen3.5, `model_type: qwen3_5`), 64 layer hibrida 3:1 (48 GDN + 16 full attention), bobot 1-bit, target T4 (sm_75, 16 GB, ~320 GB/s), jalur Mojo + CUDA kustom.
Prinsip dokumen ini: **tidak menyederhanakan**. Setiap klaim diberi status: **[terukur]** (ada data), **[dihitung]** (turunan aritmetika), **[belum diuji]** (hipotesis).

---

## 0. Definisi yang dipakai dokumen ini

**"1 bit" = 64 dimensi. "4 bit" = 256 dimensi (penuh).** Rasio 256/64 = 4 → dari situ asal penamaan "1 vs 4".

**Rantainya:** **256 → Oja / SMEC → 64** → disimpan sebagai **64 dimensi pertama (fast)**, selalu dibaca → **informasi tambahan** (komplemen yang membuat 64 setara 256) disimpan di **192** → **64 + 192 = 256**.

Jadi "64" **bukan** potongan 64 dimensi pertama dari vektor 256 mentah, melainkan **hasil kompresi**. Dan "192" bukan sisa yang terbuang — ia **informasi tambahan** yang dibaca hanya pada mode "4 bit". Ini **bukan** presisi bit per nilai dan **bukan** kuantisasi. (Lihat catatan Bab 1 di `OjaKV_Adaptive_Token_Selection.md`.)

**Konsekuensi langsung:** analisis lama yang berbasis format kuantisasi — SNR 1-bit ±6 dB, 216 B/864 B per token/layer, biaya dequant 22–37 GOP/step — **tidak berlaku dan dibuang**. Tidak ada unpack bit, tidak ada skala, tidak ada dequant di skema ini.

---

## 1. Peta risiko (urut prioritas)

| # | Risiko | Status | Akibat bila salah |
|---|---|---|---|
| **R1** | Apakah 64 dimensi hasil gating SMEC cukup mewakili 256 untuk token "1 bit" | **[belum diuji]** | ±90% konteks kehilangan informasi → kualitas runtuh, dan tidak ada jalur pemulihan |
| **R2** | Urutan UnRoPE ↔ seleksi tidak dinyatakan dokumen → belum jelas apakah seleksi punya sinyal posisi | **[belum ditetapkan]** | Seleksi bias, atau tidak bisa membedakan token identik di posisi berbeda |
| **R3** | Menggabungkan skor 64-dim dan 256-dim dalam satu softmax | **[belum diputuskan]** | Skala salah → softmax didominasi satu lebar |
| R4 | Kernel attention harus menangani dua lebar | **[dihitung]** | Regresi di jalur yang sudah lolos uji T4 (FIX-1..FIX-9) |
| R5 | Prefill 100k–300k tetap puluhan menit | **[terukur+ekstrapolasi]** | "Memproses 300k" tetap tidak praktis |
| R6 | Kriteria jangkar: §6.1 memakai reconstruction error, yang butuh $W$ | **[belum diputuskan]** | Himpunan jangkar berbeda → kualitas akhir berbeda |

---

## 2. Akibat pada kualitas

### 2.1 64 dimensi sebagai pengganti 256 — risiko utama skema

- Bab 9 mengklaim gating SMEC "menekan dimensi noise menjadi 0, sambil mendorong dimensi paling informatif ke indeks 1–64". **Klaim ini belum diuji pada Bonsai-27B** [belum diuji].
- Yang dipertaruhkan: ±90% token hanya menyumbang 64 dari 256 dimensinya. Bila gating gagal menempatkan informasi penting di 64 pertama, token itu efektif kehilangan informasi — karena pada mode "1 bit" **informasi tambahan di 192 tidak dibaca**. Satu-satunya jalur pemulihan adalah menaikkan token itu ke mode "4 bit" (membaca 64 + 192).
- Beda peran K vs V:
  - **K 64-dim** → error masuk ke $QK^\top$, menggeser **distribusi** attention, bukan sekadar besaran. Efeknya non-linear (softmax) dan bisa memindahkan massa probabilitas ke token yang salah. Ini lebih berbahaya [belum diuji].
  - **V 64-dim** → error masuk ke output attention; softmax-weighted averaging meredam sebagian, jadi toleransinya lebih tinggi [belum diuji].

### 2.2 Interaksi dengan bobot 1-bit

- Sudah ada bukti empiris kerapuhan margin: pada inferensi T4 model memilih token Tionghoa (`第一阶段`) alih-alih "tahap pertama" — margin logit menyempit akibat bobot 1,125 bpw [terukur, 2026-09-14].
- Memangkas lebar KV pada 90% token menambah tekanan pada margin yang sudah tipis. Efeknya tidak bisa diasumsikan aditif atau kecil [belum diuji].

### 2.3 UnRoPE bersifat sementara — posisi tidak boleh disimpulkan hilang

- `UnRoPE` **bukan** pembuangan informasi posisi. Namanya menyiratkan operasi yang bisa dibalik, dan §9.2 langkah 5 memang menyatakan RoPE **diterapkan kembali** sebelum masuk FlashAttention.
- Yang **tidak dinyatakan** dokumen: apakah RoPE dipasang kembali **sebelum** seleksi (langkah 4) atau sesudahnya. Keduanya mungkin [belum ditetapkan].
- Karena itu klaim "seleksi kehilangan sinyal posisi" **bukan kesimpulan yang bisa diambil** — itu asumsi saya. Yang benar: **urutan UnRoPE ↔ seleksi harus ditentukan lebih dulu**; baru setelah itu risiko posisi bisa dinilai [belum ditetapkan].

### 2.4 Seleksi dihitung sekali, dipakai 16 layer

- Rencana amortisasi gaya CSA2: klasifikasi sekali, dipakai ulang untuk seluruh 16 layer attention.
- **Akibat yang harus disadari:** bila seleksinya salah, kesalahannya **konsisten di 16 layer** — tidak ada layer lain yang mengoreksi [belum diuji].
- Trade-off: hemat 15/16 biaya seleksi vs hilangnya redundansi koreksi. Alternatif: hitung di 2–4 layer berbeda lalu gabungkan (voting).

### 2.5 Kriteria jangkar masih terbuka

- §6.1 memilih jangkar dari **reconstruction error tertinggi**, yang didefinisikan lewat proyeksi $W$ (Bab 3).
- Bab 9 menggantikan $W$ dengan SMEC, jadi kriteria itu belum punya pengganti yang ditetapkan [belum diputuskan].

---

## 3. Akibat pada komputasi

### 3.1 Tidak ada biaya dequant — beban terbesar versi lama hilang

- Karena "1 bit / 4 bit" adalah **lebar dimensi**, bukan format bit, maka **tidak ada unpack, tidak ada skala, tidak ada dequant**. Biaya dequant ±22–37 GOP/step @300k pada analisis lama **hilang seluruhnya**.
- Yang tersisa sebagai beban baru: attention harus menjalankan **dua lebar** (64 dan 256) dalam satu pass, lalu menggabungkan skornya.

### 3.2 Perbandingan beban baca (satuan: dimensi)

- Rata-rata per token per layer (bobot jangkar 1% / "4 bit" 10% / "1 bit" 89%):
  `0,01×256 + 0,10×256 + 0,89×64 ≈ 85` dimensi vs 256 → **±67% lebih sedikit** [dihitung].
- Konsisten dengan klaim §8 dokumen OjaKV (">60%").

### 3.3 QR dan state $W$ — tidak ada lagi

- Bila desain final memakai SMEC (Bab 9), tidak ada $W$ daring, tidak ada Oja's rule, tidak ada QR.
- Maka biaya QR ±315 GFLOP @300k (±1 menit) dan masalah inkonsistensi cache **hilang** [dihitung, kini moot].

---

## 4. Akibat pada pola akses memori

- Base `[seq, 64]` dibaca untuk **semua** token → **kontigu**, efisiensi bandwidth mendekati puncak.
- Residual `[seq, 192]` hanya dibaca untuk ±11% token (jangkar + "4 bit") → ini **gather parsial**, tidak kontigu. Porsinya kecil, tapi pola aksesnya bukan streaming murni.
- **Trade-off yang tak terhindarkan:** penghematan murni berasal dari **lebar**, bukan dari seleksi. Menaikkan top-K (memindahkan token dari 64 ke 256) langsung menaikkan beban baca; tidak ada katup lain yang bisa menutupnya.

---

## 5. Akibat arsitektural

### 5.1 Prefill bertahap (chunked) wajib di 300k

- Aktivasi 300k token dalam satu chunk tidak realistis. Cache di-commit bertahap → seleksi dan pemilihan jangkar harus bisa dihitung dari **data parsial**, atau ditunda sampai prefill selesai (dua lintasan, biaya O(N) satu kali).

### 5.2 GDN tidak tersentuh

- 48 layer GDN: state 48 × 128 × 128 × 4 B ≈ **144 MiB, konstan** [dihitung] — tidak tumbuh dengan konteks, jadi bukan masalah kapasitas. Tetapi skema ini **tidak memberi manfaat apa pun pada 75% layer**. Bila kelak ingin menghemat lebih jauh, sasarannya adalah state GDN, dan itu masalah yang berbeda.

---

## 6. Akibat pada kode & verifikasi

### 6.1 Kernel yang harus diubah adalah kernel yang paling rapuh

- `gqa_attention_sm75_gpu` (SMEM 320 float, reduksi stage-2 `lane < 8`, hanya benar untuk `head_dim == 256`) — FIX-8 menambahkan `raise` justru karena kekakuan itu.
- Menambahkan **lebar 64 di samping 256** di kernel yang sama = kompleksitas baru di titik paling sensitif → **risiko regresi** pada jalur yang sudah lolos uji T4.
- Alternatif yang lebih aman: dua jalur terpisah (attention 64-dim untuk seleksi + tingkat "1 bit", attention 256-dim utuh untuk "4 bit" dan jangkar), lalu gabung skornya — dengan biaya penggabungan softmax lintas jalur.

### 6.2 Berkas yang akan terdampak

`main.mojo` (max_seq, alur cache), `src/models/qwen3_5/attention.mojo` (AttentionKVCache), `src/ops.mojo` (launcher), `src/kernels/elementwise_sm75.mojo` (kernel GQA).
`SMECAdapter` sendiri **di luar** jalur inferensi Mojo — ia dilatih offline.

### 6.3 Verifikasi hanya lewat Kaggle

- Toolchain Mojo lokal rusak (`unable to locate module 'stdlib'`), jadi setiap verifikasi = 1 run Kaggle ±11 menit + kuota.
- **Akibat:** eksperimen harus dirancang untuk **sedikit iterasi** — satu run menguji beberapa konfigurasi sekaligus (batch top-K/threshold), bukan satu konfigurasi per run.

---

## 7. Batas yang tidak bisa dihindari

1. **Lantai = 64 dimensi per token per layer.** Tidak bisa dikurangi tanpa membuang token sama sekali.
2. **Plafon overhead ±51,5 ms/step** [terukur: 60,66 ms − 8,75 bobot − 0,36 KV @1.820 token]. Bahkan bila KV gratis dibaca → ±16,6 tok/s di 100k.
3. **Prefill 100k–300k tetap puluhan menit** (GDN ±7,5 ms/token → ±12,5 menit @100k, ±37 menit @300k) [terukur+ekstrapolasi]. Skema KV tidak menyentuhnya.
4. **`max_seq` masih 4.096**, dan kemampuan konteks panjang model belum diverifikasi dari checkpoint.
5. **SMEC butuh fine-tuning offline.** Ini tidak bisa ditambahkan saat inferensi. Tanpa training, 64 dimensi pertama tidak punya jaminan informatif — dan kekhawatiran §3.1 ("membuang 192 dimensi terakhir itu buruk") kembali berlaku penuh.

---

## 8. Eksperimen penentu — sebelum menulis kode produksi

| # | Eksperimen | Kriteria lulus | Biaya |
|---|---|---|---|
| **E1** | Ukur kualitas saat ±90% token hanya memakai 64 dimensi: PPL + tugas retrieval konteks panjang | Penurunan dapat diterima (mis. PPL < 2%) | Offline |
| **E2** | Presisi seleksi: top-10% dari 64 dim hasil SMEC vs top-10% attention penuh; ukur juga dua urutan — RoPE dipasang **sebelum** vs **sesudah** seleksi | Overlap > 90%, dan salah satu urutan menang signifikan | Offline |
| **E3** | Mikrobenchmark bandwidth T4: base 64 kontigu (100% token) + residual 192 gather (11% token) | ≥ 80% bandwidth puncak | 1 run Kaggle |
| **E4** | Uji normalisasi softmax lintas lebar (64 vs 256): apakah $\sqrt{d}$ per-lebar cukup, atau perlu kalibrasi | Distribusi attention tidak didominasi satu lebar | Offline + 1 run |

Urutan: **E1 dan E2 lebih dulu** — keduanya offline, tidak menyentuh jalur produksi, dan keduanya bisa mematikan atau mengubah desain.

---

## 9. Yang sudah terjawab (tidak perlu diputuskan lagi)

- **Format "1-bit"/"4-bit"** → lebar dimensi **64 / 256**, bukan presisi bit.
- **Bagaimana 64 dihasilkan?** → **dua jalur**: Oja (proyeksi $W$, daring) atau SMEC (gating, *offline*). Belum ditetapkan yang mana; keduanya bermuara pada 64 dimensi pertama yang sama. Jadi NBN/ECR (Bab 7, berbasis proyeksi) tetap relevan bila jalur Oja yang dipakai.
- **State $W$, Oja's rule, QR, inkonsistensi cache** → tidak ada pada desain final.
- **Biaya dequant 1-bit/4-bit** → tidak ada.
- **Risiko posisi pada seleksi** → **belum bisa dinilai**: UnRoPE sementara dan reversibel (§9.2 langkah 5 memasang RoPE kembali); urutan UnRoPE ↔ seleksi belum dinyatakan. Lihat §2.3.
