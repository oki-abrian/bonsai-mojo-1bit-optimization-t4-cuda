# Hasil Tes Pemisah — Bonsai-27B 1-bit (Kaggle T4)

Tanggal: 2026-09-12 · Kernel: `okiabrian/bonsai-infer-susah` · Sampling T=0.70, rep 1.10, window 256, seed 1234

## Tujuan

Memisahkan dua hipotesis yang sebelumnya masih bercampur:

- **A.** "penalaran rusak" — model tidak sanggup menalar beruntun
- **B.** "generasi panjang rusak" — jalur decode memburuk seiring posisi token

## Desain

Tiga prompt dijalankan berurutan dalam satu kernel. Tiap prompt di proses `bonsai_infer`
terpisah, jadi cache KV selalu mulai kosong (tidak ada kontaminasi antar prompt).

| # | Prompt | Budget | Yang diuji |
|---|---|---|---|
| 1 | 5 nama buah tropis | 96 | sanity: bisakah koheren sama sekali? |
| 2 | cerita pendek 250 kata | 512 | output panjang **tanpa** penalaran |
| 3 | soal kolam (pipa A/B) | 512 | output panjang **dengan** penalaran |

## Hasil kuantitatif

| Run | Ter-generate | ID unik | Rasio | Siklus mulai @ | `</think>` @ | TOP2 di batas prefill |
|---|---|---|---|---|---|---|
| #1 fakta | 96 | 72 | **0,75** | tidak ada | — | 8160 (16,64) \| 77264 (15,0) |
| #2 cerita | 512 | 144 | **0,28** | token 75 | — | 8160 (16,66) \| 77264 (14,76) |
| #3 penalaran | 512 | 108 | **0,21** | token 207 | token 168 | 8160 (19,31) \| 1596 (16,33) |

Rasio id unik per jendela 50 token (1,00 = tidak ada pengulangan):

```
RUN 1: 0.88 0.83
RUN 2: 0.84 0.76 0.58 0.64 0.58 0.66 0.48 0.40 0.38 0.42
RUN 3: 0.74 0.76 0.72 0.54 0.42 0.22 0.20 0.20 0.10 0.16
```

## Kesimpulan

**Hipotesis B yang benar — degenerasi dikendalikan oleh PANJANG output, bukan penalaran.**

- Run #1 (budget 96 token) tetap sehat sepanjang jendelanya: 0,83–0,88. Tidak ada siklus.
- Run #2 adalah **cerita** — nol kebutuhan penalaran — tetapi tetap runtuh, rasio turun
  bertahap 0,84 → 0,42 dan berakhir di pola `*Mata*: "Pata"` / `"Hata"` / `"Rata"`.
- Run #3 (penalaran) runtuh lebih tajam (0,74 → 0,10) tetapi **polanya sama**.

Jadi "penalaran" bukan tersangka khusus. Yang runtuh adalah **jalur decode saat posisi
memanjang**. Jendela koheren hanya sekitar **50–100 token pertama**.

**Prefill sehat, decode yang rusak.** Di batas prefill, ketiga run memilih token yang sama
dan wajar (`8160` = "Here", logit 16,6–19,3). Keputusan token pertama stabil dan benar;
kerusakan menumpuk setelahnya.

Gejala khas yang muncul di ketiga run:
- Pembuka template kaku `Here's thinking process:` — model masuk mode "rencana" dulu,
  bukan menjawab.
- Prompt berbahasa Indonesia dijawab dalam **bahasa Inggris**.
- Setelah `</think>`, output jatuh ke pengulangan angka (`1. 2. 3. 5.` … `10. 10. 10.`).
- **Tidak ada EOS** di ketiga run (`<|im_end|>` = 248046 tidak pernah muncul) — konsisten
  dengan `main.mojo` yang memang tidak punya logika berhenti di EOS.

## Temuan sampingan

1. **Mode thinking tidak bisa dimatikan dari jalur ini.** `chat_template.jinja` dataset
   memakai variabel `enable_thinking`. Kami memanggil
   `apply_chat_template(..., add_generation_prompt=True)` **tanpa** argumen itu, sehingga
   template masuk cabang `else` dan menambahkan `<think>\n`. Akibatnya run #1 (budget 96)
   menghabiskan seluruh anggaran untuk "analisis" dan **tidak pernah menyebut buahnya**.
   Untuk jawaban langsung perlu `enable_thinking=False`.
2. `tokenizer_config.json` dataset **tidak punya** kunci `chat_template`; template ada di
   berkas terpisah `chat_template.jinja` (7.764 byte). Ini penting: kalau berkas itu hilang,
   `apply_chat_template` akan gagal total, bukan sekadar salah format.
3. Tidak ada `chat_template.jinja` di salinan lokal `fix/kaggle_bonsai_model/` — salinan itu
   tidak akan bisa memakai `apply_chat_template`.

## Berkas

- Log mentah: `dist_kaggle_mojo/infer_susah/infer_susah.log`
- Log JSON (untuk detok): `dist_kaggle_mojo/infer_susah/infer_susah_log.json`
- Baca ulang kapan saja:
  `python detok_infer.py --log <log.json> --raw`
  `python detok_infer.py --log <log.json> --run 2` (satu run saja)

## Langkah berikut yang disarankan

1. **Uji batas panjang.** Jalankan prompt yang sama pada budget 64 / 128 / 192 / 256 token
   untuk memastikan di mana tepatnya kurva mulai turun. Ini memisahkan "rusak di posisi N"
   dari "rusak setelah token ke-N".
2. **Uji jalur kernel.** Bandingkan logits decode di posisi panjang dengan `BONSAI_NO_FUSE=1`
   dan `BONSAI_PREFILL_PER_TOKEN=1` — kalau hasilnya berbeda, akumulasi galat ada di kernel
   QMV fusi, bukan di bobot 1-bit.
3. **Tambahkan berhenti di EOS** di `main.mojo` supaya output tidak lanjut ke giliran palsu.
4. **Pakai `enable_thinking=False`** untuk prompt yang butuh jawaban langsung (fakta, daftar).
