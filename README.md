# Bonsai 1-Bit (W1A16 g128) Quantized Matmul & Decode for NVIDIA T4 in Mojo

Implementasi berkinerja tinggi kernel kuantisasi **1-bit biner (W1A16, group_size 128, Affine Bonsai)** yang dioptimalkan secara spesifik untuk GPU **NVIDIA T4 (arsitektur Turing, Compute Capability `sm_75`)** menggunakan bahasa pemrograman **Mojo**.

Repositori ini di-porting langsung dari optimasi low-level CUDA pada arsitektur hybrid model **Bonsai-27B** (*Gated Delta Net + SwiGLU MLP*), mempertahankan seluruh kompleksitas teknik eliminasi *bank conflict*, *branchless bit twiddling*, dan *vectorized memory coalescing*.

---

## Fitur & Optimasi Khusus NVIDIA T4 (`sm_75`)

### 1. Jalur Decode Token ($M = 1$ dan Small Batch $M \le 8$)
* **Vectorized Memory Coalescing**:
  * Pemuatan bobot tervektorisasi 32-bit (`uint32` = 32 bobot/lane) hingga 128-bit (`uint4` = 16 byte = 128 bobot = 1 grup penuh $g128$ per lane), memotong transaksi memori ke batas fisik bus GDDR6 T4.
* **Eliminasi LDS Bank Conflict SMEM**:
  * Array shared memory aktivasi diberi *padding stride* $128 \to 132$ float (`132 % 32 == 4`), menjamin 8 kolom grup pada satu warp mengakses 8 bank LDS berbeda secara simultan (zero bank conflicts).
* **Akumulator Register-M**:
  * Akumulasi tetap berada di register FP32 sepanjang loop $K$, sehingga bobot hanya dibaca **1 kali** dari DRAM per token.
* **Warp Reduction Deterministik**:
  * Reduksi intra-warp kooperatif tanpa menggunakan `atomicAdd`, menjamin **100% determinisme bitwise**.

### 2. Jalur Prefill ($M > 8$ hingga Batch Besar)
* **Tiling 2D CTA ($BM=64, BN=32, BK=64$) dengan Shared Memory 8 KiB**:
  * Menjaga ukuran shared memory tetap hemat (8 KiB dari batas 64 KiB per SM pada T4) sehingga T4 dapat mempertahankan okupansi penuh ($\ge 6$ blok/SM pada regfile 64K).
* **Eliminasi Bank Conflict `PAD = 8`**:
  * Padding `PAD = 8` pada matriks $A$ di shared memory untuk mencegah *bank conflicts* saat pembacaan sub-tile.
* **Dekuantisasi Branchless (Zero-Branch / Bit-Flip)**:
  * Tidak menggunakan percabangan `if/else` per-bit (yang memicu divergens warp).
  * Menggunakan trik bitwise langsung: $w_{\text{eff}} = s \cdot (2 \cdot \text{bit} - 1)$ serta manipulasi bit tanda IEEE FP16 (`0xBC00` XOR `(bit << 15)`).
* **Invarian Barrier Seragam (*Uniform Barrier*)**:
  * Seluruh thread CTA wajib memanggil barrier bersama-sama bahkan jika sel $(m, n)$ berada di luar batas matriks (*ragged boundary*), mencegah deadlock warp saat menangani panjang konteks ganjil.

---

## Kontrak Matematika Kuantisasi

* **Bobot Terkuantisasi ($w$)**: Tipe `uint8`, shape $[N, K/8]$. Tiap byte menyimpan 8 bobot biner dengan urutan **LSB-first**:
  $$\text{bit}_i = (\text{byte} \gg (k \pmod 8)) \ \& \ 1$$
* **Skala ($s$)**: Tipe `Float16` / `Float32`, shape $[N, (K + 127) / 128]$. Group size $G = 128$.
* **Kontrak Affine Bonsai ($b = -s$)**:
  $$w_{\text{eff}} = (2 \cdot \text{bit} - 1) \cdot s = \begin{cases} +s, & \text{jika bit} = 1 \\ -s, & \text{jika bit} = 0 \end{cases}$$
  Bias $-s$ terserap sempurna ke dalam perkalian tanpa overhead alokasi memori bias terpisah maupun operasi FMA tambahan.

---

## Struktur Repositori

```text
bonsai-1bit-t4-mojo/
├── mojoproject.toml              # Konfigurasi package & dependencies Mojo (Modular/Magic)
├── README.md                     # Dokumentasi teknis lengkap
├── src/
│   ├── __init__.mojo             # Ekspor publik package
│   ├── common.mojo               # Konstanta arsitektur hardware, geometri tile, & stride
│   ├── dequant.mojo              # Ekstraksi bit LSB-first & branchless sign-flip
│   ├── kernels/
│   │   ├── __init__.mojo
│   │   ├── prefill_sm75.mojo     # Prefill GPU kernel (BM=64, BN=32, BK=64)
│   │   └── decode_sm75.mojo      # Decode GPU kernel (Vectorized Coalesced GEMV)
│   └── ops.mojo                  # Host dispatcher & intelligent router (M<=8 vs M>8)
├── tests/
│   ├── selftest_sm75.mojo        # Uji mandiri komprehensif 43 kasus ekstrem + determinisme
│   ├── test_bonsai_shapes.mojo   # Uji khusus layer Bonsai-27B (N=11008, K=4096, Fused N=22016)
│   └── verify_differential.py    # Skrip verifikasi silang Python + NumPy FP64 reference
└── benchmarks/
    ├── bench_t4.mojo             # Micro-benchmark throughput (TFLOPS & GB/s) di T4
    └── bench_bonsai_layer.py     # Profiling latensi layer Bonsai-27B
```

---

## Menjalankan Uji & Benchmark

### 1. Menjalankan 43 Kasus Uji Ekstrem
Uji ini membandingkan komputasi kernel vs referensi FP64 di CPU serta menguji determinisme bitwise (dua eksekusi identik):
```bash
magic run mojo run tests/selftest_sm75.mojo
```

Kasus uji mencakup:
* **Ekstrim K Mini**: $K=16, 32, 48, 64$
* **Ragged N**: $N=8, 16, 27, 33, 65, 100, 127$
* **Ragged M & Decode**: $M=1, 2, 7, 8, 11, 15, 25, 33, 65, 130$
* **Batched & Weight Broadcast**: $L=2, 3, 5, 8$ dengan `broadcast_w = true/false`
* **Pola Bit Ekstrem**: All Zeros (`0x00`), All Ones (`0xFF`), Checkerboard (`0xAA`)

### 2. Menjalankan Validasi Dimensi Layer Bonsai-27B
```bash
magic run mojo run tests/test_bonsai_shapes.mojo
```

### 3. Menjalankan Benchmark Throughput & Bandwidth
```bash
magic run mojo run benchmarks/bench_t4.mojo
```

---

## Lisensi
Apache-2.0 License.
