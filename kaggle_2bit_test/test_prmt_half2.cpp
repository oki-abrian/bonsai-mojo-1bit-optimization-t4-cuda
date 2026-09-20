// Uji host untuk rancangan ekspansi 2-bit -> half2 di kernel GEMV decode.
// ==============================================================================
// Tujuan: memastikan rancangan PRMT yang dipakai untuk menggantikan loop
// skalar  `q = (word >> 2h) & 3; local += float(q-1) * x[h]`  benar-benar
// menghasilkan nilai FP16 yang sama, SEBELUM kode CUDA ditulis (kompilasi
// CUDA hanya bisa dilakukan di Kaggle, jadi salah di sini = buang satu siklus
// build).
//
// `__byte_perm(a, b, sel)` diemulasi persis sesuai konvensi yang sudah
// terbukti di kernel fp16/int8 proyek ini:
//   * byte sumber 0..3 = argumen `a`, byte 4..7 = argumen `b`
//     (byte 0 = LSB);
//   * nibble ke-i dari `sel` memilih indeks byte sumber untuk byte hasil ke-i:
//     out.byte[i] = src[(sel >> 4i) & 0xF].
// ==============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <random>

static inline uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t sel) {
    uint8_t src[8];
    memcpy(src + 0, &a, 4);
    memcpy(src + 4, &b, 4);
    uint32_t out = 0;
    uint8_t o[4];
    for (int i = 0; i < 4; ++i) o[i] = src[(sel >> (4 * i)) & 0xF];
    memcpy(&out, o, 4);
    return out;
}

// Referensi: konversi kode 2-bit q -> FP16 (q-1) untuk q ∈ {0,1,2,3}.
// Konversi float->FP16 yang BENAR (bukan hardcoded): q=3 menghasilkan +2.0
// = 0x4000, bukan 0x3C00 — konversi hardcoded pernah membuat uji ini gagal
// untuk kode yang sebenarnya tidak terpakai.
static inline uint16_t ref_half(uint32_t q) {
    float v = (float)(int)q - 1.0f;               // -1, 0, +1, +2
    uint32_t bits; memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xFFu);
    uint32_t man  = bits & 0x7FFFFFu;
    if (exp == 0 && man == 0) return (uint16_t)sign;      // +/-0
    int32_t e = exp - 127 + 15;                            // bias FP16
    // Nilai yang diuji hanya -1, 0, +1, +2 — semuanya normal di FP16, jadi
    // cabang subnormal/inf tidak perlu diimplementasi (dan pernah ditulis
    // dengan pergeseran yang salah).
    if (e >= 31 || e <= 0) {
        printf("ref_half: nilai di luar jangkauan uji (v=%g)\n", v);
        return 0xFFFFu;
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}

static inline float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    if (exp == 0) {
        if (man == 0) { uint32_t z = sign; float f; memcpy(&f, &z, 4); return f; }
        // subnormal
        float f = std::ldexp((float)man, -24);
        return sign ? -f : f;
    }
    uint32_t bits = sign | ((exp + 112u) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// RANCANGAN yang diuji: satu word U32 (16 kode, lane i di bit 2i)
// -> 8 buah half2 (masing-masing 32-bit: [0x00, H(q_even), 0x00, H(q_odd)]).
// TAB_H = byte tinggi dari half(-1), half(0), half(+1), half(+2)
//       = 0xBC, 0x00, 0x3C, 0x40  -> diindeks langsung oleh q.
static const uint32_t TAB_H = 0x40003C00u;   // byte0=0x00, byte1=0x3C? -- lihat catatan
// Hati-hati urutan byte: U32 little-endian, byte0 = LSB.
// Kita ingin TAB_H.byte[q] = H(q):
//   H(0)=0xBC, H(1)=0x00, H(2)=0x3C, H(3)=0x40
//   -> byte0=0xBC, byte1=0x00, byte2=0x3C, byte3=0x40
//   -> U32 = 0x40'3C'00'BC = 0x403C00BCu
static const uint32_t TAB_H_REAL = 0x403C00BCu;

static void expand_word_h2(uint32_t word, uint32_t out[8]) {
    const uint32_t pe0 = byte_perm(TAB_H_REAL, TAB_H_REAL,  word        & 0x33333333u);
    const uint32_t po0 = byte_perm(TAB_H_REAL, TAB_H_REAL, (word >>  2) & 0x33333333u);
    const uint32_t pe1 = byte_perm(TAB_H_REAL, TAB_H_REAL, (word >> 16) & 0x33333333u);
    const uint32_t po1 = byte_perm(TAB_H_REAL, TAB_H_REAL, (word >> 18) & 0x33333333u);
    const uint32_t D0 = byte_perm(pe0, po0, 0x5140u);   // H(q0),H(q1),H(q2),H(q3)
    const uint32_t D1 = byte_perm(pe0, po0, 0x7362u);   // H(q4),H(q5),H(q6),H(q7)
    const uint32_t D2 = byte_perm(pe1, po1, 0x5140u);   // H(q8)..H(q11)
    const uint32_t D3 = byte_perm(pe1, po1, 0x7362u);   // H(q12)..H(q15)
    const uint32_t Z  = 0u;
    out[0] = byte_perm(D0, Z, 0x1404u);   // {h(q0), h(q1)}
    out[1] = byte_perm(D0, Z, 0x3424u);   // {h(q2), h(q3)}
    out[2] = byte_perm(D1, Z, 0x1404u);   // {h(q4), h(q5)}
    out[3] = byte_perm(D1, Z, 0x3424u);   // {h(q6), h(q7)}
    out[4] = byte_perm(D2, Z, 0x1404u);   // {h(q8), h(q9)}
    out[5] = byte_perm(D2, Z, 0x3424u);   // {h(q10), h(q11)}
    out[6] = byte_perm(D3, Z, 0x1404u);   // {h(q12), h(q13)}
    out[7] = byte_perm(D3, Z, 0x3424u);   // {h(q14), h(q15)}
}

int main() {
    // 1. Uji TAB_H terhadap referensi
    for (uint32_t q = 0; q < 4; ++q) {
        uint32_t tabbytes[1]; memcpy(tabbytes, &TAB_H_REAL, 4);
        uint8_t b = ((const uint8_t*)tabbytes)[q];
        uint16_t ref = ref_half(q);
        uint8_t ref_hi = (uint8_t)(ref >> 8);
        if (b != ref_hi) {
            printf("GAGAL TAB_H: q=%u byte=0x%02X ref_hi=0x%02X\n", q, b, ref_hi);
            return 1;
        }
    }
    printf("[OK] TAB_H = 0x403C00BC -> H(q) cocok dengan byte tinggi half(q-1)\n");

    // 2. Uji acak: 20000 word, 16 kode per word
    std::mt19937 rng(12345);
    int bad = 0;
    for (int t = 0; t < 20000; ++t) {
        uint32_t word = rng();
        // q=3 tidak terpakai di checkpoint, tapi tetap diuji kelengkapannya.
        if (t % 2 == 0) {
            // paksa hanya kode 0..2 supaya sesuai kontrak bobot nyata
            uint32_t w = 0;
            for (int i = 0; i < 16; ++i) w |= (rng() % 3u) << (2 * i);
            word = w;
        }
        uint32_t out[8];
        expand_word_h2(word, out);
        for (int i = 0; i < 8; ++i) {
            uint8_t o[4]; memcpy(o, &out[i], 4);
            uint16_t h_lo = (uint16_t)(o[0] | (o[1] << 8));   // half genap
            uint16_t h_hi = (uint16_t)(o[2] | (o[3] << 8));   // half ganjil
            uint32_t q_lo = (word >> (2 * (2 * i))) & 3u;
            uint32_t q_hi = (word >> (2 * (2 * i + 1))) & 3u;
            uint16_t r_lo = ref_half(q_lo);
            uint16_t r_hi = ref_half(q_hi);
            if (h_lo != r_lo || h_hi != r_hi) {
                if (bad < 5) {
                    printf("GAGAL word=0x%08X out[%d]: lo=0x%04X(ref 0x%04X) "
                           "hi=0x%04X(ref 0x%04X)\n", word, i, h_lo, r_lo, h_hi, r_hi);
                }
                ++bad;
            }
        }
    }
    if (bad) { printf("[GAGAL] %d ketidakcocokan\n", bad); return 1; }
    printf("[OK] 20000 word x 16 kode: 0 ketidakcocokan (nilai float: ");
    printf("%g %g %g)\n", half_to_float(ref_half(0)), half_to_float(ref_half(1)),
           half_to_float(ref_half(2)));

    // 3. Uji khusus: word=0 (semua q=0 -> semua -1)
    uint32_t out[8]; expand_word_h2(0u, out);
    for (int i = 0; i < 8; ++i) {
        uint8_t o[4]; memcpy(o, &out[i], 4);
        uint16_t h_lo = (uint16_t)(o[0] | (o[1] << 8));
        uint16_t h_hi = (uint16_t)(o[2] | (o[3] << 8));
        if (h_lo != 0xBC00u || h_hi != 0xBC00u) {
            printf("[GAGAL] word=0 out[%d] = %04X %04X (harap BC00 BC00)\n",
                   i, h_lo, h_hi);
            return 1;
        }
    }
    printf("[OK] word=0 -> 16 bobot -1.0 (0xBC00), bukan 0: q=0 itu kode SAH\n");
    printf("SEMUA UJI LULUS\n");
    return 0;
}
