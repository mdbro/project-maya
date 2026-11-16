// src/mm3/mm3_native_cpu_fast.cpp - the CPU lane's reworked native NVFP4 expert kernel (mm3_native_cpu_fast.hpp).
// Built without -ffast-math and with C++20's ISO mode (no FMA contraction): the scalar epilogues are the shared kernel's
// expressions, so the INT8 path's bits are its bits.
#include "mm3_native_cpu_fast.hpp"

#include <immintrin.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
#define MM3_CPU_FAST_TARGET __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx512dq")))
#else
#define MM3_CPU_FAST_TARGET
#endif

namespace mm3::cpu_fast {
namespace {

constexpr int kUnit = 144;   // an RG16 unit: 16 rows x 16 inputs, 128 code bytes then 16 E4M3 scale bytes

float bf16_float(uint16_t x) { return std::bit_cast<float>(uint32_t(x) << 16); }
// The shared kernel's float_bf16 (fast_math.h): round to nearest even, NaN quieted.
uint16_t float_bf16(float f) {
    const uint32_t u = std::bit_cast<uint32_t>(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) return uint16_t((u >> 16) | 0x40U);
    return uint16_t((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}

// canonical_math.h's E4M3 / E2M1 helpers (the 5090's activation rule), on the host.
int32_t e4m3_scaled(unsigned word) {
    const unsigned e = (word >> 3) & 15U, m = word & 7U;
    return e == 0 ? int32_t(m) : int32_t((8U + m) << (e - 1U));
}
float e4m3_value(unsigned word) { return float(e4m3_scaled(word)) * (1.0F / 512.0F); }
float rint_rn(float x) { const float magic = 12582912.0F; return (x + magic) - magic; }
uint8_t e4m3_rn_satfinite(float x) {
    const uint32_t u = std::bit_cast<uint32_t>(x) & 0x7FFFFFFFU;
    if (u > 0x7F800000U) return 0x7F;
    if (u >= 0x43E00000U) return 0x7E;
    const int exp = int(u >> 23);
    if (exp < 127 - 6) return uint8_t(int(rint_rn(std::bit_cast<float>(u) * 512.0F)));
    const uint32_t mant = u & 0x7FFFFFU, rest = mant & 0xFFFFFU;
    uint32_t keep = mant >> 20;
    int e = exp - 127 + 7;
    if (rest > 0x80000U || (rest == 0x80000U && (keep & 1U) != 0)) {
        if (++keep == 8U) { keep = 0; ++e; }
    }
    if (e > 15 || (e == 15 && keep > 6U)) return 0x7E;
    return uint8_t((unsigned(e) << 3) | keep);
}
uint8_t e2m1_rn_satfinite(float x) {
    const uint32_t bits = std::bit_cast<uint32_t>(x);
    const unsigned sign = (bits >> 31) != 0 ? 8U : 0U;
    const float a = std::bit_cast<float>(bits & 0x7FFFFFFFU);
    unsigned code;
    if (a <= 0.25F) code = 0;
    else if (a < 0.75F) code = 1;
    else if (a <= 1.25F) code = 2;
    else if (a < 1.75F) code = 3;
    else if (a <= 2.5F) code = 4;
    else if (a < 3.5F) code = 5;
    else if (a <= 5.0F) code = 6;
    else code = 7;
    return uint8_t(sign | code);
}
int e2m1_x2(unsigned code) {
    const int mag = int((0x0C08060403020100ULL >> (8U * (code & 7U))) & 0xFFU);
    return (code & 8U) != 0 ? -mag : mag;
}

// ---- quantization, one block of 16 BF16 values

void quant_block_scalar(Act act, const uint16_t* x, QBlock& out) {
    std::memset(out.q, 0, 16);
    out.off = 0;
    float m = 0;
    bool invalid = false;
    for (int k = 0; k < 16; ++k) {
        const float v = bf16_float(x[k]);
        invalid |= !std::isfinite(v);
        m = std::max(m, std::fabs(v));
    }
    if (invalid) { out.scale = std::numeric_limits<float>::quiet_NaN(); return; }
    int sum = 0;
    if (act == kInt8) {   // (fast_cpu.cpp's quant)
        const float s = m / 127.f;
        out.scale = s;
        for (int k = 0; k < 16; ++k) {
            out.q[k] = s ? int8_t(std::clamp(int(std::nearbyint(bf16_float(x[k]) / s)), -127, 127)) : 0;
            sum += out.q[k];
        }
    } else {              // (canonical_math.h's quantize_a4_codes, input scale 1)
        const uint8_t word = e4m3_rn_satfinite(m / 6.0F);
        if (e4m3_scaled(word) == 0) { out.scale = 0; return; }
        const float d = e4m3_value(word);
        out.scale = d * 0.5F;
        for (int k = 0; k < 16; ++k) {
            out.q[k] = int8_t(e2m1_x2(e2m1_rn_satfinite(bf16_float(x[k]) / d)));
            sum += out.q[k];
        }
    }
    out.off = 12 * sum;
}

MM3_CPU_FAST_TARGET void quant_block(Act act, const uint16_t* x, QBlock& out) {
    const __m512i u = _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) x)), 16);
    const __m512i exp = _mm512_set1_epi32(0x7f800000);
    if (_mm512_cmpeq_epi32_mask(_mm512_and_si512(u, exp), exp)) {   // (an Inf or NaN in the block)
        std::memset(out.q, 0, 16);
        out.off = 0;
        out.scale = std::numeric_limits<float>::quiet_NaN();
        return;
    }
    const __m512 v = _mm512_castsi512_ps(u);
    const float m = _mm512_reduce_max_ps(_mm512_castsi512_ps(_mm512_and_si512(u, _mm512_set1_epi32(0x7fffffff))));
    __m512i q;
    if (act == kInt8) {
        const float s = m / 127.f;
        out.scale = s;
        if (!s) { std::memset(out.q, 0, 16); out.off = 0; return; }
        const __m512 r = _mm512_roundscale_ps(_mm512_div_ps(v, _mm512_set1_ps(s)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q = _mm512_max_epi32(_mm512_min_epi32(_mm512_cvtps_epi32(r), _mm512_set1_epi32(127)), _mm512_set1_epi32(-127));
    } else {
        const uint8_t word = e4m3_rn_satfinite(m / 6.0F);
        if (e4m3_scaled(word) == 0) { std::memset(out.q, 0, 16); out.off = 0; out.scale = 0; return; }
        const float d = e4m3_value(word);
        out.scale = d * 0.5F;
        const bool threshold = act == kNvfp4Threshold;
        const __m512 c = threshold ? v : _mm512_div_ps(v, _mm512_set1_ps(d));
        const __m512 a = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(c), _mm512_set1_epi32(0x7fffffff)));
        // E4M3 scales and all seven E2M1 midpoints are binary rationals with at most
        // seven significant bits. Their products are exact in float, including ties.
        const float t = threshold ? d : 1.0f;
        const __m512i one = _mm512_set1_epi32(1);
        __m512i idx = _mm512_setzero_si512();
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.25f * t), _CMP_GT_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(0.75f * t), _CMP_GE_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(1.25f * t), _CMP_GT_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(1.75f * t), _CMP_GE_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(2.5f * t), _CMP_GT_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(3.5f * t), _CMP_GE_OQ), idx, one);
        idx = _mm512_mask_add_epi32(idx, _mm512_cmp_ps_mask(a, _mm512_set1_ps(5.0f * t), _CMP_GT_OQ), idx, one);
        const __m512i table = _mm512_setr_epi32(0, 1, 2, 3, 4, 6, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0);
        const __m512i mag = _mm512_permutexvar_epi32(idx, table);
        const __mmask16 neg = _mm512_movepi32_mask(_mm512_castps_si512(c));   // (the sign bit, -0 included)
        q = _mm512_mask_sub_epi32(mag, neg, _mm512_setzero_si512(), mag);
    }
    _mm_storeu_si128((__m128i*) out.q, _mm512_cvtepi32_epi8(q));
    out.off = 12 * _mm512_reduce_add_epi32(q);
}

// ---- the weights' sums

// Twice the E2M1 value plus 12 for each code (the codes 8..15 are the negative values; 8 is -0).
MM3_CPU_FAST_TARGET inline __m512i weight_table() {
    return _mm512_broadcast_i32x4(_mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0));
}
// A unit's 16 E4M3 row scales times one half (for doubled codes), as the shared kernel's table holds them: subnormal
// m / 1024, normal ((e + 119) << 23 | m << 20), 0x7f NaN; the sign bit masked.
MM3_CPU_FAST_TARGET inline __m512 unit_scales(const uint8_t* s) {
    const __m512i x = _mm512_and_si512(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) s)), _mm512_set1_epi32(0x7f));
    const __m512 normal = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_slli_epi32(x, 20), _mm512_set1_epi32(119 << 23)));
    const __m512 subnormal = _mm512_mul_ps(_mm512_cvtepi32_ps(x), _mm512_set1_ps(1.f / 1024.f));
    __m512 r = _mm512_mask_blend_ps(_mm512_cmplt_epu32_mask(x, _mm512_set1_epi32(8)), normal, subnormal);
    return _mm512_mask_blend_ps(_mm512_cmpeq_epi32_mask(x, _mm512_set1_epi32(0x7f)), r,
                                _mm512_castsi512_ps(_mm512_set1_epi32(0x7fc00000)));
}

// Row group rg of a matrix (blocks = K / 16 units a row group), C columns: out[c][r] = row r's dot product with column c
// over all K inputs: per block the exact int32 sum of doubled weight code x activation code, then
// acc = fma(sum, weight scale x activation scale, acc) block by block - the shared kernel's order.
template <int C>
MM3_CPU_FAST_TARGET void rows(const uint8_t* matrix, int blocks, int rg, const QBlock* const* x, float out[8][16]) {
    const __m512i lut = weight_table(), nib = _mm512_set1_epi8(15);
    __m512 acc[C];
    for (int c = 0; c < C; ++c) acc[c] = _mm512_setzero_ps();
    const uint8_t* unit = matrix + size_t(rg) * blocks * kUnit;
    for (int b = 0; b < blocks; ++b, unit += kUnit) {
        const __m512 scale = unit_scales(unit + 128);
        // bytes 32q + 4(row % 8) + t hold input 4q + t of rows (row % 8) (low nibble) and 8 + row % 8 (high nibble): a
        // 64-byte load holds quarters 2j and 2j + 1; each quarter's 64 codes, rows 0-7 then 8-15, make one register whose
        // 32-bit lane r is row r's four inputs
        const __m512i v0 = _mm512_loadu_si512(unit), v1 = _mm512_loadu_si512(unit + 64);
        const __m512i lo0 = _mm512_and_si512(v0, nib), hi0 = _mm512_and_si512(_mm512_srli_epi16(v0, 4), nib);
        const __m512i lo1 = _mm512_and_si512(v1, nib), hi1 = _mm512_and_si512(_mm512_srli_epi16(v1, 4), nib);
        const __m512i w0 = _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo0, hi0, 0x44));
        const __m512i w1 = _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo0, hi0, 0xEE));
        const __m512i w2 = _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo1, hi1, 0x44));
        const __m512i w3 = _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo1, hi1, 0xEE));
        for (int c = 0; c < C; ++c) {
            const QBlock& a = x[c][b];
            int32_t q[4];
            std::memcpy(q, a.q, 16);
            __m512i s = _mm512_set1_epi32(-a.off);   // (sum (w + 12) a - 12 sum a = sum w a)
            s = _mm512_dpbusd_epi32(s, w0, _mm512_set1_epi32(q[0]));
            s = _mm512_dpbusd_epi32(s, w1, _mm512_set1_epi32(q[1]));
            s = _mm512_dpbusd_epi32(s, w2, _mm512_set1_epi32(q[2]));
            s = _mm512_dpbusd_epi32(s, w3, _mm512_set1_epi32(q[3]));
            acc[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(scale, _mm512_set1_ps(a.scale)), acc[c]);
        }
    }
    for (int c = 0; c < C; ++c) _mm512_storeu_ps(out[c], acc[c]);
}

void rows_any(const uint8_t* matrix, int blocks, int rg, const QBlock* const* x, int cols, float out[8][16]) {
    switch (cols) {
    case 1: rows<1>(matrix, blocks, rg, x, out); break;
    case 2: rows<2>(matrix, blocks, rg, x, out); break;
    case 3: rows<3>(matrix, blocks, rg, x, out); break;
    case 4: rows<4>(matrix, blocks, rg, x, out); break;
    case 5: rows<5>(matrix, blocks, rg, x, out); break;
    case 6: rows<6>(matrix, blocks, rg, x, out); break;
    case 7: rows<7>(matrix, blocks, rg, x, out); break;
    default: rows<8>(matrix, blocks, rg, x, out); break;
    }
}

// Candidate code deliberately does not replace/refactor rows(): execution0 is
// the frozen old loop, so an exact A/B can catch errors in these helpers too.
struct DecodedUnit { __m512i w0, w1, w2, w3; __m512 scale; };
MM3_CPU_FAST_TARGET inline DecodedUnit decode_unit(const uint8_t* unit) {
    const __m512i lut = weight_table(), nib = _mm512_set1_epi8(15);
    const __m512i v0 = _mm512_loadu_si512(unit), v1 = _mm512_loadu_si512(unit + 64);
    const __m512i lo0 = _mm512_and_si512(v0, nib), hi0 = _mm512_and_si512(_mm512_srli_epi16(v0, 4), nib);
    const __m512i lo1 = _mm512_and_si512(v1, nib), hi1 = _mm512_and_si512(_mm512_srli_epi16(v1, 4), nib);
    return {_mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo0, hi0, 0x44)),
            _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo0, hi0, 0xEE)),
            _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo1, hi1, 0x44)),
            _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lo1, hi1, 0xEE)), unit_scales(unit + 128)};
}
struct BroadcastBlock { __m512i off, q0, q1, q2, q3; __m512 scale; };
MM3_CPU_FAST_TARGET inline BroadcastBlock broadcast_block(const QBlock& a) {
    int32_t q[4];
    std::memcpy(q, a.q, 16);
    return {_mm512_set1_epi32(-a.off), _mm512_set1_epi32(q[0]), _mm512_set1_epi32(q[1]),
            _mm512_set1_epi32(q[2]), _mm512_set1_epi32(q[3]), _mm512_set1_ps(a.scale)};
}
MM3_CPU_FAST_TARGET inline __m512i dot_block(const DecodedUnit& w, const BroadcastBlock& a) {
    __m512i sum = _mm512_dpbusd_epi32(a.off, w.w0, a.q0);
    sum = _mm512_dpbusd_epi32(sum, w.w1, a.q1);
    sum = _mm512_dpbusd_epi32(sum, w.w2, a.q2);
    return _mm512_dpbusd_epi32(sum, w.w3, a.q3);
}

template <int C>
MM3_CPU_FAST_TARGET void rows_unrolled(const uint8_t* matrix, int blocks, int rg,
                                      const QBlock* const* x, float out[8][16]) {
    __m512 acc[C];
    for (int c = 0; c < C; ++c) acc[c] = _mm512_setzero_ps();
    const uint8_t* unit = matrix + size_t(rg) * blocks * kUnit;
    int b = 0;
    for (; b + 1 < blocks; b += 2, unit += 2 * kUnit) {
        const DecodedUnit w0 = decode_unit(unit), w1 = decode_unit(unit + kUnit);
        for (int c = 0; c < C; ++c) {
            const BroadcastBlock a0 = broadcast_block(x[c][b]), a1 = broadcast_block(x[c][b + 1]);
            const __m512i s0 = dot_block(w0, a0), s1 = dot_block(w1, a1);
            // Same per-output b-order and scale-product rounding as rows().
            acc[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s0), _mm512_mul_ps(w0.scale, a0.scale), acc[c]);
            acc[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s1), _mm512_mul_ps(w1.scale, a1.scale), acc[c]);
        }
    }
    if (b < blocks) {
        const DecodedUnit w = decode_unit(unit);
        for (int c = 0; c < C; ++c) {
            const BroadcastBlock a = broadcast_block(x[c][b]);
            acc[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot_block(w, a)), _mm512_mul_ps(w.scale, a.scale), acc[c]);
        }
    }
    for (int c = 0; c < C; ++c) _mm512_storeu_ps(out[c], acc[c]);
}

template <int C>
MM3_CPU_FAST_TARGET void rows_paired(const uint8_t* matrix, int blocks, int rg,
                                    const QBlock* const* x, float out0[8][16], float out1[8][16]) {
    __m512 acc0[C], acc1[C];
    for (int c = 0; c < C; ++c) acc0[c] = acc1[c] = _mm512_setzero_ps();
    const uint8_t* unit0 = matrix + size_t(rg) * blocks * kUnit;
    const uint8_t* unit1 = unit0 + size_t(blocks) * kUnit;
    for (int b = 0; b < blocks; ++b, unit0 += kUnit, unit1 += kUnit) {
        const DecodedUnit w0 = decode_unit(unit0), w1 = decode_unit(unit1);
        for (int c = 0; c < C; ++c) {
            const BroadcastBlock a = broadcast_block(x[c][b]);
            const __m512i s0 = dot_block(w0, a), s1 = dot_block(w1, a);
            acc0[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s0), _mm512_mul_ps(w0.scale, a.scale), acc0[c]);
            acc1[c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s1), _mm512_mul_ps(w1.scale, a.scale), acc1[c]);
        }
    }
    for (int c = 0; c < C; ++c) {
        _mm512_storeu_ps(out0[c], acc0[c]);
        _mm512_storeu_ps(out1[c], acc1[c]);
    }
}

void rows_execution(const uint8_t* matrix, int blocks, int rg, const QBlock* const* x,
                    int cols, float out[8][16], Execution execution) {
    if ((int(execution) & kUnrollBlocks) && cols <= 2) {
        if (cols == 1) rows_unrolled<1>(matrix, blocks, rg, x, out);
        else rows_unrolled<2>(matrix, blocks, rg, x, out);
    } else rows_any(matrix, blocks, rg, x, cols, out);
}

// SwiGLU-OAI to BF16: the shared kernel's expression, character for character.
uint16_t swiglu(float gate, float up, const float* multiplier) {
    float g = std::min(gate * multiplier[0], 7.f);
    float u = std::clamp(up * multiplier[1], -7.f, 7.f);
    return float_bf16(g / (1.f + std::exp(-1.702f * g)) * (u + 1.f));
}

}  // namespace

bool supported() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

MM3_CPU_FAST_TARGET void to_bf16(const float* x, int n, uint16_t* out) {
    int i = 0;
    const __m512i exp = _mm512_set1_epi32(0x7f800000), mant = _mm512_set1_epi32(0x7fffff);
    for (; i + 16 <= n; i += 16) {
        const __m512i u = _mm512_loadu_si512(x + i);
        const __m512i r = _mm512_srli_epi32(
            _mm512_add_epi32(_mm512_add_epi32(u, _mm512_set1_epi32(0x7fff)), _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))), 16);
        const __mmask16 nan = _mm512_cmpeq_epi32_mask(_mm512_and_si512(u, exp), exp) & _mm512_test_epi32_mask(u, mant);
        const __m512i v = _mm512_mask_or_epi32(r, nan, _mm512_srli_epi32(u, 16), _mm512_set1_epi32(0x40));
        _mm256_storeu_si256((__m256i*) (out + i), _mm512_cvtepi32_epi16(v));
    }
    for (; i < n; ++i) {   // (the native CPU adapter's bf())
        uint32_t u = std::bit_cast<uint32_t>(x[i]);
        if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) { out[i] = uint16_t((u >> 16) | 0x40u); continue; }
        u += 0x7fffu + ((u >> 16) & 1u);
        out[i] = uint16_t(u >> 16);
    }
}

void quant(Act act, const uint16_t* x, int n, QBlock* out) {
    for (int b = 0; b < n / 16; ++b) quant_block(act, x + 16 * b, out[b]);
}
void quant_scalar(Act act, const uint16_t* x, int n, QBlock* out) {
    for (int b = 0; b < n / 16; ++b) quant_block_scalar(act, x + 16 * b, out[b]);
}
uint8_t nvfp4_scale_word(float amax) { return e4m3_rn_satfinite(amax / 6.0F); }

void gate_up_pair_rows(const Job& job, int h, int p, float a[8][16], float b[8][16], Execution execution) {
    if (execution == kExecutionBaseline) {
        rows_any(job.record, h / 16, 2 * p, job.x, job.columns, a);
        rows_any(job.record, h / 16, 2 * p + 1, job.x, job.columns, b);
    } else if ((int(execution) & kGatePair) && job.columns <= 4) {
        switch (job.columns) {
        case 1: rows_paired<1>(job.record, h / 16, 2 * p, job.x, a, b); break;
        case 2: rows_paired<2>(job.record, h / 16, 2 * p, job.x, a, b); break;
        case 3: rows_paired<3>(job.record, h / 16, 2 * p, job.x, a, b); break;
        default: rows_paired<4>(job.record, h / 16, 2 * p, job.x, a, b); break;
        }
    } else {
        rows_execution(job.record, h / 16, 2 * p, job.x, job.columns, a, execution);
        rows_execution(job.record, h / 16, 2 * p + 1, job.x, job.columns, b, execution);
    }
}

void gate_up_pair(Act act, const Job& job, int h, int i, int p, Execution execution) {
    (void) i;
    float a[8][16], b[8][16];
    if (execution == kExecutionBaseline) {
        rows_any(job.record, h / 16, 2 * p, job.x, job.columns, a);
        rows_any(job.record, h / 16, 2 * p + 1, job.x, job.columns, b);
    } else gate_up_pair_rows(job, h, p, a, b, execution);
    for (int n = 0; n < job.columns; ++n) {
        alignas(64) uint16_t hidden[16];
        for (int r = 0; r < 8; ++r) {
            hidden[r] = swiglu(a[n][2 * r], a[n][2 * r + 1], job.multiplier);
            hidden[8 + r] = swiglu(b[n][2 * r], b[n][2 * r + 1], job.multiplier);
        }
        quant_block(act, hidden, job.hidden[n][p]);
    }
}

void down_group(const Job& job, int h, int i, int r, Execution execution) {
    const uint8_t* down = job.record + size_t(2 * i / 16) * (h / 16) * kUnit;
    float v[8][16];
    if (execution == kExecutionBaseline) rows_any(down, i / 16, r, job.hidden, job.columns, v);
    else rows_execution(down, i / 16, r, job.hidden, job.columns, v, execution);
    for (int n = 0; n < job.columns; ++n)
        for (int k = 0; k < 16; ++k) job.y[n][r * 16 + k] = float_bf16(v[n][k] * job.multiplier[2]);
}

Span gate_up_span(int h, int i, int p0, int p1) {
    (void) i;
    const size_t group = size_t(h / 16) * kUnit;
    return {size_t(2 * p0) * group, size_t(2 * (p1 - p0)) * group};
}
Span down_span(int h, int i, int r0, int r1) {
    const size_t base = size_t(2 * i / 16) * (h / 16) * kUnit, group = size_t(i / 16) * kUnit;
    return {base + size_t(r0) * group, size_t(r1 - r0) * group};
}

size_t touch(const uint8_t* p, size_t n, const std::atomic<bool>* stop) {
    // one 8-byte read a 64-byte line (4 independent sums keep many lines in flight), the piece's last byte as well, so
    // a piece that does not start on a line still brings in its last line
    uint64_t a = 0, b = 0, c = 0, d = 0;
    size_t done = 0;
    while (done < n) {
        if (stop && stop->load(std::memory_order_relaxed)) break;
        const size_t piece = std::min<size_t>(4096, n - done);
        const uint8_t* q = p + done;
        size_t k = 0;
        for (; k + 256 <= piece; k += 256) {
            uint64_t w[4];
            std::memcpy(&w[0], q + k, 8);
            std::memcpy(&w[1], q + k + 64, 8);
            std::memcpy(&w[2], q + k + 128, 8);
            std::memcpy(&w[3], q + k + 192, 8);
            a += w[0], b += w[1], c += w[2], d += w[3];
        }
        for (; k < piece; k += 64) a += q[k];
        d += q[piece - 1];
        done += piece;
    }
    static std::atomic<uint64_t> sink{0};
    sink.fetch_xor(a ^ b ^ c ^ d, std::memory_order_relaxed);
    return done;
}

}  // namespace mm3::cpu_fast
