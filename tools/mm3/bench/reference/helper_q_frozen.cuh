// Frozen oracle from 7a2006db39ff66813e958b9cae30a631f68feebe:src/mm3/mm3_helper_q.cuh; only namespace/include renames.
// Kernel item 5: the helper cards' expert kernel with NVFP4 activations on int8 tensor cores (helper_q).
//
// The user's decision of 2026-11-13 (CLAUDE.md, Kernels): "All lanes 4-bit": NVFP4 activations for the experts on every
// lane, the 3090 and 4080 by exact FP4 x FP4 products on int8 tensor cores, with the 5090's already-quantized activations
// sent to every lane; adopted by tokens per second against each lane's current kernel. A precision change: a short
// usability check before adoption.
//
// Arithmetic ("W4A4"): the activations are the 5090's own NVFP4 encoding (infernix canon::quantize_a4_codes, global scale
// 1: BF16 values, an E4M3 scale per 16 = RNE-satfinite(amax / 6), E2M1 codes RNE-satfinite of value / scale), and the
// hidden vector is rounded to BF16 and encoded the same way, as the 5090's FP4 kernel does (native_fp4.cu). Doubled E2M1
// values are the integers -12..12 for weights and activations alike, so one m16n8k16 int8 tensor-core product sums 16
// inputs exactly in int32 (|sum| <= 16 x 144). Each such sum is widened to FP32 (exact) and multiplied by the product of
// the two half scales (E4M3 / 2 each: their product has at most 8 significant bits, exact) in one fmaf into the FP32
// accumulator. The warps of a group of 16 output rows split the inputs and add up at the end, as the tensor-core kernel
// (mm3_helper_kernels.cuh, project2) does; SwiGLU-OAI and the outputs' BF16 rounding are project2's.
//
// Layouts. An activation column on the card ("q layout"): K int8 doubled values, then K / 16 E4M3 scale bytes (input: 6528
// bytes a column; hidden: 3264). The transfer format of a row from the 5090 ("packed"): K / 2 code bytes (code 2i in the
// low nibble of byte i, 2i+1 in the high), then K / 16 scale bytes: 3456 bytes against 12,288 for the BF16 row.
// The weights are read as stored (RG16): one 32-bit word of a unit holds rows g (low nibbles) and g+8 (high) at four
// consecutive inputs, which is exactly an int8 A fragment's two registers once decoded.
#pragma once
#include "helper_tc_frozen.cuh"
#include <cstddef>
#include <cstdint>

namespace mm3::helper_q_frozen {
constexpr int kH = 6144, kI = 3072, kUnit = 144, kMaxColumns = 8;
constexpr std::size_t kInStride = kH + kH / 16;        // a q-layout input column: 6528 bytes
constexpr std::size_t kHidStride = kI + kI / 16;       // a q-layout hidden column: 3264 bytes
constexpr std::size_t kPackedRow = kH / 2 + kH / 16;   // a packed row from the 5090: 3456 bytes
constexpr std::size_t kJobHidden = kMaxColumns * kHidStride;   // a job's hidden columns: 26,112 bytes (256-aligned)
static_assert(kInStride % 16 == 0 && kHidStride % 16 == 0 && kPackedRow % 16 == 0 && kJobHidden % 256 == 0, "16-byte pieces");
// warps (1, 2, 4, 8) a group of 16 output rows; loads (1, 2, 4, 8) units a batch, as project2
inline bool valid(int warps, int loads) {
    return (warps == 1 || warps == 2 || warps == 4 || warps == 8) && (loads == 1 || loads == 2 || loads == 4 || loads == 8);
}
}  // namespace mm3::helper_q_frozen

#if !defined(__CUDACC__)
// The CPU helper tests compile the production helper without a CUDA compiler (tests/mm3_helper_fake_source.cmake) and
// never select these kernels: the same API, every launch refused.
namespace mm3::helper_q_frozen {
using infernix::ops::minimax_ffn::FastJob;
inline cudaError_t quant_bf16(const std::uint16_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t expand_packed(const std::uint8_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t ffn(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, std::uint8_t*, void*, cudaStream_t) {
    return cudaErrorInvalidValue;
}
inline cudaError_t ffn2(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, void*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t pack_bf16(const std::uint16_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t expand_bf16(const std::uint8_t*, std::uint16_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t rows_to_expanded(const float*, int, int, std::uint16_t*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t ffn3(int, int, const FastJob*, int, const std::uint16_t*, const std::uint16_t*, void*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t ffn_int_bits(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, std::uint8_t*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t expand_packed_codes(const std::uint8_t*, std::uint8_t*, int, bool, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t quant_bf16_codes(const std::uint16_t*, std::uint8_t*, int, bool, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t rows_to_codes(const float*, int, int, std::uint8_t*, bool, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t rows_to_formats(const float*, std::size_t, int, std::uint8_t*, std::uint16_t*, std::uint8_t*, std::uint8_t*, cudaStream_t) { return cudaErrorInvalidValue; }
}  // namespace mm3::helper_q_frozen
#else
#include "ops/common/canonical_math.h"
namespace mm3::helper_q_frozen {
using infernix::ops::minimax_ffn::FastJob;
using infernix::ops::minimax_ffn::float_bf16;
namespace detail {
namespace canon = infernix::ops::canon;
// Scale 0x7f is the invalid-activation marker. Keep it NaN through both
// projections so helper collection/recovery can observe a nonfinite result.
__device__ __forceinline__ void fill_scale_table(float* table, int threads) {
    for (int i = threadIdx.x; i < mm3::helper_tc_frozen::detail::kScaleTable; i += threads)
        table[i] = (unsigned(i) & 0x7fu) == 0x7fu ? __uint_as_float(0x7fc00000u)
            : mm3::helper_tc_frozen::detail::half_scale(unsigned(i) & 0x7fu);
    __syncthreads();
}
__device__ __forceinline__ canon::A4Codes checked_codes(const std::uint16_t* v) {
    canon::A4Codes q = canon::quantize_a4_codes(v, 1.0f);
    bool finite = true;
#pragma unroll
    for (int k = 0; k < 16; ++k) finite = finite && (v[k] & 0x7f80u) != 0x7f80u;
    if (!finite) q.scale_word = 0x7f;
    return q;
}
using mm3::helper_tc_frozen::detail::kScaleTable;
constexpr std::size_t kDownOffset = mm3::helper_tc_frozen::kDownOffset;

// Four E2M1 nibbles, one in each byte -> exact doubled values in E4M3.
// Keep negative zero: it is a representation detail, not a change of value.
__device__ __forceinline__ unsigned fp8_codes_quad(unsigned codes) {
    const unsigned select = (codes & 7u) | ((codes >> 4) & 0x70u) |
                            ((codes >> 8) & 0x700u) | ((codes >> 12) & 0x7000u);
    return __byte_perm(0x44403800u, 0x54504c48u, select) | ((codes & 0x08080808u) << 4);
}
template <bool Fp8> __device__ __forceinline__ unsigned code_quad(unsigned codes) {
    if constexpr (Fp8) return fp8_codes_quad(codes);
    return canon::e2m1_x2_quad(codes);
}
// One 16-value block of BF16 values encoded (canon::quantize_a4_codes, the 5090's rule) into its q layout: 16 doubled
// values and the scale byte.
template <bool Fp8 = false> __device__ __forceinline__ void encode_block(const std::uint16_t* v, std::uint8_t* codes16, std::uint8_t* scale) {
    const canon::A4Codes q = checked_codes(v);
    unsigned w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
        w[k] = code_quad<Fp8>(unsigned(q.code[4 * k]) | (unsigned(q.code[4 * k + 1]) << 8) | (unsigned(q.code[4 * k + 2]) << 16) |
                                   (unsigned(q.code[4 * k + 3]) << 24));
    *reinterpret_cast<uint4*>(codes16) = make_uint4(w[0], w[1], w[2], w[3]);
    *scale = q.scale_word;
}
// Store one canonical block in any requested representation. Row pointers
// already name their destination row; shared by worker and queued preparation.
__device__ __forceinline__ void store_codes(const canon::A4Codes& q, int b, std::uint8_t* packed, std::uint16_t* bf16,
                                           std::uint8_t* s8, std::uint8_t* f8) {
    unsigned codes[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
        codes[k] = unsigned(q.code[4 * k]) | (unsigned(q.code[4 * k + 1]) << 8) |
                   (unsigned(q.code[4 * k + 2]) << 16) | (unsigned(q.code[4 * k + 3]) << 24);
    if (packed) {
        unsigned lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            lo |= (unsigned(q.code[2 * k]) | (unsigned(q.code[2 * k + 1]) << 4)) << (8 * k);
            hi |= (unsigned(q.code[8 + 2 * k]) | (unsigned(q.code[8 + 2 * k + 1]) << 4)) << (8 * k);
        }
        *reinterpret_cast<uint2*>(packed + b * 8) = make_uint2(lo, hi);
        packed[kH / 2 + b] = q.scale_word;
    }
    if (bf16) {
        const float scale = mm3::helper_tc_frozen::detail::half_scale(q.scale_word & 0x7fu);
#pragma unroll
        for (int k = 0; k < 16; ++k) bf16[b * 16 + k] = q.scale_word == 0x7f ? std::uint16_t(0x7fc0)
            : float_bf16(float(canon::e2m1_x2(q.code[k])) * scale);
    }
    if (s8) {
        *reinterpret_cast<uint4*>(s8 + b * 16) = make_uint4(code_quad<false>(codes[0]), code_quad<false>(codes[1]),
                                                         code_quad<false>(codes[2]), code_quad<false>(codes[3]));
        s8[kH + b] = q.scale_word;
    }
    if (f8) {
        *reinterpret_cast<uint4*>(f8 + b * 16) = make_uint4(code_quad<true>(codes[0]), code_quad<true>(codes[1]),
                                                         code_quad<true>(codes[2]), code_quad<true>(codes[3]));
        f8[kH + b] = q.scale_word;
    }
}
// BF16 columns [columns][kH] -> q layout [columns][kInStride]; a thread a block of 16
template <bool Fp8 = false>
__global__ void quant_bf16_kernel(const std::uint16_t* x, std::uint8_t* q, int columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns * (kH / 16)) return;
    const int col = t / (kH / 16), b = t % (kH / 16);
    std::uint8_t* dst = q + std::size_t(col) * kInStride;
    encode_block<Fp8>(x + std::size_t(col) * kH + b * 16, dst + b * 16, dst + kH + b);
}
// packed rows [columns][kPackedRow] -> q layout; a thread a block of 16
template <bool Fp8 = false>
__global__ void expand_packed_kernel(const std::uint8_t* packed, std::uint8_t* q, int columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns * (kH / 16)) return;
    const int col = t / (kH / 16), b = t % (kH / 16);
    const std::uint8_t* src = packed + std::size_t(col) * kPackedRow;
    const uint2 v = *reinterpret_cast<const uint2*>(src + b * 8);   // 16 codes, two a byte
    unsigned w[4];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const unsigned word = k ? v.y : v.x;
        const unsigned lo = word & 0x0f0f0f0fu, hi = (word >> 4) & 0x0f0f0f0fu;   // codes 0,2,4,6 and 1,3,5,7 of 8
        w[2 * k] = code_quad<Fp8>(__byte_perm(lo, hi, 0x5140u));
        w[2 * k + 1] = code_quad<Fp8>(__byte_perm(lo, hi, 0x7362u));
    }
    std::uint8_t* dst = q + std::size_t(col) * kInStride;
    *reinterpret_cast<uint4*>(dst + b * 16) = make_uint4(w[0], w[1], w[2], w[3]);
    dst[kH + b] = src[kH / 2 + b];
}
// D = A B: a 16 x 16 tile of int8 times 16 x 8 int8, int32 result (A: row g inputs 4c..4c+3 in a0, row g+8 in a1;
// B: column g inputs 4c..4c+3; D: row g columns 2c, 2c+1 in d0, d1, row g+8 in d2, d3)
__device__ __forceinline__ void tile_s8(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned b0) {
    d0 = d1 = d2 = d3 = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0));
}
// E4M3 exactly represents all doubled E2M1 values. A k32 MMA is issued with
// its upper 16 products zero, keeping each original per-16 scale independent.
__device__ __forceinline__ void tile_fp8(float& d0, float& d1, float& d2, float& d3, unsigned a0, unsigned a1, unsigned b0) {
    d0 = d1 = d2 = d3 = 0.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(0u), "r"(0u), "r"(b0), "r"(0u));
#else
    // A device image without FP8 must never silently return zero products.
    asm volatile("trap;");
#endif
}
// The sums of one warp's share of the inputs for 16 output rows of one job: each unit's int32 product scaled by the
// two half scales into FP32, P units' loads issued before their use (project2's batching).
// |d| <= 16 * 12 * 12 = 2304. In [2^23,2^24), consecutive FP32 bit
// patterns differ by one; the integer addition therefore encodes 12582912+d
// exactly, including negative d (unsigned arithmetic is modulo 2^32).
template <bool Bits> __device__ __forceinline__ float dot_float(int d) {
    if constexpr (Bits) return __uint_as_float(0x4b400000u + unsigned(d)) - 12582912.0f;
    return float(d);
}
template <int S, bool Down, int P, bool Bits, bool Fp8>
__device__ __forceinline__ void sums(const std::uint8_t* unit, const std::uint8_t* xcol, const std::uint8_t* scol0,
                                     const std::uint8_t* scol1, unsigned keep, const float* scale, int warp, float& s0,
                                     float& s1, float& s2, float& s3) {
    constexpr int K = Down ? kI : kH, blocks = K / 16, per = blocks / S;
    static_assert(blocks % S == 0 && per % P == 0, "a warp's share of units must split into batches of P");
    const int lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int first = per * warp;
    unit += std::size_t(first) * kUnit + c * 32 + g * 4;
    const unsigned* xw = reinterpret_cast<const unsigned*>(xcol + std::size_t(first) * 16 + 4 * c);
#pragma unroll 1
    for (int b = 0; b < per; b += P, unit += P * kUnit, xw += P * 4) {
        unsigned w[P], sa[P], sb[P], bx[P], x0[P], x1[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            w[p] = __ldg(reinterpret_cast<const unsigned*>(unit + p * kUnit));
            sa[p] = __ldg(unit + p * kUnit + 128 - c * 32 - g * 4 + g);   // (the unit's scale bytes: rows g and g+8)
            sb[p] = __ldg(unit + p * kUnit + 136 - c * 32 - g * 4 + g);
            bx[p] = __ldg(xw + p * 4);
            x0[p] = __ldg(scol0 + first + b + p);
            x1[p] = __ldg(scol1 + first + b + p);
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const unsigned a0 = code_quad<Fp8>(w[p] & 0x0f0f0f0fu), a1 = code_quad<Fp8>((w[p] >> 4) & 0x0f0f0f0fu);
            float f0, f1, f2, f3;
            if constexpr (Fp8) tile_fp8(f0, f1, f2, f3, a0, a1, bx[p] & keep);
            else {
                int d0, d1, d2, d3;
                tile_s8(d0, d1, d2, d3, a0, a1, bx[p] & keep);
                f0 = dot_float<Bits>(d0); f1 = dot_float<Bits>(d1); f2 = dot_float<Bits>(d2); f3 = dot_float<Bits>(d3);
            }
            const float w0 = scale[sa[p]], w1 = scale[sb[p]], v0 = scale[x0[p]], v1 = scale[x1[p]];
            s0 = fmaf(f0, w0 * v0, s0); s1 = fmaf(f1, w0 * v1, s1);
            s2 = fmaf(f2, w1 * v0, s2); s3 = fmaf(f3, w1 * v1, s3);
        }
    }
}
// The up projection (gate and up interleaved by row), two groups of 16 record rows a block (2 S warps), so the block holds
// 16 consecutive hidden units and encodes them itself (as native_fp4.cu's up kernel does); grid (kI / 16, jobs).
template <int S, int P, bool Bits, bool Fp8>
__device__ __forceinline__ void up_block(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* q_in,
                                         std::uint8_t* hidden_q, int j, int pair) {
    const FastJob job = jobs[j];
    if (job.columns == 0) return;
    __shared__ float scale[kScaleTable];
    __shared__ float part[2][S][4][32];
    __shared__ std::uint16_t hid[kMaxColumns][16];
    fill_scale_table(scale, 64 * S);
    const int warp = (threadIdx.x >> 5) % S, local = (threadIdx.x >> 5) / S, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int group = 2 * pair + local;
    const std::size_t col0 = std::size_t(job.x - x_base) / kH;   // (the job's first column in the pass)
    const bool live = g < job.columns;
    const unsigned keep = live ? ~0u : 0u;
    const std::uint8_t* xcol = q_in + (col0 + (live ? g : 0)) * kInStride;
    const std::uint8_t* scol0 = q_in + (col0 + (2 * c < job.columns ? 2 * c : 0)) * kInStride + kH;
    const std::uint8_t* scol1 = q_in + (col0 + (2 * c + 1 < job.columns ? 2 * c + 1 : 0)) * kInStride + kH;
    const std::uint8_t* unit = job.record + std::size_t(group) * (kH / 16) * kUnit;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    sums<S, false, P, Bits, Fp8>(unit, xcol, scol0, scol1, keep, scale, warp, s0, s1, s2, s3);
    if constexpr (S > 1) {
        part[local][warp][0][lane] = s0; part[local][warp][1][lane] = s1; part[local][warp][2][lane] = s2; part[local][warp][3][lane] = s3;
    }
    __syncthreads();
    if (warp == 0) {
        if constexpr (S > 1) {
#pragma unroll
            for (int w = 1; w < S; ++w) { s0 += part[local][w][0][lane]; s1 += part[local][w][1][lane]; s2 += part[local][w][2][lane]; s3 += part[local][w][3][lane]; }
        }
        // record rows g and g+8 of this group: an even row is a gate row, the next one its up row (project2's epilogue)
        const float u0 = __shfl_xor_sync(0xffffffffu, s0, 4), u1 = __shfl_xor_sync(0xffffffffu, s1, 4);
        const float u2 = __shfl_xor_sync(0xffffffffu, s2, 4), u3 = __shfl_xor_sync(0xffffffffu, s3, 4);
        if (!(g & 1)) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int col = 2 * c + v;
                if (col >= job.columns) continue;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const float raw_gate = (half ? (v ? s3 : s2) : (v ? s1 : s0)) * job.multiplier[0];
                    const float raw_up = (half ? (v ? u3 : u2) : (v ? u1 : u0)) * job.multiplier[1];
                    const float gate = fminf(raw_gate, 7.f);
                    const float up = fminf(fmaxf(raw_up, -7.f), 7.f);
                    // fmin/fmax choose the finite operand on NaN; preserve the
                    // error before clamping so collection can request recovery.
                    hid[col][8 * local + g / 2 + 4 * half] = isfinite(raw_gate) && isfinite(raw_up)
                        ? float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f)) : std::uint16_t(0x7fc0);
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < job.columns) {   // the 16 hidden units of this block, one column a thread, encoded as the 5090 does
        const int col = threadIdx.x;
        std::uint8_t* dst = hidden_q + std::size_t(j) * kJobHidden + std::size_t(col) * kHidStride;
        encode_block<Fp8>(hid[col], dst + pair * 16, dst + kI + pair);
    }
}
// The down projection; grid (kH / 16, jobs), S warps a group of 16 output rows.
template <int S, int P, bool Bits, bool Fp8>
__device__ __forceinline__ void down_block(const FastJob* jobs, const std::uint8_t* hidden_q, int j, int group) {
    const FastJob job = jobs[j];
    if (job.columns == 0) return;
    __shared__ float scale[kScaleTable];
    __shared__ float part[S][4][32];
    fill_scale_table(scale, 32 * S);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const bool live = g < job.columns;
    const unsigned keep = live ? ~0u : 0u;
    const std::uint8_t* hq = hidden_q + std::size_t(j) * kJobHidden;
    const std::uint8_t* xcol = hq + std::size_t(live ? g : 0) * kHidStride;
    const std::uint8_t* scol0 = hq + std::size_t(2 * c < job.columns ? 2 * c : 0) * kHidStride + kI;
    const std::uint8_t* scol1 = hq + std::size_t(2 * c + 1 < job.columns ? 2 * c + 1 : 0) * kHidStride + kI;
    const std::uint8_t* unit = job.record + kDownOffset + std::size_t(group) * (kI / 16) * kUnit;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    sums<S, true, P, Bits, Fp8>(unit, xcol, scol0, scol1, keep, scale, warp, s0, s1, s2, s3);
    if constexpr (S > 1) {
        part[warp][0][lane] = s0; part[warp][1][lane] = s1; part[warp][2][lane] = s2; part[warp][3][lane] = s3;
        __syncthreads();
        if (warp) return;
#pragma unroll
        for (int w = 1; w < S; ++w) { s0 += part[w][0][lane]; s1 += part[w][1][lane]; s2 += part[w][2][lane]; s3 += part[w][3][lane]; }
    }
#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int col = 2 * c + v;
        if (col >= job.columns) continue;
        std::uint16_t* y = job.y + std::size_t(col) * kH + group * 16 + g;
        y[0] = float_bf16((v ? s1 : s0) * job.multiplier[2]);
        y[8] = float_bf16((v ? s3 : s2) * job.multiplier[2]);
    }
}
// Standard and queued launches share these exact block bodies. The queued
// wrappers visit only the device-published job count, rather than padding the
// graph to 64 jobs. The barrier retires shared scratch before the next item.
template <int S, int P, bool Bits, bool Fp8>
__global__ void __launch_bounds__(64 * S) up_kernel(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in, std::uint8_t* hidden) {
    up_block<S, P, Bits, Fp8>(jobs, x_base, in, hidden, int(blockIdx.y), int(blockIdx.x));
}
template <int S, int P, bool Bits, bool Fp8>
__global__ void __launch_bounds__(32 * S) down_kernel(const FastJob* jobs, const std::uint8_t* hidden) {
    down_block<S, P, Bits, Fp8>(jobs, hidden, int(blockIdx.y), int(blockIdx.x));
}
template <int S, int P, bool Bits, bool Fp8>
inline cudaError_t ffn_sp(const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in, std::uint8_t* hidden_q,
                          cudaStream_t stream) {
    up_kernel<S, P, Bits, Fp8><<<dim3(kI / 16, count), 64 * S, 0, stream>>>(jobs, x_base, q_in, hidden_q);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    down_kernel<S, P, Bits, Fp8><<<dim3(kH / 16, count), 32 * S, 0, stream>>>(jobs, hidden_q);
    return cudaGetLastError();
}
template <int S, bool Bits = false, bool Fp8 = false>
inline cudaError_t ffn_s(int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                         std::uint8_t* hidden_q, cudaStream_t stream) {
    switch (loads) {
        case 2: return ffn_sp<S, 2, Bits, Fp8>(jobs, count, x_base, q_in, hidden_q, stream);
        case 4: return ffn_sp<S, 4, Bits, Fp8>(jobs, count, x_base, q_in, hidden_q, stream);
        case 8: return ffn_sp<S, 8, Bits, Fp8>(jobs, count, x_base, q_in, hidden_q, stream);
        default: return ffn_sp<S, 1, Bits, Fp8>(jobs, count, x_base, q_in, hidden_q, stream);
    }
}
}  // namespace detail

// Asynchronous on the stream. BF16 columns [columns][kH] (as NativeHelper::submit lays them out) -> q layout.
inline cudaError_t quant_bf16(const std::uint16_t* x, std::uint8_t* q, int columns, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    const int n = columns * (kH / 16);
    detail::quant_bf16_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(x, q, columns);
    return cudaGetLastError();
}
// Packed rows from the 5090 [columns][kPackedRow] -> q layout.
inline cudaError_t expand_packed(const std::uint8_t* packed, std::uint8_t* q, int columns, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    const int n = columns * (kH / 16);
    detail::expand_packed_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(packed, q, columns);
    return cudaGetLastError();
}
// As above, with byte codes already in the selected MMA representation.
inline cudaError_t expand_packed_codes(const std::uint8_t* packed, std::uint8_t* q, int columns, bool fp8, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    if (!packed || !q) return cudaErrorInvalidValue;
    const int n = columns * (kH / 16);
    if (fp8) detail::expand_packed_kernel<true><<<(n + 127) / 128, 128, 0, stream>>>(packed, q, columns);
    else detail::expand_packed_kernel<false><<<(n + 127) / 128, 128, 0, stream>>>(packed, q, columns);
    return cudaGetLastError();
}
inline cudaError_t quant_bf16_codes(const std::uint16_t* x, std::uint8_t* q, int columns, bool fp8, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    if (!x || !q) return cudaErrorInvalidValue;
    const int n = columns * (kH / 16);
    if (fp8) detail::quant_bf16_kernel<true><<<(n + 127) / 128, 128, 0, stream>>>(x, q, columns);
    else detail::quant_bf16_kernel<false><<<(n + 127) / 128, 128, 0, stream>>>(x, q, columns);
    return cudaGetLastError();
}
// `count` jobs of widths 1..8 (0: inactive), their inputs in q layout: job j's column k at q_in + ((job.x - x_base) / kH
// + k) * kInStride; hidden_q: count * kJobHidden bytes. warps / loads as project2's (valid()).
inline cudaError_t ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                       std::uint8_t* hidden_q, void* /*unused*/, cudaStream_t stream) {
    if (!jobs || !q_in || !hidden_q || count < 1 || count > 65535 || !valid(warps, loads)) return cudaErrorInvalidValue;
    switch (warps) {
        case 1: return detail::ffn_s<1>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 2: return detail::ffn_s<2>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 4: return detail::ffn_s<4>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        default: return detail::ffn_s<8>(loads, jobs, count, x_base, q_in, hidden_q, stream);
    }
}
inline cudaError_t ffn_int_bits(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                                std::uint8_t* hidden_q, cudaStream_t stream) {
    if (!jobs || !q_in || !hidden_q || count < 1 || count > 65535 || !valid(warps, loads)) return cudaErrorInvalidValue;
    switch (warps) {
        case 1: return detail::ffn_s<1, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 2: return detail::ffn_s<2, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 4: return detail::ffn_s<4, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        default: return detail::ffn_s<8, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
    }
}
inline cudaError_t ffn_fp8(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                           std::uint8_t* hidden_q, cudaStream_t stream) {
    if (!jobs || !q_in || !hidden_q || count < 1 || count > 65535 || !valid(warps, loads)) return cudaErrorInvalidValue;
    switch (warps) {
        case 1: return detail::ffn_s<1, false, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 2: return detail::ffn_s<2, false, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        case 4: return detail::ffn_s<4, false, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
        default: return detail::ffn_s<8, false, true>(loads, jobs, count, x_base, q_in, hidden_q, stream);
    }
}
namespace detail {
// ---- Variant 2 (kernel item 5b, helper_kernel 81/82/84/88; 180 + ... with the wide graphs): the same arithmetic, on
// today's BF16 tensor-core product instead of the int8 one. Each NVFP4 activation is a doubled E2M1 value (at most two
// significant bits) times its half scale (at most four), so the product holds at most six significant bits: exact in
// BF16. A unit's 16 products then share that scale, each an integer of at most 144 times it, and their sum an integer
// of at most 2,304 times it: exact in the FP32 accumulation of mma.sync bf16 (sixteen bits of span below 24). So the
// tensor-core result of a unit is exactly variant 1's int32 sum times the column's half scale, and each unit adds
// fmaf(d, w, s) - the same exact product d_int * v * w as variant 1's fmaf(float(d_int), w * v, s) (w * v has at most
// eight significant bits: exact) rounded once with the same s, in the same order: the outputs are variant 1's bit for
// bit at the same warps and loads. There is no int-to-float conversion (variant 1's four a unit run at a quarter of the
// FP32 rate on sm_86 and sm_89), and the activations are read from the 5090's packed rows directly (no expansion pass).
// The up kernel encodes its 16 hidden units per column with 16 threads (the same codes and scale as encode_block) and
// writes them as BF16 values (code * half scale, exact) where today's down kernel reads its hidden vector, so the down
// projection is helper_tc's project2 unchanged.
//
// Four packed codes of one column (bytes c and 4 + c of the unit's eight: codes 2c, 2c+1, 2c+8, 2c+9) in the low nibbles
// of a word's bytes, in that order: decode4's input.
__device__ __forceinline__ unsigned packed_codes(uint2 v, int c) {
    const unsigned lo = __byte_perm(v.x & 0x0f0f0f0fu, (v.x >> 4) & 0x0f0f0f0fu, unsigned(c) | (unsigned(c + 4) << 4));
    const unsigned hi = __byte_perm(v.y & 0x0f0f0f0fu, (v.y >> 4) & 0x0f0f0f0fu, unsigned(c) | (unsigned(c + 4) << 4));
    return __byte_perm(lo, hi, 0x5410u);
}
// a BF16 pair times a BF16 pair, rounded to nearest (exact here); fma with -0 as the PTX's bf16x2 multiply
__device__ __forceinline__ unsigned mul_bf16x2(unsigned a, unsigned b) {
    unsigned d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}
// the half scale of every byte value as a BF16 pair (exact: at most four significant bits), beside the FP32 table
__device__ __forceinline__ void fill_scale_pairs(unsigned* pairs, int threads) {
    for (int i = threadIdx.x; i < kScaleTable; i += threads) {
        const unsigned b = (unsigned(i) & 0x7fu) == 0x7fu ? 0x7fc0u : __float_as_uint(mm3::helper_tc_frozen::detail::half_scale(unsigned(i) & 0x7fu)) >> 16;   // (exact: low bits zero)
        pairs[i] = b | (b << 16);
    }
}
// The up projection, variant 2: as up_kernel (two groups of 16 record rows a block, S warps each), its inputs from
// packed rows [columns][kPackedRow] (job j's column k at packed + ((job.x - x_base) / kH + k) * kPackedRow), its hidden
// vector to the tensor-core kernel's workspace (BF16 [job][kMaxColumns][kI], helper_tc::kJobWorkspace a job).
template <int S, int P, bool Expanded = false>
__device__ __forceinline__ void up2_block(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* packed,
                                          std::uint8_t* workspace, int j, int pair) {
    namespace tcd = mm3::helper_tc_frozen::detail;
    constexpr int blocks = kH / 16, per = blocks / S;
    static_assert(blocks % S == 0 && per % P == 0, "a warp's share of units must split into batches of P");
    const FastJob job = jobs[j];
    if (job.columns == 0) return;
    __shared__ float scale[kScaleTable];
    __shared__ unsigned scale2[kScaleTable];
    __shared__ float part[2][S][4][32];
    __shared__ std::uint16_t hid[kMaxColumns][16];
    fill_scale_pairs(scale2, 64 * S);
    fill_scale_table(scale, 64 * S);   // (its barrier also covers scale2)
    const int warp = (threadIdx.x >> 5) % S, local = (threadIdx.x >> 5) / S, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int group = 2 * pair + local;
    const std::size_t col0 = std::size_t(job.x - x_base) / kH;
    const bool live = g < job.columns;
    const unsigned keep = live ? ~0u : 0u;
    const std::uint8_t* row = packed + (col0 + (live ? g : 0)) * (Expanded ? kH * 2 : kPackedRow);   // (column g's packed row)
    const std::uint8_t* unit = job.record + (std::size_t(group) * blocks + std::size_t(per) * warp) * kUnit;
    const int at = (c >> 1) * 32 + g * 4, shift = 16 * (c & 1), first = per * warp;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll 1
    for (int b = 0; b < per; b += P, unit += P * kUnit) {
        unsigned lo[P], hi[P], sa[P], sb[P], xs[P];
        uint2 xc[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {
            lo[p] = __ldg(reinterpret_cast<const unsigned*>(unit + p * kUnit + at));
            hi[p] = __ldg(reinterpret_cast<const unsigned*>(unit + p * kUnit + at + 64));
            sa[p] = __ldg(unit + p * kUnit + 128 + g); sb[p] = __ldg(unit + p * kUnit + 136 + g);
            if constexpr (Expanded) {
                const auto* v = reinterpret_cast<const std::uint16_t*>(row) + std::size_t(first + b + p) * 16;
                xc[p] = make_uint2(__ldg(reinterpret_cast<const unsigned*>(v + 2 * c)),
                                   __ldg(reinterpret_cast<const unsigned*>(v + 8 + 2 * c)));
            } else {
                xc[p] = __ldg(reinterpret_cast<const uint2*>(row + std::size_t(first + b + p) * 8));
                xs[p] = __ldg(row + kH / 2 + first + b + p);
            }
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {
            const unsigned word = ((lo[p] >> shift) & 0xffffu) | ((hi[p] >> shift) << 16);
            unsigned a0, a1, a2, a3, x01, x89;
            tcd::decode4(word & 0x0f0f0f0fu, a0, a2);
            tcd::decode4((word >> 4) & 0x0f0f0f0fu, a1, a3);
            if constexpr (Expanded) { x01 = xc[p].x; x89 = xc[p].y; }
            else { tcd::decode4(packed_codes(xc[p], c), x01, x89); }
            const unsigned v = Expanded ? 0 : scale2[xs[p]];
            const float w0 = scale[sa[p]], w1 = scale[sb[p]];
            float d0, d1, d2, d3;
            tcd::tile(d0, d1, d2, d3, a0, a1, a2, a3, (Expanded ? x01 : mul_bf16x2(x01, v)) & keep, (Expanded ? x89 : mul_bf16x2(x89, v)) & keep);
            s0 = fmaf(d0, w0, s0); s1 = fmaf(d1, w0, s1);
            s2 = fmaf(d2, w1, s2); s3 = fmaf(d3, w1, s3);
        }
    }
    if constexpr (S > 1) {
        part[local][warp][0][lane] = s0; part[local][warp][1][lane] = s1; part[local][warp][2][lane] = s2; part[local][warp][3][lane] = s3;
    }
    __syncthreads();
    if (warp == 0) {   // (up_kernel's epilogue, unchanged)
        if constexpr (S > 1) {
#pragma unroll
            for (int w = 1; w < S; ++w) { s0 += part[local][w][0][lane]; s1 += part[local][w][1][lane]; s2 += part[local][w][2][lane]; s3 += part[local][w][3][lane]; }
        }
        const float u0 = __shfl_xor_sync(0xffffffffu, s0, 4), u1 = __shfl_xor_sync(0xffffffffu, s1, 4);
        const float u2 = __shfl_xor_sync(0xffffffffu, s2, 4), u3 = __shfl_xor_sync(0xffffffffu, s3, 4);
        if (!(g & 1)) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int col = 2 * c + v;
                if (col >= job.columns) continue;
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const float raw_gate = (half ? (v ? s3 : s2) : (v ? s1 : s0)) * job.multiplier[0];
                    const float raw_up = (half ? (v ? u3 : u2) : (v ? u1 : u0)) * job.multiplier[1];
                    const float gate = fminf(raw_gate, 7.f);
                    const float up = fminf(fmaxf(raw_up, -7.f), 7.f);
                    // fmin/fmax choose the finite operand on NaN; preserve the
                    // error before clamping so collection can request recovery.
                    hid[col][8 * local + g / 2 + 4 * half] = isfinite(raw_gate) && isfinite(raw_up)
                        ? float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f)) : std::uint16_t(0x7fc0);
                }
            }
        }
    }
    __syncthreads();
    // the 16 hidden units of each column: 16 threads a column, canon::quantize_a4_codes's rule element by element (the
    // largest magnitude by a 16-lane maximum - order-free - then the same scale word in every thread, then each its own
    // code), written as code * half scale in BF16 (exact) where project2's down kernel reads the hidden vector
    std::uint16_t* hidden = reinterpret_cast<std::uint16_t*>(workspace + std::size_t(j) * mm3::helper_tc_frozen::kJobWorkspace);
    for (int base = 0; base < 16 * kMaxColumns; base += 64 * S) {
        const int t = base + int(threadIdx.x), col = t >> 4, k = t & 15;
        const bool mine = col < job.columns;
        const std::uint16_t h = mine ? hid[col][k] : std::uint16_t(0);
        float amax = canon::f32_from_bits(canon::f32_bits(canon::bf16_to_f32(h)) & 0x7FFFFFFFu);
        unsigned invalid = (h & 0x7f80u) == 0x7f80u;
#pragma unroll
        for (int m = 8; m >= 1; m >>= 1) {
            const float o = __shfl_xor_sync(0xffffffffu, amax, m, 16);
            amax = o > amax ? o : amax;
            invalid |= __shfl_xor_sync(0xffffffffu, invalid, m, 16);
        }
        if (mine) {
            const std::uint8_t word = canon::a4_scale_word(amax, 1.0f);
            const int code2 = canon::e4m3_scaled(word) == 0 ? 0 : canon::e2m1_x2(canon::a4_code(canon::bf16_to_f32(h), word, 1.0f));
            hidden[std::size_t(col) * kI + pair * 16 + k] = invalid ? std::uint16_t(0x7fc0) : float_bf16(float(code2) * scale[word]);
        }
    }
}
// BF16 columns [columns][kH] -> packed rows (the 5090's pack_rows' layout and encoding from BF16): variant 2's inputs when
// a pass's rows came as BF16; a thread a block of 16.
template <int = 0>
__global__ void pack_bf16_kernel(const std::uint16_t* x, std::uint8_t* packed, int columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns * (kH / 16)) return;
    const int col = t / (kH / 16), b = t % (kH / 16);
    const canon::A4Codes q = checked_codes(x + std::size_t(col) * kH + b * 16);
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        lo |= (unsigned(q.code[2 * k]) | (unsigned(q.code[2 * k + 1]) << 4)) << (8 * k);
        hi |= (unsigned(q.code[8 + 2 * k]) | (unsigned(q.code[8 + 2 * k + 1]) << 4)) << (8 * k);
    }
    std::uint8_t* dst = packed + std::size_t(col) * kPackedRow;
    *reinterpret_cast<uint2*>(dst + b * 8) = make_uint2(lo, hi);
    dst[kH / 2 + b] = q.scale_word;
}
template <int S, int P, bool Expanded = false>
__global__ void __launch_bounds__(64 * S) up2_kernel(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in, std::uint8_t* workspace) {
    up2_block<S, P, Expanded>(jobs, x_base, in, workspace, int(blockIdx.y), int(blockIdx.x));
}
// Family 0 INT8, 1 INT8 bit conversion, 2 FP8, 3 inline BF16 decode,
// 4 preexpanded BF16. Input layout is the ordinary family's input layout.
template <int S, int P, int Family, bool Down, bool Sm89Build>
__global__ void __launch_bounds__(Down ? 32 * S : 64 * S) queued_kernel(
    const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in, std::uint8_t* workspace,
    const int* count, const int* kind, int wanted) {
    if (*reinterpret_cast<const volatile int*>(kind) != wanted) return;
    const int n = *reinterpret_cast<const volatile int*>(count);
    if (n < 1 || n > 64) return;
    constexpr int groups = Down ? kH / 16 : kI / 16;
    for (int item = int(blockIdx.x); item < n * groups; item += int(gridDim.x)) {
        const int j = item / groups, group = item % groups;
        if constexpr (Family < 3) {
            if constexpr (Down) down_block<S, P, Family == 1, Family == 2>(jobs, workspace, j, group);
            else up_block<S, P, Family == 1, Family == 2>(jobs, x_base, in, workspace, j, group);
        } else {
            if constexpr (Down) mm3::helper_tc_frozen::detail::project2_block<S, true, P>(jobs, workspace, j, group);
            else up2_block<S, P, Family == 4>(jobs, x_base, in, workspace, j, group);
        }
        __syncthreads();
    }
}
template <int S, int P, int Family, bool Sm89Build>
inline cudaError_t queued_sp(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in, std::uint8_t* workspace,
                            const int* count, const int* kind, int wanted, int blocks, cudaStream_t stream) {
    queued_kernel<S, P, Family, false, Sm89Build><<<blocks, 64 * S, 0, stream>>>(jobs, x_base, in, workspace, count, kind, wanted);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    queued_kernel<S, P, Family, true, Sm89Build><<<blocks, 32 * S, 0, stream>>>(jobs, x_base, in, workspace, count, kind, wanted);
    return cudaGetLastError();
}
template <int S, int Family, bool Sm89Build>
inline cudaError_t queued_s(int loads, const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in, std::uint8_t* workspace,
                           const int* count, const int* kind, int wanted, int blocks, cudaStream_t stream) {
    switch (loads) {
        case 2: return queued_sp<S, 2, Family, Sm89Build>(jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        case 4: return queued_sp<S, 4, Family, Sm89Build>(jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        case 8: return queued_sp<S, 8, Family, Sm89Build>(jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        default: return queued_sp<S, 1, Family, Sm89Build>(jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
    }
}
template <int Family, bool Sm89Build>
inline cudaError_t queued_family(int warps, int loads, const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in,
                                std::uint8_t* workspace, const int* count, const int* kind, int wanted, int blocks, cudaStream_t stream) {
    switch (warps) {
        case 1: return queued_s<1, Family, Sm89Build>(loads, jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        case 2: return queued_s<2, Family, Sm89Build>(loads, jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        case 4: return queued_s<4, Family, Sm89Build>(loads, jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
        default: return queued_s<8, Family, Sm89Build>(loads, jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
    }
}
template <int S, int P, bool Expanded = false>
inline cudaError_t ffn2_sp(const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* packed, void* workspace,
                           cudaStream_t stream) {
    auto* ws = static_cast<std::uint8_t*>(workspace);
    up2_kernel<S, P, Expanded><<<dim3(kI / 16, count), 64 * S, 0, stream>>>(jobs, x_base, packed, ws);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    mm3::helper_tc_frozen::detail::project2<S, true, P><<<dim3(kH / 16, count), 32 * S, 0, stream>>>(jobs, ws);
    return cudaGetLastError();
}
template <int S, bool Expanded = false>
inline cudaError_t ffn2_s(int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* packed,
                          void* workspace, cudaStream_t stream) {
    switch (loads) {
        case 2: return ffn2_sp<S, 2, Expanded>(jobs, count, x_base, packed, workspace, stream);
        case 4: return ffn2_sp<S, 4, Expanded>(jobs, count, x_base, packed, workspace, stream);
        case 8: return ffn2_sp<S, 8, Expanded>(jobs, count, x_base, packed, workspace, stream);
        default: return ffn2_sp<S, 1, Expanded>(jobs, count, x_base, packed, workspace, stream);
    }
}
}  // namespace detail

// Distinct template identity keeps the sm_89 fatbin's CUDA launch stubs
// separate from the ordinary sm_86/sm_120 translation unit. FP8 is instantiated
// only in the explicit sm_89 wrapper, never in an unsupported sm_86 image.
template <bool Sm89Build = false>
inline cudaError_t ffn_queued(int family, int warps, int loads, const FastJob* jobs, const std::uint16_t* x_base,
                              const std::uint8_t* in, void* workspace, const int* count, const int* kind,
                              int wanted, int blocks, cudaStream_t stream) {
    if (!valid(warps, loads) || !jobs || !x_base || !in || !workspace || !count || !kind || blocks < 1 || wanted < 0 || wanted > 1)
        return cudaErrorInvalidValue;
    auto* ws = static_cast<std::uint8_t*>(workspace);
    switch (family) {
        case 0: return detail::queued_family<0, Sm89Build>(warps, loads, jobs, x_base, in, ws, count, kind, wanted, blocks, stream);
        case 1: return detail::queued_family<1, Sm89Build>(warps, loads, jobs, x_base, in, ws, count, kind, wanted, blocks, stream);
        case 2:
            if constexpr (Sm89Build) return detail::queued_family<2, true>(warps, loads, jobs, x_base, in, ws, count, kind, wanted, blocks, stream);
            else return cudaErrorInvalidValue;
        case 3: return detail::queued_family<3, Sm89Build>(warps, loads, jobs, x_base, in, ws, count, kind, wanted, blocks, stream);
        case 4: return detail::queued_family<4, Sm89Build>(warps, loads, jobs, x_base, in, ws, count, kind, wanted, blocks, stream);
        default: return cudaErrorInvalidValue;
    }
}
// Variant 2: `count` jobs, their inputs as packed rows (job j's column k at packed + ((job.x - x_base) / kH + k) *
// kPackedRow); workspace: helper_tc::workspace_bytes(count), 256-aligned (the hidden vectors, as the tensor-core kernel's).
// The outputs equal ffn's (variant 1) bit for bit at the same warps and loads.
inline cudaError_t ffn2(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* packed,
                        void* workspace, cudaStream_t stream) {
    if (!jobs || !packed || !workspace || count < 1 || count > 65535 || !valid(warps, loads) ||
        (reinterpret_cast<std::uintptr_t>(workspace) & 255))
        return cudaErrorInvalidValue;
    switch (warps) {
        case 1: return detail::ffn2_s<1>(loads, jobs, count, x_base, packed, workspace, stream);
        case 2: return detail::ffn2_s<2>(loads, jobs, count, x_base, packed, workspace, stream);
        case 4: return detail::ffn2_s<4>(loads, jobs, count, x_base, packed, workspace, stream);
        default: return detail::ffn2_s<8>(loads, jobs, count, x_base, packed, workspace, stream);
    }
}
// Expand once: each BF16 value exactly represents the original code times its E4M3 scale.
namespace detail {
template <int = 0> __global__ void expand_bf16_kernel(const std::uint8_t* packed, std::uint16_t* out, int columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns * (kH / 16)) return;
    const int col = t / (kH / 16), b = t % (kH / 16);
    const auto* row = packed + std::size_t(col) * kPackedRow;
    const float half_scale = mm3::helper_tc_frozen::detail::half_scale(row[kH / 2 + b] & 0x7f);
    for (int k = 0; k < 16; ++k) {
        const unsigned byte = row[b * 8 + k / 2];
        const unsigned code = (byte >> (4 * (k & 1))) & 15;
        out[std::size_t(col) * kH + b * 16 + k] = row[kH / 2 + b] == 0x7f ? std::uint16_t(0x7fc0) : float_bf16(float(canon::e2m1_x2(code)) * half_scale);
    }
}
template <int = 0> __global__ void rows_to_expanded_kernel(const float* x, int stride, int rows, std::uint16_t* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * (kH / 16)) return;
    const int r = t / (kH / 16), b = t % (kH / 16);
    std::uint16_t v[16];
    for (int k = 0; k < 16; ++k) v[k] = canon::f32_to_bf16_rn(x[std::size_t(r) * stride + b * 16 + k]);
    const canon::A4Codes q = checked_codes(v);
    const float scale = mm3::helper_tc_frozen::detail::half_scale(q.scale_word & 0x7f);
    for (int k = 0; k < 16; ++k) out[std::size_t(r) * kH + b * 16 + k] = q.scale_word == 0x7f ? std::uint16_t(0x7fc0) : float_bf16(float(canon::e2m1_x2(q.code[k])) * scale);
}
}
namespace detail {
template <bool Fp8> __global__ void rows_to_codes_kernel(const float* x, int stride, int rows, std::uint8_t* out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * (kH / 16)) return;
    const int r = t / (kH / 16), b = t % (kH / 16);
    std::uint16_t v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = canon::f32_to_bf16_rn(x[std::size_t(r) * stride + b * 16 + k]);
    auto* row = out + std::size_t(r) * kInStride;
    encode_block<Fp8>(v, row + b * 16, row + kH + b);
}
}
inline cudaError_t rows_to_codes(const float* x, int stride, int rows, std::uint8_t* out, bool fp8, cudaStream_t stream) {
    if (rows < 1) return cudaSuccess;
    if (!x || !out || stride < kH) return cudaErrorInvalidValue;
    const int n = rows * (kH / 16);
    if (fp8) detail::rows_to_codes_kernel<true><<<(n + 127) / 128, 128, 0, stream>>>(x, stride, rows, out);
    else detail::rows_to_codes_kernel<false><<<(n + 127) / 128, 128, 0, stream>>>(x, stride, rows, out);
    return cudaGetLastError();
}
inline cudaError_t rows_to_expanded(const float* x, int stride, int rows, std::uint16_t* out, cudaStream_t stream) {
    if (rows < 1) return cudaSuccess;
    if (!x || !out || stride < kH) return cudaErrorInvalidValue;
    const int n = rows * (kH / 16);
    detail::rows_to_expanded_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(x, stride, rows, out);
    return cudaGetLastError();
}
inline cudaError_t expand_bf16(const std::uint8_t* packed, std::uint16_t* out, int columns, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    if (!packed || !out) return cudaErrorInvalidValue;
    const int n = columns * (kH / 16);
    detail::expand_bf16_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(packed, out, columns);
    return cudaGetLastError();
}
inline cudaError_t ffn3(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base,
                        const std::uint16_t* expanded, void* workspace, cudaStream_t stream) {
    if (!jobs || !expanded || !workspace || count < 1 || count > 65535 || !valid(warps, loads) ||
        (reinterpret_cast<std::uintptr_t>(workspace) & 255)) return cudaErrorInvalidValue;
    auto* in = reinterpret_cast<const std::uint8_t*>(expanded);
    switch (warps) {
        case 1: return detail::ffn2_s<1, true>(loads, jobs, count, x_base, in, workspace, stream);
        case 2: return detail::ffn2_s<2, true>(loads, jobs, count, x_base, in, workspace, stream);
        case 4: return detail::ffn2_s<4, true>(loads, jobs, count, x_base, in, workspace, stream);
        default: return detail::ffn2_s<8, true>(loads, jobs, count, x_base, in, workspace, stream);
    }
}
// BF16 columns -> packed rows on the card (variant 2's inputs for a pass whose rows came as BF16).
inline cudaError_t pack_bf16(const std::uint16_t* x, std::uint8_t* packed, int columns, cudaStream_t stream) {
    if (columns < 1) return cudaSuccess;
    const int n = columns * (kH / 16);
    detail::pack_bf16_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(x, packed, columns);
    return cudaGetLastError();
}
// Source preparation for both cards: quantize each 16-value block once,
// then write only the requested representations. Nullable outputs are independent.
// This prevents the location screen from charging repeated quantization merely
// because one card consumes expanded BF16 and the other consumes byte codes.
template <int = 0>
__global__ void rows_to_formats_kernel(const float* x, std::size_t stride, int rows,
                                      std::uint8_t* packed, std::uint16_t* bf16,
                                      std::uint8_t* s8, std::uint8_t* f8) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * (kH / 16)) return;
    const int r = t / (kH / 16), b = t % (kH / 16);
    std::uint16_t v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = detail::canon::f32_to_bf16_rn(x[std::size_t(r) * stride + b * 16 + k]);
    const detail::canon::A4Codes q = detail::checked_codes(v);
    detail::store_codes(q, b, packed ? packed + std::size_t(r) * kPackedRow : nullptr,
        bf16 ? bf16 + std::size_t(r) * kH : nullptr, s8 ? s8 + std::size_t(r) * kInStride : nullptr,
        f8 ? f8 + std::size_t(r) * kInStride : nullptr);
}
inline cudaError_t rows_to_formats(const float* x, std::size_t stride, int rows,
                                  std::uint8_t* packed, std::uint16_t* bf16,
                                  std::uint8_t* s8, std::uint8_t* f8, cudaStream_t stream) {
    if (rows < 1) return cudaSuccess;
    if (!x || stride < kH || (!packed && !bf16 && !s8 && !f8)) return cudaErrorInvalidValue;
    const int n = rows * (kH / 16);
    rows_to_formats_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(x, stride, rows, packed, bf16, s8, f8);
    return cudaGetLastError();
}
// (the 5090, helper_q_rows) F32 rows -> BF16 (round to nearest even) -> packed NVFP4 rows, the same encoding the cards
// apply to the host's BF16 rows; a thread a block of 16.
template <int = 0>
__global__ void pack_rows_kernel(const float* x, std::size_t x_stride, int rows, std::uint8_t* packed) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * (kH / 16)) return;
    const int r = t / (kH / 16), b = t % (kH / 16);
    std::uint16_t v[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) v[k] = detail::canon::f32_to_bf16_rn(x[std::size_t(r) * x_stride + b * 16 + k]);
    const detail::canon::A4Codes q = detail::checked_codes(v);
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        lo |= (unsigned(q.code[2 * k]) | (unsigned(q.code[2 * k + 1]) << 4)) << (8 * k);
        hi |= (unsigned(q.code[8 + 2 * k]) | (unsigned(q.code[8 + 2 * k + 1]) << 4)) << (8 * k);
    }
    std::uint8_t* dst = packed + std::size_t(r) * kPackedRow;
    *reinterpret_cast<uint2*>(dst + b * 8) = make_uint2(lo, hi);
    dst[kH / 2 + b] = q.scale_word;
}
inline cudaError_t pack_rows(const float* x, std::size_t x_stride, int rows, std::uint8_t* packed, cudaStream_t stream) {
    if (rows < 1) return cudaSuccess;
    const int n = rows * (kH / 16);
    pack_rows_kernel<><<<(n + 127) / 128, 128, 0, stream>>>(x, x_stride, rows, packed);
    return cudaGetLastError();
}
}  // namespace mm3::helper_q_frozen
#endif  // __CUDACC__
