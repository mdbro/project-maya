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
#include "mm3_helper_kernels.cuh"
#include <cstddef>
#include <cstdint>

namespace mm3::helper_q {
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
}  // namespace mm3::helper_q

#if !defined(__CUDACC__)
// The CPU helper tests compile the production helper without a CUDA compiler (tests/mm3_helper_fake_source.cmake) and
// never select these kernels: the same API, every launch refused.
namespace mm3::helper_q {
using infernix::ops::minimax_ffn::FastJob;
inline cudaError_t quant_bf16(const std::uint16_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t expand_packed(const std::uint8_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t ffn(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, std::uint8_t*, void*, cudaStream_t) {
    return cudaErrorInvalidValue;
}
}  // namespace mm3::helper_q
#else
#include "ops/common/canonical_math.h"
namespace mm3::helper_q {
using infernix::ops::minimax_ffn::FastJob;
using infernix::ops::minimax_ffn::float_bf16;
namespace detail {
namespace canon = infernix::ops::canon;
using mm3::helper_tc::detail::fill_scale_table;
using mm3::helper_tc::detail::kScaleTable;
constexpr std::size_t kDownOffset = mm3::helper_tc::kDownOffset;

// One 16-value block of BF16 values encoded (canon::quantize_a4_codes, the 5090's rule) into its q layout: 16 doubled
// values and the scale byte.
__device__ __forceinline__ void encode_block(const std::uint16_t* v, std::uint8_t* codes16, std::uint8_t* scale) {
    const canon::A4Codes q = canon::quantize_a4_codes(v, 1.0f);
    unsigned w[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
        w[k] = canon::e2m1_x2_quad(unsigned(q.code[4 * k]) | (unsigned(q.code[4 * k + 1]) << 8) | (unsigned(q.code[4 * k + 2]) << 16) |
                                   (unsigned(q.code[4 * k + 3]) << 24));
    *reinterpret_cast<uint4*>(codes16) = make_uint4(w[0], w[1], w[2], w[3]);
    *scale = q.scale_word;
}
// BF16 columns [columns][kH] -> q layout [columns][kInStride]; a thread a block of 16
template <int = 0>   // (a template: one definition however many files include this)
__global__ void quant_bf16_kernel(const std::uint16_t* x, std::uint8_t* q, int columns) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns * (kH / 16)) return;
    const int col = t / (kH / 16), b = t % (kH / 16);
    std::uint8_t* dst = q + std::size_t(col) * kInStride;
    encode_block(x + std::size_t(col) * kH + b * 16, dst + b * 16, dst + kH + b);
}
// packed rows [columns][kPackedRow] -> q layout; a thread a block of 16
template <int = 0>
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
        w[2 * k] = canon::e2m1_x2_quad(__byte_perm(lo, hi, 0x5140u));
        w[2 * k + 1] = canon::e2m1_x2_quad(__byte_perm(lo, hi, 0x7362u));
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
// The sums of one warp's share of the inputs for 16 output rows of one job: each unit's int32 product scaled by the
// two half scales into FP32, P units' loads issued before their use (project2's batching).
template <int S, bool Down, int P>
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
            const unsigned a0 = canon::e2m1_x2_quad(w[p] & 0x0f0f0f0fu), a1 = canon::e2m1_x2_quad((w[p] >> 4) & 0x0f0f0f0fu);
            int d0, d1, d2, d3;
            tile_s8(d0, d1, d2, d3, a0, a1, bx[p] & keep);
            const float w0 = scale[sa[p]], w1 = scale[sb[p]], v0 = scale[x0[p]], v1 = scale[x1[p]];
            s0 = fmaf(float(d0), w0 * v0, s0); s1 = fmaf(float(d1), w0 * v1, s1);
            s2 = fmaf(float(d2), w1 * v0, s2); s3 = fmaf(float(d3), w1 * v1, s3);
        }
    }
}
// The up projection (gate and up interleaved by row), two groups of 16 record rows a block (2 S warps), so the block holds
// 16 consecutive hidden units and encodes them itself (as native_fp4.cu's up kernel does); grid (kI / 16, jobs).
template <int S, int P>
__global__ void __launch_bounds__(64 * S) up_kernel(const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* q_in,
                                                    std::uint8_t* hidden_q) {
    const int j = blockIdx.y, pair = blockIdx.x;
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
    sums<S, false, P>(unit, xcol, scol0, scol1, keep, scale, warp, s0, s1, s2, s3);
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
                    const float gate = fminf((half ? (v ? s3 : s2) : (v ? s1 : s0)) * job.multiplier[0], 7.f);
                    const float up = fminf(fmaxf((half ? (v ? u3 : u2) : (v ? u1 : u0)) * job.multiplier[1], -7.f), 7.f);
                    hid[col][8 * local + g / 2 + 4 * half] = float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f));
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < job.columns) {   // the 16 hidden units of this block, one column a thread, encoded as the 5090 does
        const int col = threadIdx.x;
        std::uint8_t* dst = hidden_q + std::size_t(j) * kJobHidden + std::size_t(col) * kHidStride;
        encode_block(hid[col], dst + pair * 16, dst + kI + pair);
    }
}
// The down projection; grid (kH / 16, jobs), S warps a group of 16 output rows.
template <int S, int P>
__global__ void __launch_bounds__(32 * S) down_kernel(const FastJob* jobs, const std::uint8_t* hidden_q) {
    const int j = blockIdx.y, group = blockIdx.x;
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
    sums<S, true, P>(unit, xcol, scol0, scol1, keep, scale, warp, s0, s1, s2, s3);
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
template <int S, int P>
inline cudaError_t ffn_sp(const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in, std::uint8_t* hidden_q,
                          cudaStream_t stream) {
    up_kernel<S, P><<<dim3(kI / 16, count), 64 * S, 0, stream>>>(jobs, x_base, q_in, hidden_q);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    down_kernel<S, P><<<dim3(kH / 16, count), 32 * S, 0, stream>>>(jobs, hidden_q);
    return cudaGetLastError();
}
template <int S>
inline cudaError_t ffn_s(int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                         std::uint8_t* hidden_q, cudaStream_t stream) {
    switch (loads) {
        case 2: return ffn_sp<S, 2>(jobs, count, x_base, q_in, hidden_q, stream);
        case 4: return ffn_sp<S, 4>(jobs, count, x_base, q_in, hidden_q, stream);
        case 8: return ffn_sp<S, 8>(jobs, count, x_base, q_in, hidden_q, stream);
        default: return ffn_sp<S, 1>(jobs, count, x_base, q_in, hidden_q, stream);
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
    const detail::canon::A4Codes q = detail::canon::quantize_a4_codes(v, 1.0f);
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
}  // namespace mm3::helper_q
#endif  // __CUDACC__
