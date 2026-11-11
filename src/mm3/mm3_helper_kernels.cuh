// Expert kernels for the helper cards (RTX 3090, sm_86; RTX 4080, sm_89). The user's direction of 2026-11-11: every
// device gets its own kernel with the arithmetic that is fastest on it; no lane has to reproduce another's output.
// Plan: docs/minimax-wsl/README.md#device-kernels. Two kernels, measured against each other by
// tools/mm3/bench/helper_tc_bench.cu: helper_tc (below: tensor cores, BF16 activations used exactly) and helper_dp
// (at the end: today's dp4a arithmetic compiled per job width, same output as today).
//
// What it replaces on these cards: fast_ffn_cuda_packed (Infernix, minimax_nvfp4_ffn/fast_cuda.cu), which quantizes
// activations to INT8 and does four dp4a for every row a job serves, so an expert's cost grows with its rows.
//
// Arithmetic here ("W4 x BF16"): the stored 4-bit codes are used exactly (decoded to BF16, which holds every code
// exactly), the BF16 activations are used as they are (no quantization, no conversion), each 16 x 16 tile of weights
// is multiplied with up to eight columns in one tensor-core product with FP32 accumulation, and the tile's per-row
// E4M3 scales are applied to the FP32 result. The hidden vector is rounded to BF16 as before and used exactly.
//
// The record (record.h, RG16) is already laid out in tiles: a unit is 144 bytes = 16 output rows x 16 inputs:
//   byte[q*32 + r*4 + t], q 0..3, r 0..7, t 0..3: low nibble the code of row r at input 4q+t, high nibble row r+8's
//   byte[128 + row]: that row's E4M3 scale for these 16 inputs
// A code is sign (bit 3) and a 3-bit index into the doubled values {0,1,2,3,4,6,8,12}; a weight is code * scale / 2.
// Gate and up are interleaved by row (even rows gate, odd rows up): rows 2p and 2p+1 of a group give hidden unit p.
//
// Same FastJob contract as fast_ffn_cuda: BF16 x and y, column-major [columns, 6144]; 1 to 8 columns a job, and jobs
// of different widths may share a launch (a zero-column job is inactive). Record pointers 8-byte aligned.
#pragma once
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/fast_math.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace mm3::helper_tc {
using infernix::ops::minimax_ffn::FastJob;
using infernix::ops::minimax_ffn::float_bf16;

constexpr int kH = 6144, kI = 3072, kUnit = 144, kMaxColumns = 8;
constexpr std::size_t kDownOffset = std::size_t(2 * kI / 16) * (kH / 16) * kUnit;   // the down matrix in a record
// workspace of one job: its hidden vectors, BF16, [8][kI]
constexpr std::size_t kJobWorkspace = (std::size_t(kMaxColumns) * kI * 2 + 255) & ~std::size_t(255);
inline std::size_t workspace_bytes(int jobs) { return std::size_t(jobs) * kJobWorkspace; }

namespace detail {
// D = A B, a 16 x 16 tile of BF16 times 16 x 8 BF16, FP32 result. Per thread (g = lane / 4, c = lane % 4):
//   a0: row g, inputs 2c and 2c+1;  a1: row g+8, the same inputs;  a2, a3: the same rows, inputs 2c+8 and 2c+9
//   b0: column g, inputs 2c and 2c+1;  b1: column g, inputs 2c+8 and 2c+9
//   d0, d1: row g, columns 2c and 2c+1;  d2, d3: row g+8, the same columns
__device__ __forceinline__ void tile(float& d0, float& d1, float& d2, float& d3, unsigned a0, unsigned a1, unsigned a2,
                                     unsigned a3, unsigned b0, unsigned b1) {
    d0 = d1 = d2 = d3 = 0.f;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// Four codes, one in the low nibble of each byte, to two registers of two BF16 each: {k0, k1} and {k2, k3}.
// BF16 of the doubled values 0,1,2,3,4,6,8,12: 0000 3F80 4000 4040 4080 40C0 4100 4140; the sign goes to bit 15.
__device__ __forceinline__ void decode4(unsigned codes, unsigned& r01, unsigned& r23) {
    const unsigned pairs = codes | (codes >> 4);
    const unsigned select = ((pairs & 0xffu) | ((pairs >> 8) & 0xff00u)) & 0x7777u;
    const unsigned high = __byte_perm(0x40403F00u, 0x41414040u, select) | ((codes & 0x08080808u) << 4);
    const unsigned low = __byte_perm(0x40008000u, 0x4000C080u, select);
    r01 = __byte_perm(low, high, 0x5140u);
    r23 = __byte_perm(low, high, 0x7362u);
}
// An E4M3 scale byte (never negative or NaN in a valid record) times one half, for the doubled codes.
__device__ __forceinline__ float half_scale(unsigned x) {
    const unsigned e = x >> 3, m = x & 7u;
    return float(e ? (8u + m) << (e - 1) : m) * 0x1p-10f;
}

// One block is one group of 16 output rows of one job; its S warps split the inputs and add up at the end.
template<int S, bool Down>
__global__ void __launch_bounds__(32 * S) project(const FastJob* jobs, std::uint8_t* workspace) {
    constexpr int K = Down ? kI : kH, blocks = K / 16;
    const int j = blockIdx.y, group = blockIdx.x;
    const FastJob job = jobs[j];
    if (job.columns == 0) return;                               // an inactive slot; uniform over the block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    std::uint16_t* hidden = reinterpret_cast<std::uint16_t*>(workspace + std::size_t(j) * kJobWorkspace);
    const std::uint16_t* x = Down ? hidden : job.x;
    const bool live = g < job.columns;                          // this thread's column of the activations
    const unsigned keep = live ? ~0u : 0u;
    const unsigned* column = reinterpret_cast<const unsigned*>(x + std::size_t(live ? g : 0) * K) + c;
    const std::uint8_t* unit = job.record + (Down ? kDownOffset : 0) + (std::size_t(group) * blocks + std::size_t(blocks) * warp / S) * kUnit;
    // where this thread's codes sit in a unit
    const int quad = (c >> 1) * 32, shift = 16 * (c & 1);
    const int pair_row = ((2 * g) & 7) * 4, pair_nibble = g >= 4 ? 4 : 0;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll 1
    for (int b = blocks * warp / S; b < blocks * (warp + 1) / S; ++b, unit += kUnit) {
        unsigned a0, a1, a2, a3;
        float w0, w1;
        if constexpr (Down) {
            // rows g and g+8 share their bytes: inputs 2c, 2c+1 in one 32-bit word, inputs 2c+8, 2c+9 in another
            const unsigned lo = *reinterpret_cast<const unsigned*>(unit + quad + g * 4) >> shift;
            const unsigned hi = *reinterpret_cast<const unsigned*>(unit + quad + 64 + g * 4) >> shift;
            const unsigned word = (lo & 0xffffu) | (hi << 16);
            decode4(word & 0x0f0f0f0fu, a0, a2);
            decode4((word >> 4) & 0x0f0f0f0fu, a1, a3);
            w0 = half_scale(unit[128 + g]);
            w1 = half_scale(unit[136 + g]);
        } else {
            // tile row g is the gate row of hidden unit g (record row 2g), tile row g+8 its up row (record row 2g+1):
            // both land in one thread, so the activation needs no exchange between threads
            const unsigned long long lo = *reinterpret_cast<const unsigned long long*>(unit + quad + pair_row) >> shift;
            const unsigned long long hi = *reinterpret_cast<const unsigned long long*>(unit + quad + 64 + pair_row) >> shift;
            const unsigned gate = (unsigned(lo) & 0xffffu) | (unsigned(hi) << 16);
            const unsigned up = (unsigned(lo >> 32) & 0xffffu) | (unsigned(hi >> 32) << 16);
            decode4((gate >> pair_nibble) & 0x0f0f0f0fu, a0, a2);
            decode4((up >> pair_nibble) & 0x0f0f0f0fu, a1, a3);
            const unsigned scales = *reinterpret_cast<const std::uint16_t*>(unit + 128 + 2 * g);
            w0 = half_scale(scales & 0xffu);
            w1 = half_scale(scales >> 8);
        }
        const unsigned b0 = column[b * 8] & keep, b1 = column[b * 8 + 4] & keep;
        float d0, d1, d2, d3;
        tile(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        s0 = fmaf(d0, w0, s0); s1 = fmaf(d1, w0, s1);
        s2 = fmaf(d2, w1, s2); s3 = fmaf(d3, w1, s3);
    }
    if constexpr (S > 1) {
        __shared__ float part[S][4][32];
        part[warp][0][lane] = s0; part[warp][1][lane] = s1; part[warp][2][lane] = s2; part[warp][3][lane] = s3;
        __syncthreads();
        if (warp) return;
#pragma unroll
        for (int w = 1; w < S; ++w) { s0 += part[w][0][lane]; s1 += part[w][1][lane]; s2 += part[w][2][lane]; s3 += part[w][3][lane]; }
    }
    // this thread holds columns 2c and 2c+1 of tile rows g (s0, s1) and g+8 (s2, s3)
    if constexpr (Down) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int col = 2 * c + v;
            if (col >= job.columns) continue;
            std::uint16_t* y = job.y + std::size_t(col) * kH + group * 16 + g;
            y[0] = float_bf16((v ? s1 : s0) * job.multiplier[2]);
            y[8] = float_bf16((v ? s3 : s2) * job.multiplier[2]);
        }
    } else {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int col = 2 * c + v;
            if (col >= job.columns) continue;
            const float gate = fminf((v ? s1 : s0) * job.multiplier[0], 7.f);
            const float up = fminf(fmaxf((v ? s3 : s2) * job.multiplier[1], -7.f), 7.f);
            hidden[std::size_t(col) * kI + group * 8 + g] = float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f));
        }
    }
}

// The same product with fewer instructions a unit (variant 1): the E4M3 scales come from a 128-entry table in shared
// memory instead of being converted each time, and gate/up use the record's own row order (a tile row and the row
// eight below share their bytes, as in the down matrix), so a unit costs two 32-bit loads; the gate row and the up
// row of a hidden unit then sit in neighbouring thread groups and are brought together once, after the sums.
template<int S, bool Down>
__global__ void __launch_bounds__(32 * S) project1(const FastJob* jobs, std::uint8_t* workspace) {
    constexpr int K = Down ? kI : kH, blocks = K / 16;
    const int j = blockIdx.y, group = blockIdx.x;
    const FastJob job = jobs[j];
    if (job.columns == 0) return;
    __shared__ float scale[128];
    for (int i = threadIdx.x; i < 128; i += 32 * S) scale[i] = half_scale(unsigned(i));
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    std::uint16_t* hidden = reinterpret_cast<std::uint16_t*>(workspace + std::size_t(j) * kJobWorkspace);
    const std::uint16_t* x = Down ? hidden : job.x;
    const bool live = g < job.columns;
    const unsigned keep = live ? ~0u : 0u;
    const unsigned* column = reinterpret_cast<const unsigned*>(x + std::size_t(live ? g : 0) * K) + c;
    const std::uint8_t* unit = job.record + (Down ? kDownOffset : 0) + (std::size_t(group) * blocks + std::size_t(blocks) * warp / S) * kUnit;
    const int at = (c >> 1) * 32 + g * 4, shift = 16 * (c & 1);
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll 1
    for (int b = blocks * warp / S; b < blocks * (warp + 1) / S; ++b, unit += kUnit) {
        const unsigned lo = *reinterpret_cast<const unsigned*>(unit + at) >> shift;
        const unsigned hi = *reinterpret_cast<const unsigned*>(unit + at + 64) >> shift;
        const unsigned word = (lo & 0xffffu) | (hi << 16);
        unsigned a0, a1, a2, a3;
        decode4(word & 0x0f0f0f0fu, a0, a2);
        decode4((word >> 4) & 0x0f0f0f0fu, a1, a3);
        const float w0 = scale[unit[128 + g]], w1 = scale[unit[136 + g]];
        const unsigned b0 = column[b * 8] & keep, b1 = column[b * 8 + 4] & keep;
        float d0, d1, d2, d3;
        tile(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        s0 = fmaf(d0, w0, s0); s1 = fmaf(d1, w0, s1);
        s2 = fmaf(d2, w1, s2); s3 = fmaf(d3, w1, s3);
    }
    if constexpr (S > 1) {
        __shared__ float part[S][4][32];
        part[warp][0][lane] = s0; part[warp][1][lane] = s1; part[warp][2][lane] = s2; part[warp][3][lane] = s3;
        __syncthreads();
        if (warp) return;
#pragma unroll
        for (int w = 1; w < S; ++w) { s0 += part[w][0][lane]; s1 += part[w][1][lane]; s2 += part[w][2][lane]; s3 += part[w][3][lane]; }
    }
    if constexpr (Down) {
#pragma unroll
        for (int v = 0; v < 2; ++v) {
            const int col = 2 * c + v;
            if (col >= job.columns) continue;
            std::uint16_t* y = job.y + std::size_t(col) * kH + group * 16 + g;
            y[0] = float_bf16((v ? s1 : s0) * job.multiplier[2]);
            y[8] = float_bf16((v ? s3 : s2) * job.multiplier[2]);
        }
    } else {
        // record rows g and g+8 of this group: an even row is a gate row, the next one its up row (four lanes on)
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
                    hidden[std::size_t(col) * kI + group * 8 + g / 2 + 4 * half] = float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f));
                }
            }
        }
    }
}
}  // namespace detail

// Variant 1 of the same kernel (see project1): the same contract and the same arithmetic.
template<int S>
inline cudaError_t ffn1(const FastJob* device_jobs, int count, void* workspace, cudaStream_t stream) {
    if (!device_jobs || !workspace || count < 1 || count > 65535 || (reinterpret_cast<std::uintptr_t>(workspace) & 255)) return cudaErrorInvalidValue;
    detail::project1<S, false><<<dim3(2 * kI / 16, count), 32 * S, 0, stream>>>(device_jobs, static_cast<std::uint8_t*>(workspace));
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    detail::project1<S, true><<<dim3(kH / 16, count), 32 * S, 0, stream>>>(device_jobs, static_cast<std::uint8_t*>(workspace));
    return cudaGetLastError();
}

// Asynchronous on the stream. `count` jobs of any widths 1..8 (0: inactive); workspace of workspace_bytes(count),
// 256-byte aligned, disjoint from every operand. S: warps a group of 16 output rows (1, 2, 4 or 8), tuned per card.
template<int S>
inline cudaError_t ffn(const FastJob* device_jobs, int count, void* workspace, cudaStream_t stream) {
    if (!device_jobs || !workspace || count < 1 || count > 65535 || (reinterpret_cast<std::uintptr_t>(workspace) & 255)) return cudaErrorInvalidValue;
    detail::project<S, false><<<dim3(2 * kI / 16, count), 32 * S, 0, stream>>>(device_jobs, static_cast<std::uint8_t*>(workspace));
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
    detail::project<S, true><<<dim3(kH / 16, count), 32 * S, 0, stream>>>(device_jobs, static_cast<std::uint8_t*>(workspace));
    return cudaGetLastError();
}
}  // namespace mm3::helper_tc

// The shared dp4a arithmetic (INT8 activations in blocks of 16, exact integer products, FP32 sums), compiled for each
// job width. fast_ffn_cuda_packed loops over a job's columns at run time with its sums in an array indexed by the
// loop, which costs about as much for every further column as the whole first one. Here the width is a template
// argument: the column loop is unrolled and the sums stay in registers. Same expressions in the same order, so the
// output is meant to be bit-identical to fast_ffn_cuda_packed's (the benchmark checks it); same workspace layout.
// ffn: every job of a launch has exactly C columns (the helper already launches one group a width). ffn_any: jobs of
// any widths in one launch, each block running the code compiled for its job's width. 0 columns: inactive.
namespace mm3::helper_dp {
using infernix::ops::minimax_ffn::FastJob;
using infernix::ops::minimax_ffn::bf16_float;
using infernix::ops::minimax_ffn::float_bf16;
using mm3::helper_tc::kH;
using mm3::helper_tc::kI;
using mm3::helper_tc::kUnit;
using mm3::helper_tc::kDownOffset;
struct QBlock { signed char q[16]; float scale; };
__host__ __device__ constexpr std::size_t align256(std::size_t n) { return (n + 255) & ~std::size_t(255); }
__host__ __device__ constexpr std::size_t job_workspace(int cols) {
    return align256(std::size_t(cols) * (kH / 16) * sizeof(QBlock)) + align256(std::size_t(cols) * (kI / 16) * sizeof(QBlock)) +
           align256(std::size_t(cols) * kI * 2);
}
namespace detail {
struct Workspace { QBlock *x, *hq; std::uint16_t* hidden; };
template<int C> __device__ __forceinline__ Workspace carve(void* base, int j) {
    constexpr std::size_t nx = align256(std::size_t(C) * (kH / 16) * sizeof(QBlock)), nh = align256(std::size_t(C) * (kI / 16) * sizeof(QBlock));
    auto* p = static_cast<std::uint8_t*>(base) + std::size_t(j) * job_workspace(C);
    return {reinterpret_cast<QBlock*>(p), reinterpret_cast<QBlock*>(p + nx), reinterpret_cast<std::uint16_t*>(p + nx + nh)};
}
template<bool Hidden, int C> __global__ void quant(const FastJob* jobs, void* scratch) {
    constexpr int blocks = (Hidden ? kI : kH) / 16;
    const int j = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= C * blocks) return;
    const FastJob job = jobs[j];
    if (job.columns == 0) return;
    const Workspace w = carve<C>(scratch, j);
    const std::uint16_t* x = Hidden ? w.hidden : job.x;
    QBlock* dst = Hidden ? w.hq : w.x;
    float m = 0;
    for (int t = 0; t < 16; ++t) m = fmaxf(m, fabsf(bf16_float(x[k * 16 + t])));
    const float s = m / 127.f;
    dst[k].scale = s;
    for (int t = 0; t < 16; ++t) dst[k].q[t] = s ? (signed char)max(-127, min(127, __float2int_rn(bf16_float(x[k * 16 + t]) / s))) : 0;
}
// four codes of one row (the low or the high nibbles of a word) as four signed bytes of the doubled values
__device__ __forceinline__ int packed_weights(unsigned word, int row) {
    const unsigned codes = (word >> (row >= 8 ? 4 : 0)) & 0x0f0f0f0fu;
    const unsigned pairs = codes | (codes >> 4);
    const unsigned selector = ((pairs & 0xffu) | ((pairs >> 8) & 0xff00u)) & 0x7777u;
    const unsigned positive = __byte_perm(0x03020100u, 0x0c080604u, selector);
    const unsigned negative = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, selector);
    const unsigned sign = ((codes >> 3) & 0x01010101u) * 0xffu;
    return int((positive & ~sign) | (negative & sign));
}
__device__ __forceinline__ float packed_scale(unsigned x) {
    const unsigned e = x >> 3, m = x & 7u;
    return float(e ? (8u + m) << (e - 1) : m) * 0x1p-10f;
}
// The sums of one thread = (row, one of eight interleaved parts of the inputs) of a group of 16 output rows, for a
// job of exactly C columns; then the group's results. 128 threads a block; `partial` is the block's shared scratch.
template<bool Down, int C>
__device__ __forceinline__ void rows(const FastJob& job, const Workspace& w, float (*partial)[8][16]) {
    constexpr int K = Down ? kI : kH, blocks = K / 16;
    const int group = blockIdx.x, row = threadIdx.x % 16, part = threadIdx.x / 16;
    const std::uint8_t* matrix = job.record + (Down ? kDownOffset : 0) + std::size_t(group) * blocks * kUnit;
    const QBlock* acts = Down ? w.hq : w.x;
    float sum[C];
#pragma unroll
    for (int c = 0; c < C; ++c) sum[c] = 0.f;
    for (int b = part; b < blocks; b += 8) {
        const std::uint8_t* unit = matrix + std::size_t(b) * kUnit;
        int weights[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) weights[q] = packed_weights(*reinterpret_cast<const unsigned*>(unit + q * 32 + (row % 8) * 4), row);
        const float ws = packed_scale(unit[128 + row]);
#pragma unroll
        for (int c = 0; c < C; ++c) {
            const QBlock& a = acts[c * blocks + b];
            const int* aq = reinterpret_cast<const int*>(a.q);
            int dot = 0;
#pragma unroll
            for (int q = 0; q < 4; ++q) dot = __dp4a(weights[q], aq[q], dot);
            sum[c] += float(dot) * ws * a.scale;
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) partial[part][c][row] = sum[c];
    __syncthreads();
    if (part == 0) {
#pragma unroll
        for (int c = 0; c < C; ++c) {
            float total = 0;
            for (int p = 0; p < 8; ++p) total += partial[p][c][row];
            if constexpr (Down) job.y[std::size_t(c) * kH + group * 16 + row] = float_bf16(total * job.multiplier[2]);
            else partial[0][c][row] = total * job.multiplier[row & 1];
        }
    }
    if constexpr (!Down) {
        __syncthreads();
        if (part == 0 && row < 8) {
#pragma unroll
            for (int c = 0; c < C; ++c) {
                const float g = fminf(partial[0][c][2 * row], 7.f), u = fminf(fmaxf(partial[0][c][2 * row + 1], -7.f), 7.f);
                w.hidden[std::size_t(c) * kI + group * 8 + row] = float_bf16(g / (1.f + expf(-1.702f * g)) * (u + 1.f));
            }
        }
    }
}
// every job of the launch has C columns; the workspace is laid out for C
template<bool Down, int C> __global__ void __launch_bounds__(128) project(const FastJob* jobs, void* scratch) {
    const FastJob job = jobs[blockIdx.y];
    if (job.columns == 0) return;
    __shared__ float partial[8][8][16];
    rows<Down, C>(job, carve<C>(scratch, blockIdx.y), partial);
}
template<int C> inline cudaError_t ffn(const FastJob* jobs, int count, void* workspace, cudaStream_t stream) {
    quant<false, C><<<dim3((C * (kH / 16) + 127) / 128, count), 128, 0, stream>>>(jobs, workspace);
    project<false, C><<<dim3(2 * kI / 16, count), 128, 0, stream>>>(jobs, workspace);
    quant<true, C><<<dim3((C * (kI / 16) + 127) / 128, count), 128, 0, stream>>>(jobs, workspace);
    project<true, C><<<dim3(kH / 16, count), 128, 0, stream>>>(jobs, workspace);
    return cudaGetLastError();
}
// Jobs of any widths in one launch: the workspace is laid out for 8 columns a job, and each block runs the code
// compiled for its own job's width (a block serves one job, so the choice is uniform over the block).
template<bool Hidden> __global__ void quant_any(const FastJob* jobs, void* scratch) {
    constexpr int blocks = (Hidden ? kI : kH) / 16;
    const int j = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    const FastJob job = jobs[j];
    if (k >= job.columns * blocks) return;                     // (also an inactive job: 0 columns)
    const Workspace w = carve<8>(scratch, j);
    const std::uint16_t* x = Hidden ? w.hidden : job.x;
    QBlock* dst = Hidden ? w.hq : w.x;
    float m = 0;
    for (int t = 0; t < 16; ++t) m = fmaxf(m, fabsf(bf16_float(x[k * 16 + t])));
    const float s = m / 127.f;
    dst[k].scale = s;
    for (int t = 0; t < 16; ++t) dst[k].q[t] = s ? (signed char)max(-127, min(127, __float2int_rn(bf16_float(x[k * 16 + t]) / s))) : 0;
}
template<bool Down> __global__ void __launch_bounds__(128) project_any(const FastJob* jobs, void* scratch) {
    const FastJob job = jobs[blockIdx.y];
    if (job.columns == 0) return;
    __shared__ float partial[8][8][16];
    const Workspace w = carve<8>(scratch, blockIdx.y);
    switch (job.columns) {
        case 1: rows<Down, 1>(job, w, partial); break;
        case 2: rows<Down, 2>(job, w, partial); break;
        case 3: rows<Down, 3>(job, w, partial); break;
        case 4: rows<Down, 4>(job, w, partial); break;
        case 5: rows<Down, 5>(job, w, partial); break;
        case 6: rows<Down, 6>(job, w, partial); break;
        case 7: rows<Down, 7>(job, w, partial); break;
        default: rows<Down, 8>(job, w, partial); break;
    }
}
}  // namespace detail
// Asynchronous on the stream: `count` jobs that all have `columns` columns (or 0: inactive); workspace of
// count * job_workspace(columns), 256-byte aligned (the same sizes as fast_workspace_bytes).
inline cudaError_t ffn(const FastJob* device_jobs, int count, int columns, void* workspace, cudaStream_t stream) {
    if (!device_jobs || !workspace || count < 1 || count > 65535 || (reinterpret_cast<std::uintptr_t>(workspace) & 255)) return cudaErrorInvalidValue;
    switch (columns) {
        case 1: return detail::ffn<1>(device_jobs, count, workspace, stream);
        case 2: return detail::ffn<2>(device_jobs, count, workspace, stream);
        case 3: return detail::ffn<3>(device_jobs, count, workspace, stream);
        case 4: return detail::ffn<4>(device_jobs, count, workspace, stream);
        case 5: return detail::ffn<5>(device_jobs, count, workspace, stream);
        case 6: return detail::ffn<6>(device_jobs, count, workspace, stream);
        case 7: return detail::ffn<7>(device_jobs, count, workspace, stream);
        case 8: return detail::ffn<8>(device_jobs, count, workspace, stream);
        default: return cudaErrorInvalidValue;
    }
}
// Asynchronous on the stream: `count` jobs of any widths 1..8 (0: inactive) in one launch of four kernels; workspace
// of count * job_workspace(8), 256-byte aligned.
inline cudaError_t ffn_any(const FastJob* device_jobs, int count, void* workspace, cudaStream_t stream) {
    if (!device_jobs || !workspace || count < 1 || count > 65535 || (reinterpret_cast<std::uintptr_t>(workspace) & 255)) return cudaErrorInvalidValue;
    detail::quant_any<false><<<dim3((8 * (kH / 16) + 127) / 128, count), 128, 0, stream>>>(device_jobs, workspace);
    detail::project_any<false><<<dim3(2 * kI / 16, count), 128, 0, stream>>>(device_jobs, workspace);
    detail::quant_any<true><<<dim3((8 * (kI / 16) + 127) / 128, count), 128, 0, stream>>>(device_jobs, workspace);
    detail::project_any<true><<<dim3(kH / 16, count), 128, 0, stream>>>(device_jobs, workspace);
    return cudaGetLastError();
}
}  // namespace mm3::helper_dp
