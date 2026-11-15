// Kernel item 7 (mm3_native_fp4_fast.cuh): the donor's native FP4 expert kernel (Infernix-minimax
// ops/minimax_nvfp4_ffn/native_fp4.cu at 5c017634), copied, with each warp's loads of P steps issued before their use.
// Built for sm_120a only (the block-scaled FP4 tensor-core instruction), as the donor's library is.
#include "mm3_native_fp4_fast.cuh"
#include "ops/minimax_nvfp4_ffn/fast_math.h"
#include "ops/common/canonical_math.h"
#include "ops/common/mma.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace mm3::fp4_fast {
namespace {
namespace c = infernix::ops::canon;
using infernix::ops::minimax_ffn::float_bf16;
// (the donor's constants and layout: an encoded row of the inputs, XR bytes; of the hidden vector, HR)
constexpr int H = 6144, I = 3072, U = 144, XR = H / 2 + H / 16, HR = I / 2 + I / 16;
constexpr std::size_t GU = std::size_t(2 * I / 16) * (H / 16) * U;
__host__ __device__ constexpr std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t(255); }
__host__ __device__ constexpr int position(int b) { return (b & ~3) | ((b & 1) << 1) | ((b & 2) >> 1); }
std::size_t workspace_bytes(int columns) { return align256(std::size_t(columns) * XR) + align256(std::size_t(columns) * HR); }
__device__ void store_block(std::uint8_t* row, int codebytes, int block, const c::A4Codes& q) {
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        lo |= (unsigned(q.code[j]) | (unsigned(q.code[4 + j]) << 4)) << (8 * j);
        hi |= (unsigned(q.code[8 + j]) | (unsigned(q.code[12 + j]) << 4)) << (8 * j);
    }
    const int p = position(block);
    *reinterpret_cast<uint2*>(row + 8 * p) = make_uint2(lo, hi);
    row[codebytes + p] = q.scale_word;
}
template <int = 0>
__global__ void prepare(const FastJob* jobs, int cols, std::uint8_t* work, std::size_t per) {
    int j = blockIdx.y, task = blockIdx.x * blockDim.x + threadIdx.x;
    auto job = jobs[j];
    if (!job.columns || task >= job.columns * (H / 16)) return;
    int col = task / (H / 16), block = task % (H / 16);
    auto q = c::quantize_a4_codes(job.x + std::size_t(col) * H + block * 16, 1.f);
    store_block(work + std::size_t(j) * per + std::size_t(col) * XR, H / 2, block, q);
}
// The donor's project<Down>, its step loop batched by P.
template <bool Down, int P>
__global__ __launch_bounds__(128, 3) void project(const FastJob* jobs, int cols, std::uint8_t* work, std::size_t per) {
    constexpr int K = Down ? I : H, RB = Down ? HR : XR, CB = K / 2;
    int jobid = blockIdx.y;
    auto job = jobs[jobid];
    if (!job.columns) return;
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31, q = lane / 4, tc = lane & 3;
    int local = warp / 2, seg = warp & 1, group = blockIdx.x * 2 + local;
    auto* base = work + std::size_t(jobid) * per;
    auto* x = Down ? base + align256(std::size_t(cols) * XR) : base;
    auto* input = q < job.columns ? x + std::size_t(q) * RB : nullptr;
    const auto* matrix = job.record + (Down ? GU : 0);
    float sum[4]{};
    const int k_begin = (K / 64) * seg / 2, k_end = (K / 64) * (seg + 1) / 2;
    static_assert((K / 64) % (2 * P) == 0, "a warp's steps must split into batches of P");
#pragma unroll 1
    for (int k0 = k_begin; k0 < k_end; k0 += P) {
        unsigned w0[P], w1[P], w2[P], w3[P], s[P][4], b0[P], b1[P], sb[P];
#pragma unroll
        for (int p = 0; p < P; ++p) {   // (every load of the batch first)
            const int k = k0 + p;
            const auto* units = matrix + (std::size_t(group) * (K / 16) + 4 * k) * U;
            const auto* unit = units + 2 * (tc >> 1) * U + 64 * (tc & 1) + 4 * q;
            w0[p] = *reinterpret_cast<const unsigned*>(unit);
            w1[p] = *reinterpret_cast<const unsigned*>(unit + 32);
            w2[p] = *reinterpret_cast<const unsigned*>(unit + U);
            w3[p] = *reinterpret_cast<const unsigned*>(unit + U + 32);
#pragma unroll
            for (int b = 0; b < 4; ++b) s[p][b] = units[position(b) * U + 128 + q + 8 * (tc & 1)];
            const int p0 = 4 * k + position(2 * (tc >> 1)), p1 = 4 * k + position(2 * (tc >> 1) + 1);
            b0[p] = input ? *reinterpret_cast<const unsigned*>(input + 8 * p0 + 4 * (tc & 1)) : 0;
            b1[p] = input ? *reinterpret_cast<const unsigned*>(input + 8 * p1 + 4 * (tc & 1)) : 0;
            sb[p] = input ? *reinterpret_cast<const unsigned*>(input + CB + 4 * k) : 0;
        }
#pragma unroll
        for (int p = 0; p < P; ++p) {   // (then the donor's products, in the donor's order)
            const unsigned a0 = (w0[p] & 0x0f0f0f0fu) | ((w1[p] & 0x0f0f0f0fu) << 4), a1 = ((w0[p] >> 4) & 0x0f0f0f0fu) | (w1[p] & 0xf0f0f0f0u),
                           a2 = (w2[p] & 0x0f0f0f0fu) | ((w3[p] & 0x0f0f0f0fu) << 4), a3 = ((w2[p] >> 4) & 0x0f0f0f0fu) | (w3[p] & 0xf0f0f0f0u);
            unsigned sa = 0;
#pragma unroll
            for (int b = 0; b < 4; ++b) sa |= s[p][b] << (8 * b);
            infernix::ops::mma_nvfp4_e4m3(sum[0], sum[1], sum[2], sum[3], a0, a1, a2, a3, b0[p], b1[p], sa, sb[p]);
        }
    }
    __shared__ float part[2][4][32];
    if (seg) for (int v = 0; v < 4; ++v) part[local][v][lane] = sum[v];
    __syncthreads();
    if (!seg) for (int v = 0; v < 4; ++v) sum[v] += part[local][v][lane];
    if constexpr (Down) {
        if (seg) return;
        for (int half = 0; half < 2; ++half)
            for (int v = 0; v < 2; ++v)
                if (2 * tc + v < job.columns) job.y[std::size_t(2 * tc + v) * H + group * 16 + q + 8 * half] = float_bf16(sum[2 * half + v] * job.multiplier[2]);
    } else {
        __shared__ std::uint16_t hidden[8][16];
        if (!seg) {
            float up[4];
            for (int v = 0; v < 4; ++v) up[v] = __shfl_xor_sync(0xffffffffu, sum[v], 4);
            if (!(q & 1))
                for (int half = 0; half < 2; ++half)
                    for (int v = 0; v < 2; ++v)
                        if (2 * tc + v < job.columns) {
                            float g = fminf(sum[2 * half + v] * job.multiplier[0], 7.f);
                            float u = fminf(fmaxf(up[2 * half + v] * job.multiplier[1], -7.f), 7.f);
                            hidden[2 * tc + v][8 * local + q / 2 + 4 * half] = float_bf16(g / (1.f + expf(-1.702f * g)) * (u + 1.f));
                        }
        }
        __syncthreads();
        if (threadIdx.x < job.columns) {
            int col = threadIdx.x;
            auto enc = c::quantize_a4_codes(hidden[col], 1.f);
            auto* row = base + align256(std::size_t(cols) * XR) + std::size_t(col) * HR;
            store_block(row, I / 2, blockIdx.x, enc);
        }
    }
}
void check() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template <int P> void run(const FastJob* jobs, int count, int cols, std::uint8_t* work, std::size_t per, cudaStream_t stream) {
    prepare<><<<dim3((cols * (H / 16) + 127) / 128, count), 128, 0, stream>>>(jobs, cols, work, per); check();
    project<false, P><<<dim3(2 * I / 32, count), 128, 0, stream>>>(jobs, cols, work, per); check();
    project<true, P><<<dim3(H / 32, count), 128, 0, stream>>>(jobs, cols, work, per); check();
}
}  // namespace

void ffn(const FastJob* jobs, int count, int cols, void* work, int loads, cudaStream_t stream) {
    if (cols < 1 || cols > 8) throw std::invalid_argument("native FP4 columns must be 1..8");
    const std::size_t per = workspace_bytes(cols);
    if (!jobs || !work || count < 1 || count > 65535 || (reinterpret_cast<std::uintptr_t>(work) & 255) || !valid_loads(loads))
        throw std::invalid_argument("invalid native FP4 operands");
    auto* w = static_cast<std::uint8_t*>(work);
    switch (loads) {
        case 2: run<2>(jobs, count, cols, w, per, stream); break;
        case 4: run<4>(jobs, count, cols, w, per, stream); break;
        case 8: run<8>(jobs, count, cols, w, per, stream); break;
        default: run<1>(jobs, count, cols, w, per, stream); break;
    }
}
}  // namespace mm3::fp4_fast
