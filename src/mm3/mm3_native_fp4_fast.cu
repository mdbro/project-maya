// Kernel item 7 (mm3_native_fp4_fast.cuh): the donor's native FP4 expert kernel (Infernix-minimax
// ops/minimax_nvfp4_ffn/native_fp4.cu at 5c017634), copied, with each warp's loads of P steps issued before their use.
// Built for sm_120a only (the block-scaled FP4 tensor-core instruction), as the donor's library is.
#include "mm3_native_fp4_fast.cuh"
#include "mm3_native_fp4_rows.cuh"
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
template <bool Down, int P, bool ParallelHidden = false, bool SharedRows = false>
__global__ __launch_bounds__(128, 3) void project(const FastJob* jobs, int cols, std::uint8_t* work, std::size_t per,
                                                const std::uint8_t* packed_rows = nullptr, const int* row_ids = nullptr,
                                                int packed_row_count = 0) {
    constexpr int K = Down ? I : H, RB = Down ? HR : XR, CB = K / 2;
    int jobid = blockIdx.y;
    auto job = jobs[jobid];
    if (!job.columns) return;
    int warp = threadIdx.x / 32, lane = threadIdx.x & 31, q = lane / 4, tc = lane & 3;
    int local = warp / 2, seg = warp & 1, group = blockIdx.x * 2 + local;
    auto* base = work + std::size_t(jobid) * per;
    auto* x = Down ? base + align256(std::size_t(cols) * XR) : base;
    const auto* input = q < job.columns ? x + std::size_t(q) * RB : nullptr;
    if constexpr (!Down && SharedRows) {
        input = nullptr;
        if (q < job.columns) {
            const int row = row_ids[jobid*cols+q];
            // Refuse invalid device metadata rather than reading another row's memory.
            if (row < 0 || row >= packed_row_count) asm volatile("trap;");
            input = packed_rows + std::size_t(row)*XR;
        }
    }
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
        if constexpr (ParallelHidden) {
            const int col = threadIdx.x / 16, k = threadIdx.x & 15;
            const bool live = col < job.columns;
            const std::uint16_t h = live ? hidden[col][k] : 0;
            const float f = c::bf16_to_f32(h);
            float amax = c::f32_from_bits(c::f32_bits(f) & 0x7fffffffu);
            // Canonical quantizer ignores NaN in its maximum.
            if (!(amax > 0.f)) amax = 0.f;
#pragma unroll
            for (int m = 8; m; m >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, amax, m, 16);
                amax = other > amax ? other : amax;
            }
            const unsigned scale = c::a4_scale_word(amax, 1.f);
            const unsigned code = c::e4m3_scaled(scale) == 0 ? 0 : c::a4_code(f, scale, 1.f);
            unsigned lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned a = __shfl_sync(0xffffffffu, code, j, 16);
                const unsigned b = __shfl_sync(0xffffffffu, code, 4+j, 16);
                const unsigned d = __shfl_sync(0xffffffffu, code, 8+j, 16);
                const unsigned e = __shfl_sync(0xffffffffu, code, 12+j, 16);
                lo |= (a | (b << 4)) << (8*j); hi |= (d | (e << 4)) << (8*j);
            }
            if (live && k == 0) {
                auto* row = base + align256(std::size_t(cols) * XR) + std::size_t(col) * HR;
                const int p = position(blockIdx.x);
                *reinterpret_cast<uint2*>(row + 8*p) = make_uint2(lo, hi);
                row[I/2+p] = scale;
            }
        } else if (threadIdx.x < job.columns) {
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
template <int P, bool ParallelHidden = false> void run(const FastJob* jobs, int count, int cols, std::uint8_t* work, std::size_t per, cudaStream_t stream) {
    prepare<><<<dim3((cols * (H / 16) + 127) / 128, count), 128, 0, stream>>>(jobs, cols, work, per); check();
    project<false, P, ParallelHidden><<<dim3(2 * I / 32, count), 128, 0, stream>>>(jobs, cols, work, per); check();
    project<true, P><<<dim3(H / 32, count), 128, 0, stream>>>(jobs, cols, work, per); check();
}
template <int P, bool ParallelHidden = false>
void run_rows(const FastJob* jobs, int count, int cols, std::uint8_t* work, std::size_t per,
              const std::uint8_t* packed_rows, const int* row_ids, int row_count, cudaStream_t stream) {
    project<false, P, ParallelHidden, true><<<dim3(2*I/32,count),128,0,stream>>>(jobs,cols,work,per,packed_rows,row_ids,row_count); check();
    project<true, P><<<dim3(H/32,count),128,0,stream>>>(jobs,cols,work,per); check();
}
__global__ void pack_rows_kernel(const float* x, int rows, int ldx, std::uint8_t* packed) {
    const int group = (blockIdx.x*blockDim.x+threadIdx.x)/16;
    const int row = group/(H/16), block = group%(H/16);
    const bool live = row < rows;
    const float value = live ? x[std::size_t(row)*ldx+16*block+(threadIdx.x&15)] : 0.f;
    fp4_rows::pack_block(value, live ? packed+std::size_t(row)*XR : packed, block, live);
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
        case 12: run<4, true>(jobs, count, cols, w, per, stream); break;
        case 8: run<8>(jobs, count, cols, w, per, stream); break;
        default: run<1>(jobs, count, cols, w, per, stream); break;
    }
}
void ffn_rows(const FastJob* jobs, int count, int cols, void* work, int loads,
              const std::uint8_t* packed_rows, const int* row_ids, int row_count, cudaStream_t stream) {
    if (!jobs || !work || !packed_rows || !row_ids || row_count < 1 || cols < 1 || cols > 8 || count < 1 || count > 65535 ||
        (reinterpret_cast<std::uintptr_t>(work)&255) || (reinterpret_cast<std::uintptr_t>(packed_rows)&15) || !valid_loads(loads))
        throw std::invalid_argument("invalid native FP4 shared-row operands");
    auto* w=static_cast<std::uint8_t*>(work);const std::size_t per=workspace_bytes(cols);
    switch(loads) {
        case 2: run_rows<2>(jobs,count,cols,w,per,packed_rows,row_ids,row_count,stream);break;
        case 4: run_rows<4>(jobs,count,cols,w,per,packed_rows,row_ids,row_count,stream);break;
        case 8: run_rows<8>(jobs,count,cols,w,per,packed_rows,row_ids,row_count,stream);break;
        case 12: run_rows<4,true>(jobs,count,cols,w,per,packed_rows,row_ids,row_count,stream);break;
        default: run_rows<1>(jobs,count,cols,w,per,packed_rows,row_ids,row_count,stream);break;
    }
}
void pack_rows_f32(const float* x,int rows,int ldx,std::uint8_t* packed,cudaStream_t stream) {
    if(!x||!packed||rows<1||rows>16||ldx<H||(reinterpret_cast<std::uintptr_t>(packed)&15))
        throw std::invalid_argument("invalid native FP4 row encoding operands");
    pack_rows_kernel<<<(rows*H+255)/256,256,0,stream>>>(x,rows,ldx,packed);check();
}
}  // namespace mm3::fp4_fast
