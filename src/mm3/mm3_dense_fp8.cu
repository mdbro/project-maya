// src/mm3/mm3_dense_fp8.cu - mm3_dense_fp8.cuh's kernel: MXFP8 weights x MXFP8 activations on the 5090's tensor cores.
//
// Layout of one mma.sync m16n8k32 (A = 16 weight rows x 32 inputs, B = 32 inputs x 8 tokens, FP32 C 16 x 8). Lane
// (g = lane / 4, t = lane % 4). The PTX fragment of an 8-bit A gives thread (g, t) the inputs {4t .. 4t+3, 16+4t ..
// 16+4t+3} of rows g and g + 8; of B the same inputs of token g. A dot product does not depend on the order of its
// inputs as long as A and B use the same one, so this kernel feeds the instruction a permutation of each run of 32
// inputs: thread (g, t) takes the 8 consecutive bytes [8t, 8t + 8) of the run - logical inputs 4t .. 4t+3 are memory
// 8t .. 8t+3, logical 16+4t .. are memory 8t+4 .. 8t+7 - for the weight rows (one 8-byte load each) and for the
// activations alike. Each run's scale belongs to the whole run, so the permutation inside a run changes nothing.
// Scales (scale_vec::1X, thread-id 0, byte-id 0): A's row g from lane 4g (t = 0), row g + 8 from lane 4g + 1 (t = 1);
// B's token g from lane 4g. A thread loads its row's scales for 4 runs at once (4 bytes) and shifts the run's byte
// down for each instruction.
//
// A warp walks groups of 4 runs (128 bytes a row): warp w of a block of KS takes groups w, w + KS, ... of its 16 rows;
// the KS partial sums are added in shared memory in warp order. C: c0, c1 = row g, tokens 2t, 2t + 1; c2, c3 = row
// g + 8, the same tokens.
#include "mm3_dense_fp8.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace mm3::dense_fp8 {
namespace {

struct Params {
    Job jobs[kMaxJobs];
    int start[kMaxJobs + 1];   // the jobs' first block (16 rows a block)
    int njobs = 0;
    int rows = 0;
    uint8_t* packed[kMaxJobs]{};
};

__device__ __forceinline__ void mma_mx(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((unsigned short) 0),
          "h"((unsigned short) 0), "r"(sfb), "h"((unsigned short) 0), "h"((unsigned short) 0));
#else
    (void) c; (void) a0; (void) a1; (void) a2; (void) a3; (void) b0; (void) b1; (void) sfa; (void) sfb;
    __trap();
#endif
}

// one run of 32 activations of one token as the quad holds it (8 a thread, memory order): the run's UE8M0 byte and
// the thread's 8 E4M3 codes as two words (b0: its first 4, b1: its last 4). A non-finite input makes the byte 255 (NaN).
__device__ __forceinline__ void quantize8(const float4 v0, const float4 v1, uint32_t& b0, uint32_t& b1,
                                          uint32_t& sbyte) {
    float m = 0.0f;
    const float a[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float x = fabsf(a[i]);
        m = (x != x) ? INFINITY : fmaxf(m, x);
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    float mul = 1.0f;
    if (!(m <= 3.4028234664e38f)) {
        sbyte = 255u;
    } else if (m == 0.0f) {
        sbyte = 0u;
    } else {
        const uint32_t b = __float_as_uint(__fdiv_rn(m, 448.0f));
        int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
        e = max(-127, min(126, e));
        sbyte = (uint32_t) (e + 127);
        mul = __uint_as_float((uint32_t) (127 - e) << 23);
    }
    const __nv_fp8x2_storage_t p0 = __nv_cvt_float2_to_fp8x2(make_float2(a[0] * mul, a[1] * mul), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t p1 = __nv_cvt_float2_to_fp8x2(make_float2(a[2] * mul, a[3] * mul), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t p2 = __nv_cvt_float2_to_fp8x2(make_float2(a[4] * mul, a[5] * mul), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t p3 = __nv_cvt_float2_to_fp8x2(make_float2(a[6] * mul, a[7] * mul), __NV_SATFINITE, __NV_E4M3);
    b0 = (uint32_t) p0 | ((uint32_t) p1 << 16);
    b1 = (uint32_t) p2 | ((uint32_t) p3 << 16);
}

// NB: 8-token tiles (1: rows <= 8, 2: rows <= 16); KS: warps a block.
template <int NB, int KS, bool Packed = false>
__global__ void __launch_bounds__(KS * 32) mxq_kernel(const __grid_constant__ Params p) {
    __shared__ float red[KS][NB * 4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    int j = 0;
    while (j + 1 < p.njobs && (int) blockIdx.x >= p.start[j + 1]) ++j;
    const Job& jb = p.jobs[j];
    const int r0 = ((int) blockIdx.x - p.start[j]) * 16;
    const int K = jb.K, ks = K >> 5, groups = K >> 7;
    const uint8_t* wa = jb.w + (size_t) (r0 + g) * K + 8 * t;
    const uint8_t* wb = wa + (size_t) 8 * K;
    const uint8_t* sr = jb.sc + (size_t) (r0 + g + 8 * (t & 1)) * ks;
    const int rows = p.rows;
    const int first_token = blockIdx.y * (8 * NB);
    const float* xr[NB];
    bool live[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int tok = first_token + nb * 8 + g;
        live[nb] = tok < rows;
        xr[nb] = jb.x + (size_t) (live[nb] ? tok : 0) * jb.ldx + 8 * t;
    }
    float c[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[nb][i] = 0.0f;
#pragma unroll 1
    for (int G = warp; G < groups; G += KS) {
        const size_t off = (size_t) G * 128;
        uint2 A[4], B[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            A[u] = __ldg((const uint2*) (wa + off + 32 * u));
            B[u] = __ldg((const uint2*) (wb + off + 32 * u));
        }
        const uint32_t sa = __ldg((const unsigned int*) (sr + 4 * G));
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                uint32_t b0, b1, sb;
                if constexpr (Packed) {
                    const int tok = first_token + nb * 8 + g;
                    if (live[nb]) {
                        const uint8_t* row = p.packed[j] + (size_t)tok * ((K + ks + 15) & ~15);
                        const uint2 v = __ldg((const uint2*)(row + off + 32 * u + 8 * t));
                        b0 = v.x; b1 = v.y;
                        sb = row[K + (off >> 5) + u];
                    } else { b0 = b1 = sb = 0; }
                } else {
                    const float4 x0 = live[nb] ? __ldg((const float4*)(xr[nb] + off + 32*u)) : make_float4(0,0,0,0);
                    const float4 x1 = live[nb] ? __ldg((const float4*)(xr[nb] + off + 32*u + 4)) : make_float4(0,0,0,0);
                    quantize8(x0, x1, b0, b1, sb);
                }
                mma_mx(c[nb], A[u].x, B[u].x, A[u].y, B[u].y, b0, b1, sa >> (8 * u), sb);
            }
        }
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][nb * 4 + i][lane] = c[nb][i];
    __syncthreads();
    for (int i = threadIdx.x; i < NB * 4 * 32; i += KS * 32) {
        const int reg = i >> 5, l = i & 31;
        float v = red[0][reg][l];
#pragma unroll
        for (int w = 1; w < KS; ++w) v += red[w][reg][l];
        const int nb = reg >> 2, q = reg & 3;
        const int row = r0 + (l >> 2) + ((q & 2) ? 8 : 0);
        const int tok = first_token + nb * 8 + 2 * (l & 3) + (q & 1);
        if (tok < rows) {
            float* yp = jb.y + (size_t) tok * jb.ldy + row;
            *yp = jb.beta != 0.0f ? jb.beta * *yp + v : v;
        }
    }
}

template <int NB, int KS>
__global__ void __launch_bounds__(KS * 32) mxq_baseline(const __grid_constant__ Params p) {
    __shared__ float red[KS][NB * 4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    int j = 0;
    while (j + 1 < p.njobs && (int) blockIdx.x >= p.start[j + 1]) ++j;
    const Job& jb = p.jobs[j];
    const int r0 = ((int) blockIdx.x - p.start[j]) * 16;
    const int K = jb.K, ks = K >> 5, groups = K >> 7;
    const uint8_t* wa = jb.w + (size_t) (r0 + g) * K + 8 * t;
    const uint8_t* wb = wa + (size_t) 8 * K;
    const uint8_t* sr = jb.sc + (size_t) (r0 + g + 8 * (t & 1)) * ks;
    const int rows = p.rows;
    const float* xr[NB];
    bool live[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int tok = nb * 8 + g;
        live[nb] = tok < rows;
        xr[nb] = jb.x + (size_t) (live[nb] ? tok : 0) * jb.ldx + 8 * t;
    }
    float c[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[nb][i] = 0.0f;
#pragma unroll 1
    for (int G = warp; G < groups; G += KS) {
        const size_t off = (size_t) G * 128;
        uint2 A[4], B[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            A[u] = __ldg((const uint2*) (wa + off + 32 * u));
            B[u] = __ldg((const uint2*) (wb + off + 32 * u));
        }
        const uint32_t sa = __ldg((const unsigned int*) (sr + 4 * G));
        float4 X[NB][4][2];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (live[nb]) {
                    X[nb][u][0] = __ldg((const float4*) (xr[nb] + off + 32 * u));
                    X[nb][u][1] = __ldg((const float4*) (xr[nb] + off + 32 * u + 4));
                } else {
                    X[nb][u][0] = make_float4(0.f, 0.f, 0.f, 0.f);
                    X[nb][u][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                uint32_t b0, b1, sb;
                quantize8(X[nb][u][0], X[nb][u][1], b0, b1, sb);
                mma_mx(c[nb], A[u].x, B[u].x, A[u].y, B[u].y, b0, b1, sa >> (8 * u), sb);
            }
        }
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][nb * 4 + i][lane] = c[nb][i];
    __syncthreads();
    for (int i = threadIdx.x; i < NB * 4 * 32; i += KS * 32) {
        const int reg = i >> 5, l = i & 31;
        float v = red[0][reg][l];
#pragma unroll
        for (int w = 1; w < KS; ++w) v += red[w][reg][l];
        const int nb = reg >> 2, q = reg & 3;
        const int row = r0 + (l >> 2) + ((q & 2) ? 8 : 0);
        const int tok = nb * 8 + 2 * (l & 3) + (q & 1);
        if (tok < rows) {
            float* yp = jb.y + (size_t) tok * jb.ldy + row;
            *yp = jb.beta != 0.0f ? jb.beta * *yp + v : v;
        }
    }
}

// Four lanes encode a 32-value block using the same quantizer and lane order as mxq_kernel.
__global__ void pack_x(const float* x, int ldx, int K, int rows, uint8_t* out) {
    const int lane = threadIdx.x & 3;
    const int block = (blockIdx.x * blockDim.x + threadIdx.x) / 4;
    const int blocks_per_row = K / 32;
    const bool live = block < rows * blocks_per_row;
    const int row = live ? block / blocks_per_row : 0, b = live ? block % blocks_per_row : 0;
    const float* src = x + (size_t)row * ldx + 32*b + 8*lane;
    const float4 a0 = live ? *reinterpret_cast<const float4*>(src) : make_float4(0,0,0,0);
    const float4 a1 = live ? *reinterpret_cast<const float4*>(src+4) : make_float4(0,0,0,0);
    uint32_t a, z, scale;
    quantize8(a0, a1, a, z, scale);
    if (!live) return;
    uint8_t* dst = out + (size_t)row * ((K + blocks_per_row + 15) & ~15);
    *reinterpret_cast<uint2*>(dst + 32*b + 8*lane) = make_uint2(a, z);
    if (!lane) dst[K+b] = scale;
}

// ---- the staged kernel (kernel 1): the same products and quantization, the weights brought in by cp.async in whole
// 128-byte lines (8 lanes a line, 4 rows an instruction - kernel 0's 8-byte fragment loads touch 8 lines an instruction,
// one 32-byte sector each) into S stages of the warp's own shared memory, the next groups' copies in flight while a
// group is multiplied; the scales likewise (a 4-byte copy a row). The fragments are then read from shared memory (rows
// 160 bytes apart: two wavefronts a 64-bit read, the least for 256 bytes). Same reduction order as kernel 0, but not
// the same bits (the instruction order of the products is the same; the results agree bit for bit in practice - the
// bench reports it).
constexpr int kRowPitch = 160;                       // bytes a staged row (128 + 32: conflict-free reads)
constexpr int kStageBytes = 16 * kRowPitch + 16 * 4; // 16 rows of weights, then 16 scale words
__device__ __forceinline__ void cp16(void* sm, const void* gm) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(sm);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(a), "l"(gm));
}
__device__ __forceinline__ void cp4(void* sm, const void* gm) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(sm);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(a), "l"(gm));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

template <int NB, int KS, int S, bool Packed = false>
__global__ void __launch_bounds__(KS * 32) mxq2_kernel(const __grid_constant__ Params p) {
    constexpr int kWarpBytes = S * kStageBytes;
    constexpr int kRedBytes = KS * NB * 4 * 32 * 4;
    __shared__ __align__(16) unsigned char smem[(KS * kWarpBytes > kRedBytes) ? KS * kWarpBytes : kRedBytes];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    int j = 0;
    while (j + 1 < p.njobs && (int) blockIdx.x >= p.start[j + 1]) ++j;
    const Job& jb = p.jobs[j];
    const int r0 = ((int) blockIdx.x - p.start[j]) * 16;
    const int K = jb.K, ks = K >> 5, groups = K >> 7;
    unsigned char* mine = smem + warp * kWarpBytes;
    // this lane's copies of a group: rows lane / 8 + 4 i (i < 4), 16-byte chunk lane % 8; lanes 0-15 a scale word each
    const uint8_t* wsrc = jb.w + (size_t) (r0 + (lane >> 3)) * K + 16 * (lane & 7);
    const uint8_t* ssrc = jb.sc + (size_t) (r0 + (lane & 15)) * ks;
    const int ng = warp < groups ? (groups - warp + KS - 1) / KS : 0;
    auto issue = [&](int it) {   // group warp + it KS into stage it % S
        if (it < ng) {
            const int G = warp + it * KS;
            unsigned char* st = mine + (it % S) * kStageBytes;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                cp16(st + ((lane >> 3) + 4 * i) * kRowPitch + 16 * (lane & 7), wsrc + (size_t) (4 * i) * K + (size_t) G * 128);
            if (lane < 16) cp4(st + 16 * kRowPitch + 4 * lane, ssrc + 4 * G);
        }
        cp_commit();   // (an empty group past the end keeps the wait counts uniform)
    };
    const int rows = p.rows;
    const float* xr[NB];
    bool live[NB];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb) {
        const int tok = nb * 8 + g;
        live[nb] = tok < rows;
        xr[nb] = jb.x + (size_t) (live[nb] ? tok : 0) * jb.ldx + 8 * t;
    }
    float c[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[nb][i] = 0.0f;
#pragma unroll
    for (int i = 0; i < S - 1; ++i) issue(i);
#pragma unroll 1
    for (int it = 0; it < ng; ++it) {
        issue(it + S - 1);
        const size_t off = (size_t) (warp + it * KS) * 128;
        float4 X[NB][4][2];   // (the activations' loads go out before the wait)
        uint2 Xp[NB][4];
        uint32_t Sp[NB][4];
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if constexpr (Packed) {
                    if (live[nb]) {
                        const uint8_t* row = p.packed[j] + (size_t)(nb * 8 + g) * ((K + ks + 15) & ~15);
                        Xp[nb][u] = __ldg((const uint2*)(row + off + 32 * u + 8 * t));
                        Sp[nb][u] = row[K + (off >> 5) + u];
                    } else {
                        Xp[nb][u] = make_uint2(0u, 0u);
                        Sp[nb][u] = 0;
                    }
                } else if (live[nb]) {
                    X[nb][u][0] = __ldg((const float4*) (xr[nb] + off + 32 * u));
                    X[nb][u][1] = __ldg((const float4*) (xr[nb] + off + 32 * u + 4));
                } else {
                    X[nb][u][0] = make_float4(0.f, 0.f, 0.f, 0.f);
                    X[nb][u][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
        cp_wait<S - 1>();
        __syncwarp();
        const unsigned char* st = mine + (it % S) * kStageBytes;
        const uint32_t sa = *(const uint32_t*) (st + 16 * kRowPitch + 4 * (g + 8 * (t & 1)));
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const uint2 A = *(const uint2*) (st + g * kRowPitch + 32 * u + 8 * t);
            const uint2 B = *(const uint2*) (st + (g + 8) * kRowPitch + 32 * u + 8 * t);
#pragma unroll
            for (int nb = 0; nb < NB; ++nb) {
                uint32_t b0, b1, sb;
                if constexpr (Packed) {
                    b0 = Xp[nb][u].x; b1 = Xp[nb][u].y; sb = Sp[nb][u];
                } else quantize8(X[nb][u][0], X[nb][u][1], b0, b1, sb);
                mma_mx(c[nb], A.x, B.x, A.y, B.y, b0, b1, sa >> (8 * u), sb);
            }
        }
        __syncwarp();   // (every lane's reads of this stage done before a later issue refills it)
    }
    cp_wait<0>();
    __syncthreads();   // (the stages are free: the reduction reuses the buffer)
    float* red = (float*) smem;   // [KS][NB * 4][32]
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[(warp * NB * 4 + nb * 4 + i) * 32 + lane] = c[nb][i];
    __syncthreads();
    for (int i = threadIdx.x; i < NB * 4 * 32; i += KS * 32) {
        const int reg = i >> 5, l = i & 31;
        float v = red[reg * 32 + l];
#pragma unroll
        for (int w = 1; w < KS; ++w) v += red[(w * NB * 4 + reg) * 32 + l];
        const int nb = reg >> 2, q = reg & 3;
        const int row = r0 + (l >> 2) + ((q & 2) ? 8 : 0);
        const int tok = nb * 8 + 2 * (l & 3) + (q & 1);
        if (tok < rows) {
            float* yp = jb.y + (size_t) tok * jb.ldy + row;
            *yp = jb.beta != 0.0f ? jb.beta * *yp + v : v;
        }
    }
}

std::atomic<int> g_kernel{0};        // the built-in kernel: 0 fragment loads, 1 staged, 2 by rows (set_kernel)
std::atomic<int> g_supported[16];   // per device: 0 unknown, 1 yes, 2 no
std::atomic<int> g_sms[16];         // per device: its SM count (0: not read yet)

template <int NB>
cudaError_t go(const Params& p, int blocks, int ks, int kernel, cudaStream_t s) {
    if (kernel == 3) {
        switch (ks) {
            case 4: mxq_kernel<NB, 4, true><<<blocks, 128, 0, s>>>(p); break;
            case 8: mxq_kernel<NB, 8, true><<<blocks, 256, 0, s>>>(p); break;
            default: mxq_kernel<NB, 16, true><<<blocks, 512, 0, s>>>(p); break;
        }
        return cudaGetLastError();
    }
    if (kernel == 4) {   // ID 1's identical staged weight pipeline, pre-packed activations only.
        switch (ks) {
            case 4: mxq2_kernel<NB, 4, 3, true><<<blocks, 128, 0, s>>>(p); break;
            case 8: mxq2_kernel<NB, 8, 2, true><<<blocks, 256, 0, s>>>(p); break;
            default: mxq2_kernel<NB, 16, 1, true><<<blocks, 512, 0, s>>>(p); break;
        }
        return cudaGetLastError();
    }
    if (kernel == 1) {   // (stages a warp: 3 with 4 warps a block, 2 with 8, 1 with 16 - each block under 48 KB)
        switch (ks) {
            case 4: mxq2_kernel<NB, 4, 3><<<blocks, 128, 0, s>>>(p); break;
            case 8: mxq2_kernel<NB, 8, 2><<<blocks, 256, 0, s>>>(p); break;
            default: mxq2_kernel<NB, 16, 1><<<blocks, 512, 0, s>>>(p); break;
        }
        return cudaGetLastError();
    }
    switch (ks) {
        case 4: mxq_baseline<NB, 4><<<blocks, 128, 0, s>>>(p); break;
        case 8: mxq_baseline<NB, 8><<<blocks, 256, 0, s>>>(p); break;
        default: mxq_baseline<NB, 16><<<blocks, 512, 0, s>>>(p); break;
    }
    return cudaGetLastError();
}

}  // namespace

bool supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    int v = g_supported[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        cudaFuncAttributes fa{};
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
                        major == 12 && minor == 0 &&
                        cudaFuncGetAttributes(&fa, mxq_kernel<1, 4>) == cudaSuccess;
        cudaGetLastError();
        v = ok ? 1 : 2;
        g_supported[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

bool job_ok(const Job& j) {
    return j.w && j.sc && j.x && j.y && j.N > 0 && j.K > 0 && j.N % 16 == 0 && j.K % 128 == 0 &&
           ((uintptr_t) j.w & 15u) == 0 && ((uintptr_t) j.sc & 3u) == 0 && ((uintptr_t) j.x & 15u) == 0 &&
           j.ldx >= j.K && j.ldx % 4 == 0 && j.ldy >= j.N;
}

void set_kernel(int k) { g_kernel.store(k >= 0 && k <= 4 ? k : 0, std::memory_order_relaxed); }
int kernel() { return g_kernel.load(std::memory_order_relaxed); }

std::size_t workspace_bytes(const Job* jobs, int n, int rows) {
    if (!jobs || n < 1 || n > kMaxJobs || rows < 1 || rows > kMaxPromptRows) return 0;
    size_t bytes = 0;
    for (int i = 0; i < n; ++i) {
        if (!job_ok(jobs[i])) return 0;
        bool duplicate = false;
        for (int j = 0; j < i; ++j)
            if (jobs[i].x == jobs[j].x && jobs[i].K == jobs[j].K && jobs[i].ldx == jobs[j].ldx) duplicate = true;
        if (!duplicate) bytes += (size_t)rows * ((jobs[i].K + jobs[i].K / 32 + 15) & ~15);
    }
    return bytes;
}

bool launch(const Job* jobs, int n, int rows, cudaStream_t s, int ks, int kern, void* workspace, size_t workspace_size) {
    if (!jobs || n < 1 || n > kMaxJobs || rows < 1 || rows > kMaxPromptRows || !supported()) return false;
    Params p;
    p.njobs = n;
    p.rows = rows;
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        if (!job_ok(jobs[i])) return false;
        p.jobs[i] = jobs[i];
        p.start[i] = blocks;
        blocks += jobs[i].N / 16;
    }
    p.start[n] = blocks;
    for (int i = n + 1; i <= kMaxJobs; ++i) p.start[i] = blocks;
    if (ks == 0) {
        // (the 5090: 170 SMs; a block of 4 warps when the grid gives every SM two or more, more warps a block - a finer
        //  split of K - for the small grids: the attention's k / v and the indexer's projections alone)
        int dev = 0;
        cudaGetDevice(&dev);
        int sms = g_sms[dev & 15].load(std::memory_order_relaxed);
        if (sms <= 0) {
            if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms <= 0) sms = 170;
            g_sms[dev & 15].store(sms, std::memory_order_relaxed);
        }
        ks = blocks >= 2 * sms ? 4 : blocks >= sms / 2 ? 8 : 16;
    }
    if (ks != 4 && ks != 8 && ks != 16) return false;
    if (kern < 0) kern = g_kernel.load(std::memory_order_relaxed);
    // 2: by rows - the staged kernel up to 8 rows, the fragment-load kernel above (measured 2026-11-14 in the k3-fp8 gap,
    // the MoE layer's projections as a graph: kernel 1 123-131 us against kernel 0's 137-161 at 1-8 rows; kernel 0
    // 132-138 us against kernel 1's 157-194 at 10-16 rows, a prompt's 16-row groups)
    if (kern == 2) kern = rows <= 8 ? 1 : 0;
    if (rows > kMaxRows && kern != 0 && kern != 3) return false;
    if (rows > 32 && kern != 3) return false;
    if (rows > kMaxRows && kern == 0) {
        if(ks==4)mxq_kernel<4,4><<<blocks,128,0,s>>>(p);
        else if(ks==8)mxq_kernel<4,8><<<blocks,256,0,s>>>(p);
        else mxq_kernel<4,16><<<blocks,512,0,s>>>(p);
        return cudaGetLastError()==cudaSuccess;
    }
    if (kern != 0 && kern != 1 && kern != 3 && kern != 4) return false;
    if (kern == 3 || kern == 4) {
        const size_t needed = workspace_bytes(jobs, n, rows);
        if (!workspace || (reinterpret_cast<uintptr_t>(workspace) & 15u) || workspace_size < needed) return false;
        size_t offset = 0;
        for (int i = 0; i < n; ++i) {
            int prior = -1;
            for (int j = 0; j < i; ++j)
                if (jobs[i].x == jobs[j].x && jobs[i].K == jobs[j].K && jobs[i].ldx == jobs[j].ldx) { prior = j; break; }
            if (prior >= 0) p.packed[i] = p.packed[prior];
            else {
                p.packed[i] = static_cast<uint8_t*>(workspace) + offset;
                offset += (size_t)rows * ((jobs[i].K + jobs[i].K / 32 + 15) & ~15);
                const int threads = rows * (jobs[i].K / 32) * 4;
                pack_x<<<(threads + 127)/128, 128, 0, s>>>(jobs[i].x, jobs[i].ldx, jobs[i].K, rows, p.packed[i]);
                if (cudaGetLastError() != cudaSuccess) return false;
            }
        }
    }
    if (rows > kMaxRows) {
        // Each block multiplies one 16-weight-row tile by up to 32 tokens.
        // Adjacent token tiles reuse one input packing and the same K reduction.
        const dim3 grid(blocks, (rows + 31) / 32);
        if (ks == 4) mxq_kernel<4, 4, true><<<grid, 128, 0, s>>>(p);
        else if (ks == 8) mxq_kernel<4, 8, true><<<grid, 256, 0, s>>>(p);
        else mxq_kernel<4, 16, true><<<grid, 512, 0, s>>>(p);
        return cudaGetLastError() == cudaSuccess;
    }
    const cudaError_t e = rows <= 8 ? go<1>(p, blocks, ks, kern, s) : go<2>(p, blocks, ks, kern, s);
    return e == cudaSuccess;
}

uint8_t quantize_block_host(const float* x, uint8_t* codes) {
    float m = 0.0f;
    for (int i = 0; i < 32; ++i) {
        const float a = std::fabs(x[i]);
        m = (a != a) ? INFINITY : std::max(m, a);
    }
    float mul = 1.0f;
    uint8_t sbyte;
    if (!(m <= 3.4028234664e38f)) {
        sbyte = 255;
    } else if (m == 0.0f) {
        sbyte = 0;
    } else {
        const float q = m / 448.0f;
        uint32_t b;
        std::memcpy(&b, &q, 4);
        int e = (int) ((b >> 23) & 0xffu) - 127 + ((b & 0x7fffffu) != 0u ? 1 : 0);
        e = std::max(-127, std::min(126, e));
        sbyte = (uint8_t) (e + 127);
        const uint32_t mb = (uint32_t) (127 - e) << 23;
        std::memcpy(&mul, &mb, 4);
    }
    for (int i = 0; i < 32; ++i) codes[i] = (uint8_t) __nv_cvt_float_to_fp8(x[i] * mul, __NV_SATFINITE, __NV_E4M3);
    return sbyte;
}

}  // namespace mm3::dense_fp8
