// src/mm3/mm3_draft_fp4.cu - mm3_draft_fp4.cuh's kernels: NVFP4 weights x NVFP4 activations on the 5090's FP4 tensor
// cores, for DFlash2's projections (kernel item 6).
//
// One mma.sync m16n8k64 (A = 16 weight rows x 64 inputs, B = 64 inputs x 8 activation rows, FP32 C 16 x 8). Lane
// (g = lane / 4, t = lane % 4): A's a0 / a1 / a2 / a3 = rows g / g + 8 / g / g + 8, inputs 8t .. 8t + 7 (a0, a1) and
// 32 + 8t .. 32 + 8t + 7 (a2, a3), input 8t + j in nibble j; B's b0 / b1 = activation row g, the same inputs; the scales
// (scale_vec::4X, E4M3, byte j = inputs 16j .. 16j + 15): A's row g + 8 (t & 1) from lane (g, t), B's row g from lane
// (g, *); C: c0, c1 = weight row g, activation rows 2t, 2t + 1; c2, c3 = weight row g + 8, the same activation rows.
// (The same mapping as Infernix's ldmatrix fragments in ops/linear/nvfp4/nvfp4_a4_mma.cuh, whose A is the activations.)
#include "mm3_draft_fp4.cuh"

#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>

namespace mm3::draft_fp4 {
namespace {

struct Params {
    Job jobs[kMaxJobs];
    int start[kMaxJobs + 1];   // the jobs' first block (16 weight rows a block)
    int njobs = 0;
    int rows = 0;
};

__device__ __forceinline__ void mma_fp4(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1, uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"((unsigned short) 0),
          "h"((unsigned short) 0), "r"(sfb), "h"((unsigned short) 0), "h"((unsigned short) 0));
#else
    (void) c; (void) a0; (void) a1; (void) a2; (void) a3; (void) b0; (void) b1; (void) sfa; (void) sfb;
    __trap();
#endif
}

// 16 values (already divided by the row's global scale) -> two words of E2M1 codes (value 2i in the low nibble of byte i,
// RNE, saturating) under E4M3 scale `dec` (its value); cvt.rn.satfinite.e2m1x2.f32 puts its first operand in the high
// nibble, so each pair goes in (odd, even)
__device__ __forceinline__ void pack16(const float (&v)[16], float dec, uint32_t& lo, uint32_t& hi) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    float q[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) q[i] = dec > 0.0f ? __fdiv_rn(v[i], dec) : 0.0f;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "mov.b32 %1, {b4, b5, b6, b7};\n"
        "}\n"
        : "=r"(lo), "=r"(hi)
        : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]), "f"(q[4]), "f"(q[5]), "f"(q[6]), "f"(q[7]), "f"(q[8]), "f"(q[9]),
          "f"(q[10]), "f"(q[11]), "f"(q[12]), "f"(q[13]), "f"(q[14]), "f"(q[15]));
#else
    (void) v; (void) dec;
    lo = hi = 0;
    __trap();
#endif
}

__device__ __forceinline__ float e2m1_dev(uint32_t c) {
    const float m[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = m[c & 7u];
    return (c & 8u) ? -a : a;
}
__device__ __forceinline__ float e4m3_dev(uint32_t b) {
    __nv_fp8_e4m3 f;
    f.__x = (__nv_fp8_storage_t) b;
    return (float) f;
}

// one row a block (blockIdx.x; rows of the last 8-row tile past `rows` written as zeros), 256 threads
template <int T>
__global__ void __launch_bounds__(256) quant_kernel(const float* __restrict__ x, int rows, int ldx, int K,
                                                    uint8_t* __restrict__ xq, float* __restrict__ xgs) {
    const int r = blockIdx.x, KC = K >> 6, G16 = K >> 4;
    if (r >= rows) {   // (zero codes and scales: the MMA's padding rows, whose outputs are never stored)
        for (int i = threadIdx.x; i < T * KC * 4; i += blockDim.x) {
            const int term = i / (KC * 4), rest = i % (KC * 4), kc = rest >> 2, j = rest & 3;
            *(uint2*) (xq + x_word(KC, term, kc, r, j)) = make_uint2(0u, 0u);   // (runs j, j + 4: lane t = j's 8 bytes)
            xq[x_scale(KC, term, kc, r, j)] = 0;
        }
        if (threadIdx.x == 0) xgs[r] = 0.0f;
        return;
    }
    const float* xr = x + (size_t) r * ldx;
    __shared__ float red[8];
    float m = 0.0f;
    for (int i = threadIdx.x * 4; i < K; i += blockDim.x * 4) {
        const float4 v = *(const float4*) (xr + i);
        const float a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int k = 0; k < 4; ++k) m = (a[k] != a[k]) ? INFINITY : fmaxf(m, fabsf(a[k]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = red[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
    const bool bad = !(m <= 3.4028234664e38f);
    const float inv = (bad || m == 0.0f) ? 0.0f : __fdiv_rn(2688.0f, m);   // (6 x 448: the largest block's scale is 448)
    if (threadIdx.x == 0) xgs[r] = bad ? __int_as_float(0x7fc00000) : (m == 0.0f ? 0.0f : __fdiv_rn(m, 2688.0f));
    for (int G = threadIdx.x; G < G16; G += blockDim.x) {
        float v[16];
#pragma unroll
        for (int k = 0; k < 16; k += 4) {
            const float4 q = *(const float4*) (xr + 16 * G + k);
            v[k] = __fmul_rn(q.x, inv);
            v[k + 1] = __fmul_rn(q.y, inv);
            v[k + 2] = __fmul_rn(q.z, inv);
            v[k + 3] = __fmul_rn(q.w, inv);
        }
        const int kc = G >> 2, j = G & 3;
#pragma unroll
        for (int term = 0; term < T; ++term) {
            float a = 0.0f;
#pragma unroll
            for (int k = 0; k < 16; ++k) a = fmaxf(a, fabsf(v[k]));
            const uint8_t sb = (uint8_t) __nv_cvt_float_to_fp8(__fdiv_rn(a, 6.0f), __NV_SATFINITE, __NV_E4M3);
            const float dec = e4m3_dev(sb);
            uint32_t lo, hi;
            pack16(v, dec, lo, hi);
            *(uint32_t*) (xq + x_word(KC, term, kc, r, 2 * j)) = lo;       // inputs 16j .. 16j + 7: run 2j
            *(uint32_t*) (xq + x_word(KC, term, kc, r, 2 * j + 1)) = hi;   // 16j + 8 .. 16j + 15: run 2j + 1
            xq[x_scale(KC, term, kc, r, j)] = sb;
            if (term + 1 < T) {   // (what this term left, for the next)
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    v[k] = __fsub_rn(v[k], __fmul_rn(e2m1_dev((lo >> (4 * k)) & 15u), dec));
                    v[8 + k] = __fsub_rn(v[8 + k], __fmul_rn(e2m1_dev((hi >> (4 * k)) & 15u), dec));
                }
            }
        }
    }
}

// NB: 8-row activation tiles (1: rows <= 8, 2: rows <= 16); KS: warps a block; T: activation terms
template <int NB, int KS, int T>
__global__ void __launch_bounds__(KS * 32) w4a4_kernel(const __grid_constant__ Params p) {
    __shared__ float red[KS][NB * 4][32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    int j = 0;
    while (j + 1 < p.njobs && (int) blockIdx.x >= p.start[j + 1]) ++j;
    const Job& jb = p.jobs[j];
    const int rb = (int) blockIdx.x - p.start[j];
    const int KC = jb.m.K >> 6;
    const uint4* wt = (const uint4*) (jb.m.w + w_word(KC, rb, 0, g, t, 0));        // + 32 a chunk
    const uint32_t* st = (const uint32_t*) (jb.m.sc + w_scale(KC, rb, 0, g + 8 * (t & 1)));   // + 16 a chunk
    const size_t plane = (size_t) KC * 2 * 288;
    const uint2* xc = (const uint2*) (jb.xq + x_word(KC, 0, 0, g, t));            // + 32 a (chunk, tile)
    const uint32_t* xs = (const uint32_t*) (jb.xq + x_scale(KC, 0, 0, g, 0));     // + 8 a (chunk, tile)
    float c[NB][4];
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) c[nb][i] = 0.0f;
    constexpr int U = 4;   // (chunks whose weight loads go out together)
    int kc = warp;
#pragma unroll 1
    for (; kc + (U - 1) * KS < KC; kc += U * KS) {
        uint4 A[U];
        uint32_t sa[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            A[u] = __ldg(wt + (size_t) (kc + u * KS) * 32);
            sa[u] = __ldg(st + (size_t) (kc + u * KS) * 16);
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int k = kc + u * KS;
#pragma unroll
            for (int nb = 0; nb < NB; ++nb)
#pragma unroll
                for (int term = 0; term < T; ++term) {
                    const size_t o = term * plane;
                    const uint2 B = __ldg((const uint2*) ((const uint8_t*) (xc + (size_t) (2 * k + nb) * 32) + o));
                    const uint32_t sb = __ldg((const uint32_t*) ((const uint8_t*) (xs + (size_t) (2 * k + nb) * 8) + o));
                    mma_fp4(c[nb], A[u].x, A[u].y, A[u].z, A[u].w, B.x, B.y, sa[u], sb);
                }
        }
    }
#pragma unroll 1
    for (; kc < KC; kc += KS) {
        const uint4 A = __ldg(wt + (size_t) kc * 32);
        const uint32_t sa = __ldg(st + (size_t) kc * 16);
#pragma unroll
        for (int nb = 0; nb < NB; ++nb)
#pragma unroll
            for (int term = 0; term < T; ++term) {
                const size_t o = term * plane;
                const uint2 B = __ldg((const uint2*) ((const uint8_t*) (xc + (size_t) (2 * kc + nb) * 32) + o));
                const uint32_t sb = __ldg((const uint32_t*) ((const uint8_t*) (xs + (size_t) (2 * kc + nb) * 8) + o));
                mma_fp4(c[nb], A.x, A.y, A.z, A.w, B.x, B.y, sa, sb);
            }
    }
#pragma unroll
    for (int nb = 0; nb < NB; ++nb)
#pragma unroll
        for (int i = 0; i < 4; ++i) red[warp][nb * 4 + i][lane] = c[nb][i];
    __syncthreads();
    const int rows = p.rows;
    for (int i = threadIdx.x; i < NB * 4 * 32; i += KS * 32) {
        const int reg = i >> 5, l = i & 31;
        float v = red[0][reg][l];
#pragma unroll
        for (int w = 1; w < KS; ++w) v += red[w][reg][l];
        const int nb = reg >> 2, q = reg & 3;
        const int row = rb * 16 + (l >> 2) + ((q & 2) ? 8 : 0);
        const int tok = nb * 8 + 2 * (l & 3) + (q & 1);
        if (tok < rows) {
            v = (v * jb.m.gs) * jb.xgs[tok];
            float* yp = jb.y + (size_t) tok * jb.ldy + row;
            *yp = jb.beta != 0.0f ? jb.beta * *yp + v : v;
        }
    }
}

std::atomic<int> g_supported[16];   // per device: 0 unknown, 1 yes, 2 no
std::atomic<int> g_sms[16];         // per device: its SM count (0: not read yet)

template <int NB, int T>
cudaError_t go(const Params& p, int blocks, int ks, cudaStream_t s) {
    switch (ks) {
        case 4: w4a4_kernel<NB, 4, T><<<blocks, 128, 0, s>>>(p); break;
        case 8: w4a4_kernel<NB, 8, T><<<blocks, 256, 0, s>>>(p); break;
        default: w4a4_kernel<NB, 16, T><<<blocks, 512, 0, s>>>(p); break;
    }
    return cudaGetLastError();
}

}  // namespace

void tile_host(const uint8_t* codes, const uint8_t* scales, int N, int K, uint8_t* wt, uint8_t* st) {
    const int KC = K / 64;
    const size_t cr = (size_t) K / 2, sr = (size_t) K / 16;
    for (int rb = 0; rb < N / 16; ++rb)
        for (int kc = 0; kc < KC; ++kc) {
            for (int g = 0; g < 8; ++g)
                for (int t = 0; t < 4; ++t)
                    for (int reg = 0; reg < 4; ++reg) {
                        const int row = rb * 16 + g + ((reg & 1) ? 8 : 0);
                        // inputs kc 64 + 8t .. (+ 32 for a2, a3): bytes kc 32 + 4t .. (+ 16) of the row
                        const size_t src = (size_t) row * cr + (size_t) kc * 32 + 4 * t + ((reg & 2) ? 16 : 0);
                        std::memcpy(wt + w_word(KC, rb, kc, g, t, reg), codes + src, 4);
                    }
            for (int r = 0; r < 16; ++r)
                std::memcpy(st + w_scale(KC, rb, kc, r), scales + (size_t) (rb * 16 + r) * sr + (size_t) kc * 4, 4);
        }
}

bool supported() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    int v = g_supported[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        cudaFuncAttributes fa{};
        const bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
                        major == 12 && minor == 0 && cudaFuncGetAttributes(&fa, w4a4_kernel<1, 4, 1>) == cudaSuccess &&
                        cudaFuncGetAttributes(&fa, quant_kernel<1>) == cudaSuccess;
        cudaGetLastError();
        v = ok ? 1 : 2;
        g_supported[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

bool quantize(const float* x, int rows, int ldx, int K, int terms, uint8_t* xq, float* xgs, cudaStream_t s) {
    if (!x || !xq || !xgs || rows < 1 || rows > kMaxRows || K < 64 || K % 64 || ldx < K || ldx % 4 ||
        ((uintptr_t) x & 15u) || ((uintptr_t) xq & 15u) || (terms != 1 && terms != 2) || !supported())
        return false;
    const int blocks = (rows + 7) / 8 * 8;
    if (terms == 1) quant_kernel<1><<<blocks, 256, 0, s>>>(x, rows, ldx, K, xq, xgs);
    else quant_kernel<2><<<blocks, 256, 0, s>>>(x, rows, ldx, K, xq, xgs);
    return cudaGetLastError() == cudaSuccess;
}

bool launch(const Job* jobs, int n, int rows, int terms, cudaStream_t s, int ks) {
    if (!jobs || n < 1 || n > kMaxJobs || rows < 1 || rows > kMaxRows || (terms != 1 && terms != 2) || !supported())
        return false;
    Params p;
    p.njobs = n;
    p.rows = rows;
    int blocks = 0;
    for (int i = 0; i < n; ++i) {
        const Job& j = jobs[i];
        if (!j.m.w || !j.m.sc || !j.xq || !j.xgs || !j.y || j.m.N <= 0 || j.m.N % 16 || j.m.K < 64 || j.m.K % 64 ||
            ((uintptr_t) j.m.w & 15u) || ((uintptr_t) j.m.sc & 3u) || ((uintptr_t) j.xq & 15u) || j.ldy < j.m.N)
            return false;
        p.jobs[i] = j;
        p.start[i] = blocks;
        blocks += j.m.N / 16;
    }
    p.start[n] = blocks;
    for (int i = n + 1; i <= kMaxJobs; ++i) p.start[i] = blocks;
    if (ks == 0) {
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
    cudaError_t e;
    if (rows <= 8) e = terms == 1 ? go<1, 1>(p, blocks, ks, s) : go<1, 2>(p, blocks, ks, s);
    else e = terms == 1 ? go<2, 1>(p, blocks, ks, s) : go<2, 2>(p, blocks, ks, s);
    return e == cudaSuccess;
}

float e2m1_value(unsigned code) {
    static const float m[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float a = m[code & 7u];
    return (code & 8u) ? -a : a;
}

float e4m3_value(uint8_t b) {
    __nv_fp8_e4m3 f;
    f.__x = (__nv_fp8_storage_t) b;
    return (float) f;
}

namespace {
// E2M1 nearest, ties to the even code, saturating at 6; the sign kept (a negative value that rounds to 0 is code 8)
unsigned e2m1_code_host(float v) {
    const float a = std::fabs(v);
    unsigned c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return (std::signbit(v) ? 8u : 0u) | c;
}
}  // namespace

float quantize_row_host(const float* x, int K, int terms, uint8_t* codes, uint8_t* scales) {
    float m = 0.0f;
    for (int i = 0; i < K; ++i) m = (x[i] != x[i]) ? INFINITY : std::max(m, std::fabs(x[i]));
    const bool bad = !(m <= 3.4028234664e38f);
    const float inv = (bad || m == 0.0f) ? 0.0f : 2688.0f / m;
    for (int G = 0; G < K / 16; ++G) {
        float v[16];
        for (int k = 0; k < 16; ++k) v[k] = x[16 * G + k] * inv;
        for (int term = 0; term < terms; ++term) {
            float a = 0.0f;
            for (int k = 0; k < 16; ++k) a = std::max(a, std::fabs(v[k]));
            const uint8_t sb = (uint8_t) __nv_cvt_float_to_fp8(a / 6.0f, __NV_SATFINITE, __NV_E4M3);
            const float dec = e4m3_value(sb);
            uint8_t* cr = codes + (size_t) term * (K / 2) + 8 * G;
            for (int k = 0; k < 16; k += 2) {
                const unsigned c0 = dec > 0.0f ? e2m1_code_host(v[k] / dec) : 0u;
                const unsigned c1 = dec > 0.0f ? e2m1_code_host(v[k + 1] / dec) : 0u;
                cr[k / 2] = (uint8_t) (c0 | (c1 << 4));
                if (term + 1 < terms) {
                    const float p0 = e2m1_value(c0) * dec, p1 = e2m1_value(c1) * dec;
                    v[k] = v[k] - p0;
                    v[k + 1] = v[k + 1] - p1;
                }
            }
            scales[(size_t) term * (K / 16) + G] = sb;
        }
    }
    return bad ? std::nanf("") : (m == 0.0f ? 0.0f : m / 2688.0f);
}

}  // namespace mm3::draft_fp4
