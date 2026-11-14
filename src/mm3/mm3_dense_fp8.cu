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
template <int NB, int KS>
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

std::atomic<int> g_supported[16];   // per device: 0 unknown, 1 yes, 2 no
std::atomic<int> g_sms[16];         // per device: its SM count (0: not read yet)

template <int NB>
cudaError_t go(const Params& p, int blocks, int ks, cudaStream_t s) {
    switch (ks) {
        case 4: mxq_kernel<NB, 4><<<blocks, 128, 0, s>>>(p); break;
        case 8: mxq_kernel<NB, 8><<<blocks, 256, 0, s>>>(p); break;
        default: mxq_kernel<NB, 16><<<blocks, 512, 0, s>>>(p); break;
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
           ((uintptr_t) j.w & 7u) == 0 && ((uintptr_t) j.sc & 3u) == 0 && ((uintptr_t) j.x & 15u) == 0 &&
           j.ldx >= j.K && j.ldx % 4 == 0 && j.ldy >= j.N;
}

bool launch(const Job* jobs, int n, int rows, cudaStream_t s, int ks) {
    if (!jobs || n < 1 || n > kMaxJobs || rows < 1 || rows > kMaxRows || !supported()) return false;
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
    const cudaError_t e = rows <= 8 ? go<1>(p, blocks, ks, s) : go<2>(p, blocks, ks, s);
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
