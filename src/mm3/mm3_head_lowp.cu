// src/mm3/mm3_head_lowp.cu - kernel item 10: the output head in FP8 (MXFP8 layout) or NVFP4, weights only
// (mm3_head_lowp.cuh).
#include "mm3_head_lowp.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace mm3::head_lowp {
namespace {

// ---- quantization (once, at load)

// kFp8: a thread a 32-block: UE8M0 scale 2^s, s the least with amax / 2^s <= 448; E4M3 codes of v / 2^s (exact division
// by a power of two, then the hardware's round-to-nearest-even, saturating: nothing reaches the saturation).
__global__ void quant_fp8(const uint16_t* __restrict__ bf, int n, int k, uint8_t* __restrict__ w, uint8_t* __restrict__ sc) {
    const size_t blocks = (size_t) n * (k / 32);
    for (size_t b = blockIdx.x * (size_t) blockDim.x + threadIdx.x; b < blocks; b += (size_t) gridDim.x * blockDim.x) {
        const uint16_t* v = bf + b * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(__uint_as_float(uint32_t(v[i]) << 16)));
        int s = 0;
        if (amax > 0.0f && isfinite(amax)) {
            int ex = 0;
            const float m = frexpf(amax / 448.0f, &ex);   // amax / 448 = m 2^ex, m in [0.5, 1)
            s = m == 0.5f ? ex - 1 : ex;                   // (ceil(log2(amax / 448)))
            s = max(-127, min(127, s));
        }
        sc[b] = (uint8_t) (s + 127);
        const float inv = ldexpf(1.0f, -s);
        for (int i = 0; i < 32; ++i)
            w[b * 32 + i] = (uint8_t) __nv_cvt_float_to_fp8(__uint_as_float(uint32_t(v[i]) << 16) * inv, __NV_SATFINITE, __NV_E4M3);
    }
}

// kNvfp4: the matrix's largest magnitude (positive floats order as their bits)
__global__ void amax_kernel(const uint16_t* __restrict__ bf, size_t count, float* out) {
    float m = 0.0f;
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < count; i += (size_t) gridDim.x * blockDim.x) {
        const float a = fabsf(__uint_as_float(uint32_t(bf[i]) << 16));
        if (isfinite(a)) m = fmaxf(m, a);
    }
    for (int d = 16; d; d >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
    if ((threadIdx.x & 31) == 0) atomicMax((unsigned int*) out, __float_as_uint(m));
}
__device__ uint8_t e2m1_rn(float x) {   // canonical_math.h's e2m1_rn_satfinite: nearest even onto {0, .5, 1, 1.5, 2, 3, 4, 6}
    const unsigned sign = signbit(x) ? 8u : 0u;
    const float a = fabsf(x);
    unsigned c = a <= 0.25f ? 0u : a < 0.75f ? 1u : a <= 1.25f ? 2u : a < 1.75f ? 3u : a <= 2.5f ? 4u : a < 3.5f ? 5u : a <= 5.0f ? 6u : 7u;
    return (uint8_t) (sign | c);
}
// a thread a 16-block: E4M3 scale of block amax / (6 global) to nearest even (saturating at 448), codes E2M1 of
// v / (scale global)
__global__ void quant_nvfp4(const uint16_t* __restrict__ bf, int n, int k, const float* __restrict__ amax_p,
                            uint8_t* __restrict__ w, uint8_t* __restrict__ sc) {
    const float amax = *amax_p, global = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
    const size_t blocks = (size_t) n * (k / 16);
    for (size_t b = blockIdx.x * (size_t) blockDim.x + threadIdx.x; b < blocks; b += (size_t) gridDim.x * blockDim.x) {
        const uint16_t* v = bf + b * 16;
        float bmax = 0.0f;
        for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(__uint_as_float(uint32_t(v[i]) << 16)));
        const uint8_t s = (uint8_t) __nv_cvt_float_to_fp8(bmax / (6.0f * global), __NV_SATFINITE, __NV_E4M3);
        sc[b] = s;
        const float d = e4m3_float(s) * global;
        uint8_t* out = w + b * 8;
        for (int i = 0; i < 16; i += 2) {
            const uint8_t lo = d > 0.0f ? e2m1_rn(__uint_as_float(uint32_t(v[i]) << 16) / d) : 0,
                          hi = d > 0.0f ? e2m1_rn(__uint_as_float(uint32_t(v[i + 1]) << 16) / d) : 0;
            out[i / 2] = (uint8_t) (lo | (hi << 4));
        }
    }
}

// ---- the matrix-vector products: a warp RW rows, a lane 8 consecutive weights of each a step (256 a row a step), the
// lane's 8 activations of each column held in registers for its RW rows; each 8-weight piece summed unscaled, then
// times its block scale into the row's sum; the lanes' sums added by a fixed tree. Rows past n read the last row (their
// loads stay unconditional, so a lane keeps its rows' loads in flight); their sums are never written.

__device__ __forceinline__ void fp8x8(uint2 b, float* f) {   // 8 E4M3 (byte i: weight i) to float
    const uint32_t q[2] = {b.x, b.y};
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) ((q[h] >> (16 * p)) & 0xffffu), __NV_E4M3);
            const float2 v = __half22float2(*(const __half2*) &r);
            f[4 * h + 2 * p] = v.x;
            f[4 * h + 2 * p + 1] = v.y;
        }
}
__device__ __forceinline__ void fp4x8(uint32_t b, float* f) {   // 8 E2M1 (nibble i: weight i) to twice their value
    // the doubled magnitudes {0, 1, 2, 3, 4, 6, 8, 12} picked by byte permutes (canonical_math.h's e2m1_x2_quad)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t codes = h ? (b >> 16) : (b & 0xffffu);
        const uint32_t sel = (codes & 0x7u) | ((codes >> 4 & 0x7u) << 4) | ((codes >> 8 & 0x7u) << 8) | ((codes >> 12 & 0x7u) << 12);
        const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float m = (float) ((mag >> (8 * i)) & 0xffu);
            f[4 * h + i] = (codes >> (4 * i + 3)) & 1u ? -m : m;
        }
    }
}

template <int FMT, int NC, int RW, int WB>
__global__ void __launch_bounds__(WB * 32) head_mv(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, int n,
                                                    int k, float global, const float* __restrict__ x, int ldx,
                                                    float* __restrict__ y, int ldy, int ncol) {
    const int lane = threadIdx.x & 31, row0 = (blockIdx.x * WB + (threadIdx.x >> 5)) * RW;
    if (row0 >= n) return;
    float acc[RW][NC];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    // bytes a row, scale bytes a row
    const size_t wrow = FMT == kFp8 ? (size_t) k : (size_t) k / 2, srow = FMT == kFp8 ? (size_t) k / 32 : (size_t) k / 16;
    int rr[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) rr[r] = min(row0 + r, n - 1);
    const int steps = k / 256;
#pragma unroll 2
    for (int s = 0; s < steps; ++s) {
        const int j0 = s * 256 + lane * 8;   // (the lane's 8 weights: one 32-block (kFp8) or one 16-block (kNvfp4))
        uint2 b8[RW];
        uint32_t b4[RW], e[RW];
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            if constexpr (FMT == kFp8) b8[r] = __ldg((const uint2*) (w + rr[r] * wrow + j0));
            else b4[r] = __ldg((const uint32_t*) (w + rr[r] * wrow + j0 / 2));
            e[r] = __ldg(sc + rr[r] * srow + (FMT == kFp8 ? j0 / 32 : j0 / 16));
        }
        float xv[NC][8];
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c < ncol) {
                const float4 a = __ldg((const float4*) (x + (size_t) c * ldx + j0)), b = __ldg((const float4*) (x + (size_t) c * ldx + j0 + 4));
                xv[c][0] = a.x, xv[c][1] = a.y, xv[c][2] = a.z, xv[c][3] = a.w, xv[c][4] = b.x, xv[c][5] = b.y, xv[c][6] = b.z, xv[c][7] = b.w;
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[c][i] = 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < RW; ++r) {
            float wf[8], scale;
            if constexpr (FMT == kFp8) {
                fp8x8(b8[r], wf);
                scale = e[r] ? __uint_as_float(e[r] << 23) : __uint_as_float(0x00400000u);   // (2^(e - 127))
            } else {
                fp4x8(b4[r], wf);
                scale = 0.5f * e4m3_float((uint8_t) e[r]);   // (the codes are doubled)
            }
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) p = fmaf(wf[i], xv[c][i], p);
                acc[r][c] = fmaf(p, scale, acc[r][c]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        if (row0 + r >= n) break;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c >= ncol) break;
            float v = acc[r][c];
            for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
            if (lane == 0) y[(size_t) c * ldy + row0 + r] = FMT == kFp8 ? v : v * global;
        }
    }
}

template <int FMT, int NC>
bool launch_nc(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, int rw, int wb, cudaStream_t s) {
#define HEAD_CASE(RW, WB)                                                                                                 \
    if (rw == RW && wb == WB) {                                                                                           \
        head_mv<FMT, NC, RW, WB><<<(unsigned) ((h.n + RW * WB - 1) / (RW * WB)), WB * 32, 0, s>>>(h.w, h.sc, h.n, h.k,  \
                                                                                                 h.global, x, ldx, y, \
                                                                                                 ldy, ncol);          \
        return cudaPeekAtLastError() == cudaSuccess;                                                                     \
    }
    HEAD_CASE(1, 4) HEAD_CASE(2, 4) HEAD_CASE(4, 4) HEAD_CASE(8, 4) HEAD_CASE(2, 8) HEAD_CASE(4, 8) HEAD_CASE(4, 2)
#undef HEAD_CASE
    return false;
}
template <int FMT>
bool launch_fmt(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, int rw, int wb, cudaStream_t s) {
    if (ncol <= 1) return launch_nc<FMT, 1>(h, x, ldx, ncol, y, ldy, rw, wb, s);
    if (ncol <= 2) return launch_nc<FMT, 2>(h, x, ldx, ncol, y, ldy, rw, wb, s);
    if (ncol <= 4) return launch_nc<FMT, 4>(h, x, ldx, ncol, y, ldy, rw, wb, s);
    return launch_nc<FMT, 8>(h, x, ldx, ncol, y, ldy, rw, wb, s);
}
std::atomic<int> g_shape_mode{0};
// Original shape retained for paired timing. The candidate follows the real-head
// sweep: greater row reuse helps wide head passes without changing each sum.
void default_shape(int fmt, int ncol, int& rw, int& wb) {
    (void) fmt;
    wb = 4;
    rw = ncol <= 1 ? 4 : ncol <= 2 ? 4 : ncol <= 4 ? 4 : 2;
    if (g_shape_mode.load(std::memory_order_relaxed) == 1 && ncol >= 2) rw = 8;
}

}  // namespace

void set_shape_mode(int mode) { g_shape_mode.store(mode == 1 ? 1 : 0, std::memory_order_relaxed); }
int shape_mode() { return g_shape_mode.load(std::memory_order_relaxed); }

bool shape_ok(int fmt, int n, int k) {
    return n > 0 && k > 0 && k % 256 == 0 && (fmt == kFp8 || fmt == kNvfp4);
}
size_t weight_bytes(int fmt, int n, int k) { return fmt == kFp8 ? (size_t) n * k : (size_t) n * k / 2; }
size_t scale_bytes(int fmt, int n, int k) { return fmt == kFp8 ? (size_t) n * k / 32 : (size_t) n * k / 16; }

void release(Head& h) {
    if (h.w) cudaFree(h.w);
    if (h.sc) cudaFree(h.sc);
    if (h.amax) cudaFree(h.amax);
    h = Head{};
}

bool make(int fmt, const uint16_t* bf16, int n, int k, Head& out, cudaStream_t s, const char** err) {
    auto fail = [&](const char* why) { if (err) *err = why; release(out); return false; };
    out = Head{};
    if (!bf16 || !shape_ok(fmt, n, k)) return fail("head_lowp: unsupported head shape");
    out.fmt = fmt, out.n = n, out.k = k;
    if (cudaMalloc(&out.w, weight_bytes(fmt, n, k)) != cudaSuccess || cudaMalloc(&out.sc, scale_bytes(fmt, n, k)) != cudaSuccess) {
        cudaGetLastError();
        return fail("head_lowp: out of device memory");
    }
    if (fmt == kFp8) {
        quant_fp8<<<2048, 256, 0, s>>>(bf16, n, k, out.w, out.sc);
    } else {
        if (cudaMalloc(&out.amax, sizeof(float)) != cudaSuccess) { cudaGetLastError(); return fail("head_lowp: out of device memory"); }
        cudaMemsetAsync(out.amax, 0, sizeof(float), s);
        amax_kernel<<<2048, 256, 0, s>>>(bf16, (size_t) n * k, out.amax);
        quant_nvfp4<<<2048, 256, 0, s>>>(bf16, n, k, out.amax, out.w, out.sc);
        float amax = 0.0f;
        if (cudaMemcpyAsync(&amax, out.amax, sizeof amax, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
            cudaStreamSynchronize(s) != cudaSuccess)
            return fail("head_lowp: quantization failed");
        out.global = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;   // (the kernel's own expression: the same float)
    }
    if (cudaStreamSynchronize(s) != cudaSuccess || cudaGetLastError() != cudaSuccess) return fail("head_lowp: quantization failed");
    return true;
}

bool mv_shape(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, int rw, int wb, cudaStream_t s) {
    if (!h.w || !x || !y || ncol < 1 || ncol > 8 || ldx < h.k || ldy < h.n || (ldx % 4) || ((uintptr_t) x & 15u)) return false;
    return h.fmt == kFp8 ? launch_fmt<kFp8>(h, x, ldx, ncol, y, ldy, rw, wb, s)
                         : launch_fmt<kNvfp4>(h, x, ldx, ncol, y, ldy, rw, wb, s);
}

bool mv(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, cudaStream_t s) {
    for (int c0 = 0; c0 < ncol; c0 += 8) {   // (more than 8 columns: in runs of 8)
        const int nc = std::min(8, ncol - c0);
        int rw = 4, wb = 4;
        default_shape(h.fmt, nc, rw, wb);
        if (!mv_shape(h, x + (size_t) c0 * ldx, ldx, nc, y + (size_t) c0 * ldy, ldy, rw, wb, s)) return false;
    }
    return true;
}

}  // namespace mm3::head_lowp
