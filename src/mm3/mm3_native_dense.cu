#include "mm3_native_dense.cuh"
#include "mm3_kernels.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>

namespace strata::mm3::native_dense {
namespace {
struct Context { int k; uint8_t* mem; size_t bytes, partial_offset; cublasHandle_t blas; };
using Key = std::pair<int, cudaStream_t>;
std::mutex mu;
std::map<Key, Context> contexts;
bool get(cudaStream_t s, Context& out) {
    int device; if (cudaGetDevice(&device) != cudaSuccess) return false;
    std::lock_guard<std::mutex> lock(mu);
    auto it = contexts.find({device, s});
    if (it == contexts.end()) return false;
    out = it->second; return true;
}
__global__ void f16_bf16(const __half* x, int ldx, __nv_bfloat16* out, int k, int rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (size_t)rows * k;
    if (i < count) out[i] = __float2bfloat16(__half2float(x[(i / k) * ldx + i % k]));
}
__global__ void bf16_mv(const __nv_bfloat16* w, int k, int n, const float* x, size_t xs,
                         float* y, size_t ys, const float* bias, float alpha) {
    int lane = threadIdx.x & 31, row = blockIdx.x * 4 + threadIdx.x / 32, c = blockIdx.y;
    if (row >= n) return;
    float sum = 0;
    for (int j = lane; j < k; j += 32) sum = fmaf(__bfloat162float(w[(size_t)row * k + j]), x[(size_t)c * xs + j], sum);
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) y[(size_t)c * ys + row] = alpha * sum + (bias ? bias[row] : 0.0f);
}
// bf16_mv for several columns at once: each weight read once for all of them (bf16_mv's grid re-read w once a column:
// a verify window's or a draft block's head pass read the 2.46 GB head once per row). A warp takes RW rows and a block
// WB warps; x comes from global memory (L1/L2). Every (row, column) sum runs bf16_mv's lane order (lane l adds j = l,
// l + 32, ... in increasing order) and its reduction, so every result is bf16_mv's, bit for bit.
// Second form, 2026-11-09: the first (f18bfdd: 4 rows a warp) broke out of its row loop at n, which serialized each
// warp's four weight loads, and held the head pass at 920 GB/s at 2-8 columns against bf16_mv's 1,660 at one column.
// Every row's load is unconditional here (rows past n read the last row; their sums are never written), so a lane keeps
// RW rows' loads of several steps in flight. Head-shaped sweep (200064 x 6144, the 5090; tmp/mvbench): RW 12 / WB 4 at
// up to 4 columns 1.49-1.50 ms, RW 8 / WB 4 at 5-8 columns 1.56-1.62 ms, against 2.66-2.71 ms first and 1.48 ms for one
// column. Staging x in shared memory a K-tile at a time was slower (1.76-1.91 ms: the block syncs).
template <int NC, int RW, int WB>
__global__ void __launch_bounds__(WB * 32) bf16_mv_cols(const __nv_bfloat16* w, int k, int n, const float* x, size_t xs,
                                                        float* y, size_t ys, const float* bias, float alpha, int ncol) {
    const int lane = threadIdx.x & 31, row0 = (blockIdx.x * WB + (threadIdx.x >> 5)) * RW;
    if (row0 >= n) return;
    float sum[RW][NC];
#pragma unroll
    for (int r = 0; r < RW; ++r)
#pragma unroll
        for (int c = 0; c < NC; ++c) sum[r][c] = 0.0f;
    const __nv_bfloat16* wr[RW];
#pragma unroll
    for (int r = 0; r < RW; ++r) wr[r] = w + (size_t)min(row0 + r, n - 1) * k;
#pragma unroll 4
    for (int j = lane; j < k; j += 32) {
        float xv[NC], wv[RW];
#pragma unroll
        for (int c = 0; c < NC; ++c) xv[c] = c < ncol ? __ldg(x + (size_t)c * xs + j) : 0.0f;
#pragma unroll
        for (int r = 0; r < RW; ++r) wv[r] = __bfloat162float(wr[r][j]);
#pragma unroll
        for (int r = 0; r < RW; ++r)
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < ncol) sum[r][c] = fmaf(wv[r], xv[c], sum[r][c]);
    }
#pragma unroll
    for (int r = 0; r < RW; ++r) {
        if (row0 + r >= n) break;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (c >= ncol) break;
            float v = sum[r][c];
            for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
            if (lane == 0) y[(size_t)c * ys + row0 + r] = alpha * v + (bias ? bias[row0 + r] : 0.0f);
        }
    }
}
__global__ void post(float* y, size_t ys, int n, int cols, const float* bias, float alpha) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n * cols) { int c = i / n, r = i % n; float &v = y[(size_t)c * ys + r]; v = alpha * v + (bias ? bias[r] : 0.0f); }
}
}
size_t scratch_bytes(int k, bool allow_gemm) {
    if (k <= 0) return 0;
    return (allow_gemm ? (size_t)(kBand + kRows) * k * 2 : 0) + (size_t)16 * kBand * std::max(16, (k + 511) / 512) * sizeof(float);
}
bool prepare(cudaStream_t s, int max_k, void* p, size_t bytes, bool allow_gemm) {
    if (!p || max_k <= 0 || bytes < scratch_bytes(max_k, allow_gemm)) return false;
    {   // (MAYA_MM3_MV_COLS=0: BF16 matrices back to a pass a column - the same bits, for an A/B of the one-pass kernel)
        const char* v = std::getenv("MAYA_MM3_MV_COLS");
        if (v != nullptr && std::atoi(v) == 0) set_mv_cols(false);
    }
    int device; if (cudaGetDevice(&device) != cudaSuccess) return false;
    std::lock_guard<std::mutex> lock(mu);
    if (contexts.count({device, s})) return false;
    cublasHandle_t h = nullptr;
    if (allow_gemm) {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return false;
        if (cublasSetStream(h, s) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(h); return false; }
    }
    const size_t partial_offset = allow_gemm ? (size_t)(kBand+kRows)*max_k*2 : 0;
    contexts.emplace(Key{device, s}, Context{max_k, (uint8_t*)p, bytes, partial_offset, h}); return true;
}
void release(cudaStream_t s) {
    int device; if (cudaGetDevice(&device) != cudaSuccess) return;
    std::lock_guard<std::mutex> lock(mu);
    auto it = contexts.find({device, s});
    if (it != contexts.end()) { if (it->second.blas) cublasDestroy(it->second.blas); contexts.erase(it); }
}
namespace { std::atomic<int> g_split_n{kBand}; std::atomic<bool> g_mv_cols_off{false}; }
void set_split_n(int n) { g_split_n.store(n, std::memory_order_relaxed); }
int split_n() { return g_split_n.load(std::memory_order_relaxed); }
void set_mv_cols(bool on) { g_mv_cols_off.store(!on, std::memory_order_relaxed); }
bool mv_cols_on() { return !g_mv_cols_off.load(std::memory_order_relaxed); }
bool mv(const exl3::MvJob& j, cudaStream_t s) {
    if (!j.m || !exl3::supported(*j.m) || !j.xf || !j.y || j.ncol < 1 || j.ncol > 16) return false;
    const auto& m = *j.m;
    Context c; if (!get(s, c) || m.k > c.k) return false;
    size_t xs = j.ncol == 1 ? (size_t)m.k : j.xf_cs, ys = j.ncol == 1 ? (size_t)m.n : j.y_cs;
    if (xs < (size_t)m.k || ys < (size_t)m.n) return false;
    if (m.fmt == kFmtBF16) {
        if (j.ncol == 1 || g_mv_cols_off.load(std::memory_order_relaxed))
            bf16_mv<<<dim3((m.n + 3)/4, j.ncol), 128, 0, s>>>((const __nv_bfloat16*)m.raw, m.k, m.n, j.xf, xs, j.y, ys, j.bias, j.alpha);
        else if (j.ncol <= 4)   // (48 rows a block: 4 warps of 12 rows)
            bf16_mv_cols<4, 12, 4><<<(m.n + 47)/48, 128, 0, s>>>((const __nv_bfloat16*)m.raw, m.k, m.n, j.xf, xs, j.y, ys,
                                                                 j.bias, j.alpha, j.ncol);
        else if (j.ncol <= 8)   // (32 rows a block: 4 warps of 8 rows)
            bf16_mv_cols<8, 8, 4><<<(m.n + 31)/32, 128, 0, s>>>((const __nv_bfloat16*)m.raw, m.k, m.n, j.xf, xs, j.y, ys,
                                                                j.bias, j.alpha, j.ncol);
        else                    // (16 rows a block: 4 warps of 4 rows; not swept - beyond a verify window or a draft block)
            bf16_mv_cols<16, 4, 4><<<(m.n + 15)/16, 128, 0, s>>>((const __nv_bfloat16*)m.raw, m.k, m.n, j.xf, xs, j.y, ys,
                                                                 j.bias, j.alpha, j.ncol);
    } else {
        // For small output matrices use bounded split-K partials; large ones
        // already offer enough row blocks to saturate the device.
        float* part = m.n <= split_n() ? (float*)(c.mem + c.partial_offset) : nullptr;
        ::mm3::k::mx_rows((const uint8_t*)m.raw, m.native_scale, m.n, m.k, j.xf, j.ncol, (int)xs, j.y, (int)ys, 0.0f, part, s);
        if (j.alpha != 1.0f || j.bias) post<<<((size_t)m.n*j.ncol+255)/256, 256, 0, s>>>(j.y, ys, m.n, j.ncol, j.bias, j.alpha);
    }
    return cudaPeekAtLastError() == cudaSuccess;
}
bool gemm_rows(const exl3::Mat& m, const uint16_t* x, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
    Context c;
    if (rows <= 0) return !get(s, c) || c.blas != nullptr;
    if (!get(s, c) || !c.blas) return false;
    if (!exl3::supported(m) || m.k > c.k || !x || !y || ldx < m.k || ldy < m.n) return false;
    auto* wbuf = (__nv_bfloat16*)c.mem;
    auto* xbuf = (__nv_bfloat16*)(c.mem + (size_t)kBand*c.k*2);
    const float one = 1.0f;
    for (int r = 0; r < rows; r += kRows) {
        int nr = std::min(kRows, rows-r);
        f16_bf16<<<((size_t)nr*m.k+255)/256,256,0,s>>>((const __half*)x+(size_t)r*ldx, ldx, xbuf, m.k, nr);
        const int band = m.fmt == kFmtBF16 ? m.n : kBand;
        for (int b = 0; b < m.n; b += band) {
            int nb = std::min(band, m.n-b);
            const __nv_bfloat16* w = (const __nv_bfloat16*)m.raw + (size_t)b*m.k;
            if (m.fmt == kFmtMXFP8) {
                ::mm3::k::mxfp8_to_bf16((const uint8_t*)m.raw+(size_t)b*m.k, m.native_scale+(size_t)b*(m.k/32), nb, m.k, (uint16_t*)wbuf, s);
                w = wbuf;
            }
            if (cublasGemmEx(c.blas, CUBLAS_OP_T, CUBLAS_OP_N, nb, nr, m.k, &one,
                             w, CUDA_R_16BF, m.k, xbuf, CUDA_R_16BF, m.k, &beta,
                             y+(size_t)r*ldy+b, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) return false;
        }
    }
    return cudaPeekAtLastError() == cudaSuccess;
}
}
