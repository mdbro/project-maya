// Run on a CUDA device after the native dense sources are linked with mm3 kernels/exl3.
#include "mm3_native_dense.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace strata::mm3;
#define CUDA(x) assert((x) == cudaSuccess)
// (the head bench's weights and inputs, made on the device: 1.2 G weights would take long to fill on the host)
__global__ void bench_fill(uint16_t* w, size_t n, float scale, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        const float f = ((float)(h & 0xffff) / 32768.0f - 1.0f) * scale;
        w[i] = (uint16_t)(__float_as_uint(f) >> 16);
    }
}
__global__ void bench_fill_f(float* x, size_t n, float scale, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        x[i] = ((float)(h & 0xffff) / 32768.0f - 1.0f) * scale;
    }
}
// MM3_DENSE_BENCH=1: the target's head shape (200064 x 6144 BF16, 2.46 GB): the several-column pass against the per-column
// grid, bit for bit at 7 and 8 columns, and each one's time and GB/s (events, median of 7 after a warm-up)
static void head_bench() {
    constexpr int K = 6144, N = 200064, C = 8;
    cudaStream_t s; CUDA(cudaStreamCreate(&s));
    size_t sb = native_dense::scratch_bytes(K, false); void* sc; CUDA(cudaMalloc(&sc, sb));
    assert(native_dense::prepare(s, K, sc, sb, false));
    uint16_t* w; float *x, *y;
    CUDA(cudaMalloc(&w, (size_t)N * K * 2)); CUDA(cudaMalloc(&x, (size_t)C * K * 4)); CUDA(cudaMalloc(&y, (size_t)2 * C * N * 4));
    bench_fill<<<4096, 256, 0, s>>>(w, (size_t)N * K, 0.05f, 777u);
    bench_fill_f<<<64, 256, 0, s>>>(x, (size_t)C * K, 3.0f, 991u);
    CUDA(cudaStreamSynchronize(s));
    exl3::Mat m; m.fmt = native_dense::kFmtBF16; m.k = K; m.n = N; m.raw = w;
    cudaEvent_t e0, e1; CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
    auto run = [&](int cols, bool several, int slot) {
        native_dense::set_mv_cols(several);
        exl3::MvJob j; j.m = &m; j.xf = x; j.y = y + (size_t)slot * C * N; j.alpha = 1.0f; j.ncol = cols; j.xf_cs = K; j.y_cs = N;
        assert(native_dense::mv(j, s));
    };
    auto timed = [&](int cols, bool several) {
        run(cols, several, several ? 1 : 0);
        std::vector<float> t;
        for (int rep = 0; rep < 7; ++rep) {
            CUDA(cudaEventRecord(e0, s)); run(cols, several, several ? 1 : 0); CUDA(cudaEventRecord(e1, s));
            CUDA(cudaEventSynchronize(e1)); float ms = 0; CUDA(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        return t[3];
    };
    std::vector<float> a((size_t)C * N), b((size_t)C * N);
    for (int cols : {1, 2, 4, 7, 8}) {
        const float t_one = timed(cols, false), t_several = cols > 1 ? timed(cols, true) : t_one;
        if (cols > 1) {
            CUDA(cudaMemcpy(a.data(), y, (size_t)cols * N * 4, cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(b.data(), y + (size_t)C * N, (size_t)cols * N * 4, cudaMemcpyDeviceToHost));
            assert(std::memcmp(a.data(), b.data(), (size_t)cols * N * 4) == 0);
        }
        const double gb = (double)N * K * 2 / 1e9;
        std::printf("head bench, %d column%s: a pass a column %.3f ms; several-column pass %.3f ms (%.0f GB/s)%s\n", cols,
                    cols > 1 ? "s" : "", t_one, t_several, gb / t_several * 1e3, cols > 1 ? ", bit for bit" : "");
    }
    native_dense::set_mv_cols(true);
    native_dense::release(s); CUDA(cudaFree(sc)); CUDA(cudaFree(w)); CUDA(cudaFree(x)); CUDA(cudaFree(y));
    CUDA(cudaEventDestroy(e0)); CUDA(cudaEventDestroy(e1)); CUDA(cudaStreamDestroy(s));
}
int main() {
    if (const char* v = std::getenv("MM3_DENSE_BENCH"); v && std::atoi(v) == 1) { head_bench(); return 0; }
    constexpr int K=512, N=64, R=4;
    cudaStream_t s; CUDA(cudaStreamCreate(&s));
    size_t bytes=native_dense::scratch_bytes(K); void* scratch; CUDA(cudaMalloc(&scratch, bytes));
    assert(native_dense::prepare(s,K,scratch,bytes));
    // E4M3 0x38 = 1, E8M0 127 = 1; corresponding BF16 matrix is 1.
    uint8_t *w,*sc; float *x,*y,*bias; uint16_t *xh,*bw;
    CUDA(cudaMalloc(&w,N*K)); CUDA(cudaMalloc(&sc,N*K/32));
    CUDA(cudaMalloc(&bw,N*K*2)); CUDA(cudaMalloc(&x,R*K*4)); CUDA(cudaMalloc(&xh,R*K*2));
    CUDA(cudaMalloc(&y,R*N*4)); CUDA(cudaMalloc(&bias,N*4));
    std::vector<uint8_t> hw(N*K,0x38), hs(N*K/32,127);
    std::vector<float> hx(R*K, 0.25f), hy(R*N);
    std::vector<uint16_t> hh(R*K, 0x3400), hb(N*K, 0x3f80);
    std::fill(hw.begin()+32*K,hw.end(),0x40); // E4M3 2
    std::fill(hb.begin()+32*K,hb.end(),0x4000); // BF16 2
    std::vector<float> biases(N,3.0f);
    CUDA(cudaMemcpy(w,hw.data(),hw.size(),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(sc,hs.data(),hs.size(),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(bw,hb.data(),hb.size()*2,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(x,hx.data(),hx.size()*4,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(xh,hh.data(),hh.size()*2,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(bias,biases.data(),biases.size()*4,cudaMemcpyHostToDevice));
    for (int fmt : {native_dense::kFmtMXFP8,native_dense::kFmtBF16}) {
        exl3::Mat m; m.fmt=fmt; m.k=K; m.n=N; m.raw=fmt==5?(void*)w:(void*)bw; m.native_scale=sc;
        assert(exl3::supported(m)); auto v=exl3::view(m,32,32); assert(exl3::supported(v));
        exl3::MvJob j; j.m=&v; j.xf=x; j.y=y; j.bias=bias; j.ncol=R; j.xf_cs=K; j.y_cs=N; j.alpha=0.5f;
        assert(exl3::mv(&j,1,s)); CUDA(cudaStreamSynchronize(s));
        CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<R;++r) for(int c=0;c<32;++c) assert(std::abs(hy[r*N+c]-131.0f)<0.02f);
        CUDA(cudaMemsetAsync(y,0,R*N*4,s));
        exl3::gemm_rows(m,xh,K,R,y,N,0,s); CUDA(cudaStreamSynchronize(s));
        CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<R;++r) for(int c=0;c<N;++c) assert(std::abs(hy[r*N+c]-(c<32?128.0f:256.0f))<0.02f);
        exl3::gemm_rows(m,xh,K,R,y,N,1,s); CUDA(cudaStreamSynchronize(s));
        CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<R;++r) for(int c=0;c<N;++c) assert(std::abs(hy[r*N+c]-(c<32?256.0f:512.0f))<0.02f);
    }
    native_dense::release(s); CUDA(cudaFree(scratch));
    // Selected MV-only context: small N exercises the split-K partial buffer
    // at offset zero. Both single-token and verification width use the same
    // reference as the full workspace, and GEMM must be rejected.
    const size_t compact=native_dense::scratch_bytes(K,false);
    assert(compact < native_dense::scratch_bytes(K));
    CUDA(cudaMalloc(&scratch,compact));
    assert(!native_dense::prepare(s,K,scratch,compact-1,false));
    assert(native_dense::prepare(s,K,scratch,compact,false));
    exl3::Mat m; m.fmt=native_dense::kFmtMXFP8;m.k=K;m.n=N;m.raw=w;m.native_scale=sc;
    auto v=exl3::view(m,32,32);assert(exl3::supported(v));
    for(int cols:{1,4}) {
        CUDA(cudaMemsetAsync(y,0,R*N*4,s));
        exl3::MvJob j;j.m=&v;j.xf=x;j.y=y;j.bias=bias;j.ncol=cols;j.xf_cs=K;j.y_cs=N;j.alpha=0.5f;
        assert(native_dense::mv(j,s));CUDA(cudaStreamSynchronize(s));
        CUDA(cudaMemcpy(hy.data(),y,hy.size()*4,cudaMemcpyDeviceToHost));
        for(int r=0;r<cols;++r)for(int c=0;c<32;++c)assert(std::abs(hy[r*N+c]-131.f)<0.02f);
    }
    assert(!native_dense::gemm_rows(m,xh,K,R,y,N,0,s));
    assert(!native_dense::gemm_rows(m,xh,K,0,y,N,0,s));
    native_dense::release(s);CUDA(cudaFree(scratch));
    CUDA(cudaFree(w)); CUDA(cudaFree(sc)); CUDA(cudaFree(bw)); CUDA(cudaFree(x)); CUDA(cudaFree(xh)); CUDA(cudaFree(y)); CUDA(cudaFree(bias)); CUDA(cudaStreamDestroy(s));
    // BF16 with several columns: the one-pass kernel (each weight read once for all columns) against bf16_mv's grid (a
    // pass a column), bit for bit, for 2..16 columns - random weights and inputs, a head's width, rows not a multiple of
    // 16 (the last block's bound) and a bias with alpha.
    {
        constexpr int K2 = 6144, N2 = 1001, C2 = 16;
        cudaStream_t s2; CUDA(cudaStreamCreate(&s2));
        size_t b2 = native_dense::scratch_bytes(K2, false); void* sc2; CUDA(cudaMalloc(&sc2, b2));
        assert(native_dense::prepare(s2, K2, sc2, b2, false));
        std::vector<uint16_t> hw2((size_t)N2 * K2); std::vector<float> hx2((size_t)C2 * K2), hb2(N2);
        uint32_t st = 12345u;
        auto rnd = [&st]() { st = st * 1664525u + 1013904223u; return (float)((st >> 8) & 0xffff) / 32768.0f - 1.0f; };
        for (auto& v : hw2) { float f = rnd() * 0.05f; uint32_t u; std::memcpy(&u, &f, 4); v = (uint16_t)(u >> 16); }
        for (auto& v : hx2) v = rnd() * 3.0f;
        for (auto& v : hb2) v = rnd();
        uint16_t* dw2; float *dx2, *dy2, *db2;
        CUDA(cudaMalloc(&dw2, hw2.size() * 2)); CUDA(cudaMalloc(&dx2, hx2.size() * 4));
        CUDA(cudaMalloc(&dy2, (size_t)2 * C2 * N2 * 4)); CUDA(cudaMalloc(&db2, (size_t)N2 * 4));
        CUDA(cudaMemcpy(dw2, hw2.data(), hw2.size() * 2, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dx2, hx2.data(), hx2.size() * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(db2, hb2.data(), hb2.size() * 4, cudaMemcpyHostToDevice));
        exl3::Mat m2; m2.fmt = native_dense::kFmtBF16; m2.k = K2; m2.n = N2; m2.raw = dw2;
        std::vector<float> a((size_t)C2 * N2), b((size_t)C2 * N2);
        for (int cols = 2; cols <= C2; ++cols) {
            for (int pass = 0; pass < 2; ++pass) {
                native_dense::set_mv_cols(pass == 1);
                exl3::MvJob j; j.m = &m2; j.xf = dx2; j.y = dy2 + (size_t)pass * C2 * N2; j.bias = db2; j.alpha = 0.75f;
                j.ncol = cols; j.xf_cs = K2; j.y_cs = N2;
                assert(native_dense::mv(j, s2));
            }
            CUDA(cudaStreamSynchronize(s2));
            CUDA(cudaMemcpy(a.data(), dy2, (size_t)cols * N2 * 4, cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(b.data(), dy2 + (size_t)C2 * N2, (size_t)cols * N2 * 4, cudaMemcpyDeviceToHost));
            assert(std::memcmp(a.data(), b.data(), (size_t)cols * N2 * 4) == 0);
        }
        native_dense::set_mv_cols(true);
        native_dense::release(s2); CUDA(cudaFree(sc2)); CUDA(cudaFree(dw2)); CUDA(cudaFree(dx2)); CUDA(cudaFree(dy2));
        CUDA(cudaFree(db2)); CUDA(cudaStreamDestroy(s2));
        std::printf("native dense: BF16 several-column pass bit for bit against the per-column grid, 2..16 columns\n");
    }
}
