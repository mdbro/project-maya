// Run on a CUDA device after the native dense sources are linked with mm3 kernels/exl3.
#include "mm3_native_dense.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
using namespace strata::mm3;
#define CUDA(x) assert((x) == cudaSuccess)
int main() {
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
}
