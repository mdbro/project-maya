#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <map>
#include <mutex>

namespace strata::mm3::native_experts {
namespace donor = infernix::ops::minimax_ffn;
namespace {
constexpr int B=4, C=8;
size_t align256(size_t x) { return (x+255)&~size_t(255); }
struct Context { int h,ff; uint8_t* p; size_t bytes; };
using Key=std::pair<int,cudaStream_t>;
std::mutex mu; std::map<Key,Context> ctx;
bool lookup(cudaStream_t s, Context& c) { int d; if(cudaGetDevice(&d)!=cudaSuccess)return false; std::lock_guard<std::mutex> l(mu); auto i=ctx.find({d,s}); if(i==ctx.end())return false; c=i->second; return true; }
struct Buffers { donor::FastJob* jobs; uint16_t* in; uint16_t* out; float* partial; uint8_t* work; };
Buffers buffers(const Context& c) {
    auto* p=c.p; Buffers b; b.jobs=(donor::FastJob*)p; p+=align256(B*sizeof(donor::FastJob));
    b.in=(uint16_t*)p; p+=align256((size_t)B*C*c.h*2);
    b.out=(uint16_t*)p; p+=align256((size_t)B*C*c.h*2);
    b.partial=(float*)p; p+=align256((size_t)8*c.h*4); b.work=p; return b;
}
// mode 0: token slots, mode 1: device-grouped window, mode 2: prompt set.
struct Setup { int mode, base, h, T, k, set_e, set_start, set_n; const unsigned long long *ptr,*gp; const int *gs,*ge,*ng,*rowidx; const uint8_t* set_blob; const float* x; size_t xs; };
// Enumerate the existing, at-most-two eight-column pieces per group densely.
// Each piece contains at least one plan entry, so T*k bounds the job count.
// The device plan stays asynchronous; unused tail jobs are still allowed.
__device__ void window_piece(const Setup& a, int idx, int& g, int& start, int& n) {
    const int count=*a.ng;
    for (g=0;g<count;++g) {
        const int first=a.gs[g], len=a.gs[g+1]-first;
        const int pieces=(len>0)+(len>C);
        if (idx<pieces) { start=first+idx*C; n=min(C,len-idx*C); return; }
        idx-=pieces;
    }
    start=0; n=0;
}
__global__ void setup(Setup a, donor::FastJob* jobs, uint16_t* in, uint16_t* out) {
    int job=blockIdx.x, tid=threadIdx.x, idx=a.base+job;
    __shared__ int n,start; __shared__ const uint8_t* blob;
    if(tid==0) {
        n=0; start=0; blob=nullptr;
        if(a.mode==0) { if(idx<a.k) { blob=(const uint8_t*)a.ptr[idx]; n=blob?1:0; } }
        else if(a.mode==1) { int g; window_piece(a,idx,g,start,n); if(n) { blob=(const uint8_t*)a.gp[g]; if(!blob)n=0; } }
        else { n=a.set_n; start=a.set_start; blob=a.set_blob; }
        donor::FastJob j{}; j.record=blob?blob+sizeof(Header):nullptr; j.x=in+(size_t)job*C*a.h; j.y=out+(size_t)job*C*a.h; j.columns=n;
        if(n) { const Header* h=(const Header*)blob; for(int k=0;k<3;++k) j.multiplier[k]=1.0f/h->weight_global[k]; } jobs[job]=j;
    }
    __syncthreads();
    for(int i=tid;i<n*a.h;i+=blockDim.x) {
        int col=i/a.h, t=0;
        if(a.mode==1) t=a.ge[start+col]/8;
        else if(a.mode==2) t=a.rowidx[start+col];
        in[((size_t)job*C+col)*a.h+i%a.h]=__bfloat16_as_ushort(__float2bfloat16(a.x[(size_t)t*a.xs+i%a.h]));
    }
}
__global__ void scatter(Setup a, const donor::FastJob* jobs, const uint16_t* out, float* part, size_t ys) {
    int job=blockIdx.x, idx=a.base+job, n=jobs[job].columns; __shared__ int start;
    if (!n) return;
    if(threadIdx.x==0) { start=0; if(a.mode==1) { int g, count; window_piece(a,idx,g,start,count); } else if(a.mode==2) start=a.set_start; }
    __syncthreads();
    for(int i=threadIdx.x;i<n*a.h;i+=blockDim.x) {
        int col=i/a.h, dst=idx;
        if(a.mode==1) dst=a.ge[start+col];
        else if(a.mode==2) dst=start+col;
        part[(size_t)dst*ys+i%a.h]=__bfloat162float(__ushort_as_bfloat16(out[((size_t)job*C+col)*a.h+i%a.h]));
    }
}
__global__ void wait_publication(const unsigned* p, unsigned value) {
    if (threadIdx.x) return;
    while (*(volatile const unsigned*)p != value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}
__global__ void combine_token(exl3::MoeArgs a,const float* part,const float* sh,float* out) {
    int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=a.n_embd)return;
    float v=sh?sh[d]:0.f; for(int k=0;k<a.k;++k) if(a.plan_ptr[k]) v+=a.plan_w[k]*part[(size_t)k*a.n_embd+d];
    if(a.cpu_part && a.cpu_flag && *a.cpu_flag) v+=a.cpu_part[d]; out[d]=v;
}
__global__ void combine_window(exl3::WinArgs a) {
    int d=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y; if(d>=a.n_embd)return;
    float v=a.sh_out?a.sh_out[(size_t)t*a.n_embd+d]:0.f;
    for(int k=0;k<a.k;++k) {int e=t*8+k;if(a.plan_ptr[e])v+=a.plan_w[e]*a.part[(size_t)e*a.n_embd+d];}
    if(a.cpu_part && a.cpu_flag && a.cpu_flag[t]) v+=a.cpu_part[(size_t)t*(a.cpu_stride?a.cpu_stride:a.n_embd)+d];
    a.out[(size_t)t*a.out_stride+d]=v;
}
bool run(const Context& c,Setup a,int batches,float* part,size_t ys,cudaStream_t s) {
    auto b=buffers(c);
    try { for(int at=0;at<batches;at+=B) { a.base=at; int n=std::min(B,batches-at); setup<<<n,256,0,s>>>(a,b.jobs,b.in,b.out);
        donor::fast_ffn_cuda_packed(b.jobs,n,c.h,c.ff,C,b.work,s);
        scatter<<<n,256,0,s>>>(a,b.jobs,b.out,part,ys);
    }} catch(const std::exception& e) { std::fprintf(stderr,"native experts: %s\n",e.what()); return false; }
    return cudaPeekAtLastError()==cudaSuccess;
}
}
size_t scratch_bytes(int h,int ff) {
    if(h<=0||ff<=0)return 0;
    try { return align256(B*sizeof(donor::FastJob))+2*align256((size_t)B*C*h*2)+align256((size_t)8*h*4)+B*donor::fast_workspace_bytes(h,ff,C); } catch(...) { return 0; }
}
bool prepare(cudaStream_t s,int h,int ff,void* p,size_t bytes) {
    size_t need=scratch_bytes(h,ff); if(!need||!p||((uintptr_t)p%256)||bytes<need)return false;
    int d;if(cudaGetDevice(&d)!=cudaSuccess)return false;std::lock_guard<std::mutex> l(mu);
    return ctx.emplace(Key{d,s},Context{h,ff,(uint8_t*)p,bytes}).second;
}
void release(cudaStream_t s) { int d;if(cudaGetDevice(&d)!=cudaSuccess)return;std::lock_guard<std::mutex> l(mu);ctx.erase({d,s}); }
bool gate_up(const exl3::MoeArgs& a,const float* x,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||!x||!a.plan_ptr||a.k<1||a.k>8)return false;
    Setup z{};z.mode=0;z.h=c.h;z.k=a.k;z.ptr=a.plan_ptr;z.x=x;z.xs=c.h;
    return run(c,z,a.k,buffers(c).partial,c.h,s);
}
bool down(const exl3::MoeArgs& a,const float* sh,float* out,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||!a.plan_w||!a.plan_ptr||!out)return false;
    combine_token<<<(c.h+255)/256,256,0,s>>>(a,buffers(c).partial,sh,out);return cudaPeekAtLastError()==cudaSuccess;
}
bool window(const exl3::WinArgs& a,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||a.T<1||a.T>16||a.k<1||a.k>8||!a.grp_ptr||!a.grp_start||!a.grp_ent||!a.n_groups||!a.plan_ptr||!a.plan_w||!a.part||!a.out||!a.x)return false;
    Setup z{};z.mode=1;z.h=c.h;z.gp=a.grp_ptr;z.gs=a.grp_start;z.ge=a.grp_ent;z.ng=a.n_groups;z.x=a.x;z.xs=a.x_stride;
    if(!run(c,z,a.T*a.k,a.part,c.h,s))return false;
    if(a.wait_ptr) wait_publication<<<1,32,0,s>>>(a.wait_ptr,a.wait_val);
    combine_window<<<dim3((c.h+255)/256,a.T),256,0,s>>>(a);return cudaPeekAtLastError()==cudaSuccess;
}
bool set(const exl3::SetArgs& a,const float* x,int ldx,const int* rowidx,float* y,int ldy,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||!a.hbounds||!a.wbase||!x||!rowidx||!y||ldx<c.h||ldy<c.h)return false;
    for(int e=0;e<a.n_exp;++e) for(int start=a.hbounds[e]-a.row0;start<a.hbounds[e+1]-a.row0;start+=C) {
        Setup z{};z.mode=2;z.h=c.h;z.x=x;z.xs=ldx;z.rowidx=rowidx;z.set_start=start;z.set_n=std::min(C,a.hbounds[e+1]-a.row0-start);z.set_blob=a.wbase+(size_t)e*a.stride;
        // Only descriptor zero is active for this prompt chunk.
        auto b=buffers(c); setup<<<1,256,0,s>>>(z,b.jobs,b.in,b.out);
        try { donor::fast_ffn_cuda_packed(b.jobs,1,c.h,c.ff,C,b.work,s); } catch(const std::exception& e) {std::fprintf(stderr,"native experts: %s\n",e.what());return false;}
        scatter<<<1,256,0,s>>>(z,b.jobs,b.out,y,ldy);
    }
    return cudaPeekAtLastError()==cudaSuccess;
}
}
