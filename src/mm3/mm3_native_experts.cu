#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/native_fp4.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <cstring>
#include <map>
#include <mutex>

namespace strata::mm3::native_experts {
namespace donor = infernix::ops::minimax_ffn;
namespace {
constexpr int B=4, C=8;
size_t align256(size_t x) { return (x+255)&~size_t(255); }
struct Context { int h,ff; uint8_t* p; size_t bytes; Arithmetic arithmetic; };
using Key=std::pair<int,cudaStream_t>;
std::mutex mu; std::map<Key,Context> ctx;
bool lookup(cudaStream_t s, Context& c) { int d; if(cudaGetDevice(&d)!=cudaSuccess)return false; std::lock_guard<std::mutex> l(mu); auto i=ctx.find({d,s}); if(i==ctx.end())return false; c=i->second; return true; }
bool allowed(int d, int h, int ff, Arithmetic arithmetic) {
    if(arithmetic==Arithmetic::Dp4a)return true;
    if(arithmetic!=Arithmetic::NativeFp4||h!=6144||ff!=3072)return false;
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop,d)==cudaSuccess && prop.major==12 && prop.minor==0 &&
           std::strstr(prop.name,"RTX 5090")!=nullptr;
}
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
        const int pieces=(len+C-1)/C;
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
__global__ void wait_publication(const unsigned* p, unsigned value, const unsigned* value_at) {
    if (threadIdx.x) return;
    if (value_at) value = *(volatile const unsigned*) value_at;   // (expert_graph: the pass's value, in the device plan)
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
void launch_ffn(const Context& c,donor::FastJob* jobs,int count,void* work,cudaStream_t s) {
    if(c.arithmetic==Arithmetic::NativeFp4)donor::native_fp4_ffn(jobs,count,C,work,s);
    else donor::fast_ffn_cuda_packed(jobs,count,c.h,c.ff,C,work,s);
}
bool run(const Context& c,Setup a,int batches,float* part,size_t ys,cudaStream_t s) {
    auto b=buffers(c);
    try { for(int at=0;at<batches;at+=B) { a.base=at; int n=std::min(B,batches-at); setup<<<n,256,0,s>>>(a,b.jobs,b.in,b.out);
        launch_ffn(c,b.jobs,n,b.work,s);
        scatter<<<n,256,0,s>>>(a,b.jobs,b.out,part,ys);
    }} catch(const std::exception& e) { std::fprintf(stderr,"native experts: %s\n",e.what()); return false; }
    return cudaPeekAtLastError()==cudaSuccess;
}
}
size_t scratch_bytes(int h,int ff,Arithmetic arithmetic) {
    if(h<=0||ff<=0)return 0;
    try {
        size_t work=0;
        if(arithmetic==Arithmetic::Dp4a)work=donor::fast_workspace_bytes(h,ff,C);
        else if(arithmetic==Arithmetic::NativeFp4 && h==6144 && ff==3072)work=donor::native_fp4_workspace_bytes(C);
        else return 0;
        return align256(B*sizeof(donor::FastJob))+2*align256((size_t)B*C*h*2)+align256((size_t)8*h*4)+B*work;
    } catch(...) { return 0; }
}
bool prepare(cudaStream_t s,int h,int ff,void* p,size_t bytes,Arithmetic arithmetic) {
    size_t need=scratch_bytes(h,ff,arithmetic); if(!need||!p||((uintptr_t)p%256)||bytes<need)return false;
    int d;if(cudaGetDevice(&d)!=cudaSuccess)return false;std::lock_guard<std::mutex> l(mu);
    if(!allowed(d,h,ff,arithmetic))return false;
    return ctx.emplace(Key{d,s},Context{h,ff,(uint8_t*)p,bytes,arithmetic}).second;
}
bool set_arithmetic(cudaStream_t s,Arithmetic arithmetic) {
    int d;if(cudaGetDevice(&d)!=cudaSuccess)return false;std::lock_guard<std::mutex> l(mu);
    auto i=ctx.find({d,s});if(i==ctx.end())return false;
    const size_t need=scratch_bytes(i->second.h,i->second.ff,arithmetic);
    if(!need||i->second.bytes<need||!allowed(d,i->second.h,i->second.ff,arithmetic))return false;
    i->second.arithmetic=arithmetic;return true;
}
bool get_arithmetic(cudaStream_t s,Arithmetic& arithmetic) {
    Context c;if(!lookup(s,c))return false;arithmetic=c.arithmetic;return true;
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
bool window_scatter(const exl3::WinArgs& a,int pieces,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||a.T<1||a.T>16||a.k<1||a.k>8||pieces<0||pieces>a.T*a.k||!a.grp_ptr||!a.grp_start||!a.grp_ent||!a.n_groups||!a.part||!a.x)return false;
    Setup z{};z.mode=1;z.h=c.h;z.gp=a.grp_ptr;z.gs=a.grp_start;z.ge=a.grp_ent;z.ng=a.n_groups;z.x=a.x;z.xs=a.x_stride;
    return !pieces || run(c,z,pieces,a.part,c.h,s);
}
bool window_combine(const exl3::WinArgs& a,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||!a.plan_ptr||!a.plan_w||!a.part||!a.out||a.T<1||a.T>16||a.k<1||a.k>8)return false;
    combine_window<<<dim3((c.h+255)/256,a.T),256,0,s>>>(a);return cudaPeekAtLastError()==cudaSuccess;
}
// A diagnostic event's record (the light capture's 5090 stamps, the replay trace's): an external record while the stream
// is capturing - the expert graph's event node, recorded at each replay - and a plain record outside one, as before
// fb225ef. cudaEventRecordExternal is invalid outside a capture (cuda.h, CU_EVENT_RECORD_EXTERNAL), so the flag follows
// the stream's capture status; an invalidated capture fails the pass, as any refused record does.
static bool record_diagnostic(cudaEvent_t e,cudaStream_t s) {
    if(!e)return true;
    cudaStreamCaptureStatus cs=cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(s,&cs)!=cudaSuccess)return false;
    if(cs==cudaStreamCaptureStatusActive)return cudaEventRecordWithFlags(e,s,cudaEventRecordExternal)==cudaSuccess;
    return cs==cudaStreamCaptureStatusNone && cudaEventRecord(e,s)==cudaSuccess;
}
bool window(const exl3::WinArgs& a,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||a.T<1||a.T>16||a.k<1||a.k>8||!a.grp_ptr||!a.grp_start||!a.grp_ent||!a.n_groups||!a.plan_ptr||!a.plan_w||!a.part||!a.out||!a.x)return false;
    Setup z{};z.mode=1;z.h=c.h;z.gp=a.grp_ptr;z.gs=a.grp_start;z.ge=a.grp_ent;z.ng=a.n_groups;z.x=a.x;z.xs=a.x_stride;
    if(!run(c,z,a.T*a.k,a.part,c.h,s))return false;
    // (the diagnostic events: in a stream capture - the expert graph - the graph's event nodes, recorded at each replay;
    //  outside one - the prompt's chunks, the launches after a failed capture, the replay trace - plain records)
    if(!record_diagnostic(a.diagnostic_compute_done,s))return false;
    if(a.wait_ptr) wait_publication<<<1,32,0,s>>>(a.wait_ptr,a.wait_val,a.wait_val_ptr);
    if(!record_diagnostic(a.diagnostic_wait_done,s))return false;
    combine_window<<<dim3((c.h+255)/256,a.T),256,0,s>>>(a);return cudaPeekAtLastError()==cudaSuccess;
}
// ---- prompt_layerwise: jobs over a whole prompt's rows (see the header). The setup and scatter differ from the window
// path's only in where a column's row is read and its result written; the FFN launches are the same calls.
namespace {
size_t prompt_work_per_job(const Context& c) {
    return c.arithmetic==Arithmetic::NativeFp4 ? donor::native_fp4_workspace_bytes(C) : donor::fast_workspace_bytes(c.h,c.ff,C);
}
struct PromptScratch { donor::FastJob* jobs; uint16_t* in; uint16_t* out; uint8_t* work; size_t bytes; };
PromptScratch prompt_layout(const Context& c, void* base, int max_jobs) {
    const uintptr_t b0=(uintptr_t)base; size_t at=0; PromptScratch b{};
    b.jobs=(donor::FastJob*)(b0+at); at+=align256((size_t)max_jobs*sizeof(donor::FastJob));
    b.in=(uint16_t*)(b0+at); at+=align256((size_t)max_jobs*C*c.h*2);
    b.out=(uint16_t*)(b0+at); at+=align256((size_t)max_jobs*C*c.h*2);
    b.work=(uint8_t*)(b0+at); at+=(size_t)max_jobs*prompt_work_per_job(c);
    b.bytes=at; return b;
}
__global__ void prompt_setup(const PromptJob* src, int base, int k, int h, const uint16_t* x16, donor::FastJob* jobs, uint16_t* in,
                             uint16_t* out) {
    const int j=blockIdx.x; const PromptJob pj=src[base+j];
    if(threadIdx.x==0) {
        donor::FastJob f{}; const int n=pj.blob?pj.n:0;
        f.record=pj.blob?pj.blob+sizeof(Header):nullptr; f.x=in+(size_t)j*C*h; f.y=out+(size_t)j*C*h; f.columns=n;
        if(n) { const Header* hd=(const Header*)pj.blob; for(int q=0;q<3;++q) f.multiplier[q]=1.0f/hd->weight_global[q]; }
        jobs[j]=f;
    }
    const int n=pj.blob?pj.n:0;
    for(int i=threadIdx.x;i<n*h;i+=blockDim.x) {
        const int col=i/h, row=pj.entry[col]/k;
        in[((size_t)j*C+col)*h+i%h]=x16[(size_t)row*h+i%h];
    }
}
__global__ void prompt_scatter(const PromptJob* src, int base, int h, const uint16_t* out, uint16_t* part16) {
    const int j=blockIdx.x; const PromptJob pj=src[base+j]; const int n=pj.blob?pj.n:0;
    for(int i=threadIdx.x;i<n*h;i+=blockDim.x) {
        const int col=i/h;
        part16[(size_t)pj.entry[col]*h+i%h]=out[((size_t)j*C+col)*h+i%h];
    }
}
__global__ void prompt_bf16(const float* x, size_t n, uint16_t* out) {
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        out[i]=__bfloat16_as_ushort(__float2bfloat16(x[i]));
}
// combine_window's arithmetic (same expressions, same order), then k::add's h += out; ToY (--ablate): v written to y
// instead, h untouched (the caller projects y, then adds it). ToY false is the kernel as before the sync, unchanged.
template <bool ToY>
__global__ void prompt_combine_rows(int k, int H, const float* sh, const uint16_t* part16, const float* w,
                                    const uint8_t* served, const float* helper, const int* flag, float* h) {
    const int d=blockIdx.x*blockDim.x+threadIdx.x; const size_t r=blockIdx.y; if(d>=H)return;
    float v=sh[r*H+d];
    for(int p=0;p<k;++p) {const size_t e=r*k+p;if(served[e])v+=w[e]*__bfloat162float(__ushort_as_bfloat16(part16[e*H+d]));}
    if(helper && flag && flag[r]) v+=helper[r*H+d];
    if constexpr (ToY) h[r*H+d]=v;
    else h[r*H+d]+=v;
}
}
size_t prompt_scratch_bytes(cudaStream_t s,int max_jobs) {
    Context c; if(!lookup(s,c)||max_jobs<1)return 0;
    try { return prompt_layout(c,nullptr,max_jobs).bytes+256; } catch(...) { return 0; }
}
bool prompt_run(const PromptJob* jobs,int first,int count,const uint16_t* x16,uint16_t* part16,void* scratch,size_t bytes,int max_jobs,cudaStream_t s) {
    Context c; if(!lookup(s,c)||!jobs||!x16||!part16||!scratch||count<0||first<0||max_jobs<1)return false;
    if(!count)return true;
    void* base=(void*)(((uintptr_t)scratch+255)&~uintptr_t(255));
    try {
        const PromptScratch b=prompt_layout(c,base,max_jobs);
        if(b.bytes+256>bytes)return false;
        for(int at=0;at<count;at+=max_jobs) {
            const int n=std::min(max_jobs,count-at);
            prompt_setup<<<n,256,0,s>>>(jobs,first+at,4,c.h,x16,b.jobs,b.in,b.out);
            launch_ffn(c,b.jobs,n,b.work,s);
            prompt_scatter<<<n,256,0,s>>>(jobs,first+at,c.h,b.out,part16);
        }
    } catch(const std::exception& e) { std::fprintf(stderr,"native prompt experts: %s\n",e.what()); return false; }
    return cudaPeekAtLastError()==cudaSuccess;
}
bool prompt_rows_bf16(const float* x,int rows,int h,uint16_t* out,cudaStream_t s) {
    if(!x||!out||rows<0||h<1)return false;
    const size_t n=(size_t)rows*h; if(!n)return true;
    prompt_bf16<<<(unsigned)std::min<size_t>(4096,(n+255)/256),256,0,s>>>(x,n,out);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool prompt_combine(int rows,int k,int h_dim,const float* sh,const uint16_t* part16,const float* w,const uint8_t* served,
                    const float* helper,const int* flag,float* h,cudaStream_t s) {
    if(rows<0||k<1||k>8||h_dim<1||!sh||!part16||!w||!served||!h||rows>65535)return false;
    if(!rows)return true;
    prompt_combine_rows<false><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,h);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool prompt_combine_to(int rows,int k,int h_dim,const float* sh,const uint16_t* part16,const float* w,const uint8_t* served,
                       const float* helper,const int* flag,float* y,cudaStream_t s) {
    if(rows<0||k<1||k>8||h_dim<1||!sh||!part16||!w||!served||!y||rows>65535)return false;
    if(!rows)return true;
    prompt_combine_rows<true><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,y);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool set(const exl3::SetArgs& a,const float* x,int ldx,const int* rowidx,float* y,int ldy,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||!a.hbounds||!a.wbase||!x||!rowidx||!y||ldx<c.h||ldy<c.h)return false;
    for(int e=0;e<a.n_exp;++e) for(int start=a.hbounds[e]-a.row0;start<a.hbounds[e+1]-a.row0;start+=C) {
        Setup z{};z.mode=2;z.h=c.h;z.x=x;z.xs=ldx;z.rowidx=rowidx;z.set_start=start;z.set_n=std::min(C,a.hbounds[e+1]-a.row0-start);z.set_blob=a.wbase+(size_t)e*a.stride;
        // Only descriptor zero is active for this prompt chunk.
        auto b=buffers(c); setup<<<1,256,0,s>>>(z,b.jobs,b.in,b.out);
        try { launch_ffn(c,b.jobs,1,b.work,s); } catch(const std::exception& e) {std::fprintf(stderr,"native experts: %s\n",e.what());return false;}
        scatter<<<1,256,0,s>>>(z,b.jobs,b.out,y,ldy);
    }
    return cudaPeekAtLastError()==cudaSuccess;
}
}
