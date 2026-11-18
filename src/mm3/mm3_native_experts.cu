#include "mm3_native_fp4_fast.cuh"
#include "mm3_native_fp4_rows.cuh"
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/native_fp4.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <exception>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>

namespace strata::mm3::native_experts {
namespace donor = infernix::ops::minimax_ffn;
namespace {
constexpr int B=4, C=8;
// expert_fast (kernel item 4, 2026-11-14; a per-request switch, set_fast): 1 the jobs' input gather and output scatter
// spread over many blocks a job (one block of 256 threads ran each job's up to 8 x 6144 values); 2 also up to kMaxB
// jobs a launch instead of B. window() runs T x k job slots - the plan's piece count is on the device - so a drafting
// window of T rows ran T rounds of setup / FFN / scatter however few experts it had (the rounds past its pieces do
// nothing but launch: about 8.5 us a row in the light captures' expert compute, measured), and its fifth expert started
// a second working round (about 33 us more); with kMaxB = 32 every window of up to 8 rows is one round. The same
// arithmetic and the same bits in every mode.
constexpr int kMaxB=32;
constexpr int kWideChunk=2048;   // (values a block of the wide gather / scatter: 8 a thread)
std::atomic<int> g_fast{0};
std::atomic<int> g_fp4{0};   // (kernel item 7: fp4_kernel)
std::atomic<bool> g_input_reuse{false};
std::atomic<long long> g_reuse_enqueues{0},g_reuse_rows{0},g_reuse_unhandled{0};
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
// (room for kMaxB jobs - expert_fast 2 - the first B of them laid out as before)
Buffers buffers(const Context& c) {
    auto* p=c.p; Buffers b; b.jobs=(donor::FastJob*)p; p+=align256(kMaxB*sizeof(donor::FastJob));
    b.in=(uint16_t*)p; p+=align256((size_t)kMaxB*C*c.h*2);
    b.out=(uint16_t*)p; p+=align256((size_t)kMaxB*C*c.h*2);
    b.partial=(float*)p; p+=align256((size_t)8*c.h*4); b.work=p; return b;
}
// mode 0: token slots, mode 1: device-grouped window, mode 2: prompt set. (mode 1, expert_overlap: part 0 every group,
// part 1 the groups before *split, part 2 those from it)
struct Setup { int mode, base, h, T, k, set_e, set_start, set_n; const unsigned long long *ptr,*gp; const int *gs,*ge,*ng,*rowidx; const uint8_t* set_blob; const float* x; size_t xs; const int* split; int part; };
// Enumerate the existing, at-most-two eight-column pieces per group densely.
// Each piece contains at least one plan entry, so T*k bounds the job count.
// The device plan stays asynchronous; unused tail jobs are still allowed.
__device__ void window_piece(const Setup& a, int idx, int& g, int& start, int& n) {
    int lo=0, count=*a.ng;
    if (a.part) { const int at=min(max(*a.split,0),count); if (a.part==1) count=at; else lo=at; }
    for (g=lo;g<count;++g) {
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
// expert_fast: setup and scatter over (jobs, chunks of kWideChunk values): each block finds its job's piece itself, the
// chunk-0 block writes the job's descriptor; the same values as setup / scatter, from many blocks instead of one.
__device__ void job_piece(const Setup& a, int idx, int& n, int& start, const uint8_t*& blob) {
    n=0; start=0; blob=nullptr;
    if(a.mode==0) { if(idx<a.k) { blob=(const uint8_t*)a.ptr[idx]; n=blob?1:0; } }
    else if(a.mode==1) { int g; window_piece(a,idx,g,start,n); if(n) { blob=(const uint8_t*)a.gp[g]; if(!blob)n=0; } }
    else { n=a.set_n; start=a.set_start; blob=a.set_blob; }
}
// One stream boundary completes both independent producers: original-row
// encoding and the job descriptors/indices. Even an all-inactive first batch
// encodes the bank, so later batches or the staged overlap part can consume it.
__global__ void setup_rows(Setup a,donor::FastJob* jobs,uint16_t* out,uint8_t* packed,int* row_ids,int pack_blocks) {
    constexpr int H=::mm3::fp4_rows::kHidden, XR=::mm3::fp4_rows::kRowBytes;
    if((int)blockIdx.x<pack_blocks) {
        const int group=(blockIdx.x*blockDim.x+threadIdx.x)/16;
        const int row=group/(H/16),block=group%(H/16);
        const bool live=row<a.T;
        const float value=live?a.x[(size_t)row*a.xs+16*block+(threadIdx.x&15)]:0.f;
        ::mm3::fp4_rows::pack_block(value,live?packed+(size_t)row*XR:packed,block,live);
        return;
    }
    const int job=blockIdx.x-pack_blocks,idx=a.base+job;
    __shared__ int n,start;__shared__ const uint8_t* blob;
    if(threadIdx.x==0) {
        job_piece(a,idx,n,start,blob);
        donor::FastJob j{};j.record=blob?blob+sizeof(Header):nullptr;
        j.x=nullptr; // unused by the Maya-only row-indexed entry point
        j.y=out+(size_t)job*C*a.h;j.columns=n;
        if(n) {const Header* h=(const Header*)blob;for(int k=0;k<3;++k)j.multiplier[k]=1.0f/h->weight_global[k];}
        jobs[job]=j;
    }
    __syncthreads();
    if(threadIdx.x<C) {
        const int col=threadIdx.x;
        row_ids[job*C+col]=col<n?(a.mode==1?a.ge[start+col]/8:0):-1;
    }
}
__global__ void setup_wide(Setup a, donor::FastJob* jobs, uint16_t* in, uint16_t* out) {
    int job=blockIdx.x, idx=a.base+job;
    __shared__ int n,start; __shared__ const uint8_t* blob;
    if(threadIdx.x==0) {
        job_piece(a,idx,n,start,blob);
        if(blockIdx.y==0) {
            donor::FastJob j{}; j.record=blob?blob+sizeof(Header):nullptr; j.x=in+(size_t)job*C*a.h; j.y=out+(size_t)job*C*a.h; j.columns=n;
            if(n) { const Header* h=(const Header*)blob; for(int k=0;k<3;++k) j.multiplier[k]=1.0f/h->weight_global[k]; } jobs[job]=j;
        }
    }
    __syncthreads();
    const int lo=blockIdx.y*kWideChunk, hi=min(n*a.h,lo+kWideChunk);
    for(int i=lo+threadIdx.x;i<hi;i+=blockDim.x) {
        int col=i/a.h, t=0;
        if(a.mode==1) t=a.ge[start+col]/8;
        else if(a.mode==2) t=a.rowidx[start+col];
        in[((size_t)job*C+col)*a.h+i%a.h]=__bfloat16_as_ushort(__float2bfloat16(a.x[(size_t)t*a.xs+i%a.h]));
    }
}
__global__ void scatter_wide(Setup a, const donor::FastJob* jobs, const uint16_t* out, float* part, size_t ys) {
    int job=blockIdx.x, idx=a.base+job, n=jobs[job].columns; __shared__ int start;
    if (!n) return;
    const int lo=blockIdx.y*kWideChunk, hi=min(n*a.h,lo+kWideChunk);
    if (lo>=hi) return;
    if(threadIdx.x==0) { start=0; if(a.mode==1) { int g, count; window_piece(a,idx,g,start,count); } else if(a.mode==2) start=a.set_start; }
    __syncthreads();
    for(int i=lo+threadIdx.x;i<hi;i+=blockDim.x) {
        int col=i/a.h, dst=idx;
        if(a.mode==1) dst=a.ge[start+col];
        else if(a.mode==2) dst=start+col;
        part[(size_t)dst*ys+i%a.h]=__bfloat162float(__ushort_as_bfloat16(out[((size_t)job*C+col)*a.h+i%a.h]));
    }
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
void launch_ffn(const Context& c,donor::FastJob* jobs,int count,void* work,cudaStream_t s,cudaEvent_t down_wait=nullptr,cudaEvent_t gu_begin=nullptr,cudaEvent_t gu_end=nullptr) {
    // (kernel item 7: fp4_kernel P > 0 - the donor's kernel with its loads batched, the same bits and workspace)
    if(c.arithmetic==Arithmetic::NativeFp4) {
        const int loads=g_fp4.load(std::memory_order_relaxed);
        if(loads>0)::mm3::fp4_fast::ffn(jobs,count,C,work,loads,s,down_wait,gu_begin,gu_end); else donor::native_fp4_ffn(jobs,count,C,work,s);
    }
    else donor::fast_ffn_cuda_packed(jobs,count,c.h,c.ff,C,work,s);
}
bool run(const Context& c,Setup a,int batches,float* part,size_t ys,cudaStream_t s,int force_fast=-1,bool encode_rows=true,cudaEvent_t down_wait=nullptr,cudaEvent_t gu_begin=nullptr,cudaEvent_t gu_end=nullptr) {
    auto b=buffers(c);
    const int fast=force_fast>=0?force_fast:g_fast.load(std::memory_order_relaxed);
    const int per=fast>=2?kMaxB:B;
    const unsigned chunks=(unsigned)((C*a.h+kWideChunk-1)/kWideChunk);
    const bool reuse=g_input_reuse.load(std::memory_order_relaxed);
    if(reuse&&(c.arithmetic!=Arithmetic::NativeFp4||a.mode==2||a.T<1||a.T>16))return false;
    // Reinterpret the now-unused BF16 gather arena. No new allocation or bank.
    constexpr size_t packed_bytes=16*::mm3::fp4_rows::kRowBytes;
    static_assert(packed_bytes+kMaxB*C*sizeof(int)<=size_t(kMaxB)*C*6144*2);
    auto* packed=reinterpret_cast<uint8_t*>(b.in);
    auto* row_ids=reinterpret_cast<int*>(packed+packed_bytes);
    try { for(int at=0;at<batches;at+=per) { a.base=at; int n=std::min(per,batches-at);
        if(reuse) {
            const int pack_blocks=encode_rows&&at==0?(a.T*c.h+255)/256:0;
            setup_rows<<<pack_blocks+n,256,0,s>>>(a,b.jobs,b.out,packed,row_ids,pack_blocks);
            int loads=g_fp4.load(std::memory_order_relaxed);if(!loads)loads=1;
            ::mm3::fp4_fast::ffn_rows(b.jobs,n,C,b.work,loads,packed,row_ids,a.T,s);
            g_reuse_enqueues.fetch_add(1,std::memory_order_relaxed);
            if(pack_blocks)g_reuse_rows.fetch_add(a.T,std::memory_order_relaxed);
        } else {
            if(fast>=1) setup_wide<<<dim3(n,chunks),256,0,s>>>(a,b.jobs,b.in,b.out);
            else setup<<<n,256,0,s>>>(a,b.jobs,b.in,b.out);
            launch_ffn(c,b.jobs,n,b.work,s,down_wait,at?nullptr:gu_begin,at?nullptr:gu_end);
        }
        if(fast>=1) scatter_wide<<<dim3(n,chunks),256,0,s>>>(a,b.jobs,b.out,part,ys);
        else scatter<<<n,256,0,s>>>(a,b.jobs,b.out,part,ys);
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
        return align256(kMaxB*sizeof(donor::FastJob))+2*align256((size_t)kMaxB*C*h*2)+align256((size_t)8*h*4)+kMaxB*work;
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
void set_fast(int mode) { g_fast.store(mode<0?0:mode>2?2:mode, std::memory_order_relaxed); }
void set_fp4_kernel(int loads) { g_fp4.store(::mm3::fp4_fast::valid_loads(loads) ? loads : 0, std::memory_order_relaxed); }
int fp4_kernel() { return g_fp4.load(std::memory_order_relaxed); }
void set_input_reuse(bool enabled) {g_input_reuse.store(enabled,std::memory_order_relaxed);}
bool input_reuse() {return g_input_reuse.load(std::memory_order_relaxed);}
void input_reuse_counts(long long& enqueues,long long& rows,long long& unhandled) {
    enqueues=g_reuse_enqueues.load(std::memory_order_relaxed);
    rows=g_reuse_rows.load(std::memory_order_relaxed);
    unhandled=g_reuse_unhandled.load(std::memory_order_relaxed);
}
int fast() { return g_fast.load(std::memory_order_relaxed); }
void release(cudaStream_t s) { int d;if(cudaGetDevice(&d)!=cudaSuccess)return;std::lock_guard<std::mutex> l(mu);ctx.erase({d,s}); }
bool gate_up(const exl3::MoeArgs& a,const float* x,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||!x||!a.plan_ptr||a.k<1||a.k>8)return false;
    Setup z{};z.mode=0;z.h=c.h;z.T=1;z.k=a.k;z.ptr=a.plan_ptr;z.x=x;z.xs=c.h;
    return run(c,z,a.k,buffers(c).partial,c.h,s);
}
bool down(const exl3::MoeArgs& a,const float* sh,float* out,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||!a.plan_w||!a.plan_ptr||!out)return false;
    combine_token<<<(c.h+255)/256,256,0,s>>>(a,buffers(c).partial,sh,out);return cudaPeekAtLastError()==cudaSuccess;
}
bool window_scatter(const exl3::WinArgs& a,int pieces,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||a.T<1||a.T>16||a.k<1||a.k>8||pieces<0||pieces>a.T*a.k||!a.grp_ptr||!a.grp_start||!a.grp_ent||!a.n_groups||!a.part||!a.x)return false;
    Setup z{};z.mode=1;z.h=c.h;z.T=a.T;z.gp=a.grp_ptr;z.gs=a.grp_start;z.ge=a.grp_ent;z.ng=a.n_groups;z.x=a.x;z.xs=a.x_stride;
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
// expert_overlap's wait between its rounds: an external event node in a stream capture (the expert graph waits, at each
// replay, for the event's record before that replay's launch), a plain wait outside one.
static bool wait_split(cudaEvent_t e,cudaStream_t s) {
    cudaStreamCaptureStatus cs=cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(s,&cs)!=cudaSuccess)return false;
    if(cs==cudaStreamCaptureStatusActive)return cudaStreamWaitEvent(s,e,cudaEventWaitExternal)==cudaSuccess;
    return cs==cudaStreamCaptureStatusNone && cudaStreamWaitEvent(s,e,0)==cudaSuccess;
}
bool window(const exl3::WinArgs& a,cudaStream_t s,int active_pieces) {
    if (active_pieces < -1 || active_pieces > a.T*a.k || (active_pieces >= 0 && a.split_wait)) return false;
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||a.T<1||a.T>16||a.k<1||a.k>8||!a.grp_ptr||!a.grp_start||!a.grp_ent||!a.n_groups||!a.plan_ptr||!a.plan_w||!a.part||!a.out||!a.x)return false;
    Setup z{};z.mode=1;z.h=c.h;z.T=a.T;z.gp=a.grp_ptr;z.gs=a.grp_start;z.ge=a.grp_ent;z.ng=a.n_groups;z.x=a.x;z.xs=a.x_stride;
    if(a.down_wait && (!a.split_wait || c.arithmetic!=Arithmetic::NativeFp4 || !g_fp4.load() || g_input_reuse.load()))return false;
    if(a.split_at && a.split_wait) {
        // expert_overlap: the groups before the split (frames already in VRAM) while the rest's uploads run, then those
        // after the uploads' event. Each round has T x k job slots and runs as expert_fast 2 does (32 a launch, wide
        // gather and scatter): two rounds of T launches each would cost more than the overlap saves. Every job is the
        // same job as in one round and writes its own entries of part, so the output is the same bit for bit.
        z.split=a.split_at;
        z.part=1; if(!run(c,z,a.T*a.k,a.part,c.h,s,2))return false;
        if(!record_diagnostic(a.diagnostic_ready_done,s) || !wait_split(a.split_wait,s))return false;
        z.part=2; if(!run(c,z,a.T*a.k,a.part,c.h,s,2,false,a.down_wait,a.gu_begin,a.gu_end))return false;
    } else {
        if(a.down_wait && !wait_split(a.split_wait,s))return false;
        if(!run(c,z,active_pieces < 0 ? a.T*a.k : active_pieces,a.part,c.h,s,-1,true,a.down_wait,a.gu_begin,a.gu_end))return false;
    }
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
template <bool HasTail> __global__ void prompt_scatter(const PromptJob* src, int base, int h, const uint16_t* out, uint16_t* part16,PromptTail tail) {
    const int j=blockIdx.x; const PromptJob pj=src[base+j]; const int n=pj.blob?pj.n:0;
    for(int i=threadIdx.x;i<n*h;i+=blockDim.x) {
        const int col=i/h;
        int at=pj.entry[col];auto* dest=part16;
        if constexpr(HasTail)if(at>=tail.first){dest=tail.data;at-=tail.first;}
        dest[size_t(at)*h+i%h]=out[((size_t)j*C+col)*h+i%h];
    }
}
__global__ void prompt_bf16(const float* x, size_t n, uint16_t* out) {
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        out[i]=__bfloat16_as_ushort(__float2bfloat16(x[i]));
}
// combine_window's arithmetic (same expressions, same order), then k::add's h += out; ToY (--ablate): v written to y
// instead, h untouched (the caller projects y, then adds it). ToY false is the kernel as before the sync, unchanged.
template <bool ToY,bool HasTail=false>
__global__ void prompt_combine_rows(int k, int H, const float* sh, const uint16_t* part16, const float* w,
                                    const uint8_t* served, const float* helper, const int* flag, float* h,PromptTail tail) {
    const int d=blockIdx.x*blockDim.x+threadIdx.x; const size_t r=blockIdx.y; if(d>=H)return;
    float v=sh[r*H+d];
    for(int p=0;p<k;++p){const size_t e=r*k+p;if(served[e]){
        const auto* src=part16;size_t at=e;
        if constexpr(HasTail)if(at>=size_t(tail.first)){src=tail.data;at-=tail.first;}
        v+=w[e]*__bfloat162float(__ushort_as_bfloat16(src[at*H+d]));
    }}
    if(helper && flag && flag[r]) v+=helper[r*H+d];
    if constexpr (ToY) h[r*H+d]=v;
    else h[r*H+d]+=v;
}
}
size_t prompt_scratch_bytes(cudaStream_t s,int max_jobs) {
    Context c; if(!lookup(s,c)||max_jobs<1)return 0;
    try { return prompt_layout(c,nullptr,max_jobs).bytes+256; } catch(...) { return 0; }
}
bool prompt_run(const PromptJob* jobs,int first,int count,const uint16_t* x16,uint16_t* part16,void* scratch,size_t bytes,int max_jobs,cudaStream_t s,PromptTail tail) {
    Context c; if(!lookup(s,c)||!jobs||!x16||!part16||!scratch||count<0||first<0||max_jobs<1)return false;
    if(!count)return true;
    if(g_input_reuse.load(std::memory_order_relaxed))g_reuse_unhandled.fetch_add(1,std::memory_order_relaxed);
    void* base=(void*)(((uintptr_t)scratch+255)&~uintptr_t(255));
    try {
        const PromptScratch b=prompt_layout(c,base,max_jobs);
        if(b.bytes+256>bytes)return false;
        for(int at=0;at<count;at+=max_jobs) {
            const int n=std::min(max_jobs,count-at);
            prompt_setup<<<n,256,0,s>>>(jobs,first+at,4,c.h,x16,b.jobs,b.in,b.out);
            launch_ffn(c,b.jobs,n,b.work,s);
            if(tail.data)prompt_scatter<true><<<n,256,0,s>>>(jobs,first+at,c.h,b.out,part16,tail);
            else prompt_scatter<false><<<n,256,0,s>>>(jobs,first+at,c.h,b.out,part16,tail);
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
                    const float* helper,const int* flag,float* h,cudaStream_t s,PromptTail tail) {
    if(rows<0||k<1||k>8||h_dim<1||!sh||!part16||!w||!served||!h||rows>65535)return false;
    if(!rows)return true;
    if(tail.data)prompt_combine_rows<false,true><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,h,tail);
    else prompt_combine_rows<false><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,h,tail);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool prompt_combine_to(int rows,int k,int h_dim,const float* sh,const uint16_t* part16,const float* w,const uint8_t* served,
                       const float* helper,const int* flag,float* y,cudaStream_t s,PromptTail tail) {
    if(rows<0||k<1||k>8||h_dim<1||!sh||!part16||!w||!served||!y||rows>65535)return false;
    if(!rows)return true;
    if(tail.data)prompt_combine_rows<true,true><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,y,tail);
    else prompt_combine_rows<true><<<dim3((h_dim+255)/256,rows),256,0,s>>>(k,h_dim,sh,part16,w,served,helper,flag,y,tail);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool set(const exl3::SetArgs& a,const float* x,int ldx,const int* rowidx,float* y,int ldy,cudaStream_t s) {
    Context c;if(!lookup(s,c)||c.h!=a.n_embd||c.ff!=a.n_ff||a.limit!=7.0f||!a.hbounds||!a.wbase||!x||!rowidx||!y||ldx<c.h||ldy<c.h)return false;
    if(g_input_reuse.load(std::memory_order_relaxed))g_reuse_unhandled.fetch_add(1,std::memory_order_relaxed);
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
