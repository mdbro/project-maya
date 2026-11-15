// Standalone primary row reuse gate + full-window (setup through combine) timing.
// Parent-controlled GPU run only. No model. Synthetic exact donor records.
// Two independent stream contexts; rows up to16, >32 active/inactive-leading job
// slots, empty early overlap part, changed-input/row-ID replay, source strides.
#include "mm3_native_experts.cuh"
#include "mm3_native_fp4_fast.cuh"
#include "mm3_native_fp4_rows.cuh"
#include "ops/minimax_nvfp4_ffn/record.h"
#include "ops/common/canonical_math.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace nx=strata::mm3::native_experts;
namespace x3=strata::mm3::exl3;
namespace ff=infernix::ops::minimax_ffn;
namespace canon=infernix::ops::canon;
namespace rows=mm3::fp4_rows;
constexpr int H=6144,F=3072,TMAX=16,LD=H+16,PICKS=4,POOL=16;
static void ck(cudaError_t e,const char* what){if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));}
static void require(bool b,const char* what){if(!b)throw std::runtime_error(what);}
template<class T>T* dev(size_t n){T* p=nullptr;ck(cudaMalloc((void**)&p,n*sizeof(T)),"allocation");return p;}
template<class T>void upload(T* p,const std::vector<T>& v,cudaStream_t s){ck(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,s),"upload");ck(cudaStreamSynchronize(s),"upload done");}
template<class T>std::vector<T> read(const T* p,size_t n,cudaStream_t s){std::vector<T> v(n);ck(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,s),"read");ck(cudaStreamSynchronize(s),"read done");return v;}
static std::vector<uint8_t> pack_host(const std::vector<float>& x,int n){
    std::vector<uint8_t> out(size_t(n)*rows::kRowBytes);
    for(int r=0;r<n;++r)for(int b=0;b<H/16;++b){
        uint16_t bf[16];for(int k=0;k<16;++k)bf[k]=canon::f32_to_bf16_rn(x[size_t(r)*LD+16*b+k]);
        const auto q=canon::quantize_a4_codes(bf,1.f);const int p=rows::position(b);
        for(int k=0;k<4;++k){out[size_t(r)*rows::kRowBytes+8*p+k]=q.code[k]|(q.code[4+k]<<4);out[size_t(r)*rows::kRowBytes+8*p+4+k]=q.code[8+k]|(q.code[12+k]<<4);}
        out[size_t(r)*rows::kRowBytes+H/2+p]=q.scale_word;
    }return out;
}
// Independent serial GPU oracle for the original hardware BF16 conversion and
// canonical prepare rule, including NaN payload/sign conversion behavior.
__global__ void original_encode(const float* x,int ldx,uint8_t* out){
    const int task=blockIdx.x*blockDim.x+threadIdx.x;if(task>=TMAX*(H/16))return;
    const int r=task/(H/16),b=task%(H/16),p=rows::position(b);uint16_t bf[16];
    for(int k=0;k<16;++k)bf[k]=__bfloat16_as_ushort(__float2bfloat16(x[size_t(r)*ldx+16*b+k]));
    const auto q=canon::quantize_a4_codes(bf,1.f);
    for(int k=0;k<4;++k){out[size_t(r)*rows::kRowBytes+8*p+k]=q.code[k]|(q.code[4+k]<<4);out[size_t(r)*rows::kRowBytes+8*p+4+k]=q.code[8+k]|(q.code[12+k]<<4);}
    out[size_t(r)*rows::kRowBytes+H/2+p]=q.scale_word;
}
static void codec_gate(cudaStream_t s){
    std::vector<float> x(size_t(TMAX)*LD);
    std::mt19937 rng(191125);std::normal_distribution<float> nd(0,1);
    for(auto& f:x)f=nd(rng);
    // Finite F32 values immediately around BF16 rounding boundaries, then
    // subnormal/zero/saturation cases. These differ from quantizing F32 directly.
    const uint32_t bits[]={0x3f808000u,0x3f817fffu,0x3f818001u,0xbf808000u,0x00008000u,0x80000000u,0x7f7f0000u,0x3a800000u};
    for(size_t i=0;i<1024;++i)std::memcpy(&x[i],&bits[i%8],4);
    float* dx=dev<float>(x.size());uint8_t* packed=dev<uint8_t>(size_t(TMAX)*rows::kRowBytes+256);
    upload(dx,x,s);
    cudaGraphExec_t graph=nullptr;cudaGraph_t g=nullptr;
    ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"codec capture");
    mm3::fp4_fast::pack_rows_f32(dx,TMAX,LD,packed,s);
    ck(cudaStreamEndCapture(s,&g),"codec capture end");ck(cudaGraphInstantiate(&graph,g,0),"codec instantiate");cudaGraphDestroy(g);
    for(int pass=0;pass<2;++pass){
        if(pass)for(auto& f:x)f=-f;
        upload(dx,x,s);ck(cudaMemsetAsync(packed,0xa5,size_t(TMAX)*rows::kRowBytes+256,s),"codec guard");
        ck(cudaGraphLaunch(graph,s),"codec replay");const auto actual=read(packed,size_t(TMAX)*rows::kRowBytes+256,s);const auto expected=pack_host(x,TMAX);
        require(std::equal(expected.begin(),expected.end(),actual.begin()),"shared encoder differs from independent finite host canonical codes/scales");
        require(std::all_of(actual.begin()+expected.size(),actual.end(),[](uint8_t v){return v==0xa5;}),"shared encoder bounds guard");
    }
    const uint32_t exceptional[]={0u,0x80000000u,0x00000001u,0x7f7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0xffc12345u};
    for(size_t i=0;i<x.size();++i)std::memcpy(&x[i],&exceptional[i%8],4);
    upload(dx,x,s);auto* reference=dev<uint8_t>(size_t(TMAX)*rows::kRowBytes);
    original_encode<<<(TMAX*(H/16)+127)/128,128,0,s>>>(dx,LD,reference);
    mm3::fp4_fast::pack_rows_f32(dx,TMAX,LD,packed,s);
    require(read(reference,size_t(TMAX)*rows::kRowBytes,s)==read(packed,size_t(TMAX)*rows::kRowBytes,s),"exceptional codes/scales differ from original GPU BF16/serial canonical rule");
    cudaFree(reference);
    cudaGraphExecDestroy(graph);cudaFree(packed);cudaFree(dx);
    std::puts("codec PASS: exact independent finite host and exceptional original-GPU F32->BF16->NVFP4 codes/scales; boundaries, stride, guard, changed-input replay");
}
struct Lane{
    cudaStream_t s=nullptr;cudaEvent_t ready=nullptr;uint8_t* scratch=nullptr;float *x=nullptr,*part=nullptr,*out=nullptr;
    unsigned long long *ptr=nullptr,*gp=nullptr;float* weight=nullptr;int *gs=nullptr,*ge=nullptr,*ng=nullptr,*split=nullptr;
    Lane(){
        ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");ck(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming),"ready event");
        const auto bytes=nx::scratch_bytes(H,F,nx::Arithmetic::NativeFp4);scratch=dev<uint8_t>(bytes);
        require(nx::prepare(s,H,F,scratch,bytes,nx::Arithmetic::NativeFp4),"native context prepare");
        x=dev<float>(size_t(TMAX)*LD);part=dev<float>(size_t(TMAX)*8*H);out=dev<float>(size_t(TMAX)*H);
        ptr=dev<unsigned long long>(TMAX*8);weight=dev<float>(TMAX*8);gp=dev<unsigned long long>(TMAX*PICKS);gs=dev<int>(TMAX*PICKS+1);ge=dev<int>(TMAX*PICKS);ng=dev<int>(1);split=dev<int>(1);
    }
    ~Lane(){cudaStreamSynchronize(s);nx::release(s);for(void* p:{(void*)scratch,(void*)x,(void*)part,(void*)out,(void*)ptr,(void*)weight,(void*)gp,(void*)gs,(void*)ge,(void*)ng,(void*)split})cudaFree(p);cudaEventDestroy(ready);cudaStreamDestroy(s);}
    x3::WinArgs args(int T,int split_at){x3::WinArgs w{};w.grp_ptr=gp;w.grp_start=gs;w.grp_ent=ge;w.n_groups=ng;w.plan_ptr=ptr;w.plan_w=weight;w.x=x;w.x_stride=LD;w.part=part;w.out=out;w.out_stride=H;w.T=T;w.k=PICKS;w.n_embd=H;w.n_ff=F;w.limit=7.f;w.lay.fmt=nx::kFmt;if(split_at>=0){w.split_at=split;w.split_wait=ready;}return w;}
};
struct Plan{std::vector<unsigned long long> ptr,gp;std::vector<float> weight;std::vector<int> gs,ge;};
static Plan plan(int T,int kind,bool shuffle,uint8_t* blobs,size_t blob){
    Plan p;p.ptr.assign(TMAX*8,0);p.weight.assign(TMAX*8,0);p.gs={0};std::vector<std::vector<int>> users(POOL);
    for(int t=0;t<T;++t)for(int k=0;k<PICKS;++k){
        int e=(3*t+k)%POOL;bool active=kind!=2||t>=T/2;
        p.ptr[t*8+k]=active?(unsigned long long)(uintptr_t)(blobs+size_t(e)*blob):0;p.weight[t*8+k]=active?0.25f:0.f;
        if(kind){p.gp.push_back(p.ptr[t*8+k]);p.ge.push_back(t*8+k);p.gs.push_back(int(p.ge.size()));}
        else users[e].push_back(t*8+k);
    }
    if(!kind)for(int e=0;e<POOL;++e)if(!users[e].empty()){
        if(shuffle)std::reverse(users[e].begin(),users[e].end());
        p.gp.push_back((unsigned long long)(uintptr_t)(blobs+size_t(e)*blob));p.ge.insert(p.ge.end(),users[e].begin(),users[e].end());p.gs.push_back(int(p.ge.size()));
    }
    if(kind&&shuffle){ // Change row IDs and descriptors while preserving the same per-entry plan.
        std::reverse(p.gp.begin(),p.gp.end());std::reverse(p.ge.begin(),p.ge.end());
    }
    return p;
}
static void load_plan(Lane& a,const Plan& p,int split_at){upload(a.ptr,p.ptr,a.s);upload(a.weight,p.weight,a.s);upload(a.gp,p.gp,a.s);upload(a.gs,p.gs,a.s);upload(a.ge,p.ge,a.s);upload(a.ng,std::vector<int>{int(p.gp.size())},a.s);upload(a.split,std::vector<int>{std::max(0,split_at)},a.s);ck(cudaEventRecord(a.ready,a.s),"ready");ck(cudaStreamSynchronize(a.s),"ready done");}
static cudaGraphExec_t capture(Lane& a,const x3::WinArgs& w,bool reuse){
    nx::set_input_reuse(reuse);cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;
    ck(cudaStreamBeginCapture(a.s,cudaStreamCaptureModeThreadLocal),"window capture");require(nx::window(w,a.s),"window capture refused");ck(cudaStreamEndCapture(a.s,&g),"window end capture");ck(cudaGraphInstantiate(&e,g,0),"window instantiate");cudaGraphDestroy(g);return e;
}
static double time_graph(Lane& a,cudaGraphExec_t g,int reps){
    cudaEvent_t begin,end;ck(cudaEventCreate(&begin),"timer");ck(cudaEventCreate(&end),"timer");
    for(int i=0;i<5;++i)ck(cudaGraphLaunch(g,a.s),"warm");ck(cudaEventRecord(begin,a.s),"timer start");
    for(int i=0;i<reps;++i)ck(cudaGraphLaunch(g,a.s),"timed replay");ck(cudaEventRecord(end,a.s),"timer end");ck(cudaEventSynchronize(end),"timer wait");float ms=0;ck(cudaEventElapsedTime(&ms,begin,end),"timer elapsed");cudaEventDestroy(begin);cudaEventDestroy(end);return 1000.0*ms/reps;
}
static void token_and_scope_gate(Lane& a,uint8_t* blobs,size_t blob,const std::vector<float>& hx){
    upload(a.x,hx,a.s);auto p=plan(1,0,false,blobs,blob);load_plan(a,p,-1);
    x3::MoeArgs m{};m.plan_ptr=a.ptr;m.plan_w=a.weight;m.n_embd=H;m.n_ff=F;m.k=PICKS;m.limit=7.f;
    for(int loads:{0,4,12}){
        nx::set_fp4_kernel(loads);std::vector<float> ref;
        for(bool reuse:{false,true}){
            nx::set_input_reuse(reuse);require(nx::gate_up(m,a.x,a.s)&&nx::down(m,nullptr,a.out,a.s),"one-row gate_up/down refused");
            auto actual=read(a.out,size_t(H),a.s);if(!reuse)ref=actual;else require(std::memcmp(ref.data(),actual.data(),H*4)==0,"one-row gate_up/down mismatch");
        }
    }
    // Explicitly prove normal prompt and general-set ingestion remain supported
    // with the switch on, retaining their existing bounded representation.
    std::vector<uint16_t> hx16(H);for(int i=0;i<H;++i)hx16[i]=canon::f32_to_bf16_rn(hx[i]);
    auto* x16=dev<uint16_t>(H);auto* output16=dev<uint16_t>(H);upload(x16,hx16,a.s);
    nx::PromptJob pj{};pj.blob=blobs;pj.n=1;pj.entry[0]=0;
    auto* jobs=dev<nx::PromptJob>(1);upload(jobs,std::vector<nx::PromptJob>{pj},a.s);
    const size_t bytes=nx::prompt_scratch_bytes(a.s,1);auto* scratch=dev<uint8_t>(bytes);
    std::vector<uint16_t> prompt_ref;long long enq0,row0,unhandled0;nx::input_reuse_counts(enq0,row0,unhandled0);
    for(bool reuse:{false,true}){
        nx::set_input_reuse(reuse);require(nx::prompt_run(jobs,0,1,x16,output16,scratch,bytes,1,a.s),"prompt baseline scope refused");
        auto actual=read(output16,size_t(H),a.s);if(!reuse)prompt_ref=actual;else require(actual==prompt_ref,"prompt scope changed arithmetic");
    }
    const int bounds[2]={0,1};x3::SetArgs set{};set.wbase=blobs;set.stride=blob;set.n_exp=1;set.hbounds=bounds;set.n_embd=H;set.n_ff=F;set.limit=7.f;
    auto* ids=dev<int>(1);upload(ids,std::vector<int>{0},a.s);std::vector<float> set_ref;
    for(bool reuse:{false,true}){
        nx::set_input_reuse(reuse);require(nx::set(set,a.x,LD,ids,a.out,H,a.s),"general-set baseline scope refused");
        auto actual=read(a.out,size_t(H),a.s);if(!reuse)set_ref=actual;else require(actual==set_ref,"general-set scope changed arithmetic");
    }
    long long enq,row,unhandled;nx::input_reuse_counts(enq,row,unhandled);require(unhandled-unhandled0==2,"unhandled scope enqueue counter");
    cudaFree(ids);cudaFree(scratch);cudaFree(jobs);cudaFree(output16);cudaFree(x16);
    std::puts("one-row + scope PASS: direct gate_up/down exact at donor0/P4/P12; prompt/general-set baseline explicit and counted");
}
int main(int argc,char** argv){
    try{
        const int device=argc>1?std::atoi(argv[1]):0;ck(cudaSetDevice(device),"device");Lane a,b;codec_gate(a.s);
        const size_t rec=ff::record_stride(H,F),blob=sizeof(nx::Header)+rec;std::vector<uint8_t> host(blob*POOL);std::mt19937 rng(151125);
        for(int e=0;e<POOL;++e){nx::Header h{};for(int k=0;k<3;++k){h.weight_global[k]=50.f;h.input_global[k]=1.f;}std::memcpy(host.data()+size_t(e)*blob,&h,sizeof(h));
            auto* p=host.data()+size_t(e)*blob+sizeof(h);for(size_t i=0;i<rec;++i)p[i]=i%144<128?uint8_t(rng()):uint8_t(0x28+(rng()&7));}
        auto* blobs=dev<uint8_t>(host.size());upload(blobs,host,a.s);std::vector<float> hx(size_t(TMAX)*LD);std::normal_distribution<float> normal(0,1);for(auto& v:hx)v=normal(rng);
        token_and_scope_gate(a,blobs,blob,hx);
        nx::set_fast(2);int failures=0;std::puts("full-window us/replay, fixed synthetic expert working set (cache condition shared by arms; not DRAM bandwidth); setup+FFN+scatter+combine included");
        for(int loads:{0,4,12})for(int T:{1,2,4,8,9,16})for(int kind:{0,1,2}){
            if(kind==2&&T!=16)continue;
            // -1 unsplit,0 empty early part, midpoint two nonempty parts.
            const int parts[]={-1,0,T*PICKS/2};
            for(int split_at:parts){
                if(split_at>=0&&kind==0)continue;
                nx::set_fp4_kernel(loads);auto p=plan(T,kind,false,blobs,blob);load_plan(a,p,split_at);load_plan(b,p,split_at);upload(a.x,hx,a.s);auto hb=hx;for(auto& v:hb)v=-v;upload(b.x,hb,b.s);
                const auto wa=a.args(T,split_at),wb=b.args(T,split_at);
                const auto olda=capture(a,wa,false),olda2=capture(b,wb,false),newa=capture(a,wa,true),newb=capture(b,wb,true);
                for(int replay=0;replay<2;++replay){
                    if(replay){p=plan(T,kind,true,blobs,blob);load_plan(a,p,split_at);load_plan(b,p,split_at);std::reverse(hx.begin(),hx.end());upload(a.x,hx,a.s);hb=hx;for(auto& v:hb)v=-v;upload(b.x,hb,b.s);}
                    ck(cudaGraphLaunch(olda,a.s),"baseline A");ck(cudaGraphLaunch(olda2,b.s),"baseline B");const auto refa=read(a.out,size_t(T)*H,a.s),refb=read(b.out,size_t(T)*H,b.s);
                    // No wait between launches: contexts must own independent scratch.
                    ck(cudaGraphLaunch(newa,a.s),"reuse A");ck(cudaGraphLaunch(newb,b.s),"reuse B");const auto gota=read(a.out,size_t(T)*H,a.s),gotb=read(b.out,size_t(T)*H,b.s);
                    if(std::memcmp(refa.data(),gota.data(),refa.size()*4)||std::memcmp(refb.data(),gotb.data(),refb.size()*4))++failures;
                }
                // Small bounded timing screen; per-pass engine timing remains decisive.
                const double old_us=time_graph(a,olda,30),new_us=time_graph(a,newa,30);
                std::printf("P=%2d T=%2d layout=%d split=%3d old=%8.2f reuse=%8.2f us %s\n",loads,T,kind,split_at,old_us,new_us,failures?"CHECKFAIL":"same");
                for(auto g:{olda,olda2,newa,newb})cudaGraphExecDestroy(g);
            }
        }
        nx::set_input_reuse(false);nx::set_fp4_kernel(0);cudaFree(blobs);
        std::printf("primary input reuse %s: codec oracle, donor-ID0 equality, same-P4/P12, shuffled/repeated row IDs, inactive first32, >32jobs, empty early part, changed-input replay and independent concurrent contexts\n",failures?"FAIL":"PASS");return failures?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"primary input reuse: %s\n",e.what());return 2;}
}
