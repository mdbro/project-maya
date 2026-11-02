// Same-device production-adapter check using a real Proposal 1 RG16 record.
// Usage: mm3_native_experts_test /absolute/path/l3-e0.rg16 [device]
// Links Maya's x3/native expert sources and Infernix fast FFN sources.
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>
#include <vector>

namespace x3 = strata::mm3::exl3;
namespace nx = strata::mm3::native_experts;
namespace ff = infernix::ops::minimax_ffn;
constexpr int H=6144, F=3072, T=4, K=4;

static void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
#define CUDA(call) cuda_check((call), #call)
static void require(bool b,const char* msg) { if(!b) throw std::runtime_error(msg); }
template<class V> struct Device {
    V* p=nullptr; size_t count;
    explicit Device(size_t n):count(n) { CUDA(cudaMalloc((void**)&p,n*sizeof(V))); }
    ~Device() { if(p) (void)cudaFree(p); }
    Device(const Device&)=delete; Device& operator=(const Device&)=delete;
    void upload(const V* src,size_t n) {
        require(n<=count,"upload overflow");
        CUDA(cudaMemcpy(p,src,n*sizeof(V),cudaMemcpyHostToDevice));
        // The adapter uses a nonblocking stream. A pageable H2D copy may return
        // after staging while its default-stream DMA is still in flight.
        CUDA(cudaStreamSynchronize(nullptr));
    }
    void download(V* dst,size_t n) const { require(n<=count,"download overflow"); CUDA(cudaMemcpy(dst,p,n*sizeof(V),cudaMemcpyDeviceToHost)); }
};
static float from_bf16(uint16_t b) { uint32_t u=uint32_t(b)<<16;float x;std::memcpy(&x,&u,4);return x; }
__global__ static void selected_multipliers(ff::FastJob* job,const nx::Header* header) {
    if(threadIdx.x==0)for(int i=0;i<3;++i)job->multiplier[i]=1.0f/header->weight_global[i];
}
static void check_rows(const char* name,const std::vector<float>& got,const std::vector<float>& want) {
    require(got.size()==want.size(),"comparison size");
    for(size_t i=0;i<got.size();++i) {
        const float tolerance=0.01f*std::max(1.0f,std::fabs(want[i]));
        if(!std::isfinite(got[i])||!std::isfinite(want[i])||std::fabs(got[i]-want[i])>tolerance) {
            std::fprintf(stderr,"%s mismatch index %zu: got %.9g expected %.9g tolerance %.4g\n",name,i,got[i],want[i],tolerance);
            throw std::runtime_error(name);
        }
    }
}
int main(int argc,char** argv) {
 try {
    require(argc>=2,"record path required");
    const int device=argc>2?std::atoi(argv[2]):0; CUDA(cudaSetDevice(device));
    const size_t bytes=ff::record_stride(H,F), blob_bytes=sizeof(nx::Header)+bytes;
    std::ifstream in(argv[1],std::ios::binary|std::ios::ate); require(bool(in),"cannot open RG16 record");
    require(in.tellg()==std::streamoff(bytes+3*sizeof(float)),"record size or reciprocal trailer mismatch");
    in.seekg(0); std::vector<uint8_t> source(bytes); float mult[3];
    in.read((char*)source.data(),bytes); in.read((char*)mult,sizeof mult); require(bool(in),"short RG16 record");
    nx::Header header{};
    for(int i=0;i<3;++i) { require(std::isfinite(mult[i])&&mult[i]>0,"invalid reciprocal trailer"); header.weight_global[i]=1.0f/mult[i];header.input_global[i]=1.0f; }
    // Two physical records enable unequal device-side groups and prompt-set stride.
    Device<uint8_t> blobs(2*blob_bytes);
    CUDA(cudaMemcpy(blobs.p,&header,sizeof header,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(blobs.p+sizeof header,source.data(),bytes,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(blobs.p+blob_bytes,&header,sizeof header,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(blobs.p+blob_bytes+sizeof header,source.data(),bytes,cudaMemcpyHostToDevice));
    unsigned long long p0=(unsigned long long)(uintptr_t)blobs.p;
    unsigned long long p1=(unsigned long long)(uintptr_t)(blobs.p+blob_bytes);
    // Exactly BF16-representable, varying both by token and input index.
    const float fv[]={-0.5f,-0.25f,0.25f,0.5f};
    const uint16_t bv[]={0xbf00,0xbe80,0x3e80,0x3f00};
    std::vector<float> x(T*H);std::vector<uint16_t> xb(T*H);
    for(int t=0;t<T;++t) for(int i=0;i<H;++i) {int v=(i*3+t)%4;x[(size_t)t*H+i]=fv[v];xb[(size_t)t*H+i]=bv[v];}
    Device<float> dx(x.size());dx.upload(x.data(),x.size());Device<uint16_t> dx16(xb.size()),dy16(xb.size());dx16.upload(xb.data(),xb.size());
    cudaStream_t s;CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    // The reference uses the record trailer directly. The adapter has to recover
    // it from the reciprocal header; small BF16 boundary differences are allowed.
    ff::FastJob ref{};ref.record=blobs.p+sizeof header;ref.x=dx16.p;ref.y=dy16.p;ref.columns=T;
    for(int i=0;i<3;++i)ref.multiplier[i]=mult[i];
    Device<ff::FastJob> refjob(1);refjob.upload(&ref,1);
    Device<uint8_t> refwork(ff::fast_workspace_bytes(H,F,T));
    ff::fast_ffn_cuda(refjob.p,1,H,F,T,refwork.p,s);CUDA(cudaStreamSynchronize(s));
    std::vector<uint16_t> yr(T*H);dy16.download(yr.data(),yr.size());
    std::vector<float> reference(T*H);for(size_t i=0;i<yr.size();++i)reference[i]=from_bf16(yr[i]);
    require(std::any_of(reference.begin(),reference.end(),[](float v){return std::isfinite(v)&&v!=0;}),"reference produced no nonzero finite output");
    // Scatter must match the selected packed C=8 arithmetic exactly. Keep the
    // independent unpacked/trailer reference above for the weighted comparison;
    // its reciprocal round trip and decoder can differ at BF16 boundaries.
    refjob.upload(&ref,1);selected_multipliers<<<1,1,0,s>>>(refjob.p,(const nx::Header*)blobs.p);
    Device<uint8_t> packed_work(ff::fast_workspace_bytes(H,F,8));
    ff::fast_ffn_cuda_packed(refjob.p,1,H,F,8,packed_work.p,s);CUDA(cudaStreamSynchronize(s));
    std::vector<uint16_t> packed_y(T*H);dy16.download(packed_y.data(),packed_y.size());
    size_t scratch_bytes=nx::scratch_bytes(H,F);require(scratch_bytes>0,"expert scratch query");
    Device<uint8_t> scratch(scratch_bytes);require(nx::prepare(s,H,F,scratch.p,scratch_bytes),"expert prepare");
    Device<unsigned long long> plan(T*8), groups(2);
    Device<float> weights(T*8), shared(T*H), cpu(T*H), out(T*H), part(T*8*H);
    Device<int> starts(3), entries(5), ng(1), flags(T), rows(5);
    std::vector<float> sh(T*H,0.125f), cp(T*H);for(int t=0;t<T;++t)std::fill(cp.begin()+(size_t)t*H,cp.begin()+(size_t)(t+1)*H,0.25f*(t+1));
    shared.upload(sh.data(),sh.size());cpu.upload(cp.data(),cp.size());
    int cpu_flags[T]={1,0,1,0}; flags.upload(cpu_flags,T);
    std::vector<unsigned long long> pp(T*8,0);std::vector<float> ww(T*8,0);
    pp[0]=p0;pp[2]=p1;ww[0]=0.5f;ww[2]=0.25f;plan.upload(pp.data(),pp.size());weights.upload(ww.data(),ww.size());
    x3::MoeArgs ma{};ma.lay.fmt=nx::kFmt;ma.plan_ptr=plan.p;ma.plan_w=weights.p;ma.k=K;ma.n_embd=H;ma.n_ff=F;ma.limit=7;ma.cpu_part=cpu.p;ma.cpu_flag=flags.p;
    const int errors_before=x3::launch_errors();
    x3::moe_gate_up(ma,dx.p,nullptr,s);x3::moe_down(ma,nullptr,shared.p,out.p,s);CUDA(cudaStreamSynchronize(s));
    require(x3::launch_errors()==errors_before,"token dispatch failed");
    std::vector<float> actual(T*H),expected(T*H);out.download(actual.data(),H);
    for(int i=0;i<H;++i)expected[i]=0.125f+0.75f*reference[i]+0.25f;
    check_rows("token active",{actual.begin(),actual.begin()+H},{expected.begin(),expected.begin()+H});
    // Exercise the short final batch (five descriptors) with live and null slots.
    ma.k=5;pp[4]=p0;ww[4]=0.125f;plan.upload(pp.data(),pp.size());weights.upload(ww.data(),ww.size());
    x3::moe_gate_up(ma,dx.p,nullptr,s);x3::moe_down(ma,nullptr,shared.p,out.p,s);CUDA(cudaStreamSynchronize(s));out.download(actual.data(),H);
    for(int i=0;i<H;++i)expected[i]=0.375f+0.875f*reference[i];
    check_rows("token partial batch",{actual.begin(),actual.begin()+H},{expected.begin(),expected.begin()+H});ma.k=K;
    std::fill(pp.begin(),pp.end(),0);plan.upload(pp.data(),pp.size());
    x3::moe_gate_up(ma,dx.p,nullptr,s);x3::moe_down(ma,nullptr,shared.p,out.p,s);CUDA(cudaStreamSynchronize(s));out.download(actual.data(),H);
    for(int i=0;i<H;++i)expected[i]=0.375f;
    check_rows("token empty",{actual.begin(),actual.begin()+H},{expected.begin(),expected.begin()+H});
    // Group 0 uses one resident expert for all four tokens, group 1 adds a
    // different resident slot only for token 0. Remaining device jobs inactive.
    pp.assign(T*8,0);ww.assign(T*8,0);for(int t=0;t<T;++t){pp[t*8]=p0;ww[t*8]=0.25f;}
    pp[1]=p1;ww[1]=0.5f;plan.upload(pp.data(),pp.size());weights.upload(ww.data(),ww.size());
    const unsigned long long gp[2]={p0,p1};groups.upload(gp,2);
    const int gs[3]={0,4,5}, ge[5]={0,8,16,24,1}, two=2;starts.upload(gs,3);entries.upload(ge,5);ng.upload(&two,1);
    unsigned* host_word=nullptr;CUDA(cudaHostAlloc((void**)&host_word,sizeof(unsigned),cudaHostAllocMapped|cudaHostAllocPortable));
    unsigned* device_word=nullptr;CUDA(cudaHostGetDevicePointer((void**)&device_word,host_word,0));
    std::atomic_thread_fence(std::memory_order_seq_cst);*(volatile unsigned*)host_word=37;
    x3::WinArgs wa{};wa.lay.fmt=nx::kFmt;wa.grp_ptr=groups.p;wa.grp_start=starts.p;wa.grp_ent=entries.p;wa.n_groups=ng.p;
    wa.plan_ptr=plan.p;wa.plan_w=weights.p;wa.x=dx.p;wa.x_stride=H;wa.part=part.p;wa.sh_out=shared.p;wa.out=out.p;wa.out_stride=H;
    wa.cpu_part=cpu.p;wa.cpu_stride=H;wa.cpu_flag=flags.p;wa.wait_ptr=device_word;wa.wait_val=37;wa.T=T;wa.k=K;wa.n_embd=H;wa.n_ff=F;wa.limit=7;
    require(x3::win_experts(wa,s),"window dispatch");CUDA(cudaStreamSynchronize(s));out.download(actual.data(),actual.size());
    for(int t=0;t<T;++t)for(int i=0;i<H;++i)expected[(size_t)t*H+i]=0.125f+(t==0?0.75f:0.25f)*reference[(size_t)t*H+i]+(cpu_flags[t]?0.25f*(t+1):0);
    check_rows("window grouped",actual,expected);
    const int zero=0;ng.upload(&zero,1);pp.assign(T*8,0);plan.upload(pp.data(),pp.size());
    require(x3::win_experts(wa,s),"empty window dispatch");CUDA(cudaStreamSynchronize(s));out.download(actual.data(),actual.size());
    for(int t=0;t<T;++t)for(int i=0;i<H;++i)expected[(size_t)t*H+i]=0.125f+(cpu_flags[t]?0.25f*(t+1):0);
    check_rows("window empty",actual,expected);

    // Production window mapping: one group spans a split at nine columns,
    // a second live group has a different weight, and a third is inactive.
    // Inputs repeat the independently computed four-row reference.
    constexpr int MT=16, MK=3;
    std::vector<float> wx(MT*H), wsh(MT*H,0.125f), wcpu(MT*H), wgot(MT*H), wwant(MT*H);
    for(int t=0;t<MT;++t) { std::copy_n(x.data()+(size_t)(t%T)*H,H,wx.data()+(size_t)t*H); std::fill_n(wcpu.data()+(size_t)t*H,H,0.25f*(t+1)); }
    Device<float> wdx(wx.size()), wdsh(wsh.size()), wdcpu(wcpu.size()), wdout(wgot.size()), wdpart(MT*8*H), wdw(MT*8);
    Device<unsigned long long> wdplan(MT*8), wdgroups(3);
    Device<int> wdst(4), wdent(MT+2), wdng(1), wdflags(MT);
    wdx.upload(wx.data(),wx.size());wdsh.upload(wsh.data(),wsh.size());wdcpu.upload(wcpu.data(),wcpu.size());
    int wf[MT];for(int t=0;t<MT;++t)wf[t]=t%2==0;wdflags.upload(wf,MT);
    const unsigned long long wgp[3]={p0,p1,0};wdgroups.upload(wgp,3);const int three=3;wdng.upload(&three,1);
    cudaEvent_t done;CUDA(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));
    for(int width: {1,8,9,16}) {
        std::vector<unsigned long long> wp(MT*8,0);std::vector<float> weights_host(MT*8,0);
        std::vector<int> ents(width+2);for(int t=0;t<width;++t) { ents[t]=t*8;wp[t*8]=p0;weights_host[t*8]=0.25f; }
        ents[width]=1;ents[width+1]=2;wp[1]=p1;weights_host[1]=0.5f;
        const int bounds[4]={0,width,width+1,width+2};wdst.upload(bounds,4);wdent.upload(ents.data(),ents.size());
        wdplan.upload(wp.data(),wp.size());wdw.upload(weights_host.data(),weights_host.size());
        std::vector<float> sentinel(MT*8*H,-123.0f);wdpart.upload(sentinel.data(),sentinel.size());
        x3::WinArgs w{};w.lay.fmt=nx::kFmt;w.grp_ptr=wdgroups.p;w.grp_start=wdst.p;w.grp_ent=wdent.p;w.n_groups=wdng.p;
        w.plan_ptr=wdplan.p;w.plan_w=wdw.p;w.x=wdx.p;w.x_stride=H;w.part=wdpart.p;w.sh_out=wdsh.p;w.out=wdout.p;w.out_stride=H;
        w.cpu_part=wdcpu.p;w.cpu_stride=H;w.cpu_flag=wdflags.p;w.wait_ptr=device_word;w.wait_val=37;w.T=width;w.k=MK;w.n_embd=H;w.n_ff=F;w.limit=7;
        *(volatile unsigned*)host_word=0;std::atomic_thread_fence(std::memory_order_seq_cst);
        const bool submitted=x3::win_experts(w,s);if(!submitted)*(volatile unsigned*)host_word=37;
        require(submitted,"width window dispatch");CUDA(cudaEventRecord(done,s));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));const cudaError_t before=cudaEventQuery(done);
        std::atomic_thread_fence(std::memory_order_seq_cst);*(volatile unsigned*)host_word=37;
        require(before==cudaErrorNotReady,"window published before CPU contribution");CUDA(cudaStreamSynchronize(s));wdout.download(wgot.data(),(size_t)width*H);
        for(int t=0;t<width;++t)for(int i=0;i<H;++i)wwant[(size_t)t*H+i]=0.125f+(t==0?0.75f:0.25f)*reference[(size_t)(t%T)*H+i]+(wf[t]?0.25f*(t+1):0);
        check_rows("window widths",{wgot.begin(),wgot.begin()+(size_t)width*H},{wwant.begin(),wwant.begin()+(size_t)width*H});
        wdpart.download(sentinel.data(),sentinel.size());
        for(int t=0;t<width;++t)for(int i=0;i<H;++i) {
            const size_t ri=(size_t)(t%T)*H+i;
            const float want=from_bf16(packed_y[ri]), got=sentinel[((size_t)t*8)*H+i];
            if(got!=want) {
                std::fprintf(stderr,"window scatter width=%d token=%d element=%d got=%.9g packed=%.9g unpacked=%.9g\n",width,t,i,got,want,reference[ri]);
                throw std::runtime_error("window scatter");
            }
        }
        require(sentinel[2*H]==-123.0f,"inactive group scattered");
    }
    CUDA(cudaEventDestroy(done));
    CUDA(cudaFreeHost(host_word));
    // Global hbounds start at row 3, while row_idx and y are pass-local.
    const int hb[3]={3,7,8}, row_ids[5]={3,0,2,1,2};rows.upload(row_ids,5);
    Device<float> setout(5*H);x3::SetArgs sa{};sa.wbase=blobs.p;sa.stride=blob_bytes;sa.lay.fmt=nx::kFmt;sa.n_exp=2;sa.hbounds=hb;sa.row0=3;sa.max_rows=4;sa.n_embd=H;sa.n_ff=F;sa.limit=7;
    require(nx::set(sa,dx.p,H,rows.p,setout.p,H,s),"prompt set dispatch");CUDA(cudaStreamSynchronize(s));
    std::vector<float> setgot(5*H),setexpected(5*H);setout.download(setgot.data(),setgot.size());
    for(int j=0;j<5;++j)std::copy_n(reference.data()+(size_t)row_ids[j]*H,H,setexpected.data()+(size_t)j*H);
    check_rows("prompt set",setgot,setexpected);
    nx::release(s);CUDA(cudaStreamDestroy(s));std::puts("native expert adapter: OK");return 0;
 } catch(const std::exception& e) {std::fprintf(stderr,"native expert adapter test: %s\n",e.what());return 1;}
}
