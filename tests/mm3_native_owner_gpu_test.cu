// Real 6144x3072 same-arithmetic comparison: all-resident production adapter
// versus bounded streamed production scatter/combine on four reused frames.
// Usage: mm3_native_owner_gpu_test /path/to/l3-e0.rg16 [device] [fp4|dp4a]
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace nx=strata::mm3::native_experts;
namespace x3=strata::mm3::exl3;
namespace ff=infernix::ops::minimax_ffn;
static void ck(cudaError_t e,const char* x){if(e!=cudaSuccess)throw std::runtime_error(std::string(x)+": "+cudaGetErrorString(e));}
#define CUDA(x) ck((x),#x)
static void require(bool x,const char* m){if(!x)throw std::runtime_error(m);}
template<class T>struct Buf{T* p=nullptr;explicit Buf(size_t n){CUDA(cudaMalloc((void**)&p,n*sizeof(T)));}~Buf(){if(p)cudaFree(p);}Buf(const Buf&)=delete;};
struct Plan {
    unsigned long long pp[128]{};float w[128]{};unsigned long long gp[128]{};
    int gs[129]{},ge[128]{},ng=0;
};
int main(int argc,char**argv){try{
    require(argc>=2,"RG16 record path required");const int dev=argc>2?std::atoi(argv[2]):0;
    const std::string mode=argc>3?argv[3]:"fp4";require(mode=="fp4"||mode=="dp4a","invalid arithmetic");
    CUDA(cudaSetDevice(dev));constexpr int H=6144,F=3072,N=16,FRAMES=4,K=4;
    const size_t payload=ff::record_stride(H,F),blob=sizeof(nx::Header)+payload;
    std::ifstream in(argv[1],std::ios::binary|std::ios::ate);require(bool(in)&&in.tellg()==std::streamoff(payload+3*sizeof(float)),"invalid RG16 fixture");
    in.seekg(0);std::vector<uint8_t> bytes(payload);float mult[3];in.read((char*)bytes.data(),payload);in.read((char*)mult,sizeof mult);require(bool(in),"short RG16 fixture");
    // Pinned immutable records are used both to construct the all-resident
    // bank and as streamed sources. Vary global scales by identity.
    uint8_t* host=nullptr;CUDA(cudaMallocHost((void**)&host,N*blob));
    for(int e=0;e<N;++e){nx::Header h{};for(int i=0;i<3;++i){require(std::isfinite(mult[i])&&mult[i]>0,"invalid trailer");h.weight_global[i]=(1.f/mult[i])*(1.f+.025f*(e+1)*(i+1));h.input_global[i]=1;}
        std::memcpy(host+e*blob,&h,sizeof h);std::memcpy(host+e*blob+sizeof h,bytes.data(),payload);}
    Buf<uint8_t> resident(N*blob),frames(FRAMES*blob);CUDA(cudaMemcpy(resident.p,host,N*blob,cudaMemcpyHostToDevice));
    cudaStream_t s;CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
    auto arithmetic=mode=="fp4"?nx::Arithmetic::NativeFp4:nx::Arithmetic::Dp4a;
    Buf<uint8_t> work(nx::scratch_bytes(H,F,arithmetic));require(nx::prepare(s,H,F,work.p,nx::scratch_bytes(H,F,arithmetic),arithmetic),"adapter prepare");
    Buf<Plan> plan(1);Buf<float> dx(16*H),sh(16*H),part(16*8*H),out(16*H),ref(16*H);
    std::vector<float> input(16*H),shared(16*H),got(16*H),want(16*H);
    for(int t=0;t<16;++t)for(int i=0;i<H;++i){input[t*H+i]=float(((i*7+t*3)%17)-8)/32.f;shared[t*H+i]=float((i+t)%7-3)/64.f;}
    CUDA(cudaMemcpy(dx.p,input.data(),input.size()*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(sh.p,shared.data(),shared.size()*4,cudaMemcpyHostToDevice));CUDA(cudaStreamSynchronize(nullptr));
    x3::WinArgs a{};a.grp_ptr=plan.p->gp;a.grp_start=plan.p->gs;a.grp_ent=plan.p->ge;a.n_groups=&plan.p->ng;a.plan_ptr=plan.p->pp;a.plan_w=plan.p->w;
    a.x=dx.p;a.x_stride=H;a.part=part.p;a.sh_out=sh.p;a.out_stride=H;a.k=K;a.n_embd=H;a.n_ff=F;a.limit=7;
    // One pinned plan per pass: asynchronous H2D readers cannot observe a
    // subsequently rewritten host descriptor.
    Plan* plans=nullptr;CUDA(cudaMallocHost((void**)&plans,6*sizeof(Plan)));
    for(int T:{1,8,16}) {
        std::vector<std::vector<int>> groups(N);for(int t=0;t<T;++t)for(int k=0;k<K;++k){int e=k==0?0:1+((t*3+k-1)%15);groups[e].push_back(t*8+k);}
        auto make=[&](Plan& p,int begin,int end,bool staged){
            p=Plan{};for(int t=0;t<T;++t)for(int k=0;k<K;++k){int idx=t*8+k;p.pp[idx]=1;p.w[idx]=.07f+.035f*k+.001f*t;}
            int n=0,pieces=0;for(int e=begin;e<end;++e)if(!groups[e].empty()){
                int j=p.ng++;p.gp[j]=(unsigned long long)(uintptr_t)(staged?frames.p+(e-begin)*blob:resident.p+e*blob);p.gs[j]=n;
                for(int entry:groups[e])p.ge[n++]=entry;pieces+=(groups[e].size()+7)/8;
            }p.gs[p.ng]=n;return pieces;
        };
        a.T=T;a.out=ref.p;make(plans[0],0,N,false);CUDA(cudaMemcpyAsync(plan.p,plans,sizeof(Plan),cudaMemcpyHostToDevice,s));
        require(nx::window(a,s),"all-resident window");
        // A missed streamed write must not inherit the reference partial.
        CUDA(cudaMemsetAsync(part.p,0xff,16*8*H*sizeof(float),s));
        a.out=out.p;
        int pi=1;for(int begin=0;begin<N;begin+=FRAMES){
            int end=std::min(N,begin+FRAMES);
            for(int e=begin;e<end;++e)if(!groups[e].empty())CUDA(cudaMemcpyAsync(frames.p+(e-begin)*blob,host+e*blob,blob,cudaMemcpyHostToDevice,s));
            int pieces=make(plans[pi],begin,end,true);
            CUDA(cudaMemcpyAsync(plan.p,&plans[pi],sizeof(Plan),cudaMemcpyHostToDevice,s));
            require(nx::window_scatter(a,pieces,s),"streamed scatter");++pi;
        }
        require(nx::window_combine(a,s),"streamed combine");CUDA(cudaStreamSynchronize(s));
        CUDA(cudaMemcpy(got.data(),out.p,(size_t)T*H*4,cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(want.data(),ref.p,(size_t)T*H*4,cudaMemcpyDeviceToHost));
        bool nonzero=false;for(int i=0;i<T*H;++i){require(std::isfinite(got[i])&&std::isfinite(want[i]),"nonfinite result");if(got[i]!=shared[i])nonzero=true;if(got[i]!=want[i]){std::fprintf(stderr,"T=%d i=%d actual %.9g reference %.9g\n",T,i,got[i],want[i]);throw std::runtime_error("streamed versus resident mismatch");}}
        require(nonzero,"no routed expert contribution");std::printf("T=%d all-resident and streamed exact match\n",T);
    }
    nx::release(s);CUDA(cudaStreamDestroy(s));CUDA(cudaFreeHost(plans));CUDA(cudaFreeHost(host));return 0;
}catch(const std::exception& e){std::fprintf(stderr,"native owner GPU fixture: %s\n",e.what());return 1;}}
