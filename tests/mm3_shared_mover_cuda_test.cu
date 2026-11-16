// Manual private-allocation gate, never a model-speed result. Real expert
// payloads, both production helper graph routes, shared VMM and direct swaps.
#include "../src/mm3/mm3_native_helper.hpp"
#include "../src/mm3/mm3_native_host.hpp"
#include "../src/mm3/mm3_background_traffic.hpp"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using H=mm3::NativeHelper;
constexpr int packed_row=H::kHidden/2+H::kHidden/16;
static void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
static void ck(cudaError_t e,const char* why){if(e!=cudaSuccess)throw std::runtime_error(std::string(why)+": "+cudaGetErrorString(e));}
static int card(const char* name) {
    int n=0;ck(cudaGetDeviceCount(&n),"device count");
    for(int i=0;i<n;++i){cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,i),"device property");if(std::strstr(p.name,name))return i;}
    throw std::runtime_error(std::string("missing card ")+name);
}
struct Record {std::vector<unsigned char> bytes;float weight[3]{},input[3]{1,2,3};};
static Record read(const char* path,size_t stride) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f) && f.tellg()==std::streamoff(stride+12),"real RG16 record size");
    Record r;r.bytes.resize(stride);f.seekg(0);f.read(reinterpret_cast<char*>(r.bytes.data()),stride);
    float scale[3];f.read(reinterpret_cast<char*>(scale),12);require(bool(f),"real RG16 record read");
    for(int i=0;i<3;++i){require(std::isfinite(scale[i]) && scale[i]>0,"record scale");r.weight[i]=1.0f/scale[i];}
    return r;
}
static std::vector<float> compute(H& h,int expert,int columns,int epoch) {
    std::vector<unsigned char> x(size_t(columns)*packed_row);
    for(int t=0;t<columns;++t) {
        auto* row=x.data()+size_t(t)*packed_row;
        for(int k=0;k<H::kHidden/2;++k)row[k]=static_cast<unsigned char>(((k+t+epoch)%16)|(((3*k+epoch+1)%16)<<4));
        std::fill(row+H::kHidden/2,row+packed_row,static_cast<unsigned char>(0x20+(epoch%3)));
    }
    H::Job job{};job.layer=3;job.expert=expert;job.columns=columns;
    for(int t=0;t<columns;++t){job.token[t]=t;job.weight[t]=1.0f;}
    std::string error;const auto seq=h.submit_packed(columns,x.data(),{job},error);require(seq,"submit packed: "+error);
    std::vector<float> out(size_t(columns)*H::kHidden);
    require(h.collect(seq,out.data(),out.size(),error),"collect: "+error);
    require(std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);}),"nonfinite helper output");
    require(std::any_of(out.begin(),out.end(),[](float x){return x!=0;}),"all-zero helper output");return out;
}
static void equal(const std::vector<float>& a,const std::vector<float>& b,const char* why) {
    require(a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0,why);
}
int main(int argc,char** argv) {
    // Failure deliberately exits without freeing leased allocations whose DMA
    // retirement might be unknown. The runner provides a process watchdog.
    try {
        require(argc==3,"usage: mm3_shared_mover_cuda_test A.rg16 B.rg16");
        const int primary=card("5090"),helper=card("3090"),third=card("4080");
        const size_t stride=infernix::ops::minimax_ffn::record_stride(H::kHidden,H::kIntermediate);
        const auto a=read(argv[1],stride),b=read(argv[2],stride);
        auto* bank=new mm3::NativeHostBank(3*stride,primary,2ULL<<30,2ULL<<30,true,{helper,third});
        auto* host=reinterpret_cast<unsigned char*>(bank->data());
        std::memcpy(host,a.bytes.data(),stride);std::memcpy(host+stride,b.bytes.data(),stride);std::memset(host+2*stride,0,stride);
        bank->grant_device();
        H* cards[2]{new H(helper,4),new H(third,3)};
        std::string error;require(cards[0]->reserve_direct_spare(error),error);
        for(unsigned actor=0;actor<2;++actor) {
            auto& h=*cards[actor];
            require(h.add_expert(3,0,host+stride,stride,b.weight,b.input,error),"add old: "+error);
            require(h.add_expert(3,2,host+stride,stride,b.weight,b.input,error),"add B reference: "+error);
            require(h.add_expert(3,3,host,stride,a.weight,a.input,error),"add A reference: "+error);
            require(h.seal(error),"seal: "+error);h.set_kernel(actor?188:172,actor?88:68);
            h.set_sm89(actor==1);h.set_fast(true);require(h.set_execution(actor?4:1,error),"execution: "+error);
            h.live_x4_window(UINT64_MAX);
        }
        auto* traffic=new mm3::BackgroundTraffic;
        for(int cycle=0;cycle<2;++cycle) {
            const int old_id=cycle?1:0,new_id=cycle?0:1;const auto& next=cycle?b:a;const auto& old=cycle?a:b;
            const auto* source=host+(cycle?stride:0);
            std::array<uint64_t,2> token{};
            for(unsigned actor=0;actor<2;++actor) {
                auto& h=*cards[actor];token[actor]=h.live_x4_revoke();float w[3]{},i[3]{};
                require(h.direct_begin(3,old_id,3,new_id,source,actor?nullptr:host+2*stride,stride,next.weight,next.input,
                                       w,i,traffic,actor,actor==0,true,error),"direct begin: "+error);
                require(std::memcmp(w,old.weight,sizeof w)==0 && std::memcmp(i,old.input,sizeof i)==0,"old metadata changed");
                equal(compute(h,2,1,cycle),compute(h,2,1,cycle),"unrelated record changed with a pending lease");
                if(!actor)equal(compute(h,old_id,8,cycle),compute(h,cycle?3:2,8,cycle),"old ready record changed before publication");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            for(unsigned actor=0;actor<2;++actor) {
                const auto s=cards[actor]->live_schedule_stats();require(s.direct_h2d_bytes==size_t(cycle)*stride,"closed guard issued direct bytes");
                require(cards[actor]->live_x4_publish(token[actor],{},error),"open direct guard: "+error);
                cards[actor]->live_x4_window(UINT64_MAX); // foreground submit closed the old link window
            }
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(30);
            for(unsigned actor=0;actor<2;++actor)for(;;) {
                const auto status=cards[actor]->live_poll();
                if(status.phase==H::LivePhase::prepared)break;
                require(status.phase!=H::LivePhase::failed,"direct worker: "+status.error);
                if(std::chrono::steady_clock::now()>=until) {
                    const auto s=cards[actor]->live_schedule_stats();
                    std::fprintf(stderr,"timeout actor=%u phase=%d chunks=%llu bytes=%llu guard_closed=%llu guard_admissions=%llu budget=%zu\n",
                        actor,int(status.phase),(unsigned long long)s.direct_chunks,(unsigned long long)s.direct_h2d_bytes,
                        (unsigned long long)s.guard_closed_polls,(unsigned long long)s.guard_admissions,traffic->used());
                    throw std::runtime_error("direct worker timeout");
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            require(std::memcmp(host+2*stride,old.bytes.data(),stride)==0,"outgoing direct payload differs");
            require(std::memcmp(host,a.bytes.data(),stride)==0 && std::memcmp(host+stride,b.bytes.data(),stride)==0,"canonical source changed");
            require(traffic->healthy() && traffic->used()==0 && traffic->high()<=traffic->capacity(),"shared byte budget failed");
            for(unsigned actor=0;actor<2;++actor) {
                auto& h=*cards[actor];require(h.direct_publish(error),"publish direct: "+error);
                bool safe=false;require(h.live_retire(safe,error) && safe,"retire direct: "+error);
                require(h.records().size()==3,"resident capacity changed");
                // Changed rows and one/eight-column jobs exercise cached graphs
                // after slot rotation, including a return to the old identity.
                for(int epoch=0;epoch<3;++epoch)for(int columns:{1,8})
                    equal(compute(h,new_id,columns,epoch),compute(h,cycle?2:3,columns,epoch),"published graph has stale weights or inputs");
            }
        }
        for(unsigned actor=0;actor<2;++actor) {
            const auto s=cards[actor]->live_schedule_stats();
            require(s.direct_h2d_bytes==2*stride && s.direct_d2h_bytes==(actor?0:2*stride) && !s.host_pieces,"wrong direct traffic");
            require(s.direct_verified==(actor?2:4) && !s.direct_mismatches && cards[actor]->graph_launches()>0,"direct verification or graphs missing");
            std::printf("card=%s h2d_bytes=%llu d2h_bytes=%llu chunks=%llu verified=%llu graphs=%llu result=PASS\n",actor?"4080":"3090",
                (unsigned long long)s.direct_h2d_bytes,(unsigned long long)s.direct_d2h_bytes,(unsigned long long)s.direct_chunks,
                (unsigned long long)s.direct_verified,(unsigned long long)cards[actor]->graph_launches());
        }
        std::printf("shared_budget_high_bytes=%zu limit_bytes=%zu result=PASS\n",traffic->high(),traffic->capacity());
        delete cards[0];delete cards[1];delete traffic;delete bank;return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
