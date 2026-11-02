// Native CPU batching versus the existing serial CPU adapter, on one real RG16
// expert. Usage: mm3_native_cpu_batch_test /path/to/l3-e0.rg16 [workers]
// No cross-device arithmetic comparison. Link Maya mm3_cpu/native_cpu and
// Infernix minimax_nvfp4_ffn CPU/record sources.
#include "mm3_native_cpu.hpp"
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ff = infernix::ops::minimax_ffn;
namespace nx = strata::mm3::native_experts;
constexpr int H=6144, F=3072, T=16, E=4;
using Users=std::vector<std::vector<std::pair<int,float>>>;
using Order=std::vector<std::pair<int,int>>;
static void require(bool b,const char* why){if(!b)throw std::runtime_error(why);}
struct Workspace {
    void* data=nullptr;size_t size=0;
    explicit Workspace(size_t n):size((n+255)&~size_t(255)){
        if(!size)size=256;data=std::aligned_alloc(256,size);require(data!=nullptr,"workspace allocation");
    }
    ~Workspace(){std::free(data);}
    Workspace(const Workspace&)=delete;Workspace& operator=(const Workspace&)=delete;
};
static void compare(const char* name,const std::vector<float>& a,const std::vector<float>& b){
    require(a.size()==b.size(),"result size");
    for(size_t i=0;i<a.size();++i) if(!std::isfinite(a[i])||!std::isfinite(b[i])||a[i]!=b[i]){
        std::fprintf(stderr,"%s mismatch at %zu: batch %.9g serial %.9g\n",name,i,a[i],b[i]);throw std::runtime_error(name);
    }
}
int main(int argc,char** argv){
 try{
    require(argc>=2,"record path required");
    int workers=argc>2?std::atoi(argv[2]):3;require(workers>=0,"invalid worker count");
    const size_t n=ff::record_stride(H,F);
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);require(bool(f),"cannot open RG16 record");
    require(f.tellg()==std::streamoff(n+3*sizeof(float)),"unexpected RG16 record or trailer size");
    f.seekg(0);std::vector<uint8_t> blob(sizeof(nx::Header)+n);float inverse[3];
    f.read((char*)blob.data()+sizeof(nx::Header),n);f.read((char*)inverse,sizeof inverse);require(bool(f),"short RG16 record");
    nx::Header header{};for(int i=0;i<3;++i){require(std::isfinite(inverse[i])&&inverse[i]>0,"invalid scale trailer");header.weight_global[i]=1.f/inverse[i];header.input_global[i]=1.f;}
    std::memcpy(blob.data(),&header,sizeof header);
    // All four jobs deliberately share a real source record; their token/user
    // lists and weights differ, exercising grouping without copying 128 MiB.
    std::vector<mm3::NativeCpuExpert> experts(E);for(auto& e:experts)e.blob=blob.data();
    mm3::CpuLane lane(workers,H,F,1.702f,7.0f,64);
    Workspace batch(mm3::native_cpu_batch_workspace_bytes(H,F,64));
    Workspace serial(mm3::native_cpu_workspace_bytes(H,F,64));
    require(mm3::native_cpu_batch_workspace_bytes(H,F,64)>0,"batch scratch query");
    std::vector<float> x((size_t)T*H);
    const float values[]={-0.5f,-0.25f,0.25f,0.5f};
    for(int t=0;t<T;++t)for(int i=0;i<H;++i)x[(size_t)t*H+i]=values[(i*3+t)%4];
    // T=1, four separate jobs and nontrivial token-major reverse order.
    Users one(E);Order one_order;
    for(int e=0;e<E;++e)one[e].push_back({0,0.125f*(e+1)});
    for(int e=E-1;e>=0;--e)one_order.push_back({e,0});
    std::vector<float> a(H,0.125f),b(H,0.125f);
    require(mm3::native_cpu_run_multi_batch(lane,H,F,experts,one,x.data(),1,a.data(),&one_order,batch.data,batch.size,true),"C1 batch");
    require(mm3::native_cpu_run_multi(lane,H,F,experts,one,x.data(),1,b.data(),&one_order,serial.data,serial.size,true),"C1 serial");
    compare("C1 weighted accumulate",a,b);
    require(std::any_of(a.begin(),a.end(),[](float v){return std::isfinite(v)&&v!=0.125f;}),"C1 reference has no contribution");
    // Full 16x4 grouping: each expert has 16 users, split into two donor jobs.
    Users users(E);Order order;
    for(int e=0;e<E;++e)for(int t=0;t<T;++t)users[e].push_back({t,0.03125f*(1+e+(t%3))});
    for(int t=0;t<T;++t)for(int e=E-1;e>=0;--e)order.push_back({e,t});
    a.assign((size_t)T*H,9.f);b.assign(a.size(),-7.f);
    require(mm3::native_cpu_run_multi_batch(lane,H,F,experts,users,x.data(),T,a.data(),&order,batch.data,batch.size),"64-entry batch");
    require(mm3::native_cpu_run_multi(lane,H,F,experts,users,x.data(),T,b.data(),&order,serial.data,serial.size),"64-entry serial");
    compare("64-entry ordered mix",a,b);
    // Empty work still resets output or preserves it for accumulate.
    Users none;std::vector<mm3::NativeCpuExpert> no_experts;
    std::fill(a.begin(),a.end(),3.f);
    require(mm3::native_cpu_run_multi_batch(lane,H,F,no_experts,none,x.data(),T,a.data(),nullptr,batch.data,batch.size),"empty batch");
    require(std::all_of(a.begin(),a.end(),[](float z){return z==0.f;}),"empty batch must clear");
    std::fill(a.begin(),a.end(),3.f);
    require(mm3::native_cpu_run_multi_batch(lane,H,F,no_experts,none,x.data(),T,a.data(),nullptr,batch.data,batch.size,true),"empty accumulate");
    require(std::all_of(a.begin(),a.end(),[](float z){return z==3.f;}),"empty accumulate must preserve");
    // Mapping validation happens before compute or output publication.
    std::fill(a.begin(),a.end(),42.f);
    Users bad=one;bad[0][0].first=1;
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,bad,x.data(),1,a.data(),nullptr,batch.data,batch.size),"invalid token mapping accepted");
    bad=one;bad[0][0].second=std::numeric_limits<float>::infinity();
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,bad,x.data(),1,a.data(),nullptr,batch.data,batch.size),"nonfinite weight accepted");
    Order bad_order=one_order;bad_order[0]={E,0};
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,one,x.data(),1,a.data(),&bad_order,batch.data,batch.size),"invalid order accepted");
    bad_order=one_order;bad_order[0]=bad_order[1];
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,one,x.data(),1,a.data(),&bad_order,batch.data,batch.size),"duplicate order accepted");
    bad_order=one_order;bad_order.pop_back();
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,one,x.data(),1,a.data(),&bad_order,batch.data,batch.size),"incomplete order accepted");
    Order invalid_empty{{0,0}};
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,no_experts,none,x.data(),T,a.data(),&invalid_empty,batch.data,batch.size),"invalid empty mapping accepted");
    require(std::all_of(a.begin(),a.end(),[](float z){return z==42.f;}),"invalid mapping changed output");
    const size_t need=mm3::native_cpu_batch_workspace_bytes(H,F,E);
    require(!mm3::native_cpu_run_multi_batch(lane,H,F,experts,one,x.data(),1,a.data(),&one_order,batch.data,need-1),"undersized workspace accepted");
    std::puts("native CPU batch adapter: OK");return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"native CPU batch adapter test: %s\n",e.what());return 1;}
}
