#include "mm3_native_cpu.hpp"
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
namespace mm3 {
namespace donor=infernix::ops::minimax_ffn;
namespace {
size_t align256(size_t x){return (x+255)&~size_t(255);}
uint16_t bf(float f){uint32_t u;std::memcpy(&u,&f,4); if((u&0x7f800000u)==0x7f800000u && (u&0x7fffffu)) return uint16_t((u>>16)|0x40u);u+=0x7fffu+((u>>16)&1u);return uint16_t(u>>16);}
float f32(uint16_t v){uint32_t u=uint32_t(v)<<16;float f;std::memcpy(&f,&u,4);return f;}
void parallel(void* context,int n,donor::WorkItem work,void* arg) {
    auto& lane=*static_cast<CpuLane*>(context); lane.parallel_for(n,[&](int i){work(arg,i);});
}
}
size_t native_cpu_workspace_bytes(int h,int ff,int entries){
    if(entries<0||h<=0||ff<=0)return 0;
    try{return 2*align256((size_t)8*h*2)+align256((size_t)entries*h*2)+donor::fast_workspace_bytes(h,ff,8);}catch(...){return 0;}
}
bool native_cpu_run_multi(CpuLane& lane,int h,int ff,const std::vector<NativeCpuExpert>& ex,
                          const std::vector<std::vector<std::pair<int,float>>>& users,const float* x,int T,float* out,
                          const std::vector<std::pair<int,int>>* order,void* mem,size_t bytes,bool accumulate) {
    if(users.size()!=ex.size()||!x||!out||!mem||((uintptr_t)mem%256)||T<0)return false;
    int entries=0;for(const auto& u:users)entries+=(int)u.size();
    size_t need=native_cpu_workspace_bytes(h,ff,entries);if(!need||bytes<need)return false;
    auto* p=(uint8_t*)mem; auto* input=(uint16_t*)p;p+=align256((size_t)8*h*2);
    auto* output=(uint16_t*)p;p+=align256((size_t)8*h*2);
    auto* saved=(uint16_t*)p;p+=align256((size_t)entries*h*2);
    std::vector<std::vector<int>> offsets(ex.size());int cur=0;
    for(size_t e=0;e<ex.size();++e) {
        if(!ex[e].blob)return false;
        auto* header=(const strata::mm3::native_experts::Header*)ex[e].blob;
        donor::FastJob job{};job.record=ex[e].blob+sizeof(*header);job.x=input;job.y=output;
        for(int k=0;k<3;++k){float w=header->weight_global[k];if(!(w>0)||!std::isfinite(w))return false;job.multiplier[k]=1.f/w;}
        for(size_t start=0;start<users[e].size();start+=8){
            int n=(int)std::min<size_t>(8,users[e].size()-start);job.columns=n;
            for(int c=0;c<n;++c){int t=users[e][start+c].first;if(t<0||t>=T)return false;
                offsets[e].push_back(cur++);for(int k=0;k<h;++k)input[(size_t)c*h+k]=bf(x[(size_t)t*h+k]);}
            donor::Executor executor{&lane,parallel};
            try {donor::fast_ffn_cpu(job,h,ff,p,&executor);}catch(const std::exception& err){std::fprintf(stderr,"native CPU expert: %s\n",err.what());return false;}
            std::memcpy(saved+(size_t)(cur-n)*h,output,(size_t)n*h*2);
        }
    }
    if(!accumulate)std::fill(out,out+(size_t)T*h,0.f);
    auto add=[&](size_t e,size_t u){int t=users[e][u].first;float w=users[e][u].second;const uint16_t* y=saved+(size_t)offsets[e][u]*h;
        for(int k=0;k<h;++k)out[(size_t)t*h+k]+=w*f32(y[k]);};
    if(order){for(auto [e,u]:*order){if(e<0||u<0||(size_t)e>=ex.size()||(size_t)u>=users[e].size())return false;add(e,u);}}
    else for(size_t e=0;e<ex.size();++e)for(size_t u=0;u<users[e].size();++u)add(e,u);
    return true;
}
size_t native_cpu_batch_workspace_bytes(int h,int ff,int entries){
    if(entries<0||entries>64||h<=0||ff<=0)return 0;
    try{return 2*align256((size_t)entries*h*2)+(size_t)entries*donor::fast_workspace_bytes(h,ff,1);}
    catch(...){return 0;}
}
bool native_cpu_run_multi_batch(CpuLane& lane,int h,int ff,const std::vector<NativeCpuExpert>& ex,
                                const std::vector<std::vector<std::pair<int,float>>>& users,const float* x,int T,float* out,
                                const std::vector<std::pair<int,int>>* order,void* mem,size_t bytes,bool accumulate){
    if(users.size()!=ex.size()||!x||!out||!mem||((uintptr_t)mem%256)||T<0||h<=0||ff<=0)return false;
    size_t entries=0;
    std::vector<size_t> base(ex.size()+1,0);
    for(size_t e=0;e<users.size();++e){
        const auto& u=users[e];if(u.size()>64-entries)return false;
        for(const auto& item:u)if(item.first<0||item.first>=T||!std::isfinite(item.second))return false;
        entries+=u.size();base[e+1]=entries;
    }
    if(order){
        if(order->size()!=entries)return false;
        std::vector<uint8_t> seen(entries,0);
        for(const auto& item:*order){
            const int e=item.first,u=item.second;
            if(e<0||u<0||(size_t)e>=users.size()||(size_t)u>=users[(size_t)e].size())return false;
            const size_t index=base[(size_t)e]+(size_t)u;
            if(seen[index])return false;
            seen[index]=1;
        }
    }
    if(!entries){if(!accumulate)std::fill(out,out+(size_t)T*h,0.f);return true;}
    const size_t need=native_cpu_batch_workspace_bytes(h,ff,(int)entries);if(!need||bytes<need)return false;
    auto* p=(uint8_t*)mem;auto* input=(uint16_t*)p;p+=align256(entries*h*2);
    auto* saved=(uint16_t*)p;p+=align256(entries*h*2);
    std::vector<donor::FastJob> jobs;jobs.reserve(entries);
    std::vector<std::vector<int>> offsets(ex.size());int cur=0;
    for(size_t e=0;e<ex.size();++e){
        if(!ex[e].blob)return false;
        auto* header=(const strata::mm3::native_experts::Header*)ex[e].blob;
        for(size_t start=0;start<users[e].size();start+=8){
            donor::FastJob job{};job.record=ex[e].blob+sizeof(*header);job.columns=(int)std::min<size_t>(8,users[e].size()-start);
            job.x=input+(size_t)cur*h;job.y=saved+(size_t)cur*h;
            for(int k=0;k<3;++k){float w=header->weight_global[k];if(!(w>0)||!std::isfinite(w))return false;job.multiplier[k]=1.f/w;}
            for(int c=0;c<job.columns;++c){int t=users[e][start+c].first;if(t<0||t>=T)return false;
                offsets[e].push_back(cur);for(int k=0;k<h;++k)input[(size_t)cur*h+k]=bf(x[(size_t)t*h+k]);++cur;}
            jobs.push_back(job);
        }
    }
    donor::Executor executor{&lane,parallel};
    try{donor::fast_ffn_cpu_batch(jobs.data(),(int)jobs.size(),h,ff,p,bytes-(size_t)(p-(uint8_t*)mem),&executor);}
    catch(const std::exception& err){std::fprintf(stderr,"native CPU batch: %s\n",err.what());return false;}
    if(!accumulate)std::fill(out,out+(size_t)T*h,0.f);
    auto add=[&](size_t e,size_t u){int t=users[e][u].first;float w=users[e][u].second;const uint16_t* y=saved+(size_t)offsets[e][u]*h;
        for(int k=0;k<h;++k)out[(size_t)t*h+k]+=w*f32(y[k]);};
    if(order){for(auto [e,u]:*order){if(e<0||u<0||(size_t)e>=ex.size()||(size_t)u>=users[e].size())return false;add(e,u);}}
    else for(size_t e=0;e<ex.size();++e)for(size_t u=0;u<users[e].size();++u)add(e,u);
    return true;
}
bool native_cpu_run(CpuLane& lane,int h,int ff,const std::vector<NativeCpuExpert>& ex,const float* x,float* out,void* mem,size_t bytes){
    std::vector<std::vector<std::pair<int,float>>> users(ex.size());for(size_t i=0;i<ex.size();++i)users[i].push_back({0,ex[i].w});
    return native_cpu_run_multi(lane,h,ff,ex,users,x,1,out,nullptr,mem,bytes);
}
}
