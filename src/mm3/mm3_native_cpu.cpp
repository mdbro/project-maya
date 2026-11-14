#include "mm3_native_cpu.hpp"
#include "mm3_native_experts.cuh"
#include "mm3_native_cpu_fast.hpp"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <immintrin.h>
#include <memory>
#include <chrono>
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
namespace fast=cpu_fast;
namespace {
// A worker's fixed share of an expert (L3 prefetch and its compute): worker k of W takes gate/up pairs
// [k P / W, (k + 1) P / W) and down row groups [k D / W, (k + 1) D / W).
struct Share { int p0,p1,r0,r1; };
Share share(int k,int W,int P,int D){return {k*P/W,(k+1)*P/W,k*D/W,(k+1)*D/W};}
}
size_t native_cpu_fast_workspace_bytes(int h,int ff,int entries){
    if(entries<0||entries>64||h<=0||ff<=0||h%64||ff%64)return 0;
    const size_t q=sizeof(fast::QBlock);
    return align256((size_t)entries*(h/16)*q)+align256((size_t)entries*(ff/16)*q)+align256((size_t)entries*h*2)+align256((size_t)h*2);
}
bool native_cpu_run_multi_fast(CpuLane& lane,int act,int h,int ff,const std::vector<NativeCpuExpert>& ex,
                               const std::vector<std::vector<std::pair<int,float>>>& users,const float* x,int T,float* out,
                               const std::vector<std::pair<int,int>>* order,void* mem,size_t bytes,bool accumulate,
                               const std::vector<uint8_t>* l3){
    if(act!=fast::kInt8&&act!=fast::kNvfp4)return false;
    if(users.size()!=ex.size()||!x||!out||!mem||((uintptr_t)mem%256)||T<0||h<=0||ff<=0||h%64||ff%64)return false;
    if(l3&&l3->size()!=ex.size())return false;
    if(!fast::supported())return false;
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
    const size_t need=native_cpu_fast_workspace_bytes(h,ff,(int)entries);if(!need||bytes<need)return false;
    const int HB=h/16,FB=ff/16;
    auto* p=(uint8_t*)mem;
    auto* xq=(fast::QBlock*)p;p+=align256(entries*HB*sizeof(fast::QBlock));
    auto* hq=(fast::QBlock*)p;p+=align256(entries*FB*sizeof(fast::QBlock));
    auto* saved=(uint16_t*)p;p+=align256(entries*h*2);
    auto* xb=(uint16_t*)p;
    // each token's activations quantized once, for all of its experts
    std::vector<int> slot((size_t)T,-1);int tokens=0;
    for(const auto& u:users)for(const auto& item:u)if(slot[(size_t)item.first]<0){
        const int t=item.first;slot[(size_t)t]=tokens++;
        fast::to_bf16(x+(size_t)t*h,h,xb);fast::quant((fast::Act)act,xb,h,xq+(size_t)slot[(size_t)t]*HB);
    }
    std::vector<fast::Job> jobs;jobs.reserve(entries);
    std::vector<uint8_t> fixed;fixed.reserve(entries);
    std::vector<std::vector<int>> offsets(ex.size());int cur=0;
    for(size_t e=0;e<ex.size();++e){
        if(!ex[e].blob)return false;
        auto* header=(const strata::mm3::native_experts::Header*)ex[e].blob;
        for(size_t start=0;start<users[e].size();start+=8){
            fast::Job job;job.record=ex[e].blob+sizeof(*header);job.columns=(int)std::min<size_t>(8,users[e].size()-start);
            for(int k=0;k<3;++k){float w=header->weight_global[k];if(!(w>0)||!std::isfinite(w))return false;job.multiplier[k]=1.f/w;}
            for(int c=0;c<job.columns;++c){const int t=users[e][start+c].first;
                job.x[c]=xq+(size_t)slot[(size_t)t]*HB;job.hidden[c]=hq+(size_t)cur*FB;job.y[c]=saved+(size_t)cur*h;
                offsets[e].push_back(cur);++cur;}
            jobs.push_back(job);fixed.push_back(l3&&(*l3)[e]?1:0);
        }
    }
    // Work items: the other jobs' pairs / row groups one at a time (any thread), then W tasks for the fixed shares. Each
    // share is a range of items its worker takes from the front - first, whatever task it drew - while anyone done with
    // its own work takes from the back of the shares not yet finished (a worker asleep, parked or slow): every item runs
    // once, nearly all of them by the worker that read them ahead.
    const int W=std::clamp(lane.active()-1,0,64);
    const bool any_fixed=W>0&&std::find(fixed.begin(),fixed.end(),1)!=fixed.end();
    if(!any_fixed)std::fill(fixed.begin(),fixed.end(),0);
    std::vector<int> dyn;for(size_t j=0;j<jobs.size();++j)if(!fixed[j])dyn.push_back((int)j);
    std::vector<int> fixed_jobs;for(size_t j=0;j<jobs.size();++j)if(fixed[j])fixed_jobs.push_back((int)j);
    const fast::Act a=(fast::Act)act;
    std::unique_ptr<std::atomic<uint64_t>[]> range(new std::atomic<uint64_t>[64]);   // (front << 32 | back)
    const auto phase=[&](int per_job,int P,int D,bool down){
        for(int k=0;k<W&&any_fixed;++k){const Share sh=share(k,W,P,D);const uint64_t f=down?sh.r0:sh.p0,b=down?sh.r1:sh.p1;
            range[k].store((f<<32)|b,std::memory_order_relaxed);}
        const int nd=(int)dyn.size()*per_job,tasks=nd+(any_fixed?W:0);
        const auto item=[&](int q){   // (item q of a share: that pair / row group of every fixed job)
            for(int j:fixed_jobs){if(!down)fast::gate_up_pair(a,jobs[(size_t)j],h,ff,q);else fast::down_group(jobs[(size_t)j],h,ff,q);}
        };
        const auto take=[&](int k,bool front)->int{
            uint64_t r=range[k].load(std::memory_order_acquire);
            for(;;){
                const uint32_t f=uint32_t(r>>32),b=uint32_t(r);
                if(f>=b)return -1;
                const uint64_t next=front?((uint64_t(f+1)<<32)|b):((uint64_t(f)<<32)|(b-1));
                if(range[k].compare_exchange_weak(r,next,std::memory_order_acq_rel))return int(front?f:b-1);
            }
        };
        const std::function<void(int)> f=[&](int task){
            const int own=CpuLane::thread_slot();
            if(any_fixed&&own>=1&&own<=W)for(int q;(q=take(own-1,true))>=0;)item(q);
            if(task<nd){
                const auto& job=jobs[(size_t)dyn[(size_t)(task/per_job)]];const int q=task%per_job;
                if(!down)fast::gate_up_pair(a,job,h,ff,q);else fast::down_group(job,h,ff,q);
                return;
            }
            if(own==0){   // (the caller: give the workers up to 20 us to start on their own shares)
                const auto t0=std::chrono::steady_clock::now();
                for(;;){
                    bool started=true;
                    for(int k=0;k<W&&started;++k){const Share sh=share(k,W,P,D);
                        started=uint32_t(range[k].load(std::memory_order_relaxed)>>32)!=uint32_t(down?sh.r0:sh.p0);}
                    if(started||std::chrono::steady_clock::now()-t0>std::chrono::microseconds(20))break;
                    _mm_pause();
                }
            }
            for(bool left=true;left;){   // (then the backs of the shares still open, the owner's front untouched)
                left=false;
                for(int k=0;k<W;++k){const int q=take(k,false);if(q>=0){item(q);left=true;}}
            }
        };
        lane.parallel_for(tasks,f);
    };
    phase(FB,FB,HB,false);   // (gate/up and the hidden blocks: FB pairs a job)
    phase(HB,FB,HB,true);    // (down: HB row groups a job)
    if(!accumulate)std::fill(out,out+(size_t)T*h,0.f);
    auto add=[&](size_t e,size_t u){int t=users[e][u].first;float w=users[e][u].second;const uint16_t* y=saved+(size_t)offsets[e][u]*h;
        for(int k=0;k<h;++k)out[(size_t)t*h+k]+=w*f32(y[k]);};
    if(order){for(auto [e,u]:*order)add((size_t)e,(size_t)u);}
    else for(size_t e=0;e<ex.size();++e)for(size_t u=0;u<users[e].size();++u)add(e,u);
    return true;
}
bool native_cpu_l3_begin(CpuLane& lane,int h,int ff,const std::vector<const uint8_t*>& blobs,
                         std::atomic<unsigned long long>* bytes){
    const int W=std::clamp(lane.active()-1,0,64);
    if(W<1||blobs.empty()||h<=0||ff<=0||h%64||ff%64)return false;
    for(auto* b:blobs)if(!b)return false;
    struct State { std::vector<const uint8_t*> blobs; std::unique_ptr<std::atomic<uint8_t>[]> claimed; };
    auto state=std::make_shared<State>();
    state->blobs=blobs;state->claimed.reset(new std::atomic<uint8_t>[(size_t)W]);
    for(int k=0;k<W;++k)state->claimed[k].store(0,std::memory_order_relaxed);
    const std::atomic<bool>* stop=lane.background_stop();
    const int P=ff/16,D=h/16;
    const std::function<void(int)> f=[state,stop,bytes,W,P,D,h,ff](int){
        const int own=CpuLane::thread_slot();
        if(own<1||own>W||state->claimed[own-1].exchange(1))return;   // (each worker reads its own share once)
        const Share sh=share(own-1,W,P,D);
        unsigned long long n=0;
        for(const uint8_t* blob:state->blobs){
            const uint8_t* payload=blob+sizeof(strata::mm3::native_experts::Header);
            const auto g=fast::gate_up_span(h,ff,sh.p0,sh.p1),d=fast::down_span(h,ff,sh.r0,sh.r1);
            n+=fast::touch(payload+g.offset,g.bytes,stop);
            n+=fast::touch(payload+d.offset,d.bytes,stop);
            if(stop->load(std::memory_order_relaxed))break;
        }
        if(bytes)bytes->fetch_add(n,std::memory_order_relaxed);
    };
    return lane.run_background(4*W,f);   // (more tasks than workers: every awake worker draws at least one)
}
bool native_cpu_run(CpuLane& lane,int h,int ff,const std::vector<NativeCpuExpert>& ex,const float* x,float* out,void* mem,size_t bytes){
    std::vector<std::vector<std::pair<int,float>>> users(ex.size());for(size_t i=0;i<ex.size();++i)users[i].push_back({0,ex[i].w});
    return native_cpu_run_multi(lane,h,ff,ex,users,x,1,out,nullptr,mem,bytes);
}
}
