#include "mm3_capture_stub.hpp"
// Compile the actual production owner_window and demand methods against a
// deterministic queued-stream CUDA substitute. Host copies execute only on
// drain, so reusing pinned plan storage before retirement breaks the test.
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include "../src/mm3/mm3_native_demand_units.hpp"
#include "../src/mm3/mm3_residency_policy.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>
constexpr int kWin=16,cudaSuccess=0,cudaErrorNotReady=2,cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2;
struct Event {bool ready=false,failed=false;}; using cudaEvent_t=Event*;
using cudaError_t=int;
static std::vector<std::function<void()>> queue;
static int submission=0,fail_submission=0,fail_join=0,event_queries=0;
static int cudaMemcpyAsync(void* dst,const void* src,size_t n,int,int) {
    if(++submission==fail_submission)return 1;
    queue.emplace_back([=]{std::memcpy(dst,src,n);});return 0;
}
static int cudaSetDevice(int){return 0;}
static int cudaStreamSynchronize(int){if(fail_join)return 1;for(auto& f:queue)f();queue.clear();return 0;}
static int cudaEventQuery(Event* e){++event_queries;return e->failed?1:e->ready?0:cudaErrorNotReady;}
static const char* cudaGetErrorString(int){return "injected";}
// early_copy's reader wait (f666cf0); no frame here holds an early copy.
static int early_copy_reader_waits=0;
static int cudaStreamWaitEvent(int,Event*,unsigned){++early_copy_reader_waits;return 0;}
#define CK(x) do {if((x)!=0){err="injected";return false;}} while(0)
namespace r2=mm3::r2;
namespace x3 {struct WinArgs {
    const unsigned long long *grp_ptr=nullptr,*plan_ptr=nullptr;
    const int *grp_start=nullptr,*grp_ent=nullptr,*n_groups=nullptr;
    const float* plan_w=nullptr;float *part=nullptr,*out=nullptr,*sh_out=nullptr,*x=nullptr;
    int T=0,k=0,n_embd=1,n_ff=1;float limit=7;
};}
namespace ne {
static bool window_scatter(x3::WinArgs a,int pieces,int) {
    if(++submission==fail_submission)return false;
    queue.emplace_back([=] {
        int got=0;
        for(int g=0;g<*a.n_groups;++g) {
            const int n=a.grp_start[g+1]-a.grp_start[g];got+=(n+7)/8;
            const float id=*(const float*)(uintptr_t)a.grp_ptr[g];
            for(int j=a.grp_start[g];j<a.grp_start[g+1];++j) {
                const int e=a.grp_ent[j];a.part[e]=id+float(e/8)*.125f;
            }
        }
        assert(got==pieces);
    });return true;
}
static bool window_combine(x3::WinArgs a,int) {
    if(++submission==fail_submission)return false;
    queue.emplace_back([=]{for(int t=0;t<a.T;++t){float v=a.sh_out[t];for(int k=0;k<a.k;++k){int e=t*8+k;if(a.plan_ptr[e])v+=a.plan_w[e]*a.part[e];}a.out[t]=v;}});return true;
}}
namespace k {static void add(float* h,const float* out,size_t n,int){queue.emplace_back([=]{for(size_t i=0;i<n;++i)h[i]+=out[i];});}}
struct Ready {uint32_t primary[4]{},helper[4]{};};
struct Layer {std::vector<int> res_slot=std::vector<int>(128,-1);std::vector<const uint8_t*> host_blob=std::vector<const uint8_t*>(128);size_t blob=sizeof(float);};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,s=1,n_slots=8,n_pending=0;size_t slot_bytes=sizeof(float);
    Event events[8]{};std::array<Event*,8> slot_ev{{&events[0],&events[1],&events[2],&events[3],&events[4],&events[5],&events[6],&events[7]}};
    std::vector<uint8_t> slot_prediction;std::array<bool,8> reserved{};
    std::vector<long long> slot_owner=std::vector<long long>(8,-1),slot_pend=std::vector<long long>(8,-1),slot_used=std::vector<long long>(8,-1);
    std::vector<int> slot_foreground=std::vector<int>(8),slot_iss=std::vector<int>(8);
    std::vector<uint8_t> slot_early,slot_early_fresh;   // (early_copy, f666cf0: empty while it is off)
    std::vector<Promo> promo_q;float frames[8]{},h[16]{},out[16]{},part[128]{},shared[16]{};
    std::vector<uint8_t> hostplans=std::vector<uint8_t>(65*4096),devplan=std::vector<uint8_t>(4096);
    uint8_t* owner_plan_h=hostplans.data(),*win_plan_d=devplan.data();
    Ready ready{},*ready_h=&ready;int originals[64]{},*original_h=originals;
    unsigned* hf_original_d=nullptr;uint32_t* hf_snapshot_h=nullptr;mm3::DemandSnapshot hf_demand_seen;
};
struct Owner {
    struct {int n_layers=1,n_exp=128,k_exp=4,dim=1;float scale=1.25f;} g;
    struct {bool promote=true,routing=false;} o;
    struct {long owner_expert_groups=0,owner_passes=0,profiled=0,route_original_misses=0,route_substitutions=0,route_primary_hits=0,route_helper_hits=0,miss_union=0,promoted=0,adaptive_hits=0,promo_queries=0,promo_not_ready=0;double t_promo=0;uint64_t miss_union_bytes=0;long early_copy_waits=0,early_copy_used=0;} st;
    bool primary_adaptive=false;mm3::PrimaryResidencyPolicy adaptive_policy;
    bool native=true,hf_demand_poisoned=false,placement_failed=false,generation_active=true;long long fwd=10,native_demand_rows=5;int native_demand_block_rows=5;static constexpr float kDecay=.95f,kPromote=1.8f;static constexpr int kMaxPending=4,kPExperts=0;
    std::atomic<bool> retirement_uncertain{false};std::string promotion_error;
    std::vector<Layer>L=std::vector<Layer>(1);std::vector<Dev>D;
    std::vector<float>use_score;std::vector<long long>use_last;std::vector<uint32_t>original_prof=std::vector<uint32_t>(128),prof=std::vector<uint32_t>(128);
    std::vector<uint8_t>adaptive_main;std::array<float,128> hosts{};float resident=100;
    Owner(){for(int e=0;e<128;++e){hosts[e]=float(e+1);L[0].host_blob[e]=(uint8_t*)&hosts[e];}}
    size_t win_plan_bytes()const{return 4096;}
    uint8_t* resident_address(const Layer&,int){return (uint8_t*)&resident;}
    uint8_t* staging_address(Dev& d,int s){return (uint8_t*)&d.frames[s];}
    bool staging_reserved(const Dev& d,int s)const{return d.reserved[s];}
    bool foreground_usable(const Dev& d,size_t s)const{return r2::foreground_usable(d.slot_foreground[s]!=0,d.slot_used[s],fwd);}
    bool retire_foreground_slots(Dev& d,std::string& err){return r2::foreground_retire(d.slot_foreground,d.slot_owner,promotion_error,err,
        [&](size_t s,std::string& error){int v=cudaEventQuery(d.slot_ev[s]);if(v==cudaSuccess)return r2::Completion::ready;if(v==cudaErrorNotReady)return r2::Completion::not_ready;error="injected foreground query";return r2::Completion::failed;});}
    bool stage(const Layer& l,Dev& d,int e,int slot,std::string& err){
        return r2::foreground_issue(d.slot_owner[slot],d.slot_foreground[slot],promotion_error,err,[&](std::string& error){
            int rc=cudaMemcpyAsync(staging_address(d,slot),l.host_blob[e],l.blob,cudaMemcpyHostToDevice,d.s);
            if(rc==0 && ++submission==fail_submission)rc=1; // failed event after accepted copy
            if(rc){error="injected foreground copy/event";return false;}return true;
        });
    }
    bool on_cpu(const Layer& l,const Dev& d,int e)const {if(l.res_slot[e]>=0)return false;for(int s=0;s<8;++s)if(foreground_usable(d,s)&&d.slot_owner[s]==e)return false;return true;}
    bool adaptive_identity_pending(const Dev&,long long)const{return false;}
    bool helper_live_leased(long long)const{return false;}
    int adaptive_pending(const Dev&)const{return 0;}
    mutable int consideration_checks=0;
    bool prefetch_active(bool)const{++consideration_checks;return false;}
    static double now(){return 0;}
    void prefetch_note_selection(const Layer&,Dev&,const int*,int){}
    void prefetch_note_use(Dev&,int,size_t){}
    void prefetch_forget(Dev&,int){}
    void pmark(Dev&,int){}
    void ablate(const Dev&,float*,int) const {}   // (--ablate, mm3_model.cu: no direction loaded, nothing launched)
    double demand_decay=0.95;   // (residency_decay, 1b3ba4c: production default 0.95)
    const bool early_copy_verify=false;   // (MM3_EARLY_COPY_VERIFY unset)
    int early_copy_checks=0;
    bool early_copy_check(const Layer&,Dev&,int,int,std::string&){++early_copy_checks;return true;}
    // production's slot_early_busy (mm3_native_prefetch.inc, f666cf0), which this fixture does not include
    bool slot_early_busy(const Dev& d,int s) const {return !d.slot_early.empty() && d.slot_early[size_t(s)];}
#include "../src/mm3/mm3_native_demand.inc"
    r2::Completion optional_status=r2::Completion::ready; int optional_retires=0,foreground_resets=0;
    bool compressed_host()const{return false;}
    r2::Completion scale_optional_query(size_t,std::string&){return optional_status;}
    bool scale_optional_retire(size_t,std::string&){++optional_retires;return true;}
    bool scale_foreground_joined(Dev&,std::string&){++foreground_resets;return true;}
#include "../src/mm3/mm3_native_slot_publish.inc"
#include "../src/mm3/mm3_native_owner_window.inc"
};
static void run(int T,bool routing) {
    queue.clear();submission=fail_submission=0;Owner m;Dev d;std::string err;m.o.routing=routing;
    m.L[0].res_slot[0]=0; d.slot_owner[0]=1;d.frames[0]=m.hosts[1];
    // Four issued optional copies remain unavailable and retain their frames.
    m.o.promote=false;
    for(int i=4;i<8;++i){d.slot_pend[i]=50+i;d.slot_iss[i]=1;d.frames[i]=-999;}
    d.n_pending=4;event_queries=0;
    assert(m.publish_slots(d,err));assert(event_queries==4 && d.n_pending==4);
    std::vector<int> top(T*4),orig(T*4);std::vector<float>w(T*4);
    for(int t=0;t<T;++t)for(int k=0;k<4;++k){int i=t*4+k;top[i]=k==0?0:(k==1?1:2+(i%30));orig[i]=routing&&k==1?100:top[i];w[i]=.11f+.03f*k;d.original_h[i]=orig[i];}
    d.ready.primary[0]=3;
    const size_t ow=kWin*8*8,ogp=ow+kWin*8*4,ogs=ogp+kWin*8*8,oge=ogs+(kWin*8+1)*4,ong=oge+kWin*8*4;
    x3::WinArgs a{};a.grp_ptr=(unsigned long long*)(d.win_plan_d+ogp);a.grp_start=(int*)(d.win_plan_d+ogs);a.grp_ent=(int*)(d.win_plan_d+oge);a.n_groups=(int*)(d.win_plan_d+ong);a.plan_ptr=(unsigned long long*)d.win_plan_d;a.plan_w=(float*)(d.win_plan_d+ow);a.part=d.part;a.out=d.out;a.sh_out=d.shared;a.T=T;a.k=4;
    for(int t=0;t<T;++t)d.shared[t]=2;
    m.begin_window_demand(m.L[0],d,top.data(),T);
    assert(m.owner_window(m.L[0],d,T,top.data(),w.data(),a,err));
    for(int i=4;i<8;++i)assert(d.slot_pend[i]==50+i && d.frames[i]==-999);
    assert(m.original_prof[100]==(routing?unsigned(T):0));
    assert(m.st.owner_expert_groups>0 && m.st.owner_passes>0);
    if(T>=8)assert(m.st.owner_expert_groups>8 && m.st.owner_passes>=3);
    // DMA plans and staged frames are consumed in order only now.
    assert(cudaStreamSynchronize(1)==0);
    for(int t=0;t<T;++t){float expected=2;for(int k=0;k<4;++k){int e=top[t*4+k];float val=e==0?100:m.hosts[e];expected+=w[t*4+k]*m.g.scale*(val+t*.125f);}assert(std::fabs(d.h[t]-expected)<1e-4f);}
    for(int i=4;i<8;++i){assert(d.frames[i]==-999);d.events[i].ready=true;}
    assert(m.publish_slots(d,err));assert(d.n_pending==0);
    for(int i=4;i<8;++i)assert(d.slot_owner[i]==50+i && d.slot_pend[i]<0);
    for(int e=0;e<128;++e){unsigned n=std::count(orig.begin(),orig.end(),e);assert(m.original_prof[e]==n);}
}
static x3::WinArgs args(Dev& d,int T) {
    const size_t ow=kWin*8*8,ogp=ow+kWin*8*4,ogs=ogp+kWin*8*8,oge=ogs+(kWin*8+1)*4,ong=oge+kWin*8*4;
    x3::WinArgs a{};a.grp_ptr=(unsigned long long*)(d.win_plan_d+ogp);a.grp_start=(int*)(d.win_plan_d+ogs);a.grp_ent=(int*)(d.win_plan_d+oge);a.n_groups=(int*)(d.win_plan_d+ong);a.plan_ptr=(unsigned long long*)d.win_plan_d;a.plan_w=(float*)(d.win_plan_d+ow);a.part=d.part;a.out=d.out;a.sh_out=d.shared;a.T=T;a.k=4;return a;
}
int main(){for(int t:{1,8,16})for(bool routing:{false,true})run(t,routing);
    // Exercise the production publication bridge: a queued/running decoder
    // cannot be mistaken for a completed event from an earlier use of a frame.
    {Owner m;Dev d;std::string err;d.slot_pend[0]=42;d.slot_iss[0]=1;d.n_pending=1;
     d.events[0].ready=true;event_queries=0;m.optional_status=r2::Completion::not_ready;
     assert(m.publish_slots(d,err));assert(event_queries==0 && d.n_pending==1 && m.optional_retires==0);
     m.optional_status=r2::Completion::ready;assert(m.publish_slots(d,err));
     assert(event_queries==1 && d.n_pending==0 && d.slot_owner[0]==42 && m.optional_retires==1);
     assert(m.owner_window_join(d,err));assert(m.foreground_resets==1);
    }
    // Selected hot cache frame must survive while a colder unselected frame
    // remains available; this invokes the actual production locate LRU.
    {queue.clear();submission=fail_submission=0;Owner m;Dev d;std::string err;
        for(int i=2;i<8;++i)d.reserved[i]=true;
        d.slot_owner[0]=1;d.frames[0]=m.hosts[1];d.slot_used[0]=9;
        d.slot_owner[1]=77;d.frames[1]=m.hosts[77];d.slot_used[1]=-100;
        int top[4]={1,2,2,2};float w[4]={1,1,1,1};std::copy_n(top,4,d.original_h);
        m.begin_window_demand(m.L[0],d,top,1);assert(m.owner_window(m.L[0],d,1,top,w,args(d,1),err));
        assert(d.slot_owner[0]==1 && d.slot_owner[1]==2);assert(cudaStreamSynchronize(1)==0);
    }
    // A repeatedly demanded HOST identity is displaced by later passes. It is
    // considered once for ordinary admission using its original demand; all
    // active frames are still protected, so no optional fill is forced.
    {queue.clear();submission=fail_submission=0;Owner m;Dev d;std::string err;
        int top[32];float w[32];for(int i=0;i<32;++i){top[i]=i<4?2:3+i;w[i]=.25f;d.original_h[i]=top[i];}
        m.begin_window_demand(m.L[0],d,top,8);assert(m.owner_window(m.L[0],d,8,top,w,args(d,8),err));
        assert(m.original_prof[2]==4 && m.score(2,false)==4);
        assert(std::find(d.slot_owner.begin(),d.slot_owner.end(),2)==d.slot_owner.end());
        assert(m.consideration_checks==37 && d.promo_q.empty());
        assert(cudaStreamSynchronize(1)==0);
    }
    // Foreground publication uses the production lifecycle: an incomplete
    // event retains the frame and ownership; only a later successful query
    // makes it generally usable. A query failure poisons the lifecycle.
    {Owner m;Dev d;std::string err;d.slot_owner[0]=4;d.slot_foreground[0]=1;d.slot_used[0]=2;
        assert(m.retire_foreground_slots(d,err));assert(d.slot_foreground[0] && !m.foreground_usable(d,0));
        d.events[0].ready=true;assert(m.retire_foreground_slots(d,err));assert(!d.slot_foreground[0] && m.foreground_usable(d,0));
        d.slot_foreground[0]=1;d.events[0].failed=true;assert(!m.retire_foreground_slots(d,err));
        assert(d.slot_owner[0]==-1 && d.slot_foreground[0] && !m.promotion_error.empty());
    }
    // Failed real owner join must not retire readers or drain queued work.
    {queue.clear();submission=fail_submission=0;Owner m;Dev d;std::string err;int observed=0;
        d.slot_foreground[0]=1;d.slot_owner[0]=9;d.slot_used[0]=3;
        queue.emplace_back([&]{observed=1;});fail_join=1;
        assert(!m.owner_window_join(d,err));assert(m.placement_failed && m.retirement_uncertain.load());
        assert(d.slot_foreground[0]==1 && d.slot_owner[0]==9 && d.slot_used[0]==3 && observed==0 && queue.size()==1);
        fail_join=0;queue.clear();
    }
    // Inject each copy, event, plan upload, scatter and combine failure. No
    // failed destination is published; possibly submitted work retains ownership.
    for(int fault=1;fault<=11;++fault){queue.clear();submission=0;fail_submission=fault;Owner m;Dev d;std::string err;int top[4]={4,5,6,7};float w[4]={1,1,1,1};std::copy_n(top,4,d.original_h);x3::WinArgs a{};
        m.begin_window_demand(m.L[0],d,top,1);assert(!m.owner_window(m.L[0],d,1,top,w,a,err));assert(m.retirement_uncertain.load());if(fault<=8){assert(d.slot_owner[(fault-1)/2]==-1);assert(!m.promotion_error.empty());}}
    queue.clear();
}
