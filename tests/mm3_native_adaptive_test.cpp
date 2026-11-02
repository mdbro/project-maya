// Literal production adaptive and demand methods with delayed, controlled CUDA.
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <cstdio>
#include <string>
#include <vector>
namespace r2=mm3::r2;
using cudaError_t=int;
struct Event { size_t through=0; };
using cudaEvent_t=Event*;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaErrorUnknown=2,cudaHostAllocPortable=0,cudaEventDisableTiming=0;
enum cudaMemcpyKind {cudaMemcpyHostToDevice=1,cudaMemcpyDeviceToHost=2};
struct Copy {void* dst;const void* src;size_t bytes;};
static std::vector<Copy> copies;
static size_t completed=0;
static int submission=0, fail_submit=0, fail_query=0, fail_join=0;
static int cudaSetDevice(int){return 0;}
static int cudaHostAlloc(uint8_t** p,size_t n,unsigned){*p=static_cast<uint8_t*>(std::malloc(n));return *p?0:cudaErrorUnknown;}
static int cudaFreeHost(void* p){std::free(p);return 0;}
static int cudaEventCreateWithFlags(cudaEvent_t* e,unsigned){*e=new Event;return 0;}
static int cudaMemcpyAsync(void* d,const void* s,size_t n,int,int){
    if(++submission==fail_submit)return cudaErrorUnknown;
    copies.push_back({d,s,n});return 0;
}
static int cudaEventRecord(cudaEvent_t e,int){
    if(++submission==fail_submit)return cudaErrorUnknown;
    e->through=copies.size();return 0;
}
static int cudaEventQuery(cudaEvent_t e){return fail_query?cudaErrorUnknown:(e->through<=completed?cudaSuccess:cudaErrorNotReady);}
static void complete_all();
static int cudaStreamSynchronize(int){if(fail_join)return cudaErrorUnknown;complete_all();return 0;}
static const char* cudaGetErrorString(int){return "injected";}
static void complete_through(size_t n){while(completed<n){auto& c=copies[completed++];std::memcpy(c.dst,c.src,c.bytes);}}
static void complete_all(){complete_through(copies.size());}
static void reset_fake(){copies.clear();completed=0;submission=fail_submit=fail_query=fail_join=0;}
#define CK(x) do {if((x)!=0){err="injected CUDA failure";return false;}}while(0)
namespace ne { struct Header {float weight_global[3],input_global[3];uint8_t pad[232];}; }
namespace nf { static size_t record_stride(int,int){return 16;} }
struct Pool {size_t n;size_t size()const{return n;}size_t capacity()const{return n;}};
struct NativeHelper {
    struct Identity {int layer,expert,slot;};
    std::array<uint32_t,4> ready(int)const{return {};}
    std::vector<Identity> records()const{return {};}
    uintptr_t pool_identity()const{return 0;}
    bool export_expert(int,int,void*,size_t,float*,float*,bool&,std::string&){return false;}
    bool replace_expert(int,int,int,int,const void*,size_t,const float*,const float*,bool&,std::string&){return false;}
};
struct Ready {uint32_t primary[4]{},helper[4]{};};
struct Layer {
    bool moe=true,hf=false;int dev=0;size_t blob=272;uint8_t* res_base=nullptr;
    std::vector<int> res_slot,res_ids;std::vector<uint8_t*> res_ptr;
    std::vector<const uint8_t*> host_blob;
    std::array<uint32_t,4> helper_ready{};unsigned long long* exp_tab=nullptr;
};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    struct AdaptiveSwap {
        long long incoming=-1,victim=-1;int landing=-1;
        uint8_t *source_host=nullptr,*source_gpu=nullptr,*landing_gpu=nullptr,*target_host=nullptr;
        cudaEvent_t event=nullptr;bool submitted=false;
    };
    int id=0,s=0,s2=0,cs=1,n_slots=8,n_pending=0;size_t slot_bytes=272;
    std::vector<int> layers{0,1};std::vector<long long> slot_owner,slot_pend,slot_used;
    std::vector<int> slot_foreground,slot_iss;std::vector<Promo> promo_q;
    std::vector<uint8_t*> slot_ptr;std::vector<uint8_t> slot_reserved;
    std::array<AdaptiveSwap,2> adaptive;uint8_t* stg=nullptr;
    Ready ready;Ready* ready_h=&ready;int originals[64]{};int* original_h=originals;
    unsigned* hf_original_d=nullptr;uint32_t* hf_snapshot_h=nullptr;
    mm3::DemandSnapshot hf_demand_seen;
};
struct Model {
    struct {int n_layers=2,n_exp=4,k_exp=4,dim=1,ff=1;} g;
    struct {bool promote=true,routing=true,fast_decode=true;int chunk=16,seqs=1,resident_cap=-1;} o;
    struct {
        long experts=8,pooled=4,adaptive_queries=0,adaptive_not_ready=0,adaptive_published=0,adaptive_cancelled_promotions=0;
        long adaptive_no_gain=0,adaptive_no_slot=0,adaptive_issued=0,adaptive_reused=0,adaptive_hits=0,adaptive_source_busy=0;
        long route_original_misses=0,route_substitutions=0,route_primary_hits=0,route_helper_hits=0,miss_union=0,promoted=0;
        unsigned long long adaptive_h2d_bytes=0,adaptive_d2h_bytes=0,miss_union_bytes=0;double adaptive_host_ms=0;
    } st;
    bool native=true,graph_on=false,vis=false,generation_active=true,hf_demand_poisoned=false;
    std::atomic<bool> retirement_uncertain{false};long long fwd=10;
    static constexpr int kWin=16,kMaxPending=4;static constexpr float kPromote=1.8f,kDecay=.95f;
    std::vector<float> use_score;std::vector<long long> use_last;
    std::vector<uint32_t> original_prof=std::vector<uint32_t>(8);
    std::vector<Layer> L=std::vector<Layer>(2);std::vector<Dev> D=std::vector<Dev>(1);
    std::vector<uint8_t> host=std::vector<uint8_t>(4*272),primary=std::vector<uint8_t>(4*272),stage_mem=std::vector<uint8_t>(8*272);
    uint8_t* pool=host.data();std::unique_ptr<Pool> native_pool=std::make_unique<Pool>(Pool{host.size()});
    std::unique_ptr<NativeHelper> helper;std::unique_ptr<int> draftm;int head_dev=0;
    std::vector<void*> owned;std::string promotion_error;
    static double now(){return 0;}
    bool helper_owns(const Layer&,int)const{return false;}
    void graphs_clear(){}
    bool foreground_usable(const Dev& d,size_t i)const{return r2::foreground_usable(d.slot_foreground[i]!=0,d.slot_used[i],fwd);}
    bool stage(const Layer& l,Dev& d,int e,int s,std::string&){std::memcpy(staging_address(d,s),l.host_blob[e],l.blob);return true;}
    bool on_cpu(const Layer& l,const Dev& d,int e)const {
        if(l.res_slot[e]>=0)return false;
        const auto key=(&l-L.data())*1024+e;
        for(int s=0;s<d.n_slots;++s)if(foreground_usable(d,s)&&d.slot_owner[s]==key)return false;
        return true;
    }
    long long helper_live_a=-1;
    size_t helper_live_blob=0;bool helper_live_enabled=false;
    uint64_t helper_live_started=0,helper_live_finished=0;
    bool helper_live_settle(std::string&){return true;}
    bool helper_live_leased(long long)const{return false;}
#include "../src/mm3/mm3_native_placement.inc"
#include "../src/mm3/mm3_native_adaptive.inc"
#include "../src/mm3/mm3_native_demand.inc"
    Model(){
        auto& d=D[0];d.stg=stage_mem.data();d.slot_owner.assign(8,-1);d.slot_pend.assign(8,-1);d.slot_used.assign(8,-1);
        owned={primary.data(),stage_mem.data()};
        d.slot_foreground.assign(8,0);d.slot_iss.assign(8,0);
        for(int li=0;li<2;++li){auto& l=L[li];l.res_slot={0,1,-1,-1};l.res_ids={0,1};l.host_blob.assign(4,nullptr);
            l.res_base=primary.data()+li*2*272;
            for(int e=0;e<2;++e)std::memset(l.res_base+e*272,li*4+e,272);
            for(int e=2;e<4;++e){auto* p=host.data()+(li*2+e-2)*272;std::memset(p,li*4+e,272);l.host_blob[e]=p;}}
        std::string err;assert(adaptive_prepare(err));primary_adaptive=true;
    }
    ~Model(){for(auto* p:adaptive_host_owned)std::free(p);for(auto& x:D[0].adaptive)delete x.event;}
    void demand(int layer,int expert,int n){score_at(layer*1024+expert,fwd,n);}
    uint8_t value(int layer,int expert)const {const auto& l=L[layer];return l.res_slot[expert]>=0?*resident_address(l,l.res_slot[expert]):*l.host_blob[expert];}
    void verify()const {
        std::vector<uintptr_t> h,v;
        for(int li=0;li<2;++li){const auto& l=L[li];for(int e=0;e<4;++e){assert((l.res_slot[e]>=0)!=(l.host_blob[e]!=nullptr));assert(value(li,e)==li*4+e);
            if(l.res_slot[e]>=0){assert(l.res_ids[l.res_slot[e]]==e);v.push_back(uintptr_t(resident_address(l,l.res_slot[e])));}else h.push_back(uintptr_t(l.host_blob[e]));}}
        auto in_gpu_allocation=[&](uintptr_t p){
            const auto in=[&](const std::vector<uint8_t>& a){const auto base=uintptr_t(a.data());return p>=base && p-base<=a.size()-272 && (p-base)%272==0;};
            return in(primary)||in(stage_mem);
        };
        for(auto p:v)assert(in_gpu_allocation(p));
        for(int s=0;s<D[0].n_slots;++s)assert(in_gpu_allocation(uintptr_t(staging_address(D[0],s))));
        assert(adaptive_storage_ok(h,v));
    }
};
static void observe(Model& m,int layer,const std::vector<int>& orig,const std::vector<int>& selected){
    auto& d=m.D[0];std::copy(orig.begin(),orig.end(),d.originals);m.begin_window_demand(m.L[layer],d,selected.data(),int(orig.size()/4));
}
int main(){
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err;m.verify();
        // Real route observation counts each original occurrence, never its substitute.
        observe(m,0,{2,0,1,3,2,0,1,3},{0,0,1,3,0,0,1,3});
        assert(m.score(2,false)==2 && m.score(0,false)==2);
        m.demand(1,2,5);m.demand(0,2,4);
        const auto old_host=m.L[0].host_blob[2];const auto old_res=m.resident_address(m.L[0],0);
        assert(m.adaptive_issue(d,err));assert(m.adaptive_pending(d)==2);assert(m.st.adaptive_issued==2);
        assert(d.adaptive[0].incoming/1024!=d.adaptive[0].victim/1024);
        assert(d.adaptive[1].incoming/1024==d.adaptive[1].victim/1024);
        for(int i=0;i<3;++i){
            ++m.fwd;assert(m.adaptive_publish(d,err));assert(m.adaptive_pending(d)==2);
            // Exercise actual single/group demand and address lookup while
            // both transactions remain unavailable; byte readers stay valid.
            std::vector<int> top=i==0?std::vector<int>{2,0,1,3}:std::vector<int>{2,0,1,3,2,0,1,3};
            observe(m,0,top,top);
            std::vector<char> busy(8);unsigned long long ptr=0;
            assert(m.locate(m.L[0],d,2,busy,ptr,err) && *reinterpret_cast<uint8_t*>(ptr)==2);
            assert(m.locate(m.L[0],d,0,busy,ptr,err) && *reinterpret_cast<uint8_t*>(ptr)==0);
            const auto queued=d.promo_q.size();m.consider_promotion(m.L[0],d,2);assert(d.promo_q.size()==queued);
            m.verify();assert(m.L[0].host_blob[2]==old_host);assert(m.resident_address(m.L[0],0)==old_res);
        }
        const auto score=m.score(2,false);
        complete_through(d.adaptive[0].event->through);assert(m.adaptive_publish(d,err));
        assert(m.adaptive_pending(d)==1);m.verify();
        complete_all();assert(m.adaptive_publish(d,err));assert(!m.adaptive_pending(d));m.verify();
        assert(m.L[0].res_slot[2]>=0 && m.score(2,false)==score);assert(m.adaptive_host_spare[0]==old_host || m.adaptive_host_spare[1]==old_host);
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err;m.demand(0,2,9);
        m.generation_active=false;assert(m.adaptive_issue(d,err));assert(!m.adaptive_pending(d));m.generation_active=true;
        // A ready staging copy avoids H2D; its old slot cannot be observed while reserved.
        std::memcpy(m.staging_address(d,3),m.L[0].host_blob[2],272);d.slot_owner[3]=2;
        assert(m.adaptive_issue(d,err));assert(m.st.adaptive_reused==1 && m.st.adaptive_h2d_bytes==0);
        assert(d.slot_owner[3]==-1 && m.staging_reserved(d,3));complete_all();assert(m.adaptive_publish(d,err));m.verify();
        assert(m.L[0].res_slot[2]>=0 && d.slot_owner[3]>=0);
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err;m.demand(0,2,9);
        const auto* old_host=m.L[0].host_blob[2];assert(m.adaptive_issue(d,err));
        const int extra=d.adaptive[0].landing==4?5:4;
        std::memcpy(m.staging_address(d,extra),old_host,272);
        d.slot_owner[extra]=2;d.slot_foreground[extra]=1;
        complete_all();assert(m.adaptive_publish(d,err));
        assert(m.adaptive_pending(d)==1 && m.st.adaptive_source_busy==1);
        assert(m.L[0].host_blob[2]==old_host && d.slot_owner[extra]==2);m.verify();
        std::string sticky;
        assert(r2::foreground_retire(d.slot_foreground,d.slot_owner,sticky,err,
               [](size_t,std::string&){return r2::Completion::ready;}));
        assert(m.adaptive_publish(d,err));assert(!m.adaptive_pending(d));
        assert(d.slot_owner[extra]==-1 && m.L[0].res_slot[2]>=0);m.verify();
    }
    reset_fake();
    {
        Model m;std::vector<uintptr_t> hosts,residents;
        for(const auto& l:m.L)for(int e=0;e<4;++e)
            if(l.res_slot[e]>=0)residents.push_back(uintptr_t(m.resident_address(l,l.res_slot[e])));
            else hosts.push_back(uintptr_t(l.host_blob[e]));
        assert(m.adaptive_storage_ok(hosts,residents));
        auto duplicated=hosts;duplicated[0]=duplicated[1];assert(!m.adaptive_storage_ok(duplicated,residents));
        duplicated=residents;duplicated[0]=duplicated[1];assert(!m.adaptive_storage_ok(hosts,duplicated));
        std::vector<uint8_t> alien(272);
        auto outside=hosts;outside[0]=uintptr_t(alien.data());assert(!m.adaptive_storage_ok(outside,residents));
        outside=residents;outside[0]=uintptr_t(alien.data());assert(!m.adaptive_storage_ok(hosts,outside));
        auto* frame=m.D[0].slot_ptr[0];m.D[0].slot_ptr[0]=alien.data();
        assert(!m.adaptive_storage_ok(hosts,residents));m.D[0].slot_ptr[0]=frame;
        auto* spare=m.adaptive_host_spare[0];m.adaptive_host_spare[0]=m.adaptive_host_spare[1];
        assert(!m.adaptive_storage_ok(hosts,residents));m.adaptive_host_spare[0]=spare;
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err,report;m.demand(0,2,9);
        assert(m.adaptive_issue(d,err));assert(m.adaptive_pending(d)==1 && completed==0);
        const int foreground=d.adaptive[0].landing==4?5:4;
        std::memcpy(m.staging_address(d,foreground),m.L[0].host_blob[2],272);
        d.slot_owner[foreground]=2;d.slot_foreground[foreground]=1;
        m.primary_adaptive=false;assert(m.adaptive_issue(d,err));assert(m.adaptive_pending(d)==1 && completed==0);
        // Read-only placement inspection reports reserved staging but must not
        // join or expose the unfinished transaction.
        assert(m.placement({},false,report,err));
        assert(m.adaptive_pending(d)==1 && completed==0 && report.find("\"staging_frames\"")!=std::string::npos);
        // The real explicit placement method joins the copy stream, publishes,
        // resets slot state, and reapplies the chosen fixed resident profile.
        std::vector<uint32_t> profile{10,9,0,0,8,7,0,0};
        assert(m.placement(profile,true,report,err));assert(!m.adaptive_pending(d));m.verify();
        assert(m.L[0].res_slot[0]>=0 && m.L[0].res_slot[1]>=0);
        assert(m.L[1].res_slot[0]>=0 && m.L[1].res_slot[1]>=0);
        assert(report.find("\"ownership_complete\":true")!=std::string::npos);
        for(int s=0;s<d.n_slots;++s)assert(!m.staging_reserved(d,s) && d.slot_owner[s]<0);
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err,report;m.demand(0,2,9);
        assert(m.adaptive_issue(d,err));fail_join=1;
        assert(!m.placement({10,9,0,0,8,7,0,0},true,report,err));
        assert(m.placement_failed && m.retirement_uncertain && m.adaptive_pending(d)==1 && completed==0);
        fail_join=0;assert(!m.adaptive_publish(d,err));
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err,report;m.demand(0,2,9);
        assert(m.adaptive_issue(d,err));fail_query=1;
        assert(!m.placement({10,9,0,0,8,7,0,0},true,report,err));
        assert(m.placement_failed && m.retirement_uncertain && m.adaptive_pending(d)==1);
        assert(completed==copies.size());fail_query=0;assert(!m.adaptive_publish(d,err));
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err;m.demand(0,2,8);m.demand(1,2,6);
        // Submitted promotion holds A's source: that identity cannot be a swap candidate.
        d.slot_pend[0]=2;d.slot_iss[0]=1;d.n_pending=1;
        assert(cudaMemcpyAsync(m.staging_address(d,0),m.L[0].host_blob[2],272,cudaMemcpyHostToDevice,d.cs)==cudaSuccess);
        assert(m.adaptive_issue(d,err));assert(!m.adaptive_identity_pending(d,2));assert(m.adaptive_identity_pending(d,1026));
        assert(d.n_pending+m.adaptive_pending(d)<=4);complete_all();assert(m.adaptive_publish(d,err));m.verify();
        std::string sticky;
        assert(r2::publish(d.slot_pend,d.slot_iss,d.slot_owner,d.slot_used,d.n_pending,m.fwd,sticky,err,
                           [](size_t,std::string&){return r2::Completion::ready;}));
        ++m.fwd;assert(m.adaptive_issue(d,err));assert(m.adaptive_identity_pending(d,2));
        assert(m.st.adaptive_reused==1);complete_all();assert(m.adaptive_publish(d,err));m.verify();
    }
    reset_fake();
    {
        Model m;auto& d=m.D[0];std::string err;m.demand(0,2,9);
        // Four queued promotions create pressure. Cancel an unissued A descriptor and preserve accounting.
        for(int s=0;s<4;++s){const int key=s==0?2:1027;d.slot_pend[s]=key;d.slot_iss[s]=0;d.promo_q.push_back({s,m.L[key/1024].host_blob[key%1024],272});}
        d.n_pending=4;assert(m.adaptive_issue(d,err));assert(m.adaptive_identity_pending(d,2));
        assert(d.n_pending+m.adaptive_pending(d)<=4 && m.st.adaptive_cancelled_promotions>=1);
        for(auto q:d.promo_q)assert(q.src!=m.L[0].host_blob[2]);
        complete_all();assert(m.adaptive_publish(d,err));m.verify();
    }
    for(int fail=1;fail<=3;++fail){reset_fake();Model m;std::string err;m.demand(0,2,9);fail_submit=fail;
        assert(!m.adaptive_issue(m.D[0],err));assert(m.placement_failed && m.retirement_uncertain);assert(m.adaptive_pending(m.D[0])==1);
        const auto count=copies.size();fail_submit=0;
        assert(!m.adaptive_issue(m.D[0],err) && !m.adaptive_publish(m.D[0],err));
        assert(copies.size()==count && m.adaptive_pending(m.D[0])==1);}
    reset_fake();
    {Model m;std::string err;m.demand(0,2,9);m.demand(1,2,8);fail_submit=5;
     assert(!m.adaptive_issue(m.D[0],err));assert(m.adaptive_pending(m.D[0])==2);
     assert(m.D[0].adaptive[0].submitted && !m.D[0].adaptive[1].submitted);
     complete_all();assert(!m.adaptive_publish(m.D[0],err));assert(m.adaptive_pending(m.D[0])==2);}
    reset_fake();
    {Model m;std::string err;m.demand(0,2,9);assert(m.adaptive_issue(m.D[0],err));fail_query=1;
     assert(!m.adaptive_publish(m.D[0],err));assert(m.placement_failed && m.retirement_uncertain && m.adaptive_pending(m.D[0])==1);
     fail_query=0;complete_all();assert(!m.adaptive_publish(m.D[0],err));assert(m.adaptive_pending(m.D[0])==1);}
}
