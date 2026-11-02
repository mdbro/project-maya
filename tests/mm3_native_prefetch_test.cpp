// Exercises the literal production prefetch and publication methods with
// controlled copy completion and failures. No GPU or duplicate policy.
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace r2=mm3::r2;
constexpr int kWin=16, cudaSuccess=0, cudaErrorNotReady=1, cudaHostAllocPortable=0,
              cudaEventDisableTiming=0, cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
using cudaError_t=int;using cudaStream_t=int;
struct Event { bool ready=false,fail=false; };using cudaEvent_t=Event*;
static int fail_step=0, step=0, event_queries=0, fail_join=0;
static int cudaSetDevice(int){return 0;}
static int cudaHostAlloc(uint8_t** p,size_t n,unsigned){*p=(uint8_t*)std::malloc(n);return *p?0:2;}
static int cudaEventCreateWithFlags(Event** e,unsigned){*e=new Event;return 0;}
static int cudaEventRecord(Event* e,int){if(++step==fail_step)return 2;e->ready=false;return 0;}
static int cudaStreamWaitEvent(int,Event*,unsigned){return ++step==fail_step?2:0;}
static int cudaStreamSynchronize(int){return fail_join?2:0;}
static int cudaPeekAtLastError(){return 0;}
static int cudaMemcpyAsync(void* dst,const void* src,size_t n,int kind,int){
    if(kind==cudaMemcpyHostToDevice && ++step==fail_step)return 2;
    std::memcpy(dst,src,n);return 0;
}
static int cudaEventQuery(Event* e){++event_queries;return e->fail?2:e->ready?0:cudaErrorNotReady;}
static const char* cudaGetErrorString(int){return "injected";}
#define CK(x) do {if((x)!=0){err="injected CUDA failure";return false;}}while(0)
namespace k { static void rows_dot(const float*,int,int,const float*,int,float*,int){} }
namespace resident {
    struct Ready {uint32_t primary[4]{},helper[4]{};};enum class Policy {Off};
    static int route(const float*,const float*,int,Policy,float,const Ready*,int*,float*,int*,void*,int){return 0;}
}
struct Layer {
    bool moe=true,hf=false;int dev=0;size_t blob=8;float* gate=nullptr;float* bias=nullptr;
    std::vector<int> res_slot=std::vector<int>(128,-1);std::vector<const uint8_t*> host_blob=std::vector<const uint8_t*>(128,nullptr);
};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,s=1,s2=3,cs=2,n_slots=8,n_pending=0;size_t slot_bytes=8;
    std::vector<long long> slot_owner=std::vector<long long>(8,-1),slot_pend=std::vector<long long>(8,-1),slot_used=std::vector<long long>(8,-1);
    std::vector<uint8_t> slot_iss=std::vector<uint8_t>(8),slot_foreground=std::vector<uint8_t>(8),slot_prediction=std::vector<uint8_t>(8);
    std::vector<Promo> promo_q;std::array<Event,8> events{};std::vector<Event*> slot_ev;
    Event fork_event{},join_event{};Event* fev=&fork_event;Event* jev=&join_event;
    float* prediction_logits=nullptr;int* prediction_ids=nullptr;float* prediction_weights=nullptr;
    uint8_t* prediction_host=nullptr;Event* prediction_uploads=nullptr;uint8_t stage[64]{};float* xn=nullptr;
    resident::Ready ready{},*ready_h=&ready;int originals[64]{},*original_h=originals;
    unsigned* hf_original_d=nullptr;uint32_t* hf_snapshot_h=nullptr;mm3::DemandSnapshot hf_demand_seen;
    Dev(){for(auto& e:events)slot_ev.push_back(&e);}
};
struct Model {
    struct {int n_layers=2,n_exp=128,k_exp=4,dim=1;} g;
    struct {int seqs=1,tp=0,chunk=16;bool fast_decode=true,promote=true,routing=true,routing_prefetch=true;} o;
    struct {long prefetch_predictions=0,prefetch_rows=0,prefetch_admitted=0,prefetch_cancelled=0,prefetch_useful=0,prefetch_unused=0,prefetch_late=0,promo_queries=0,promo_not_ready=0;
        long route_original_misses=0,route_substitutions=0,route_primary_hits=0,route_helper_hits=0,miss_union=0,promoted=0,adaptive_hits=0;
        uint64_t prefetch_issued_bytes=0,prefetch_useful_bytes=0,prefetch_unused_bytes=0,prefetch_late_bytes=0,miss_union_bytes=0;
        double prefetch_policy_ms=0,t_promo=0;} st;
    bool native=true,graph_on=false,primary_adaptive=false,generation_active=true,vis=false,placement_failed=false;
    std::atomic<bool> retirement_uncertain{false};
    int native_pool=1,helper=1;std::string helper_dispatch_current="worker",promotion_error;
    long long fwd=10,lease=-1,adaptive=-1;static constexpr int kMaxPending=4;
    static constexpr float kDecay=.95f,kPromote=1.8f;
    std::vector<int> rseq;std::vector<Layer> L=std::vector<Layer>(2);std::vector<Dev> D=std::vector<Dev>(1);
    std::vector<float> use_score;std::vector<long long> use_last;std::vector<uint32_t> original_prof=std::vector<uint32_t>(256);
    std::vector<uint8_t> adaptive_main;bool hf_demand_poisoned=false;
    std::array<std::array<uint8_t,8>,128> hosts{};std::vector<void*> allocations;
    Model(){for(int e=0;e<128;++e)L[1].host_blob[e]=hosts[e].data();}
    ~Model(){for(void* p:allocations)std::free(p);auto& d=D[0];std::free(d.prediction_host);delete d.prediction_uploads;}
    template<class T> bool dalloc(int,T*& p,size_t n,std::string&){p=(T*)std::calloc(n,sizeof(T));allocations.push_back(p);return p;}
    bool staging_reserved(const Dev&,int)const{return false;}
    uint8_t* staging_address(Dev& d,int s){return d.stage+s*d.slot_bytes;}
    uint8_t* resident_address(const Layer&,int){return nullptr;}
    bool foreground_usable(const Dev& d,size_t s)const{return r2::foreground_usable(d.slot_foreground[s]!=0,d.slot_used[s],fwd);}
    bool stage(const Layer& l,Dev& d,int e,int s,std::string&){std::memcpy(staging_address(d,s),l.host_blob[e],l.blob);return true;}
    bool on_cpu(const Layer& l,const Dev& d,int e)const {
        if(helper_owns(l,e) || l.res_slot[e]>=0)return false;
        const long long key=(&l-L.data())*1024+e;
        for(int s=0;s<d.n_slots;++s)if(foreground_usable(d,s) && d.slot_owner[s]==key)return false;
        return true;
    }
    bool helper_owns(const Layer&,int e)const{return e==1;}
    bool helper_live_leased(long long k)const{return k==lease;}
    bool adaptive_identity_pending(const Dev&,long long k)const{return k==adaptive;}
    int adaptive_pending(const Dev&)const{return adaptive>=0;}
    cudaStream_t fork(Dev& d){return d.s2;}
    void join(Dev&){}
    static double now(){return 0;}
#include "../src/mm3/mm3_native_demand.inc"
#include "../src/mm3/mm3_native_prefetch.inc"
    bool compressed_host()const{return false;}
    r2::Completion scale_optional_query(size_t,std::string&){return r2::Completion::ready;}
    bool scale_optional_retire(size_t,std::string&){return true;}
    bool scale_foreground_joined(std::string&){return true;}
#include "../src/mm3/mm3_native_slot_publish.inc"
    void prediction(const std::vector<int>& ids,const std::vector<float>& weights) {
        auto& d=D[0];std::memcpy(d.prediction_host,ids.data(),ids.size()*4);
        std::memcpy(d.prediction_host+kWin*4*4,weights.data(),weights.size()*4);
    }
};
int main(){
    Model m;auto& d=m.D[0];std::string err;assert(m.prefetch_prepare(err));
    // Whole-group weighted support chooses 7, despite each row individually
    // preferring different experts. The helper-owned expert is ineligible.
    m.prediction({2,7,1,3, 4,7,1,5},{.40f,.30f,.20f,.10f, .40f,.30f,.20f,.10f});
    d.slot_used.assign(8,10);d.slot_foreground[4]=1;
    assert(m.prefetch_join_readers(d,err));
    assert(!d.slot_foreground[4] && d.slot_used[4]==9);
    std::vector<char> busy(8);busy[4]=1;d.slot_used[5]=10; // current-reader protections
    m.prefetch_admit(m.L[0],d,2,busy);
    assert(d.promo_q.size()==1 && d.slot_pend[6]==1024+7 && d.n_pending==1);
    assert(d.slot_owner[6]==-1 && d.slot_prediction[6]==1);
    assert(m.prefetch_issue_one(d,err));assert(d.slot_iss[6] && d.promo_q.empty());
    assert(m.st.prefetch_issued_bytes==8);
    // Publication is query-only: unresolved data stays unavailable and no
    // optional wait or extra submission is made. It becomes ready only after
    // a later successful query, before constructing the next snapshot.
    const int submitted=step;assert(m.publish_slots(d,err));
    assert(d.slot_owner[6]<0 && d.slot_pend[6]>=0 && step==submitted);
    int selected[]={7,0,2,3};m.prefetch_note_selection(m.L[1],d,selected,1);
    assert(m.st.prefetch_late==1 && d.slot_prediction[6]==4);
    d.events[6].ready=true;assert(m.publish_slots(d,err));
    assert(d.slot_owner[6]==1024+7 && d.slot_pend[6]<0 && d.n_pending==0);
    m.prefetch_note_selection(m.L[1],d,selected,1);assert(m.st.prefetch_useful==0);
    assert(m.score(1024+7,false)==0); // prediction and publication grant no demand credit
    std::copy(selected,selected+4,d.original_h);d.ready.primary[0]|=1u<<7;
    m.begin_window_demand(m.L[1],d,selected,1);assert(m.score(1024+7,false)==1);
    unsigned long long ptr=0;std::vector<char> actual_busy(8);
    assert(m.locate(m.L[1],d,7,actual_busy,ptr,err));assert(m.st.prefetch_useful==1);
    // A stale unissued descriptor is released exactly; helper/adaptive leases
    // and all current readers stay protected during prediction admission.
    d.slot_owner[6]=-1;d.slot_prediction[6]=0;d.slot_pend[7]=1030;d.slot_iss[7]=0;d.n_pending=1;
    d.promo_q.push_back({7,m.hosts[6].data(),8});m.lease=1024+7;
    m.prefetch_admit(m.L[0],d,2,busy);
    assert(m.st.prefetch_cancelled==1 && d.n_pending==1 && d.promo_q.size()==1);
    assert(d.slot_pend[d.promo_q[0].slot]!=1024+7);
    // An issued copy blocks all new optional submissions, even if the queue
    // contains an unissued descriptor. It is never waited on.
    d.slot_pend[4]=1040;d.slot_iss[4]=1;++d.n_pending;
    const int before=step;assert(m.prefetch_issue_one(d,err));assert(step==before && d.promo_q.size()==1);
    // Failed optional submission poisons the lifecycle and retains the source
    // and frame even when failure follows partial submission.
    d.slot_pend[4]=-1;d.slot_iss[4]=0;--d.n_pending;fail_step=step+3;
    const int slot=d.promo_q[0].slot;assert(!m.prefetch_issue_one(d,err));
    assert(!m.promotion_error.empty() && d.slot_pend[slot]>=0 && d.slot_owner[slot]<0);
    // Every operation in the fence/copy/event chain can fail, including the
    // final event after a DMA may have begun. The source and destination stay
    // reserved and the lifecycle remains poisoned in every case.
    for(int fault=1;fault<=4;++fault) {
        step=0;fail_step=0;Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        x.prediction({9,8,7,6},{.4f,.3f,.2f,.1f});std::vector<char> free(8);
        x.prefetch_admit(x.L[0],a,1,free);assert(a.promo_q.size()==1);
        const int s=a.promo_q[0].slot;fail_step=fault;
        assert(!x.prefetch_issue_one(a,e));assert(!x.promotion_error.empty());
        assert(a.slot_pend[s]==1024+9 && a.slot_owner[s]<0 && a.n_pending==1);
        assert(!x.publish_slots(a,e));
    }
    // A failed completion query cannot publish or release a pending frame.
    {
        step=fail_step=0;Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        x.prediction({9,8,7,6},{.4f,.3f,.2f,.1f});std::vector<char> free(8);
        x.prefetch_admit(x.L[0],a,1,free);const int s=a.promo_q[0].slot;
        assert(x.prefetch_issue_one(a,e));a.events[s].fail=true;
        assert(!x.publish_slots(a,e));assert(!x.promotion_error.empty());
        assert(a.slot_pend[s]==1024+9 && a.slot_owner[s]<0 && a.n_pending==1);
    }
    // Four prefill copies can carry into generation. No new descriptor is
    // admitted at that limit, and adaptive copy activity prevents issue.
    {
        step=fail_step=0;Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        x.prediction({9,8,7,6},{.4f,.3f,.2f,.1f});std::vector<char> free(8);
        for(int s=0;s<4;++s){a.slot_pend[s]=2000+s;a.slot_iss[s]=1;}a.n_pending=4;
        x.prefetch_admit(x.L[0],a,1,free);assert(a.promo_q.empty() && a.n_pending==4);
        a.slot_pend[3]=-1;a.slot_iss[3]=0;--a.n_pending;
        x.prefetch_admit(x.L[0],a,1,free);assert(a.promo_q.size()==1);
        x.adaptive=3000;const int n=step;assert(x.prefetch_issue_one(a,e));assert(step==n);
    }
    // Sixteen rows produce one record for the entire group, using summed
    // support and stable expert-ID tie breaking. A pending identity, an active
    // helper lease and an adaptive source are all excluded as candidates.
    {
        Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        std::vector<int> ids;std::vector<float> weights;
        for(int t=0;t<16;++t) {ids.insert(ids.end(),{9,8,7,6});weights.insert(weights.end(),{.4f,.3f,.2f,.1f});}
        x.prediction(ids,weights);a.slot_pend[0]=1024+9;a.slot_iss[0]=1;a.n_pending=1;
        x.lease=1024+8;x.adaptive=1024+7;std::vector<char> free(8);
        x.prefetch_admit(x.L[0],a,16,free);
        assert(a.promo_q.size()==1 && a.n_pending==2 && a.slot_pend[a.promo_q[0].slot]==1024+6);
    }
    {
        Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        x.prediction({9,8,7,6},{.4f,.4f,.1f,.1f});std::vector<char> free(8);
        x.prefetch_admit(x.L[0],a,1,free);
        assert(a.promo_q.size()==1 && a.slot_pend[a.promo_q[0].slot]==1024+8);
    }
    // Explicit guards exclude graph, pipeline and parallel paths, and R-off
    // makes prediction and generation arbitration inactive.
    {
        Model x;assert(x.prefetch_active(true));assert(!x.prefetch_active(false));
        x.o.routing=false;assert(!x.prefetch_active(true));x.o.routing=true;
        x.rseq.push_back(0);assert(!x.prefetch_active(true));x.rseq.clear();
        x.graph_on=true;assert(!x.prefetch_active(true));
    }
    // An unissued prediction selected as a miss is neither transferred nor
    // late, and is cancelled before the target layer's ready snapshot.
    {
        Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        x.prediction({9,8,7,6},{.4f,.3f,.2f,.1f});std::vector<char> free(8);
        x.prefetch_admit(x.L[0],a,1,free);const int s=a.promo_q[0].slot;
        int selected_unissued[]={9,0,2,3};x.prefetch_note_selection(x.L[1],a,selected_unissued,1);
        assert(x.st.prefetch_late==0 && x.st.prefetch_unused==0 && a.slot_prediction[s]==1);
        assert(x.prefetch_begin_layer(a,e));assert(a.promo_q.empty() && a.n_pending==0);
        assert(a.slot_pend[s]<0 && a.slot_prediction[s]==0 && x.st.prefetch_cancelled==1);
        assert(x.st.prefetch_late_bytes==0 && x.st.prefetch_unused_bytes==0);
    }
    {
        Model x;auto& a=x.D[0];std::string e;a.slot_foreground[4]=1;a.slot_used[4]=x.fwd;
        fail_join=1;assert(!x.prefetch_join_readers(a,e));fail_join=0;
        assert(x.placement_failed && x.retirement_uncertain.load());
        assert(a.slot_foreground[4] && a.slot_used[4]==x.fwd);
    }
    {
        Model x;std::string e;fail_join=1;assert(!x.prefetch_control_join(x.D[0].s,e));fail_join=0;
        assert(x.placement_failed && x.retirement_uncertain.load());
    }
    // Both sides of each side-stream handoff are necessary for the later
    // primary-stream join to prove reader retirement.
    for(int enabled=0;enabled<2;++enabled)for(int routing=0;routing<2;++routing)for(int branch=0;branch<2;++branch)for(int fault=1;fault<=2;++fault) {
        Model x;x.o.routing_prefetch=enabled;x.o.routing=routing;std::string e;step=0;fail_step=fault;cudaStream_t stream=0;
        const bool ok=branch ? x.native_window_join(x.D[0],e) : x.native_window_fork(x.D[0],stream,e);
        assert(!ok && x.placement_failed && x.retirement_uncertain.load());fail_step=0;
    }
    {
        Model x;x.D[0].s2=0;std::string e;cudaStream_t stream=0;step=0;
        assert(x.native_window_fork(x.D[0],stream,e) && stream==x.D[0].s);
        assert(x.native_window_join(x.D[0],e) && step==0);
    }
    // The exact method called by both completion and discard cancels only an
    // unissued prediction; an issued prediction and ordinary demand retain
    // their separate frame/source ownership at the request boundary.
    {
        Model x;auto& a=x.D[0];std::string e;assert(x.prefetch_prepare(e));
        a.slot_pend[4]=1100;a.slot_iss[4]=1;a.slot_prediction[4]=1;
        a.slot_pend[5]=1101;a.slot_iss[5]=0;a.slot_prediction[5]=1;
        a.slot_pend[6]=1102;a.slot_iss[6]=0;a.n_pending=3;
        a.promo_q.push_back({5,x.hosts[0].data(),8});a.promo_q.push_back({6,x.hosts[1].data(),8});
        x.prefetch_end_generation();
        assert(a.n_pending==2 && a.slot_pend[5]<0 && a.promo_q.size()==1 && a.promo_q[0].slot==6);
        assert(a.slot_pend[4]==1100 && a.slot_iss[4] && a.slot_pend[6]==1102);
    }
    return 0;
}
