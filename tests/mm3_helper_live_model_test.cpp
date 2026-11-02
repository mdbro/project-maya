// Exercises the literal production Model helper-live methods with controlled
// worker completion, foreground readers and source ownership.
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
using cudaError_t=int;
constexpr int cudaSuccess=0,cudaHostAllocPortable=1;
static int fake_selected_device=-1;
static int cudaSetDevice(int id){fake_selected_device=id;return 0;}
static int cudaHostAlloc(uint8_t** p,size_t n,unsigned){*p=static_cast<uint8_t*>(std::malloc(n));return *p?0:1;}
static int fake_host_frees=0;
static int cudaFreeHost(void* p){++fake_host_frees;std::free(p);return 0;}
static const char* cudaGetErrorString(int){return "injected";}
namespace ne { struct Header {float weight_global[3],input_global[3];uint8_t pad[232];}; }
namespace nf { static size_t record_stride(int,int){return 16;} }
struct NativeHelper {
    enum class LivePhase {idle,preparing,prepared,uploading,uploaded,published,recycling,complete,failed};
    struct LiveStatus {LivePhase phase;bool storage_unsafe;std::string error;};
    struct Identity {int layer,expert,slot;};
    enum class Fault {none,begin,withdraw,publish,recycle,retire};
    LivePhase phase=LivePhase::idle; int ready_layer=0,ready_expert=1; bool busy=false,auto_advance=false;
    Fault fault=Fault::none;mutable int records_calls=0;
    uint8_t *host=nullptr,*incoming=nullptr,*outgoing=nullptr;
    std::array<uint8_t,16> record{}; int dst=-1,dst_layer=-1;
    NativeHelper(){record.fill(0x22);}
    std::vector<Identity> records()const {++records_calls;return ready_expert>=0?std::vector<Identity>{{ready_layer,ready_expert,0}}:std::vector<Identity>{};}
    bool live_begin(int old_layer,int old,int new_layer,int next,void* h,void* i,void* o,size_t n,const float*,const float*,float* w,float* x,std::string& e) {
        if(fault==Fault::begin){e="injected begin failure";return false;}
        assert(phase==LivePhase::idle && old_layer==ready_layer && old==ready_expert && n==16);
        host=(uint8_t*)h;incoming=(uint8_t*)i;outgoing=(uint8_t*)o;dst=next;dst_layer=new_layer;
        for(int j=0;j<3;++j){w[j]=float(j+1);x[j]=float(j+4);}
        phase=LivePhase::preparing;busy=true;return true;
    }
    void finish_prepare(){assert(phase==LivePhase::preparing);std::memcpy(incoming,host,16);std::memcpy(outgoing,record.data(),16);busy=false;phase=LivePhase::prepared;}
    bool live_withdraw(std::string& e){if(fault==Fault::withdraw){e="injected withdraw failure";return false;}assert(phase==LivePhase::prepared);ready_expert=-1;busy=true;phase=LivePhase::uploading;return true;}
    void finish_upload(){assert(phase==LivePhase::uploading);std::memcpy(record.data(),incoming,16);busy=false;phase=LivePhase::uploaded;}
    bool live_publish(std::string& e){if(fault==Fault::publish){e="injected publish failure";return false;}assert(phase==LivePhase::uploaded);ready_layer=dst_layer;ready_expert=dst;phase=LivePhase::published;return true;}
    bool live_recycle(std::string& e){if(fault==Fault::recycle){e="injected recycle failure";return false;}assert(phase==LivePhase::published);busy=true;phase=LivePhase::recycling;return true;}
    void finish_recycle(){assert(phase==LivePhase::recycling);std::memcpy(host,outgoing,16);busy=false;phase=LivePhase::complete;}
    LiveStatus live_poll(){if(auto_advance && phase==LivePhase::uploading)finish_upload();else if(auto_advance && phase==LivePhase::recycling)finish_recycle();return {phase,phase!=LivePhase::idle,phase==LivePhase::failed?"injected asynchronous failure":""};}
    bool live_retire(bool& safe,std::string& e){if(fault==Fault::retire){safe=false;e="injected retire failure";return false;}safe=!busy && (phase==LivePhase::complete || phase==LivePhase::failed);if(safe)phase=LivePhase::idle;return safe;}
    bool live_cancel(bool& safe,std::string&){phase=LivePhase::failed;safe=!busy;if(safe)phase=LivePhase::idle;return safe;}
};
struct Layer {
    bool moe=true,hf=false;int dev=0;size_t blob=272;
    std::vector<int> res_slot=std::vector<int>(4,-1);
    std::vector<const uint8_t*> host_blob=std::vector<const uint8_t*>(4,nullptr);
    std::array<uint32_t,4> helper_ready{};
};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,n_slots=2;std::vector<int> layers{0};
    std::vector<long long> slot_pend=std::vector<long long>(2,-1),slot_owner=std::vector<long long>(2,-1);
    std::vector<uint8_t> slot_foreground=std::vector<uint8_t>(2,0);
    std::vector<Promo> promo_q;
};
struct Model {
    struct {int dim=1,ff=1,n_exp=4;} g;
    struct {int native_stage_prefix=0,seqs=1,chunk=4;bool fast_decode=true;} o;
    static constexpr float kPromote=1.8f;static constexpr int kWin=16;
    bool placement_failed=false,generation_active=true;std::atomic<bool> retirement_uncertain{false};
    bool native=true,native_pool=true,graph_on=false,vis=false,adaptive_available=true;
    int head_dev=0;std::vector<Layer>L=std::vector<Layer>(1);std::vector<Dev>D=std::vector<Dev>(1);
    std::unique_ptr<NativeHelper> helper=std::make_unique<NativeHelper>();
    std::array<float,4> scores{{5.0f,0.0f,1.0f,0.0f}};
    alignas(ne::Header) std::array<uint8_t,272> a{},c{};long long pending_adaptive=-1;
    bool adaptive_supported()const{return adaptive_available;}
    bool adaptive_identity_pending(const Dev&,long long k)const{return pending_adaptive==k;}
    bool helper_owns(const Layer& l,int e)const{return (l.helper_ready[e/32]>>(e%32))&1u;}
    float score(long long k,bool){return scores[size_t(k%1024)];}
    uint64_t helper_queue_pending=0;
    void prefetch_forget(Dev&,int){}
#include "../src/mm3/mm3_native_helper_live.inc"
    Model(){
        L[0].host_blob[0]=a.data();L[0].host_blob[2]=c.data();L[0].res_slot[3]=0;
        L[0].helper_ready[0]=1u<<1;
        auto* h=reinterpret_cast<ne::Header*>(a.data());
        for(int i=0;i<3;++i){h->weight_global[i]=float(i+7);h->input_global[i]=float(i+10);}
        std::memset(a.data()+256,0x11,16);
    }
    ~Model(){for(auto* p:helper_live_buffers)if(p)cudaFreeHost(p);}
};
static void stage_suffix_owner() {
    Model m;std::string error;
    // Give the literal production methods a dense prefix owner and a suffix
    // owner whose first MoE layer is 3, with a real nonzero layer identity.
    m.adaptive_available=false;m.o.native_stage_prefix=3;m.head_dev=1;
    m.L.resize(4);m.L[3]=m.L[0];m.L[3].dev=1;
    for(int i=0;i<3;++i)m.L[i].moe=false;
    m.D.resize(2);m.D[0].layers={0,1,2};m.D[0].n_slots=0;
    Dev& front=m.D[0];Dev& suffix=m.D[1];
    suffix.id=11;suffix.layers={3};suffix.n_slots=4;
    suffix.slot_pend.resize(4,-1);suffix.slot_owner.resize(4,-1);suffix.slot_foreground.resize(4,0);
    m.helper->ready_layer=3;

    assert(m.helper_live_mode_supported());
    m.o.native_stage_prefix=2;assert(!m.helper_live_mode_supported());m.o.native_stage_prefix=3;
    m.native=false;assert(!m.helper_live_mode_supported());m.native=true;
    m.native_pool=false;assert(!m.helper_live_mode_supported());m.native_pool=true;
    m.graph_on=true;assert(!m.helper_live_mode_supported());m.graph_on=false;
    m.o.seqs=2;assert(!m.helper_live_mode_supported());m.o.seqs=1;
    m.o.fast_decode=false;assert(!m.helper_live_mode_supported());m.o.fast_decode=true;
    m.o.chunk=17;assert(!m.helper_live_mode_supported());m.o.chunk=4;
    m.vis=true;assert(!m.helper_live_mode_supported());m.vis=false;
    suffix.n_slots=3;assert(!m.helper_live_mode_supported());suffix.n_slots=4;
    m.head_dev=0;assert(!m.helper_live_mode_supported());m.head_dev=1;
    assert(m.helper_live_prepare(error) && fake_selected_device==11);
    m.helper_live_enabled=true;
    const long long incoming=3*1024,victim=incoming+1;
    int scans=m.helper->records_calls;
    assert(m.helper_live_issue(front,error));
    assert(m.helper_live_a<0 && m.helper->phase==NativeHelper::LivePhase::idle && m.helper->records_calls==scans);
    // A suffix-owned pending reader blocks admission even with prefix idle.
    suffix.slot_pend[3]=incoming;assert(m.helper_live_issue(suffix,error) && m.helper_live_a<0);
    suffix.slot_pend[3]=-1;assert(m.helper_live_issue(suffix,error));
    assert(m.helper_live_a==incoming && m.helper_live_b==victim);
    assert(m.helper_live_leased(incoming) && m.helper_live_leased(victim));
    assert(m.L[3].host_blob[0]==m.a.data() && m.helper_owns(m.L[3],1));

    m.helper->finish_prepare();scans=m.helper->records_calls;
    assert(m.helper_live_progress(front,error));
    assert(m.helper->phase==NativeHelper::LivePhase::prepared && m.helper->records_calls==scans);
    assert(m.helper_owns(m.L[3],1) && !m.L[3].host_blob[1]);
    assert(m.helper_live_progress(suffix,error));
    assert(m.helper_live_withdrawals==1 && !m.helper_owns(m.L[3],1));
    assert(m.L[3].host_blob[1]==m.helper_live_buffers[1]);

    m.helper->finish_upload();suffix.slot_owner[3]=incoming;suffix.slot_foreground[3]=1;
    assert(m.helper_live_progress(front,error));assert(m.helper_live_progress(suffix,error));
    assert(m.helper->phase==NativeHelper::LivePhase::uploaded && !m.helper_owns(m.L[3],0));
    assert(m.L[3].host_blob[0]==m.a.data() && m.helper_live_leased(incoming));
    suffix.slot_foreground[3]=0;assert(m.helper_live_progress(suffix,error));
    assert(m.helper_live_publications==1 && m.helper_owns(m.L[3],0));
    assert(!m.L[3].host_blob[0] && suffix.slot_owner[3]==-1);
    assert(m.helper->ready_layer==3 && m.helper->ready_expert==0);
    assert(m.L[3].host_blob[1]==m.helper_live_buffers[1]);

    m.helper->finish_recycle();scans=m.helper->records_calls;
    assert(m.helper_live_progress(front,error));
    assert(m.helper->phase==NativeHelper::LivePhase::complete && m.helper->records_calls==scans);
    // Production maintenance must progress D[head_dev], or it cannot retire.
    assert(m.helper_live_settle(error));
    assert(m.helper_live_a<0 && m.helper_live_finished==1);
    assert(m.L[3].host_blob[1]==m.a.data() && m.helper_owns(m.L[3],0));
    assert(m.a[256]==0x22 && m.helper->record[0]==0x11);
}
int main(){
    stage_suffix_owner();
    std::string e;Model m;auto& d=m.D[0];
    assert(m.helper_live_prepare(e));assert(m.helper_live_blob==272 && m.helper_live_capacity==1);
    m.helper_live_enabled=true;
    m.helper_queue_pending=1;
    assert(m.helper_live_issue(d,e) && m.helper_live_a<0);
    m.helper_queue_pending=0;
    // Admission respects both pending optional work and foreground HOST readers.
    d.slot_pend[0]=0;assert(m.helper_live_issue(d,e) && m.helper_live_a<0);d.slot_pend[0]=-1;
    d.slot_owner[0]=0;d.slot_foreground[0]=1;assert(m.helper_live_issue(d,e) && m.helper_live_a<0);
    d.slot_foreground[0]=0;m.pending_adaptive=0;assert(m.helper_live_issue(d,e) && m.helper_live_a<0);
    m.pending_adaptive=-1;assert(m.helper_live_issue(d,e));
    assert(m.helper_live_a==0 && m.helper_live_b==1 && m.helper_live_leased(0) && m.helper_live_leased(1));
    assert(m.L[0].host_blob[0]==m.a.data() && m.helper_owns(m.L[0],1));
    int scans=m.helper->records_calls;
    for(int i=0;i<10;++i)assert(m.helper_live_progress(d,e) && m.helper->phase==NativeHelper::LivePhase::preparing);
    assert(m.helper->records_calls==scans); // pending polling never builds a roster
    m.helper->finish_prepare();m.helper_live_enabled=false; // stop admission, finish lease
    m.helper_queue_pending=1;
    assert(m.helper_live_progress(d,e) && m.helper_owns(m.L[0],1));
    assert(m.helper->phase==NativeHelper::LivePhase::prepared && m.helper->records_calls==scans);
    assert(!m.helper_live_settle(e));
    m.helper_queue_pending=0;
    assert(m.helper_live_progress(d,e));
    assert(m.helper->records_calls==scans+2 && m.helper_live_withdrawals==1 && m.helper_live_publications==0);
    assert(!m.helper_owns(m.L[0],1) && m.L[0].host_blob[1]==m.helper_live_buffers[1]);
    assert(m.helper->records().empty() && m.helper->phase==NativeHelper::LivePhase::uploading);
    m.helper->finish_upload();d.slot_owner[0]=0;d.slot_foreground[0]=1;
    scans=m.helper->records_calls;
    for(int i=0;i<10;++i)assert(m.helper_live_progress(d,e) && !m.helper_owns(m.L[0],0));
    assert(m.helper->records_calls==scans); // a source reader delays mutation, not the forward
    d.slot_foreground[0]=0;assert(m.helper_live_progress(d,e));
    assert(m.helper->records_calls==scans+2 && m.helper_live_publications==1);
    assert(m.helper_owns(m.L[0],0) && !m.L[0].host_blob[0] && d.slot_owner[0]==-1);
    auto* bh=reinterpret_cast<const ne::Header*>(m.L[0].host_blob[1]);
    assert(bh->weight_global[0]==1 && bh->input_global[2]==6);
    m.helper->finish_recycle();d.slot_owner[1]=1;d.slot_foreground[1]=1;
    scans=m.helper->records_calls;
    for(int i=0;i<10;++i)assert(m.helper_live_progress(d,e) && m.L[0].host_blob[1]==m.helper_live_buffers[1]);
    assert(m.helper->records_calls==scans);
    d.slot_foreground[1]=0;assert(m.helper_live_progress(d,e));
    assert(m.helper->records_calls==scans+1);
    assert(m.helper_live_a<0 && m.L[0].host_blob[1]==m.a.data() && m.helper_live_finished==1);
    assert(m.a[256]==0x22 && reinterpret_cast<ne::Header*>(m.a.data())->weight_global[2]==3);
    assert(m.helper->record[0]==0x11 && m.helper_live_issue(d,e) && m.helper_live_a<0);
    // Teardown never releases borrowed banks while the fake CPU/GPU worker is active.
    m.helper_live_enabled=true;m.scores={{0,0,8,0}};d.slot_owner[1]=-1;
    assert(m.helper_live_issue(d,e) && m.helper_live_a==2);
    assert(!m.helper_live_cancel_retired());m.helper->busy=false;
    assert(m.helper_live_cancel_retired());

    // A and B share one Layer in all cases, so rollback and paired bit changes
    // exercise the real aliasing shape. Each failed API retains its detail.
    using F=NativeHelper::Fault;
    for(auto fault:{F::begin,F::withdraw,F::publish,F::recycle,F::retire}) {
        Model n;auto& nd=n.D[0];std::string error;
        assert(n.helper_live_prepare(error));n.helper_live_enabled=true;n.helper->fault=fault;
        if(fault==F::begin) {
            assert(!n.helper_live_issue(nd,error));
            assert(n.helper_live_a<0 && n.L[0].host_blob[0]==n.a.data() && n.helper_owns(n.L[0],1));
        } else {
            assert(n.helper_live_issue(nd,error));n.helper->finish_prepare();
            if(fault==F::withdraw) {
                assert(!n.helper_live_progress(nd,error));
                assert(n.L[0].host_blob[0]==n.a.data() && !n.L[0].host_blob[1] && n.helper_owns(n.L[0],1));
            } else {
                assert(n.helper_live_progress(nd,error));n.helper->finish_upload();
                if(fault==F::publish) {
                    assert(!n.helper_live_progress(nd,error));
                    assert(n.L[0].host_blob[0]==n.a.data() && n.L[0].host_blob[1]==n.helper_live_buffers[1]);
                    assert(!n.helper_owns(n.L[0],0) && !n.helper_owns(n.L[0],1));
                } else if(fault==F::recycle) {
                    assert(!n.helper_live_progress(nd,error));
                    assert(!n.L[0].host_blob[0] && n.helper_owns(n.L[0],0));
                    assert(n.L[0].host_blob[1]==n.helper_live_buffers[1]);
                } else {
                    assert(n.helper_live_progress(nd,error));n.helper->finish_recycle();
                    assert(!n.helper_live_progress(nd,error));
                    assert(!n.L[0].host_blob[0] && n.helper_owns(n.L[0],0));
                    assert(n.L[0].host_blob[1]==n.helper_live_buffers[1]);
                }
            }
        }
        assert(n.placement_failed && error.find("injected")!=std::string::npos);
        assert(n.helper_live_cancel_retired()); // external lease ended before Model destruction
    }
    // Inject worker failure after withdrawal and after publication. Teardown
    // must retain the complete Model and both portable buffers while busy.
    for(bool during_recycle:{false,true}) {
        auto* n=new Model;auto& nd=n->D[0];std::string error;
        assert(n->helper_live_prepare(error));n->helper_live_enabled=true;
        assert(n->helper_live_issue(nd,error));n->helper->finish_prepare();assert(n->helper_live_progress(nd,error));
        if(during_recycle){n->helper->finish_upload();assert(n->helper_live_progress(nd,error));}
        n->helper->phase=NativeHelper::LivePhase::failed;n->helper->busy=true;
        assert(!n->helper_live_progress(nd,error) && n->placement_failed);
        const int frees_before=fake_host_frees;
        assert(!n->helper_live_cancel_retired() && fake_host_frees==frees_before);
        assert(n->L[0].host_blob[1]==n->helper_live_buffers[1]);
        if(during_recycle)assert(n->helper_owns(n->L[0],0) && !n->L[0].host_blob[0]);
        else assert(!n->helper_owns(n->L[0],0) && n->L[0].host_blob[0]==n->a.data());
        n->helper->busy=false;assert(n->helper_live_cancel_retired());delete n;
        assert(fake_host_frees==frees_before+2);
    }
    // Explicit maintenance can settle a withdrawn exchange using observed
    // completions before a placement ownership audit.
    {
        Model n;std::string error;auto& nd=n.D[0];assert(n.helper_live_prepare(error));n.helper_live_enabled=true;
        assert(n.helper_live_issue(nd,error));n.helper->finish_prepare();assert(n.helper_live_progress(nd,error));
        assert(n.helper_live_withdrawn && !n.helper_live_published);
        n.helper->auto_advance=true;assert(n.helper_live_settle(error));
        assert(n.helper_live_a<0 && n.helper_live_finished==1 && n.helper_owns(n.L[0],0));
        assert(n.L[0].host_blob[1]==n.a.data() && n.a[256]==0x22);
    }
}
