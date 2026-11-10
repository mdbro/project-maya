#include "mm3_capture_stub.hpp"
// Controlled model owner compiling the literal production observation,
// cache-reader and admission methods. No GPU or duplicate scoring algorithm.
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include "../src/mm3/mm3_native_demand_units.hpp"
#include "../src/mm3/mm3_residency_policy.hpp"
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define CK(x) do { if ((x)!=0) {err="injected CUDA failure";return false;} } while(0)
constexpr int cudaMemcpyDeviceToHost=1;
static int cudaSetDevice(int){return 0;}
static int cudaStreamSynchronize(int){return 0;}
static int cudaMemcpyAsync(void* a,const void* b,size_t n,int,int){std::memcpy(a,b,n);return 0;}
// early_copy's reader wait (f666cf0) compiles against these; no frame here holds an early copy.
using cudaError_t=int;using cudaEvent_t=int;constexpr int cudaSuccess=0;
static int early_copy_reader_waits=0;
static int cudaStreamWaitEvent(int,cudaEvent_t,unsigned){++early_copy_reader_waits;return 0;}
static const char* cudaGetErrorString(int){return "injected";}
namespace r2=mm3::r2;
struct Ready {uint32_t primary[4]{},helper[4]{};};
struct Layer {std::vector<int> res_slot;std::vector<const uint8_t*> host_blob;size_t blob=1;};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,s=0,n_slots=3,n_pending=0;size_t slot_bytes=1;
    std::vector<long long> slot_owner{-1,-1,-1},slot_pend{-1,-1,-1},slot_used{0,0,0};
    std::vector<int> slot_foreground{0,0,0},slot_iss{0,0,0};
    std::vector<uint8_t> slot_early,slot_early_fresh;std::vector<cudaEvent_t> slot_ev;   // (early_copy off: empty)
    std::vector<Promo> promo_q;
    uint8_t storage[3]{};uint8_t* stg=storage;
    Ready ready;Ready* ready_h=&ready;int originals[64]{};int* original_h=originals;
    unsigned* hf_original_d=nullptr;uint32_t* hf_snapshot_h=nullptr;
    mm3::DemandSnapshot hf_demand_seen;
};
struct Owner {
    struct {int n_layers=1,n_exp=8,k_exp=4;} g;
    struct {bool promote=true,routing=true;} o;
    struct {long route_original_misses=0,route_substitutions=0,route_primary_hits=0,route_helper_hits=0,miss_union=0,promoted=0,adaptive_hits=0;unsigned long long miss_union_bytes=0;long early_copy_waits=0,early_copy_used=0;} st;
    bool primary_adaptive=false;mm3::PrimaryResidencyPolicy adaptive_policy;
    bool native=true,hf_demand_poisoned=false,generation_active=true;long long fwd=10,native_demand_rows=0;int native_demand_block_rows=0;static constexpr float kDecay=.95f,kPromote=1.8f;static constexpr int kMaxPending=4;
    std::vector<Layer> L{Layer{}};std::vector<Dev> D;
    std::vector<float> use_score;std::vector<long long> use_last;std::vector<uint32_t> original_prof=std::vector<uint32_t>(8);
    std::vector<uint8_t> adaptive_main;
    double demand_decay=0.95;   // (residency_decay, 1b3ba4c: production default 0.95)
    bool placement_failed=false;std::atomic<bool> retirement_uncertain{false};
    const bool early_copy_verify=false;   // (MM3_EARLY_COPY_VERIFY unset)
    int early_copy_checks=0;
    bool early_copy_check(const Layer&,Dev&,int,int,std::string&){++early_copy_checks;return true;}
    // production's slot_early_busy (mm3_native_prefetch.inc, f666cf0), which this fixture does not include
    bool slot_early_busy(const Dev& d,int s) const {return !d.slot_early.empty() && d.slot_early[size_t(s)];}
    uint8_t host[8]{};
    Owner(){L[0].res_slot.assign(8,-1);L[0].res_slot[0]=0;L[0].host_blob.assign(8,host);}
    uint8_t* resident_address(const Layer&,int){return host;}
    bool foreground_usable(const Dev& d,size_t i) const{return r2::foreground_usable(d.slot_foreground[i]!=0,d.slot_used[i],fwd);}
    bool staging_reserved(const Dev&,int)const{return false;}
    uint8_t* staging_address(const Dev& d,int s)const{return d.stg+s*d.slot_bytes;}
    bool adaptive_identity_pending(const Dev&,long long)const{return false;}
    int adaptive_pending(const Dev&)const{return 0;}
    bool stage(Layer&,Dev&,int,int,std::string&){return true;}
    bool on_cpu(const Layer& l,const Dev& d,int e) const {
        if(l.res_slot[e]>=0 || (d.ready.helper[e/32] & (1u<<(e%32))))return false;
        for(int i=0;i<d.n_slots;++i) if(foreground_usable(d,i)&&d.slot_owner[i]==e)return false;
        return true;
    }
    bool helper_live_leased(long long)const{return false;}
    bool prefetch_active(bool)const{return false;}
    void prefetch_forget(Dev&,int){}
    void prefetch_note_selection(const Layer&,Dev&,const int*,int){}
    void prefetch_note_use(Dev&,int,size_t){}
#include "../src/mm3/mm3_native_demand.inc"
};
static void run_observation(Owner& o,Dev& d,const std::vector<int>& orig,const std::vector<int>& selected) {
    assert(orig.size()==selected.size() && orig.size()%4==0);
    if(o.generation_active) {o.native_demand_block_rows=int(orig.size()/4);o.native_demand_rows+=o.native_demand_block_rows;}
    std::copy(orig.begin(),orig.end(),d.originals);
    o.begin_window_demand(o.L[0],d,selected.data(),int(orig.size()/4));
}
int main(){
    { // One helper opportunity per five evaluated rows, never per chunk/call.
        mm3::NativeAdmissionBudget budget;
        for(int row=1;row<5;++row) {budget.advance(row,1);assert(!budget.available());}
        budget.advance(5,1);assert(budget.available());budget.spend();assert(!budget.available());
        budget.advance(5,1);assert(!budget.available()); // prefill leaves row clock unchanged
        budget.advance(500,1);assert(budget.credit==1); // idle history cannot create an unbounded burst
        mm3::NativeAdmissionBudget primary;primary.advance(5,8);
        for(int i=0;i<7;++i) {assert(primary.available(true));primary.spend();}
        assert(!primary.available(true) && primary.available());primary.spend();
        primary.advance(6,8);assert(primary.available() && !primary.available(true));
    }
    bool poisoned=false, armed=false;
    { mm3::DemandBoundary invalid_request{poisoned,armed}; }
    assert(!poisoned);
    { mm3::DemandBoundary complete{poisoned,armed}; armed=true; complete.complete=true; }
    assert(!poisoned);
    armed=false;
    { mm3::DemandBoundary failed{poisoned,armed}; armed=true; }
    assert(poisoned);
    Owner o;Dev d;std::string err;
    // Original A=4 is omitted; selected B=5 is resident in a staging slot.
    // Original 0 is primary, 1 helper, 2 pending. All get one credit.
    d.slot_owner[0]=5;d.slot_pend[1]=2;d.n_pending=1;d.ready.primary[0]=(1u<<0)|(1u<<5);d.ready.helper[0]=1u<<1;
    std::vector<int> orig{4,0,1,2},selected{5,0,1,2};
    run_observation(o,d,orig,selected);
    for(int e:{0,1,2,4}) assert(std::fabs(o.score(e,false)-mm3::native_demand_credit(1))<1e-6f);
    assert(o.score(5,false)==0 && d.slot_used[0]==o.fwd);
    std::vector<char> busy(3);unsigned long long ptr=0;
    assert(o.locate(o.L[0],d,5,busy,ptr,err));
    assert(busy[0] && o.score(5,false)==0);
    std::vector<int> cpu_of(8,-1); cpu_of[2]=0;
    o.finish_window_demand(o.L[0],d,selected.data(),1,cpu_of);
    assert(std::fabs(o.score(4,false)-mm3::native_demand_credit(1))<1e-6f && d.promo_q.empty());
    // Grouped repeated A contributes once per occurrence, while a selected B
    // remains protected even if A's new score is high enough for admission.
    ++o.fwd;orig={4,0,1,2,4,0,1,2,4,0,1,2};selected={5,0,1,2,5,0,1,2,5,0,1,2};
    run_observation(o,d,orig,selected);
    assert(std::fabs(o.score(4,false)-(mm3::native_demand_credit(1)*mm3::native_demand_decay(3)+3*mm3::native_demand_credit(3)))<1e-6f);
    assert(o.score(5,false)==0 && d.slot_used[0]==o.fwd);
    o.finish_window_demand(o.L[0],d,selected.data(),3,cpu_of);
    assert(d.promo_q.size()==1 && d.slot_pend[2]==4 && d.slot_owner[0]==5);
    o.finish_window_demand(o.L[0],d,selected.data(),3,cpu_of);
    assert(std::fabs(o.score(4,false)-(mm3::native_demand_credit(1)*mm3::native_demand_decay(3)+3*mm3::native_demand_credit(3)))<1e-6f && d.promo_q.size()==1);
    // Publication is the actual nonblocking production transition, and adds
    // no score. An already-pending original also received its occurrences.
    d.slot_iss[2]=1;std::string sticky;
    assert(r2::publish(d.slot_pend,d.slot_iss,d.slot_owner,d.slot_used,d.n_pending,o.fwd,sticky,err,
                       [](size_t,std::string&){return r2::Completion::ready;}));
    assert(d.slot_owner[2]==4 && std::fabs(o.score(4,false)-(mm3::native_demand_credit(1)*mm3::native_demand_decay(3)+3*mm3::native_demand_credit(3)))<1e-6f);
    assert(std::fabs(o.score(2,false)-(mm3::native_demand_credit(1)*mm3::native_demand_decay(3)+3*mm3::native_demand_credit(3)))<1e-6f);
    // Routing off and on use the same original credit; execution/admission is
    // never another credit, including an already resident expert.
    ++o.fwd;o.o.routing=false;orig={4,0,1,3};selected=orig;
    run_observation(o,d,orig,selected);const float a=o.score(4,false);
    o.finish_window_demand(o.L[0],d,selected.data(),1,cpu_of);
    assert(o.score(4,false)==a && std::fabs(o.score(3,false)-mm3::native_demand_credit(1))<1e-6f);
    // With routing off, a selected CPU expert's two real occurrences are
    // credited before the actual grouped admission transition.
    Owner off;Dev od;off.o.routing=false;od.ready.primary[0]=1u;od.ready.helper[0]=2u;
    std::vector<int> straight{4,0,1,3,4,0,1,3};run_observation(off,od,straight,straight);
    std::vector<int> off_cpu(8,-1);off_cpu[4]=0;off.finish_window_demand(off.L[0],od,straight.data(),2,off_cpu);
    assert(std::fabs(off.score(4,false)-2*mm3::native_demand_credit(2))<1e-6f && od.promo_q.size()==1 && od.slot_pend[0]==4);
    // HF snapshots consume prefill deltas without leaking them into generation.
    Owner hf;Dev hd;unsigned histogram[8]{};hd.hf_original_d=histogram;
    uint32_t shot[8]{};hf.generation_active=false;histogram[4]=9;
    assert(hf.queue_hf_demand(hd,shot,err) && hf.consume_hf_demand(hd,shot,10,err));
    assert(hf.score(4,false)==0);
    hf.generation_active=true;hf.native_demand_rows=5;hf.native_demand_block_rows=5;
    histogram[4]=12;histogram[5]=1;
    assert(hf.queue_hf_demand(hd,shot,err) && hf.consume_hf_demand(hd,shot,11,err));
    assert(std::fabs(hf.score(4,false)-3)<1e-5 && std::fabs(hf.score(5,false)-1)<1e-5);
    assert(hf.consume_hf_demand(hd,shot,11,err) && std::fabs(hf.score(4,false)-3)<1e-5);
    // Prefill does not age an existing generation history.
    hf.generation_active=false;histogram[4]+=12;
    assert(hf.queue_hf_demand(hd,shot,err) && hf.consume_hf_demand(hd,shot,12,err));
    assert(std::fabs(hf.score(4,false)-3)<1e-5);
    // Priced primary admission owns optional fills; legacy and disabled
    // primary retain ordinary promotion when the same hot expert is observed.
    {
        Owner q;Dev d;q.native_demand_rows=5;q.native_demand_block_rows=5;
        q.score_at(4,5,10);q.primary_adaptive=true;q.adaptive_policy.legacy=false;
        q.consider_promotion(q.L[0],d,4);assert(d.promo_q.empty());
        q.adaptive_policy.legacy=true;q.consider_promotion(q.L[0],d,4);
        assert(d.promo_q.size()==1 && d.slot_pend[0]==4);
        Owner off;Dev od;off.native_demand_rows=5;off.native_demand_block_rows=5;
        off.score_at(4,5,10);off.primary_adaptive=false;off.adaptive_policy.legacy=false;
        off.consider_promotion(off.L[0],od,4);assert(od.promo_q.size()==1);
    }
    // Five rows reproduce the old recurrence, and stationary frequency is
    // independent of block width (uniform credit within each block).
    assert(std::fabs(mm3::native_demand_decay(5)-.95f)<1e-7f);
    assert(std::fabs(mm3::native_demand_credit(5)-1.f)<1e-7f);
    for(int width:{1,5,16}) {
        double transient=0;
        for(int rows=0;rows<80;rows+=width)
            transient=transient*mm3::native_demand_decay(width)+width*mm3::native_demand_credit(width)*.25;
        assert(std::fabs(transient-25*(1-mm3::native_demand_decay(80)))<.0001);
        double value=0;
        for(int i=0;i<10000;++i)
            value=value*mm3::native_demand_decay(width)+width*mm3::native_demand_credit(width)*.25;
        assert(std::fabs(value-25)<.001);
    }
}
