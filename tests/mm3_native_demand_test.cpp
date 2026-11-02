// Controlled model owner compiling the literal production observation,
// cache-reader and admission methods. No GPU or duplicate scoring algorithm.
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <algorithm>
#include <array>
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
namespace r2=mm3::r2;
struct Ready {uint32_t primary[4]{},helper[4]{};};
struct Layer {std::vector<int> res_slot;std::vector<const uint8_t*> host_blob;size_t blob=1;};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,s=0,n_slots=3,n_pending=0;size_t slot_bytes=1;
    std::vector<long long> slot_owner{-1,-1,-1},slot_pend{-1,-1,-1},slot_used{0,0,0};
    std::vector<int> slot_foreground{0,0,0},slot_iss{0,0,0};
    std::vector<Promo> promo_q;
    uint8_t storage[3]{};uint8_t* stg=storage;
    Ready ready;Ready* ready_h=&ready;int originals[64]{};int* original_h=originals;
    unsigned* hf_original_d=nullptr;uint32_t* hf_snapshot_h=nullptr;
    mm3::DemandSnapshot hf_demand_seen;
};
struct Owner {
    struct {int n_layers=1,n_exp=8,k_exp=4;} g;
    struct {bool promote=true,routing=true;} o;
    struct {long route_original_misses=0,route_substitutions=0,route_primary_hits=0,route_helper_hits=0,miss_union=0,promoted=0;unsigned long long miss_union_bytes=0;} st;
    bool native=true,hf_demand_poisoned=false;long long fwd=10;static constexpr float kDecay=.95f,kPromote=1.8f;static constexpr int kMaxPending=4;
    std::vector<Layer> L{Layer{}};std::vector<Dev> D;
    std::vector<float> use_score;std::vector<long long> use_last;std::vector<uint32_t> original_prof=std::vector<uint32_t>(8);
    uint8_t host[8]{};
    Owner(){L[0].res_slot.assign(8,-1);L[0].res_slot[0]=0;L[0].host_blob.assign(8,host);}
    uint8_t* resident_address(const Layer&,int){return host;}
    bool foreground_usable(const Dev& d,size_t i) const{return r2::foreground_usable(d.slot_foreground[i]!=0,d.slot_used[i],fwd);}
    bool stage(Layer&,Dev&,int,int,std::string&){return true;}
    bool on_cpu(const Layer& l,const Dev& d,int e) const {
        if(l.res_slot[e]>=0 || (d.ready.helper[e/32] & (1u<<(e%32))))return false;
        for(int i=0;i<d.n_slots;++i) if(foreground_usable(d,i)&&d.slot_owner[i]==e)return false;
        return true;
    }
#include "../src/mm3/mm3_native_demand.inc"
};
static void run_observation(Owner& o,Dev& d,const std::vector<int>& orig,const std::vector<int>& selected) {
    assert(orig.size()==selected.size() && orig.size()%4==0);
    std::copy(orig.begin(),orig.end(),d.originals);
    o.begin_window_demand(o.L[0],d,selected.data(),int(orig.size()/4));
}
int main(){
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
    for(int e:{0,1,2,4}) assert(o.score(e,false)==1);
    assert(o.score(5,false)==0 && d.slot_used[0]==o.fwd);
    std::vector<char> busy(3);unsigned long long ptr=0;
    assert(o.locate(o.L[0],d,5,busy,ptr,err));
    assert(busy[0] && o.score(5,false)==0);
    std::vector<int> cpu_of(8,-1); cpu_of[2]=0;
    o.finish_window_demand(o.L[0],d,selected.data(),1,cpu_of);
    assert(o.score(4,false)==1 && d.promo_q.empty());
    // Grouped repeated A contributes once per occurrence, while a selected B
    // remains protected even if A's new score is high enough for admission.
    ++o.fwd;orig={4,0,1,2,4,0,1,2,4,0,1,2};selected={5,0,1,2,5,0,1,2,5,0,1,2};
    run_observation(o,d,orig,selected);
    assert(std::fabs(o.score(4,false)-3.95f)<1e-6f);
    assert(o.score(5,false)==0 && d.slot_used[0]==o.fwd);
    o.finish_window_demand(o.L[0],d,selected.data(),3,cpu_of);
    assert(d.promo_q.size()==1 && d.slot_pend[2]==4 && d.slot_owner[0]==5);
    o.finish_window_demand(o.L[0],d,selected.data(),3,cpu_of);
    assert(std::fabs(o.score(4,false)-3.95f)<1e-6f && d.promo_q.size()==1);
    // Publication is the actual nonblocking production transition, and adds
    // no score. An already-pending original also received its occurrences.
    d.slot_iss[2]=1;std::string sticky;
    assert(r2::publish(d.slot_pend,d.slot_iss,d.slot_owner,d.slot_used,d.n_pending,o.fwd,sticky,err,
                       [](size_t,std::string&){return r2::Completion::ready;}));
    assert(d.slot_owner[2]==4 && std::fabs(o.score(4,false)-3.95f)<1e-6f);
    assert(std::fabs(o.score(2,false)-3.95f)<1e-6f);
    // Routing off and on use the same original credit; execution/admission is
    // never another credit, including an already resident expert.
    ++o.fwd;o.o.routing=false;orig={4,0,1,3};selected=orig;
    run_observation(o,d,orig,selected);const float a=o.score(4,false);
    o.finish_window_demand(o.L[0],d,selected.data(),1,cpu_of);
    assert(o.score(4,false)==a && o.score(3,false)==1);
    // With routing off, a selected CPU expert's two real occurrences are
    // credited before the actual grouped admission transition.
    Owner off;Dev od;off.o.routing=false;od.ready.primary[0]=1u;od.ready.helper[0]=2u;
    std::vector<int> straight{4,0,1,3,4,0,1,3};run_observation(off,od,straight,straight);
    std::vector<int> off_cpu(8,-1);off_cpu[4]=0;off.finish_window_demand(off.L[0],od,straight.data(),2,off_cpu);
    assert(off.score(4,false)==2 && od.promo_q.size()==1 && od.slot_pend[0]==4);
    // HF device counters can be consumed at their original stage ticks even
    // when the caller joins later; duplicate snapshots never add credit.
    Owner hf;Dev hd;unsigned histogram[8]{};hd.hf_original_d=histogram;
    uint32_t shot1[8]{},shot2[8]{};histogram[4]=2;assert(hf.queue_hf_demand(hd,shot1,err));
    histogram[4]=5;histogram[5]=1;assert(hf.queue_hf_demand(hd,shot2,err));
    assert(hf.consume_hf_demand(hd,shot1,10,err));
    assert(hf.consume_hf_demand(hd,shot2,11,err));
    assert(std::fabs(hf.score_at(4,11,0)-4.9f)<1e-6f && hf.score_at(5,11,0)==1);
    assert(hf.consume_hf_demand(hd,shot2,11,err) && hf.score_at(4,11,0)==4.9f);
}
