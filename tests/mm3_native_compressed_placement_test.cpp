// Actual CPU codec with fake CUDA copies through the literal production placement method.
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
using cudaStream_t = int;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
using cudaError_t = int;
constexpr int cudaSuccess=0, cudaHostAllocPortable=1;
static int copy_calls=0, fail_copy=0, allocs=0, frees=0;
static const char* cudaGetErrorString(int){return "injected";}
static int cudaSetDevice(int){return 0;}
static int cudaStreamSynchronize(int){return 0;}
static int cudaMemcpyAsync(void* d,const void* s,size_t n,cudaMemcpyKind,int){
    if (++copy_calls==fail_copy) return 1;
    std::memcpy(d,s,n);return 0;
}
static int cudaHostAlloc(uint8_t** p,size_t n,unsigned){*p=(uint8_t*)std::malloc(n);++allocs;return *p?0:1;}
static int cudaFreeHost(void* p){++frees;std::free(p);return 0;}
namespace ne { struct Header {float weight_global[3],input_global[3];uint8_t reserved[232];}; static_assert(sizeof(Header)==256); }
namespace mm3::scale_store {
constexpr size_t kRawBytes=31850752,kScratchBytes=31850496,kPinnedBytes=240648192;
bool encode(const uint8_t*,size_t,uint8_t*,size_t,size_t&,std::string&);
bool decode_raw(const uint8_t*,size_t,uint8_t*,size_t,std::string&);
}
namespace ss=mm3::scale_store;
namespace nf { static size_t record_stride(int,int){return ss::kRawBytes-256;} }
struct NativeHelper {
    struct Identity {int layer,expert,slot;};
    std::vector<Identity> records()const{return {};}
    std::array<uint32_t,4> ready(int)const{return {};}
    uintptr_t pool_identity()const{return 0;}
    bool export_expert(int,int,void*,size_t,float*,float*,bool&,std::string&){return false;}
    bool replace_expert(int,int,int,int,const void*,size_t,const float*,const float*,bool&,std::string&){return false;}
};
struct Layer {bool moe=true,hf=false;int dev=0;size_t blob=ss::kRawBytes;std::vector<int> res_slot,res_ids;uint8_t* res_base=nullptr;std::vector<uint8_t*> res_ptr;std::vector<const uint8_t*> host_blob;std::array<uint32_t,4> helper_ready{};unsigned long long* exp_tab=nullptr;};
struct Dev {int id=0,s=1,s2=2,cs=3,n_slots=0;float* win_cpu=nullptr;uint8_t *owner_plan_h=nullptr,*win_plan_d=nullptr;std::vector<int> layers{0},promo_q,slot_owner,slot_pend,slot_used,slot_foreground;std::vector<long long> slot_iss;int n_pending=0;};
struct Pool {size_t n;size_t size()const{return n;}size_t capacity()const{return n;}};
struct TestModel {
    struct {int n_layers=1,n_exp=4,dim=6144,ff=3072;} g;
    struct {int chunk=4,resident_cap=-1;bool native_cooperative=false;} o;
    struct {int experts=4,pooled=2;} st;
    size_t native_cpu_bytes=0;int scale_device=0;static constexpr int kWin=16;size_t win_plan_bytes()const{return 4096;}
    bool native=true;std::unique_ptr<Pool> native_pool;std::atomic<bool> retirement_uncertain{false};
    std::vector<Layer> L=std::vector<Layer>(1);std::vector<Dev>D=std::vector<Dev>(1);int head_dev=0;
    std::unique_ptr<NativeHelper>helper;uint8_t* pool=nullptr;std::vector<void*>owned;std::unique_ptr<int>draftm;
    std::string promotion_error;std::vector<float>use_score;std::vector<int>use_last;std::vector<uint8_t>adaptive_main;
    size_t adaptive_blob=0;bool primary_adaptive=false;std::array<uint8_t*,2> adaptive_host_owned{};
    std::vector<uint8_t> host,primary;size_t host_stride=0;std::array<size_t,4> sizes{};
    bool compressed_host()const{return true;}
    size_t host_record_bytes(const Layer&)const{return host_stride;}
    size_t scale_encode_scratch_bytes()const{return 3538944+70000;}
    bool scale_drain(std::string&){return true;}bool scale_joined(Dev&,std::string&){return true;}
    bool adaptive_host_contains(uintptr_t,size_t)const{return false;}
    bool adaptive_storage_ok(const std::vector<uintptr_t>&,const std::vector<uintptr_t>&)const{return true;}
    bool adaptive_publish(Dev&,std::string&){return true;}int adaptive_pending(const Dev&)const{return 0;}
    uint8_t* staging_address(const Dev&,int)const{return nullptr;}bool staging_reserved(const Dev&,int)const{return false;}
    bool helper_owns(const Layer&,int)const{return false;}
    bool baseline_hf_demand(std::string&){return true;}void graphs_clear(){}
    static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
    uint64_t helper_queue_pending=0;void helper_queue_cancel_pending(){helper_queue_pending=0;placement_failed=true;}
    long long helper_live_a=-1;size_t helper_live_blob=0;bool helper_live_enabled=false;
    uint64_t helper_live_started=0,helper_live_finished=0;
    bool helper_live_settle(std::string&){return true;}
    void prefetch_discard_joined(Dev&){}
    // Distinct exact records, including deliberately high entropy scales for id 0.
    static void raw(int id,std::vector<uint8_t>& v) {
        v.assign(ss::kRawBytes,uint8_t(17+id));uint32_t x=1234567;
        for(size_t i=256;i<v.size();i+=144) for(size_t j=0;j<16;++j) {
            x^=x<<13;x^=x>>17;x^=x<<5;
            v[i+128+j]=id==0?uint8_t(x):(id==1?uint8_t((i/144+j)%31):uint8_t(id));
        }
        std::memcpy(v.data(),&id,sizeof(id));
    }
    TestModel() {
        std::string err;std::vector<uint8_t> v,buf(ss::kRawBytes+4096);
        for(int id=0;id<4;++id){raw(id,v);assert(ss::encode(v.data(),v.size(),buf.data(),buf.size(),sizes[id],err));host_stride=std::max(host_stride,sizes[id]);}
        assert(sizes[0]>sizes[2] && sizes[1]>sizes[2]);
        primary.resize(2*ss::kRawBytes);host.resize(2*host_stride);pool=host.data();native_pool=std::make_unique<Pool>(Pool{host.size()});
        auto& l=L[0];l.res_ids={0,1};l.res_slot={0,1,-1,-1};l.res_base=primary.data();l.host_blob.assign(4,nullptr);
        for(int id=0;id<2;++id){raw(id,v);std::memcpy(primary.data()+id*ss::kRawBytes,v.data(),v.size());}
        for(int id=2;id<4;++id){raw(id,v);size_t n=0;auto* dst=pool+(id-2)*host_stride;assert(ss::encode(v.data(),v.size(),dst,host_stride,n,err));l.host_blob[id]=dst;}
        owned={primary.data()};
    }
#include "../src/mm3/mm3_native_placement.inc"
    void verify() {
        std::vector<uint8_t> want,got(ss::kRawBytes);std::string err;
        std::vector<const uint8_t*> slots;
        for(int id=0;id<4;++id) {
            raw(id,want);const auto& l=L[0];
            if(l.res_slot[id]>=0) {assert(!l.host_blob[id]);assert(std::memcmp(resident_address(l,l.res_slot[id]),want.data(),want.size())==0);}
            else {assert(l.host_blob[id]);slots.push_back(l.host_blob[id]);assert(ss::decode_raw(l.host_blob[id],host_stride,got.data(),got.size(),err));assert(got==want);}
        }
        std::sort(slots.begin(),slots.end());assert(slots.size()==2 && slots[0]==pool && slots[1]==pool+host_stride);
    }
};
int main() {
    // The production report must count actual suffix plans, not both devices.
    {TestModel m;std::string report,err;uint8_t plan=0;
     m.D[0].id=0;m.D[0].owner_plan_h=&plan;m.D[0].win_plan_d=&plan;
     Dev prefix;prefix.id=1;prefix.layers.clear();m.D.insert(m.D.begin(),prefix);
     m.head_dev=1;m.L[0].dev=1;
     assert(m.placement({},false,report,err));m.verify();
     assert(report.find("\"codec_cuda_device\":0")!=std::string::npos);
     assert(report.find("\"window_cpu_bytes\":0")!=std::string::npos);
     assert(report.find("\"owner_plan_host_bytes\":266240")!=std::string::npos);
     assert(report.find("\"owner_plan_device_bytes\":4096")!=std::string::npos);
    }
    {TestModel m;std::string report,err;std::vector<uint32_t> a{0,9,10,0};
     assert(m.placement(a,true,report,err));m.verify();assert(m.L[0].res_slot[2]>=0);
     auto& d=m.D[0];d.n_slots=2;d.slot_owner={7,8};d.slot_pend={9,-1};d.slot_used={2,2};
     d.slot_foreground={1,1};d.slot_iss={1,0};d.n_pending=1;d.promo_q={7};
     assert(m.placement(a,true,report,err));m.verify();assert(report.find("\"primary_changed\":0")!=std::string::npos);
     assert(d.n_pending==0 && d.promo_q.empty() && d.slot_owner[0]==-1 && d.slot_pend[0]==-1 && !d.slot_foreground[0]);
     std::vector<uint32_t>b{20,0,0,10};assert(m.placement(b,true,report,err));m.verify();
    }
    {TestModel m;std::string report,err;auto source=m.L[0].host_blob[2];
     std::vector<uint8_t> before(source,source+m.sizes[2]),resident(m.primary.begin(),m.primary.begin()+ss::kRawBytes);
     m.host_stride=m.sizes[2];copy_calls=0;std::vector<uint32_t> a{0,9,10,0};
     assert(!m.placement(a,true,report,err));assert(!m.placement_failed && !m.retirement_uncertain);
     assert(std::memcmp(source,before.data(),before.size())==0 && std::memcmp(m.primary.data(),resident.data(),resident.size())==0);
     assert(m.L[0].host_blob[2]==source && m.L[0].res_slot[0]>=0 && copy_calls==1);
    }
    {TestModel m;std::string report,err;std::vector<uint32_t> a{0,9,10,0};copy_calls=0;fail_copy=2;
     assert(!m.placement(a,true,report,err));assert(m.placement_failed && m.retirement_uncertain);fail_copy=0;
    }
    std::puts("production compressed placement: OK");
}
