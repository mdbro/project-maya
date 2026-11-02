// Exercises the production placement method with small, inspectable storage.
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
namespace nf { static size_t record_stride(int,int){return 16;} }
struct NativeHelper {
    struct Identity {int layer,expert,slot;};
    struct Record {int layer,expert;std::array<uint8_t,16> data;std::array<float,3> weight,input;};
    std::vector<Record> r;
    bool fail_export=false,fail_replace=false,poisoned=false;
    std::vector<Identity> records() const { std::vector<Identity> v;if(!poisoned)for(size_t i=0;i<r.size();++i)v.push_back({r[i].layer,r[i].expert,int(i)});return v; }
    std::array<uint32_t,4> ready(int layer) const {std::array<uint32_t,4>a{};if(!poisoned)for(auto& x:r)if(x.layer==layer)a[x.expert/32]|=1u<<(x.expert%32);return a;}
    uintptr_t pool_identity() const{return uintptr_t(this);}
    bool export_expert(int l,int e,void* p,size_t n,float* w,float* x,bool& unsafe,std::string& err){
        unsafe=false;if(fail_export){poisoned=true;err="export";return false;}
        for(auto& a:r)if(a.layer==l&&a.expert==e){assert(n==16);std::memcpy(p,a.data.data(),16);std::copy(a.weight.begin(),a.weight.end(),w);std::copy(a.input.begin(),a.input.end(),x);return true;}return false;
    }
    bool replace_expert(int ol,int oe,int nl,int ne,const void* p,size_t n,const float* w,const float* x,bool& unsafe,std::string& err){
        unsafe=false;if(fail_replace){poisoned=true;err="replace";return false;}
        for(auto& a:r)if(a.layer==ol&&a.expert==oe){assert(n==16);a.layer=nl;a.expert=ne;std::memcpy(a.data.data(),p,16);std::copy_n(w,3,a.weight.begin());std::copy_n(x,3,a.input.begin());return true;}return false;
    }
};
struct Layer {bool moe=true,hf=false;int dev=0;size_t blob=272;std::vector<int> res_slot,res_ids;uint8_t* res_base=nullptr;std::vector<uint8_t*> res_ptr;std::vector<const uint8_t*> host_blob;std::array<uint32_t,4> helper_ready{};unsigned long long* exp_tab=nullptr;};
struct Dev {int id=0,s=1,s2=2,cs=3,n_slots=0;std::vector<int> layers{0,1},promo_q,slot_owner,slot_pend,slot_used,slot_foreground;std::vector<long long> slot_iss;int n_pending=0;};
struct Pool {size_t n;size_t size()const{return n;}size_t capacity()const{return n;}};
struct TestModel {
    struct {int n_layers=2,n_exp=4,dim=1,ff=1;} g;
    struct {int chunk=4,resident_cap=-1;} o;
    struct {int experts=8,pooled=4;} st;
    bool native=true;std::unique_ptr<Pool> native_pool;std::atomic<bool> retirement_uncertain{false};
    std::vector<Layer>L;std::vector<Dev>D;int head_dev=0;std::unique_ptr<NativeHelper>helper;uint8_t* pool=nullptr;
    std::vector<void*>owned;std::unique_ptr<int>draftm;std::string promotion_error;
    std::vector<float>use_score;std::vector<int>use_last;
    std::vector<uint8_t> adaptive_main;size_t adaptive_blob=0;bool primary_adaptive=false;
    std::array<uint8_t*,2> adaptive_host_owned{};
    bool adaptive_host_contains(uintptr_t,size_t) const{return false;}
    bool adaptive_storage_ok(const std::vector<uintptr_t>&,const std::vector<uintptr_t>&) const{return true;}
    bool adaptive_publish(Dev&,std::string&){return true;}
    int adaptive_pending(const Dev&)const{return 0;}
    uint8_t* staging_address(const Dev&,int)const{return nullptr;}
    bool staging_reserved(const Dev&,int)const{return false;}
    std::vector<uint8_t> host,primary0,primary1;
    bool helper_owns(const Layer& l,int e)const{return (l.helper_ready[e/32]>>(e%32))&1;}
    bool baseline_fail=false;
    bool baseline_hf_demand(std::string& error){if(baseline_fail){error="HF baseline injected";return false;}return true;}
    void graphs_clear(){}
    static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
    static void put(uint8_t* p,int id){std::memset(p,0,272);auto* h=(ne::Header*)p;for(int i=0;i<3;++i)h->weight_global[i]=h->input_global[i]=float(id+1);std::memset(p+256,id,16);}
    TestModel():L(2),D(1),helper(new NativeHelper),host(4*272),primary0(2*272),primary1(272){
        native_pool=std::make_unique<Pool>(Pool{host.size()});pool=host.data();
        for(auto& l:L){l.res_slot.assign(4,-1);l.host_blob.assign(4,nullptr);}
        L[0].res_base=primary0.data();L[0].res_ids={0,1};L[0].res_slot[0]=0;L[0].res_slot[1]=1;put(primary0.data(),0);put(primary0.data()+272,1);
        L[1].res_base=primary1.data();L[1].res_ids={0};L[1].res_slot[0]=0;put(primary1.data(),4);
        int ids[]={2,3,6,7};for(int i=0;i<4;++i){put(host.data()+i*272,ids[i]);L[ids[i]/4].host_blob[ids[i]%4]=host.data()+i*272;}
        NativeHelper::Record r{};r.layer=1;r.expert=1;r.data.fill(5);r.weight.fill(6);r.input.fill(6);helper->r.push_back(r);L[1].helper_ready=helper->ready(1);
        owned={primary0.data(),primary1.data()};
    }
    void make_first_layer_full(unsigned long long* table) {
        primary0.resize(4*272);L[0].res_base=primary0.data();owned[0]=primary0.data();
        L[0].res_ids={0,1,2,3};for(int e=0;e<4;++e){L[0].res_slot[e]=e;put(primary0.data()+e*272,e);L[0].host_blob[e]=nullptr;}
        L[0].hf=true;L[0].exp_tab=table;st.pooled=2;
    }
    long long helper_live_a=-1;
    size_t helper_live_blob=0;bool helper_live_enabled=false;
    uint64_t helper_live_started=0,helper_live_finished=0;
    int helper_live_settle_calls=0;
    bool helper_live_settle(std::string&){++helper_live_settle_calls;return true;}
    bool helper_live_leased(long long)const{return false;}
#include "../src/mm3/mm3_native_placement.inc"
    void verify() {
        int owners=0;
        for(int li=0;li<2;++li)for(int e=0;e<4;++e){int id=li*4+e;const auto& l=L[li];const uint8_t* p=nullptr;
            if(l.res_slot[e]>=0){p=resident_address(l,l.res_slot[e]);++owners;}
            if(l.host_blob[e]){assert(!p);p=l.host_blob[e];++owners;}
            if(helper_owns(l,e)){assert(!p);auto it=std::find_if(helper->r.begin(),helper->r.end(),[&](const auto&r){return r.layer==li&&r.expert==e;});assert(it!=helper->r.end());for(auto b:it->data)assert(b==id);++owners;continue;}
            assert(p);assert(p[256]==id);auto* h=(const ne::Header*)p;assert(h->weight_global[0]==float(id+1));
        }assert(owners==8);
    }
};
int main(){
    for(bool apply : {false,true}) {
        TestModel m;std::string report,error;
        m.helper_live_a=2;
        m.D[0].slot_foreground={1};m.D[0].slot_owner={2};
        assert(!mm3::r2::foreground_retire(m.D[0].slot_foreground,m.D[0].slot_owner,m.promotion_error,error,
            [](size_t,std::string& message){message="injected foreground query failure";return mm3::r2::Completion::failed;}));
        const int old_copies=copy_calls;
        std::vector<uint32_t> profile=apply?std::vector<uint32_t>{1,1,1,1,1,1,1,1}:std::vector<uint32_t>{};
        assert(!m.placement(profile,true,report,error));
        assert(m.placement_failed && m.helper_live_settle_calls==0 && copy_calls==old_copies);
        assert(m.D[0].slot_foreground[0]==1 && error==m.promotion_error);
        m.verify();
    }
    {
        TestModel m;const auto host=m.pool;const auto primary=m.owned;const auto hp=m.helper->pool_identity();std::string report,error;
        m.D[0].n_pending=2;m.D[0].slot_pend={1,2};m.D[0].slot_used={20,21};m.D[0].slot_iss={3,4};m.D[0].slot_foreground={1,1};
        std::vector<uint32_t> c{0,0,100,80,0,90,70,1};
        assert(m.placement(c,true,report,error));m.verify();assert(m.pool==host&&m.owned==primary&&m.helper->pool_identity()==hp);
        assert(m.D[0].n_pending==0);
        assert(m.L[0].res_slot[2]>=0&&m.L[1].res_slot[1]>=0&&m.L[0].res_slot[3]>=0);
        assert(report.find("\"ownership_complete\":true")!=std::string::npos);
        std::vector<uint32_t> next{100,90,0,0,80,0,2,1};
        assert(m.placement(next,true,report,error));m.verify();
        assert(m.pool==host&&m.owned==primary&&m.helper->pool_identity()==hp);
    }
    {
        TestModel m;std::string report,error;
        m.D[0].layers={0};Dev d;d.id=1;d.layers={1};m.D.push_back(d);m.L[1].dev=1;m.head_dev=1;
        std::vector<uint32_t> c{0,0,100,80,0,90,70,1};
        assert(m.placement(c,true,report,error));m.verify();
        assert(m.L[0].res_ids.size()==2&&m.L[1].res_ids.size()==1);
        for(auto p:m.L[0].res_ptr)assert(p>=m.primary0.data()&&p<m.primary0.data()+m.primary0.size());
        for(auto p:m.L[1].res_ptr)assert(p>=m.primary1.data()&&p<m.primary1.data()+m.primary1.size());
        for(auto r:m.helper->records())assert(r.layer==1);
    }
    {
        TestModel m;std::string report,error;unsigned long long table[4]{};m.make_first_layer_full(table);
        std::vector<uint32_t> c{100,99,98,97,0,0,90,1};
        assert(m.placement(c,true,report,error));m.verify();assert(m.L[0].hf);
        for(int e=0;e<4;++e)assert(table[e]==uintptr_t(m.resident_address(m.L[0],m.L[0].res_slot[e])));
    }
    {
        TestModel m;std::string report,error;unsigned long long table[4]{};m.make_first_layer_full(table);
        std::vector<uint32_t> c{100,99,98,0,0,90,80,70};
        assert(m.placement(c,true,report,error));m.verify();assert(!m.L[0].hf);
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,10,90,0,0,1,2};
        assert(m.placement(c,false,report,error));m.verify();assert(m.helper_owns(m.L[0],3));
        assert(m.L[0].res_slot[0]>=0&&m.L[0].res_slot[1]>=0&&m.L[1].res_slot[0]>=0);
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,100,80,0,90,70,1};
        fail_copy=2;copy_calls=0;assert(!m.placement(c,true,report,error));assert(m.placement_failed&&m.retirement_uncertain);fail_copy=0;
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,10,90,0,0,1,2};m.helper->fail_export=true;
        assert(!m.placement(c,false,report,error));assert(m.placement_failed);
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,10,90,0,0,1,2};m.helper->fail_replace=true;
        assert(!m.placement(c,false,report,error));assert(m.placement_failed);
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,10,90,0,0,1,2};m.o.resident_cap=1;
        assert(!m.placement(c,true,report,error));assert(!m.placement_failed);m.verify();
        assert(error.find("cannot fill")!=std::string::npos);
    }
    {
        TestModel m;std::string report,error;std::vector<uint32_t> c{0,0,10,90,0,0,1,2};
        m.baseline_fail=true;
        assert(!m.placement(c,false,report,error));
        assert(m.placement_failed && m.retirement_uncertain && error.find("HF baseline")!=std::string::npos);
    }
    std::puts("production native placement: OK");
}
