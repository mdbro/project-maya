// Controlled operations around the exact resident loader loop and destructor gate.
#include "../src/mm3/mm3_resident_loader.hpp"
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace mm3::r2;
struct Fake {
    enum Fault { none, invalid_record, copy, record, wait, create, join } fault=none;
    bool force_join_failure=false;
    int alloc=0, freed=0, created=0, destroyed=0, assembled=0, copies=0, drops=0, joins=0, records=0;
    std::vector<uint8_t*> allocations;
    std::vector<int*> events;
    ResidentOps ops() {
        ResidentOps o;
        o.set_device=[]{};
        o.allocate_pinned=[&](uint8_t** p,size_t n) { *p=(uint8_t*)std::malloc(n); assert(*p); allocations.push_back(*p); ++alloc; return true; };
        o.create_event=[&](int b)->void* { if(fault==create && b==1)throw std::runtime_error("event create"); auto* p=new int(1); events.push_back(p); ++created; return p; };
        o.record_event=[&](void*) { if(fault==record && ++records==3) throw std::runtime_error("event submission"); };
        o.wait_event=[&](void*) { if(fault==wait) throw std::runtime_error("event wait"); };
        o.assemble=[&](const ResidentJob&,uint8_t*) { ++assembled; if(fault==invalid_record) throw std::runtime_error("invalid record"); };
        o.copy=[&](const ResidentJob&,uint8_t*,bool async) { assert(async); ++copies; if(fault==copy) throw std::runtime_error("copy submission"); };
        o.drop_source=[&](const ResidentJob&) { ++drops; };
        o.join=[&] { ++joins; return fault!=join && !force_join_failure; };
        o.free_pinned=[&](uint8_t* p) { ++freed; for(auto& a:allocations) if(a==p) a=nullptr; std::free(p); };
        o.destroy_event=[&](void* p) { ++destroyed; for(auto& e:events) if(e==p)e=nullptr; delete static_cast<int*>(p); };
        return o;
    }
    ~Fake() { for(auto* p:allocations) std::free(p); for(auto* e:events) delete e; }
};
struct Owner { bool* destroyed; ~Owner(){*destroyed=true;} };
int main() {
    const int ids[2]={1,2};
    for(auto fault:{Fake::none,Fake::invalid_record,Fake::copy,Fake::record,Fake::wait,Fake::create,Fake::join}) {
        Fake f; f.fault=fault;
        std::vector<ResidentBatch> batches{{nullptr,16,{{&ids[0],0},{&ids[1],1}}}};
        std::atomic<bool> uncertain{false}; std::string error;
        auto start=[](std::vector<std::thread>& t,auto fn){ t.emplace_back(std::move(fn)); };
        const bool loaded=load_residents(batches,[&](auto&){return f.ops();},start,uncertain,error);
        assert(loaded==(fault==Fake::none));
        assert(f.joins==1 && f.alloc==2 && f.created==(fault==Fake::create?1:2));
        if(fault==Fake::join) {
            assert(uncertain && f.freed==0 && f.destroyed==0);
            bool destroyed=false; auto* owner=new Owner{&destroyed};
            std::vector<int> devices{0};
            destroy_after_retire(owner,[&]{return retire(uncertain.load(),devices,[](int){return true;});},
                                  [](Owner* p){delete p;},[]{});
            assert(!destroyed); // whole production owner remains reachable
            delete owner; // controlled fixture has no real CUDA work
        } else {
            assert(!uncertain && f.freed==2 && f.destroyed==(fault==Fake::create?1:2));
            if(fault!=Fake::none) assert(!error.empty());
            if(fault==Fake::invalid_record) assert(error.find("invalid record")!=std::string::npos && f.copies==0);
            if(fault==Fake::copy) assert(f.drops==0);
        }
    }
    // A validation exception followed by failed retirement keeps both staging
    // and the entire owner; the original validation error survives.
    {
        Fake f; f.fault=Fake::invalid_record; f.force_join_failure=true;
        std::vector<ResidentBatch> batches{{nullptr,16,{{&ids[0],0}}}};
        std::atomic<bool> uncertain{false}; std::string error;
        auto start=[](std::vector<std::thread>& t,auto fn){t.emplace_back(std::move(fn));};
        assert(!load_residents(batches,[&](auto&){return f.ops();},start,uncertain,error));
        assert(uncertain && f.freed==0 && f.destroyed==0 && error.find("invalid record")!=std::string::npos);
    }
    // Failure creating a later worker joins the already-started worker, reports
    // the error, and does not publish a partially loaded model.
    {
        Fake f; std::vector<ResidentBatch> batches{{nullptr,16,{{&ids[0],0}}},{nullptr,16,{{&ids[1],0}}}};
        std::atomic<bool> uncertain{false}; std::string error; int starts=0;
        auto start=[&](std::vector<std::thread>& t,auto fn){if(++starts==2)throw std::runtime_error("start injected");t.emplace_back(std::move(fn));};
        assert(!load_residents(batches,[&](auto&){return f.ops();},start,uncertain,error));
        assert(starts==2 && f.joins==1 && f.drops==1 && error.find("start injected")!=std::string::npos);
    }
}
