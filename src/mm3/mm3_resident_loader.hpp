#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mm3::r2 {
// Opaque record identity and staging operations keep the production loop itself
// independent of CUDA so tests can inject failures at every ownership boundary.
struct ResidentJob { const void* layer; size_t slot; };
struct ResidentBatch { void* device; size_t max_bytes; std::vector<ResidentJob> jobs; };
struct ResidentOps {
    std::function<void()> set_device;
    std::function<bool(uint8_t**,size_t)> allocate_pinned;
    std::function<void*(int)> create_event;
    std::function<void(void*)> record_event, wait_event;
    std::function<void(const ResidentJob&,uint8_t*)> assemble;
    std::function<void(const ResidentJob&,uint8_t*,bool)> copy;
    std::function<void(const ResidentJob&)> drop_source;
    std::function<bool()> join;
    std::function<void(uint8_t*)> free_pinned;
    std::function<void(void*)> destroy_event;
};

template<class MakeOps, class Start>
bool load_residents(std::vector<ResidentBatch>& batches, MakeOps make_ops, Start start,
                    std::atomic<bool>& uncertain, std::string& error) {
    std::vector<std::thread> threads;
    std::mutex mu;
    std::string worker_error;
    auto report = [&](const std::string& text) {
        std::lock_guard<std::mutex> lock(mu);
        if (worker_error.empty()) worker_error=text;
    };
    try {
        for (auto& batch : batches) start(threads, [&, bp=&batch] {
            ResidentOps ops;
            try { ops=make_ops(*bp); }
            catch(const std::exception& e) { report(e.what()); return; }
            catch(...) { report("unknown resident worker setup failure"); return; }
            uint8_t* pinned[2]={nullptr,nullptr};
            void* events[2]={nullptr,nullptr};
            std::vector<uint8_t>* pageable=nullptr;
            bool device_set=false;
            std::string failure;
            try {
                ops.set_device(); device_set=true;
                if (bp->max_bytes) {
                    bool all_pinned=true;
                    for(int b=0;b<2;++b) {
                        if (!ops.allocate_pinned(&pinned[b],bp->max_bytes)) all_pinned=false;
                        events[b]=ops.create_event(b);
                        ops.record_event(events[b]);
                    }
                    if(!all_pinned) pageable=new std::vector<uint8_t>(bp->max_bytes);
                    int n=0;
                    for(const auto& job:bp->jobs) {
                        uint8_t* source;
                        if(all_pinned) {
                            const int b=n&1;
                            ops.wait_event(events[b]); source=pinned[b];
                            ops.assemble(job,source);
                            ops.copy(job,source,true);
                            ops.record_event(events[b]);
                        } else {
                            source=pageable->data();
                            ops.assemble(job,source);
                            ops.copy(job,source,false);
                        }
                        ops.drop_source(job); ++n;
                    }
                }
            } catch(const std::exception& e) { failure=e.what(); }
              catch(...) { failure="unknown resident load worker failure"; }
            bool retired=false;
            try { retired=device_set && ops.join(); } catch (...) { retired=false; }
            if (!retired) {
                uncertain.store(true);
                if(failure.empty()) failure="resident stream retirement failed";
                // Event, pinned and pageable memory remains owned by CUDA.
            } else {
                delete pageable;
                for(int b=0;b<2;++b) {
                    if(pinned[b]) ops.free_pinned(pinned[b]);
                    if(events[b]) ops.destroy_event(events[b]);
                }
            }
            if(!failure.empty()) report(failure);
        });
    } catch(const std::exception& e) { report(std::string("resident worker start: ")+e.what()); }
      catch(...) { report("unknown resident worker start failure"); }
    for(auto& thread:threads) thread.join();
    if(worker_error.empty()) return true;
    error="copying the resident experts failed: "+worker_error;
    return false;
}
template<class Pointer, class Retire, class Dispose, class Retain>
void destroy_after_retire(Pointer p, Retire retire, Dispose dispose, Retain retain) {
    if (!p) return;
    if (retire()) dispose(p); else retain();
}
}
