// Compile with mm3_native_helper_fake.cpp, MM3_HELPER_TEST_HOOKS and a short
// MM3_HELPER_TEST_DEADLINE_MS. Exercises the production helper transitions.
#include "../src/mm3/mm3_native_helper.hpp"
#include "../src/mm3/mm3_background_traffic.hpp"
#include "../src/mm3/mm3_host_resort.hpp"
#include "cuda_runtime_api.h"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace infernix::ops::minimax_ffn {
size_t record_bytes(int,int) { return 144; }
size_t record_stride(int,int) { return 144; }
size_t fast_workspace_bytes(int,int,int) { return 256; }
void fast_ffn_cuda_packed(const FastJob*,int,int,int,int,void*,cudaStream_t) {}
}
static std::atomic<bool> block_cpu{false}, entered_cpu{false}, fail_cpu{false};
static std::atomic<bool> block_export{false}, block_upload{false}, entered_export{false}, entered_upload{false};
static std::atomic<bool> block_worker_start{false}, entered_worker_start{false}, fail_worker_start{false}, exited_worker{false};
namespace mm3 {
void native_helper_test_cpu_copy(bool) {
    entered_cpu=true;
    while (block_cpu) std::this_thread::yield();
    if (fail_cpu) throw std::runtime_error("injected CPU copy failure");
}
void native_helper_test_gpu_copy(bool upload) {
    if (upload) { entered_upload=true; while(block_upload) std::this_thread::yield(); }
    else { entered_export=true; while(block_export) std::this_thread::yield(); }
}
void native_helper_test_worker_start() {
    entered_worker_start=true;
    while (block_worker_start) std::this_thread::yield();
    if (fail_worker_start) throw std::runtime_error("injected helper worker startup failure");
}
void native_helper_test_worker_exit() { exited_worker=true; }
}
using H=mm3::NativeHelper;
using P=H::LivePhase;
static std::vector<unsigned char> record(unsigned char code) {
    std::vector<unsigned char> x(144,code); std::fill(x.begin()+128,x.end(),0x38); return x;
}
static H* setup() {
    auto* h=new H(0,2); auto b=record(0x22), c=record(0x33);
    float w[3]={1,2,3},i[3]={4,5,6}; std::string e;
    assert(h->add_expert(1,3,b.data(),b.size(),w,i,e));
    assert(h->add_expert(2,4,c.data(),c.size(),w,i,e));
    assert(h->seal(e)); return h;
}
static H::LiveStatus await(H& h,P phase) {
    auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    H::LiveStatus s;
    do { s=h.live_poll(); if(s.phase==phase) return s; std::this_thread::yield(); }
    while(std::chrono::steady_clock::now()<until && s.phase!=P::failed);
    assert(false && "unexpected live phase or timeout"); return s;
}
static bool begin(H& h,std::vector<unsigned char>& host,std::vector<unsigned char>& in,
                  std::vector<unsigned char>& out,std::string& e) {
    float w[3]={7,8,9},i[3]={10,11,12},ow[3]{},oi[3]{};
    bool ok=h.live_begin(1,3,5,6,host.data(),in.data(),out.data(),host.size(),w,i,ow,oi,e);
    if (ok) assert(ow[0]==1 && ow[2]==3 && oi[0]==4 && oi[2]==6);
    return ok;
}
int main() {
    std::string e; bool safe=false;
    // The direct 3090 route uses a spare slot and no CPU payload copier. Old
    // compute remains legal until the complete two-direction transaction is
    // published at a joined boundary; the old slot becomes the next spare.
    {
        H h(0,3);assert(h.reserve_direct_spare(e));
        auto old=record(0x22),other=record(0x33),incoming=record(0x11),out=record(0);
        float w[3]={1,2,3},ig[3]={4,5,6},nw[3]={7,8,9},ni[3]={10,11,12},ow[3]{},oi[3]{};
        assert(h.add_expert(1,3,old.data(),old.size(),w,ig,e));
        assert(h.add_expert(2,4,other.data(),other.size(),w,ig,e));
        assert(!h.add_expert(3,7,other.data(),other.size(),w,ig,e));
        assert(h.seal(e));assert(h.direct_spare()==2);
        mm3::BackgroundTraffic budget;entered_cpu=false;
        h.live_x4_window(UINT64_MAX);const auto generation=h.live_x4_revoke();
        assert(h.direct_begin(1,3,5,6,incoming.data(),out.data(),incoming.size(),nw,ni,ow,oi,&budget,0,true,true,e));
        bool same=false,unsafe=false;
        assert(!h.verify_expert(1,3,old.data(),old.size(),w,ig,same,e));
        assert(!h.rewrite_expert(1,3,other.data(),other.size(),w,ig,unsafe,e));
        assert(!h.direct_publish(e));
        assert(h.ready(1)[0]==(1u<<3) && !h.ready(5)[0]);
        assert(h.live_x4_publish(generation,{},e));await(h,P::prepared);
        assert(!entered_cpu && incoming==record(0x11) && out==old && budget.used()==0);
        assert(ow[0]==1 && oi[2]==6 && h.ready(1)[0]==(1u<<3));
        std::vector<uint16_t> x(H::kHidden);std::vector<float> y(H::kHidden);
        H::Job j{};j.layer=1;j.expert=3;j.columns=1;j.weight[0]=1;
        auto seq=h.submit(1,x.data(),{j},e);assert(seq && !h.direct_publish(e));
        assert(h.collect(seq,y.data(),y.size(),e));assert(h.direct_publish(e));
        assert(h.direct_spare()==0 && !h.ready(1)[0] && h.ready(5)[0]==(1u<<6));
        assert(h.live_retire(safe,e) && safe);
        std::vector<unsigned char> back(144);
        assert(h.export_expert(5,6,back.data(),back.size(),ow,oi,unsafe,e));
        assert(back==incoming && ow[0]==7 && oi[2]==12);
        const auto s=h.live_schedule_stats();
        assert(s.direct_h2d_bytes==144 && s.direct_d2h_bytes==144 && s.direct_chunks==1);
        assert(s.host_pieces==0 && s.verified==2);
    }
    // A copy cache needs no victim download or spare. Withdrawal precedes the
    // overwrite, while unrelated records and their compute stay available.
    {
        H* h=setup();auto incoming=record(0x44);float w[3]={1,2,3},ig[3]={4,5,6},ow[3]{},oi[3]{};
        mm3::BackgroundTraffic budget;entered_cpu=false;h->live_x4_window(UINT64_MAX);
        const auto generation=h->live_x4_revoke();
        assert(h->direct_begin(1,3,5,6,incoming.data(),nullptr,incoming.size(),w,ig,ow,oi,&budget,1,false,true,e));
        assert(!h->ready(1)[0] && h->ready(2)[0]==(1u<<4));
        assert(h->live_x4_publish(generation,{},e));await(*h,P::prepared);
        assert(!entered_cpu && h->direct_publish(e) && h->live_retire(safe,e) && safe);
        const auto s=h->live_schedule_stats();assert(s.direct_h2d_bytes==144 && s.direct_d2h_bytes==0 && s.host_pieces==0);
        delete h;
    }
    // The independent HOST preparer honors foreground CPU ownership and an
    // actually pending upload event. Cancelling while closed touches no bytes.
    {
        std::unique_ptr<H> h(setup()),third(setup());mm3::BackgroundTraffic budget;
        std::vector<unsigned char> a(300000,0x21),b(300000,0x43),to_a(a.size(),0),to_b(b.size(),0);
        mm3::HostResorter worker(0,budget,*h,*third);
        const auto token=worker.guard.revoke().token;worker.cpu_busy=true;
        assert(worker.begin(to_a.data(),a.data(),to_b.data(),b.data(),a.size(),true));
        const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(10);
        while(std::chrono::steady_clock::now()<until)std::this_thread::yield();
        assert(worker.chunks==0 && worker.state()==mm3::HostResorter::running);
        fake_stall=true;worker.cpu_busy=false;
        assert(worker.guard.publish(token,{reinterpret_cast<void*>(1),nullptr,nullptr,nullptr}));
        const auto query_until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!worker.event_queries && std::chrono::steady_clock::now()<query_until)std::this_thread::yield();
        assert(worker.event_queries && worker.chunks==0);fake_stall=false;
        while(worker.state()==mm3::HostResorter::running && std::chrono::steady_clock::now()<query_until)std::this_thread::yield();
        assert(worker.state()==mm3::HostResorter::complete && to_a==a && to_b==b);
        assert(worker.copied_bytes==a.size()*2 && worker.chunks==3 && !budget.used());
        worker.published();(void)worker.guard.revoke();
        assert(worker.begin(to_b.data(),a.data(),to_a.data(),b.data(),a.size(),false));
        assert(worker.stop_and_retire() && worker.chunks==3 && to_a==a && to_b==b);
    }
    // An accepted transfer with an unprovable completion poisons the common
    // budget and retains its admitted storage instead of publishing new weights.
    {
        auto* h=new H(0,2);assert(h->reserve_direct_spare(e));auto old=record(0x22);
        float w[3]={1,2,3},ig[3]={4,5,6},ow[3]{},oi[3]{};
        assert(h->add_expert(1,3,old.data(),old.size(),w,ig,e) && h->seal(e));
        auto* incoming=new std::vector<unsigned char>(record(0x11));auto* out=new std::vector<unsigned char>(record(0));
        auto* budget=new mm3::BackgroundTraffic;
        h->live_x4_window(UINT64_MAX);fake_stall=true;fake_fail_record=true;
        assert(h->direct_begin(1,3,5,6,incoming->data(),out->data(),incoming->size(),w,ig,ow,oi,budget,0,true,false,e));
        await(*h,P::failed);
        assert(!budget->healthy() && budget->used()==288 && !budget->acquire(1,1));
        assert(!h->direct_publish(e) && !h->live_retire(safe,e) && !safe);
        delete h;fake_stall=false;fake_fail_record=false; // unretired borrowed storage intentionally retained
    }
    // Operation queued while scheduling is off, but permission already revoked
    // for enabling the guard: it must enter the chunk path after publication,
    // rather than issuing an old full-record call behind the new foreground work.
    {
        H* h=setup();auto host=record(0x11),in=record(0),out=record(0);
        const auto generation=h->live_x4_revoke();assert(begin(*h,host,in,out,e));
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!h->live_schedule_stats().guard_failed_claims && std::chrono::steady_clock::now()<until)std::this_thread::yield();
        assert(h->live_schedule_stats().guard_failed_claims && h->live_schedule_stats().x4_pieces==0);
        h->set_live_schedule(1);h->live_x4_window(UINT64_MAX);h->live_host_window(0,UINT64_MAX);
        assert(h->live_x4_publish(generation,{},e));await(*h,P::prepared);
        assert(h->live_schedule_stats().x4_pieces==1 && out==record(0x22));
        assert(h->live_withdraw(e));await(*h,P::uploaded);assert(h->live_publish(e));
        assert(h->live_recycle(e));await(*h,P::complete);assert(h->live_retire(safe,e)&&safe);delete h;
    }
    // Cancellation of a worker waiting on a closed generation remains bounded
    // and issues no payload. The borrowed buffers retire through the old lease.
    {
        H* h=setup();auto host=record(0x11),in=record(0),out=record(0);
        h->set_live_schedule(1);h->live_x4_window(UINT64_MAX);h->live_host_window(0,UINT64_MAX);
        (void)h->live_x4_revoke();assert(begin(*h,host,in,out,e));
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!h->live_schedule_stats().guard_failed_claims && std::chrono::steady_clock::now()<until)std::this_thread::yield();
        assert(h->live_schedule_stats().guard_failed_claims);
        do {h->live_cancel(safe,e);if(safe)break;std::this_thread::yield();}while(std::chrono::steady_clock::now()<until);
        assert(safe && h->live_schedule_stats().x4_pieces==0);delete h;
    }
    // The actual copy worker treats a guard-event query error as a failed
    // lease, not permission to issue the optional copy. No graph/compute path
    // is needed for this CPU failure fixture; CUDA handles are retained fakes.
    {
        H* h=setup();auto host=record(0x11),in=record(0),out=record(0);
        h->set_live_schedule(1);h->live_x4_window(UINT64_MAX);h->live_host_window(0,UINT64_MAX);
        const auto generation=h->live_x4_revoke();
        assert(h->live_x4_publish(generation,{reinterpret_cast<void*>(1),nullptr,nullptr,nullptr},e));
        fake_fail_query=true;assert(begin(*h,host,in,out,e));
        const auto status=await(*h,P::failed);fake_fail_query=false;
        assert(status.error.find("quiet upload guard")!=std::string::npos);
        const auto counts=h->live_schedule_stats();
        assert(counts.guard_event_errors==1 && counts.guard_admissions==0 && counts.x4_pieces==0);
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        do {h->live_cancel(safe,e);if(safe)break;std::this_thread::yield();}while(std::chrono::steady_clock::now()<until);
        assert(safe && h->ready(5)[0]==0);delete h;
    }
    // Queue-only startup retains refresh/roster ownership but reserves no
    // worker compute I/O or workspace. Snapshot must follow live publication.
    {
        H h(0,2,false); auto b=record(0x22), host=record(0x11),in=record(0),out=record(0);
        float w[3]={1,2,3},ig[3]={4,5,6};assert(h.add_expert(1,3,b.data(),b.size(),w,ig,e));assert(h.seal(e));
        assert(!h.worker_compute_available()&&h.bytes().device_scratch==0&&h.bytes().pinned==0);
        std::array<H::QueueIdentity,128> snap{};assert(h.queue_snapshot(1,snap,e));
        assert(snap[3].slot>=0&&snap[3].multiplier[0]==1.f&&snap[3].multiplier[1]==.5f&&snap[4].slot<0);
        H::Job j{};j.layer=1;j.expert=3;j.columns=1;j.weight[0]=1;
        std::vector<uint16_t> x(H::kHidden);assert(!h.submit(1,x.data(),{j},e));
        assert(begin(h,host,in,out,e));await(h,P::prepared);assert(h.live_withdraw(e));
        assert(h.queue_snapshot(1,snap,e)&&snap[3].slot<0);await(h,P::uploaded);assert(h.live_publish(e));
        assert(h.queue_snapshot(5,snap,e)&&snap[6].slot>=0&&snap[6].multiplier[0]==1.f/7.f);
        assert(h.live_recycle(e));await(h,P::complete);assert(h.live_retire(safe,e)&&safe);
    }
    {
        H* h=setup(); auto host=record(0x11),in=record(0),out=record(0);
        const auto pool=h->pool_identity(); block_cpu=true; entered_cpu=false;
        assert(begin(*h,host,in,out,e));
        while(!entered_cpu) std::this_thread::yield();
        assert(h->live_poll().phase==P::preparing && h->ready(1)[0]==(1u<<3));
        assert(!h->live_withdraw(e) && !h->live_retire(safe,e) && !safe);
        float w[3],i[3]; bool unsafe=false;
        assert(!h->export_expert(1,3,out.data(),out.size(),w,i,unsafe,e));
        // The required compute worker is independent of the blocked CPU copy.
        std::vector<uint16_t> x(H::kHidden); std::vector<float> y(H::kHidden);
        H::Job j{}; j.layer=1;j.expert=3;j.columns=1;j.weight[0]=1;
        auto seq=h->submit(1,x.data(),{j},e); assert(seq);
        block_cpu=false; await(*h,P::prepared);
        assert(in==record(0x11) && out==record(0x22));
        assert(!h->live_retire(safe,e) && !safe && h->live_poll().storage_unsafe);
        assert(!h->live_withdraw(e)); // active pass must be collected first
        assert(h->collect(seq,y.data(),y.size(),e));
        fake_stall=true;
        assert(h->live_withdraw(e));
        assert(h->ready(1)[0]==0 && h->ready(5)[0]==0);
        assert(!h->live_publish(e));
        fake_stall=false; await(*h,P::uploaded);
        assert(!h->live_retire(safe,e) && !safe && h->live_poll().storage_unsafe);
        assert(h->ready(5)[0]==0 && h->live_publish(e));
        assert(!h->live_retire(safe,e) && !safe && h->live_poll().storage_unsafe);
        assert(h->ready(5)[0]==(1u<<6) && h->ready(2)[0]==(1u<<4));
        block_cpu=true; entered_cpu=false; assert(h->live_recycle(e));
        while(!entered_cpu) std::this_thread::yield();
        assert(!h->live_retire(safe,e) && !safe && host==record(0x11));
        block_cpu=false; await(*h,P::complete);
        assert(host==record(0x22) && h->live_retire(safe,e) && safe);
        assert(h->pool_identity()==pool);
        assert(h->export_expert(5,6,in.data(),in.size(),w,i,unsafe,e));
        assert(in==record(0x11) && w[0]==7 && w[2]==9 && i[0]==10 && i[2]==12);
        delete h;
    }
    // Gate each optional operation independently from the CUDA compute stream.
    // B stays readable during export; after withdrawal C executes during upload
    // and again while the separate CPU worker is recycling outgoing B.
    {
        H* h=setup(); auto host=record(0x11),in=record(0),out=record(0);
        std::vector<uint16_t> x(H::kHidden); std::vector<float> y(H::kHidden);
        H::Job b{}; b.layer=1;b.expert=3;b.columns=1;b.weight[0]=1;
        H::Job c{}; c.layer=2;c.expert=4;c.columns=1;c.weight[0]=1;
        entered_export=false; block_export=true;
        assert(begin(*h,host,in,out,e)); while(!entered_export) std::this_thread::yield();
        assert(h->ready(1)[0]==(1u<<3));
        auto seq=h->submit(1,x.data(),{b},e); assert(seq && h->collect(seq,y.data(),y.size(),e));
        assert(h->live_poll().phase==P::preparing);
        block_export=false; await(*h,P::prepared);
        entered_upload=false; block_upload=true; assert(h->live_withdraw(e));
        while(!entered_upload) std::this_thread::yield();
        assert(h->ready(1)[0]==0 && h->ready(5)[0]==0);
        seq=h->submit(1,x.data(),{c},e); assert(seq && h->collect(seq,y.data(),y.size(),e));
        assert(h->live_poll().phase==P::uploading && !h->live_publish(e));
        block_upload=false; await(*h,P::uploaded); assert(h->live_publish(e));
        entered_cpu=false; block_cpu=true; assert(h->live_recycle(e));
        while(!entered_cpu) std::this_thread::yield();
        seq=h->submit(1,x.data(),{c},e); assert(seq && h->collect(seq,y.data(),y.size(),e));
        assert(h->live_poll().phase==P::recycling && !h->live_retire(safe,e) && !safe);
        block_cpu=false; await(*h,P::complete); assert(h->live_retire(safe,e) && safe);
        delete h;
    }
    // An early compute-worker exception may not free the pool while the live
    // stream is blocked. Cancellation ends the lease only after that stream exits.
    {
        block_worker_start=true; entered_worker_start=false; exited_worker=false; fail_worker_start=true;
        H* h=setup(); while(!entered_worker_start) std::this_thread::yield();
        auto host=record(0x11),in=record(0),out=record(0);
        entered_export=false; block_export=true; const int frees_before=fake_frees;
        assert(begin(*h,host,in,out,e)); while(!entered_export) std::this_thread::yield();
        block_worker_start=false; await(*h,P::failed); while(!exited_worker) std::this_thread::yield();
        assert(fake_frees==frees_before && !h->live_cancel(safe,e) && !safe);
        assert(!h->live_withdraw(e) && !h->live_publish(e));
        block_export=false;
        auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!h->live_cancel(safe,e) && std::chrono::steady_clock::now()<limit) std::this_thread::yield();
        assert(safe && fake_frees==frees_before);
        fail_worker_start=false; delete h;
    }
    // Validation and all CUDA failure boundaries poison without publishing A.
    for(int fault=0;fault<5;++fault) {
        H* h=setup(); auto* host=new std::vector<unsigned char>(record(0x11));
        auto* in=new std::vector<unsigned char>(record(0)); auto* out=new std::vector<unsigned char>(record(0));
        fake_not_portable=(fault==0); fake_fail_copy=(fault==1);
        fake_fail_record=(fault==2); fake_fail_query=(fault==3); fail_cpu=(fault==4);
        if(fault>=1 && fault<=3) fake_stall=true;
        assert(begin(*h,*host,*in,*out,e)); await(*h,P::failed);
        // Allow both independent workers to reach their completion reports.
        auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        do { h->live_retire(safe,e); if(safe) break; std::this_thread::yield(); }
        while(std::chrono::steady_clock::now()<limit && fault!=1 && fault!=2 && fault!=3);
        if(fault>=1 && fault<=3) assert(!safe); // failed stream cannot prove retirement
        assert(h->ready(5)[0]==0 && h->records().empty());
        delete h;
        fake_not_portable=false; fake_fail_copy=false; fake_fail_record=false; fake_fail_query=false;
        fake_stall=false; fail_cpu=false;
        if(safe) { delete host; delete in; delete out; } // otherwise deliberately retained
    }
    // Upload failure occurs after B has been withdrawn; it must not publish A.
    {
        H* h=setup(); auto* host=new std::vector<unsigned char>(record(0x11));
        auto* in=new std::vector<unsigned char>(record(0)); auto* out=new std::vector<unsigned char>(record(0));
        assert(begin(*h,*host,*in,*out,e)); await(*h,P::prepared);
        fake_stall=true; fake_fail_record=true;
        assert(h->live_withdraw(e)); await(*h,P::failed);
        assert(!h->live_publish(e) && !h->live_retire(safe,e) && !safe);
        delete h; fake_stall=false; fake_fail_record=false;
        // The borrowed buffers remain owned because GPU retirement was unproven.
    }
    // A stalled CPU worker also receives a bounded failure. Its buffers cannot
    // be released until that worker actually exits its memcpy/hook.
    {
        H* h=setup(); auto host=record(0x11),in=record(0),out=record(0);
        block_cpu=true; entered_cpu=false; assert(begin(*h,host,in,out,e));
        while(!entered_cpu) std::this_thread::yield();
        await(*h,P::failed); assert(!h->live_retire(safe,e) && !safe);
        block_cpu=false;
        auto limit=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!h->live_retire(safe,e) && std::chrono::steady_clock::now()<limit) std::this_thread::yield();
        assert(safe); delete h;
    }
    // An optional call stuck inside CUDA remains observable and borrowed storage
    // must survive both bounded progress failure and helper destruction.
    {
        H* h=setup(); auto* host=new std::vector<unsigned char>(record(0x11));
        auto* in=new std::vector<unsigned char>(record(0)); auto* out=new std::vector<unsigned char>(record(0));
        fake_block_query=true; int before=fake_queries;
        assert(begin(*h,*host,*in,*out,e));
        while(fake_queries==before) std::this_thread::yield();
        await(*h,P::failed); assert(!h->live_retire(safe,e) && !safe);
        auto start=std::chrono::steady_clock::now(); delete h;
        assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
        assert(host->at(0)==0x11 && in->at(0)==0x11); fake_block_query=false;
        // Intentionally keep the three borrowed buffers for the detached worker.
    }
}
