// Manual real-CUDA guard gate; private synthetic RG16 records, no model state.
// Root runs this only in its coordinated idle GPU window. No timing claim:
// primary probe work proves independent stream progress, not expert latency.
// Deterministic create/record/query faults and exact claim races live in CPU
// fixtures; this gate covers real cross-device queries and production copies.
// Usage: mm3_upload_guard_cuda_test [--primary N] [--helper N]
#include "mm3_native_helper.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using H=mm3::NativeHelper;
using Clock=std::chrono::steady_clock;
constexpr int kLayer=3,kHidden=H::kHidden,kPackedRow=kHidden/2+kHidden/16;
constexpr size_t kProbeBytes=4096,kPiece=512u<<10;
void require(bool ok,const std::string& what) {if(!ok)throw std::runtime_error(what);}
void ck(cudaError_t e,const char* what) {
    if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
#define CUDA(x) ck((x),#x)
int find_card(const char* name) {
    int count=0;CUDA(cudaGetDeviceCount(&count));
    for(int d=0;d<count;++d) {cudaDeviceProp p{};CUDA(cudaGetDeviceProperties(&p,d));if(std::strstr(p.name,name))return d;}
    throw std::runtime_error(std::string("missing GPU ")+name);
}
template<class Ready> void wait_for(Ready&& ready,const char* what,int ms=3000) {
    const auto until=Clock::now()+std::chrono::milliseconds(ms);
    while(!ready()) {require(Clock::now()<until,std::string("timeout: ")+what);std::this_thread::sleep_for(std::chrono::microseconds(100));}
}
bool event_ready(cudaEvent_t event) {
    const auto e=cudaEventQuery(event);if(e==cudaSuccess)return true;
    if(e!=cudaErrorNotReady)ck(e,"query fixture event");return false;
}
__global__ void primary_probe(unsigned* value) {if(threadIdx.x==0 && blockIdx.x==0)++*value;}

// A host callback holds only its own nonblocking stream and makes no CUDA calls.
// Its fail-open timeout is a test failure, never a successful dependency gate.
struct Hold {
    std::mutex mutex;std::condition_variable cv;
    bool started=false,released=false,timed_out=false;
    uint64_t entered_ns=0,released_ns=0;
    static void CUDART_CB callback(void* data) {
        auto& h=*static_cast<Hold*>(data);std::unique_lock<std::mutex> lock(h.mutex);
        h.started=true;h.entered_ns=H::clock_ns();h.cv.notify_all();
        if(!h.cv.wait_for(lock,std::chrono::seconds(3),[&]{return h.released;})) {h.timed_out=true;h.released=true;}
    }
    void await() {
        std::unique_lock<std::mutex> lock(mutex);
        require(cv.wait_for(lock,std::chrono::seconds(2),[&]{return started;}),"held stream callback did not start");
    }
    bool held() {std::lock_guard<std::mutex> lock(mutex);return started&&!released&&!timed_out;}
    void release() {std::lock_guard<std::mutex> lock(mutex);released_ns=H::clock_ns();released=true;cv.notify_all();}
    void check() {std::lock_guard<std::mutex> lock(mutex);require(!timed_out,"held stream auto-released: gate is invalid");}
};
struct Record {std::vector<uint8_t> bytes;std::array<float,3> weight{},input{};};
enum class Case { demand,early,exchange,empty,stale,cpu };
const char* case_name(Case c) {
    switch(c) {case Case::demand:return "held_d_us";case Case::early:return "held_d_cs";
        case Case::exchange:return "held_d_xs";case Case::empty:return "no_upload";
        case Case::stale:return "rerecord_stale_generation";case Case::cpu:return "cpu_busy";}
    return "invalid";
}

// Heap-owned; any uncertain CUDA failure exits without freeing borrowed buffers
// or event handles. A process watchdog bounds even an accidentally blocking API.
struct Fixture {
    int primary,device,current=0,swapped_slot=-1,stable_slot=-1;
    H* helper=nullptr;size_t stride=0;uintptr_t pool=0;H::Bytes allocation{};
    std::array<Record,3> records;
    uint8_t *host=nullptr,*incoming=nullptr,*outgoing=nullptr;
    std::array<cudaStream_t,4> stream{};
    std::array<cudaEvent_t,4> marker{};
    cudaEvent_t origin=nullptr,probe_done=nullptr;
    uint8_t *probe_source=nullptr,*probe_target=nullptr;
    unsigned *counter=nullptr,*counter_back=nullptr;
    unsigned expected_counter=0;
    std::array<std::vector<uint8_t>,2> rows;
    std::array<std::array<std::vector<float>,2>,3> expected;
    uint64_t last_generation=0,probe_observed_ns=0;
    unsigned checks=0;
    Fixture(int p,int d):primary(p),device(d) {}
    void open() {helper->live_x4_window(UINT64_MAX);helper->live_host_window(0,UINT64_MAX);}
    std::array<void*,4> events() const {return {marker[0],marker[1],marker[2],marker[3]};}
    void record_markers() {
        CUDA(cudaSetDevice(primary));for(size_t i=0;i<4;++i)CUDA(cudaEventRecord(marker[i],stream[i]));
    }
    void publish(uint64_t generation) {
        std::string error;require(helper->live_x4_publish(generation,events(),error),"publish guard: "+error);last_generation=generation;
    }
    void ready_markers() {
        const auto generation=helper->live_x4_revoke();record_markers();
        for(auto event:marker)wait_for([&]{return event_ready(event);},"empty marker ready");
        publish(generation);
    }
    void probe() {
        CUDA(cudaSetDevice(primary));
        primary_probe<<<1,32,0,stream[0]>>>(counter);CUDA(cudaGetLastError());
        CUDA(cudaMemcpyAsync(counter_back,counter,sizeof(unsigned),cudaMemcpyDeviceToHost,stream[0]));
        CUDA(cudaEventRecord(probe_done,stream[0]));
        wait_for([&]{return event_ready(probe_done);},"independent primary compute");
        probe_observed_ns=H::clock_ns();require(*counter_back==++expected_counter,"independent primary output mismatch");
    }
    void init() {
        cudaDeviceProp p{},d{};CUDA(cudaGetDeviceProperties(&p,primary));CUDA(cudaGetDeviceProperties(&d,device));
        require(primary!=device&&std::strstr(p.name,"5090")&&std::strstr(d.name,"3090"),"requires distinct primary 5090 and helper 3090");
        CUDA(cudaSetDevice(device));helper=new H(device,2);stride=helper->record_stride();
        helper->set_occupancy(false);helper->set_fast(true);helper->set_zero_copy(1);
        helper->set_kernel(172,68);helper->set_loads(8);helper->set_sm89(false);
        CUDA(cudaHostAlloc(&host,stride,cudaHostAllocPortable));
        CUDA(cudaHostAlloc(&incoming,stride,cudaHostAllocPortable));
        CUDA(cudaHostAlloc(&outgoing,stride,cudaHostAllocPortable));
        std::mt19937 rng(20261116);std::string error;
        for(int id=0;id<3;++id) {
            auto& r=records[id];r.bytes.resize(stride);
            for(auto& b:r.bytes)b=uint8_t(rng());
            // The retained RG16 format consists of 128 code bytes and sixteen
            // finite E4M3 scales per 144-byte unit, plus stride padding.
            for(size_t u=0;u+144<=stride;u+=144)for(size_t k=128;k<144;++k)r.bytes[u+k]=uint8_t(0x28+rng()%24);
            r.weight={2.0f+0.125f*id,2.5f,3.0f};r.input={1.0f+id,2.0f+id,3.0f+id};
            if(id!=1) {std::memcpy(incoming,r.bytes.data(),stride);
                require(helper->add_expert(kLayer,id,incoming,stride,r.weight.data(),r.input.data(),error),"add expert: "+error);}
        }
        require(helper->seal(error),"seal: "+error);require(helper->set_execution(1,error),"execution: "+error);
        helper->set_live_schedule(1);pool=helper->pool_identity();allocation=helper->bytes();
        for(const auto& id:helper->records()) {if(id.expert==0)swapped_slot=id.slot;if(id.expert==2)stable_slot=id.slot;}
        require(pool&&swapped_slot>=0&&stable_slot>=0&&swapped_slot!=stable_slot,"initial slot identities");
        std::memcpy(host,records[1].bytes.data(),stride);
        for(int wi=0;wi<2;++wi) {const int width=wi?3:1;rows[wi].resize(size_t(width)*kPackedRow);
            for(int t=0;t<width;++t) {auto* row=rows[wi].data()+size_t(t)*kPackedRow;
                for(int b=0;b<kHidden/2;++b)row[b]=uint8_t(rng());
                for(int b=0;b<kHidden/16;++b)row[kHidden/2+b]=uint8_t(0x28+rng()%16);}}
        CUDA(cudaSetDevice(primary));
        for(auto& s:stream)CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
        for(auto& e:marker)CUDA(cudaEventCreate(&e));
        CUDA(cudaEventCreate(&origin));CUDA(cudaEventCreate(&probe_done));
        CUDA(cudaHostAlloc(&probe_source,4*kProbeBytes,cudaHostAllocPortable));
        CUDA(cudaHostAlloc(&counter_back,sizeof(unsigned),cudaHostAllocPortable));
        CUDA(cudaMalloc(&probe_target,4*kProbeBytes));CUDA(cudaMalloc(&counter,sizeof(unsigned)));
        for(size_t b=0;b<4*kProbeBytes;++b)probe_source[b]=uint8_t(31+b*17);
        CUDA(cudaMemset(counter,0,sizeof(unsigned)));
        // Prewarm primary modules, all stream copy routes, helper widths/graphs,
        // live workers and both record identities before holding any stream.
        probe();
        for(size_t i=0;i<4;++i)CUDA(cudaMemcpyAsync(probe_target+i*kProbeBytes,probe_source+i*kProbeBytes,kProbeBytes,cudaMemcpyHostToDevice,stream[i]));
        ready_markers();
        for(int wi=0;wi<2;++wi) {expected[0][wi]=compute(0,wi);expected[2][wi]=compute(2,wi);}
        allocation=helper->bytes(); // Q1 workspace and both graph variants are now allocated
        open();begin();finish(false);
        for(int wi=0;wi<2;++wi)expected[1][wi]=compute(1,wi);
        require(expected[0][0]!=expected[1][0],"synthetic replacement outputs must differ");
        open();begin();finish(true);
        require(current==0,"prewarm did not restore starting identity");
        std::printf("PREWARM PASS primary=%d helper=%d pool=%llu stride=%zu slots=%d,%d schedule=1 kernel=172/68 execution=1 zero_copy=1 graph=1\n",
                    primary,device,(unsigned long long)pool,stride,swapped_slot,stable_slot);std::fflush(stdout);
    }
    uint64_t submit(int expert,int wi) {
        const int width=wi?3:1;H::Job job{};job.layer=kLayer;job.expert=expert;job.columns=width;
        for(int c=0;c<width;++c){job.token[c]=c;job.weight[c]=0.125f;}
        std::string error;const auto seq=helper->submit_packed(width,rows[wi].data(),{job},error);
        require(seq!=0,"submit packed: "+error);return seq;
    }
    std::vector<float> collect(uint64_t seq,int wi) {
        const int width=wi?3:1;std::vector<float> out(size_t(width)*kHidden);std::string error;
        wait_for([&]{return helper->done(seq);},"helper graph completion");
        H::PassTiming t;std::vector<H::Refused> refused;
        require(helper->collect(seq,out.data(),out.size(),error,&t,&refused),"collect: "+error);
        require(refused.empty()&&t.input_format==1&&t.kernel==(wi?172:68)&&t.execution==(wi?1:0)&&
                t.multi==bool(wi)&&t.graph&&!t.graph_fallback&&!t.gpu_measured&&!t.sm89,"production graph/format/execution receipt mismatch");
        require(std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);}),"nonfinite helper output");
        require(std::any_of(out.begin(),out.end(),[](float x){return x!=0;}),"zero-only helper output");
        std::printf("ROUTE seq=%llu columns=%d kernel=%d execution=%d format=%d graph=1 zero_copy=%d submit_ns=%llu ready_ns=%llu collect_ns=%llu\n",
                    (unsigned long long)seq,width,t.kernel,t.execution,t.input_format,helper->zero_copy(),
                    (unsigned long long)t.submit_enter_ns,(unsigned long long)t.ready_observed_ns,(unsigned long long)t.collect_done_ns);
        return out;
    }
    std::vector<float> compute(int expert,int wi) {return collect(submit(expert,wi),wi);}
    void exact(const std::vector<float>& got,const std::vector<float>& want,const char* what) {
        require(got.size()==want.size()&&std::memcmp(got.data(),want.data(),got.size()*sizeof(float))==0,what);
    }
    void wait(H::LivePhase desired) {
        wait_for([&]{const auto s=helper->live_poll();require(s.phase!=H::LivePhase::failed,"live helper failed: "+s.error);return s.phase==desired;},"live helper phase",5000);
    }
    void begin() {
        const int next=1-current;const auto& r=records[next];
        require(std::memcmp(host,r.bytes.data(),stride)==0,"HOST incoming identity before begin");
        std::array<float,3> weight{},input{};std::string error;
        require(helper->live_begin(kLayer,current,kLayer,next,host,incoming,outgoing,stride,r.weight.data(),r.input.data(),
                                  weight.data(),input.data(),error),"live begin: "+error);
        require(weight==records[current].weight&&input==records[current].input,"exported original scales");
    }
    void no_copy(const H::LiveScheduleStats& before,const char* what) {
        const auto until=Clock::now()+std::chrono::milliseconds(25);
        do {const auto s=helper->live_schedule_stats();
            require(s.x4_pieces==before.x4_pieces&&s.guard_admissions==before.guard_admissions,std::string("copy/admission while ")+what);
            require(helper->live_poll().phase==H::LivePhase::preparing,std::string("unexpected phase while ")+what);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }while(Clock::now()<until);
    }
    void finish(bool check_output) {
        const int old=current,next=1-current;std::string error;wait(H::LivePhase::prepared);
        require(std::memcmp(incoming,records[next].bytes.data(),stride)==0&&std::memcmp(outgoing,records[old].bytes.data(),stride)==0,"prepared incoming/export payloads");
        require(helper->ready(kLayer)[0]==((1u<<old)|(1u<<2)),"prepared ready identities");
        require(helper->live_withdraw(error),"withdraw: "+error);wait(H::LivePhase::uploaded);
        require(helper->ready(kLayer)[0]==(1u<<2),"uploaded record published before owner transition");
        require(helper->live_publish(error),"publish replacement: "+error);
        require(helper->ready(kLayer)[0]==((1u<<next)|(1u<<2)),"replacement ready identities");
        require(helper->live_recycle(error),"recycle: "+error);wait(H::LivePhase::complete);
        bool safe=false;require(helper->live_retire(safe,error)&&safe,"retire lease: "+error);current=next;
        require(std::memcmp(host,records[old].bytes.data(),stride)==0,"recycled HOST identity");
        const auto bytes=helper->bytes();require(helper->pool_identity()==pool&&bytes.pool==allocation.pool&&
            bytes.device_scratch==allocation.device_scratch&&bytes.pinned==allocation.pinned,"helper allocation identity changed");
        const auto ids=helper->records();require(ids.size()==2,"helper record count changed");
        for(const auto& id:ids)require(id.layer==kLayer&&((id.expert==current&&id.slot==swapped_slot)||(id.expert==2&&id.slot==stable_slot)),"record/frame identity changed");
        std::array<H::QueueIdentity,128> snapshot{};require(helper->queue_snapshot(kLayer,snapshot,error),"queue identity snapshot: "+error);
        for(int i=0;i<3;++i)require(snapshot[current].multiplier[i]==1.0f/records[current].weight[i],"replacement reciprocal scale changed");
        // Readbacks are after lease retirement, outside held/concurrency windows.
        CUDA(cudaSetDevice(device));std::vector<uint8_t> back(stride);
        for(int id:{current,2}) {CUDA(cudaMemcpy(back.data(),reinterpret_cast<const uint8_t*>(pool)+size_t(id==2?stable_slot:swapped_slot)*stride,stride,cudaMemcpyDeviceToHost));
            require(back==records[id].bytes,"retired GPU payload identity mismatch");}
        if(check_output)for(int wi=0;wi<2;++wi) {exact(compute(current,wi),expected[current][wi],"replacement graph output changed");
            exact(compute(2,wi),expected[2][wi],"unchanged graph output changed");}
    }
    void run(Case which) {
        const char* name=case_name(which);const auto before=helper->live_schedule_stats();const int old=current;
        helper->live_cpu_busy(false);open();CUDA(cudaSetDevice(primary));
        CUDA(cudaEventRecord(origin,stream[0]));wait_for([&]{return event_ready(origin);},"case origin");
        const uint64_t start_ns=H::clock_ns();uint64_t generation=helper->live_x4_revoke(),old_generation=0;
        Hold* hold=nullptr;int held=-1;bool stale_rejected=false;
        if(which==Case::demand||which==Case::stale)held=1;
        else if(which==Case::early)held=2;else if(which==Case::exchange)held=3;
        if(which==Case::stale) {
            record_markers();for(auto event:marker)wait_for([&]{return event_ready(event);},"old completed markers");
            publish(generation);old_generation=generation;generation=helper->live_x4_revoke();
        }
        if(held>=0) {
            // A different payload on every held-stream case prevents a prior
            // setup copy from satisfying the post-retirement byte check.
            for(size_t b=0;b<kProbeBytes;++b)probe_source[size_t(held)*kProbeBytes+b]=uint8_t(47+checks*19+b*13);
            hold=new Hold;CUDA(cudaLaunchHostFunc(stream[size_t(held)],Hold::callback,hold));
            CUDA(cudaMemcpyAsync(probe_target+size_t(held)*kProbeBytes,probe_source+size_t(held)*kProbeBytes,kProbeBytes,cudaMemcpyHostToDevice,stream[size_t(held)]));
        }
        record_markers();if(hold) {hold->await();require(hold->held()&&!event_ready(marker[size_t(held)]),"marker did not remain pending");}
        if(which==Case::stale) {
            std::string error;stale_rejected=!helper->live_x4_publish(old_generation,events(),error);
            require(stale_rejected&&!error.empty(),"old generation reopened rerecorded markers");
            begin();no_copy(before,"new generation closed");publish(generation);
        } else {publish(generation);if(which==Case::cpu)helper->live_cpu_busy(true);begin();}
        probe();if(hold)require(hold->held(),"primary probe completed only after upload release");
        if(hold) {
            wait_for([&]{return helper->live_schedule_stats().guard_failed_claims>before.guard_failed_claims;},"worker observed pending marker");
            no_copy(before,"independent marker held");require(hold->held(),"held stream expired during observation");
            hold->release();wait_for([&]{return event_ready(marker[size_t(held)]);},"released upload marker");hold->check();
        } else if(which==Case::cpu) {
            no_copy(before,"CPU busy");helper->live_cpu_busy(false);
        }
        finish(true);
        const auto after=helper->live_schedule_stats();const uint64_t pieces=(stride+kPiece-1)/kPiece;
        require(after.x4_pieces-before.x4_pieces==2*pieces,"piece count for complete replacement");
        require(after.guard_event_errors==before.guard_event_errors&&after.mismatches==before.mismatches&&after.verified==before.verified,"unexpected guard error or mode-2 traffic");
        require(after.guard_admissions-before.guard_admissions>=2*pieces,"missing enabled-guard admissions");
        CUDA(cudaSetDevice(primary));
        if(held>=0) {
            std::array<uint8_t,kProbeBytes> actual{};
            CUDA(cudaMemcpy(actual.data(),probe_target+size_t(held)*kProbeBytes,kProbeBytes,cudaMemcpyDeviceToHost));
            require(std::memcmp(actual.data(),probe_source+size_t(held)*kProbeBytes,kProbeBytes)==0,"retired primary upload payload mismatch");
            std::printf("UPLOAD case=%s source=%llu destination=%llu bytes=%zu exact=1\n",name,
                        (unsigned long long)reinterpret_cast<uintptr_t>(probe_source+size_t(held)*kProbeBytes),
                        (unsigned long long)reinterpret_cast<uintptr_t>(probe_target+size_t(held)*kProbeBytes),kProbeBytes);
        }
        for(size_t i=0;i<4;++i) {wait_for([&]{return event_ready(marker[i]);},"case marker retirement");float ms=0;CUDA(cudaEventElapsedTime(&ms,origin,marker[i]));
            std::printf("ENDPOINT case=%s stream=%zu primary_origin_ms=%.6f\n",name,i,ms);}
        std::printf("CASE PASS name=%s old=%d new=%d generation=%llu old_generation=%llu stale_publication_rejected=%d marker_mask=15 host_start_ns=%llu primary_probe_observed_ns=%llu release_ns=%llu retired_ns=%llu pieces=%llu admissions=%llu failed_claims=%llu stale_claims=%llu revocations=%llu publications=%llu prior_admitted_observations=%llu prior_inflight_observations=%llu\n",
                    name,old,current,(unsigned long long)generation,(unsigned long long)old_generation,int(stale_rejected),
                    (unsigned long long)start_ns,(unsigned long long)probe_observed_ns,(unsigned long long)(hold?hold->released_ns:0),(unsigned long long)H::clock_ns(),
                    (unsigned long long)(after.x4_pieces-before.x4_pieces),(unsigned long long)(after.guard_admissions-before.guard_admissions),
                    (unsigned long long)(after.guard_failed_claims-before.guard_failed_claims),(unsigned long long)(after.guard_stale_claims-before.guard_stale_claims),
                    (unsigned long long)(after.guard_revocations-before.guard_revocations),(unsigned long long)(after.guard_publications-before.guard_publications),
                    (unsigned long long)(after.guard_already_admitted-before.guard_already_admitted),(unsigned long long)(after.guard_already_inflight-before.guard_already_inflight));
        delete hold;++checks;std::fflush(stdout);
    }
    void run_collect() {
        ready_markers();helper->live_cpu_busy(false);open();
        // Submit closes the transfer window synchronously. Do not reopen it
        // until after collect; pass_busy itself may clear at worker completion.
        // live_begin requires no active compute owner, so begin with an already
        // closed window before submitting the independent retained expert.
        helper->live_x4_window(0);const auto before=helper->live_schedule_stats();begin();
        const auto seq=submit(2,1);
        no_copy(before,"helper submitted, window closed");
        exact(collect(seq,1),expected[2][1],"concurrent stable helper graph output");
        no_copy(before,"helper collected, window still closed");
        open();finish(true);const auto after=helper->live_schedule_stats();
        require(after.x4_pieces-before.x4_pieces==2*((stride+kPiece-1)/kPiece)&&after.guard_event_errors==before.guard_event_errors,"collect transition copy receipt");
        std::printf("CASE PASS name=helper_submit_collect sequence=%llu generation=%llu pieces=%llu admissions=%llu retired_ns=%llu explicit_window_reopen=1\n",
                    (unsigned long long)seq,(unsigned long long)last_generation,(unsigned long long)(after.x4_pieces-before.x4_pieces),
                    (unsigned long long)(after.guard_admissions-before.guard_admissions),(unsigned long long)H::clock_ns());
        ++checks;std::fflush(stdout);
    }
    void cleanup() {
        helper->live_x4_guard(nullptr);delete helper;helper=nullptr;
        CUDA(cudaSetDevice(primary));for(auto s:stream)CUDA(cudaStreamSynchronize(s));
        for(auto e:marker)CUDA(cudaEventDestroy(e));CUDA(cudaEventDestroy(origin));CUDA(cudaEventDestroy(probe_done));
        for(auto s:stream)CUDA(cudaStreamDestroy(s));CUDA(cudaFree(counter));CUDA(cudaFree(probe_target));
        CUDA(cudaFreeHost(counter_back));CUDA(cudaFreeHost(probe_source));
        CUDA(cudaFreeHost(host));CUDA(cudaFreeHost(incoming));CUDA(cudaFreeHost(outgoing));
    }
};
} // namespace

int main(int argc,char** argv) {
    // Uses no CUDA and deliberately exits without destructors if a runtime API
    // or a broken dependency blocks. No uncertain allocation is freed early.
    std::thread([]{std::this_thread::sleep_for(std::chrono::seconds(120));std::fputs("upload guard CUDA gate: overall timeout\n",stderr);std::fflush(stderr);std::_Exit(2);}).detach();
    try {
        int primary=-1,helper=-1;
        for(int i=1;i<argc;++i) {const std::string a=argv[i];
            if(a=="--primary"&&i+1<argc)primary=std::stoi(argv[++i]);
            else if(a=="--helper"&&i+1<argc)helper=std::stoi(argv[++i]);
            else throw std::runtime_error("usage: mm3_upload_guard_cuda_test [--primary N] [--helper N]");}
        if(primary<0)primary=find_card("5090");if(helper<0)helper=find_card("3090");
        auto* fixture=new Fixture(primary,helper);fixture->init();
        for(auto c:{Case::demand,Case::early,Case::exchange,Case::empty,Case::stale,Case::cpu})fixture->run(c);
        fixture->run_collect();const auto checks=fixture->checks;fixture->cleanup();delete fixture;
        std::printf("RESULT PASS cases=%u; real guard/helper ordering and exact payload identity; no model performance claim\n",checks);return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"RESULT FAIL upload guard CUDA gate: %s\n",e.what());std::fflush(stderr);std::_Exit(1);
    }
}
