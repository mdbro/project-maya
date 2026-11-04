#include "mm3_capture_stub.hpp"
// Literal production bridge and slot publication with the real CUDA Service.
// Each fault case exits its own process, retaining quarantined resources.
#include "mm3/mm3_native_scale_store.hpp"
#include "mm3/mm3_r2_lifecycle.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
namespace ss=mm3::scale_store;namespace r2=mm3::r2;
static bool isolated_child=false;
static void req(bool v,const std::string& why){if(!v){if(isolated_child){std::fprintf(stderr,"bridge child: %s\n",why.c_str());std::fflush(nullptr);_Exit(1);}throw std::runtime_error(why);}}
static void ck(cudaError_t rc){req(rc==cudaSuccess,cudaGetErrorString(rc));}
static bool fail_query=false,fail_join=false;static int query_calls=0;
static cudaError_t test_query(cudaEvent_t e){++query_calls;return fail_query?cudaErrorUnknown:cudaEventQuery(e);}
static cudaError_t test_join(cudaStream_t s){return fail_join?cudaErrorUnknown:cudaStreamSynchronize(s);}
struct Gate {std::atomic<bool> entered{false},release{false},auto_released{false};};
static void hold(Gate& g){
    g.entered=true;const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(4);
    while(!g.release.load()) {
        if(std::chrono::steady_clock::now()>=until) {g.auto_released=true;g.release=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
static void gate(void* p,int){hold(*static_cast<Gate*>(p));}
static void CUDART_CB stream_gate(void* p){hold(*static_cast<Gate*>(p));}
static void wait_for(const std::function<bool()>& f){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);while(!f()){req(std::chrono::steady_clock::now()<end,"deadline");std::this_thread::sleep_for(std::chrono::milliseconds(1));}}
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,n_pending=0;cudaStream_t s=nullptr,s2=nullptr,cs=nullptr;
    std::vector<cudaEvent_t>slot_ev;std::vector<long long>slot_owner,slot_pend,slot_used;
    std::vector<int>slot_iss;std::vector<uint8_t>slot_prediction;std::vector<Promo>promo_q;
    uint8_t* dest=nullptr;
};
#define cudaEventQuery test_query
#define cudaStreamSynchronize test_join
struct Model {
    std::unique_ptr<ss::Service>scale_service;size_t host_stride=0;int scale_device=-1;
    struct {int native_stage_prefix=0;} o;
    bool native=true,placement_failed=false;std::atomic<bool>retirement_uncertain{false};std::string promotion_error;
    long long fwd=1;struct {long promo_queries=0,promo_not_ready=0;double t_promo=0;} st;
    bool compressed_host()const{return true;}bool prefetch_active(bool)const{return false;}
    static double now(){return 0;}
    uint8_t* staging_address(Dev& d,int slot){return d.dest+size_t(slot)*ss::kRawBytes;}
#include "../src/mm3/mm3_native_scale_bridge.inc"
#include "../src/mm3/mm3_native_maintenance.inc"
#include "../src/mm3/mm3_native_slot_publish.inc"
};
#undef cudaEventQuery
#undef cudaStreamSynchronize
static void child(int device,const std::string& mode){
    isolated_child=true;
    ck(cudaSetDevice(device));Dev d;d.id=device;ck(cudaStreamCreateWithFlags(&d.s,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&d.cs,cudaStreamNonBlocking));
    int device_count=0;ck(cudaGetDeviceCount(&device_count));req(device_count>=2,"two CUDA devices required");
    Dev prefix;prefix.id=device==0 ? 1 : 0;
    ck(cudaSetDevice(prefix.id));ck(cudaStreamCreateWithFlags(&prefix.s,cudaStreamNonBlocking));ck(cudaStreamCreateWithFlags(&prefix.s2,cudaStreamNonBlocking));
    ck(cudaSetDevice(device));
    uint8_t *fg=nullptr,*opt=nullptr,*fg_dest=nullptr,*encoded=nullptr;ck(cudaMalloc(&fg,ss::kScratchBytes));ck(cudaMalloc(&opt,ss::kScratchBytes));ck(cudaMalloc(&d.dest,ss::kRawBytes));ck(cudaMalloc(&fg_dest,ss::kRawBytes));
    ck(cudaMallocHost(&encoded,ss::kRawBytes+1048576));std::vector<uint8_t> raw(ss::kRawBytes,17);size_t encoded_bytes=0;std::string err;
    req(ss::encode(raw.data(),raw.size(),encoded,ss::kRawBytes+1048576,encoded_bytes,err),err);
    // The child exits without unwinding on assertion failures. Static gates
    // also remain valid if an unexpected exception unwinds to main's catch.
    static Gate g;ss::TestHooks hooks{&g,nullptr,gate};Model m;m.host_stride=encoded_bytes;m.scale_device=device;
    m.scale_service=std::make_unique<ss::Service>(device,fg,d.s,opt,d.cs,30000,err,&hooks);req(m.scale_service->healthy(),err);
    // CUDA may synchronize the context while lazily loading the interleave
    // module. Load it before holding a callback, as in the nonblocking fixture.
    req(m.scale_service->issue_foreground(fg_dest,encoded,encoded_bytes),m.scale_service->error());
    ck(cudaStreamSynchronize(d.s));req(m.scale_foreground_joined(d,err),err);
    cudaEvent_t event=nullptr;ck(cudaEventCreateWithFlags(&event,cudaEventDisableTiming));
    ck(cudaEventRecord(event,d.cs));ck(cudaStreamSynchronize(d.cs)); // stale ready event
    d.slot_ev={event};d.slot_owner={-1};d.slot_pend={42};d.slot_used={-1};d.slot_iss={0};d.n_pending=1;
    d.promo_q.push_back({0,encoded,ss::kRawBytes});req(m.issue_promotions(d,err),err);
    wait_for([&]{return g.entered.load();});req(m.scale_service->optional_state(0)==ss::OptionalState::Running,"optional not running");
    query_calls=0;req(m.publish_slots(d,err),err);req(query_calls==0 && d.n_pending==1 && d.slot_owner[0]==-1,"stale event published");
    // Foreground decoding proceeds while optional is held and its lease is reset only after the joined primary stream.
    static Gate fg_gate;ck(cudaLaunchHostFunc(d.s,stream_gate,&fg_gate));
    const auto kick=cudaStreamQuery(d.s);req(kick==cudaSuccess || kick==cudaErrorNotReady,"foreground gate progress query");
    wait_for([&]{return fg_gate.entered.load();});
    req(m.scale_service->issue_foreground(fg_dest,encoded,encoded_bytes),m.scale_service->error());
    req(m.scale_joined(prefix,err) && m.scale_foreground_joined(prefix,err),err);
    req(m.scale_service->foreground_used()>0 && m.scale_service->optional_state(0)==ss::OptionalState::Running,"prefix retired suffix leases");
    // Maintenance drains CPU submitters, so release the optional worker first.
    // The suffix foreground GPU reader remains held independently throughout
    // the actual prefix-device stream joins and codec retirement check.
    g.release=true;wait_for([&]{return m.scale_service->optional_state(0)==ss::OptionalState::Submitted || !m.scale_service->healthy();});
    req(m.scale_service->optional_state(0)==ss::OptionalState::Submitted,m.scale_service->error());
    req(m.scale_maintenance_join(prefix,err),err);
    req(m.scale_service->foreground_used()>0 && m.scale_service->optional_state(0)==ss::OptionalState::Submitted &&
        d.n_pending==1 && d.slot_owner[0]==-1,"prefix maintenance retired suffix leases");
    ck(cudaSetDevice(device));req(cudaStreamQuery(d.s)==cudaErrorNotReady,"suffix foreground gate did not hold stream");(void)cudaGetLastError();
    req(!g.auto_released && !fg_gate.auto_released,"fixture gate timed out before prefix maintenance proof");
    fg_gate.release=true;
    if(mode=="query") {fail_query=true;req(!m.publish_slots(d,err),"query failure accepted");req(m.placement_failed && m.retirement_uncertain && !m.promotion_error.empty(),"query failure not quarantined");std::fflush(nullptr);_Exit(0);}
    if(mode=="join") {fail_join=true;req(!m.scale_maintenance_join(d,err),"join failure accepted");req(m.placement_failed && m.retirement_uncertain,"join failure not quarantined");std::fflush(nullptr);_Exit(0);}
    wait_for([&]{req(m.publish_slots(d,err),err);return d.n_pending==0;});
    req(d.slot_owner[0]==42 && m.scale_service->optional_state(0)==ss::OptionalState::Free,"optional not retired before publication");
    req(m.scale_service->foreground_used()>0,"foreground lease retired before suffix maintenance");
    req(m.scale_maintenance_join(d,err),err);req(m.scale_service->foreground_used()==0,"suffix maintenance retained foreground lease");
    req(m.scale_service->release_after_gpu_join(),m.scale_service->error());
    ck(cudaEventDestroy(event));ck(cudaFreeHost(encoded));ck(cudaFree(fg_dest));ck(cudaFree(d.dest));ck(cudaFree(opt));ck(cudaFree(fg));ck(cudaStreamDestroy(d.cs));ck(cudaStreamDestroy(d.s));
    ck(cudaSetDevice(prefix.id));ck(cudaStreamDestroy(prefix.s2));ck(cudaStreamDestroy(prefix.s));
}
int main(int argc,char** argv)try {
    req(argc==2||argc==3,"usage: test DEVICE [CASE]");int device=std::atoi(argv[1]);
    if(argc==3){child(device,argv[2]);return 0;}
    for(const char* mode:{"success","query","join"}) {pid_t pid=fork();req(pid>=0,"fork");if(!pid){execl(argv[0],argv[0],argv[1],mode,(char*)nullptr);_Exit(127);}int status=0;req(waitpid(pid,&status,0)==pid && WIFEXITED(status) && WEXITSTATUS(status)==0,std::string("failed: ")+mode);std::printf("PASS bridge %s\n",mode);}
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"bridge: %s\n",e.what());if(isolated_child){std::fflush(nullptr);_Exit(1);}return 1;}
 catch(...){std::fprintf(stderr,"bridge: unexpected exception\n");if(isolated_child){std::fflush(nullptr);_Exit(1);}return 1;}
