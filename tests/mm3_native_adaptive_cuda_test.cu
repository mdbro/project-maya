// Manual real-CUDA check of literal adaptive publication and ownership.
// This uses tiny artificial records and stand-in byte-reader kernels; it is not
// a full Model::chunk, DSpark, model-output or performance qualification.
// Run only when the primary coordinator has authorized use of the device.
#include <cuda_runtime.h>
#include "../src/mm3/mm3_native_host.hpp"
#include "../src/mm3/mm3_residency_policy.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static void check(cudaError_t s,const char* what) {
    if(s!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(s));
}
#define CUDA(x) check((x),#x)
#define REQUIRE(x) do {if(!(x))throw std::runtime_error("check failed: " #x);}while(0)
namespace ne { struct Header {uint8_t bytes[16];}; }
namespace nf { static size_t record_stride(int,int){return 16;} }
constexpr size_t kRecord=32;
struct Pool {size_t n;size_t size()const{return n;}};
struct Layer {
    bool moe=true,hf=false;size_t blob=kRecord;uint8_t* res_base=nullptr;
    std::vector<int> res_slot,res_ids;std::vector<uint8_t*> res_ptr;
    std::vector<const uint8_t*> host_blob;
};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    struct AdaptiveSwap {
        long long incoming=-1,victim=-1;int landing=-1;
        uint8_t *source_host=nullptr,*source_gpu=nullptr,*landing_gpu=nullptr,*target_host=nullptr;
        cudaEvent_t event=nullptr;bool submitted=false;
    };
    int id=0,n_slots=4,n_pending=0;size_t slot_bytes=kRecord;
    cudaStream_t cs=nullptr,s=nullptr;
    std::vector<int> layers{0};std::vector<long long> slot_owner,slot_pend,slot_used;
    std::vector<int> slot_foreground,slot_iss;std::vector<Promo> promo_q;
    std::vector<uint8_t*> slot_ptr;std::vector<uint8_t> slot_reserved;
    std::array<AdaptiveSwap,2> adaptive;uint8_t* stg=nullptr;
};
struct Model {
    struct {int n_layers=1,n_exp=4,dim=1,ff=1;} g;
    struct {bool fast_decode=true;int chunk=16,seqs=1;} o;
    struct {
        long adaptive_queries=0,adaptive_not_ready=0,adaptive_published=0,adaptive_cancelled_promotions=0;
        long adaptive_no_gain=0,adaptive_no_slot=0,adaptive_issued=0,adaptive_reused=0,adaptive_source_busy=0;
        unsigned long long adaptive_h2d_bytes=0,adaptive_d2h_bytes=0;double adaptive_host_ms=0;
    } st;
    bool native=true,graph_on=false,vis=false,placement_failed=false,generation_active=true;
    std::atomic<bool> retirement_uncertain{false};long long fwd=1;
    static constexpr int kWin=16,kMaxPending=4;static constexpr float kPromote=1.8f,kDecay=.95f;
    double lane_cpu1=0;
    std::vector<Layer> L{Layer{}};std::vector<Dev> D{Dev{}};
    uint8_t* pool=nullptr;std::unique_ptr<Pool> native_pool;
    std::array<float,4> scores{{0,0,12,10}};
    static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
    uint8_t* resident_address(const Layer& l,int s)const{return l.res_ptr.empty()?l.res_base+size_t(s)*l.blob:l.res_ptr.at(size_t(s));}
    float score(long long key,bool)const{return scores.at(size_t(key%1024));}
    bool helper_live_leased(long long)const{return false;}
    void prefetch_forget(Dev&,int){}
    void prefetch_external_cancel(Dev&,int){}
#include "../src/mm3/mm3_native_adaptive.inc"
};

struct Hold {std::mutex mu;std::condition_variable cv;bool started=false,released=false,timed_out=false;};
static void CUDART_CB hold_copy_stream(void* p) {
    auto& h=*static_cast<Hold*>(p);std::unique_lock<std::mutex> lock(h.mu);
    h.started=true;h.cv.notify_all();
    if(!h.cv.wait_for(lock,std::chrono::seconds(5),[&]{return h.released;})) {h.timed_out=true;h.released=true;}
}
struct ReleaseGuard {
    Hold* h;
    void release(){std::lock_guard<std::mutex> lock(h->mu);h->released=true;h->cv.notify_all();}
    void started(){std::unique_lock<std::mutex> lock(h->mu);REQUIRE(h->cv.wait_for(lock,std::chrono::seconds(2),[&]{return h->started;}));}
    bool held(){std::lock_guard<std::mutex> lock(h->mu);return h->started&&!h->released&&!h->timed_out;}
    ~ReleaseGuard(){release();}
};
__global__ void read_record(const uint8_t* input,uint8_t* output,int width) {
    int i=int(threadIdx.x);if(i<width)output[i]=input[i];
}
static void wait_ready(cudaEvent_t e) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    for(;;){const auto s=cudaEventQuery(e);if(s==cudaSuccess)return;
        if(s!=cudaErrorNotReady)check(s,"cudaEventQuery");
        if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("independent byte reader did not finish");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));}
}
int main(int argc,char** argv) {
    try {
        Model m;m.adaptive_policy.legacy=argc<=3 || std::string(argv[3])!="priced";auto& d=m.D[0];d.id=argc>1?std::atoi(argv[1]):0;CUDA(cudaSetDevice(d.id));
        uint8_t *primary=nullptr,*stage=nullptr,*host=nullptr,*result_d=nullptr,*result_h=nullptr;
        std::unique_ptr<mm3::NativeHostBank> canonical;
        if(argc>2 && std::string(argv[2])=="vmm") {
            canonical=std::make_unique<mm3::NativeHostBank>(2*kRecord,d.id);
            host=reinterpret_cast<uint8_t*>(canonical->data());
        } else CUDA(cudaHostAlloc(&host,2*kRecord,cudaHostAllocPortable));
        CUDA(cudaHostAlloc(&result_h,4,cudaHostAllocPortable));
        CUDA(cudaMalloc(&primary,2*kRecord));CUDA(cudaMalloc(&stage,4*kRecord));CUDA(cudaMalloc(&result_d,4));
        CUDA(cudaStreamCreateWithFlags(&d.s,cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&d.cs,cudaStreamNonBlocking));
        cudaEvent_t reader_done=nullptr;CUDA(cudaEventCreateWithFlags(&reader_done,cudaEventDisableTiming));
        std::array<uint8_t,2*kRecord> initial{};
        for(size_t i=0;i<kRecord;++i){initial[i]=10;initial[kRecord+i]=11;host[i]=12;host[kRecord+i]=13;}
        CUDA(cudaMemcpyAsync(primary,initial.data(),initial.size(),cudaMemcpyHostToDevice,d.s));
        CUDA(cudaMemcpyAsync(stage,host,kRecord,cudaMemcpyHostToDevice,d.s));
        // Eagerly load this kernel before holding the copy stream.
        read_record<<<1,4,0,d.s>>>(primary,result_d,4);CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(d.s));
        m.pool=host;m.native_pool=std::make_unique<Pool>(Pool{2*kRecord});
        auto& l=m.L[0];l.res_base=primary;l.res_ids={0,1};l.res_slot={0,1,-1,-1};l.host_blob={nullptr,nullptr,host,host+kRecord};
        d.stg=stage;d.slot_owner={2,-1,-1,-1};d.slot_pend.assign(4,-1);d.slot_used.assign(4,-1);
        d.slot_foreground.assign(4,0);d.slot_iss.assign(4,0);
        std::string err;REQUIRE(m.adaptive_prepare(err));m.primary_adaptive=true;
        REQUIRE(m.adaptive_blob==kRecord && m.adaptive_host_owned.size()*m.adaptive_blob==2*kRecord);
        // A heap owner intentionally survives to process exit on exceptional
        // paths, so a queued callback can never outlive its referenced object.
        auto* held=new Hold;
        {
            ReleaseGuard release{held};CUDA(cudaLaunchHostFunc(d.cs,hold_copy_stream,held));
            if(!m.adaptive_policy.legacy) {
                REQUIRE(m.adaptive_retain_joined(d,1,err));
                REQUIRE(m.adaptive_pending(d)==1 && m.st.adaptive_reused==1);
            }
            REQUIRE(m.adaptive_issue(d,err));REQUIRE(m.adaptive_pending(d)==2);
            REQUIRE(m.st.adaptive_reused==1 && m.st.adaptive_h2d_bytes==kRecord && m.st.adaptive_d2h_bytes==2*kRecord);
            auto* old_host=l.host_blob[2];auto* old_primary=m.resident_address(l,l.res_slot[0]);
            release.started();REQUIRE(release.held());
            for(int width:{1,4,1,4}) {
                ++m.fwd;
                read_record<<<1,4,0,d.s>>>(old_primary,result_d,width);CUDA(cudaGetLastError());
                CUDA(cudaMemcpyAsync(result_h,result_d,size_t(width),cudaMemcpyDeviceToHost,d.s));
                CUDA(cudaEventRecord(reader_done,d.s));wait_ready(reader_done);
                REQUIRE(release.held());for(int i=0;i<width;++i)REQUIRE(result_h[i]==10);
                REQUIRE(*old_host==12 && m.adaptive_publish(d,err));
                REQUIRE(m.adaptive_pending(d)==2 && l.host_blob[2]==old_host && l.res_slot[0]>=0);
            }
            REQUIRE(m.st.adaptive_not_ready>=8 && release.held());
            release.release();CUDA(cudaStreamSynchronize(d.cs));
            REQUIRE(m.adaptive_publish(d,err));REQUIRE(m.adaptive_pending(d)==0);
            REQUIRE(l.res_slot[2]>=0 && l.res_slot[3]>=0 && l.host_blob[2]==nullptr && l.host_blob[3]==nullptr);
            REQUIRE(l.host_blob[0] && l.host_blob[1]);
            REQUIRE(std::all_of(l.host_blob[0],l.host_blob[0]+kRecord,[](uint8_t v){return v==10;}));
            REQUIRE(std::all_of(l.host_blob[1],l.host_blob[1]+kRecord,[](uint8_t v){return v==11;}));
            std::array<uint8_t,kRecord> result{};
            CUDA(cudaMemcpy(result.data(),m.resident_address(l,l.res_slot[2]),kRecord,cudaMemcpyDeviceToHost));
            REQUIRE(std::all_of(result.begin(),result.end(),[](uint8_t v){return v==12;}));
            CUDA(cudaMemcpy(result.data(),m.resident_address(l,l.res_slot[3]),kRecord,cudaMemcpyDeviceToHost));
            REQUIRE(std::all_of(result.begin(),result.end(),[](uint8_t v){return v==13;}));
            std::vector<uintptr_t> hosts{uintptr_t(l.host_blob[0]),uintptr_t(l.host_blob[1])};
            std::vector<uintptr_t> residents{uintptr_t(m.resident_address(l,l.res_slot[2])),uintptr_t(m.resident_address(l,l.res_slot[3]))};
            REQUIRE(m.adaptive_storage_ok(hosts,residents));
            // A second exchange writes directly into the former canonical
            // HOST frames, now spares. In vmm mode these are the actual
            // CPU/primary-only HOST VMM mapping, not portable pinned memory.
            m.scores={20,18,0,0};++m.fwd;
            REQUIRE(m.adaptive_issue(d,err) && m.adaptive_pending(d)==2);
            CUDA(cudaStreamSynchronize(d.cs));REQUIRE(m.adaptive_publish(d,err));
            REQUIRE(l.res_slot[0]>=0 && l.res_slot[1]>=0);
            for(int e=2;e<4;++e)for(size_t b=0;b<kRecord;++b)REQUIRE(l.host_blob[e][b]==10+e);
            for(int e=0;e<2;++e) {
                CUDA(cudaMemcpy(result.data(),m.resident_address(l,l.res_slot[e]),kRecord,cudaMemcpyDeviceToHost));
                REQUIRE(std::all_of(result.begin(),result.end(),[&](uint8_t v){return v==10+e;}));
            }
            hosts={uintptr_t(l.host_blob[2]),uintptr_t(l.host_blob[3])};
            residents={uintptr_t(m.resident_address(l,l.res_slot[0])),uintptr_t(m.resident_address(l,l.res_slot[1]))};
            REQUIRE(m.adaptive_storage_ok(hosts,residents));
        }
        delete held;
        for(auto& x:d.adaptive)CUDA(cudaEventDestroy(x.event));
        for(auto* p:m.adaptive_host_owned)CUDA(cudaFreeHost(p));
        CUDA(cudaEventDestroy(reader_done));CUDA(cudaStreamDestroy(d.s));CUDA(cudaStreamDestroy(d.cs));
        CUDA(cudaFree(primary));CUDA(cudaFree(stage));CUDA(cudaFree(result_d));
        if(!canonical)CUDA(cudaFreeHost(host));canonical.reset();CUDA(cudaFreeHost(result_h));
        std::puts("native adaptive publication and independent CUDA byte readers: OK");return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"native adaptive CUDA fixture: %s\n",e.what());return 1;}
}
