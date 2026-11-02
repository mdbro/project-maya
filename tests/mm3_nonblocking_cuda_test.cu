// Manual, tiny real-CUDA integration check for nonblocking promotion
// publication. Uses no model weights. Never run during an unrelated model run.
#include "mm3_r2_lifecycle.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using mm3::r2::Completion;
static void cuda_check(cudaError_t e,const char* call) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(call)+": "+cudaGetErrorString(e));
}
#define CUDA(x) cuda_check((x),#x)
#define REQUIRE(x) do { if(!(x)) throw std::runtime_error("check failed: " #x); } while(0)
__global__ void increment(unsigned* out) { ++*out; }
struct Hold {
    std::mutex mu;
    std::condition_variable cv;
    bool started=false, released=false, auto_released=false;
};
// This callback is host-only: it does not call CUDA. Its own timeout ensures
// even an accidentally blocking publication cannot hold a GPU kernel running.
static void CUDART_CB hold_callback(void* data) {
    auto& hold=*static_cast<Hold*>(data);
    std::unique_lock<std::mutex> lock(hold.mu);
    hold.started=true; hold.cv.notify_all();
    if(!hold.cv.wait_for(lock,std::chrono::milliseconds(900),[&]{return hold.released;})) {
        hold.auto_released=true; hold.released=true;
    }
}
struct ReleaseGuard {
    Hold* hold;
    explicit ReleaseGuard(Hold* p) : hold(p) {}
    void wait_started() {
        std::unique_lock<std::mutex> lock(hold->mu);
        if(!hold->cv.wait_for(lock,std::chrono::milliseconds(900),[&]{return hold->started;}))
            throw std::runtime_error("copy-stream host callback did not start");
    }
    bool held() {
        std::lock_guard<std::mutex> lock(hold->mu);
        return hold->started && !hold->released && !hold->auto_released;
    }
    void release() {
        std::lock_guard<std::mutex> lock(hold->mu);
        hold->released=true; hold->cv.notify_all();
    }
    ~ReleaseGuard() { release(); }
};
static void wait_ready(cudaEvent_t event) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    for (;;) {
        const cudaError_t e=cudaEventQuery(event);
        if(e==cudaSuccess)return;
        if(e!=cudaErrorNotReady)cuda_check(e,"cudaEventQuery");
        if(std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("independent work did not complete within 3 seconds");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
struct Promo { int slot; };
int main(int argc,char** argv) {
    try {
        const int device=argc>1?std::atoi(argv[1]):0;
        CUDA(cudaSetDevice(device));
        unsigned *host_src=nullptr,*host_counter=nullptr,*target=nullptr,*counter=nullptr;
        CUDA(cudaHostAlloc(&host_src,sizeof(unsigned),cudaHostAllocPortable));
        CUDA(cudaHostAlloc(&host_counter,sizeof(unsigned),cudaHostAllocPortable));
        CUDA(cudaMalloc(&target,sizeof(unsigned))); CUDA(cudaMalloc(&counter,sizeof(unsigned)));
        *host_src=12345;
        CUDA(cudaMemset(target,0,sizeof(unsigned))); CUDA(cudaMemset(counter,0,sizeof(unsigned)));
        CUDA(cudaDeviceSynchronize());
        cudaStream_t copy_stream=nullptr,compute_stream=nullptr;
        cudaEvent_t copy_done=nullptr,compute_done=nullptr;
        CUDA(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&compute_stream,cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&copy_done,cudaEventDisableTiming));
        CUDA(cudaEventCreateWithFlags(&compute_done,cudaEventDisableTiming));
        // Load the compute kernel before holding a callback; module loading is
        // test setup, not the independent work whose progress is being checked.
        increment<<<1,1,0,compute_stream>>>(counter); CUDA(cudaGetLastError());
        CUDA(cudaMemsetAsync(counter,0,sizeof(unsigned),compute_stream));
        CUDA(cudaStreamSynchronize(compute_stream));
        // If an error exits before the callback runs, retain its small host
        // state until process exit. On success it is freed after copy_done.
        auto* held=new Hold;
        {
            ReleaseGuard release(held);
            CUDA(cudaLaunchHostFunc(copy_stream,hold_callback,held));
            std::vector<Promo> queue{{0}};
            std::vector<long long> pending{77},issued{0},owner{-1},used{-1};
            std::string sticky,error;int n=1,queries=0;
            auto query=[&](size_t,std::string& e) {
                ++queries; const cudaError_t status=cudaEventQuery(copy_done);
                if(status==cudaSuccess)return Completion::ready;
                if(status==cudaErrorNotReady)return Completion::not_ready;
                e=cudaGetErrorString(status);return Completion::failed;
            };
            // An unrecorded (therefore already complete) CUDA event cannot
            // publish a pending slot without a successful submission marker.
            REQUIRE(mm3::r2::publish(pending,issued,owner,used,n,1LL,sticky,error,query));
            REQUIRE(queries==0 && owner[0]==-1 && n==1);
            REQUIRE(mm3::r2::issue(queue,issued,sticky,error,[&](const Promo&,std::string& e) {
                cudaError_t status=cudaMemcpyAsync(target,host_src,sizeof(unsigned),cudaMemcpyHostToDevice,copy_stream);
                if(status==cudaSuccess) status=cudaEventRecord(copy_done,copy_stream);
                if(status==cudaSuccess)return true;
                e=cudaGetErrorString(status);return false;
            }));
            release.wait_started();
            REQUIRE(release.held());
            for(long long boundary:{1LL,2LL,2LL,3LL}) {
                // Stand-in work exercises an actual independent CUDA stream at
                // ordinary and grouped boundaries while the copy is blocked.
                increment<<<1,1,0,compute_stream>>>(counter);
                CUDA(cudaGetLastError()); CUDA(cudaEventRecord(compute_done,compute_stream));
                wait_ready(compute_done);
                REQUIRE(release.held());
                const int before=queries;
                REQUIRE(mm3::r2::publish(pending,issued,owner,used,n,boundary,sticky,error,query));
                REQUIRE(queries==before+1 && owner[0]==-1 && pending[0]==77 && issued[0]==1 && n==1);
                CUDA(cudaPeekAtLastError());
                REQUIRE(!mm3::r2::promotion_slot_available(false,true,0LL,boundary));
                REQUIRE(mm3::r2::snapshot_visible(false) && owner[0]!=77); // selected expert remains HOST-served
            }
            CUDA(cudaMemcpyAsync(host_counter,counter,sizeof(unsigned),cudaMemcpyDeviceToHost,compute_stream));
            CUDA(cudaEventRecord(compute_done,compute_stream)); wait_ready(compute_done);
            REQUIRE(*host_counter==4 && release.held());
            release.release(); wait_ready(copy_done);
            REQUIRE(mm3::r2::publish(pending,issued,owner,used,n,3LL,sticky,error,query));
            REQUIRE(owner[0]==77 && pending[0]==-1 && issued[0]==0 && n==0);
            unsigned actual=0; CUDA(cudaMemcpy(&actual,target,sizeof(unsigned),cudaMemcpyDeviceToHost));
            REQUIRE(actual==*host_src);
        }
        delete held; // copy_done proves the host callback has returned
        CUDA(cudaEventDestroy(copy_done)); CUDA(cudaEventDestroy(compute_done));
        CUDA(cudaStreamDestroy(copy_stream)); CUDA(cudaStreamDestroy(compute_stream));
        CUDA(cudaFree(target)); CUDA(cudaFree(counter));
        CUDA(cudaFreeHost(host_src)); CUDA(cudaFreeHost(host_counter));
        std::puts("nonblocking promotion and independent CUDA work: OK");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"nonblocking CUDA fixture: %s\n",e.what()); return 1; }
}
