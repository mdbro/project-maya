#pragma once
#include "mm3_background_traffic.hpp"
#include "mm3_native_helper.hpp"
#include "mm3_upload_guard.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace mm3 {
// A separate canonical-storage preparation job. It is never an implicit stage
// of a GPU swap: an incoming swap can use only an already-published region record.
// Caller leases both old sources and both free destinations until completion and
// joined publication. No worker changes a canonical pointer.
class HostResorter {
public:
    enum State {idle,running,complete,failed};
    HostResorter(int device,BackgroundTraffic& traffic,NativeHelper& helper,NativeHelper& third)
        :device_(device),traffic_(traffic),helper_(helper),third_(third),worker_([this]{run();}) {}
    ~HostResorter(){stop_.store(true);cv_.notify_all();if(worker_.joinable())worker_.join();}
    bool begin(void* into_region,const void* private_source,void* into_private,const void* region_source,
               size_t bytes,bool verify) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(state_.load()!=idle || stop_.load())return false;
        dst_[0]=into_region;dst_[1]=into_private;src_[0]=private_source;src_[1]=region_source;
        bytes_=bytes;verify_=verify;error_.clear();queued_=true;state_.store(running,std::memory_order_release);
        cv_.notify_all();return true;
    }
    State state()const {return state_.load(std::memory_order_acquire);}
    std::string error()const {std::lock_guard<std::mutex> lock(mutex_);return error_;}
    void published(){state_.store(idle,std::memory_order_release);}
    UploadGuard guard;
    std::atomic<bool> cpu_busy{false};
    std::atomic<uint64_t> copied_bytes{0},chunks{0},copy_ns{0},event_queries{0};
    bool stop_and_retire() {
        stop_.store(true,std::memory_order_release);cv_.notify_all();
        const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        while(!exited_.load(std::memory_order_acquire) && std::chrono::steady_clock::now()<until)
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        return exited_.load(std::memory_order_acquire);
    }
private:
    static uint64_t now_ns(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());}
    void run() {
        for(;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return queued_ || stop_.load();});
                if(stop_.load())break;queued_=false;
            }
            std::string error;
            if(cudaSetDevice(device_)!=cudaSuccess)error="HOST re-sort CUDA context failed";
            UploadGuard::PollCache cache;
            for(size_t off=0;error.empty() && off<bytes_;) {
                if(stop_.load(std::memory_order_acquire)){error="HOST re-sort cancelled";break;}
                if(!traffic_.healthy()){error="shared background traffic is poisoned";break;}
                if(cpu_busy.load(std::memory_order_acquire) || helper_.foreground_busy() || third_.foreground_busy()) {
                    std::this_thread::yield();continue;
                }
                const auto claim=guard.claim([&](void* event) {
                    event_queries.fetch_add(1,std::memory_order_relaxed);
                    const auto q=cudaEventQuery(static_cast<cudaEvent_t>(event));
                    return q==cudaSuccess?UploadGuard::Query::ready:q==cudaErrorNotReady?UploadGuard::Query::pending:UploadGuard::Query::error;
                },cache,now_ns(),20000);
                if(claim.result==UploadGuard::Result::error){error="HOST re-sort upload-event query failed";traffic_.poison();break;}
                if(claim.result!=UploadGuard::Result::admitted && claim.result!=UploadGuard::Result::unrestricted) {
                    std::this_thread::yield();continue;
                }
                struct Release{UploadGuard& g;uint64_t token;~Release(){g.release(token);}}release{guard,claim.token};
                const size_t n=std::min(size_t(128)<<10,bytes_-off),traffic=n*(verify_?8:4);
                if(!traffic_.acquire(2,traffic)) {std::this_thread::yield();continue;}
                if(!guard.current(claim.token) || cpu_busy.load(std::memory_order_acquire) ||
                   helper_.foreground_busy() || third_.foreground_busy()) {traffic_.release(2,traffic);continue;}
                const auto started=now_ns();
                for(int i=0;i<2;++i) {
                    auto* dst=static_cast<uint8_t*>(dst_[i])+off;const auto* src=static_cast<const uint8_t*>(src_[i])+off;
                    std::memcpy(dst,src,n);
                    if(verify_ && std::memcmp(dst,src,n))error="HOST re-sort payload mismatch";
                }
                copied_bytes.fetch_add(2*n,std::memory_order_relaxed);chunks.fetch_add(1,std::memory_order_relaxed);
                copy_ns.fetch_add(now_ns()-started,std::memory_order_relaxed);traffic_.release(2,traffic);
                off+=n;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);error_=error;
                state_.store(error.empty()?complete:failed,std::memory_order_release);
            }
        }
        exited_.store(true,std::memory_order_release);
    }
    int device_;
    BackgroundTraffic& traffic_;NativeHelper& helper_;NativeHelper& third_;
    mutable std::mutex mutex_;std::condition_variable cv_;
    std::atomic<State> state_{idle};std::atomic<bool> stop_{false},exited_{false};
    bool queued_=false,verify_=false;size_t bytes_=0;
    void* dst_[2]{};const void* src_[2]{};std::string error_;
    std::thread worker_;
};
} // namespace mm3
