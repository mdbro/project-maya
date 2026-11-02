#include "mm3_native_scale_store.hpp"
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace mm3::scale_store {
namespace {
__global__ void interleave(const uint8_t* codes,const uint8_t* scales,uint8_t* out) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    constexpr size_t units=kScaleBytes/16;
    if(i<units*144) {size_t u=i/144,o=i%144;out[i]=o<128?codes[u*128+o]:scales[u*16+o-128];}
}
}
struct Service::Impl {
    using Clock=std::chrono::steady_clock;
    struct Job { int frame=-1; OptionalState state=OptionalState::Free; void* dst=nullptr; const uint8_t* src=nullptr; size_t cap=0; cudaEvent_t event=nullptr; };
    int device; uint8_t* fg_scratch; uint8_t* opt_scratch; cudaStream_t fg_stream,opt_stream;
    unsigned deadline_ms; TestHooks hooks{}; uint8_t* pinned=nullptr; cudaEvent_t progress=nullptr;
    mutable std::mutex mu; std::condition_variable cv,done;
    bool stop=false, failed=false; std::string why;
    std::array<Job,kOptionalBuffers> jobs{}; std::vector<std::thread> fg_threads; std::thread opt_thread;
    unsigned generation=0; int remaining=0; bool decode_bad=false;
    View current{}; uint8_t* current_dst=nullptr; size_t used=0;
    Impl(int dev,void* fg,cudaStream_t fs,void* opt,cudaStream_t os,unsigned ms,const TestHooks* h)
      :device(dev),fg_scratch(static_cast<uint8_t*>(fg)),opt_scratch(static_cast<uint8_t*>(opt)),fg_stream(fs),opt_stream(os),deadline_ms(ms),hooks(h ? *h : TestHooks{}) {}
    void fail_locked(const std::string& s) { if(!failed) {failed=true;why=s;} done.notify_all(); }
    void fail(const std::string& s) {std::lock_guard<std::mutex> l(mu);fail_locked(s);}
    bool cuda_ok(cudaError_t rc,const char* what) {if(rc==cudaSuccess)return true;fail(std::string(what)+": "+cudaGetErrorString(rc));return false;}
    bool before(Operation op,const char* what) {
        if(!hooks.before)return true;
        try {return cuda_ok(hooks.before(hooks.context,op),what);}
        catch(const std::exception& e) {fail(std::string(what)+": hook exception: "+e.what());return false;}
        catch(...) {fail(std::string(what)+": unknown hook exception");return false;}
    }
    void fg_worker(int id) {
        unsigned seen=0;
        for(;;) {
            std::unique_lock<std::mutex> l(mu);cv.wait(l,[&]{return stop||seen!=generation;});
            if(stop)return;seen=generation;View v=current;uint8_t* dst=current_dst;l.unlock();
            std::string e;bool ok=true;
            try {for(size_t b=id;b<kBlocks;b+=4) if(!decode_block(v,b,dst,e)) {ok=false;break;}}
            catch(const std::exception& ex) {ok=false;e=ex.what();}
            catch(...) {ok=false;e="unknown decoder exception";}
            l.lock();if(!ok){decode_bad=true;fail_locked("foreground decode: "+e);} --remaining;done.notify_all();
        }
    }
    bool submit(void* dst,const View& v,uint8_t* scales,uint8_t* scratch,cudaStream_t s,cudaEvent_t event,bool progress_query) {
        if(!before(Operation::HeaderCopy,"inject header copy") || !cuda_ok(cudaMemcpyAsync(dst,v.header,kHeaderBytes,cudaMemcpyHostToDevice,s),"copy native header"))return false;
        if(!before(Operation::CodeCopy,"inject code copy") || !cuda_ok(cudaMemcpyAsync(scratch,v.codes,kCodeBytes,cudaMemcpyHostToDevice,s),"copy native codes"))return false;
        if(progress_query) {
            if(!before(Operation::ProgressEventRecord,"inject progress event") || !cuda_ok(cudaEventRecord(progress,s),"record code progress"))return false;
            if(!before(Operation::ProgressEventQuery,"inject progress query"))return false;
            cudaError_t rc=cudaEventQuery(progress);
            if(rc!=cudaSuccess && rc!=cudaErrorNotReady) return cuda_ok(rc,"query code progress");
        }
        if(!before(Operation::ScaleCopy,"inject scale copy") || !cuda_ok(cudaMemcpyAsync(scratch+kCodeBytes,scales,kScaleBytes,cudaMemcpyHostToDevice,s),"copy native scales"))return false;
        constexpr size_t bytes=kRawBytes-kHeaderBytes;
        if(!before(Operation::Interleave,"inject interleave"))return false;
        interleave<<<(bytes+255)/256,256,0,s>>>(scratch,scratch+kCodeBytes,static_cast<uint8_t*>(dst)+kHeaderBytes);
        if(!cuda_ok(cudaGetLastError(),"launch RG16 interleave"))return false;
        if(event&&(!before(Operation::CompletionEventRecord,"inject completion event") || !cuda_ok(cudaEventRecord(event,s),"record optional completion")))return false;
        return true;
    }
    void optional_worker() {
        if(!cuda_ok(cudaSetDevice(device),"optional worker cudaSetDevice"))return;
        for(;;) {
            std::unique_lock<std::mutex> l(mu);
            cv.wait(l,[&]{if(stop)return true;for(const auto& j:jobs)if(j.state==OptionalState::Queued)return true;return false;});
            if(stop)return;
            size_t i=0;while(i<jobs.size()&&jobs[i].state!=OptionalState::Queued)++i;
            Job job=jobs[i];jobs[i].state=OptionalState::Running;l.unlock();
            bool ok=false;
            try {
                if(hooks.optional_gate)hooks.optional_gate(hooks.context,job.frame);
                {std::lock_guard<std::mutex> guard(mu);ok=!failed;}
                View v;std::string e;if(ok)ok=view(job.src,job.cap,v,e);
                uint8_t* dst=pinned+(kForegroundBuffers+i)*kScaleBytes;
                if(ok)for(size_t b=0;b<kBlocks;++b)if(!decode_block(v,b,dst,e)){ok=false;break;}
                if(!ok && !e.empty())fail("optional decode: "+e);
                {std::lock_guard<std::mutex> guard(mu);ok=ok&&!failed;}
                if(ok)ok=submit(job.dst,v,dst,opt_scratch,opt_stream,job.event,false);
            } catch(const std::exception& e) {fail(std::string("optional worker exception: ")+e.what());ok=false;}
              catch(...) {fail("unknown optional worker exception");ok=false;}
            l.lock(); jobs[i].state=(ok&&!failed)?OptionalState::Submitted:OptionalState::Failed;done.notify_all();
        }
    }
    bool wait_cpu(std::unique_lock<std::mutex>& l) {
        auto pending=[&]{if(remaining)return true;for(const auto& j:jobs)if(j.state==OptionalState::Queued||j.state==OptionalState::Running)return true;return false;};
        if(!done.wait_until(l,Clock::now()+std::chrono::milliseconds(deadline_ms),[&]{return !pending();})) {fail_locked("scale worker progress deadline exceeded; ownership quarantined");return false;}
        return !failed;
    }
};
Service::Service(int dev,void* fg,cudaStream_t fs,void* opt,cudaStream_t os,unsigned ms,std::string& err,const TestHooks* hooks) {
    p_=new Impl(dev,fg,fs,opt,os,ms,hooks);
    if(!fg||!opt||fg==opt||!fs||!os||fs==os||!ms) {p_->fail("invalid scale service streams, scratch or deadline");err=p_->why;return;}
    if(!p_->cuda_ok(cudaSetDevice(dev),"scale service cudaSetDevice") ||
       !p_->cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&p_->pinned),kPinnedBytes),"allocate decoded scale buffers") ||
       !p_->cuda_ok(cudaEventCreateWithFlags(&p_->progress,cudaEventDisableTiming),"allocate progress event")) {err=p_->why;return;}
    try {
        for(int i=0;i<4;++i)p_->fg_threads.emplace_back([p=p_,i]{
            try {p->fg_worker(i);} catch(const std::exception& e) {p->fail(std::string("foreground worker exception: ")+e.what());}
            catch(...) {p->fail("unknown foreground worker exception");}
        });
        p_->opt_thread=std::thread([p=p_]{
            try {p->optional_worker();}
            catch(const std::exception& e) {p->fail(std::string("optional worker exception: ")+e.what());}
            catch(...) {p->fail("unknown optional worker exception");}
            std::lock_guard<std::mutex> l(p->mu);
            if(p->failed)for(auto& j:p->jobs)if(j.state==OptionalState::Running||j.state==OptionalState::Queued)j.state=OptionalState::Failed;
            p->done.notify_all();
        });
    }
    catch(const std::exception& e){p_->fail(std::string("create scale workers: ")+e.what());}
    if(!healthy())err=error();
}
// Never free possibly active CPU or GPU readers on a failed/forgotten release.
Service::~Service() {p_=nullptr;}
bool Service::healthy() const {if(!p_)return false;std::lock_guard<std::mutex> l(p_->mu);return !p_->failed;}
std::string Service::error() const {if(!p_)return "scale service released";std::lock_guard<std::mutex> l(p_->mu);return p_->why;}
size_t Service::foreground_used() const {if(!p_)return 0;std::lock_guard<std::mutex> l(p_->mu);return p_->used;}
bool Service::issue_foreground(void* dst,const uint8_t* src,size_t cap) {
    if(!p_)return false;View v;std::string e;
    if(!dst||!view(src,cap,v,e)){p_->fail("foreground source: "+e);return false;}
    uint8_t* decoded;
    {std::lock_guard<std::mutex> l(p_->mu);
     if(p_->failed)return false;
     if(p_->remaining||p_->used>=kForegroundBuffers){p_->fail_locked("foreground decoded buffer exhausted or decoder busy");return false;}
     decoded=p_->pinned+p_->used++*kScaleBytes;p_->current=v;p_->current_dst=decoded;p_->decode_bad=false;p_->remaining=4;++p_->generation;p_->cv.notify_all();}
    // Start workers before transferring header and codes; query is a progress
    // kick, never a CUDA wait. Decode must finish before scale transfer.
    if(!p_->cuda_ok(cudaSetDevice(p_->device),"foreground cudaSetDevice") ||
       !p_->before(Operation::HeaderCopy,"inject foreground header copy") ||
       !p_->cuda_ok(cudaMemcpyAsync(dst,v.header,kHeaderBytes,cudaMemcpyHostToDevice,p_->fg_stream),"copy foreground header") ||
       !p_->before(Operation::CodeCopy,"inject foreground code copy") ||
       !p_->cuda_ok(cudaMemcpyAsync(p_->fg_scratch,v.codes,kCodeBytes,cudaMemcpyHostToDevice,p_->fg_stream),"copy foreground codes") ||
       !p_->before(Operation::ProgressEventRecord,"inject foreground progress event") ||
       !p_->cuda_ok(cudaEventRecord(p_->progress,p_->fg_stream),"record foreground progress"))return false;
    if(!p_->before(Operation::ProgressEventQuery,"inject foreground progress query"))return false;
    cudaError_t q=cudaEventQuery(p_->progress);if(q!=cudaSuccess&&q!=cudaErrorNotReady)return p_->cuda_ok(q,"query foreground progress");
    {std::unique_lock<std::mutex> l(p_->mu);
     if(!p_->done.wait_until(l,Impl::Clock::now()+std::chrono::milliseconds(p_->deadline_ms),[&]{return p_->remaining==0;})) {p_->fail_locked("foreground decoder progress deadline exceeded; ownership quarantined");return false;}
     if(p_->failed||p_->decode_bad)return false;}
    if(!p_->before(Operation::ScaleCopy,"inject foreground scale copy") || !p_->cuda_ok(cudaMemcpyAsync(p_->fg_scratch+kCodeBytes,decoded,kScaleBytes,cudaMemcpyHostToDevice,p_->fg_stream),"copy foreground scales"))return false;
    constexpr size_t bytes=kRawBytes-kHeaderBytes;
    if(!p_->before(Operation::Interleave,"inject foreground interleave"))return false;
    interleave<<<(bytes+255)/256,256,0,p_->fg_stream>>>(p_->fg_scratch,p_->fg_scratch+kCodeBytes,static_cast<uint8_t*>(dst)+kHeaderBytes);
    return p_->cuda_ok(cudaGetLastError(),"launch foreground RG16 interleave");
}
bool Service::reset_foreground_after_join(){if(!p_)return false;std::lock_guard<std::mutex> l(p_->mu);if(p_->failed||p_->remaining)return false;p_->used=0;return true;}
bool Service::enqueue_optional(int frame,void* dst,const uint8_t* src,size_t cap,cudaEvent_t event) {
    if(!p_)return false;std::lock_guard<std::mutex> l(p_->mu);if(p_->failed)return false;
    if(frame<0||!dst||!src||!event){p_->fail_locked("invalid optional scale job");return false;}
    for(const auto& j:p_->jobs)if(j.frame==frame&&j.state!=OptionalState::Free){p_->fail_locked("optional frame already leased");return false;}
    for(auto& j:p_->jobs)if(j.state==OptionalState::Free){j={frame,OptionalState::Queued,dst,src,cap,event};p_->cv.notify_all();return true;}
    return false; // bounded backpressure; no ownership transferred
}
OptionalState Service::optional_state(int frame) const {if(!p_)return OptionalState::Free;std::lock_guard<std::mutex> l(p_->mu);for(const auto& j:p_->jobs)if(j.frame==frame)return j.state;return OptionalState::Free;}
bool Service::retire_optional(int frame){if(!p_)return false;std::lock_guard<std::mutex> l(p_->mu);for(auto& j:p_->jobs)if(j.frame==frame){if(j.state!=OptionalState::Submitted)return false;j=Impl::Job{};return true;}return false;}
bool Service::drain_cpu(){if(!p_)return false;std::unique_lock<std::mutex> l(p_->mu);return p_->wait_cpu(l);}
bool Service::release_after_gpu_join() {
    if(!p_)return false;auto* p=p_;
    {std::unique_lock<std::mutex> l(p->mu);if(!p->wait_cpu(l))return false;
     for(const auto& j:p->jobs)if(j.state!=OptionalState::Free)return false;
     p->stop=true;p->cv.notify_all();}
    for(auto& t:p->fg_threads)if(t.joinable())t.join();if(p->opt_thread.joinable())p->opt_thread.join();
    if(!p->cuda_ok(cudaSetDevice(p->device),"release cudaSetDevice") || !p->cuda_ok(cudaEventDestroy(p->progress),"release progress event") || !p->cuda_ok(cudaFreeHost(p->pinned),"release decoded buffers"))return false;
    p_=nullptr;delete p;return true;
}
}
