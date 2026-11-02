// Ordered helper queue adapted from Maya DS4 helper_window/helper_loop and
// helper_enqueue. Persistent pool and joined reader/slot lifetimes follow the
// Infernix ExpertHelper/PeerExpertExecutor ownership boundary. MiniMax's RG16
// FFN implementation is supplied by Infernix's minimax_nvfp4_ffn; this class
// deliberately owns no routing or refill policy.
#include "mm3_native_helper.hpp"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace mm3 {
namespace {
namespace mf = infernix::ops::minimax_ffn;
using DeviceJob = mf::FastJob;
constexpr size_t kBufferElements = size_t(NativeHelper::kMaxJobs) * NativeHelper::kHidden;
constexpr int kLayers = 60, kExperts = 128;
// A helper cannot use the CPU bank for its exclusive records. Fail the request
// on a stalled device and quarantine its state instead of attempting a fallback.
#ifdef MM3_HELPER_TEST_DEADLINE_MS
constexpr auto kProgressDeadline = std::chrono::milliseconds(MM3_HELPER_TEST_DEADLINE_MS);
#else
constexpr auto kProgressDeadline = std::chrono::seconds(30);
#endif
constexpr auto kShutdownDeadline = std::chrono::seconds(1);
uint64_t host_now_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}
struct DeviceGuard {
    int previous = -1;
    explicit DeviceGuard(int id) { ck(cudaGetDevice(&previous), "get device"); ck(cudaSetDevice(id), "set helper device"); }
    ~DeviceGuard() { if (previous >= 0) (void)cudaSetDevice(previous); }
};
float from_bf16(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16; float f; std::memcpy(&f, &bits, sizeof(f)); return f;
}
}

struct NativeHelper::Impl {
    struct Record { int slot = -1; float weight[3]{}, input[3]{}; };
    int device, capacity, count = 0;
    size_t stride = 0, workspace_bytes = 0;
    Bytes allocated{};
    uint8_t *pool = nullptr, *workspace = nullptr;
    uint16_t *d_x = nullptr, *d_y = nullptr, *h_x = nullptr, *h_y = nullptr;
    DeviceJob *d_jobs = nullptr, *h_jobs = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t done_event = nullptr;
    std::array<Record, kLayers * kExperts> records{};
    std::array<Job, kMaxJobs> pending_jobs{};
    int pending_n = 0, pending_tokens = 0, pending_columns = 0;
    std::array<size_t, kMaxJobs> pending_offsets{};
    uint64_t next = 0, active = 0;
    bool sealed = false, queued = false, finished = false, quit = false, poisoned = false, unsafe = false;
    bool worker_exited = false;
    bool collector_claimed = false; // exactly one retirement owner per sequence
    PassTiming pass_timing{}; // protected by mu except worker-local endpoints
    std::string failure;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;

    bool refreshing = false, refresh_queued = false, refresh_done = false, refresh_validate = false, refresh_invalid = false;
    void* refresh_dst = nullptr;
    const void* refresh_src = nullptr;
    size_t refresh_bytes = 0;
    cudaMemcpyKind refresh_kind = cudaMemcpyHostToDevice;

    // The worker owns CUDA submission and event polling, so an API call that
    // never returns cannot hold mu or prevent bounded caller/shutdown waits.
    bool transfer(std::unique_lock<std::mutex>& lock, void* dst, const void* src,
                  size_t bytes, cudaMemcpyKind kind, bool& staging_unsafe, std::string& error, bool validate=false) {
        refresh_validate=validate; refresh_invalid=false; refresh_dst=dst; refresh_src=src; refresh_bytes=bytes; refresh_kind=kind;
        refreshing=true; refresh_done=false; refresh_queued=true; cv.notify_one();
        if (!cv.wait_for(lock,kProgressDeadline,[&]{ return refresh_done || poisoned; })) {
            poisoned=true; unsafe=true; failure=error="helper refresh progress deadline exceeded";
            staging_unsafe=true; return false;
        }
        if (poisoned) { staging_unsafe=unsafe; error=failure; return false; }
        refreshing=false;
        if (refresh_invalid) { error="helper refresh requires portable pinned staging"; return false; }
        return true;
    }

    Impl(int dev, int cap) : device(dev), capacity(cap) {
        if (dev < 0 || cap < 1 || cap > kLayers * kExperts) throw std::invalid_argument("invalid helper device/capacity");
        stride = mf::record_stride(kHidden, kIntermediate);
        // submit() admits at most four picks for each of 16 tokens. Jobs of
        // equal width launch as one group, groups run in order on one stream.
        // Bound the workspace by the largest possible group, not 64 width-8 jobs.
        constexpr int max_columns_total = kMaxTokens * 4;
        static_assert(max_columns_total == kMaxJobs, "helper IO and routing column capacities differ");
        for (int cols = 1; cols <= kMaxColumns; ++cols)
            workspace_bytes = std::max(workspace_bytes, size_t(max_columns_total / cols) *
                mf::fast_workspace_bytes(kHidden, kIntermediate, cols));
        allocated.pool = stride * size_t(capacity);
        const size_t io = kBufferElements * sizeof(uint16_t);
        const size_t jobs = size_t(kMaxJobs) * sizeof(DeviceJob);
        allocated.device_scratch = 2 * io + jobs + workspace_bytes;
        allocated.pinned = 2 * io + jobs;
        DeviceGuard guard(device);
        try {
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create helper stream");
            ck(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), "create helper event");
            ck(cudaMalloc(&pool, allocated.pool), "allocate helper records");
            ck(cudaMalloc(&workspace, workspace_bytes), "allocate helper workspace");
            ck(cudaMalloc(&d_x, io), "allocate helper inputs"); ck(cudaMalloc(&d_y, io), "allocate helper outputs");
            ck(cudaMalloc(&d_jobs, jobs), "allocate helper jobs");
            ck(cudaMallocHost(&h_x, io), "allocate pinned helper inputs"); ck(cudaMallocHost(&h_y, io), "allocate pinned helper outputs");
            ck(cudaMallocHost(&h_jobs, jobs), "allocate pinned helper jobs");
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (unsafe) return; // Unknown outstanding writer: quarantine all its buffers.
        if (stream && cudaStreamQuery(stream) != cudaSuccess) {
            unsafe = true; return; // A failed join cannot authorize freeing buffers.
        }
        if (done_event) cudaEventDestroy(done_event);
        if (stream) cudaStreamDestroy(stream);
        if (pool) cudaFree(pool); if (workspace) cudaFree(workspace);
        if (d_x) cudaFree(d_x); if (d_y) cudaFree(d_y); if (d_jobs) cudaFree(d_jobs);
        if (h_x) cudaFreeHost(h_x); if (h_y) cudaFreeHost(h_y); if (h_jobs) cudaFreeHost(h_jobs);
    }
    ~Impl() {
        { std::lock_guard<std::mutex> lock(mu); quit = true; }
        cv.notify_all(); if (worker.joinable()) worker.join();
    }
    void run() {
        struct ExitNotice {
            Impl* p;
            ~ExitNotice() {
                // Cleanup can itself block in a CUDA API. Keep it on this
                // heap-owned worker so the public destructor can time out and
                // retain the complete Impl instead of blocking or freeing it.
                try { DeviceGuard guard(p->device); p->release(); }
                catch (...) { p->unsafe = true; }
                std::lock_guard<std::mutex> lock(p->mu);
                p->worker_exited = true;
                p->cv.notify_all();
            }
        } exit_notice{this};
        try {
            DeviceGuard guard(device);
            for (;;) {
                int n; bool refresh;
                uint64_t worker_start_ns = 0, event_record_begin_ns = 0, event_record_end_ns = 0;
                uint64_t last_not_ready_ns = 0, ready_observed_ns = 0;
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return quit || queued || refresh_queued; });
                    if (quit && !queued && !refresh_queued) break;
                    refresh=refresh_queued; refresh_queued=false;
                    n = pending_n; if (!refresh) queued = false;
                    if (!refresh) worker_start_ns = host_now_ns();
                }
                if (refresh && refresh_validate) {
                    unsigned flags=0;
                    const bool invalid=!refresh_src || cudaHostGetFlags(&flags,const_cast<void*>(refresh_src))!=cudaSuccess || !(flags&cudaHostAllocPortable);
                    { std::lock_guard<std::mutex> lock(mu); refresh_invalid=invalid; refresh_done=true; }
                    cv.notify_all();
                    continue;
                }
                std::string error;
                bool uncertain = false;
                try {
                    if (refresh) {
                        ck(cudaMemcpyAsync(refresh_dst,refresh_src,refresh_bytes,refresh_kind,stream), "submit helper refresh copy");
                    } else {
                    const size_t io = size_t(pending_columns) * kHidden * sizeof(uint16_t);
                    ck(cudaMemcpyAsync(d_x, h_x, io, cudaMemcpyHostToDevice, stream), "copy helper input");
                    ck(cudaMemcpyAsync(d_jobs, h_jobs, size_t(n) * sizeof(DeviceJob), cudaMemcpyHostToDevice, stream), "copy helper jobs");
                    // Jobs are sorted by width; the shared fast API requires a
                    // common width for every descriptor in a launch. Workspace
                    // is reused only by consecutive launches on this stream.
                    for (int first = 0; first < n;) {
                        const int cols = pending_jobs[first].columns;
                        int end = first + 1;
                        while (end < n && pending_jobs[end].columns == cols) ++end;
                        if (size_t(end-first) * mf::fast_workspace_bytes(kHidden,kIntermediate,cols) > workspace_bytes)
                            throw std::runtime_error("helper width group exceeds workspace");
                        mf::fast_ffn_cuda_packed(d_jobs + first, end - first, kHidden, kIntermediate, cols, workspace, stream);
                        first = end;
                    }
                    ck(cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream), "copy helper output");
                    }
                    if (!refresh) event_record_begin_ns = host_now_ns();
                    ck(cudaEventRecord(done_event, stream), "record helper event");
                    if (!refresh) event_record_end_ns = host_now_ns();
                    const auto started = std::chrono::steady_clock::now();
                    const auto deadline = started + kProgressDeadline;
                    for (;;) {
                        const uint64_t query_begin_ns = refresh ? 0 : host_now_ns();
                        cudaError_t status = cudaEventQuery(done_event);
                        if (status == cudaSuccess) {
                            if (!refresh) ready_observed_ns = host_now_ns();
                            break;
                        }
                        if (status != cudaErrorNotReady) ck(status, "query helper event");
                        if (!refresh) last_not_ready_ns = query_begin_ns;
                        {
                            std::lock_guard<std::mutex> lock(mu);
                            if (quit) throw std::runtime_error("helper shutdown before completion");
                        }
                        const auto now = std::chrono::steady_clock::now();
                        if (now >= deadline)
                            throw std::runtime_error("helper progress deadline exceeded");
                        // Typical helper service is well below a millisecond.
                        // Avoid imposing a millisecond sleep on each request;
                        // only back off once the short active-poll window passes.
                        if (now - started < std::chrono::milliseconds(1))
                            std::this_thread::yield();
                        else
                            std::this_thread::sleep_for(std::chrono::microseconds(50));
                    }
                } catch (const std::exception& e) {
                    error = e.what();
                    // An error or failed event does not prove the stream has
                    // retired; quarantine on any failed stream synchronization.
                    uncertain = cudaStreamQuery(stream) != cudaSuccess;
                } catch (...) {
                    error = "unknown helper failure";
                    uncertain = cudaStreamQuery(stream) != cudaSuccess;
                }
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (!error.empty()) { failure = error; poisoned = true; unsafe |= uncertain; }
                    if (refresh) refresh_done = true;
                    else {
                        pass_timing.worker_start_ns = worker_start_ns;
                        pass_timing.event_record_begin_ns = event_record_begin_ns;
                        pass_timing.event_record_end_ns = event_record_end_ns;
                        pass_timing.last_not_ready_ns = last_not_ready_ns;
                        pass_timing.ready_observed_ns = ready_observed_ns;
                        pass_timing.worker_published_ns = host_now_ns();
                        finished = true;
                    }
                }
                cv.notify_all();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mu);
            failure = e.what(); poisoned = true; finished = true;
            cv.notify_all();
        }
    }
};

NativeHelper::NativeHelper(int device, int capacity) : p_(std::make_unique<Impl>(device, capacity)) {}
NativeHelper::~NativeHelper() {
    if (!p_) return;
    Impl* p = p_.get();
    {
        std::unique_lock<std::mutex> lock(p->mu);
        p->quit = true;
        if (!p->worker.joinable()) {
            // An unsealed helper still needs CUDA cleanup under the same
            // bounded ownership rule as a sealed helper.
            try { p->worker = std::thread([p] { p->run(); }); }
            catch (...) { (void)p_.release(); return; }
        }
        p->cv.notify_all();
        if (p->worker.joinable() &&
            !p->cv.wait_for(lock, kShutdownDeadline, [&] { return p->worker_exited; })) {
            // A CUDA API itself may fail to return. The worker holds a reference
            // to the complete heap-owned Impl; detach only after making that
            // object immortal, so neither its synchronization primitives nor
            // any GPU-visible storage can be destroyed under the worker.
            p->worker.detach();
            (void)p_.release();
            return;
        }
    }
}

bool NativeHelper::add_expert(int layer, int expert, const void* record, size_t bytes,
                              const float weight[3], const float input[3], std::string& error) {
    auto& p = *p_;
    if (p.sealed || p.poisoned || layer < 0 || layer >= kLayers || expert < 0 || expert >= kExperts ||
        !record || !weight || !input || bytes != p.stride || p.count == p.capacity ||
        p.records[size_t(layer) * kExperts + expert].slot >= 0) {
        error = "invalid or duplicate helper expert, record size, or sealed/full pool"; return false;
    }
    for (int i = 0; i < 3; ++i) if (!std::isfinite(weight[i]) || weight[i] <= 0 || !std::isfinite(input[i]) || input[i] <= 0 || (!std::isfinite(1.0f / weight[i]) || 1.0f / weight[i] == 0.0f)) {
        error = "invalid helper expert global scale"; return false;
    }
    // RG16 records consist of 128 code bytes followed by 16 E4M3
    // block scales in each unit; code nibbles are all valid E2M1 values.
    const auto* source = static_cast<const uint8_t*>(record);
    const size_t payload = mf::record_bytes(kHidden, kIntermediate);
    for (size_t off = 0; off < payload; off += 144) for (int row = 0; row < 16; ++row) {
        const uint8_t scale = source[off + 128 + row];
        if ((scale & 0x80u) || scale == 0x7fu) { error = "invalid RG16 block scale"; return false; }
    }
    try {
        DeviceGuard guard(p.device);
        ck(cudaMemcpy(p.pool + size_t(p.count) * p.stride, record, bytes, cudaMemcpyHostToDevice), "copy helper record");
    } catch (const std::exception& e) { error = e.what(); p.poisoned = true; p.unsafe = true; return false; }
    auto& r = p.records[size_t(layer) * kExperts + expert];
    r.slot = p.count++; std::copy_n(weight, 3, r.weight); std::copy_n(input, 3, r.input);
    return true;
}

bool NativeHelper::seal(std::string& error) {
    auto& p = *p_;
    if (p.sealed || p.poisoned) { error = "helper already sealed or failed"; return false; }
    try { p.worker = std::thread([&p] { p.run(); }); p.sealed = true; return true; }
    catch (const std::exception& e) { error = e.what(); return false; }
}
std::array<uint32_t, 4> NativeHelper::ready(int layer) const {
    std::array<uint32_t, 4> result{};
    std::lock_guard<std::mutex> lock(p_->mu);
    if (!p_->sealed || p_->poisoned || p_->refreshing || layer < 0 || layer >= kLayers) return result;
    for (int e = 0; e < kExperts; ++e) if (p_->records[size_t(layer) * kExperts + e].slot >= 0)
        result[e >> 5] |= uint32_t(1) << (e & 31);
    return result;
}
NativeHelper::Bytes NativeHelper::bytes() const { return p_->allocated; }

std::vector<NativeHelper::Identity> NativeHelper::records() const {
    std::lock_guard<std::mutex> lock(p_->mu);
    std::vector<Identity> result;
    if (!p_->sealed || p_->poisoned || p_->refreshing) return result;
    for (int layer=0;layer<kLayers;++layer) for (int expert=0;expert<kExperts;++expert) {
        const auto& r=p_->records[size_t(layer)*kExperts+expert];
        if (r.slot>=0) result.push_back({layer,expert,r.slot});
    }
    return result;
}
uintptr_t NativeHelper::pool_identity() const { return reinterpret_cast<uintptr_t>(p_->pool); }
size_t NativeHelper::record_stride() const { return p_->stride; }

namespace {
bool refresh_payload(const void* record, const float weight[3], const float input[3], std::string& error) {
    if (!weight || !input) { error="missing helper expert scales"; return false; }
    for (int i=0;i<3;++i) if (!std::isfinite(weight[i]) || weight[i]<=0 || !std::isfinite(input[i]) || input[i]<=0 ||
        !std::isfinite(1.0f/weight[i]) || 1.0f/weight[i]==0.0f) {
        error="invalid helper expert global scale"; return false;
    }
    const auto* source=static_cast<const uint8_t*>(record);
    const size_t payload=mf::record_bytes(NativeHelper::kHidden,NativeHelper::kIntermediate);
    for (size_t off=0;off<payload;off+=144) for (int row=0;row<16;++row) {
        const uint8_t scale=source[off+128+row];
        if ((scale&0x80u) || scale==0x7fu) { error="invalid RG16 block scale"; return false; }
    }
    return true;
}
bool valid_identity(int layer,int expert) { return layer>=0 && layer<kLayers && expert>=0 && expert<kExperts; }
}

bool NativeHelper::export_expert(int layer,int expert,void* staging,size_t bytes,
                                 float weight[3],float input[3],bool& staging_unsafe,std::string& error) {
    staging_unsafe=false; auto& p=*p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.refreshing || !valid_identity(layer,expert) || bytes!=p.stride || !weight || !input) {
        error="helper refresh unavailable, active, or invalid export"; return false;
    }
    const auto& r=p.records[size_t(layer)*kExperts+expert];
    if (r.slot<0) { error="helper export record is not ready"; return false; }
    if (!p.transfer(lock,nullptr,staging,0,cudaMemcpyDeviceToHost,staging_unsafe,error,true)) return false;
    if (!p.transfer(lock,staging,p.pool+size_t(r.slot)*p.stride,bytes,cudaMemcpyDeviceToHost,staging_unsafe,error)) return false;
    std::copy_n(r.weight,3,weight);std::copy_n(r.input,3,input);return true;
}
bool NativeHelper::replace_expert(int old_layer,int old_expert,int new_layer,int new_expert,
                                  const void* staging,size_t bytes,const float weight[3],const float input[3],
                                  bool& staging_unsafe,std::string& error) {
    staging_unsafe=false; auto& p=*p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.refreshing || !valid_identity(old_layer,old_expert) ||
        !valid_identity(new_layer,new_expert) || bytes!=p.stride) {
        error="helper refresh unavailable, active, or invalid replacement"; return false;
    }
    auto& old=p.records[size_t(old_layer)*kExperts+old_expert];
    auto& replacement=p.records[size_t(new_layer)*kExperts+new_expert];
    if (old.slot<0 || replacement.slot>=0) { error="helper replacement source absent or destination already ready"; return false; }
    if (!p.transfer(lock,nullptr,staging,0,cudaMemcpyHostToDevice,staging_unsafe,error,true) ||
        !refresh_payload(staging,weight,input,error)) return false;
    const int slot=old.slot;
    old.slot=-1; // no partially overwritten record is ever published
    if (!p.transfer(lock,p.pool+size_t(slot)*p.stride,staging,bytes,cudaMemcpyHostToDevice,staging_unsafe,error)) return false;
    replacement.slot=slot;std::copy_n(weight,3,replacement.weight);std::copy_n(input,3,replacement.input);
    return true;
}

uint64_t NativeHelper::submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error) {
    const uint64_t submit_enter_ns = host_now_ns();
    auto& p = *p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.refreshing || !x || tokens < 1 || tokens > kMaxTokens || jobs.empty() || jobs.size() > kMaxJobs) {
        error = "helper unavailable, request already active, or invalid request size"; return 0;
    }
    int per_token[kMaxTokens]{};
    int total_columns = 0;
    const int layer = jobs.front().layer;
    for (const Job& j : jobs) {
        if (j.layer != layer || j.layer < 0 || j.layer >= kLayers || j.expert < 0 || j.expert >= kExperts || j.columns < 1 || j.columns > kMaxColumns ||
            p.records[size_t(j.layer) * kExperts + j.expert].slot < 0) { error = "invalid or unready helper job"; return 0; }
        total_columns += j.columns;
        if (total_columns > kMaxTokens * 4) { error = "helper column capacity exceeded"; return 0; }
        for (int c = 0; c < j.columns; ++c) {
            if (j.token[c] < 0 || j.token[c] >= tokens || !std::isfinite(j.weight[c]) || j.weight[c] < 0 || ++per_token[j.token[c]] > 4) {
                error = "invalid helper token/weight or more than four picks"; return 0;
            }
        }
    }
    // Each token may name an expert at most once, including across grouped jobs.
    for (size_t i = 0; i < jobs.size(); ++i) for (int c = 0; c < jobs[i].columns; ++c) {
        for (size_t j = 0; j <= i; ++j) {
            if (jobs[j].expert != jobs[i].expert) continue;
            const int stop = j == i ? c : jobs[j].columns;
            for (int k = 0; k < stop; ++k) if (jobs[j].token[k] == jobs[i].token[c]) {
                error = "duplicate expert for helper token"; return 0;
            }
        }
        const uint16_t* row = x + size_t(jobs[i].token[c]) * kHidden;
        for (int k = 0; k < kHidden; ++k) if ((row[k] & 0x7f80u) == 0x7f80u) {
            error = "nonfinite helper activation"; return 0;
        }
    }
    size_t i = 0, offset = 0;
    for (int cols = 1; cols <= kMaxColumns; ++cols) for (const Job& j : jobs) {
        if (j.columns != cols) continue;
        const auto& r = p.records[size_t(j.layer) * kExperts + j.expert];
        p.pending_jobs[i] = j;
        auto& d = p.h_jobs[i]; d = {};
        d.record = p.pool + size_t(r.slot) * p.stride;
        for (int k = 0; k < 3; ++k) d.multiplier[k] = 1.0f / r.weight[k];
        p.pending_offsets[i] = offset;
        d.x = p.d_x + offset; d.y = p.d_y + offset; d.columns = j.columns;
        for (int c = 0; c < j.columns; ++c)
            std::memcpy(p.h_x + offset + size_t(c) * kHidden, x + size_t(j.token[c]) * kHidden, size_t(kHidden) * sizeof(uint16_t));
        offset += size_t(j.columns) * kHidden;
        ++i;
    }
    p.pending_columns = int(offset / kHidden);
    if (++p.next == 0) ++p.next;
    p.active = p.next; p.pending_n = int(jobs.size()); p.pending_tokens = tokens; p.finished = false;
    p.collector_claimed = false;
    p.pass_timing = {};
    p.pass_timing.submit_enter_ns = submit_enter_ns;
    p.pass_timing.enqueue_ns = host_now_ns();
    p.queued = true;
    p.cv.notify_one();
    p.pass_timing.submit_published_ns = host_now_ns();
    return p.active;
}

bool NativeHelper::collect(uint64_t seq, float* out, size_t n, std::string& error, PassTiming* timing) {
    const uint64_t collect_enter_ns = host_now_ns();
    auto& p = *p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!seq || seq != p.active || !out || n < size_t(p.pending_tokens) * kHidden) { error = "invalid helper sequence or output"; return false; }
    if (p.collector_claimed) { error = "helper sequence already has a collector"; return false; }
    p.collector_claimed = true;
    const uint64_t collect_wait_begin_ns = host_now_ns();
    if (!p.cv.wait_for(lock, kProgressDeadline, [&] { return p.finished || p.poisoned; })) {
        p.failure = "helper progress deadline exceeded";
        p.poisoned = true;
        p.unsafe = true; // unknown writer: never reuse or free its allocations
        error = p.failure;
        return false;
    }
    const uint64_t collect_wait_end_ns = host_now_ns();
    if (p.poisoned) { error = p.failure; return false; }
    if (p.active != seq) { error = "helper sequence changed before retirement"; return false; }
    std::fill_n(out, size_t(p.pending_tokens) * kHidden, 0.0f);
    for (int i = 0; i < p.pending_n; ++i) {
        const Job& j = p.pending_jobs[i];
        for (int c = 0; c < j.columns; ++c) {
            float* dst = out + size_t(j.token[c]) * kHidden;
            const uint16_t* src = p.h_y + p.pending_offsets[i] + size_t(c) * kHidden;
            for (int k = 0; k < kHidden; ++k) {
                if ((src[k] & 0x7f80u) == 0x7f80u) {
                    p.failure = "nonfinite helper output"; p.poisoned = true; error = p.failure; return false;
                }
                const float sum = dst[k] + j.weight[c] * from_bf16(src[k]);
                if (!std::isfinite(sum)) {
                    p.failure = "nonfinite helper weighted sum"; p.poisoned = true; error = p.failure; return false;
                }
                dst[k] = sum;
            }
        }
    }
    if (timing) {
        p.pass_timing.collect_enter_ns = collect_enter_ns;
        p.pass_timing.collect_wait_begin_ns = collect_wait_begin_ns;
        p.pass_timing.collect_wait_end_ns = collect_wait_end_ns;
        p.pass_timing.collect_done_ns = host_now_ns();
        *timing = p.pass_timing;
    }
    p.active = 0; p.finished = false; p.collector_claimed = false; return true;
}
}
