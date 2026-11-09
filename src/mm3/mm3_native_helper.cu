// Ordered helper queue adapted from Maya DS4 helper_window/helper_loop and
// helper_enqueue. Persistent pool and joined reader/slot lifetimes follow the
// Infernix ExpertHelper/PeerExpertExecutor ownership boundary. MiniMax's RG16
// FFN implementation is supplied by Infernix's minimax_nvfp4_ffn; this class
// deliberately owns no routing or refill policy.
#include "mm3_native_helper.hpp"
#include "mm3_capture.hpp"
#include "mm3_kernels.cuh"
#include "mm3_helper_merge.hpp"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace mm3 {
#ifdef MM3_HELPER_TEST_HOOKS
// Test-only hook is outside the production build and runs on the CPU worker.
void native_helper_test_cpu_copy(bool recycle);
void native_helper_test_gpu_copy(bool upload);
void native_helper_test_worker_start();
void native_helper_test_worker_exit();
#endif
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
// One thread polls a host word until the next pass claims the stream, or a bound passes. Without
// the bound a lost host would leave the stream occupied. It reads no record and writes nothing.
__global__ void occupancy_kernel(const volatile unsigned* word, unsigned expected, unsigned long long cycles) {
    const unsigned long long begin = clock64();
    while (*word == expected && clock64() - begin < cycles) {}
}
bool refresh_payload(const void*, const float[3], const float[3], std::string&);
bool valid_identity(int, int);
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
    std::array<cudaEvent_t,4> replay_events{};
    bool probe_timing=false;
    std::array<Record, kLayers * kExperts> records{};
    std::array<Job, kMaxJobs> pending_jobs{};
    int pending_n = 0, pending_tokens = 0, pending_columns = 0;
    std::array<size_t, kMaxJobs> pending_offsets{};
    uint64_t next = 0, active = 0;
    bool worker_compute = true;
    bool sealed = false, queued = false, finished = false, quit = false, poisoned = false, unsafe = false;
    bool worker_exited = false, cleanup_exited = false;
    bool collector_claimed = false; // exactly one retirement owner per sequence
    PassTiming pass_timing{}; // protected by mu except worker-local endpoints
    std::string failure;
    std::string pending_capture_context; // snapshotted on submit under mu
    mutable std::mutex traffic_mu;
    WeightTraffic traffic{};
    void weight_issue(bool upload,size_t bytes) {
        std::lock_guard<std::mutex> lock(traffic_mu);
        if(upload) {++traffic.h2d_issued_ops;traffic.h2d_issued_bytes+=bytes;}
        else {++traffic.d2h_issued_ops;traffic.d2h_issued_bytes+=bytes;}
    }
    void weight_failure() {std::lock_guard<std::mutex> lock(traffic_mu);++traffic.copy_failures;}
    void weight_replacement_complete() {std::lock_guard<std::mutex> lock(traffic_mu);++traffic.replacements_completed;}
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;
    // helper_fast: lock-free mirrors of queued and finished that a waiter may spin on before its
    // condition-variable wait, whose predicate still decides; and one graph per job count for one-column
    // passes, built on first use by the worker thread (the only user of the stream).
    std::atomic<bool> fast_mode{false}, queued_flag{false}, finished_flag{false};
    std::atomic<uint64_t> graph_count{0};
    // helper_zero_copy: the graph's three host copies (inputs and descriptors up, results down) as copy kernels on
    // device views of the pinned buffers - on WSL2 a graph's host copy node costs ~13-18 us, and helper_graph_bench on
    // the 3090 measured -18 us a launch (2026-11-09). Its graphs are a second set ([1]), so switching rebuilds nothing.
    // (2: also check each such pass: the three buffers read back through the copy engine against the bytes moved)
    std::atomic<int> zero_copy_mode{0};
    std::atomic<uint64_t> zc_checked{0}, zc_mismatches{0};
    std::vector<uint8_t> zc_back;   // (the check's read-back, worker thread)
    void *v_x = nullptr, *v_y = nullptr, *v_jobs = nullptr;   // device views of h_x, h_y, h_jobs (worker thread)
    std::array<std::array<cudaGraphExec_t, kMaxJobs + 1>, 2> graph_exec{};
    std::array<std::array<bool, kMaxJobs + 1>, 2> graph_failed{};
    static void spin_until(const std::atomic<bool>& flag, std::chrono::microseconds limit) {
        const auto started = std::chrono::steady_clock::now();
        while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() - started < limit) {}
    }
    // Capture the pass's submission for n one-column jobs: inputs and descriptors to the device, the grouped
    // expert launch, the result back. Buffers and workspace are fixed, so only the descriptors' contents change.
    // zc: the copies as kernels on the pinned buffers' device views (copy_words: whole 4-byte words) instead of
    // memcpy nodes. A graph that fails to build is not retried; its passes take the separate calls.
    static_assert(sizeof(DeviceJob) % 4 == 0 && (kHidden * sizeof(uint16_t)) % 4 == 0, "copy_words copies whole words");
    bool views_ready() {
        if (v_x && v_y && v_jobs) return true;
        void *a = nullptr, *b = nullptr, *c = nullptr;
        if (cudaHostGetDevicePointer(&a, h_x, 0) != cudaSuccess || cudaHostGetDevicePointer(&b, h_y, 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&c, h_jobs, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        v_x = a; v_y = b; v_jobs = c;
        return true;
    }
    bool graph_ready(int n, bool zc) {
        if (n < 1 || n > kMaxJobs || graph_failed[zc][size_t(n)]) return false;
        cudaGraphExec_t& exec = graph_exec[zc][size_t(n)];
        if (exec) return true;
        const size_t io = size_t(n) * kHidden * sizeof(uint16_t), jobs = size_t(n) * sizeof(DeviceJob);
        if (size_t(n) * mf::fast_workspace_bytes(kHidden, kIntermediate, 1) > workspace_bytes || (zc && !views_ready())) {
            graph_failed[zc][size_t(n)] = true; return false;
        }
        cudaGraph_t graph = nullptr;
        bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            if (zc) k::copy_words(v_x, d_x, io, stream, v_jobs, d_jobs, jobs);   // (a failed launch fails the capture)
            else ok = cudaMemcpyAsync(d_x, h_x, io, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                      cudaMemcpyAsync(d_jobs, h_jobs, jobs, cudaMemcpyHostToDevice, stream) == cudaSuccess;
            if (ok) mf::fast_ffn_cuda_packed(d_jobs, n, kHidden, kIntermediate, 1, workspace, stream);
            if (ok && zc) k::copy_words(d_y, v_y, io, stream);
            else ok = ok && cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
            const bool ended = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
            ok = ok && ended && graph && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
        }
        if (graph) cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        if (!ok) { exec = nullptr; graph_failed[zc][size_t(n)] = true; }
        return ok;
    }
    void graph_destroy() {
        for (auto& set : graph_exec) for (auto& g : set) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
    }

    // Allocated and started only for an explicitly requested live replacement.
    // Neither thread holds mu during CPU copies or CUDA calls. A stuck call can
    // therefore be reported by live_poll and quarantined on destruction.
    struct Live {
        LivePhase phase = LivePhase::idle;
        std::thread cpu, gpu;
        std::condition_variable cv;
        bool stop = false, cpu_exited = false, gpu_exited = false;
        bool cpu_queued = false, gpu_queued = false, cpu_busy = false, gpu_busy = false;
        bool cpu_done = false, gpu_done = false, gpu_uncertain = false;
        bool gpu_failure_recorded = false;
        bool recycle = false, upload = false;
        int old_id = -1, new_id = -1, slot = -1;
        void *host = nullptr, *incoming = nullptr, *outgoing = nullptr;
        float weight[3]{}, input[3]{};
        std::string capture_context; // captured when the lease was submitted
        std::chrono::steady_clock::time_point started{};
        std::string error;
    };
    std::unique_ptr<Live> live;

    void live_fail_locked(const std::string& error, bool uncertain) {
        auto& l=*live;
        if(uncertain && l.gpu_busy && !l.gpu_failure_recorded) {weight_failure();l.gpu_failure_recorded=true;}
        if (l.phase != LivePhase::failed) l.error=error;
        l.phase=LivePhase::failed; l.gpu_uncertain |= uncertain;
        poisoned=true; unsafe |= uncertain; failure=l.error;
    }
    void live_advance_locked() {
        auto& l=*live;
        if (l.phase == LivePhase::failed || l.phase == LivePhase::idle) return;
        if (poisoned) { live_fail_locked(failure.empty()?"helper failed during live replacement":failure,false); return; }
        if ((l.cpu_busy || l.gpu_busy) && std::chrono::steady_clock::now()-l.started >= kProgressDeadline) {
            live_fail_locked("live helper progress deadline exceeded", l.gpu_busy);
            return;
        }
        if (l.phase==LivePhase::preparing && l.cpu_done && l.gpu_done) l.phase=LivePhase::prepared;
        else if (l.phase==LivePhase::uploading && l.gpu_done) l.phase=LivePhase::uploaded;
        else if (l.phase==LivePhase::recycling && l.cpu_done) l.phase=LivePhase::complete;
    }
    void live_cpu_run() {
        for (;;) {
            bool recycle; void *host, *incoming, *outgoing; std::string capture_context;
            {
                std::unique_lock<std::mutex> lock(mu);
                live->cv.wait(lock,[&]{return live->stop || live->cpu_queued;});
                if (live->stop && !live->cpu_queued) break;
                auto& l=*live; recycle=l.recycle; host=l.host; incoming=l.incoming; outgoing=l.outgoing;
                capture_context=l.capture_context;
                l.cpu_queued=false;
            }
            std::string error;
            if(!capture_context.empty())nvtxRangePushA((capture_context+(recycle?" phase=helper_live_cpu_recycle":" phase=helper_live_cpu_prepare")).c_str());
            try {
#ifdef MM3_HELPER_TEST_HOOKS
                native_helper_test_cpu_copy(recycle);
#endif
                if (recycle) std::memcpy(host,outgoing,stride);
                else {
                    std::memcpy(incoming,host,stride);
                    if (!refresh_payload(incoming,live->weight,live->input,error) && error.empty()) error="invalid live helper payload";
                }
            } catch (const std::exception& e) { error=e.what(); }
              catch (...) { error="unknown live helper CPU failure"; }
            if(!capture_context.empty())nvtxRangePop();
            {
                std::lock_guard<std::mutex> lock(mu);
                auto& l=*live; l.cpu_busy=false; l.cpu_done=error.empty();
                if (!error.empty()) live_fail_locked(error,false);
                live_advance_locked();
            }
            cv.notify_all();
        }
        { std::lock_guard<std::mutex> lock(mu); live->cpu_exited=true; }
        cv.notify_all();
    }
    void live_gpu_run() {
        cudaStream_t copy_stream=nullptr; cudaEvent_t event=nullptr;
        for (;;) {
            bool upload; void *incoming, *outgoing; int slot; std::string capture_context;
            {
                std::unique_lock<std::mutex> lock(mu);
                live->cv.wait(lock,[&]{return live->stop || live->gpu_queued;});
                if (live->stop && !live->gpu_queued) break;
                auto& l=*live; upload=l.upload; incoming=l.incoming; outgoing=l.outgoing; slot=l.slot;
                capture_context=l.capture_context;
                l.gpu_queued=false;
            }
            std::string error; bool uncertain=false, submitted=false;
            if(!capture_context.empty())nvtxRangePushA((capture_context+(upload?" phase=helper_live_h2d":" phase=helper_live_d2h")).c_str());
            try {
                DeviceGuard guard(device);
                if (!copy_stream) ck(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking),"create live helper stream");
                if (!event) ck(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"create live helper event");
                unsigned flags=0;
                ck(cudaHostGetFlags(&flags,incoming),"inspect incoming staging");
                if (!(flags&cudaHostAllocPortable)) throw std::runtime_error("incoming staging is not portable pinned memory");
                ck(cudaHostGetFlags(&flags,outgoing),"inspect outgoing staging");
                if (!(flags&cudaHostAllocPortable)) throw std::runtime_error("outgoing staging is not portable pinned memory");
                // A failed submission can still leave stream state uncertain.
                submitted=true;
                ck(cudaMemcpyAsync(upload ? pool+size_t(slot)*stride : outgoing,
                                   upload ? incoming : pool+size_t(slot)*stride, stride,
                                   upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,copy_stream),"submit live helper copy");
                weight_issue(upload,stride);
                if(!capture_context.empty())capture::detail_event("helper_weight_copy_issue",
                    "\"submission_context\":"+capture::quote(capture_context)+",\"operation\":\"live\",\"direction\":"+
                    capture::quote(upload?"h2d":"d2h")+",\"bytes\":"+std::to_string(stride)+",\"slot\":"+std::to_string(slot));
                ck(cudaEventRecord(event,copy_stream),"record live helper event");
#ifdef MM3_HELPER_TEST_HOOKS
                // Gate this operation's completion independently of compute.
                native_helper_test_gpu_copy(upload);
#endif
                const auto deadline=std::chrono::steady_clock::now()+kProgressDeadline;
                for (;;) {
                    const auto status=cudaEventQuery(event);
                    if (status==cudaSuccess) break;
                    if (status!=cudaErrorNotReady) ck(status,"query live helper event");
                    if (std::chrono::steady_clock::now()>=deadline) throw std::runtime_error("live helper progress deadline exceeded");
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
            } catch (const std::exception& e) {
                error=e.what();
                if (submitted) { try { DeviceGuard guard(device); uncertain=cudaStreamQuery(copy_stream)!=cudaSuccess; } catch (...) { uncertain=true; } }
            } catch (...) { error="unknown live helper CUDA failure"; uncertain=submitted; }
            if(!capture_context.empty())nvtxRangePop();
            if(!capture_context.empty())capture::detail_event("helper_weight_copy_completion",
                "\"submission_context\":"+capture::quote(capture_context)+",\"operation\":\"live\",\"direction\":"+
                capture::quote(upload?"h2d":"d2h")+",\"bytes\":"+std::to_string(stride)+",\"slot\":"+std::to_string(slot)+
                ",\"status\":"+capture::quote(error.empty()?"completed":"failed")+",\"error\":"+capture::quote(error));
            {
                std::lock_guard<std::mutex> lock(mu);
                auto& l=*live; l.gpu_busy=false; l.gpu_done=error.empty();
                if(!error.empty() && !l.gpu_failure_recorded) {weight_failure();l.gpu_failure_recorded=true;}
                if (!error.empty()) live_fail_locked(error,uncertain);
                live_advance_locked();
            }
            cv.notify_all();
        }
        // If retirement is uncertain, retain the stream/event as well as pool.
        bool uncertain;
        { std::lock_guard<std::mutex> lock(mu); uncertain=live->gpu_uncertain; }
        if (!uncertain) {
            try { DeviceGuard guard(device); if (event) cudaEventDestroy(event); if (copy_stream) cudaStreamDestroy(copy_stream); }
            catch (...) { std::lock_guard<std::mutex> lock(mu); unsafe=true; live->gpu_uncertain=true; }
        }
        { std::lock_guard<std::mutex> lock(mu); live->gpu_exited=true; }
        cv.notify_all();
    }

    bool refreshing = false, refresh_queued = false, refresh_done = false, refresh_validate = false, refresh_invalid = false;
    bool refresh_failure_recorded = false; // guarded by mu, one count per operation
    void* refresh_dst = nullptr;
    const void* refresh_src = nullptr;
    size_t refresh_bytes = 0;
    cudaMemcpyKind refresh_kind = cudaMemcpyHostToDevice;
    std::string refresh_capture_context; // protected by mu

    // The worker owns CUDA submission and event polling, so an API call that
    // never returns cannot hold mu or prevent bounded caller/shutdown waits.
    bool transfer(std::unique_lock<std::mutex>& lock, void* dst, const void* src,
                  size_t bytes, cudaMemcpyKind kind, bool& staging_unsafe, std::string& error, bool validate=false) {
        refresh_validate=validate; refresh_invalid=false; refresh_failure_recorded=false; refresh_dst=dst; refresh_src=src; refresh_bytes=bytes; refresh_kind=kind;
        refresh_capture_context=capture::detailed()?capture::context():std::string{};
        refreshing=true; refresh_done=false; refresh_queued=true; cv.notify_one();
        if (!cv.wait_for(lock,kProgressDeadline,[&]{ return refresh_done || poisoned; })) {
            if(!validate && !refresh_failure_recorded) {weight_failure();refresh_failure_recorded=true;}
            poisoned=true; unsafe=true; failure=error="helper refresh progress deadline exceeded";
            staging_unsafe=true; return false;
        }
        if (poisoned) { staging_unsafe=unsafe; error=failure; return false; }
        refreshing=false;
        if (refresh_invalid) { error="helper refresh requires portable pinned staging"; return false; }
        return true;
    }

    // Occupancy fills the gaps between passes on this same stream, so a pass never queues behind
    // it: the worker releases the spinner just before submitting. A separate continuously running
    // kernel held the clocks but made each pass wait about 2.9 ms longer (helper-occupancy-20261107).
    // About 20 ms at 2 GHz and 130 ms at 300 MHz, well beyond any gap inside generation.
    static constexpr unsigned long long kOccupancyCycles = 40000000;
    std::atomic<bool> occupancy_want{false};
    std::atomic<uint64_t> occupancy_count{0};
    unsigned* h_occupancy = nullptr;
    bool occupancy_armed = false;  // worker thread only

    void occupancy_release() {
        if (h_occupancy) __atomic_add_fetch(h_occupancy, 1u, __ATOMIC_SEQ_CST);
    }
    void occupancy_arm() {
        const unsigned expected = __atomic_load_n(h_occupancy, __ATOMIC_SEQ_CST);
        occupancy_kernel<<<1, 1, 0, stream>>>(h_occupancy, expected, kOccupancyCycles);
        if (cudaGetLastError() != cudaSuccess) { occupancy_want.store(false); return; }
        (void)cudaStreamQuery(stream);  // WSL groups launches; start this one now
        (void)cudaGetLastError();
        occupancy_armed = true;
        occupancy_count.fetch_add(1, std::memory_order_relaxed);
    }

    Impl(int dev, int cap, bool compute) : device(dev), capacity(cap), worker_compute(compute) {
        if (dev < 0 || cap < 1 || cap > kLayers * kExperts) throw std::invalid_argument("invalid helper device/capacity");
        stride = mf::record_stride(kHidden, kIntermediate);
        // submit() admits at most four picks for each of 16 tokens. Jobs of
        // equal width launch as one group, groups run in order on one stream.
        // Bound the workspace by the largest possible group, not 64 width-8 jobs.
        constexpr int max_columns_total = kMaxTokens * 4;
        static_assert(max_columns_total == kMaxJobs, "helper IO and routing column capacities differ");
        for (int cols = 1; compute && cols <= kMaxColumns; ++cols)
            workspace_bytes = std::max(workspace_bytes, size_t(max_columns_total / cols) *
                mf::fast_workspace_bytes(kHidden, kIntermediate, cols));
        allocated.pool = stride * size_t(capacity);
        const size_t io = kBufferElements * sizeof(uint16_t);
        const size_t jobs = size_t(kMaxJobs) * sizeof(DeviceJob);
        allocated.device_scratch = compute ? 2 * io + jobs + workspace_bytes : 0;
        allocated.pinned = compute ? 2 * io + jobs : 0;
        DeviceGuard guard(device);
        try {
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create helper stream");
            ck(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), "create helper event");
            const char* replay=std::getenv("MM3_REPLAY_CAPTURE");
            if(replay && std::string(replay)=="1")for(auto& event:replay_events)
                ck(cudaEventCreateWithFlags(&event,0),"create helper replay timing event");
            ck(cudaMalloc(&pool, allocated.pool), "allocate helper records");
            if(compute) {ck(cudaMalloc(&workspace, workspace_bytes), "allocate helper workspace");
            ck(cudaMalloc(&d_x, io), "allocate helper inputs"); ck(cudaMalloc(&d_y, io), "allocate helper outputs");
            ck(cudaMalloc(&d_jobs, jobs), "allocate helper jobs");
            ck(cudaMallocHost(&h_x, io), "allocate pinned helper inputs"); ck(cudaMallocHost(&h_y, io), "allocate pinned helper outputs");
            ck(cudaMallocHost(&h_jobs, jobs), "allocate pinned helper jobs");
            ck(cudaMallocHost(reinterpret_cast<void**>(&h_occupancy), 64), "allocate occupancy word");
            *h_occupancy = 0;}
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (unsafe) return; // Unknown outstanding writer: quarantine all its buffers.
        occupancy_release();
        if (h_occupancy && stream && occupancy_count.load() && cudaStreamSynchronize(stream) != cudaSuccess) {
            unsafe = true; return;
        }
        if (stream && cudaStreamQuery(stream) != cudaSuccess) {
            unsafe = true; return; // A failed join cannot authorize freeing buffers.
        }
        if (done_event) cudaEventDestroy(done_event);
        for(auto event:replay_events)if(event)cudaEventDestroy(event);
        graph_destroy();
        if (stream) cudaStreamDestroy(stream);
        if (pool) cudaFree(pool); if (workspace) cudaFree(workspace);
        if (d_x) cudaFree(d_x); if (d_y) cudaFree(d_y); if (d_jobs) cudaFree(d_jobs);
        if (h_x) cudaFreeHost(h_x); if (h_y) cudaFreeHost(h_y); if (h_jobs) cudaFreeHost(h_jobs); if (h_occupancy) cudaFreeHost(h_occupancy);
    }
    ~Impl() {
        { std::lock_guard<std::mutex> lock(mu); quit = true; }
        cv.notify_all(); if (worker.joinable()) worker.join();
    }
    void run() {
        struct ExitNotice {
            Impl* p;
            ~ExitNotice() {
                // Even an early startup failure must not free the pool: the
                // independent live stream may already be reading it. The
                // destructor coordinates both workers before physical cleanup.
                { std::lock_guard<std::mutex> lock(p->mu); p->worker_exited = true; }
#ifdef MM3_HELPER_TEST_HOOKS
                native_helper_test_worker_exit();
#endif
                p->cv.notify_all();
            }
        } exit_notice{this};
        try {
#ifdef MM3_HELPER_TEST_HOOKS
            native_helper_test_worker_start();
#endif
            DeviceGuard guard(device);
            for (;;) {
                int n; bool refresh; std::string worker_capture_context;
                uint64_t worker_start_ns = 0, event_record_begin_ns = 0, event_record_end_ns = 0;
                uint64_t last_not_ready_ns = 0, ready_observed_ns = 0;
                float activation_ms=0,compute_ms=0,result_ms=0;
                bool gpu_timed=false;
                if (fast_mode.load(std::memory_order_relaxed)) spin_until(queued_flag, std::chrono::microseconds(2000));
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return quit || queued || refresh_queued; });
                    if (quit && !queued && !refresh_queued) break;
                    refresh=refresh_queued; refresh_queued=false;
                    n = pending_n; if (!refresh) { queued = false; queued_flag.store(false, std::memory_order_relaxed); }
                    if(!refresh) worker_capture_context=pending_capture_context;
                    else worker_capture_context=refresh_capture_context;
                    if (!refresh) worker_start_ns = host_now_ns();
                    gpu_timed=!refresh && replay_events[0] && (probe_timing || capture::detailed());
                }
                if (refresh && refresh_validate) {
                    unsigned flags=0;
                    const bool invalid=!refresh_src || cudaHostGetFlags(&flags,const_cast<void*>(refresh_src))!=cudaSuccess || !(flags&cudaHostAllocPortable);
                    { std::lock_guard<std::mutex> lock(mu); refresh_invalid=invalid; refresh_done=true; }
                    cv.notify_all();
                    continue;
                }
                std::string error;
                bool uncertain = false, zc_check = false;
                if(!worker_capture_context.empty()) nvtxRangePushA((worker_capture_context+" phase=helper_worker_cuda").c_str());
                if (occupancy_armed) { occupancy_release(); occupancy_armed = false; }
                try {
                    if (refresh) {
                        ck(cudaMemcpyAsync(refresh_dst,refresh_src,refresh_bytes,refresh_kind,stream), "submit helper refresh copy");
                        weight_issue(refresh_kind==cudaMemcpyHostToDevice,refresh_bytes);
                        if(!worker_capture_context.empty())capture::detail_event("helper_weight_copy_issue",
                            "\"submission_context\":"+capture::quote(worker_capture_context)+",\"operation\":\"placement\",\"direction\":"+capture::quote(refresh_kind==cudaMemcpyHostToDevice?"h2d":"d2h")+
                            ",\"bytes\":"+std::to_string(refresh_bytes));
                    } else if (const int zc_mode = zero_copy_mode.load(std::memory_order_relaxed);
                               !gpu_timed && fast_mode.load(std::memory_order_relaxed) && pending_columns == n && graph_ready(n, zc_mode > 0)) {
                        // every job has one column (the total equals the job count): one launch replaces five calls
                        ck(cudaGraphLaunch(graph_exec[zc_mode > 0][size_t(n)], stream), "launch helper graph");
                        zc_check = zc_mode == 2;
                        graph_count.fetch_add(1, std::memory_order_relaxed);
                    } else {
                    const size_t io = size_t(pending_columns) * kHidden * sizeof(uint16_t);
                    if(gpu_timed)ck(cudaEventRecord(replay_events[0],stream),"helper activation timing");
                    ck(cudaMemcpyAsync(d_x, h_x, io, cudaMemcpyHostToDevice, stream), "copy helper input");
                    ck(cudaMemcpyAsync(d_jobs, h_jobs, size_t(n) * sizeof(DeviceJob), cudaMemcpyHostToDevice, stream), "copy helper jobs");
                    if(gpu_timed)ck(cudaEventRecord(replay_events[1],stream),"helper compute timing");
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
                    if(gpu_timed)ck(cudaEventRecord(replay_events[2],stream),"helper result timing");
                    ck(cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream), "copy helper output");
                    if(gpu_timed)ck(cudaEventRecord(replay_events[3],stream),"helper finish timing");
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
                    if (zc_check) {
                        // helper_zero_copy=2: what the copy kernels moved, read back through the copy engine (the
                        // stream has retired; the host writes the inputs only between passes, so they are as sent)
                        const size_t io = size_t(n) * kHidden * sizeof(uint16_t), jb = size_t(n) * sizeof(DeviceJob);
                        zc_back.resize(io);
                        ck(cudaMemcpy(zc_back.data(), d_x, io, cudaMemcpyDeviceToHost), "check helper zero-copy input");
                        bool same = std::memcmp(zc_back.data(), h_x, io) == 0;
                        ck(cudaMemcpy(zc_back.data(), d_jobs, jb, cudaMemcpyDeviceToHost), "check helper zero-copy jobs");
                        same = same && std::memcmp(zc_back.data(), h_jobs, jb) == 0;
                        ck(cudaMemcpy(zc_back.data(), d_y, io, cudaMemcpyDeviceToHost), "check helper zero-copy result");
                        same = same && std::memcmp(zc_back.data(), h_y, io) == 0;
                        (same ? zc_checked : zc_mismatches).fetch_add(1, std::memory_order_relaxed);
                    }
                    if(gpu_timed) {
                        ck(cudaEventElapsedTime(&activation_ms,replay_events[0],replay_events[1]),"helper activation elapsed");
                        ck(cudaEventElapsedTime(&compute_ms,replay_events[1],replay_events[2]),"helper compute elapsed");
                        ck(cudaEventElapsedTime(&result_ms,replay_events[2],replay_events[3]),"helper result elapsed");
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
                if(!worker_capture_context.empty()) nvtxRangePop();
                if(refresh && !worker_capture_context.empty())capture::detail_event("helper_weight_copy_completion",
                    "\"submission_context\":"+capture::quote(worker_capture_context)+",\"operation\":\"placement\",\"direction\":"+
                    capture::quote(refresh_kind==cudaMemcpyHostToDevice?"h2d":"d2h")+",\"bytes\":"+std::to_string(refresh_bytes)+
                    ",\"status\":"+capture::quote(error.empty()?"completed":"failed")+",\"error\":"+capture::quote(error));
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if(refresh && !error.empty() && !refresh_failure_recorded) {weight_failure();refresh_failure_recorded=true;}
                    if (!error.empty()) { failure = error; poisoned = true; unsafe |= uncertain; }
                    if (refresh) refresh_done = true;
                    else {
                        pass_timing.worker_start_ns = worker_start_ns;
                        pass_timing.gpu_measured=gpu_timed;
                        pass_timing.activation_gpu_ms=activation_ms;pass_timing.compute_gpu_ms=compute_ms;pass_timing.result_gpu_ms=result_ms;
                        pass_timing.event_record_begin_ns = event_record_begin_ns;
                        pass_timing.event_record_end_ns = event_record_end_ns;
                        pass_timing.last_not_ready_ns = last_not_ready_ns;
                        pass_timing.ready_observed_ns = ready_observed_ns;
                        pass_timing.worker_published_ns = host_now_ns();
                        finished = true;
                        finished_flag.store(true, std::memory_order_release);
                    }
                }
                cv.notify_all();
                if (!refresh && error.empty() && h_occupancy && occupancy_want.load(std::memory_order_relaxed)) occupancy_arm();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mu);
            failure = e.what(); poisoned = true; finished = true;
            cv.notify_all();
        }
    }
};

NativeHelper::NativeHelper(int device, int capacity, bool compute) : p_(std::make_unique<Impl>(device, capacity, compute)) {}
void NativeHelper::set_fast(bool on) { p_->fast_mode.store(on); }
bool NativeHelper::fast() const { return p_->fast_mode.load(); }
void NativeHelper::set_zero_copy(int mode) { p_->zero_copy_mode.store(mode); }
int NativeHelper::zero_copy() const { return p_->zero_copy_mode.load(); }
uint64_t NativeHelper::zero_copy_checked() const { return p_->zc_checked.load(std::memory_order_relaxed); }
uint64_t NativeHelper::zero_copy_mismatches() const { return p_->zc_mismatches.load(std::memory_order_relaxed); }
uint64_t NativeHelper::graph_launches() const { return p_->graph_count.load(); }
void NativeHelper::set_occupancy(bool on) {
    p_->occupancy_want.store(on);
    if (!on) p_->occupancy_release();
}
bool NativeHelper::occupancy() const { return p_->occupancy_want.load(); }
uint64_t NativeHelper::occupancy_launches() const { return p_->occupancy_count.load(std::memory_order_relaxed); }

NativeHelper::~NativeHelper() {
    if (!p_) return;
    Impl* p = p_.get();
    p->occupancy_want.store(false);
    p->occupancy_release();
    if (p->live) {
        bool retired;
        {
            std::unique_lock<std::mutex> lock(p->mu);
            p->live->stop=true; p->live->cv.notify_all();
            retired=p->cv.wait_for(lock,kShutdownDeadline,[&] {
                return (!p->live->cpu.joinable() || p->live->cpu_exited) &&
                       (!p->live->gpu.joinable() || p->live->gpu_exited);
            });
            if (!retired) { p->unsafe=true; p->quit=true; p->cv.notify_all(); }
        }
        if (!retired) {
            // Threads can still hold both Impl and borrowed external storage.
            // live_retire must have told the caller to retain that storage.
            if (p->live->cpu.joinable()) p->live->cpu.detach();
            if (p->live->gpu.joinable()) p->live->gpu.detach();
            if (p->worker.joinable()) p->worker.detach();
            (void)p_.release(); return;
        }
        if (p->live->cpu.joinable()) p->live->cpu.join();
        if (p->live->gpu.joinable()) p->live->gpu.join();
    }
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
    // Both live workers have exited and the compute worker has exited. Join it
    // before freeing anything, then run potentially blocking CUDA cleanup on a
    // separate thread so shutdown remains bounded if a driver call hangs.
    if (p->worker.joinable()) p->worker.join();
    std::thread cleanup;
    try {
        cleanup=std::thread([p] {
            try { DeviceGuard guard(p->device); p->release(); }
            catch (...) { std::lock_guard<std::mutex> lock(p->mu); p->unsafe=true; }
            { std::lock_guard<std::mutex> lock(p->mu); p->cleanup_exited=true; }
            p->cv.notify_all();
        });
    } catch (...) { (void)p_.release(); return; }
    {
        std::unique_lock<std::mutex> lock(p->mu);
        if (!p->cv.wait_for(lock,kShutdownDeadline,[&] { return p->cleanup_exited; })) {
            cleanup.detach(); (void)p_.release(); return;
        }
    }
    cleanup.join();
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
        p.weight_issue(true,bytes);
    } catch (const std::exception& e) { p.weight_failure(); error = e.what(); p.poisoned = true; p.unsafe = true; return false; }
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
NativeHelper::Bytes NativeHelper::planned_bytes(int capacity, bool compute) {
    Bytes b{};
    b.pool = mf::record_stride(kHidden,kIntermediate)*size_t(capacity);
    const size_t io=kBufferElements*sizeof(uint16_t);
    const size_t jobs=size_t(kMaxJobs)*sizeof(DeviceJob);
    size_t workspace=0;
    for(int cols=1;compute && cols<=kMaxColumns;++cols)
        workspace=std::max(workspace,size_t(kMaxTokens*4/cols)*mf::fast_workspace_bytes(kHidden,kIntermediate,cols));
    b.device_scratch=compute ? 2*io+jobs+workspace : 0;
    b.pinned=compute ? 2*io+jobs : 0;
    return b;
}

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
bool NativeHelper::worker_compute_available() const { return p_->worker_compute; }
bool NativeHelper::replay_probe_timing(bool enabled,std::string& error) {
    std::lock_guard<std::mutex> lock(p_->mu);
    if(p_->active || p_->queued || (enabled && !p_->replay_events[0])) {
        error="helper probe timers unavailable or helper active";return false;
    }
    p_->probe_timing=enabled;return true;
}
bool NativeHelper::queue_snapshot(int layer,std::array<QueueIdentity,128>& out,std::string& error) const {
    std::lock_guard<std::mutex> lock(p_->mu);out={};
    if(!p_->sealed||p_->poisoned||p_->refreshing||layer<0||layer>=kLayers) {error="helper snapshot unavailable";return false;}
    for(int e=0;e<kExperts;++e){const auto& r=p_->records[size_t(layer)*kExperts+e];out[e].slot=r.slot;
        if(r.slot>=0)for(int k=0;k<3;++k)out[e].multiplier[k]=1.0f/r.weight[k];}
    return true;
}
uintptr_t NativeHelper::pool_identity() const { return reinterpret_cast<uintptr_t>(p_->pool); }
size_t NativeHelper::record_stride() const { return p_->stride; }
NativeHelper::WeightTraffic NativeHelper::weight_traffic() const {
    std::lock_guard<std::mutex> lock(p_->traffic_mu); return p_->traffic;
}

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
    if (!p.sealed || p.poisoned || p.active || p.refreshing || (p.live && p.live->phase!=LivePhase::idle) || !valid_identity(layer,expert) || bytes!=p.stride || !weight || !input) {
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
    if (!p.sealed || p.poisoned || p.active || p.refreshing || (p.live && p.live->phase!=LivePhase::idle) || !valid_identity(old_layer,old_expert) ||
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
    p.weight_replacement_complete();
    return true;
}

bool NativeHelper::live_begin(int ol,int oe,int nl,int ne,void* host,void* incoming,void* outgoing,size_t bytes,
                              const float weight[3],const float input[3],float old_weight[3],float old_input[3],std::string& error) {
    auto& p=*p_; std::unique_lock<std::mutex> lock(p.mu);
    auto separate=[&](const void* a,const void* b) {
        const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
        return x<y ? y-x>=bytes : x-y>=bytes;
    };
    if (!p.sealed || p.poisoned || p.active || p.refreshing || !valid_identity(ol,oe) || !valid_identity(nl,ne) ||
        !host || !incoming || !outgoing || !weight || !input || !old_weight || !old_input || bytes!=p.stride ||
        !separate(host,incoming) || !separate(host,outgoing) || !separate(incoming,outgoing) ||
        (p.live && (p.live->phase!=LivePhase::idle || p.live->stop))) {
        error="live helper unavailable, active, or invalid buffers/identity"; return false;
    }
    auto& old=p.records[size_t(ol)*kExperts+oe]; auto& replacement=p.records[size_t(nl)*kExperts+ne];
    if (old.slot<0 || replacement.slot>=0) { error="live helper source absent or destination already ready"; return false; }
    for (int i=0;i<3;++i) if (!std::isfinite(weight[i]) || weight[i]<=0 || !std::isfinite(input[i]) || input[i]<=0 ||
                                !std::isfinite(1.0f/weight[i]) || 1.0f/weight[i]==0) {
        error="invalid live helper scales"; return false;
    }
    if (!p.live) {
        try {
            p.live=std::make_unique<Impl::Live>();
            p.live->cpu=std::thread([&p]{p.live_cpu_run();});
            p.live->gpu=std::thread([&p]{p.live_gpu_run();});
        } catch (const std::exception& e) {
            if (p.live) { p.live->stop=true; p.live->cv.notify_all(); }
            error=std::string("start live helper workers: ")+e.what(); return false;
        }
    }
    auto& l=*p.live;
    l.old_id=ol*kExperts+oe; l.new_id=nl*kExperts+ne; l.slot=old.slot;
    l.capture_context=capture::detailed()?capture::context():std::string{};
    l.host=host; l.incoming=incoming; l.outgoing=outgoing;
    std::copy_n(weight,3,l.weight); std::copy_n(input,3,l.input);
    std::copy_n(old.weight,3,old_weight); std::copy_n(old.input,3,old_input);
    l.phase=LivePhase::preparing; l.error.clear(); l.cpu_done=l.gpu_done=false;
    l.gpu_failure_recorded=false;
    l.cpu_busy=l.gpu_busy=true; l.cpu_queued=l.gpu_queued=true;
    l.recycle=l.upload=false; l.started=std::chrono::steady_clock::now();
    l.cv.notify_all(); return true;
}

NativeHelper::LiveStatus NativeHelper::live_poll() {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.live) return {};
    auto& l=*p.live; p.live_advance_locked();
    // The lease persists between stages even while both workers are quiescent.
    return {l.phase,l.phase!=LivePhase::idle,l.error};
}
bool NativeHelper::live_withdraw(std::string& error) {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.live) { error="no live helper operation"; return false; }
    auto& l=*p.live; p.live_advance_locked();
    if (p.poisoned || p.active || p.refreshing || l.phase!=LivePhase::prepared ||
        p.records[l.old_id].slot!=l.slot || p.records[l.new_id].slot>=0) {
        error="live helper not prepared, active, or ownership changed"; return false;
    }
    p.records[l.old_id].slot=-1;
    l.gpu_failure_recorded=false;
    l.phase=LivePhase::uploading; l.upload=true; l.gpu_done=false; l.gpu_busy=true; l.gpu_queued=true;
    l.started=std::chrono::steady_clock::now(); l.cv.notify_all(); return true;
}
bool NativeHelper::live_publish(std::string& error) {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.live) { error="no live helper operation"; return false; }
    auto& l=*p.live; p.live_advance_locked();
    if (p.poisoned || p.active || p.refreshing || l.phase!=LivePhase::uploaded || p.records[l.new_id].slot>=0) {
        error="live helper upload incomplete or active"; return false;
    }
    auto& r=p.records[l.new_id]; r.slot=l.slot;
    std::copy_n(l.weight,3,r.weight); std::copy_n(l.input,3,r.input);
    l.phase=LivePhase::published; return true;
}
bool NativeHelper::live_recycle(std::string& error) {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.live || p.poisoned || p.live->phase!=LivePhase::published) { error="live helper not published"; return false; }
    auto& l=*p.live; l.phase=LivePhase::recycling; l.recycle=true; l.cpu_done=false; l.cpu_busy=true; l.cpu_queued=true;
    l.started=std::chrono::steady_clock::now(); l.cv.notify_all(); return true;
}
bool NativeHelper::live_retire(bool& safe,std::string& error) {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu); safe=false;
    if (!p.live || p.live->phase==LivePhase::idle) { safe=true; return true; }
    auto& l=*p.live; p.live_advance_locked();
    if (l.cpu_busy || l.gpu_busy || l.gpu_uncertain || (l.phase!=LivePhase::complete && l.phase!=LivePhase::failed)) {
        error=l.error.empty()?"live helper operation has not retired":l.error; return false;
    }
    if(l.phase==LivePhase::complete)p.weight_replacement_complete();
    l.phase=LivePhase::idle; l.host=l.incoming=l.outgoing=nullptr; safe=true; return true;
}
bool NativeHelper::live_cancel(bool& safe,std::string& error) {
    auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu); safe=false;
    if (!p.live || p.live->phase==LivePhase::idle) { safe=true; return true; }
    auto& l=*p.live;
    // Existing queued copies still run to a proven completion. No later stage
    // can be submitted once poisoned, and this method never waits for them.
    if (l.phase!=LivePhase::failed) p.live_fail_locked("live helper cancelled",false);
    if (l.cpu_busy || l.gpu_busy || l.gpu_uncertain) {
        error=l.error; return false;
    }
    l.phase=LivePhase::idle; l.host=l.incoming=l.outgoing=nullptr; safe=true; return true;
}

uint64_t NativeHelper::submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error) {
    const uint64_t submit_enter_ns = host_now_ns();
    auto& p = *p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.worker_compute || !p.sealed || p.poisoned || p.active || p.refreshing || !x || tokens < 1 || tokens > kMaxTokens || jobs.empty() || jobs.size() > kMaxJobs) {
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
    p.finished_flag.store(false, std::memory_order_relaxed);
    p.collector_claimed = false;
    p.pass_timing = {};
    p.pending_capture_context = capture::detailed() ? capture::context() : std::string{};
    p.pass_timing.submit_enter_ns = submit_enter_ns;
    p.pass_timing.enqueue_ns = host_now_ns();
    p.queued = true;
    p.queued_flag.store(true, std::memory_order_release);
    p.cv.notify_one();
    p.pass_timing.submit_published_ns = host_now_ns();
    return p.active;
}

bool NativeHelper::collect(uint64_t seq, float* out, size_t n, std::string& error, PassTiming* timing) {
    const uint64_t collect_enter_ns = host_now_ns();
    auto& p = *p_;
    // helper_fast: spin before taking the lock, which the worker needs to publish; the wait below decides.
    uint64_t spin_begin_ns = 0;
    if (p.fast_mode.load(std::memory_order_relaxed)) { spin_begin_ns = host_now_ns(); Impl::spin_until(p.finished_flag, std::chrono::microseconds(2000)); }
    std::unique_lock<std::mutex> lock(p.mu);
    if (!seq || seq != p.active || !out || n < size_t(p.pending_tokens) * kHidden) { error = "invalid helper sequence or output"; return false; }
    if (p.collector_claimed) { error = "helper sequence already has a collector"; return false; }
    p.collector_claimed = true;
    const uint64_t collect_wait_begin_ns = spin_begin_ns ? spin_begin_ns : host_now_ns();
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
            const unsigned flags = helper_merge::merge_column(dst, src, j.weight[c], size_t(kHidden));
            if (flags & helper_merge::kNonfiniteInput) {
                p.failure = "nonfinite helper output"; p.poisoned = true; error = p.failure; return false;
            }
            if (flags & helper_merge::kNonfiniteSum) {
                p.failure = "nonfinite helper weighted sum"; p.poisoned = true; error = p.failure; return false;
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
    p.active = 0; p.finished = false; p.finished_flag.store(false, std::memory_order_relaxed); p.collector_claimed = false; return true;
}
}
