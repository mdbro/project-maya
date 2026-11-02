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
    bool collector_claimed = false; // exactly one retirement owner per sequence
    std::string failure;
    std::mutex mu;
    std::condition_variable cv;
    std::thread worker;

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
        if (stream && cudaStreamSynchronize(stream) != cudaSuccess) {
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
        try { DeviceGuard guard(device); release(); } catch (...) { /* No safe device context: retain buffers. */ }
    }
    void run() {
        try {
            DeviceGuard guard(device);
            for (;;) {
                int n;
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return quit || queued; });
                    if (quit && !queued) break;
                    n = pending_n; queued = false;
                }
                std::string error;
                bool uncertain = false;
                try {
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
                    ck(cudaEventRecord(done_event, stream), "record helper event");
                    ck(cudaEventSynchronize(done_event), "wait helper event");
                } catch (const std::exception& e) {
                    error = e.what();
                    // An error or failed event does not prove the stream has
                    // retired; quarantine on any failed stream synchronization.
                    uncertain = cudaStreamSynchronize(stream) != cudaSuccess;
                } catch (...) {
                    error = "unknown helper failure";
                    uncertain = cudaStreamSynchronize(stream) != cudaSuccess;
                }
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (!error.empty()) { failure = error; poisoned = true; unsafe |= uncertain; }
                    finished = true;
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
NativeHelper::~NativeHelper() = default;

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
    if (!p_->sealed || p_->poisoned || layer < 0 || layer >= kLayers) return result;
    for (int e = 0; e < kExperts; ++e) if (p_->records[size_t(layer) * kExperts + e].slot >= 0)
        result[e >> 5] |= uint32_t(1) << (e & 31);
    return result;
}
NativeHelper::Bytes NativeHelper::bytes() const { return p_->allocated; }

uint64_t NativeHelper::submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error) {
    auto& p = *p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || !x || tokens < 1 || tokens > kMaxTokens || jobs.empty() || jobs.size() > kMaxJobs) {
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
    p.active = p.next; p.pending_n = int(jobs.size()); p.pending_tokens = tokens; p.finished = false; p.queued = true;
    p.collector_claimed = false;
    p.cv.notify_one(); return p.active;
}

bool NativeHelper::collect(uint64_t seq, float* out, size_t n, std::string& error) {
    auto& p = *p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!seq || seq != p.active || !out || n < size_t(p.pending_tokens) * kHidden) { error = "invalid helper sequence or output"; return false; }
    if (p.collector_claimed) { error = "helper sequence already has a collector"; return false; }
    p.collector_claimed = true;
    p.cv.wait(lock, [&] { return p.finished || p.poisoned; });
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
    p.active = 0; p.finished = false; p.collector_claimed = false; return true;
}
}
