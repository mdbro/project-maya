// Ordered helper queue adapted from Maya DS4 helper_window/helper_loop and
// helper_enqueue. Persistent pool and joined reader/slot lifetimes follow the
// Infernix ExpertHelper/PeerExpertExecutor ownership boundary. MiniMax's RG16
// FFN implementation is supplied by Infernix's minimax_nvfp4_ffn; this class
// deliberately owns no routing or refill policy.
#include "mm3_native_helper.hpp"
#include "mm3_capture.hpp"
#include "mm3_kernels.cuh"
#include "mm3_helper_merge.hpp"
#include "mm3_helper_kernels.cuh"
#include "mm3_helper_q.cuh"
#include "mm3_helper_sm89.hpp"
#include "mm3_helper_finite.hpp"
#include "mm3_upload_guard.hpp"
#include "mm3_background_traffic.hpp"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

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
// a pass's inputs and outputs: one row of kHidden a column, four columns a token
size_t buffer_elements(int pass_tokens) { return size_t(4 * pass_tokens) * NativeHelper::kHidden; }
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
bool refresh_scales(const void*, const float[3], const float[3], size_t, size_t, std::string&);
bool valid_identity(int, int);
}

struct NativeHelper::Impl {
    struct Record { int slot = -1; float weight[3]{}, input[3]{}; };
    int device, capacity, count = 0;
    int direct_spare_slot=-1;
    uint8_t* probe_destination = nullptr;  // 4080 diagnostic copy target, outside its expert roster
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
    int pass_tokens = kMaxTokens;            // a pass's capacity (configured_pass_tokens at construction)
    std::vector<Job> pending_jobs;           // [4 * pass_tokens]
    int pending_n = 0, pending_tokens = 0, pending_columns = 0;
    std::vector<size_t> pending_offsets;     // [4 * pass_tokens]
    // collect's merge: the pass's columns in merge order, and the columns it refused (reserved once: no allocation a pass)
    std::vector<helper_merge::PassColumn> merge_cols;
    std::vector<helper_merge::Flagged> merge_refused;
    // pass column index (merge order: jobs, then their columns) -> job i, column c
    std::pair<int, int> column_at(size_t index) const {
        for (int i = 0; i < pending_n; ++i) {
            const size_t w = size_t(pending_jobs[size_t(i)].columns);
            if (index < w) return {i, int(index)};
            index -= w;
        }
        return {-1, -1};
    }
    // (error path only: collect, under mu, the pass retired) why the merge refused column c of job i: the card, the job,
    // the bad elements of h_y, the input row as h_x holds it now, the multipliers, and the same column read back from the
    // card's own buffers (d_y the kernel's output, d_x its input) through the copy engine. The kernels give finite results
    // for finite inputs and any record bytes (nonfinite-helper.md, 2026-11-14), so where the first non-finite value shows
    // says where it entered: only in h_y (d_y finite, d_y != h_y) - on the way to host memory; in d_y too, with d_x = h_x
    // and |x| ordinary - the card's computation, its record or a multiplier; d_x != h_x or |x| out of range - the input
    // after submit's check. bad: h_y's non-finite elements. Diagnostic only: the caller decides the reaction (a refused
    // column recomputed on another lane, or the helper poisoned).
    std::string nonfinite_detail(int i, int c, int* bad = nullptr) {
        const Job& j = pending_jobs[size_t(i)];
        const size_t at = pending_offsets[size_t(i)] + size_t(c) * kHidden;
        const auto& r = records[size_t(j.layer) * kExperts + j.expert];
        struct Scan { int count = 0, first = -1; unsigned bits = 0; };
        const auto scan = [](const uint16_t* v) {
            Scan s;
            for (int k = 0; k < kHidden; ++k)
                if ((v[k] & 0x7f80u) == 0x7f80u) { if (s.first < 0) { s.first = k; s.bits = v[k]; } ++s.count; }
            return s;
        };
        const Scan hy = scan(h_y + at), hx = scan(h_x + at);
        if (bad) *bad = hy.count;
        float x_max = 0.f;
        for (int k = 0; k < kHidden; ++k) {
            const uint32_t bits = uint32_t(h_x[at + size_t(k)]) << 16;
            float v; std::memcpy(&v, &bits, sizeof v);
            if (std::isfinite(v)) x_max = std::max(x_max, std::fabs(v));
        }
        std::string device_side;
        try {
            DeviceGuard guard(device);
            const size_t bytes = size_t(kHidden) * sizeof(uint16_t);
            std::vector<uint16_t> back(static_cast<size_t>(kHidden));
            ck(cudaMemcpy(back.data(), d_y + at, bytes, cudaMemcpyDeviceToHost), "read back helper output");
            const Scan dy = scan(back.data());
            const bool y_same = std::memcmp(back.data(), h_y + at, bytes) == 0;
            char at_first[48] = "";   // (d_y at h_y's first bad element: one flipped bit apart, or not)
            if (hy.first >= 0) std::snprintf(at_first, sizeof at_first, ", d_y[%d] = 0x%04x", hy.first, unsigned(back[size_t(hy.first)]));
            ck(cudaMemcpy(back.data(), d_x + at, bytes, cudaMemcpyDeviceToHost), "read back helper input");
            const bool x_same = std::memcmp(back.data(), h_x + at, bytes) == 0;
            device_side = "d_y nonfinite " + std::to_string(dy.count) + at_first + (y_same ? ", d_y = h_y" : ", d_y != h_y") +
                          (x_same ? ", d_x = h_x" : ", d_x != h_x");
        } catch (const std::exception& e) {
            (void)cudaGetLastError();
            device_side = std::string("read-back failed: ") + e.what();
        }
        char text[640];
        std::snprintf(text, sizeof text, " (device %d layer %d expert %d slot %d column %d of %d token %d weight %g;"
                      " h_y nonfinite %d of %d, first [%d] = 0x%04x; h_x nonfinite %d, largest finite |x| %g;"
                      " multipliers %g %g %g; %s)", device, j.layer, j.expert, r.slot, c, j.columns, j.token[c],
                      double(j.weight[c]), hy.count, int(kHidden), hy.first, hy.bits, hx.count, double(x_max),
                      double(1.0f / r.weight[0]), double(1.0f / r.weight[1]), double(1.0f / r.weight[2]), device_side.c_str());
        return text;
    }
    // MM3_HELPER_INJECT_NONFINITE=LAYER:EXPERT (a check of the recompute path on a GPU; nothing happens unless it is set):
    // the first column of the first job naming that expert gets a NaN in h_y before the merge, once per process, and
    // only in a collect that can recover (with a refused list) - calibration, replay and the third pool's fill check,
    // which collect without one, would be poisoned by it.
    void inject_nonfinite_once() { inject_nonfinite_into(pending_jobs.data(), pending_n, h_y, pending_offsets.data()); }
    // (also the hand-off's merge: its jobs and columns; offsets[i] is job i's first element in y)
    static void inject_nonfinite_into(const Job* jobs, int n, uint16_t* y, const size_t* offsets) {
        static const std::pair<int, int> target = [] {
            const char* v = std::getenv("MM3_HELPER_INJECT_NONFINITE");
            int layer = -1, expert = -1;
            if (!v || std::sscanf(v, "%d:%d", &layer, &expert) != 2) layer = expert = -1;
            return std::make_pair(layer, expert);
        }();
        static std::atomic<bool> injected{false};
        if (target.first < 0 || injected.load(std::memory_order_relaxed)) return;
        for (int i = 0; i < n; ++i) {
            const Job& j = jobs[size_t(i)];
            if (j.layer == target.first && j.expert == target.second && !injected.exchange(true)) {
                y[offsets[size_t(i)] + 7] = 0x7fc0;
                return;
            }
        }
    }
    // (verify_expert, rewrite_expert) the record of a ready expert, and not the slot a live replacement is copying
    bool repair_target(const Record& r, std::string& error) const {
        if (r.slot < 0) { error = "helper record check: the expert is not ready on this card"; return false; }
        if (live && live->phase != LivePhase::idle && (live->slot == r.slot ||
            (live->direct && live->direct_old_slot == r.slot))) {
            error = "helper record check: the expert is the live replacement's"; return false;
        }
        return true;
    }
    uint64_t next = 0, active = 0;
    std::atomic<uint64_t> records_version{0};   // (helper_handoff: bumped on every record or scale change)
    bool handoff_active = false;                // (helper_handoff: a released pass in flight; under mu)
    bool worker_compute = true;
    bool sealed = false, queued = false, finished = false, quit = false, poisoned = false, unsafe = false;
    bool worker_exited = false, cleanup_exited = false;
    bool collector_claimed = false; // exactly one retirement owner per sequence
    PassTiming pass_timing{}; // protected by mu except worker-local endpoints
    std::string failure;
    bool pending_stage_events=false, graph_stage_events=false;
    std::array<cudaEvent_t,5> stage_events{};cudaEvent_t stage_clock_ref=nullptr;
    bool mark_stage(int i) {return !graph_stage_events || cudaEventRecordWithFlags(stage_events[i],stream,cudaEventRecordExternal)==cudaSuccess;}
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
    // helper_kernel: this card's own kernel in place of the shared dp4a kernel (see set_kernel). With the tensor-core
    // kernel or the any-width dp4a kernel a pass with a job of more than one column takes one launch for all its jobs
    // instead of one a width, and with 100 added to the value that pass is one CUDA graph (inputs up, kernels,
    // results down), built on first use for its job count and column total. A pass of one-column jobs keeps its
    // graph, of which each kernel has its own set (index 0 the shared kernel, 1 to 4 the tensor-core warp counts 1,
    // 2, 4, 8, 5 today's arithmetic compiled per width, 6 the same in the any-width launch).
    std::atomic<int> kernel_multi{0}, kernel_one{0};
    // helper_kernel_loads (set_loads): the tensor-core kernel's loads a batch, P units a warp (project2); 1 variant 1
    // (ffn1). Read once a pass by the worker and passed to every launch and graph of that pass.
    std::atomic<int> kernel_loads{1};
    std::atomic<int> execution_mode{0};
    std::atomic<uint64_t> kernel_count{0};
    static int kernel_base(int v) { return v >= 100 ? v - 100 : v; }
    // Kernel IDs reserve four consecutive warp choices per arithmetic family.
    static int warp_index(int v) { return v == 1 ? 0 : v == 2 ? 1 : v == 4 ? 2 : v == 8 ? 3 : -1; }
    static int kernel_index(int v) {
        v = kernel_base(v);
        if (const int i = warp_index(v); i >= 0) return 1 + i;
        if (v == 16) return 5;
        if (v == 32) return 6;
        for (int family = 0; family < 5; ++family) {
            const int base = family == 0 ? 64 : family == 1 ? 80 : family == 2 ? 16 : family == 3 ? 32 : 48;
            if (const int i = warp_index(v - base); i >= 0) return 7 + family * 4 + i;
        }
        return 0;
    }
    static bool kernel_tensor(int v) { const int i = kernel_index(v); return i >= 1 && i <= 4; }
    static bool kernel_any(int v) { return kernel_index(v) == 6; }
    static bool kernel_q(int v) { return kernel_index(v) >= 7; }
    static bool kernel_q2(int v) { const int i = kernel_index(v); return i >= 11 && i <= 14; }
    static bool kernel_fp8(int v) { const int i = kernel_index(v); return i >= 15 && i <= 18; }
    static bool kernel_bits(int v) { const int i = kernel_index(v); return i >= 19 && i <= 22; }
    static bool kernel_q3(int v) { const int i = kernel_index(v); return i >= 23 && i <= 26; }
    static int q_warps(int v) { return 1 << ((kernel_index(v) - 7) % 4); }
    static bool kernel_wide_graph(int v) { return v >= 100 && (kernel_tensor(v) || kernel_any(v) || kernel_q(v)); }
    static bool kernel_valid(int v) { return kernel_index(v) != 0 && (v < 100 || kernel_wide_graph(v)); }
    static bool execution_valid(int mode, int kernel) {
        if (mode == 0) return true;
        if (mode < 0 || mode > 7) return false;
        const int i = kernel_index(kernel);
        return kernel_q2(kernel) || (i >= 7 && i <= 10 && !(mode & 4));
    }
    static int effective_execution(int mode, int kernel, int jobs, int columns) {
        const int i = kernel_index(kernel);
        return i >= 7 && i <= 10 && columns == jobs ? mode & ~1 : mode;
    }
    static bool loads_valid(int p) { return p == 1 || p == 2 || p == 4 || p == 8; }
    static int loads_index(int p) { return p == 2 ? 1 : p == 4 ? 2 : p == 8 ? 3 : 0; }
    static bool input_valid(int kernel, int format) {
        if (format == 0) return true;
        if (format == 1) return kernel_q(kernel);
        if (format == 2) return kernel_q3(kernel);
        return format == 3 && kernel_q(kernel) && !kernel_q2(kernel) && !kernel_q3(kernel);
    }
    // Graphs are distinct for arithmetic, loads, architecture and all four input
    // formats. Addresses remain fixed for the lifetime of each captured graph.
    static constexpr size_t kKernels = 27, kGraphSlots = kKernels * 4 * 2 * 4;
    static size_t graph_slot(int kernel, int loads, bool sm89 = false, int format = 0) {
        return size_t(kernel_index(kernel)) + ((kernel_tensor(kernel) || kernel_q(kernel)) ? kKernels * size_t(loads_index(loads)) : 0) +
               kKernels * 4 * size_t(sm89) + kKernels * 8 * size_t(format);
    }
    std::atomic<bool> sm89_mode{false};
    bool sm89_device = false;
    bool use_sm89(bool asked) const { return asked && sm89_device && mm3::helper_sm89::built(); }
    std::uint8_t *d_q = nullptr, *d_hq = nullptr, *d_qp = nullptr;
    std::uint16_t* d_expanded = nullptr;
    std::atomic<size_t> q_scratch_bytes{0};
    int pending_format = 0, pending_kernel = 0, pending_loads = 1, pending_execution = 0;
    bool pending_sm89 = false;
    bool ensure_q(int kernel, int format) {
        const size_t cols = size_t(4 * pass_tokens);
        const auto alloc = [&](auto*& pointer, size_t bytes) {
            if (pointer) return true;
            if (cudaMalloc(reinterpret_cast<void**>(&pointer), bytes) != cudaSuccess) {
                (void)cudaGetLastError(); pointer = nullptr; return false;
            }
            q_scratch_bytes.fetch_add(bytes, std::memory_order_relaxed);
            return true;
        };
        if (kernel_q2(kernel)) return format == 1 || alloc(d_qp, cols * mm3::helper_q::kPackedRow);
        if (kernel_q3(kernel)) return format == 2 ||
            (alloc(d_expanded, cols * kHidden * sizeof(uint16_t)) &&
             (format == 1 || alloc(d_qp, cols * mm3::helper_q::kPackedRow)));
        return (format == 3 || alloc(d_q, cols * mm3::helper_q::kInStride)) &&
               alloc(d_hq, cols * mm3::helper_q::kJobHidden);
    }
    static size_t input_bytes(int columns, int format) {
        return size_t(columns) * (format == 1 ? mm3::helper_q::kPackedRow :
                                 format == 3 ? mm3::helper_q::kInStride : size_t(kHidden) * sizeof(uint16_t));
    }
    cudaError_t q_launch(int kernel, int loads, const DeviceJob* jobs, int n, int columns, int format, bool sm89, int execution) {
        namespace hq = mm3::helper_q;
        namespace a89 = mm3::helper_sm89;
        if (!input_valid(kernel, format) || !execution_valid(execution, kernel) || (kernel_fp8(kernel) && !sm89)) return cudaErrorInvalidValue;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(d_x);
        if (kernel_q2(kernel) || kernel_q3(kernel)) {
            if (mm3::helper_tc::workspace_bytes(n) > workspace_bytes) return cudaErrorMemoryAllocation;
            const std::uint8_t* packed = bytes;
            if (format == 0) {
                if (!d_qp) return cudaErrorMemoryAllocation;
                const cudaError_t e = sm89 ? a89::q2_pack_bf16(d_x, d_qp, columns, stream) : hq::pack_bf16(d_x, d_qp, columns, stream);
                if (e != cudaSuccess) return e;
                packed = d_qp;
            }
            if (kernel_q2(kernel)) {
                if (execution) return sm89 ? a89::q2_ffn_exec(execution, q_warps(kernel), loads, jobs, n, d_x, packed, workspace, stream)
                                           : hq::ffn2_exec(execution, q_warps(kernel), loads, jobs, n, d_x, packed, workspace, stream);
                return sm89 ? a89::q2_ffn(q_warps(kernel), loads, jobs, n, d_x, packed, workspace, stream)
                            : hq::ffn2(q_warps(kernel), loads, jobs, n, d_x, packed, workspace, stream);
            }
            const std::uint16_t* expanded = d_x;
            if (format != 2) {
                if (!d_expanded) return cudaErrorMemoryAllocation;
                const cudaError_t e = sm89 ? a89::q_expand_bf16(packed, d_expanded, columns, stream)
                                          : hq::expand_bf16(packed, d_expanded, columns, stream);
                if (e != cudaSuccess) return e;
                expanded = d_expanded;
            }
            return sm89 ? a89::q3_ffn(q_warps(kernel), loads, jobs, n, d_x, expanded, workspace, stream)
                        : hq::ffn3(q_warps(kernel), loads, jobs, n, d_x, expanded, workspace, stream);
        }
        if (!d_hq || (format != 3 && !d_q)) return cudaErrorMemoryAllocation;
        const std::uint8_t* in = bytes;
        if (format != 3) {
            const bool fp8 = kernel_fp8(kernel);
            const cudaError_t e = format == 1
                ? (sm89 ? a89::q_expand_packed_codes(bytes, d_q, columns, fp8, stream) : hq::expand_packed_codes(bytes, d_q, columns, fp8, stream))
                : (sm89 ? a89::q_quant_bf16_codes(d_x, d_q, columns, fp8, stream) : hq::quant_bf16_codes(d_x, d_q, columns, fp8, stream));
            if (e != cudaSuccess) return e;
            in = d_q;
        }
        // Q1 bit1 is useful only for jobs with multiple columns. An entirely
        // one-column pass must launch the original specialization, so it also
        // avoids the parallel encoder's register requirements. Preserve bit2.
        // Mixed passes keep bit1 and apply the per-job width guard in up_block.
        const int dispatched_execution = effective_execution(execution, kernel, n, columns);
        if (dispatched_execution) return sm89 ? a89::q_ffn_exec(dispatched_execution, q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream)
                                              : hq::ffn_exec(dispatched_execution, q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream);
        if (kernel_fp8(kernel)) return a89::q_fp8_ffn(q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream);
        if (kernel_bits(kernel))
            return sm89 ? a89::q_bits_ffn(q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream)
                        : hq::ffn_int_bits(q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream);
        return sm89 ? a89::q_ffn(q_warps(kernel), loads, jobs, n, d_x, in, d_hq, stream)
                    : hq::ffn(q_warps(kernel), loads, jobs, n, d_x, in, d_hq, nullptr, stream);
    }
    template<int S> cudaError_t tensor_launch_s(int loads, const DeviceJob* jobs, int n) {
        namespace tc = mm3::helper_tc;
        switch (loads) {
            case 2: return tc::ffn2<S, 2>(jobs, n, workspace, stream);
            case 4: return tc::ffn2<S, 4>(jobs, n, workspace, stream);
            case 8: return tc::ffn2<S, 8>(jobs, n, workspace, stream);
            default: return tc::ffn1<S>(jobs, n, workspace, stream);   // (1: variant 1, the A/B's baseline)
        }
    }
    cudaError_t tensor_launch(int warps, int loads, const DeviceJob* jobs, int n, bool sm89 = false) {
        namespace tc = mm3::helper_tc;
        if (tc::workspace_bytes(n) > workspace_bytes) return cudaErrorMemoryAllocation;
        if (sm89) return mm3::helper_sm89::tensor(kernel_base(warps), loads, jobs, n, workspace, stream);
        switch (kernel_base(warps)) {
            // (variant 1 of the kernel, or with loads 2, 4 or 8 variant 2: the same output with the loads batched)
            case 1: return tensor_launch_s<1>(loads, jobs, n);
            case 2: return tensor_launch_s<2>(loads, jobs, n);
            case 4: return tensor_launch_s<4>(loads, jobs, n);
            default: return tensor_launch_s<8>(loads, jobs, n);
        }
    }
    struct GraphBank {
        std::array<std::array<std::array<cudaGraphExec_t, kMaxJobs + 1>, 2>, kGraphSlots> exec{};
        std::array<std::array<std::array<bool, kMaxJobs + 1>, 2>, kGraphSlots> failed{};
    };
    std::array<std::unique_ptr<GraphBank>, 16> graph_banks{};
    GraphBank& graph_bank(int execution) {
        auto& b = graph_banks[size_t(execution)+(graph_stage_events?8:0)];
        if (!b) b = std::make_unique<GraphBank>();
        return *b;
    }
    // the graphs of passes with multi-column jobs, by kernel slot (kernel and loads, graph_slot), zero-copy, job count
    // and column total (worker thread only; a null entry: the build failed and is not tried again)
    std::unordered_map<uint64_t, cudaGraphExec_t> wide_graphs;
    cudaGraphExec_t wide_graph(int n, int columns, bool zc, int kernel, int loads, bool sm89 = false, int format = 0, int execution = 0) {
        if (!kernel_wide_graph(kernel) || n < 1 || n > kMaxJobs || columns <= n || columns > kMaxJobs) return nullptr;
        if (!input_valid(kernel, format)) return nullptr;
        if (kernel_q(kernel) && !ensure_q(kernel, format)) return nullptr;
        static_assert(kGraphSlots <= 1023 && kMaxJobs < 128, "the wide graph key's fields");
        const uint64_t key = uint64_t(graph_stage_events) << 40 | uint64_t(execution) << 32 | uint64_t(graph_slot(kernel, loads, sm89, format)) << 22 |
                             uint64_t(zc) << 21 | uint64_t(n) << 12 | uint64_t(columns);
        if (const auto it = wide_graphs.find(key); it != wide_graphs.end()) return it->second;
        cudaGraphExec_t exec = nullptr;
        if (wide_graphs.size() < 4096) {
            const size_t io = size_t(columns) * kHidden * sizeof(uint16_t), jobs = size_t(n) * sizeof(DeviceJob);
            const size_t in = input_bytes(columns, format);
            cudaGraph_t graph = nullptr;
            bool ok = (!zc || views_ready()) && cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            if (ok) {
                ok=mark_stage(0);
                if (ok && zc) k::copy_words(v_x, d_x, in, stream, v_jobs, d_jobs, jobs);   // (a failed launch fails the capture)
                else ok = ok && cudaMemcpyAsync(d_x, h_x, in, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                          cudaMemcpyAsync(d_jobs, h_jobs, jobs, cudaMemcpyHostToDevice, stream) == cudaSuccess;
                ok=ok && mark_stage(1);
                if (ok) ok = (kernel_any(kernel) ? any_launch(d_jobs, n) :
                              kernel_q(kernel) ? q_launch(kernel, loads, d_jobs, n, columns, format, use_sm89(sm89), execution) :
                              tensor_launch(kernel, loads, d_jobs, n, use_sm89(sm89))) == cudaSuccess;
                ok=ok && mark_stage(2);
                if (ok && zc) k::copy_words(d_y, v_y, io, stream);
                else ok = ok && cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
                ok=ok && mark_stage(3);
                const bool ended = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
                ok = ok && ended && graph && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
            }
            if (graph) cudaGraphDestroy(graph);
            (void)cudaGetLastError();
            if (!ok) exec = nullptr;
        }
        if (!exec) std::fprintf(stderr, "MM3 warning: helper device=%d kernel=%d input_format=%d wide graph unavailable; using separate launches (reported in PassTiming)\n", device, kernel, format);
        wide_graphs.emplace(key, exec);
        return exec;
    }
    cudaError_t any_launch(const DeviceJob* jobs, int n) {
        if (size_t(n) * mm3::helper_dp::job_workspace(8) > workspace_bytes) return cudaErrorMemoryAllocation;
        return mm3::helper_dp::ffn_any(jobs, n, workspace, stream);
    }
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
    bool graph_ready(int n, bool zc, int warps, int loads, bool sm89 = false, int format = 0, int execution = 0) {
        const size_t kernel = size_t(kernel_index(warps)), slot = graph_slot(warps, loads, sm89, format);
        auto& bank = graph_bank(execution);
        if (n < 1 || n > kMaxJobs || bank.failed[slot][zc][size_t(n)]) return false;
        cudaGraphExec_t& exec = bank.exec[slot][zc][size_t(n)];
        if (exec) return true;
        const size_t io = size_t(n) * kHidden * sizeof(uint16_t), jobs = size_t(n) * sizeof(DeviceJob);
        const size_t in = input_bytes(n, format);
        if (size_t(n) * mf::fast_workspace_bytes(kHidden, kIntermediate, 1) > workspace_bytes || (zc && !views_ready()) ||
            (!input_valid(warps, format)) || (kernel_q(warps) && !ensure_q(warps, format))) {
            bank.failed[slot][zc][size_t(n)] = true; return false;
        }
        cudaGraph_t graph = nullptr;
        bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        if (ok) {
            ok=mark_stage(0);
            if (ok && zc) k::copy_words(v_x, d_x, in, stream, v_jobs, d_jobs, jobs);   // (a failed launch fails the capture)
            else ok = ok && cudaMemcpyAsync(d_x, h_x, in, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                      cudaMemcpyAsync(d_jobs, h_jobs, jobs, cudaMemcpyHostToDevice, stream) == cudaSuccess;
            ok=ok && mark_stage(1);
            if (ok && kernel_q(warps)) ok = q_launch(warps, loads, d_jobs, n, n, format, use_sm89(sm89), execution) == cudaSuccess;
            else if (ok && kernel_tensor(warps)) ok = tensor_launch(warps, loads, d_jobs, n, use_sm89(sm89)) == cudaSuccess;
            else if (ok && kernel) ok = mm3::helper_dp::ffn(d_jobs, n, 1, workspace, stream) == cudaSuccess;
            else if (ok) mf::fast_ffn_cuda_packed(d_jobs, n, kHidden, kIntermediate, 1, workspace, stream);
            ok=ok && mark_stage(2);
            if (ok && zc) k::copy_words(d_y, v_y, io, stream);
            else ok = ok && cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream) == cudaSuccess;
            ok=ok && mark_stage(3);
            const bool ended = cudaStreamEndCapture(stream, &graph) == cudaSuccess;
            ok = ok && ended && graph && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
        }
        if (graph) cudaGraphDestroy(graph);
        (void)cudaGetLastError();
        if (!ok) {
            exec = nullptr; bank.failed[slot][zc][size_t(n)] = true;
            std::fprintf(stderr, "MM3 warning: helper device=%d kernel=%d input_format=%d graph capture failed; using separate launches (reported in PassTiming)\n", device, warps, format);
        }
        return ok;
    }
    void graph_destroy() {
        for (auto& bank : graph_banks) if (bank) {
            for (auto& kernel : bank->exec) for (auto& set : kernel) for (auto& g : set) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
            bank.reset();
        }
        for (auto& entry : wide_graphs) if (entry.second) cudaGraphExecDestroy(entry.second);
        wide_graphs.clear();
    }

    // exchange_schedule: the live copies in pieces inside windows the model opens. The workers read the windows and the
    // pass flag lock-free; the host's critical path only stores numbers. kX4Piece is about 0.09 ms on the 3090's x4
    // link (5.7 GB/s); kHostPiece is split over kHostThreads threads.
    static constexpr size_t kX4Piece = size_t(512) << 10, kHostPiece = size_t(4) << 20;
    static constexpr int kHostThreads = 4;
    std::atomic<int> sched_mode{0};
    std::atomic<uint64_t> x4_until{0}, host_open{0}, host_close{0};
    std::atomic<uint64_t> live_progress_ns{0};   // a scheduled worker's last sign of life (waiting for a window is not a stall)
    std::atomic<bool> pass_busy{false}, x4_piece_busy{false};
    UploadGuard upload_guard;
    UploadGuard::PollCache upload_poll_cache; // live GPU-copy worker only
    std::atomic<bool> x4_piece_submitted{false}; // CUDA accepted the admitted piece; completion not yet observed
    std::atomic<bool> cpu_busy{false};            // (exchange_quiet: the CPU lane is running a job)
    std::atomic<uint64_t> sx4_pieces{0}, shost_pieces{0}, sx4_ns{0}, shost_ns{0}, soverlaps{0}, sverified{0}, smismatches{0};
    std::atomic<uint64_t> sguard_revocations{0}, sguard_publications{0}, sguard_admissions{0};
    std::atomic<uint64_t> sguard_failed_claims{0}, sguard_stale_claims{0}, sguard_event_errors{0};
    std::atomic<uint64_t> sguard_closed_polls{0}, sguard_pending_polls{0}, sguard_deferred_polls{0};
    std::atomic<uint64_t> sguard_event_queries{0}, sguard_query_free_admissions{0};
    std::atomic<uint64_t> sdirect_h2d_bytes{0},sdirect_d2h_bytes{0},sdirect_chunks{0},sdirect_span_ns{0};
    std::atomic<uint64_t> sdirect_verified{0},sdirect_mismatches{0};
    std::atomic<uint64_t> sguard_already_admitted{0}, sguard_already_inflight{0};
    // The high bit tags a unique chunk start until its first foreground
    // boundary. Tagging the start prevents a late CAS from claiming a later
    // chunk. Zero means no measured direct-copy chunk is live.
    static constexpr uint64_t kOpenChunk = uint64_t(1) << 63;
    std::atomic<uint64_t> first_direct_foreground{0}, direct_tail_ns{0};
    void note_direct_foreground() {
        uint64_t open = first_direct_foreground.load(std::memory_order_relaxed);
        if (open & kOpenChunk)
            first_direct_foreground.compare_exchange_strong(open, host_now_ns(), std::memory_order_relaxed);
    }
    uint64_t revoke_upload_guard(bool disable=false) {
        const auto previous=disable?upload_guard.disable():upload_guard.revoke();
        if (!disable && previous.admitted) note_direct_foreground();
        sguard_revocations.fetch_add(1,std::memory_order_relaxed);
        if(previous.admitted) {
            sguard_already_admitted.fetch_add(1,std::memory_order_relaxed);
        }
        // Also include an unscheduled copy which predates enabling the guard.
        if(x4_piece_submitted.load(std::memory_order_acquire))
            sguard_already_inflight.fetch_add(1,std::memory_order_relaxed);
        return previous.token;
    }
    // A persistent pool for the scheduled host copies: the CPU worker copies part 0, the others the rest.
    struct CopyPool {
        std::vector<std::thread> threads;
        std::mutex m; std::condition_variable cv;
        uint8_t* dst = nullptr; const uint8_t* src = nullptr; size_t bytes = 0;
        uint64_t generation = 0; bool quit = false; int parts = 1;
        std::atomic<int> left{0};
        void start(int n) {
            parts = n;
            for (int i = 1; i < n; ++i) threads.emplace_back([this, i] { work(i); });
        }
        void part(int i) {
            const size_t per = ((bytes + size_t(parts) - 1) / size_t(parts) + 4095) & ~size_t(4095);
            const size_t a = std::min(bytes, per * size_t(i)), b = std::min(bytes, a + per);
            if (b > a) std::memcpy(dst + a, src + a, b - a);
        }
        void work(int i) {
            uint64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock<std::mutex> lock(m);
                    cv.wait(lock, [&] { return quit || generation != seen; });
                    if (quit) return;
                    seen = generation;
                }
                part(i);
                left.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
        void copy(void* d, const void* s, size_t n) {
            {
                std::lock_guard<std::mutex> lock(m);
                dst = static_cast<uint8_t*>(d); src = static_cast<const uint8_t*>(s); bytes = n;
                left.store(parts - 1, std::memory_order_release); ++generation;
            }
            cv.notify_all();
            part(0);
            while (left.load(std::memory_order_acquire) > 0) std::this_thread::yield();
        }
        ~CopyPool() {
            { std::lock_guard<std::mutex> lock(m); quit = true; }
            cv.notify_all();
            for (auto& t : threads) if (t.joinable()) t.join();
        }
    };

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
        bool direct=false, direct_download=false, direct_verify=false, probe=false;
        int direct_old_slot=-1;
        BackgroundTraffic* budget=nullptr;
        unsigned actor=0;
        int old_id = -1, new_id = -1, slot = -1;
        void *host = nullptr, *incoming = nullptr, *outgoing = nullptr;
        float weight[3]{}, input[3]{};
        std::string capture_context; // captured when the lease was submitted
        std::chrono::steady_clock::time_point started{};
        std::string error;
        std::unique_ptr<CopyPool> copier;   // exchange_schedule's host copies, started on first use (CPU worker only)
    };
    std::unique_ptr<Live> live;

    // exchange_schedule, worker side. A stop or failure is seen between pieces without ever blocking on mu.
    bool live_should_stop() {
        std::unique_lock<std::mutex> lock(mu, std::try_to_lock);
        return lock.owns_lock() && (live->stop || poisoned || live->phase == LivePhase::failed);
    }
    // Waits until the next x4 piece may start: no pass submitted and unfinished, and the piece expected to end inside
    // the open window. Unscheduled permission is also tagged, so enabling the
    // guard cannot mistake a previously obtained permission for a new claim.
    bool x4_wait(uint64_t est_ns,uint64_t& token,bool& limited) {
        token=0;limited=false;
        const uint64_t began = host_now_ns();
        for (unsigned spins = 0;; ++spins) {
            const bool scheduled=sched_mode.load(std::memory_order_relaxed)!=0 || live->direct;
            const uint64_t now = host_now_ns();
            if (!scheduled || (!pass_busy.load(std::memory_order_acquire) && now + est_ns <= x4_until.load(std::memory_order_acquire) &&
                !cpu_busy.load(std::memory_order_acquire))) {
                cudaError_t failure=cudaSuccess;unsigned queries=0;
                const auto claim=upload_guard.claim([&](void* event) {
                    ++queries;
                    const auto status=cudaEventQuery(static_cast<cudaEvent_t>(event));
                    if(status==cudaSuccess)return UploadGuard::Query::ready;
                    if(status==cudaErrorNotReady)return UploadGuard::Query::pending;
                    failure=status;return UploadGuard::Query::error;
                },upload_poll_cache,now,20000); // at most one pending-query pass per 20 us/generation
                if(queries)sguard_event_queries.fetch_add(queries,std::memory_order_relaxed);
                if(claim.result==UploadGuard::Result::closed)sguard_closed_polls.fetch_add(1,std::memory_order_relaxed);
                if(claim.result==UploadGuard::Result::pending)sguard_pending_polls.fetch_add(1,std::memory_order_relaxed);
                if(claim.result==UploadGuard::Result::deferred)sguard_deferred_polls.fetch_add(1,std::memory_order_relaxed);
                if(claim.result==UploadGuard::Result::error) {
                    sguard_event_errors.fetch_add(1,std::memory_order_relaxed);
                    ck(failure,"query quiet upload guard"); // live_gpu_run reports/poisons the failed lease
                }
                if(claim.result==UploadGuard::Result::admitted || claim.result==UploadGuard::Result::unrestricted) {
                    const bool guarded=claim.result==UploadGuard::Result::admitted;
                    if(guarded) {
                        sguard_admissions.fetch_add(1,std::memory_order_relaxed);
                        if(!queries)sguard_query_free_admissions.fetch_add(1,std::memory_order_relaxed);
                    }
                    const bool check_window=guarded || sched_mode.load(std::memory_order_relaxed)!=0;
                    if(upload_guard.current(claim.token) && (!check_window || (!pass_busy.load(std::memory_order_acquire) &&
                       !cpu_busy.load(std::memory_order_acquire) && host_now_ns()+est_ns<=x4_until.load(std::memory_order_acquire)))) {
                        token=claim.token;limited=guarded;return true;
                    }
                    if(!upload_guard.current(claim.token))sguard_stale_claims.fetch_add(1,std::memory_order_relaxed);
                    upload_guard.release(claim.token);
                }
                if(claim.result==UploadGuard::Result::stale)sguard_stale_claims.fetch_add(1,std::memory_order_relaxed);
                sguard_failed_claims.fetch_add(1,std::memory_order_relaxed);
            }
            if ((spins & 255) == 0) { live_progress_ns.store(now, std::memory_order_relaxed); if (live_should_stop()) return false; }
            // a window opens a few hundred microseconds after each pass's collect: spin, then back off when none comes
            if (now - began < 2000000) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    bool host_wait(uint64_t est_ns) {
        for (unsigned spins = 0;; ++spins) {
            if (sched_mode.load(std::memory_order_relaxed) == 0) return true;
            const uint64_t now = host_now_ns();
            if (now >= host_open.load(std::memory_order_acquire) && now + est_ns <= host_close.load(std::memory_order_acquire)) return true;
            if ((spins & 63) == 0) { live_progress_ns.store(now, std::memory_order_relaxed); if (live_should_stop()) return false; }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    // One host-to-host copy of a live payload: at once (mode 0), or in pieces inside host windows, each piece then passed
    // to check(begin,end) while it is hot (the copy-in's payload checks). Mode 2 compares the whole copy.
    template <class Check>
    void live_host_copy(void* dst, const void* src, size_t bytes, uint64_t& est_ns, Check&& check) {
        if (sched_mode.load(std::memory_order_relaxed) == 0) { std::memcpy(dst, src, bytes); check(size_t(0), bytes); return; }
        if (!live->copier) { live->copier = std::make_unique<CopyPool>(); live->copier->start(kHostThreads); }
        auto* d = static_cast<uint8_t*>(dst); const auto* s = static_cast<const uint8_t*>(src);
        for (size_t off = 0; off < bytes;) {
            if (!host_wait(est_ns)) throw std::runtime_error("live helper stopped while its host copy was scheduled");
            const bool scheduled = sched_mode.load(std::memory_order_relaxed) != 0;
            const size_t n = scheduled ? std::min(kHostPiece, bytes - off) : bytes - off;
            const uint64_t t0 = host_now_ns();
            live->copier->copy(d + off, s + off, n);
            check(off, off + n);
            const uint64_t dt = host_now_ns() - t0;
            if (n == kHostPiece) est_ns = (3 * est_ns + dt) / 4;
            shost_pieces.fetch_add(1, std::memory_order_relaxed); shost_ns.fetch_add(dt, std::memory_order_relaxed);
            live_progress_ns.store(host_now_ns(), std::memory_order_relaxed);
            off += n;
        }
        if (sched_mode.load(std::memory_order_relaxed) == 2) {
            if (std::memcmp(dst, src, bytes) != 0) { smismatches.fetch_add(1); throw std::runtime_error("live helper host copy differs from its source"); }
            sverified.fetch_add(1);
        }
    }

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
        // A scheduled worker stamps live_progress_ns per piece and while it waits for a window (zero unless scheduled).
        auto last = l.started;
        const auto stamped = std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(live_progress_ns.load(std::memory_order_relaxed))));
        if (stamped > last) last = stamped;
        if ((l.cpu_busy || l.gpu_busy) && std::chrono::steady_clock::now()-last >= kProgressDeadline) {
            live_fail_locked("live helper progress deadline exceeded", l.gpu_busy);
            return;
        }
        if (l.phase==LivePhase::preparing && l.cpu_done && l.gpu_done) l.phase=LivePhase::prepared;
        else if (l.phase==LivePhase::uploading && l.gpu_done) l.phase=LivePhase::uploaded;
        else if (l.phase==LivePhase::recycling && l.cpu_done) l.phase=LivePhase::complete;
    }
    void live_cpu_run() {
        uint64_t host_est_ns = 250000;   // exchange_schedule: a host piece's measured duration (four threads), first guess
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
                if (recycle) live_host_copy(host,outgoing,stride,host_est_ns,[](size_t,size_t) {});
                else if (sched_mode.load(std::memory_order_relaxed) == 0) {
                    std::memcpy(incoming,host,stride);
                    if (!refresh_payload(incoming,live->weight,live->input,error) && error.empty()) error="invalid live helper payload";
                } else {
                    // (exchange_schedule: refresh_payload's checks piece by piece, inside the same windows, on whole units)
                    size_t checked=0;
                    live_host_copy(incoming,host,stride,host_est_ns,[&](size_t,size_t end) {
                        const size_t whole=end/144*144;
                        if (error.empty() && whole>checked) {
                            if (!refresh_scales(incoming,live->weight,live->input,checked,whole,error) && error.empty())
                                error="invalid live helper payload";
                            checked=whole;
                        }
                    });
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
    // exchange_schedule: one live device copy in x4 pieces, each started inside an x4 window and joined before the next.
    void live_x4_copy(uint8_t* dst, const uint8_t* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream,
                      cudaEvent_t event, uint64_t& est_ns) {
        for (size_t off = 0; off < bytes;) {
            uint64_t token=0;bool limited=false;
            if (!x4_wait(est_ns,token,limited)) throw std::runtime_error("live helper stopped while its device copy was scheduled");
            struct ReleaseClaim { UploadGuard& guard; uint64_t token; ~ReleaseClaim(){guard.release(token);} } claim_end{upload_guard,token};
            if(token && !upload_guard.current(token)) {
                sguard_stale_claims.fetch_add(1,std::memory_order_relaxed);
                sguard_failed_claims.fetch_add(1,std::memory_order_relaxed);
                continue;
            }
            const bool scheduled = limited || sched_mode.load(std::memory_order_relaxed) != 0;
            const size_t n = scheduled ? std::min(kX4Piece, bytes - off) : bytes - off;
            const uint64_t t0 = host_now_ns();
            x4_piece_busy.store(true, std::memory_order_release);
            ck(cudaMemcpyAsync(dst + off, src + off, n, kind, stream), "submit live helper piece");
            x4_piece_submitted.store(true,std::memory_order_release);
            ck(cudaEventRecord(event, stream), "record live helper piece");
            const auto deadline = std::chrono::steady_clock::now() + kProgressDeadline;
            for (;;) {
                const auto status = cudaEventQuery(event);
                if (status == cudaSuccess) break;
                if (status != cudaErrorNotReady) ck(status, "query live helper piece");
                if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("live helper progress deadline exceeded");
                std::this_thread::yield();   // (a piece takes about 0.09 ms: observe its end promptly)
            }
            x4_piece_submitted.store(false,std::memory_order_release);
            x4_piece_busy.store(false, std::memory_order_release);
            const uint64_t dt = host_now_ns() - t0;
            if (n == kX4Piece) est_ns = (3 * est_ns + dt) / 4;
            sx4_pieces.fetch_add(1, std::memory_order_relaxed); sx4_ns.fetch_add(dt, std::memory_order_relaxed);
            live_progress_ns.store(host_now_ns(), std::memory_order_relaxed);
            off += n;
        }
    }
    void live_direct_copy(const uint8_t* incoming,uint8_t* outgoing,uint8_t* destination,int old_slot,bool download,
                          BackgroundTraffic* budget,unsigned actor,cudaStream_t up,cudaStream_t down,
                          cudaEvent_t up_done,cudaEvent_t down_done,uint64_t& estimate_ns) {
        if(!budget || actor>1)throw std::runtime_error("direct mover has no shared traffic budget");
        const size_t piece=actor==1 ? size_t(128)<<10 : kX4Piece;
        weight_issue(true,stride);if(download)weight_issue(false,stride);
        for(size_t off=0;off<stride;) {
            if(!budget->healthy())throw std::runtime_error("shared background traffic is poisoned");
            uint64_t token=0;bool limited=false;
            if(!x4_wait(estimate_ns,token,limited))throw std::runtime_error("direct mover stopped before a chunk");
            struct Release {UploadGuard& guard;uint64_t token;~Release(){guard.release(token);}} claim{upload_guard,token};
            const size_t n=std::min(piece,stride-off),traffic=n*(download?2:1);
            if(!budget->acquire(actor,traffic)) {std::this_thread::yield();continue;}
            if(!upload_guard.current(token) || cpu_busy.load(std::memory_order_acquire) || pass_busy.load(std::memory_order_acquire)) {
                budget->release(actor,traffic);continue;
            }
            const uint64_t began=host_now_ns();
            first_direct_foreground.store(kOpenChunk | began,std::memory_order_release);
            if (!upload_guard.current(token) || cpu_busy.load(std::memory_order_acquire) || pass_busy.load(std::memory_order_acquire)) {
                first_direct_foreground.store(0,std::memory_order_release);
                budget->release(actor,traffic);
                continue;
            }
            x4_piece_busy.store(true,std::memory_order_release);
            // Separate destinations and streams permit the 3090's two directions
            // to overlap. The old ready slot remains immutable until publication.
            ck(cudaMemcpyAsync(destination+off,incoming+off,n,cudaMemcpyHostToDevice,up),"direct upload chunk");
            sdirect_h2d_bytes.fetch_add(n,std::memory_order_relaxed);
            x4_piece_submitted.store(true,std::memory_order_release);
            ck(cudaEventRecord(up_done,up),"direct upload chunk completion");
            if(download) {
                ck(cudaMemcpyAsync(outgoing+off,pool+size_t(old_slot)*stride+off,n,cudaMemcpyDeviceToHost,down),"direct download chunk");
                sdirect_d2h_bytes.fetch_add(n,std::memory_order_relaxed);
                ck(cudaEventRecord(down_done,down),"direct download chunk completion");
            }
            const auto deadline=std::chrono::steady_clock::now()+kProgressDeadline;
            for(;;) {
                const auto a=cudaEventQuery(up_done),b=download?cudaEventQuery(down_done):cudaSuccess;
                if(a==cudaSuccess && b==cudaSuccess)break;
                if(a!=cudaSuccess && a!=cudaErrorNotReady)ck(a,"direct upload progress");
                if(b!=cudaSuccess && b!=cudaErrorNotReady)ck(b,"direct download progress");
                if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("direct chunk progress deadline exceeded");
                std::this_thread::yield();
            }
            x4_piece_submitted.store(false,std::memory_order_release);
            x4_piece_busy.store(false,std::memory_order_release);
            const uint64_t end = host_now_ns();
            const uint64_t boundary = first_direct_foreground.exchange(0,std::memory_order_acq_rel);
            if (boundary >= began && boundary < end) direct_tail_ns.fetch_add(end-boundary,std::memory_order_relaxed);
            budget->release(actor,traffic);
            const uint64_t dt=end-began;
            if(n==piece)estimate_ns=(3*estimate_ns+dt)/4;
            sdirect_chunks.fetch_add(1,std::memory_order_relaxed);
            sdirect_span_ns.fetch_add(dt,std::memory_order_relaxed);
            sx4_pieces.fetch_add(download?2:1,std::memory_order_relaxed);
            sx4_ns.fetch_add(dt,std::memory_order_relaxed); // elapsed pair span, not sum of overlapping directions
            live_progress_ns.store(host_now_ns(),std::memory_order_relaxed);
            off+=n;
        }
    }
    void live_gpu_run() {
        cudaStream_t copy_stream=nullptr,down_stream=nullptr; cudaEvent_t event=nullptr,down_event=nullptr;
        uint64_t x4_est_ns = 110000;     // exchange_schedule: an x4 piece's measured duration, first guess
        std::vector<uint8_t> verify_back; // (mode 2: the slot read back)
        for (;;) {
            bool upload,direct,download,verify,probe; void *incoming, *outgoing; int slot,old_slot;
            BackgroundTraffic* budget=nullptr;unsigned actor=0;std::string capture_context;
            {
                std::unique_lock<std::mutex> lock(mu);
                live->cv.wait(lock,[&]{return live->stop || live->gpu_queued;});
                if (live->stop && !live->gpu_queued) break;
                auto& l=*live; upload=l.upload; incoming=l.incoming; outgoing=l.outgoing; slot=l.slot;
                direct=l.direct;probe=l.probe;download=l.direct_download;verify=l.direct_verify;old_slot=l.direct_old_slot;
                budget=l.budget;actor=l.actor;
                capture_context=l.capture_context;
                l.gpu_queued=false;
            }
            std::string error; bool uncertain=false, submitted=false;
            if(!capture_context.empty())nvtxRangePushA((capture_context+(upload?" phase=helper_live_h2d":" phase=helper_live_d2h")).c_str());
            try {
                DeviceGuard guard(device);
                if (!copy_stream) ck(cudaStreamCreateWithFlags(&copy_stream,cudaStreamNonBlocking),"create live helper stream");
                if (!event) ck(cudaEventCreateWithFlags(&event,cudaEventDisableTiming),"create live helper event");
                if(direct) {
                    if(download && !down_stream)ck(cudaStreamCreateWithFlags(&down_stream,cudaStreamNonBlocking),"create direct download stream");
                    if(download && !down_event)ck(cudaEventCreateWithFlags(&down_event,cudaEventDisableTiming),"create direct download event");
                    submitted=true;
                    uint8_t* dst = probe && !download ? probe_destination : pool + size_t(slot)*stride;
                    live_direct_copy(static_cast<const uint8_t*>(incoming),static_cast<uint8_t*>(outgoing),dst,old_slot,
                                     download,budget,actor,copy_stream,down_stream,event,down_event,x4_est_ns);
                    if(verify) {
                        verify_back.resize(stride);
                        ck(cudaMemcpyAsync(verify_back.data(),dst,stride,cudaMemcpyDeviceToHost,copy_stream),"check direct incoming payload");
                        ck(cudaStreamSynchronize(copy_stream),"join direct incoming check");
                        if(std::memcmp(verify_back.data(),incoming,stride)!=0) {
                            smismatches.fetch_add(1);sdirect_mismatches.fetch_add(1);throw std::runtime_error("direct incoming payload differs");
                        }
                        sverified.fetch_add(1);sdirect_verified.fetch_add(1);
                        if(download) {
                            ck(cudaMemcpyAsync(verify_back.data(),pool+size_t(old_slot)*stride,stride,cudaMemcpyDeviceToHost,down_stream),"check direct outgoing payload");
                            ck(cudaStreamSynchronize(down_stream),"join direct outgoing check");
                            if(std::memcmp(verify_back.data(),outgoing,stride)!=0) {
                                smismatches.fetch_add(1);sdirect_mismatches.fetch_add(1);throw std::runtime_error("direct outgoing payload differs");
                            }
                            sverified.fetch_add(1);sdirect_verified.fetch_add(1);
                        }
                    }
                } else {
                unsigned flags=0;
                ck(cudaHostGetFlags(&flags,incoming),"inspect incoming staging");
                if (!(flags&cudaHostAllocPortable)) throw std::runtime_error("incoming staging is not portable pinned memory");
                ck(cudaHostGetFlags(&flags,outgoing),"inspect outgoing staging");
                if (!(flags&cudaHostAllocPortable)) throw std::runtime_error("outgoing staging is not portable pinned memory");
                // A failed submission can still leave stream state uncertain.
                submitted=true;
                const auto verify_copy=[&] {
                    if (sched_mode.load(std::memory_order_relaxed) != 2)return;
                    verify_back.resize(stride);
                    ck(cudaMemcpyAsync(verify_back.data(), pool + size_t(slot) * stride, stride, cudaMemcpyDeviceToHost, copy_stream), "check live helper copy");
                    ck(cudaStreamSynchronize(copy_stream), "join live helper check");
                    if (std::memcmp(verify_back.data(), upload ? incoming : outgoing, stride) != 0) {
                        smismatches.fetch_add(1);throw std::runtime_error("live helper device copy differs from its source");
                    }
                    sverified.fetch_add(1);
                };
                if (sched_mode.load(std::memory_order_relaxed) != 0) {
                    uint8_t* const dst = upload ? pool + size_t(slot) * stride : static_cast<uint8_t*>(outgoing);
                    const uint8_t* const src = upload ? static_cast<const uint8_t*>(incoming) : pool + size_t(slot) * stride;
                    weight_issue(upload,stride);
                    if(!capture_context.empty())capture::detail_event("helper_weight_copy_issue",
                        "\"submission_context\":"+capture::quote(capture_context)+",\"operation\":\"live\",\"direction\":"+
                        capture::quote(upload?"h2d":"d2h")+",\"bytes\":"+std::to_string(stride)+",\"slot\":"+std::to_string(slot)+",\"scheduled\":1");
                    live_x4_copy(dst, src, stride, upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, copy_stream, event, x4_est_ns);
#ifdef MM3_HELPER_TEST_HOOKS
                    native_helper_test_gpu_copy(upload);
#endif
                    verify_copy();
                } else {
                    for(;;) {
                        uint64_t token=0;bool limited=false;
                        if(!x4_wait(x4_est_ns,token,limited))throw std::runtime_error("live helper stopped before its device copy");
                        struct ReleaseClaim {UploadGuard& guard;uint64_t token;~ReleaseClaim(){guard.release(token);}} claim_end{upload_guard,token};
                        if(!upload_guard.current(token)) {
                            sguard_stale_claims.fetch_add(1,std::memory_order_relaxed);
                            sguard_failed_claims.fetch_add(1,std::memory_order_relaxed);continue;
                        }
                        if(limited || sched_mode.load(std::memory_order_relaxed)!=0) {
                            // Scheduling was enabled after this operation was queued.
                            // Give back the bulk admission and enter the chunk path.
                            upload_guard.release(token);
                            weight_issue(upload,stride);
                            if(!capture_context.empty())capture::detail_event("helper_weight_copy_issue",
                                "\"submission_context\":"+capture::quote(capture_context)+",\"operation\":\"live\",\"direction\":"+
                                capture::quote(upload?"h2d":"d2h")+",\"bytes\":"+std::to_string(stride)+",\"slot\":"+std::to_string(slot)+",\"scheduled\":1,\"queued_before_enable\":true");
                            live_x4_copy(upload?pool+size_t(slot)*stride:static_cast<uint8_t*>(outgoing),
                                         upload?static_cast<const uint8_t*>(incoming):pool+size_t(slot)*stride,
                                         stride,upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,copy_stream,event,x4_est_ns);
#ifdef MM3_HELPER_TEST_HOOKS
                            native_helper_test_gpu_copy(upload);
#endif
                            verify_copy();break;
                        }
                        x4_piece_busy.store(true,std::memory_order_release);
                        ck(cudaMemcpyAsync(upload ? pool+size_t(slot)*stride : outgoing,
                                           upload ? incoming : pool+size_t(slot)*stride, stride,
                                           upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost,copy_stream),"submit live helper copy");
                        x4_piece_submitted.store(true,std::memory_order_release);
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
                        x4_piece_submitted.store(false,std::memory_order_release);
                        x4_piece_busy.store(false,std::memory_order_release);
                        break;
                    }
                }   // (sched_mode 0)
                }   // legacy staging transfer
            } catch (const std::exception& e) {
                error=e.what();
                x4_piece_submitted.store(false,std::memory_order_release);
                x4_piece_busy.store(false, std::memory_order_release);
                if(direct && budget)budget->poison();
                if (submitted) { try { DeviceGuard guard(device); uncertain=cudaStreamQuery(copy_stream)!=cudaSuccess ||
                    (down_stream && cudaStreamQuery(down_stream)!=cudaSuccess); } catch (...) { uncertain=true; } }
            } catch (...) { error="unknown live helper CUDA failure"; uncertain=submitted;
                if(direct && budget)budget->poison();
                x4_piece_submitted.store(false,std::memory_order_release);x4_piece_busy.store(false,std::memory_order_release); }
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
            try { DeviceGuard guard(device); if (event) cudaEventDestroy(event); if (copy_stream) cudaStreamDestroy(copy_stream);
                if(down_event)cudaEventDestroy(down_event);if(down_stream)cudaStreamDestroy(down_stream); }
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
        static_assert(kMaxTokens * 4 == kMaxJobs, "helper IO and routing column capacities differ");
        pass_tokens = configured_pass_tokens();
        const int max_columns_total = pass_tokens * 4;
        pending_jobs.resize(size_t(max_columns_total)); pending_offsets.resize(size_t(max_columns_total));
        merge_cols.reserve(size_t(max_columns_total)); merge_refused.reserve(size_t(max_columns_total));
        for (int cols = 1; compute && cols <= kMaxColumns; ++cols)
            workspace_bytes = std::max(workspace_bytes, size_t(max_columns_total / cols) *
                mf::fast_workspace_bytes(kHidden, kIntermediate, cols));
        if (compute) workspace_bytes = std::max({workspace_bytes, mm3::helper_tc::workspace_bytes(max_columns_total),
                                                 size_t(max_columns_total) * mm3::helper_dp::job_workspace(8)});
        allocated.pool = stride * size_t(capacity);
        const size_t io = buffer_elements(pass_tokens) * sizeof(uint16_t);
        const size_t jobs = size_t(max_columns_total) * sizeof(DeviceJob);
        allocated.device_scratch = compute ? 2 * io + jobs + workspace_bytes : 0;
        allocated.pinned = compute ? 2 * io + jobs : 0;
        DeviceGuard guard(device);
        try {
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create helper stream");
#if defined(__CUDACC__)
            {   // (kernel item 5: whether this card can run the sm_89 build - the RTX 4080; not in the CPU tests' fake API)
                int major = 0, minor = 0;
                if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess)
                    sm89_device = major == 8 && minor == 9;
                (void)cudaGetLastError();
            }
#endif
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
        if(stage_clock_ref)cudaEventDestroy(stage_clock_ref);
        for(auto event:stage_events)if(event)cudaEventDestroy(event);
        for(auto event:replay_events)if(event)cudaEventDestroy(event);
        graph_destroy();
        if (stream) cudaStreamDestroy(stream);
        if (probe_destination) cudaFree(probe_destination);
        if (pool) cudaFree(pool); if (workspace) cudaFree(workspace);
        if (d_x) cudaFree(d_x); if (d_y) cudaFree(d_y); if (d_jobs) cudaFree(d_jobs);
        if (d_q) cudaFree(d_q); if (d_hq) cudaFree(d_hq); if (d_qp) cudaFree(d_qp); if (d_expanded) cudaFree(d_expanded);
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
                    graph_stage_events=!refresh && pending_stage_events;
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
                const int pass_loads = pending_loads;
                const bool pass_sm89 = pending_sm89;
                const int pass_format = refresh ? 0 : pending_format;
                const int pass_kernel = pending_kernel;
                const int pass_execution = pending_execution;
                float finish_tail_ms=0,card_clock_ms=0;
                bool used_graph = false;
                try {
                    if(graph_stage_events && !stage_events[0]){
                        for(auto& e:stage_events)ck(cudaEventCreate(&e),"helper graph timing event");
                        ck(cudaEventCreate(&stage_clock_ref),"helper clock epoch");
                        ck(cudaEventRecord(stage_clock_ref,stream),"start helper clock epoch");
                    }
                    if (refresh) {
                        ck(cudaMemcpyAsync(refresh_dst,refresh_src,refresh_bytes,refresh_kind,stream), "submit helper refresh copy");
                        weight_issue(refresh_kind==cudaMemcpyHostToDevice,refresh_bytes);
                        if(!worker_capture_context.empty())capture::detail_event("helper_weight_copy_issue",
                            "\"submission_context\":"+capture::quote(worker_capture_context)+",\"operation\":\"placement\",\"direction\":"+capture::quote(refresh_kind==cudaMemcpyHostToDevice?"h2d":"d2h")+
                            ",\"bytes\":"+std::to_string(refresh_bytes));
                    } else if (const int zc_mode = zero_copy_mode.load(std::memory_order_relaxed), one_kernel = pass_kernel;
                               !gpu_timed && fast_mode.load(std::memory_order_relaxed) && pending_columns == n &&
                               graph_ready(n, zc_mode > 0, one_kernel, pass_loads, pass_sm89, pass_format, pass_execution)) {
                        // every job has one column (the total equals the job count): one launch replaces five calls
                        ck(cudaGraphLaunch(graph_bank(pass_execution).exec[graph_slot(one_kernel, pass_loads, pass_sm89, pass_format)][zc_mode > 0][size_t(n)], stream), "launch helper graph");
                        zc_check = zc_mode == 2;
                        used_graph = true;
                        graph_count.fetch_add(1, std::memory_order_relaxed);
                        if (kernel_index(one_kernel)) kernel_count.fetch_add(1, std::memory_order_relaxed);
                    } else if (cudaGraphExec_t wide = !gpu_timed && fast_mode.load(std::memory_order_relaxed) && pending_columns > n
                                   ? wide_graph(n, pending_columns, zero_copy_mode.load(std::memory_order_relaxed) > 0, pass_kernel, pass_loads,
                                                pass_sm89, pass_format, pass_execution) : nullptr; wide) {
                        // a pass with multi-column jobs as one graph (set_kernel, 100 added)
                        ck(cudaGraphLaunch(wide, stream), "launch helper wide graph");
                        zc_check = zero_copy_mode.load(std::memory_order_relaxed) == 2;
                        used_graph = true;
                        graph_count.fetch_add(1, std::memory_order_relaxed);
                        kernel_count.fetch_add(1, std::memory_order_relaxed);
                    } else {
                    const size_t io = size_t(pending_columns) * kHidden * sizeof(uint16_t);
                    if(gpu_timed)ck(cudaEventRecord(replay_events[0],stream),"helper activation timing");
                    ck(cudaMemcpyAsync(d_x, h_x, input_bytes(pending_columns, pass_format), cudaMemcpyHostToDevice, stream), "copy helper input");
                    ck(cudaMemcpyAsync(d_jobs, h_jobs, size_t(n) * sizeof(DeviceJob), cudaMemcpyHostToDevice, stream), "copy helper jobs");
                    if(gpu_timed)ck(cudaEventRecord(replay_events[1],stream),"helper compute timing");
                    // This card's own kernel takes jobs of every width in one launch.
                    if (!input_valid(pass_kernel, pass_format)) throw std::runtime_error("helper input representation does not match kernel");
                    if (kernel_q(pass_kernel)) {
                        if (!ensure_q(pass_kernel, pass_format)) throw std::runtime_error("helper NVFP4-activation buffers could not be allocated");
                        ck(q_launch(pass_kernel, pass_loads, d_jobs, n, pending_columns, pass_format, use_sm89(pass_sm89), pass_execution), "launch helper NVFP4-activation kernel");
                        kernel_count.fetch_add(1, std::memory_order_relaxed);
                    } else if (kernel_tensor(pass_kernel)) {
                        ck(tensor_launch(pass_kernel, pass_loads, d_jobs, n, use_sm89(pass_sm89)), "launch helper tensor-core kernel");
                        kernel_count.fetch_add(1, std::memory_order_relaxed);
                    } else if (kernel_any(pass_kernel)) {
                        ck(any_launch(d_jobs, n), "launch helper any-width kernel");
                        kernel_count.fetch_add(1, std::memory_order_relaxed);
                    } else
                    // Jobs are sorted by width; the shared fast API requires a
                    // common width for every descriptor in a launch. Workspace
                    // is reused only by consecutive launches on this stream.
                    for (int first = 0; first < n;) {
                        const int cols = pending_jobs[first].columns;
                        int end = first + 1;
                        while (end < n && pending_jobs[end].columns == cols) ++end;
                        if (size_t(end-first) * mf::fast_workspace_bytes(kHidden,kIntermediate,cols) > workspace_bytes)
                            throw std::runtime_error("helper width group exceeds workspace");
                        if (kernel_index(pass_kernel)) {      // (16: today's arithmetic compiled for this width)
                            ck(mm3::helper_dp::ffn(d_jobs + first, end - first, cols, workspace, stream), "launch helper width kernel");
                            kernel_count.fetch_add(1, std::memory_order_relaxed);
                        } else mf::fast_ffn_cuda_packed(d_jobs + first, end - first, kHidden, kIntermediate, cols, workspace, stream);
                        first = end;
                    }
                    if(gpu_timed)ck(cudaEventRecord(replay_events[2],stream),"helper result timing");
                    ck(cudaMemcpyAsync(h_y, d_y, io, cudaMemcpyDeviceToHost, stream), "copy helper output");
                    if(gpu_timed)ck(cudaEventRecord(replay_events[3],stream),"helper finish timing");
                    }
                    if (!refresh) event_record_begin_ns = host_now_ns();
                    const auto finished=graph_stage_events && used_graph ? stage_events[4] : done_event;
                    ck(cudaEventRecord(finished, stream), "record helper event");
                    if (!refresh) event_record_end_ns = host_now_ns();
                    const auto started = std::chrono::steady_clock::now();
                    const auto deadline = started + kProgressDeadline;
                    for (;;) {
                        const uint64_t query_begin_ns = refresh ? 0 : host_now_ns();
                        cudaError_t status = cudaEventQuery(finished);
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
                        const size_t io = size_t(pending_columns) * kHidden * sizeof(uint16_t), jb = size_t(n) * sizeof(DeviceJob);
                        const size_t in = input_bytes(pending_columns, pass_format);
                        zc_back.resize(io);
                        ck(cudaMemcpy(zc_back.data(), d_x, in, cudaMemcpyDeviceToHost), "check helper zero-copy input");
                        bool same = std::memcmp(zc_back.data(), h_x, in) == 0;
                        ck(cudaMemcpy(zc_back.data(), d_jobs, jb, cudaMemcpyDeviceToHost), "check helper zero-copy jobs");
                        same = same && std::memcmp(zc_back.data(), h_jobs, jb) == 0;
                        ck(cudaMemcpy(zc_back.data(), d_y, io, cudaMemcpyDeviceToHost), "check helper zero-copy result");
                        same = same && std::memcmp(zc_back.data(), h_y, io) == 0;
                        (same ? zc_checked : zc_mismatches).fetch_add(1, std::memory_order_relaxed);
                    }
                    if(graph_stage_events && used_graph) {
                        ck(cudaEventElapsedTime(&activation_ms,stage_events[0],stage_events[1]),"graph input timing");
                        ck(cudaEventElapsedTime(&compute_ms,stage_events[1],stage_events[2]),"graph compute timing");
                        ck(cudaEventElapsedTime(&result_ms,stage_events[2],stage_events[3]),"graph result timing");
                        ck(cudaEventElapsedTime(&card_clock_ms,stage_clock_ref,stage_events[4]),"helper clock calibration");
                        ck(cudaEventElapsedTime(&finish_tail_ms,stage_events[3],stage_events[4]),"graph tail timing");
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
                        pass_timing.input_format = pass_format;
                        pass_timing.kernel = pass_kernel;
                        pass_timing.execution = pass_execution;
                        pass_timing.multi = pending_columns > n;
                        pass_timing.sm89 = use_sm89(pass_sm89);
                        pass_timing.graph = used_graph;
                        pass_timing.graph_fallback = !gpu_timed && fast_mode.load(std::memory_order_relaxed) && !used_graph &&
                            n <= kMaxJobs && pending_columns <= kMaxJobs &&
                            (pending_columns == n || kernel_wide_graph(pass_kernel));
                        pass_timing.gpu_measured=gpu_timed || (graph_stage_events && used_graph);
                        pass_timing.staged_graph_timing=graph_stage_events && used_graph;pass_timing.finish_tail_ms=finish_tail_ms;pass_timing.card_clock_ms=card_clock_ms;
                        pass_timing.activation_gpu_ms=activation_ms;pass_timing.compute_gpu_ms=compute_ms;pass_timing.result_gpu_ms=result_ms;
                        pass_timing.event_record_begin_ns = event_record_begin_ns;
                        pass_timing.event_record_end_ns = event_record_end_ns;
                        pass_timing.last_not_ready_ns = last_not_ready_ns;
                        pass_timing.ready_observed_ns = ready_observed_ns;
                        pass_timing.worker_published_ns = host_now_ns();
                        finished = true;
                        finished_flag.store(true, std::memory_order_release);
                        pass_busy.store(false, std::memory_order_release);
                    }
                }
                cv.notify_all();
                if (!refresh && error.empty() && h_occupancy && occupancy_want.load(std::memory_order_relaxed)) occupancy_arm();
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mu);
            failure = e.what(); poisoned = true; finished = true;
            pass_busy.store(false, std::memory_order_release);
            cv.notify_all();
        }
    }
};

NativeHelper::NativeHelper(int device, int capacity, bool compute) : p_(std::make_unique<Impl>(device, capacity, compute)) {}
void NativeHelper::set_fast(bool on) { p_->fast_mode.store(on); }
bool NativeHelper::fast() const { return p_->fast_mode.load(); }
void NativeHelper::set_zero_copy(int mode) { p_->zero_copy_mode.store(mode); }
void NativeHelper::set_kernel(int multi, int one) {
    p_->kernel_multi.store(Impl::kernel_valid(multi) ? multi : 0); p_->kernel_one.store(Impl::kernel_index(one) ? Impl::kernel_base(one) : 0);
}
int NativeHelper::kernel_multi() const { return p_->kernel_multi.load(); }
int NativeHelper::kernel_one() const { return p_->kernel_one.load(); }
int NativeHelper::expanded_format(int requested) const {
    const int a = kernel_one(), b = kernel_multi();
    if ((requested == 2 || requested == 4) && Impl::kernel_q3(a) && Impl::kernel_q3(b)) return 2;
    if ((requested == 3 || requested == 4) && Impl::input_valid(a, 3) && Impl::input_valid(b, 3) &&
        Impl::kernel_fp8(a) == Impl::kernel_fp8(b)) return 3;
    return 0;
}
bool NativeHelper::expanded_fp8() const {
    return Impl::kernel_fp8(kernel_one()) && Impl::kernel_fp8(kernel_multi());
}
void NativeHelper::set_loads(int loads) { p_->kernel_loads.store(Impl::loads_valid(loads) ? loads : 1); }
int NativeHelper::loads() const { return p_->kernel_loads.load(); }
bool NativeHelper::set_execution(int mode, std::string& error) {
    auto& p = *p_; std::lock_guard<std::mutex> lock(p.mu);
    if (mode < 0 || mode > 7 || !Impl::execution_valid(mode, p.kernel_one.load()) || !Impl::execution_valid(mode, p.kernel_multi.load())) {
        error = "helper execution accepts 0..3 for Q1 and 0..7 for Q2 in both width modes; bit4 requires Q2"; return false;
    }
    if (mode == p.execution_mode.load()) return true; // no mutation; a background refresh may continue
    if (!p.worker_compute || p.poisoned || p.unsafe || p.active || p.queued || p.refreshing || p.handoff_active) {
        error = "helper execution mode changes require healthy idle ownership"; return false;
    }
    p.execution_mode.store(mode, std::memory_order_release);
    return true;
}
int NativeHelper::execution() const { return p_->execution_mode.load(std::memory_order_acquire); }
uint64_t NativeHelper::kernel_launches() const { return p_->kernel_count.load(); }
int NativeHelper::zero_copy() const { return p_->zero_copy_mode.load(); }
uint64_t NativeHelper::zero_copy_checked() const { return p_->zc_checked.load(std::memory_order_relaxed); }
uint64_t NativeHelper::zero_copy_mismatches() const { return p_->zc_mismatches.load(std::memory_order_relaxed); }
uint64_t NativeHelper::graph_launches() const { return p_->graph_count.load(); }
void NativeHelper::set_occupancy(bool on) {
    p_->occupancy_want.store(on);
    if (!on) p_->occupancy_release();
}
bool NativeHelper::occupancy() const { return p_->occupancy_want.load(); }
void NativeHelper::set_live_schedule(int mode) { p_->sched_mode.store(mode < 0 ? 0 : mode > 2 ? 2 : mode); }
int NativeHelper::live_schedule() const { return p_->sched_mode.load(); }
void NativeHelper::live_x4_window(uint64_t until_ns) { p_->x4_until.store(until_ns, std::memory_order_release); }
uint64_t NativeHelper::live_x4_revoke() {return p_->revoke_upload_guard();}
bool NativeHelper::live_x4_publish(uint64_t generation,const std::array<void*,4>& events,std::string& error) {
    if(!p_->upload_guard.publish(generation,events)) {error="quiet upload guard publication is stale or not revoked";return false;}
    p_->sguard_publications.fetch_add(1,std::memory_order_relaxed);return true;
}
void NativeHelper::live_x4_guard_failed(const std::string& error) {
    (void)p_->revoke_upload_guard();
    p_->sguard_event_errors.fetch_add(1,std::memory_order_relaxed);
    poison(error);
}
void NativeHelper::live_x4_guard(void* uploads_done) {
    if(!uploads_done) {(void)p_->revoke_upload_guard(true);return;}
    const auto generation=live_x4_revoke();std::string error;
    if(!live_x4_publish(generation,{uploads_done,nullptr,nullptr,nullptr},error))poison(error);
}
void NativeHelper::live_cpu_busy(bool busy) {
    p_->cpu_busy.store(busy, std::memory_order_release);
    if (busy) p_->note_direct_foreground();
}
void NativeHelper::live_host_window(uint64_t open_ns, uint64_t close_ns) {
    p_->host_close.store(close_ns, std::memory_order_release); p_->host_open.store(open_ns, std::memory_order_release);
}
uint64_t NativeHelper::clock_ns() { return host_now_ns(); }
NativeHelper::LiveScheduleStats NativeHelper::live_schedule_stats() const {
    LiveScheduleStats s;
    s.x4_pieces = p_->sx4_pieces.load(); s.host_pieces = p_->shost_pieces.load();
    s.x4_ns = p_->sx4_ns.load(); s.host_ns = p_->shost_ns.load(); s.job_overlaps = p_->soverlaps.load();
    s.verified = p_->sverified.load(); s.mismatches = p_->smismatches.load();
    s.guard_revocations=p_->sguard_revocations.load();s.guard_publications=p_->sguard_publications.load();
    s.guard_admissions=p_->sguard_admissions.load();s.guard_failed_claims=p_->sguard_failed_claims.load();
    s.guard_stale_claims=p_->sguard_stale_claims.load();s.guard_event_errors=p_->sguard_event_errors.load();
    s.guard_closed_polls=p_->sguard_closed_polls.load();s.guard_pending_polls=p_->sguard_pending_polls.load();
    s.guard_deferred_polls=p_->sguard_deferred_polls.load();s.guard_event_queries=p_->sguard_event_queries.load();
    s.guard_query_free_admissions=p_->sguard_query_free_admissions.load();
    s.direct_h2d_bytes=p_->sdirect_h2d_bytes.load();s.direct_d2h_bytes=p_->sdirect_d2h_bytes.load();
    s.direct_tail_ns=p_->direct_tail_ns.load();
    s.direct_chunks=p_->sdirect_chunks.load();s.direct_span_ns=p_->sdirect_span_ns.load();
    s.direct_verified=p_->sdirect_verified.load();s.direct_mismatches=p_->sdirect_mismatches.load();
    s.guard_already_admitted=p_->sguard_already_admitted.load();s.guard_already_inflight=p_->sguard_already_inflight.load();
    return s;
}
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
        !record || !weight || !input || bytes != p.stride || p.count == p.capacity-(p.direct_spare_slot>=0?1:0) ||
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
    p.records_version.fetch_add(1, std::memory_order_acq_rel);
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
NativeHelper::Bytes NativeHelper::bytes() const { auto b = p_->allocated; b.device_scratch += p_->q_scratch_bytes.load(std::memory_order_relaxed); return b; }
int NativeHelper::configured_pass_tokens() {
    const char* v = std::getenv("MM3_HELPER_PASS_TOKENS");
    if (!v || !*v) return kMaxTokens;
    char* end = nullptr; const long n = std::strtol(v, &end, 10);
    if (*end || n < kMaxTokens || n > kMaxPassTokens || n % kMaxTokens)
        throw std::invalid_argument("MM3_HELPER_PASS_TOKENS must be a multiple of 16 from 16 to 256");
    return int(n);
}
int NativeHelper::pass_tokens() const { return p_->pass_tokens; }
NativeHelper::Bytes NativeHelper::planned_bytes(int capacity, bool compute) {
    Bytes b{};
    b.pool = mf::record_stride(kHidden,kIntermediate)*size_t(capacity);
    const int pass=configured_pass_tokens();
    const size_t io=buffer_elements(pass)*sizeof(uint16_t);
    const size_t jobs=size_t(4*pass)*sizeof(DeviceJob);
    size_t workspace=0;
    for(int cols=1;compute && cols<=kMaxColumns;++cols)
        workspace=std::max(workspace,size_t(pass*4/cols)*mf::fast_workspace_bytes(kHidden,kIntermediate,cols));
    if(compute)workspace=std::max({workspace,mm3::helper_tc::workspace_bytes(pass*4),size_t(pass*4)*mm3::helper_dp::job_workspace(8)});
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
// refresh_payload's checks in parts (exchange_schedule): the global scales, then the RG16 units in [begin,end) of the
// payload, both multiples of the 144-byte unit; over consecutive ranges covering it, exactly refresh_payload's checks.
bool refresh_scales(const void* record, const float weight[3], const float input[3], size_t begin, size_t end, std::string& error) {
    if (begin == 0) {
        if (!weight || !input) { error="missing helper expert scales"; return false; }
        for (int i=0;i<3;++i) if (!std::isfinite(weight[i]) || weight[i]<=0 || !std::isfinite(input[i]) || input[i]<=0 ||
            !std::isfinite(1.0f/weight[i]) || 1.0f/weight[i]==0.0f) {
            error="invalid helper expert global scale"; return false;
        }
    }
    const auto* source=static_cast<const uint8_t*>(record);
    const size_t payload=mf::record_bytes(NativeHelper::kHidden,NativeHelper::kIntermediate);
    for (size_t off=begin;off<payload && off<end;off+=144) for (int row=0;row<16;++row) {
        const uint8_t scale=source[off+128+row];
        if ((scale&0x80u) || scale==0x7fu) { error="invalid RG16 block scale"; return false; }
    }
    return true;
}
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
// A live replacement's copies write only the slot it took from its victim and read only that slot, so any other slot a
// record names can be read, or rewritten while it is not ready, beside them.
bool NativeHelper::verify_expert(int layer,int expert,const void* expected,size_t bytes,const float weight[3],
                                 const float input[3],bool& same,std::string& error) {
    same=false; auto& p=*p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.refreshing || !valid_identity(layer,expert) || !expected || bytes!=p.stride ||
        !weight || !input) { error="helper record check unavailable, active, or invalid"; return false; }
    const auto& r=p.records[size_t(layer)*kExperts+expert];
    if (!p.repair_target(r,error)) return false;
    try {
        DeviceGuard guard(p.device);
        std::vector<uint8_t> back(p.stride);
        ck(cudaMemcpy(back.data(),p.pool+size_t(r.slot)*p.stride,p.stride,cudaMemcpyDeviceToHost),"read back helper record");
        p.weight_issue(false,p.stride);
        same=std::memcmp(back.data(),expected,p.stride)==0 && std::equal(weight,weight+3,r.weight) && std::equal(input,input+3,r.input);
    } catch (const std::exception& e) { (void)cudaGetLastError(); error=e.what(); return false; }
    return true;
}
bool NativeHelper::rewrite_expert(int layer,int expert,const void* staging,size_t bytes,const float weight[3],
                                  const float input[3],bool& staging_unsafe,std::string& error) {
    staging_unsafe=false; auto& p=*p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.handoff_active || p.refreshing || !valid_identity(layer,expert) || bytes!=p.stride || !weight || !input) {
        error="helper record rewrite unavailable, active, or invalid"; return false;
    }
    auto& r=p.records[size_t(layer)*kExperts+expert];
    if (!p.repair_target(r,error)) return false;
    if (!p.transfer(lock,nullptr,staging,0,cudaMemcpyHostToDevice,staging_unsafe,error,true) ||
        !refresh_payload(staging,weight,input,error)) return false;
    const int slot=r.slot;
    r.slot=-1; // (not ready while it is written: no pass can name it)
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
    if (!p.transfer(lock,p.pool+size_t(slot)*p.stride,staging,bytes,cudaMemcpyHostToDevice,staging_unsafe,error)) return false;
    r.slot=slot;std::copy_n(weight,3,r.weight);std::copy_n(input,3,r.input);
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
    p.weight_replacement_complete();
    return true;
}
void NativeHelper::poison(const std::string& why) {
    auto& p=*p_;
    { std::lock_guard<std::mutex> lock(p.mu); if (!p.poisoned) p.failure=why; p.poisoned=true; }
    p.cv.notify_all();
}
int NativeHelper::device() const { return p_->device; }
bool NativeHelper::replace_expert(int old_layer,int old_expert,int new_layer,int new_expert,
                                  const void* staging,size_t bytes,const float weight[3],const float input[3],
                                  bool& staging_unsafe,std::string& error) {
    staging_unsafe=false; auto& p=*p_; std::unique_lock<std::mutex> lock(p.mu);
    if (!p.sealed || p.poisoned || p.active || p.handoff_active || p.refreshing || (p.live && p.live->phase!=LivePhase::idle) || !valid_identity(old_layer,old_expert) ||
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
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
    if (!p.transfer(lock,p.pool+size_t(slot)*p.stride,staging,bytes,cudaMemcpyHostToDevice,staging_unsafe,error)) return false;
    replacement.slot=slot;std::copy_n(weight,3,replacement.weight);std::copy_n(input,3,replacement.input);
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
    p.weight_replacement_complete();
    return true;
}

bool NativeHelper::reserve_direct_spare(std::string& error) {
    auto& p=*p_;std::lock_guard<std::mutex> lock(p.mu);
    if(p.sealed || p.count || p.capacity<2 || p.poisoned) {
        error="direct spare must be reserved before filling a pool of at least two records";return false;
    }
    p.direct_spare_slot=p.capacity-1;return true;
}
int NativeHelper::direct_spare()const {
    std::lock_guard<std::mutex> lock(p_->mu);return p_->direct_spare_slot;
}
bool NativeHelper::foreground_busy()const {return p_->pass_busy.load(std::memory_order_acquire);}
bool NativeHelper::direct_begin(int ol,int oe,int nl,int ne,const void* incoming,void* outgoing,size_t bytes,
                                const float weight[3],const float input[3],float old_weight[3],float old_input[3],
                                BackgroundTraffic* budget,unsigned actor,bool download,bool verify,std::string& error,bool probe) {
    auto& p=*p_;std::unique_lock<std::mutex> lock(p.mu);
    const auto a=reinterpret_cast<uintptr_t>(incoming),b=reinterpret_cast<uintptr_t>(outgoing);
    const bool separate=!download || (a<b?b-a>=bytes:a-b>=bytes);
    if(!p.sealed || p.poisoned || p.active || p.handoff_active || p.refreshing || !valid_identity(ol,oe) ||
       !valid_identity(nl,ne) || !incoming || (download && !outgoing) || !separate || !weight || !input ||
       !old_weight || !old_input || bytes!=p.stride || !budget || !budget->healthy() || actor>1 ||
       (download && p.direct_spare_slot<0) || (p.live && (p.live->phase!=LivePhase::idle || p.live->stop))) {
        error="direct helper unavailable, active, or invalid lease";return false;
    }
    auto& old=p.records[size_t(ol)*kExperts+oe];auto& replacement=p.records[size_t(nl)*kExperts+ne];
    if(old.slot<0 || replacement.slot>=0 || (download && old.slot==p.direct_spare_slot)) {
        error="direct helper source absent or destination already occupied";return false;
    }
    for(int i=0;i<3;++i)if(!std::isfinite(weight[i]) || weight[i]<=0 || !std::isfinite(input[i]) || input[i]<=0 ||
                          !std::isfinite(1.0f/weight[i]) || 1.0f/weight[i]==0) {
        error="invalid direct helper scales";return false;
    }
    if(!p.live) {
        try {
            p.live=std::make_unique<Impl::Live>();
            p.live->cpu=std::thread([&p]{p.live_cpu_run();}); // idle in direct mode; no payload is queued
            p.live->gpu=std::thread([&p]{p.live_gpu_run();});
        } catch(const std::exception& e) {
            if(p.live) {p.live->stop=true;p.live->cv.notify_all();}
            error=std::string("start direct helper worker: ")+e.what();return false;
        }
    }
    if (probe && !download && !p.probe_destination) {
        DeviceGuard guard(p.device);
        if (cudaMalloc(&p.probe_destination, p.stride) != cudaSuccess) {
            error = "unable to allocate the measured copy probe's scratch expert";
            return false;
        }
    }
    auto& l=*p.live;l.probe=probe;l.direct=true;l.direct_download=download;l.direct_verify=verify;l.budget=budget;l.actor=actor;
    l.old_id=ol*kExperts+oe;l.new_id=nl*kExperts+ne;l.direct_old_slot=old.slot;
    l.slot=download?p.direct_spare_slot:old.slot;
    l.host=nullptr;l.incoming=const_cast<void*>(incoming);l.outgoing=outgoing;
    l.capture_context=capture::detailed()?capture::context():std::string{};
    std::copy_n(weight,3,l.weight);std::copy_n(input,3,l.input);
    std::copy_n(old.weight,3,old_weight);std::copy_n(old.input,3,old_input);
    l.phase=LivePhase::preparing;l.error.clear();l.cpu_done=true;l.gpu_done=false;
    l.cpu_busy=l.cpu_queued=false;l.gpu_busy=l.gpu_queued=true;l.gpu_failure_recorded=false;
    l.recycle=false;l.upload=true;l.started=std::chrono::steady_clock::now();
    if(!download && !probe) {old.slot=-1;p.records_version.fetch_add(1,std::memory_order_acq_rel);}
    l.cv.notify_all();return true;
}
bool NativeHelper::direct_publish(std::string& error) {
    auto& p=*p_;std::lock_guard<std::mutex> lock(p.mu);
    if(!p.live) {error="no direct helper lease";return false;}
    auto& l=*p.live;p.live_advance_locked();
    if(!l.direct || p.poisoned || p.active || p.handoff_active || p.refreshing || l.phase!=LivePhase::prepared ||
       p.records[l.new_id].slot>=0 || p.records[l.old_id].slot!=((l.direct_download || l.probe)?l.direct_old_slot:-1) ||
       (l.direct_download && p.direct_spare_slot!=l.slot)) {
        error="direct helper payload not retired or identity changed";return false;
    }
    if (l.probe) {
        l.phase = LivePhase::complete;
        return true;
    }
    p.records[l.old_id].slot=-1;
    auto& r=p.records[l.new_id];r.slot=l.slot;
    std::copy_n(l.weight,3,r.weight);std::copy_n(l.input,3,r.input);
    if(l.direct_download)p.direct_spare_slot=l.direct_old_slot;
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
    l.phase=LivePhase::complete;return true;
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
    l.direct=false;l.direct_download=false;l.direct_verify=false;l.budget=nullptr;
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
    if (p.poisoned || p.active || p.handoff_active || p.refreshing || l.phase!=LivePhase::prepared ||
        p.records[l.old_id].slot!=l.slot || p.records[l.new_id].slot>=0) {
        error="live helper not prepared, active, or ownership changed"; return false;
    }
    p.records[l.old_id].slot=-1;
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
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
    p.records_version.fetch_add(1,std::memory_order_acq_rel);
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
    if(l.phase==LivePhase::complete && !(l.direct && l.probe))p.weight_replacement_complete();
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
    return submit_rows(tokens, x, 0, jobs, error);
}
uint64_t NativeHelper::submit_packed(int tokens, const uint8_t* packed, const std::vector<Job>& jobs, std::string& error) {
    return submit_rows(tokens, packed, 1, jobs, error);
}
uint64_t NativeHelper::submit_expanded(int tokens, const void* rows, int format, const std::vector<Job>& jobs, std::string& error) {
    if (format != 2 && format != 3) { error = "expanded helper input format must be 2 (BF16 values) or 3 (byte codes)"; return 0; }
    return submit_rows(tokens, rows, format, jobs, error);
}
uint64_t NativeHelper::submit_rows(int tokens, const void* rows, int format, const std::vector<Job>& jobs, std::string& error) {
    const uint64_t submit_enter_ns = host_now_ns();
    auto& p = *p_; std::lock_guard<std::mutex> lock(p.mu);
    if (!p.worker_compute || !p.sealed || p.poisoned || p.active || p.refreshing || !rows || tokens < 1 || tokens > p.pass_tokens ||
        jobs.empty() || jobs.size() > size_t(4 * p.pass_tokens)) {
        // (which of them: the request of 2026-11-11 15:01 that failed here on the lab's path left no way to tell)
        error = "helper unavailable, request already active, or invalid request size (device " + std::to_string(p.device) +
                ": compute " + std::to_string(int(p.worker_compute)) + " sealed " + std::to_string(int(p.sealed)) + " poisoned " + std::to_string(int(p.poisoned)) +
                " active " + std::to_string(p.active) + " refreshing " + std::to_string(int(p.refreshing)) + " tokens " + std::to_string(tokens) +
                " of " + std::to_string(p.pass_tokens) + " jobs " + std::to_string(jobs.size()) + ")";
        return 0;
    }
    std::vector<int> per_token(size_t(tokens), 0);
    int total_columns = 0;
    const int layer = jobs.front().layer;
    for (const Job& j : jobs) {
        if (j.layer != layer || j.layer < 0 || j.layer >= kLayers || j.expert < 0 || j.expert >= kExperts || j.columns < 1 || j.columns > kMaxColumns ||
            p.records[size_t(j.layer) * kExperts + j.expert].slot < 0) { error = "invalid or unready helper job"; return 0; }
        total_columns += j.columns;
        if (total_columns > tokens * 4) { error = "helper column capacity exceeded"; return 0; }
        for (int c = 0; c < j.columns; ++c) {
            if (j.token[c] < 0 || j.token[c] >= tokens || !std::isfinite(j.weight[c]) || j.weight[c] < 0 || ++per_token[j.token[c]] > 4) {
                error = "invalid helper token/weight or more than four picks"; return 0;
            }
        }
    }
    const int selected_kernel = (total_columns == int(jobs.size()) ? p.kernel_one : p.kernel_multi).load(std::memory_order_relaxed);
    const bool selected_sm89 = p.use_sm89(p.sm89_mode.load(std::memory_order_relaxed));
    const int requested_execution = p.execution_mode.load(std::memory_order_relaxed);
    if (!Impl::execution_valid(requested_execution, selected_kernel)) {
        error = "helper execution flags are incompatible with the selected kernel family"; return 0;
    }
    // Snapshot the effective dispatch, so one-column Q1 policy1 reports0 and
    // reuses execution0's graph bank instead of capturing an identical graph.
    const int selected_execution = Impl::effective_execution(requested_execution, selected_kernel, int(jobs.size()), total_columns);
    if (!Impl::input_valid(selected_kernel, format)) {
        error = "helper input format " + std::to_string(format) + " is incompatible with kernel " + std::to_string(selected_kernel); return 0;
    }
    if (Impl::kernel_fp8(selected_kernel) && !selected_sm89) {
        error = "FP8 helper kernel requires the sm_89 build enabled on a compute-capability 8.9 device"; return 0;
    }
    const size_t row_bytes = Impl::input_bytes(1, format);
    const auto* input = static_cast<const uint8_t*>(rows);
    // Each token may name an expert at most once, including across grouped jobs (a bit per token and expert: a long
    // pass has up to 1,024 columns).
    std::vector<uint64_t> named(size_t(tokens) * 2, 0);   // two words (128 experts) a token
    // each token's row is checked once, at its first column, in one vectorized pass (mm3_helper_finite.hpp); the
    // checks run in the same order as before, so a request is refused with the same error (0 unchecked, 1 finite)
    std::vector<uint8_t> row_ok(size_t(tokens), 0);
    for (size_t i = 0; i < jobs.size(); ++i) for (int c = 0; c < jobs[i].columns; ++c) {
        uint64_t& w = named[size_t(jobs[i].token[c]) * 2 + size_t(jobs[i].expert) / 64];
        const uint64_t bit = uint64_t(1) << (jobs[i].expert % 64);
        if (w & bit) { error = "duplicate expert for helper token"; return 0; }
        w |= bit;
        uint8_t& ok = row_ok[size_t(jobs[i].token[c])];
        if (!ok) {
            const auto* row = input + size_t(jobs[i].token[c]) * row_bytes;
            const bool finite = (format == 0 || format == 2)
                ? mm3::helper_finite::row_finite(reinterpret_cast<const uint16_t*>(row), size_t(kHidden))
                : std::find(row + (format == 1 ? kHidden / 2 : kHidden), row + row_bytes, uint8_t(0x7f)) == row + row_bytes;
            if (!finite) {
                error = "nonfinite helper activation"; return 0;
            }
            ok = 1;
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
        for (int c = 0; c < j.columns; ++c) {
            std::memcpy(reinterpret_cast<uint8_t*>(p.h_x) + (offset / kHidden + size_t(c)) * row_bytes,
                        input + size_t(j.token[c]) * row_bytes, row_bytes);
        }
        offset += size_t(j.columns) * kHidden;
        ++i;
    }
    p.pending_columns = int(offset / kHidden);
    p.pending_format = format;
    p.pending_kernel = selected_kernel;
    p.pending_loads = p.kernel_loads.load(std::memory_order_relaxed);
    p.pending_execution = selected_execution;
    p.pending_sm89 = selected_sm89;
    if (++p.next == 0) ++p.next;
    p.active = p.next; p.pending_n = int(jobs.size()); p.pending_tokens = tokens; p.finished = false;
    p.finished_flag.store(false, std::memory_order_relaxed);
    p.collector_claimed = false;
    p.pass_timing = {};
    p.pending_stage_events=capture::light::full() && capture::light::g_helper_stages;
    p.pending_capture_context = capture::detailed() ? capture::context() : std::string{};
    p.pass_timing.submit_enter_ns = submit_enter_ns;
    p.pass_timing.enqueue_ns = host_now_ns();
    // exchange_schedule: no live x4 piece starts beside this pass; one already in flight is counted
    p.x4_until.store(0, std::memory_order_release);
    if (p.x4_piece_busy.load(std::memory_order_acquire)) {
        p.soverlaps.fetch_add(1, std::memory_order_relaxed);
        p.note_direct_foreground();
    }
    p.pass_busy.store(true, std::memory_order_release);
    p.queued = true;
    p.queued_flag.store(true, std::memory_order_release);
    p.cv.notify_one();
    p.pass_timing.submit_published_ns = host_now_ns();
    return p.active;
}

bool NativeHelper::done(uint64_t seq) const {
    auto& p = *p_;
    if (p.finished_flag.load(std::memory_order_acquire)) return true;
    std::lock_guard<std::mutex> lock(p.mu);
    return !seq || seq != p.active || p.finished || p.poisoned;
}
bool NativeHelper::collect(uint64_t seq, float* out, size_t n, std::string& error, PassTiming* timing,
                           std::vector<Refused>* refused) {
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
    // The pass's columns in today's merge order (jobs, then their columns); merge_pass zero-fills and adds them.
    p.merge_cols.clear();
    for (int i = 0; i < p.pending_n; ++i) {
        const Job& j = p.pending_jobs[i];
        for (int c = 0; c < j.columns; ++c)
            p.merge_cols.push_back({j.token[c], j.weight[c], p.h_y + p.pending_offsets[i] + size_t(c) * kHidden});
    }
    if (refused) { refused->clear(); p.inject_nonfinite_once(); }
    const auto first = helper_merge::merge_pass(out, p.pending_tokens, size_t(kHidden), p.merge_cols.data(), p.merge_cols.size(),
                                                refused ? &p.merge_refused : nullptr);
    if (first.flags) {   // (no refused list: the helper poisoned, as before; the message keeps its first words)
        const auto [i, c] = p.column_at(first.index);
        p.failure = std::string(first.flags & helper_merge::kNonfiniteInput ? "nonfinite helper output" : "nonfinite helper weighted sum") +
                    p.nonfinite_detail(i, c);
        p.poisoned = true; error = p.failure; return false;
    }
    if (refused) for (const auto& f : p.merge_refused) {   // (left out of the output: the caller recomputes them)
        const auto [i, c] = p.column_at(f.index);
        const Job& j = p.pending_jobs[size_t(i)];
        Refused r;
        r.layer = j.layer; r.expert = j.expert; r.slot = p.records[size_t(j.layer) * kExperts + j.expert].slot;
        r.token = j.token[c]; r.weight = j.weight[c]; r.sum = !(f.flags & helper_merge::kNonfiniteInput);
        r.detail = std::string(r.sum ? "nonfinite helper weighted sum" : "nonfinite helper output") + p.nonfinite_detail(i, c, &r.bad);
        refused->push_back(std::move(r));
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

void NativeHelper::set_sm89(bool on) { p_->sm89_mode.store(on); }
bool NativeHelper::sm89() const { return p_->sm89_mode.load() && p_->use_sm89(true); }
bool NativeHelper::sm89_capable() const { return p_->use_sm89(true); }
uint64_t NativeHelper::records_version() const { return p_->records_version.load(std::memory_order_acquire); }
void NativeHelper::handoff_busy(bool busy) {
    auto& p = *p_;
    {
        std::lock_guard<std::mutex> lock(p.mu);
        if (busy && p.execution_mode.load()) throw std::logic_error("helper execution variants are unavailable in queued handoff");
        p.handoff_active = busy;
    }
    if (busy) {   // (exchange_schedule: no live x4 piece starts beside a released pass; one already in flight is counted)
        p.x4_until.store(0, std::memory_order_release);
        if (p.x4_piece_busy.load(std::memory_order_acquire)) {
            p.soverlaps.fetch_add(1, std::memory_order_relaxed);
            p.note_direct_foreground();
        }
    }
    p.pass_busy.store(busy, std::memory_order_release);
}
bool NativeHelper::merge_external(int tokens, const std::vector<Job>& jobs, uint16_t* columns, float* out, size_t n,
                                  std::string& error, std::vector<Refused>* refused) {
    auto& p = *p_;
    if (!columns || !out || tokens < 1 || tokens > p.pass_tokens || n < size_t(tokens) * kHidden || jobs.size() > size_t(4 * p.pass_tokens)) {
        error = "invalid helper hand-off merge"; return false;
    }
    std::vector<helper_merge::PassColumn> cols;
    std::vector<size_t> offsets(jobs.size());
    std::vector<std::pair<int, int>> at;   // column index -> job, its column
    size_t offset = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        offsets[i] = offset;
        for (int c = 0; c < jobs[i].columns; ++c) {
            if (jobs[i].token[c] < 0 || jobs[i].token[c] >= tokens) { error = "invalid helper hand-off column"; return false; }
            cols.push_back({jobs[i].token[c], jobs[i].weight[c], columns + offset + size_t(c) * kHidden});
            at.push_back({int(i), c});
        }
        offset += size_t(jobs[i].columns) * kHidden;
    }
    if (refused) { refused->clear(); Impl::inject_nonfinite_into(jobs.data(), int(jobs.size()), columns, offsets.data()); }
    std::vector<helper_merge::Flagged> flagged;
    const auto first = helper_merge::merge_pass(out, tokens, size_t(kHidden), cols.data(), cols.size(), refused ? &flagged : nullptr);
    const auto describe = [&](size_t index, int* bad) {
        const auto [i, c] = at[index];
        const Job& j = jobs[size_t(i)];
        const uint16_t* y = cols[index].src;
        int count = 0, first_bad = -1;
        for (int k = 0; k < kHidden; ++k)
            if ((y[k] & 0x7f80u) == 0x7f80u) { if (first_bad < 0) first_bad = k; ++count; }
        if (bad) *bad = count;
        int slot = -1;
        { std::lock_guard<std::mutex> lock(p.mu); slot = p.records[size_t(j.layer) * kExperts + j.expert].slot; }
        char text[320];
        std::snprintf(text, sizeof text, " (device %d, hand-off pass: layer %d expert %d slot %d column %d of %d token %d weight %g;"
                      " columns nonfinite %d of %d, first [%d])", p.device, j.layer, j.expert, slot, c, j.columns, j.token[c],
                      double(j.weight[c]), count, int(kHidden), first_bad);
        return std::string(text);
    };
    if (first.flags) {
        error = std::string(first.flags & helper_merge::kNonfiniteInput ? "nonfinite helper output" : "nonfinite helper weighted sum") +
                describe(first.index, nullptr);
        return false;
    }
    if (refused) for (const auto& f : flagged) {
        const auto [i, c] = at[f.index];
        const Job& j = jobs[size_t(i)];
        Refused r;
        r.layer = j.layer; r.expert = j.expert;
        { std::lock_guard<std::mutex> lock(p.mu); r.slot = p.records[size_t(j.layer) * kExperts + j.expert].slot; }
        r.token = j.token[c]; r.weight = j.weight[c]; r.sum = !(f.flags & helper_merge::kNonfiniteInput);
        r.detail = std::string(r.sum ? "nonfinite helper weighted sum" : "nonfinite helper output") + describe(f.index, &r.bad);
        refused->push_back(std::move(r));
    }
    return true;
}
}
