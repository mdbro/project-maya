// The helper hand-off (kernel item 1, helper_handoff; off unless a request asks): each helper card's work for a whole
// step queued ahead on the card and released by a word in mapped memory.
//
// The user, 2026-11-10: "I do agree that it would be a good idea in general for the 3090 to queue each layer's work ahead
// and release it with a word in shared memory"; chose "Yes, build it first" (2026-11-12). Design notes and the frozen
// prediction: /mnt/wsl/transfer/minimax-wsl/runs/kernel1-handoff-20261114/.
//
// Today a helper card's pass starts after the 5090's preparation has ended, the host has joined it, planned the pass,
// converted the rows to BF16 and submitted; a worker thread then launches the card's graph and polls its event, and the
// collector merges. Here, at the forward's first MoE layer the host launches one graph a card for the whole step: for
// every MoE layer in order, a take kernel that polls that layer's request word, the card's expert kernels, and a result
// kernel that writes the columns back and publishes a done word. The 5090 releases each layer itself: a request kernel on
// its router branch, right after the routing, builds the card's jobs from the picks (the host plan's grouping and order,
// plan_jobs below), writes them and the BF16 rows into the request slot, then the word. The host still joins, plans the
// 5090's and the CPU lane's share, and merges the card's columns (helper_merge, as NativeHelper::collect does), so the
// arithmetic and every sum's order are today's; only who starts the card's work, and when, changes.
//
// Per card: kSlots request and result slots (pinned, mapped on every card), its own device buffers and stream (the
// NativeHelper's worker keeps its own for the prompt reader, the transfers and the check of helper_handoff=2), and one
// graph per (layer list, one row or windows, kernels).
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <cuda_runtime_api.h>

#if defined(__CUDACC__)
#define MM3_HD __host__ __device__
#else
#define MM3_HD
#endif

namespace mm3::handoff {
constexpr int kSlots = 4, kMaxRows = 16, kMaxJobs = 64, kMaxColumns = 8, kHidden = 6144, kExperts = 128, kLayers = 60;
constexpr int kTakeBlocks = 16, kResultBlocks = 16, kThreads = 256;

// One expert's columns on a card, as the host plan builds a NativeHelper::Job, with the record named.
struct Job {
    unsigned long long record = 0;   // the record's address on the card (pool + slot * stride)
    float multiplier[3] = {};        // 1 / weight_global, gate / up / down (NativeHelper::queue_snapshot)
    int expert = 0, columns = 0;
    int token[kMaxColumns] = {};
    float weight[kMaxColumns] = {};
};
// A layer's table entry for one expert on one card (pinned; the request kernel reads the layer's 128).
struct Entry { unsigned long long record = 0; float multiplier[3] = {}; int ready = 0; };
enum Status : int { kServed = 0, kBadRequest = 1, kAborted = 2, kBadRecord = 3, kExpired = 4, kNonfiniteRow = 5 };

// A request slot: written by the 5090 (jobs, rows, header, then the word), read by the card's take kernel and, in
// helper_handoff=2, by the host.
struct alignas(256) Request {
    unsigned long long word = 0;   // last: (step << 8) | m
    int count = 0, columns = 0, rows = 0, kind = 0, status = 0, layer = -1;   // kind 0: every job one column
    Job jobs[kMaxJobs];
    alignas(16) uint16_t x[kMaxRows][kHidden];   // the pass's rows, BF16 (the host's rounding), whatever the card's jobs
};
// A result slot: written by the card's result kernel (columns, header, then the word), read by the host.
struct alignas(256) Result {
    unsigned long long word = 0;   // last: the request's word
    int status = 0, count = 0, columns = 0, pad = 0;
    unsigned long long t_seen = 0, t_done = 0;   // the card's %globaltimer: request seen, columns written
    alignas(16) uint16_t y[kMaxJobs][kHidden];  // the jobs' columns in order (jobs, then their columns)
};
// The per-pass snapshot the host writes before the 5090's preparation (pinned; one, like the ready mask): each card's
// word for this pass and which experts each card serves.
struct Snapshot {
    unsigned long long word[2] = {};
    uint32_t helper[4] = {}, third[4] = {};   // card 0 (the 3090): ready there; card 1 (the 4080): held, not elsewhere
};
inline MM3_HD bool bit(const uint32_t* m, int e) { return (m[e >> 5] >> (e & 31)) & 1u; }

// The jobs of one card from a pass's picks, in the host plan's order (window_post: token by token, pick by pick; a pick
// joins the first job of its expert with fewer than kMaxColumns columns, else opens one), then in NativeHelper::submit's
// order (stable by width, 1 to 8): the order the host merges in. take(t, i, e): the pick goes to this card. per_token4:
// at most four picks a token (the 4080's rule in window_post; the 3090's picks are at most k). Weights are mixture *
// scale, as window_post computes them. Returns the job count, -1 past kMaxJobs. scratch: kMaxJobs jobs.
template <class Take>
MM3_HD inline int plan_jobs(const int* top, const float* w, int T, int k, float scale, bool per_token4, Take take,
                            Job* scratch, Job* out) {
    int n = 0;
    int per_token[kMaxRows] = {};
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < k; ++i) {
            const int e = top[t * k + i];
            if (!take(t, i, e)) continue;
            if (per_token4) {
                if (per_token[t] >= 4) continue;
                ++per_token[t];
            }
            int j = 0;
            while (j < n && !(scratch[j].expert == e && scratch[j].columns < kMaxColumns)) ++j;
            if (j == n) {
                if (n == kMaxJobs) return -1;
                scratch[n] = Job{};
                scratch[n].expert = e;
                ++n;
            }
            const int c = scratch[j].columns++;
            scratch[j].token[c] = t;
            scratch[j].weight[c] = w[t * k + i] * scale;
        }
    int o = 0;
    for (int cols = 1; cols <= kMaxColumns; ++cols)
        for (int j = 0; j < n; ++j)
            if (scratch[j].columns == cols) out[o++] = scratch[j];
    return n;
}

// The 5090's request kernel (on the router branch, after the routing): both cards' jobs and rows, then their words.
struct RequestArgs {
    const int* top = nullptr;          // [T, k] the routing's picks (device)
    const float* w = nullptr;          // [T, k] their mixture weights (device)
    const float* xn = nullptr;         // [T, kHidden] the normed rows (device, F32)
    int T = 1, k = 4, layer = 0;
    float scale = 1.0f;                // routed_scaling_factor
    const Snapshot* snapshot = nullptr;           // device view of the pinned snapshot
    const Entry* table[2] = {nullptr, nullptr};   // device views: this layer's 128 entries per card (null: card off)
    Request* request[2] = {nullptr, nullptr};     // device views: this pass's slot per card (null: card off)
};
bool launch_request(const RequestArgs& a, cudaStream_t stream);

// One helper card's queue.
class Lane {
public:
    // Tensor-core warps (1, 2, 4, 8) for passes of one-column jobs and for the others, and the loads a batch (1, 2, 4,
    // 8): NativeHelper::set_kernel's tensor-core values (100 added or not) and set_loads.
    struct Kernels { int one = 4, multi = 8, loads = 8; };
    // device: the card; pool, stride, capacity: its NativeHelper's records (a request naming anything else is refused
    // on the card); tables: pinned [kLayers][kExperts] this lane fills (Lane allocates it).
    Lane(int device, uintptr_t pool, size_t stride, int capacity);
    ~Lane();
    Lane(const Lane&) = delete;
    Lane& operator=(const Lane&) = delete;
    // Launches the next step's graph: the MoE layers of one forward in order (at most 255), for one-row forwards or for
    // windows. Builds it on first use. The previous step must have been ended (end_step).
    bool arm(const std::vector<int>& layers, bool windows, Kernels k, std::string& err);
    bool armed() const { return armed_; }
    unsigned long long word(int m) const { return (step_ << 8) | unsigned(m & 255); }
    Request* request(int m) { return requests_ + (m % kSlots); }
    Result* result(int m) { return results_ + (m % kSlots); }
    Entry* table(int layer) { return tables_ + size_t(layer) * kExperts; }
    // Spins until m's result is published or the deadline passes (then false and err). status: the result's.
    bool wait(int m, double deadline_s, int& status, uint64_t& seen_ns, uint64_t& wait_ns, std::string& err);
    // The rest of the armed step skipped on the card (its abort word); then a polling kernel holds the card's clocks until
    // the next arm, if occupancy is on (the spinner's job between steps). Idempotent.
    void end_step(bool occupancy);
    int device() const { return device_; }
    uint64_t steps() const { return step_; }
    uint64_t captures() const { return captures_; }
    double capture_ms() const { return capture_ms_; }
    // (helper_handoff=2) the lane's device result columns read back, for a mismatch report
    bool read_back(int columns, std::vector<uint16_t>& out, std::string& err);

private:
    struct State;
    int device_ = -1, sms_ = 0, capacity_ = 0;
    uintptr_t pool_ = 0;
    size_t stride_ = 0;
    cudaStream_t stream_ = nullptr;
    Request* requests_ = nullptr;   // pinned, mapped (portable): kSlots
    Result* results_ = nullptr;     // pinned, mapped (portable): kSlots
    Entry* tables_ = nullptr;       // pinned, mapped (portable): kLayers * kExperts
    unsigned long long* control_ = nullptr;   // pinned: [0] the abort word (a step number), [1] the occupancy word
    State* state_ = nullptr;        // device
    uint16_t *d_rows_ = nullptr, *d_x_ = nullptr, *d_y_ = nullptr;
    void* d_jobs_ = nullptr;        // FastJob[kMaxJobs]
    uint8_t* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    unsigned long long step_ = 0;
    bool armed_ = false, occupancy_armed_ = false, broken_ = false;
    uint64_t captures_ = 0;
    double capture_ms_ = 0;
    std::map<std::tuple<std::vector<int>, bool, int, int, int>, cudaGraphExec_t> graphs_;
    cudaGraphExec_t build(const std::vector<int>& layers, bool windows, Kernels k, std::string& err);
    void release() noexcept;
};
}  // namespace mm3::handoff
#undef MM3_HD
