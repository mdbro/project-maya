// Standalone bench of kernel item 1, the helper hand-off (src/mm3/mm3_helper_handoff.hpp), on today's cards: the 5090 and
// one helper card (the 3090, or the 4080), with the engine's own code - NativeHelper's worker path with serving's settings
// (CUDA graphs, zero-copy copies, the tensor-core kernel with serving's warps and loads) against the hand-off's lane and
// the 5090's request kernel. Not the engine: no attention, no CPU lane, no other card; the 5090's preparation is a
// routing stand-in, a wait for the shared expert's remainder and the readback.
//
// Each pass, on the 5090: a kernel marks the routing's end in mapped memory (the common origin, seen by a spinning host
// thread), then on the main stream a wait of --shared-us (the shared expert's remainder after the routing) and the
// readback of the picks, weights and rows to pinned memory; in the hand-off arm the request kernel runs on a side stream
// right after the routing, as on the router branch.
//   today:    the host waits for the 5090 (the join), plans the card's jobs from the readback (window_post's rule),
//             converts the rows, submits, and collects (NativeHelper::submit / collect): merged rows on the host.
//   hand-off: one step graph a --passes passes armed on the card; the host waits for the pass's word and merges
//             (NativeHelper::merge_external): merged rows on the host. It joins the 5090 as well (not on this path).
// Times are host steady-clock ms from the routing's end as the host saw it; the card's own span (request seen to columns
// written) from its globaltimer. Every pass of the first step of each case is checked: the hand-off's merged rows against
// today's for the same inputs, bit for bit, and the card's jobs as the 5090 wrote them against the host's plan.
//
//   build:  cmake --build build-local --target mm3_helper_handoff_bench -j 15     (not while a timed request runs)
//   run:    build-local/mm3_helper_handoff_bench --card 3090 [--steps 20] [--passes 57] [--shared-us 38] [--load]
//           build-local/mm3_helper_handoff_bench --card 4080 ...
//           (cards are found by name; nothing else may use them; --load adds a 5090 upload stream at full speed)
#include "mm3_native_helper.hpp"
#include "mm3_helper_handoff.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace ho = mm3::handoff;
using H = mm3::NativeHelper;
constexpr int kHidden = H::kHidden, kK = 4, kExperts = 8, kLayer = 3;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__global__ void mark_kernel(unsigned* flag, unsigned value) { __threadfence_system(); *(volatile unsigned*) flag = value; __threadfence_system(); }
__global__ void wait_kernel(unsigned long long ns) { const unsigned long long t0 = gtime(); while (gtime() - t0 < ns) {} }
__global__ void readback_kernel(const int* top, const float* w, const float* xn, int* rtop, float* rw, float* rx, int T) {
    const int n = T * kK, m = T * kHidden;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) {
        rx[i] = xn[i];
        if (i < n) { rtop[i] = top[i]; rw[i] = w[i]; }
    }
}

struct Case { const char* name; int T; int per_row; int pool; };   // per_row picks a row on the card, out of pool experts
struct Stat { std::vector<double> v; void add(double x) { v.push_back(x); }
    double q(double p) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[size_t(std::min<double>(double(v.size() - 1), p * double(v.size() - 1) + 0.5))]; } };

// window_post's plan of the card's picks (then submit orders them by width itself)
std::vector<H::Job> host_plan(const int* top, const float* w, int T, float scale, const uint32_t* mask) {
    std::vector<H::Job> jobs;
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < kK; ++i) {
            const int x = top[t * kK + i];
            if (!ho::bit(mask, x)) continue;
            auto it = std::find_if(jobs.begin(), jobs.end(), [&](const H::Job& j) { return j.expert == x && j.columns < H::kMaxColumns; });
            if (it == jobs.end()) { H::Job j; j.layer = kLayer; j.expert = x; jobs.push_back(j); it = jobs.end() - 1; }
            const int c = it->columns++; it->token[c] = t; it->weight[c] = w[t * kK + i] * scale;
        }
    return jobs;
}
std::vector<H::Job> by_width(const std::vector<H::Job>& jobs) {
    std::vector<H::Job> out;
    for (int cols = 1; cols <= H::kMaxColumns; ++cols) for (const auto& j : jobs) if (j.columns == cols) out.push_back(j);
    return out;
}
int find_card(const char* name) {
    int n = 0;
    ck(cudaGetDeviceCount(&n), "device count");
    for (int i = 0; i < n; ++i) { cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, i), "properties"); if (std::strstr(p.name, name)) return i; }
    throw std::runtime_error(std::string("no ") + name);
}
}  // namespace

int main(int argc, char** argv) {
    std::string card = "3090";
    int steps = 20, passes = 57;
    double shared_us = 38;
    bool load = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--card") card = next();
        else if (a == "--steps") steps = std::stoi(next());
        else if (a == "--passes") passes = std::stoi(next());
        else if (a == "--shared-us") shared_us = std::stod(next());
        else if (a == "--load") load = true;
        else { std::fprintf(stderr, "usage: %s --card 3090|4080 [--steps N] [--passes N] [--shared-us US] [--load]\n", argv[0]); return 2; }
    }
    try {
        const bool third = card == "4080";
        const int lane_index = third ? 1 : 0;
        const int primary = find_card("RTX 5090"), dev = find_card(third ? "RTX 4080" : "RTX 3090");
        cudaDeviceProp cp{};
        ck(cudaGetDeviceProperties(&cp, dev), "card properties");
        std::printf("helper_handoff_bench: 5090 = CUDA%d, card = CUDA%d (%s), %d steps of %d passes, shared expert remainder %.0f us%s\n",
                    primary, dev, cp.name, steps, passes, shared_us, load ? ", 5090 uploading" : "");
        // the card's records and its helper, with serving's settings for that card (mm3_sample.hpp)
        H helper(dev, kExperts);
        const size_t stride = helper.record_stride();
        std::mt19937 rng(20261114);
        uint8_t* staging = nullptr;
        ck(cudaSetDevice(dev), "set card");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&staging), stride, cudaHostAllocPortable), "staging");
        std::string err;
        for (int e = 0; e < kExperts; ++e) {
            for (size_t b = 0; b < stride; ++b) staging[b] = uint8_t(rng());
            for (size_t off = 0; off + 144 <= stride; off += 144) for (int r = 0; r < 16; ++r) staging[off + 128 + r] = uint8_t(0x28 + rng() % 24);
            const float wg[3] = {2.0f + 0.1f * float(e), 2.5f, 3.0f}, ig[3] = {1.0f, 1.0f, 1.0f};
            if (!helper.add_expert(kLayer, e, staging, stride, wg, ig, err)) throw std::runtime_error("add_expert: " + err);
        }
        if (!helper.seal(err)) throw std::runtime_error("seal: " + err);
        helper.set_fast(true); helper.set_zero_copy(1); helper.set_occupancy(true);
        if (third) { helper.set_kernel(108, 8); helper.set_loads(4); } else { helper.set_kernel(108, 4); helper.set_loads(8); }
        ho::Lane lane(dev, helper.pool_identity(), stride, kExperts);
        std::array<H::QueueIdentity, 128> ids{};
        if (!helper.queue_snapshot(kLayer, ids, err)) throw std::runtime_error("queue_snapshot: " + err);
        uint32_t mask[4] = {};
        for (int e = 0; e < 128; ++e) {
            ho::Entry en;
            en.ready = ids[size_t(e)].slot >= 0;
            en.record = en.ready ? helper.pool_identity() + size_t(ids[size_t(e)].slot) * stride : 0;
            for (int q = 0; q < 3; ++q) en.multiplier[q] = ids[size_t(e)].multiplier[q];
            lane.table(kLayer)[e] = en;
            if (en.ready) mask[e >> 5] |= 1u << (e & 31);
        }
        const ho::Lane::Kernels kern = third ? ho::Lane::Kernels{8, 8, 4} : ho::Lane::Kernels{4, 8, 8};
        // the 5090's buffers
        ck(cudaSetDevice(primary), "set 5090");
        cudaStream_t s_main, s_side, s_load;
        ck(cudaStreamCreateWithFlags(&s_main, cudaStreamNonBlocking), "stream");
        ck(cudaStreamCreateWithFlags(&s_side, cudaStreamNonBlocking), "stream");
        ck(cudaStreamCreateWithFlags(&s_load, cudaStreamNonBlocking), "stream");
        cudaEvent_t fork, join;
        ck(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "event");
        ck(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "event");
        int *top = nullptr, *rtop = nullptr; float *w = nullptr, *xn = nullptr, *rw = nullptr, *rx = nullptr;
        unsigned* flag = nullptr; ho::Snapshot* snap = nullptr;
        ck(cudaMalloc(&top, ho::kMaxRows * kK * sizeof(int)), "top");
        ck(cudaMalloc(&w, ho::kMaxRows * kK * sizeof(float)), "w");
        ck(cudaMalloc(&xn, size_t(ho::kMaxRows) * kHidden * sizeof(float)), "xn");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&rtop), ho::kMaxRows * kK * sizeof(int), cudaHostAllocPortable | cudaHostAllocMapped), "rtop");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&rw), ho::kMaxRows * kK * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "rw");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&rx), size_t(ho::kMaxRows) * kHidden * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "rx");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&flag), 64, cudaHostAllocPortable | cudaHostAllocMapped), "flag");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&snap), sizeof(ho::Snapshot), cudaHostAllocPortable | cudaHostAllocMapped), "snapshot");
        *flag = 0;
        // --load: the 5090 uploads 64 MiB pieces back to back while the bench runs
        std::atomic<bool> stop{false};
        std::thread loader;
        void *up_src = nullptr, *up_dst = nullptr;
        if (load) {
            ck(cudaHostAlloc(&up_src, size_t(64) << 20, cudaHostAllocPortable), "upload source");
            ck(cudaMalloc(&up_dst, size_t(64) << 20), "upload destination");
            loader = std::thread([&] {
                cudaSetDevice(primary);
                while (!stop.load()) { cudaMemcpyAsync(up_dst, up_src, size_t(64) << 20, cudaMemcpyHostToDevice, s_load); cudaStreamSynchronize(s_load); }
            });
        }
        const float scale = 2.0f;
        const Case cases[] = {{"one row, 1 job", 1, 1, 8}, {"one row, 2 jobs", 1, 2, 8}, {"one row, 3 jobs", 1, 3, 8}, {"one row, 4 jobs", 1, 4, 8},
                              {"4 rows, 2 a row of 4", 4, 2, 4}, {"8 rows, 2 a row of 6", 8, 2, 6}};
        std::printf("\n%-24s %-9s %10s %10s %10s %10s %10s %8s %8s\n", "case (picks on the card)", "jobs/cols", "today", "hand-off",
                    "saved", "word seen", "card span", "exact", "jobs ok");
        std::printf("%-24s %-9s %10s %10s %10s %10s %10s %8s %8s\n", "", "", "merged ms", "merged ms", "ms", "ms", "ms", "", "");
        unsigned seq = 0;
        std::vector<int> htop(ho::kMaxRows * kK);
        std::vector<float> hw(ho::kMaxRows * kK), hx(size_t(ho::kMaxRows) * kHidden);
        std::vector<uint16_t> bf(size_t(ho::kMaxRows) * kHidden);
        for (const Case& c : cases) {
            std::mt19937 in_rng(7 + c.T * 31 + c.per_row);
            // the passes' inputs (shared by the two arms): picks, weights, rows
            struct Pass { std::vector<int> top; std::vector<float> w, x; };
            std::vector<Pass> inputs(static_cast<size_t>(passes));
            for (auto& p : inputs) {
                p.top.resize(size_t(c.T * kK)); p.w.resize(size_t(c.T * kK)); p.x.resize(size_t(c.T) * kHidden);
                for (int t = 0; t < c.T; ++t) {
                    std::vector<int> row;
                    while (int(row.size()) < c.per_row) { const int e = int(in_rng() % unsigned(c.pool)); if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
                    while (int(row.size()) < kK) row.push_back(100 + int(row.size()) + 4 * t % 20);   // (picks elsewhere: not the card's)
                    std::shuffle(row.begin(), row.end(), in_rng);
                    for (int i = 0; i < kK; ++i) { p.top[size_t(t * kK + i)] = row[size_t(i)]; p.w[size_t(t * kK + i)] = std::uniform_real_distribution<float>(0.05f, 0.5f)(in_rng); }
                }
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& v : p.x) v = nd(in_rng);
            }
            const auto upload = [&](const Pass& p) {
                ck(cudaMemcpyAsync(top, p.top.data(), p.top.size() * sizeof(int), cudaMemcpyHostToDevice, s_main), "top up");
                ck(cudaMemcpyAsync(w, p.w.data(), p.w.size() * sizeof(float), cudaMemcpyHostToDevice, s_main), "w up");
                ck(cudaMemcpyAsync(xn, p.x.data(), p.x.size() * sizeof(float), cudaMemcpyHostToDevice, s_main), "x up");
                ck(cudaStreamSynchronize(s_main), "inputs");
            };
            const auto prepare = [&](bool request, int m) {   // the 5090's part of a pass; returns the mark's value
                const unsigned value = ++seq;
                mark_kernel<<<1, 1, 0, s_main>>>(flag, value);
                ck(cudaEventRecord(fork, s_main), "fork");
                if (request) {
                    ck(cudaStreamWaitEvent(s_side, fork, 0), "side");
                    ho::RequestArgs a;
                    a.top = top; a.w = w; a.xn = xn; a.T = c.T; a.k = kK; a.layer = kLayer; a.scale = scale; a.snapshot = snap;
                    a.table[lane_index] = lane.table(kLayer); a.request[lane_index] = lane.request(m);
                    if (!ho::launch_request(a, s_side)) throw std::runtime_error("launch_request");
                    ck(cudaEventRecord(join, s_side), "join");
                }
                wait_kernel<<<1, 1, 0, s_main>>>((unsigned long long) (shared_us * 1000));
                if (request) ck(cudaStreamWaitEvent(s_main, join, 0), "join wait");
                readback_kernel<<<16, 256, 0, s_main>>>(top, w, xn, rtop, rw, rx, c.T);
                ck(cudaGetLastError(), "5090 launches");
                (void) cudaStreamQuery(s_main);
                return value;
            };
            const auto seen = [&](unsigned value) { while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) != value) {} return now_ms(); };
            // ---- today: join, plan, convert, submit, collect
            Stat today, today_join;
            std::vector<std::vector<float>> reference;
            int njobs = 0, ncols = 0;
            for (int s = 0; s < steps; ++s)
                for (int m = 0; m < passes; ++m) {
                    const Pass& p = inputs[size_t(m)];
                    upload(p);
                    const unsigned v = prepare(false, m);
                    const double t0 = seen(v);
                    ck(cudaStreamSynchronize(s_main), "join");
                    const double tj = now_ms();
                    const auto jobs = host_plan(rtop, rw, c.T, scale, mask);
                    for (size_t i = 0; i < size_t(c.T) * kHidden; ++i) { uint32_t u; std::memcpy(&u, rx + i, 4); bf[i] = uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
                    std::vector<float> merged(size_t(c.T) * kHidden);
                    const uint64_t q = helper.submit(c.T, bf.data(), jobs, err);
                    if (!q || !helper.collect(q, merged.data(), merged.size(), err)) throw std::runtime_error("today's path: " + err);
                    const double t1 = now_ms();
                    if (s > 0) { today.add(t1 - t0); today_join.add(tj - t0); }
                    if (s == 0) { reference.push_back(merged); njobs = int(jobs.size()); ncols = 0; for (auto& j : jobs) ncols += j.columns; }
                }
            // ---- the hand-off: a step graph a step, the 5090 releases each pass
            Stat handoff, word, span, ho_join;
            int exact = 0, checked = 0, jobs_ok = 0;
            double arm_ms = 0;
            std::vector<int> layers(static_cast<size_t>(passes));
            for (int m = 0; m < passes; ++m) layers[size_t(m)] = m;
            for (int s = 0; s < steps; ++s) {
                const double ta = now_ms();
                if (!lane.arm(layers, c.T > 1, kern, err)) throw std::runtime_error("arm: " + err);
                if (s > 0) arm_ms += now_ms() - ta;
                for (int m = 0; m < passes; ++m) {
                    const Pass& p = inputs[size_t(m)];
                    upload(p);
                    *snap = ho::Snapshot{};
                    snap->word[lane_index] = lane.word(m);
                    for (int q = 0; q < 4; ++q) (lane_index == 0 ? snap->helper : snap->third)[q] = mask[q];
                    helper.handoff_busy(true);
                    const unsigned v = prepare(true, m);
                    const double t0 = seen(v);
                    int status = 0; uint64_t seen_ns = 0, wait_ns = 0;
                    if (!lane.wait(m, 2.0, status, seen_ns, wait_ns, err)) throw std::runtime_error(err);
                    const double tw = now_ms();
                    helper.handoff_busy(false);
                    if (status != ho::kServed) throw std::runtime_error("hand-off status " + std::to_string(status));
                    const auto plan = host_plan(p.top.data(), p.w.data(), c.T, scale, mask);
                    const auto ordered = by_width(plan);
                    std::vector<float> merged(size_t(c.T) * kHidden);
                    std::vector<H::Refused> refused;
                    if (!helper.merge_external(c.T, ordered, &lane.result(m)->y[0][0], merged.data(), merged.size(), err, &refused))
                        throw std::runtime_error("merge: " + err);
                    const double t1 = now_ms();
                    ck(cudaStreamSynchronize(s_main), "join");
                    const double tj = now_ms();
                    const ho::Result* r = lane.result(m);
                    if (s > 0) { handoff.add(t1 - t0); word.add(tw - t0); span.add(double(r->t_done - r->t_seen) * 1e-6); ho_join.add(tj - t0); }
                    if (s == 0) {
                        ++checked;
                        exact += std::memcmp(merged.data(), reference[size_t(m)].data(), merged.size() * sizeof(float)) == 0 && refused.empty();
                        const ho::Request* rq = lane.request(m);
                        bool same = rq->count == int(ordered.size());
                        for (size_t j = 0; same && j < ordered.size(); ++j) {
                            const ho::Job& dj = rq->jobs[j];
                            same = dj.expert == ordered[j].expert && dj.columns == ordered[j].columns &&
                                   dj.record == helper.pool_identity() + size_t(ids[size_t(dj.expert)].slot) * stride;
                            for (int q = 0; same && q < dj.columns; ++q)
                                same = dj.token[q] == ordered[j].token[q] && std::memcmp(&dj.weight[q], &ordered[j].weight[q], 4) == 0;
                        }
                        jobs_ok += same;
                    }
                }
                lane.end_step(true);
            }
            std::printf("%-24s %3d/%-5d %10.4f %10.4f %10.4f %10.4f %10.4f %4d/%-3d %4d/%-3d\n", c.name, njobs, ncols, today.q(0.5),
                        handoff.q(0.5), today.q(0.5) - handoff.q(0.5), word.q(0.5), span.q(0.5), exact, checked, jobs_ok, checked);
            std::printf("%-24s %-9s p10-p90: today %.4f-%.4f, hand-off %.4f-%.4f; join today %.4f, hand-off %.4f; arm %.3f ms a step\n", "", "",
                        today.q(0.1), today.q(0.9), handoff.q(0.1), handoff.q(0.9), today_join.q(0.5), ho_join.q(0.5),
                        steps > 1 ? arm_ms / double(steps - 1) : 0.0);
        }
        stop = true;
        if (loader.joinable()) loader.join();
        std::printf("\nlane: %llu steps, %llu step graphs built (%.1f ms)\n", (unsigned long long) lane.steps(),
                    (unsigned long long) lane.captures(), lane.capture_ms());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helper_handoff_bench: %s\n", e.what());
        return 1;
    }
}
