// tests/mm3_native_expert_fast_test.cu - kernel item 4's standalone check and timing: the 5090's window expert phase
// (native_experts::window, the FP4 path the drafting windows run) under expert_fast 0, 1 and 2 (mm3_native_experts.cuh):
// every mode's output must equal mode 0's bit for bit, and each mode is timed as a CUDA graph (the expert graph replays
// it that way), by rows and distinct experts. Synthetic records: random E2M1 codes with safe E4M3 scales, unit global
// scales (the arithmetic is the kernels'; this checks the launches' mapping, not an oracle).
// Usage: mm3_native_expert_fast_test [device]   (an idle RTX 5090; about 0.7 GB). Exit 0: bit-identical; 1: a mismatch.
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/record.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace x3 = strata::mm3::exl3;
namespace nx = strata::mm3::native_experts;
namespace ff = infernix::ops::minimax_ffn;
constexpr int H = 6144, F = 3072, K = 4, POOL = 16;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
template <class V> V* dev(size_t n) { V* p = nullptr; ck(cudaMalloc((void**) &p, n * sizeof(V)), "cudaMalloc"); return p; }

int main(int argc, char** argv) {
    try {
        const int device = argc > 1 ? std::atoi(argv[1]) : 0;
        ck(cudaSetDevice(device), "set device");
        std::mt19937_64 rng(20261114);
        // POOL synthetic expert blobs: the 256-byte header (unit global scales), then the RG16 record: units of 144 bytes,
        // 128 code bytes and 16 E4M3 scale bytes (here 0x30-0x37: 0.5 to 0.94)
        const size_t rec = ff::record_stride(H, F), blob = sizeof(nx::Header) + rec;
        std::vector<uint8_t> host(blob * POOL);
        for (int e = 0; e < POOL; ++e) {
            nx::Header h{};
            for (int i = 0; i < 3; ++i) { h.weight_global[i] = 1.0f; h.input_global[i] = 1.0f; }
            std::memcpy(&host[(size_t) e * blob], &h, sizeof h);
            uint8_t* r = &host[(size_t) e * blob + sizeof h];
            for (size_t i = 0; i < rec; ++i) r[i] = (i % 144) < 128 ? uint8_t(rng()) : uint8_t(0x30 + (rng() & 7));
        }
        uint8_t* blobs = dev<uint8_t>(host.size());
        ck(cudaMemcpy(blobs, host.data(), host.size(), cudaMemcpyHostToDevice), "upload blobs");
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        const size_t sb = nx::scratch_bytes(H, F, nx::Arithmetic::NativeFp4);
        uint8_t* scratch = dev<uint8_t>(sb);
        if (!nx::prepare(s, H, F, scratch, sb, nx::Arithmetic::NativeFp4)) throw std::runtime_error("native FP4 prepare (an RTX 5090?)");
        constexpr int TMAX = 8;
        float* x = dev<float>((size_t) TMAX * H);
        float* part = dev<float>((size_t) TMAX * 8 * H);
        float* out = dev<float>((size_t) TMAX * H);
        unsigned long long* plan_ptr = dev<unsigned long long>(TMAX * 8);
        float* plan_w = dev<float>(TMAX * 8);
        unsigned long long* gp = dev<unsigned long long>(TMAX * 8);
        int* gs = dev<int>(TMAX * 8 + 1);
        int* ge = dev<int>(TMAX * 8);
        int* ng = dev<int>(1);
        {
            std::vector<float> hx((size_t) TMAX * H);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& v : hx) v = nd(rng);
            ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        }
        cudaEvent_t a, b;
        ck(cudaEventCreate(&a), "event");
        ck(cudaEventCreate(&b), "event");
        int failures = 0;
        std::printf("mm3_native_expert_fast_test: the 5090's window expert phase, microseconds a replay (CUDA graph, measured), "
                    "by rows and distinct experts; outputs compared with expert_fast 0 bit for bit\n");
        std::printf("  %4s %8s %6s %10s %10s %10s %8s\n", "rows", "experts", "picks", "mode 0", "mode 1", "mode 2", "same?");
        for (int T : {1, 2, 4, 8}) {
            for (int E : {1, 3, 6, 10}) {
                if (E > T * K) continue;
                // the plan: each row picks min(K, E) distinct experts of the first E, weights 1/K; unused slots empty
                std::vector<unsigned long long> pp(TMAX * 8, 0);
                std::vector<float> pw(TMAX * 8, 0.f);
                std::vector<std::vector<int>> users(E);
                std::vector<int> order(E);
                for (int t = 0; t < T; ++t) {
                    std::iota(order.begin(), order.end(), 0);
                    std::shuffle(order.begin(), order.end(), rng);
                    for (int i = 0; i < std::min(K, E); ++i) {
                        const int e = order[(size_t) i];
                        pp[(size_t) t * 8 + i] = (unsigned long long) (uintptr_t) (blobs + (size_t) e * blob);
                        pw[(size_t) t * 8 + i] = 1.0f / K;
                        users[(size_t) e].push_back(t * 8 + i);
                    }
                }
                std::vector<unsigned long long> hgp;
                std::vector<int> hgs{0}, hge;
                int picks = 0;
                for (int e = 0; e < E; ++e) {
                    if (users[(size_t) e].empty()) continue;
                    hgp.push_back((unsigned long long) (uintptr_t) (blobs + (size_t) e * blob));
                    for (int v : users[(size_t) e]) hge.push_back(v);
                    hgs.push_back((int) hge.size());
                    picks += (int) users[(size_t) e].size();
                }
                const int groups = (int) hgp.size();
                ck(cudaMemcpy(plan_ptr, pp.data(), pp.size() * 8, cudaMemcpyHostToDevice), "plan");
                ck(cudaMemcpy(plan_w, pw.data(), pw.size() * 4, cudaMemcpyHostToDevice), "plan w");
                ck(cudaMemcpy(gp, hgp.data(), hgp.size() * 8, cudaMemcpyHostToDevice), "groups");
                ck(cudaMemcpy(gs, hgs.data(), hgs.size() * 4, cudaMemcpyHostToDevice), "group starts");
                ck(cudaMemcpy(ge, hge.data(), hge.size() * 4, cudaMemcpyHostToDevice), "group entries");
                ck(cudaMemcpy(ng, &groups, 4, cudaMemcpyHostToDevice), "group count");
                x3::WinArgs w{};
                w.grp_ptr = gp; w.grp_start = gs; w.grp_ent = ge; w.n_groups = ng;
                w.plan_ptr = plan_ptr; w.plan_w = plan_w; w.x = x; w.x_stride = H; w.part = part; w.out = out;
                w.out_stride = H; w.lay.fmt = nx::kFmt; w.T = T; w.k = K; w.n_embd = H; w.n_ff = F; w.limit = 7.0f;
                std::vector<float> ref((size_t) T * H), got((size_t) T * H);
                double us[3] = {0, 0, 0};
                bool same = true;
                for (int mode = 0; mode < 3; ++mode) {
                    nx::set_fast(mode);
                    ck(cudaMemsetAsync(out, 0xff, (size_t) T * H * 4, s), "poison");
                    if (!nx::window(w, s)) throw std::runtime_error("window refused");
                    ck(cudaStreamSynchronize(s), "window");
                    ck(cudaMemcpy(mode == 0 ? ref.data() : got.data(), out, (size_t) T * H * 4, cudaMemcpyDeviceToHost), "read");
                    if (mode && std::memcmp(ref.data(), got.data(), ref.size() * 4) != 0) { same = false; ++failures; }
                    // as a graph, the way the expert graph replays it
                    cudaGraph_t gr = nullptr;
                    cudaGraphExec_t ex = nullptr;
                    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                    if (!nx::window(w, s)) throw std::runtime_error("window refused in capture");
                    ck(cudaStreamEndCapture(s, &gr), "end capture");
                    ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                    for (int i = 0; i < 10; ++i) ck(cudaGraphLaunch(ex, s), "warm");
                    ck(cudaEventRecord(a, s), "record");
                    constexpr int reps = 200;
                    for (int i = 0; i < reps; ++i) ck(cudaGraphLaunch(ex, s), "replay");
                    ck(cudaEventRecord(b, s), "record");
                    ck(cudaEventSynchronize(b), "sync");
                    float ms = 0;
                    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
                    us[mode] = 1000.0 * ms / reps;
                    cudaGraphExecDestroy(ex);
                    cudaGraphDestroy(gr);
                }
                std::printf("  %4d %8d %6d %10.2f %10.2f %10.2f %8s\n", T, groups, picks, us[0], us[1], us[2], same ? "yes" : "NO");
            }
        }
        nx::set_fast(0);
        std::printf("  (synthetic records in VRAM, 31.85 MB each, more than the L2 holds; the times are the GPU's alone, without the\n"
                    "   engine's uploads and host submission)\n");
        std::printf("mm3_native_expert_fast_test: %s\n", failures ? "FAIL (a mode's output differs from mode 0's)" : "ok");
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm3_native_expert_fast_test: %s\n", e.what());
        return 2;
    }
}
