// --ablate on the native serving path (upstream c5842c5's refusal ablation, built into our four residual adds by the
// port onto minimax-wsl/local, 2026-11-13): the pieces that run on a GPU, without a model (random rows).
// Usage: mm3_native_ablate_test [device]   (exit 0: all hold; run on a GPU in an engine-start gap, not beside serving)
//  1. The prompt reader's combine written out (prompt_combine_to, the path a request with the projection on takes) and
//     then added with k::add gives prompt_combine's h bit for bit: the ablation path changes nothing but the
//     projection itself. (With no direction file the reader calls prompt_combine as before the sync: the default path
//     is the same kernel; prompt_combine_rows<false> is its body unchanged.)
//  2. y may be the shared expert's own buffer (the reader writes its combine into sh in place): the same bits as a
//     separate y.
//  3. The projection lands: combine, k::project_out with a unit direction, then k::add, against a double-precision
//     host reference (relative error), and the residual write left has no component along the direction.
//  4. Inside a captured CUDA graph (the expert graph's and the preparation graph's projection): the replay gives the
//     direct launches' bits, and the graph captured without the projection has exactly one node fewer (the default
//     graph is the graph before the sync: the projection is the only node --ablate adds).
// The engine-level check (the same tokens and per-layer prompt hashes as the binary before the sync, with no direction
// file; a direction loaded but off; and the graphs' verify modes with it on) is tools/mm3/ablate_identity_check.py.
#include "mm3_kernels.cuh"
#include "mm3_native_experts.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ne = strata::mm3::native_experts;
namespace k = mm3::k;
constexpr int H = 6144, K = 4;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
#define CUDA(call) ck((call), #call)
static void require(bool b, const std::string& msg) { if (!b) throw std::runtime_error(msg); }
template <class V> struct Dev {
    V* p = nullptr;
    explicit Dev(size_t n) { CUDA(cudaMalloc((void**) &p, std::max<size_t>(1, n) * sizeof(V))); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
};
static uint16_t bf16_rne(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static float from_bf16(uint16_t b) { uint32_t u = uint32_t(b) << 16; float x; std::memcpy(&x, &u, 4); return x; }

static size_t differ(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
    return n;
}

static int run(int rows) {
    std::mt19937 rng(20261113u + (unsigned) rows);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> uw(0.05f, 0.5f);
    std::bernoulli_distribution coin(0.7);
    cudaStream_t s; CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // (every copy on the kernels' stream, joined before use: see mm3_native_prompt_test's note)
    auto up = [&](void* dst, const void* src, size_t bytes) { CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s)); };
    auto down = [&](void* dst, const void* src, size_t bytes) {
        CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s)); CUDA(cudaStreamSynchronize(s));
    };
    const size_t n = (size_t) rows * H;
    std::vector<float> h0(n), sh(n), helper(n), w((size_t) rows * K), r(H);
    std::vector<uint16_t> part((size_t) rows * K * H);
    std::vector<uint8_t> served((size_t) rows * K);
    std::vector<int> flag((size_t) rows);
    for (auto& x : h0) x = 4.0f * nd(rng);
    for (auto& x : sh) x = 0.5f * nd(rng);
    for (auto& x : helper) x = 0.3f * nd(rng);
    for (auto& x : part) x = bf16_rne(0.4f * nd(rng));
    for (auto& x : w) x = uw(rng);
    for (auto& x : served) x = coin(rng) ? 1 : 0;
    for (int t = 0; t < rows; ++t) flag[(size_t) t] = t % 3 == 1;
    double rn = 0;
    for (auto& x : r) { x = nd(rng); rn += (double) x * x; }
    for (auto& x : r) x = (float) (x / std::sqrt(rn));   // (a unit direction, as mm3::ablate::load gives)

    float* m_helper = nullptr; int* m_flag = nullptr;   // (mapped, as prompt_read keeps the 3090's sums)
    CUDA(cudaHostAlloc((void**) &m_helper, n * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA(cudaHostAlloc((void**) &m_flag, (size_t) rows * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(m_helper, helper.data(), n * 4);
    std::memcpy(m_flag, flag.data(), (size_t) rows * 4);
    Dev<float> d_h(n), d_sh(n), d_y(n), d_w(w.size()), d_r(H);
    Dev<uint16_t> d_part(part.size());
    Dev<uint8_t> d_served(served.size());
    up(d_part.p, part.data(), part.size() * 2);
    up(d_w.p, w.data(), w.size() * 4);
    up(d_served.p, served.data(), served.size());
    up(d_r.p, r.data(), H * 4);
    auto reset = [&]() { up(d_h.p, h0.data(), n * 4); up(d_sh.p, sh.data(), n * 4); };

    // 1. the default combine (h += v) against the written-out one plus k::add
    reset();
    require(ne::prompt_combine(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_h.p, s), "prompt_combine");
    std::vector<float> want(n); down(want.data(), d_h.p, n * 4);
    reset();
    require(ne::prompt_combine_to(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_y.p, s), "prompt_combine_to");
    k::add(d_h.p, d_y.p, n, s);
    std::vector<float> got(n); down(got.data(), d_h.p, n * 4);
    if (size_t bad = differ(got, want)) {
        std::fprintf(stderr, "rows %d: combine written out then added differs from prompt_combine at %zu of %zu\n", rows, bad, n);
        return 1;
    }
    // 2. y = sh in place (the reader's form)
    reset();
    require(ne::prompt_combine_to(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_sh.p, s), "in place");
    k::add(d_h.p, d_sh.p, n, s);
    down(got.data(), d_h.p, n * 4);
    if (size_t bad = differ(got, want)) {
        std::fprintf(stderr, "rows %d: the in-place combine differs at %zu of %zu\n", rows, bad, n);
        return 1;
    }
    // 3. the projection: h0 + (v - (r . v) r), against the host in double
    reset();
    require(ne::prompt_combine_to(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_sh.p, s), "projected");
    k::project_out(d_sh.p, d_r.p, 1.0f, rows, H, s);
    k::add(d_h.p, d_sh.p, n, s);
    down(got.data(), d_h.p, n * 4);
    double worst = 0, worst_along = 0;
    for (int t = 0; t < rows; ++t) {
        std::vector<double> v(H);
        double dot = 0;
        for (int d = 0; d < H; ++d) {
            const size_t i = (size_t) t * H + d;
            double x = sh[i];
            for (int p = 0; p < K; ++p) {
                const size_t e = (size_t) t * K + p;
                if (served[e]) x += (double) w[e] * from_bf16(part[e * H + d]);
            }
            if (flag[(size_t) t]) x += helper[i];
            v[(size_t) d] = x;
            dot += (double) r[(size_t) d] * x;
        }
        double along = 0, scale = 0;
        for (int d = 0; d < H; ++d) {
            const size_t i = (size_t) t * H + d;
            const double ref = h0[i] + (v[(size_t) d] - dot * r[(size_t) d]);
            worst = std::max(worst, std::abs((double) got[i] - ref) / (1.0 + std::abs(ref)));
            along += (double) r[(size_t) d] * ((double) got[i] - h0[i]);
            scale = std::max(scale, std::abs(v[(size_t) d]));
        }
        worst_along = std::max(worst_along, std::abs(along) / (1.0 + scale));
    }
    if (!(worst < 1e-5) || !(worst_along < 1e-4)) {
        std::fprintf(stderr, "rows %d: projected combine: relative error %.3g (want < 1e-5), left along r %.3g (want < 1e-4)\n",
                     rows, worst, worst_along);
        return 1;
    }
    // 4. in a captured graph: the projection and the add, replayed, against direct launches; one node fewer without it
    std::vector<float> direct(got);
    auto capture = [&](bool project, size_t& nodes) {
        cudaGraph_t gr = nullptr; cudaGraphExec_t ex = nullptr;
        CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        if (project) k::project_out(d_sh.p, d_r.p, 1.0f, rows, H, s);
        k::add(d_h.p, d_sh.p, n, s);
        CUDA(cudaStreamEndCapture(s, &gr));
        CUDA(cudaGraphGetNodes(gr, nullptr, &nodes));
        CUDA(cudaGraphInstantiate(&ex, gr, 0));
        CUDA(cudaGraphDestroy(gr));
        return ex;
    };
    size_t with = 0, without = 0;
    cudaGraphExec_t g_on = capture(true, with), g_off = capture(false, without);
    reset();
    require(ne::prompt_combine_to(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_sh.p, s), "graph combine");
    CUDA(cudaGraphLaunch(g_on, s));
    down(got.data(), d_h.p, n * 4);
    if (size_t bad = differ(got, direct)) {
        std::fprintf(stderr, "rows %d: the graph's projection differs from the direct launches at %zu of %zu\n", rows, bad, n);
        return 1;
    }
    reset();
    require(ne::prompt_combine_to(rows, K, H, d_sh.p, d_part.p, d_w.p, d_served.p, m_helper, m_flag, d_sh.p, s), "graph combine");
    CUDA(cudaGraphLaunch(g_off, s));
    down(got.data(), d_h.p, n * 4);
    if (size_t bad = differ(got, want)) {
        std::fprintf(stderr, "rows %d: the graph without the projection differs from prompt_combine at %zu of %zu\n", rows, bad, n);
        return 1;
    }
    if (with != without + 1) {
        std::fprintf(stderr, "rows %d: graph nodes %zu with the projection, %zu without (want one more)\n", rows, with, without);
        return 1;
    }
    CUDA(cudaGraphExecDestroy(g_on)); CUDA(cudaGraphExecDestroy(g_off));
    CUDA(cudaFreeHost(m_helper)); CUDA(cudaFreeHost(m_flag));
    CUDA(cudaStreamDestroy(s));
    std::printf("rows %d: combine identity, in place, projection (rel %.2g, along %.2g), graph (%zu / %zu nodes): ok\n", rows,
                worst, worst_along, with, without);
    return 0;
}

int main(int argc, char** argv) {
    try {
        CUDA(cudaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
        for (int rows : {1, 3, 16, 33, 160})
            if (run(rows)) return 1;
        std::printf("mm3_native_ablate_test: ok\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm3_native_ablate_test: %s\n", e.what());
        return 1;
    }
}
