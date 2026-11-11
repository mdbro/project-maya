// prompt_layerwise's expert kernels against the window path's, bit for bit (no model needed: random RG16 records).
// Usage: mm3_native_prompt_test [device]   (every arithmetic the device runs: dp4a always, native FP4 on an RTX 5090)
//  1. Column independence: each column's partial is the same bits whatever job it is in, at whatever position, with
//     however many columns and jobs a launch (1 to 64).
//  2. The same partials as the window path (native_experts::window on a 16-row plan), entry by entry.
//  3. The combine and residual add: prompt_combine against window's combine plus k::add's h += out, with 3090 sums
//     added for flagged rows, bit for bit.
//  4. prompt_rows_bf16 against the host's round to nearest even (window_post's helper input).
#include "mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/record.h"
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

namespace x3 = strata::mm3::exl3;
namespace ne = strata::mm3::native_experts;
namespace ff = infernix::ops::minimax_ffn;
constexpr int H = 6144, F = 3072, K = 4;

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

// a random record: random E2M1 codes, positive E4M3 block scales in a moderate range
static std::vector<uint8_t> random_blob(std::mt19937& rng, float wg) {
    const size_t rec = ff::record_stride(H, F);
    std::vector<uint8_t> b(sizeof(ne::Header) + rec, 0);
    ne::Header hd{};
    for (int i = 0; i < 3; ++i) { hd.weight_global[i] = wg * (1.0f + 0.25f * i); hd.input_global[i] = 1.0f; }
    std::memcpy(b.data(), &hd, sizeof hd);
    uint8_t* r = b.data() + sizeof(ne::Header);
    const size_t units = ff::record_bytes(H, F) / 144;
    std::uniform_int_distribution<int> code(0, 255), scale(0x30, 0x40);
    for (size_t u = 0; u < units; ++u) {
        uint8_t* unit = r + u * 144;
        for (int i = 0; i < 128; ++i) unit[i] = (uint8_t) code(rng);
        for (int i = 0; i < 16; ++i) unit[128 + i] = (uint8_t) scale(rng);
    }
    return b;
}

static int run(int device, ne::Arithmetic arith, const char* name) {
    std::mt19937 rng(1234);
    cudaStream_t s; CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // Every copy on the kernels' own stream and joined before use. A blocking cudaMemcpy from pageable memory runs on the
    // legacy stream and may return before its DMA lands, and a nonblocking stream does not wait for it: the first form of
    // this test let prompt_combine read a partly uploaded buffer (fp4 run, 2026-11-10 16:31).
    auto copy = [&](void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
        CUDA(cudaMemcpyAsync(dst, src, bytes, kind, s));
        CUDA(cudaStreamSynchronize(s));
    };
    const size_t nb = ne::scratch_bytes(H, F, arith);
    require(nb > 0, "scratch size");
    Dev<uint8_t> arena(nb + 256);
    void* base = (void*) (((uintptr_t) arena.p + 255) & ~uintptr_t(255));
    require(ne::prepare(s, H, F, base, nb, arith), "prepare");
    // two experts
    const int E = 2;
    std::vector<std::vector<uint8_t>> host;
    for (int e = 0; e < E; ++e) host.push_back(random_blob(rng, 0.5f * (e + 1)));
    const size_t blob = host[0].size();
    Dev<uint8_t> blobs(E * blob);
    for (int e = 0; e < E; ++e) copy(blobs.p + e * blob, host[(size_t) e].data(), blob, cudaMemcpyHostToDevice);
    // 16 rows (the window path's limit), four picks each: picks alternate between the two experts and a third
    // "elsewhere" (the 3090) per row, so a row's picks are 5090-served or not
    const int T = 16, R = T;
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> x((size_t) R * H);
    for (auto& v : x) v = uni(rng) * 0.5f;
    std::vector<uint16_t> x16((size_t) R * H);
    for (size_t i = 0; i < x.size(); ++i) x16[i] = bf16_rne(x[i]);
    // entries: row r pick i -> expert (r + i) % 5, four distinct a row (0 and 1 on this GPU, 2 to 4 elsewhere)
    std::vector<int> expert_of((size_t) R * K);
    std::vector<float> w((size_t) R * K);
    for (int r = 0; r < R; ++r)
        for (int i = 0; i < K; ++i) { expert_of[(size_t) r * K + i] = (r + i) % 5; w[(size_t) r * K + i] = 0.1f + 0.05f * i + 0.01f * r; }
    Dev<uint16_t> dx16((size_t) R * H);
    Dev<float> dx((size_t) R * H);
    copy(dx16.p, x16.data(), x16.size() * 2, cudaMemcpyHostToDevice);
    copy(dx.p, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    // 4: bf16 conversion
    {
        Dev<uint16_t> got((size_t) R * H);
        require(ne::prompt_rows_bf16(dx.p, R, H, got.p, s), "prompt_rows_bf16");
        std::vector<uint16_t> h((size_t) R * H);
        CUDA(cudaStreamSynchronize(s));
        copy(h.data(), got.p, h.size() * 2, cudaMemcpyDeviceToHost);
        require(h == x16, "prompt_rows_bf16 differs from round to nearest even");
    }
    // 1: prompt_run under several job compositions
    const int max_jobs = 64;
    const size_t sb = ne::prompt_scratch_bytes(s, max_jobs);
    require(sb > 0, "prompt scratch");
    Dev<uint8_t> scratch(sb);
    const size_t entries = (size_t) R * K;
    auto compose = [&](int width, bool reverse, int mj) {
        std::vector<ne::PromptJob> jobs;
        for (int e = 0; e < E; ++e) {
            std::vector<int> v;
            for (size_t k = 0; k < entries; ++k) if (expert_of[k] == e) v.push_back((int) k);
            if (reverse) std::reverse(v.begin(), v.end());
            for (size_t a = 0; a < v.size(); a += (size_t) width) {
                ne::PromptJob j{};
                j.blob = blobs.p + e * blob; j.n = (int) std::min<size_t>((size_t) width, v.size() - a);
                for (int c = 0; c < j.n; ++c) j.entry[c] = v[a + (size_t) c];
                jobs.push_back(j);
                if (jobs.size() % 3 == 1) { ne::PromptJob idle{}; jobs.push_back(idle); }   // (inactive descriptors between)
            }
        }
        Dev<ne::PromptJob> dj(jobs.size());
        copy(dj.p, jobs.data(), jobs.size() * sizeof(ne::PromptJob), cudaMemcpyHostToDevice);
        Dev<uint16_t> part(entries * H);
        CUDA(cudaMemsetAsync(part.p, 0, entries * H * 2, s));
        CUDA(cudaStreamSynchronize(s));
        require(ne::prompt_run(dj.p, 0, (int) jobs.size(), dx16.p, part.p, scratch.p, sb, mj, s), "prompt_run");
        CUDA(cudaStreamSynchronize(s));
        std::vector<uint16_t> h(entries * H);
        copy(h.data(), part.p, h.size() * 2, cudaMemcpyDeviceToHost);
        return h;
    };
    const auto ref = compose(1, false, 1);
    int checks = 0;
    for (int width : {1, 3, 5, 8})
        for (bool rev : {false, true})
            for (int mj : {1, 2, 7, 64}) {
                const auto got = compose(width, rev, mj);
                for (size_t k = 0; k < entries; ++k) {
                    if (expert_of[k] >= E) continue;
                    if (std::memcmp(got.data() + k * H, ref.data() + k * H, (size_t) H * 2) != 0) {
                        std::fprintf(stderr, "%s: entry %zu differs at width %d reverse %d max_jobs %d\n", name, k, width, (int) rev, mj);
                        return 1;
                    }
                }
                ++checks;
            }
    // 2 and 3: the window path on the same 16 rows, plan entries t * 8 + i
    std::vector<unsigned long long> plan_ptr((size_t) T * 8, 0), grp_ptr;
    std::vector<float> plan_w((size_t) T * 8, 0.0f);
    std::vector<int> grp_start, grp_ent;
    for (int e = 0; e < E; ++e) {
        grp_ptr.push_back((unsigned long long) (uintptr_t) (blobs.p + e * blob));
        grp_start.push_back((int) grp_ent.size());
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i)
                if (expert_of[(size_t) t * K + i] == e) grp_ent.push_back(t * 8 + i);
    }
    grp_start.push_back((int) grp_ent.size());
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) {
            const int e = expert_of[(size_t) t * K + i];
            plan_w[(size_t) t * 8 + i] = w[(size_t) t * K + i];
            if (e < E) plan_ptr[(size_t) t * 8 + i] = (unsigned long long) (uintptr_t) (blobs.p + e * blob);
        }
    // split the window's groups into pieces of at most 16 entries (window_piece takes two eight-column pieces a group)
    {
        std::vector<unsigned long long> gp2; std::vector<int> gs2, ge2;
        for (size_t g2 = 0; g2 + 1 < grp_start.size(); ++g2)
            for (int a = grp_start[g2]; a < grp_start[g2 + 1]; a += 16) {
                gp2.push_back(grp_ptr[g2]); gs2.push_back((int) ge2.size());
                for (int q = a; q < std::min(a + 16, grp_start[g2 + 1]); ++q) ge2.push_back(grp_ent[(size_t) q]);
            }
        gs2.push_back((int) ge2.size());
        grp_ptr = gp2; grp_start = gs2; grp_ent = ge2;
    }
    const int ngroups = (int) grp_ptr.size();
    Dev<unsigned long long> d_plan_ptr(plan_ptr.size()), d_grp_ptr(grp_ptr.size());
    Dev<float> d_plan_w(plan_w.size());
    Dev<int> d_gs(grp_start.size()), d_ge(grp_ent.size()), d_ng(1);
    copy(d_plan_ptr.p, plan_ptr.data(), plan_ptr.size() * 8, cudaMemcpyHostToDevice);
    copy(d_grp_ptr.p, grp_ptr.data(), grp_ptr.size() * 8, cudaMemcpyHostToDevice);
    copy(d_plan_w.p, plan_w.data(), plan_w.size() * 4, cudaMemcpyHostToDevice);
    copy(d_gs.p, grp_start.data(), grp_start.size() * 4, cudaMemcpyHostToDevice);
    copy(d_ge.p, grp_ent.data(), grp_ent.size() * 4, cudaMemcpyHostToDevice);
    copy(d_ng.p, &ngroups, 4, cudaMemcpyHostToDevice);
    std::vector<float> sh((size_t) T * H), h0((size_t) T * H), helper((size_t) T * H);
    for (auto& v : sh) v = uni(rng); for (auto& v : h0) v = uni(rng) * 2; for (auto& v : helper) v = uni(rng) * 0.1f;
    // (signed zeros where the CPU-part add and the residual add are sensitive)
    sh[5] = -0.0f; helper[5] = -0.0f; h0[7] = -0.0f;
    std::vector<int> flag((size_t) T);
    for (int t = 0; t < T; ++t) flag[(size_t) t] = 0;
    for (int t = 0; t < T; ++t) for (int i = 0; i < K; ++i) if (expert_of[(size_t) t * K + i] >= E) flag[(size_t) t] = 1;
    std::vector<float> cpu_part((size_t) T * H, 0.0f);   // window_post: zero-filled, then += the 3090's rows
    for (size_t i = 0; i < cpu_part.size(); ++i) cpu_part[i] += helper[i];
    Dev<float> d_sh(sh.size()), d_out(sh.size()), d_h(sh.size()), d_part((size_t) T * 8 * H), d_cpu(sh.size());
    Dev<int> d_flag(T);
    copy(d_sh.p, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice);
    copy(d_cpu.p, cpu_part.data(), cpu_part.size() * 4, cudaMemcpyHostToDevice);
    copy(d_flag.p, flag.data(), flag.size() * 4, cudaMemcpyHostToDevice);
    x3::WinArgs a;
    a.grp_ptr = d_grp_ptr.p; a.grp_start = d_gs.p; a.grp_ent = d_ge.p; a.n_groups = d_ng.p;
    a.plan_ptr = d_plan_ptr.p; a.plan_w = d_plan_w.p; a.x = dx.p; a.x_stride = H; a.part = d_part.p; a.sh_out = d_sh.p;
    a.out = d_out.p; a.out_stride = H; a.T = T; a.k = K; a.n_embd = H; a.n_ff = F; a.limit = 7.0f;
    a.cpu_part = d_cpu.p; a.cpu_stride = H; a.cpu_flag = d_flag.p;
    require(ne::window(a, s), "window");
    CUDA(cudaStreamSynchronize(s));
    std::vector<float> wpart((size_t) T * 8 * H), wout(sh.size());
    copy(wpart.data(), d_part.p, wpart.size() * 4, cudaMemcpyDeviceToHost);
    copy(wout.data(), d_out.p, wout.size() * 4, cudaMemcpyDeviceToHost);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i) {
            if (expert_of[(size_t) t * K + i] >= E) continue;
            const uint16_t* pr = ref.data() + ((size_t) t * K + i) * H;
            const float* pw = wpart.data() + ((size_t) t * 8 + i) * H;
            for (int d = 0; d < H; ++d) {
                const float v = from_bf16(pr[d]);
                if (std::memcmp(&v, pw + d, 4) != 0) {
                    std::fprintf(stderr, "%s: row %d pick %d element %d: prompt %.9g window %.9g\n", name, t, i, d, v, pw[d]);
                    return 1;
                }
            }
        }
    // the window's h after k::add: h0 + out
    std::vector<float> want_h(sh.size());
    for (size_t i = 0; i < want_h.size(); ++i) { float v = h0[i]; v += wout[i]; want_h[i] = v; }
    // prompt_combine on the prompt partials, the 3090 sums as prompt_read keeps them (0 + its rows), mapped
    float* m_helper = nullptr; int* m_flag = nullptr;
    CUDA(cudaHostAlloc((void**) &m_helper, sh.size() * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    CUDA(cudaHostAlloc((void**) &m_flag, T * 4, cudaHostAllocMapped | cudaHostAllocPortable));
    for (size_t i = 0; i < sh.size(); ++i) { float v = 0.0f; v += helper[i]; m_helper[i] = v; }
    for (int t = 0; t < T; ++t) m_flag[t] = flag[(size_t) t];
    std::vector<float> pwk((size_t) T * K); std::vector<uint8_t> served((size_t) T * K);
    for (size_t k = 0; k < pwk.size(); ++k) { pwk[k] = w[k]; served[k] = expert_of[k] < E; }
    Dev<float> d_pw(pwk.size()); Dev<uint8_t> d_served(served.size()); Dev<uint16_t> d_part16(entries * H);
    copy(d_pw.p, pwk.data(), pwk.size() * 4, cudaMemcpyHostToDevice);
    copy(d_served.p, served.data(), served.size(), cudaMemcpyHostToDevice);
    copy(d_part16.p, ref.data(), entries * H * 2, cudaMemcpyHostToDevice);
    copy(d_h.p, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice);
    require(ne::prompt_combine(T, K, H, d_sh.p, d_part16.p, d_pw.p, d_served.p, m_helper, m_flag, d_h.p, s), "prompt_combine");
    CUDA(cudaStreamSynchronize(s));
    std::vector<float> got_h(sh.size());
    copy(got_h.data(), d_h.p, got_h.size() * 4, cudaMemcpyDeviceToHost);
    size_t bad = 0, first_bad = got_h.size();
    for (size_t i = 0; i < got_h.size(); ++i)
        if (std::memcmp(&got_h[i], &want_h[i], 4) != 0) { ++bad; if (first_bad == got_h.size()) first_bad = i; }
    if (bad) {
        // what each side should be, from the same inputs on the host (combine_window's order, with and without FMA)
        const size_t i = first_bad, t = i / H, d = i % H;
        std::fprintf(stderr, "%s: combine differs at %zu of %zu elements; first: row %zu element %zu: prompt %.9g window %.9g\n",
                     name, bad, got_h.size(), t, d, got_h[i], want_h[i]);
        std::fprintf(stderr, "  h0 %.9g sh %.9g flag %d helper %.9g window out %.9g\n", h0[i], sh[i], flag[t], helper[i], wout[i]);
        float v = sh[i], vf = sh[i];
        for (int k = 0; k < K; ++k) {
            const size_t e = t * K + k;
            const float pp = from_bf16(ref[e * H + d]), pw = wpart[(t * 8 + k) * H + d];
            std::fprintf(stderr, "  pick %d expert %d served %d w %.6g prompt part %.9g window part %.9g\n", k, expert_of[e],
                         (int) served[e], pwk[e], pp, pw);
            if (served[e]) { v += pwk[e] * pp; vf = std::fma(pwk[e], pp, vf); }
        }
        if (flag[t]) { v += helper[i]; vf += helper[i]; }
        std::fprintf(stderr, "  host: out %.9g (fma %.9g), h0 + out %.9g (fma %.9g)\n", v, vf, h0[i] + v, h0[i] + vf);
        for (size_t j = 0, shown = 0; j < got_h.size() && shown < 6; ++j)
            if (std::memcmp(&got_h[j], &want_h[j], 4) != 0) {
                std::fprintf(stderr, "  differs at row %zu element %zu: prompt %.9g window %.9g\n", j / H, j % H, got_h[j], want_h[j]); ++shown;
            }
        return 1;
    }
    cudaFreeHost(m_helper); cudaFreeHost(m_flag);
    ne::release(s);
    CUDA(cudaStreamDestroy(s));
    std::printf("%s on device %d: %d job compositions, window partials and combine: bitwise equal\n", name, device, checks + 1);
    return 0;
}

int main(int argc, char** argv) {
    try {
        const int device = argc > 1 ? std::atoi(argv[1]) : 0;
        CUDA(cudaSetDevice(device));
        cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop, device));
        int rc = run(device, ne::Arithmetic::Dp4a, "dp4a");
        if (rc) return rc;
        if (prop.major == 12 && prop.minor == 0 && std::strstr(prop.name, "RTX 5090")) rc = run(device, ne::Arithmetic::NativeFp4, "fp4");
        else std::printf("fp4: skipped on %s\n", prop.name);
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm3_native_prompt_test: %s\n", e.what());
        return 2;
    }
}
