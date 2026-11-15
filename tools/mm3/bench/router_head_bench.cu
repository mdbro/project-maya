// tools/mm3/bench/router_head_bench.cu - kernel items 9 (the router) and 10 (the output head), standalone on the 5090.
// Needs the GPU to itself (about 4.5 GB free: run it in an engine gap, never beside a timed request).
//
//   router: rows_dot (the gate, F32 [128 x 6144]) at 8 rows a block (today) and 4, 2, 1 - its outputs compared bit for bit;
//           rows_dot2 (the router and the next layer's prediction in one launch) against two launches, bit for bit;
//           the routing kernel; and the preparation's two branches as one CUDA graph - the router's (rows_dot, route,
//           and at one row early_copy's prediction rows_dot + route) on a side stream beside the shared expert's
//           (MXFP8 gate, up [3072 x 6144], SwiGLU, down [6144 x 3072]) - so the graph's time shows whether the router's
//           branch is the one a pass waits for;
//   head:   today's BF16 kernels (mv: bf16_mv, bf16_mv_cols) against the FP8 and NVFP4 heads (mm3_head_lowp) at 1-8
//           columns; with --sweep each low-precision kernel's shapes; checks: the quantizers against a host reference
//           (sampled blocks, bit for bit), each kernel against a double-precision sum of its own dequantized weights
//           (sampled rows), and the low-precision logits against the BF16 ones (relative RMS, largest difference, top-1
//           and top-16 agreement, KL of the temperature-1 distributions) on two kinds of columns: random, and "peaked"
//           (a few head rows mixed, so some tokens stand out as in a real distribution).
// The head is the checkpoint's own (--pack, the native pack) or random BF16 (--random-head).
// Build: cmake target mm3_router_head_bench. Run: mm3_router_head_bench [--router] [--head] [--sweep] [--pack DIR]
// [--dense-kernel 0..3] selects the router comparison's faithful MXFP8 shared expert (default 2).
// Exit: 0 every gate passed; 1 an error (CUDA, the pack); 2 a router launch shape's outputs differ from 8 rows a block
// (the router_kernel arms are then invalid); 3 a head quantizer or kernel-arithmetic check failed (head_lowp invalid).
#include "mm3_head_lowp.cuh"
#include "mm3_dense_fp8.cuh"
#include "mm3_kernels.cuh"
#include "mm3_native_dense.cuh"
#include "mm3_pack.hpp"
#include "mm3_resident_router.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nd = strata::mm3::native_dense;
namespace x3 = strata::mm3::exl3;
namespace k = mm3::k;
namespace hl = mm3::head_lowp;
namespace strata::mm3::exl3 {   // (only the native dense formats are linked here, as tools/mm3/native_dense_bench.cu)
bool supported(const Mat& m) {
    return (m.fmt == nd::kFmtMXFP8 && m.raw && m.native_scale && m.k % 32 == 0) || (m.fmt == nd::kFmtBF16 && m.raw && m.k > 0);
}
}  // namespace strata::mm3::exl3

namespace {
bool g_router_bits_ok = true;   // (exit 2 when a launch shape's router outputs differ)
bool g_head_checks_ok = true;   // (exit 3 when a quantizer or a kernel's arithmetic check fails)
void ck(cudaError_t e, const char* op) { if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e)); }
// Host buffers may be pageable; return only when this stream's copy is done.
cudaError_t copy_sync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    return e == cudaSuccess ? cudaStreamSynchronize(stream) : e;
}
template <class T> T* dmalloc(size_t n) { void* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc"); return (T*) p; }
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float f32(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
template <class F> double time_ms(cudaStream_t s, int iters, F&& launch) {
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "event");
    ck(cudaEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) launch();
    ck(cudaEventRecord(a, s), "record");
    for (int i = 0; i < iters; ++i) launch();
    ck(cudaEventRecord(b, s), "record");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

// ---------------------------------------------------------------- router
void router(cudaStream_t s, int dense_kernel) {
    constexpr int N = 128, K = 6144, F = 3072;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd01(0.f, 1.f);
    std::vector<float> w((size_t) 2 * N * K), x((size_t) 8 * K);
    for (auto& v : w) v = nd01(rng) * 0.02f;
    for (auto& v : x) v = nd01(rng);
    float* dw = dmalloc<float>(w.size());
    float* dx = dmalloc<float>(x.size());
    float* dy = dmalloc<float>((size_t) 8 * 2 * N);
    float* dy2 = dmalloc<float>((size_t) 8 * 2 * N);
    ck(copy_sync(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice, s), "copy");
    ck(copy_sync(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice, s), "copy");
    std::printf("== router: rows_dot (the gate, F32 128 x 6144; 3.1 MB a pass), us a launch (events, 500 back to back)\n");
    std::printf("%5s %10s %10s %10s %10s   %s\n", "rows", "8 a block", "4 a block", "2 a block", "1 a block", "bits vs 8 a block");
    for (int T : {1, 2, 4, 8}) {
        std::vector<float> ref((size_t) T * N), got((size_t) T * N);
        double us[4];
        bool same = true;
        const int warps[4] = {8, 4, 2, 1};
        for (int i = 0; i < 4; ++i) {
            k::set_rows_dot_warps(warps[i]);
            us[i] = 1000 * time_ms(s, 500, [&] { k::rows_dot(dw, N, K, dx, T, dy, s); });
            ck(copy_sync(i ? got.data() : ref.data(), dy, (size_t) T * N * 4, cudaMemcpyDeviceToHost, s), "copy");
            if (i) same = same && std::memcmp(got.data(), ref.data(), got.size() * 4) == 0;
        }
        g_router_bits_ok = g_router_bits_ok && same;
        std::printf("%5d %10.2f %10.2f %10.2f %10.2f   %s\n", T, us[0], us[1], us[2], us[3], same ? "identical" : "DIFFERENT");
    }
    // rows_dot2: the router and the next layer's prediction (two gates) in one launch, against two launches
    std::printf("== router + prediction: rows_dot2 (256 rows, one launch) against two rows_dot launches, us\n");
    for (int warps : {8, 1}) {
        k::set_rows_dot_warps(warps);
        for (int T : {1, 8}) {
            const double two = 1000 * time_ms(s, 500, [&] {
                k::rows_dot(dw, N, K, dx, T, dy, s);
                k::rows_dot(dw + (size_t) N * K, N, K, dx, T, dy + (size_t) T * N, s);
            });
            std::vector<float> a((size_t) 2 * T * N), b((size_t) 2 * T * N);
            ck(copy_sync(a.data(), dy, a.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
            const double one = 1000 * time_ms(s, 500, [&] { k::rows_dot2(dw, N, dw + (size_t) N * K, N, K, dx, T, dy2, dy2 + (size_t) T * N, s); });
            ck(copy_sync(b.data(), dy2, b.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
            const bool same2 = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
            g_router_bits_ok = g_router_bits_ok && same2;
            std::printf("  %d a block, %d rows: two launches %.2f, one launch %.2f   %s\n", warps, T, two, one,
                        same2 ? "identical" : "DIFFERENT");
        }
    }
    // the routing kernel (the Prior bonus with a ready set, dampened weights, the per-row budget)
    float* bias = dmalloc<float>(N);
    ck(cudaMemsetAsync(bias, 0, N * 4, s), "memset");
    mm3::resident::Ready ready{};
    for (int e = 0; e < N; e += 3) ready.primary[e / 32] |= 1u << (e % 32);
    mm3::resident::Ready* dready = dmalloc<mm3::resident::Ready>(1);
    ck(copy_sync(dready, &ready, sizeof ready, cudaMemcpyHostToDevice, s), "copy");
    int* ids = dmalloc<int>(8 * 4);
    float* wts = dmalloc<float>(8 * 4);
    int* orig = dmalloc<int>(8 * 4);
    for (int T : {1, 8}) {
        const double us = 1000 * time_ms(s, 500, [&] {
            ck(mm3::resident::route(dy, bias, T, mm3::resident::Policy::Prior, 0.30f, dready, ids, wts, orig, nullptr, s, true,
                                    nullptr, 0.5f, nullptr), "route");
        });
        std::printf("== route (Prior 0.30 dampened, budget 0.5), %d rows: %.2f us\n", T, us);
    }
    // the preparation's two branches as one graph: side stream router (+ prediction) beside the shared expert
    uint8_t* sgw = dmalloc<uint8_t>((size_t) 2 * F * K);
    uint8_t* sgs = dmalloc<uint8_t>((size_t) 2 * F * K / 32);
    uint8_t* sdw = dmalloc<uint8_t>((size_t) K * F);
    uint8_t* sds = dmalloc<uint8_t>((size_t) K * F / 32);
    ck(cudaMemsetAsync(sgw, 0x38, (size_t) 2 * F * K, s), "memset");
    ck(cudaMemsetAsync(sgs, 120, (size_t) 2 * F * K / 32, s), "memset");
    ck(cudaMemsetAsync(sdw, 0x30, (size_t) K * F, s), "memset");
    ck(cudaMemsetAsync(sds, 120, (size_t) K * F / 32, s), "memset");
    float* g = dmalloc<float>((size_t) 8 * F);
    float* u = dmalloc<float>((size_t) 8 * F);
    float* sh = dmalloc<float>((size_t) 8 * K);
    namespace f8 = mm3::dense_fp8;
    if (!f8::supported()) throw std::runtime_error("corrected shared-expert comparison needs an sm_120 GPU");
    f8::Job shared_jobs[2];
    for (int j = 0; j < 2; ++j) {
        shared_jobs[j].w = sgw + size_t(j)*F*K; shared_jobs[j].sc = sgs + size_t(j)*F*K/32;
        shared_jobs[j].N = F; shared_jobs[j].K = K; shared_jobs[j].x = dx; shared_jobs[j].ldx = K;
        shared_jobs[j].y = j ? u : g; shared_jobs[j].ldy = F;
    }
    const size_t packed_bytes = f8::workspace_bytes(shared_jobs, 2, 8);
    uint8_t* packed = dmalloc<uint8_t>(packed_bytes);
    std::printf("   shared expert uses MXFP8 activations, dense kernel %d (checkpoint recipe)\n", dense_kernel);
    cudaStream_t side;
    ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "stream");
    cudaEvent_t fork, join;
    ck(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "event");
    std::printf("== the preparation's branches as one graph, us a replay (shared expert on the main stream; router beside it)\n");
    std::printf("%5s %-28s %12s %12s %12s %12s\n", "rows", "router branch", "shared only", "router only", "both, 8/blk", "both, 1/blk");
    for (int T : {1, 2, 4, 8}) {
        for (int pred = 0; pred < (T == 1 ? 2 : 1); ++pred) {
            auto graph_us = [&](bool shared, bool rout, int warps) {
                k::set_rows_dot_warps(warps);
                cudaGraph_t gr;
                cudaGraphExec_t ex;
                ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                ck(cudaEventRecord(fork, s), "fork");
                ck(cudaStreamWaitEvent(side, fork, 0), "wait");
                if (rout) {
                    if (pred) k::rows_dot2(dw, N, dw + (size_t) N * K, N, K, dx, T, dy, dy + (size_t) T * N, side);
                    else k::rows_dot(dw, N, K, dx, T, dy, side);
                    ck(mm3::resident::route(dy, bias, T, mm3::resident::Policy::Prior, 0.30f, dready, ids, wts, orig, nullptr,
                                            side, true, nullptr, 0.5f, nullptr), "route");
                    if (pred) ck(mm3::resident::route(dy + (size_t) T * N, bias, T, mm3::resident::Policy::Off, 0, nullptr,
                                                      ids, wts, nullptr, nullptr, side), "route");
                }
                if (shared) {
                    if (!f8::launch(shared_jobs, 2, T, s, 0, dense_kernel, packed, packed_bytes))
                        throw std::runtime_error("shared gate/up MXFP8 launch refused");
                    k::swigluoai(g, u, (size_t) T * F, 1.702f, 7.0f, g, nullptr, s);
                    f8::Job down;
                    down.w = sdw; down.sc = sds; down.N = K; down.K = F;
                    down.x = g; down.ldx = F; down.y = sh; down.ldy = K;
                    if (!f8::launch(&down, 1, T, s, 0, dense_kernel, packed, packed_bytes))
                        throw std::runtime_error("shared down MXFP8 launch refused");
                }
                ck(cudaEventRecord(join, side), "join");
                ck(cudaStreamWaitEvent(s, join, 0), "wait");
                ck(cudaStreamEndCapture(s, &gr), "end capture");
                ck(cudaGraphInstantiate(&ex, gr, 0), "instantiate");
                const double us = 1000 * time_ms(s, 300, [&] { ck(cudaGraphLaunch(ex, s), "launch"); });
                cudaGraphExecDestroy(ex);
                cudaGraphDestroy(gr);
                return us;
            };
            std::printf("%5d %-28s %12.2f %12.2f %12.2f %12.2f\n", T, pred ? "router + prediction (rows_dot2)" : "router",
                        graph_us(true, false, 8), graph_us(false, true, 8), graph_us(true, true, 8), graph_us(true, true, 1));
        }
    }
    k::set_rows_dot_warps(8);
    cudaFree(packed);
}

// ---------------------------------------------------------------- head
struct Stats { double rel_rms = 0, max_abs = 0, top1 = 0, top16 = 0, kl = 0; };
Stats compare_logits(const std::vector<float>& ref, const std::vector<float>& got, int V, int cols) {
    Stats st;
    double num = 0, den = 0;
    for (int c = 0; c < cols; ++c) {
        const float* a = ref.data() + (size_t) c * V;
        const float* b = got.data() + (size_t) c * V;
        std::vector<int> ia(V), ib(V);
        for (int i = 0; i < V; ++i) { ia[i] = ib[i] = i; num += double(a[i] - b[i]) * (a[i] - b[i]); den += double(a[i]) * a[i];
            st.max_abs = std::max(st.max_abs, (double) std::fabs(a[i] - b[i])); }
        std::partial_sort(ia.begin(), ia.begin() + 16, ia.end(), [&](int p, int q) { return a[p] > a[q]; });
        std::partial_sort(ib.begin(), ib.begin() + 16, ib.end(), [&](int p, int q) { return b[p] > b[q]; });
        st.top1 += ia[0] == ib[0];
        int overlap = 0;
        for (int i = 0; i < 16; ++i) for (int j = 0; j < 16; ++j) overlap += ia[i] == ib[j];
        st.top16 += overlap / 16.0;
        double ma = -1e30, mb = -1e30;
        for (int i = 0; i < V; ++i) { ma = std::max(ma, (double) a[i]); mb = std::max(mb, (double) b[i]); }
        double za = 0, zb = 0;
        for (int i = 0; i < V; ++i) { za += std::exp(a[i] - ma); zb += std::exp(b[i] - mb); }
        double kl = 0;
        for (int i = 0; i < V; ++i) {
            const double pa = std::exp(a[i] - ma) / za, lpa = a[i] - ma - std::log(za), lpb = b[i] - mb - std::log(zb);
            kl += pa * (lpa - lpb);
        }
        st.kl += kl;
    }
    st.rel_rms = std::sqrt(num / std::max(den, 1e-30));
    st.top1 /= cols;
    st.top16 /= cols;
    st.kl /= cols;
    return st;
}

void head(cudaStream_t s, const std::string& pack_dir, bool random_head, bool sweep) {
    int V = 200064, D = 6144;
    std::vector<uint16_t> host;
    if (!random_head) {
        mm3::Pack pk;
        std::string err;
        if (!pk.load(pack_dir, err)) throw std::runtime_error("pack: " + err);
        const mm3::Tensor* t = pk.get("language_model.lm_head.weight");
        if (!t || t->dtype != "BF16" || t->shape.size() != 2) throw std::runtime_error("pack: no BF16 lm_head");
        V = (int) t->shape[0];
        D = (int) t->shape[1];
        host.resize((size_t) V * D);
        std::memcpy(host.data(), pk.data(*t), host.size() * 2);
    } else {
        host.resize((size_t) V * D);
        std::mt19937 rng(9);
        std::normal_distribution<float> nd01(0.f, 1.f);
        for (auto& v : host) v = bf16(nd01(rng) * 0.02f);
    }
    std::printf("== head [%d x %d] (%s), BF16 %.2f GB\n", V, D, random_head ? "random" : "the checkpoint's", host.size() * 2 / 1e9);
    uint16_t* dh = dmalloc<uint16_t>(host.size());
    ck(copy_sync(dh, host.data(), host.size() * 2, cudaMemcpyHostToDevice, s), "copy head");
    hl::Head h8, h4;
    const char* err = nullptr;
    if (!hl::make(hl::kFp8, dh, V, D, h8, s, &err) || !hl::make(hl::kNvfp4, dh, V, D, h4, s, &err)) throw std::runtime_error(err);
    std::printf("   FP8 head %.2f GB, NVFP4 head %.2f GB (global scale %.6g)\n",
                (hl::weight_bytes(1, V, D) + hl::scale_bytes(1, V, D)) / 1e9, (hl::weight_bytes(2, V, D) + hl::scale_bytes(2, V, D)) / 1e9, h4.global);
    // quantizers against a host reference on sampled blocks (bit for bit)
    {
        std::vector<uint8_t> w8(hl::weight_bytes(1, V, D)), s8(hl::scale_bytes(1, V, D)), w4(hl::weight_bytes(2, V, D)), s4(hl::scale_bytes(2, V, D));
        ck(copy_sync(w8.data(), h8.w, w8.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(s8.data(), h8.sc, s8.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(w4.data(), h4.w, w4.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(s4.data(), h4.sc, s4.size(), cudaMemcpyDeviceToHost, s), "copy");
        std::mt19937 rng(1);
        long bad8 = 0, bad4 = 0, checked = 0;
        double err8 = 0, err4 = 0, ref2 = 0;
        float global_max = 0.f;
        for (uint16_t b : host) global_max = std::max(global_max, std::fabs(f32(b)));
        const float expected_global = global_max > 0.f ? global_max / 2688.f : 1.f;
        if (h4.global != expected_global) ++bad4;
        for (int it = 0; it < 20000; ++it) {
            const int r = int(rng() % V), j = int(rng() % D);
            const float v = f32(host[(size_t) r * D + j]);
            // FP8: scale 2^s with amax / 2^s <= 448 and s least; code = E4M3 nearest even of v / 2^s
            const int b = j / 32;
            float amax = 0;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(f32(host[(size_t) r * D + b * 32 + i])));
            int sexp = 0;
            if (amax > 0) { int ex; const float m = std::frexp(amax / 448.0f, &ex); sexp = m == 0.5f ? ex - 1 : ex; }
            const float q8 = hl::weight_host(h8, w8.data(), s8.data(), r, j);
            // the nearest E4M3 value (ties to even) of v / 2^s, times 2^s
            const float scaled = std::ldexp(v, -sexp);
            float best = 0, bestd = 1e30f; int bestc = 0;
            for (int c = 0; c < 256; ++c) {
                const float f = hl::e4m3_float((uint8_t) c);
                if (!std::isfinite(f)) continue;
                const float d = std::fabs(f - scaled);
                if (d < bestd || (d == bestd && (c & 1) == 0 && (bestc & 1))) { bestd = d; best = f; bestc = c; }
            }
            if (s8[(size_t) r * (D / 32) + b] != uint8_t(sexp + 127) || q8 != std::ldexp(best, sexp)) ++bad8;
            const float q4 = hl::weight_host(h4, w4.data(), s4.data(), r, j);
            // Independent NVFP4 oracle: enumerate representable scale/code
            // values and break exact ties by the code's low bit (nearest even).
            const int b4 = j/16;
            float max4 = 0.f;
            for (int i = 0; i < 16; ++i) max4 = std::max(max4, std::fabs(f32(host[size_t(r)*D + b4*16+i])));
            const float scaled_max = max4 / (6.f * expected_global);
            float distance = INFINITY;
            int expected_scale = 0;
            for (int c = 0; c <= 0x7e; ++c) {
                const float delta = std::fabs(hl::e4m3_float(uint8_t(c)) - scaled_max);
                if (delta < distance || (delta == distance && !(c&1) && (expected_scale&1))) {
                    distance = delta; expected_scale = c;
                }
            }
            const float divisor = hl::e4m3_float(uint8_t(expected_scale)) * expected_global;
            const float magnitude = divisor > 0.f ? std::fabs(v/divisor) : 0.f;
            distance = INFINITY;
            int expected_code = 0;
            for (int c = 0; c < 8; ++c) {
                const float delta = std::fabs(hl::e2m1_float(uint8_t(c)) - magnitude);
                if (delta < distance || (delta == distance && !(c&1) && (expected_code&1))) {
                    distance = delta; expected_code = c;
                }
            }
            if (divisor > 0.f && std::signbit(v)) expected_code |= 8;
            const uint8_t packed = w4[size_t(r)*(D/2)+j/2];
            const unsigned actual_code = j&1 ? packed>>4 : packed&15;
            if (s4[size_t(r)*(D/16)+b4] != expected_scale || actual_code != unsigned(expected_code)) ++bad4;
            err8 += double(q8 - v) * (q8 - v);
            err4 += double(q4 - v) * (q4 - v);
            ref2 += double(v) * v;
            ++checked;
        }
        g_head_checks_ok = g_head_checks_ok && bad8 == 0 && bad4 == 0;
        std::printf("   quantizer checks: FP8 %ld and NVFP4 %ld of %ld sampled weights differ from host codes/scales (NVFP4 global also checked); weight error, relative RMS: "
                    "FP8 %.3f%%, NVFP4 %.3f%%\n", bad8, bad4, checked, 100 * std::sqrt(err8 / ref2), 100 * std::sqrt(err4 / ref2));
    }
    // columns: random and peaked
    std::mt19937 rng(3);
    std::normal_distribution<float> nd01(0.f, 1.f);
    std::vector<float> xr((size_t) 8 * D), xp((size_t) 8 * D);
    for (auto& v : xr) v = nd01(rng);
    for (int c = 0; c < 8; ++c) {   // (a mix of 3 head rows plus noise, scaled so the top logits stand 10-20 above the rest)
        float* x = xp.data() + (size_t) c * D;
        for (int j = 0; j < D; ++j) x[j] = 0.3f * nd01(rng);
        for (int t = 0; t < 3; ++t) {
            const int row = int(rng() % V);
            double norm = 0;
            for (int j = 0; j < D; ++j) norm += double(f32(host[(size_t) row * D + j])) * f32(host[(size_t) row * D + j]);
            const float a = float((12.0 + 4 * t) / std::max(norm, 1e-20));
            for (int j = 0; j < D; ++j) x[j] += a * f32(host[(size_t) row * D + j]);
        }
    }
    float* dx = dmalloc<float>((size_t) 8 * D);
    float* y16 = dmalloc<float>((size_t) 8 * V);
    float* y8 = dmalloc<float>((size_t) 8 * V);
    float* y4 = dmalloc<float>((size_t) 8 * V);
    void* scratch = nullptr;
    const size_t sb = nd::scratch_bytes(D, false);
    ck(cudaMalloc(&scratch, sb), "scratch");
    if (!nd::prepare(s, D, scratch, sb, false)) throw std::runtime_error("native_dense prepare");
    x3::Mat m;
    m.fmt = nd::kFmtBF16, m.raw = dh, m.k = D, m.n = V;
    auto bf16_head = [&](const float* x, int nc, float* y) {
        x3::MvJob j;
        j.m = &m, j.xf = x, j.y = y, j.ncol = nc, j.xf_cs = (size_t) D, j.y_cs = (size_t) V;
        if (!nd::mv(j, s)) throw std::runtime_error("bf16 mv");
    };
    // each kernel against a double-precision sum of its own weights (sampled rows): its arithmetic
    {
        ck(copy_sync(dx, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice, s), "copy");
        bf16_head(dx, 3, y16);
        if (!hl::mv(h8, dx, D, 3, y8, V, s) || !hl::mv(h4, dx, D, 3, y4, V, s)) throw std::runtime_error("head_lowp mv");
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) 3 * V), b((size_t) 3 * V), c4((size_t) 3 * V);
        ck(copy_sync(a.data(), y16, a.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(b.data(), y8, b.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(c4.data(), y4, c4.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        std::vector<uint8_t> w8(hl::weight_bytes(1, V, D)), s8(hl::scale_bytes(1, V, D)), w4(hl::weight_bytes(2, V, D)), s4(hl::scale_bytes(2, V, D));
        ck(copy_sync(w8.data(), h8.w, w8.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(s8.data(), h8.sc, s8.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(w4.data(), h4.w, w4.size(), cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(s4.data(), h4.sc, s4.size(), cudaMemcpyDeviceToHost, s), "copy");
        double e16 = 0, e8 = 0, e4 = 0, n16 = 0, n8 = 0, n4 = 0;
        for (int it = 0; it < 600; ++it) {
            const int r = int(rng() % V), c = it % 3;
            double s16 = 0, s8d = 0, s4d = 0, a16 = 0, a8 = 0, a4 = 0;
            for (int j = 0; j < D; ++j) {
                const double xv = xr[(size_t) c * D + j];
                const double w16 = f32(host[(size_t) r * D + j]), w8d = hl::weight_host(h8, w8.data(), s8.data(), r, j),
                             w4d = hl::weight_host(h4, w4.data(), s4.data(), r, j);
                s16 += w16 * xv, s8d += w8d * xv, s4d += w4d * xv;
                a16 += std::fabs(w16 * xv), a8 += std::fabs(w8d * xv), a4 += std::fabs(w4d * xv);
            }
            e16 = std::max(e16, std::fabs(a[(size_t) c * V + r] - s16) / a16);
            e8 = std::max(e8, std::fabs(b[(size_t) c * V + r] - s8d) / a8);
            e4 = std::max(e4, std::fabs(c4[(size_t) c * V + r] - s4d) / a4);
            n16 += a16, n8 += a8, n4 += a4;
        }
        g_head_checks_ok = g_head_checks_ok && e16 < 1e-4 && e8 < 1e-4 && e4 < 1e-4;
        std::printf("   kernel arithmetic, largest |result - exact sum of its own weights| / sum |w x| over 600 sampled rows: "
                    "BF16 %.2e, FP8 %.2e, NVFP4 %.2e (FP32 rounding only: about 1e-6)\n", e16, e8, e4);
    }
    // logits against BF16
    for (int kind = 0; kind < 2; ++kind) {
        ck(copy_sync(dx, (kind ? xp : xr).data(), xr.size() * 4, cudaMemcpyHostToDevice, s), "copy");
        bf16_head(dx, 8, y16);
        if (!hl::mv(h8, dx, D, 8, y8, V, s) || !hl::mv(h4, dx, D, 8, y4, V, s)) throw std::runtime_error("head_lowp mv");
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> a((size_t) 8 * V), b((size_t) 8 * V), c4((size_t) 8 * V);
        ck(copy_sync(a.data(), y16, a.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(b.data(), y8, b.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        ck(copy_sync(c4.data(), y4, c4.size() * 4, cudaMemcpyDeviceToHost, s), "copy");
        for (int f = 0; f < 2; ++f) {
            const Stats st = compare_logits(a, f ? c4 : b, V, 8);
            std::printf("   %s columns, %s against BF16: logits relative RMS %.4f%%, largest |difference| %.4f, top-1 agree %.0f%%, "
                        "top-16 overlap %.1f%%, KL %.2e nats\n", kind ? "peaked" : "random", f ? "NVFP4" : "FP8  ", 100 * st.rel_rms,
                        st.max_abs, 100 * st.top1, 100 * st.top16, st.kl);
        }
    }
    // timing
    std::printf("== head pass, ms (events, 30 back to back): today's BF16 kernels against the low-precision heads\n");
    std::printf("%5s %10s %10s %10s %10s %10s   %s\n", "cols", "BF16", "FP8 old", "FP8 shape1", "NVFP4 old", "NVFP4 shape1", "shape bits");
    ck(copy_sync(dx, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice, s), "copy");
    for (int nc = 1; nc <= 8; ++nc) {
        hl::set_shape_mode(0);
        const double t16 = time_ms(s, 30, [&] { bf16_head(dx, nc, y16); });
        const double t8 = time_ms(s, 30, [&] { hl::mv(h8, dx, D, nc, y8, V, s); });
        const double t4 = time_ms(s, 30, [&] { hl::mv(h4, dx, D, nc, y4, V, s); });
        std::vector<float> baseline8(size_t(nc)*V), baseline4(baseline8.size()), candidate(baseline8.size());
        ck(copy_sync(baseline8.data(), y8, baseline8.size()*4, cudaMemcpyDeviceToHost, s), "shape baseline FP8");
        ck(copy_sync(baseline4.data(), y4, baseline4.size()*4, cudaMemcpyDeviceToHost, s), "shape baseline NVFP4");
        hl::set_shape_mode(1);
        const double t8new = time_ms(s, 30, [&] { if (!hl::mv(h8, dx, D, nc, y8, V, s)) throw std::runtime_error("head shape1 FP8"); });
        ck(copy_sync(candidate.data(), y8, candidate.size()*4, cudaMemcpyDeviceToHost, s), "shape candidate FP8");
        bool same = std::memcmp(candidate.data(), baseline8.data(), candidate.size()*4) == 0;
        const double t4new = time_ms(s, 30, [&] { if (!hl::mv(h4, dx, D, nc, y4, V, s)) throw std::runtime_error("head shape1 NVFP4"); });
        ck(copy_sync(candidate.data(), y4, candidate.size()*4, cudaMemcpyDeviceToHost, s), "shape candidate NVFP4");
        same &= std::memcmp(candidate.data(), baseline4.data(), candidate.size()*4) == 0;
        g_head_checks_ok &= same;
        std::printf("%5d %10.3f %10.3f %10.3f %10.3f %10.3f   %s\n", nc, t16, t8, t8new, t4, t4new, same ? "identical" : "DIFFERENT");
    }
    hl::set_shape_mode(0);
    if (sweep) {
        std::printf("== low-precision kernel shapes (rows a warp x warps a block), ms\n");
        const int shapes[][2] = {{1, 4}, {2, 4}, {4, 4}, {8, 4}, {2, 8}, {4, 8}, {4, 2}};
        for (int f = 1; f <= 2; ++f)
            for (int nc : {1, 2, 4, 8}) {
                std::printf("  %s %d cols:", f == 1 ? "FP8  " : "NVFP4", nc);
                for (auto& sh : shapes) {
                    const hl::Head& hh = f == 1 ? h8 : h4;
                    const double t = time_ms(s, 20, [&] { if (!hl::mv_shape(hh, dx, D, nc, y8, V, sh[0], sh[1], s)) throw std::runtime_error("shape"); });
                    std::printf("  %dx%d %.3f", sh[0], sh[1], t);
                }
                std::printf("\n");
            }
    }
    nd::release(s);
    hl::release(h8);
    hl::release(h4);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        bool do_router = false, do_head = false, sweep = false, random_head = false;
        int dense_kernel = 2;
        std::string pack = "/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-2/native-pack";
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--router") do_router = true;
            else if (a == "--head") do_head = true;
            else if (a == "--sweep") sweep = true;
            else if (a == "--random-head") random_head = true;
            else if (a == "--pack" && i + 1 < argc) pack = argv[++i];
            else if (a == "--dense-kernel" && i + 1 < argc) dense_kernel = std::stoi(argv[++i]);
            else throw std::invalid_argument("unknown option " + a);
        }
        if (!do_router && !do_head) do_router = do_head = true;
        if (dense_kernel < 0 || dense_kernel > 3) throw std::invalid_argument("dense kernel must be 0..3");
        cudaStream_t s;
        ck(cudaSetDevice(0), "device");
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        if (do_router) router(s, dense_kernel);
        if (do_head) head(s, pack, random_head, sweep);
        std::printf("== gates: router bits %s; head quantizer and arithmetic checks %s\n", g_router_bits_ok ? "identical" : "DIFFERENT",
                    g_head_checks_ok ? "pass" : "FAIL");
        return !g_router_bits_ok ? 2 : !g_head_checks_ok ? 3 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "router_head_bench: %s\n", e.what());
        return 1;
    }
}
