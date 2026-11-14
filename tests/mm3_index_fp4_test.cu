// tests/mm3_index_fp4_test.cu - --index-keys fp4 (mm3_kernels.cuh: index_cache_bytes, index_norm_rope, block_scores)
// and the sparse attention's K / V formats, without a model (synthetic keys: Gaussian, 6 outlier dims x20 with fixed
// signs, a topic direction a 64-position segment, normed and RoPE'd by the engine's own kernels).
// Usage: mm3_index_fp4_test [device]   (run on a GPU in an engine-start gap, not beside serving: up to ~0.8 GB)
//  a. The FP4 index cache's bytes (index_norm_rope, fmt kKvFp4, rotated) = a host NVFP4 quantization of the rotated
//     keys - scale E4M3 (round to nearest even, saturated) of max(amax / 6, 2^-9) a group of 16, codes E2M1 (round to
//     nearest even) of value / scale - from the keys' F32 values (norm, RoPE: qk_norm_rope's query path, the same
//     norm_rope) rotated on the host in had_block's order; and = qk_norm_rope's FP4 K rows (put_row) of the same keys;
//     the BF16 index cache = those F32 values rounded; every decoded value (code x scale) a BF16 value.
//  b. block_scores over the FP4 cache (rotated queries) = block_scores over a BF16 cache holding the decoded keys, bit
//     for bit, at T = 1 (the token kernel), 8 and 16 (the tiled kernel) and 64 (the wide kernel), 4K and 32K positions;
//     and a query's FP4 scores the same bits from all three kernels.
//  c. The kept blocks (select_blocks, 16 a query and head) with FP4 keys (rotated, and unrotated) against the exact
//     path's (BF16 keys, unrotated): per query and head the blocks in common, and how often the set is the same.
//  d. Timings (measured: cudaEvent over CUDA graphs of many calls, warm): block_scores and index_norm_rope, BF16 vs FP4,
//     T = 1, 8, 16, the queries at the context's end (every block scored), 4K and 32K positions.
//  e. The sparse attention (k::sparse_attn, the engine's 16 splits) over K / V BF16, INT8 rotated (the built-in) and FP4
//     rotated, 32K positions, T = 1, 8, 16, 16 kept blocks: timings, and the INT8 / FP4 outputs' error against BF16's.
// Exit 0: every exactness check (a, b) holds; 1: one failed; 2: a CUDA error.
#include "mm3_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace k = mm3::k;

namespace {

constexpr int Hi = 4, kRot = 64, kBlk = 128, kTopk = 16, kChunk = 4096;
constexpr float kTheta = 5.0e6f, kEps = 1e-6f;
constexpr int kTopics = 32, kSeg = 64, kNOut = 6;
constexpr int kLayersInTurn = 60;   // (d: the index caches read in turn, one a layer, as a token reads them)
constexpr int kCalls = 200, kLaunches = 20;

int g_fail = 0;
std::mt19937_64 g_rng(20261114);
std::vector<std::string> g_out[5];   // the sections' lines (a .. e), printed at the end

#define CK(x)                                                                                        \
    do {                                                                                             \
        const cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));   \
            std::exit(2);                                                                            \
        }                                                                                            \
    } while (0)

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T) + 16));
    return p;
}
template <typename T>
T* dev(const std::vector<T>& v) {
    T* d = dalloc<T>(v.size());
    CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <typename T>
std::vector<T> host(const T* d, size_t n) {
    std::vector<T> v(n);
    CK(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
std::string fmt(const char* f, ...) {
    char b[512];
    va_list a;
    va_start(a, f);
    std::vsnprintf(b, sizeof b, f, a);
    va_end(a);
    return b;
}
void out(int sec, const std::string& s) { g_out[sec].push_back(s); }

// ---- the host's own number formats (independent of the kernels' code)
uint32_t fbits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float e4m3_f(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float a = e == 0 ? m * std::ldexp(1.0f, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return (b & 0x80) ? -a : a;
}
float e2m1_f(int c) {
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}
// E4M3 of x >= 0: the nearest finite value, ties to the even code, saturated at 448
uint8_t e4m3_rne_sat(float x) {
    static std::vector<double> val = [] {
        std::vector<double> v(127);
        for (int c = 0; c < 127; ++c) v[(size_t) c] = e4m3_f((uint8_t) c);
        return v;
    }();
    if (!(x < 448.0f)) return 126;
    const int hi = (int) (std::lower_bound(val.begin(), val.end(), (double) x) - val.begin());
    if (val[(size_t) hi] == (double) x || hi == 0) return (uint8_t) hi;
    const int lo = hi - 1;
    const double dl = (double) x - val[(size_t) lo], dh = val[(size_t) hi] - (double) x;
    return (uint8_t) (dl < dh ? lo : dh < dl ? hi : (lo % 2 == 0 ? lo : hi));
}
// E2M1 of q (|q| <= 6): the nearest magnitude, ties to the even code; bit 3 the sign (of -0 too)
int e2m1_rne(float q) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double a = std::fabs((double) q);
    int best = 0;
    for (int c = 1; c < 8; ++c) {
        const double dc = std::fabs(a - mag[c]), db = std::fabs(a - mag[best]);
        if (dc < db || (dc == db && c % 2 == 0)) best = c;
    }
    return best | (std::signbit(q) ? 8 : 0);
}
uint16_t bf16_rne(float f) {
    uint32_t u = fbits(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// the 128-point Sylvester transform over sqrt(128) in F32, in had_block's order (butterflies m = 1 .. 64, then the scale)
void had_f32(float* v) {
    for (int m = 1; m < 128; m <<= 1)
        for (int i = 0; i < 128; ++i)
            if (!(i & m)) {
                const float a = v[i], b = v[i | m];
                v[i] = a + b;
                v[i | m] = a - b;
            }
    for (int i = 0; i < 128; ++i) v[i] = v[i] * 0.088388347648f;
}
// the host NVFP4 row of 128 values: 64 code bytes (value 2i in the low nibble), 8 E4M3 scale bytes
void nvfp4_row(const float* v, uint8_t* code, uint8_t* sc) {
    int c[128];
    for (int g = 0; g < 8; ++g) {
        float am = 0.0f;
        for (int i = 0; i < 16; ++i) am = std::max(am, std::fabs(v[g * 16 + i]));
        const uint8_t sb = e4m3_rne_sat(std::max(am / 6.0f, 0.001953125f));
        const float s = e4m3_f(sb);
        sc[g] = sb;
        for (int i = 0; i < 16; ++i) c[g * 16 + i] = e2m1_rne(std::min(std::max(v[g * 16 + i] / s, -6.0f), 6.0f));
    }
    for (int i = 0; i < 64; ++i) code[i] = (uint8_t) (c[2 * i] | (c[2 * i + 1] << 4));
}

// ---- synthetic rows
struct Gen {
    int outl[kNOut];
    float sgn[kNOut];
    std::vector<float> topic;   // [kTopics][128]
};
Gen make_gen() {
    Gen g;
    std::vector<int> dims(128);
    for (int i = 0; i < 128; ++i) dims[(size_t) i] = i;
    std::shuffle(dims.begin(), dims.end(), g_rng);
    for (int i = 0; i < kNOut; ++i) {
        g.outl[i] = dims[(size_t) i];
        g.sgn[i] = (g_rng() & 1) ? 1.0f : -1.0f;
    }
    std::normal_distribution<float> n01(0.0f, 1.0f);
    g.topic.resize((size_t) kTopics * 128);
    for (auto& x : g.topic) x = n01(g_rng);
    return g;
}
// 128 raw values: Gaussian, plus amp x topic z (z < 0: none), the outlier dims 20 x (their sign + a quarter Gaussian)
void raw_row(const Gen& g, int z, float amp, float* x) {
    std::normal_distribution<float> n01(0.0f, 1.0f);
    for (int d = 0; d < 128; ++d) x[d] = n01(g_rng) + (z >= 0 ? amp * g.topic[(size_t) z * 128 + d] : 0.0f);
    for (int i = 0; i < kNOut; ++i) x[g.outl[i]] = 20.0f * (g.sgn[i] + 0.25f * n01(g_rng));
}

// the mean time of one call in microseconds: calls f(0) .. f(calls - 1) captured into a CUDA graph (the engine's decode
// runs its kernels in graphs), launched warm, then timed over kLaunches launches
double graph_us(cudaStream_t s, const std::function<void(int)>& f, int calls = kCalls) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < calls; ++i) f(i);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&ge, g, 0));
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int l = 0; l < kLaunches; ++l) CK(cudaGraphLaunch(ge, s));
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaGraphExecDestroy(ge));
    CK(cudaGraphDestroy(g));
    return 1000.0 * ms / ((double) calls * kLaunches);
}

struct Ctx {
    cudaStream_t s = nullptr;
    float *one = nullptr, *inv = nullptr;
    size_t l2 = 0;
};

// the index cache ic (format fmt, rot) of positions 0 .. ctx - 1 from the raw keys d_ik [ctx][128], as the engine
// writes it (index_norm_rope, chunks of kChunk tokens; the queries' scratch d_iq unused after)
void fill_index(const Ctx& C, uint16_t* ic, int fmt, bool rot, const float* d_ik, const int* d_pos, int ctx,
                float* d_iq) {
    for (int p0 = 0; p0 < ctx; p0 += kChunk) {
        const int n = std::min(kChunk, ctx - p0);
        k::index_norm_rope(d_iq, d_ik + (size_t) p0 * 128, C.one, C.one, d_pos + p0, n, Hi, 128, kRot, C.inv, kEps, ic,
                           C.s, fmt, rot);
    }
    CK(cudaStreamSynchronize(C.s));
    CK(cudaGetLastError());
}

// ---------------------------------------------------------------- one context length: a, b, c, d
void run_ctx(const Ctx& C, int ctx) {
    const int nblk = ctx / kBlk;
    std::fprintf(stderr, "ctx %d: keys\n", ctx);
    const Gen gen = make_gen();
    std::vector<int> seg_topic((size_t) (ctx + kSeg - 1) / kSeg);
    for (auto& z : seg_topic) z = (int) (g_rng() % kTopics);
    std::vector<float> ik((size_t) ctx * 128);
    for (int p = 0; p < ctx; ++p) raw_row(gen, seg_topic[(size_t) p / kSeg], 1.5f, &ik[(size_t) p * 128]);
    std::vector<int> pos_all((size_t) ctx);
    for (int p = 0; p < ctx; ++p) pos_all[(size_t) p] = p;
    float* d_ik = dev(ik);
    int* d_pos = dev(pos_all);
    float* d_iq = dalloc<float>((size_t) kChunk * Hi * 128);
    CK(cudaMemset(d_iq, 0, (size_t) kChunk * Hi * 128 * 4));
    const size_t bB = k::index_cache_bytes(k::kKvBf16, (size_t) ctx), bF = k::index_cache_bytes(k::kKvFp4, (size_t) ctx);
    uint16_t* icB = (uint16_t*) dalloc<uint8_t>(bB);    // BF16, unrotated: the exact path
    uint16_t* icF = (uint16_t*) dalloc<uint8_t>(bF);    // FP4, rotated
    uint16_t* icF0 = (uint16_t*) dalloc<uint8_t>(bF);   // FP4, unrotated
    uint16_t* icD = (uint16_t*) dalloc<uint8_t>(bB);    // BF16 holding icF's decoded (rotated) keys
    fill_index(C, icB, k::kKvBf16, false, d_ik, d_pos, ctx, d_iq);
    fill_index(C, icF, k::kKvFp4, true, d_ik, d_pos, ctx, d_iq);
    fill_index(C, icF0, k::kKvFp4, false, d_ik, d_pos, ctx, d_iq);
    const std::vector<uint8_t> hF = host((const uint8_t*) icF, bF);

    // ---- a. the FP4 cache's bytes
    {
        std::fprintf(stderr, "ctx %d: a\n", ctx);
        // the keys' F32 values (norm, RoPE): qk_norm_rope's query path - norm_rope, as index_kernel's key path
        std::vector<float> xf((size_t) ctx * 128);
        float* d_q = dalloc<float>((size_t) kChunk * 128);
        for (int p0 = 0; p0 < ctx; p0 += kChunk) {
            const int n = std::min(kChunk, ctx - p0);
            CK(cudaMemcpyAsync(d_q, d_ik + (size_t) p0 * 128, (size_t) n * 128 * 4, cudaMemcpyDeviceToDevice, C.s));
            k::qk_norm_rope(d_q, nullptr, nullptr, C.one, C.one, d_pos + p0, n, 1, 0, 128, kRot, C.inv, kEps, nullptr,
                            nullptr, C.s);
            CK(cudaMemcpyAsync(&xf[(size_t) p0 * 128], d_q, (size_t) n * 128 * 4, cudaMemcpyDeviceToHost, C.s));
        }
        // qk_norm_rope's FP4 K rows (put_row, rotated) of the same keys: one KV head, V the same rows unrotated
        uint8_t* kvb = dalloc<uint8_t>(k::kv_bytes(k::kKvFp4, k::kKvFp4, (size_t) ctx));
        uint16_t *kc16 = nullptr, *vc16 = nullptr;
        k::KvQ kq;
        k::kv_carve(kvb, (size_t) ctx, k::kKvFp4, k::kKvFp4, 1, kc16, vc16, kq);
        for (int p0 = 0; p0 < ctx; p0 += kChunk) {
            const int n = std::min(kChunk, ctx - p0);
            k::qk_norm_rope(nullptr, d_ik + (size_t) p0 * 128, d_ik + (size_t) p0 * 128, C.one, C.one, d_pos + p0, n,
                            0, 1, 128, kRot, C.inv, kEps, nullptr, nullptr, C.s, 0, kq);
        }
        CK(cudaStreamSynchronize(C.s));
        CK(cudaGetLastError());
        const std::vector<uint8_t> kcode = host(kq.kc, (size_t) ctx * 64), ksc = host(kq.ks, (size_t) ctx * 8);
        const std::vector<uint16_t> hB = host(icB, (size_t) ctx * 128);
        size_t d_put = 0, d_ref = 0, d_bf = 0, d_dec = 0, shown = 0;
        for (int p = 0; p < ctx; ++p) {
            const uint8_t* fc = &hF[k::ix_code_off(p)];
            const uint8_t* fs = &hF[k::ix_scale_off(p)];
            for (int i = 0; i < 64; ++i) d_put += fc[i] != kcode[(size_t) p * 64 + i];
            for (int i = 0; i < 8; ++i) d_put += fs[i] != ksc[(size_t) p * 8 + i];
            float v[128];
            std::memcpy(v, &xf[(size_t) p * 128], sizeof v);
            for (int d = 0; d < 128; ++d) d_bf += hB[(size_t) p * 128 + d] != bf16_rne(v[d]);
            had_f32(v);
            uint8_t rc[64], rs[8];
            nvfp4_row(v, rc, rs);
            for (int i = 0; i < 72; ++i) {
                const bool bad = i < 64 ? fc[i] != rc[i] : fs[i - 64] != rs[i - 64];
                if (!bad) continue;
                ++d_ref;
                if (shown++ < 6) {   // (the value, its scale, the quotient: is it at a rounding boundary?)
                    const int g = i < 64 ? (2 * i) / 16 : i - 64;
                    const float s = e4m3_f(rs[g]);
                    out(0, fmt("    mismatch ctx %d position %d byte %d: device %02x host %02x (group %d scale %g, "
                               "values / scale %.9g %.9g)", ctx, p, i, i < 64 ? fc[i] : fs[i - 64], i < 64 ? rc[i] : rs[i - 64],
                               g, s, i < 64 ? v[2 * i] / s : 0.0, i < 64 ? v[2 * i + 1] / s : 0.0));
                }
            }
            for (int d = 0; d < 128; ++d) {
                const float x = e2m1_f((fc[d / 2] >> (4 * (d & 1))) & 15) * e4m3_f(fs[d / 16]);
                d_dec += (fbits(x) & 0xffffu) != 0u;
            }
        }
        const size_t nb = (size_t) ctx * 72, nv = (size_t) ctx * 128;
        auto line = [&](const char* what, size_t bad, size_t n) {
            out(0, fmt("  ctx %6d  %-62s %8zu of %9zu differ  %s", ctx, what, bad, n, bad ? "FAIL" : "ok"));
            if (bad) ++g_fail;
        };
        line("FP4 cache bytes vs qk_norm_rope's FP4 K rows (put_row)", d_put, nb);
        line("FP4 cache bytes vs the host NVFP4 reference (rotated keys)", d_ref, nb);
        line("BF16 cache vs the keys' F32 values rounded to BF16", d_bf, nv);
        line("FP4 decoded values (code x scale) not BF16 values", d_dec, nv);
        CK(cudaFree(d_q));
        CK(cudaFree(kvb));
    }

    // the BF16 cache of the decoded keys (icD), for b
    {
        std::vector<uint16_t> dec((size_t) ctx * 128);
        for (int p = 0; p < ctx; ++p)
            for (int d = 0; d < 128; ++d) {
                const uint8_t* fc = &hF[k::ix_code_off(p)];
                const uint8_t* fs = &hF[k::ix_scale_off(p)];
                dec[(size_t) p * 128 + d] = (uint16_t) (fbits(e2m1_f((fc[d / 2] >> (4 * (d & 1))) & 15) * e4m3_f(fs[d / 16])) >> 16);
            }
        CK(cudaMemcpy(icD, dec.data(), dec.size() * 2, cudaMemcpyHostToDevice));
    }

    // ---- b. block_scores FP4 = BF16 over the decoded keys, bit for bit; a query's FP4 scores the same from each kernel
    {
        std::fprintf(stderr, "ctx %d: b\n", ctx);
        constexpr int kTmax = 64;
        std::vector<float> qpool((size_t) kTmax * Hi * 128);   // the token at position p: row ctx - 1 - p
        for (int r = 0; r < kTmax; ++r)
            for (int h = 0; h < Hi; ++h) raw_row(gen, (int) (g_rng() % kTopics), 2.0f, &qpool[((size_t) r * Hi + h) * 128]);
        float* dq = dalloc<float>((size_t) kTmax * Hi * 128);
        int* dp = dalloc<int>(kTmax);
        float* bsF = dalloc<float>((size_t) Hi * kTmax * nblk);
        float* bsD = dalloc<float>((size_t) Hi * kTmax * nblk);
        const int Ts[4] = {1, 8, 16, 64};
        std::vector<std::vector<float>> rowsF(4);   // [T idx][Hi][T][nblk]
        for (int ti = 0; ti < 4; ++ti) {
            const int T = Ts[ti];
            std::vector<float> q((size_t) T * Hi * 128);
            std::vector<int> pos((size_t) T);
            for (int t = 0; t < T; ++t) {
                pos[(size_t) t] = ctx - T + t;
                std::memcpy(&q[(size_t) t * Hi * 128], &qpool[(size_t) (T - 1 - t) * Hi * 128], (size_t) Hi * 128 * 4);
            }
            CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dp, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
            // (the tokens' keys written again at their positions: the same bytes - the queries rotated and rounded)
            k::index_norm_rope(dq, d_ik + (size_t) (ctx - T) * 128, C.one, C.one, dp, T, Hi, 128, kRot, C.inv, kEps, icF,
                               C.s, k::kKvFp4, true);
            CK(cudaMemsetAsync(bsF, 0xff, (size_t) Hi * T * nblk * 4, C.s));
            CK(cudaMemsetAsync(bsD, 0xff, (size_t) Hi * T * nblk * 4, C.s));
            k::block_scores(dq, icF, dp, T, Hi, nblk, kBlk, bsF, C.s, k::kKvFp4);
            k::block_scores(dq, icD, dp, T, Hi, nblk, kBlk, bsD, C.s, k::kKvBf16);
            CK(cudaStreamSynchronize(C.s));
            CK(cudaGetLastError());
            const std::vector<float> f = host(bsF, (size_t) Hi * T * nblk), g = host(bsD, (size_t) Hi * T * nblk);
            size_t bad = 0, n = 0;
            for (int h = 0; h < Hi; ++h)
                for (int t = 0; t < T; ++t)
                    for (int b = 0; b <= pos[(size_t) t] / kBlk; ++b, ++n) {
                        const size_t i = ((size_t) h * T + t) * nblk + b;
                        bad += fbits(f[i]) != fbits(g[i]);
                    }
            const char* kern = T == 1 ? "token kernel" : T >= 32 ? "wide kernel" : "tiled kernel";
            out(1, fmt("  ctx %6d  T %2d (%-12s)  FP4 vs BF16 over the decoded keys: %8zu of %8zu scores differ  %s",
                       ctx, T, kern, bad, n, bad ? "FAIL" : "ok"));
            if (bad) ++g_fail;
            rowsF[(size_t) ti] = f;
        }
        // the same query (position) in each run: the T = 64 run's row (wide kernel) against T = 1, 8, 16's, and T = 1's
        // (token kernel) against T = 8's (tiled)
        auto cmp_rows = [&](int ta, int tb, const char* what) {
            const int TA = Ts[ta], TB = Ts[tb];
            size_t bad = 0, n = 0;
            for (int t = 0; t < TA; ++t) {
                const int p = ctx - TA + t, tb_row = p - (ctx - TB);
                for (int h = 0; h < Hi; ++h)
                    for (int b = 0; b <= p / kBlk; ++b, ++n)
                        bad += fbits(rowsF[(size_t) ta][((size_t) h * TA + t) * nblk + b]) !=
                               fbits(rowsF[(size_t) tb][((size_t) h * TB + tb_row) * nblk + b]);
            }
            out(1, fmt("  ctx %6d  %-58s %8zu of %8zu scores differ  %s", ctx, what, bad, n, bad ? "FAIL" : "ok"));
            if (bad) ++g_fail;
        };
        cmp_rows(0, 1, "FP4, the same queries: token kernel vs tiled kernel");
        cmp_rows(0, 3, "FP4, the same queries: token kernel vs wide kernel");
        cmp_rows(1, 3, "FP4, the same queries: tiled kernel (T 8) vs wide kernel");
        cmp_rows(2, 3, "FP4, the same queries: tiled kernel (T 16) vs wide kernel");
        for (void* p : {(void*) dq, (void*) dp, (void*) bsF, (void*) bsD}) CK(cudaFree(p));
    }

    // ---- c. the kept blocks against the exact path (BF16 keys, unrotated)
    {
        std::fprintf(stderr, "ctx %d: c\n", ctx);
        const int NQ = 512;
        std::vector<int> pos((size_t) NQ);
        for (auto& p : pos) p = 16 * kBlk + (int) (g_rng() % (uint64_t) (ctx - 16 * kBlk));
        std::sort(pos.begin(), pos.end());
        std::vector<float> q((size_t) NQ * Hi * 128), kq((size_t) NQ * 128);
        for (int t = 0; t < NQ; ++t) {
            for (int h = 0; h < Hi; ++h) raw_row(gen, (int) (g_rng() % kTopics), 2.0f, &q[((size_t) t * Hi + h) * 128]);
            std::memcpy(&kq[(size_t) t * 128], &ik[(size_t) pos[(size_t) t] * 128], 128 * 4);   // (its own key again)
        }
        int* dp = dev(pos);
        float* dk = dev(kq);
        float* dq = dalloc<float>(q.size());
        float* bs = dalloc<float>((size_t) Hi * NQ * nblk);
        int* kept = dalloc<int>((size_t) Hi * NQ * kTopk);
        auto run = [&](uint16_t* ic, int f, bool rot) {
            CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            k::index_norm_rope(dq, dk, C.one, C.one, dp, NQ, Hi, 128, kRot, C.inv, kEps, ic, C.s, f, rot);
            k::block_scores(dq, ic, dp, NQ, Hi, nblk, kBlk, bs, C.s, f);
            k::select_blocks(bs, dp, NQ, Hi, nblk, kBlk, kTopk, kept, C.s);
            CK(cudaStreamSynchronize(C.s));
            CK(cudaGetLastError());
            return host(kept, (size_t) Hi * NQ * kTopk);
        };
        const std::vector<int> ref = run(icB, k::kKvBf16, false);
        struct V {
            const char* name;
            uint16_t* ic;
            bool rot;
        };
        for (const V& v : {V{"FP4 keys, rotated (the option's built-in)", icF, true}, V{"FP4 keys, unrotated (--kv-rot off)", icF0, false}}) {
            const std::vector<int> got = run(v.ic, k::kKvFp4, v.rot);
            double sum = 0;
            int mn = kTopk, same = 0, ge15 = 0;
            for (int i = 0; i < Hi * NQ; ++i) {
                std::vector<int> a(&ref[(size_t) i * kTopk], &ref[(size_t) i * kTopk] + kTopk);
                std::vector<int> b(&got[(size_t) i * kTopk], &got[(size_t) i * kTopk] + kTopk);
                std::sort(a.begin(), a.end());
                std::sort(b.begin(), b.end());
                std::vector<int> c;
                std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(c));
                const int m = (int) std::count_if(c.begin(), c.end(), [](int x) { return x >= 0; });
                sum += m;
                mn = std::min(mn, m);
                same += m == kTopk;
                ge15 += m >= kTopk - 1;
            }
            out(2, fmt("  %6d  %-42s %6.2f of 16 %6d %8.1f%% %8.1f%%", ctx, v.name, sum / (Hi * NQ), mn,
                       100.0 * same / (Hi * NQ), 100.0 * ge15 / (Hi * NQ)));
        }
        for (void* p : {(void*) dp, (void*) dk, (void*) dq, (void*) bs, (void*) kept}) CK(cudaFree(p));
    }

    // ---- d. timings: block_scores and index_norm_rope, BF16 vs FP4
    {
        std::fprintf(stderr, "ctx %d: d\n", ctx);
        // kLayersInTurn copies of each cache (the same keys), read in turn as a token's layers read theirs
        std::vector<uint16_t*> LB(kLayersInTurn), LF(kLayersInTurn);
        uint8_t* bigB = dalloc<uint8_t>(bB * kLayersInTurn);
        uint8_t* bigF = dalloc<uint8_t>(bF * kLayersInTurn);
        for (int l = 0; l < kLayersInTurn; ++l) {
            LB[(size_t) l] = (uint16_t*) (bigB + (size_t) l * bB);
            LF[(size_t) l] = (uint16_t*) (bigF + (size_t) l * bF);
            CK(cudaMemcpy(LB[(size_t) l], icB, bB, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(LF[(size_t) l], icF, bF, cudaMemcpyDeviceToDevice));
        }
        float* dqB = dalloc<float>((size_t) 16 * Hi * 128);
        float* dqF = dalloc<float>((size_t) 16 * Hi * 128);
        float* dqT = dalloc<float>((size_t) 16 * Hi * 128);   // (index_norm_rope's timing: its queries in place)
        int* dp = dalloc<int>(16);
        float* bs = dalloc<float>((size_t) Hi * 16 * nblk);
        for (int T : {1, 8, 16}) {
            std::vector<float> q((size_t) T * Hi * 128);
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < Hi; ++h) raw_row(gen, (int) (g_rng() % kTopics), 2.0f, &q[((size_t) t * Hi + h) * 128]);
            std::vector<int> pos((size_t) T);
            for (int t = 0; t < T; ++t) pos[(size_t) t] = ctx - T + t;
            CK(cudaMemcpy(dp, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqB, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dqF, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            const float* kT = d_ik + (size_t) (ctx - T) * 128;
            k::index_norm_rope(dqB, kT, C.one, C.one, dp, T, Hi, 128, kRot, C.inv, kEps, LB[0], C.s, k::kKvBf16, false);
            k::index_norm_rope(dqF, kT, C.one, C.one, dp, T, Hi, 128, kRot, C.inv, kEps, LF[0], C.s, k::kKvFp4, true);
            CK(cudaStreamSynchronize(C.s));
            const double sB1 = graph_us(C.s, [&](int) { k::block_scores(dqB, LB[0], dp, T, Hi, nblk, kBlk, bs, C.s, k::kKvBf16); });
            const double sF1 = graph_us(C.s, [&](int) { k::block_scores(dqF, LF[0], dp, T, Hi, nblk, kBlk, bs, C.s, k::kKvFp4); });
            const double sBn = graph_us(C.s, [&](int i) {
                k::block_scores(dqB, LB[(size_t) (i % kLayersInTurn)], dp, T, Hi, nblk, kBlk, bs, C.s, k::kKvBf16);
            });
            const double sFn = graph_us(C.s, [&](int i) {
                k::block_scores(dqF, LF[(size_t) (i % kLayersInTurn)], dp, T, Hi, nblk, kBlk, bs, C.s, k::kKvFp4);
            });
            CK(cudaMemcpy(dqT, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            const double rB = graph_us(C.s, [&](int i) {
                k::index_norm_rope(dqT, kT, C.one, C.one, dp, T, Hi, 128, kRot, C.inv, kEps,
                                   LB[(size_t) (i % kLayersInTurn)], C.s, k::kKvBf16, false);
            });
            CK(cudaMemcpy(dqT, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
            const double rF = graph_us(C.s, [&](int i) {
                k::index_norm_rope(dqT, kT, C.one, C.one, dp, T, Hi, 128, kRot, C.inv, kEps,
                                   LF[(size_t) (i % kLayersInTurn)], C.s, k::kKvFp4, true);
            });
            auto row = [&](const char* what, double b, double f) {
                out(3, fmt("  %6d  %2d  %-46s %9.2f %9.2f %8.2f", ctx, T, what, b, f, f / b));
            };
            row("index_norm_rope (write keys, queries)", rB, rF);
            row("block_scores, one cache (stays in L2)", sB1, sF1);
            row("block_scores, 60 caches read in turn", sBn, sFn);
        }
        for (void* p : {(void*) bigB, (void*) bigF, (void*) dqB, (void*) dqF, (void*) dqT, (void*) dp, (void*) bs})
            CK(cudaFree(p));
    }
    for (void* p : {(void*) d_ik, (void*) d_pos, (void*) d_iq, (void*) icB, (void*) icF, (void*) icF0, (void*) icD})
        CK(cudaFree(p));
}

// ---------------------------------------------------------------- e. the sparse attention over the K / V formats
void run_attn(const Ctx& C) {
    constexpr int ctx = 32768, Hkv = 4, Hq = 64, kR = 16, kSplits = 16, kTm = 16;
    std::fprintf(stderr, "e: keys\n");
    // K: each KV head its own outlier dims (Gaussian, 6 dims x20); V: Gaussian
    std::vector<float> kk((size_t) ctx * Hkv * 128), vv(kk.size());
    {
        Gen gk[Hkv];
        for (auto& g : gk) g = make_gen();
        std::normal_distribution<float> n01(0.0f, 1.0f);
        for (int p = 0; p < ctx; ++p)
            for (int h = 0; h < Hkv; ++h) {
                raw_row(gk[h], -1, 0.0f, &kk[((size_t) p * Hkv + h) * 128]);
                for (int d = 0; d < 128; ++d) vv[((size_t) p * Hkv + h) * 128 + d] = n01(g_rng);
            }
    }
    float* dk = dev(kk);
    float* dv = dev(vv);
    std::vector<int> pos_all((size_t) ctx);
    for (int p = 0; p < ctx; ++p) pos_all[(size_t) p] = p;
    int* dpa = dev(pos_all);
    struct F {
        const char* name;
        int kf, vf, rot;
        int nl = 1;
        uint8_t* buf = nullptr;
        size_t bytes = 0;
        std::vector<uint16_t*> kc, vc;
        std::vector<k::KvQ> q;
        float* out = nullptr;
    };
    F fs[3] = {{"BF16 (the reference)", k::kKvBf16, k::kKvBf16, 0},
               {"INT8 rotated (built-in)", k::kKvI8, k::kKvI8, 3},
               {"FP4 rotated", k::kKvFp4, k::kKvFp4, 3}};
    for (F& f : fs) {
        const size_t rows = (size_t) ctx * Hkv;
        f.bytes = k::kv_bytes(f.kf, f.vf, rows);
        // (copies of the cache, read in turn with different kept blocks: more than the L2 holds, as a token's layers)
        f.nl = (int) std::min<size_t>(8, std::max<size_t>(1, (size_t) std::ceil(2.5 * (double) C.l2 / (double) f.bytes)));
        f.buf = dalloc<uint8_t>(f.bytes * (size_t) f.nl);
        f.kc.resize((size_t) f.nl);
        f.vc.resize((size_t) f.nl);
        f.q.resize((size_t) f.nl);
        for (int l = 0; l < f.nl; ++l)
            k::kv_carve(f.buf + (size_t) l * f.bytes, rows, f.kf, f.vf, f.rot, f.kc[(size_t) l], f.vc[(size_t) l],
                        f.q[(size_t) l]);
        for (int p0 = 0; p0 < ctx; p0 += 1024)   // (as the prompt path writes them: qk_norm_rope, K normed and RoPE'd)
            k::qk_norm_rope(nullptr, dk + (size_t) p0 * Hkv * 128, dv + (size_t) p0 * Hkv * 128, C.one, C.one, dpa + p0,
                            1024, 0, Hkv, 128, kRot, C.inv, kEps, f.kc[0], f.vc[0], C.s, 0, f.q[0]);
        CK(cudaStreamSynchronize(C.s));
        for (int l = 1; l < f.nl; ++l)
            CK(cudaMemcpy(f.buf + (size_t) l * f.bytes, f.buf, f.bytes, cudaMemcpyDeviceToDevice));
        f.out = dalloc<float>((size_t) kTm * Hq * 128);
    }
    float* part = dalloc<float>(k::attn_part_floats(kTm, Hkv, kSplits));
    float* dq = dalloc<float>((size_t) kTm * Hq * 128);
    int* dp = dalloc<int>(kTm);
    int* dkept = dalloc<int>((size_t) kR * Hkv * kTm * kTopk);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    for (int T : {1, 8, 16}) {
        std::fprintf(stderr, "e: T %d\n", T);
        std::vector<float> q((size_t) T * Hq * 128);
        for (auto& x : q) x = n01(g_rng);
        std::vector<int> pos((size_t) T);
        for (int t = 0; t < T; ++t) pos[(size_t) t] = ctx - T + t;
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp, pos.data(), (size_t) T * 4, cudaMemcpyHostToDevice));
        k::qk_norm_rope(dq, nullptr, nullptr, C.one, C.one, dp, T, Hq, 0, 128, kRot, C.inv, kEps, nullptr, nullptr, C.s);
        // kR sets of kept blocks: the query's own and 15 others, each set its own blocks (together all of them)
        std::vector<int> kept((size_t) kR * Hkv * T * kTopk);
        std::vector<int> perm(ctx / kBlk - 1);
        for (int i = 0; i < (int) perm.size(); ++i) perm[(size_t) i] = i;
        std::shuffle(perm.begin(), perm.end(), g_rng);
        for (int r = 0; r < kR; ++r)
            for (int g = 0; g < Hkv; ++g)
                for (int t = 0; t < T; ++t) {
                    int* o = &kept[(((size_t) r * Hkv + g) * T + t) * kTopk];
                    o[0] = pos[(size_t) t] / kBlk;
                    for (int i = 1; i < kTopk; ++i)
                        o[i] = perm[(size_t) ((r * 15 + (i - 1) + g * 61) % (int) perm.size())];
                }
        CK(cudaMemcpy(dkept, kept.data(), kept.size() * 4, cudaMemcpyHostToDevice));
        const size_t kstride = (size_t) Hkv * T * kTopk;
        const float scale = 1.0f / std::sqrt(128.0f);
        double us1[3], usn[3];
        for (int fi = 0; fi < 3; ++fi) {
            F& f = fs[fi];
            k::sparse_attn(dq, f.kc[0], f.vc[0], dkept, dp, T, Hq, Hkv, kTopk, kBlk, scale, f.out, C.s, f.q[0],
                           k::KvHost{}, kSplits, part);
            us1[fi] = graph_us(C.s, [&](int) {
                k::sparse_attn(dq, f.kc[0], f.vc[0], dkept, dp, T, Hq, Hkv, kTopk, kBlk, scale, f.out, C.s, f.q[0],
                               k::KvHost{}, kSplits, part);
            });
            usn[fi] = graph_us(C.s, [&](int i) {
                const int l = i % f.nl, r = (i / f.nl) % kR;
                k::sparse_attn(dq, f.kc[(size_t) l], f.vc[(size_t) l], dkept + (size_t) r * kstride, dp, T, Hq, Hkv,
                               kTopk, kBlk, scale, f.out, C.s, f.q[(size_t) l], k::KvHost{}, kSplits, part);
            });
            // (the output for the error: layer 0, kept set 0 again - the timing runs overwrote it with other sets')
            k::sparse_attn(dq, f.kc[0], f.vc[0], dkept, dp, T, Hq, Hkv, kTopk, kBlk, scale, f.out, C.s, f.q[0],
                           k::KvHost{}, kSplits, part);
        }
        CK(cudaStreamSynchronize(C.s));
        CK(cudaGetLastError());
        const std::vector<float> ref = host(fs[0].out, (size_t) T * Hq * 128);
        for (int fi = 0; fi < 3; ++fi) {
            const std::vector<float> o = host(fs[fi].out, (size_t) T * Hq * 128);
            double se = 0, sr = 0, worst = 0;
            for (int th = 0; th < T * Hq; ++th) {
                double a = 0, b = 0;
                for (int d = 0; d < 128; ++d) {
                    const double x = (double) o[(size_t) th * 128 + d] - ref[(size_t) th * 128 + d];
                    a += x * x;
                    b += (double) ref[(size_t) th * 128 + d] * ref[(size_t) th * 128 + d];
                }
                se += a;
                sr += b;
                worst = std::max(worst, std::sqrt(a / std::max(b, 1e-30)));
            }
            out(4, fmt("  %2d  %-26s %7.0f %10.2f %10.2f %10.2e %10.2e   (%d %s)", T, fs[fi].name,
                       (double) k::kv_bytes(fs[fi].kf, fs[fi].vf, (size_t) Hkv), us1[fi], usn[fi],
                       std::sqrt(se / std::max(sr, 1e-30)), worst, fs[fi].nl, fs[fi].nl == 1 ? "copy" : "copies"));
        }
    }
    for (F& f : fs) {
        CK(cudaFree(f.buf));
        CK(cudaFree(f.out));
    }
    for (void* p : {(void*) dk, (void*) dv, (void*) dpa, (void*) part, (void*) dq, (void*) dp, (void*) dkept})
        CK(cudaFree(p));
}

}  // namespace

int main(int argc, char** argv) {
    const int devi = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(devi));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, devi));
    Ctx C;
    C.l2 = (size_t) prop.l2CacheSize;
    CK(cudaStreamCreate(&C.s));   // (a blocking stream: the legacy stream's copies and it ordered)
    {
        std::vector<float> ones(128, 1.0f), inv((size_t) kRot / 2);
        k::rope_inv_freq(kTheta, kRot, inv.data());
        C.one = dev(ones);
        C.inv = dev(inv);
    }
    std::printf("mm3_index_fp4_test on CUDA%d %s (L2 %zu MB)\n", devi, prop.name, C.l2 >> 20);
    for (const char* e : {"MAYA_MM3_MSA_TOK", "MAYA_MM3_MSA_WIDE", "MM3_MSA_OLD", "MAYA_MM3_ATTN_WIDE"})
        if (std::getenv(e)) std::printf("  note: %s=%s is set (the kernels named below may not be the ones run)\n", e, std::getenv(e));
    for (int ctx : {4096, 32768}) run_ctx(C, ctx);
    run_attn(C);

    std::printf("\na. FP4 index cache, exactness (index_norm_rope --index-keys fp4, rows rotated)\n");
    for (const auto& s : g_out[0]) std::printf("%s\n", s.c_str());
    std::printf("\nb. block_scores over FP4 keys, exactness (queries rotated; scores compared as F32 bits)\n");
    for (const auto& s : g_out[1]) std::printf("%s\n", s.c_str());
    std::printf("\nc. kept blocks against the exact path (BF16 keys, unrotated), 512 queries x 4 heads at random positions"
                " past block 16\n");
    std::printf("  %6s  %-42s %12s %6s %9s %9s\n", "ctx", "keys", "mean match", "min", "same set", ">= 15");
    for (const auto& s : g_out[2]) std::printf("%s\n", s.c_str());
    std::printf("\nd. indexer timings, measured on %s: microseconds a call (CUDA graphs of %d calls, %d launches, warm);"
                " queries at the context's end, every block scored\n", prop.name, kCalls, kLaunches);
    std::printf("  %6s  %2s  %-46s %9s %9s %8s\n", "ctx", "T", "kernel", "BF16 us", "FP4 us", "FP4/BF16");
    for (const auto& s : g_out[3]) std::printf("%s\n", s.c_str());
    std::printf("\ne. sparse attention over the K / V formats, ctx 32768, 64 query heads over 4 KV heads, 16 kept blocks "
                "of 128, 16 splits (as the engine); times measured on %s (microseconds a call, CUDA graphs of %d calls);\n"
                "   error against BF16's output: relative RMS over all outputs, and the worst query head's relative "
                "error\n", prop.name, kCalls);
    std::printf("  %2s  %-26s %7s %10s %10s %10s %10s\n", "T", "K / V", "B a pos", "us 1 cache", "us in turn",
                "rel RMS", "worst head");
    for (const auto& s : g_out[4]) std::printf("%s\n", s.c_str());
    std::printf("   (B a pos: one layer's K and V bytes a position, 4 KV heads; 'one cache': the same kept blocks every "
                "call, in L2; 'in turn': copies of the cache and 16 sets of kept blocks read in turn, more than L2)\n");
    std::printf("\nmm3_index_fp4_test: %s (%d exactness failures)\n", g_fail ? "FAIL" : "ok", g_fail);
    return g_fail ? 1 : 0;
}
