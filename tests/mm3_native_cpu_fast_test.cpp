// tests/mm3_native_cpu_fast_test.cpp - the CPU lane's reworked native kernel (mm3_native_cpu_fast.hpp), host only.
//  1. quantizers: the vector code against the scalar code, bit for bit, on random and edge-case BF16 blocks; the NVFP4
//     quantizer against the 5090's rule itself (canonical_math.h quantize_a4_codes, input scale 1): same scale word and
//     same codes for every finite block;
//  2. INT8 path: every output bit equal to the shared kernel's (native_cpu_run_multi_batch -> fast_ffn_cpu_batch), on
//     synthetic RG16 experts, for 1 to 12 columns an expert, several experts, the token order, accumulate, and with the
//     L3 prefetch's fixed shares (after a background read through the lane);
//  3. NVFP4 path: every output bit equal to a scalar reference of the same arithmetic (the block products exact in
//     integers, FP32 in the shared kernel's order), and close to the exact NVFP4 x NVFP4 product in double;
//  4. a non-finite input stays non-finite; refused shapes leave the output untouched.
// Usage: mm3_native_cpu_fast_test [workers]
#include "mm3_native_cpu.hpp"
#include "mm3_native_cpu_fast.hpp"
#include "mm3_native_experts.cuh"
#include "ops/common/canonical_math.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fast = mm3::cpu_fast;
namespace nx = strata::mm3::native_experts;
namespace canon = infernix::ops::canon;
constexpr int H = 6144, F = 3072;
using Users = std::vector<std::vector<std::pair<int, float>>>;
using Order = std::vector<std::pair<int, int>>;

static void require(bool b, const std::string& why) { if (!b) throw std::runtime_error(why); }
static uint16_t bf(float f) {
    uint32_t u = std::bit_cast<uint32_t>(f);
    if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x7fffffu)) return uint16_t((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return uint16_t(u >> 16);
}
static float f32(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }

struct Aligned {
    uint8_t* p = nullptr;
    size_t n = 0;
    explicit Aligned(size_t bytes) : n((bytes + 4095) & ~size_t(4095)) {
        p = (uint8_t*) std::aligned_alloc(4096, n);
        require(p != nullptr, "allocation");
        std::memset(p, 0, n);
    }
    ~Aligned() { std::free(p); }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
};

// A synthetic expert: native header + RG16 payload with random codes and E4M3 row scales (zero, subnormal and normal,
// never the NaN word or a sign).
static void make_expert(uint8_t* blob, std::mt19937& rng, float global) {
    nx::Header header{};
    for (int k = 0; k < 3; ++k) { header.weight_global[k] = global * (1.f + 0.25f * k); header.input_global[k] = 1.f; }
    std::memcpy(blob, &header, sizeof header);
    uint8_t* rec = blob + sizeof header;
    const size_t units = size_t(3) * H * F / 256;
    std::uniform_int_distribution<int> byte(0, 255), pick(0, 99), normal(0x18, 0x30), sub(0, 7);
    for (size_t u = 0; u < units; ++u) {
        uint8_t* unit = rec + u * 144;
        for (int k = 0; k < 128; ++k) unit[k] = (uint8_t) byte(rng);
        for (int r = 0; r < 16; ++r) {
            const int c = pick(rng);
            unit[128 + r] = (uint8_t) (c < 2 ? 0 : c < 5 ? sub(rng) : normal(rng));
        }
    }
}

// ---- 1. quantizers

static void random_block(std::mt19937& rng, uint16_t* v, int kind) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> scale_exp(-30.f, 20.f);
    const float s = std::ldexp(1.f, (int) scale_exp(rng));
    for (int k = 0; k < 16; ++k) v[k] = bf(nd(rng) * s);
    if (kind == 1) std::fill(v, v + 16, (uint16_t) 0);
    if (kind == 2) for (int k = 0; k < 16; k += 3) v[k] = 0x8000;                    // (negative zeros)
    if (kind == 3) for (int k = 0; k < 16; ++k) v[k] = uint16_t(k + 1);              // (BF16 subnormals)
    if (kind == 4) v[5] = 0x7f80;                                                    // (+Inf)
    if (kind == 5) v[9] = 0x7fc1;                                                    // (NaN)
    if (kind == 6) for (int k = 0; k < 16; ++k) v[k] = bf((k % 2 ? -1.f : 1.f) * 0.25f * k);  // (E2M1 ties)
    if (kind == 7) for (int k = 0; k < 16; ++k) v[k] = bf(3.0e38f * (k % 2 ? -1 : 1));       // (near the top)
}

static void test_quantizers() {
    std::mt19937 rng(7);
    long finite_blocks = 0;
    for (int it = 0; it < 200000; ++it) {
        uint16_t v[16];
        random_block(rng, v, it % 23 < 8 ? it % 23 : 8);
        for (int act : {fast::kInt8, fast::kNvfp4}) {
            fast::QBlock a{}, b{};
            fast::quant((fast::Act) act, v, 16, &a);
            fast::quant_scalar((fast::Act) act, v, 16, &b);
            require(std::memcmp(a.q, b.q, 16) == 0 && a.off == b.off &&
                    std::bit_cast<uint32_t>(a.scale) == std::bit_cast<uint32_t>(b.scale),
                    "quantizer vector/scalar mismatch, act " + std::to_string(act) + " block " + std::to_string(it));
        }
        bool finite = true;
        for (int k = 0; k < 16; ++k) finite &= std::isfinite(f32(v[k]));
        if (!finite) {
            fast::QBlock a{};
            fast::quant(fast::kNvfp4, v, 16, &a);
            require(std::isnan(a.scale), "a non-finite block must get a NaN scale");
            continue;
        }
        ++finite_blocks;
        const canon::A4Codes ref = canon::quantize_a4_codes(v, 1.0f);   // (the 5090's rule)
        fast::QBlock a{};
        fast::quant(fast::kNvfp4, v, 16, &a);
        const float ref_scale = canon::e4m3_scaled(ref.scale_word) ? canon::e4m3_value(ref.scale_word) * 0.5f : 0.f;
        require(std::bit_cast<uint32_t>(a.scale) == std::bit_cast<uint32_t>(ref_scale), "NVFP4 scale differs from the 5090 rule");
        int sum = 0;
        for (int k = 0; k < 16; ++k) {
            const int c2 = canon::e4m3_scaled(ref.scale_word) ? canon::e2m1_x2(ref.code[k]) : 0;
            require(a.q[k] == c2, "NVFP4 code differs from the 5090 rule at block " + std::to_string(it));
            sum += c2;
        }
        require(a.off == 12 * sum, "NVFP4 offset");
        require(fast::nvfp4_scale_word(0.f) == 0 && ref.scale_word == canon::e4m3_rn_satfinite(
                    [&] { float m = 0; for (int k = 0; k < 16; ++k) m = std::max(m, std::fabs(f32(v[k]))); return m; }() / 6.f),
                "scale word");
    }
    // FP32 -> BF16 against the adapter's conversion, including NaN payloads, Inf and values that round up to Inf
    std::vector<float> x(4096 + 7);
    std::uniform_int_distribution<uint32_t> bits;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::bit_cast<float>(bits(rng));
    x[3] = std::bit_cast<float>(0x7f7fffffu); x[4] = std::bit_cast<float>(0xff7fffffu); x[5] = std::bit_cast<float>(0x7f800001u);
    std::vector<uint16_t> y(x.size());
    fast::to_bf16(x.data(), (int) x.size(), y.data());
    for (size_t i = 0; i < x.size(); ++i) require(y[i] == bf(x[i]), "to_bf16 differs at " + std::to_string(i));
    std::printf("quantizers: vector = scalar on 400000 blocks; NVFP4 = the 5090 rule on %ld finite blocks; BF16 OK\n", finite_blocks);
}

// ---- 3. the NVFP4 path's scalar reference: same quantization, the block products in integers, FP32 in the shared
// kernel's order (per block: fma(sum, weight scale x activation scale, acc))
static float unit_scale(uint8_t b) {
    const uint32_t x = b & 0x7f, e = x >> 3, m = x & 7;
    if (x == 0x7f) return std::numeric_limits<float>::quiet_NaN();
    if (e == 0) return float(m) * (1.f / 1024.f);
    return std::bit_cast<float>(((e + 119u) << 23) | (m << 20));
}
static int code2(uint8_t nibble) { static const int t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}; return t[nibble & 15]; }
// row r of a matrix with K inputs (rows of 16-row groups), against blocks a (K/16): float and double
static void ref_row(const uint8_t* matrix, int K, int r, const fast::QBlock* a, float& out, double& exact) {
    const int blocks = K / 16, rg = r / 16, row = r % 16;
    float acc = 0;
    double ex = 0;
    for (int b = 0; b < blocks; ++b) {
        const uint8_t* unit = matrix + (size_t(rg) * blocks + b) * 144;
        int sum = 0;
        for (int q = 0; q < 4; ++q)
            for (int t = 0; t < 4; ++t) {
                const uint8_t byte = unit[q * 32 + (row % 8) * 4 + t];
                sum += code2(row >= 8 ? byte >> 4 : byte & 15) * a[b].q[q * 4 + t];
            }
        const float ws = unit_scale(unit[128 + row]);
        acc = std::fma(float(sum), ws * a[b].scale, acc);
        ex += double(sum) * double(ws) * double(a[b].scale);
    }
    out = acc;
    exact = ex;
}
// worst (if given): the largest difference of a gate/up row's FP32 result from its exact value, over those values' RMS
static void ref_expert(fast::Act act, const uint8_t* blob, const float* x, uint16_t* y, double* worst) {
    auto* header = (const nx::Header*) blob;
    const uint8_t* rec = blob + sizeof(nx::Header);
    const float m0 = 1.f / header->weight_global[0], m1 = 1.f / header->weight_global[1], m2 = 1.f / header->weight_global[2];
    std::vector<uint16_t> xb(H);
    for (int k = 0; k < H; ++k) xb[k] = bf(x[k]);
    std::vector<fast::QBlock> xq(H / 16), hq(F / 16);
    fast::quant_scalar(act, xb.data(), H, xq.data());
    std::vector<uint16_t> hidden(F);
    double err = 0, sq = 0;
    for (int j = 0; j < F; ++j) {
        float g, u;
        double ge, ue;
        ref_row(rec, H, 2 * j, xq.data(), g, ge);
        ref_row(rec, H, 2 * j + 1, xq.data(), u, ue);
        err = std::max({err, std::fabs(double(g) - ge), std::fabs(double(u) - ue)});
        sq += ge * ge + ue * ue;
        float gg = std::min(g * m0, 7.f);
        float uu = std::clamp(u * m1, -7.f, 7.f);
        hidden[j] = canon::f32_to_bf16_rn(gg / (1.f + std::exp(-1.702f * gg)) * (uu + 1.f));
    }
    if (worst) *worst = err / std::sqrt(sq / (2.0 * F));
    fast::quant_scalar(act, hidden.data(), F, hq.data());
    const uint8_t* down = rec + size_t(2 * F / 16) * (H / 16) * 144;
    for (int r = 0; r < H; ++r) {
        float v;
        double ve;
        ref_row(down, F, r, hq.data(), v, ve);
        y[r] = canon::f32_to_bf16_rn(v * m2);
    }
}

static void compare(const char* name, const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), std::string(name) + ": size");
    for (size_t i = 0; i < a.size(); ++i)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) {
            std::fprintf(stderr, "%s: first difference at %zu: %.9g against %.9g\n", name, i, a[i], b[i]);
            throw std::runtime_error(name);
        }
}

int main(int argc, char** argv) {
    try {
        if (!fast::supported()) { std::puts("mm3_native_cpu_fast_test: no AVX-512 VNNI here - skipped"); return 0; }
        const int workers = argc > 1 ? std::atoi(argv[1]) : 8;
        test_quantizers();
        // experts
        constexpr int E = 4, T = 12;
        const size_t blob = sizeof(nx::Header) + size_t(3) * H * F * 9 / 16;
        std::vector<std::unique_ptr<Aligned>> mem;
        std::vector<mm3::NativeCpuExpert> experts(E);
        std::mt19937 rng(11);
        for (int e = 0; e < E; ++e) {
            mem.push_back(std::make_unique<Aligned>(blob));
            make_expert(mem.back()->p, rng, 40.f + 13.f * e);
            experts[(size_t) e].blob = mem.back()->p;
        }
        std::vector<float> x((size_t) T * H);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng) * 0.7f;
        for (int k = 0; k < H; k += 97) x[(size_t) 3 * H + k] = 0.f;   // (some exact zeros)
        mm3::CpuLane lane(workers, H, F, 1.702f, 7.0f, 64);
        const size_t wbytes = mm3::native_cpu_batch_workspace_bytes(H, F, 64);
        require(mm3::native_cpu_fast_workspace_bytes(H, F, 64) <= wbytes, "fast workspace exceeds the engine's");
        Aligned ws(wbytes);
        // 2. INT8: bits equal to the shared kernel's, over several shapes
        struct Shape { const char* name; Users users; bool order; bool accumulate; };
        std::vector<Shape> shapes;
        { Users u(1); u[0].push_back({0, 0.375f}); shapes.push_back({"one column", u, false, false}); }
        { Users u(E); for (int e = 0; e < E; ++e) u[(size_t) e].push_back({0, 0.125f * (e + 1)}); shapes.push_back({"four experts, one row", u, true, true}); }
        { Users u(2); for (int t = 0; t < 3; ++t) { u[0].push_back({t, 0.2f + 0.1f * t}); u[1].push_back({2 - t, 0.3f}); } shapes.push_back({"three columns", u, true, false}); }
        { Users u(E); for (int e = 0; e < E; ++e) for (int t = 0; t < 8; ++t) if ((t + e) % 3) u[(size_t) e].push_back({t, 0.05f * (1 + (t + e) % 5)}); shapes.push_back({"up to 8 columns", u, true, false}); }
        { Users u(2); for (int t = 0; t < T; ++t) { u[0].push_back({t, 0.1f}); if (t % 2) u[1].push_back({t, 0.2f}); } shapes.push_back({"12 columns: two jobs", u, true, true}); }
        for (auto& s : shapes) {
            Order order;
            if (s.order) for (int t = 0; t < T; ++t) for (int e = (int) s.users.size() - 1; e >= 0; --e)
                for (size_t k = 0; k < s.users[(size_t) e].size(); ++k) if (s.users[(size_t) e][k].first == t) order.push_back({e, (int) k});
            std::vector<mm3::NativeCpuExpert> ex(experts.begin(), experts.begin() + (long) s.users.size());
            std::vector<float> ref((size_t) T * H, 0.25f), got(ref.size(), 0.25f), l3got(ref.size(), 0.25f);
            require(mm3::native_cpu_run_multi_batch(lane, H, F, ex, s.users, x.data(), T, ref.data(), s.order ? &order : nullptr, ws.p, ws.n, s.accumulate), std::string(s.name) + ": shared kernel");
            require(mm3::native_cpu_run_multi_fast(lane, fast::kInt8, H, F, ex, s.users, x.data(), T, got.data(), s.order ? &order : nullptr, ws.p, ws.n, s.accumulate), std::string(s.name) + ": fast INT8");
            compare((std::string("INT8 against the shared kernel: ") + s.name).c_str(), got, ref);
            // the L3 prefetch's fixed shares: a background read of the first expert(s), then the compute with them fixed
            std::vector<const uint8_t*> blobs;
            std::vector<uint8_t> flags(ex.size(), 0);
            for (size_t e = 0; e < ex.size() && e < 2; ++e) { blobs.push_back(ex[e].blob); flags[e] = 1; }
            std::atomic<unsigned long long> read{0};
            require(mm3::native_cpu_l3_begin(lane, H, F, blobs, &read), "L3 prefetch did not start");
            for (volatile int spin = 0; spin < 2000000; ++spin) {}
            lane.stop_background();
            require(mm3::native_cpu_run_multi_fast(lane, fast::kInt8, H, F, ex, s.users, x.data(), T, l3got.data(), s.order ? &order : nullptr, ws.p, ws.n, s.accumulate, &flags), std::string(s.name) + ": fast INT8, fixed shares");
            compare((std::string("INT8 with the L3 prefetch's shares: ") + s.name).c_str(), l3got, ref);
            std::printf("INT8 = shared kernel, bit for bit: %s (prefetch read %.1f MB before stopping)\n", s.name, read.load() / 1e6);
        }
        // 3. NVFP4 against its scalar reference, bit for bit, and against the exact product
        {
            Users u(2);
            u[0] = {{0, 0.6f}, {1, 0.4f}, {5, 0.25f}};
            u[1] = {{1, 0.3f}};
            std::vector<mm3::NativeCpuExpert> ex(experts.begin(), experts.begin() + 2);
            std::vector<float> got((size_t) T * H, 0.f), want((size_t) T * H, 0.f), fixed((size_t) T * H, 0.f);
            require(mm3::native_cpu_run_multi_fast(lane, fast::kNvfp4, H, F, ex, u, x.data(), T, got.data(), nullptr, ws.p, ws.n), "fast NVFP4");
            double worst = 0;
            std::vector<uint16_t> y(H);
            for (size_t e = 0; e < ex.size(); ++e)
                for (auto [t, w] : u[e]) {
                    ref_expert(fast::kNvfp4, ex[e].blob, x.data() + (size_t) t * H, y.data(), e == 0 && t == 0 ? &worst : nullptr);
                    for (int k = 0; k < H; ++k) want[(size_t) t * H + k] += w * f32(y[(size_t) k]);
                }
            compare("NVFP4 against its scalar reference", got, want);
            std::vector<uint8_t> flags{1, 0};
            require(mm3::native_cpu_l3_begin(lane, H, F, {ex[0].blob}, nullptr), "L3 prefetch did not start");
            lane.stop_background();
            require(mm3::native_cpu_run_multi_fast(lane, fast::kNvfp4, H, F, ex, u, x.data(), T, fixed.data(), nullptr, ws.p, ws.n, false, &flags), "fast NVFP4 fixed");
            compare("NVFP4 with the L3 prefetch's shares", fixed, want);
            require(worst < 1e-4, "NVFP4 gate/up rows stray from the exact NVFP4 product: " + std::to_string(worst));
            // INT8 against NVFP4 on the same expert: different activations, outputs of the same size
            std::vector<float> i8((size_t) T * H, 0.f);
            require(mm3::native_cpu_run_multi_fast(lane, fast::kInt8, H, F, ex, u, x.data(), T, i8.data(), nullptr, ws.p, ws.n), "fast INT8 (comparison)");
            double num = 0, den = 0;
            for (size_t i = 0; i < i8.size(); ++i) { num += double(got[i] - i8[i]) * (got[i] - i8[i]); den += double(i8[i]) * i8[i]; }
            std::printf("NVFP4 = scalar reference, bit for bit; gate/up rows within %.2e (relative) of the exact NVFP4 product; "
                        "NVFP4 against INT8 activations: %.2f%% relative RMS (synthetic expert)\n", worst, 100 * std::sqrt(num / std::max(den, 1e-30)));
        }
        // 4. non-finite input, refused shapes
        {
            std::vector<float> xn(x.begin(), x.begin() + H);
            xn[100] = std::numeric_limits<float>::quiet_NaN();
            Users u(1);
            u[0].push_back({0, 1.f});
            std::vector<mm3::NativeCpuExpert> ex(experts.begin(), experts.begin() + 1);
            for (int act : {fast::kInt8, fast::kNvfp4}) {
                std::vector<float> o(H, 0.f);
                require(mm3::native_cpu_run_multi_fast(lane, act, H, F, ex, u, xn.data(), 1, o.data(), nullptr, ws.p, ws.n), "non-finite run");
                require(std::any_of(o.begin(), o.end(), [](float v) { return !std::isfinite(v); }), "a NaN input must leave a non-finite output");
            }
            std::vector<float> o(H, 42.f);
            Users bad = u;
            bad[0][0].first = 1;
            require(!mm3::native_cpu_run_multi_fast(lane, fast::kInt8, H, F, ex, bad, x.data(), 1, o.data(), nullptr, ws.p, ws.n), "bad token accepted");
            require(!mm3::native_cpu_run_multi_fast(lane, 3, H, F, ex, u, x.data(), 1, o.data(), nullptr, ws.p, ws.n), "bad act accepted");
            require(!mm3::native_cpu_run_multi_fast(lane, fast::kInt8, H, F, ex, u, x.data(), 1, o.data(), nullptr, ws.p, 4096), "small workspace accepted");
            require(std::all_of(o.begin(), o.end(), [](float v) { return v == 42.f; }), "a refused call changed the output");
        }
        std::puts("mm3_native_cpu_fast_test: OK");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm3_native_cpu_fast_test: %s\n", e.what());
        return 1;
    }
}
