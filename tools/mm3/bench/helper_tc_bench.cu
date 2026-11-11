// Standalone benchmark of the helper cards' expert kernels (plan: docs/minimax-wsl/README.md#device-kernels).
// Today's kernel (fast_ffn_cuda_packed, INT8 activations, dp4a per row) against the two candidates of
// src/mm3/mm3_helper_kernels.cuh on one card, on synthetic expert records in the engine's RG16 format:
//   dp  today's arithmetic compiled per job width (meant to give today's output bit for bit): one launch group a
//       width as today, or (dp any) jobs of every width in one launch
//   tc  the tensor-core kernel (BF16 activations used exactly), with 1, 2, 4 or 8 warps a row group
//
//   build:  see build.sh beside this file
//   run:    CUDA_VISIBLE_DEVICES=<one card's UUID> ./helper_tc_bench [--experts 16] [--iters 100]
//
// Reported: microseconds a launch and a job for 1 to 16 jobs of 1, 2, 4 and 8 columns; a pass shaped like drafting
// (jobs of mixed widths; today's kernel needs one launch a width, as the engine does); and each kernel's error
// against the exact product of the stored weights (double precision on the host, nothing rounded but the inputs).
// Timing has no data dependence, so synthetic records are enough for time. The error figures use weights drawn the
// way NVFP4 stores Gaussian weights and Gaussian activations with outliers; the engine's fidelity meter is the test
// on real data.
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/fast_math.h"
#include "mm3_helper_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::ops::minimax_ffn {
// The donor defines this beside its CPU kernel (fast_cpu.cpp, which needs AVX-512 flags); the same sizes here.
std::size_t fast_workspace_bytes(int h, int i, int cols) {
    auto al = [](std::size_t n) { return (n + 255) & ~std::size_t(255); };
    return al(std::size_t(cols) * (h / 16) * 20) + al(std::size_t(cols) * (i / 16) * 20) + al(std::size_t(cols) * i * 2);
}
}  // namespace infernix::ops::minimax_ffn

namespace {
namespace mf = infernix::ops::minimax_ffn;
namespace tc = mm3::helper_tc;
using mf::FastJob;
constexpr int H = tc::kH, I = tc::kI, U = tc::kUnit, MAXJ = 16, MAXC = 8;
constexpr std::size_t kRecord = (std::size_t(2 * I / 16) * (H / 16) + std::size_t(H / 16) * (I / 16)) * U;   // 31,850,496

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
struct Rng {
    std::uint64_t s[2];
    explicit Rng(std::uint64_t seed) { s[0] = seed * 0x9E3779B97F4A7C15ull + 1; s[1] = seed ^ 0xD1B54A32D192ED03ull; for (int i = 0; i < 8; ++i) next(); }
    std::uint64_t next() {
        std::uint64_t a = s[0], b = s[1]; s[0] = b; a ^= a << 23; s[1] = a ^ b ^ (a >> 17) ^ (b >> 26); return s[1] + b;
    }
    double uniform() { return double(next() >> 11) * 0x1p-53; }
    double normal() { double u = uniform(), v = uniform(); return std::sqrt(-2.0 * std::log(u + 1e-300)) * std::cos(6.283185307179586 * v); }
};
// A record: every code byte holds two codes drawn with the frequencies NVFP4 gives Gaussian weights scaled by the
// largest of 16 (index 0..7 of the doubled values 0,1,2,3,4,6,8,12), a random sign; scales E4M3 in 0.25 .. 1.9.
void make_record(std::uint8_t* rec, std::uint64_t seed) {
    static std::uint8_t pick[256]; static bool ready = false;
    if (!ready) {
        const double p[8] = {0.10, 0.19, 0.17, 0.14, 0.17, 0.13, 0.07, 0.03}; double acc = 0; int at = 0;
        for (int m = 0; m < 8; ++m) { acc += p[m]; for (; at < int(acc * 256 + 0.5) && at < 256; ++at) pick[at] = std::uint8_t(m); }
        for (; at < 256; ++at) pick[at] = 7;
        ready = true;
    }
    Rng r(seed);
    const std::size_t units = kRecord / U;
    for (std::size_t u = 0; u < units; ++u) {
        std::uint8_t* unit = rec + u * U;
        for (int i = 0; i < 128; i += 2) {
            const std::uint64_t z = r.next();
            for (int k = 0; k < 2; ++k) {
                const unsigned lo = pick[(z >> (16 * k)) & 255] | ((z >> (40 + 2 * k)) & 1) << 3;
                const unsigned hi = pick[(z >> (16 * k + 8)) & 255] | ((z >> (41 + 2 * k)) & 1) << 3;
                unit[i + k] = std::uint8_t(lo | hi << 4);
            }
        }
        const std::uint64_t a = r.next(), b = r.next();
        for (int row = 0; row < 16; ++row) {
            const std::uint64_t z = row < 8 ? a >> (8 * row) : b >> (8 * (row - 8));
            unit[128 + row] = std::uint8_t((5 + (z & 255) % 3) << 3 | ((z >> 3) & 7));
        }
    }
}
double e4m3(unsigned x) { const int e = x >> 3, m = x & 7; return e ? std::ldexp(double(8 + m), e - 10) : std::ldexp(double(m), -9); }
// one row of a matrix of the record, decoded exactly: weight = code * scale / 2
void decode_row(const std::uint8_t* matrix, int blocks, int row, std::vector<double>& out) {
    static const int T[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int group = row / 16, r = row % 16;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = matrix + (std::size_t(group) * blocks + b) * U;
        const double scale = 0.5 * e4m3(unit[128 + r]);
        for (int k = 0; k < 16; ++k) {
            const unsigned byte = unit[(k / 4) * 32 + (r % 8) * 4 + (k % 4)], code = r < 8 ? byte & 15 : byte >> 4;
            out[std::size_t(b) * 16 + k] = (code & 8 ? -1.0 : 1.0) * T[code & 7] * scale;
        }
    }
}
// y = down(act(gate x, up x)) in double: the stored weights and the BF16 inputs exactly. round_hidden: the hidden
// vector rounded to BF16, as every kernel of the engine stores it; otherwise nothing is rounded.
void reference(const std::uint8_t* rec, const float* mult, const std::uint16_t* x, int cols, bool round_hidden, std::vector<double>& y) {
    std::vector<double> xs(std::size_t(cols) * H), hidden(std::size_t(cols) * I), w(H), w2(H);
    for (std::size_t i = 0; i < xs.size(); ++i) xs[i] = mf::bf16_float(x[i]);
    for (int p = 0; p < I; ++p) {
        decode_row(rec, H / 16, 2 * p, w);
        decode_row(rec, H / 16, 2 * p + 1, w2);
        for (int c = 0; c < cols; ++c) {
            double g = 0, u = 0; const double* xc = xs.data() + std::size_t(c) * H;
            for (int k = 0; k < H; ++k) { g += w[k] * xc[k]; u += w2[k] * xc[k]; }
            g = std::min(g * double(mult[0]), 7.0);
            u = std::min(std::max(u * double(mult[1]), -7.0), 7.0);
            const double value = g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0);
            hidden[std::size_t(c) * I + p] = round_hidden ? double(mf::bf16_float(mf::float_bf16(float(value)))) : value;
        }
    }
    y.assign(std::size_t(cols) * H, 0.0);
    std::vector<double> wd(I);
    for (int row = 0; row < H; ++row) {
        decode_row(rec + tc::kDownOffset, I / 16, row, wd);
        for (int c = 0; c < cols; ++c) {
            double s = 0; const double* hc = hidden.data() + std::size_t(c) * I;
            for (int k = 0; k < I; ++k) s += wd[k] * hc[k];
            y[std::size_t(c) * H + row] = s * double(mult[2]);
        }
    }
}
double rel_rms(const std::uint16_t* got, const std::vector<double>& want) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < want.size(); ++i) { const double d = double(mf::bf16_float(got[i])) - want[i]; num += d * d; den += want[i] * want[i]; }
    return std::sqrt(num / den);
}

struct Bench {
    int experts = 16, iters = 100;
    cudaStream_t stream = nullptr;
    std::uint8_t* d_records = nullptr; std::uint16_t *d_x = nullptr, *d_y = nullptr; FastJob* d_jobs = nullptr; void* d_work = nullptr;
    std::vector<std::uint16_t> h_x;
    std::vector<std::uint8_t> h_record0;
    const float mult[3] = {1.f / 200.f, 1.f / 200.f, 1.f / 100.f};   // gate, up and output of order one for these records
    static constexpr int kTables = 8;                     // job tables that start at different experts

    // table t: job j uses expert (t * 5 + j) mod experts, its own x and y, the given width
    void set_jobs(const std::vector<int>& widths) {
        std::vector<FastJob> jobs(std::size_t(kTables) * MAXJ);
        for (int t = 0; t < kTables; ++t)
            for (int j = 0; j < int(widths.size()); ++j) {
                FastJob& f = jobs[std::size_t(t) * MAXJ + j];
                f.record = d_records + std::size_t((t * 5 + j) % experts) * kRecord;
                f.multiplier[0] = mult[0]; f.multiplier[1] = mult[1]; f.multiplier[2] = mult[2];
                f.x = d_x + std::size_t(j) * MAXC * H; f.y = d_y + std::size_t(j) * MAXC * H; f.columns = widths[j];
            }
        ck(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(FastJob), cudaMemcpyHostToDevice), "jobs");
    }
    template<class F> double time_us(F&& launch) {
        for (int i = 0; i < 10; ++i) launch(i % kTables);
        ck(cudaStreamSynchronize(stream), "sync");
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) { launch(i % kTables); ck(cudaStreamSynchronize(stream), "sync"); }
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    }
    // today's kernel: one launch for every run of jobs with the same width (the engine sorts jobs by width)
    void baseline(int table, const std::vector<int>& widths) {
        const FastJob* jobs = d_jobs + std::size_t(table) * MAXJ;
        for (int first = 0, n = int(widths.size()); first < n;) {
            int end = first + 1;
            while (end < n && widths[end] == widths[first]) ++end;
            mf::fast_ffn_cuda_packed(jobs + first, end - first, H, I, widths[first], d_work, stream);
            first = end;
        }
    }
    // today's arithmetic compiled per width: the same launch groups
    void dp(int table, const std::vector<int>& widths) {
        const FastJob* jobs = d_jobs + std::size_t(table) * MAXJ;
        for (int first = 0, n = int(widths.size()); first < n;) {
            int end = first + 1;
            while (end < n && widths[end] == widths[first]) ++end;
            ck(mm3::helper_dp::ffn(jobs + first, end - first, widths[first], d_work, stream), "dp launch");
            first = end;
        }
    }
    void dp_any(int table, int n) { ck(mm3::helper_dp::ffn_any(d_jobs + std::size_t(table) * MAXJ, n, d_work, stream), "dp any-width launch"); }
    template<int S> void tensor(int table, int n) { ck(tc::ffn<S>(d_jobs + std::size_t(table) * MAXJ, n, d_work, stream), "tensor-core launch"); }
    template<int S> void tensor1(int table, int n) { ck(tc::ffn1<S>(d_jobs + std::size_t(table) * MAXJ, n, d_work, stream), "tensor-core launch, variant 1"); }

    void run() {
        cudaDeviceProp prop{};
        int dev = 0; ck(cudaGetDevice(&dev), "device"); ck(cudaGetDeviceProperties(&prop, dev), "properties");
        std::printf("card: %s (sm_%d%d), %d SMs, %.1f GiB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
        ck(cudaStreamCreate(&stream), "stream");
        ck(cudaMalloc(&d_records, std::size_t(experts) * kRecord), "records");
        ck(cudaMalloc(&d_x, std::size_t(MAXJ) * MAXC * H * 2), "x"); ck(cudaMalloc(&d_y, std::size_t(MAXJ) * MAXC * H * 2), "y");
        ck(cudaMalloc(&d_jobs, std::size_t(kTables) * MAXJ * sizeof(FastJob)), "jobs");
        const std::size_t work = std::max(std::size_t(MAXJ) * mf::fast_workspace_bytes(H, I, MAXC), tc::workspace_bytes(MAXJ));
        ck(cudaMalloc(&d_work, work), "workspace");
        {
            std::vector<std::uint8_t> rec(kRecord);
            for (int e = 0; e < experts; ++e) {
                make_record(rec.data(), 1000 + e);
                if (e == 0) h_record0 = rec;
                ck(cudaMemcpy(d_records + std::size_t(e) * kRecord, rec.data(), kRecord, cudaMemcpyHostToDevice), "upload record");
            }
        }
        // activations: N(0,1), one value in a hundred ten times larger
        h_x.resize(std::size_t(MAXJ) * MAXC * H);
        Rng r(7);
        for (auto& v : h_x) { double z = r.normal(); if (r.uniform() < 0.01) z *= 10; v = mf::float_bf16(float(z)); }
        ck(cudaMemcpy(d_x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice), "upload x");
        std::printf("%d synthetic experts of %.2f MB on the card, %d timed launches a figure\n\n", experts, kRecord / 1e6, iters);
        // bring the card's clocks up before anything is timed: two seconds of back-to-back launches
        {
            set_jobs(std::vector<int>(8, 2));
            const auto t0 = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2)) { baseline(0, std::vector<int>(8, 2)); ck(cudaStreamSynchronize(stream), "sync"); }
        }

        // 1. error against the exact product (job 0 of table 0 is expert 0), eight columns
        {
            set_jobs({8, 8});
            // want: nothing rounded. stored: the hidden vector rounded to BF16 (what a kernel with exact arithmetic gives)
            std::vector<double> want, stored;
            reference(h_record0.data(), mult, h_x.data(), 8, false, want);
            reference(h_record0.data(), mult, h_x.data(), 8, true, stored);
            std::vector<std::uint16_t> got(std::size_t(8) * H), base(std::size_t(8) * H), ideal(std::size_t(8) * H);
            for (std::size_t i = 0; i < want.size(); ++i) ideal[i] = mf::float_bf16(float(stored[i]));
            ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear");
            baseline(0, {8, 8}); ck(cudaStreamSynchronize(stream), "sync");
            ck(cudaMemcpy(base.data(), d_y, base.size() * 2, cudaMemcpyDeviceToHost), "read");
            std::printf("error against the exact product, relative RMS over 8 columns x %d outputs:\n", H);
            std::printf("  exact arithmetic with the hidden vector and the result in BF16 (the least): %.5f%%\n", 100 * rel_rms(ideal.data(), want));
            std::printf("  today's kernel (INT8 activations, dp4a):                           %.5f%%\n", 100 * rel_rms(base.data(), want));
            {
                ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear");
                dp(0, {8, 8}); ck(cudaStreamSynchronize(stream), "sync");
                ck(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost), "read");
                std::size_t differ = 0; for (std::size_t i = 0; i < got.size(); ++i) differ += got[i] != base[i];
                std::printf("  today's arithmetic compiled per width (dp):                        %.5f%%  (%zu of %zu outputs differ from today's kernel)\n",
                            100 * rel_rms(got.data(), want), differ, got.size());
                for (int width : {1, 3, 5}) {   // other widths against today's kernel, bit for bit
                    const std::vector<int> ws = {width, width};
                    set_jobs(ws);
                    std::vector<std::uint16_t> a(std::size_t(width) * H), b(std::size_t(width) * H);
                    ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear"); baseline(0, ws); ck(cudaStreamSynchronize(stream), "sync");
                    ck(cudaMemcpy(a.data(), d_y, a.size() * 2, cudaMemcpyDeviceToHost), "read");
                    ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear"); dp(0, ws); ck(cudaStreamSynchronize(stream), "sync");
                    ck(cudaMemcpy(b.data(), d_y, b.size() * 2, cudaMemcpyDeviceToHost), "read");
                    std::size_t d = 0; for (std::size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
                    std::printf("    width %d: %zu of %zu outputs differ from today's kernel\n", width, d, a.size());
                }
                {   // jobs of mixed widths in one launch against today's kernel (one launch group a width), the whole buffer
                    const std::vector<int> ws = {1, 1, 2, 3, 5, 8};
                    set_jobs(ws);
                    std::vector<std::uint16_t> a(std::size_t(MAXJ) * MAXC * H), b(a.size());
                    ck(cudaMemset(d_y, 0, a.size() * 2), "clear"); baseline(0, ws); ck(cudaStreamSynchronize(stream), "sync");
                    ck(cudaMemcpy(a.data(), d_y, a.size() * 2, cudaMemcpyDeviceToHost), "read");
                    ck(cudaMemset(d_y, 0, a.size() * 2), "clear"); dp_any(0, int(ws.size())); ck(cudaStreamSynchronize(stream), "sync");
                    ck(cudaMemcpy(b.data(), d_y, b.size() * 2, cudaMemcpyDeviceToHost), "read");
                    std::size_t d = 0, nonzero = 0; for (std::size_t i = 0; i < a.size(); ++i) { d += a[i] != b[i]; nonzero += a[i] != 0; }
                    std::printf("    widths 1 1 2 3 5 8 in one launch (dp any): %zu outputs differ from today's kernel (%zu nonzero outputs)\n", d, nonzero);
                }
                set_jobs({8, 8});
            }
            const char* names[4] = {"1", "2", "4", "8"};
            for (int s = 0; s < 4; ++s) {
                ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear");
                if (s == 0) tensor<1>(0, 2); else if (s == 1) tensor<2>(0, 2); else if (s == 2) tensor<4>(0, 2); else tensor<8>(0, 2);
                ck(cudaStreamSynchronize(stream), "sync");
                ck(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost), "read");
                std::size_t differ = 0; for (std::size_t i = 0; i < got.size(); ++i) differ += got[i] != ideal[i];
                std::printf("  tensor-core kernel, %s warp(s) a row group (BF16 activations exact):  %.5f%%  (%zu of %zu outputs differ from that least-error result)\n",
                            names[s], 100 * rel_rms(got.data(), want), differ, got.size());
            }
            for (int s = 0; s < 3; ++s) {
                ck(cudaMemset(d_y, 0, std::size_t(MAXJ) * MAXC * H * 2), "clear");
                if (s == 0) tensor1<2>(0, 2); else if (s == 1) tensor1<4>(0, 2); else tensor1<8>(0, 2);
                ck(cudaStreamSynchronize(stream), "sync");
                ck(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost), "read");
                std::size_t differ = 0; for (std::size_t i = 0; i < got.size(); ++i) differ += got[i] != ideal[i];
                std::printf("  tensor-core kernel, variant 1, %s warps a row group:                  %.5f%%  (%zu of %zu outputs differ from that least-error result)\n",
                            names[s + 1], 100 * rel_rms(got.data(), want), differ, got.size());
            }
            // a narrow job beside a wide one in the same launch: columns past a job's width must stay untouched
            set_jobs({3, 8});
            ck(cudaMemset(d_y, 0xff, std::size_t(MAXJ) * MAXC * H * 2), "fill");
            for (int variant = 0; variant < 2; ++variant) {
                ck(cudaMemset(d_y, 0xff, std::size_t(MAXJ) * MAXC * H * 2), "fill");
                if (variant) tensor1<4>(0, 2); else tensor<4>(0, 2);
                ck(cudaStreamSynchronize(stream), "sync");
                ck(cudaMemcpy(got.data(), d_y, got.size() * 2, cudaMemcpyDeviceToHost), "read");
                std::size_t wrong = 0, touched = 0;
                for (std::size_t i = 0; i < got.size(); ++i) { if (i < std::size_t(3) * H) wrong += got[i] != ideal[i] && std::fabs(mf::bf16_float(got[i]) - want[i]) > 0.02 * (std::fabs(want[i]) + 0.05); else touched += got[i] != 0xffff; }
                std::printf("  a 3-column job beside an 8-column one, variant %d: %zu outputs off by more than 2%%, %zu written past its width\n", variant, wrong, touched);
            }
            std::printf("\n");
        }

        // 2. time by jobs a launch and columns a job
        std::printf("microseconds a launch: today's kernel; today's arithmetic compiled per width (dp) and the same in the any-width launch; the tensor-core kernel with 1, 2, 4, 8 warps a row group; its variant 1 with 2, 4, 8\n");
        std::printf("%5s %5s | %8s | %8s %8s | %8s %8s %8s %8s | %8s %8s %8s | a job: today, dp, best tc | dp and best tc against today\n", "jobs", "cols", "today", "dp", "dp any", "tc S=1", "S=2", "S=4", "S=8", "tc1 S=2", "S=4", "S=8");
        for (int cols : {1, 2, 4, 8})
            for (int n : {1, 2, 4, 8, 16}) {
                const std::vector<int> widths(std::size_t(n), cols);
                set_jobs(widths);
                const double base = time_us([&](int t) { baseline(t, widths); }), td = time_us([&](int t) { dp(t, widths); }), ta = time_us([&](int t) { dp_any(t, n); });
                const double t1 = time_us([&](int t) { tensor<1>(t, n); }), t2 = time_us([&](int t) { tensor<2>(t, n); });
                const double t4 = time_us([&](int t) { tensor<4>(t, n); }), t8 = time_us([&](int t) { tensor<8>(t, n); });
                const double v2 = time_us([&](int t) { tensor1<2>(t, n); }), v4 = time_us([&](int t) { tensor1<4>(t, n); }), v8 = time_us([&](int t) { tensor1<8>(t, n); });
                const double best = std::min(std::min(std::min(t1, t2), std::min(t4, t8)), std::min(v2, std::min(v4, v8)));
                std::printf("%5d %5d | %8.1f | %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f | %7.1f %7.1f %7.1f | x%.2f x%.2f\n", n, cols, base, td, ta, t1, t2, t4, t8, v2, v4, v8,
                            base / n, std::min(td, ta) / n, best / n, base / std::min(td, ta), base / best);
                std::fflush(stdout);
            }
        // 3. passes shaped like drafting: jobs of mixed widths, sorted by width as the engine sorts them
        std::printf("\npasses of mixed widths (today and dp: one launch group a width; dp any and tensor-core: one launch)\n");
        const std::vector<std::vector<int>> passes = {
            {1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 6}, {1, 1, 1, 2, 2, 3, 3, 4}, {1, 1, 2, 2, 4, 8}, {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5}};
        for (const auto& widths : passes) {
            set_jobs(widths);
            const int n = int(widths.size()); int total = 0; std::string shape;
            for (int w : widths) { total += w; shape += std::to_string(w) + " "; }
            const double base = time_us([&](int t) { baseline(t, widths); }), td = time_us([&](int t) { dp(t, widths); }), ta = time_us([&](int t) { dp_any(t, n); });
            const double t2 = time_us([&](int t) { tensor<2>(t, n); }), t4 = time_us([&](int t) { tensor<4>(t, n); }), t8 = time_us([&](int t) { tensor<8>(t, n); });
            const double v2 = time_us([&](int t) { tensor1<2>(t, n); }), v4 = time_us([&](int t) { tensor1<4>(t, n); }), v8 = time_us([&](int t) { tensor1<8>(t, n); });
            std::printf("  %2d jobs, %2d columns (%s): today %8.1f | dp %8.1f  dp any %8.1f | tc S=2 %8.1f  S=4 %8.1f  S=8 %8.1f | tc1 S=2 %8.1f  S=4 %8.1f  S=8 %8.1f | x%.2f x%.2f x%.2f\n", n, total, shape.c_str(),
                        base, td, ta, t2, t4, t8, v2, v4, v8, base / td, base / ta, base / std::min(std::min(t2, std::min(t4, t8)), std::min(v2, std::min(v4, v8))));
            std::fflush(stdout);
        }
    }
};
}  // namespace

int main(int argc, char** argv) {
    Bench b;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string k = argv[i];
        if (k == "--experts") b.experts = std::atoi(argv[i + 1]);
        else if (k == "--iters") b.iters = std::atoi(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (b.experts < 4 || b.iters < 1) { std::fprintf(stderr, "--experts at least 4 (16 or more gives every job of a launch its own), --iters at least 1\n"); return 2; }
    try {
        ck(cudaSetDeviceFlags(cudaDeviceScheduleSpin), "schedule flag");
        b.run();
    } catch (const std::exception& e) { std::fprintf(stderr, "FAILED: %s\n", e.what()); return 1; }
    return 0;
}
