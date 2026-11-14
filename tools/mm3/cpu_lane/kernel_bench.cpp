// tools/mm3/cpu_lane/kernel_bench.cpp - the CPU lane's expert kernels, standalone (kernel item 2). CPU only: no GPU.
//
// Times one CPU job (an expert, 1 to 8 columns) through the engine's own lane (mm3::CpuLane, pinned workers, the caller
// one of the threads) for:
//   shared  the service's fast_ffn_cpu_batch (today's kernel, native_cpu_run_multi_batch);
//   int8    the reworked kernel, INT8 activations (the same output bits);
//   nvfp4   the reworked kernel, NVFP4 activations;
// with the record
//   dram       a different record each call, from a pool far larger than the L3 (the engine's case without a prefetch);
//   l3         the same record each call (warm: the ceiling of what a prefetch can give);
//   prefetched a different record each call, read ahead by the lane's workers (native_cpu_l3_begin) until all of it is
//              read, then computed with the fixed shares - the L3 prefetch as the engine would run it, minus the wait.
// Synthetic RG16 experts (random codes and E4M3 scales): the kernels' time does not depend on the values.
// Build: cmake target mm3_cpu_lane_bench (EXCLUDE_FROM_ALL).
// Run:   mm3_cpu_lane_bench [--threads 9,15] [--records 24] [--calls 40] [--quiet-lock PATH]
#include "mm3_native_cpu.hpp"
#include "mm3_native_cpu_fast.hpp"
#include "mm3_native_experts.cuh"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fast = mm3::cpu_fast;
namespace nx = strata::mm3::native_experts;
using clk = std::chrono::steady_clock;
constexpr int H = 6144, F = 3072;

static std::string quiet_lock;
static void wait_quiet() {
    while (!quiet_lock.empty() && access(quiet_lock.c_str(), F_OK) == 0) {
        std::fprintf(stderr, "quiet lock held: waiting\n");
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

int main(int argc, char** argv) {
    std::vector<int> thread_counts{9, 15};
    int records = 24, calls = 40;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--threads") { thread_counts.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) thread_counts.push_back(std::atoi(t.c_str())); }
        else if (a == "--records") records = std::atoi(next().c_str());
        else if (a == "--calls") calls = std::atoi(next().c_str());
        else if (a == "--quiet-lock") quiet_lock = next();
    }
    if (!fast::supported()) { std::puts("no AVX-512 VNNI"); return 1; }
    const size_t blob = (sizeof(nx::Header) + size_t(3) * H * F * 9 / 16 + 4095) & ~size_t(4095);
    uint8_t* pool = (uint8_t*) std::aligned_alloc(4096, blob * records);
    if (!pool) { std::puts("allocation failed"); return 1; }
    std::mt19937 rng(3);
    for (int r = 0; r < records; ++r) {
        uint8_t* b = pool + blob * r;
        nx::Header header{};
        for (int k = 0; k < 3; ++k) { header.weight_global[k] = 50.f; header.input_global[k] = 1.f; }
        std::memcpy(b, &header, sizeof header);
        uint32_t* w = (uint32_t*) (b + sizeof header);
        const size_t n = size_t(3) * H * F * 9 / 16 / 4;
        for (size_t i = 0; i < n; ++i) w[i] = rng();
        // scale bytes: valid E4M3 (no NaN word, no sign)
        uint8_t* rec = b + sizeof header;
        for (size_t u = 0; u < size_t(3) * H * F / 256; ++u) for (int k = 0; k < 16; ++k) rec[u * 144 + 128 + k] = uint8_t(0x18 + (rng() % 24));
    }
    std::vector<float> x(8 * H);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    const int max_threads = *std::max_element(thread_counts.begin(), thread_counts.end());
    mm3::CpuLane lane(max_threads - 1, H, F, 1.702f, 7.0f, 64);
    const size_t wbytes = mm3::native_cpu_batch_workspace_bytes(H, F, 64);
    void* ws = std::aligned_alloc(256, (wbytes + 255) & ~size_t(255));
    const size_t payload = size_t(3) * H * F * 9 / 16;
    std::printf("CPU job time, ms (median of %d calls), one expert of 31.85 MB; lane threads include the caller\n", calls);
    std::printf("%7s %5s %-8s %10s %10s %10s %14s\n", "threads", "cols", "kernel", "dram", "l3", "prefetched", "prefetch read");
    unsigned rot = 0;
    for (int threads : thread_counts) {
        lane.set_active(threads);
        for (int cols : {1, 2, 4, 8}) {
            std::vector<std::vector<std::pair<int, float>>> users(1);
            for (int c = 0; c < cols; ++c) users[0].push_back({c, 0.25f});
            std::vector<float> out((size_t) 8 * H);
            for (int kernel = 0; kernel < 3; ++kernel) {
                wait_quiet();
                auto run = [&](const uint8_t* b, const std::vector<uint8_t>* fixed) {
                    std::vector<mm3::NativeCpuExpert> ex{{b, 0.25f}};
                    const bool ok = kernel == 0 ? mm3::native_cpu_run_multi_batch(lane, H, F, ex, users, x.data(), cols, out.data(), nullptr, ws, wbytes)
                                                : mm3::native_cpu_run_multi_fast(lane, kernel, H, F, ex, users, x.data(), cols, out.data(), nullptr, ws, wbytes, false, fixed);
                    if (!ok) { std::puts("kernel refused"); std::exit(1); }
                };
                std::vector<double> dram, l3, pre, pre_read;
                for (int c = 0; c < calls + 3; ++c) {   // (the first three of each: warm-up)
                    const uint8_t* b = pool + blob * (rot++ % records);
                    auto t0 = clk::now();
                    run(b, nullptr);
                    if (c >= 3) dram.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
                }
                const uint8_t* hot = pool + blob * (rot++ % records);
                for (int c = 0; c < calls + 3; ++c) {
                    auto t0 = clk::now();
                    run(hot, nullptr);
                    if (c >= 3) l3.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
                }
                if (kernel > 0) {
                    const std::vector<uint8_t> fixed{1};
                    for (int c = 0; c < calls + 3; ++c) {
                        const uint8_t* b = pool + blob * (rot++ % records);
                        std::atomic<unsigned long long> read{0};
                        auto p0 = clk::now();
                        if (!mm3::native_cpu_l3_begin(lane, H, F, {b}, &read)) { std::puts("prefetch refused"); return 1; }
                        while (read.load() < payload && clk::now() - p0 < std::chrono::milliseconds(20)) std::this_thread::yield();
                        const double pr = std::chrono::duration<double, std::milli>(clk::now() - p0).count();
                        lane.stop_background();
                        auto t0 = clk::now();
                        run(b, &fixed);
                        if (c >= 3) { pre.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count()); pre_read.push_back(pr); }
                    }
                }
                static const char* names[] = {"shared", "int8", "nvfp4"};
                std::printf("%7d %5d %-8s %10.3f %10.3f %10s %14s\n", threads, cols, names[kernel], median(dram), median(l3),
                            kernel ? std::to_string(median(pre)).substr(0, 5).c_str() : "-",
                            kernel ? (std::to_string(median(pre_read)).substr(0, 5) + " ms").c_str() : "-");
                std::fflush(stdout);
            }
        }
    }
    std::free(ws);
    std::free(pool);
    return 0;
}
