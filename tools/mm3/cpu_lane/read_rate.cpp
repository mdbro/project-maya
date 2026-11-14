// tools/mm3/cpu_lane/read_rate.cpp - why does the CPU alone read host memory at about 60 GB/s, and 9 readers at 79?
// (kernel item 2 of the MiniMax plan). CPU only: no GPU, no CUDA.
//
// Build: g++ -O2 -mavx512f -mavx512bw -std=c++20 -pthread read_rate.cpp -o read_rate
// Run:   read_rate [rounds] [seconds a measurement] [--quiet-lock PATH] [--sets LIST]
//
// Each measurement streams expert-sized records (31.85 MB, the CPU lane's job) from a pool far larger than the L3, on N
// threads pinned to guest vCPUs, for a fixed time, and reports the sum of the threads' read rates (GB/s, 1e9 bytes) and
// each thread's own rate. Two access patterns:
//   slice: thread t reads the t-th contiguous 1/N of each record (the earlier bandwidth_*.cu benchmarks' pattern);
//   jobs:  the CPU lane's pattern - each record's 55,296-byte row groups (384 units of 144 bytes) taken in turn from one
//          shared counter, so the threads interleave on row groups.
// The per-thread rates tell where the threads ran: on Zen 5 each CCD reaches DRAM through its own Infinity Fabric link,
// so threads sharing one CCD share that link's rate, and a thread alone on the other CCD reads several times faster
// than its peers. Pages: 4 KiB (the default) or 2 MiB (MADV_HUGEPAGE), measured both ways.
// Readers are pinned to the odd vCPUs by default: the serving engine's idle CPU-lane workers are pinned to the even ones
// (2..28) and wake every 50 ms.
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr size_t kRecord = 31850496;        // 3 * 6144 * 3072 * 9 / 16 (the RG16 payload of one expert)
constexpr size_t kRowGroup = 384 * 144;     // gate/up row group: h/16 units of 144 bytes
constexpr size_t kRecords = 24;             // 764 MB a pool: six times the 128 MB of L3 in both CCDs
using clk = std::chrono::steady_clock;
std::string quiet_lock;
bool quiet_held() { return !quiet_lock.empty() && access(quiet_lock.c_str(), F_OK) == 0; }

void pin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

struct Pool {
    uint8_t* base = nullptr;
    size_t bytes = 0;
    bool huge = false;
    size_t records = kRecords;
    explicit Pool(bool h, size_t n = kRecords) : huge(h), records(n) {
        bytes = records * kRecord;
        const size_t align = size_t(2) << 20;
        bytes = (bytes + align - 1) / align * align;
        void* p = mmap(nullptr, bytes + align, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
        base = (uint8_t*) (((uintptr_t) p + align - 1) / align * align);
        madvise(base, bytes, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE);
        auto* w = (uint64_t*) base;   // (written, so no page is the shared zero page)
        for (size_t i = 0; i < bytes / 8; ++i) w[i] = (i + 1) * 0x9E3779B97F4A7C15ull;
    }
    uint8_t* record(size_t r) const { return base + (r % records) * kRecord; }
};

// One 64-byte line a load, four lines an iteration, summed so the loads cannot be dropped.
__attribute__((target("avx512f,avx512bw"))) inline __m512i read_span(const uint8_t* p, size_t n, __m512i acc) {
    __m512i a = acc, b = _mm512_setzero_si512(), c = b, d = b;
    size_t i = 0;
    for (; i + 256 <= n; i += 256) {
        a = _mm512_add_epi64(a, _mm512_load_si512(p + i));
        b = _mm512_add_epi64(b, _mm512_load_si512(p + i + 64));
        c = _mm512_add_epi64(c, _mm512_load_si512(p + i + 128));
        d = _mm512_add_epi64(d, _mm512_load_si512(p + i + 192));
    }
    for (; i + 64 <= n; i += 64) a = _mm512_add_epi64(a, _mm512_load_si512(p + i));
    return _mm512_add_epi64(_mm512_add_epi64(a, b), _mm512_add_epi64(c, d));
}

struct Result { double total = 0; std::vector<double> per; };

__attribute__((target("avx512f,avx512bw"))) Result measure(const Pool& pool, const std::vector<int>& cpus, bool jobs,
                                                         double seconds) {
    const int n = (int) cpus.size();
    std::atomic<int> ready{0};
    std::atomic<bool> go{false}, stop{false};
    std::atomic<uint64_t> ticket{0};
    std::vector<double> bytes(n, 0);
    std::vector<uint64_t> sink(n, 0);
    std::vector<std::thread> th;
    const size_t groups = kRecord / kRowGroup;   // 576 row groups a record (the last partial one left out)
    const size_t slice = n ? kRecord / n / 64 * 64 : 0;
    for (int t = 0; t < n; ++t)
        th.emplace_back([&, t] {
            pin(cpus[t]);
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) _mm_pause();
            __m512i acc = _mm512_setzero_si512();
            double b = 0;
            if (jobs) {
                for (;;) {
                    if (stop.load(std::memory_order_relaxed)) break;
                    const uint64_t j = ticket.fetch_add(1, std::memory_order_relaxed);
                    const uint8_t* p = pool.record(j / groups) + (j % groups) * kRowGroup;
                    acc = read_span(p, kRowGroup, acc);
                    b += kRowGroup;
                }
            } else {
                for (size_t r = (size_t) t * 7;; ++r) {   // (threads start on different records)
                    if (stop.load(std::memory_order_relaxed)) break;
                    acc = read_span(pool.record(r) + (size_t) t * slice, slice, acc);
                    b += slice;
                }
            }
            alignas(64) uint64_t v[8];
            _mm512_store_si512(v, acc);
            sink[t] = v[0] ^ v[7];
            bytes[t] = b;
        });
    while (ready.load() < n) std::this_thread::yield();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const auto t0 = clk::now();
    go.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& x : th) x.join();
    const double el = std::chrono::duration<double>(clk::now() - t0).count();
    Result r;
    uint64_t x = 0;
    for (int t = 0; t < n; ++t) {
        r.per.push_back(bytes[t] / el / 1e9);
        r.total += bytes[t] / el / 1e9;
        x ^= sink[t];
    }
    if (x == 42) std::printf(" ");
    return r;
}

std::vector<int> odd(int n) { std::vector<int> v; for (int i = 0; i < n; ++i) v.push_back(2 * i + 1); return v; }
std::vector<int> unpinned(int n) { return std::vector<int>(n, -1); }


// Which CCD a vCPU runs on, seen from inside the VM: one thread pinned to it rereads a 24 MB buffer (fits the L3 of
// either CCD: 96 MB on the V-Cache CCD, 32 MB on the other) and a 64 MB buffer (fits only the V-Cache CCD's), while
// `spin` other threads keep that many more vCPUs busy (spinning, no memory traffic), so the host places busy vCPUs as
// it would under load. 64 MB at close to the 24 MB rate: the thread ran on the V-Cache CCD; at DRAM rate: the other.
__attribute__((target("avx512f,avx512bw"))) double reread(const uint8_t* p, size_t n, double seconds) {
    __m512i acc = _mm512_setzero_si512();
    for (int w = 0; w < 3; ++w) acc = read_span(p, n, acc);
    const auto t0 = clk::now();
    double b = 0, el = 0;
    do { acc = read_span(p, n, acc); b += n; el = std::chrono::duration<double>(clk::now() - t0).count(); } while (el < seconds);
    alignas(64) uint64_t v[8];
    _mm512_store_si512(v, acc);
    if ((v[0] ^ v[5]) == 42) std::printf(" ");
    return b / el / 1e9;
}
int ccd_probe(const Pool& pool, int rounds) {
    std::printf("CCD probe: GB/s rereading 24 MB and 64 MB on one pinned thread, with N other vCPUs spinning\n");
    std::printf("%5s %6s", "vCPU", "spin");
    for (int r = 0; r < rounds; ++r) std::printf("   24MB  64MB");
    std::printf("\n");
    for (int spin : {0, 8, 14}) {
        for (int k = 0; k < 30; ++k) {
            std::printf("%5d %6d", k, spin);
            for (int r = 0; r < rounds; ++r) {
                while (quiet_held()) std::this_thread::sleep_for(std::chrono::seconds(10));
                std::atomic<bool> stop{false};
                std::vector<std::thread> sp;
                for (int i = 0, c = 0; i < spin && c < 30; ++c) {
                    if (c == k) continue;
                    sp.emplace_back([&, c] { pin(c); while (!stop.load(std::memory_order_relaxed)) _mm_pause(); });
                    ++i;
                }
                double a = 0, b = 0;
                std::thread t([&] { pin(k); a = reread(pool.base, size_t(24) << 20, 0.15); b = reread(pool.base, size_t(64) << 20, 0.2); });
                t.join();
                stop = true;
                for (auto& x : sp) x.join();
                std::printf("  %5.1f %5.1f", a, b);
                std::fflush(stdout);
            }
            std::printf("\n");
        }
    }
    return 0;
}

struct Case { std::string name; std::vector<int> cpus; bool jobs; bool huge; std::vector<Result> runs; };

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
}  // namespace

int main(int argc, char** argv) {
    int rounds = 4;
    double seconds = 0.5;
    std::string sets = "count";
    size_t records = kRecords;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quiet-lock" && i + 1 < argc) quiet_lock = argv[++i];
        else if (a == "--sets" && i + 1 < argc) sets = argv[++i];
        else if (a == "--records" && i + 1 < argc) records = (size_t) std::atoi(argv[++i]);
        else if (i == 1) rounds = std::atoi(argv[i]);
        else seconds = std::atof(argv[i]);
    }
    const bool want_huge = sets == "huge";
    Pool small(false, records);
    if (sets == "ccd") return ccd_probe(small, rounds);
    const auto h0 = clk::now();
    Pool huge(true, want_huge ? 8 : 1);   // (2 MiB pages: 8 records, 255 MB - the host's free memory is short, and a
                                          //  2 MiB fault may compact; the engine's bank itself is 4 KiB shmem pages)
    if (want_huge) std::fprintf(stderr, "2 MiB pool: %.1f s to fault in\n", std::chrono::duration<double>(clk::now() - h0).count());
    std::vector<Case> cases;
    if (sets == "count") {   // thread count 1..15, slice and jobs patterns, 4 KiB pages; 2 MiB pages at a few counts
        for (int n : {1, 2, 4, 6, 8, 9, 10, 12, 15}) {
            cases.push_back({std::to_string(n) + " odd vCPUs, slices", odd(n), false, false, {}});
            cases.push_back({std::to_string(n) + " odd vCPUs, row-group jobs", odd(n), true, false, {}});
        }
        for (int n : {8, 9, 15}) cases.push_back({std::to_string(n) + " unpinned, slices", unpinned(n), false, false, {}});
    } else if (sets == "huge") {   // 4 KiB against 2 MiB pages, slices
        for (int n : {1, 2, 8, 9, 15}) {
            cases.push_back({std::to_string(n) + " odd vCPUs, slices, 4 KiB pages", odd(n), false, false, {}});
            cases.push_back({std::to_string(n) + " odd vCPUs, slices, 2 MiB pages", odd(n), false, true, {}});
        }
    } else if (sets == "pairs") {   // two readers: vCPU 1 and each other odd vCPU
        for (int k = 3; k < 30; k += 2) cases.push_back({"vCPUs 1 + " + std::to_string(k), {1, k}, false, false, {}});
    } else if (sets == "nine") {   // the 9-reader case repeated, per thread
        for (int k : {7, 8, 9, 10, 11}) {
            cases.push_back({std::to_string(k) + " odd vCPUs, slices", odd(k), false, false, {}});
            cases.push_back({std::to_string(k) + " odd vCPUs, row-group jobs", odd(k), true, false, {}});
        }
    }
    const int n = (int) cases.size();
    for (int r = 0; r < rounds; ++r) {
        for (int j = 0; j < n; ++j) {
            Case& c = cases[(size_t) ((j + r * 3) % n)];
            while (quiet_held()) { std::fprintf(stderr, "quiet lock held: waiting\n"); std::this_thread::sleep_for(std::chrono::seconds(10)); }
            const Pool& p = c.huge ? huge : small;
            (void) measure(p, c.cpus, c.jobs, 0.1);   // (warm the threads and the pages' translations)
            c.runs.push_back(measure(p, c.cpus, c.jobs, seconds));
        }
        std::fprintf(stderr, "round %d of %d done\n", r + 1, rounds);
    }
    std::printf("read rate, GB/s (1e9 bytes/s), %d rounds of %.2f s; per round the sum over threads, then each thread's "
                "rate in the round with the median sum\n", rounds, seconds);
    for (auto& c : cases) {
        std::vector<double> tot;
        for (auto& r : c.runs) tot.push_back(r.total);
        const double med = median(tot);
        size_t pick = 0;
        for (size_t i = 0; i < c.runs.size(); ++i) if (c.runs[i].total == med) pick = i;
        std::printf("%-40s median %6.1f  by round:", c.name.c_str(), med);
        for (double t : tot) std::printf(" %5.1f", t);
        std::printf("   threads:");
        for (double t : c.runs[pick].per) std::printf(" %4.1f", t);
        std::printf("\n");
    }
}
