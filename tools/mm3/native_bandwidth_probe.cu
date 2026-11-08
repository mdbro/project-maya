// Host-memory bandwidth shared by CPU reads and GPU uploads, alone and together. No model, no engine state touched.
//
// Readers: CPU threads summing a large pageable buffer (well past the last-level cache), and 32 MB host-to-device
// copies from pinned memory on each listed GPU, the size of one expert record. Each case runs every listed reader at
// once for a fixed time and reports each reader's GB/s. Device buffers are small so they fit beside a resident engine.
//
//   native_bandwidth_probe <seconds> <cpu threads> <primary gpu> <helper gpu>
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
constexpr std::size_t MiB = std::size_t{1} << 20;
constexpr std::size_t RECORD = 32 * MiB;

struct CopyReader {
    int device; std::size_t device_bytes; cudaStream_t stream = nullptr; void* dst = nullptr; const void* src = nullptr; cudaEvent_t a = nullptr, b = nullptr;
    CopyReader(int dev, std::size_t bytes, const void* host) : device(dev), device_bytes(bytes), src(host) {
        ck(cudaSetDevice(device), "set device"); ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        ck(cudaMalloc(&dst, device_bytes), "device buffer"); ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event");
    }
    ~CopyReader() { cudaSetDevice(device); cudaFree(dst); cudaStreamDestroy(stream); cudaEventDestroy(a); cudaEventDestroy(b); }
    double run(double seconds, std::atomic<bool>& go) {
        ck(cudaSetDevice(device), "set device");
        while (!go.load()) {}
        const auto end = Clock::now() + std::chrono::duration<double>(seconds);
        std::size_t bytes = 0, slot = 0;
        ck(cudaEventRecord(a, stream), "record");
        while (Clock::now() < end) {
            for (int i = 0; i < 4; ++i) {   // keep a few copies queued so the engine never idles
                char* to = static_cast<char*>(dst) + (slot++ % (device_bytes / RECORD)) * RECORD;
                ck(cudaMemcpyAsync(to, src, RECORD, cudaMemcpyHostToDevice, stream), "copy");
                bytes += RECORD;
            }
            ck(cudaStreamSynchronize(stream), "sync");
        }
        ck(cudaEventRecord(b, stream), "record"); ck(cudaEventSynchronize(b), "sync");
        float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "elapsed");
        return bytes / (ms * 1e-3) / 1e9;
    }
};

static double cpu_read(const std::uint64_t* data, std::size_t words, int threads, double seconds, std::atomic<bool>& go) {
    std::vector<std::thread> pool; std::vector<std::size_t> read(threads, 0); std::atomic<std::uint64_t> sink{0};
    for (int t = 0; t < threads; ++t) pool.emplace_back([&, t] {
        const std::size_t span = words / threads, begin = t * span;
        std::uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0; std::size_t n = 0;
        while (!go.load()) {}
        const auto end = Clock::now() + std::chrono::duration<double>(seconds);
        while (Clock::now() < end) {
            const std::uint64_t* p = data + begin;
            for (std::size_t i = 0; i + 4 <= span; i += 4) { s0 += p[i]; s1 += p[i + 1]; s2 += p[i + 2]; s3 += p[i + 3]; }
            n += span;
        }
        read[t] = n * sizeof(std::uint64_t); sink += s0 + s1 + s2 + s3;
    });
    const auto began = Clock::now();
    for (auto& th : pool) th.join();
    const double elapsed = std::chrono::duration<double>(Clock::now() - began).count();
    std::size_t total = 0; for (auto r : read) total += r;
    if (sink.load() == 42) std::printf(" ");
    return total / elapsed / 1e9;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: native_bandwidth_probe <seconds> <cpu threads> <primary gpu> <helper gpu>");
        const double seconds = std::atof(argv[1]); const int threads = std::atoi(argv[2]); const int primary = std::atoi(argv[3]), helper = std::atoi(argv[4]);
        if (!(seconds > 0 && seconds <= 30) || threads < 1 || threads > 64) throw std::runtime_error("invalid seconds or threads");
        const std::size_t cpu_bytes = 2048 * MiB;
        std::vector<std::uint64_t> cpu_data(cpu_bytes / sizeof(std::uint64_t));
        for (std::size_t i = 0; i < cpu_data.size(); ++i) cpu_data[i] = i * 2654435761u;
        void* pinned_primary = nullptr; void* pinned_helper = nullptr;
        ck(cudaHostAlloc(&pinned_primary, RECORD, cudaHostAllocPortable), "pinned"); ck(cudaHostAlloc(&pinned_helper, RECORD, cudaHostAllocPortable), "pinned");
        std::memset(pinned_primary, 1, RECORD); std::memset(pinned_helper, 2, RECORD);
        CopyReader p(primary, 8 * RECORD, pinned_primary), h(helper, 2 * RECORD, pinned_helper);
        struct Case { const char* name; bool cpu, gp, gh; };
        const Case cases[] = {{"cpu", true, false, false}, {"primary", false, true, false}, {"helper", false, false, true},
                              {"cpu+primary", true, true, false}, {"cpu+helper", true, false, true}, {"primary+helper", false, true, true},
                              {"cpu+primary+helper", true, true, true}};
        std::printf("{\"seconds\": %.1f, \"cpu_threads\": %d, \"record_mib\": 32, \"cases\": [\n", seconds, threads);
        bool first = true;
        for (int repeat = 0; repeat < 2; ++repeat) for (const auto& c : cases) {
            std::atomic<bool> go{false}; double cpu = 0, gp = 0, gh = 0;
            std::thread tp, th;
            if (c.gp) tp = std::thread([&] { gp = p.run(seconds, go); });
            if (c.gh) th = std::thread([&] { gh = h.run(seconds, go); });
            std::thread tc;
            if (c.cpu) tc = std::thread([&] { cpu = cpu_read(cpu_data.data(), cpu_data.size(), threads, seconds, go); });
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            go.store(true);
            if (tc.joinable()) tc.join(); if (tp.joinable()) tp.join(); if (th.joinable()) th.join();
            std::printf("%s {\"case\": \"%s\", \"repeat\": %d, \"cpu_gbps\": %.2f, \"primary_gbps\": %.2f, \"helper_gbps\": %.2f, \"total_gbps\": %.2f}\n",
                        first ? " " : ",", c.name, repeat, cpu, gp, gh, cpu + gp + gh);
            first = false; std::fflush(stdout);
        }
        std::printf("]}\n");
        cudaFreeHost(pinned_primary); cudaFreeHost(pinned_helper);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "native_bandwidth_probe: %s\n", e.what()); return 1; }
}
