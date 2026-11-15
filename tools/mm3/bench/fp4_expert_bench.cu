// Standalone bench of kernel item 7 on the RTX 5090: the donor's FP4 expert kernel (Infernix-minimax native_fp4_ffn,
// what serving runs) against src/mm3/mm3_native_fp4_fast.cu (the same kernel with each warp's loads of P steps batched),
// per pass shape: 1, 2, 4, 8 rows (each job's columns) and 1 to 16 jobs (experts) a call.
//
// Reported per shape: microseconds a call (CUDA graphs of the call's three kernels, replayed round-robin over 16 record sets
// so no expert is still in the 5090's L2 when it is read again; CUDA events), the experts' bytes over that time against
// the card's measured read bandwidth (a streaming read of the records), and the check:
// every variant's outputs against the donor's, bit for bit (the batching changes only when loads are issued).
//
//   build:  cmake --build <build dir> --target mm3_fp4_expert_bench -j 15        (not while a timed request runs)
//   run:    <build dir>/mm3_fp4_expert_bench [--iters 200] [--profile]           (about 0.6 GB on the 5090)
//   profile (one clock, the kernels one by one):
//           ncu --clock-control base --nvtx --nvtx-include "regex:fp4 (donor|batched P=4) .*/" --set full -o fp4-profile \
//               <build dir>/mm3_fp4_expert_bench --profile
//   (--profile: each shape and variant once, outside graphs, inside an NVTX range "fp4 <variant> <jobs>x<rows>"; no '/'
//   in the names, which ncu reads as nesting)
#include "mm3_native_fp4_fast.cuh"
#include "ops/minimax_nvfp4_ffn/native_fp4.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mf = infernix::ops::minimax_ffn;
using mf::FastJob;
constexpr int H = 6144, I = 3072, POOL = 16, C = 8;
void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
__global__ void read_kernel(const uint4* p, std::size_t n, unsigned* sink) {
    unsigned acc = 0;
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x) {
        const uint4 v = __ldcs(p + i); acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) *sink = acc;
}
// Host buffers may be pageable; return only when this stream's copy is done.
cudaError_t copy_sync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, kind, stream);
    return e == cudaSuccess ? cudaStreamSynchronize(stream) : e;
}
int find_5090() {
    int n = 0; ck(cudaGetDeviceCount(&n), "count");
    for (int i = 0; i < n; ++i) { cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, i), "props"); if (std::strstr(p.name, "RTX 5090")) return i; }
    throw std::runtime_error("no RTX 5090");
}
}  // namespace

int main(int argc, char** argv) {
    int iters = 200; bool profile = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
        else if (a == "--profile") profile = true;
        else { std::fprintf(stderr, "usage: %s [--iters N] [--profile]\n", argv[0]); return 2; }
    }
    try {
        const int dev = find_5090();
        ck(cudaSetDevice(dev), "set device");
        cudaStream_t s; ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, dev), "props");
        const std::size_t rec = mf::record_stride(H, I), payload = mf::record_bytes(H, I);
        std::printf("fp4_expert_bench: CUDA%d %s, %d SMs; %d synthetic experts of %.2f MB; %d replays a shape\n", dev, prop.name,
                    prop.multiProcessorCount, POOL, payload / 1e6, iters);
        std::mt19937_64 rng(20261114);
        std::vector<std::uint8_t> host(rec * POOL);
        for (std::size_t i = 0; i < host.size(); ++i) host[i] = (i % rec) < payload && (i % 144) >= 128 ? std::uint8_t(0x30 + (rng() & 7)) : std::uint8_t(rng());
        std::uint8_t* records = nullptr;
        ck(cudaMalloc(&records, host.size()), "records");
        ck(copy_sync(records, host.data(), host.size(), cudaMemcpyHostToDevice, s), "records up");
        cudaEvent_t e0, e1; ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
        // the card's read bandwidth: the records streamed
        unsigned* sink = nullptr; ck(cudaMalloc(&sink, 4), "sink");
        const std::size_t n16 = host.size() / 16;
        read_kernel<<<prop.multiProcessorCount * 8, 512, 0, s>>>(reinterpret_cast<const uint4*>(records), n16, sink);
        ck(cudaEventRecord(e0, s), "e0");
        for (int i = 0; i < 10; ++i) read_kernel<<<prop.multiProcessorCount * 8, 512, 0, s>>>(reinterpret_cast<const uint4*>(records), n16, sink);
        ck(cudaEventRecord(e1, s), "e1"); ck(cudaEventSynchronize(e1), "read");
        float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        const double gbs = 10.0 * host.size() / (ms * 1e-3) / 1e9;
        std::printf("read bandwidth: %.0f GB/s (the records streamed); one expert at that rate: %.1f us\n\n", gbs, payload / (gbs * 1e3));
        // jobs' inputs and outputs, workspace
        constexpr int kJobs = 16;
        std::uint16_t *x = nullptr, *y = nullptr;
        ck(cudaMalloc(&x, std::size_t(kJobs) * C * H * 2), "x"); ck(cudaMalloc(&y, std::size_t(kJobs) * C * H * 2), "y");
        std::vector<std::uint16_t> xh(std::size_t(kJobs) * C * H);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xh) { float f = nd(rng); std::uint32_t u; std::memcpy(&u, &f, 4); v = std::uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
        ck(copy_sync(x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice, s), "x up");
        const std::size_t per = mf::native_fp4_workspace_bytes(C);
        void* work = nullptr; ck(cudaMalloc(&work, per * kJobs), "work");
        constexpr int kSets = 16;   // (record sets the replays rotate over: >= 7 other experts read between two uses of one)
        FastJob* jobs = nullptr; ck(cudaMalloc(&jobs, sizeof(FastJob) * kJobs * kSets), "jobs");
        const int variants[] = {0, 1, 2, 4, 8, 12};   // 0 the donor; P the batched copy
        const auto name = [](int v) { return v == 12 ? std::string("P=4 parallel") : v ? "batched P=" + std::to_string(v) : std::string("donor"); };
        std::printf("%-6s %-5s", "rows", "jobs");
        for (int v : variants) std::printf(" %14s", name(v).c_str());
        std::printf("   (us a call; in brackets the experts' bytes a us as a share of the read bandwidth)\n");
        int bad = 0;
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 8}) for (int njobs : {1, 2, 4, 8, 16}) {
            std::vector<FastJob> hj(std::size_t(kJobs) * kSets);
            for (int set = 0; set < kSets; ++set)
                for (int j = 0; j < njobs; ++j) {
                    FastJob f{}; f.record = records + rec * ((set * njobs + j) % POOL);
                    for (int k = 0; k < 3; ++k) f.multiplier[k] = 1.0f;
                    f.x = x + std::size_t(j) * C * H; f.y = y + std::size_t(j) * C * H; f.columns = rows;
                    hj[std::size_t(set) * kJobs + j] = f;
                }
            ck(copy_sync(jobs, hj.data(), hj.size() * sizeof(FastJob), cudaMemcpyHostToDevice, s), "jobs up");
            const auto call_set = [&](int v, int set) {
                FastJob* js = jobs + std::size_t(set) * kJobs;
                if (v) mm3::fp4_fast::ffn(js, njobs, C, work, v, s); else mf::native_fp4_ffn(js, njobs, C, work, s);
            };
            const auto call = [&](int v) { call_set(v, 0); };
            std::vector<std::uint16_t> want(std::size_t(njobs) * C * H), got(want.size());
            std::vector<std::uint8_t> packed_ref(per * njobs), packed_got(packed_ref.size());
            const auto packing_equal = [&] {
                constexpr size_t xr = H/2 + H/16, hr = I/2 + I/16;
                const size_t hidden_offset = (size_t(C) * xr + 255) & ~size_t(255);
                for (int j = 0; j < njobs; ++j)
                    for (int c = 0; c < rows; ++c) {
                        const size_t input = size_t(j) * per + c * xr;
                        const size_t hidden = size_t(j) * per + hidden_offset + c * hr;
                        if (std::memcmp(packed_ref.data()+input, packed_got.data()+input, xr) ||
                            std::memcmp(packed_ref.data()+hidden, packed_got.data()+hidden, hr)) return false;
                    }
                return true;
            };
            std::printf("%-6d %-5d", rows, njobs);
            for (int v : variants) {
                ck(cudaMemsetAsync(y, 0, want.size() * 2, s), "clear");
                call(v); ck(cudaStreamSynchronize(s), "call");
                ck(copy_sync(v ? got.data() : want.data(), y, want.size() * 2, cudaMemcpyDeviceToHost, s), "y back");
                ck(copy_sync(v ? packed_got.data() : packed_ref.data(), work, packed_ref.size(), cudaMemcpyDeviceToHost, s), "packing back");
                const bool same = !v || (got == want && packing_equal());
                bad += !same;
                if (profile) {
                    const std::string range = "fp4 " + name(v) + " " + std::to_string(njobs) + "x" + std::to_string(rows);
                    nvtxRangePushA(range.c_str()); call(v); ck(cudaStreamSynchronize(s), "profiled call"); nvtxRangePop();
                    std::printf(" %14s", same ? "same" : "DIFFERS");
                    continue;
                }
                cudaGraphExec_t ge[kSets] = {};
                for (int set = 0; set < kSets; ++set) {
                    cudaGraph_t g = nullptr;
                    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                    call_set(v, set);
                    ck(cudaStreamEndCapture(s, &g), "end capture"); ck(cudaGraphInstantiate(&ge[set], g, 0), "instantiate");
                    cudaGraphDestroy(g);
                }
                for (int i = 0; i < kSets; ++i) ck(cudaGraphLaunch(ge[i], s), "warm");
                ck(cudaEventRecord(e0, s), "e0");
                for (int i = 0; i < iters; ++i) ck(cudaGraphLaunch(ge[i % kSets], s), "replay");
                ck(cudaEventRecord(e1, s), "e1"); ck(cudaEventSynchronize(e1), "replays");
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                // Replay correctness is checked separately from uncaptured calls.
                ck(cudaGraphLaunch(ge[0], s), "check replay");
                ck(cudaStreamSynchronize(s), "check replay done");
                ck(copy_sync(got.data(), y, want.size()*2, cudaMemcpyDeviceToHost, s), "replay outputs");
                ck(copy_sync(packed_got.data(), work, packed_ref.size(), cudaMemcpyDeviceToHost, s), "replay packing");
                if (got != want || !packing_equal()) {
                    ++bad;
                    std::fprintf(stderr, "graph output/packing mismatch: %s %dx%d\n", name(v).c_str(), njobs, rows);
                }
                const double us = 1000.0 * ms / iters;
                char cell[32];
                std::snprintf(cell, sizeof cell, "%6.1f (%3.0f%%)%s", us, 100.0 * njobs * payload / (us * 1e3) / gbs, same ? "" : "!");
                std::printf(" %14s", cell);
                for (auto g : ge) cudaGraphExecDestroy(g);
            }
            std::printf("\n");
        }
        std::printf("\n%s\n", bad ? "fp4_expert_bench: a batched variant DIFFERS from the donor (marked !)" :
                                    "fp4_expert_bench: every variant's outputs and NVFP4 input/hidden codes+scales equal the donor's (direct and graph replay)");
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fp4_expert_bench: %s\n", e.what());
        return 1;
    }
}
