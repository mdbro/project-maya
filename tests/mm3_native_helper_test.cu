// Focused helper lifetime/group/scatter test. Link mm3_native_helper.cu with
// Infernix minimax_nvfp4_ffn record and fast CUDA implementation.
#include "../src/mm3/mm3_native_helper.hpp"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error.c_str()); return 1; } } while (0)
int main(int argc, char** argv) {
    using H = mm3::NativeHelper;
    std::string error;
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t n = infernix::ops::minimax_ffn::record_stride(H::kHidden, H::kIntermediate);
    std::vector<unsigned char> record(n, 0);
    for (size_t o = 0; o < n; o += 144) for (int i = 0; i < 16; ++i) record[o + 128 + i] = 0x38;
    const float scales[3] = {1, 1, 1};
    void* staged = nullptr;
    CHECK(cudaHostAlloc(&staged, n, cudaHostAllocPortable) == cudaSuccess);
    std::memcpy(staged, record.data(), n);
    // passes of up to 64 tokens (the layer-by-layer prompt read's long passes); everything below 16 tokens is unchanged
    setenv("MM3_HELPER_PASS_TOKENS", "64", 1);
    H helper(device, 4);
    CHECK(helper.pass_tokens() == 64);
    CHECK(!helper.add_expert(3, 4, staged, n + 256, scales, scales, error));
    CHECK(helper.add_expert(3, 4, staged, n, scales, scales, error));
    CHECK(helper.add_expert(3, 5, staged, n, scales, scales, error));
    CHECK(helper.add_expert(3, 7, staged, n, scales, scales, error));
    CHECK(!helper.add_expert(3, 5, staged, n, scales, scales, error));
    // Nonzero codes produce finite positive FFN values, to exercise overflow
    // in weighted accumulation separately from a nonfinite FFN output.
    auto* payload = static_cast<unsigned char*>(staged);
    std::memset(payload, 0x11, n);
    for (size_t o = 0; o < n; o += 144) for (int i = 0; i < 16; ++i) payload[o + 128 + i] = 0x38;
    CHECK(helper.add_expert(3, 6, staged, n, scales, scales, error));
    CHECK(cudaFreeHost(staged) == cudaSuccess);
    CHECK(helper.seal(error));
    CHECK(helper.ready(3)[0] == ((1u << 4) | (1u << 5) | (1u << 6) | (1u << 7)));
    CHECK(helper.ready(4)[0] == 0);
    std::vector<uint16_t> x(H::kMaxTokens * H::kHidden, 0x3f80); // BF16 1
    H::Job a{}; a.layer = 3; a.expert = 4; a.columns = 2; a.token[0] = 0; a.token[1] = 1; a.weight[0] = a.weight[1] = 0.5f;
    H::Job b{}; b.layer = 3; b.expert = 5; b.columns = 1; b.token[0] = 1; b.weight[0] = 0.75f;
    const std::vector<H::Job> jobs{a, b};
    H::Job excess{};excess.layer=3;excess.expert=4;excess.columns=5;
    for(int c=0;c<5;++c){excess.token[c]=0;excess.weight[c]=1.f;}
    CHECK(helper.submit(1,x.data(),{excess},error)==0);
    auto seq = helper.submit(3, x.data(), jobs, error); CHECK(seq != 0);
    CHECK(helper.submit(3, x.data(), jobs, error) == 0);
    std::vector<float> out(3 * H::kHidden, 1.0f);
    CHECK(!helper.collect(seq + 1, out.data(), out.size(), error));
    CHECK(helper.collect(seq, out.data(), out.size(), error));
    for (float value : out) CHECK(value == 0.0f);
    CHECK(!helper.collect(seq, out.data(), out.size(), error));
    seq = helper.submit(3, x.data(), jobs, error); CHECK(seq != 0);
    std::atomic<bool> go{false}; bool collected[2]{};
    std::vector<float> out2(out.size()); std::string errors[2];
    std::thread t1([&] { while (!go.load(std::memory_order_acquire)) {} collected[0] = helper.collect(seq, out.data(), out.size(), errors[0]); });
    std::thread t2([&] { while (!go.load(std::memory_order_acquire)) {} collected[1] = helper.collect(seq, out2.data(), out2.size(), errors[1]); });
    go.store(true, std::memory_order_release); t1.join(); t2.join();
    CHECK(collected[0] != collected[1]);
    auto duplicate = jobs; duplicate.push_back(b);
    CHECK(helper.submit(3, x.data(), duplicate, error) == 0);
    // Four distinct experts across all 16 tokens occupy every one of the 64
    // routed columns. Expert 6 has nonzero weights but a zero mixture weight;
    // the other three have all-zero codes, so every scattered result is zero.
    // Exercise both maximum-width grouping and the 64-job width-one endpoint.
    std::vector<float> full_out(size_t(H::kMaxTokens)*H::kHidden,1.f);
    for(int width : {H::kMaxColumns,1}) {
        std::vector<H::Job> full;
        for(int expert : {4,5,6,7}) for(int first=0;first<H::kMaxTokens;first+=width) {
            H::Job job{};job.layer=3;job.expert=expert;job.columns=width;
            for(int c=0;c<width;++c){job.token[c]=first+c;job.weight[c]=expert==6?0.f:1.f;}
            full.push_back(job);
        }
        CHECK(full.size()==size_t(64/width));
        seq=helper.submit(H::kMaxTokens,x.data(),full,error);CHECK(seq!=0);
        CHECK(helper.collect(seq,full_out.data(),full_out.size(),error));
        for(float value:full_out)CHECK(value==0.f);
    }
    // Occupancy fills the gaps on the helper's own stream. Each pass must release it: results stay
    // the same, and no pass waits out the spinner's bound of at least 20 ms.
    CHECK(!helper.occupancy());
    helper.set_occupancy(true);
    CHECK(helper.occupancy());
    std::vector<double> occupied_ms;
    for (int pass = 0; pass < 200; ++pass) {
        // The gap lets the spinner start, so the next pass really has something to release.
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        const auto began = std::chrono::steady_clock::now();
        seq = helper.submit(3, x.data(), jobs, error); CHECK(seq != 0);
        CHECK(helper.collect(seq, out.data(), out.size(), error));
        occupied_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
        for (float value : out) CHECK(value == 0.0f);
    }
    std::sort(occupied_ms.begin(), occupied_ms.end());
    CHECK(occupied_ms[occupied_ms.size() / 2] < 10.0);
    CHECK(helper.occupancy_launches() >= 199);
    helper.set_occupancy(false);
    CHECK(!helper.occupancy());
    std::printf("occupied pass median %.3f ms, launches %llu\n", occupied_ms[occupied_ms.size() / 2], (unsigned long long)helper.occupancy_launches());
    // helper_fast's graphs for one-column passes: the separate calls, the graphs with host copy nodes and the zero-copy
    // graphs (helper_zero_copy=2: each pass's buffers read back through the copy engine) give the same bits, for 1 to 4
    // jobs. Varied inputs and the nonzero expert 6 between all-zero ones, so a misplaced row or descriptor shows.
    {
        std::vector<uint16_t> xv(size_t(H::kMaxTokens) * H::kHidden);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = uint16_t(0x3c00u + ((uint32_t(i) * 2654435761u) >> 22));   // 2^-7 .. 2
        const int experts[4] = {6, 4, 6, 7};
        for (int jobs_n = 1; jobs_n <= 4; ++jobs_n) {
            std::vector<H::Job> one;
            for (int j = 0; j < jobs_n; ++j) {
                H::Job job{}; job.layer = 3; job.expert = experts[j]; job.columns = 1; job.token[0] = 2 * j + 1; job.weight[0] = 0.375f;
                one.push_back(job);
            }
            std::vector<float> got[3];
            for (int arm = 0; arm < 3; ++arm) {   // 0 separate calls, 1 copy-node graphs, 2 zero-copy graphs, checked
                helper.set_fast(arm > 0);
                helper.set_zero_copy(arm == 2 ? 2 : 0);
                const uint64_t launches = helper.graph_launches(), checked = helper.zero_copy_checked();
                got[arm].assign(size_t(H::kMaxTokens) * H::kHidden, 1.f);
                seq = helper.submit(H::kMaxTokens, xv.data(), one, error); CHECK(seq != 0);
                CHECK(helper.collect(seq, got[arm].data(), got[arm].size(), error));
                CHECK(helper.graph_launches() == launches + (arm > 0 ? 1 : 0));
                CHECK(helper.zero_copy_checked() == checked + (arm == 2 ? 1 : 0));
            }
            CHECK(helper.zero_copy_mismatches() == 0);
            bool nonzero = false;
            for (float value : got[0]) nonzero |= value != 0.f;
            CHECK(nonzero);
            CHECK(std::memcmp(got[0].data(), got[1].data(), got[0].size() * sizeof(float)) == 0);
            CHECK(std::memcmp(got[0].data(), got[2].data(), got[0].size() * sizeof(float)) == 0);
        }
        helper.set_fast(false); helper.set_zero_copy(0);
        std::printf("helper graphs: copy nodes and zero-copy match the separate calls for 1-4 jobs (%llu checked)\n",
                    (unsigned long long)helper.zero_copy_checked());
    }
    // A long pass: four 16-token groups' jobs in one pass (tokens shifted by 16 a group) give the same bits as four
    // one-group passes. Mixed widths, the nonzero expert 6 among all-zero ones, varied inputs and weights.
    {
        constexpr int G = 4, T = 16;
        std::vector<uint16_t> xv(size_t(G * T) * H::kHidden);
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = uint16_t(0x3c00u + ((uint32_t(i) * 2246822519u) >> 22));
        std::vector<std::vector<H::Job>> groups(G);
        for (int g = 0; g < G; ++g) {
            const int widths[4] = {1 + g, 8, 3, 2 + (g % 3)};
            const int experts[4] = {6, 4, 7, 5};
            for (int k = 0; k < 4; ++k) for (int first = (g + k) % 3; first + widths[k] <= T; first += widths[k] + 1) {
                H::Job job{}; job.layer = 3; job.expert = experts[k]; job.columns = widths[k];
                for (int c = 0; c < widths[k]; ++c) { job.token[c] = first + c; job.weight[c] = 0.125f * float(1 + ((g + c + k) % 5)); }
                groups[size_t(g)].push_back(job);
            }
        }
        std::vector<float> one(size_t(G * T) * H::kHidden, 1.f), longp(size_t(G * T) * H::kHidden, 1.f);
        std::vector<H::Job> all;
        for (int g = 0; g < G; ++g) {
            seq = helper.submit(T, xv.data() + size_t(g * T) * H::kHidden, groups[size_t(g)], error); CHECK(seq != 0);
            CHECK(helper.collect(seq, one.data() + size_t(g * T) * H::kHidden, size_t(T) * H::kHidden, error));
            for (H::Job job : groups[size_t(g)]) { for (int c = 0; c < job.columns; ++c) job.token[c] += g * T; all.push_back(job); }
        }
        seq = helper.submit(G * T, xv.data(), all, error); CHECK(seq != 0);
        CHECK(helper.collect(seq, longp.data(), longp.size(), error));
        bool nonzero = false;
        for (float value : one) nonzero |= value != 0.f;
        CHECK(nonzero);
        CHECK(std::memcmp(one.data(), longp.data(), one.size() * sizeof(float)) == 0);
        std::vector<uint16_t> x65(size_t(65) * H::kHidden, 0x3f80);
        H::Job j65{}; j65.layer = 3; j65.expert = 4; j65.columns = 1; j65.token[0] = 64; j65.weight[0] = 1.f;
        CHECK(helper.submit(65, x65.data(), {j65}, error) == 0);   // above the pass capacity
        std::printf("long pass: %zu jobs over %d tokens equal four one-group passes bit for bit\n", all.size(), G * T);
    }
    H::Job overflow{}; overflow.layer = 3; overflow.expert = 6; overflow.columns = 1;
    overflow.token[0] = 0; overflow.weight[0] = std::numeric_limits<float>::max();
    {   // with a refused list (the user's decision of 2026-11-14): the column is left out and listed, the helper stays usable
        std::vector<H::Refused> refused;
        seq = helper.submit(3, x.data(), {overflow}, error); CHECK(seq != 0);
        CHECK(helper.collect(seq, out.data(), out.size(), error, nullptr, &refused));
        CHECK(refused.size() == 1 && refused[0].sum && refused[0].layer == 3 && refused[0].expert == 6 && refused[0].token == 0);
        CHECK(refused[0].bad == 0);                                                          // (its values were finite)
        CHECK(refused[0].detail.rfind("nonfinite helper weighted sum (device ", 0) == 0);
        CHECK(refused[0].detail.find("d_y = h_y, d_x = h_x") != std::string::npos);        // (a finite column, read back as sent)
        bool zero = true;
        for (float v : out) zero &= v == 0.f;
        CHECK(zero);                                                                          // (its row holds no column)
        std::printf("refused detail: %s\n", refused[0].detail.c_str());
        seq = helper.submit(3, x.data(), jobs, error); CHECK(seq != 0);                       // (not poisoned)
        CHECK(helper.collect(seq, out.data(), out.size(), error, nullptr, &refused) && refused.empty());
    }
    seq = helper.submit(3, x.data(), {overflow}, error); CHECK(seq != 0);
    CHECK(!helper.collect(seq, out.data(), out.size(), error));
    CHECK(error.rfind("nonfinite helper weighted sum (device ", 0) == 0);   // (nonfinite_detail follows)
    CHECK(error.find("d_y = h_y, d_x = h_x") != std::string::npos);        // (a finite column, read back as sent)
    std::printf("nonfinite detail: %s\n", error.c_str());
    CHECK(helper.submit(3, x.data(), jobs, error) == 0); // failed output poisons helper
    std::puts("native helper: OK"); return 0;
}
