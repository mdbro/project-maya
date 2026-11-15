// The helper hand-off (helper_handoff, kernel item 1) on the CPU, with the production helper compiled against the fake
// CUDA API (tests/mm3_helper_fake_source.cmake) and the fake card's expert below:
//   - plan_jobs (the 5090's request kernel plans each card's jobs with it) against window_post's plan and
//     NativeHelper::submit's order, for random routings, masks and widths: the same jobs, columns, rows and weights in
//     the same order - the order the host merges in, so the merged rows are today's;
//   - NativeHelper::merge_external (the host's merge of a hand-off pass) against collect for the same jobs and columns,
//     bit for bit, and its refused list for a non-finite column as collect's;
//   - records_version moves with every record change, and handoff_busy keeps a rewrite, a replacement and a live
//     withdrawal from starting while a released pass is in flight;
//   - the slots' layout (sizes the 5090 and the cards agree on).
// The kernels themselves (request, take, the expert kernels' grid form, result) run on the cards: tests/mm3_native_helper_test
// is not extended here; the standalone bench (tools/mm3/bench/helper_handoff_bench.cu) checks them bit for bit.
#include "../src/mm3/mm3_native_helper.hpp"
#include "../src/mm3/mm3_helper_handoff.hpp"
#include "../src/mm3/mm3_helper_q.cuh"
#include "cuda_runtime_api.h"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) { std::fprintf(stderr, "mm3_helper_handoff_test:%d: FAIL: %s\n", line, what); ++failures; }
}
#define CHECK(x) check((x), #x, __LINE__)
uint16_t bf16(float f) { uint32_t b; std::memcpy(&b, &f, 4); return uint16_t(b >> 16); }
constexpr unsigned char kBad = 0x70;
}

namespace infernix::ops::minimax_ffn {
size_t record_bytes(int, int) { return 144; }
size_t record_stride(int, int) { return 144; }
size_t fast_workspace_bytes(int, int, int) { return 256; }
// The fake card's expert: column c of a job holds (record code) / 16 + c / 64 everywhere (so columns differ), and the kBad
// record's first column a NaN at element 5.
void fast_ffn_cuda_packed(const FastJob* jobs, int n, int h, int, int, void*, cudaStream_t) {
    for (int j = 0; j < n; ++j)
        for (int c = 0; c < jobs[j].columns; ++c) {
            uint16_t* y = jobs[j].y + size_t(c) * size_t(h);
            const unsigned char code = jobs[j].record[0];
            std::fill_n(y, size_t(h), bf16(float(code) / 16.0f + float(c) / 64.0f));
            if (code == kBad && c == 0) y[5] = 0x7fc0;
        }
}
}

using H = mm3::NativeHelper;
namespace ho = mm3::handoff;

// window_post's plan for one card (the loop over tokens and picks, a pick joining the first job of its expert with room)
// followed by NativeHelper::submit's order (widths 1..8, each in creation order): the reference plan_jobs must match.
static std::vector<H::Job> reference(const std::vector<int>& top, const std::vector<float>& w, int T, int k, float scale,
                                     bool four, const std::vector<uint8_t>& mine) {
    std::vector<H::Job> jobs;
    std::vector<int> per(size_t(T), 0);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < k; ++i) {
            const int x = top[size_t(t * k + i)];
            if (!mine[size_t(x)]) continue;
            if (four) { if (per[size_t(t)] >= 4) continue; ++per[size_t(t)]; }
            const float wt = w[size_t(t * k + i)] * scale;
            auto it = std::find_if(jobs.begin(), jobs.end(), [&](const H::Job& j) { return j.expert == x && j.columns < H::kMaxColumns; });
            if (it == jobs.end()) { H::Job j; j.expert = x; jobs.push_back(j); it = jobs.end() - 1; }
            const int c = it->columns++; it->token[c] = t; it->weight[c] = wt;
        }
    std::vector<H::Job> out;
    for (int cols = 1; cols <= H::kMaxColumns; ++cols) for (const auto& j : jobs) if (j.columns == cols) out.push_back(j);
    return out;
}

static void test_plan() {
    std::mt19937 rng(20261114);
    int cases = 0, wide = 0;
    for (int round = 0; round < 4000; ++round) {
        const int T = 1 + int(rng() % 16), k = 4;
        const float scale = 2.5f;
        std::vector<int> top(size_t(T * k));
        std::vector<float> w(size_t(T * k));
        // few distinct experts in some rounds, so jobs fill past eight columns and split
        const int pool = round % 3 == 0 ? 6 : 128;
        for (int t = 0; t < T; ++t) {
            std::vector<int> row;
            while (int(row.size()) < k) { const int e = int(rng() % unsigned(pool)) * (128 / pool); if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e); }
            for (int i = 0; i < k; ++i) { top[size_t(t * k + i)] = row[size_t(i)]; w[size_t(t * k + i)] = std::uniform_real_distribution<float>(0.01f, 1.0f)(rng); }
        }
        std::vector<uint8_t> helper(128), third(128);
        uint32_t hm[4] = {}, tm[4] = {};
        for (int e = 0; e < 128; ++e) {
            helper[size_t(e)] = rng() % 3 == 0;
            third[size_t(e)] = !helper[size_t(e)] && rng() % 4 == 0;
            if (helper[size_t(e)]) hm[e >> 5] |= 1u << (e & 31);
            if (third[size_t(e)]) tm[e >> 5] |= 1u << (e & 31);
        }
        for (int c = 0; c < 2; ++c) {
            std::vector<ho::Job> scratch(ho::kMaxJobs), out(ho::kMaxJobs);
            const auto take = [&](int, int, int e) { return c == 0 ? ho::bit(hm, e) : (!ho::bit(hm, e) && ho::bit(tm, e)); };
            const int n = ho::plan_jobs(top.data(), w.data(), T, k, scale, c == 1, take, scratch.data(), out.data());
            const auto ref = reference(top, w, T, k, scale, c == 1, c == 0 ? helper : third);
            bool same = n == int(ref.size());
            for (int j = 0; same && j < n; ++j) {
                same = out[size_t(j)].expert == ref[size_t(j)].expert && out[size_t(j)].columns == ref[size_t(j)].columns;
                for (int q = 0; same && q < ref[size_t(j)].columns; ++q)
                    same = out[size_t(j)].token[q] == ref[size_t(j)].token[q] &&
                           std::memcmp(&out[size_t(j)].weight[q], &ref[size_t(j)].weight[q], sizeof(float)) == 0;
                wide += same && ref[size_t(j)].columns == H::kMaxColumns;
            }
            CHECK(same);
            ++cases;
        }
    }
    std::printf("plan_jobs: %d cases matched window_post's plan in submit's order (%d jobs filled to eight columns)\n", cases, wide);
    CHECK(wide > 0);
}

static std::vector<unsigned char> record(unsigned char code) {
    std::vector<unsigned char> x(144, code);
    std::fill(x.begin() + 128, x.end(), 0x38);
    return x;
}

static void test_merge() {
    std::string err;
    H h(0, 6);
    const float sw[3] = {1, 2, 3}, si[3] = {4, 5, 6};
    for (int e = 0; e < 4; ++e) { auto r = record(e == 3 ? kBad : (unsigned char) (0x10 * (e + 1))); CHECK(h.add_expert(2, e, r.data(), r.size(), sw, si, err)); }
    const uint64_t v0 = h.records_version();
    CHECK(v0 == 4);
    CHECK(h.seal(err));
    // three tokens; expert 0 on all three (one job of three columns), expert 1 on token 2, expert 2 on tokens 0 and 1
    H::Job a{}; a.layer = 2; a.expert = 0; a.columns = 3; a.token[0] = 0; a.token[1] = 1; a.token[2] = 2; a.weight[0] = 0.5f; a.weight[1] = 0.25f; a.weight[2] = 1.5f;
    H::Job b{}; b.layer = 2; b.expert = 1; b.columns = 1; b.token[0] = 2; b.weight[0] = 0.75f;
    H::Job c{}; c.layer = 2; c.expert = 2; c.columns = 2; c.token[0] = 0; c.token[1] = 1; c.weight[0] = 1.25f; c.weight[1] = 0.125f;
    const std::vector<H::Job> plan{a, b, c};
    std::vector<H::Job> ordered;   // (submit's order: by width)
    for (int cols = 1; cols <= H::kMaxColumns; ++cols) for (const auto& j : plan) if (j.columns == cols) ordered.push_back(j);
    std::vector<uint16_t> x(size_t(3) * H::kHidden, bf16(0.5f));
    std::vector<float> want(size_t(3) * H::kHidden), got(size_t(3) * H::kHidden, 7.0f);
    const uint64_t seq = h.submit(3, x.data(), plan, err);
    CHECK(seq != 0 && h.collect(seq, want.data(), want.size(), err));
    // the same columns as the card's result slot holds them: jobs in submit's order, then their columns
    std::vector<uint16_t> columns;
    for (const auto& j : ordered)
        for (int q = 0; q < j.columns; ++q) {
            const unsigned char code = (unsigned char) (0x10 * (j.expert + 1));
            columns.insert(columns.end(), size_t(H::kHidden), bf16(float(code) / 16.0f + float(q) / 64.0f));
        }
    std::vector<H::Refused> refused;
    CHECK(h.merge_external(3, ordered, columns.data(), got.data(), got.size(), err, &refused));
    CHECK(refused.empty());
    CHECK(std::memcmp(want.data(), got.data(), want.size() * sizeof(float)) == 0);
    // a NaN column (expert 3, the kBad record): refused as collect refuses it, its row rebuilt from the token's others
    H::Job d{}; d.layer = 2; d.expert = 3; d.columns = 1; d.token[0] = 1; d.weight[0] = 1.0f;
    const std::vector<H::Job> plan2{b, d};
    std::vector<float> want2(size_t(3) * H::kHidden), got2(size_t(3) * H::kHidden);
    std::vector<H::Refused> refused_collect;
    const uint64_t seq2 = h.submit(3, x.data(), plan2, err);
    CHECK(seq2 != 0 && h.collect(seq2, want2.data(), want2.size(), err, nullptr, &refused_collect));
    std::vector<uint16_t> columns2(size_t(2) * H::kHidden, bf16(float(0x20) / 16.0f));
    std::fill_n(columns2.begin() + H::kHidden, H::kHidden, bf16(float(kBad) / 16.0f));
    columns2[size_t(H::kHidden) + 5] = 0x7fc0;
    CHECK(h.merge_external(3, plan2, columns2.data(), got2.data(), got2.size(), err, &refused));
    CHECK(refused.size() == 1 && refused_collect.size() == 1);
    if (refused.size() == 1 && refused_collect.size() == 1) {
        CHECK(refused[0].expert == 3 && refused[0].token == 1 && !refused[0].sum && refused[0].bad == 1 && refused[0].slot == refused_collect[0].slot);
        CHECK(refused[0].detail.rfind("nonfinite helper output (device 0, hand-off pass: layer 2 expert 3 ", 0) == 0);
        std::printf("refused: %s\n", refused[0].detail.c_str());
    }
    CHECK(std::memcmp(want2.data(), got2.data(), want2.size() * sizeof(float)) == 0);
    // without a list: fails with collect's first words (the caller poisons, as collect does)
    std::fill_n(columns2.begin() + H::kHidden, H::kHidden, bf16(float(kBad) / 16.0f));
    columns2[size_t(H::kHidden) + 5] = 0x7fc0;
    CHECK(!h.merge_external(3, plan2, columns2.data(), got2.data(), got2.size(), err));
    CHECK(err.rfind("nonfinite helper output (device 0, hand-off pass:", 0) == 0);
    // a column naming a row past the pass is refused as invalid
    H::Job bad_row = b; bad_row.token[0] = 3;
    CHECK(!h.merge_external(3, {bad_row}, columns2.data(), got2.data(), got2.size(), err));
    // records_version and handoff_busy: a rewrite waits for the released pass, then moves the version twice
    auto r1 = record(0x20);
    bool unsafe = false;
    h.handoff_busy(true);
    CHECK(!h.rewrite_expert(2, 1, r1.data(), r1.size(), sw, si, unsafe, err));
    CHECK(!h.replace_expert(2, 1, 5, 7, r1.data(), r1.size(), sw, si, unsafe, err));
    h.handoff_busy(false);
    const uint64_t v1 = h.records_version();
    CHECK(h.rewrite_expert(2, 1, r1.data(), r1.size(), sw, si, unsafe, err));
    CHECK(h.records_version() == v1 + 2);
    CHECK(h.replace_expert(2, 1, 5, 7, r1.data(), r1.size(), sw, si, unsafe, err));
    CHECK(h.records_version() == v1 + 4);
    CHECK(h.ready(5)[0] == (1u << 7) && !(h.ready(2)[0] & 2u));
}

// kernel item 5 on the fake API: the NVFP4-activation kernel's values accepted by set_kernel; packed rows from the 5090
// refused unless both kernels are that kernel, and a row carrying the 5090's non-finite mark (scale 0x7f) refused as a
// non-finite BF16 row is (the kernels themselves run only on a card: tools/mm3/bench/helper_q_bench.cu)
static void test_q_inputs() {
    std::string err;
    H h(0, 4);
    const float sw[3] = {1, 2, 3}, si[3] = {4, 5, 6};
    auto r = record(0x10);
    CHECK(h.add_expert(2, 0, r.data(), r.size(), sw, si, err));
    CHECK(h.seal(err));
    std::vector<uint8_t> packed(size_t(2) * mm3::helper_q::kPackedRow, 0x11);
    H::Job j{}; j.layer = 2; j.expert = 0; j.columns = 1; j.token[0] = 1; j.weight[0] = 1.0f;
    CHECK(h.submit_packed(2, packed.data(), {j}, err) == 0 && err.find("NVFP4-activation kernel") != std::string::npos);
    h.set_kernel(172, 72);
    CHECK(h.kernel_multi() == 172 && h.kernel_one() == 72);
    h.set_kernel(165, 66);
    CHECK(h.kernel_multi() == 165 && h.kernel_one() == 66);
    packed[size_t(mm3::helper_q::kPackedRow) + 3072 + 5] = 0x7f;   // (token 1's block 5: the non-finite mark)
    CHECK(h.submit_packed(2, packed.data(), {j}, err) == 0 && err == "nonfinite helper activation");
    CHECK(!h.sm89_capable() && !h.sm89());   // (the fake card is no sm_89 card; the CPU build has no sm_89 kernels)
}

int main() {
    static_assert(sizeof(ho::Job) == 96, "the 5090 and the cards share the job layout");
    static_assert(offsetof(ho::Request, x) % 16 == 0 && offsetof(ho::Result, y) % 16 == 0, "rows and columns move in 16-byte pieces");
    static_assert(sizeof(ho::Request::x) == size_t(ho::kMaxRows) * ho::kHidden * 2 && sizeof(ho::Result::y) == size_t(ho::kMaxJobs) * ho::kHidden * 2, "slot sizes");
    test_plan();
    test_merge();
    test_q_inputs();
    if (failures) { std::fprintf(stderr, "mm3_helper_handoff_test: %d failure(s)\n", failures); return 1; }
    std::puts("mm3_helper_handoff_test: OK");
    return 0;
}
