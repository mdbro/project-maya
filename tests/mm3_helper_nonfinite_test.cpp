// A helper card's non-finite result column (the user's decision of 2026-11-14: "It should recompute that column on another
// lane and carry on, make sure it logs if it happens"), on the CPU with the production helper compiled against the fake
// CUDA API (tests/mm3_helper_fake_source.cmake), the fake card's expert defined below:
//   - collect with a refused list: the bad column is left out and listed with its detail (where it went bad), the other
//     rows untouched, the helper not poisoned and usable for the next pass;
//   - collect without one: poisoned as before, the message keeping its first words with the detail after them;
//   - poison(), and read_record: a slot read back while a live replacement is in flight (export_expert refuses then);
//   - mm3_helper_recover.hpp: the record sources' order, the guarded add of the CPU lane's column, the log line.
// The model's glue (mm3_native_helper_recover.inc: the record read, the CPU lane's run) needs the model and is not here.
#include "../src/mm3/mm3_native_helper.hpp"
#include "../src/mm3/mm3_helper_recover.hpp"
#include "cuda_runtime_api.h"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
int failures = 0;
void check(bool ok, const char* what, int line) {
    if (!ok) { std::fprintf(stderr, "mm3_helper_nonfinite_test:%d: FAIL: %s\n", line, what); ++failures; }
}
#define CHECK(x) check((x), #x, __LINE__)
constexpr unsigned char kBad = 0x30;   // the record whose column carries one bad element
uint16_t g_bad_value = 0x7fc0;         // (NaN; or a huge finite value)
int g_bad_index = 5;
uint16_t bf16(float f) { uint32_t b; std::memcpy(&b, &f, 4); return uint16_t(b >> 16); }
}

namespace infernix::ops::minimax_ffn {
size_t record_bytes(int, int) { return 144; }
size_t record_stride(int, int) { return 144; }
size_t fast_workspace_bytes(int, int, int) { return 256; }
// The fake card's expert: every element of a column is (the record's first code byte) / 16, and the kBad record's
// column holds g_bad_value at g_bad_index. Device memory is host memory here, so y is written directly.
void fast_ffn_cuda_packed(const FastJob* jobs, int n, int h, int, int, void*, cudaStream_t) {
    for (int j = 0; j < n; ++j)
        for (int c = 0; c < jobs[j].columns; ++c) {
            uint16_t* y = jobs[j].y + size_t(c) * size_t(h);
            const unsigned char code = jobs[j].record[0];
            std::fill_n(y, size_t(h), bf16(float(code) / 16.0f));
            if (code == kBad) y[g_bad_index] = g_bad_value;
        }
}
}

using H = mm3::NativeHelper;
static std::vector<unsigned char> record(unsigned char code) {
    std::vector<unsigned char> x(144, code);
    std::fill(x.begin() + 128, x.end(), 0x38);   // (valid E4M3 block scales)
    return x;
}
static H* setup(int capacity = 4) {
    auto* h = new H(0, capacity);
    const float w[3] = {1, 2, 3}, in[3] = {4, 5, 6};
    std::string err;
    auto a = record(0x10), b = record(0x20), c = record(kBad);
    CHECK(h->add_expert(1, 3, a.data(), a.size(), w, in, err));   // columns of 1.0
    CHECK(h->add_expert(1, 4, b.data(), b.size(), w, in, err));   // columns of 2.0
    CHECK(h->add_expert(1, 5, c.data(), c.size(), w, in, err));   // columns of 3.0, one element bad
    CHECK(h->seal(err));
    return h;
}
// two tokens: expert 3 on both (weights 1, 0.5), expert 5 on token 0 (weight 1), expert 4 on token 1 (weight 2)
static std::vector<H::Job> pass_jobs() {
    H::Job a{}; a.layer = 1; a.expert = 3; a.columns = 2; a.token[0] = 0; a.token[1] = 1; a.weight[0] = 1.0f; a.weight[1] = 0.5f;
    H::Job b{}; b.layer = 1; b.expert = 5; b.columns = 1; b.token[0] = 0; b.weight[0] = 1.0f;
    H::Job c{}; c.layer = 1; c.expert = 4; c.columns = 1; c.token[0] = 1; c.weight[0] = 2.0f;
    return {a, b, c};
}
static bool row_is(const std::vector<float>& out, int t, float v) {
    for (int k = 0; k < H::kHidden; ++k) if (out[size_t(t) * H::kHidden + size_t(k)] != v) return false;
    return true;
}

int main() {
    std::string err;
    std::vector<uint16_t> x(size_t(2) * H::kHidden, bf16(0.25f));
    std::vector<float> out(size_t(2) * H::kHidden, 9.0f);
    {   // with a refused list: the NaN column left out and listed; the helper usable afterwards
        H* h = setup();
        std::vector<H::Refused> refused;
        uint64_t seq = h->submit(2, x.data(), pass_jobs(), err);
        CHECK(seq != 0);
        CHECK(h->collect(seq, out.data(), out.size(), err, nullptr, &refused));
        CHECK(refused.size() == 1);
        if (!refused.empty()) {
            const H::Refused& r = refused[0];
            CHECK(r.layer == 1 && r.expert == 5 && r.slot >= 0 && r.token == 0 && r.weight == 1.0f && !r.sum && r.bad == 1);
            CHECK(r.detail.rfind("nonfinite helper output (device 0 layer 1 expert 5 ", 0) == 0);
            CHECK(r.detail.find("h_y nonfinite 1 of 6144, first [5] = 0x7fc0") != std::string::npos);
            CHECK(r.detail.find("d_y nonfinite 1, d_y[5] = 0x7fc0, d_y = h_y, d_x = h_x") != std::string::npos);
            std::printf("refused: %s\n", r.detail.c_str());
        }
        CHECK(row_is(out, 0, 1.0f));          // token 0: expert 3's column alone (expert 5's left out)
        CHECK(row_is(out, 1, 0.5f + 4.0f));   // token 1: untouched by the refusal
        CHECK(h->ready(1)[0] == ((1u << 3) | (1u << 4) | (1u << 5)));   // (not poisoned: every record still ready)
        H::Job one{}; one.layer = 1; one.expert = 4; one.columns = 1; one.token[0] = 0; one.weight[0] = 1.0f;
        seq = h->submit(1, x.data(), {one}, err);
        CHECK(seq != 0);
        CHECK(h->collect(seq, out.data(), out.size(), err, nullptr, &refused) && refused.empty() && row_is(out, 0, 2.0f));
        // a finite but huge value whose weighted sum overflows: refused with the sum flag and no bad element
        g_bad_value = bf16(3.0e38f);
        H::Job big{}; big.layer = 1; big.expert = 5; big.columns = 1; big.token[0] = 0; big.weight[0] = 4.0f;
        seq = h->submit(1, x.data(), {one, big}, err);
        CHECK(seq != 0);
        CHECK(h->collect(seq, out.data(), out.size(), err, nullptr, &refused));
        CHECK(refused.size() == 1 && refused[0].sum && refused[0].bad == 0 && refused[0].weight == 4.0f);
        CHECK(!refused.empty() && refused[0].detail.rfind("nonfinite helper weighted sum (device 0 ", 0) == 0);
        CHECK(row_is(out, 0, 2.0f));
        g_bad_value = 0x7fc0;
        delete h;
    }
    {   // without a list: poisoned as before, the message's first words kept
        H* h = setup();
        const uint64_t seq = h->submit(2, x.data(), pass_jobs(), err);
        CHECK(seq != 0);
        CHECK(!h->collect(seq, out.data(), out.size(), err));
        CHECK(err.rfind("nonfinite helper output (device 0 layer 1 expert 5 ", 0) == 0);
        CHECK(h->ready(1)[0] == 0 && h->records().empty());
        CHECK(h->submit(2, x.data(), pass_jobs(), err) == 0 && err.find("poisoned 1") != std::string::npos);
        delete h;
    }
    {   // poison(): the caller's recompute failed too
        H* h = setup();
        h->poison("nonfinite helper output (device 0 ...); recomputing it on the CPU lane failed: test");
        CHECK(h->submit(2, x.data(), pass_jobs(), err) == 0 && err.find("poisoned 1") != std::string::npos);
        CHECK(h->device() == 0);
        delete h;
    }
    {   // read_record: the slot read back, also while a live replacement is in flight (export_expert refuses then)
        H* h = setup(4);
        std::vector<unsigned char> staging(144, 0);
        float w[3] = {}, in[3] = {};
        bool unsafe = false;
        CHECK(h->read_record(1, 4, staging.data(), staging.size(), w, in, unsafe, err) && !unsafe);
        CHECK(staging == record(0x20) && w[0] == 1 && w[2] == 3 && in[1] == 5);
        CHECK(!h->read_record(1, 7, staging.data(), staging.size(), w, in, unsafe, err));   // (no such record)
        // a lease: expert (1,3) out, (2,9) in
        auto host = record(0x40);
        std::vector<unsigned char> incoming(144), outgoing(144);
        float ow[3] = {}, oi[3] = {};
        const float nw[3] = {1, 1, 1}, ni[3] = {1, 1, 1};
        CHECK(h->live_begin(1, 3, 2, 9, host.data(), incoming.data(), outgoing.data(), 144, nw, ni, ow, oi, err));
        std::fill(staging.begin(), staging.end(), 0);
        CHECK(!h->export_expert(1, 4, staging.data(), staging.size(), w, in, unsafe, err));   // (export_expert: an idle lease only)
        CHECK(h->read_record(1, 4, staging.data(), staging.size(), w, in, unsafe, err) && !unsafe);
        CHECK(staging == record(0x20));
        CHECK(h->read_record(1, 5, staging.data(), staging.size(), w, in, unsafe, err) && staging == record(kBad));
        // with a pass in flight: refused
        const uint64_t seq = h->submit(2, x.data(), {pass_jobs()[2]}, err);
        CHECK(seq != 0);
        CHECK(!h->read_record(1, 4, staging.data(), staging.size(), w, in, unsafe, err) && err.find("active") != std::string::npos);
        std::vector<H::Refused> refused;
        CHECK(h->collect(seq, out.data(), out.size(), err, nullptr, &refused) && refused.empty());
        bool safe = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!(h->live_cancel(safe, err) && safe) && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        CHECK(safe);
        delete h;
    }
    {   // mm3_helper_recover.hpp
        namespace hr = mm3::helper_recover;
        using S = hr::Source;
        CHECK((hr::sources(false, true, false) == std::vector<S>{S::pack, S::card}));                 // 3090: pack, its slot
        CHECK((hr::sources(false, false, false) == std::vector<S>{S::card}));
        CHECK((hr::sources(false, true, true) == std::vector<S>{S::pack, S::card}));                  // (a 3090 record is never the bank's)
        CHECK((hr::sources(true, true, true) == std::vector<S>{S::pack, S::host_bank, S::card}));     // 4080: pack, the bank's
        CHECK((hr::sources(true, false, true) == std::vector<S>{S::host_bank, S::card}));
        std::vector<float> row(8, 1.0f), col(8, 0.5f);
        CHECK(hr::add_column(row.data(), col.data(), 8) == hr::Add::ok && row[3] == 1.5f);
        col[2] = NAN;
        CHECK(hr::add_column(row.data(), col.data(), 8) == hr::Add::nonfinite_column && row[2] == 1.5f && row[3] == 1.5f);   // (not added)
        col[2] = 3.0e38f; row[2] = 3.0e38f;
        CHECK(hr::add_column(row.data(), col.data(), 8) == hr::Add::nonfinite_sum);
        hr::Event e;
        e.card = "3090 helper"; e.device = 1; e.device_name = "NVIDIA GeForce RTX 3090"; e.where = "decode";
        e.layer = 37; e.expert = 85; e.slot = 12; e.token = 2; e.rows = 5; e.position = 1407; e.weight = 0.25f; e.bad = 1;
        e.detail = " (device 1 layer 37 ...)"; e.source = S::pack; e.ms = 41.3; e.count = 1;
        const std::string line = hr::event_line(e);
        std::printf("event line: %s\n", line.c_str());
        for (const char* part : {"#1", "3090 helper CUDA1 (NVIDIA GeForce RTX 3090)", "decode", "layer 37 expert 85 slot 12",
                                 "row 2 of 5", "position 1407", "1 of 6144 elements non-finite", "41.3 ms",
                                 "the native pack's record", "detail:  (device 1 layer 37"})
            CHECK(line.find(part) != std::string::npos);
        e.source = S::card; e.fallback = "pack failed: invalid E4M3 block scale"; e.sum = true;
        const std::string fb = hr::event_line(e);
        CHECK(fb.find("the card's slot exported") != std::string::npos && fb.find("(pack failed: invalid E4M3") != std::string::npos);
        CHECK(fb.find("weighted sum overflowed") != std::string::npos);
    }
    if (failures) { std::fprintf(stderr, "mm3_helper_nonfinite_test: %d failure(s)\n", failures); return 1; }
    std::puts("mm3_helper_nonfinite_test: OK");
    return 0;
}
