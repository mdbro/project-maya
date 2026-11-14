// The helper's result merge (src/mm3/mm3_helper_merge.hpp) against the scalar loop it replaced in NativeHelper::collect:
// bit-identical outputs on random columns (normal, subnormal, large and zero values, negative weights), the same flags for
// non-finite inputs and sums, and with MM3_MERGE_BENCH=1 the time of an 8-row window's merge (16 columns of 3072).
#include "mm3_helper_merge.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr size_t kHidden = 3072;

float from_bf16(uint16_t v) {
    uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// The replaced loop, verbatim but for its error returns: 0 ok, 1 a non-finite input, 2 a non-finite sum (first bad element).
int scalar(float* dst, const uint16_t* src, float w, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        if ((src[k] & 0x7f80u) == 0x7f80u) return 1;
        const float sum = dst[k] + w * from_bf16(src[k]);
        if (!std::isfinite(sum)) return 2;
        dst[k] = sum;
    }
    return 0;
}

uint16_t to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    return uint16_t(bits >> 16);
}

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}
}  // namespace

int main() {
    std::mt19937 rng(20261109);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> kind(0, 9);
    for (int trial = 0; trial < 400; ++trial) {
        std::vector<float> a(kHidden), b;
        std::vector<uint16_t> src(kHidden);
        for (size_t k = 0; k < kHidden; ++k) {
            a[k] = normal(rng) * (trial % 3 == 0 ? 1e-3f : 1.0f);
            float v = normal(rng);
            switch (kind(rng)) {
                case 0: v *= 1e-39f; break;   // subnormal in float, after bf16 truncation
                case 1: v *= 1e30f; break;
                case 2: v = 0.0f; break;
                case 3: v = -0.0f; break;
                default: break;
            }
            src[k] = to_bf16(v);
        }
        b = a;
        const float w = (trial % 2 ? -1.0f : 1.0f) * std::abs(normal(rng));
        const int s = scalar(a.data(), src.data(), w, kHidden);
        const unsigned f = mm3::helper_merge::merge_column(b.data(), src.data(), w, kHidden);
        if (s == 0) {
            check(f == 0, "finite column flagged");
            check(std::memcmp(a.data(), b.data(), kHidden * sizeof(float)) == 0, "merged column differs from the scalar loop");
        } else {
            check(f != 0, "non-finite column not flagged");
        }
    }
    {   // a non-finite input, and an overflowing sum, each flagged
        std::vector<float> dst(kHidden, 1.0f);
        std::vector<uint16_t> src(kHidden, to_bf16(0.5f));
        src[1234] = 0x7f80u;   // +inf
        check(mm3::helper_merge::merge_column(dst.data(), src.data(), 1.0f, kHidden) & mm3::helper_merge::kNonfiniteInput,
              "infinite input not flagged");
        std::vector<float> big(kHidden, 3.0e38f);
        std::vector<uint16_t> half(kHidden, to_bf16(2.0e38f));
        const unsigned f = mm3::helper_merge::merge_column(big.data(), half.data(), 1.0f, kHidden);
        check(f == mm3::helper_merge::kNonfiniteSum, "overflowing sum not flagged alone");
    }
    {   // merge_pass (2026-11-14): a pass of columns over tokens, with and without a refused list
        namespace hm = mm3::helper_merge;
        constexpr int T = 3;
        constexpr size_t C = 7;
        const int tokens[C] = {0, 1, 2, 0, 1, 0, 2};
        std::vector<std::vector<uint16_t>> cols(C, std::vector<uint16_t>(kHidden));
        std::vector<float> weights(C);
        for (size_t c = 0; c < C; ++c) {
            weights[c] = 0.25f * float(c + 1);
            for (size_t k = 0; k < kHidden; ++k) cols[c][k] = to_bf16(normal(rng));
        }
        const auto pass = [&](std::vector<hm::PassColumn>& v) {
            v.clear();
            for (size_t c = 0; c < C; ++c) v.push_back({tokens[c], weights[c], cols[c].data()});
        };
        // the reference: column by column in pass order, as collect merged before merge_pass, skipping `left_out`
        const auto reference = [&](std::vector<float>& out, const std::vector<size_t>& left_out) {
            out.assign(size_t(T) * kHidden, 0.0f);
            for (size_t c = 0; c < C; ++c) {
                if (std::find(left_out.begin(), left_out.end(), c) != left_out.end()) continue;
                hm::merge_column(out.data() + size_t(tokens[c]) * kHidden, cols[c].data(), weights[c], kHidden);
            }
        };
        std::vector<hm::PassColumn> v;
        std::vector<hm::Flagged> refused;
        std::vector<float> got(size_t(T) * kHidden, 7.0f), want;
        // clean: bit-identical to the column-by-column merge, with or without a refused list; nothing refused
        pass(v); reference(want, {});
        check(hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), nullptr).flags == 0, "clean pass flagged");
        check(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0, "clean pass differs from the column merge");
        std::fill(got.begin(), got.end(), 7.0f);
        check(hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), &refused).flags == 0 && refused.empty(), "clean pass refused");
        check(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0, "clean pass with a list differs");
        // column 3 (token 0) a NaN at one element, column 5 (token 0, after it) clean: without a list the first flag stops
        // the merge; with one, column 3 alone is refused (column 5's sum flags in the first pass only because the row
        // already held NaN) and token 0's row is the sum of columns 0 and 5; rows 1 and 2 untouched
        cols[3][100] = 0x7fc0u;
        pass(v);
        const hm::Flagged first = hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), nullptr);
        check(first.index == 3 && (first.flags & hm::kNonfiniteInput), "first flagged column not reported");   // (its sum flags too)
        check(hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), &refused).flags == 0, "a refused pass returned a flag");
        check(refused.size() == 1 && refused[0].index == 3 && (refused[0].flags & hm::kNonfiniteInput), "NaN column not refused alone");
        reference(want, {3});
        check(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0, "rebuilt row differs from the sum without the column");
        // two bad columns in one row and one in another: each refused, in pass order
        cols[5][7] = 0xff80u;   // -inf
        cols[2][3000] = 0x7f80u;
        pass(v);
        check(hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), &refused).flags == 0, "a refused pass returned a flag");
        check(refused.size() == 3 && refused[0].index == 2 && refused[1].index == 3 && refused[2].index == 5, "three bad columns not refused");
        reference(want, {2, 3, 5});
        check(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0, "rows without three bad columns differ");
        // a finite column whose weighted sum overflows: refused with the sum flag, its row without it
        cols[2][3000] = to_bf16(1.0f); cols[3][100] = to_bf16(1.0f); cols[5][7] = to_bf16(1.0f);
        cols[4].assign(kHidden, to_bf16(2.0e38f));
        weights[4] = 2.0f;
        pass(v);
        check(hm::merge_pass(got.data(), T, kHidden, v.data(), v.size(), &refused).flags == 0 && refused.size() == 1 &&
              refused[0].index == 4 && refused[0].flags == hm::kNonfiniteSum, "overflowing column not refused with the sum flag");
        reference(want, {4});
        check(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0, "row without the overflowing column differs");
    }
    if (const char* bench = std::getenv("MM3_MERGE_BENCH"); bench && bench[0] == '1') {
        const int columns = 16, reps = 2000;
        std::vector<float> out(columns * kHidden), out2;
        std::vector<uint16_t> src(columns * kHidden);
        for (auto& v : src) v = to_bf16(normal(rng));
        for (auto& v : out) v = normal(rng);
        out2 = out;
        auto time = [&](auto&& body) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; ++r) body();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        const double old_us = time([&] {
            for (int c = 0; c < columns; ++c) scalar(out.data() + c * kHidden, src.data() + c * kHidden, 0.25f, kHidden);
        });
        const double new_us = time([&] {
            for (int c = 0; c < columns; ++c)
                mm3::helper_merge::merge_column(out2.data() + c * kHidden, src.data() + c * kHidden, 0.25f, kHidden);
        });
        std::printf("merge of %d columns x %zu: scalar %.2f us, branch-free %.2f us\n", columns, kHidden, old_us, new_us);
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("mm3_helper_merge_test: ok\n");
    return 0;
}
