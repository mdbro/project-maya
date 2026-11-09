// The helper's result merge (src/mm3/mm3_helper_merge.hpp) against the scalar loop it replaced in NativeHelper::collect:
// bit-identical outputs on random columns (normal, subnormal, large and zero values, negative weights), the same flags for
// non-finite inputs and sums, and with MM3_MERGE_BENCH=1 the time of an 8-row window's merge (16 columns of 3072).
#include "mm3_helper_merge.hpp"

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
