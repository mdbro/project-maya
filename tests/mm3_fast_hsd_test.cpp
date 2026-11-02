// CPU checks for the target-relative Fast-HSD cutoff and target replacement.
#include <cassert>
#include <cmath>
#include "mm3_sample.hpp"

using namespace mm3::sample;

int main() {
    Req r;
    r.temperature = 1; r.top_k = 0; r.top_p = 0.95f;
    const float logits[] = {0.0f, -0.69314718f, -1.38629436f, -5.0f};
    auto d = dist(logits, 4, r, {});
    assert(d.size() == 3); // fourth token is outside the top-p support
    assert(fast_hsd_accepts(d, 0, 1));
    assert(!fast_hsd_accepts(d, 1, 1));
    assert(fast_hsd_accepts(d, 1, 0.49));
    assert(!fast_hsd_accepts(d, 2, 0.3));
    assert(!fast_hsd_accepts(d, 3, 0.001));
    assert(!fast_hsd_accepts(d, 0, 0));
    // A rejected token in support remains drawable: replacement is p, not a
    // draw excluding the draft or a residual. Check its observed frequency.
    std::mt19937_64 rng(17);
    int n = 0;
    for (int i = 0; i < 30000; ++i) n += draw(d, rng) == 2;
    assert(std::abs(double(n) / 30000 - d[2].second) < 0.012);
}
