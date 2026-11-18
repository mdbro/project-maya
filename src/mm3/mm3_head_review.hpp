#pragma once

#include <array>
#include <cmath>
#include <sstream>
#include <string>

namespace mm3::head_review {

// Raw target logits before sampling transforms. The three contiguous comparison rows
// come from the same final normalized hidden vector as the BF16 generation row.
inline bool record(const float* base, const float* comparisons, int vocab, int token,
                   long long index, long long position, bool reasoning, std::string& json) {
    if (!base || !comparisons || vocab < 1 || token < 0 || token >= vocab) return false;
    const std::array<const float*,4> rows{base, comparisons, comparisons+vocab, comparisons+2*vocab};
    std::array<int,4> top{};
    for (size_t h=0; h<rows.size(); ++h) {
        for (int i=0; i<vocab; ++i) {
            if (!std::isfinite(rows[h][i])) return false;
            if (rows[h][i]>rows[h][top[h]]) top[h]=i;
        }
    }
    std::array<double,4> partition{}, difference{};
    for (int i=0; i<vocab; ++i) {
        const double weight=std::exp(double(base[i])-base[top[0]]);
        partition[0]+=weight;
        for (size_t h=1; h<rows.size(); ++h) {
            partition[h]+=std::exp(double(rows[h][i])-rows[h][top[h]]);
            difference[h]+=weight*(double(base[i])-rows[h][i]);
        }
    }
    std::ostringstream out;
    out.precision(17);
    out << "{\"type\":\"head_comparison\",\"i\":" << index << ",\"pos\":" << position
        << ",\"id\":" << token << ",\"channel\":\"" << (reasoning?"reasoning":"final")
        << "\",\"vocab\":" << vocab << ",\"base_top1\":" << top[0];
    const char* names[]={"base","bf16_repeat","fp8","dense8"};
    for (size_t h=1; h<rows.size(); ++h) {
        const double kl=difference[h]/partition[0] + (double(rows[h][top[h]])-base[top[0]])
                        + std::log(partition[h]/partition[0]);
        out << ",\"" << names[h] << "\":{\"kl\":" << kl << ",\"top1\":" << top[h] << "}";
    }
    out << '}';
    json=out.str();
    return true;
}

} // namespace mm3::head_review
