// tests/mm3_nucleus_test.cu - step 6's GPU nucleus: topk_rows (kCandCap candidates) and row_stats on the GPU, then
// mm3::sample::nucleus_from over them, against mm3::sample::dist() over the whole row - the host path it replaces - at the
// model's sampling (temperature 1, top_p 0.95, no top_k, min_p or penalties): the same distribution, bit for bit,
// wherever nucleus_from settles; the other rows fall back to the whole row. Rows from flat to peaked, with ties, at the
// vocabulary's size. row_stats' maximum must equal the host's and its total agree to rounding.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "mm3_kernels.cuh"
#include "mm3_sample.hpp"

namespace k = mm3::k;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main() {
    constexpr int V = 200064, R = 8, ROUNDS = 16;
    std::mt19937_64 rng(20261109);
    mm3::sample::Req req;
    req.temperature = 1.0f;
    req.top_p = 0.95f;
    req.top_k = 0;
    req.min_p = 0.0f;
    float* d_lg = nullptr;
    k::CandRow* c = nullptr;
    k::RowStats* st = nullptr;
    CK(cudaMalloc(&d_lg, (size_t) R * V * sizeof(float)));
    CK(cudaHostAlloc((void**) &c, R * sizeof(k::CandRow), cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &st, R * sizeof(k::RowStats), cudaHostAllocMapped | cudaHostAllocPortable));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    const float scales[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 10.0f};   // (the logits' spread: flat to peaked)
    std::vector<float> h((size_t) R * V);
    int rows = 0, settled = 0, fell_back = 0, mismatched = 0;
    double worst_total = 0.0;
    for (int round = 0; round < ROUNDS; ++round) {
        for (int r = 0; r < R; ++r) {
            const float sc = scales[(r + round) % 8];
            std::normal_distribution<float> nd(0.0f, sc);
            std::uniform_int_distribution<int> ui(0, V - 1);
            float* x = h.data() + (size_t) r * V;
            if ((r + round) % 2 == 0) {
                // Zipf-shaped rows, as a language model's are: logit -a log(rank) over a shuffled vocabulary, a from 0.9 to
                // 2.3 - nuclei from a few tokens to past the candidates (the 256 and 512 counts' boundaries)
                const float a = 0.9f + 0.2f * (float) ((r + 3 * round) % 8);
                std::vector<int> perm(V);
                for (int i = 0; i < V; ++i) perm[(size_t) i] = i;
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int i = 0; i < V; ++i) x[perm[(size_t) i]] = -a * std::log(1.0f + (float) i) + 0.01f * nd(rng) / sc;
            } else {
                for (int i = 0; i < V; ++i) x[i] = nd(rng);
                for (int b = 0; b < (round + r) % 5; ++b) x[ui(rng)] = 4.0f * sc + (float) b;   // (a few strong tokens)
            }
            if (round % 3 == 0)
                for (int t = 0; t < 4; ++t) x[ui(rng)] = x[ui(rng)];   // (ties)
        }
        CK(cudaMemcpy(d_lg, h.data(), (size_t) R * V * sizeof(float), cudaMemcpyHostToDevice));
        k::topk_rows(d_lg, R, V, k::kCandCap, nullptr, 0, c, s);
        k::row_stats(d_lg, R, V, st, s);
        CK(cudaStreamSynchronize(s));
        for (int r = 0; r < R; ++r) {
            ++rows;
            const float* x = h.data() + (size_t) r * V;
            const auto ref = mm3::sample::dist(x, V, req, {});
            float mx = -INFINITY;
            for (int i = 0; i < V; ++i) mx = std::max(mx, x[i]);
            double tot = 0.0;
            for (int i = 0; i < V; ++i) tot += std::exp((double) (x[i] - mx));
            if (st[r].max != mx || !st[r].finite) {
                std::printf("row %d: row_stats maximum %.9g (finite %d) against the host's %.9g\n", rows, st[r].max, st[r].finite, mx);
                return 1;
            }
            worst_total = std::max(worst_total, std::abs(st[r].total - tot) / tot);
            if (c[r].n >= 1 && c[r].n <= k::kCandCap) {
                std::vector<std::pair<int, float>> top((size_t) c[r].n);
                for (int i = 0; i < c[r].n; ++i) top[(size_t) i] = {c[r].id[i], c[r].v[i]};
                std::vector<std::pair<int, double>> out;
                if (mm3::sample::nucleus_from(top, st[r].max, st[r].total, req.top_p, (size_t) V, out)) {
                    ++settled;
                    bool same = out.size() == ref.size();
                    for (size_t i = 0; same && i < out.size(); ++i)
                        same = out[i].first == ref[i].first && out[i].second == ref[i].second;
                    if (!same) {
                        ++mismatched;
                        std::printf("row %d (spread %.1f): nucleus of %zu against dist()'s %zu\n", rows,
                                    scales[(r + round) % 8], out.size(), ref.size());
                    }
                    continue;
                }
            }
            ++fell_back;
        }
    }
    std::printf("GPU nucleus: %d rows: %d settled from candidates (%d differ from dist()), %d fall back to the whole row; "
                "row_stats' total within %.2e of the host's\n", rows, settled, mismatched, fell_back, worst_total);
    cudaFree(d_lg);
    cudaFreeHost(c);
    cudaFreeHost(st);
    cudaStreamDestroy(s);
    return mismatched == 0 && settled > 0 && worst_total < 1e-12 ? 0 : 1;
}
