// tests/mm3_topk_split_test.cu - topk_rows split over many blocks a row (slices, then a merge) against the block-a-row form:
// the same CandRows bit for bit (n, ids, logits, the extra ids' logits), for 16 and 512 candidates over 1-16 rows. Rows:
// normal and Zipf-shaped logits, logits rounded to a few levels (ties across slices, and slices with more ties than
// kCandCap, which go back to the whole-row pass), constant rows (more ties than kCandCap: n -1), masked (-inf) runs, a NaN,
// and vocabularies smaller than the candidates. MM3_TOPK_BENCH=1 also times both forms at 1, 7 and 8 rows.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "mm3_kernels.cuh"

namespace k = mm3::k;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static bool same_row(const k::CandRow& a, const k::CandRow& b, int n_extra) {
    if (a.n != b.n) return false;
    if (a.n > 0 && (std::memcmp(a.id, b.id, sizeof(int) * a.n) != 0 || std::memcmp(a.v, b.v, sizeof(float) * a.n) != 0)) return false;
    return std::memcmp(a.ex, b.ex, sizeof(float) * n_extra) == 0;
}

int main() {
    constexpr int VMAX = 200064, RMAX = 16;
    std::mt19937_64 rng(20261109);
    float* d_lg = nullptr;
    int* d_extra = nullptr;
    void* scratch = nullptr;
    k::CandRow *a = nullptr, *b = nullptr;
    CK(cudaMalloc(&d_lg, (size_t) RMAX * VMAX * sizeof(float)));
    CK(cudaMalloc(&d_extra, k::kCandExtra * sizeof(int)));
    CK(cudaMalloc(&scratch, k::topk_rows_scratch(RMAX)));
    CK(cudaHostAlloc((void**) &a, RMAX * sizeof(k::CandRow), cudaHostAllocMapped | cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &b, RMAX * sizeof(k::CandRow), cudaHostAllocMapped | cudaHostAllocPortable));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::vector<float> h((size_t) RMAX * VMAX);
    int checked = 0, failed = 0, overflow = 0, ties_back = 0;
    const int Vs[] = {200064, 200064, 200064, 4099, 600, 100, 1};
    const int Rs[] = {1, 2, 7, 8, 16};
    for (int V : Vs)
        for (int R : Rs)
            for (int kc : {16, 512})
                for (int shape = 0; shape < 6; ++shape) {
                    std::normal_distribution<float> nd(0.0f, 1.0f + shape);
                    std::uniform_int_distribution<int> ui(0, V - 1);
                    for (int r = 0; r < R; ++r) {
                        float* x = h.data() + (size_t) r * V;
                        switch (shape) {
                            case 0: for (int i = 0; i < V; ++i) x[i] = nd(rng); break;
                            case 1: {   // Zipf over a shuffled vocabulary
                                std::vector<int> perm(V);
                                for (int i = 0; i < V; ++i) perm[i] = i;
                                std::shuffle(perm.begin(), perm.end(), rng);
                                for (int i = 0; i < V; ++i) x[perm[i]] = -1.3f * std::log(1.0f + (float) i);
                                break;
                            }
                            case 2: for (int i = 0; i < V; ++i) x[i] = std::round(nd(rng) * 2.0f) / 2.0f; break;   // few levels
                            case 3: for (int i = 0; i < V; ++i) x[i] = (r % 2) ? 1.5f : std::round(nd(rng));      // constant rows
                                    break;
                            case 4: for (int i = 0; i < V; ++i) x[i] = (i / 997) % 3 == 0 ? -INFINITY : nd(rng); break;
                            default: for (int i = 0; i < V; ++i) x[i] = nd(rng);
                                     x[ui(rng)] = NAN;
                                     x[ui(rng)] = -0.0f;
                        }
                    }
                    const int n_extra = (shape % 2) ? std::min(5, V) : 0;
                    std::vector<int> extra(k::kCandExtra);
                    for (int j = 0; j < n_extra; ++j) extra[j] = ui(rng);
                    CK(cudaMemcpy(d_extra, extra.data(), k::kCandExtra * sizeof(int), cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(d_lg, h.data(), (size_t) R * V * sizeof(float), cudaMemcpyHostToDevice));
                    std::memset(a, 0x5a, R * sizeof(k::CandRow));
                    std::memset(b, 0xa5, R * sizeof(k::CandRow));
                    k::topk_rows(d_lg, R, V, kc, d_extra, n_extra, a, s);
                    k::topk_rows(d_lg, R, V, kc, d_extra, n_extra, b, scratch, s);
                    CK(cudaStreamSynchronize(s));
                    for (int r = 0; r < R; ++r) {
                        ++checked;
                        if (a[r].n < 0) ++overflow;
                        if (shape == 2 || shape == 3) ++ties_back;
                        if (!same_row(a[r], b[r], n_extra)) {
                            ++failed;
                            if (failed <= 10)
                                std::printf("MISMATCH V %d rows %d kc %d shape %d row %d: n %d against %d\n", V, R, kc, shape, r, a[r].n, b[r].n);
                        }
                    }
                }
    std::printf("topk split: %d rows checked, %d differ (%d with more ties than kCandCap, %d from tied shapes)\n", checked, failed,
                overflow, ties_back);
    if (const char* bench = std::getenv("MM3_TOPK_BENCH"); bench && bench[0] == '1') {
        std::normal_distribution<float> nd(0.0f, 2.0f);
        for (size_t i = 0; i < (size_t) 8 * VMAX; ++i) h[i] = nd(rng);
        CK(cudaMemcpy(d_lg, h.data(), (size_t) 8 * VMAX * sizeof(float), cudaMemcpyHostToDevice));
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        for (auto [rows, kc] : {std::pair{1, 512}, std::pair{7, 16}, std::pair{8, 512}}) {
            float t[2] = {0, 0};
            for (int form = 0; form < 2; ++form)
                for (int rep = 0; rep < 11; ++rep) {
                    float ms = 0;
                    CK(cudaEventRecord(e0, s));
                    if (form == 0) k::topk_rows(d_lg, rows, VMAX, kc, nullptr, 0, a, s);
                    else k::topk_rows(d_lg, rows, VMAX, kc, nullptr, 0, b, scratch, s);
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    CK(cudaEventElapsedTime(&ms, e0, e1));
                    if (rep) t[form] += ms / 10;
                }
            std::printf("  %d row%s, %d candidates: block a row %.3f ms, split %.3f ms\n", rows, rows > 1 ? "s" : "", kc, t[0], t[1]);
        }
    }
    cudaFree(d_lg);
    cudaFree(d_extra);
    cudaFree(scratch);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaStreamDestroy(s);
    return failed == 0 ? 0 : 1;
}
