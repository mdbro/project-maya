// src/mm3/mm3_route_floor.hpp - the per-row budget on lost router mass in the serving routing kernel
// (mm3_resident_router.cu, Policy::Prior), with the routing lab's semantics (mm3_route_lab.hpp rule 5 with route_priced
// off; tools/mm3/routing_lab/settings_schema.py route_floor: "the most of a row's original top-four router mass that its
// substitutions may give up; applied cheapest first until the next one would go over"). The setting is the floor, 1
// minus the budget. One source for the kernel (__device__) and its CPU check (tests/mm3_route_lab_test.cpp), so the
// check compares the code the kernel runs, not a copy of it.
//
// A row's original four O (the router's own, rank order: sigmoid plus bias, higher first, ties to the smaller id) and
// the bonus's four S (the Prior bonus's selection, adjusted order) differ by pairs: the displaced originals by sigmoid
// ascending (ties smaller id) zipped with the newcomers by sigmoid descending (ties smaller id), so each later pair
// gives up at least as much mass. The pairs are applied in that order while the row keeps at least floor times its
// top-four sigmoid sum (double arithmetic, the lab's and the replay's), and the first pair that would take it under
// stops them. The executed four are O with the applied pairs' newcomers in their displaced originals' places, ordered
// by the adjusted score (higher first, ties smaller id): the order the kernel's weights sum in, as the lab's.
#pragma once

#if defined(__CUDACC__)
#define MM3_ROUTE_FLOOR_HD __host__ __device__
#else
#define MM3_ROUTE_FLOOR_HD
#endif

namespace mm3::resident {

MM3_ROUTE_FLOOR_HD inline bool floor_better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }

// O, pO, aO: the original four (rank order), their sigmoids and their adjusted scores (sigmoid + bias + the bonus where
// ready: the kernel's score arithmetic). S, pS, aS: the bonus's four (adjusted order) and theirs. floor in (0, 1].
// Out: ex, pex - the executed four and their sigmoids, adjusted order. Returns the pairs applied.
MM3_ROUTE_FLOOR_HD inline int route_floor(const int* O, const float* pO, const float* aO, const int* S, const float* pS,
                                          const float* aS, float row_floor, int* ex, float* pex) {
    int D[4], N[4];
    float pD[4], pN[4], aN[4];
    int nd = 0, nn = 0;
    for (int i = 0; i < 4; ++i) {
        bool in = false;
        for (int j = 0; j < 4; ++j) in |= O[i] == S[j];
        if (!in) { D[nd] = O[i]; pD[nd] = pO[i]; ++nd; }
    }
    for (int i = 0; i < 4; ++i) {
        bool in = false;
        for (int j = 0; j < 4; ++j) in |= S[i] == O[j];
        if (!in) { N[nn] = S[i]; pN[nn] = pS[i]; aN[nn] = aS[i]; ++nn; }
    }
    // (insertion sorts: D by sigmoid ascending, N by sigmoid descending, ties to the smaller id)
    for (int i = 1; i < nd; ++i)
        for (int j = i; j > 0 && (pD[j] < pD[j - 1] || (pD[j] == pD[j - 1] && D[j] < D[j - 1])); --j) {
            const int t = D[j]; D[j] = D[j - 1]; D[j - 1] = t;
            const float u = pD[j]; pD[j] = pD[j - 1]; pD[j - 1] = u;
        }
    for (int i = 1; i < nn; ++i)
        for (int j = i; j > 0 && (pN[j] > pN[j - 1] || (pN[j] == pN[j - 1] && N[j] < N[j - 1])); --j) {
            const int t = N[j]; N[j] = N[j - 1]; N[j - 1] = t;
            const float u = pN[j]; pN[j] = pN[j - 1]; pN[j - 1] = u;
            const float w = aN[j]; aN[j] = aN[j - 1]; aN[j - 1] = w;
        }
    const int pairs = nd < nn ? nd : nn;
    const double total = double(pO[0]) + double(pO[1]) + double(pO[2]) + double(pO[3]);
    double kept = total;
    int applied = 0;
    for (; applied < pairs; ++applied) {
        const double next = kept - double(pD[applied]) + double(pN[applied]);
        if (next < double(row_floor) * total) break;
        kept = next;
    }
    float aex[4];
    for (int i = 0; i < 4; ++i) {
        ex[i] = O[i]; pex[i] = pO[i]; aex[i] = aO[i];
        for (int j = 0; j < applied; ++j)
            if (O[i] == D[j]) { ex[i] = N[j]; pex[i] = pN[j]; aex[i] = aN[j]; }
    }
    for (int i = 1; i < 4; ++i)
        for (int j = i; j > 0 && floor_better(aex[j], ex[j], aex[j - 1], ex[j - 1]); --j) {
            const int t = ex[j]; ex[j] = ex[j - 1]; ex[j - 1] = t;
            const float u = pex[j]; pex[j] = pex[j - 1]; pex[j - 1] = u;
            const float w = aex[j]; aex[j] = aex[j - 1]; aex[j - 1] = w;
        }
    return applied;
}

}  // namespace mm3::resident
