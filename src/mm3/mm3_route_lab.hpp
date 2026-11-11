// MiniMax routing lab: the per-request routing rules as a pure host function (no CUDA, no engine state), so they can be
// tested on the CPU and validated exactly against the router kernel (mm3_resident_router.cu).
//
// Inputs per row are the router kernel's own capture of the scores ([128][3]: sigmoid p, p + bias, adjusted), never a
// host recomputation: device expf and the host's exp can differ in the last bit. With the device's p and p + bias, the
// selection (p + bias, plus the bonus on ready experts, higher first, ties to the smaller id) and the weights (p over a
// left-to-right float sum) are the kernel's IEEE arithmetic exactly, so with every rule off route_pass reproduces
// resident::route with Policy::Prior bit for bit (reference_prior below is that rule on its own).
//
// The rules, in the order the contract fixes within a row (routing-lab-20261110/CONTRACT.md):
//   1. layer drop: if the sum of the router's own top-four sigmoids is below route_layer_drop, the row's routed experts at
//      this layer contribute nothing (every pick skipped: not loaded, weight 0).
//   2. protect: the strongest N original picks (router rank, sigmoid + bias) are fixed; if not ready they are loaded.
//   3. candidates: the bonus (route_bonus_draft for passes of more than one row unless -1, else routing_prior_bonus, times
//      route_layer_scale) selects the remaining picks from all other experts. Its moved picks are pairs, as in the
//      re-routed replay's reroute_floor (tools/mm3/native_reroute_replay.py): the displaced originals by sigmoid
//      ascending (ties smaller id) zipped with the newcomers by sigmoid descending (ties smaller id).
//   4. priced (route_priced > 0): pass-level, as the replay's Sim.priced_route. A load (an executed expert not ready)
//      is removed only if every row of the pass that executes it has a candidate pair displacing it; its time saved is
//      route_price_staged if its copy is in flight, else route_price_first if it is the pass's only load, else
//      route_price_more; its cost is the sum over those rows of (p[displaced] - p[newcomer]) / the row's top-four sum.
//      Removals are taken in order of saved / cost (ties: the smaller expert id) while that is at least route_priced;
//      with route_floor on, a removal that would take any of its rows under the floor is not made.
//   5. floor (route_floor > 0, priced off): per row, the pairs are applied cheapest first (the zip order: each later
//      pair loses at least as much mass) while the row keeps at least floor of its top-four mass; it stops at the first
//      pair that would break it (the replay's rule).
//      With neither priced nor floor, every pair is applied (the bonus's own routing as a set).
//   6. k_drop (route_k_drop > 0): an original pick at rank 4 (route_k_drop_ranks 1) or ranks 3-4 (2) that is still
//      executed, not ready, not protected, and whose original weight p / top-four sum is below route_k_drop is skipped.
//   7. k_widen (route_k_widen > 0): if the row is still exactly the original four (nothing substituted or dropped),
//      exactly one of them is not ready and not protected, ranks 5 and 6 are ready, and the top six sigmoids are all at
//      least route_k_widen, the unready pick is replaced by ranks 5 and 6: five experts.
//   8. weights (route_weights): 0 today's (own p over the executed set's p sum); 1 dampen (own p over the original
//      top-four sum, no renormalisation); 2 amplify (retained originals as in 1, a substitute takes its displaced
//      original's weight; the two widened picks split the replaced weight in proportion to their p).
// Executed picks are ordered by the bonus's score (adjusted, higher first, ties smaller id), then skipped picks. With no
// rule acting this is the kernel's selection order, so the plan's float sums are today's.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace mm3::route_lab {

constexpr int kE = 128;   // experts
constexpr int kK = 4;     // the router's picks
constexpr int kMax = 5;   // picks a row can execute (k_widen)

struct Settings {
    bool on = false;            // route_lab
    float bonus = 0.0f;         // routing_prior_bonus (the router's value in prior mode)
    float bonus_draft = -1.0f;  // route_bonus_draft: passes of more than one row; -1: bonus
    float prompt_bonus = -1.0f; // route_prompt_bonus: the router's bonus while the prompt is read; -1: bonus
    bool prompt = false;        // route_lab_prompt: the chunked prefill's passes through the lab too (with
                                // prompt_bonus, else bonus; never bonus_draft)
    std::vector<float> layer_scale;   // route_layer_scale by model layer (empty or 1: unscaled)
    float floor = 0.0f;         // route_floor
    int protect = 0;            // route_protect: 0, 1 or 2
    float priced = 0.0f;        // route_priced: ms per unit of router mass; 0 off
    float price_first = 0.60f, price_more = 0.32f, price_staged = 0.10f;
    float layer_drop = 0.0f;    // route_layer_drop: on the top-four sigmoid sum (0..4); 0 off
    int weights = 0;            // route_weights: 0, 1 or 2
    float k_drop = 0.0f;        // route_k_drop: original weight threshold; 0 off
    int k_drop_ranks = 1;       // route_k_drop_ranks: 1 (rank 4) or 2 (ranks 3 and 4)
    float k_widen = 0.0f;       // route_k_widen: top-six sigmoid threshold; 0 off
    int stats = 1;              // route_stats: the token records' route objects, 0 none, 1 per row, 2 with "layers"
                                // (diagnostics only: not compared by ==, so a change keeps the reused state)
    // whether a rule beyond the bonus can act (the bonus itself is today's rule)
    bool any_rule() const {
        bool scaled = false;
        for (float x : layer_scale) scaled |= x != 1.0f;
        return bonus_draft >= 0.0f || scaled || floor > 0.0f || protect > 0 || priced > 0.0f || layer_drop > 0.0f ||
               weights != 0 || k_drop > 0.0f || k_widen > 0.0f;
    }
    bool operator==(const Settings& o) const {
        return on == o.on && bonus == o.bonus && bonus_draft == o.bonus_draft && prompt_bonus == o.prompt_bonus &&
               prompt == o.prompt &&
               layer_scale == o.layer_scale &&
               floor == o.floor && protect == o.protect && priced == o.priced && price_first == o.price_first &&
               price_more == o.price_more && price_staged == o.price_staged && layer_drop == o.layer_drop &&
               weights == o.weights && k_drop == o.k_drop && k_drop_ranks == o.k_drop_ranks && k_widen == o.k_widen;
    }
    bool operator!=(const Settings& o) const { return !(*this == o); }
};

// One bit per expert: bit (e % 32) of word (e / 32), as resident::Ready.
struct Masks {
    uint32_t ready[4]{};      // the router's ready snapshot (primary | helper): exactly what the kernel saw
    uint32_t resident[4]{};   // resident on the 5090 (the rest of ready's primary part is staging frames)
    uint32_t helper[4]{};     // ready on the 3090
    uint32_t in_flight[4]{};  // not ready, a copy to the 5090 issued and not yet visible
};
inline bool bit(const uint32_t* m, int e) { return (m[e >> 5] >> (e & 31)) & 1u; }
inline void set_bit(uint32_t* m, int e) { m[e >> 5] |= 1u << (e & 31); }

// One row's result at one layer.
struct Row {
    int slots = kK;          // slots used: 4, or 5 when widened
    int ids[kMax]{};         // executed picks first (bonus order), then skipped originals
    float w[kMax]{};         // 0 for a skipped slot
    bool skip[kMax]{};       // skipped: not loaded, no contribution
    int original[kK]{};      // the router's own four, rank order
    bool invalid = false;    // the kernel marked the row invalid (non-finite input); the lab leaves it untouched
    // diagnostics
    int executed = 0, moved = 0, loads = 0, staged = 0, in_flight = 0, dropped = 0;
    bool widened = false, layer_dropped = false;
    unsigned changed = 0;    // the rules that changed this row at this layer (kChanged bits)
    double kept = 1.0;       // executed p mass over the original four's (0 for a dropped layer)
    float orig_sum = 0.0f;   // the original four's p sum, rank order, float (the kernel's dampen sum)
};

// Row::changed: a rule changed the row's picks or weights at a layer. bonus: a pair applied; protect: without it a
// protected pick would have been displaced; floor: the floor kept a pair out; priced: the price kept a pair out (not
// the floor); layer_drop, k_drop, widen: the rule acted; weights: the weights differ (bitwise) from route_weights 0's.
enum : unsigned { kBonus = 1, kProtect = 2, kFloor = 4, kPriced = 8, kLayerDrop = 16, kKDrop = 32, kWiden = 64, kWeights = 128 };
constexpr int kRules = 8;
inline const char* rule_name(int i) {
    static const char* names[kRules] = {"bonus", "protect", "floor", "priced", "layer_drop", "k_drop", "widen", "weights"};
    return names[i];
}

struct PassStats {
    int loads = 0, in_flight = 0;   // distinct experts the pass loads after the rules (not ready; copy in flight)
    double est_ms = 0.0;            // priced load cost: first + more x (loads - 1) + staged x in_flight
    int removals = 0;               // priced removals made
};

namespace detail {
inline bool better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }
// the k best of sc by better(), skipping excluded experts, in order
inline void select(const float* sc, const bool* excluded, int k, int* out) {
    bool taken[kE] = {};
    for (int r = 0; r < k; ++r) {
        float best = -std::numeric_limits<float>::infinity();
        int bi = 0x7fffffff;
        for (int e = 0; e < kE; ++e) {
            if (taken[e] || (excluded && excluded[e])) continue;
            if (better(sc[e], e, best, bi)) { best = sc[e]; bi = e; }
        }
        out[r] = bi;
        if (bi >= 0 && bi < kE) taken[bi] = true;
    }
}
inline bool contains(const int* a, int n, int x) {
    for (int i = 0; i < n; ++i) if (a[i] == x) return true;
    return false;
}
}  // namespace detail

inline float layer_scale(const Settings& s, int layer) {
    return layer >= 0 && size_t(layer) < s.layer_scale.size() ? s.layer_scale[size_t(layer)] : 1.0f;
}
// the bonus of a pass of `rows` rows at model layer `layer`
inline float bonus_for(const Settings& s, int rows, int layer) {
    const float b = rows > 1 && s.bonus_draft >= 0.0f ? s.bonus_draft : s.bonus;
    const float scale = layer_scale(s, layer);
    return scale == 1.0f ? b : b * scale;
}

// resident::route with Policy::Prior and `bonus` (0: the router's own four), from the kernel's captured row scores
// ([128][3]) and its ready snapshot.
inline void reference_prior(const float* row, const uint32_t* ready, float bonus, int ids[kK], float w[kK]) {
    float sc[kE];
    for (int e = 0; e < kE; ++e) {
        const float base = row[size_t(e) * 3 + 1];
        sc[e] = bonus > 0.0f ? base + (bit(ready, e) ? bonus : 0.0f) : base;
    }
    detail::select(sc, nullptr, kK, ids);
    float p[kK];
    for (int i = 0; i < kK; ++i) p[i] = row[size_t(ids[i]) * 3];
    const float sum = p[0] + p[1] + p[2] + p[3];
    for (int i = 0; i < kK; ++i) w[i] = sum > 0.0f ? p[i] / sum : 0.25f;
}

// The pass's rules. scores: [rows][128][3] (the kernel's capture); layer: the model layer; invalid (may be null): rows
// the kernel marked invalid, left as they are. out: rows entries.
inline void route_pass(const Settings& s, int layer, int rows, const float* scores, const Masks& m, const bool* invalid,
                       Row* out, PassStats* ps = nullptr) {
    const float b = bonus_for(s, rows, layer);
    const int P = std::max(0, std::min(2, s.protect));
    struct Work {
        int O[6];               // router ranks 1-6
        double total;           // top-four p sum, double (the replay's mass arithmetic)
        double kept;
        int S[kK];              // the bonus's picks, adjusted order
        int nd = 0;             // pairs
        int d[kK], n[kK];
        bool applied[kK] = {};
        float adj[kE];
        const float* row;
    };
    std::vector<Work> W(static_cast<size_t>(rows));
    const auto ready = [&](int e) { return bit(m.ready, e); };
    const auto flying = [&](int e) { return !bit(m.ready, e) && bit(m.in_flight, e); };
    for (int r = 0; r < rows; ++r) {
        Row& o = out[r];
        o = Row{};
        o.invalid = invalid && invalid[r];
        if (o.invalid) continue;
        Work& k = W[size_t(r)];
        k.row = scores + size_t(r) * kE * 3;
        const auto p = [&](int e) { return k.row[size_t(e) * 3]; };
        float base[kE];
        for (int e = 0; e < kE; ++e) base[e] = k.row[size_t(e) * 3 + 1];
        detail::select(base, nullptr, 6, k.O);
        for (int i = 0; i < kK; ++i) o.original[i] = k.O[i];
        o.orig_sum = p(k.O[0]) + p(k.O[1]) + p(k.O[2]) + p(k.O[3]);
        k.total = double(p(k.O[0])) + double(p(k.O[1])) + double(p(k.O[2])) + double(p(k.O[3]));
        k.kept = k.total;
        // 1. layer drop
        if (s.layer_drop > 0.0f && o.orig_sum < s.layer_drop) {
            o.layer_dropped = true;
            o.changed |= kLayerDrop;
            continue;
        }
        // 2-3. protect, then the bonus over the rest
        for (int e = 0; e < kE; ++e) k.adj[e] = b > 0.0f ? base[e] + (ready(e) ? b : 0.0f) : base[e];
        bool excluded[kE] = {};
        for (int i = 0; i < P; ++i) excluded[k.O[i]] = true;
        int rest[kK];
        detail::select(k.adj, excluded, kK - P, rest);
        if (P > 0) {   // (would the bonus alone have displaced a protected pick?)
            int plain[kK];
            detail::select(k.adj, nullptr, kK, plain);
            for (int i = 0; i < P; ++i) if (!detail::contains(plain, kK, k.O[i])) o.changed |= kProtect;
        }
        int S[kK];
        for (int i = 0; i < P; ++i) S[i] = k.O[i];
        for (int i = P; i < kK; ++i) S[i] = rest[i - P];
        std::sort(S, S + kK, [&](int x, int y) { return detail::better(k.adj[x], x, k.adj[y], y); });
        std::copy(S, S + kK, k.S);
        int D[kK], N[kK], nd = 0, nn = 0;
        for (int i = 0; i < kK; ++i) if (!detail::contains(S, kK, k.O[i])) D[nd++] = k.O[i];
        for (int i = 0; i < kK; ++i) if (!detail::contains(k.O, kK, S[i])) N[nn++] = S[i];
        std::sort(D, D + nd, [&](int x, int y) { return p(x) < p(y) || (p(x) == p(y) && x < y); });
        std::sort(N, N + nn, [&](int x, int y) { return p(x) > p(y) || (p(x) == p(y) && x < y); });
        k.nd = std::min(nd, nn);
        for (int i = 0; i < k.nd; ++i) { k.d[i] = D[i]; k.n[i] = N[i]; }
    }
    // the executed set of row r with its applied pairs (O with d -> n), before the k rules
    const auto executes = [&](int r, int e) {
        const Work& k = W[size_t(r)];
        for (int i = 0; i < k.nd; ++i) {
            if (k.applied[i] && k.n[i] == e) return true;
            if (k.applied[i] && k.d[i] == e) return false;
        }
        return detail::contains(k.O, kK, e);
    };
    const auto live = [&](int r) { return !out[r].invalid && !out[r].layer_dropped; };
    int removals = 0;
    if (s.priced > 0.0f) {
        for (;;) {
            bool used[kE] = {};
            for (int r = 0; r < rows; ++r) if (live(r))
                for (int e = 0; e < kE; ++e) if (executes(r, e)) used[e] = true;
            int nloads = 0;
            for (int e = 0; e < kE; ++e) nloads += used[e] && !ready(e) && !flying(e);
            double best_value = 0.0;
            int best = -1;
            for (int x = 0; x < kE; ++x) {
                if (!used[x] || ready(x)) continue;
                const double saved = flying(x) ? s.price_staged : nloads == 1 ? s.price_first : s.price_more;
                double cost = 0.0;
                bool ok = true;
                for (int r = 0; r < rows && ok; ++r) {
                    if (!live(r) || !executes(r, x)) continue;
                    const Work& k = W[size_t(r)];
                    int at = -1;
                    for (int i = 0; i < k.nd; ++i) if (!k.applied[i] && k.d[i] == x) at = i;
                    if (at < 0) { ok = false; break; }
                    const double px = k.row[size_t(x) * 3], pn = k.row[size_t(k.n[at]) * 3];
                    if (s.floor > 0.0f && k.kept - px + pn < double(s.floor) * k.total) { ok = false; break; }
                    cost += k.total > 0.0 ? (px - pn) / k.total : 0.0;
                }
                if (!ok) continue;
                const double value = cost > 1e-12 ? saved / cost : std::numeric_limits<double>::infinity();
                if (value >= double(s.priced) && (best < 0 || value > best_value)) { best = x; best_value = value; }
            }
            if (best < 0) {   // (the pairs left out: by the floor, else by the price)
                for (int r = 0; r < rows; ++r) {
                    if (!live(r)) continue;
                    const Work& k = W[size_t(r)];
                    for (int i = 0; i < k.nd; ++i) {
                        if (k.applied[i]) continue;
                        const double nk = k.kept - double(k.row[size_t(k.d[i]) * 3]) + double(k.row[size_t(k.n[i]) * 3]);
                        out[r].changed |= s.floor > 0.0f && nk < double(s.floor) * k.total ? kFloor : kPriced;
                    }
                }
                break;
            }
            ++removals;
            for (int r = 0; r < rows; ++r) {
                if (!live(r) || !executes(r, best)) continue;
                Work& k = W[size_t(r)];
                for (int i = 0; i < k.nd; ++i) if (!k.applied[i] && k.d[i] == best) {
                    k.applied[i] = true;
                    k.kept = k.kept - double(k.row[size_t(best) * 3]) + double(k.row[size_t(k.n[i]) * 3]);
                    break;
                }
            }
        }
    } else {
        for (int r = 0; r < rows; ++r) {
            if (!live(r)) continue;
            Work& k = W[size_t(r)];
            for (int i = 0; i < k.nd; ++i) {
                if (s.floor > 0.0f) {
                    const double nk = k.kept - double(k.row[size_t(k.d[i]) * 3]) + double(k.row[size_t(k.n[i]) * 3]);
                    if (nk < double(s.floor) * k.total) { out[r].changed |= kFloor; break; }
                    k.kept = nk;
                }
                k.applied[i] = true;
            }
        }
    }
    // per row: the k rules, the order, the weights, the diagnostics
    for (int r = 0; r < rows; ++r) {
        Row& o = out[r];
        if (o.invalid) continue;
        const Work& k = W[size_t(r)];
        const auto p = [&](int e) { return k.row[size_t(e) * 3]; };
        if (o.layer_dropped) {
            o.slots = kK;
            for (int i = 0; i < kK; ++i) { o.ids[i] = k.O[i]; o.w[i] = 0.0f; o.skip[i] = true; }
            o.kept = 0.0;
            continue;
        }
        int ex[kMax], nex = 0;      // executed
        int sk[kK], nsk = 0;        // skipped originals
        int sub_of[kE];             // a substitute's displaced original
        std::fill(sub_of, sub_of + kE, -1);
        for (int i = 0; i < k.nd; ++i) if (k.applied[i]) sub_of[k.n[i]] = k.d[i];
        bool any_applied = false;
        for (int i = 0; i < k.nd; ++i) any_applied |= k.applied[i];
        for (int i = 0; i < kK; ++i) {
            const int e = k.O[i];
            bool replaced = false;
            for (int j = 0; j < k.nd; ++j) if (k.applied[j] && k.d[j] == e) {
                ex[nex++] = k.n[j];
                replaced = true;
            }
            if (replaced) continue;
            // 6. k_drop
            const bool in_ranks = i >= kK - std::max(1, std::min(2, s.k_drop_ranks));
            if (s.k_drop > 0.0f && in_ranks && i >= P && !ready(e) && o.orig_sum > 0.0f && p(e) / o.orig_sum < s.k_drop) {
                sk[nsk++] = e;
                ++o.dropped;
                continue;
            }
            ex[nex++] = e;
        }
        // 7. k_widen
        int widen_out = -1;
        if (s.k_widen > 0.0f && !any_applied && o.dropped == 0) {
            int unready = 0, u = -1, rank_u = -1;
            for (int i = 0; i < kK; ++i) if (!ready(k.O[i])) { ++unready; u = k.O[i]; rank_u = i; }
            bool high = true;
            for (int i = 0; i < 6; ++i) high &= p(k.O[i]) >= s.k_widen;
            if (unready == 1 && rank_u >= P && ready(k.O[4]) && ready(k.O[5]) && high) {
                nex = 0;
                for (int i = 0; i < kK; ++i) if (k.O[i] != u) ex[nex++] = k.O[i];
                ex[nex++] = k.O[4];
                ex[nex++] = k.O[5];
                widen_out = u;
                o.widened = true;
            }
        }
        std::sort(ex, ex + nex, [&](int x, int y) { return detail::better(k.adj[x], x, k.adj[y], y); });
        // 8. weights
        float w[kMax] = {};
        if (s.weights == 0) {
            float sum = 0.0f;
            for (int i = 0; i < nex; ++i) sum = i ? sum + p(ex[i]) : p(ex[i]);
            for (int i = 0; i < nex; ++i) w[i] = sum > 0.0f ? p(ex[i]) / sum : 1.0f / float(nex);
        } else {
            const float t = o.orig_sum;
            const auto own = [&](int e) { return t > 0.0f ? p(e) / t : 0.25f; };
            for (int i = 0; i < nex; ++i) {
                const int e = ex[i];
                if (s.weights == 1 || detail::contains(k.O, kK, e)) { w[i] = own(e); continue; }
                if (sub_of[e] >= 0) { w[i] = own(sub_of[e]); continue; }
                // a widened pick (ranks 5 and 6): the replaced pick's weight, split by their p
                const float p5 = p(k.O[4]), p6 = p(k.O[5]), pair = p5 + p6;
                w[i] = pair > 0.0f ? own(widen_out) * (p(e) / pair) : own(widen_out) * 0.5f;
            }
        }
        if (s.weights != 0) {   // (today's weights for the same picks, to see whether the mode changed any)
            float sum = 0.0f;
            for (int i = 0; i < nex; ++i) sum = i ? sum + p(ex[i]) : p(ex[i]);
            for (int i = 0; i < nex; ++i) {
                const float w0 = sum > 0.0f ? p(ex[i]) / sum : 1.0f / float(nex);
                if (std::memcmp(&w0, &w[i], sizeof w0) != 0) o.changed |= kWeights;
            }
        }
        if (any_applied) o.changed |= kBonus;
        if (o.dropped) o.changed |= kKDrop;
        if (o.widened) o.changed |= kWiden;
        o.slots = std::max(kK, nex + nsk);
        int at = 0;
        for (int i = 0; i < nex; ++i, ++at) { o.ids[at] = ex[i]; o.w[at] = w[i]; o.skip[at] = false; }
        for (int i = 0; i < nsk; ++i, ++at) { o.ids[at] = sk[i]; o.w[at] = 0.0f; o.skip[at] = true; }
        // diagnostics
        o.executed = nex;
        double mass = 0.0;
        for (int i = 0; i < nex; ++i) {
            const int e = ex[i];
            mass += double(p(e));
            o.moved += !detail::contains(k.O, kK, e);
            if (!ready(e)) { ++o.loads; o.in_flight += bit(m.in_flight, e); }
            else if (!bit(m.resident, e) && !bit(m.helper, e)) ++o.staged;
        }
        o.kept = k.total > 0.0 ? mass / k.total : 1.0;
    }
    if (ps) {
        *ps = PassStats{};
        bool load[kE] = {}, fly[kE] = {};
        for (int r = 0; r < rows; ++r) {
            const Row& o = out[r];
            if (o.invalid) continue;
            for (int i = 0; i < o.slots; ++i) {
                if (o.skip[i] || bit(m.ready, o.ids[i])) continue;
                (bit(m.in_flight, o.ids[i]) ? fly : load)[o.ids[i]] = true;
            }
        }
        for (int e = 0; e < kE; ++e) { ps->loads += load[e]; ps->in_flight += fly[e]; }
        ps->est_ms = (ps->loads > 0 ? s.price_first + s.price_more * (ps->loads - 1) : 0.0) + s.price_staged * ps->in_flight;
        ps->removals = removals;
    }
}

// ---- diagnostics over a forward's MoE layers, for one row (the token record's "route") and over a request

struct RowStats {
    int layers = 0, picks = 0, moved = 0, loads = 0, staged = 0, in_flight = 0, dropped = 0, widened = 0,
        layers_dropped = 0, under_08 = 0, kept_n = 0;
    double kept_sum = 0.0, min_kept = std::numeric_limits<double>::infinity(), est_ms = 0.0;
    int changed[kRules] = {};   // layers each rule changed
    // by MoE layer ordinal (route_stats 2)
    std::vector<float> layer_kept;
    std::vector<int> layer_moved, layer_k, layer_loads, layer_staged, layer_pass_loads;
    void clear(int n_moe) {
        *this = RowStats{};
        layer_kept.assign(size_t(std::max(0, n_moe)), 1.0f);
        layer_moved.assign(size_t(std::max(0, n_moe)), 0);
        layer_k.assign(size_t(std::max(0, n_moe)), kK);
        layer_loads.assign(size_t(std::max(0, n_moe)), 0);
        layer_staged.assign(size_t(std::max(0, n_moe)), 0);
        layer_pass_loads.assign(size_t(std::max(0, n_moe)), 0);
    }
    // pass_est_ms, pass_loads: the pass's (PassStats), the same for every row of a forward
    void add(const Row& r, int moe, double pass_est_ms, int pass_loads = 0) {
        ++layers;
        est_ms += pass_est_ms;
        if (moe >= 0 && size_t(moe) < layer_pass_loads.size()) layer_pass_loads[size_t(moe)] = pass_loads;
        if (r.invalid) return;
        for (int i = 0; i < kRules; ++i) changed[i] += (r.changed >> i) & 1u;
        picks += r.executed;
        moved += r.moved;
        loads += r.loads;
        staged += r.staged;
        in_flight += r.in_flight;
        dropped += r.dropped;
        widened += r.widened;
        layers_dropped += r.layer_dropped;
        if (!r.layer_dropped) {
            kept_sum += r.kept;
            ++kept_n;
            min_kept = std::min(min_kept, r.kept);
            under_08 += r.kept < 0.8;
        }
        if (moe >= 0 && size_t(moe) < layer_kept.size()) {
            layer_kept[size_t(moe)] = float(r.kept);
            layer_moved[size_t(moe)] = r.moved;
            layer_k[size_t(moe)] = r.executed;
            layer_loads[size_t(moe)] = r.loads;
            layer_staged[size_t(moe)] = r.staged;
        }
    }
};

// the request summary: every row the lab routed (draft rows later rejected included: the token records' route objects
// give the committed tokens' own), each layer pass once
struct RequestStats {
    long rows = 0, row_layers = 0, picks = 0, moved = 0, loads = 0, staged = 0, in_flight = 0, dropped = 0,
         widened = 0, layers_dropped = 0, under_08 = 0, kept_n = 0, forwards = 0, passes = 0, pass_loads = 0,
         kernel_checked = 0, kernel_mismatch = 0, identity_mismatch = 0, removals = 0, helper_second = 0,
         fifth = 0;   // widened rows' fifth picks the plan dispatched (5090, CPU lane or 3090)
    double kept_sum = 0.0, min_kept = std::numeric_limits<double>::infinity(), est_ms = 0.0;
    long changed[kRules] = {}, changed_any = 0;   // row-layers each rule changed, and any rule
    void add(const Row& r) {
        ++row_layers;
        if (r.invalid) return;
        for (int i = 0; i < kRules; ++i) changed[i] += (r.changed >> i) & 1u;
        changed_any += r.changed != 0;
        picks += r.executed; moved += r.moved; loads += r.loads; staged += r.staged; in_flight += r.in_flight;
        dropped += r.dropped; widened += r.widened; layers_dropped += r.layer_dropped;
        if (!r.layer_dropped) {
            kept_sum += r.kept;
            ++kept_n;
            min_kept = std::min(min_kept, r.kept);
            under_08 += r.kept < 0.8;
        }
    }
    void add(const PassStats& p) { ++passes; pass_loads += p.loads; est_ms += p.est_ms; removals += p.removals; }
};

namespace detail {
inline std::string num(double x) {
    if (!std::isfinite(x)) return "null";
    char b[40];
    std::snprintf(b, sizeof b, "%.9g", x);
    return b;
}
template <class T> std::string changed_json(const T* c) {
    std::string s = "{";
    for (int i = 0; i < kRules; ++i) s += std::string(i ? "," : "") + "\"" + rule_name(i) + "\":" + std::to_string(c[i]);
    return s + "}";
}
template <class T> std::string arr(const std::vector<T>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ','; s += num(double(v[i])); }
    return s + "]";
}
}  // namespace detail

// the token record's "route" object (level 2 adds "layers")
inline std::string row_json(const RowStats& r, int level) {
    using detail::num;
    std::string s = "{\"picks\":" + std::to_string(r.picks) + ",\"moved\":" + std::to_string(r.moved) +
                    ",\"mass_kept\":" + num(r.kept_n ? r.kept_sum / r.kept_n : 1.0) +
                    ",\"min_kept\":" + num(r.kept_n ? r.min_kept : 1.0) + ",\"rows_under_08\":" + std::to_string(r.under_08) +
                    ",\"loads\":" + std::to_string(r.loads) + ",\"staged\":" + std::to_string(r.staged) +
                    ",\"in_flight\":" + std::to_string(r.in_flight) + ",\"dropped\":" + std::to_string(r.dropped) +
                    ",\"widened\":" + std::to_string(r.widened) + ",\"layers_dropped\":" + std::to_string(r.layers_dropped) +
                    ",\"layer_count\":" + std::to_string(r.layers) + ",\"est_ms\":" + num(r.est_ms) +
                    ",\"changed\":" + detail::changed_json(r.changed);
    if (level >= 2)
        s += ",\"layers\":{\"kept\":" + detail::arr(r.layer_kept) + ",\"moved\":" + detail::arr(r.layer_moved) +
             ",\"k\":" + detail::arr(r.layer_k) + ",\"loads\":" + detail::arr(r.layer_loads) +
             ",\"staged\":" + detail::arr(r.layer_staged) + ",\"pass_loads\":" + detail::arr(r.layer_pass_loads) + "}";
    return s + "}";
}

inline std::string request_json(const RequestStats& q) {
    using detail::num;
    return "{\"rows\":" + std::to_string(q.rows) + ",\"forwards\":" + std::to_string(q.forwards) +
           ",\"row_layers\":" + std::to_string(q.row_layers) + ",\"picks\":" + std::to_string(q.picks) +
           ",\"moved\":" + std::to_string(q.moved) + ",\"mass_kept\":" + num(q.kept_n ? q.kept_sum / q.kept_n : 1.0) +
           ",\"min_kept\":" + num(q.kept_n ? q.min_kept : 1.0) + ",\"rows_under_08\":" + std::to_string(q.under_08) +
           ",\"loads\":" + std::to_string(q.loads) + ",\"staged\":" + std::to_string(q.staged) +
           ",\"in_flight\":" + std::to_string(q.in_flight) + ",\"dropped\":" + std::to_string(q.dropped) +
           ",\"widened\":" + std::to_string(q.widened) + ",\"layers_dropped\":" + std::to_string(q.layers_dropped) +
           ",\"passes\":" + std::to_string(q.passes) + ",\"pass_loads\":" + std::to_string(q.pass_loads) +
           ",\"est_ms\":" + num(q.est_ms) + ",\"priced_removals\":" + std::to_string(q.removals) +
           ",\"changed\":" + detail::changed_json(q.changed) + ",\"changed_any\":" + std::to_string(q.changed_any) +
           ",\"fifth_executed\":" + std::to_string(q.fifth) +
           ",\"helper_second_submissions\":" + std::to_string(q.helper_second) +
           ",\"kernel_rows_checked\":" + std::to_string(q.kernel_checked) +
           ",\"kernel_mismatch_rows\":" + std::to_string(q.kernel_mismatch) +
           ",\"identity_mismatch_rows\":" + std::to_string(q.identity_mismatch) + "}";
}

inline std::string settings_json(const Settings& s) {
    using detail::num;
    std::string scale = "[";
    for (size_t i = 0; i < s.layer_scale.size(); ++i) { if (i) scale += ','; scale += num(s.layer_scale[i]); }
    scale += "]";
    return "{\"route_lab\":" + std::to_string(s.on ? 1 : 0) + ",\"routing_prior_bonus\":" + num(s.bonus) +
           ",\"route_bonus_draft\":" + num(s.bonus_draft) + ",\"route_prompt_bonus\":" + num(s.prompt_bonus) +
           ",\"route_lab_prompt\":" + std::to_string(s.prompt ? 1 : 0) +
           ",\"route_layer_scale\":" + scale +
           ",\"route_floor\":" + num(s.floor) + ",\"route_protect\":" + std::to_string(s.protect) +
           ",\"route_priced\":" + num(s.priced) + ",\"route_price_first\":" + num(s.price_first) +
           ",\"route_price_more\":" + num(s.price_more) + ",\"route_price_staged\":" + num(s.price_staged) +
           ",\"route_layer_drop\":" + num(s.layer_drop) + ",\"route_weights\":" + std::to_string(s.weights) +
           ",\"route_k_drop\":" + num(s.k_drop) + ",\"route_k_drop_ranks\":" + std::to_string(s.k_drop_ranks) +
           ",\"route_k_widen\":" + num(s.k_widen) + ",\"route_stats\":" + std::to_string(s.stats) + "}";
}

// ---- the request line's values (mm3_main.cpp)

// a float in [lo, hi]; false on anything else
inline bool parse_float(const std::string& v, float lo, float hi, float& out) {
    if (v.empty()) return false;
    char* end = nullptr;
    const float x = std::strtof(v.c_str(), &end);
    if (end == v.c_str() || *end || !std::isfinite(x) || x < lo || x > hi) return false;
    out = x;
    return true;
}
// route_layer_scale: n_moe comma-separated floats for the MoE layers in order, or groups "a-b:x" / "a:x" by model layer
// number, comma-separated (layers not named keep 1). moe_layers: the model numbers of the MoE layers, in order. The
// result is indexed by model layer (size n_layers); each value in [0, 16].
inline bool parse_layer_scale(const std::string& v, const std::vector<int>& moe_layers, int n_layers,
                              std::vector<float>& out) {
    if (v.empty() || n_layers <= 0) return false;
    std::vector<float> scale(size_t(n_layers), 1.0f);
    std::vector<std::string> parts;
    for (size_t at = 0;;) {
        const size_t c = v.find(',', at);
        parts.push_back(v.substr(at, c == std::string::npos ? std::string::npos : c - at));
        if (c == std::string::npos) break;
        at = c + 1;
    }
    const bool groups = v.find(':') != std::string::npos;
    if (!groups) {
        if (parts.size() != moe_layers.size()) return false;
        for (size_t i = 0; i < parts.size(); ++i) {
            float x;
            if (!parse_float(parts[i], 0.0f, 16.0f, x)) return false;
            const int li = moe_layers[i];
            if (li < 0 || li >= n_layers) return false;
            scale[size_t(li)] = x;
        }
    } else {
        for (const std::string& part : parts) {
            const size_t colon = part.find(':');
            if (colon == std::string::npos || colon == 0) return false;
            const std::string range = part.substr(0, colon);
            float x;
            if (!parse_float(part.substr(colon + 1), 0.0f, 16.0f, x)) return false;
            const size_t dash = range.find('-');
            char* end = nullptr;
            const long a = std::strtol(range.c_str(), &end, 10);
            if (end == range.c_str() || (dash == std::string::npos ? *end != 0 : size_t(end - range.c_str()) != dash))
                return false;
            long z = a;
            if (dash != std::string::npos) {
                const std::string hi = range.substr(dash + 1);
                z = std::strtol(hi.c_str(), &end, 10);
                if (hi.empty() || end == hi.c_str() || *end) return false;
            }
            if (a < 0 || z < a || z >= n_layers) return false;
            for (long li = a; li <= z; ++li)
                if (std::find(moe_layers.begin(), moe_layers.end(), int(li)) != moe_layers.end()) scale[size_t(li)] = x;
        }
    }
    out.swap(scale);
    return true;
}

}  // namespace mm3::route_lab
