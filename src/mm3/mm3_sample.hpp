// src/mm3/mm3_sample.hpp - maya-mm3's sampler (mm3_main.cpp's serve loop): a request's sampling settings, the next
// token's distribution from a row of logits or from its candidates (Model::forward_cands), draws, and the residual of
// speculative sampling.  Header-only, so tests/mm3_sampler_test.cpp checks it against itself.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mm3_model.hpp"

namespace mm3::sample {

struct Req {
    long long max_new = 0;
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
    unsigned long long seed = 0;
    int pen_n = 0;
    float pen_repeat = 1.0f, pen_freq = 0.0f, pen_present = 0.0f;
    long long think_budget = 0;
    int think_end = -1;
    int spec = -1;          // spec=0|1: this request drafts or not (-1: as the engine was started)
    int draft_width = -1;   // maximum draft positions; -1 uses the loaded block
    int fresh_context = 0; // independent request: rebuild target and draft context
    int state_test_reject_first = 0; // opt-in controlled serving diagnostic
    bool state_test_given = false;
    bool verify_segment_given = false;
    int stage_overlap = -1; // omitted: startup stage default
    int verify_segment = 0; // 0: one verification call; 1..7: ordered full-logit segments
    float conf = -1.0f;     // dspark_conf=F: the drafts' confidence floor for this request (-1: the engine's)
    std::string spec_policy = "existing"; // existing, sampled, lenience, fast_hsd (diagnostic cutoff)
    float spec_rho = -1.0f;
    float spec_ell = 0.4f;
    std::string routing;      // empty: startup policy; otherwise off, substitute or prior
    bool routing_given = false;
    int routing_prefetch = -1; // negative: startup value
    int routing_dampen = -1; // negative: startup value
    float routing_ratio = -1.0f; // negative: startup ratio
    bool routing_ratio_given = false;
    float routing_prior_bonus = -1.0f;
    bool routing_prior_bonus_given = false;
    int helper_live = -1; // omitted request uses static helper
    int cooperative_stage_reuse = -1; // omitted request restores startup setting
    double cooperative_cpu_scale = -1; // omitted request restores startup CPU price scale
    int primary_submit_query = -1; // omitted request restores startup setting
    int cpu_first = -1; // omitted request restores startup setting
    int helper_occupancy = 0; // experimental; off unless this request asks
    int helper_fast = 0; // experimental; off unless this request asks
    int helper_zero_copy = 0; // experimental; off unless this request asks (helper_fast's graphs only)
    int early_copy = 0; // experimental; off unless this request asks
    int early_copy_path = 0; // 0 the lean early copy, 1 its first form (A/B only)
    int prep_graph = 0; // experimental; off unless this request asks (2: verify, diagnostic)
    int expert_graph = 0; // experimental; off unless this request asks (2: verify, diagnostic)
    int dflash2_debug = 0; // (DFlash2 diagnostic) a proposal's parts switched off; 0 none
    int spec_plan = 0; // (step 6, planner v1) 1: the acceptance-aware verify width from --spec-plan's tables; 0: the floor
    int gpu_nucleus = 0; // (step 6) 1: temperature-1 top_p nuclei from GPU candidates and row stats; 0: whole rows on the host
    int dense_split = 0; // 1: dense mv splits K up to 512 outputs; off unless this request asks
    int exchange_schedule = 0; // 1: the exchanges' copies in pieces in idle time, 2: and checked; off unless this request asks
    double residency_decay = -1; // the native score's decay per five rows (negative: 0.95)
    double helper_max_exchanges = -1; // the 3090's exchange credit per five generated rows (negative: 1)
    int exchange_host_wait = -1; // forward ends a scheduled host copy waits for its window (negative: 2)
    std::string primary_policy;
    int primary_horizon = -1, primary_max_exchanges = -1;
    double primary_service_ms = -1, primary_h2d_ms = -1, primary_d2h_ms = -1;
    double primary_interference_ms = -1, primary_margin_ms = -1;
    int primary_adaptive = -1; // explicit 0/1; absent requests retain the fixed default
    std::string primary_arithmetic; // absent: startup arithmetic, not the preceding request
    std::string helper_dispatch; // absent: startup dispatch, not the preceding request
    std::string file;       // GENI: the pictures
    std::vector<int> ids;
    int rid = -1;           // rid=N: the request's id (--parallel: every line it causes carries it)
};

inline bool penalised(const Req& r) {
    return r.pen_n != 0 && (r.pen_repeat != 1.0f || r.pen_freq != 0.0f || r.pen_present != 0.0f);
}

// Full-vocabulary nucleus candidate. The old full sort sums all exponentials
// in sorted order; this sums them in input order. Positive double sums and the
// CDF have bounded rounding error. If either side of the selected cutoff is
// within the conservative bound, fall back to the old sorted reduction.
// Returned probabilities are always normalized in sorted retained order.
inline bool adaptive_nucleus(std::vector<std::pair<float,int>>& v, float top_p,
                             std::vector<std::pair<int,double>>& out) {
    const size_t n = v.size();
    if (n < 1024 || !(top_p > 0.0f && top_p < 1.0f)) return false;
    const auto before = [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    for (const auto& item : v) if (!std::isfinite(item.first)) return false;
    const float maximum = std::min_element(v.begin(), v.end(), before)->first;
    double total = 0;
    for (const auto& item : v) total += std::exp((double)(item.first - maximum));
    if (!std::isfinite(total) || !(total > 0)) return false;
    // Every weight is at most one. If even the lower bound on the number of
    // survivors is broad, avoid repeated partitions and use the full sort.
    const size_t limit = n / 4;
    if (std::ceil((double)top_p * total) >= (double)limit) return false;
    const double eps_n = (double)n * std::numeric_limits<double>::epsilon();
    if (eps_n >= 0.001) return false;
    const double guard = 8.0 * eps_n / (1.0 - 2.0 * eps_n);
    for (size_t count = std::min<size_t>(256, limit); count < n && count <= limit;) {
        std::nth_element(v.begin(), v.begin() + (long)count, v.end(), before);
        std::sort(v.begin(), v.begin() + (long)count, before);
        double cumulative = 0;
        for (size_t i = 0; i < count; ++i) {
            const double previous = cumulative;
            cumulative += std::exp((double)(v[i].first - maximum)) / total;
            if (cumulative >= (double)top_p) {
                if (std::abs(previous - (double)top_p) <= guard ||
                    std::abs(cumulative - (double)top_p) <= guard) return false;
                std::vector<double> weights(i + 1);
                double retained = 0;
                for (size_t j = 0; j <= i; ++j)
                    retained += (weights[j] = std::exp((double)(v[j].first - maximum)));
                out.resize(i + 1);
                for (size_t j = 0; j <= i; ++j) out[j] = {v[j].second, weights[j] / retained};
                return true;
            }
        }
        if (cumulative >= (double)top_p - guard || count >= limit) return false;
        count = std::min(limit, count * 2);
    }
    return false;
}

// (step 6, the GPU nucleus) adaptive_nucleus's result from a row's candidates instead of the whole row: top, the row's
// largest logits sorted as adaptive_nucleus sorts (logit desc, ties to the lower id; Model's topk_rows), and the row's
// maximum and total (row_stats: the sum of exp(x - maximum) over all n logits). It walks the same counts (256, 512, ...)
// over the same sorted prefix, so its sums and guard checks are adaptive_nucleus's, the total aside. total_rel_err bounds
// the total's relative error against adaptive_nucleus's (k::kRowStatsRelErr for the GPU's FP32 exponentials): every
// cumulative is then within that of adaptive_nucleus's, so the guard band widens to 8 times it - outside the band each
// comparison with top_p comes out as adaptive_nucleus's, inside it the row falls back - and the limit check takes the
// total's upper bound. false wherever adaptive_nucleus would need more than the candidates or might decline: the caller
// then takes the whole row, which decides exactly as before. Temperature 1, no top_k, min_p or penalties (the caller's).
inline bool nucleus_from(const std::vector<std::pair<int, float>>& top, float maximum, double total, float top_p, size_t n,
                         std::vector<std::pair<int, double>>& out, double total_rel_err = 0.0) {
    if (n < 1024 || !(top_p > 0.0f && top_p < 1.0f) || top.empty()) return false;
    if (!std::isfinite(total) || !(total > 0)) return false;
    const size_t limit = n / 4;
    if (std::ceil((double)top_p * total * (1.0 + 2.0 * total_rel_err)) >= (double)limit) return false;
    const double eps_n = (double)n * std::numeric_limits<double>::epsilon();
    if (eps_n >= 0.001) return false;
    const double guard = std::max(8.0 * eps_n / (1.0 - 2.0 * eps_n), 8.0 * total_rel_err);
    double cumulative = 0;
    size_t i = 0;
    for (size_t count = std::min<size_t>(256, limit); count < n && count <= limit;) {
        if (count > top.size()) return false;   // (adaptive_nucleus would sort further than the candidates reach)
        for (; i < count; ++i) {
            const double previous = cumulative;
            cumulative += std::exp((double)(top[i].second - maximum)) / total;
            if (cumulative >= (double)top_p) {
                if (std::abs(previous - (double)top_p) <= guard || std::abs(cumulative - (double)top_p) <= guard) return false;
                std::vector<double> weights(i + 1);
                double retained = 0;
                for (size_t j = 0; j <= i; ++j) retained += (weights[j] = std::exp((double)(top[j].second - maximum)));
                out.resize(i + 1);
                for (size_t j = 0; j <= i; ++j) out[j] = {top[j].first, weights[j] / retained};
                return true;
            }
        }
        if (cumulative >= (double)top_p - guard || count >= limit) return false;
        count = std::min(limit, count * 2);
    }
    return false;
}

// the next token's distribution from a row's candidate logits - (id, logit) pairs holding every id the distribution
// can hold: the whole row, or Model::forward_cands' kc largest with their ties - and the logits of the extra ids (the
// penalised ones not among them): penalties over the history's last pen_n tokens (llama.cpp's), then greedy
// (temperature <= 0: the argmax alone, ties to the lower id) or temperature, top_k, top_p, min_p - (token, probability)
// pairs, the most likely first (ties: the lower id first), summing to 1.  Over the candidates it is the whole row's
// distribution whenever kc >= top_k + extra.size() and extra holds every penalised id.
inline std::vector<std::pair<int, double>> dist_c(std::vector<std::pair<int, float>> c, const std::vector<int>& extra,
                                                  const std::vector<float>& ex, const Req& r,
                                                  const std::vector<int>& hist) {
    if (penalised(r)) {
        std::unordered_map<int, int> cnt;
        const size_t n = r.pen_n < 0 ? hist.size() : std::min(hist.size(), (size_t) r.pen_n);
        for (size_t i = hist.size() - n; i < hist.size(); ++i) ++cnt[hist[i]];
        std::unordered_map<int, size_t> at;
        for (size_t i = 0; i < c.size(); ++i) at[c[i].first] = i;
        for (const auto& [t, k] : cnt) {
            auto it = at.find(t);
            size_t i = 0;
            if (it != at.end()) {
                i = it->second;
            } else {   // (a penalised id below the candidates: its logit from extra, a candidate now)
                const auto e = std::find(extra.begin(), extra.end(), t);
                if (e == extra.end()) continue;   // (outside the vocabulary)
                c.push_back({t, ex[(size_t) (e - extra.begin())]});
                i = c.size() - 1;
            }
            float& x = c[i].second;
            if (r.pen_repeat > 0.0f && r.pen_repeat != 1.0f) x = x > 0 ? x / r.pen_repeat : x * r.pen_repeat;
            x -= r.pen_freq * (float) k + r.pen_present;
        }
    }
    if (!(r.temperature > 0.0f)) {
        size_t b = 0;
        for (size_t i = 1; i < c.size(); ++i)
            if (c[i].second > c[b].second || (c[i].second == c[b].second && c[i].first < c[b].first)) b = i;
        return {{c[b].first, 1.0}};
    }
    std::vector<std::pair<float, int>> v;
    v.reserve(c.size());
    for (const auto& [id, x] : c) v.push_back({x / r.temperature, id});
    const size_t k = r.top_k > 0 ? std::min((size_t) r.top_k, v.size()) : v.size();
    const auto before = [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    std::vector<std::pair<int,double>> adaptive;
    if (r.top_k <= 0 && r.min_p == 0.0f && adaptive_nucleus(v, r.top_p, adaptive)) return adaptive;
    if (r.top_k <= 0) std::sort(v.begin(), v.end(), before);
    else std::partial_sort(v.begin(), v.begin() + (long) k, v.end(), before);
    const auto& cs = v;
    double sum = 0;
    std::vector<double> p(k);
    for (size_t i = 0; i < k; ++i) sum += (p[i] = std::exp((double) (cs[i].first - cs[0].first)));
    size_t keep = k;
    double cum = 0;
    for (size_t i = 0; i < k; ++i) {
        cum += p[i] / sum;
        if (r.min_p > 0.0f && p[i] < r.min_p * p[0]) {
            keep = std::max<size_t>(1, i);
            break;
        }
        if (r.top_p < 1.0f && cum >= r.top_p) {
            keep = i + 1;
            break;
        }
    }
    double s2 = 0;
    for (size_t i = 0; i < keep; ++i) s2 += p[i];
    std::vector<std::pair<int, double>> out(keep);
    for (size_t i = 0; i < keep; ++i) out[i] = {cs[i].second, p[i] / s2};
    return out;
}

// ... from a whole row of logits
inline std::vector<std::pair<int, double>> dist(const float* row, int V, const Req& r, const std::vector<int>& hist) {
    std::vector<std::pair<int, float>> c((size_t) V);
    for (int i = 0; i < V; ++i) c[(size_t) i] = {i, row[i]};
    return dist_c(std::move(c), {}, {}, r, hist);
}

// a draw from d, without `skip` (renormalised; -1: none)
inline int draw(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double tot = 0;
    for (const auto& [t, q] : d)
        if (t != skip) tot += q;
    if (!(tot > 0)) return d[0].first;
    double u = std::uniform_real_distribution<double>(0.0, tot)(rng);
    int last = d[0].first;
    for (const auto& [t, q] : d) {
        if (t == skip) continue;
        last = t;
        u -= q;
        if (u <= 0) return t;
    }
    return last;
}

// Fast-HSD speccascade_accepts/min_p_allowed_set, pinned revision
// 66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17. d is the already truncated
// target distribution, so an absent draft is outside its support.
inline bool fast_hsd_accepts(const std::vector<std::pair<int, double>>& d, int draft, double rho) {
    if (d.empty() || !(rho > 0.0 && rho <= 1.0)) return false;
    double maximum = 0.0;
    for (const auto& item : d) maximum = std::max(maximum, item.second);
    for (const auto& item : d)
        if (item.first == draft) return item.second > 0.0 && item.second >= rho * maximum;
    return false;
}

// The proposal must be the filtered, normalized distribution actually sampled by DSpark.
inline double lenience_accept_prob(double px, double qx, double ell) {
    if (!(ell > 0.0 && ell <= 1.0) || !(qx > 0.0) || !(px > 0.0)) return 0.0;
    return std::min(1.0, px / (ell * qx));
}

// a draw from max(0, p - q) renormalised (the token after a sampled draft the target turned down: speculative
// sampling's residual, which leaves the token the target's own sample); p itself when that is empty
inline int draw_residual(const std::vector<std::pair<int, double>>& p, const std::vector<std::pair<int, float>>& q,
                         std::mt19937_64& rng) {
    std::vector<std::pair<int, double>> d;
    for (const auto& [t, pt] : p) {
        double qt = 0;
        for (const auto& [u, qu] : q)
            if (u == t) qt = qu;
        if (pt > qt) d.push_back({t, pt - qt});
    }
    return d.empty() ? draw(p, rng) : draw(d, rng);
}

// a forward's candidates for these rows (mm3::Model::forward_cands): kc = top_k + the penalised ids (the last pen_n
// tokens of the history and the window's - every row's own window within them); false: the whole rows instead (no
// top_k, or the whole history penalised)
inline bool cands_for(const Req& r, const std::vector<int>& hist, const std::vector<int>& more, mm3::Cands& c) {
    c.extra.clear();
    if (penalised(r)) {
        if (r.pen_n < 0) return false;
        std::vector<int> all(hist.end() - (long) std::min(hist.size(), (size_t) r.pen_n), hist.end());
        all.insert(all.end(), more.begin(), more.end());
        for (int t : all)
            if (std::find(c.extra.begin(), c.extra.end(), t) == c.extra.end()) c.extra.push_back(t);
    }
    if (r.temperature > 0.0f && !(r.top_k > 0)) return false;
    c.kc = (r.temperature > 0.0f ? r.top_k : 1) + (int) c.extra.size();
    return true;
}

}  // namespace mm3::sample
