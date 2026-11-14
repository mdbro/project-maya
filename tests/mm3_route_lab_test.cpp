// mm3_route_lab.hpp (the routing lab's rules, routing-lab-20261110/CONTRACT.md) on the CPU:
//   - exactness: with every rule off, route_pass and reference_prior equal a literal CPU model of the router kernel
//     (mm3_resident_router.cu, Policy Prior and Off) bit for bit, over many random rows, ties included;
//   - each rule alone at its edges, the rules in combination, the invariants over random settings;
//   - the floor against a port of the re-routed replay's reroute_floor (tools/mm3/native_reroute_replay.py);
//   - the serving kernel's per-row budget (mm3_route_floor.hpp, the code mm3_resident_router.cu runs, with the kernel's
//     selection around it) against the lab path's route_floor on recorded router inputs (tests/data/
//     mm3_router_recorded.bin, from tests/mm3_route_floor_fixture.py; argv[1] or MM3_ROUTER_RECORDED: another file of
//     that format, e.g. every recorded row) and on random rows with ties: the same four, order and weights bit for bit.
// ROUTE_LAB_NEGATE=<line> negates the CHECK on that line (the mutation runs: each negated check must fail the test).
#include "mm3_route_lab.hpp"
#include "mm3_route_floor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace rl = mm3::route_lab;

static int g_fail = 0;
static const int g_negate = std::getenv("ROUTE_LAB_NEGATE") ? std::atoi(std::getenv("ROUTE_LAB_NEGATE")) : -1;
static bool check_at(bool ok, int line, const char* what) {
    if (line == g_negate) ok = !ok;
    if (!ok) {
        if (g_fail < 20) std::fprintf(stderr, "FAIL line %d: %s\n", line, what);
        ++g_fail;
    }
    return ok;
}
#define CHECK(c) check_at(bool(c), __LINE__, #c)

static uint32_t fbits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

// ---- the router kernel, literally (one row; the warp's reduction is an argmax with ties to the smaller id)
struct Kernel {
    int ids[4], original[4];
    float w[4];
    float capture[128 * 3];
};
static bool kbetter(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }
static void kernel_row(const float* logits, const float* bias, const uint32_t* primary, const uint32_t* helper, bool prior,
                       float parameter, Kernel& k) {
    float prob[128], score[128];
    for (int e = 0; e < 128; ++e) {
        prob[e] = 1.0f / (1.0f + expf(-logits[e]));
        score[e] = prob[e] + bias[e];
    }
    const auto ready = [&](int e) { return rl::bit(primary, e) || rl::bit(helper, e); };
    int ranked[8];
    float ranked_prob[8];
    for (int rank = 0; rank < 8; ++rank) {
        float best = -INFINITY, bp = 0.0f;
        int bi = 0x7fffffff;
        for (int e = 0; e < 128; ++e) if (kbetter(score[e], e, best, bi)) { best = score[e]; bi = e; bp = prob[e]; }
        ranked[rank] = bi;
        ranked_prob[rank] = bp;
        score[bi] = -INFINITY;
    }
    int selected[4] = {ranked[0], ranked[1], ranked[2], ranked[3]};
    float selected_prob[4] = {ranked_prob[0], ranked_prob[1], ranked_prob[2], ranked_prob[3]};
    if (prior && parameter > 0.0f) {
        for (int e = 0; e < 128; ++e) score[e] = prob[e] + bias[e] + (ready(e) ? parameter : 0.0f);
        for (int rank = 0; rank < 4; ++rank) {
            float best = -INFINITY, bp = 0.0f;
            int bi = 0x7fffffff;
            for (int e = 0; e < 128; ++e) if (kbetter(score[e], e, best, bi)) { best = score[e]; bi = e; bp = prob[e]; }
            selected[rank] = bi;
            selected_prob[rank] = bp;
            score[bi] = -INFINITY;
        }
    }
    for (int id = 0; id < 128; ++id) {
        const float p = prob[id], base = p + bias[id];
        const float adjusted = prior && parameter > 0.0f ? base + (ready(id) ? parameter : 0.0f) : base;
        k.capture[id * 3] = p;
        k.capture[id * 3 + 1] = base;
        k.capture[id * 3 + 2] = adjusted;
    }
    const float sum = selected_prob[0] + selected_prob[1] + selected_prob[2] + selected_prob[3];
    for (int i = 0; i < 4; ++i) {
        k.ids[i] = selected[i];
        k.original[i] = ranked[i];
        k.w[i] = sum > 0.0f ? selected_prob[i] / sum : 0.25f;
    }
}

// ---- hand-built rows: p and bias given directly (base = p + bias)
struct Pass {
    int rows = 1;
    std::vector<float> scores;   // [rows][128][3]
    rl::Masks m{};
    explicit Pass(int r = 1) : rows(r), scores(size_t(r) * 128 * 3, 0.0f) {}
    void set(int row, int e, float p, float bias = 0.0f) {
        float* s = scores.data() + (size_t(row) * 128 + e) * 3;
        s[0] = p;
        s[1] = p + bias;
        s[2] = s[1];
    }
    // every expert a small distinct p below `top` (ranked by id: smaller id higher)
    void fill(int row, float top = 0.05f) { for (int e = 0; e < 128; ++e) set(row, e, top * (1.0f - float(e) / 256.0f)); }
    void ready(int e, bool helper = false, bool resident = true) {
        rl::set_bit(m.ready, e);
        if (helper) rl::set_bit(m.helper, e);
        else if (resident) rl::set_bit(m.resident, e);
    }
    void flying(int e) { rl::set_bit(m.in_flight, e); }
    std::vector<rl::Row> run(const rl::Settings& s, int layer = 10, rl::PassStats* ps = nullptr) const {
        std::vector<rl::Row> out(static_cast<size_t>(rows));
        rl::route_pass(s, layer, rows, scores.data(), m, nullptr, out.data(), ps);
        return out;
    }
};
static std::set<int> executed(const rl::Row& r) {
    std::set<int> s;
    for (int i = 0; i < r.slots; ++i) if (!r.skip[i]) s.insert(r.ids[i]);
    return s;
}
static float weight_of(const rl::Row& r, int e) {
    for (int i = 0; i < r.slots; ++i) if (r.ids[i] == e && !r.skip[i]) return r.w[i];
    return -1.0f;
}
static bool near(double a, double b, double tol = 1e-6) { return std::fabs(a - b) <= tol; }

// the standard hand row: originals 0..3 with p .9 .8 .7 .6, rank 5 and 6 (experts 4, 5) .5 and .45, the rest below
static Pass standard(int rows = 1) {
    Pass q(rows);
    for (int r = 0; r < rows; ++r) {
        q.fill(r);
        q.set(r, 0, 0.9f); q.set(r, 1, 0.8f); q.set(r, 2, 0.7f); q.set(r, 3, 0.6f); q.set(r, 4, 0.5f); q.set(r, 5, 0.45f);
    }
    return q;
}

// ---- exactness against the kernel
static void test_exact_random() {
    std::mt19937 rng(20261110);
    std::normal_distribution<float> logit(0.0f, 1.5f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    const float bonuses[] = {0.0f, 0.05f, 0.2f, 0.25f, 0.3f, 0.4f, 1.0f, 0.123456f};
    long rows_checked = 0, ties = 0;
    for (int it = 0; it < 4000; ++it) {
        const int T = it % 3 == 0 ? 1 + int(uni(rng) * 16) % 16 : 1;
        const float bonus = bonuses[it % 8];
        const bool prior = it % 11 != 0;   // (Off: the router's own four)
        const int mode = it % 5;           // 0 normal, 1 coarse (ties), 2 bias large, 3 none ready, 4 all ready
        uint32_t primary[4] = {}, helper[4] = {};
        const float q = uni(rng);
        for (int e = 0; e < 128; ++e) {
            if (mode == 3) continue;
            if (mode == 4 || uni(rng) < q) (uni(rng) < 0.3f ? helper : primary)[e >> 5] |= 1u << (e & 31);
        }
        rl::Masks m{};
        for (int w = 0; w < 4; ++w) { m.ready[w] = primary[w] | helper[w]; m.helper[w] = helper[w]; m.resident[w] = primary[w]; }
        std::vector<float> scores(size_t(T) * 384);
        std::vector<Kernel> K(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t) {
            float lg[128], bias[128];
            for (int e = 0; e < 128; ++e) {
                lg[e] = logit(rng);
                bias[e] = mode == 2 ? 2.0f * (uni(rng) - 0.5f) : 0.1f * (uni(rng) - 0.5f);
                if (mode == 1) { lg[e] = std::round(lg[e] * 2.0f) / 2.0f; bias[e] = uni(rng) < 0.5f ? 0.0f : 0.25f; }
            }
            if (mode == 0 && it % 7 == 0) {   // near ties: one ulp apart in bias
                lg[1] = lg[0];
                bias[1] = std::nextafter(bias[0], -1.0f);
            }
            kernel_row(lg, bias, primary, helper, prior, bonus, K[size_t(t)]);
            std::memcpy(scores.data() + size_t(t) * 384, K[size_t(t)].capture, sizeof K[0].capture);
            std::set<uint32_t> seen;
            for (int e = 0; e < 128; ++e) ties += !seen.insert(fbits(K[size_t(t)].capture[e * 3 + 1])).second;
        }
        rl::Settings s;
        s.on = true;
        s.bonus = prior ? bonus : 0.0f;
        std::vector<rl::Row> out(static_cast<size_t>(T));
        rl::route_pass(s, 3 + it % 57, T, scores.data(), m, nullptr, out.data());
        for (int t = 0; t < T; ++t) {
            const Kernel& k = K[size_t(t)];
            const rl::Row& r = out[size_t(t)];
            int ref_ids[4];
            float ref_w[4];
            rl::reference_prior(scores.data() + size_t(t) * 384, m.ready, s.bonus, ref_ids, ref_w);
            bool same = r.slots == 4, ref_same = true;
            for (int i = 0; i < 4; ++i) {
                same &= r.ids[i] == k.ids[i] && fbits(r.w[i]) == fbits(k.w[i]) && !r.skip[i] && r.original[i] == k.original[i];
                ref_same &= ref_ids[i] == k.ids[i] && fbits(ref_w[i]) == fbits(k.w[i]);
            }
            CHECK(same);
            CHECK(ref_same);
            CHECK(r.executed == 4);
            CHECK((r.changed & ~rl::kBonus) == 0);   // (no lab rule acted; the bonus is today's)
            ++rows_checked;
        }
    }
    CHECK(rows_checked > 10000);
    CHECK(ties > 1000);   // (the tie rule was exercised)
    std::printf("exactness: %ld rows equal the kernel model bit for bit (%ld equal base scores seen)\n", rows_checked, ties);
}

// ---- each rule alone
static void test_layer_drop() {
    Pass q = standard();
    rl::Settings s;
    s.on = true;
    const float sum = 0.9f + 0.8f + 0.7f + 0.6f;
    s.layer_drop = sum;   // not below: kept
    auto out = q.run(s);
    CHECK(!out[0].layer_dropped && out[0].executed == 4);
    s.layer_drop = std::nextafter(sum, 10.0f);   // just above: dropped
    out = q.run(s);
    CHECK(out[0].layer_dropped);
    CHECK(out[0].changed == rl::kLayerDrop);
    CHECK(out[0].executed == 0 && out[0].kept == 0.0);
    bool all = true;
    for (int i = 0; i < out[0].slots; ++i) all &= out[0].skip[i] && out[0].w[i] == 0.0f;
    CHECK(all);
    CHECK(executed(out[0]).empty());
    rl::PassStats ps;
    q.run(s, 10, &ps);
    CHECK(ps.loads == 0 && ps.est_ms == 0.0);   // (a dropped layer loads nothing)
}

static void test_protect() {
    Pass q = standard();
    for (int e = 2; e <= 10; ++e) q.ready(e);   // 0 and 1 unready
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;
    auto out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{2, 3, 4, 5}));
    CHECK(out[0].moved == 2 && out[0].loads == 0);
    CHECK(out[0].changed == rl::kBonus);
    s.protect = 1;
    out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{0, 2, 3, 4}));
    CHECK(out[0].loads == 1);
    CHECK(out[0].changed == (rl::kBonus | rl::kProtect));
    s.protect = 2;
    out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{0, 1, 2, 3}));
    CHECK(out[0].moved == 0 && out[0].loads == 2);
}

static void test_floor() {
    Pass q = standard();
    for (int e = 2; e <= 10; ++e) q.ready(e);
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;
    // pairs: (1 -> 4) loses .3 of 3.0 (kept .9), then (0 -> 5) loses .45 (kept .75)
    s.floor = 0.8f;
    auto out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{0, 2, 3, 4}));
    CHECK(out[0].changed == (rl::kBonus | rl::kFloor));
    CHECK(near(out[0].kept, (0.9 + 0.7 + 0.6 + 0.5) / 3.0, 1e-6));
    s.floor = 0.95f;
    out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{0, 1, 2, 3}));
    s.floor = 0.74f;
    out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{2, 3, 4, 5}));
    CHECK(out[0].changed == rl::kBonus);   // (the floor kept nothing out)
    s.floor = 0.76f;
    out = q.run(s);
    CHECK((executed(out[0]) == std::set<int>{0, 2, 3, 4}));
    s.floor = 1.0f;   // nothing that loses mass
    out = q.run(s);
    CHECK(out[0].moved == 0);
}

static void test_priced() {
    // 0 and 1 unready; the bonus pairs (1 -> 4) and (0 -> 5); expert 5's p .1 makes 0's removal costly
    Pass q = standard();
    q.set(0, 5, 0.1f);
    for (int e = 2; e <= 5; ++e) q.ready(e);
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;
    s.priced = 3.0f;   // first: 1 at .32 / (.3 / 3) = 3.2; then 0 alone at .6 / (.8 / 3) = 2.25: stop
    rl::PassStats ps;
    auto out = q.run(s, 10, &ps);
    CHECK((executed(out[0]) == std::set<int>{0, 2, 3, 4}));
    CHECK(ps.removals == 1 && ps.loads == 1);
    CHECK(near(ps.est_ms, 0.60, 1e-6));
    CHECK(out[0].changed == (rl::kBonus | rl::kPriced));
    s.priced = 2.0f;   // both: 3.2, then 2.25
    out = q.run(s, 10, &ps);
    CHECK((executed(out[0]) == std::set<int>{2, 3, 4, 5}));
    CHECK(ps.removals == 2 && ps.loads == 0 && ps.est_ms == 0.0);
    s.priced = 3.3f;   // neither (3.2 and 1.2 below)
    out = q.run(s, 10, &ps);
    CHECK(ps.removals == 0 && out[0].moved == 0);
    CHECK(near(ps.est_ms, 0.60 + 0.32, 1e-6));
    // a copy in flight saves price_staged only: 1 at .1 / .1 = 1.0; 0, now the only load, .6 / .2667 = 2.25 first
    Pass f = q;
    f.flying(1);
    s.priced = 2.0f;
    out = f.run(s, 10, &ps);
    CHECK((executed(out[0]) == std::set<int>{1, 2, 3, 5}));
    CHECK(out[0].in_flight == 1 && ps.in_flight == 1 && ps.loads == 0);
    // with the floor: 1's removal (kept .9) is allowed at 0.85, 0's (kept .667) is not
    s.priced = 0.5f;
    s.floor = 0.85f;
    out = q.run(s, 10, &ps);
    CHECK((executed(out[0]) == std::set<int>{0, 2, 3, 4}));
    CHECK(out[0].kept >= 0.85);
    CHECK(out[0].changed == (rl::kBonus | rl::kFloor));   // (0's removal was kept out by the floor, not the price)
    // drafting: a load another row executes without a candidate is never removed (that row can't move it)
    Pass d(2);
    for (int r = 0; r < 2; ++r) {
        d.fill(r);
        d.set(r, 0, 0.9f); d.set(r, 1, 0.8f); d.set(r, 2, 0.7f); d.set(r, 3, 0.6f);
    }
    d.set(0, 4, 0.5f);   // row 0: expert 4 (p .5) is its candidate for 3
    d.set(1, 4, 0.001f);
    d.ready(4);          // only 4 is ready
    rl::Settings sd;
    sd.on = true;
    sd.bonus = 1.0f;
    sd.bonus_draft = 0.5f;   // row 1: .001 + .5 < .6, 4 does not displace 3 there; row 0: .5 + .5 > .6, it does
    sd.priced = 1e-6f;
    out = d.run(sd, 10, &ps);
    CHECK(executed(out[0]).count(3) == 1);   // 3 stays in row 0: row 1 needs it anyway
    CHECK(ps.removals == 0);
    sd.priced = 0.0f;   // without pricing row 0 substitutes alone
    out = d.run(sd, 10, &ps);
    CHECK(executed(out[0]).count(3) == 0 && executed(out[0]).count(4) == 1);
    CHECK(executed(out[1]).count(3) == 1);
}

static void test_k_drop() {
    Pass q = standard();
    for (int e = 0; e <= 2; ++e) q.ready(e);   // 3 unready; its weight .6 / 3.0 = .2
    rl::Settings s;
    s.on = true;
    s.k_drop = 0.21f;
    auto out = q.run(s);
    CHECK(out[0].dropped == 1 && out[0].executed == 3);
    CHECK(out[0].changed == rl::kKDrop);
    CHECK(executed(out[0]).count(3) == 0);
    CHECK(out[0].loads == 0);
    float sum = 0.0f;
    for (int i = 0; i < out[0].slots; ++i) sum += out[0].w[i];
    CHECK(near(sum, 1.0, 1e-6));   // mode 0: renormalised over three
    s.weights = 1;
    out = q.run(s);
    CHECK(near(weight_of(out[0], 0), 0.9 / 3.0, 1e-6) && out[0].dropped == 1);   // dampen: the dropped weight is lost
    s.weights = 0;
    s.k_drop = 0.19f;   // above its weight: kept
    out = q.run(s);
    CHECK(out[0].dropped == 0);
    // rank 3 (expert 2) unready too: only with ranks 3-4
    Pass r = standard();
    r.ready(0); r.ready(1);
    s.k_drop = 0.24f;   // 2's weight .7 / 3 = .233, 3's .2
    out = r.run(s);
    CHECK(out[0].dropped == 1 && executed(out[0]).count(2) == 1);
    s.k_drop_ranks = 2;
    out = r.run(s);
    CHECK(out[0].dropped == 2 && out[0].executed == 2);
    // a ready pick is never dropped
    Pass a = standard();
    for (int e = 0; e < 4; ++e) a.ready(e);
    out = a.run(s);
    CHECK(out[0].dropped == 0);
}

static void test_k_widen() {
    Pass q = standard();
    for (int e : {0, 1, 2, 4, 5}) q.ready(e);   // 3 unready, ranks 5-6 (4, 5) ready
    rl::Settings s;
    s.on = true;
    s.k_widen = 0.45f;
    auto out = q.run(s);
    CHECK(out[0].widened && out[0].slots == 5 && out[0].executed == 5);
    CHECK(out[0].changed == rl::kWiden);
    CHECK((executed(out[0]) == std::set<int>{0, 1, 2, 4, 5}));
    CHECK(out[0].loads == 0 && out[0].moved == 2);
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) sum += out[0].w[i];
    CHECK(near(sum, 1.0, 1e-6));
    s.weights = 2;   // amplify: 4 and 5 split 3's weight .2 by their p
    out = q.run(s);
    CHECK(near(weight_of(out[0], 4), 0.2 * 0.5 / 0.95, 1e-6) && near(weight_of(out[0], 5), 0.2 * 0.45 / 0.95, 1e-6));
    CHECK(near(weight_of(out[0], 0), 0.3, 1e-6));
    s.weights = 0;
    s.k_widen = 0.46f;   // rank 6 below
    out = q.run(s);
    CHECK(!out[0].widened && out[0].slots == 4);
    s.k_widen = 0.45f;
    Pass u = standard();   // rank 5 not ready
    for (int e : {0, 1, 2, 5}) u.ready(e);
    out = u.run(s);
    CHECK(!out[0].widened);
    Pass two = standard();   // two unready
    for (int e : {0, 1, 4, 5}) two.ready(e);
    out = two.run(s);
    CHECK(!out[0].widened);
    Pass prot = standard();   // the unready pick protected
    for (int e : {1, 2, 3, 4, 5}) prot.ready(e);
    s.protect = 1;
    out = prot.run(s);
    CHECK(!out[0].widened && executed(out[0]).count(0) == 1);
    s.protect = 0;
    out = prot.run(s);
    CHECK(out[0].widened);
    s.bonus = 1.0f;   // the bonus substitutes 3 first: no widening
    out = q.run(s);
    CHECK(!out[0].widened && out[0].moved == 1);
    s.bonus = 0.0f;
    s.k_drop = 0.21f;   // drop runs first and wins
    out = q.run(s);
    CHECK(!out[0].widened && out[0].dropped == 1);
}

static void test_weights() {
    Pass q = standard();
    for (int e = 2; e <= 10; ++e) q.ready(e);
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;   // pairs (1 -> 4), (0 -> 5): executed 2 3 4 5
    s.weights = 1;
    auto out = q.run(s);
    const float t = 0.9f + 0.8f + 0.7f + 0.6f;
    CHECK(fbits(weight_of(out[0], 4)) == fbits(0.5f / t) && fbits(weight_of(out[0], 5)) == fbits(0.45f / t));
    CHECK(fbits(weight_of(out[0], 2)) == fbits(0.7f / t));
    s.weights = 2;
    out = q.run(s);
    CHECK(fbits(weight_of(out[0], 4)) == fbits(0.8f / t) && fbits(weight_of(out[0], 5)) == fbits(0.9f / t));
    CHECK(fbits(weight_of(out[0], 3)) == fbits(0.6f / t));
    CHECK(out[0].changed == (rl::kBonus | rl::kWeights));
    s.weights = 0;
    out = q.run(s);
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) sum += out[0].w[i];
    CHECK(near(sum, 1.0, 1e-6));
}

static void test_bonus_modes() {
    Pass one = standard(1), two = standard(2);
    for (int e = 2; e <= 10; ++e) { one.ready(e); two.ready(e); }
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;
    s.bonus_draft = 0.0f;
    CHECK(one.run(s)[0].moved == 2);
    auto o2 = two.run(s);
    CHECK(o2[0].moved == 0 && o2[1].moved == 0);
    s.bonus_draft = -1.0f;
    CHECK(two.run(s)[1].moved == 2);
    s.layer_scale.assign(60, 1.0f);
    s.layer_scale[10] = 0.0f;
    CHECK(one.run(s, 10)[0].moved == 0);
    CHECK(one.run(s, 11)[0].moved == 2);
    s.layer_scale[11] = 0.1f;   // 1.0 x 0.1: .5 + .1 < .8, .45 + .1 < .9: no move; 3 stays too
    CHECK(one.run(s, 11)[0].moved == 0);
}

// ---- the floor against the replay's reroute_floor (as sets)
static std::vector<int> replay_floor(const float* row, const uint32_t* ready, float bonus, float floor) {
    float base[128], adj[128];
    for (int e = 0; e < 128; ++e) {
        base[e] = row[e * 3 + 1];
        adj[e] = bonus > 0 ? base[e] + (rl::bit(ready, e) ? bonus : 0.0f) : base[e];
    }
    const auto top4 = [](const float* v) {
        std::vector<int> idx(128);
        for (int i = 0; i < 128; ++i) idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
        idx.resize(4);
        return idx;
    };
    const auto O = top4(base), S = top4(adj);
    const auto sig = [&](int e) { return row[e * 3]; };
    std::vector<int> D, N;
    for (int e : O) if (std::find(S.begin(), S.end(), e) == S.end()) D.push_back(e);
    for (int e : S) if (std::find(O.begin(), O.end(), e) == O.end()) N.push_back(e);
    std::sort(D.begin(), D.end(), [&](int a, int b) { return sig(a) < sig(b) || (sig(a) == sig(b) && a < b); });
    std::sort(N.begin(), N.end(), [&](int a, int b) { return sig(a) > sig(b) || (sig(a) == sig(b) && a < b); });
    if (D.empty()) return S;
    const double total = double(sig(O[0])) + double(sig(O[1])) + double(sig(O[2])) + double(sig(O[3]));
    double kept = total;
    std::vector<int> out = O;
    for (size_t i = 0; i < D.size() && i < N.size(); ++i) {
        const double nk = kept - double(sig(D[i])) + double(sig(N[i]));
        if (nk < double(floor) * total) break;
        *std::find(out.begin(), out.end(), D[i]) = N[i];
        kept = nk;
    }
    return out;
}
static void test_floor_vs_replay() {
    std::mt19937 rng(7);
    std::normal_distribution<float> logit(0.0f, 1.5f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    long same = 0, n = 0;
    for (int it = 0; it < 3000; ++it) {
        uint32_t primary[4] = {}, helper[4] = {};
        for (int e = 0; e < 128; ++e) if (uni(rng) < 0.6f) primary[e >> 5] |= 1u << (e & 31);
        rl::Masks m{};
        for (int w = 0; w < 4; ++w) m.ready[w] = m.resident[w] = primary[w];
        float lg[128], bias[128];
        for (int e = 0; e < 128; ++e) { lg[e] = logit(rng); bias[e] = 0.1f * (uni(rng) - 0.5f); }
        Kernel k;
        kernel_row(lg, bias, primary, helper, true, 0.0f, k);
        rl::Settings s;
        s.on = true;
        s.bonus = 0.1f + 0.5f * uni(rng);
        s.floor = 0.6f + 0.35f * uni(rng);
        rl::Row r;
        rl::route_pass(s, 5, 1, k.capture, m, nullptr, &r);
        const auto want = replay_floor(k.capture, m.ready, s.bonus, s.floor);
        same += executed(r) == std::set<int>(want.begin(), want.end());
        ++n;
    }
    CHECK(same == n);
    std::printf("floor vs the replay's reroute_floor: %ld of %ld rows the same set\n", same, n);
}

// ---- invariants over random settings and rows
static void test_invariants() {
    std::mt19937 rng(99);
    std::normal_distribution<float> logit(0.0f, 1.5f);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    long rows = 0, moved = 0, dropped = 0, widened = 0, layer_dropped = 0, floored = 0;
    for (int it = 0; it < 3000; ++it) {
        const int T = it % 2 ? 1 : 1 + int(uni(rng) * 16) % 16;
        rl::Settings s;
        s.on = true;
        s.bonus = uni(rng) < 0.2f ? 0.0f : uni(rng);
        if (uni(rng) < 0.3f) s.bonus_draft = uni(rng);
        s.floor = uni(rng) < 0.5f ? 0.0f : 0.5f + 0.5f * uni(rng);
        s.protect = int(uni(rng) * 3) % 3;
        s.priced = uni(rng) < 0.6f ? 0.0f : 4.0f * uni(rng);
        s.layer_drop = uni(rng) < 0.8f ? 0.0f : 3.0f + 0.9f * uni(rng);   // (top-four sigmoid sums run about 3 to 4)
        s.weights = int(uni(rng) * 3) % 3;
        s.k_drop = uni(rng) < 0.6f ? 0.0f : 0.3f * uni(rng);
        s.k_drop_ranks = 1 + int(uni(rng) * 2) % 2;
        s.k_widen = uni(rng) < 0.6f ? 0.0f : 0.5f * uni(rng);
        rl::Masks m{};
        const float q = uni(rng);
        for (int e = 0; e < 128; ++e) {
            if (uni(rng) < q) { rl::set_bit(m.ready, e); rl::set_bit(uni(rng) < 0.3f ? m.helper : m.resident, e); }
            else if (uni(rng) < 0.05f) rl::set_bit(m.in_flight, e);
        }
        std::vector<float> scores(size_t(T) * 384);
        std::vector<Kernel> K(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t) {
            float lg[128], bias[128];
            for (int e = 0; e < 128; ++e) { lg[e] = logit(rng); bias[e] = 0.1f * (uni(rng) - 0.5f); }
            kernel_row(lg, bias, m.ready, m.helper, true, 0.0f, K[size_t(t)]);
            std::memcpy(scores.data() + size_t(t) * 384, K[size_t(t)].capture, sizeof K[0].capture);
        }
        std::vector<rl::Row> out(static_cast<size_t>(T));
        rl::PassStats ps;
        rl::route_pass(s, 3 + it % 57, T, scores.data(), m, nullptr, out.data(), &ps);
        for (int t = 0; t < T; ++t) {
            const rl::Row& r = out[size_t(t)];
            const float* row = scores.data() + size_t(t) * 384;
            ++rows;
            // distinct valid ids, finite non-negative weights, skipped weight 0
            std::set<int> ids;
            bool valid = true, finite = true;
            for (int i = 0; i < r.slots; ++i) {
                valid &= r.ids[i] >= 0 && r.ids[i] < 128;
                ids.insert(r.ids[i]);
                finite &= std::isfinite(r.w[i]) && r.w[i] >= 0.0f && (!r.skip[i] || r.w[i] == 0.0f);
            }
            CHECK(valid && int(ids.size()) == r.slots);
            CHECK(finite);
            CHECK(r.slots == 4 || (r.slots == 5 && r.widened));
            // a rule reported as changing the row is on
            CHECK(!(r.changed & rl::kFloor) || s.floor > 0.0f);
            CHECK(!(r.changed & rl::kPriced) || s.priced > 0.0f);
            CHECK(!(r.changed & rl::kProtect) || s.protect > 0);
            CHECK(!(r.changed & rl::kLayerDrop) || s.layer_drop > 0.0f);
            CHECK(!(r.changed & rl::kKDrop) || (s.k_drop > 0.0f && r.dropped > 0));
            CHECK(!(r.changed & rl::kWiden) || (s.k_widen > 0.0f && r.widened));
            CHECK(!(r.changed & rl::kWeights) || s.weights != 0);
            CHECK(!(r.changed & rl::kBonus) || r.moved > 0);
            for (int i = 0; i < 4; ++i) CHECK(r.original[i] == K[size_t(t)].original[i]);
            // a dropped layer contributes nothing
            if (r.layer_dropped) {
                ++layer_dropped;
                CHECK(r.executed == 0 && executed(r).empty());
                continue;
            }
            const auto ex = executed(r);
            CHECK(int(ex.size()) == r.executed);
            // a protected rank is never replaced (and never dropped)
            for (int i = 0; i < s.protect; ++i) CHECK(ex.count(r.original[i]) == 1);
            // skipped picks are unready originals at the dropped ranks
            for (int i = 0; i < r.slots; ++i)
                if (r.skip[i]) {
                    const int e = r.ids[i];
                    int rank = -1;
                    for (int j = 0; j < 4; ++j) if (r.original[j] == e) rank = j;
                    CHECK(rank >= 4 - s.k_drop_ranks && rank >= s.protect && !rl::bit(m.ready, e));
                }
            // under the floor only without a substitution: the substituted mass (dropped picks added back) keeps it
            if (s.floor > 0.0f && r.moved > 0 && !r.widened) {
                double mass = 0.0, total = 0.0;
                for (int i = 0; i < 4; ++i) total += double(row[r.original[i] * 3]);
                for (int i = 0; i < r.slots; ++i) mass += double(row[r.ids[i] * 3]);
                CHECK(mass >= double(s.floor) * total - 1e-9);
                ++floored;
            }
            // an executed newcomer is ready; an executed unready pick is an original
            for (int e : ex) {
                bool orig = false;
                for (int j = 0; j < 4; ++j) orig |= r.original[j] == e;
                CHECK(orig || rl::bit(m.ready, e));
            }
            // the loads count the unready executed picks
            int loads = 0;
            for (int e : ex) loads += !rl::bit(m.ready, e);
            CHECK(loads == r.loads);
            moved += r.moved;
            dropped += r.dropped;
            widened += r.widened;
        }
    }
    CHECK(moved > 0 && dropped > 0 && widened > 0 && layer_dropped > 0 && floored > 0);
    std::printf("invariants: %ld rows (%ld moved picks, %ld dropped, %ld widened, %ld layers dropped, %ld floored rows)\n",
                rows, moved, dropped, widened, layer_dropped, floored);
}

// ---- stats, JSON, parsing
static void test_stats_and_parse() {
    Pass q = standard();
    for (int e = 2; e <= 10; ++e) q.ready(e);
    rl::Settings s;
    s.on = true;
    s.bonus = 1.0f;
    rl::PassStats ps;
    const auto out = q.run(s, 10, &ps);
    rl::RowStats rs;
    rs.clear(57);
    rs.add(out[0], 7, ps.est_ms, ps.loads);
    rs.add(out[0], 8, ps.est_ms, 3);
    CHECK(rs.layers == 2 && rs.picks == 8 && rs.moved == 4 && rs.kept_n == 2);
    CHECK(rs.layer_k[7] == 4 && rs.layer_moved[8] == 2);
    CHECK(rs.changed[0] == 2 && rs.changed[1] == 0);
    CHECK(rl::row_json(rs, 1).find("\"changed\":{\"bonus\":2,") != std::string::npos);
    const std::string j1 = rl::row_json(rs, 1), j2 = rl::row_json(rs, 2);
    CHECK(j1.find("\"mass_kept\":") != std::string::npos && j1.find("\"layers\":{") == std::string::npos);
    CHECK(j2.find("\"layers\":{\"kept\":[") != std::string::npos && j2.find("\"pass_loads\":[") != std::string::npos);
    CHECK(rs.layer_pass_loads[7] == ps.loads && rs.layer_pass_loads[8] == 3 && rs.layer_pass_loads[9] == 0);
    CHECK(j1.find(' ') == std::string::npos && j2.find(' ') == std::string::npos);   // (one token on the STAT line)
    rl::RequestStats rq;
    rq.add(out[0]);
    rq.add(out[0]);
    rq.add(ps);
    CHECK(rq.row_layers == 2 && rq.picks == 8 && rq.moved == 4 && rq.passes == 1);
    CHECK(rs.layer_staged[7] == out[0].staged);
    CHECK(rl::request_json(rq).find(' ') == std::string::npos && rl::settings_json(s).find(' ') == std::string::npos);
    std::vector<int> moe;
    for (int i = 3; i < 60; ++i) moe.push_back(i);
    std::vector<float> sc;
    std::string list;
    for (int i = 0; i < 57; ++i) list += (i ? "," : "") + std::string(i == 0 ? "0.5" : "1");
    CHECK(rl::parse_layer_scale(list, moe, 60, sc) && sc.size() == 60 && sc[3] == 0.5f && sc[4] == 1.0f && sc[0] == 1.0f);
    CHECK(!rl::parse_layer_scale("1,1,1", moe, 60, sc));
    CHECK(rl::parse_layer_scale("3-12:0.5,13-32:1.5", moe, 60, sc) && sc[12] == 0.5f && sc[13] == 1.5f && sc[33] == 1.0f);
    CHECK(rl::parse_layer_scale("40:2", moe, 60, sc) && sc[40] == 2.0f && sc[41] == 1.0f);
    CHECK(!rl::parse_layer_scale("3-12:17", moe, 60, sc) && !rl::parse_layer_scale("12-3:1", moe, 60, sc));
    CHECK(!rl::parse_layer_scale("a-b:1", moe, 60, sc) && !rl::parse_layer_scale("3-70:1", moe, 60, sc));
    float x = 0;
    CHECK(rl::parse_float("0.8", 0, 1, x) && x == 0.8f && !rl::parse_float("1.5", 0, 1, x) && !rl::parse_float("nan", 0, 1, x));
    CHECK(!rl::parse_float("0.5x", 0, 1, x) && !rl::parse_float("", 0, 1, x));
    rl::Settings d;
    CHECK(!d.any_rule() && !d.on && !d.prompt && d.stats == 1);
    rl::Settings e = d;
    e.prompt = true;
    CHECK(e != d && !e.any_rule());   // (a change of route_lab_prompt clears the reused state; it is no rule by itself)
    e = d;
    e.stats = 2;
    CHECK(e == d);                    // (route_stats does not)
    CHECK(rl::settings_json(s).find("\"route_lab_prompt\":0") != std::string::npos);
}

// ---- the serving kernel's per-row budget against the lab path's route_floor
// The kernel from its captured scores (the sigmoid p and p + bias of every expert), as its warp reductions decide: the
// router's eight by p + bias, the Prior bonus's four by that plus the bonus where ready, the budget through route_floor
// (the kernel's own source), and its weights (the executed set's p sum, or the original four's with dampen).
struct ServingRow {
    int ids[4];
    float w[4];
    int pairs = -1;   // pairs the budget applied (-1: no budget)
    int offered = -1; // pairs the bonus offered (route_floor's pairs_out: the serving counter counts a row with pairs < offered)
};
static ServingRow serving_row(const float* p, const float* base, const uint32_t* ready, float bonus, float floor, bool dampen) {
    ServingRow out;
    float score[128];
    for (int e = 0; e < 128; ++e) score[e] = base[e];
    int ranked[8];
    float ranked_prob[8], ranked_score[8];
    for (int rank = 0; rank < 8; ++rank) {
        float best = -INFINITY, bp = 0.0f;
        int bi = 0x7fffffff;
        for (int e = 0; e < 128; ++e) if (kbetter(score[e], e, best, bi)) { best = score[e]; bi = e; bp = p[e]; }
        ranked[rank] = bi; ranked_prob[rank] = bp; ranked_score[rank] = best;
        score[bi] = -INFINITY;
    }
    int sel[4] = {ranked[0], ranked[1], ranked[2], ranked[3]};
    float selp[4] = {ranked_prob[0], ranked_prob[1], ranked_prob[2], ranked_prob[3]};
    float sels[4] = {ranked_score[0], ranked_score[1], ranked_score[2], ranked_score[3]};
    if (bonus > 0.0f) {
        for (int e = 0; e < 128; ++e) score[e] = base[e] + (rl::bit(ready, e) ? bonus : 0.0f);
        for (int rank = 0; rank < 4; ++rank) {
            float best = -INFINITY, bp = 0.0f;
            int bi = 0x7fffffff;
            for (int e = 0; e < 128; ++e) if (kbetter(score[e], e, best, bi)) { best = score[e]; bi = e; bp = p[e]; }
            sel[rank] = bi; selp[rank] = bp; sels[rank] = best;
            score[bi] = -INFINITY;
        }
        if (floor > 0.0f) {
            float oadj[4];
            for (int r = 0; r < 4; ++r) oadj[r] = ranked_score[r] + (rl::bit(ready, ranked[r]) ? bonus : 0.0f);
            int ex[4];
            float pex[4];
            out.pairs = mm3::resident::route_floor(ranked, ranked_prob, oadj, sel, selp, sels, floor, ex, pex, &out.offered);
            for (int r = 0; r < 4; ++r) { sel[r] = ex[r]; selp[r] = pex[r]; }
        }
    }
    float sum = selp[0] + selp[1] + selp[2] + selp[3];
    if (dampen && bonus > 0.0f) sum = ranked_prob[0] + ranked_prob[1] + ranked_prob[2] + ranked_prob[3];
    for (int i = 0; i < 4; ++i) { out.ids[i] = sel[i]; out.w[i] = sum > 0.0f ? selp[i] / sum : 0.25f; }
    return out;
}
// the lab path on the same row: the kernel's capture ([128][3]: p, p + bias, plus the bonus where ready) through
// route_pass with route_floor and route_weights 0 or 1 (dampen), every other rule off
static rl::Row lab_row(const float* p, const float* base, const uint32_t* ready, float bonus, float floor, bool dampen) {
    std::vector<float> row(128 * 3);
    for (int e = 0; e < 128; ++e) {
        row[size_t(e) * 3] = p[e];
        row[size_t(e) * 3 + 1] = base[e];
        row[size_t(e) * 3 + 2] = bonus > 0.0f ? base[e] + (rl::bit(ready, e) ? bonus : 0.0f) : base[e];
    }
    rl::Settings s;
    s.on = true;
    s.bonus = bonus;
    s.floor = floor;
    s.weights = dampen ? 1 : 0;
    rl::Masks m{};
    for (int w = 0; w < 4; ++w) m.ready[w] = ready[w];
    rl::Row out;
    rl::route_pass(s, -1, 1, row.data(), m, nullptr, &out);
    return out;
}
struct FloorTally {
    long rows = 0, compared = 0, mismatch = 0, binding = 0, moved = 0;
    long counted = 0, withheld = 0, count_mismatch = 0;   // (the serving counter: rows it counts, substitutions withheld)
};
// one row at every budget and bonus: the serving kernel's four, order and weights against the lab's
static void floor_row(const float* p, const float* base, const uint32_t* ready, FloorTally& tally) {
    static const float floors[] = {0.0f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f, 1.0f};   // (budgets 1 down to 0)
    static const float bonuses[] = {0.0f, 0.1f, 0.2f, 0.3f, 1.0f};
    ++tally.rows;
    for (float bonus : bonuses)
        for (float floor : floors)
            for (int dampen = 0; dampen < 2; ++dampen) {
                const ServingRow k = serving_row(p, base, ready, bonus, floor, dampen != 0);
                const rl::Row lab = lab_row(p, base, ready, bonus, floor, dampen != 0);
                bool same = lab.slots == 4 && !lab.invalid;
                for (int i = 0; i < 4 && same; ++i)
                    same = !lab.skip[i] && lab.ids[i] == k.ids[i] && fbits(lab.w[i]) == fbits(k.w[i]);
                ++tally.compared;
                if (!same) {
                    if (tally.mismatch < 5)
                        std::fprintf(stderr, "floor mismatch: bonus %g floor %g dampen %d: kernel %d %d %d %d, lab %d %d %d %d\n",
                                     bonus, floor, dampen, k.ids[0], k.ids[1], k.ids[2], k.ids[3], lab.ids[0], lab.ids[1],
                                     lab.ids[2], lab.ids[3]);
                    ++tally.mismatch;
                }
                if (dampen == 0 && floor > 0.0f && bonus > 0.0f) {
                    tally.moved += lab.moved > 0;
                    tally.binding += (lab.changed & rl::kFloor) != 0;   // (the budget kept a pair out of this row)
                    // the serving counter (route_floor_rows): a row with fewer pairs applied than offered - the lab's
                    // kFloor, row for row; route_floor_withheld adds offered - applied
                    const bool limited = k.offered >= 0 && k.pairs < k.offered;
                    tally.counted += limited;
                    if (limited) tally.withheld += k.offered - k.pairs;
                    tally.count_mismatch += limited != ((lab.changed & rl::kFloor) != 0);
                }
            }
}
static std::string recorded_path(int argc, char** argv) {
    if (argc > 1) return argv[1];
    if (const char* v = std::getenv("MM3_ROUTER_RECORDED")) return v;
    const std::string here = __FILE__;   // (CMake passes the source's absolute path)
    return here.substr(0, here.find_last_of('/') + 1) + "data/mm3_router_recorded.bin";
}
static void test_serving_floor_recorded(int argc, char** argv) {
    const std::string path = recorded_path(argc, argv);
    std::FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr);   // (the recorded rows are part of the test: a missing file fails it)
    if (!f) { std::fprintf(stderr, "no recorded router rows at %s\n", path.c_str()); return; }
    char magic[8] = {};
    uint32_t n = 0;
    const bool header = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "MM3RRI1", 8) == 0 && std::fread(&n, 4, 1, f) == 1;
    CHECK(header && n > 0);
    FloorTally tally;
    std::vector<float> p(128), base(128);
    uint32_t primary[4], helper[4], ready[4];
    for (uint32_t r = 0; header && r < n; ++r) {
        if (std::fread(p.data(), 4, 128, f) != 128 || std::fread(base.data(), 4, 128, f) != 128 ||
            std::fread(primary, 4, 4, f) != 4 || std::fread(helper, 4, 4, f) != 4) { CHECK(false); break; }
        for (int w = 0; w < 4; ++w) ready[w] = primary[w] | helper[w];
        floor_row(p.data(), base.data(), ready, tally);
    }
    std::fclose(f);
    std::printf("serving per-row budget on %ld recorded rows (%s): %ld comparisons, %ld mismatches; the budget kept a "
                "pair out in %ld of %ld row-settings with a moved pick\n", tally.rows, path.c_str(), tally.compared,
                tally.mismatch, tally.binding, tally.moved);
    CHECK(tally.rows == long(n) && tally.mismatch == 0);
    CHECK(tally.binding > 0 && tally.binding < tally.moved);   // (the budget acts on these rows, and not on all of them)
    std::printf("serving budget counter on the recorded rows: %ld rows limited (%ld substitutions withheld), %ld differ from "
                "the lab's floor flag\n", tally.counted, tally.withheld, tally.count_mismatch);
    CHECK(tally.count_mismatch == 0 && tally.counted == tally.binding && tally.withheld >= tally.counted);
}
static void test_serving_floor_random() {
    std::mt19937 rng(20261112);
    std::normal_distribution<float> logit(-2.0f, 1.5f), bias(0.0f, 0.3f);
    FloorTally tally;
    std::vector<float> p(128), base(128);
    for (int row = 0; row < 3000; ++row) {
        const bool ties = row % 3 == 0;   // (coarse values: equal sigmoids and equal scores, the ties' rules)
        for (int e = 0; e < 128; ++e) {
            float x = logit(rng), b = bias(rng);
            if (ties) { x = std::round(x * 2.0f) / 2.0f; b = std::round(b * 4.0f) / 4.0f; }
            p[size_t(e)] = 1.0f / (1.0f + std::exp(-x));
            base[size_t(e)] = p[size_t(e)] + b;
        }
        uint32_t ready[4];
        for (int w = 0; w < 4; ++w) ready[w] = (uint32_t) rng() & (uint32_t) rng();   // (about a quarter ready)
        floor_row(p.data(), base.data(), ready, tally);
    }
    std::printf("serving per-row budget on %ld random rows: %ld comparisons, %ld mismatches\n", tally.rows, tally.compared,
                tally.mismatch);
    CHECK(tally.mismatch == 0 && tally.binding > 0);
    CHECK(tally.count_mismatch == 0 && tally.counted == tally.binding);   // (the serving counter, as on the recorded rows)
}

int main(int argc, char** argv) {
    test_serving_floor_recorded(argc, argv);
    test_serving_floor_random();
    test_exact_random();
    test_layer_drop();
    test_protect();
    test_floor();
    test_priced();
    test_k_drop();
    test_k_widen();
    test_weights();
    test_bonus_modes();
    test_floor_vs_replay();
    test_invariants();
    test_stats_and_parse();
    if (g_fail) {
        std::fprintf(stderr, "mm3_route_lab_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("mm3_route_lab_test: all checks passed\n");
    return 0;
}
