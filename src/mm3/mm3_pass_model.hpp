#pragma once
// The bandwidth-aware pass model (plan step 6; pass_model, a per-request switch, off by default). A layer pass of the
// cooperative path from the plan's end (t = 0, ms): each lane's finish from its jobs' bytes and measured rates plus fixed
// start and chain costs, and the pass's finish, the later of the 5090 and the last host lane's publication. It replaces
// the CPU price (cooperative_cpu_scale, a multiplier on the startup calibration's CPU lane) where the switch is on.
//
//   host    the CPU lane, the 5090's uploads, the exchanges' downloads and (one row) the early copy read or write host
//           memory: each has its rate alone and its draw; while the running flows would draw more than host_gbps, each
//           runs slower in proportion (host_flows)
//   uploads one record each, chained on the 5090's upload link from copy_start[path]: in windows (path 0) behind the
//           unscheduled pending exchanges' uploads (exchange_up records each) at copy_gbps[0]; at one row (path 1) at
//           copy_gbps[1], the link shared with the early copy
//   downs   the unscheduled pending exchanges' victims, exchange_down records each, at d2h_gbps
//   early   at one row with the early copy on, early_draw GB/s of host memory all pass long
//   cpu     the host's CPU lane from cpu_start[rows]: cpu_fixed + experts x record / cpu_gbps + extra columns x
//           cpu_column of work alone
//   5090    ready groups can run beside uploads. A split pass ends at max(ready-group finish, upload finish) plus
//           late-group work; a non-split pass keeps its single-round dependency. Each group run pays its setup cost.
//   3090    helper_base[rows] + experts x record / helper_gbps + columns x helper_column (+ helper_lease with a lease)
//   4080    third_base[rows] + experts x record / third_gbps + columns x third_column
//   pass    max(5090, last host lane + publish[rows])
//
// Rates are GB/s of record bytes; times ms. The constants come from tools/mm3/pass_model/fit_pass_model.py (fitted on
// light captures; mm3_pass_model_fit.hpp holds the fitted set and golden cases); tools/mm3/pass_model/pass_model.py is
// the same model in Python, term for term, for fitting and checking. Rows index 1..16.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mm3::pass_model {

constexpr int kRows = 17;
constexpr double kNone = std::numeric_limits<double>::quiet_NaN();   // (a lane with no job)

struct Rates {
    double model_version = 2.0; // split-ready/late model; older external rate files must be refitted
    double record_bytes = 31850752.0;
    std::array<double, 2> copy_gbps{54.0, 25.0};
    std::array<double, 2> copy_start{0.0, 0.0};
    double cpu_gbps = 56.5, cpu_fixed = 0.0, cpu_column = 0.02;
    std::array<double, kRows> cpu_start{};
    double host_gbps = 80.0, d2h_gbps = 53.0, exchange_up = 0.25, exchange_down = 1.0, early_draw = 0.0;
    std::array<double, 2> reuse{0.19, 0.38};   // (uses from its frame within 8 forwards an upload buys, by path)
    double gpu_gbps = 1679.0, gpu_pick = 0.009, gpu_round = 0.033;
    std::array<double, kRows> gpu_start{}, gpu_start_ready{}, gpu_base{};
    double helper_gbps = 900.0, helper_column = 0.005, helper_lease = 0.0;
    std::array<double, kRows> helper_base{};
    double third_gbps = 700.0, third_column = 0.005;
    std::array<double, kRows> third_base{}, publish{};
};

struct Load {
    int rows = 1;
    int cpu_experts = 0, cpu_columns = 0;
    int copies = 0;            // records the 5090 waits for in this pass (uploaded now)
    int exchanges = 0;         // the 5090's pending exchanges whose copies run during the pass (exchange_schedule 0)
    int ready = 0;             // experts the 5090 reads from staging frames an earlier copy filled
    int early = 0;             // 1: the early copy runs this pass (one row, early_copy on)
    int gpu_experts = 0, gpu_picks = 0;
    int helper_experts = 0, helper_columns = 0, helper_lease = 0;
    int third_experts = 0, third_columns = 0;
    int gpu_rounds_of = 4;     // experts a round of the 5090's expert phase (4; 32 with expert_fast 2)
    bool overlap = false;     // actual eligible route, not merely the request's switch
    int late_experts = 0, late_picks = 0; // groups whose demand upload has not retired
    // Explicit queued exchange traffic, in records. Negative means a historical capture has only pending-count
    // evidence and uses the fitted aggregate estimate. Scheduled work is never zeroed just because of its flag.
    double exchange_up_records = -1, exchange_down_records = -1;
    // Signed start offset relative to plan end. A copy submitted while planning can start before that origin.
    // NaN uses the fitted offset; an observed offset is useful for isolating rate error from submission error.
    double copy_begin = kNone;
};

struct Finish {
    double cpu = kNone, copies = kNone, gpu = 0, helper = kNone, third = kNone, host = kNone, passed = 0;
};

inline double row_of(const std::array<double, kRows>& table, int rows) { return table[size_t(std::min(std::max(rows, 1), kRows - 1))]; }

// Flows sharing host memory: each (begin ms, work MB, rate alone GB/s, host memory drawn alone GB/s). While the running
// flows would draw more than host_gbps, each runs at its rate times host_gbps / their total draw. Returns each flow's end
// (ms); a flow with no work ends at its begin. (pass_model.py host_flows, step for step)
struct Flow { double begin, work, rate, draw; };
template <size_t N>
inline std::array<double, N> host_flows(const std::array<Flow, N>& f, double host_gbps) {
    std::array<double, N> ends{}, left{};
    std::array<bool, N> done{};
    for (size_t i = 0; i < N; ++i) {
        left[i] = f[i].work;
        if (f[i].work <= 0) { ends[i] = f[i].begin; done[i] = true; }
    }
    double t = 0;
    bool any = false;
    for (size_t i = 0; i < N; ++i)
        if (!done[i]) { t = any ? std::min(t, f[i].begin) : f[i].begin; any = true; }
    for (;;) {
        bool all = true, active = false, later = false;
        double draw = 0, next = 0;
        for (size_t i = 0; i < N; ++i) {
            if (done[i]) continue;
            all = false;
            if (f[i].begin <= t) { active = true; draw += f[i].draw; }
            else { next = later ? std::min(next, f[i].begin) : f[i].begin; later = true; }
        }
        if (all) break;
        if (!active) { t = next; continue; }
        const double scale = std::min(1.0, host_gbps / draw);
        double step = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < N; ++i)
            if (!done[i] && f[i].begin <= t) step = std::min(step, left[i] / (f[i].rate * scale));
        if (later) step = std::min(step, next - t);
        for (size_t i = 0; i < N; ++i) {
            if (done[i] || f[i].begin > t) continue;
            const double rate = f[i].rate * scale;
            if (left[i] / rate <= step) { left[i] = 0; done[i] = true; ends[i] = t + step; }
            else left[i] -= rate * step;
        }
        t += step;
    }
    return ends;
}

inline Finish predict(const Load& load, const Rates& r) {
    Finish f;
    const int path = load.rows == 1 ? 1 : 0;
    const double mb = r.record_bytes * 1e-6;   // (record MB; MB / GB/s = ms)
    const double cpu_ms = load.cpu_experts ? (r.cpu_fixed + load.cpu_experts * mb / r.cpu_gbps +
                                              std::max(0, load.cpu_columns - load.cpu_experts) * r.cpu_column) : 0.0;
    const double exchange_up = load.exchange_up_records >= 0 ? load.exchange_up_records * r.exchange_up :
                               (path == 0 ? load.exchanges * r.exchange_up : 0.0);
    const double exchange_down = load.exchange_down_records >= 0 ? load.exchange_down_records * r.exchange_down :
                                 load.exchanges * r.exchange_down;
    const double up = (load.copies + exchange_up) * mb;
    const double down = exchange_down * mb;
    const double early = (load.early && path == 1) ? r.early_draw * 1000.0 : 0.0;   // (a flow that outlasts the pass)
    const double begin = std::isfinite(load.copy_begin) ? load.copy_begin : r.copy_start[size_t(path)];
    const double rate = r.copy_gbps[size_t(path)];
    const auto ends = host_flows<4>({{{row_of(r.cpu_start, load.rows), cpu_ms * r.cpu_gbps, r.cpu_gbps, r.cpu_gbps},
                                      {begin, up, rate, rate},
                                      {begin, down, r.d2h_gbps, r.d2h_gbps},
                                      {begin, early, r.early_draw, r.early_draw}}}, r.host_gbps);
    const double ready_start = row_of(load.ready ? r.gpu_start_ready : r.gpu_start, load.rows);
    const int per = std::max(1, load.gpu_rounds_of);
    const auto compute = [&](int experts, int picks) {
        const int rounds = std::max(0, (experts + per - 1) / per - 1);
        return row_of(r.gpu_base, load.rows) + experts * mb / r.gpu_gbps + picks * r.gpu_pick + rounds * r.gpu_round;
    };
    const int late = std::clamp(load.late_experts, 0, load.gpu_experts);
    const int late_picks = std::clamp(load.late_picks, 0, load.gpu_picks);
    if (load.overlap && late > 0 && late < load.gpu_experts) {
        const double ready_done = ready_start + compute(load.gpu_experts - late, load.gpu_picks - late_picks);
        f.gpu = std::max(ready_done, ends[1]) + compute(late, late_picks);
    } else {
        const double start = load.copies ? std::max(ready_start, ends[1]) : ready_start;
        f.gpu = start + compute(load.gpu_experts, load.gpu_picks);
    }
    if (load.cpu_experts) f.cpu = ends[0];
    if (up > 0) f.copies = ends[1];
    if (load.helper_experts)
        f.helper = row_of(r.helper_base, load.rows) + load.helper_experts * mb / r.helper_gbps + load.helper_columns * r.helper_column +
                   (load.helper_lease ? r.helper_lease : 0.0);
    if (load.third_experts)
        f.third = row_of(r.third_base, load.rows) + load.third_experts * mb / r.third_gbps + load.third_columns * r.third_column;
    bool any = false;
    double host = 0;
    for (const double x : {f.cpu, f.helper, f.third})
        if (!std::isnan(x)) { host = any ? std::max(host, x) : x; any = true; }
    if (any) f.host = host;
    f.passed = any ? std::max(f.gpu, host + row_of(r.publish, load.rows)) : f.gpu;
    return f;
}

// The window's staging by the model (pass_model 1): the same choice as plan_batch_staging (mm3_batch_staging.hpp) -
// which of the CPU lane's eligible experts to upload to the 5090 this pass, greedily, each step the one that lowers the
// objective most, no step without a strict gain, at most max_new - but priced by predict() instead of the calibration's
// additive lanes with the CPU price. The objective: the pass's predicted finish less what the uploads buy later, each
// reuse[path] uses from its frame worth use_value ms (the saving of one use by an expert on the 5090 instead of the CPU
// lane or a copy). `base` holds the pass with every candidate on the CPU lane; candidates are (expert id, its columns)
// in expert id order.
struct StagingPlan {
    bool valid = false;
    const char* status = "invalid input";
    std::vector<int> selected;
    Finish before, after;
    double credit = 0;   // (ms: the later uses the selected uploads buy, as the objective counted them)
};

inline bool rates_valid(const Rates& r) {
    auto pos = [](double x) { return std::isfinite(x) && x > 0; };
    auto nonneg = [](double x) { return std::isfinite(x) && x >= 0; };
    if (r.model_version != 2.0 || !pos(r.record_bytes) || !pos(r.copy_gbps[0]) || !pos(r.copy_gbps[1]) || !pos(r.cpu_gbps) || !pos(r.gpu_gbps) ||
        !pos(r.helper_gbps) || !pos(r.third_gbps) || !pos(r.host_gbps) || !pos(r.d2h_gbps))
        return false;
    if (!std::isfinite(r.copy_start[0]) || !std::isfinite(r.copy_start[1])) return false;
    for (const double x : {r.cpu_fixed, r.cpu_column, r.gpu_pick, r.gpu_round, r.helper_column,
                           r.helper_lease, r.third_column, r.exchange_up, r.exchange_down, r.early_draw, r.reuse[0], r.reuse[1]})
        if (!nonneg(x)) return false;
    for (const auto* t : {&r.cpu_start, &r.gpu_start, &r.gpu_start_ready, &r.gpu_base, &r.helper_base, &r.third_base, &r.publish})
        for (int i = 1; i < kRows; ++i) if (!nonneg((*t)[size_t(i)])) return false;
    return true;
}

inline StagingPlan plan_staging(const Load& base, const std::vector<std::pair<int, int>>& candidates, int max_new, const Rates& r,
                                double use_value = 0.0) {
    StagingPlan out;
    if (base.rows < 1 || base.rows > 16 || max_new < 0 || !rates_valid(r) || !std::isfinite(use_value) || use_value < 0) return out;
    const double each = r.reuse[base.rows == 1 ? 1 : 0] * use_value;   // (the objective's credit an upload)
    int columns = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].second < 1 || (i && candidates[i].first <= candidates[i - 1].first)) return out;
        columns += candidates[i].second;
    }
    if (int(candidates.size()) > base.cpu_experts || columns > base.cpu_columns) return out;
    Load current = base;
    out.before = out.after = predict(current, r);
    double objective = out.before.passed;
    std::vector<char> taken(candidates.size(), 0);
    for (int step = 0; step < max_new; ++step) {
        int best = -1;
        Finish best_f = out.after;
        double best_objective = objective;
        Load best_load = current;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (taken[i]) continue;
            Load trial = current;
            trial.cpu_experts -= 1;
            trial.cpu_columns -= candidates[i].second;
            trial.copies += 1;
            trial.gpu_experts += 1;
            trial.gpu_picks += candidates[i].second;
            trial.late_experts += 1;
            trial.late_picks += candidates[i].second;
            const Finish f = predict(trial, r);
            if (!std::isfinite(f.passed)) { out.status = "model overflow"; return out; }
            const double value = f.passed - double(out.selected.size() + 1) * each;
            const double tol = 1e-12 * std::max({1.0, std::abs(objective), std::abs(best_objective), std::abs(value)});
            if (value < best_objective - tol) { best = int(i); best_f = f; best_objective = value; best_load = trial; }
        }
        if (best < 0) break;   // (comparing against the current objective: no addition wins a tie)
        taken[size_t(best)] = 1;
        out.selected.push_back(candidates[size_t(best)].first);
        current = best_load;
        out.after = best_f;
        objective = best_objective;
    }
    out.credit = double(out.selected.size()) * each;
    out.valid = true;
    out.status = "ok";
    return out;
}

// The model's constants from a fit file (fit_pass_model.py --text): one line a constant, its name and its value(s),
// '#' comments. Every constant of Rates must appear once; the result is checked by rates_valid.
inline bool load_rates(const std::string& path, Rates& out, std::string& err) {
    std::ifstream in(path);
    if (!in) { err = "cannot open " + path; return false; }
    Rates r;
    struct Field { const char* name; double* at; size_t n; bool seen; };
    std::vector<Field> fields = {
        {"model_version", &r.model_version, 1, false},
        {"record_bytes", &r.record_bytes, 1, false}, {"copy_gbps", r.copy_gbps.data(), 2, false}, {"copy_start", r.copy_start.data(), 2, false},
        {"cpu_gbps", &r.cpu_gbps, 1, false}, {"cpu_fixed", &r.cpu_fixed, 1, false}, {"cpu_column", &r.cpu_column, 1, false},
        {"cpu_start", r.cpu_start.data(), kRows, false}, {"host_gbps", &r.host_gbps, 1, false}, {"d2h_gbps", &r.d2h_gbps, 1, false},
        {"exchange_up", &r.exchange_up, 1, false}, {"exchange_down", &r.exchange_down, 1, false}, {"early_draw", &r.early_draw, 1, false},
        {"reuse", r.reuse.data(), 2, false},
        {"gpu_gbps", &r.gpu_gbps, 1, false}, {"gpu_pick", &r.gpu_pick, 1, false}, {"gpu_round", &r.gpu_round, 1, false},
        {"gpu_start", r.gpu_start.data(), kRows, false}, {"gpu_start_ready", r.gpu_start_ready.data(), kRows, false},
        {"gpu_base", r.gpu_base.data(), kRows, false}, {"helper_gbps", &r.helper_gbps, 1, false}, {"helper_column", &r.helper_column, 1, false},
        {"helper_lease", &r.helper_lease, 1, false}, {"helper_base", r.helper_base.data(), kRows, false},
        {"third_gbps", &r.third_gbps, 1, false}, {"third_column", &r.third_column, 1, false}, {"third_base", r.third_base.data(), kRows, false},
        {"publish", r.publish.data(), kRows, false}};
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        ++number;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream s(line);
        std::string name;
        if (!(s >> name)) continue;
        auto it = std::find_if(fields.begin(), fields.end(), [&](const Field& f) { return name == f.name; });
        if (it == fields.end() || it->seen) { err = path + ":" + std::to_string(number) + ": unknown or repeated constant " + name; return false; }
        for (size_t i = 0; i < it->n; ++i)
            if (!(s >> it->at[i])) { err = path + ":" + std::to_string(number) + ": " + name + " needs " + std::to_string(it->n) + " values"; return false; }
        std::string extra;
        if (s >> extra) { err = path + ":" + std::to_string(number) + ": too many values for " + name; return false; }
        it->seen = true;
    }
    for (const auto& f : fields)
        if (!f.seen) { err = path + ": constant " + std::string(f.name) + " missing"; return false; }
    if (!rates_valid(r)) { err = path + ": constants out of range"; return false; }
    out = r;
    return true;
}

}  // namespace mm3::pass_model
