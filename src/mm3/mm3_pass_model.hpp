#pragma once
// The bandwidth-aware pass model (plan step 6; pass_model, a per-request switch, off by default). A layer pass of the
// cooperative path from the plan's end (t = 0, ms): each lane's finish from its jobs' bytes and measured rates plus fixed
// start and chain costs, and the pass's finish, the later of the 5090 and the last host lane's publication. It replaces
// the CPU price (cooperative_cpu_scale, a multiplier on the startup calibration's CPU lane) where the switch is on.
//
//   copies  the 5090's uploads, one record each, chained on its copy engine from copy_start: alone at copy_gbps[path]
//           (path 0: demand copies in windows of 2+ rows, behind the pending exchanges' copies; path 1: one-row passes)
//   cpu     the host's CPU lane from cpu_start[rows]: cpu_fixed + experts x record / cpu_gbps + extra columns x
//           cpu_column of work alone. Copies and the CPU lane both read host memory: while both run, each progresses
//           at its share of its own rate (cpu_share, copy_share); the one left runs alone again.
//   5090    from the later of gpu_start[rows] (gpu_start_ready[rows] when it reads an expert an earlier copy staged)
//           and the end of the copies it waits for: gpu_base[rows] + experts x record / gpu_gbps + picks x gpu_pick +
//           extra rounds x gpu_round (rounds of gpu_rounds_of experts)
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
    double record_bytes = 31850752.0;
    std::array<double, 2> copy_gbps{54.0, 25.0};
    std::array<double, 2> copy_start{0.0, 0.0};
    double cpu_gbps = 56.5, cpu_fixed = 0.0, cpu_column = 0.02;
    std::array<double, kRows> cpu_start{};
    double cpu_share = 0.6, copy_share = 0.6;
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
    int exchanges = 0;         // the 5090's pending exchanges (their copies run ahead of a window's demand copies)
    int ready = 0;             // experts the 5090 reads from staging frames an earlier copy filled
    int gpu_experts = 0, gpu_picks = 0;
    int helper_experts = 0, helper_columns = 0, helper_lease = 0;
    int third_experts = 0, third_columns = 0;
    int gpu_rounds_of = 4;     // experts a round of the 5090's expert phase (4; 32 with expert_fast 2)
};

struct Finish {
    double cpu = kNone, copies = kNone, gpu = 0, helper = kNone, third = kNone, host = kNone, passed = 0;
};

inline double row_of(const std::array<double, kRows>& table, int rows) { return table[size_t(std::min(std::max(rows, 1), kRows - 1))]; }

// The CPU lane's and the copy chain's ends when each needs `work` ms alone and each progresses at its share while the
// other runs (the earlier one alone until the later one starts). A flow with no work ends at its begin.
inline std::pair<double, double> shared_ends(double cpu_begin, double cpu_work, double copy_begin, double copy_work,
                                             double cpu_share, double copy_share) {
    if (cpu_work <= 0 || copy_work <= 0)
        return {cpu_begin + std::max(cpu_work, 0.0), copy_begin + std::max(copy_work, 0.0)};
    const bool copy_first = copy_begin <= cpu_begin;
    const double first_work = copy_first ? copy_work : cpu_work, first_begin = copy_first ? copy_begin : cpu_begin,
                 first_share = copy_first ? copy_share : cpu_share;
    const double second_work = copy_first ? cpu_work : copy_work, second_begin = copy_first ? cpu_begin : copy_begin,
                 second_share = copy_first ? cpu_share : copy_share;
    const double alone = second_begin - first_begin;
    double first_end, second_end;
    if (first_work <= alone) {
        first_end = first_begin + first_work;
        second_end = second_begin + second_work;
    } else {
        const double first_left = first_work - alone;
        const double t_first = first_left / first_share, t_second = second_work / second_share;
        if (t_first <= t_second) {
            first_end = second_begin + t_first;
            second_end = first_end + (second_work - second_share * t_first);
        } else {
            second_end = second_begin + t_second;
            first_end = second_end + (first_left - first_share * t_second);
        }
    }
    return copy_first ? std::make_pair(second_end, first_end) : std::make_pair(first_end, second_end);
}

inline Finish predict(const Load& load, const Rates& r) {
    Finish f;
    const int path = load.rows == 1 ? 1 : 0;
    const double mb = r.record_bytes * 1e-6;   // (record MB; MB / GB/s = ms)
    const double cpu_work = load.cpu_experts ? (r.cpu_fixed + load.cpu_experts * mb / r.cpu_gbps +
                                                std::max(0, load.cpu_columns - load.cpu_experts) * r.cpu_column) : 0.0;
    const int chain = load.copies + (path == 0 ? load.exchanges : 0);
    const double copy_work = chain * mb / r.copy_gbps[size_t(path)];
    const auto ends = shared_ends(row_of(r.cpu_start, load.rows), cpu_work, r.copy_start[size_t(path)], copy_work, r.cpu_share, r.copy_share);
    const double ready_start = row_of(load.ready ? r.gpu_start_ready : r.gpu_start, load.rows);
    const double start = load.copies ? std::max(ready_start, ends.second) : ready_start;
    const int per = std::max(1, load.gpu_rounds_of);
    const int rounds = std::max(0, (load.gpu_experts + per - 1) / per - 1);
    f.gpu = start + row_of(r.gpu_base, load.rows) + load.gpu_experts * mb / r.gpu_gbps + load.gpu_picks * r.gpu_pick + rounds * r.gpu_round;
    if (load.cpu_experts) f.cpu = ends.first;
    if (chain) f.copies = ends.second;
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
// which of the CPU lane's eligible experts to upload to the 5090 this pass, greedily, each step the one that shortens the
// pass's predicted finish most, no step without a strict gain, at most max_new - but priced by predict() instead of the
// calibration's additive lanes with the CPU price. `base` holds the pass with every candidate on the CPU lane;
// candidates are (expert id, its columns) in expert id order.
struct StagingPlan {
    bool valid = false;
    const char* status = "invalid input";
    std::vector<int> selected;
    Finish before, after;
};

inline bool rates_valid(const Rates& r) {
    auto pos = [](double x) { return std::isfinite(x) && x > 0; };
    auto nonneg = [](double x) { return std::isfinite(x) && x >= 0; };
    if (!pos(r.record_bytes) || !pos(r.copy_gbps[0]) || !pos(r.copy_gbps[1]) || !pos(r.cpu_gbps) || !pos(r.gpu_gbps) ||
        !pos(r.helper_gbps) || !pos(r.third_gbps) || !pos(r.cpu_share) || r.cpu_share > 1 || !pos(r.copy_share) || r.copy_share > 1)
        return false;
    for (const double x : {r.copy_start[0], r.copy_start[1], r.cpu_fixed, r.cpu_column, r.gpu_pick, r.gpu_round, r.helper_column,
                           r.helper_lease, r.third_column})
        if (!nonneg(x)) return false;
    for (const auto* t : {&r.cpu_start, &r.gpu_start, &r.gpu_start_ready, &r.gpu_base, &r.helper_base, &r.third_base, &r.publish})
        for (int i = 1; i < kRows; ++i) if (!nonneg((*t)[size_t(i)])) return false;
    return true;
}

inline StagingPlan plan_staging(const Load& base, const std::vector<std::pair<int, int>>& candidates, int max_new, const Rates& r) {
    StagingPlan out;
    if (base.rows < 1 || base.rows > 16 || max_new < 0 || !rates_valid(r)) return out;
    int columns = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].second < 1 || (i && candidates[i].first <= candidates[i - 1].first)) return out;
        columns += candidates[i].second;
    }
    if (int(candidates.size()) > base.cpu_experts || columns > base.cpu_columns) return out;
    Load current = base;
    out.before = out.after = predict(current, r);
    std::vector<char> taken(candidates.size(), 0);
    for (int step = 0; step < max_new; ++step) {
        int best = -1;
        Finish best_f = out.after;
        Load best_load = current;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (taken[i]) continue;
            Load trial = current;
            trial.cpu_experts -= 1;
            trial.cpu_columns -= candidates[i].second;
            trial.copies += 1;
            trial.gpu_experts += 1;
            trial.gpu_picks += candidates[i].second;
            const Finish f = predict(trial, r);
            if (!std::isfinite(f.passed)) { out.status = "model overflow"; return out; }
            const double tol = 1e-12 * std::max({1.0, std::abs(out.after.passed), std::abs(best_f.passed), std::abs(f.passed)});
            if (f.passed < best_f.passed - tol) { best = int(i); best_f = f; best_load = trial; }
        }
        if (best < 0) break;   // (comparing against the current finish: no addition wins a tie)
        taken[size_t(best)] = 1;
        out.selected.push_back(candidates[size_t(best)].first);
        current = best_load;
        out.after = best_f;
    }
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
        {"record_bytes", &r.record_bytes, 1, false}, {"copy_gbps", r.copy_gbps.data(), 2, false}, {"copy_start", r.copy_start.data(), 2, false},
        {"cpu_gbps", &r.cpu_gbps, 1, false}, {"cpu_fixed", &r.cpu_fixed, 1, false}, {"cpu_column", &r.cpu_column, 1, false},
        {"cpu_start", r.cpu_start.data(), kRows, false}, {"cpu_share", &r.cpu_share, 1, false}, {"copy_share", &r.copy_share, 1, false},
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
