// The bandwidth-aware pass model (src/mm3/mm3_pass_model.hpp): the fitted constants reproduce the Python model's golden
// cases (tools/mm3/pass_model/fit_pass_model.py writes both), the shared host-memory flows behave, the staging planner
// stages only for a strict gain, and a constants file round-trips.
#include "../src/mm3/mm3_pass_model.hpp"
#include "../src/mm3/mm3_pass_model_fit.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

using namespace mm3::pass_model;

static bool same(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
    return std::abs(a - b) <= 1e-9 * std::max(1.0, std::abs(b));
}

int main() {
    const Rates r = fitted_rates();
    assert(rates_valid(r));
    size_t n = 0;
    const GoldenCase* cases = golden_cases(n);
    assert(n >= 6);
    for (size_t i = 0; i < n; ++i) {
        const Finish f = predict(cases[i].load, r);
        if (!same(f.cpu, cases[i].cpu) || !same(f.copies, cases[i].copies) || !same(f.gpu, cases[i].gpu) ||
            !same(f.helper, cases[i].helper) || !same(f.third, cases[i].third) || !same(f.passed, cases[i].passed)) {
            std::fprintf(stderr, "golden case %zu: C++ %.12g %.12g %.12g %.12g %.12g %.12g\n", i, f.cpu, f.copies, f.gpu, f.helper, f.third, f.passed);
            return 1;
        }
    }
    // shared flows: alone when the other has no work; the CPU slows only while copies run; symmetric in the order of starts
    {
        auto e = shared_ends(0.04, 0.5, 0.0, 0.0, 0.6, 0.7);
        assert(same(e.first, 0.54) && same(e.second, 0.0));
        e = shared_ends(0.0, 0.5, 0.0, 0.3, 0.5, 0.5);   // copy done at 0.6 (CPU at 0.3 of 0.5), CPU alone 0.2 more
        assert(same(e.second, 0.6) && same(e.first, 0.8));
        e = shared_ends(0.1, 0.2, 0.0, 1.0, 0.5, 0.5);   // copy alone 0.1, then CPU done at 0.1 + 0.4; copy rest alone
        assert(same(e.first, 0.5) && same(e.second, 0.5 + (0.9 - 0.2)));
        const auto a = shared_ends(0.0, 0.7, 0.2, 0.4, 0.6, 0.8), b = shared_ends(0.2, 0.4, 0.0, 0.7, 0.8, 0.6);
        assert(same(a.first, b.second) && same(a.second, b.first));
    }
    // staging: a CPU lane far behind is relieved; nothing is staged when another lane sets the finish; max_new holds
    {
        Rates t = r;
        Load base; base.rows = 4; base.cpu_experts = 3; base.cpu_columns = 4; base.gpu_experts = 2; base.gpu_picks = 3;
        base.helper_experts = 2; base.helper_columns = 3;
        const auto p = plan_staging(base, {{5, 1}, {9, 2}, {11, 1}}, 4, t);
        assert(p.valid && !p.selected.empty() && p.after.passed < p.before.passed);
        assert(p.selected.size() <= 3);
        const auto one = plan_staging(base, {{5, 1}, {9, 2}, {11, 1}}, 1, t);
        assert(one.valid && one.selected.size() == 1);
        Load slow = base; slow.third_experts = 40; slow.third_columns = 40;   // (the 4080 far behind everything)
        const auto q = plan_staging(slow, {{5, 1}, {9, 2}, {11, 1}}, 4, t);
        assert(q.valid && q.selected.empty() && same(q.after.passed, q.before.passed));
        assert(!plan_staging(base, {{9, 2}, {5, 1}}, 2, t).valid);            // (not in id order)
        assert(!plan_staging(base, {{5, 1}, {9, 2}, {11, 1}, {12, 1}}, 2, t).valid);   // (more candidates than CPU experts)
        Rates bad = t; bad.cpu_share = 0;
        assert(!plan_staging(base, {{5, 1}}, 1, bad).valid);
    }
    // a constants file: written in the fit script's format, read back exactly; a missing constant is refused
    {
        const std::string path = "mm3_pass_model_test_rates.txt";
        {
            std::ofstream o(path);
            auto line = [&](const char* name, const double* v, size_t k) {
                o << name; char buf[64];
                for (size_t i = 0; i < k; ++i) { std::snprintf(buf, sizeof buf, " %.17g", v[i]); o << buf; }
                o << "\n";
            };
            o << "# test\n";
            line("record_bytes", &r.record_bytes, 1); line("copy_gbps", r.copy_gbps.data(), 2); line("copy_start", r.copy_start.data(), 2);
            line("cpu_gbps", &r.cpu_gbps, 1); line("cpu_fixed", &r.cpu_fixed, 1); line("cpu_column", &r.cpu_column, 1);
            line("cpu_start", r.cpu_start.data(), kRows); line("cpu_share", &r.cpu_share, 1); line("copy_share", &r.copy_share, 1);
            line("gpu_gbps", &r.gpu_gbps, 1); line("gpu_pick", &r.gpu_pick, 1); line("gpu_round", &r.gpu_round, 1);
            line("gpu_start", r.gpu_start.data(), kRows); line("gpu_start_ready", r.gpu_start_ready.data(), kRows);
            line("gpu_base", r.gpu_base.data(), kRows); line("helper_gbps", &r.helper_gbps, 1); line("helper_column", &r.helper_column, 1);
            line("helper_lease", &r.helper_lease, 1); line("helper_base", r.helper_base.data(), kRows);
            line("third_gbps", &r.third_gbps, 1); line("third_column", &r.third_column, 1); line("third_base", r.third_base.data(), kRows);
            line("publish", r.publish.data(), kRows);
        }
        Rates back; std::string err;
        assert(load_rates(path, back, err));
        for (size_t i = 0; i < n; ++i) assert(same(predict(cases[i].load, back).passed, cases[i].passed));
        {
            std::ofstream o(path);
            o << "record_bytes 31850752\n";
        }
        assert(!load_rates(path, back, err) && err.find("missing") != std::string::npos);
        std::remove(path.c_str());
    }
    std::puts("mm3_pass_model_test: ok");
    return 0;
}
