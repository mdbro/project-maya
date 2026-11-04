#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace mm3 {

// Approximate wall times in milliseconds. Per-expert costs include all of its
// columns and any <=8-column job chunks; the three service paths can overlap.
struct BatchStagingCosts {
    bool valid = false;
    std::array<double,17> cpu_extra_ms{}, gpu_extra_ms{}, helper_extra_ms{}, gpu_base_ms{};
    double cpu_base_ms = 0, helper_base_ms = 0, h2d_ms = 0, d2h_ms = 0;
};

struct BatchStagingPlan {
    bool valid = false;
    const char* status = "invalid input";
    std::vector<int> selected;
    double predicted_cpu_ms = 0, predicted_gpu_ms = 0, predicted_helper_ms = 0;
    double critical_before_ms = 0, critical_after_ms = 0;
};

// Each nonzero uses entry belongs to exactly one of CPU, GPU and helper.
// Candidates are a subset of CPU; selecting an expert moves its whole group to
// GPU, with one H2D charge, without changing any routed columns or weights.
inline BatchStagingPlan plan_batch_staging(const std::array<int,128>& uses,
        const std::vector<int>& cpu, const std::vector<int>& gpu, const std::vector<int>& helper,
        const std::vector<int>& candidates, int max_new, int rows, const BatchStagingCosts& costs) {
    BatchStagingPlan out;
    if(rows<1 || rows>16 || max_new<0) return out;
    if(!costs.valid) {out.status="costs unavailable";return out;}
    auto good=[](double x){return std::isfinite(x) && x>=0;};
    if(!good(costs.cpu_base_ms) || !good(costs.helper_base_ms) || !good(costs.h2d_ms) ||
       !good(costs.d2h_ms) || !good(costs.gpu_base_ms[size_t(rows)])) {
        out.status="invalid costs";return out;
    }
    for(int n=1;n<=rows;++n) if(!good(costs.cpu_extra_ms[size_t(n)]) ||
        !good(costs.gpu_extra_ms[size_t(n)]) || !good(costs.helper_extra_ms[size_t(n)])) {
        out.status="invalid costs";return out;
    }
    std::array<int,128> owner{};
    auto partition=[&](const std::vector<int>& ids,int kind) {
        for(int id:ids) {
            if(id<0 || id>=128 || owner[size_t(id)] || uses[size_t(id)]<1 || uses[size_t(id)]>rows) return false;
            owner[size_t(id)]=kind;
        }
        return true;
    };
    if(!partition(cpu,1) || !partition(gpu,2) || !partition(helper,3)) return out;
    for(size_t i=0;i<uses.size();++i)
        if(uses[i]<0 || uses[i]>rows || ((uses[i]!=0)!=(owner[i]!=0))) return out;
    std::array<bool,128> eligible{};
    for(int id:candidates) {
        if(id<0 || id>=128 || owner[size_t(id)]!=1 || eligible[size_t(id)]) return out;
        eligible[size_t(id)]=true;
    }
    // Sum in expert ID order so equivalent row and input-list permutations
    // produce the same floating-point decisions.
    auto evaluate=[&](const std::array<bool,128>& selected) {
        double c=0,g=costs.gpu_base_ms[size_t(rows)],h=0;
        int nc=0,nh=0,uploads=0;
        for(size_t id=0;id<uses.size();++id) if(owner[id]) {
            const size_t n=size_t(uses[id]);
            if(owner[id]==1 && !selected[id]) {++nc;c+=costs.cpu_extra_ms[n];}
            else if(owner[id]==2 || (owner[id]==1 && selected[id])) {
                g+=costs.gpu_extra_ms[n]; if(owner[id]==1)++uploads;
            } else if(owner[id]==3) {++nh;h+=costs.helper_extra_ms[n];}
        }
        if(nc)c+=costs.cpu_base_ms;
        if(nh)h+=costs.helper_base_ms;
        g+=uploads*costs.h2d_ms;
        return std::array<double,4>{c,g,h,std::max({c,g,h})};
    };
    std::array<bool,128> selected{};
    auto current=evaluate(selected);
    if(!std::isfinite(current[3])) {out.status="cost overflow";return out;}
    out.critical_before_ms=current[3];
    for(int step=0;step<max_new;++step) {
        int best=-1;std::array<double,4> best_values=current;
        for(int id=0;id<128;++id) if(eligible[size_t(id)] && !selected[size_t(id)]) {
            selected[size_t(id)]=true;
            const auto trial=evaluate(selected);
            selected[size_t(id)]=false;
            if(!std::isfinite(trial[3])) {out.status="cost overflow";return out;}
            const double tol=1e-12*std::max({1.0,std::abs(current[3]),std::abs(best_values[3]),std::abs(trial[3])});
            if(trial[3] < best_values[3]-tol) {best=id;best_values=trial;}
        }
        // Comparing against the current objective makes no addition win ties.
        if(best<0)break;
        selected[size_t(best)]=true;out.selected.push_back(best);current=best_values;
    }
    out.predicted_cpu_ms=current[0];out.predicted_gpu_ms=current[1];out.predicted_helper_ms=current[2];
    out.critical_after_ms=current[3];out.valid=true;out.status="ok";return out;
}
}
