// Included in Model::Impl. pass_model (plan step 6): the window's staging priced by the bandwidth-aware pass model
// (mm3_pass_model.hpp) instead of the startup calibration's additive lanes with the CPU price (plan_batch_staging).
//
// The pass as the planner sees it at its decision: the CPU lane's experts (cp) and their columns, the 5090's (gp: resident,
// or in a staging frame an earlier copy filled - "ready"), the 3090's (hp), the 4080's (counted before they leave the
// plan), the 5090's pending exchanges when they copy beside the pass (exchange_schedule 0), the one-row early copy, the
// 3090's lease, and the 5090's rounds
// (expert_fast 2: one round of up to 32 jobs). The result has plan_batch_staging's form; its predicted_* are the model's
// finishes in ms after the plan's end (the CPU lane's, the 5090's, the 3090's), critical_* the pass's predicted finish
// before and after the staging.
//
// It also keeps the model's per-use saving of an expert on the 5090 (pass_use_value): on each window it planned, how much
// sooner the pass would finish if one of its remaining CPU experts (one column), or one of its copies, were resident
// instead - 0 where neither binds - as a running average (weight 0.02 a sample). An upload's later uses from its frame
// (reuse[path] of them) are credited at that value in the staging's objective, and pass_model 2 prices the exchanges by it.
double pass_use_value() const {
    if (pass_service_samples >= 32) return pass_service_ms;
    const auto& r = pass_rates;   // (before 32 windows: the CPU lane's work for one expert beside an upload, less the 5090's)
    const double mb = r.record_bytes * 1e-6, share = std::min(1.0, r.host_gbps / (r.cpu_gbps + r.copy_gbps[0]));
    return std::max(0.0, mb / (r.cpu_gbps * share) - (mb / r.gpu_gbps + r.gpu_pick));
}
// Scheduled transactions can retain queued copies across a pass boundary. Count the queued prefix until its
// completion is observed, not zero merely because scheduling is enabled. Partial pieces have no individual event,
// so these are upper bounds; they intentionally do not include unissued future pieces.
std::array<double,2> pass_exchange_records(const Dev& d) const {
    std::array<double,2> out{};
    for (const auto& x : d.adaptive) if (x.incoming >= 0 && x.submitted) {
        if (!x.scheduled) {
            out[0] += double(x.submitted_h2d)/double(adaptive_blob);
            out[1] += double(x.submitted_d2h)/double(adaptive_blob); continue;
        }
        if (!x.h2d_done) out[0] += x.h2d_total ? double(x.h2d_sent) / double(adaptive_blob) : (x.h2d_issued ? 1.0 : 0.0);
        out[1] += double(x.d2h_done) / double(adaptive_blob);
    }
    return out;
}
BatchStagingPlan pass_model_staging(const Layer& l, const Dev& d, int T, const std::array<int,128>& uses,
                                    const std::vector<int>& cp, const std::vector<int>& gp, const std::vector<int>& hp,
                                    const std::vector<int>& cand, int maximum, int third_experts, int third_columns) {
    BatchStagingPlan out;
    pass_model::Load ld;
    ld.rows = T;
    for (int e : cp) { ++ld.cpu_experts; ld.cpu_columns += uses[size_t(e)]; }
    for (int e : gp) {
        ++ld.gpu_experts; ld.gpu_picks += uses[size_t(e)];
        if (l.res_slot[size_t(e)] < 0) {
            ++ld.ready;
            const long long key = (long long) (&l - L.data()) * 1024 + e;
            for (int slot=0; slot<d.n_slots; ++slot) if (d.slot_owner[size_t(slot)] == key && d.slot_foreground[size_t(slot)]) {
                ++ld.late_experts; ld.late_picks += uses[size_t(e)]; break;
            }
        }
    }
    for (int e : hp) { ++ld.helper_experts; ld.helper_columns += uses[size_t(e)]; }
    ld.third_experts = third_experts;
    ld.third_columns = third_columns;
    ld.exchanges = adaptive_pending(d);
    const auto traffic = pass_exchange_records(d);
    ld.exchange_up_records = traffic[0]; ld.exchange_down_records = traffic[1];
    ld.early = T == 1 && early_copy_on() && o.early_copy != 3 ? 1 : 0; // expected policy draw, not an issued-copy receipt
    ld.helper_lease = helper_live_a >= 0 ? 1 : 0;
    ld.gpu_rounds_of = ne::fast() >= 2 ? 32 : 4;
    ld.overlap = overlap_pass;
    std::vector<int> ids(cand);
    std::sort(ids.begin(), ids.end());
    std::vector<std::pair<int,int>> candidates;
    for (int e : ids) {
        if (std::find(cp.begin(), cp.end(), e) == cp.end()) { out.status = "a staging candidate is not on the CPU lane"; return out; }
        candidates.push_back({e, uses[size_t(e)]});
    }
    const auto plan = pass_model::plan_staging(ld, candidates, maximum, pass_rates, pass_use_value());
    if (!plan.valid) { out.status = plan.status; return out; }
    const auto finite = [](double x) { return std::isnan(x) ? 0.0 : x; };
    out.valid = true;
    out.status = "ok";
    out.selected = plan.selected;
    out.predicted_cpu_ms = finite(plan.after.cpu);
    out.predicted_gpu_ms = plan.after.gpu;
    out.predicted_helper_ms = finite(plan.after.helper);
    out.critical_before_ms = plan.before.passed;
    out.critical_after_ms = plan.after.passed;
    ++st.pass_model_passes;
    st.pass_model_staged += long(plan.selected.size());
    {
        pass_model::Load after = ld;
        for (int e : plan.selected) {
            after.cpu_experts -= 1; after.cpu_columns -= uses[size_t(e)];
            after.copies += 1; after.gpu_experts += 1; after.gpu_picks += uses[size_t(e)];
            after.late_experts += 1; after.late_picks += uses[size_t(e)];
        }
        const auto sample = [&](const pass_model::Load& x) {
            const double saving = std::max(0.0, plan.after.passed - pass_model::predict(x, pass_rates).passed);
            pass_service_ms = pass_service_samples ? 0.98 * pass_service_ms + 0.02 * saving : saving;
            ++pass_service_samples;
        };
        for (int e : cp) if (std::find(plan.selected.begin(), plan.selected.end(), e) == plan.selected.end()) {
            pass_model::Load x = after; // the entire grouped expert, not a fictitious one-column removal
            x.cpu_experts -= 1; x.cpu_columns -= uses[size_t(e)]; x.gpu_experts += 1; x.gpu_picks += uses[size_t(e)];
            sample(x); break; // one bounded representative sample, as before
        }
        if (after.copies) {        // (one copied expert resident instead: the copy gone, its compute stays)
            pass_model::Load x = after;
            x.copies -= 1;
            if (!plan.selected.empty()) {x.late_experts -= 1; x.late_picks -= uses[size_t(plan.selected.front())];}
            sample(x);
        }
    }
    return out;
}
