// ---- expert_graph (experimental, off unless a request asks): a generation window's expert phase on the 5090 - the
// plan's copy, the grouped expert kernels (setup, the FFN's launches, scatter), the wait for the host's publication of
// the CPU lane's and the 3090's rows, the combine and the residual add - captured once per window size and wait variant
// and replayed. Launched one by one, the host submits them more slowly than the 5090 runs them: a median 77 us a pass
// of the stream idle between them (traced, perpass-trace-new-20261109), and since helper_zero_copy the 5090's own path
// binds on about half the passes (evidence.md#after-zero-copy-bound).
// Nothing a replay takes changes from pass to pass. The plan's contents reach the device through the graph's first
// kernel, which reads its pinned buffer through a device view (a host copy node costs ~13 us on WSL2). The one value
// that does change, the wait word's expected value, rides in the plan's free tail, where the graph's wait reads it. The
// kernels take every expert, frame and weight from the plan in device memory, so no layer is captured: one graph per
// window size and wait variant (and arithmetic, device and buffers, which do not move within a load).
// Everything stateful stays on the host, in today's order, before the launch: the join, the retirements and exchanges,
// planning, demand uploads and frame waits, early_copy's issue and the 3090's submission. After it: the one query, the
// CPU lane, the collect and the publication.
// expert_graph 2 (verify, diagnostic): after the publication, the replay's combine output and residual are kept, the
// residual restored, the launches run on the same plan (their results are the layer's from there), and both compared
// bit for bit.
std::map<std::vector<long long>, cudaGraphExec_t> expert_graphs;
bool expert_graph_broken = false;             // (a capture failed: the launches from then on, this load)
static constexpr size_t kPlanWaitAt = 3600;   // (the plan's free tail, 8-byte aligned: 3,592 of win_plan_bytes()'s 3,648 used)
void expert_graphs_clear() {
    for (auto& kv : expert_graphs)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    expert_graphs.clear();
}
bool expert_graph_eligible(const Layer& l, int T, bool regular) const {
    return o.expert_graph > 0 && !lab.on && !expert_graph_broken && native && o.native_cooperative && regular && generation_active &&
           T >= 1 && T <= kWin && l.moe && !l.hf && l.lay.fmt == ne::kFmt && rseq.empty() && D.size() == 1 && !pf.on &&
           !replay_trace_active && !replay_fixed_batch && !capture::detailed() && !prefetch_active(regular) &&
           !predict_diag_on() && !o.cpu_first && helper_dispatch_current != "queue" && win_plan_bytes() >= kPlanWaitAt + 8;
}
// (events: the light capture's expert-phase events, recorded by the graph's event nodes; null without one - a graph
//  with nodes and one without are different graphs, so nothing changes with the capture off)
std::vector<long long> expert_graph_key(const Dev& d, int T, bool with_cpu, const void* compute_event = nullptr,
                                        const void* wait_event = nullptr) const {
    const auto at = [](const void* p) { return (long long) (uintptr_t) p; };
    ne::Arithmetic arithmetic{};
    (void) ne::get_arithmetic(d.s, arithmetic);
    return {(long long) d.id, (long long) T, with_cpu ? 1LL : 0LL, (long long) arithmetic, at(d.s), at(d.win_plan_h_dev),
            at(d.win_plan_d), at(d.xn), at(d.win_h), at(d.win_part), at(d.sh), at(d.out), at(d.h), at(d.win_cpu),
            at(compute_event), at(wait_event)};
}
// The verify buffers prep_graph_verify uses (the same sizes), made on first use.
bool graph_verify_buffers(Dev& d, std::string& err) {
    if (d.pv_save) return true;
    const size_t cap = (size_t) kWin * (3 * g.dim + 4 * g.k_exp) * 4;   // (a full window's rows, and a prediction)
    if (!dalloc(d.id, d.pv_save, (size_t) kWin * g.dim, err) || !dalloc(d.id, d.pv_a, cap, err) || !dalloc(d.id, d.pv_b, cap, err))
        return false;
    CK(cudaMallocHost(&d.pv_h, 2 * cap));
    return true;
}
// The expert phase through its graph: a replay, or on a key's first use the sequence captured and the new graph
// launched. `added` says whether the residual add ran inside (the caller then skips its own after the publication).
// A capture that fails ran nothing: the launches run instead, and the graphs stay off for the load.
bool expert_graph_run(Dev& d, int T, x3::WinArgs a, bool with_cpu, bool& added, std::string& err) {
    added = false;
    const auto launches = [&]() -> bool {   // today's sequence, the plan's copy included (the graph path skipped it)
        if (cudaMemcpyAsync(d.win_plan_d, d.win_plan_h, win_plan_bytes(), cudaMemcpyHostToDevice, d.s) != cudaSuccess) return false;
        return x3::win_experts(a, d.s);
    };
    if (!d.win_plan_h_dev && !expert_graph_broken) {
        void* view = nullptr;
        if (cudaHostGetDevicePointer(&view, d.win_plan_h, 0) == cudaSuccess) d.win_plan_h_dev = view;
        else {
            cudaGetLastError(); expert_graph_broken = true;
            warn("expert_graph", "expert_graph off: no device view of the plan's pinned buffer - the expert phases' kernels "
                                 "launched one by one for this load");
            std::fprintf(stderr, "maya-mm3: WARNING: %s\n", warnings.back().second.c_str());
        }
    }
    if (expert_graph_broken) return launches();
    x3::WinArgs ga = a;
    if (with_cpu) {
        *(volatile unsigned*) (d.win_plan_h + kPlanWaitAt) = a.wait_val;   // (the replay's wait reads it from the device plan)
        ga.wait_val_ptr = (const unsigned*) (d.win_plan_d + kPlanWaitAt);
    }
    const std::vector<long long> key = expert_graph_key(d, T, with_cpu, ga.diagnostic_compute_done, ga.diagnostic_wait_done);
    auto it = expert_graphs.find(key);
    if (it == expert_graphs.end()) {
        const double t0 = now();
        cudaGraph_t gr = nullptr;
        cudaGraphExec_t ex = nullptr;
        const bool began = cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
        bool ran = false;
        if (began) {
            k::copy_words(d.win_plan_h_dev, d.win_plan_d, win_plan_bytes(), d.s);
            ran = x3::win_experts(ga, d.s);
            if (ran) k::add(d.h, d.out, (size_t) T * g.dim, d.s);
        }
        const cudaError_t ended = began ? cudaStreamEndCapture(d.s, &gr) : cudaErrorUnknown;   // (the stream leaves capture)
        const bool ok = ran && ended == cudaSuccess && gr != nullptr && cudaGraphInstantiate(&ex, gr, 0) == cudaSuccess;
        if (gr) cudaGraphDestroy(gr);
        if (!ok) {
            if (ex) cudaGraphExecDestroy(ex);
            cudaGetLastError();
            expert_graph_broken = true;
            ++st.expert_graph_fallbacks;
            warn("expert_graph", "expert_graph off: an expert phase did not capture - its kernels launched one by one for the "
                                 "rest of this load");
            std::fprintf(stderr, "maya-mm3: WARNING: %s\n", warnings.back().second.c_str());
            return launches();
        }
        it = expert_graphs.emplace(key, ex).first;
        ++st.expert_graph_captures;
        st.expert_graph_capture_ms += 1000 * (now() - t0);
    }
    if (o.expert_graph == 2) {   // (verify: the residual before the replay, restored for the launches after the publication)
        if (!graph_verify_buffers(d, err)) return false;
        CK(cudaMemcpyAsync(d.pv_save, d.h, (size_t) T * g.dim * 4, cudaMemcpyDeviceToDevice, d.s));
    }
    CK(cudaGraphLaunch(it->second, d.s));
    ++st.expert_graph_launches;
    added = true;
    return true;
}
// expert_graph 2, after the publication: the replay's out and h kept, h restored, the launches run (the wait word already
// holds a.wait_val, so their wait passes at once; they read the same plan, frames and published rows), their out and h
// kept, both compared bit for bit. The launches' results stand.
bool expert_graph_verify(Layer& l, Dev& d, int T, const x3::WinArgs& a, std::string& err) {
    const size_t nd = (size_t) T * g.dim * 4;
    CK(cudaMemcpyAsync(d.pv_a, d.out, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaMemcpyAsync(d.pv_a + nd, d.h, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaMemcpyAsync(d.h, d.pv_save, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaMemcpyAsync(d.win_plan_d, d.win_plan_h, win_plan_bytes(), cudaMemcpyHostToDevice, d.s));
    x3::WinArgs va = a;   // (the replay's light-capture events keep the replay's times)
    va.diagnostic_compute_done = va.diagnostic_wait_done = nullptr;
    if (!x3::win_experts(va, d.s)) {err = "win_experts (expert_graph verify)"; return false;}
    k::add(d.h, d.out, (size_t) T * g.dim, d.s);
    CK(cudaMemcpyAsync(d.pv_b, d.out, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaMemcpyAsync(d.pv_b + nd, d.h, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaMemcpyAsync(d.pv_h, d.pv_a, 2 * nd, cudaMemcpyDeviceToHost, d.s));
    CK(cudaMemcpyAsync(d.pv_h + 2 * nd, d.pv_b, 2 * nd, cudaMemcpyDeviceToHost, d.s));
    CK(cudaStreamSynchronize(d.s));
    if (std::memcmp(d.pv_h, d.pv_h + 2 * nd, 2 * nd) == 0) {
        ++st.expert_graph_verified;
        return true;
    }
    ++st.expert_graph_mismatches;
    const bool out_same = std::memcmp(d.pv_h, d.pv_h + 2 * nd, nd) == 0;
    std::fprintf(stderr, "maya-mm3: expert_graph verify: layer %lld, %d rows at position %d: the replay's %s differ from the launches'\n",
                 (long long) (&l - L.data()), T, cur_pos, out_same ? "h" : "out");
    return true;
}
