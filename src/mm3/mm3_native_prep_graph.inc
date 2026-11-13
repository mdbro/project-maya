// ---- prep_graph (experimental, off unless a request asks): an MoE layer's window preparation - one row at T=1, or a
// drafting window's rows; its attention, the router's and the shared expert's work on both streams, and the readbacks to
// the host - captured once per window size as a CUDA graph
// and replayed: the same kernels with the same arguments, so the same bits, without the ~22 launches and four
// cross-stream handoffs the preparation costs a layer (2026-11-08, HAGS on: the 5090 spends ~68 of its ~285 us of
// preparation between kernels). The host's part stays on the host: before each replay the ready mask is built exactly
// as route_layer builds it, and the graph's copy reads it. A graph is keyed by every captured argument that can move:
// the layer, the indexer's launch shape (the load's most blocks: the kernels skip the blocks past each query's own, so
// one graph serves every position), the residual buffer, the routing policy and its parameter, whether the CPU lane's
// rows are read back, early_copy's prediction, the caches' and the side stream's addresses, and whether --ablate's
// projection of the attention's write is on (window_pre's ablate(); never, without a direction file). Everything else the
// launches take is fixed at load; the positions are read from d.pos.
// prep_graph 2 (verify, diagnostic): every replay checked against the launches - the residual saved, the graph replayed
// and its results kept, the residual restored, the launches run (theirs are the layer's results from there), and both
// results compared bit for bit.
std::map<std::vector<long long>, cudaGraphExec_t> prep_graphs;
bool prep_graph_broken = false;   // (a capture failed: the launches from then on, this load)
// While a preparation is captured, its five small host copies (the ready mask up; the routing, the original ids and the
// CPU lane's rows down) are kernels on device views of the same pinned buffers: in a graph, host copy nodes cost ~13 us
// each on WSL2 and undid the boundaries the graph saves (prep_shape_bench, 2026-11-08). The host still reads after the
// stream's synchronization, so nothing it sees changes.
bool prep_zero_copy = false;
bool prep_graph_views(Dev& d) {
    if (d.ready_h_dev && d.original_h_dev && d.win_rb_dev) return true;
    void *a = nullptr, *b = nullptr, *c = nullptr;
    if (cudaHostGetDevicePointer(&a, d.ready_h, 0) != cudaSuccess || cudaHostGetDevicePointer(&b, d.original_h, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(&c, d.win_rb, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    d.ready_h_dev = a;
    d.original_h_dev = b;
    d.win_rb_dev = c;
    return true;
}
void prep_graphs_clear() {
    for (auto& kv : prep_graphs)
        if (kv.second) cudaGraphExecDestroy(kv.second);
    prep_graphs.clear();
    expert_graphs_clear();   // (every change that drops these - arithmetic, dispatch, dense_split, reload - drops those)
}
bool prep_graph_eligible(const Layer& l, int T, bool regular) {
    // (generation windows only: one row, or a drafting window's rows; a prompt's chunks launch as before)
    return o.prep_graph > 0 && !lab.on && !prep_graph_broken && native && regular && generation_active && T >= 1 && T <= kWin && l.moe &&
           !l.hf && rseq.empty() &&
           D.size() == 1 && l.kvh.ring == 0 && !pf.on && !replay_trace_active && !capture::detailed() &&
           !prefetch_active(regular) && !predict_diag_on() && !(early_copy_on() && o.early_copy_legacy) &&
           helper_dispatch_current != "queue" && !abl_capture_now();
}
// early_copy's prediction (the next layer's router on this layer's input, its top-k, its ids to the host) is part of a
// T=1 preparation whenever the early copy is on, in its lean form, and a next MoE layer on this device exists.
bool early_copy_in_graph(const Layer& l, int T) const {
    const size_t li = size_t(&l - L.data());
    return early_copy_on() && T == 1 && !o.early_copy_legacy && li + 1 < L.size() && L[li + 1].moe && !L[li + 1].hf &&
           L[li + 1].dev == l.dev;
}
// Its buffers, event and the device view its graph copy writes are made before any capture: nothing allocates inside one.
bool prep_predict_ready(Dev& d, std::string& err) {
    if (!early_copy_ready(d, err)) return false;
    if (!d.prediction_host_dev) {
        void* view = nullptr;
        if (cudaHostGetDevicePointer(&view, d.prediction_host, 0) == cudaSuccess) d.prediction_host_dev = view;
        else cudaGetLastError();   // (without a view the capture keeps the host copy)
    }
    return true;
}
std::vector<long long> prep_graph_key(const Layer& l, const Dev& d, int di, int T) const {
    const auto policy = !o.routing ? resident::Policy::Off :
                        o.routing_prior ? resident::Policy::Prior : resident::Policy::Substitute;
    const float parameter = policy == resident::Policy::Prior ? router_bonus(T) : o.routing_ratio;
    const float floor_value = router_floor(policy, false);   // (the graphs run without the lab: prep_graph_eligible)
    uint32_t bits = 0, floor_bits = 0;
    std::memcpy(&bits, &parameter, sizeof bits);
    std::memcpy(&floor_bits, &floor_value, sizeof floor_bits);
    const auto at = [](const void* p) { return (long long) (uintptr_t) p; };
    return {(long long) (&l - L.data()), (long long) d.nblk, (long long) di, (long long) T, at(d.h), at(d.s2),
            (long long) policy, (long long) bits, o.routing_dampen ? 1LL : 0LL, (lane || helper) ? 1LL : 0LL, at(l.kc),
            at(l.vc), at(l.ic), at(l.kvq.kc), at(l.kvq.vc), (long long) attn_splits,
            early_copy_in_graph(l, T) ? 1LL : 0LL, at(d.prediction_host_dev), (long long) floor_bits,
            abl_any(d, T) ? 1LL : 0LL};   // (--ablate: the attention's write projected in the graph or not)
}
// The ready mask a window's routing reads (route_layer copies it to the device): the layer's resident experts, its
// staged frames whose copies are visible, and the helper's ready set.
void route_ready_host(const Layer& l, Dev& d) {
    // prompt_layerwise: the layer's groups are queued without a join, so an earlier group's copy of this pinned mask may
    // still be pending: the mask is written once a layer (the residents do not change during the read) and then left
    if (prompt_reading) {
        const int layer = int(&l-L.data());
        if (prompt_mask_layer == layer) return;
        prompt_mask_layer = layer;
    }
    *d.ready_h = {};
    for (int e : l.res_ids) d.ready_h->primary[e/32] |= 1u << (e%32);
    const long long layer = (&l-L.data());
    // prompt_layerwise / prompt_check: a prompt's routing sees the published residents only, never a staged frame
    if (prompt_reading || (prompt_check_mode && !generation_active)) {
        // (prompt_check 2: also the uploaded sets of the two-pass read under test, as the chunked reference's mask)
        if (!prompt_reading && prompt_check_mode == 2 && (size_t) layer < prompt_u_bits.size())
            for (int k = 0; k < 4; ++k) d.ready_h->primary[k] |= prompt_u_bits[(size_t) layer][(size_t) k];
        std::copy(l.helper_ready.begin(),l.helper_ready.end(),d.ready_h->helper);
        return;
    }
    for (size_t slot=0;slot<d.slot_owner.size();++slot) {
        const long long key=d.slot_owner[slot];
        if (!r2::snapshot_visible(d.slot_foreground[slot]!=0) || key<0 || key/1024!=layer) continue;
        const int e = (int)(key%1024); d.ready_h->primary[e/32] |= 1u << (e%32);
    }
    std::copy(l.helper_ready.begin(),l.helper_ready.end(),d.ready_h->helper);
    // third: its held experts count as ready, in the helper's bits (routing asks only "on the primary or elsewhere")
    if(third_on)for(size_t k=0;k<4;++k)d.ready_h->helper[k]|=l.third_ready[k];
}
// The preparation through its graph: a replay, or on a key's first use the launches captured (route_layer builds the
// mask then) and the new graph launched. A capture that fails ran nothing: the launches run instead, and the graphs
// stay off for the load.
template <class F>
bool prep_graph_run(Layer& l, Dev& d, int di, int T, F& body, std::string& err) {
    const bool predict = early_copy_in_graph(l, T);
    if (predict && !prep_predict_ready(d, err)) return false;
    const std::vector<long long> key = prep_graph_key(l, d, di, T);
    auto it = prep_graphs.find(key);
    if (it != prep_graphs.end()) {
        route_ready_host(l, d);   // (the mask the replay's copy reads)
        if (o.prep_graph == 2) return prep_graph_verify(l, d, T, it->second, body, err, predict);
        CK(cudaGraphLaunch(it->second, d.s));
        ++st.prep_graph_launches;
        if (predict) ++st.prefetch_predictions;
        return true;
    }
    const double t0 = now();
    cudaGraph_t gr = nullptr;
    cudaGraphExec_t ex = nullptr;
    const bool views = prep_graph_views(d);   // (without them the capture keeps the copies)
    const bool began = cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    prep_zero_copy = began && views;
    const bool ran = began && body();
    prep_zero_copy = false;
    const cudaError_t ended = began ? cudaStreamEndCapture(d.s, &gr) : cudaErrorUnknown;   // (the stream leaves capture)
    const bool ok = ran && ended == cudaSuccess && gr != nullptr && cudaGraphInstantiate(&ex, gr, 0) == cudaSuccess;
    if (gr) cudaGraphDestroy(gr);
    if (!ok) {
        if (ex) cudaGraphExecDestroy(ex);
        cudaGetLastError();
        prep_graph_broken = true;
        ++st.prep_graph_fallbacks;
        warn("prep_graph", "prep_graph off: a preparation did not capture (" + (err.empty() ? std::string("CUDA") : err) +
                           ") - its kernels launched one by one for the rest of this load");
        std::fprintf(stderr, "maya-mm3: WARNING: %s\n", warnings.back().second.c_str());
        err.clear();
        return body();
    }
    prep_graphs.emplace(key, ex);
    ++st.prep_graph_captures;
    if (views) ++st.prep_graph_zero_copy;
    CK(cudaGraphLaunch(ex, d.s));
    st.prep_graph_capture_ms += 1000 * (now() - t0);
    ++st.prep_graph_launches;
    if (predict) ++st.prefetch_predictions;
    return true;
}
// prep_graph 2: the replay against the launches, bit for bit, on the layer's results (h, xn, sh, top, original, w, and
// early_copy's prediction ids when the graph holds the prediction).
template <class F>
bool prep_graph_verify(const Layer& l, Dev& d, int T, cudaGraphExec_t ex, F& body, std::string& err, bool predict = false) {
    const size_t nd = (size_t) T * g.dim * 4, nk = (size_t) T * g.k_exp * 4;
    const size_t m = predict ? 7 : 6;
    const void* from[7] = {d.h, d.xn, d.sh, d.top, d.original_d, d.w, d.prediction_ids};
    const size_t n[7] = {nd, nd, nd, nk, nk, nk, (size_t) g.k_exp * 4};
    size_t bytes = 0;
    for (size_t i = 0; i < m; ++i) bytes += n[i];
    if (!graph_verify_buffers(d, err)) return false;   // (shared with expert_graph_verify)
    const auto keep = [&](uint8_t* to) {
        cudaError_t r = cudaSuccess;
        for (size_t i = 0, at = 0; i < m && r == cudaSuccess; at += n[i], ++i)
            r = cudaMemcpyAsync(to + at, from[i], n[i], cudaMemcpyDeviceToDevice, d.s);
        return r;
    };
    CK(cudaMemcpyAsync(d.pv_save, d.h, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaGraphLaunch(ex, d.s));
    CK(keep(d.pv_a));
    CK(cudaMemcpyAsync(d.h, d.pv_save, nd, cudaMemcpyDeviceToDevice, d.s));
    CK(cudaStreamSynchronize(d.s));   // (the replay's copy has read the mask before route_layer builds it again)
    if (!body()) return false;
    CK(keep(d.pv_b));
    CK(cudaMemcpyAsync(d.pv_h, d.pv_a, bytes, cudaMemcpyDeviceToHost, d.s));
    CK(cudaMemcpyAsync(d.pv_h + bytes, d.pv_b, bytes, cudaMemcpyDeviceToHost, d.s));
    CK(cudaStreamSynchronize(d.s));
    ++st.prep_graph_launches;
    if (std::memcmp(d.pv_h, d.pv_h + bytes, bytes) == 0) {
        ++st.prep_graph_verified;
        return true;
    }
    ++st.prep_graph_mismatches;
    static const char* names[7] = {"h", "xn", "sh", "top", "original", "w", "prediction"};
    std::string which;
    for (size_t i = 0, at = 0; i < m; at += n[i], ++i)
        if (std::memcmp(d.pv_h + at, d.pv_h + bytes + at, n[i]) != 0) which += std::string(which.empty() ? "" : ",") + names[i];
    std::fprintf(stderr, "maya-mm3: prep_graph verify: layer %lld, %d rows at position %d: the replay's %s differ from the launches'\n",
                 (long long) (&l - L.data()), T, cur_pos, which.c_str());
    return true;
}
