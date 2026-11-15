// ---- kernel item 2 (2026-11-14): the CPU lane's reworked kernel and its L3 prefetch, a request's cpu_kernel and
// cpu_l3_prefetch (Model::set_cpu_kernel, set_cpu_l3_prefetch; both off by default).
//
// cpu_kernel 1 / 2 runs the CPU lane's window jobs on mm3_native_cpu_fast (1: INT8 activations, the shared kernel's bits;
// 2: NVFP4 activations by the 5090's rule).
//
// cpu_l3_prefetch, drafting windows only (one row has early_copy, which uploads the predicted expert instead): DRAM is
// shared by the CPU lane and the 5090's uploads (about 80 GB/s together, 41 for the upload beside a CPU job against 57.4
// alone), and between a layer's last upload and the next layer's join it sits idle while the GPU runs the next layer's
// attention (a median 0.48 ms in the adopted serving captures, kernel2-cpu-20261114/l3-bound). So:
//  - each window layer also predicts the next layer's experts (that layer's router on this layer's input, as early_copy
//    does at one row), in the preparation (its graph when prep_graph is on), the ids to the host with the routing;
//  - at the layer's join the ids are kept (l3_note); after its planning an event marks its demand copies (l3_arm_uploads);
//  - before the next layer's join the host waits for that event (DRAM is then free; the host had only the join left to
//    wait for), picks up to two predicted experts that would be served from host memory - fewest predicted rows first,
//    the CPU lane's likeliest - and the lane's idle workers read them into their caches, each worker the share it
//    computes (native_cpu_l3_begin: a worker's L2 and its CCD's L3), until the join (l3_begin / l3_end);
//  - the next layer's CPU job then runs those experts with the fixed shares (native_cpu_run_multi_fast's l3 flags), from
//    the caches, and the pass's uploads keep DRAM to themselves. An expert read ahead and then uploaded instead cost only
//    idle DRAM time.
// STAT: cpu_l3_* (Stats).
bool cpu_fast_control() const { return native && lane && native_cpu_workspace && cpu_fast::supported(); }
bool l3_on() const {   // (cpu_l3 first: the built-in path pays one test)
    return cpu_l3 && cpu_fast_control() && o.native_cooperative && generation_active && D.size() == 1 && rseq.empty() &&
           !o.routing_prefetch;
}
// The next layer's prediction in this window layer's preparation: two rows or more, the next layer an MoE layer on this
// device, and no other user of the prediction buffers (MM3_PREDICT_DIAG, routing_prefetch).
bool l3_predict_on(const Layer& l, int T, bool regular) const {
    const size_t li = size_t(&l - L.data());
    return l3_on() && T >= 2 && T <= kWin && l.moe && !l.hf && li + 1 < L.size() && L[li + 1].moe && !L[li + 1].hf &&
           L[li + 1].dev == l.dev && !prefetch_active(regular) && !predict_diag_on();
}
// On the router's side stream (joined before the routing readback): the next layer's router on this layer's input, its
// top-k, the ids to the pinned host buffer - by a kernel on its device view inside prep_graph's capture.
bool l3_predict(Layer& l, Dev& d, int T, cudaStream_t xs, std::string& err) {
    Layer& next = L[size_t(&l - L.data()) + 1];
    if (!predict_diag_ready(d, err)) return false;   // (made before any capture: prep_graph_run's prep_predict_ready)
    k::rows_dot(next.gate, g.n_exp, g.dim, d.xn, T, d.prediction_logits, xs);
    cudaError_t status = cudaPeekAtLastError();
    if (status == cudaSuccess)
        status = resident::route(d.prediction_logits, next.bias, T, resident::Policy::Off, 0, nullptr, d.prediction_ids,
                                 d.prediction_weights, nullptr, nullptr, xs);
    if (status == cudaSuccess) {
        if (prep_zero_copy && d.prediction_host_dev) {
            k::copy_words(d.prediction_ids, d.prediction_host_dev, size_t(T) * g.k_exp * 4, xs);
            status = cudaPeekAtLastError();
        } else {
            status = cudaMemcpyAsync(d.prediction_host, d.prediction_ids, size_t(T) * g.k_exp * 4, cudaMemcpyDeviceToHost, xs);
        }
    }
    if (status != cudaSuccess) { err = std::string("cpu_l3_prefetch prediction: ") + cudaGetErrorString(status); return false; }
    return true;
}
// After the layer's join: its prediction of the next layer, kept (the next preparation overwrites the buffer).
void l3_note(const Layer& l, const Dev& d, int T, bool regular) {
    if (!l3_predict_on(l, T, regular) || !d.prediction_host) return;
    const int* ids = reinterpret_cast<const int*>(d.prediction_host);
    l3_next_ids.assign(ids, ids + size_t(T) * g.k_exp);
    l3_next_layer = int(&l - L.data()) + 1;
    l3_next_fwd = (long long) fwd;
    l3_next_rows = T;
    ++st.cpu_l3_predictions;
}
// After the layer's planning: an event behind its demand copies (none: DRAM is free once the CPU job is done).
void l3_arm_uploads(const Layer& l, Dev& d, int T, bool regular, long staged_before) {
    d.l3_uploads_armed = false;
    if (!l3_predict_on(l, T, regular) || st.staged == staged_before) return;
    if (!d.l3_uploads && cudaEventCreateWithFlags(&d.l3_uploads, cudaEventDisableTiming) != cudaSuccess) { cudaGetLastError(); return; }
    if (cudaEventRecord(d.l3_uploads, overlap_copied ? d.us : d.s) == cudaSuccess) d.l3_uploads_armed = true;   // (expert_overlap: their stream)
    else cudaGetLastError();
}
// A layer's read-ahead experts, counted once its CPU jobs ran: those they took (used) and those they did not (unused).
void l3_account() {
    if (!l3_blobs.empty()) {   // (the read-ahead experts its CPU job took, and those it did not)
        st.cpu_l3_used += l3_used;
        st.cpu_l3_unused += (long) l3_blobs.size() - l3_used;
    }
    l3_blobs.clear();
    l3_layer = -1;
    l3_used = 0;
}
// Before layer l's join: read its predicted host experts ahead on the lane's idle workers.
void l3_begin(Layer& l, Dev& d) {
    l3_account();
    const size_t li = size_t(&l - L.data());
    if (!l3_on() || l3_next_layer != int(li) || l3_next_fwd != (long long) fwd || !l.moe || l.hf) return;
    l3_next_layer = -1;
    if (d.l3_uploads_armed) {   // (the uploads have DRAM until they end; the host has only the join to wait for)
        const double t0 = now();
        if (cudaEventSynchronize(d.l3_uploads) != cudaSuccess) cudaGetLastError();
        st.cpu_l3_wait_ms += 1000 * (now() - t0);
        d.l3_uploads_armed = false;
    }
    l3_layer = int(li);   // (a prediction for this layer: its CPU experts not read ahead count as missed)
    l3_fwd = (long long) fwd;
    std::array<int, 128> uses{};
    for (int id : l3_next_ids) if (id >= 0 && id < g.n_exp && id < 128) ++uses[size_t(id)];
    std::vector<int> cand;
    for (int e = 0; e < g.n_exp && e < 128; ++e)
        if (uses[size_t(e)] && pooled(l, e) && on_cpu(l, d, e) && !third_holds(l, e)) cand.push_back(e);
    std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return uses[size_t(a)] < uses[size_t(b)]; });
    if (cand.size() > 2) cand.resize(2);
    std::vector<const uint8_t*> blobs;
    for (int e : cand) blobs.push_back(l.host_blob[size_t(e)]);
    if (blobs.empty() || !native_cpu_l3_begin(*lane, g.dim, g.ff, blobs, &l3_bytes)) return;
    l3_blobs = blobs;
    l3_used = 0;
    l3_running = true;
    ++st.cpu_l3_started;
    st.cpu_l3_experts += (long) blobs.size();
}
// After the join: the workers stop reading (within 4 KiB each) before the planning issues this layer's uploads.
void l3_end() {
    if (!l3_running) return;
    lane->stop_background();
    l3_running = false;
    st.cpu_l3_bytes += l3_bytes.exchange(0);
}
// The CPU lane's jobs of a window pass on the reworked kernel: which experts were read ahead for this layer.
bool lane_run_fast(const std::vector<NativeCpuExpert>& jobs, const std::vector<std::vector<std::pair<int, float>>>& users,
                   const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order) {
    std::vector<uint8_t> fixed;
    if (generation_active && l3_layer >= 0 && l3_layer == lane_layer && l3_fwd == (long long) fwd) {
        fixed.assign(jobs.size(), 0);
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (std::find(l3_blobs.begin(), l3_blobs.end(), jobs[i].blob) != l3_blobs.end()) { fixed[i] = 1; ++l3_used; }
            else ++st.cpu_l3_missed;
        }
    }
    ++st.cpu_fast_calls;
    return native_cpu_run_multi_fast(*lane, cpu_kernel_mode >= 2 ? cpu_kernel_mode : 1, g.dim, g.ff, jobs, users, x, T, out, order,
                                     native_cpu_workspace, native_cpu_bytes, false, fixed.empty() ? nullptr : &fixed);
}
