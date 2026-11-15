// Included inside Model::Impl. The helper hand-off (helper_handoff, kernel item 1; off unless a request asks): each helper
// card's work for a whole step queued ahead and released by the 5090 through a word in mapped memory
// (mm3_helper_handoff.hpp; design notes in /mnt/wsl/transfer/minimax-wsl/runs/kernel1-handoff-20261114/).
//
// A forward that uses it (handoff_step): at its first MoE layer each card in use is armed with the step's graph. At each
// MoE layer's preparation the host writes the snapshot (each card's word, which experts each card serves: the 3090's
// ready set, the 4080's held experts that are on no other card and in no usable frame - the same tests window_post
// applies, taken here) and marks the cards busy; the preparation's router branch carries the request kernel, which
// releases both cards. window_post plans with the snapshot's sets (so the host's jobs are the card's, by construction),
// submits nothing to the cards, and after the CPU lane waits for each card's word and merges its columns as collect does.
int handoff_blocks = 2;                                // request-only 3090 grid choice; the 4080 stays at 2
int handoff_mode = 0;                                   // the request's helper_handoff: 0 off, 1 on, 2 on and every pass checked
std::unique_ptr<handoff::Lane> handoff_lane[2];         // 0: the helper (3090), 1: the third pool (4080)
handoff::Snapshot* handoff_snap = nullptr;              // pinned, portable mapped: the pass's snapshot (one, like the ready mask)
bool handoff_step = false;                              // this forward's helper work goes through the hand-off
bool handoff_armed = false;                             // its cards have been armed (at its first MoE layer)
bool handoff_on[2] = {};                                // the cards taking part in this step
bool handoff_windows = false;                           // this step's rows: more than one
std::vector<int> handoff_layers;                        // this step's MoE layers in order
std::vector<int> handoff_index;                         // layer -> its index in the step (-1: none)
uint64_t handoff_version[2] = {~0ull, ~0ull};           // the records' version each card's pinned tables were written at
uint64_t handoff_release_ns = 0;                        // (light records) when this pass's preparation carrying the release was launched
std::string handoff_skip_reason;                        // why the last forward asked for did not use it (STAT warning)
// The cards' clocks on the host's: a card stamps its columns' publication (%globaltimer) and the host sees the word; when
// the host was already spinning (it waited), host seen - card stamp is the offset plus the one-way latency. The smallest of
// a window of such samples is used for the next window (the cards drift about 14 ppm, 1 us in 64 passes), so a pass's
// card end can be placed on the host's clock even when the host looked only after its CPU lane: the light record's lane
// end then means what a worker's publication meant (when the host could know), not when the host happened to look.
struct HandoffClock { int64_t offset = 0, window_min = std::numeric_limits<int64_t>::max(); int samples = 0; bool known = false; };
HandoffClock handoff_clock[2];
uint64_t handoff_card_end(int c, uint64_t card_ns, uint64_t seen_ns, uint64_t wait_ns) {
    HandoffClock& k = handoff_clock[c];
    if (card_ns && wait_ns > 2000) {   // (the host spun at least 2 us: it saw the word as soon as it landed)
        k.window_min = std::min<int64_t>(k.window_min, int64_t(seen_ns) - int64_t(card_ns));
        if (++k.samples >= 64 || (!k.known && k.samples >= 8)) { k.offset = k.window_min; k.known = true; k.window_min = std::numeric_limits<int64_t>::max(); k.samples = 0; }
    }
    if (!k.known || !card_ns) return seen_ns;
    const int64_t at = int64_t(card_ns) + k.offset;
    return at > 0 && uint64_t(at) < seen_ns ? uint64_t(at) : seen_ns;
}

NativeHelper* handoff_helper(int c) const { return c == 0 ? helper.get() : third.get(); }
// Full worker-kernel identity, including corrected NVFP4 arithmetic and its
// source representation. The handoff graph never substitutes BF16 activations.
bool handoff_kernels(int c, handoff::Lane::Kernels& k) const {
    NativeHelper* h = handoff_helper(c);
    if (!h || !handoff::kernel_warps(h->kernel_one()) || !handoff::kernel_warps(h->kernel_multi())) return false;
    k.one = h->kernel_one(); k.multi = h->kernel_multi(); k.loads = h->loads();
    k.grid_blocks_per_sm = c == 0 ? handoff_blocks : 2;
    const bool q = handoff::kernel_family(k.one) >= 0;
    if (q != (handoff::kernel_family(k.multi) >= 0)) return false;
    k.format = q && q_rows_mode > 0 ? (h->expanded_format(q_rows_mode) ? h->expanded_format(q_rows_mode) : 1) : 0;
    k.sm89 = q && h->sm89();
    if ((handoff::kernel_family(k.one) == 2 || handoff::kernel_family(k.multi) == 2) && !k.sm89) return false;
    return true;
}
int handoff_layer_index(const Layer& l) const {
    const size_t li = size_t(&l - L.data());
    return li < handoff_index.size() ? handoff_index[li] : -1;
}
bool handoff_pass(const Layer& l) const { return handoff_step && handoff_layer_index(l) >= 0; }
// (window_post's plan in a hand-off pass) the snapshot's sets
bool handoff_helper_serves(int e) const { return handoff_on[0] && handoff::bit(handoff_snap->helper, e); }
bool handoff_third_serves(int e) const { return handoff_on[1] && handoff::bit(handoff_snap->third, e); }

// Whether this forward uses it, and the step's layers; the cards are armed later, at the first MoE layer.
bool handoff_forward_begin(int T, bool whole_graph, std::string& err) {
    handoff_step = handoff_armed = handoff_inflight = false;
    handoff_on[0] = handoff_on[1] = false;
    if (handoff_mode <= 0 || !generation_active) return true;   // (a prompt's forwards: the worker path, not counted)
    const auto skip = [&](const char* why) { handoff_skip_reason = why; ++st.handoff_skipped; return true; };
    if (!native || !o.native_cooperative || !generation_active || !helper || T < 1 || T > kWin || D.size() != 1 || !rseq.empty() ||
        !o.fast_decode || whole_graph)
        return skip("not a one-card native cooperative generation forward");
    if (lab.on || capture::detailed() || replay_trace_active || replay_fixed_batch || pf.on || predict_diag_on() ||
        helper_dispatch_current == "queue" || (early_copy_on() && o.early_copy_legacy) || lab_pass || o.prep_graph == 2)
        return skip("a diagnostic path is on (route lab, detailed capture, replay, prefetch, queue dispatch, the legacy early copy or prep_graph 2)");
    handoff::Lane::Kernels k[2];
    const bool on[2] = {handoff_kernels(0, k[0]), third_on && third_filled && !third_failed && handoff_kernels(1, k[1])};
    if (!on[0]) return skip("the 3090 kernel or representation is not supported by the queued handoff");
    if (!handoff_snap) {
        CK(cudaSetDevice(D[0].id));
        CK(cudaHostAlloc((void**) &handoff_snap, sizeof(handoff::Snapshot), cudaHostAllocPortable | cudaHostAllocMapped));
        *handoff_snap = handoff::Snapshot{};
    }
    for (int c = 0; c < 2; ++c) {
        if (!on[c] || handoff_lane[c]) continue;
        NativeHelper* h = handoff_helper(c);
        try {
            handoff_lane[c] = std::make_unique<handoff::Lane>(h->device(), h->pool_identity(), h->record_stride(),
                                                              int(h->bytes().pool / h->record_stride()));
        } catch (const std::exception& e) { err = e.what(); return false; }
        handoff_version[c] = ~0ull;
        info.push_back("helper hand-off: card CUDA" + std::to_string(h->device()) + " lane ready");
    }
    CK(cudaSetDevice(D[0].id));
    handoff_layers.clear();
    handoff_index.assign(L.size(), -1);
    for (size_t li = 0; li < L.size(); ++li)
        if (L[li].moe && !L[li].hf && L[li].dev == head_dev) { handoff_index[li] = int(handoff_layers.size()); handoff_layers.push_back(int(li)); }
    if (handoff_layers.empty() || handoff_layers.size() > 255) return skip("no MoE layer on the primary");
    handoff_on[0] = on[0];
    handoff_on[1] = on[1];
    handoff_windows = T > 1;
    handoff_step = true;
    ++st.handoff_steps;
    return true;
}
// The pinned tables of record addresses, rewritten when a card's records moved (a swap, a placement, a repair).
void handoff_tables(int c) {
    NativeHelper* h = handoff_helper(c);
    const uint64_t v = h->records_version();
    if (v == handoff_version[c]) return;
    const uintptr_t pool = h->pool_identity();
    const size_t stride = h->record_stride();
    std::array<NativeHelper::QueueIdentity, 128> ids{};
    for (int li = 0; li < g.n_layers && li < handoff::kLayers; ++li) {
        handoff::Entry* t = handoff_lane[c]->table(li);
        std::string ignored;
        if (!L[size_t(li)].moe || !h->queue_snapshot(li, ids, ignored)) { std::fill_n(t, handoff::kExperts, handoff::Entry{}); continue; }
        for (int e = 0; e < handoff::kExperts; ++e) {
            handoff::Entry en;
            en.ready = ids[size_t(e)].slot >= 0;
            en.record = en.ready ? (unsigned long long) (pool + size_t(ids[size_t(e)].slot) * stride) : 0ull;
            for (int q = 0; q < 3; ++q) en.multiplier[q] = ids[size_t(e)].multiplier[q];
            t[e] = en;
        }
    }
    handoff_version[c] = v;
}
// window_pre of a hand-off pass, before the preparation is launched: arm at the step's first layer, the snapshot, busy.
bool handoff_pre(Layer& l, Dev& d, std::string& err) {
    const int m = handoff_layer_index(l);
    for (int c = 0; c < 2; ++c) if (handoff_on[c]) handoff_tables(c);
    if (!handoff_armed) {
        for (int c = 0; c < 2; ++c) {
            if (!handoff_on[c]) continue;
            handoff::Lane::Kernels k;
            handoff_kernels(c, k);
            if (!handoff_lane[c]->arm(handoff_layers, handoff_windows, k, err)) return false;
        }
        CK(cudaSetDevice(d.id));
        handoff_armed = true;
    }
    handoff::Snapshot& s = *handoff_snap;
    s = handoff::Snapshot{};
    const size_t li = size_t(&l - L.data());
    for (int c = 0; c < 2; ++c) if (handoff_on[c]) s.word[c] = handoff_lane[c]->word(m);
    for (int e = 0; e < g.n_exp && e < handoff::kExperts; ++e) {
        if (handoff_on[0] && helper_owns(l, e) && handoff_lane[0]->table(int(li))[e].ready) s.helper[e >> 5] |= 1u << (e & 31);
        else if (handoff_on[1] && third_holds(l, e) && on_cpu(l, d, e) && handoff_lane[1]->table(int(li))[e].ready)
            s.third[e >> 5] |= 1u << (e & 31);
    }
    for (int c = 0; c < 2; ++c) if (handoff_on[c]) handoff_helper(c)->handoff_busy(true);
    handoff_inflight = true;
    handoff_collected = false;
    handoff_release_ns = helper_clock_ns();
    return true;
}
// The router branch's request kernel (window_pre's body, after the routing).
bool handoff_request(Layer& l, Dev& d, int T, cudaStream_t xs, std::string& err) {
    const int m = handoff_layer_index(l);
    const size_t li = size_t(&l - L.data());
    handoff::RequestArgs a;
    a.top = d.top; a.w = d.w; a.xn = d.xn; a.T = T; a.k = g.k_exp; a.layer = int(li); a.scale = g.scale;
    a.snapshot = handoff_snap;
    for (int c = 0; c < 2; ++c)
        if (handoff_on[c]) {
            handoff::Lane::Kernels k;
            if (!handoff_kernels(c, k)) { err = "helper hand-off: kernel selection changed during an armed forward"; return false; }
            a.table[c] = handoff_lane[c]->table(int(li)); a.request[c] = handoff_lane[c]->request(m);
            a.format[c] = k.format; a.fp8[c] = handoff::kernel_family(k.one) == 2;
        }
    if (!handoff::launch_request(a, xs)) { err = "helper hand-off: the request kernel did not launch"; return false; }
    return true;
}
// NativeHelper::submit's order: by width, 1 to 8, each in the plan's order (the order the card's columns come in).
static std::vector<NativeHelper::Job> handoff_submit_order(const std::vector<NativeHelper::Job>& jobs) {
    std::vector<NativeHelper::Job> out;
    out.reserve(jobs.size());
    for (int cols = 1; cols <= NativeHelper::kMaxColumns; ++cols)
        for (const auto& j : jobs) if (j.columns == cols) out.push_back(j);
    return out;
}
// (helper_handoff=2) the card's jobs as the 5090 wrote them against the host's: count, experts, widths, rows, weights (bits),
// records and multipliers.
bool handoff_same_jobs(int c, int m, const std::vector<NativeHelper::Job>& jobs, std::string& why) {
    const handoff::Request* r = handoff_lane[c]->request(m);
    if (r->count != int(jobs.size())) { why = "job count " + std::to_string(r->count) + " against " + std::to_string(jobs.size()); return false; }
    NativeHelper* h = handoff_helper(c);
    std::array<NativeHelper::QueueIdentity, 128> ids{};
    std::string ignored;
    if (!jobs.empty() && !h->queue_snapshot(jobs.front().layer, ids, ignored)) { why = "no table"; return false; }
    for (size_t i = 0; i < jobs.size(); ++i) {
        const handoff::Job& dj = r->jobs[i];
        const NativeHelper::Job& hj = jobs[i];
        bool same = dj.expert == hj.expert && dj.columns == hj.columns;
        for (int q = 0; same && q < hj.columns; ++q)
            same = dj.token[q] == hj.token[q] && std::memcmp(&dj.weight[q], &hj.weight[q], sizeof(float)) == 0;
        const auto& id = ids[size_t(hj.expert)];
        same = same && id.slot >= 0 && dj.record == (unsigned long long) (h->pool_identity() + size_t(id.slot) * h->record_stride()) &&
               std::memcmp(dj.multiplier, id.multiplier, sizeof dj.multiplier) == 0;
        if (!same) { why = "job " + std::to_string(i) + " (expert " + std::to_string(hj.expert) + ")"; return false; }
    }
    return true;
}
// window_post of a hand-off pass, after the CPU lane: card c's word, then its columns merged into result (zero-filled
// rows plus the columns, as collect gives them). jobs: the host plan's jobs of that card (creation order).
bool handoff_collect(int c, Layer& l, int T, const std::vector<NativeHelper::Job>& jobs, const float* rx, std::vector<float>& result,
                     std::vector<NativeHelper::Refused>& refused, uint64_t& seen_ns, uint64_t& wait_begin_ns, uint64_t& card_end_ns,
                     std::string& err) {
    const int m = handoff_layer_index(l);
    int status = 0;
    uint64_t wait_ns = 0;
    wait_begin_ns = helper_clock_ns();
    const bool waited = handoff_lane[c]->wait(m, 2.0, status, seen_ns, wait_ns, err);
    if (!waited) { native_cpu_failed = true; return false; }   // (the card may still write the slot: nothing reuses it)
    handoff_helper(c)->handoff_busy(false);
    card_end_ns = handoff_card_end(c, handoff_lane[c]->result(m)->t_done, seen_ns, wait_ns);
    st.handoff_wait_ms += double(wait_ns) * 1e-6;
    if (status != handoff::kServed) {
        err = std::string("helper hand-off: card CUDA") + std::to_string(handoff_lane[c]->device()) + " layer " +
              std::to_string(&l - L.data()) + ": " +
              (status == handoff::kNonfiniteRow ? "nonfinite helper activation" : status == handoff::kExpired ? "the card's wait expired" :
               status == handoff::kAborted ? "the step was aborted" : status == handoff::kBadRecord ? "a record outside the card's pool" :
               "an invalid request");
        return false;
    }
    refused.clear();
    if (jobs.empty()) return true;
    const auto ordered = handoff_submit_order(jobs);
    handoff::Result* r = handoff_lane[c]->result(m);
    result.resize((size_t) T * g.dim);
    if (!handoff_helper(c)->merge_external(T, ordered, &r->y[0][0], result.data(), result.size(), err, &refused)) return false;
    NativeHelper::PassTiming actual{};
    actual.input_format = r->format; actual.kernel = r->kernel; actual.multi = r->multi != 0;
    actual.sm89 = r->sm89 != 0; actual.graph = true;
    note_helper_route(actual, c == 1);
    if (c == 0) { ++st.handoff_passes; st.handoff_jobs += (long) jobs.size(); }
    else { ++st.handoff_third_passes; st.handoff_third_jobs += (long) jobs.size(); }
    if (handoff_mode == 2) {   // the same jobs through today's worker path, the merged rows compared bit for bit
        std::string why;
        bool same = handoff_same_jobs(c, m, ordered, why);
        if (same && refused.empty()) {
            // The request slot is owned until this collection retires. Submit
            // those exact represented rows to the worker, including transport.
            const auto* request = handoff_lane[c]->request(m);
            const auto* input = reinterpret_cast<const uint8_t*>(request->x);
            std::vector<float> reference((size_t) T * g.dim);
            std::string werr;
            const uint64_t seq = request->format == 0
                ? handoff_helper(c)->submit(T, reinterpret_cast<const uint16_t*>(input), jobs, werr)
                : request->format == 1 ? handoff_helper(c)->submit_packed(T, input, jobs, werr)
                : handoff_helper(c)->submit_expanded(T, input, request->format, jobs, werr);
            if (!seq || !handoff_helper(c)->collect(seq, reference.data(), reference.size(), werr)) { err = "helper hand-off check: " + werr; return false; }
            same = std::memcmp(reference.data(), result.data(), reference.size() * sizeof(float)) == 0;
            if (!same) why = "merged rows";
        }
        if (same) ++st.handoff_verified;
        else {
            ++st.handoff_mismatches;
            std::fprintf(stderr, "maya-mm3: helper_handoff check: card CUDA%d layer %lld, %d rows: the hand-off's %s differ from today's path\n",
                         handoff_lane[c]->device(), (long long) (&l - L.data()), T, why.c_str());
        }
    }
    return true;
}
// window_post of a hand-off pass: card c's columns collected where today's collect of that card sits (the 3090 after the
// CPU lane, then the 4080), and added to the CPU lane's rows (cpu_rows given); or, for a pass with no CPU or card job,
// the cards' empty passes only waited for (cpu_rows null), so their slots are free for reuse. A card not in the step is
// left to today's path.
bool handoff_collected = false;
bool handoff_inflight = false;   // (a released pass not yet collected: helper_live_progress waits)
bool handoff_collect_card(int c, Layer& l, Dev& d, int T, const float* rx, const std::vector<NativeHelper::Job>& jobs, float* cpu_rows,
                          int* cpu_flags, bool light_on, std::string& err) {
    if (!handoff_on[c]) return true;
    if (!jobs.empty() && !cpu_rows) { err = "helper hand-off: card jobs without the CPU rows"; return false; }
    const size_t li = size_t(&l - L.data());
    std::vector<float>& result = c == 0 ? d.helper_result : d.third_result;
    std::vector<NativeHelper::Refused>& refused = d.helper_refused;
    uint64_t seen = 0, wait_begin = 0, card_end = 0;
    capture::Range range(c == 0 ? "helper_collect" : "third_collect");
    if (!handoff_collect(c, l, T, jobs, rx, result, refused, seen, wait_begin, card_end, err)) return false;
    if (jobs.empty()) return true;
    // a non-finite column: left out by the merge and recomputed on the CPU lane (mm3_native_helper_recover.inc)
    if (!refused.empty() &&
        !helper_recover(*handoff_helper(c), c == 1, refused, rx, nullptr, result.data(), "decode", 0, T, window_position(T), err)) {
        native_cpu_failed = true;
        return false;
    }
    const uint64_t merged = helper_clock_ns();
    for (size_t i = 0; i < result.size(); ++i) cpu_rows[i] += result[i];
    int columns = 0;
    for (const auto& j : jobs) for (int q = 0; q < j.columns; ++q) { cpu_flags[j.token[q]] = 1; ++columns; }
    if (c == 0) {
        st.helper_experts += (long) jobs.size();
        exchange_x4_window(kX4AfterCollectNs);   // (exchange_schedule: the 3090 is idle until the next release)
        if (light_on) {   // (the card's lane in the light record: released at the preparation's launch, seen done by the host)
            light_draft.collect_end = merged; light_draft.helper_enqueue = handoff_release_ns;
            light_draft.helper_worker_start = 0; light_draft.helper_record_begin = 0; light_draft.helper_record_end = 0;
            light_draft.helper_last_not_ready = 0; light_draft.helper_ready_observed = seen; light_draft.helper_worker_published = card_end;
            light_draft.helper_wait_begin = wait_begin; light_draft.helper_wait_end = seen;
        }
    } else {
        ++st.third_passes; st.third_jobs += (long) jobs.size();
        if (light_on) capture::light::third_pass(light_draft.begin, int(li), int(jobs.size()), columns, handoff_release_ns, 0, card_end,
                                                 wait_begin, seen, merged);
    }
    return true;
}
// Both cards, in today's order; the second is skipped (its busy mark cleared) when the first fails.
bool handoff_collect_pass(Layer& l, Dev& d, int T, const float* rx, const std::vector<NativeHelper::Job>& helper_jobs,
                          const std::vector<NativeHelper::Job>& third_jobs, float* cpu_rows, int* cpu_flags, bool light_on,
                          int only, std::string& err) {
    handoff_collected = true;
    for (int c = 0; c < 2; ++c) {
        if (only >= 0 && c != only) continue;
        if (!handoff_collect_card(c, l, d, T, rx, c == 0 ? helper_jobs : third_jobs, cpu_rows, cpu_flags, light_on, err)) {
            for (int q = c + 1; q < 2; ++q) if (handoff_on[q]) handoff_helper(q)->handoff_busy(false);
            handoff_inflight = false;
            return false;
        }
    }
    if (only != 0) handoff_inflight = false;   // (both cards collected: the 4080's call, or the call for both)
    return true;
}
// The forward's end, or its failure: the rest of the step skipped on the cards (nothing is left waiting), the busy marks
// cleared, and the clock holders armed until the next step.
void handoff_forward_end() {
    if (!handoff_step) return;
    for (int c = 0; c < 2; ++c) {
        if (!handoff_on[c]) continue;
        handoff_helper(c)->handoff_busy(false);
        handoff_lane[c]->end_step(handoff_helper(c)->occupancy());
    }
    handoff_step = handoff_armed = handoff_inflight = false;
}
void handoff_release() {
    for (auto& lane : handoff_lane) lane.reset();
    if (handoff_snap) { cudaFreeHost(handoff_snap); handoff_snap = nullptr; }
}
