// ---- route_lab (mm3_route_lab.hpp; routing-lab-20261110/CONTRACT.md): the routing lab's per-request rules, applied on
// the host to generation passes (one row and verify windows) of the native cooperative path. The router kernel runs as
// today and also hands its scores to the host (route_layer); window_post then, after its join and before anything reads
// the routes, recomputes today's rule from those scores and checks it against the kernel bit for bit, runs the
// request's rules, and rewrites the host routes the plan is built from. A skipped pick (k_drop, layer drop) keeps a
// valid id in the routes but is never planned: not staged, not run, weight 0. A widened row's fifth pick takes plan slot
// 4 (the plan and the 5090's window kernels hold eight a row); the 3090 takes at most four picks a token, so a token's
// fifth 3090 pick goes in a second helper submission in the same pass, collected before the combine is released.
// Fully resident (host-free) layers plan on the device: the lab reads their routes back, and writes ids and weights back
// only where the rules changed them (in practice only layer drop can act there: every expert is ready). Graphs are off
// while the lab is on. With route_lab 0 none of this runs.
// The diagnostics reach the token records through mm3_route_record.hpp (reroute-replay's hook, API.md 1.5): a forward's
// first lab pass moves the serial and sizes the rows, its last MoE layer writes each row's route object (route_stats 1,
// 2 with "layers") and the request's settings and summary.
route_lab::Settings lab;                        // the request's settings; lab.on: route_lab=1
route_lab::RequestStats lab_req;                // the request's summary: every row the lab routed
route_lab::RequestStats lab_req_prompt;         // ... of which the prompt's (route_lab_prompt)
std::vector<route_lab::RowStats> lab_rows;      // the last lab forward's rows
long long lab_rows_fwd = -1;                    // the forward lab_rows belong to
std::vector<int> lab_moe;                       // model layer -> MoE ordinal (-1: dense)
int lab_n_moe = 0;
// the pass window_post is planning
bool lab_pass = false;
int lab_pass_k = 4;                             // picks a row in the plan: 4, or 5 with a widened row
std::array<uint8_t, kWin * 8> lab_skip{};       // [row][slot]: planned nothing
std::array<int, kWin> lab_extra{};              // a widened row's fifth pick (-1: none)
std::array<float, kWin> lab_extra_w{};
std::vector<NativeHelper::Job> lab_helper_second;   // 3090 picks past four a token
std::array<route_lab::Row, kWin> lab_out{};

// generation passes, and with route_lab_prompt the chunked prefill's (the layer-by-layer reader is refused at the request)
bool lab_active(const Layer& l, int T) const {
    return lab.on && native && o.native_cooperative && (generation_active || lab.prompt) && rseq.empty() && D.size() == 1 &&
           l.moe && T >= 1 && T <= kWin && !replay_fixed_batch && !prompt_reading;
}
float lab_kernel_bonus() const { return o.routing && o.routing_prior ? o.routing_prior_bonus : 0.0f; }
// the bonus the router kernel used for this pass (route_prompt_bonus while a lab request reads its prompt)
float lab_pass_kernel_bonus() const { return o.routing && o.routing_prior ? router_bonus() : 0.0f; }
// The Prior bonus while the prompt is read, for every request (routing_prompt_bonus; -1: the generation's bonus).
// The user's decision of 2026-11-11: the prompt is read by the unmodified router, so serving sends 0.
float prompt_bonus = -1.0f;
// the router kernel's Prior bonus now. While the prompt is read: a lab request's route_prompt_bonus if it gives one,
// else routing_prompt_bonus if the request gives one; otherwise, and for every generated token, routing_prior_bonus.
float router_bonus() const {
    if (!generation_active) {
        if (lab.on && lab.prompt_bonus >= 0.0f) return lab.prompt_bonus;
        if (prompt_bonus >= 0.0f) return prompt_bonus;
    }
    return o.routing_prior_bonus;
}
bool lab_buffers(Dev& d, std::string& err) {
    CK(cudaSetDevice(d.id));
    if (!d.lab_scores_d) CK(cudaMalloc(&d.lab_scores_d, size_t(kWin) * 128 * 3 * sizeof(float)));
    if (!d.lab_scores_h) CK(cudaMallocHost(&d.lab_scores_h, size_t(kWin) * 128 * 3 * sizeof(float)));
    if (!d.lab_hf_h) CK(cudaMallocHost(&d.lab_hf_h, size_t(kWin) * 4 * 12));
    return true;
}
void lab_map_layers() {
    if (lab_moe.size() == L.size()) return;
    lab_moe.assign(L.size(), -1);
    lab_n_moe = 0;
    for (size_t i = 0; i < L.size(); ++i) if (L[i].moe) lab_moe[i] = lab_n_moe++;
}
// The masks at the routing (route_layer, after route_ready_host): the snapshot the kernel reads, and the experts with a
// copy to the 5090 issued and not yet visible (a foreground copy pending publication, or a pending promotion).
void lab_snapshot(const Layer& l, Dev& d) {
    route_lab::Masks m{};
    for (int w = 0; w < 4; ++w) {
        m.ready[w] = d.ready_h->primary[w] | d.ready_h->helper[w];
        m.helper[w] = d.ready_h->helper[w];
    }
    for (int e : l.res_ids) route_lab::set_bit(m.resident, e);
    const long long layer = &l - L.data();
    for (size_t s = 0; s < d.slot_owner.size(); ++s) {
        const long long own = d.slot_owner[s], pend = d.slot_pend[s];
        if (own >= 0 && own / 1024 == layer && !r2::snapshot_visible(d.slot_foreground[s] != 0))
            route_lab::set_bit(m.in_flight, int(own % 1024));
        if (pend >= 0 && pend / 1024 == layer) route_lab::set_bit(m.in_flight, int(pend % 1024));
    }
    for (int w = 0; w < 4; ++w) m.in_flight[w] &= ~m.ready[w];
    d.lab_masks = m;
}
void lab_snapshot_all_ready(Dev& d) {
    route_lab::Masks m{};
    for (int w = 0; w < 4; ++w) m.ready[w] = m.resident[w] = 0xffffffffu;
    d.lab_masks = m;
}
// The pass's rules over the kernel's routes ids [T][4] and weights [T][4] (host arrays, rewritten in place). Fills
// lab_out and the pass's skip / extra arrays, checks the kernel, and records the rows' diagnostics.
void lab_route(const Layer& l, Dev& d, int T, int* ids, float* w) {
    lab_map_layers();
    const int li = int(&l - L.data());
    const float kb = lab_pass_kernel_bonus();
    const bool prompt = !generation_active;
    route_lab::Settings s = lab;
    s.bonus = kb;
    if (prompt) s.bonus_draft = -1.0f;   // (prompt chunks: the prompt's bonus, never the drafting one)
    bool invalid[kWin] = {};
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 4; ++i)   // (the kernel's mark: ids -1 and NaN weights; a host-free layer's ids are guarded to 0)
            invalid[t] |= ids[t * 4 + i] < 0 || ids[t * 4 + i] >= 128 || !std::isfinite(w[t * 4 + i]);
    route_lab::PassStats ps{};
    route_lab::route_pass(s, li, T, d.lab_scores_h, d.lab_masks, invalid, lab_out.data(), &ps);
    const bool rules = s.any_rule();
    long checked = 0, mismatch = 0, identity = 0;
    for (int t = 0; t < T; ++t) {
        if (invalid[t]) continue;
        int ref_ids[4]; float ref_w[4];
        route_lab::reference_prior(d.lab_scores_h + size_t(t) * 128 * 3, d.lab_masks.ready, kb, ref_ids, ref_w);
        ++checked;
        mismatch += std::memcmp(ref_ids, ids + t * 4, sizeof ref_ids) != 0 || std::memcmp(ref_w, w + t * 4, sizeof ref_w) != 0;
        if (!rules) {
            const route_lab::Row& r = lab_out[size_t(t)];
            identity += r.slots != 4 || std::memcmp(r.ids, ids + t * 4, 4 * sizeof(int)) != 0 ||
                        std::memcmp(r.w, w + t * 4, 4 * sizeof(float)) != 0;
        }
    }
    for (route_lab::RequestStats* q : {&lab_req, prompt ? &lab_req_prompt : nullptr}) {
        if (!q) continue;
        q->kernel_checked += checked;
        q->kernel_mismatch += mismatch;
        q->identity_mismatch += identity;
        q->add(ps);
    }
    auto& record = route_record::forward();
    if (lab_rows_fwd != fwd) {   // (a forward's first lab pass)
        lab_rows_fwd = fwd;
        lab_rows.assign(size_t(T), route_lab::RowStats{});
        for (auto& r : lab_rows) r.clear(lab_n_moe);
        lab_req.rows += T;
        ++lab_req.forwards;
        if (prompt) { lab_req_prompt.rows += T; ++lab_req_prompt.forwards; }
        ++record.serial;
        record.rows = T;
        record.row.assign(size_t(T), std::string());
    }
    std::fill(lab_skip.begin(), lab_skip.end(), 0);
    std::fill(lab_extra.begin(), lab_extra.end(), -1);
    lab_pass_k = 4;
    for (int t = 0; t < T; ++t) {
        const route_lab::Row& r = lab_out[size_t(t)];
        if (size_t(t) < lab_rows.size()) lab_rows[size_t(t)].add(r, lab_moe[size_t(li)], ps.est_ms, ps.loads);
        lab_req.add(r);
        if (prompt) lab_req_prompt.add(r);
        if (r.invalid) continue;
        for (int i = 0; i < 4; ++i) {
            ids[t * 4 + i] = r.ids[i];
            w[t * 4 + i] = r.w[i];
            lab_skip[size_t(t) * 8 + i] = r.skip[i] ? 1 : 0;
        }
        if (r.slots > 4) {
            lab_extra[size_t(t)] = r.ids[4];
            lab_extra_w[size_t(t)] = r.w[4];
            lab_pass_k = 5;
        }
    }
    if (lab_moe[size_t(li)] == lab_n_moe - 1) {   // (the forward's last MoE layer: its rows are complete)
        for (int t = 0; t < T && size_t(t) < record.row.size(); ++t)
            record.row[size_t(t)] = lab.stats > 0 ? route_lab::row_json(lab_rows[size_t(t)], lab.stats) : std::string();
        route_record::request().settings = route_lab::settings_json(lab_settings_now());
        route_record::request().summary = lab_summary_json();
    }
    if (capture::detailed() || lab_dump_rows()) {
        std::ostringstream out;
        out.precision(9);
        out << "\"layer\":" << li << ",\"rows\":" << T << ",\"prompt\":" << (prompt ? 1 : 0) << ",\"kernel_bonus\":" << kb
            << ",\"pass_bonus\":"
            << route_lab::bonus_for(s, T, li) << ",\"kernel_mismatch_rows\":" << mismatch << ",\"identity_mismatch_rows\":"
            << identity << ",\"pass_loads\":" << ps.loads << ",\"pass_in_flight\":" << ps.in_flight << ",\"est_ms\":"
            << ps.est_ms << ",\"row\":[";
        for (int t = 0; t < T; ++t) {
            const route_lab::Row& r = lab_out[size_t(t)];
            if (t) out << ',';
            out << "{\"original\":[" << r.original[0] << ',' << r.original[1] << ',' << r.original[2] << ','
                << r.original[3] << "],\"ids\":[";
            for (int i = 0; i < r.slots; ++i) out << (i ? "," : "") << r.ids[i];
            out << "],\"w\":[";
            for (int i = 0; i < r.slots; ++i) out << (i ? "," : "") << r.w[i];
            out << "],\"skip\":[";
            for (int i = 0; i < r.slots; ++i) out << (i ? "," : "") << (r.skip[i] ? 1 : 0);
            out << "],\"ready\":[";   // (the original four's, then the executed ids', at the routing)
            for (int i = 0; i < 4; ++i) out << (i ? "," : "") << (route_lab::bit(d.lab_masks.ready, r.original[i]) ? 1 : 0);
            out << "],\"ready_ids\":[";
            for (int i = 0; i < r.slots; ++i) out << (i ? "," : "") << (route_lab::bit(d.lab_masks.ready, r.ids[i]) ? 1 : 0);
            out << "],\"kept\":" << r.kept << ",\"moved\":" << r.moved << ",\"loads\":" << r.loads << ",\"dropped\":"
                << r.dropped << ",\"widened\":" << (r.widened ? 1 : 0) << ",\"layer_dropped\":" << (r.layer_dropped ? 1 : 0)
                << ",\"invalid\":" << (r.invalid ? 1 : 0) << ",\"changed\":" << r.changed << '}';
        }
        out << ']';
        if (capture::detailed()) capture::detail_event("route_lab_pass", out.str());
        if (FILE* f = lab_dump_rows() ? lab_dump() : nullptr) {   // (the same object a line, flushed at each forward's end)
            std::fprintf(f, "{\"fwd\":%lld,%s}\n", fwd, out.str().c_str());
            if (lab_moe[size_t(li)] == lab_n_moe - 1) std::fflush(f);
        }
    }
}
route_lab::Settings lab_settings_now() const {
    route_lab::Settings s = lab;
    s.bonus = lab_kernel_bonus();
    return s;
}
// the done record's route object: every lab-routed row, and "prompt" for the prompt's rows among them
std::string lab_summary_json() const {
    std::string all = route_lab::request_json(lab_req);
    all.pop_back();
    return all + ",\"prompt\":" + route_lab::request_json(lab_req_prompt) + "}";
}
// MM3_ROUTE_LAB_DUMP=<file> (diagnostic): a readiness line a host-planned generation or prompt pass, lab on or off (about
// 150 bytes); with MM3_ROUTE_LAB_DUMP_ROWS=1 also every lab pass's rows (about 300 bytes a row). Appended, flushed at
// each forward's end.
bool lab_dump_rows() const {
    static const bool rows = [] { const char* v = std::getenv("MM3_ROUTE_LAB_DUMP_ROWS"); return v && std::string(v) == "1"; }();
    return rows && lab_dump();
}
FILE* lab_dump() const {
    static FILE* f = [] {
        const char* path = std::getenv("MM3_ROUTE_LAB_DUMP");
        return path && *path ? std::fopen(path, "a") : nullptr;
    }();
    return f;
}
// MM3_ROUTE_LAB_DUMP, every host-planned generation pass, the lab on or off: what the pass executes against the router's
// ready snapshot, from the same arrays the plan reads (the readiness comparison of the lab path with the serving kernel)
void lab_observe(const Layer& l, const Dev& d, int T, const int* rtop) {
    FILE* f = lab_dump();
    if (!f || !native || !o.native_cooperative || prompt_reading || l.hf || T < 1 || T > kWin) return;
    const auto ready = [&](int e) { return route_lab::bit(d.ready_h->primary, e) || route_lab::bit(d.ready_h->helper, e); };
    long picks = 0, moved = 0, orig_unready = 0, loads = 0;
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < g.k_exp; ++i) orig_unready += !ready(d.original_h[t * g.k_exp + i]);
        for (int i = 0; i < (lab_pass ? lab_pass_k : g.k_exp); ++i) {
            if (lab_skipped(t, i)) continue;
            const int e = lab_pick(rtop, t, i);
            ++picks;
            loads += !ready(e);
            moved += std::find(d.original_h + t * g.k_exp, d.original_h + (t + 1) * g.k_exp, e) == d.original_h + (t + 1) * g.k_exp;
        }
    }
    std::fprintf(f, "{\"observe\":1,\"fwd\":%lld,\"layer\":%d,\"rows\":%d,\"prompt\":%d,\"lab\":%d,\"picks\":%ld,"
                    "\"moved\":%ld,\"orig_unready\":%ld,\"loads\":%ld}\n", fwd, int(&l - L.data()), T, generation_active ? 0 : 1,
                 lab_pass ? 1 : 0, picks, moved, orig_unready, loads);
    if (lab_moe.size() == L.size() && lab_moe[size_t(&l - L.data())] == lab_n_moe - 1) std::fflush(f);
}
// window_post, after the plan: a widened row's fifth pick dispatched, and the done record's summary current at a
// forward's last MoE layer
void lab_after_plan(const Layer& l, int fifth) {
    if (!lab_pass) return;
    lab_req.fifth += fifth;
    if (!generation_active) lab_req_prompt.fifth += fifth;
    if (lab_moe.size() == L.size() && lab_moe[size_t(&l - L.data())] == lab_n_moe - 1)
        route_record::request().summary = lab_summary_json();
}
// window_post, a host-planned layer: the lab on this pass, or today's arrays untouched
void lab_window(const Layer& l, Dev& d, int T, int* rtop, float* rw) {
    lab_pass = lab_active(l, T) && !l.hf;
    lab_pass_k = g.k_exp;
    lab_helper_second.clear();
    if (!lab_pass) return;
    lab_route(l, d, T, rtop, rw);
}
// a pick of the pass's plan (slot i of row t: 0-3 the routes, 4 the widened fifth) and whether it is planned
int lab_pick(const int* rtop, int t, int i) const { return i < g.k_exp ? rtop[t * g.k_exp + i] : lab_extra[size_t(t)]; }
bool lab_skipped(int t, int i) const {
    if (!lab_pass) return false;
    return i < g.k_exp ? lab_skip[size_t(t) * 8 + size_t(i)] != 0 : lab_extra[size_t(t)] < 0;
}
bool lab_extra_uses(int e) const {
    if (!lab_pass || lab_pass_k <= g.k_exp) return false;
    for (int x : lab_extra) if (x == e) return true;
    return false;
}
// window_post, a host-free layer, before its device plan: the lab's routes written over the kernel's where they differ
bool lab_hf(const Layer& l, Dev& d, int T, std::string& err) {
    if (!lab_active(l, T)) return true;
    int* ids = (int*) d.lab_hf_h;
    float* w = (float*) (d.lab_hf_h + size_t(kWin) * 16);
    float* wout = (float*) (d.lab_hf_h + size_t(kWin) * 32);
    CK(cudaSetDevice(d.id));
    CK(cudaMemcpyAsync(ids, d.top, size_t(T) * 4 * sizeof(int), cudaMemcpyDeviceToHost, d.s));
    CK(cudaMemcpyAsync(w, d.w, size_t(T) * 4 * sizeof(float), cudaMemcpyDeviceToHost, d.s));
    CK(cudaStreamSynchronize(d.s));
    std::vector<int> before(ids, ids + T * 4);
    std::vector<float> wbefore(w, w + T * 4);
    lab_route(l, d, T, ids, w);
    bool changed = std::memcmp(before.data(), ids, size_t(T) * 4 * sizeof(int)) != 0 ||
                   std::memcmp(wbefore.data(), w, size_t(T) * 4 * sizeof(float)) != 0;
    for (int t = 0; t < T && !changed; ++t) changed |= lab_out[size_t(t)].slots > 4;
    if (!changed) return true;
    for (int t = 0; t < T; ++t)
        if (lab_out[size_t(t)].slots > 4) {   // (no unready pick on a fully resident layer: widen cannot act)
            err = "route_lab: a widened row on a host-free layer";
            return false;
        }
    std::memcpy(wout, w, size_t(T) * 4 * sizeof(float));
    CK(cudaMemcpyAsync(d.top, ids, size_t(T) * 4 * sizeof(int), cudaMemcpyHostToDevice, d.s));
    CK(cudaMemcpyAsync(d.w, wout, size_t(T) * 4 * sizeof(float), cudaMemcpyHostToDevice, d.s));
    CK(cudaStreamSynchronize(d.s));   // (the pinned arrays are reused by the next host-free layer)
    return true;
}
// window_post, after the first 3090 collection: the picks past four a token, as a second submission
bool lab_helper_second_pass(int T, const float* rx, float* cpu_rows, int* cpu_flags, std::string& err) {
    if (lab_helper_second.empty()) return true;
    std::vector<uint16_t> input((size_t) T * g.dim);
    for (size_t i = 0; i < input.size(); ++i) {
        uint32_t u;
        std::memcpy(&u, rx + i, 4);
        input[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    const uint64_t seq = helper->submit(T, input.data(), lab_helper_second, err);
    if (!seq) return false;
    std::vector<float> result((size_t) T * g.dim);
    if (!helper->collect(seq, result.data(), result.size(), err)) return false;
    for (size_t i = 0; i < result.size(); ++i) cpu_rows[i] += result[i];
    for (const auto& j : lab_helper_second) for (int c = 0; c < j.columns; ++c) cpu_flags[j.token[c]] = 1;
    st.helper_experts += (long) lab_helper_second.size();
    ++lab_req.helper_second;
    lab_helper_second.clear();
    return true;
}
