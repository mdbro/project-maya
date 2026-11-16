// Included in Model::Impl after mm3_native_third.inc. PLACEMENT reference FILE (mm3_reference_state.hpp): a stored
// reference state forced onto this engine start, for benchmarks. Serving never sends it.
//
// Two starts of one binary and one set of start options differ in what they measured and in how much VRAM was free: each
// measures its own lane costs (twice: at load, then inside the first request), and the primary's resident count follows
// the free VRAM at load. Those decide, in every pass, whether a host expert runs on the CPU lane or is copied to the
// primary, and what the primary exchanges. The in-process checkpoint (PLACEMENT snapshot / restore) carries them within
// one start only. This command loads a checkpoint from a file instead, sets the third card's roster with it, and clears
// the few things a checkpoint does not hold:
//   - forced from the file: the primary's, the helper's and the third card's rosters (the host bank is the rest), the
//     demand scores, the admission credits, the residency policy, the lane costs and the grouped staging table, and the
//     flag that says the lane costs are final (so the first request does not measure them again);
//   - cleared, as restore clears them: in-flight helper swaps and primary exchanges are completed, staging frames, the
//     promotion queue and the captured graphs are dropped; the serve loop drops the KV prefix;
//   - cleared here: the pass model's running average, the exchange schedule's wait state (the serve loop also drops the
//     prompt lookup's policy).
// Refused before anything changes when the file does not fit this start: another geometry or arithmetic, or another
// count on any card (the primary's count can be capped at start with --native-resident-max). A start that has served
// nothing yet takes it too: the adaptive residency the state expects is prepared first. What stays this start's own:
// which physical slot or host frame each expert sits in, the time axes, and each lane's real speed.
// The reply is the restore reply with the state's identity and the third card's fields, so the tools' signature of a
// restore reply (native_calibrate.learned_state_signature) is the same on every start that took the same file.
bool reference_apply(const std::string& path, std::string& report, std::string& err) {
    if(!native || !o.native_cooperative || !helper || D.size()!=1) {err="reference state: needs the native cooperative engine with a helper";return false;}
    std::vector<uint8_t> moe;
    for(const auto& l:L)moe.push_back(l.moe?1:0);
    mm3::ReferenceState r;
    if(!mm3::read_reference_state(path,moe,g.n_exp,r,err))return false;
    if(!r.calibrated || !r.costs.valid || r.batch_calibration.empty() ||
       r.calibration_arithmetic!=r.arithmetic || r.batch_arithmetic!=r.arithmetic) {
        err="reference state: requires final compatible lane and grouped calibration";return false;
    }
    if(r.dim!=g.dim || r.ff!=g.ff || r.resident_cap!=o.resident_cap) {err="reference state: another model geometry or per-layer cap";return false;}
    if(r.arithmetic!=native_primary_current) {
        err="reference state: the primary's arithmetic is "+native_primary_current+", the state's "+r.arithmetic;return false;
    }
    if(r.shared_bytes!=(native_shared_pool?native_shared_pool->size():0) || (native_shared_pool && r.version<2)) {
        err="reference state: shared HOST membership and policy state are missing or incompatible";return false;
    }
    if(native_shared_pool && r.shared_keys.size()>shared_frames.size()-2) {
        err="reference state: shared HOST members consume the required free landings";return false;
    }
    size_t primary_capacity=0;
    for(const auto& l:L)if(l.moe)primary_capacity+=l.res_ids.size();
    const size_t helper_capacity=helper_live_capacity ? helper_live_capacity : helper->records().size();
    const size_t third_now=third && !third_failed ? size_t(third_capacity) : 0;
    if(r.primary.size()!=primary_capacity || r.helper.size()!=helper_capacity || r.third.size()!=third_now) {
        err="reference state: it holds "+std::to_string(r.primary.size())+" experts on the primary, "+std::to_string(r.helper.size())+
            " on the helper and "+std::to_string(r.third.size())+" on the third card; this start "+std::to_string(primary_capacity)+", "+
            std::to_string(helper_capacity)+" and "+std::to_string(third_now)+
            " (a larger primary count can be capped at start: --native-resident-max)";
        return false;
    }
    for(int id:r.helper)if(L[size_t(id/g.n_exp)].dev!=head_dev) {err="reference state: a helper expert outside the head device's layers";return false;}
    // A start that has served no request has not prepared the primary's adaptive residency yet (its first request with
    // primary_adaptive does: two host records of exchange scratch and the frame table, no resident capacity). The state
    // was saved with it on, so it is prepared here, as that request would: a fresh start then takes the state
    // (2026-11-14: the first measured start refused it for this).
    if(r.primary_adaptive && !adaptive_prepare(err)) {
        err="reference state: the primary's adaptive residency could not be prepared ("+err+")";return false;
    }
    LearnedCheckpoint c;
    c.layers=g.n_layers;c.experts=g.n_exp;c.dim=g.dim;c.ff=g.ff;c.resident_cap=o.resident_cap;
    c.snapshot_fwd=fwd;c.snapshot_demand_rows=native_demand_rows;
    c.arithmetic=r.arithmetic;c.calibration_arithmetic=r.calibration_arithmetic;
    for(const auto& l:L)c.hf_layers.push_back(l.hf?1:0);
    c.primary=r.primary;c.helper=r.helper;c.third=r.third;c.scores=r.scores;
    c.adaptive_bits.assign(adaptive_main.size(),0);
    for(int id:r.adaptive_main) {
        if(size_t(id)>=c.adaptive_bits.size()) {err="reference state: an adaptive mark outside this start's table";return false;}
        c.adaptive_bits[size_t(id)]=1;
    }
    c.primary_credit=r.primary_credit;c.helper_credit=r.helper_credit;
    c.policy.legacy=r.policy_legacy;c.policy.horizon=r.policy_horizon;c.policy.max_exchanges=r.policy_max_exchanges;
    c.policy.service_saved_ms=r.policy_service_saved_ms;c.policy.h2d_ms=r.policy_h2d_ms;c.policy.d2h_ms=r.policy_d2h_ms;
    c.policy.interference_ms=r.policy_interference_ms;c.policy.margin_ms=r.policy_margin_ms;
    c.primary_enabled=r.primary_adaptive;
    c.cpu1=r.cpu1;c.cpu_add=r.cpu_add;c.copy=r.copy;c.cpu_scale=r.cpu_scale;
    c.retime.n=r.retime[0];c.retime.sx=r.retime[1];c.retime.sxx=r.retime[2];c.retime.st=r.retime[3];c.retime.sxt=r.retime[4];
    c.retime.model=r.retime[5];c.retime.experts=int(r.retime[6]);c.retime.win=int(r.retime[7]);c.retime.dir=int(r.retime[8]);
    c.retime.changes=int(r.retime[9]);c.retime.prev_c1=r.retime[10];c.retime.prev_add=r.retime[11];
    c.batch_costs=r.costs;c.batch_calibration=r.batch_calibration;c.batch_arithmetic=r.batch_arithmetic;
    c.shared_bytes=r.shared_bytes;c.shared_keys=r.shared_keys;c.free_time=r.free_time;
    c.calibrated=r.calibrated;c.valid=true;
    // From here the engine changes. The checkpoint this start held, if any, is replaced: later restores return here.
    learned_checkpoint=std::move(c);
    std::string restored;
    if(!learned_state("restore",restored,err))return false;

    if(restored.empty() || restored.back()!='}') {err="reference state: unexpected restore report";return false;}
    restored.pop_back();
    report=restored+",\"reference_state\":\""+r.id+"\""+"}";
    return true;
}
