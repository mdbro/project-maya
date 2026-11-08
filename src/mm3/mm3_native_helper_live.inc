// Included in Model::Impl after mm3_native_adaptive.inc. One bounded, opt-in
// helper exchange; policy uses the existing original-demand scores unchanged.
bool helper_live_enabled=false;
// Fixture diagnostic only: no new optional helper lease. A lease already started
// still progresses, and helper computation is unaffected.
bool helper_admissions_held=false;
mm3::NativeAdmissionBudget helper_admission_budget;
size_t helper_live_blob=0, helper_live_capacity=0;
std::array<uint8_t*,2> helper_live_buffers{}; // incoming, outgoing; full records
long long helper_live_a=-1, helper_live_b=-1;
uint8_t* helper_live_host=nullptr; // A's original canonical HOST frame
bool helper_live_withdrawn=false, helper_live_published=false;
uint64_t helper_live_started=0, helper_live_withdrawals=0, helper_live_publications=0, helper_live_finished=0;

bool helper_live_leased(long long key) const {
    return key>=0 && (key==helper_live_a || key==helper_live_b);
}
bool helper_live_contains(uintptr_t address,size_t bytes) const {
    for(auto* p:helper_live_buffers) if(p && address>=uintptr_t(p) && bytes<=helper_live_blob &&
        address-uintptr_t(p)<=helper_live_blob-bytes)return true;
    return false;
}
bool helper_live_mode_supported() const {
    return adaptive_supported() || (o.native_stage_prefix==3 && native && native_pool && D.size()==2 && head_dev==1 &&
        !graph_on && o.seqs==1 && o.fast_decode && o.chunk<=kWin && !vis && D[head_dev].n_slots>=4);
}
bool helper_live_prepare(std::string& err) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(!helper_live_mode_supported() || !helper) {err="live helper requires a healthy single-target native helper";return false;}
    if(helper_live_blob)return true;
    const size_t blob=sizeof(ne::Header)+nf::record_stride(g.dim,g.ff);
    for(const auto& l:L)if(l.moe && l.blob!=blob) {err="live helper record geometry mismatch";return false;}
    auto status=cudaSetDevice(D[head_dev].id);
    if(status!=cudaSuccess) {err=std::string("live helper set device: ")+cudaGetErrorString(status);return false;}
    for(auto& p:helper_live_buffers) {
        status=cudaHostAlloc(&p,blob,cudaHostAllocPortable);
        if(status!=cudaSuccess) {
            err=std::string("live helper portable buffer: ")+cudaGetErrorString(status);
            for(auto& q:helper_live_buffers)if(q) {cudaFreeHost(q);q=nullptr;}
            return false;
        }
    }
    helper_live_capacity=helper->records().size();
    if(!helper_live_capacity) {
        err="live helper has no ready records";
        for(auto& q:helper_live_buffers)if(q) {cudaFreeHost(q);q=nullptr;}
        return false;
    }
    helper_live_blob=blob;
    return true;
}
bool helper_live_source_busy(const Dev& d,long long key,const uint8_t* source) const {
    if(adaptive_identity_pending(d,key))return true;
    for(int s=0;s<d.n_slots;++s)
        if(d.slot_pend[s]==key || (d.slot_owner[s]==key && (d.slot_foreground[s] || slot_early_busy(d,s))))return true;
    for(const auto& q:d.promo_q)if(q.src==source)return true;
    return false;
}
bool helper_live_bad(const char* why,std::string& err) {
    err=why;placement_failed=true;return false;
}
bool helper_live_api_bad(const char* why,std::string& err) {
    const std::string detail=err;
    err=why;if(!detail.empty())err+=std::string(": ")+detail;
    placement_failed=true;return false;
}
// Constant-time check for routine progress polls, including per-layer polls.
bool helper_live_verify_pair(std::string& err) {
    if(helper_live_a<0 || helper_live_b<0 || !helper_live_host)return helper_live_bad("live helper lease metadata missing",err);
    auto& a=L[size_t(helper_live_a/1024)];auto& b=L[size_t(helper_live_b/1024)];
    const int ae=int(helper_live_a%1024),be=int(helper_live_b%1024);
    const bool ah=helper_owns(a,ae),bh=helper_owns(b,be);
    const uint8_t* ap=a.host_blob[ae];const uint8_t* bp=b.host_blob[be];
    if(a.res_slot[ae]>=0 || b.res_slot[be]>=0 ||
       ah!=helper_live_published || bh==helper_live_withdrawn ||
       ap!=(helper_live_published?nullptr:helper_live_host) ||
       bp!=(helper_live_withdrawn?helper_live_buffers[1]:nullptr))
        return helper_live_bad("live helper canonical ownership changed",err);
    return true;
}
// Full helper roster audit runs only alongside actual ownership mutations (or
// explicit maintenance), never merely because a copy has not completed.
bool helper_live_verify(std::string& err) {
    if(!helper_live_verify_pair(err))return false;
    const auto records=helper->records();
    const size_t expected=helper_live_capacity-(helper_live_withdrawn && !helper_live_published ? 1:0);
    if(records.size()!=expected)return helper_live_bad("live helper ready capacity mismatch",err);
    for(const auto& r:records)if(!helper_owns(L[size_t(r.layer)],r.expert))
        return helper_live_bad("live helper identity snapshot mismatch",err);
    return true;
}
// Existing foreground events identify every unretired primary-GPU HOST reader;
// CPU jobs and helper compute are joined before this method is called.
bool helper_live_progress(Dev& d,std::string& err) {
    if(&d!=&D[(size_t)head_dev])return true;
    if(helper_live_a<0)return true;
    // The queued consumer bypasses NativeHelper::active. Its complete reader
    // lease must retire before any withdrawal/publication can mutate a slot.
    if(helper_queue_pending)return true;
    if(placement_failed || retirement_uncertain.load())return helper_live_bad("live helper ownership is poisoned",err);
    const auto status=helper->live_poll();
    using P=NativeHelper::LivePhase;
    if(status.phase==P::failed || status.phase==P::idle) {
        if(capture::detailed())capture::detail_event("helper_exchange_error","\"error\":"+capture::quote(status.error));
        err=status.error.empty()?"live helper operation failed":status.error;placement_failed=true;return false;
    }
    auto& a=L[size_t(helper_live_a/1024)];auto& b=L[size_t(helper_live_b/1024)];
    const int ae=int(helper_live_a%1024),be=int(helper_live_b%1024);
    if(!helper_live_verify_pair(err))return false;
    if(status.phase==P::prepared) {
        if(!helper_live_verify(err))return false;
        // The current helper snapshot remains valid until all readers join.
        b.host_blob[be]=helper_live_buffers[1];
        if(!helper->live_withdraw(err)) {b.host_blob[be]=nullptr;return helper_live_api_bad("live helper withdrawal failed",err);}
        b.helper_ready[be/32]&=~(uint32_t(1)<<(be%32)); helper_live_withdrawn=true;++helper_live_withdrawals;
        capture::light::helper_event(capture::light::kHelperWithdrawn,helper_live_a,helper_live_b);
        if(capture::detailed())capture::detail_event("helper_exchange_withdraw","\"incoming\":"+std::to_string(helper_live_a)+",\"victim\":"+std::to_string(helper_live_b));
        return helper_live_verify(err);
    }
    if(status.phase==P::uploaded) {
        if(helper_live_source_busy(d,helper_live_a,helper_live_host))return true;
        if(!helper_live_verify(err))return false;
        if(!helper->live_publish(err))return helper_live_api_bad("live helper publication failed",err);
        a.helper_ready[ae/32]|=uint32_t(1)<<(ae%32);
        a.host_blob[ae]=nullptr;helper_live_published=true;++helper_live_publications;
        capture::light::helper_event(capture::light::kHelperPublished,helper_live_a,helper_live_b);
        if(capture::detailed())capture::detail_event("helper_exchange_publish","\"incoming\":"+std::to_string(helper_live_a)+",\"victim\":"+std::to_string(helper_live_b));
        // The old HOST frame can now receive B's header; the worker copies its
        // payload only after A's HOST owner and all source readers have gone.
        std::memcpy(helper_live_host,helper_live_buffers[1],sizeof(ne::Header));
        for(int s=0;s<d.n_slots;++s)if(d.slot_owner[s]==helper_live_a) {
            prefetch_forget(d,s);d.slot_owner[s]=-1;
        }
        if(!helper->live_recycle(err))return helper_live_api_bad("live helper recycle submission failed",err);
        return helper_live_verify(err);
    }
    if(status.phase==P::complete) {
        if(helper_live_source_busy(d,helper_live_b,helper_live_buffers[1]))return true;
        if(!helper_live_verify(err))return false;
        b.host_blob[be]=helper_live_host;
        bool safe=false;
        if(!helper->live_retire(safe,err) || !safe) {b.host_blob[be]=helper_live_buffers[1];return helper_live_api_bad("live helper retirement failed",err);}
        capture::light::helper_event(capture::light::kHelperRetired,helper_live_a,helper_live_b);
        if(capture::detailed())capture::detail_event("helper_exchange_retire","\"incoming\":"+std::to_string(helper_live_a)+",\"victim\":"+std::to_string(helper_live_b)+",\"host_frame\":"+std::to_string(uintptr_t(helper_live_host)));
        helper_live_a=helper_live_b=-1;helper_live_host=nullptr;
        helper_live_withdrawn=helper_live_published=false;++helper_live_finished;
    }
    return true;
}
bool helper_live_issue(Dev& d,std::string& err) {
    if(&d!=&D[(size_t)head_dev])return true;
    if(!helper_live_enabled || !generation_active || helper_live_a>=0 || helper_queue_pending || helper_admissions_held)return true;
    helper_admission_budget.advance(native_demand_rows,1.0);
    if(!helper_admission_budget.available())return true;
    if(!helper_live_blob || placement_failed || retirement_uncertain.load())return helper_live_bad("live helper is not prepared",err);
    std::vector<ResidencyCandidate> incoming_candidates,victim_candidates;
    for(int li:d.layers) {
        auto& l=L[size_t(li)];if(!l.moe || l.hf || l.dev!=head_dev)continue;
        for(int e=0;e<g.n_exp;++e) {
            const long long key=static_cast<long long>(li)*1024+e;
            const uint8_t* host=l.host_blob[e];
            if(helper_live_source_busy(d,key,host))continue;
            const float value=score(key,false);
            if(helper_owns(l,e)) {
                victim_candidates.push_back({key,value,-1});
            } else if(l.res_slot[e]<0 && host)incoming_candidates.push_back({key,value,-1});
        }
    }
    const auto choice=choose_incumbent_helper(incoming_candidates,victim_candidates);
    trace_residency_decision("helper",incoming_candidates,victim_candidates,choice,-1,0);
    if(!choice)return true;
    const long long incoming=choice.incoming,victim=choice.victim;
    const float hot=score(incoming,false),cold=score(victim,false);
    auto& a=L[size_t(incoming/1024)];auto& b=L[size_t(victim/1024)];
    const int ae=int(incoming%1024),be=int(victim%1024);
    auto* host=const_cast<uint8_t*>(a.host_blob[ae]);
    const auto* ah=reinterpret_cast<const ne::Header*>(host);
    auto* bh=reinterpret_cast<ne::Header*>(helper_live_buffers[1]);
    std::memset(bh,0,sizeof(*bh));
    const uint64_t light_submit=capture::light::stamp();
    if(!helper->live_begin(int(victim/1024),be,int(incoming/1024),ae,
        host+sizeof(ne::Header),helper_live_buffers[0]+sizeof(ne::Header),
        helper_live_buffers[1]+sizeof(ne::Header),helper_live_blob-sizeof(ne::Header),
        ah->weight_global,ah->input_global,bh->weight_global,bh->input_global,err))
        return helper_live_api_bad("live helper begin failed",err);
    helper_live_a=incoming;helper_live_b=victim;helper_live_host=host;helper_admission_budget.spend();
    capture::light::helper_event(capture::light::kHelperLease,incoming,victim,light_submit);
    if(capture::detailed()) {int slot=-1;for(const auto& id:helper->records())if(id.layer==int(victim/1024) && id.expert==be) {slot=id.slot;break;}
        std::ostringstream out;out<<"\"incoming\":"<<incoming<<",\"victim\":"<<victim<<",\"helper_slot\":"<<slot
        <<",\"incoming_score\":"<<hot<<",\"victim_score\":"<<cold<<",\"host_frame\":"<<uintptr_t(host)
        <<",\"incoming_buffer\":"<<uintptr_t(helper_live_buffers[0])<<",\"outgoing_buffer\":"<<uintptr_t(helper_live_buffers[1])
        <<",\"planned_h2d_bytes\":"<<(helper_live_blob-sizeof(ne::Header))<<",\"planned_d2h_bytes\":"<<(helper_live_blob-sizeof(ne::Header))<<",\"status\":\"lease_started\"";
        capture::detail_event("helper_exchange_issue",out.str());}
    helper_live_withdrawn=helper_live_published=false;++helper_live_started;
    return helper_live_verify(err);
}
// Explicit maintenance only: callers first join all external readers and clear
// their foreground flags. Generation never waits here. Progress is bounded even
// if the helper's CPU/GPU worker or a driver call does not return.
bool helper_live_settle(std::string& err) {
    if(helper_queue_pending){err="live helper maintenance has an outstanding queued reader";return false;}
    if(helper_live_a<0)return true;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(helper_live_a>=0) {
        if(!helper_live_progress(D[head_dev],err))return false;
        if(helper_live_a<0)return true;
        if(std::chrono::steady_clock::now()>=deadline)return helper_live_bad("live helper maintenance deadline exceeded",err);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return true;
}
// Called after Model's other readers are retired, before destroying any bank.
// A false result requires retaining the entire Model, including both buffers.
bool helper_live_cancel_retired() {
    if(!helper || helper_live_a<0)return true;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
    bool safe=false;std::string error;
    do {
        if(helper->live_cancel(safe,error) && safe)return true;
        if(std::chrono::steady_clock::now()>=deadline)return false;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    } while(true);
}
