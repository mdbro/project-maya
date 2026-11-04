// Included inside Model::Impl. Native, single-target-GPU main-bank replacement.
// Sources remain canonical until both asynchronous copies have completed. A
// completed transaction rotates pointers; it never overwrites a live last copy.
bool primary_adaptive = false;
mm3::PrimaryResidencyPolicy adaptive_policy;
long long adaptive_budget_fwd = -1;
int adaptive_budget_used = 0;
mm3::NativeAdmissionBudget primary_admission_budget;
size_t adaptive_blob = 0;
std::array<uint8_t*,2> adaptive_host_owned{}, adaptive_host_spare{};
std::vector<uint8_t> adaptive_main;
std::vector<uintptr_t> adaptive_gpu_frames; // immutable original main+staging frames
// Physical HOST copies are separate from the exclusive host_blob owner map.
std::unique_ptr<NativeHostBank> primary_backing_bank;
std::vector<uint8_t*> primary_backing;

bool primary_backing_contains(uintptr_t a,size_t n) const {
    if(!primary_backing_bank)return false;
    const uintptr_t begin=uintptr_t(primary_backing_bank->data());
    const size_t bytes=primary_backing_bank->size();
    return a>=begin && n<=bytes && a-begin<=bytes-n;
}
uint8_t*& primary_backing_for(int li,int e) {return primary_backing.at(size_t(li)*g.n_exp+e);}

uint8_t* staging_address(const Dev& d, int slot) const {
    return d.slot_ptr.empty() ? d.stg+size_t(slot)*d.slot_bytes : d.slot_ptr.at(size_t(slot));
}
bool staging_reserved(const Dev& d, int slot) const {
    return !d.slot_reserved.empty() && d.slot_reserved.at(size_t(slot));
}
int adaptive_pending(const Dev& d) const {
    int n=0;for(const auto& x:d.adaptive)n+=x.incoming>=0;return n;
}
bool adaptive_identity_pending(const Dev& d, long long key) const {
    for(const auto& x:d.adaptive)if(x.incoming==key || x.victim==key)return true;
    return false;
}
bool adaptive_host_contains(uintptr_t address,size_t bytes) const {
    for(auto* p:adaptive_host_owned)if(p && address>=uintptr_t(p) &&
        bytes<=adaptive_blob && address-uintptr_t(p)<=adaptive_blob-bytes)return true;
    return false;
}
bool adaptive_storage_ok(const std::vector<uintptr_t>& hosts,const std::vector<uintptr_t>& residents) const {
    if(!adaptive_blob)return true;
    auto h=hosts,v=residents;
    const auto in_host=[&](uintptr_t a) {
        const uintptr_t begin=uintptr_t(pool);const size_t bytes=native_pool->size();
        return (a>=begin && bytes>=adaptive_blob && a-begin<=bytes-adaptive_blob) ||
            adaptive_host_contains(a,adaptive_blob) || primary_backing_contains(a,adaptive_blob);
    };
    for(auto a:h)if(!in_host(a))return false;
    if(primary_backing_bank) for(auto* p:primary_backing) if(p) {
        if(!in_host(uintptr_t(p)))return false;h.push_back(uintptr_t(p));
    }
    for(auto* p:adaptive_host_spare) {if(!p || !in_host(uintptr_t(p)))return false;h.push_back(uintptr_t(p));}
    for(const auto& d:D)for(int s=0;s<d.n_slots;++s)v.push_back(uintptr_t(staging_address(d,s)));
    const auto separate=[&](std::vector<uintptr_t>& addresses) {
        std::sort(addresses.begin(),addresses.end());
        for(size_t i=1;i<addresses.size();++i)if(addresses[i]-addresses[i-1]<adaptive_blob)return false;
        return true;
    };
    return separate(h) && separate(v) && v==adaptive_gpu_frames;
}
bool adaptive_supported() const {
    return native && native_pool && D.size()==1 && !graph_on && o.seqs==1 &&
           o.fast_decode && o.chunk<=kWin && !vis && D[0].n_slots>=4;
}
bool adaptive_fail(const std::string& what, cudaError_t status, std::string& err,bool unsafe=true) {
    err=what+": "+cudaGetErrorString(status);placement_failed=true;
    if(capture::detailed()) capture::detail_event("cache_error","\"operation\":"+capture::quote(what)+",\"error\":"+capture::quote(cudaGetErrorString(status)));
    if(unsafe)retirement_uncertain.store(true);
    return false;
}
// Explicit control preparation is outside generation timing, retained by both
// arms. No GPU frame or resident capacity is added; two HOST records are bounded
// exchange scratch. Their physical addresses can later become canonical HOST.
bool adaptive_prepare(std::string& err) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(!adaptive_supported()) {err="adaptive primary requires healthy single-target native window serving";return false;}
    if(adaptive_blob)return true;
    const size_t blob=sizeof(ne::Header)+nf::record_stride(g.dim,g.ff);
    for(const auto& l:L)if(l.moe && l.blob!=blob) {err="adaptive record geometry mismatch";return false;}
    auto& d=D[0];
    const auto device=cudaSetDevice(d.id);
    if(device!=cudaSuccess)return adaptive_fail("adaptive set device",device,err,false);
    for(size_t i=0;i<adaptive_host_owned.size();++i) {
        auto status=cudaHostAlloc(&adaptive_host_owned[i],blob,cudaHostAllocPortable);
        if(status!=cudaSuccess)return adaptive_fail("adaptive HOST spare",status,err,false);
        adaptive_host_spare[i]=adaptive_host_owned[i];
        status=cudaEventCreateWithFlags(&d.adaptive[i].event,cudaEventDisableTiming);
        if(status!=cudaSuccess)return adaptive_fail("adaptive completion event",status,err,false);
    }
    d.slot_ptr.resize(size_t(d.n_slots));d.slot_reserved.assign(size_t(d.n_slots),0);
    for(int s=0;s<d.n_slots;++s)d.slot_ptr[s]=d.stg+size_t(s)*d.slot_bytes;
    for(auto& l:L)if(l.moe && l.res_ptr.empty())
        for(size_t s=0;s<l.res_ids.size();++s)l.res_ptr.push_back(l.res_base+s*l.blob);
    for(const auto& l:L)if(l.moe)for(auto* p:l.res_ptr)adaptive_gpu_frames.push_back(uintptr_t(p));
    for(auto* p:d.slot_ptr)adaptive_gpu_frames.push_back(uintptr_t(p));
    std::sort(adaptive_gpu_frames.begin(),adaptive_gpu_frames.end());
    adaptive_main.assign(size_t(g.n_layers)*g.n_exp,0);
    adaptive_blob=blob;
    return true;
}

// Called after a primary stream join, or between target forwards. At a layer
// join the current layer's already-frozen routing snapshot must be protected.
// Queries never wait, spin, or change credit. Submitted descriptors retain every
// source and destination until successful publication or model quarantine.
bool adaptive_publish(Dev& d,std::string& err,int protected_layer=-1) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    for(size_t i=0;i<d.adaptive.size();++i) {
        auto& x=d.adaptive[i];if(x.incoming<0)continue;
        if(protected_layer>=0 && (x.incoming/1024==protected_layer || x.victim/1024==protected_layer))continue;
        if(!x.submitted) {err="adaptive transaction lacks successful submission";placement_failed=true;return false;}
        ++st.adaptive_queries;
        const auto status=cudaEventQuery(x.event);
        if(status==cudaErrorNotReady) {++st.adaptive_not_ready;continue;}
        if(status!=cudaSuccess)return adaptive_fail("adaptive completion",status,err);
        bool foreground_source=false;
        for(int s=0;s<d.n_slots;++s)
            if(d.slot_owner[s]==x.incoming && d.slot_foreground[s])foreground_source=true;
        if(foreground_source) {++st.adaptive_source_busy;continue;}
        auto& a=L[size_t(x.incoming/1024)];auto& b=L[size_t(x.victim/1024)];
        const int ae=int(x.incoming%1024),be=int(x.victim%1024),bs=b.res_slot[be];
        if(a.res_slot[ae]>=0 || a.host_blob[ae]!=x.source_host || bs<0 ||
           resident_address(b,bs)!=x.source_gpu || !staging_reserved(d,x.landing) ||
           staging_address(d,x.landing)!=x.landing_gpu ||
           (primary_backing_bank ? primary_backing_for(int(x.victim/1024),be)!=x.target_host : adaptive_host_spare[i]!=x.target_host)) {
            err="adaptive publication ownership changed";placement_failed=true;return false;
        }
        for(const auto& q:d.promo_q)if(q.src==x.source_host) {
            err="adaptive source still held by queued promotion";placement_failed=true;return false;
        }
        for(int s=0;s<d.n_slots;++s)if(d.slot_pend[s]==x.incoming) {
            err="adaptive source still held by pending promotion";placement_failed=true;return false;
        }
        // Remove the victim while preserving the reverse slot mapping of the
        // last record. This also works when A and B belong to the same layer.
        const int last=b.res_ids.back();
        b.res_ids[size_t(bs)]=last;b.res_ptr[size_t(bs)]=b.res_ptr.back();b.res_slot[last]=bs;
        b.res_ids.pop_back();b.res_ptr.pop_back();b.res_slot[be]=-1;
        a.res_slot[ae]=int(a.res_ids.size());a.res_ids.push_back(ae);a.res_ptr.push_back(x.landing_gpu);
        if(primary_backing_bank) {
            if(!primary_backing_ledger::rotate(a.host_blob[ae],primary_backing_for(int(x.incoming/1024),ae),
                                               b.host_blob[be],primary_backing_for(int(x.victim/1024),be))) {
                err="adaptive backing rotation failed";placement_failed=true;return false;
            }
        } else {a.host_blob[ae]=nullptr;b.host_blob[be]=x.target_host;adaptive_host_spare[i]=x.source_host;}
        d.slot_ptr[size_t(x.landing)]=x.source_gpu;
        d.slot_owner[size_t(x.landing)]=x.victim;
        d.slot_used[size_t(x.landing)]=fwd-1;
        d.slot_reserved[size_t(x.landing)]=0;
        // An A demand may have obtained a separate foreground copy while its
        // optional exchange was pending. It is now a redundant cache entry;
        // do not pin that staging frame through A's future selected-reader uses.
        for(int s=0;s<d.n_slots;++s)if(s!=x.landing && d.slot_owner[s]==x.incoming)d.slot_owner[s]=-1;
        adaptive_main[size_t(x.incoming/1024)*g.n_exp+ae]=1;
        adaptive_main[size_t(x.victim/1024)*g.n_exp+be]=0;
        if(capture::detailed()) {
            std::ostringstream out; out << "\"incoming\":"<<x.incoming<<",\"victim\":"<<x.victim
                <<",\"landing_slot\":"<<x.landing<<",\"resident_frame\":"<<uintptr_t(x.landing_gpu)
                <<",\"working_frame\":"<<uintptr_t(x.source_gpu)<<",\"victim_host_frame\":"<<uintptr_t(x.target_host)
                <<",\"source_host_frame\":"<<uintptr_t(x.source_host)<<",\"status\":\"published\"";
            capture::detail_event("primary_exchange_publish",out.str());
        }
        ++st.adaptive_published;
        const auto event=x.event;x={};x.event=event;
    }
    return true;
}

// Unissued descriptors may be cancelled; submitted descriptors may not. Return
// the freed slot, preserving the ordinary promotion lifecycle's counters.
int adaptive_cancel_queued(Dev& d,int requested_slot=-1) {
    auto best=d.promo_q.end();float least=0;
    for(auto it=d.promo_q.begin();it!=d.promo_q.end();++it) {
        const int s=it->slot;
        if(d.slot_iss[s] || staging_reserved(d,s) || (requested_slot>=0 && s!=requested_slot))continue;
        const float value=score(d.slot_pend[s],false);
        if(best==d.promo_q.end() || value<least) {best=it;least=value;}
    }
    if(best==d.promo_q.end())return -1;
    const int s=best->slot;
    if(capture::detailed()) capture::detail_event("promotion_cancel","\"slot\":"+std::to_string(s)+",\"incoming\":"+std::to_string(d.slot_pend[s])+",\"status\":\"cancelled_unsubmitted\"");
    prefetch_external_cancel(d,s);d.promo_q.erase(best);
    d.slot_pend[s]=-1;d.slot_iss[s]=0;d.slot_owner[s]=-1;--d.n_pending;
    ++st.adaptive_cancelled_promotions;return s;
}

// At the existing joined target-forward end, before ordinary queued promotion
// submissions. At most two swaps and four optional fills total are outstanding.
// Admission occurs in generation; the existing scores include original demand
// from both prefill and generation, with 0.95 decay per target-forward call.
bool adaptive_issue_legacy(Dev& d,std::string& err) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(!primary_adaptive || !generation_active)return true;
    if(!adaptive_blob || placement_failed) {err="adaptive primary is not prepared";return false;}
    const double began=now();
    for(size_t tx=0;tx<d.adaptive.size();++tx) {
        auto& x=d.adaptive[tx];if(x.incoming>=0)continue;
        long long incoming=-1,victim=-1;float hot=kPromote,cold=0;
        for(int li:d.layers) {
            auto& l=L[size_t(li)];if(!l.moe || l.hf)continue;
            for(int e=0;e<g.n_exp;++e) {
                const long long key=static_cast<long long>(li)*1024+e;
                if(adaptive_identity_pending(d,key) || helper_live_leased(key))continue;
                const float value=score(key,false);
                if(l.res_slot[e]>=0) {
                    if(victim<0 || value<cold) {victim=key;cold=value;}
                } else if(l.host_blob[e] && value>=hot) {
                    bool submitted_source=false;
                    for(int s=0;s<d.n_slots;++s)
                        if(d.slot_pend[s]==key && d.slot_iss[s])submitted_source=true;
                    if(submitted_source)continue;
                    if(incoming<0 || value>hot) {incoming=key;hot=value;}
                }
            }
        }
        if(incoming<0 || victim<0 || !(hot>1.25f*cold+1.0f)) {++st.adaptive_no_gain;break;}
        int slot=-1;bool reuse=false;
        // Reuse a completed ordinary landing before paying another H2D copy.
        for(int s=0;s<d.n_slots;++s)if(!staging_reserved(d,s) && !d.slot_foreground[s] &&
            d.slot_pend[s]<0 && d.slot_owner[s]==incoming) {slot=s;reuse=true;break;}
        // Cancel an unissued copy of this source even if a ready copy exists.
        for(int s=0;s<d.n_slots;++s)if(d.slot_pend[s]==incoming && !d.slot_iss[s]) {
            const int freed=adaptive_cancel_queued(d,s);if(slot<0)slot=freed;break;
        }
        if(d.n_pending+adaptive_pending(d)>=kMaxPending && adaptive_cancel_queued(d)<0) {
            ++st.adaptive_no_slot;break;
        }
        if(slot<0)for(int s=0;s<d.n_slots;++s)if(!staging_reserved(d,s) &&
            !d.slot_foreground[s] && d.slot_pend[s]<0 &&
            (slot<0 || d.slot_used[s]<d.slot_used[slot]))slot=s;
        if(slot<0)slot=adaptive_cancel_queued(d);
        if(slot<0) {++st.adaptive_no_slot;break;}
        auto& a=L[size_t(incoming/1024)];auto& b=L[size_t(victim/1024)];
        const int ae=int(incoming%1024),be=int(victim%1024);
        if(primary_backing_bank && (!primary_backing_for(int(victim/1024),be) ||
                                   primary_backing_for(int(incoming/1024),ae))) {
            err="adaptive backing ledger mismatch before submission";placement_failed=true;return false;
        }
        x.incoming=incoming;x.victim=victim;x.landing=slot;x.submitted=false;
        x.source_host=const_cast<uint8_t*>(a.host_blob[ae]);
        x.source_gpu=resident_address(b,b.res_slot[be]);x.landing_gpu=staging_address(d,slot);
        x.target_host=primary_backing_bank ? primary_backing_for(int(victim/1024),be) : adaptive_host_spare[tx];
        prefetch_forget(d,slot);d.slot_reserved[slot]=1;d.slot_owner[slot]=-1;
        auto status=cudaSuccess;
        if(!reuse)status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess && !primary_backing_bank)status=cudaMemcpyAsync(x.target_host,x.source_gpu,adaptive_blob,cudaMemcpyDeviceToHost,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.event,d.cs);
        if(status!=cudaSuccess)return adaptive_fail("adaptive copy/event submission",status,err);
        x.submitted=true;++st.adaptive_issued;
        if(capture::detailed()) {std::ostringstream out;out<<"\"policy\":\"legacy\",\"incoming\":"<<incoming<<",\"victim\":"<<victim
            <<",\"incoming_score\":"<<hot<<",\"victim_score\":"<<cold<<",\"reuse\":"<<(reuse?1:0)
            <<",\"landing_slot\":"<<slot<<",\"source_host_frame\":"<<uintptr_t(x.source_host)
            <<",\"landing_frame\":"<<uintptr_t(x.landing_gpu)<<",\"victim_frame\":"<<uintptr_t(x.source_gpu)
            <<",\"target_host_frame\":"<<uintptr_t(x.target_host)<<",\"h2d_bytes\":"<<(reuse?0:adaptive_blob)
            <<",\"d2h_bytes\":"<<(primary_backing_bank?0:adaptive_blob)<<",\"status\":\"submitted\"";capture::detail_event("primary_exchange_issue",out.str());}
        st.adaptive_h2d_bytes+=reuse?0:adaptive_blob;st.adaptive_d2h_bytes+=primary_backing_bank?0:adaptive_blob;
        st.adaptive_reused+=reuse;
    }
    st.adaptive_host_ms+=1000*(now()-began);
    return true;
}

double adaptive_service_ms() const {
    if(adaptive_policy.service_saved_ms>0)return adaptive_policy.service_saved_ms;
    // lane_cpu1 is an observed complete CPU service, not an H2D measurement.
    // Use the measured grouped marginal when available; it is a service proxy,
    // not isolated critical-path savings under contention.
    if(batch_staging_costs.valid)
        return std::max(0.05,o.cooperative_cpu_scale*batch_staging_costs.cpu_extra_ms[1]-batch_staging_costs.gpu_extra_ms[1]);
    return lane_cpu1>0 ? std::max(0.05,0.5*lane_cpu1) : 1.0;
}
double adaptive_h2d_ms() const {
    return adaptive_policy.h2d_ms>0 ? adaptive_policy.h2d_ms :
        (batch_staging_costs.valid ? batch_staging_costs.h2d_ms : 2.0);
}
double adaptive_d2h_ms() const {
    if(primary_backing_bank)return 0;
    return adaptive_policy.d2h_ms>0 ? adaptive_policy.d2h_ms :
        (batch_staging_costs.valid ? batch_staging_costs.d2h_ms : 2.0);
}

// early_layer >= 0: an existing routing join has retired preceding readers.
// Only retain already-ready copies from other layers: the current layer's
// ready snapshot was frozen before that join. Forward-end (-1) may also fill.
// Publication occurs separately at joined boundaries, never inside selection.
bool adaptive_issue_priced(Dev& d,int early_layer,std::string& err) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(!primary_adaptive || !generation_active)return true;
    if(!adaptive_blob) {err="adaptive primary is not prepared";return false;}
    if(adaptive_budget_fwd!=fwd) {adaptive_budget_fwd=fwd;adaptive_budget_used=0;}
    primary_admission_budget.advance(native_demand_rows,adaptive_policy.max_exchanges);
    const double began=now();
    const double benefit_per_score=(1.0-std::pow(double(kDecay),adaptive_policy.horizon))*adaptive_service_ms();
    if(early_layer<0 && primary_admission_budget.available() && adaptive_pending(d)==int(d.adaptive.size()))
        ++st.adaptive_no_slot; // An unfinished exchange can consume the reserved opportunity.
    for(size_t tx=0;tx<d.adaptive.size();++tx) {
        if(!primary_admission_budget.available(early_layer>=0))break;
        auto& x=d.adaptive[tx];if(x.incoming>=0)continue;
        if(early_layer>=0) {
            // Scores are nonnegative, so an incoming candidate that cannot
            // pay its own costs cannot beat even a zero-score victim.
            bool plausible=false;
            for(int s=0;s<d.n_slots;++s) {
                const long long key=d.slot_owner[s];
                if(key<0 || key/1024==early_layer || staging_reserved(d,s) || d.slot_foreground[s] ||
                   d.slot_pend[s]>=0 || adaptive_identity_pending(d,key) || helper_live_leased(key))continue;
                if(score(key,false)*benefit_per_score>adaptive_d2h_ms()+
                   adaptive_policy.interference_ms+adaptive_policy.margin_ms) {plausible=true;break;}
            }
            if(!plausible)break;
        }
        long long victim=-1;float cold=0;
        for(int li:d.layers) {
            auto& l=L[size_t(li)];if(!l.moe || l.hf)continue;
            for(int e:l.res_ids) {
                const long long key=static_cast<long long>(li)*1024+e;
                if(adaptive_identity_pending(d,key) || helper_live_leased(key))continue;
                const float value=score(key,false);
                if(victim<0 || value<cold) {victim=key;cold=value;}
            }
        }
        if(victim<0) {++st.adaptive_no_gain;break;}
        long long incoming=-1;int chosen_slot=-1;bool reuse=false;double best=0;
        const auto consider=[&](long long key,int ready_slot) {
            if(key<0 || key/1024==early_layer || adaptive_identity_pending(d,key) || helper_live_leased(key))return;
            auto& a=L[size_t(key/1024)];const int e=int(key%1024);
            if(a.res_slot[e]>=0 || !a.host_blob[e])return;
            for(int s=0;s<d.n_slots;++s)if(d.slot_pend[s]==key && d.slot_iss[s])return;
            const double gain=(score(key,false)-cold)*benefit_per_score-adaptive_d2h_ms()-
                (ready_slot>=0?0:adaptive_h2d_ms())-adaptive_policy.interference_ms-adaptive_policy.margin_ms;
            if(gain>best) {best=gain;incoming=key;chosen_slot=ready_slot;reuse=ready_slot>=0;}
        };
        // At a layer boundary do not scan the bank unless there is a frame to
        // retain. This bounds policy work and never adds a transfer just there.
        for(int s=0;s<d.n_slots;++s)if(!staging_reserved(d,s) && !d.slot_foreground[s] &&
            d.slot_pend[s]<0 && d.slot_owner[s]>=0)consider(d.slot_owner[s],s);
        if(early_layer<0)for(int li:d.layers) {
            auto& l=L[size_t(li)];if(!l.moe || l.hf)continue;
            for(int e=0;e<g.n_exp;++e)if(l.res_slot[e]<0 && l.host_blob[e]) {
                const long long key=static_cast<long long>(li)*1024+e;
                bool ready=false;
                for(int s=0;s<d.n_slots;++s)if(!staging_reserved(d,s) && !d.slot_foreground[s] &&
                    d.slot_pend[s]<0 && d.slot_owner[s]==key) {ready=true;break;}
                if(!ready)consider(key,-1);
            }
        }
        if(incoming<0) {++st.adaptive_no_gain;break;}
        // Release only unsubmitted promotions; issued operations hold their
        // source until normal completion. The optional-work budget is shared.
        for(int s=0;s<d.n_slots;++s)if(d.slot_pend[s]==incoming && !d.slot_iss[s]) {
            const int freed=adaptive_cancel_queued(d,s);if(chosen_slot<0)chosen_slot=freed;break;
        }
        if(d.n_pending+adaptive_pending(d)>=kMaxPending && adaptive_cancel_queued(d)<0) {
            ++st.adaptive_no_slot;break;
        }
        if(chosen_slot<0)for(int s=0;s<d.n_slots;++s)if(!staging_reserved(d,s) &&
            !d.slot_foreground[s] && d.slot_pend[s]<0 &&
            (chosen_slot<0 || d.slot_used[s]<d.slot_used[chosen_slot]))chosen_slot=s;
        if(chosen_slot<0)chosen_slot=adaptive_cancel_queued(d);
        if(chosen_slot<0) {++st.adaptive_no_slot;break;}
        auto& a=L[size_t(incoming/1024)];auto& b=L[size_t(victim/1024)];
        const int ae=int(incoming%1024),be=int(victim%1024);
        if(primary_backing_bank && (!primary_backing_for(int(victim/1024),be) ||
                                   primary_backing_for(int(incoming/1024),ae))) {
            err="adaptive backing ledger mismatch before submission";placement_failed=true;return false;
        }
        x.incoming=incoming;x.victim=victim;x.landing=chosen_slot;x.submitted=false;
        x.source_host=const_cast<uint8_t*>(a.host_blob[ae]);x.source_gpu=resident_address(b,b.res_slot[be]);
        x.landing_gpu=staging_address(d,chosen_slot);x.target_host=primary_backing_bank ? primary_backing_for(int(victim/1024),be) : adaptive_host_spare[tx];
        prefetch_forget(d,chosen_slot);d.slot_reserved[chosen_slot]=1;d.slot_owner[chosen_slot]=-1;
        auto status=cudaSuccess;
        if(!reuse)status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess && !primary_backing_bank)status=cudaMemcpyAsync(x.target_host,x.source_gpu,adaptive_blob,cudaMemcpyDeviceToHost,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.event,d.cs);
        if(status!=cudaSuccess)return adaptive_fail("adaptive copy/event submission",status,err);
        x.submitted=true;++st.adaptive_issued;++adaptive_budget_used;primary_admission_budget.spend();st.adaptive_reused+=reuse;
        if(capture::detailed()) {
            std::ostringstream out; out<<"\"incoming\":"<<incoming<<",\"victim\":"<<victim
                <<",\"incoming_score\":"<<score(incoming,false)<<",\"victim_score\":"<<cold
                <<",\"estimated_gain_ms\":"<<best<<",\"service_saved_ms\":"<<adaptive_service_ms()
                <<",\"h2d_price_ms\":"<<adaptive_h2d_ms()<<",\"d2h_price_ms\":"<<adaptive_d2h_ms()
                <<",\"admission_credit\":"<<primary_admission_budget.credit<<",\"demand_rows\":"<<native_demand_rows
                <<",\"interference_ms\":"<<adaptive_policy.interference_ms<<",\"margin_ms\":"<<adaptive_policy.margin_ms
                <<",\"horizon\":"<<adaptive_policy.horizon<<",\"budget_used\":"<<adaptive_budget_used
                <<",\"budget_max\":"<<adaptive_policy.max_exchanges<<",\"early_layer\":"<<early_layer
                <<",\"reuse\":"<<(reuse?1:0)<<",\"landing_slot\":"<<chosen_slot
                <<",\"source_host_frame\":"<<uintptr_t(x.source_host)<<",\"landing_frame\":"<<uintptr_t(x.landing_gpu)
                <<",\"victim_frame\":"<<uintptr_t(x.source_gpu)<<",\"target_host_frame\":"<<uintptr_t(x.target_host)
                <<",\"h2d_bytes\":"<<(reuse?0:adaptive_blob)<<",\"d2h_bytes\":"<<(primary_backing_bank?0:adaptive_blob)<<",\"status\":\"submitted\"";
            capture::detail_event("primary_exchange_issue",out.str());
        }
        st.adaptive_h2d_bytes+=reuse?0:adaptive_blob;st.adaptive_d2h_bytes+=primary_backing_bank?0:adaptive_blob;
    }
    st.adaptive_host_ms+=1000*(now()-began);
    return true;
}
bool adaptive_issue(Dev& d,std::string& err) {
    return adaptive_policy.legacy ? adaptive_issue_legacy(d,err) : adaptive_issue_priced(d,-1,err);
}
bool adaptive_retain_joined(Dev& d,int current_layer,std::string& err) {
    if(adaptive_policy.legacy || !primary_adaptive || !generation_active)return true;
    return adaptive_issue_priced(d,current_layer,err);
}
