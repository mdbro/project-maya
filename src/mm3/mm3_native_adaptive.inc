// Included inside Model::Impl. Native, single-target-GPU main-bank replacement.
// Sources remain canonical until both asynchronous copies have completed. A
// completed transaction rotates pointers; it never overwrites a live last copy.
bool primary_adaptive = false;
size_t adaptive_blob = 0;
std::array<uint8_t*,2> adaptive_host_owned{}, adaptive_host_spare{};
std::vector<uint8_t> adaptive_main;
std::vector<uintptr_t> adaptive_gpu_frames; // immutable original main+staging frames

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
            adaptive_host_contains(a,adaptive_blob);
    };
    for(auto a:h)if(!in_host(a))return false;
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

// Called only between target forwards, or after an explicit placement join.
// Queries never wait, spin, or change credit. Submitted descriptors retain every
// source and destination until successful publication or model quarantine.
bool adaptive_publish(Dev& d,std::string& err) {
    if(placement_failed || retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    for(size_t i=0;i<d.adaptive.size();++i) {
        auto& x=d.adaptive[i];if(x.incoming<0)continue;
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
           staging_address(d,x.landing)!=x.landing_gpu || adaptive_host_spare[i]!=x.target_host) {
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
        a.host_blob[ae]=nullptr;b.host_blob[be]=x.target_host;
        adaptive_host_spare[i]=x.source_host;
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
    const int s=best->slot;d.promo_q.erase(best);
    d.slot_pend[s]=-1;d.slot_iss[s]=0;d.slot_owner[s]=-1;--d.n_pending;
    ++st.adaptive_cancelled_promotions;return s;
}

// At the existing joined target-forward end, before ordinary queued promotion
// submissions. At most two swaps and four optional fills total are outstanding.
// Admission occurs in generation; the existing scores include original demand
// from both prefill and generation, with 0.95 decay per target-forward call.
bool adaptive_issue(Dev& d,std::string& err) {
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
        x.incoming=incoming;x.victim=victim;x.landing=slot;x.submitted=false;
        x.source_host=const_cast<uint8_t*>(a.host_blob[ae]);
        x.source_gpu=resident_address(b,b.res_slot[be]);x.landing_gpu=staging_address(d,slot);
        x.target_host=adaptive_host_spare[tx];
        d.slot_reserved[slot]=1;d.slot_owner[slot]=-1;
        auto status=cudaSuccess;
        if(!reuse)status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess)status=cudaMemcpyAsync(x.target_host,x.source_gpu,adaptive_blob,cudaMemcpyDeviceToHost,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.event,d.cs);
        if(status!=cudaSuccess)return adaptive_fail("adaptive copy/event submission",status,err);
        x.submitted=true;++st.adaptive_issued;
        st.adaptive_h2d_bytes+=reuse?0:adaptive_blob;st.adaptive_d2h_bytes+=adaptive_blob;
        st.adaptive_reused+=reuse;
    }
    st.adaptive_host_ms+=1000*(now()-began);
    return true;
}
