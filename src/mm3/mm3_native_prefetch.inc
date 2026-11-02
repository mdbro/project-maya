// Bounded next-layer prediction for the regular single-device window path.
// The buffers are prepared only when explicitly enabled and retained for A/B.
bool prefetch_supported() const {
    return native && native_pool && D.size()==1 && o.seqs==1 && !o.tp && o.fast_decode &&
           o.chunk<=kWin && o.promote && !graph_on && !primary_adaptive &&
           !vis && helper && helper_dispatch_current=="worker" && D[0].n_slots==8 && D[0].cs;
}
bool prefetch_active(bool regular) const {
    return regular && o.routing_prefetch && o.routing && generation_active &&
           prefetch_supported() && rseq.empty();
}
bool prefetch_prepare(std::string& err) {
    if (!prefetch_supported()) {err="routing prefetch requires fixed single-sequence native window worker serving with eight staging frames";return false;}
    Dev& d=D[0];
    CK(cudaSetDevice(d.id));
    if (!d.prediction_logits && !dalloc(d.id,d.prediction_logits,size_t(kWin)*g.n_exp,err)) return false;
    if (!d.prediction_ids && !dalloc(d.id,d.prediction_ids,size_t(kWin)*g.k_exp,err)) return false;
    if (!d.prediction_weights && !dalloc(d.id,d.prediction_weights,size_t(kWin)*g.k_exp,err)) return false;
    if (!d.prediction_host) CK(cudaHostAlloc(&d.prediction_host,size_t(kWin)*g.k_exp*8,cudaHostAllocPortable));
    if (!d.prediction_uploads) CK(cudaEventCreateWithFlags(&d.prediction_uploads,cudaEventDisableTiming));
    if (d.slot_prediction.empty()) d.slot_prediction.assign(size_t(d.n_slots),0);
    return true;
}
// Retire only an unsubmitted descriptor. In-flight copies retain their frame
// and HOST source regardless of prediction priority.
bool prefetch_cancel_one(Dev& d,bool prediction_only=false) {
    for (auto it=d.promo_q.begin();it!=d.promo_q.end();++it) {
        const int s=it->slot;
        if (s<0 || s>=d.n_slots || d.slot_iss[size_t(s)] || d.slot_pend[size_t(s)]<0 ||
            staging_reserved(d,s) || (prediction_only &&
            (d.slot_prediction.empty() || d.slot_prediction[size_t(s)]==0))) continue;
        d.promo_q.erase(it);d.slot_pend[size_t(s)]=-1;d.slot_owner[size_t(s)]=-1;
        d.slot_iss[size_t(s)]=0;--d.n_pending;
        if (!d.slot_prediction.empty()) d.slot_prediction[size_t(s)]=0;
        ++st.prefetch_cancelled;return true;
    }
    return false;
}
void prefetch_trim_queue(Dev& d) {
    while (d.promo_q.size()>1 && prefetch_cancel_one(d)) {}
}
// An unissued prediction targeted this layer. It can no longer arrive before
// this layer's ready snapshot and must not drift into a later layer or token.
bool prefetch_begin_layer(Dev& d,std::string& err) {
    while(prefetch_cancel_one(d,true)) {}
    prefetch_trim_queue(d);
    return publish_slots(d,err);
}
void prefetch_end_generation() {
    if(o.routing_prefetch)for(auto& d:D)while(prefetch_cancel_one(d,true)) {}
}
void prefetch_forget(Dev& d,int s) {
    if (d.slot_prediction.empty()) return;
    const auto state=d.slot_prediction[size_t(s)];
    if (state==2 || state==4) {++st.prefetch_unused;st.prefetch_unused_bytes+=d.slot_bytes;}
    d.slot_prediction[size_t(s)]=0;
}
void prefetch_external_cancel(Dev& d,int s) {
    if(!d.slot_prediction.empty() && d.slot_prediction[size_t(s)])++st.prefetch_cancelled;
    prefetch_forget(d,s);
}
// Explicit maintenance has joined both streams. A submitted pending copy is
// complete there; an unsubmitted descriptor was never transferred.
void prefetch_discard_joined(Dev& d) {
    for(size_t s=0;s<d.slot_prediction.size();++s) {
        if(d.slot_prediction[s]==1 && d.slot_iss[s]) {
            ++st.prefetch_unused;st.prefetch_unused_bytes+=d.slot_bytes;
        } else prefetch_forget(d,int(s));
        d.slot_prediction[s]=0;
    }
}
void prefetch_note_selection(const Layer& l,Dev& d,const int* top,int T) {
    if(d.slot_prediction.empty())return;
    const long long base=static_cast<long long>(&l-L.data())*1024;
    for(int s=0;s<d.n_slots;++s) {
        auto& state=d.slot_prediction[size_t(s)];
        if(!state || state==3)continue;
        const long long key=d.slot_pend[size_t(s)]>=0?d.slot_pend[size_t(s)]:d.slot_owner[size_t(s)];
        if(key<base || key>=base+g.n_exp)continue;
        const int e=int(key-base);
        if(std::find(top,top+T*g.k_exp,e)==top+T*g.k_exp)continue;
        if(d.slot_pend[size_t(s)]>=0 && d.slot_iss[size_t(s)] && state==1)
            {state=4;++st.prefetch_late;st.prefetch_late_bytes+=l.blob;}
    }
}
// Attribution is made at the actual primary-cache service caller, after the
// helper and permanent-resident lanes have had their normal precedence.
void prefetch_note_use(Dev& d,int s,size_t bytes) {
    if(d.slot_prediction.empty())return;
    auto& state=d.slot_prediction[size_t(s)];
    if(state==2 || state==4) {state=3;++st.prefetch_useful;st.prefetch_useful_bytes+=bytes;}
}
// The existing successful d.s join precedes this call. It proves only the
// previous primary-stream readers and foreground writes retired; d.cs and all
// helper/adaptive source leases remain independently reserved.
void prefetch_retire_joined_readers(Dev& d) {
    std::fill(d.slot_foreground.begin(),d.slot_foreground.end(),0);
    std::fill(d.slot_used.begin(),d.slot_used.end(),fwd-1);
}
bool prefetch_join_readers(Dev& d,std::string& err) {
    const cudaError_t status=cudaStreamSynchronize(d.s);
    if(status!=cudaSuccess) {
        err=std::string("prefetch reader join failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    prefetch_retire_joined_readers(d);return true;
}
bool prefetch_control_join(cudaStream_t stream,std::string& err) {
    const cudaError_t status=cudaStreamSynchronize(stream);
    if(status!=cudaSuccess) {
        err=std::string("prefetch control join failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    return true;
}
// The regular window has two side-stream branches. Their event handoffs are
// part of the reader-retirement proof, so a failed record/wait quarantines the
// model instead of allowing a later d.s join to claim that s2 was joined.
bool prefetch_stream_fork(Dev& d,cudaStream_t& stream,std::string& err) {
    stream=d.s;
    if(!d.s2)return true;
    cudaError_t status=cudaEventRecord(d.fev,d.s);
    if(status==cudaSuccess)status=cudaStreamWaitEvent(d.s2,d.fev,0);
    if(status!=cudaSuccess) {
        err=std::string("prefetch side-stream fork failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    stream=d.s2;return true;
}
bool prefetch_stream_join(Dev& d,std::string& err) {
    if(!d.s2)return true;
    cudaError_t status=cudaEventRecord(d.jev,d.s2);
    if(status==cudaSuccess)status=cudaStreamWaitEvent(d.s,d.jev,0);
    if(status!=cudaSuccess) {
        err=std::string("prefetch side-stream join failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    return true;
}
// Every native window uses the same checked handoffs, including routing and
// prefetch off. Legacy nonnative callers retain their existing behavior.
bool native_window_fork(Dev& d,cudaStream_t& stream,std::string& err) {
    if(native)return prefetch_stream_fork(d,stream,err);
    stream=fork(d);return true;
}
bool native_window_join(Dev& d,std::string& err) {
    if(native)return prefetch_stream_join(d,err);
    join(d);return true;
}
bool prefetch_predict(Layer& l,Dev& d,int T,cudaStream_t stream,std::string& err) {
    const size_t li=size_t(&l-L.data());
    if (li+1>=L.size()) return true;
    Layer& next=L[li+1];
    if (!next.moe || next.hf || next.dev!=l.dev) return true;
    k::rows_dot(next.gate,g.n_exp,g.dim,d.xn,T,d.prediction_logits,stream);
    CK(cudaPeekAtLastError());
    CK(resident::route(d.prediction_logits,next.bias,T,resident::Policy::Off,0,nullptr,
                       d.prediction_ids,d.prediction_weights,nullptr,nullptr,stream));
    ++st.prefetch_predictions;st.prefetch_rows+=T;
    return true;
}
bool prefetch_readback(Layer& l,Dev& d,int T,std::string& err) {
    const size_t li=size_t(&l-L.data());
    if (li+1>=L.size() || !L[li+1].moe || L[li+1].hf || L[li+1].dev!=l.dev) return true;
    const size_t n=size_t(T)*g.k_exp*4;
    CK(cudaMemcpyAsync(d.prediction_host,d.prediction_ids,n,cudaMemcpyDeviceToHost,d.s));
    CK(cudaMemcpyAsync(d.prediction_host+size_t(kWin)*g.k_exp*4,d.prediction_weights,n,cudaMemcpyDeviceToHost,d.s));
    return true;
}
// One selection for the entire group. Its source must be canonical HOST and
// neither ready nor pending; no demand score is read or changed here.
void prefetch_admit(Layer& l,Dev& d,int T,const std::vector<char>& busy) {
    const size_t li=size_t(&l-L.data());
    if (li+1>=L.size())return;
    Layer& n=L[li+1];if(!n.moe || n.hf || n.dev!=l.dev)return;
    const double began=now();
    const int* ids=reinterpret_cast<const int*>(d.prediction_host);
    const float* weights=reinterpret_cast<const float*>(d.prediction_host+size_t(kWin)*g.k_exp*4);
    std::array<double,128> support{};
    for (int i=0;i<T*g.k_exp;++i) {
        if(ids[i]<0 || ids[i]>=g.n_exp || !std::isfinite(weights[i]) || weights[i]<0) return;
        support[size_t(ids[i])]+=weights[i];
    }
    int best=-1;
    for(int e=0;e<g.n_exp;++e) {
        const long long key=static_cast<long long>(li+1)*1024+e;
        if(support[size_t(e)]<=0 || n.res_slot[size_t(e)]>=0 || helper_owns(n,e) ||
           n.host_blob.empty() || !n.host_blob[size_t(e)] || helper_live_leased(key) ||
           adaptive_identity_pending(d,key) ||
           std::find(d.slot_owner.begin(),d.slot_owner.end(),key)!=d.slot_owner.end() ||
           std::find(d.slot_pend.begin(),d.slot_pend.end(),key)!=d.slot_pend.end())continue;
        if(best<0 || support[size_t(e)]>support[size_t(best)])best=e;
    }
    if(best<0) {st.prefetch_policy_ms+=(now()-began)*1000;return;}
    // Existing prefill copies may still be in flight. Do not add another
    // queued descriptor when the four-record global bound is already full.
    if (!d.promo_q.empty()) prefetch_cancel_one(d);
    if (d.n_pending+adaptive_pending(d)>=kMaxPending) {st.prefetch_policy_ms+=(now()-began)*1000;return;}
    int slot=-1;
    for(int s=d.n_slots/2;s<d.n_slots;++s)
        if(!busy[size_t(s)] && !staging_reserved(d,s) && !d.slot_foreground[size_t(s)] &&
           d.slot_pend[size_t(s)]<0 && d.slot_used[size_t(s)]!=fwd &&
           (slot<0 || d.slot_used[size_t(s)]<d.slot_used[size_t(slot)]))slot=s;
    if(slot>=0) {
        prefetch_forget(d,slot);
        d.slot_owner[size_t(slot)]=-1;d.slot_pend[size_t(slot)]=static_cast<long long>(li+1)*1024+best;
        d.slot_iss[size_t(slot)]=0;d.slot_prediction[size_t(slot)]=1;++d.n_pending;
        d.promo_q.push_back({slot,n.host_blob[size_t(best)],n.blob});++st.prefetch_admitted;
    }
    st.prefetch_policy_ms+=(now()-began)*1000;
}
// Called after current foreground H2D uploads have been enqueued, before its
// experts. One event fences exactly those uploads and no optional DMA is put
// behind an outstanding optional copy.
bool prefetch_issue_one(Dev& d,std::string& err) {
    if(d.promo_q.empty())return true;
    if(adaptive_pending(d))return true;
    for(int s=0;s<d.n_slots;++s) if(d.slot_pend[size_t(s)]>=0 && d.slot_iss[size_t(s)])return true;
    if(d.promo_q.size()!=1) {err="prefetch optional queue is not bounded";promotion_error=err;return false;}
    return r2::issue(d.promo_q,d.slot_iss,promotion_error,err,[&](const Dev::Promo& q,std::string& error) {
        cudaError_t status=cudaEventRecord(d.prediction_uploads,d.s);
        if(status==cudaSuccess)status=cudaStreamWaitEvent(d.cs,d.prediction_uploads,0);
        if(status==cudaSuccess)status=cudaMemcpyAsync(staging_address(d,q.slot),q.src,q.bytes,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(d.slot_ev[size_t(q.slot)],d.cs);
        if(status!=cudaSuccess) {error=std::string("prefetch optional submission failed: ")+cudaGetErrorString(status);return false;}
        if(d.slot_prediction[size_t(q.slot)]==1)st.prefetch_issued_bytes+=q.bytes;
        return true;
    });
}
