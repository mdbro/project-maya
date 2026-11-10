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
    // Every reassignment of a frame passes here: an early copy that was never read is counted unused.
    if(!d.slot_early_fresh.empty() && d.slot_early_fresh[size_t(s)]) {d.slot_early_fresh[size_t(s)]=0;++st.early_copy_unused;}
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
        if(!busy[size_t(s)] && !staging_reserved(d,s) && !d.slot_foreground[size_t(s)] && !slot_early_busy(d,s) &&
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
// MM3_PREDICT_DIAG=1: a predict-only diagnostic of the cross-layer prediction above. During
// single-sequence generation it runs prefetch_predict and prefetch_readback under any residency
// mode, adaptive included, never admits or issues an upload, and logs each layer's prediction
// (made at the previous layer) on that layer's routes record for offline accuracy.
bool predict_diag_on() const {
    return predict_diag && generation_active && native && D.size()==1 && !o.routing_prefetch && rseq.empty();
}
bool predict_diag_ready(Dev& d,std::string& err) {
    if (d.prediction_logits && d.prediction_ids && d.prediction_weights && d.prediction_host) return true;
    if (!d.prediction_logits && !dalloc(d.id,d.prediction_logits,size_t(kWin)*g.n_exp,err)) return false;
    if (!d.prediction_ids && !dalloc(d.id,d.prediction_ids,size_t(kWin)*g.k_exp,err)) return false;
    if (!d.prediction_weights && !dalloc(d.id,d.prediction_weights,size_t(kWin)*g.k_exp,err)) return false;
    if (!d.prediction_host) CK(cudaHostAlloc(&d.prediction_host,size_t(kWin)*g.k_exp*8,cudaHostAllocPortable));
    return true;
}
// Called after the layer's join, which completed the readback made in this layer's prepare step:
// the host buffer then holds this layer's prediction for the next one. The next layer's prepare
// step overwrites it, so it is kept here until that layer's routes record.
void predict_diag_rotate(const Dev& d,int li,int T) {
    predict_diag_this.clear();predict_diag_this_layer=-1;
    if(!predict_diag_on())return;
    if(predict_diag_next_layer==li && predict_diag_next_fwd==static_cast<long long>(fwd)) {
        predict_diag_this.swap(predict_diag_next);predict_diag_this_layer=li;
    }
    predict_diag_next.clear();predict_diag_next_layer=-1;
    const size_t next=size_t(li)+1;
    if(next<L.size() && !L[size_t(li)].hf && L[next].moe && !L[next].hf && L[next].dev==L[size_t(li)].dev && d.prediction_host) {
        const int* ids=reinterpret_cast<const int*>(d.prediction_host);
        predict_diag_next.assign(ids,ids+size_t(T)*g.k_exp);
        predict_diag_next_layer=int(next);predict_diag_next_fwd=static_cast<long long>(fwd);
    }
}
// early_copy (experimental A/B control, off by default). At T=1 the next layer's likely misses are known a layer
// early: the prediction above applies layer l+1's router to layer l's input, and holds 91% of layer l+1's misses
// (predict-diag-20261108). When layer l has queued no demand copy (mode 2: at most one) and nothing else of ours is
// pending on the copy engines (no exchange, no promotion), the first predicted HOST expert of layer l+1 is copied on the copy stream
// into the least recently used free staging frame, fenced behind this layer's own uploads. The frame is owned at
// once, so layer l+1's routing sees the expert as ready and the planner serves it from the frame; its reader waits
// on the frame's event (locate). Until that event is observed complete the frame is no one else's to take.
bool slot_early_busy(const Dev& d,int s) const {return !d.slot_early.empty() && d.slot_early[size_t(s)];}
bool early_copy_on() const {
    return o.early_copy && native && o.native_cooperative && generation_active && D.size()==1 && rseq.empty() &&
           !o.routing_prefetch && D[0].cs && !D[0].slot_early.empty();
}
bool early_copy_ready(Dev& d,std::string& err) {
    if(!predict_diag_ready(d,err))return false;
    if(!d.prediction_uploads) CK(cudaEventCreateWithFlags(&d.prediction_uploads,cudaEventDisableTiming));
    return true;
}
// MM3_EARLY_COPY_VERIFY: waits for the frame's copy, reads the frame back and compares it with the HOST record it came from.
// A difference is a placement failure: the frame was written by something other than its early copy.
bool early_copy_check(const Layer& l,Dev& d,int e,int s,std::string& err) {
    if(early_copy_check_buffer.size()<l.blob)early_copy_check_buffer.resize(l.blob);
    cudaError_t status=cudaEventSynchronize(d.slot_ev[size_t(s)]);
    if(status==cudaSuccess)status=cudaMemcpy(early_copy_check_buffer.data(),staging_address(d,s),l.blob,cudaMemcpyDeviceToHost);
    if(status!=cudaSuccess) {
        err=std::string("early copy verification failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    if(l.host_blob.empty() || !l.host_blob[size_t(e)] || std::memcmp(early_copy_check_buffer.data(),l.host_blob[size_t(e)],l.blob)!=0) {
        err="early copy frame differs from its HOST record";placement_failed=true;retirement_uncertain.store(true);return false;
    }
    ++st.early_copy_verified;
    return true;
}
// Query-only, at a layer's join: an early copy observed complete leaves an ordinary staged frame.
bool early_copy_retire(Dev& d,std::string& err) {
    for(int s=0;s<d.n_slots;++s) if(slot_early_busy(d,s)) {
        const cudaError_t status=cudaEventQuery(d.slot_ev[size_t(s)]);
        if(status==cudaSuccess) d.slot_early[size_t(s)]=0;
        else if(status!=cudaErrorNotReady) {
            err=std::string("early copy completion failed: ")+cudaGetErrorString(status);
            placement_failed=true;retirement_uncertain.store(true);return false;
        }
    }
    return true;
}
// The early copy's prediction at T=1: the next layer's router product, its top-k and its ids to the host, captured once per
// layer into a graph on the router's side stream, so it costs the host one launch instead of four calls; the side stream's
// join before the routing readback covers it. Any capture failure falls back to the separate calls for that layer.
// Detailed capture only: the forward and layer whose prediction the capture scratch's second row holds (early_copy_issue
// records it). The real routing at one row writes the first row only.
long long early_prediction_fwd=-1;
int early_prediction_layer=-1;
bool early_copy_predict(Layer& l,Dev& d,cudaStream_t xs,std::string& err) {
    const size_t li=size_t(&l-L.data());
    if(li+1>=L.size())return true;
    Layer& next=L[li+1];
    if(!next.moe || next.hf || next.dev!=l.dev)return true;
    // Inside prep_graph's capture the calls join the preparation's graph (captures cannot nest), and the ids go down by a
    // kernel on the pinned buffer's device view, as the routing does; the host reads them after the same join.
    cudaStreamCaptureStatus capturing=cudaStreamCaptureStatusNone;
    if(cudaStreamIsCapturing(xs,&capturing)!=cudaSuccess) {(void)cudaGetLastError();capturing=cudaStreamCaptureStatusNone;}
    if(capturing==cudaStreamCaptureStatusActive) {
        k::rows_dot(next.gate,g.n_exp,g.dim,d.xn,1,d.prediction_logits,xs);
        cudaError_t status=cudaPeekAtLastError();
        if(status==cudaSuccess)status=resident::route(d.prediction_logits,next.bias,1,resident::Policy::Off,0,nullptr,
                                                     d.prediction_ids,d.prediction_weights,nullptr,nullptr,xs);
        if(status==cudaSuccess) {
            if(prep_zero_copy && d.prediction_host_dev) {
                k::copy_words(d.prediction_ids,d.prediction_host_dev,size_t(g.k_exp)*4,xs);
                status=cudaPeekAtLastError();
            } else {
                status=cudaMemcpyAsync(d.prediction_host,d.prediction_ids,size_t(g.k_exp)*4,cudaMemcpyDeviceToHost,xs);
            }
        }
        if(status!=cudaSuccess) {err=std::string("early copy prediction in the preparation's graph: ")+cudaGetErrorString(status);return false;}
        return true;   // (counted at each launch of the graph: prep_graph_run)
    }
    if(capture::detailed() && d.capture_scores_d && d.capture_scores_h) {   // (never inside a capture: handled above)
        // Detailed capture: the same calls launched directly (no graph), the router also writing its score triplets
        // (sigmoid, sigmoid + bias, the same) for all 128 experts into the capture scratch's second row, and those to the
        // host on the same side stream, so early_copy_issue can record the whole prediction. The ids are unchanged.
        float* scores=d.capture_scores_d+size_t(g.n_exp)*3;
        k::rows_dot(next.gate,g.n_exp,g.dim,d.xn,1,d.prediction_logits,xs);
        cudaError_t status=cudaPeekAtLastError();
        if(status==cudaSuccess)status=resident::route(d.prediction_logits,next.bias,1,resident::Policy::Off,0,nullptr,
                                                     d.prediction_ids,d.prediction_weights,nullptr,nullptr,xs,false,scores);
        if(status==cudaSuccess)status=cudaMemcpyAsync(d.prediction_host,d.prediction_ids,size_t(g.k_exp)*4,cudaMemcpyDeviceToHost,xs);
        if(status==cudaSuccess)status=cudaMemcpyAsync(d.capture_scores_h+size_t(g.n_exp)*3,scores,size_t(g.n_exp)*3*sizeof(float),
                                                      cudaMemcpyDeviceToHost,xs);
        if(status!=cudaSuccess) {err=std::string("early copy prediction (detailed capture): ")+cudaGetErrorString(status);return false;}
        early_prediction_fwd=static_cast<long long>(fwd);early_prediction_layer=int(li);
        ++st.prefetch_predictions;
        return true;
    }
    if(d.prediction_graphs.size()!=L.size()) {d.prediction_graphs.assign(L.size(),nullptr);d.prediction_graph_failed.assign(L.size(),0);}
    cudaGraphExec_t& exec=d.prediction_graphs[li];
    const auto calls=[&](cudaStream_t s)->cudaError_t {
        k::rows_dot(next.gate,g.n_exp,g.dim,d.xn,1,d.prediction_logits,s);
        cudaError_t status=cudaPeekAtLastError();
        if(status==cudaSuccess)status=resident::route(d.prediction_logits,next.bias,1,resident::Policy::Off,0,nullptr,
                                                     d.prediction_ids,d.prediction_weights,nullptr,nullptr,s);
        if(status==cudaSuccess)status=cudaMemcpyAsync(d.prediction_host,d.prediction_ids,size_t(g.k_exp)*4,cudaMemcpyDeviceToHost,s);
        return status;
    };
    if(!exec && !d.prediction_graph_failed[li]) {
        cudaGraph_t graph=nullptr;
        cudaError_t status=cudaStreamBeginCapture(xs,cudaStreamCaptureModeThreadLocal);
        if(status==cudaSuccess) {
            const cudaError_t captured=calls(xs);
            status=cudaStreamEndCapture(xs,&graph);
            if(status==cudaSuccess)status=captured;
            if(status==cudaSuccess)status=cudaGraphInstantiate(&exec,graph,0);
            if(graph)cudaGraphDestroy(graph);
        }
        if(status!=cudaSuccess) {exec=nullptr;d.prediction_graph_failed[li]=1;(void)cudaGetLastError();}
    }
    if(exec) {CK(cudaGraphLaunch(exec,xs));++st.prefetch_predictions;return true;}
    CK(calls(xs));
    ++st.prefetch_predictions;
    return true;
}
// Called after layer l's plan upload is queued on d.s and before its experts. busy holds this layer's readers.
// Detailed capture only: the layer's prediction for the next one and what became of it, for offline re-decision (the
// re-routed replay). base is the router's sigmoid + bias for all 128 experts, as its own capture computed them.
void early_copy_record(const Layer& l,const Dev& d,long staged_before,const char* outcome,int pick,int slot) {
    if(!capture::detailed())return;
    const size_t li=size_t(&l-L.data());
    const int* ids=reinterpret_cast<const int*>(d.prediction_host);
    std::ostringstream out;out.precision(9);
    out<<"\"target_layer\":"<<li+1<<",\"outcome\":\""<<outcome<<"\",\"expert\":"<<pick<<",\"slot\":"<<slot
       <<",\"demand_copies\":"<<(st.staged-staged_before)<<",\"exchanges_pending\":"<<adaptive_pending(d)<<",\"predicted\":[";
    for(int i=0;i<g.k_exp;++i){if(i)out<<',';out<<ids[i];}
    out<<"],\"base\":";
    if(early_prediction_layer==int(li) && early_prediction_fwd==static_cast<long long>(fwd) && d.capture_scores_h) {
        const float* sc=d.capture_scores_h+size_t(g.n_exp)*3;
        out<<'[';
        for(int e=0;e<g.n_exp;++e){if(e)out<<',';out<<sc[size_t(e)*3+1];}
        out<<']';
    } else out<<"null";
    capture::detail_event("early_copy_prediction",out.str());
}
bool early_copy_issue(Layer& l,Dev& d,int T,const std::vector<char>& busy,long staged_before,std::string& err) {
    if(!early_copy_on() || T!=1 || l.hf || !d.prediction_host || !d.prediction_uploads)return true;
    const size_t li=size_t(&l-L.data());
    if(li+1>=L.size())return true;
    Layer& n=L[li+1];
    if(!n.moe || n.hf || n.dev!=l.dev || n.host_blob.empty())return true;
    // diagnostic mode 3: the prediction runs and is read back, nothing is copied
    if(o.early_copy==3) {early_copy_record(l,d,staged_before,"mode3",-1,-1);return true;}
    // Mode 1 copies only on an idle copy engine; mode 2 also behind this layer's one demand copy.
    // (exchange_schedule: an exchange holds the copy stream only while its upload may be in flight; its victim's
    //  download runs on its own stream, in idle time)
    if(st.staged-staged_before>(o.early_copy>=2 ? 1 : 0) || (exchange_schedule ? adaptive_copy_pending(d) : adaptive_pending(d)) ||
       d.n_pending || !d.promo_q.empty()) {++st.early_copy_blocked;early_copy_record(l,d,staged_before,"blocked",-1,-1);return true;}
    const int* ids=reinterpret_cast<const int*>(d.prediction_host);
    int pick=-1;
    for(int i=0;i<g.k_exp && pick<0;++i) {
        const int e=ids[i];
        if(e<0 || e>=g.n_exp) {early_copy_record(l,d,staged_before,"malformed",-1,-1);return true;}   // a malformed prediction issues nothing
        const long long key=static_cast<long long>(li+1)*1024+e;
        if(n.res_slot[size_t(e)]>=0 || helper_owns(n,e) || !n.host_blob[size_t(e)] || helper_live_leased(key) ||
           adaptive_identity_pending(d,key) ||
           std::find(d.slot_owner.begin(),d.slot_owner.end(),key)!=d.slot_owner.end() ||
           std::find(d.slot_pend.begin(),d.slot_pend.end(),key)!=d.slot_pend.end())continue;
        pick=e;
    }
    if(pick<0) {early_copy_record(l,d,staged_before,"no_candidate",-1,-1);return true;}
    int slot=-1;
    for(int s=0;s<d.n_slots;++s)
        if(!busy[size_t(s)] && !staging_reserved(d,s) && !d.slot_foreground[size_t(s)] && d.slot_pend[size_t(s)]<0 &&
           !slot_early_busy(d,s) && (slot<0 || d.slot_used[size_t(s)]<d.slot_used[size_t(slot)]))slot=s;
    if(slot<0) {++st.early_copy_no_frame;early_copy_record(l,d,staged_before,"no_frame",pick,-1);return true;}
    const long long key=static_cast<long long>(li+1)*1024+pick;
    const long long displaced=d.slot_owner[size_t(slot)];
    // The layer join retired every earlier reader on d.s; this layer's readers are in busy. The fence orders the copy
    // after this layer's plan upload, and this layer's experts, submitted after it, run beside the copy.
    // With a demand copy queued on this layer (mode 2), the fence keeps the early copy behind it, so the demand copy keeps the
    // link to itself; with none, nothing on d.s concerns the chosen frame and the copy goes at once (two host calls, not four).
    cudaError_t status=cudaSuccess;
    if(st.staged!=staged_before || o.early_copy_legacy) {
        status=cudaEventRecord(d.prediction_uploads,d.s);
        if(status==cudaSuccess)status=cudaStreamWaitEvent(d.cs,d.prediction_uploads,0);
    }
    if(status==cudaSuccess)status=cudaMemcpyAsync(staging_address(d,slot),n.host_blob[size_t(pick)],n.blob,cudaMemcpyHostToDevice,d.cs);
    if(status==cudaSuccess)status=cudaEventRecord(d.slot_ev[size_t(slot)],d.cs);
    if(status!=cudaSuccess) {
        err=std::string("early copy submission failed: ")+cudaGetErrorString(status);
        placement_failed=true;retirement_uncertain.store(true);return false;
    }
    prefetch_forget(d,slot);
    d.slot_owner[size_t(slot)]=key;d.slot_used[size_t(slot)]=fwd;
    d.slot_early[size_t(slot)]=1;d.slot_early_fresh[size_t(slot)]=1;
    ++st.early_copies;st.early_copy_bytes+=static_cast<long long>(n.blob);
    if(capture::detailed()) {
        std::ostringstream out;out<<"\"layer\":"<<li+1<<",\"expert\":"<<pick<<",\"slot\":"<<slot<<",\"displaced\":"<<displaced
            <<",\"source_host_frame\":"<<uintptr_t(n.host_blob[size_t(pick)])<<",\"destination_frame\":"<<uintptr_t(staging_address(d,slot))
            <<",\"bytes\":"<<n.blob<<",\"status\":\"submitted\"";
        capture::detail_event("early_copy",out.str());
        early_copy_record(l,d,staged_before,"copied",pick,slot);
    }
    return true;
}
