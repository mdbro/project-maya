// Production queue owner methods, shared verbatim with the CPU fixture.
bool helper_queue_begin(Layer& l,Dev& d,int T,cudaStream_t stream,std::string& err) {
    if(placement_failed || retirement_uncertain.load() || !helper_queue || helper_queue_pending || T<1 || T>kWin || helper_queue_next==UINT64_MAX) {
        err="queued helper unavailable or ticket exhausted";return false;
    }
    std::array<NativeHelper::QueueIdentity,128> snap{};
    if(!helper->queue_snapshot(int(&l-L.data()),snap,err))return false;
    const auto pool=helper->pool_identity(),stride=helper->record_stride();
    for(int e=0;e<g.n_exp;++e) {
        const bool ready=helper_owns(l,e);
        if(ready!=(snap[e].slot>=0)) {err="queued helper ready snapshot differs from canonical ownership";return false;}
        auto& rec=helper_queue_table_h[e];rec={};
        if(ready){rec.helper_record=pool+size_t(snap[e].slot)*stride;
            std::copy_n(snap[e].multiplier,3,rec.multiplier);}
    }
    const auto seq=++helper_queue_next;
    if(!helper_queue->schedule(seq,T,err)) {
        if(!helper_queue->storage_safe())retirement_uncertain.store(true);
        placement_failed=true;return false;
    }
    helper_queue_pending=seq;
    CK(cudaMemcpyAsync(helper_queue_table_d,helper_queue_table_h,g.n_exp*sizeof(nf::QueueRecord),cudaMemcpyHostToDevice,stream));
    if(!helper_queue_launch_picks(d,T,stream,err))return false;
    return helper_queue->publish(seq,helper_queue_picks_d,helper_queue_table_d,g.n_exp,d.xn,stream,err);
}
bool helper_queue_finish(int T,float* result,std::string& err) {
    if(!helper_queue || !helper_queue_pending){err="queued helper request missing";return false;}
    const bool ok=helper_queue->finish(helper_queue_pending,result,size_t(T)*g.dim,10000,err);
    if(helper_queue->storage_safe())helper_queue_pending=0;
    else retirement_uncertain.store(true);
    if(!ok)placement_failed=true;
    return ok;
}
void helper_queue_cancel_pending() {
    if(!helper_queue || !helper_queue_pending)return;
    std::string ignored;std::vector<float> discard(size_t(kWin)*g.dim);
    (void)helper_queue->abort(helper_queue_pending,ignored);
    (void)helper_queue->finish(helper_queue_pending,discard.data(),discard.size(),10000,ignored);
    if(helper_queue->storage_safe())helper_queue_pending=0;
    else retirement_uncertain.store(true);
    placement_failed=true;
}
// An internal failed queued forward is never resumed, even if cancellation
// proves both writers retired. An uncertain writer retains the entire Model.
bool helper_queue_failed_window(bool queued_route) {
    helper_queue_cancel_pending();if(queued_route)placement_failed=true;return false;
}
bool helper_queue_destroy_safe() const {
    return !retirement_uncertain.load() && (!helper_queue || helper_queue->storage_safe());
}
