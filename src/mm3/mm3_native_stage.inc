    // Two adjacent parts of one DSpark verification window. Pinned parity
    // buffers are retained until both streams drain, including all failures.
    bool stage_window(const int* ids,int n,int pos0,bool overlap,const std::function<bool(int)>& after_a,
                      int& completed_rows,float* logits,std::string& err) {
        if(hf_demand_poisoned) {err="native HF demand boundary previously failed";return false;}
        hf_demand_active = false;
        mm3::DemandBoundary demand_boundary{hf_demand_poisoned,hf_demand_active};
        completed_rows=0;
        if (stage_poisoned) {err="native stage retirement previously failed";return false;}
        if (o.native_stage_prefix!=3 || !native || D.size()!=2 || head_dev!=1 || !draftm || !draft_running ||
            !ids || !logits || n<1 || n>8 || pos0<0 || pos0+n>o.max_ctx || !rseq.empty()) {
            err="invalid native stage window"; return false;
        }
        Dev& front=D[0]; Dev& back=D[1];
        struct Drain {
            Dev& a; Dev& b; bool& poisoned; std::atomic<bool>& uncertain; bool finished=false; bool committed=false;
            bool finish() {
                if(finished) return !poisoned;
                finished=true; bool ok=true;
                if(cudaSetDevice(a.id)!=cudaSuccess) ok=false;
                else {if(cudaStreamSynchronize(a.s)!=cudaSuccess) ok=false;
                      if(a.s2 && cudaStreamSynchronize(a.s2)!=cudaSuccess) ok=false;}
                if(cudaSetDevice(b.id)!=cudaSuccess) ok=false;
                else {if(cudaStreamSynchronize(b.s)!=cudaSuccess) ok=false;
                      if(b.s2 && cudaStreamSynchronize(b.s2)!=cudaSuccess) ok=false;}
                if(!ok) {poisoned=true; uncertain.store(true);} return ok;
            }
            ~Drain(){if(!committed) {poisoned=true; uncertain.store(true);} (void)finish();}
        } drain{front,back,stage_poisoned,retirement_uncertain};
        if(!reset_hf_route_errors(err))return false;
        CK(cudaSetDevice(front.id));
        for(int i=0;i<2;++i)
            if(!stage_hidden[i] || !stage_ready[i]) {err="native stage handoff buffers are unavailable";return false;}
        long long front_tick[2] = {}, back_tick[2] = {};
        bool front_saved[2] = {}, back_saved[2] = {};
        auto run_prefix = [&](int slot,int start,int T)->bool {
            front_tick[slot]=fwd;
            CK(cudaSetDevice(front.id));
            cur_pos=pos0+start+T-1;
            if(!embed(ids+start,T,front.emb_h,err)) return false;
            for(int j=0;j<T;++j) front.pos_h[j]=pos0+start+j;
            CK(cudaMemcpyAsync(front.pos,front.pos_h,(size_t)T*4,cudaMemcpyHostToDevice,front.s));
            CK(cudaMemcpyAsync(front.h,front.emb_h,(size_t)T*g.dim*4,cudaMemcpyHostToDevice,front.s));
            for(int li=0;li<o.native_stage_prefix;++li) {
                if(!layer(li,T,cur_pos,err)) return false;
                k::submit(front.s);
            }
            CK(cudaMemcpyAsync(stage_hidden[slot],front.h,(size_t)T*g.dim*4,cudaMemcpyDeviceToHost,front.s));
            if(!queue_hf_demand(front,front.hf_stage_h[slot],err))return false;
            front_saved[slot]=true;
            CK(cudaEventRecord(stage_ready[slot],front.s));
            return true;
        };
        auto run_suffix = [&](int slot,int start,int T)->bool {
            CK(cudaSetDevice(front.id));
            const double wait_begin=now(); CK(cudaEventSynchronize(stage_ready[slot]));
            if(slot==1) st.stage_b_wait_ms+=1e3*(now()-wait_begin);
            CK(cudaSetDevice(back.id)); ++fwd; back_tick[slot]=fwd;
            if (!refresh_slots(back, err) || !helper_live_progress(back,err)) return false;
            cur_pos=pos0+start+T-1;
            for(int j=0;j<T;++j) back.pos_h[j]=pos0+start+j;
            CK(cudaMemcpyAsync(back.pos,back.pos_h,(size_t)T*4,cudaMemcpyHostToDevice,back.s));
            CK(cudaMemcpyAsync(back.h,stage_hidden[slot],(size_t)T*g.dim*4,cudaMemcpyHostToDevice,back.s));
            for(int li=o.native_stage_prefix;li<g.n_layers;++li) {
                if(!layer(li,T,pos0+start+T-1,err)) return false;
                k::submit(back.s);
                if(tap_of[(size_t)li]>=0) CK(cudaMemcpyAsync(taps+(size_t)tap_of[(size_t)li]*T*g.dim,
                    back.h,(size_t)T*g.dim*4,cudaMemcpyDeviceToDevice,back.s));
            }
            k::rms(back.h,norm1,T,g.dim,g.eps,back.xn,back.xn16,back.s);
            const int cols=std::max(1,x3::mv_cols(head));
            for(int c=0;c<T;c+=cols) {
                x3::MvJob j; j.m=&head;j.xf=back.xn+(size_t)c*g.dim;j.y=back.lg+(size_t)c*g.vocab;
                j.ncol=std::min(cols,T-c);j.xf_cs=g.dim;j.y_cs=g.vocab;
                if(!x3::mv(&j,1,back.s)){err="stage head";return false;}
            }
            CK(cudaMemcpyAsync(back.lg_h,back.lg,(size_t)T*g.vocab*4,cudaMemcpyDeviceToHost,back.s));
            if(!queue_hf_demand(back,back.hf_stage_h[slot],err) || !queue_hf_route_errors(back,err))return false;
            back_saved[slot]=true;
            CK(cudaStreamSynchronize(back.s));
            if(!check_hf_route_errors(back,err))return false;
            std::memcpy(logits+(size_t)start*g.vocab,back.lg_h,(size_t)T*g.vocab*4);
            if(!draftm->context(taps,T,pos0+start,err)) return false;
            ++st.stage_segments;st.stage_rows+=T;completed_rows+=T;
            return true;
        };
        // Keep the production ordering in a CUDA-independent scheduler so
        // failure/branch fixtures exercise these exact transitions.
        struct ScheduleOps {
            decltype(run_prefix)& prefix_fn; decltype(run_suffix)& suffix_fn;
            Dev& front; Stats& st; std::string& err;
            const std::function<bool(int)>& after_a; double submit_begin=0, suffix_begin=0;
            bool prefix(int slot,int start,int count) {return prefix_fn(slot,start,count);}
            bool suffix(int slot,int start,int count) {return suffix_fn(slot,start,count);}
            bool await_a() {
                CK(cudaSetDevice(front.id)); CK(cudaEventSynchronize(stage_event(0))); return true;
            }
            cudaEvent_t* events;
            cudaEvent_t stage_event(int slot) {return events[slot];}
            void begin_prefix_b() {submit_begin=now();}
            void submitted_prefix_b(int count) {st.stage_b_prefix_rows+=count;st.stage_prefix_b_submit_ms+=1e3*(now()-submit_begin);}
            bool observe_b_start() {
                CK(cudaSetDevice(front.id));const auto q=cudaEventQuery(stage_event(1));
                if(q==cudaErrorNotReady) {++st.stage_b_pending;(void)cudaGetLastError();}
                else if(q!=cudaSuccess) {err="stage B event query failed";return false;}
                return true;
            }
            void begin_suffix_a() {suffix_begin=now();}
            void end_suffix_a() {st.stage_suffix_a_ms+=1e3*(now()-suffix_begin);}
            bool observe_b_end() {
                CK(cudaSetDevice(front.id));const auto q=cudaEventQuery(stage_event(1));
                if(q==cudaSuccess)++st.stage_b_ready_before_a_end;
                else if(q==cudaErrorNotReady)(void)cudaGetLastError();
                else {err="stage B completion query failed";return false;}
                return true;
            }
            bool continue_b(int a) {return !after_a || after_a(a);}
            void serial_prefix_b(int n) {st.stage_b_prefix_rows+=n;}
            void completed_b(int n) {st.stage_b_suffix_completed_rows+=n;}
            void skipped_b(int n,bool overlap) {st.stage_b_suffix_skipped_rows+=n;if(overlap)st.stage_b_prefix_discarded_rows+=n;}
        } schedule{run_prefix,run_suffix,front,st,err,after_a,0,0,stage_ready};
        if(!native_stage_schedule(n,overlap,schedule))return false;
        if(!helper_live_progress(back,err) || !helper_live_issue(back,err))return false;
        for(Dev& d:D) if(!d.promo_q.empty()) {CK(cudaSetDevice(d.id));if (!issue_promotions(d, err)) return false;}
        if(!drain.finish()) {err="native stage stream retirement failed";return false;}
        if(const int e=x3::launch_errors()) {
            stage_poisoned=true;retirement_uncertain.store(true);
            err="native stage launch errors: "+std::to_string(e);return false;
        }
        // Both stage streams have retired. Consume each stream's cumulative
        // original-ID snapshot at its own submission tick, including B-prefix
        // work whose suffix was skipped. No optional promotion is joined here.
        for(int slot=0;slot<2;++slot) {
            if(front_saved[slot] && !consume_hf_demand(front,front.hf_stage_h[slot],front_tick[slot],err))return false;
            if(back_saved[slot] && !consume_hf_demand(back,back.hf_stage_h[slot],back_tick[slot],err))return false;
        }
        demand_boundary.complete=true;
        ++st.stage_windows;
        drain.committed=true;
        return true;
    }

