    // The two bounded handoff slots and their queued work belong to one lease.
    // No caller pointers survive an API call.
    struct StageState {
        uint64_t ticket=0;
        int ids[8]={}, n=0, pos=0, completed=0;
        bool submitted[2]={}, ready[2]={}, suffix[2]={}, suffix_launched[2]={};
        bool front_saved[2]={}, back_saved[2]={};
        long long front_tick[2]={}, back_tick[2]={};
        int suffix_slot=-1, suffix_next_layer=0;
        bool suffix_submitted_all=false, suffix_ready=false;
        double suffix_begin=0;
        bool active=false;
    } stage_state;

    bool stage_drain() noexcept {
        bool ok=true;
        for(Dev& d:D) {
            if(cudaSetDevice(d.id)!=cudaSuccess) {ok=false;continue;}
            if(cudaStreamSynchronize(d.s)!=cudaSuccess)ok=false;
            if(d.s2 && cudaStreamSynchronize(d.s2)!=cudaSuccess)ok=false;
        }
        return ok;
    }
    void stage_fault() noexcept {
        stage_poisoned=true;retirement_uncertain.store(true);
        if(hf_demand_active)hf_demand_poisoned=true;
        (void)stage_drain();
    }
    struct StageGuard {
        bool& poisoned; std::atomic<bool>& uncertain; bool& demand_poisoned; bool& demand_active;
        std::vector<Dev>& devices; bool committed=false;
        ~StageGuard() noexcept {
            if(committed)return;
            poisoned=true;uncertain.store(true);
            if(demand_active)demand_poisoned=true;
            for(Dev& d:devices) {
                if(cudaSetDevice(d.id)!=cudaSuccess)continue;
                (void)cudaStreamSynchronize(d.s);
                if(d.s2)(void)cudaStreamSynchronize(d.s2);
            }
        }
    };
    bool stage_check(uint64_t ticket,std::string& err) const {
        if(stage_poisoned || retirement_uncertain.load() || hf_demand_poisoned) {
            err="native stage ownership is poisoned";return false;
        }
        if(!stage_state.active || !ticket || stage_state.ticket!=ticket) {err="stale native stage ticket";return false;}
        return true;
    }
    int stage_count(int slot) const {return slot==0?std::min(4,stage_state.n):stage_state.n-std::min(4,stage_state.n);}
    int stage_start(int slot) const {return slot==0?0:std::min(4,stage_state.n);}

    bool stage_begin(const int* ids,int n,int pos,uint64_t& ticket,std::string& err) {
        ticket=0;
        if(stage_poisoned || retirement_uncertain.load() || hf_demand_poisoned) {err="native stage ownership is poisoned";return false;}
        if(stage_state.active) {err="native stage lease is already active";return false;}
        if(o.native_stage_prefix!=3 || !native || D.size()!=2 || head_dev!=1 || !draftm || !draft_running ||
           !ids || n<1 || n>8 || pos<0 || pos>o.max_ctx-n || !rseq.empty()) {
            err="invalid native stage window";return false;
        }
        for(int i=0;i<2;++i)if(!stage_hidden[i] || !stage_ready[i]) {err="native stage handoff buffers are unavailable";return false;}
        static std::atomic<uint64_t> next{0};
        uint64_t previous=next.load();
        do {
            if(previous==UINT64_MAX) {err="native stage ticket space exhausted";return false;}
        } while(!next.compare_exchange_weak(previous,previous+1));
        const uint64_t generation=previous+1;
        stage_state=StageState{};stage_state.active=true;stage_state.ticket=generation;stage_state.n=n;stage_state.pos=pos;
        std::copy(ids,ids+n,stage_state.ids);
        hf_demand_active=false;
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        if(!reset_hf_route_errors(err))return false;
        ticket=generation;guard.committed=true;return true;
    }
    bool stage_prefix_submit(uint64_t ticket,int slot,std::string& err) {
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || !stage_count(slot) || stage_state.submitted[slot] ||
           (slot==1 && (!stage_state.submitted[0] || !stage_state.ready[0]))) {
            err="invalid native stage prefix order";return false;
        }
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& front=D[0];const int start=stage_start(slot), T=stage_count(slot);
        const double submit_begin=now();
        stage_state.front_tick[slot]=fwd;
        CK(cudaSetDevice(front.id));
        cur_pos=stage_state.pos+start+T-1;
        if(!embed(stage_state.ids+start,T,front.emb_h,err))return false;
        for(int j=0;j<T;++j)front.pos_h[j]=stage_state.pos+start+j;
        CK(cudaMemcpyAsync(front.pos,front.pos_h,(size_t)T*4,cudaMemcpyHostToDevice,front.s));
        CK(cudaMemcpyAsync(front.h,front.emb_h,(size_t)T*g.dim*4,cudaMemcpyHostToDevice,front.s));
        for(int li=0;li<o.native_stage_prefix;++li) {
            if(!layer(li,T,cur_pos,err))return false;
            k::submit(front.s);
        }
        CK(cudaMemcpyAsync(stage_hidden[slot],front.h,(size_t)T*g.dim*4,cudaMemcpyDeviceToHost,front.s));
        if(!queue_hf_demand(front,front.hf_stage_h[slot],err))return false;
        stage_state.front_saved[slot]=true;
        CK(cudaEventRecord(stage_ready[slot],front.s));
        stage_state.submitted[slot]=true;
        if(slot==1) {st.stage_b_prefix_rows+=T;if(!stage_state.suffix[0])st.stage_prefix_b_submit_ms+=1e3*(now()-submit_begin);}
        guard.committed=true;return true;
    }
    bool stage_prefix_ready(uint64_t ticket,int slot,bool wait,bool& ready,std::string& err) {
        ready=false;
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || !stage_state.submitted[slot]) {err="native stage prefix was not submitted";return false;}
        if(stage_state.ready[slot]) {ready=true;return true;}
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        CK(cudaSetDevice(D[0].id));
        if(wait) {
            const double begin=now();CK(cudaEventSynchronize(stage_ready[slot]));
            if(slot==1)st.stage_b_wait_ms+=1e3*(now()-begin);
            ready=true;
        } else {
            auto q=cudaEventQuery(stage_ready[slot]);
            if(q==cudaSuccess)ready=true;
            else if(q==cudaErrorNotReady)(void)cudaGetLastError();
            else {err="native stage event query failed";return false;}
        }
        stage_state.ready[slot]=ready;guard.committed=true;return true;
    }
    // One foreground owner advances the suffix; each service call submits one
    // layer. All CUDA operations and their host-side state remain serialized.
    bool stage_suffix_launch(uint64_t ticket,int slot,std::string& err) {
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || !stage_state.submitted[slot] || stage_state.suffix[slot] ||
           stage_state.suffix_slot!=-1 || (slot==1 && !stage_state.suffix[0])) {
            err="invalid native stage suffix order";return false;
        }
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& front=D[0];Dev& back=D[1];const int start=stage_start(slot),T=stage_count(slot);
        if(slot==0 && stage_state.submitted[1]) {
            CK(cudaSetDevice(front.id));auto q=cudaEventQuery(stage_ready[1]);
            if(q==cudaErrorNotReady) {++st.stage_b_pending;(void)cudaGetLastError();}
            else if(q!=cudaSuccess) {err="stage B event query failed";return false;}
        }
        stage_state.suffix_begin=now();
        bool ready=false;if(!stage_prefix_ready(ticket,slot,true,ready,err))return false;
        CK(cudaSetDevice(back.id));++fwd;stage_state.back_tick[slot]=fwd;
        if(!refresh_slots(back,err) || !helper_live_progress(back,err))return false;
        cur_pos=stage_state.pos+start+T-1;
        for(int j=0;j<T;++j)back.pos_h[j]=stage_state.pos+start+j;
        CK(cudaMemcpyAsync(back.pos,back.pos_h,(size_t)T*4,cudaMemcpyHostToDevice,back.s));
        CK(cudaMemcpyAsync(back.h,stage_hidden[slot],(size_t)T*g.dim*4,cudaMemcpyHostToDevice,back.s));
        stage_state.suffix_slot=slot;stage_state.suffix_launched[slot]=true;
        stage_state.suffix_next_layer=o.native_stage_prefix;
        stage_state.suffix_submitted_all=false;stage_state.suffix_ready=false;
        guard.committed=true;return true;
    }
    bool stage_suffix_service(uint64_t ticket,int slot,bool& submitted_all,std::string& err) {
        submitted_all=false;
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || stage_state.suffix_slot!=slot || stage_state.suffix_submitted_all) {
            err="invalid native stage suffix service";return false;
        }
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& back=D[1];const int start=stage_start(slot),T=stage_count(slot);
        CK(cudaSetDevice(back.id));
        cur_pos=stage_state.pos+start+T-1;
        const int li=stage_state.suffix_next_layer;
        if(!layer(li,T,cur_pos,err))return false;
        k::submit(back.s);
        if(tap_of[(size_t)li]>=0)CK(cudaMemcpyAsync(taps+(size_t)tap_of[(size_t)li]*T*g.dim,
            back.h,(size_t)T*g.dim*4,cudaMemcpyDeviceToDevice,back.s));
        ++stage_state.suffix_next_layer;
        if(stage_state.suffix_next_layer==g.n_layers) {
            k::rms(back.h,norm1,T,g.dim,g.eps,back.xn,back.xn16,back.s);
            const int cols=std::max(1,x3::mv_cols(head));
            for(int c=0;c<T;c+=cols) {
                x3::MvJob j;j.m=&head;j.xf=back.xn+(size_t)c*g.dim;j.y=back.lg+(size_t)c*g.vocab;
                j.ncol=std::min(cols,T-c);j.xf_cs=g.dim;j.y_cs=g.vocab;
                if(!x3::mv(&j,1,back.s)){err="stage head";return false;}
            }
            CK(cudaMemcpyAsync(back.lg_h,back.lg,(size_t)T*g.vocab*4,cudaMemcpyDeviceToHost,back.s));
            if(!queue_hf_demand(back,back.hf_stage_h[slot],err) || !queue_hf_route_errors(back,err))return false;
            stage_state.back_saved[slot]=true;
            stage_state.suffix_submitted_all=true;submitted_all=true;
        }
        guard.committed=true;return true;
    }
    bool stage_suffix_ready(uint64_t ticket,int slot,bool wait,bool& ready,std::string& err) {
        ready=false;
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || stage_state.suffix_slot!=slot) {err="native stage suffix was not launched";return false;}
        if(!stage_state.suffix_submitted_all)return true;
        if(stage_state.suffix_ready) {ready=true;return true;}
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& back=D[1];CK(cudaSetDevice(back.id));
        if(wait) {CK(cudaStreamSynchronize(back.s));ready=true;}
        else {
            const auto q=cudaStreamQuery(back.s);
            if(q==cudaSuccess)ready=true;
            else if(q==cudaErrorNotReady)(void)cudaGetLastError();
            else {err="native stage suffix query failed";return false;}
        }
        stage_state.suffix_ready=ready;guard.committed=true;return true;
    }
    bool stage_suffix_finish(uint64_t ticket,int slot,float* logits,size_t capacity,std::string& err) {
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || stage_state.suffix_slot!=slot || !stage_state.suffix_submitted_all ||
           !stage_state.suffix_ready || !logits || capacity<(size_t)stage_count(slot)*g.vocab) {
            err="native stage suffix is not ready or output capacity is invalid";return false;
        }
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& front=D[0];Dev& back=D[1];const int start=stage_start(slot),T=stage_count(slot);
        CK(cudaSetDevice(back.id));
        if(!check_hf_route_errors(back,err))return false;
        std::memcpy(logits,back.lg_h,(size_t)T*g.vocab*4);
        if(!draftm->context(taps,T,stage_state.pos+start,err))return false;
        ++st.stage_segments;st.stage_rows+=T;stage_state.completed+=T;stage_state.suffix[slot]=true;
        stage_state.suffix_slot=-1;
        if(slot==1)st.stage_b_suffix_completed_rows+=T;
        if(slot==0) {
            st.stage_suffix_a_ms+=1e3*(now()-stage_state.suffix_begin);
            if(stage_state.submitted[1]) {
                CK(cudaSetDevice(front.id));auto q=cudaEventQuery(stage_ready[1]);
                if(q==cudaSuccess)++st.stage_b_ready_before_a_end;
                else if(q==cudaErrorNotReady)(void)cudaGetLastError();
                else {err="stage B completion query failed";return false;}
            }
        }
        guard.committed=true;return true;
    }
    bool stage_suffix_execute(uint64_t ticket,int slot,float* logits,size_t capacity,std::string& err) {
        if(!stage_check(ticket,err))return false;
        if(slot<0 || slot>1 || !logits || capacity<(size_t)stage_count(slot)*g.vocab) {
            err="invalid native stage suffix output";return false;
        }
        if(!stage_suffix_launch(ticket,slot,err))return false;
        bool all=false;while(!all)if(!stage_suffix_service(ticket,slot,all,err))return false;
        bool ready=false;if(!stage_suffix_ready(ticket,slot,true,ready,err))return false;
        return stage_suffix_finish(ticket,slot,logits,capacity,err);
    }
    bool stage_finish(uint64_t ticket,int& completed,std::string& err) {
        completed=0;if(!stage_check(ticket,err))return false;
        completed=stage_state.completed;
        StageGuard guard{stage_poisoned,retirement_uncertain,hf_demand_poisoned,hf_demand_active,D};
        Dev& back=D[1];
        // Cancellation may leave a partially submitted suffix. Capture its
        // original demand on the same stream before draining and consuming it.
        if(stage_state.suffix_slot!=-1 && !stage_state.back_saved[stage_state.suffix_slot]) {
            const int slot=stage_state.suffix_slot;
            CK(cudaSetDevice(back.id));
            if(!queue_hf_demand(back,back.hf_stage_h[slot],err) || !queue_hf_route_errors(back,err))return false;
            stage_state.back_saved[slot]=true;
        }
        if(!helper_live_progress(back,err) || !helper_live_issue(back,err))return false;
        for(Dev& d:D)if(!d.promo_q.empty()) {CK(cudaSetDevice(d.id));if(!issue_promotions(d,err))return false;}
        if(!stage_drain()) {err="native stage stream retirement failed";return false;}
        if(stage_state.suffix_slot!=-1) {
            CK(cudaSetDevice(back.id));
            if(!check_hf_route_errors(back,err))return false;
        }
        if(const int e=x3::launch_errors()) {err="native stage launch errors: "+std::to_string(e);return false;}
        for(int slot=0;slot<2;++slot) {
            if(stage_state.front_saved[slot] && !consume_hf_demand(D[0],D[0].hf_stage_h[slot],stage_state.front_tick[slot],err))return false;
            if(stage_state.back_saved[slot] && !consume_hf_demand(back,back.hf_stage_h[slot],stage_state.back_tick[slot],err))return false;
        }
        const int b=stage_count(1);
        if(b && !stage_state.suffix[1]) {
            if(stage_state.suffix[0] && !stage_state.suffix_launched[1])st.stage_b_suffix_skipped_rows+=b;
            if(stage_state.submitted[1])st.stage_b_prefix_discarded_rows+=b;
        }
        ++st.stage_windows;stage_state.active=false;hf_demand_active=false;
        guard.committed=true;return true;
    }
    bool stage_window(const int* ids,int n,int pos,bool overlap,const std::function<bool(int)>& after_a,
                      int& completed,float* logits,std::string& err) {
        completed=0;if(!logits) {err="invalid native stage output";return false;}
        uint64_t ticket=0;if(!stage_begin(ids,n,pos,ticket,err))return false;
        try {
            using OwnerPointer=decltype(this);
            struct Ops {
                OwnerPointer self;uint64_t ticket;float* logits;std::string& err;int& completed;
                const std::function<bool(int)>& after_a;
                bool prefix(int slot,int,int) {return self->stage_prefix_submit(ticket,slot,err);}
                bool suffix(int slot,int start,int count) {
                    const bool ok=self->stage_suffix_execute(ticket,slot,logits+(size_t)start*self->g.vocab,(size_t)count*self->g.vocab,err);
                    completed=self->stage_state.completed;return ok;
                }
                bool await_a() {bool ready=false;return self->stage_prefix_ready(ticket,0,true,ready,err);}
                void begin_prefix_b() {} void submitted_prefix_b(int) {} bool observe_b_start() {return true;}
                void begin_suffix_a() {} void end_suffix_a() {} bool observe_b_end() {return true;}
                bool continue_b(int a) {return !after_a || after_a(a);}
                void serial_prefix_b(int) {} void completed_b(int) {} void skipped_b(int,bool) {}
            } op{this,ticket,logits,err,completed,after_a};
            if(!native_stage_schedule(n,overlap,op))return false;
            return stage_finish(ticket,completed,err);
        } catch(...) {stage_fault();throw;}
    }
