    // Routing readback joins only the primary stream. Optional copy-stream
    // work remains independently leased, including on failure.
    bool owner_window_join(Dev& d, std::string& err) {
        const cudaError_t rc=cudaStreamSynchronize(d.s);
        if(rc!=cudaSuccess) {
            err=std::string("owner window join failed: ")+cudaGetErrorString(rc);
            placement_failed=true;retirement_uncertain.store(true);return false;
        }
        return true;
    }
    // The existing routing readback join has retired earlier foreground readers.
    // Each pinned descriptor remains immutable until the next such join.
    bool owner_window(Layer& l, Dev& d, int T, const int* top, const float* weights,
                      x3::WinArgs a, std::string& err) {
        struct Group { int expert; unsigned long long ptr; std::vector<int> entries; };
        std::vector<Group> ready, missing;
        const size_t li=(size_t)(&l-L.data());
        std::vector<Group> all;
        for(int t=0;t<T;++t) for(int k=0;k<g.k_exp;++k) {
            const int e=top[t*g.k_exp+k];
            auto it=std::find_if(all.begin(),all.end(),[&](const Group& x){return x.expert==e;});
            if(it==all.end()) { all.push_back({e,0,{}}); it=all.end()-1; }
            it->entries.push_back(t*8+k);
            ++prof[li*g.n_exp+e];
        }
        st.profiled+=(long)T*g.k_exp;
        st.owner_expert_groups+=(long)all.size();
        if(!retire_foreground_slots(d,err)) {placement_failed=true;retirement_uncertain.store(true);return false;}
        for(auto& x:all) {
            if(l.res_slot[x.expert]>=0) x.ptr=(unsigned long long)resident_address(l,l.res_slot[x.expert]);
            else for(int s=0;s<d.n_slots;++s) if(!staging_reserved(d,s) && d.slot_pend[s]<0 &&
                    foreground_usable(d,(size_t)s) && d.slot_owner[s]==(long long)li*1024+x.expert) {
                x.ptr=(unsigned long long)staging_address(d,s); d.slot_used[s]=fwd;
                prefetch_note_use(d,s,l.blob); break;
            }
            (x.ptr ? ready : missing).push_back(x);
        }
        int available=0;
        for(int s=0;s<d.n_slots;++s)
            available+=!staging_reserved(d,s) && d.slot_pend[s]<0 && foreground_usable(d,(size_t)s);
        if(!missing.empty() && !available) {err="owner window has no available foreground frame";return false;}
        std::vector<int> host_serviced((size_t)g.n_exp,-1);
        for(const auto& x:missing)host_serviced[(size_t)x.expert]=0;
        const size_t off_w=(size_t)kWin*8*8, off_gp=off_w+(size_t)kWin*8*4,
                     off_gs=off_gp+(size_t)kWin*8*8, off_ge=off_gs+((size_t)kWin*8+1)*4,
                     off_ng=off_ge+(size_t)kWin*8*4;
        int pass=0;
        auto submit=[&](const std::vector<Group>& groups)->bool {
            if(groups.empty())return true;
            if(pass>=65) {err="owner window plan capacity exceeded";return false;}
            uint8_t* p=d.owner_plan_h+(size_t)pass*win_plan_bytes();
            std::memset(p,0,win_plan_bytes());
            auto* pp=(unsigned long long*)p; auto* pw=(float*)(p+off_w);
            auto* gp=(unsigned long long*)(p+off_gp); auto* gs=(int*)(p+off_gs);
            auto* ge=(int*)(p+off_ge); auto* ng=(int*)(p+off_ng);
            // Full combination mask and weights are identical in every pass.
            for(int t=0;t<T;++t)for(int k=0;k<g.k_exp;++k) {
                int entry=t*8+k;pp[entry]=1;pw[entry]=weights[t*g.k_exp+k]*g.scale;
            }
            int n=0,pieces=0;
            for(size_t i=0;i<groups.size();++i) {
                gp[i]=groups[i].ptr;gs[i]=n;
                for(int e:groups[i].entries)ge[n++]=e;
                pieces+=int((groups[i].entries.size()+7)/8);
            }
            gs[groups.size()]=n;*ng=(int)groups.size();
            cudaError_t rc=cudaMemcpyAsync(d.win_plan_d,p,win_plan_bytes(),cudaMemcpyHostToDevice,d.s);
            if(rc!=cudaSuccess || !ne::window_scatter(a,pieces,d.s)) {
                err="owner window plan/scatter submission failed";placement_failed=true;retirement_uncertain.store(true);return false;
            }
            ++pass;++st.owner_passes;return true;
        };
        if(!submit(ready))return false;
        for(size_t at=0;at<missing.size();) {
            // Ready cached readers were submitted above. Each later pass's
            // overwrites follow the preceding scatter on the same stream.
            // Reset only this pass's busy set; locate retains its usual LRU
            // and pending/foreground ownership rules.
            std::vector<char> busy((size_t)d.n_slots,0);
            std::vector<Group> batch;
            for(int j=0;j<available && at<missing.size();++j,++at) {
                auto x=missing[at];
                if(!locate(l,d,x.expert,busy,x.ptr,err)) {
                    placement_failed=true;retirement_uncertain.store(true);return false;
                }
                batch.push_back(std::move(x));
            }
            if(!submit(batch))return false;
        }
        finish_window_demand(l,d,top,T,host_serviced);
        if(!ne::window_combine(a,d.s)) {err="owner window combine submission failed";placement_failed=true;retirement_uncertain.store(true);return false;}
        k::add(d.h,d.out,(size_t)T*g.dim,d.s);pmark(d,kPExperts);
        return true;
    }
