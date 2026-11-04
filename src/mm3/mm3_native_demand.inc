// Production routing observation, cache-reader protection and promotion scoring.
    bool locate(Layer& l, Dev& d, int e, std::vector<char>& busy, unsigned long long& ptr, std::string& err) {
        const int rs = l.res_slot[(size_t) e];
        if (rs >= 0) {
            ptr = (unsigned long long) resident_address(l, rs);
            return true;
        }
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        int at = -1;
        for (int s = 0; s < d.n_slots; ++s)
            if (foreground_usable(d,(size_t)s) && d.slot_owner[(size_t) s] == key) at = s;
        if(at>=0 && native)prefetch_note_use(d,at,l.blob);
        if (at >= 0 && !native) score(key, true);   // (a promoted expert's use keeps it)
        if (at < 0) {
            for (int s = 0; s < d.n_slots; ++s)
                if (!staging_reserved(d,s) && !busy[(size_t) s] && d.slot_pend[(size_t) s] < 0 && foreground_usable(d,(size_t)s) &&
                    (at < 0 || d.slot_used[(size_t) s] < d.slot_used[(size_t) at]))
                    at = s;
            if (at < 0) {
                err = "no free staging slot";
                return false;
            }
            if(native)prefetch_forget(d,at);
            if (!stage(l, d, e, at, err)) return false;
            d.slot_owner[(size_t) at] = key;
        }
        busy[(size_t) at] = 1;
        d.slot_used[(size_t) at] = fwd;
        ptr = (unsigned long long) staging_address(d,at);
        return true;
    }


    // Live scores start/reset at zero. Lazy aging remains 0.95 per forward.
    float score_at(long long key, long long at, uint32_t credit) {
        if (use_score.empty()) {
            use_score.assign((size_t) g.n_layers * 1024, 0.0f);
            use_last.assign(use_score.size(), 0);
        }
        float& sc = use_score[(size_t) key];
        sc *= std::pow(kDecay, (float) (at - use_last[(size_t) key]));
        use_last[(size_t) key] = at;
        sc += (float)credit;
        return sc;
    }
    float score(long long key, bool add) { return score_at(key, fwd, add ? 1u : 0u); }

    // the promotions' copies on the copy stream - at a forward's end: during it, each 20+ MB copy would hold up the
    // forward's own small copies behind it on the GPU's one host -> device engine (its plans, the CPU lane's parts)

    void consider_promotion(const Layer& l, Dev& d, int e) {
        if (prefetch_active(true) && !d.promo_q.empty()) return;
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        if (native && (adaptive_identity_pending(d,key) || helper_live_leased(key))) return;
        if(native && (std::find(d.slot_pend.begin(),d.slot_pend.end(),key)!=d.slot_pend.end() ||
                      std::find(d.slot_owner.begin(),d.slot_owner.end(),key)!=d.slot_owner.end()))return;
        const float sc = score(key, !native);
        if (!o.promote || l.host_blob.empty() || l.host_blob[(size_t) e] == nullptr || sc < kPromote ||
            d.n_pending + adaptive_pending(d) >= kMaxPending || d.n_slots == 0)
            return;
        int best = -1;
        float bs = sc;
        for (int s = 0; s < d.n_slots; ++s) {
            if(prefetch_active(true) && s<d.n_slots/2)continue;
            if (staging_reserved(d,s) || !r2::promotion_slot_available(d.slot_foreground[(size_t)s]!=0,d.slot_pend[(size_t)s]>=0,
                                              d.slot_used[(size_t)s],fwd)) continue;
            const long long ok = d.slot_owner[(size_t) s];
            const float os = ok < 0 ? -1.0f : score(ok, false);
            if (os < bs) {
                bs = os;
                best = s;
            }
        }
        if (best < 0) return;
        if(capture::detailed()) {std::ostringstream out;out<<"\"incoming\":"<<key<<",\"victim\":"<<d.slot_owner[best]
            <<",\"slot\":"<<best<<",\"destination_frame\":"<<uintptr_t(staging_address(d,best))
            <<",\"source_host_frame\":"<<uintptr_t(l.host_blob[e])<<",\"incoming_score\":"<<sc
            <<",\"victim_score\":"<<bs<<",\"bytes\":"<<l.blob<<",\"status\":\"queued\"";
            capture::detail_event("promotion_queue",out.str());}
        if(native)prefetch_forget(d,best);
        d.slot_owner[(size_t) best] = -1;
        d.slot_pend[(size_t) best] = key;
        d.slot_iss[(size_t) best] = 0;
        ++d.n_pending;
        d.promo_q.push_back({best, l.host_blob[(size_t) e], l.blob});
        ++st.promoted;
    }

    // the CPU lane's experts: the EXL3 lane's rows, or a GGUF's (mm3_cpu_gg.cpp) - one layer's experts a call

    void observe_routes(const Layer& l, Dev& d, const int* top, int T) {
        if (!native) return;
        auto ready = [](const uint32_t* bits,int e) { return (bits[e/32] >> (e%32)) & 1u; };
        bool missed[128]{};
        for (int i=0;i<T*g.k_exp;++i) {
            const int e=top[i], orig=d.original_h[i];
            if (orig>=0 && orig<g.n_exp) {
                ++original_prof[(size_t)(&l-L.data())*g.n_exp+orig];
                score((long long)(&l-L.data())*1024+orig, true);
                st.route_original_misses += !ready(d.ready_h->primary,orig) && !ready(d.ready_h->helper,orig);
            }
            bool originally_selected=false;
            const int row=(i/g.k_exp)*g.k_exp;
            for(int j=0;j<g.k_exp;++j) if(d.original_h[row+j]==e) originally_selected=true;
            st.route_substitutions += !originally_selected;
            if (ready(d.ready_h->primary,e)) ++st.route_primary_hits;
            else if (ready(d.ready_h->helper,e)) ++st.route_helper_hits;
            else if(!missed[e]) {
                missed[e]=true; ++st.miss_union; st.miss_union_bytes+=l.blob;
            }
            if (adaptive_main.size()==size_t(g.n_layers*g.n_exp) && l.res_slot[e]>=0 &&
                adaptive_main[size_t(&l-L.data())*g.n_exp+e]) ++st.adaptive_hits;
            // Protect all selected cached records before any CPU-demand
            // promotion chooses a replacement slot.
            const long long key=(&l-L.data())*1024+e;
            for(int j=0;j<d.n_slots;++j) if(r2::snapshot_visible(d.slot_foreground[j]!=0) && d.slot_owner[j]==key) d.slot_used[j]=fwd;
        }
    }

    void recover_originals(const Layer& l, Dev& d, const int* top, int T) {
        if (!native || !o.routing) return;
        bool seen[128]{};
        for(int i=0;i<T*g.k_exp;++i) {
            const int e=d.original_h[i];
            if(e<0||e>=128||seen[e])continue;
            seen[e]=true;
            if(std::find(top,top+T*g.k_exp,e)==top+T*g.k_exp && on_cpu(l,d,e)) consider_promotion(l,d,e);
        }
    }

    // Called by window_post for both ordinary T=1 and grouped T>1. The
    // caller builds its real selected plan between these two transitions.
    void begin_window_demand(const Layer& l, Dev& d, const int* top, int T) {
        if(native)prefetch_note_selection(l,d,top,T);
        observe_routes(l,d,top,T);
    }
    void finish_window_demand(const Layer& l, Dev& d, const int* top, int T,
                              const std::vector<int>& cpu_of) {
        if(native) for(int e=0;e<g.n_exp;++e) if(cpu_of[(size_t)e]>=0) consider_promotion(l,d,e);
        recover_originals(l,d,top,T);
    }

    // whether expert e of layer l is the CPU lane's this token: neither in VRAM nor in one of d's staging slots

    // D2H is enqueued on the consuming stream before an existing join. It never
    // waits on the optional promotion stream. Each snapshot represents one
    // forward tick (or one recorded stage segment) and carries that tick.
    bool queue_hf_demand(Dev& d, uint32_t* host, std::string& err) {
        if (!native || !d.hf_original_d) return true;
        if (!host) { err="native HF demand snapshot storage missing"; return false; }
        CK(cudaSetDevice(d.id));
        CK(cudaMemcpyAsync(host,d.hf_original_d,(size_t)g.n_layers*g.n_exp*sizeof(uint32_t),
                           cudaMemcpyDeviceToHost,d.s));
        return true;
    }
    bool consume_hf_demand(Dev& d, const uint32_t* host, long long at, std::string& err) {
        if (!native || !d.hf_original_d) return true;
        if (!d.hf_demand_seen.consume(host,(size_t)g.n_layers,(size_t)g.n_exp,at,
            [&](size_t li,size_t e,uint32_t n,long long tick) {
                score_at((long long)li*1024+(long long)e,tick,n);
            })) {err="native HF demand snapshot geometry mismatch";return false;}
        return true;
    }
    // Only used at the explicit joined placement/reset boundary. Bring the
    // cumulative device baseline current before the caller zeroes live scores.
    bool baseline_hf_demand(std::string& err) {
        if (hf_demand_poisoned) {err="native HF demand boundary previously failed";return false;}
        for (Dev& d : D) if (!queue_hf_demand(d,d.hf_snapshot_h,err)) return false;
        for (Dev& d : D) if (native && d.hf_original_d) {
            CK(cudaSetDevice(d.id)); CK(cudaStreamSynchronize(d.s));
            if (!consume_hf_demand(d,d.hf_snapshot_h,fwd,err)) return false;
        }
        return true;
    }
