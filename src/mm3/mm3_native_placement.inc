// Included inside Model::Impl. Joined calibration changes identities in the
// existing allocations; it does not add a generation-time admission policy.
bool placement_failed = false;

uint8_t* resident_address(const Layer& l, int slot) const {
    return l.res_ptr.empty() ? l.res_base + size_t(slot) * l.blob : l.res_ptr.at(size_t(slot));
}

bool placement(const std::vector<uint32_t>& counts, bool primary, std::string& report, std::string& err) {
    if(helper_queue_pending) {
        helper_queue_cancel_pending();
        err="placement encountered an outstanding queued helper request; engine cannot resume";return false;
    }
    if(!promotion_error.empty()) {
        err=promotion_error;placement_failed=true;return false;
    }
    if (!native || !native_pool || o.chunk > 16 || placement_failed || retirement_uncertain.load()) {
        err = "native placement requires a healthy, complete native HOST bank and chunk <=16";
        return false;
    }
    if (!counts.empty() && (counts.size() != size_t(g.n_layers)*g.n_exp ||
        std::none_of(counts.begin(), counts.end(), [](uint32_t n) { return n != 0; }))) {
        err = "placement profile has wrong geometry or no generation demand"; return false;
    }
    const double started = now();
    const int total = g.n_layers * g.n_exp;
    auto key = [&](int li, int e) { return li*g.n_exp+e; };
    auto eligible_helper = [&](int li) { return L[li].moe && L[li].dev == head_dev; };
    auto helper_bits = [&] {
        for (int li=0; li<g.n_layers; ++li)
            L[li].helper_ready = helper ? helper->ready(li) : std::array<uint32_t,4>{};
    };
    auto ownership = [&]() {
        std::vector<uintptr_t> hosts, residents;
        size_t n = 0;
        for (int li=0; li<g.n_layers; ++li) {
            const auto& l=L[li]; if(!l.moe) continue;
            if(l.host_blob.size()!=size_t(g.n_exp) || l.res_slot.size()!=size_t(g.n_exp)) return false;
            for(int e=0;e<g.n_exp;++e) {
                const bool p=l.res_slot[e]>=0, h=helper_owns(l,e), b=l.host_blob[e]!=nullptr;
                if(int(p)+int(h)+int(b)!=1) return false;
                if(p) {
                    const int sl=l.res_slot[e];
                    if(sl>=int(l.res_ids.size()) || l.res_ids[sl]!=e) return false;
                    residents.push_back(uintptr_t(resident_address(l,sl)));
                }
                if(b) {
                    const auto a=uintptr_t(l.host_blob[e]), begin=uintptr_t(pool);
                    if((a<begin || a+host_record_bytes(l)>begin+native_pool->size()) && !adaptive_host_contains(a,host_record_bytes(l))) return false;
                    hosts.push_back(a);
                }
                ++n;
            }
        }
        std::sort(hosts.begin(),hosts.end()); std::sort(residents.begin(),residents.end());
        return n==size_t(st.experts) && hosts.size()==size_t(st.pooled) &&
            std::adjacent_find(hosts.begin(),hosts.end())==hosts.end() &&
            std::adjacent_find(residents.begin(),residents.end())==residents.end() && adaptive_storage_ok(hosts,residents);
    };
    if(!scale_drain(err))return false;
    // Explicit placement/inspection is outside generation. Complete the live
    // helper ledger before checking its stable HOST/helper cardinalities.
    if(helper_live_a>=0) {
        for(auto& d:D) {
            if(cudaSetDevice(d.id)!=cudaSuccess) {err="live helper placement device failure";retirement_uncertain.store(true);return false;}
            for(auto stream:{d.s,d.s2,d.cs}) if(stream && cudaStreamSynchronize(stream)!=cudaSuccess) {
                err="live helper placement reader retirement failed";retirement_uncertain.store(true);return false;
            }
            std::fill(d.slot_foreground.begin(),d.slot_foreground.end(),0);
        }
        if(!helper_live_settle(err))return false;
    }
    helper_bits();
    if(!ownership()) {err="incomplete or duplicate exclusive placement before refresh";placement_failed=true;return false;}
    std::vector<int> old_primary, old_helper;
    for(int li=0;li<g.n_layers;++li) for(int e:L[li].res_ids) old_primary.push_back(key(li,e));
    if(helper) for(const auto& x:helper->records()) old_helper.push_back(key(x.layer,x.expert));
    uint64_t bytes=0;
    size_t scratch_bytes=0;
    uint8_t *incoming=nullptr, *outgoing=nullptr;
    bool unsafe=false, mutated=false;
    // Error ownership is conservative: an uncertain CUDA operation retains both
    // portable buffers and the whole model. A failed mutation poisons execution.
    auto failed = [&]() {
        placement_failed = placement_failed || mutated || unsafe;
        if(unsafe) retirement_uncertain.store(true);
        if(!unsafe && !mutated) { if(incoming) cudaFreeHost(incoming); if(outgoing) cudaFreeHost(outgoing); }
        return false;
    };
    auto ck = [&](cudaError_t s, const char* operation) {
        if(s==cudaSuccess) return true;
        err=std::string(operation)+": "+cudaGetErrorString(s); unsafe=true; return false;
    };
    auto copy = [&](Dev& d, void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
        if(!ck(cudaSetDevice(d.id),"refresh set primary device") ||
           !ck(cudaMemcpyAsync(dst,src,n,kind,d.s),"refresh primary copy") ||
           !ck(cudaStreamSynchronize(d.s),"refresh primary copy retirement")) return false;
        bytes+=n; return true;
    };
    try {
    if(!counts.empty()) {
        // Inference is serialized by the service FIFO. Join every stream that
        // could hold a resident/HOST reader before swapping exclusive storage.
        for(auto& d:D) {
            if(!ck(cudaSetDevice(d.id),"refresh set device")) return failed();
            for(auto s:{d.s,d.s2,d.cs}) if(s && !ck(cudaStreamSynchronize(s),"refresh reader retirement")) return failed();
            // The explicit successful join retires foreground source readers
            // even when their ordinary query boundary has not run yet.
            if(!scale_joined(d,err))return failed();
            std::fill(d.slot_foreground.begin(),d.slot_foreground.end(),0);
            if(!adaptive_publish(d,err))return failed();
            if(adaptive_pending(d)) {err="adaptive transaction not retired by explicit placement join";unsafe=true;return failed();}
            prefetch_discard_joined(d);
            d.promo_q.clear();
            d.n_pending=0;
            std::fill(d.slot_used.begin(),d.slot_used.end(),0);
            std::fill(d.slot_owner.begin(),d.slot_owner.end(),-1);
            std::fill(d.slot_pend.begin(),d.slot_pend.end(),-1);
            std::fill(d.slot_iss.begin(),d.slot_iss.end(),false);
            std::fill(d.slot_foreground.begin(),d.slot_foreground.end(),0);
        }
        if(!promotion_error.empty()) {err=promotion_error;return failed();}
        graphs_clear();
        if (!baseline_hf_demand(err)) {unsafe=true;return failed();}
        std::fill(use_score.begin(),use_score.end(),0.0f);
        std::fill(use_last.begin(),use_last.end(),0);
        std::fill(adaptive_main.begin(),adaptive_main.end(),0);
        const size_t stride=nf::record_stride(g.dim,g.ff), blob=sizeof(ne::Header)+stride;
        for(const auto& l:L) if(l.moe && l.blob!=blob) {err="native refresh record size mismatch";return failed();}
        if(!ck(cudaHostAlloc(&incoming,blob,cudaHostAllocPortable),"refresh incoming staging") ||
           !ck(cudaHostAlloc(&outgoing,blob,cudaHostAllocPortable),"refresh outgoing staging")) return failed();
        scratch_bytes=2*blob + (compressed_host()?host_stride+scale_encode_scratch_bytes():0);
        std::vector<uint8_t> encoded(compressed_host()?host_stride:0);
        auto export_helper = [&](int id, uint8_t* dst) {
            std::memset(dst,0,sizeof(ne::Header)); auto* h=reinterpret_cast<ne::Header*>(dst);
            bool source_unsafe=false;
            const bool ok=helper->export_expert(id/g.n_exp,id%g.n_exp,dst+sizeof(ne::Header),stride,
                                               h->weight_global,h->input_global,source_unsafe,err);
            unsafe |= source_unsafe; if(!ok)placement_failed=true; if(ok)bytes+=stride; return ok;
        };
        auto replace_helper = [&](int old_id, int new_id, const uint8_t* src) {
            const auto* h=reinterpret_cast<const ne::Header*>(src); bool source_unsafe=false;
            const bool ok=helper->replace_expert(old_id/g.n_exp,old_id%g.n_exp,new_id/g.n_exp,new_id%g.n_exp,
                src+sizeof(ne::Header),stride,h->weight_global,h->input_global,source_unsafe,err);
            unsafe |= source_unsafe; if(!ok)placement_failed=true; if(ok)bytes+=stride; return ok;
        };
        if(primary) for(auto& d:D) {
            struct Slot {int id;uint8_t* ptr;};
            std::vector<Slot> slots; std::vector<int> candidates;
            for(int li:d.layers) if(L[li].moe) {
                const auto& l=L[li];
                for(size_t s=0;s<l.res_ids.size();++s)slots.push_back({key(li,l.res_ids[s]),resident_address(l,int(s))});
                for(int e=0;e<g.n_exp;++e)candidates.push_back(key(li,e));
            }
            std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return counts[a]>counts[b];});
            std::vector<int> desired; std::vector<int> per_layer(g.n_layers);
            const int cap=o.resident_cap<0 ? g.n_exp : std::min(o.resident_cap,g.n_exp);
            for(int id:candidates)if(per_layer[id/g.n_exp]<cap && desired.size()<slots.size()) {
                desired.push_back(id);++per_layer[id/g.n_exp];
            }
            if(desired.size()!=slots.size()) {err="profile cannot fill fixed primary capacity";return failed();}
            std::vector<uint8_t> wanted(total,0), already(total,0);
            for(int id:desired)wanted[id]=1;
            for(const auto& s:slots)already[s.id]=1;
            size_t victim=0;
            for(int id:desired) {
                if(already[id])continue;
                while(victim<slots.size() && wanted[slots[victim].id])++victim;
                if(victim==slots.size()) {err="no primary exchange slot";return failed();}
                auto& s=slots[victim++];
                auto& old=L[s.id/g.n_exp]; auto& next=L[id/g.n_exp];
                const int oe=s.id%g.n_exp, nexp=id%g.n_exp;
                const bool from_helper=helper_owns(next,nexp);
                const uint8_t* host=next.host_blob[nexp];
                if(!from_helper && !host) {err="primary replacement lacks source owner";return failed();}
                if(from_helper) {if(!export_helper(id,incoming))return failed();}
                else if(compressed_host()) {if(!::mm3::scale_store::decode_raw(host,host_stride,incoming,blob,err))return failed();}
                else std::memcpy(incoming,host,blob);
                if(!copy(d,outgoing,s.ptr,blob,cudaMemcpyDeviceToHost))return failed();
                if(!from_helper && compressed_host()) {size_t n=0;if(!::mm3::scale_store::encode(outgoing,blob,encoded.data(),encoded.size(),n,err))return failed();}
                mutated=true;
                if(!copy(d,s.ptr,incoming,blob,cudaMemcpyHostToDevice))return failed();
                if(from_helper) {if(!replace_helper(id,s.id,outgoing))return failed();helper_bits();}
                else {
                    if(compressed_host()) std::memcpy(const_cast<uint8_t*>(host),encoded.data(),host_stride);
                    else std::memcpy(const_cast<uint8_t*>(host),outgoing,blob);
                    old.host_blob[oe]=host;next.host_blob[nexp]=nullptr;
                }
                already[s.id]=0;already[id]=1;s.id=id;
            }
            // Publish the complete fixed-bank assignment only after every copy
            // completed. Original allocations remain owned in `owned`.
            for(int li:d.layers)if(L[li].moe) {
                L[li].res_ids.clear();L[li].res_ptr.clear();
                std::fill(L[li].res_slot.begin(),L[li].res_slot.end(),-1);
            }
            for(const auto& s:slots) {
                auto& l=L[s.id/g.n_exp];const int e=s.id%g.n_exp;
                l.res_slot[e]=int(l.res_ids.size());l.res_ids.push_back(e);l.res_ptr.push_back(s.ptr);
            }
            for(int li:d.layers)if(L[li].hf) {
                auto& l=L[li];
                if(l.res_ids.size()!=size_t(g.n_exp)) {l.hf=false;continue;}
                std::vector<unsigned long long> table(g.n_exp);
                for(int e=0;e<g.n_exp;++e)table[e]=(unsigned long long)resident_address(l,l.res_slot[e]);
                std::memcpy(incoming,table.data(),table.size()*sizeof(table[0]));
                if(!copy(d,l.exp_tab,incoming,table.size()*sizeof(table[0]),cudaMemcpyHostToDevice))return failed();
            }
        }
        if(helper) {
            const auto records=helper->records();
            std::vector<int> candidates;
            // Same tie order as initial helper placement: expert, then layer.
            for(int e=0;e<g.n_exp;++e)for(int li=0;li<g.n_layers;++li)
                if(eligible_helper(li) && L[li].res_slot[e]<0)candidates.push_back(key(li,e));
            std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return counts[a]>counts[b];});
            if(candidates.size()<records.size()) {err="insufficient helper complement";return failed();}
            candidates.resize(records.size());
            std::vector<uint8_t> wanted(total,0), already(total,0);
            for(int id:candidates)wanted[id]=1;
            for(const auto& r:records)already[key(r.layer,r.expert)]=1;
            size_t victim=0;
            for(int id:candidates) {
                if(already[id])continue;
                while(victim<records.size() && wanted[key(records[victim].layer,records[victim].expert)])++victim;
                if(victim==records.size()) {err="no helper exchange slot";return failed();}
                const auto r=records[victim++];const int old_id=key(r.layer,r.expert);
                auto& next=L[id/g.n_exp];const int e=id%g.n_exp;
                auto* host=const_cast<uint8_t*>(next.host_blob[e]);
                if(!host) {err="helper replacement lacks exclusive HOST source";return failed();}
                std::memcpy(incoming,host,blob);
                if(!export_helper(old_id,outgoing))return failed();
                mutated=true;
                if(!replace_helper(old_id,id,incoming))return failed();
                std::memcpy(host,outgoing,blob);
                L[r.layer].host_blob[r.expert]=host;next.host_blob[e]=nullptr;
                helper_bits();already[old_id]=0;already[id]=1;
            }
        }
        if(!ownership()) {err="exclusive ownership incomplete after refresh";mutated=true;return failed();}
        cudaFreeHost(incoming);cudaFreeHost(outgoing);incoming=outgoing=nullptr;
    }
    } catch(const std::exception& e) {err=std::string("native placement exception: ")+e.what();return failed();}
      catch(...) {err="native placement unknown exception";return failed();}
    auto fnv = [](const std::vector<uint32_t>& v) {
        uint64_t h=14695981039346656037ULL;
        for(uint32_t n:v)for(int b=0;b<4;++b){h^=(n>>(8*b))&255;h*=1099511628211ULL;}
        char out[17];std::snprintf(out,sizeof(out),"%016llx",(unsigned long long)h);return std::string(out);
    };
    std::ostringstream json;
    json << "{\"profile_fnv64\":\"" << fnv(counts) << "\",\"primary_ids\":[";
    bool comma=false;size_t retained_primary=0, retained_helper=0, nprimary=0;
    for(int li=0;li<g.n_layers;++li) for(size_t s=0;s<L[li].res_ids.size();++s) {
        int e=L[li].res_ids[s];if(comma)json<<',';comma=true;
        json<<'['<<li<<','<<e<<','<<D[L[li].dev].id<<','<<uintptr_t(resident_address(L[li],int(s)))<<']';
        retained_primary+=std::find(old_primary.begin(),old_primary.end(),key(li,e))!=old_primary.end();++nprimary;
    }
    json << "],\"helper_ids\":[";comma=false;
    const auto hr=helper ? helper->records() : std::vector<NativeHelper::Identity>{};
    for(const auto& r:hr) {
        if(comma)json<<',';comma=true;json<<'['<<r.layer<<','<<r.expert<<','<<r.slot<<']';
        retained_helper+=std::find(old_helper.begin(),old_helper.end(),key(r.layer,r.expert))!=old_helper.end();
    }
    json << "],\"experts_per_layer\":"<<g.n_exp<<",\"resident_cap\":"<<o.resident_cap
         <<",\"helper_eligible_cuda_device\":"<<D[head_dev].id<<",\"layer_devices\":[";
    for(int li=0;li<g.n_layers;++li){if(li)json<<',';json<<D[L[li].dev].id;}
    json << "],\"layer_moe\":[";
    for(int li=0;li<g.n_layers;++li){if(li)json<<',';json<<(L[li].moe ? "true" : "false");}
    json << "],\"allocations\":[";
    for(size_t i=0;i<owned.size();++i){if(i)json<<',';json<<uintptr_t(owned[i]);}
    json << "],\"staging_frames\":[";comma=false;
    for(const auto& d:D)for(int s=0;s<d.n_slots;++s) {
        if(comma)json<<',';comma=true;
        json<<'['<<d.id<<','<<s<<','<<uintptr_t(staging_address(d,s))<<','<<d.slot_owner[s]<<','<<(staging_reserved(d,s)?1:0)<<']';
    }
    json << "],\"target_object\":"<<uintptr_t(this)<<",\"draft_object\":"<<uintptr_t(draftm.get())
         <<",\"host_address\":"<<uintptr_t(pool)<<",\"host_bytes\":"<<native_pool->capacity()
         <<",\"host_encoded\":"<<(compressed_host()?"true":"false")
         <<",\"host_encoded_stride\":"<<(compressed_host()?host_stride:0)
         <<",\"host_physical_payload_bytes\":"<<native_pool->size()
         <<",\"host_logical_expert_bytes\":"<<(size_t(st.pooled)*::mm3::scale_store::kRawBytes)
         <<",\"codec_device_scratch_bytes\":"<<(compressed_host()?2*::mm3::scale_store::kScratchBytes:0)
         <<",\"codec_host_scratch_bytes\":"<<(compressed_host()?::mm3::scale_store::kPinnedBytes:0)
         <<",\"native_cooperative\":"<<(o.native_cooperative?"true":"false")
         <<",\"native_cpu_workspace_bytes\":"<<native_cpu_bytes
         <<",\"window_cpu_bytes\":"<<((!native || o.native_cooperative)?D.size()*((size_t)kWin*g.dim+kWin+4)*4:0)
         <<",\"owner_plan_host_bytes\":"<<((native && !o.native_cooperative)?D.size()*65*win_plan_bytes():0)
         <<",\"owner_plan_device_bytes\":"<<((native && !o.native_cooperative)?D.size()*win_plan_bytes():0)
         <<",\"adaptive_host_scratch_bytes\":"<<adaptive_blob*adaptive_host_owned.size()
         <<",\"primary_adaptive\":"<<(primary_adaptive?"true":"false")
         <<",\"helper_live\":"<<(helper_live_enabled?"true":"false")
         <<",\"helper_live_scratch_bytes\":"<<2*helper_live_blob
         <<",\"helper_live_started\":"<<helper_live_started
         <<",\"helper_live_finished\":"<<helper_live_finished
         <<",\"host_records\":"<<st.pooled<<",\"helper_pool\":"<<(helper ? helper->pool_identity() : 0)
         <<",\"primary_retained\":"<<retained_primary<<",\"primary_changed\":"<<(nprimary-retained_primary)
         <<",\"helper_retained\":"<<retained_helper<<",\"helper_changed\":"<<(hr.size()-retained_helper)
         <<",\"transferred_bytes\":"<<bytes<<",\"exchange_scratch_bytes\":"<<scratch_bytes
         <<",\"duration_ms\":"<<(1000*(now()-started))<<",\"ownership_complete\":true}";
    report=json.str();return true;
}
