// Explicit idle setup only. host_blob remains the exclusive logical owner;
// primary_backing records a second physical copy for each primary identity.
bool configure_primary_backing(bool enable,std::string& report,std::string& err) {
    if(!adaptive_supported() || compressed_host() || !o.native_cooperative || !helper ||
       placement_failed || retirement_uncertain.load()) {
        err="primary backing requires healthy single-primary raw HOST cooperative serving";return false;
    }
    if(enable==bool(primary_backing_bank)) {
        std::ostringstream out;out<<"{\"enabled\":"<<(enable?"true":"false")
            <<",\"physical_bytes\":"<<(primary_backing_bank?primary_backing_bank->capacity():0)
            <<",\"valid_primary_records\":"<<std::count_if(primary_backing.begin(),primary_backing.end(),[](const uint8_t* p){return p;})<<'}';
        report=out.str();return true;
    }
    if(!adaptive_prepare(err))return false;
    std::string joined;
    if(!placement({},false,joined,err,nullptr,nullptr,true))return false;
    const size_t blob=adaptive_blob;
    size_t count=0;for(const auto& l:L)if(l.moe)count+=l.res_ids.size();
    if(!count || count>std::numeric_limits<size_t>::max()/blob) {err="invalid primary backing capacity";return false;}
    constexpr uint64_t reserve=16ULL<<30;
    if(enable) {
        NativeHostMemorySnapshot before;
        try {
            before=native_host_reserve_snapshot();
            primary_backing_bank=std::make_unique<NativeHostBank>(count*blob,D[0].id,reserve,reserve);
            primary_backing.assign(size_t(g.n_layers)*g.n_exp,nullptr);
        } catch(const std::exception& ex) {
            primary_backing_bank.reset();primary_backing.clear();err=std::string("primary backing allocation: ")+ex.what();return false;
        }
        auto& d=D[0];size_t index=0;
        for(int li=0;li<g.n_layers;++li) {
            const auto& l=L[li];if(!l.moe)continue;
            for(size_t s=0;s<l.res_ids.size();++s) {
                auto* dest=reinterpret_cast<uint8_t*>(primary_backing_bank->data())+index++*blob;
                auto status=cudaSetDevice(d.id);
                if(status==cudaSuccess)status=cudaMemcpyAsync(dest,resident_address(l,int(s)),blob,cudaMemcpyDeviceToHost,d.s);
                if(status==cudaSuccess)status=cudaStreamSynchronize(d.s);
                if(status!=cudaSuccess) {
                    // The failed submission or retirement may still reference this bank.
                    err=std::string("primary backing setup copy: ")+cudaGetErrorString(status);
                    placement_failed=true;retirement_uncertain.store(true);return false;
                }
                primary_backing_for(li,l.res_ids[s])=dest;
            }
        }
        NativeHostMemorySnapshot after;
        try {after=native_host_reserve_snapshot();}
        catch(const std::exception& ex) {
            primary_backing.clear();primary_backing_bank.reset();
            err=std::string("primary backing post-allocation memory check: ")+ex.what();return false;
        }
        if(after.available_physical<reserve || after.available_commit<reserve) {
            primary_backing.clear();primary_backing_bank.reset();
            err="primary backing allocation left less than the 16 GiB physical/commit reserve";return false;
        }
        std::ostringstream out;out<<"{\"enabled\":true,\"logical_host_records\":"<<st.pooled
            <<",\"logical_primary_records\":"<<count<<",\"logical_helper_records\":"<<helper->records().size()
            <<",\"valid_primary_records\":"<<count<<",\"physical_payload_bytes\":"<<count*blob
            <<",\"physical_bytes\":"<<primary_backing_bank->capacity()<<",\"setup_d2h_bytes\":"<<count*blob
            <<",\"physical_reserve_bytes\":"<<reserve<<",\"commit_reserve_bytes\":"<<reserve
            <<",\"pre_allocation_effective_physical\":"<<before.available_physical
            <<",\"pre_allocation_effective_commit\":"<<before.available_commit
            <<",\"pre_allocation_guest_physical\":"<<before.guest_available_physical
            <<",\"windows_host\":"<<(before.windows_host?"true":"false")
            <<",\"pre_allocation_windows_physical\":"<<before.windows_available_physical
            <<",\"pre_allocation_windows_commit\":"<<before.windows_available_commit
            <<",\"post_allocation_effective_physical\":"<<after.available_physical
            <<",\"post_allocation_effective_commit\":"<<after.available_commit
            <<",\"post_allocation_guest_physical\":"<<after.guest_available_physical
            <<",\"post_allocation_windows_physical\":"<<after.windows_available_physical
            <<",\"post_allocation_windows_commit\":"<<after.windows_available_commit<<'}';
        report=out.str();return true;
    }
    // A backing-bank frame may now own a HOST expert. Pair it with a primary
    // backing frame outside the bank, copy its bytes and update only the HOST
    // pointer. This retires all references to the bank before release.
    std::vector<const uint8_t**> bank_hosts;
    std::vector<uint8_t*> free_frames;
    for(auto& l:L)if(l.moe)for(auto& p:l.host_blob)
        if(p && primary_backing_contains(uintptr_t(p),blob))bank_hosts.push_back(&p);
    for(auto* p:primary_backing)if(p && !primary_backing_contains(uintptr_t(p),blob))free_frames.push_back(p);
    if(!primary_backing_ledger::repatriate(bank_hosts,free_frames,blob)) {
        err="primary backing repatriation ledger mismatch";placement_failed=true;return false;
    }
    const size_t capacity=primary_backing_bank->capacity();
    const size_t moved=bank_hosts.size();
    primary_backing.clear();primary_backing_bank.reset();
    std::ostringstream out;out<<"{\"enabled\":false,\"logical_host_records\":"<<st.pooled
        <<",\"logical_primary_records\":"<<count<<",\"logical_helper_records\":"<<helper->records().size()
        <<",\"released_physical_bytes\":"<<capacity<<",\"repatriated_host_records\":"<<moved<<'}';
    report=out.str();return true;
}
