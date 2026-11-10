    // CPU workers can still be submitting optional copies until drained. Keep
    // their source bank and GPU scratch quarantined on any uncertain join.
    bool scale_maintenance_join(Dev& d, std::string& err) {
        return ::mm3::r2::execution_boundary(native || compressed_host() || o.native_stage_prefix,
                                             retirement_uncertain,[&] {
            if(!scale_drain(err))return false;
            auto checked = [&](cudaError_t rc, const char* operation) {
                if(rc==cudaSuccess)return true;
                if(native || compressed_host() || o.native_stage_prefix) {
                    placement_failed=true;retirement_uncertain.store(true);
                }
                err=std::string(operation)+": "+cudaGetErrorString(rc);
                return false;
            };
            if(!checked(cudaSetDevice(d.id),"maintenance set device") ||
               !checked(cudaStreamSynchronize(d.s),"maintenance primary join") ||
               (d.s2 && !checked(cudaStreamSynchronize(d.s2),"maintenance secondary join")))return false;
            if(!compressed_host() && !issue_promotions(d,err))return false;
            if(d.cs && !checked(cudaStreamSynchronize(d.cs),"maintenance copy join"))return false;
            if(d.xs && !exchange_downloads_join(d,err))return false;   // (exchange_schedule: the victims' downloads)
            return scale_joined(d,err);
        });
    }
