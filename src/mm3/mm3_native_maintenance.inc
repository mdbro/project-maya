    // CPU workers can still be submitting optional copies until drained. Keep
    // their source bank and GPU scratch quarantined on any uncertain join.
    bool scale_maintenance_join(Dev& d, std::string& err) {
        if(!scale_drain(err))return false;
        auto checked = [&](cudaError_t rc, const char* operation) {
            if(rc==cudaSuccess)return true;
            err=std::string(operation)+": "+cudaGetErrorString(rc);
            if(compressed_host()) {placement_failed=true;retirement_uncertain.store(true);}
            return false;
        };
        if(!checked(cudaSetDevice(d.id),"maintenance set device") ||
           !checked(cudaStreamSynchronize(d.s),"maintenance primary join") ||
           (d.s2 && !checked(cudaStreamSynchronize(d.s2),"maintenance secondary join")))return false;
        if(!compressed_host() && !issue_promotions(d,err))return false;
        if(d.cs && !checked(cudaStreamSynchronize(d.cs),"maintenance copy join"))return false;
        return scale_joined(d,err);
    }
