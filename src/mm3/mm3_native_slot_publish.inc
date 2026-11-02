    // Query-only publication is safe between layer snapshots: it never
    // submits a queued copy or waits for one that has not completed.
    bool publish_slots(Dev& d, std::string& err) {
        auto query = [&](size_t slot, std::string& error, const char* operation) {
            const auto service_status=scale_optional_query(slot,error);
            if(service_status!=r2::Completion::ready)return service_status;
            const cudaError_t status=cudaEventQuery(d.slot_ev[slot]);
            if (status==cudaSuccess) return scale_optional_retire(slot,error) ? r2::Completion::ready : r2::Completion::failed;
            if (status==cudaErrorNotReady) return r2::Completion::not_ready;
            error=std::string(operation)+cudaGetErrorString(status);
            return r2::Completion::failed;
        };
        if (d.n_pending == 0) return true;
        const double tw = now();
        std::vector<long long> prior_pending;
        if(!d.slot_prediction.empty())prior_pending=d.slot_pend;
        const bool ok = r2::publish(d.slot_pend, d.slot_iss, d.slot_owner, d.slot_used,
                                    d.n_pending, fwd, promotion_error, err,
            [&](size_t slot, std::string& error) {
                ++st.promo_queries;
                const auto status=query(slot,error,"promotion completion failed: ");
                st.promo_not_ready += status==r2::Completion::not_ready;
                return status;
            });
        st.t_promo += now() - tw;
        if(!ok && compressed_host()) {placement_failed=true;retirement_uncertain.store(true);}
        if(ok && !d.slot_prediction.empty())for(size_t s=0;s<d.slot_pend.size();++s)
            if(prior_pending[s]>=0 && d.slot_pend[s]<0 && d.slot_prediction[s]==1)d.slot_prediction[s]=2;
        return ok;
    }
