    bool scale_drain(std::string& err) {
        if(!scale_service || scale_service->drain_cpu())return true;
        err=scale_service->error();placement_failed=true;retirement_uncertain.store(true);return false;
    }
    bool scale_joined(Dev& d, std::string& err) {
        if(!scale_service || d.id!=scale_device)return true;
        for(size_t i=0;i<d.slot_ev.size();++i) {
            auto state=scale_service->optional_state(int(i));
            if(state==::mm3::scale_store::OptionalState::Submitted) {
                if(cudaEventQuery(d.slot_ev[i])!=cudaSuccess || !scale_service->retire_optional(int(i))) {
                    err="compressed optional retirement failed";placement_failed=true;retirement_uncertain.store(true);return false;
                }
            } else if(state!=::mm3::scale_store::OptionalState::Free) {
                err="compressed optional worker did not retire";placement_failed=true;retirement_uncertain.store(true);return false;
            }
        }
        if(!scale_service->reset_foreground_after_join()) {err=scale_service->error();placement_failed=true;retirement_uncertain.store(true);return false;}
        return true;
    }
    bool scale_foreground_joined(Dev& d, std::string& err) {
        if(!scale_service || d.id!=scale_device || scale_service->reset_foreground_after_join())return true;
        err=scale_service->error();placement_failed=true;retirement_uncertain.store(true);return false;
    }
    r2::Completion scale_optional_query(size_t slot, std::string& err) {
        if(!scale_service)return r2::Completion::ready;
        auto state=scale_service->optional_state(int(slot));
        if(state==::mm3::scale_store::OptionalState::Queued || state==::mm3::scale_store::OptionalState::Running)return r2::Completion::not_ready;
        if(state!=::mm3::scale_store::OptionalState::Submitted) {err=scale_service->error();if(err.empty())err="invalid compressed optional state";placement_failed=true;retirement_uncertain.store(true);return r2::Completion::failed;}
        return r2::Completion::ready;
    }
    bool scale_optional_retire(size_t slot, std::string& err) {
        if(!scale_service || scale_service->retire_optional(int(slot)))return true;
        err=scale_service->error();placement_failed=true;retirement_uncertain.store(true);return false;
    }

    bool issue_promotions(Dev& d, std::string& err) {
        if(prefetch_active(true))return true; // issued only behind the current layer's foreground uploads
        if(compressed_host()) {
            if(!d.promo_q.empty() && d.id!=scale_device) {err="compressed promotion submitted on non-owner device";placement_failed=true;return false;}
            while(!d.promo_q.empty()) {
                const auto& q=d.promo_q.front();
                if(!scale_service->enqueue_optional(q.slot,staging_address(d,q.slot),q.src,host_stride,d.slot_ev[size_t(q.slot)])) {
                    if(scale_service->healthy()) return true; // bounded backpressure: retain the queue
                    err=promotion_error=scale_service->error(); placement_failed=true;retirement_uncertain.store(true);return false;
                }
                d.slot_iss[size_t(q.slot)]=1; d.promo_q.erase(d.promo_q.begin());
            }
            return true;
        }
        return r2::issue(d.promo_q, d.slot_iss, promotion_error, err,
            [&](const Dev::Promo& q, std::string& error) {
                cudaError_t status = cudaMemcpyAsync(staging_address(d,q.slot), q.src, q.bytes,
                                                      cudaMemcpyHostToDevice, d.cs);
                if (status == cudaSuccess) status = cudaEventRecord(d.slot_ev[(size_t) q.slot], d.cs);
                if (status == cudaSuccess) {
                    if(capture::detailed()) {std::ostringstream out;out<<"\"slot\":"<<q.slot<<",\"incoming\":"<<d.slot_pend[size_t(q.slot)]
                        <<",\"source_host_frame\":"<<uintptr_t(q.src)<<",\"destination_frame\":"<<uintptr_t(staging_address(d,q.slot))
                        <<",\"direction\":\"h2d\",\"bytes\":"<<q.bytes<<",\"status\":\"submitted\"";
                        capture::detail_event("promotion_copy",out.str());}
                    return true;
                }
                error = std::string("promotion submission failed: ") + cudaGetErrorString(status);
                if(capture::detailed())capture::detail_event("promotion_copy_error","\"slot\":"+std::to_string(q.slot)+",\"error\":"+capture::quote(error));
                return false;
            });
    }

