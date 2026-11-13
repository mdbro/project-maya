// Opt-in ordered residency evidence. File I/O can perturb timing; disable the
// marker for measured speed requests. No CUDA wait/query is added by tracing.
std::ostringstream residency_trace;
std::string residency_legacy_path, residency_legacy_buffer;
bool residency_trace_active=false;
bool replay_trace_active=false;
std::array<cudaEvent_t,13> replay_gpu_events{};
long long replay_gpu_forward=-1;int replay_gpu_layer=-1,replay_gpu_rows=0;
bool replay_gpu_pending=false;
bool trace_gpu_finished(std::string& err) {
    if(!replay_trace_active || !replay_gpu_pending)return true;
    float compute=0,wait=0,reduce=0,plan=0;
    // Called only after an existing primary-stream join. No new wait/query.
    if(cudaEventElapsedTime(&compute,replay_gpu_events[0],replay_gpu_events[1])!=cudaSuccess ||
       cudaEventElapsedTime(&wait,replay_gpu_events[1],replay_gpu_events[2])!=cudaSuccess ||
       cudaEventElapsedTime(&reduce,replay_gpu_events[2],replay_gpu_events[3])!=cudaSuccess ||
       cudaEventElapsedTime(&plan,replay_gpu_events[11],replay_gpu_events[12])!=cudaSuccess) {
        err="replay GPU timing not retired at the existing join";return false;
    }
    std::ostringstream out;out.precision(17);
    out<<"\"measured_forward\":"<<replay_gpu_forward<<",\"measured_layer\":"<<replay_gpu_layer
       <<",\"rows\":"<<replay_gpu_rows<<",\"compute_ms\":"<<compute<<",\"wait_ms\":"<<wait<<",\"reduce_ms\":"<<reduce<<",\"plan_upload_ms\":"<<plan;
    capture::detail_event("primary_service_observation",out.str());replay_gpu_pending=false;return true;
}
void trace_emit() {
    const std::string row=residency_trace.str();
    if(capture::detailed()) capture::detail_event("residency", "\"record\":"+row);
    if(!residency_legacy_path.empty()) residency_legacy_buffer += row+"\n";
    residency_trace.str(""); residency_trace.clear();
}

uint64_t residency_trace_request=0, residency_trace_event=0;
Stats residency_trace_before{};
double residency_trace_layer_begin=0;

void trace_numbers(const int* values,size_t n) {
    residency_trace << '[';
    for(size_t i=0;i<n;++i) {if(i)residency_trace<<',';residency_trace<<values[i];}
    residency_trace << ']';
}
void trace_ownership(const Layer& l) {
    residency_trace << "\"primary\":[";
    for(size_t i=0;i<l.res_ids.size();++i) {if(i)residency_trace<<',';residency_trace<<l.res_ids[i];}
    residency_trace << "],\"helper\":[";
    bool first=true;
    for(int e=0;e<g.n_exp;++e)if(helper_owns(l,e)) {if(!first)residency_trace<<',';residency_trace<<e;first=false;}
    residency_trace << "],\"host\":[";first=true;
    for(int e=0;e<g.n_exp;++e)if(l.host_blob[e]) {if(!first)residency_trace<<',';residency_trace<<e;first=false;}
    residency_trace << "],\"primary_frames\":[";
    for(size_t i=0;i<l.res_ids.size();++i) {if(i)residency_trace<<',';residency_trace<<'['<<l.res_ids[i]<<','<<uintptr_t(resident_address(l,int(i)))<<']';}
    residency_trace << "],\"host_frames\":["; first=true;
    for(int e=0;e<g.n_exp;++e)if(l.host_blob[e]) {if(!first)residency_trace<<',';first=false;residency_trace<<'['<<e<<','<<uintptr_t(l.host_blob[e])<<']';}
    residency_trace << "],\"primary_backing_frames\":[";first=true;
    if(primary_backing_bank)for(int e:l.res_ids) {
        if(!first)residency_trace<<',';first=false;
        residency_trace<<'['<<e<<','<<uintptr_t(primary_backing_for(int(&l-L.data()),e))<<']';
    }
    residency_trace << ']';
}
void trace_pending(const Dev& d) {
    residency_trace << ",\"staging\":[";
    for(int s=0;s<d.n_slots;++s) {
        if(s)residency_trace<<',';
        residency_trace << '[' << d.slot_owner[s] << ',' << d.slot_pend[s] << ','
            << int(d.slot_iss[s]) << ',' << int(d.slot_foreground[s]) << ','
            << staging_reserved(d,s) << ',' << d.slot_used[s] << ',' << uintptr_t(staging_address(d,s)) << ']';
    }
    residency_trace << "],\"exchanges\":[";
    for(size_t i=0;i<d.adaptive.size();++i) {
        if(i)residency_trace<<',';const auto& x=d.adaptive[i];
        residency_trace << '['<<x.incoming<<','<<x.victim<<','<<x.landing<<','<<x.submitted<<']';
    }
    residency_trace << "],\"helper_exchange\":["<<helper_live_a<<','<<helper_live_b<<','
        <<helper_live_withdrawn<<','<<helper_live_published<<']';
}
void trace_replay_state() {
    if(!replay_trace_active)return;
    residency_trace<<",\"replay_state\":{\"row_clock\":"<<native_demand_rows
        <<",\"block_rows\":"<<native_demand_block_rows
        <<",\"primary_credit\":"<<primary_admission_budget.credit
        <<",\"primary_credit_row\":"<<primary_admission_budget.last_row
        <<",\"helper_credit\":"<<helper_admission_budget.credit
        <<",\"helper_credit_row\":"<<helper_admission_budget.last_row
        <<",\"budget_forward\":"<<adaptive_budget_fwd<<",\"budget_used\":"<<adaptive_budget_used
        <<",\"scores\":[";
    bool first=true;
    for(size_t i=0;i<use_score.size();++i)if(use_score[i]!=0 || use_last[i]!=0) {
        if(!first)residency_trace<<',';first=false;
        residency_trace<<'['<<i<<','<<use_score[i]<<','<<use_last[i]<<']';
    }
    residency_trace<<"],\"adaptive_main\":[";first=true;
    for(size_t i=0;i<adaptive_main.size();++i)if(adaptive_main[i]) {
        if(!first)residency_trace<<',';first=false;residency_trace<<i;
    }
    residency_trace<<"],\"host_spares\":[";
    for(size_t i=0;i<adaptive_host_spare.size();++i) {if(i)residency_trace<<',';residency_trace<<uintptr_t(adaptive_host_spare[i]);}
    residency_trace<<"],\"helper_buffers\":["<<uintptr_t(helper_live_buffers[0])<<','<<uintptr_t(helper_live_buffers[1])
        <<"],\"helper_source\":"<<uintptr_t(helper_live_host)<<",\"helper_queue_pending\":"<<helper_queue_pending
        <<",\"pending_sources\":[";first=true;
    for(const auto& d:D)for(const auto& x:d.adaptive)if(x.incoming>=0) {
        if(!first)residency_trace<<',';first=false;
        residency_trace<<'['<<x.incoming<<','<<x.victim<<','<<x.landing<<','<<x.submitted<<','
            <<uintptr_t(x.source_host)<<','<<uintptr_t(x.source_gpu)<<','<<uintptr_t(x.landing_gpu)<<','<<uintptr_t(x.target_host)<<']';
    }
    residency_trace<<"]}";
}
void trace_replay_boundary(const char* kind,int layer=-1,int rows=0) {
    if(!replay_trace_active)return;
    residency_trace<<"{\"type\":\"boundary\",\"request\":"<<residency_trace_request
        <<",\"forward\":"<<fwd<<",\"kind\":"<<capture::quote(kind)<<",\"layer\":"<<layer
        <<",\"rows\":"<<rows<<",\"generation\":"<<generation_active
        <<",\"row_clock\":"<<native_demand_rows<<",\"time_s\":"<<now();
    if(!D.empty())trace_pending(D[size_t(head_dev)]);
    residency_trace<<'}';trace_emit();
}
void trace_residency_decision(const char* kind,const std::vector<ResidencyCandidate>& incoming,
        const std::vector<ResidencyCandidate>& victims,const ResidencyChoice& choice,
        int protected_layer,double benefit_per_score) {
    if(!replay_trace_active)return;
    residency_trace<<"{\"type\":\"policy_decision\",\"request\":"<<residency_trace_request
        <<",\"forward\":"<<fwd<<",\"kind\":"<<capture::quote(kind)<<",\"protected_layer\":"<<protected_layer
        <<",\"row_clock\":"<<native_demand_rows<<",\"benefit_per_score\":"<<benefit_per_score;
    const auto list=[&](const char* name,const std::vector<ResidencyCandidate>& values) {
        residency_trace<<",\""<<name<<"\":[";
        for(size_t i=0;i<values.size();++i) {if(i)residency_trace<<',';
            residency_trace<<'['<<values[i].key<<','<<values[i].score<<','<<values[i].ready_slot<<']';}
        residency_trace<<']';
    };
    list("incoming",incoming);list("victims",victims);
    residency_trace<<",\"choice\":["<<choice.incoming<<','<<choice.victim<<','<<choice.ready_slot<<','<<choice.value_ms
        <<"],\"primary_credit\":"<<primary_admission_budget.credit<<",\"helper_credit\":"<<helper_admission_budget.credit;
    if(!D.empty())trace_pending(D[size_t(head_dev)]);
    residency_trace<<'}';trace_emit();
}
// The same complete starting record is available at joined setup boundaries.
// Snapshot serialization adds no timed-layer instrumentation or CUDA events.
void trace_start_record() {
    residency_trace.precision(replay_trace_active?17:12);
    residency_trace << "{\"type\":\"request\",\"schema\":1,\"request\":"<<residency_trace_request
        <<",\"adaptive_issued\":"<<st.adaptive_issued<<",\"adaptive_published\":"<<st.adaptive_published
        <<",\"adaptive_reused\":"<<st.adaptive_reused<<",\"adaptive_h2d_bytes\":"<<st.adaptive_h2d_bytes
        <<",\"adaptive_d2h_bytes\":"<<st.adaptive_d2h_bytes
        <<",\"forward\":"<<fwd<<",\"generation_row_clock\":"<<native_demand_rows<<",\"time_s\":"<<now()<<",\"record_bytes\":"
        <<sizeof(ne::Header)+nf::record_stride(g.dim,g.ff)
        <<",\"lane_cpu1_ms\":"<<lane_cpu1<<",\"lane_cpu_add_ms\":"<<lane_cpu_add
        <<",\"lane_upload_ffn_ms\":"<<lane_copy<<",\"primary_adaptive\":"<<primary_adaptive
        <<",\"cooperative_cpu_scale\":"<<o.cooperative_cpu_scale
        <<",\"primary_submit_query\":"<<(o.primary_submit_query?"true":"false")
        <<",\"cpu_participants\":"<<(lane?lane->threads():0)
        <<",\"replay_schema\":"<<(replay_trace_active?1:0)
        <<",\"replay_supported\":"<<(native && o.native_cooperative && D.size()==1 && !graph_on &&
             !o.native_stage_prefix && !compressed_host() && !adaptive_policy.legacy &&
             helper_dispatch_current=="worker" && o.cooperative_stage_reuse ? "true":"false")
        <<",\"dimension\":"<<g.dim<<",\"expert_count\":"<<g.n_exp<<",\"layer_count\":"<<g.n_layers
        <<",\"helper_live\":"<<helper_live_enabled<<",\"compressed_host\":"<<compressed_host()
        <<",\"host_free_layer_count\":"<<std::count_if(L.begin(),L.end(),[](const Layer& l){return l.moe && l.hf;})
        <<",\"score_scratch_device_bytes\":"<<std::count_if(D.begin(),D.end(),[](const Dev& d){return d.capture_scores_d!=nullptr;})*size_t(kWin)*128*3*sizeof(float)
        <<",\"score_scratch_pinned_bytes\":"<<std::count_if(D.begin(),D.end(),[](const Dev& d){return d.capture_scores_h!=nullptr;})*size_t(kWin)*128*3*sizeof(float)
        <<",\"policy\":{\"legacy\":"<<adaptive_policy.legacy<<",\"horizon\":"<<adaptive_policy.horizon
        <<",\"max_exchanges\":"<<adaptive_policy.max_exchanges
        <<",\"service_saved_ms\":"<<adaptive_service_ms()<<",\"h2d_ms\":"<<adaptive_h2d_ms()
        <<",\"d2h_ms\":"<<adaptive_d2h_ms()<<",\"configured_h2d_ms\":"<<adaptive_policy.h2d_ms
        <<",\"configured_service_saved_ms\":"<<adaptive_policy.service_saved_ms
        <<",\"configured_d2h_ms\":"<<adaptive_policy.d2h_ms
        <<",\"primary_backing\":"<<(primary_backing_bank?"true":"false")<<",\"interference_ms\":"<<adaptive_policy.interference_ms
        <<",\"margin_ms\":"<<adaptive_policy.margin_ms<<",\"horizon_unit\":\"five_generation_rows\",\"age_unit\":\"generation_rows\"}"
        <<",\"batch_staging_calibration\":"<<(batch_staging_calibration.empty()?"null":batch_staging_calibration)
        <<",\"layers\":[";
    bool first=true;
    for(size_t li=0;li<L.size();++li)if(L[li].moe) {
        if(!first)residency_trace<<',';first=false;
        residency_trace << "{\"layer\":"<<li<<',';trace_ownership(L[li]);residency_trace<<'}';
    }
    residency_trace << "],\"helper_frames\":[";
    if(helper) {bool sep=false;for(const auto& id:helper->records()) {if(sep)residency_trace<<',';sep=true;residency_trace<<'['<<id.layer<<','<<id.expert<<','<<id.slot<<']';}}
    residency_trace << ']';
    if(!D.empty()) trace_pending(D[size_t(head_dev)]);
    trace_replay_state();
    residency_trace << '}';
}
std::string replay_snapshot_record() {
    std::ostringstream saved;
    residency_trace.swap(saved);
    const bool active=replay_trace_active;
    replay_trace_active=true;
    trace_start_record();
    const std::string record=residency_trace.str();
    residency_trace.swap(saved);
    replay_trace_active=active;
    return record;
}
bool trace_begin(std::string& err) {
    residency_trace.str(""); residency_trace.clear(); residency_legacy_path.clear(); residency_legacy_buffer.clear();
    residency_trace_active=false;replay_trace_active=false;
    ++residency_trace_request;
    // (the switches read once; the marker file is looked for at each request)
    static const ::mm3::EnvValue path_env=::mm3::env_value("MM3_RESIDENCY_TRACE");
    static const ::mm3::EnvValue marker_env=::mm3::env_value("MM3_RESIDENCY_TRACE_ENABLE");
    static const ::mm3::EnvValue replay_env=::mm3::env_value("MM3_REPLAY_CAPTURE");
    const char* path=path_env.c_str();
    const char* marker=marker_env.c_str();
    if(!native)return true;
    if(!light_gpu_prepare(err))return false;   // (a light capture's 5090 stamps: their events made now, not while recording)
    if(path && *path && marker && *marker && std::ifstream(marker).good()) residency_legacy_path=path;
    residency_trace_active=capture::detailed() || !residency_legacy_path.empty();
    if(!residency_trace_active)return true;
    const char* replay=replay_env.c_str();
    replay_trace_active=replay && std::string(replay)=="1" && capture::detailed();
    if(replay_trace_active) {
        if(cudaSetDevice(D[size_t(head_dev)].id)!=cudaSuccess) {err="replay timer device";return false;}
        for(auto& event:replay_gpu_events)if(!event && cudaEventCreate(&event)!=cudaSuccess) {err="replay timer allocation";return false;}
    }
    if(capture::detailed()) for(auto& d:D) {
        const size_t bytes=size_t(kWin)*128*3*sizeof(float);
        if(!d.capture_scores_d || !d.capture_scores_h) {
            if(cudaSetDevice(d.id)!=cudaSuccess ||
               (!d.capture_scores_d && cudaMalloc(&d.capture_scores_d,bytes)!=cudaSuccess) ||
               (!d.capture_scores_h && cudaMallocHost(&d.capture_scores_h,bytes)!=cudaSuccess)) {
                err="detailed capture score scratch allocation failed";return false;
            }
        }
    }
    trace_start_record();
    trace_emit();
    return true;
}
bool trace_end(std::string& err) {
    if(!residency_trace_active)return true;
    residency_trace<<"{\"type\":\"end\",\"request\":"<<residency_trace_request
        <<",\"forward\":"<<fwd<<",\"time_s\":"<<now()
        <<",\"adaptive_issued\":"<<st.adaptive_issued<<",\"adaptive_published\":"<<st.adaptive_published
        <<",\"adaptive_reused\":"<<st.adaptive_reused<<",\"adaptive_h2d_bytes\":"<<st.adaptive_h2d_bytes
        <<",\"adaptive_d2h_bytes\":"<<st.adaptive_d2h_bytes;
    if(replay_trace_active) {
        trace_replay_state();residency_trace<<",\"layers\":[";bool first=true;
        for(size_t li=0;li<L.size();++li)if(L[li].moe) {
            if(!first)residency_trace<<',';first=false;
            residency_trace<<"{\"layer\":"<<li<<',';trace_ownership(L[li]);residency_trace<<'}';
        }
        residency_trace<<"],\"helper_frames\":[";bool sep=false;
        if(helper)for(const auto& id:helper->records()) {if(sep)residency_trace<<',';sep=true;residency_trace<<'['<<id.layer<<','<<id.expert<<','<<id.slot<<']';}
        residency_trace<<']';if(!D.empty())trace_pending(D[size_t(head_dev)]);
    }
    residency_trace<<'}';
    trace_emit();
    if(!residency_legacy_path.empty()) { std::ofstream out(residency_legacy_path,std::ios::app); out<<residency_legacy_buffer; if(!out) {err="residency trace write failed";return false;} }
    residency_trace_active=false;replay_trace_active=false;
    return true;
}
void trace_routes(const Layer& l,const Dev& d,const int* top,int T) {
    if(!residency_trace_active)return;
    residency_trace_before=st;residency_trace_layer_begin=now();
    residency_trace << "{\"type\":\"routes\",\"request\":"<<residency_trace_request
        <<",\"event\":"<<++residency_trace_event<<",\"forward\":"<<fwd
        <<",\"generation\":"<<generation_active<<",\"time_s\":"<<residency_trace_layer_begin
        <<",\"layer\":"<<(&l-L.data())<<",\"rows\":"<<T<<",\"positions\":";
    trace_numbers(d.pos_h,size_t(T));
    residency_trace<<",\"original\":";trace_numbers(d.original_h,size_t(T*g.k_exp));
    residency_trace<<",\"executed\":";trace_numbers(top,size_t(T*g.k_exp));
    if(predict_diag_this_layer==int(&l-L.data()) && !predict_diag_this.empty()) {
        residency_trace<<",\"predicted_from_previous_layer\":";trace_numbers(predict_diag_this.data(),predict_diag_this.size());
    }
    if(capture::detailed() && d.capture_scores_h) {
        residency_trace<<",\"all_score_triplets_sigmoid_base_prior\":[";
        for(int i=0;i<T*128*3;++i) {if(i)residency_trace<<',';residency_trace<<d.capture_scores_h[i];}
        residency_trace<<']';
    }
    residency_trace<<',';trace_ownership(l);trace_pending(d);
    residency_trace<<",\"ready_primary\":[";
    for(int j=0;j<4;++j){if(j)residency_trace<<',';residency_trace<<d.ready_h->primary[j];}
    residency_trace<<"],\"ready_helper\":[";
    for(int j=0;j<4;++j){if(j)residency_trace<<',';residency_trace<<d.ready_h->helper[j];}
    residency_trace<<"]}";
    trace_emit();
}
bool trace_service(const Dev& d,std::string& err) {
    if(!residency_trace_active)return true;
    const auto& b=residency_trace_before;
    residency_trace << "{\"type\":\"service\",\"request\":"<<residency_trace_request
        <<",\"event\":"<<residency_trace_event<<",\"forward\":"<<fwd
        <<",\"time_s\":"<<now()<<",\"host_service_ms\":"<<1000*(now()-residency_trace_layer_begin)
        <<",\"cpu_helper_lane_ms\":"<<1000*(st.t_lane-b.t_lane)
        <<",\"coop_plan_ms\":"<<st.coop_plan_ms-b.coop_plan_ms
        <<",\"coop_cpu_run_ms\":"<<st.coop_cpu_run_ms-b.coop_cpu_run_ms
        <<",\"coop_cpu_predicted_ms\":"<<st.coop_cpu_predicted_ms-b.coop_cpu_predicted_ms
        <<",\"coop_cpu_columns\":"<<st.coop_cpu_columns-b.coop_cpu_columns
        <<",\"coop_cpu_chunks\":"<<st.coop_cpu_chunks-b.coop_cpu_chunks
        <<",\"coop_primary_columns\":"<<st.coop_primary_columns-b.coop_primary_columns
        <<",\"coop_primary_groups\":"<<st.coop_primary_groups-b.coop_primary_groups
        <<",\"coop_primary_chunks\":"<<st.coop_primary_chunks-b.coop_primary_chunks
        <<",\"coop_staging_columns\":"<<st.coop_staging_columns-b.coop_staging_columns
        <<",\"coop_helper_columns\":"<<st.coop_helper_columns-b.coop_helper_columns
        <<",\"cpu_jobs\":"<<st.cpu_experts-b.cpu_experts
        <<",\"helper_jobs\":"<<st.helper_experts-b.helper_experts
        <<",\"helper_collect_ms\":"<<st.helper_collect_block_ms-b.helper_collect_block_ms
        <<",\"helper_host_upper_ms\":"<<st.helper_host_upper_ms-b.helper_host_upper_ms
        <<",\"demand_uploads\":"<<st.staged-b.staged
        <<",\"adaptive_issued\":"<<st.adaptive_issued
        <<",\"adaptive_published\":"<<st.adaptive_published
        <<",\"adaptive_reused\":"<<st.adaptive_reused
        <<",\"adaptive_h2d_bytes\":"<<st.adaptive_h2d_bytes
        <<",\"adaptive_d2h_bytes\":"<<st.adaptive_d2h_bytes;
    trace_pending(d);residency_trace<<"}";
    trace_emit();
    return true;
}
