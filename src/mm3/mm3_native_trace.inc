// Opt-in ordered residency evidence. File I/O can perturb timing; disable the
// marker for measured speed requests. No CUDA wait/query is added by tracing.
std::ofstream residency_trace;
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
    residency_trace << ']';
}
void trace_pending(const Dev& d) {
    residency_trace << ",\"staging\":[";
    for(int s=0;s<d.n_slots;++s) {
        if(s)residency_trace<<',';
        residency_trace << '[' << d.slot_owner[s] << ',' << d.slot_pend[s] << ','
            << int(d.slot_iss[s]) << ',' << int(d.slot_foreground[s]) << ','
            << staging_reserved(d,s) << ',' << d.slot_used[s] << ']';
    }
    residency_trace << "],\"exchanges\":[";
    for(size_t i=0;i<d.adaptive.size();++i) {
        if(i)residency_trace<<',';const auto& x=d.adaptive[i];
        residency_trace << '['<<x.incoming<<','<<x.victim<<','<<x.landing<<','<<x.submitted<<']';
    }
    residency_trace << "],\"helper_exchange\":["<<helper_live_a<<','<<helper_live_b<<','
        <<helper_live_withdrawn<<','<<helper_live_published<<']';
}
bool trace_begin(std::string& err) {
    if(residency_trace.is_open())residency_trace.close();
    residency_trace.clear();
    ++residency_trace_request;
    const char* path=::mm3::env("MM3_RESIDENCY_TRACE");
    const char* marker=::mm3::env("MM3_RESIDENCY_TRACE_ENABLE");
    if(!native || !path || !*path || !marker || !*marker || !std::ifstream(marker).good())return true;
    residency_trace.open(path,std::ios::app);
    if(!residency_trace) {err="cannot open requested residency trace";return false;}
    residency_trace.precision(12);
    residency_trace << "{\"type\":\"request\",\"schema\":1,\"request\":"<<residency_trace_request
        <<",\"forward\":"<<fwd<<",\"time_s\":"<<now()<<",\"record_bytes\":"
        <<sizeof(ne::Header)+nf::record_stride(g.dim,g.ff)
        <<",\"lane_cpu1_ms\":"<<lane_cpu1<<",\"lane_cpu_add_ms\":"<<lane_cpu_add
        <<",\"lane_upload_ffn_ms\":"<<lane_copy<<",\"primary_adaptive\":"<<primary_adaptive
        <<",\"helper_live\":"<<helper_live_enabled<<",\"compressed_host\":"<<compressed_host()
        <<",\"policy\":{\"legacy\":"<<adaptive_policy.legacy<<",\"horizon\":"<<adaptive_policy.horizon
        <<",\"max_exchanges\":"<<adaptive_policy.max_exchanges
        <<",\"service_saved_ms\":"<<adaptive_service_ms()<<",\"h2d_ms\":"<<adaptive_policy.h2d_ms
        <<",\"d2h_ms\":"<<adaptive_policy.d2h_ms<<",\"interference_ms\":"<<adaptive_policy.interference_ms
        <<",\"margin_ms\":"<<adaptive_policy.margin_ms<<'}'
        <<",\"layers\":[";
    bool first=true;
    for(size_t li=0;li<L.size();++li)if(L[li].moe) {
        if(!first)residency_trace<<',';first=false;
        residency_trace << "{\"layer\":"<<li<<',';trace_ownership(L[li]);residency_trace<<'}';
    }
    residency_trace << "]}\n";
    residency_trace.flush();
    if(!residency_trace) {err="residency trace header write failed";return false;}
    return true;
}
bool trace_end(std::string& err) {
    if(!residency_trace.is_open())return true;
    residency_trace<<"{\"type\":\"end\",\"request\":"<<residency_trace_request
        <<",\"forward\":"<<fwd<<",\"time_s\":"<<now()
        <<",\"adaptive_issued\":"<<st.adaptive_issued<<",\"adaptive_published\":"<<st.adaptive_published
        <<",\"adaptive_reused\":"<<st.adaptive_reused<<",\"adaptive_h2d_bytes\":"<<st.adaptive_h2d_bytes
        <<",\"adaptive_d2h_bytes\":"<<st.adaptive_d2h_bytes<<"}\n";
    residency_trace.flush();
    if(!residency_trace) {err="residency trace completion write failed";return false;}
    return true;
}
void trace_routes(const Layer& l,const Dev& d,const int* top,int T) {
    if(!residency_trace.is_open())return;
    residency_trace_before=st;residency_trace_layer_begin=now();
    residency_trace << "{\"type\":\"routes\",\"request\":"<<residency_trace_request
        <<",\"event\":"<<++residency_trace_event<<",\"forward\":"<<fwd
        <<",\"generation\":"<<generation_active<<",\"time_s\":"<<residency_trace_layer_begin
        <<",\"layer\":"<<(&l-L.data())<<",\"rows\":"<<T<<",\"positions\":";
    trace_numbers(d.pos_h,size_t(T));
    residency_trace<<",\"original\":";trace_numbers(d.original_h,size_t(T*g.k_exp));
    residency_trace<<",\"executed\":";trace_numbers(top,size_t(T*g.k_exp));
    residency_trace<<',';trace_ownership(l);trace_pending(d);
    residency_trace<<",\"ready_primary\":[";
    for(int j=0;j<4;++j){if(j)residency_trace<<',';residency_trace<<d.ready_h->primary[j];}
    residency_trace<<"],\"ready_helper\":[";
    for(int j=0;j<4;++j){if(j)residency_trace<<',';residency_trace<<d.ready_h->helper[j];}
    residency_trace<<"]}\n";
}
bool trace_service(const Dev& d,std::string& err) {
    if(!residency_trace.is_open())return true;
    const auto& b=residency_trace_before;
    residency_trace << "{\"type\":\"service\",\"request\":"<<residency_trace_request
        <<",\"event\":"<<residency_trace_event<<",\"forward\":"<<fwd
        <<",\"time_s\":"<<now()<<",\"host_service_ms\":"<<1000*(now()-residency_trace_layer_begin)
        <<",\"cpu_helper_lane_ms\":"<<1000*(st.t_lane-b.t_lane)
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
    trace_pending(d);residency_trace<<"}\n";
    if(!residency_trace) {err="residency trace write failed";return false;}
    return true;
}
