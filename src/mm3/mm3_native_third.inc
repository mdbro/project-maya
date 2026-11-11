// The third card's fixed pool (--native-third DEVICE, --native-third-gb GB; off unless both are given).
//
// Copies of HOST experts on one more CUDA device, held in a second NativeHelper. The HOST keeps owning every such
// expert: its record stays in the bank and no ownership rule changes, so a request without third=1 runs the engine
// exactly as it was. With third=1:
//   - a window's pick of a held expert that would otherwise go to the CPU lane or to a staging frame is served by the
//     third card, as one more submission beside the helper's, and its result is added on the host like the helper's;
//   - routing counts held experts as ready (the helper's bits of the ready mask);
//   - the exchange, staging and early-copy policies leave held experts where they are.
// The pool is filled once a load (PLACEMENT third <profile>) with the profile's hottest experts that the HOST owns at
// that moment, and never changes: no exchange touches it. A held expert that a later request without third=1 moves to
// the 5090 or the 3090 is simply served there; the copy on the third card stays valid, because records never change.
std::unique_ptr<NativeHelper> third;
int third_capacity = 0;
bool third_filled = false;
bool third_failed = false;   // a fill failed: the pool stays allocated and unused for the rest of the load
bool third_on = false;   // this generation uses the pool

// Held by the pool and in use for this generation (policy filters: leave it where it is).
bool third_holds(const Layer& l, int e) const {
    return third_on && e >= 0 && e < g.n_exp && ((l.third_ready[e/32] >> (e%32)) & 1u);
}

// Served by the third card in this pass: held, and neither resident, the helper's, nor in a usable staging frame.
bool third_serves(const Layer& l, const Dev& d, int e) const {
    return third_holds(l, e) && on_cpu(l, d, e);
}

bool third_fill(const std::vector<uint32_t>& counts, std::string& report, std::string& err) {
    if(!third) {err="this load has no third pool (--native-third, --native-third-gb)";return false;}
    if(third_filled || third_failed) {err=third_failed?"an earlier fill of the third pool failed; it is unused for this load":"the third pool is filled once a load";return false;}
    if(counts.size()!=size_t(g.n_layers)*size_t(g.n_exp)) {err="third: the profile's size does not match the model";return false;}
    if(placement_failed || retirement_uncertain.load()) {err="third: the engine's placement state is not healthy";return false;}
    struct Pick { uint32_t score; int li, e; };
    std::vector<Pick> picks;
    for(int e=0;e<g.n_exp;++e)for(int li=0;li<g.n_layers;++li) {
        const Layer& l=L[size_t(li)];
        if(l.moe && !l.hf && l.dev==head_dev && l.res_slot[size_t(e)]<0 && !helper_owns(l,e) && pooled(l,e))
            picks.push_back({counts[size_t(li)*size_t(g.n_exp)+size_t(e)],li,e});
    }
    std::stable_sort(picks.begin(),picks.end(),[](const Pick& a,const Pick& b){return a.score>b.score;});
    if(int(picks.size())<third_capacity) {err="third: fewer HOST experts than the pool holds";return false;}
    const size_t stride=nf::record_stride(g.dim,g.ff), blob=sizeof(ne::Header)+stride;
    const double began=now();
    uint8_t* stage=nullptr;
    if(cudaSetDevice(o.native_third)!=cudaSuccess || cudaHostAlloc(&stage,blob,cudaHostAllocPortable)!=cudaSuccess) {
        cudaSetDevice(D[size_t(head_dev)].id);err="third: no portable staging record";return false;
    }
    bool good=true;
    std::ostringstream ids;
    for(int i=0;i<third_capacity && good;++i) {
        const auto pick=picks[size_t(i)];Layer& l=L[size_t(pick.li)];
        if(l.blob!=blob) {err="third: a record is not the shared NVFP4 layout";good=false;break;}
        std::memcpy(stage,l.host_blob[size_t(pick.e)],blob);
        const auto* h=reinterpret_cast<const ne::Header*>(stage);
        good=third->add_expert(pick.li,pick.e,stage+sizeof(ne::Header),stride,h->weight_global,h->input_global,err);
        if(good) {
            l.third_ready[size_t(pick.e/32)]|=1u<<(pick.e%32);
            ids<<(i?",":"")<<'['<<pick.li<<','<<pick.e<<','<<pick.score<<']';
        }
    }
    good=good && third->seal(err);
    // Read every record back and compare it with the HOST's: bytes and scales. The link is slow (about 17 ms a
    // record each way) and this pool is never rewritten, so the check is paid once a load.
    long verified=0,mismatched=0;
    for(int i=0;i<third_capacity && good;++i) {
        const auto pick=picks[size_t(i)];const Layer& l=L[size_t(pick.li)];
        const auto* h=reinterpret_cast<const ne::Header*>(l.host_blob[size_t(pick.e)]);
        float weight[3]={},input[3]={};bool unsafe=false;
        good=third->export_expert(pick.li,pick.e,stage,stride,weight,input,unsafe,err);
        if(!good) {if(unsafe)stage=nullptr;break;}   // (an uncertain reader: the staging record is left allocated)
        const bool same=std::memcmp(stage,l.host_blob[size_t(pick.e)]+sizeof(ne::Header),stride)==0 &&
            std::memcmp(weight,h->weight_global,sizeof weight)==0 && std::memcmp(input,h->input_global,sizeof input)==0;
        ++verified;mismatched+=!same;
    }
    if(good && mismatched) {err="third: "+std::to_string(mismatched)+" records read back differ from the HOST's";good=false;}
    // Compute cross-check: a few held experts on the same rows through the CPU lane (from the HOST record) and
    // through the third card. Reported, not judged here: the largest difference and the largest output.
    double check_diff=0,check_peak=0;int checked=0;
    if(good && lane && native_cpu_workspace) {
        const int rows=2;
        std::vector<float> input(size_t(rows)*g.dim),out_cpu(input.size()),out_third(input.size());
        std::vector<uint16_t> bf16(input.size());
        for(size_t i=0;i<input.size();++i) {
            input[i]=.01f*float(int((i*37)%101)-50);uint32_t u;std::memcpy(&u,&input[i],4);
            bf16[i]=uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
        }
        for(int k=0;k<8 && good;++k) {
            const auto pick=picks[size_t(k)*size_t(third_capacity)/8];Layer& l=L[size_t(pick.li)];
            std::vector<CpuExpert> jobs{cpu_expert(l,pick.e,1.0f)};
            std::vector<std::vector<std::pair<int,float>>> users(1);std::vector<std::pair<int,int>> order;
            for(int t=0;t<rows;++t) {users[0].push_back({t,1.0f});order.push_back({0,t});}
            std::fill(out_cpu.begin(),out_cpu.end(),0.0f);
            lane_run_multi(jobs,users,input.data(),rows,out_cpu.data(),&order);
            if(native_cpu_failed) {err="third: the CPU lane failed in the cross-check";good=false;break;}
            NativeHelper::Job job;job.layer=pick.li;job.expert=pick.e;job.columns=rows;
            for(int t=0;t<rows;++t) {job.token[t]=t;job.weight[t]=1.0f;}
            const auto seq=third->submit(rows,bf16.data(),{job},err);
            good=seq && third->collect(seq,out_third.data(),out_third.size(),err);
            if(!good)break;
            for(size_t i=0;i<out_cpu.size();++i) {
                check_diff=std::max(check_diff,double(std::fabs(out_cpu[i]-out_third[i])));
                check_peak=std::max(check_peak,double(std::fabs(out_cpu[i])));
            }
            ++checked;
        }
    }
    // A failed copy may leave an uncertain reader of the borrowed staging record: keep it allocated then.
    if(good && stage)cudaFreeHost(stage);
    cudaSetDevice(D[size_t(head_dev)].id);
    if(!good) {
        // The pool is left allocated (a failed copy may still be read) and is never used: no mask bit stays set.
        for(auto& l:L)l.third_ready={};
        third_failed=true;third_on=false;
        if(err.empty())err="third: the pool could not be filled";
        return false;
    }
    third_filled=true;
    const auto bytes=third->bytes();
    std::ostringstream out;out.precision(6);
    // (the server checks this digest against the bytes it sent: FNV-1a over the profile's bytes, as placement reports it)
    uint64_t digest=14695981039346656037ull;
    for(uint32_t n:counts)for(int b=0;b<4;++b){digest^=(n>>(8*b))&255u;digest*=1099511628211ull;}
    char hex[17];std::snprintf(hex,sizeof hex,"%016llx",(unsigned long long)digest);
    out<<"{\"profile_fnv64\":\""<<hex<<"\",\"third_device\":"<<o.native_third<<",\"third_capacity\":"<<third_capacity<<",\"third_pool_bytes\":"<<bytes.pool
       <<",\"third_scratch_bytes\":"<<bytes.device_scratch<<",\"third_pinned_bytes\":"<<bytes.pinned
       <<",\"third_fill_ms\":"<<1000*(now()-began)<<",\"third_lowest_score\":"<<picks[size_t(third_capacity-1)].score
       <<",\"third_check_experts\":"<<checked<<",\"third_check_max_diff\":"<<check_diff<<",\"third_check_peak\":"<<check_peak
       <<",\"third_verified\":"<<verified<<",\"third_mismatched\":"<<mismatched<<",\"third_ids\":["<<ids.str()<<"],\"ownership_complete\":true}";
    report=out.str();
    return true;
}
