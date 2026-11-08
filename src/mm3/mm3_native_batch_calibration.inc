// Included in Model::Impl. Explicit setup only: no calibration or waiting in
// window_post. Reuses the actual CPU batch, grouped primary and helper paths.
bool calibrate_batch_staging(std::string& err) {
    batch_staging_costs.valid=false;
    if(!native || !o.native_cooperative || D.size()!=1 || !helper ||
       helper_dispatch_current!="worker" || D[0].n_slots<4 || o.chunk<kWin || !lane || !native_cpu_workspace) {
        err="grouped service calibration requires the native cooperative worker path";return false;
    }
    auto& d=D[0];
    const auto ck=[&](cudaError_t status,const char* what) {
        if(status==cudaSuccess)return true;
        err=std::string(what)+": "+cudaGetErrorString(status);
        retirement_uncertain.store(true);return false;
    };
    if(!ck(cudaSetDevice(d.id),"calibration device") ||
       !ck(cudaStreamSynchronize(d.s),"calibration initial join"))return false;
    // CPU misses draw from a bank much larger than the CPU caches. Repeatedly
    // timing the same four records measures a different working set. Rotate
    // existing HOST records without allocating or flushing any weight copy.
    std::vector<std::pair<const Layer*,int>> cpu_records;
    for(const auto& l:L)if(l.moe)for(int e=0;e<g.n_exp && cpu_records.size()<64;++e)
        if(l.host_blob[e])cpu_records.push_back({&l,e});
    std::vector<NativeHelper::Identity> hr;
    const auto helper_records=helper->records();
    // A real helper submission serves one layer. Find four ready records from
    // one layer rather than assuming the first four global ranks share it.
    for(int li=0;li<g.n_layers && hr.size()<4;++li) {
        hr.clear();
        for(const auto& record:helper_records)if(record.layer==li && hr.size()<4)hr.push_back(record);
    }
    if(cpu_records.size()<4 || hr.size()<4) {err="grouped service calibration needs four HOST and helper records";return false;}
    const std::vector<std::pair<const Layer*,int>> records(cpu_records.begin(),cpu_records.begin()+4);
    size_t cpu_cursor=cpu_records.size()/2;
    const size_t blob=records[0].first->blob;
    if(batch_calibration_host) {err="unretired calibration buffer";return false;}
    if(!ck(cudaHostAlloc(&batch_calibration_host,blob,cudaHostAllocPortable),"calibration readback buffer"))return false;
    std::vector<float> input(size_t(kWin)*g.dim),output(input.size());
    std::vector<uint16_t> helper_input(input.size());
    for(size_t i=0;i<input.size();++i) {
        input[i]=0.01f*float(int((i*37)%101)-50);
        uint32_t u;std::memcpy(&u,&input[i],4);helper_input[i]=uint16_t((u+0x7fffu+((u>>16)&1u))>>16);
    }
    // Synchronous setup copies retire the local input before it can be freed.
    if(!ck(cudaMemcpy(d.xn,input.data(),input.size()*4,cudaMemcpyHostToDevice),"calibration input") ||
       !ck(cudaMemsetAsync(d.sh,0,input.size()*4,d.s),"calibration shared output"))return false;
    for(int i=0;i<4;++i)if(!ck(cudaMemcpyAsync(staging_address(d,i),records[i].first->host_blob[records[i].second],
            blob,cudaMemcpyHostToDevice,d.s),"calibration staged source"))return false;
    if(!ck(cudaStreamSynchronize(d.s),"calibration source join"))return false;
    const size_t off_w=size_t(kWin)*8*8, off_gp=off_w+size_t(kWin)*8*4,
        off_gs=off_gp+size_t(kWin)*8*8, off_ge=off_gs+(size_t(kWin)*8+1)*4,
        off_ng=off_ge+size_t(kWin)*8*4;
    auto gpu=[&](int rows,int groups)->bool {
        std::memset(d.win_plan_h,0,win_plan_bytes());
        auto* pp=reinterpret_cast<unsigned long long*>(d.win_plan_h);
        auto* pw=reinterpret_cast<float*>(d.win_plan_h+off_w);
        auto* gp=reinterpret_cast<unsigned long long*>(d.win_plan_h+off_gp);
        auto* gs=reinterpret_cast<int*>(d.win_plan_h+off_gs);
        auto* ge=reinterpret_cast<int*>(d.win_plan_h+off_ge);
        *reinterpret_cast<int*>(d.win_plan_h+off_ng)=groups;
        for(int e=0;e<groups;++e) {
            gp[e]=reinterpret_cast<unsigned long long>(staging_address(d,e));gs[e]=e*rows;
            for(int t=0;t<rows;++t) {ge[e*rows+t]=t*8+e;pp[t*8+e]=gp[e];pw[t*8+e]=1.0f/groups;}
        }
        gs[groups]=groups*rows;
        if(!ck(cudaMemcpyAsync(d.win_plan_d,d.win_plan_h,win_plan_bytes(),cudaMemcpyHostToDevice,d.s),"calibration GPU plan"))return false;
        x3::WinArgs a;
        a.grp_ptr=reinterpret_cast<const unsigned long long*>(d.win_plan_d+off_gp);
        a.grp_start=reinterpret_cast<const int*>(d.win_plan_d+off_gs);
        a.grp_ent=reinterpret_cast<const int*>(d.win_plan_d+off_ge);
        a.n_groups=reinterpret_cast<const int*>(d.win_plan_d+off_ng);
        a.plan_ptr=reinterpret_cast<const unsigned long long*>(d.win_plan_d);
        a.plan_w=reinterpret_cast<const float*>(d.win_plan_d+off_w);
        a.x=d.xn;a.x_stride=g.dim;a.h=d.win_h;a.part=d.win_part;a.sh_out=d.sh;
        a.out=d.out;a.out_stride=g.dim;a.lay=records[0].first->lay;
        a.T=rows;a.k=g.k_exp;a.n_embd=g.dim;a.n_ff=g.ff;a.limit=g.limit;
        if(!x3::win_experts(a,d.s)) {err="calibration grouped GPU submission";retirement_uncertain.store(true);return false;}
        return ck(cudaStreamSynchronize(d.s),"calibration grouped GPU join");
    };
    auto cpu=[&](int rows,int groups)->bool {
        std::vector<CpuExpert> jobs;std::vector<std::vector<std::pair<int,float>>> users(groups);
        std::vector<std::pair<int,int>> order;
        for(int e=0;e<groups;++e) {
            const auto& record=cpu_records[cpu_cursor++%cpu_records.size()];
            jobs.push_back(cpu_expert(*record.first,record.second,1.0f));
            for(int t=0;t<rows;++t)users[e].push_back({t,1.0f/groups});
        }
        for(int t=0;t<rows;++t)for(int e=0;e<groups;++e)order.push_back({e,t});
        lane_run_multi(jobs,users,input.data(),rows,output.data(),&order);
        if(native_cpu_failed)err="calibration grouped CPU service";
        return !native_cpu_failed;
    };
    auto assist=[&](int rows,int groups)->bool {
        std::vector<NativeHelper::Job> jobs;
        for(int e=0;e<groups;++e)for(int start=0;start<rows;start+=NativeHelper::kMaxColumns) {
            NativeHelper::Job job;job.layer=hr[e].layer;job.expert=hr[e].expert;
            job.columns=std::min(NativeHelper::kMaxColumns,rows-start);
            for(int c=0;c<job.columns;++c) {job.token[c]=start+c;job.weight[c]=1.0f/groups;}
            jobs.push_back(job);
        }
        const auto seq=helper->submit(rows,helper_input.data(),jobs,err);
        if(!seq || !helper->collect(seq,output.data(),size_t(rows)*g.dim,err))return false;
        return ck(cudaSetDevice(d.id),"calibration restore primary device");
    };
    auto measure=[&](auto&& fn,double& ms)->bool {
        if(!fn())return false; // local warm-up, excluded from the three samples
        std::array<double,3> times;
        for(auto& t:times) {const double start=now();if(!fn())return false;t=1000*(now()-start);}
        std::sort(times.begin(),times.end());ms=times[1];return std::isfinite(ms)&&ms>0;
    };
    BatchStagingCosts costs;
    std::array<double,17> c1{},c4{},g4{},h1{},h4{};
    constexpr std::array<int,5> widths{1,2,4,8,16};
    // The 5090 drops into a slow state when it idles between short bursts: on 2026-11-08 back-to-back calibrations measured its
    // H2D at 4.4 ms and its grouped expert at 0.86 ms, against 0.61 and 0.03, and a planner kept such a table. So all of its
    // measurements come first, together, each right after a moment of sustained copies and grouped kernels; an implausible
    // result is measured again after a longer warm-up.
    double primary_warmup_ms=0;int primary_remeasured=0;
    auto warm_primary=[&](double seconds)->bool {
        const double begin=now();
        do {
            if(!ck(cudaMemcpyAsync(staging_address(d,0),records[0].first->host_blob[records[0].second],blob,cudaMemcpyHostToDevice,d.s),
                   "calibration warm-up H2D") || !gpu(16,4))return false;   // gpu() joins the stream
        } while(now()-begin<seconds);
        primary_warmup_ms+=1000*(now()-begin);
        return true;
    };
    auto measure_primary=[&](double warm_seconds)->bool {
        if(!warm_primary(warm_seconds))return false;
        for(int rows=1;rows<=16;++rows)
            if(!measure([&]{return gpu(rows,0);},costs.gpu_base_ms[rows]))return false;
        for(int rows:widths)
            if(!warm_primary(0.02) || !measure([&]{return gpu(rows,4);},g4[rows]))return false;
        if(!warm_primary(0.05))return false;
        return measure([&]{return ck(cudaMemcpyAsync(staging_address(d,0),records[0].first->host_blob[records[0].second],
                  blob,cudaMemcpyHostToDevice,d.s),"calibration H2D") && ck(cudaStreamSynchronize(d.s),"calibration H2D join");},costs.h2d_ms) &&
               measure([&]{return ck(cudaMemcpyAsync(batch_calibration_host,staging_address(d,0),blob,cudaMemcpyDeviceToHost,d.s),"calibration D2H") &&
                  ck(cudaStreamSynchronize(d.s),"calibration D2H join");},costs.d2h_ms);
    };
    auto primary_plausible=[&] {
        return costs.h2d_ms<1.0 && costs.d2h_ms<1.0 && (g4[1]-costs.gpu_base_ms[1])/4.0<0.2;
    };
    if(!measure_primary(0.1))return false;
    while(!primary_plausible() && primary_remeasured<3) {
        ++primary_remeasured;
        if(!measure_primary(0.5))return false;
    }
    // The earlier lane warm-up exercises CPU/primary work only. Give the helper
    // its own bounded sustained work before measuring isolated short passes.
    const double helper_warmup_start=now();size_t helper_warmup_passes=0;
    do {if(!assist(kWin,4))return false;++helper_warmup_passes;}
        while(now()-helper_warmup_start<0.5);
    const double helper_warmup_ms=1000*(now()-helper_warmup_start);
    for(int rows:widths) {
        if(!measure([&]{return cpu(rows,1);},c1[rows]) ||
           !measure([&]{return cpu(rows,4);},c4[rows]) ||
           !measure([&]{return assist(rows,1);},h1[rows]) ||
           !measure([&]{return assist(rows,4);},h4[rows]))return false;
        costs.cpu_extra_ms[rows]=std::max(0.0,(c4[rows]-c1[rows])/3.0);
        costs.gpu_extra_ms[rows]=std::max(0.0,(g4[rows]-costs.gpu_base_ms[rows])/4.0);
        costs.helper_extra_ms[rows]=std::max(0.0,(h4[rows]-h1[rows])/3.0);
    }
    costs.cpu_base_ms=std::max(0.0,c1[1]-costs.cpu_extra_ms[1]);
    costs.helper_base_ms=std::max(0.0,h1[1]-costs.helper_extra_ms[1]);
    for(size_t i=1;i<widths.size();++i)for(int n=widths[i-1]+1;n<widths[i];++n) {
        const double fraction=double(n-widths[i-1])/(widths[i]-widths[i-1]);
        for(auto* values:{&costs.cpu_extra_ms,&costs.gpu_extra_ms,&costs.helper_extra_ms})
            (*values)[n]=(*values)[widths[i-1]]*(1-fraction)+(*values)[widths[i]]*fraction;
    }
    if(!ck(cudaFreeHost(batch_calibration_host),"calibration buffer release"))return false;
    batch_calibration_host=nullptr;
    std::fill(d.slot_owner.begin(),d.slot_owner.end(),-1);
    costs.valid=true;batch_staging_costs=costs;batch_staging_arithmetic=native_primary_current;
    std::ostringstream report;report.precision(17);
    report<<"{\"units\":\"host wall milliseconds, joined setup measurements\",\"arithmetic\":"<<capture::quote(native_primary_current)
          <<",\"temporary_pinned_bytes\":"<<blob<<",\"h2d_ms\":"<<costs.h2d_ms<<",\"d2h_ms\":"<<costs.d2h_ms
          <<",\"cpu_rotating_records\":"<<cpu_records.size()<<",\"cpu_record_working_set_bytes\":"<<cpu_records.size()*blob
          <<",\"helper_warmup_ms\":"<<helper_warmup_ms<<",\"helper_warmup_passes\":"<<helper_warmup_passes
          <<",\"primary_warmup_ms\":"<<primary_warmup_ms<<",\"primary_remeasured\":"<<primary_remeasured
          <<",\"primary_plausible\":"<<(primary_plausible()?"true":"false")
          <<",\"cpu_base_ms\":"<<costs.cpu_base_ms<<",\"helper_base_ms\":"<<costs.helper_base_ms<<",\"samples\":[";
    bool comma=false;
    for(int rows:widths) {if(comma)report<<',';comma=true;
        report<<"{\"columns\":"<<rows<<",\"cpu1_ms\":"<<c1[rows]<<",\"cpu4_ms\":"<<c4[rows]
              <<",\"gpu0_ms\":"<<costs.gpu_base_ms[rows]<<",\"gpu4_ms\":"<<g4[rows]
              <<",\"helper1_ms\":"<<h1[rows]<<",\"helper4_ms\":"<<h4[rows]<<'}';}
    report<<']';
    const auto array=[&](const char* name,const std::array<double,17>& values) {
        report<<",\""<<name<<"\":[";
        for(size_t i=0;i<values.size();++i) {if(i)report<<',';report<<values[i];}report<<']';
    };
    array("cpu_extra_ms",costs.cpu_extra_ms);array("gpu_extra_ms",costs.gpu_extra_ms);
    array("helper_extra_ms",costs.helper_extra_ms);array("gpu_base_ms",costs.gpu_base_ms);
    report<<",\"model\":\"additive grouped-service approximation; max of overlapping lanes; interpolation at unsampled widths; contention and mixed-group fit not qualified\"}";
    batch_staging_calibration=report.str();info.push_back("grouped staging calibration "+batch_staging_calibration);
    if(capture::enabled())capture::event("batch_staging_calibration","\"calibration\":"+batch_staging_calibration);
    return true;
}
