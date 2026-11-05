// Explicit, joined setup measurements. Raw hardware observations are separate
// from BatchStagingCosts and never overwrite policy calibration or CPU prices.
bool measure_replay_services(std::string& report,std::string& err) {
    if(!adaptive_supported() || !helper || !lane || !native_cpu_workspace || helper_dispatch_current!="worker") {
        err="replay service probe requires native cooperative worker execution";return false;
    }
    auto& d=D[0];
    std::string joined;
    if(!learned_state("snapshot",joined,err))return false;
    const auto ck=[&](cudaError_t rc,const char* what) {
        if(rc==cudaSuccess)return true;
        err=std::string(what)+": "+cudaGetErrorString(rc);retirement_uncertain.store(true);return false;
    };
    std::vector<std::pair<const Layer*,int>> cpu_records,primary_records;
    for(const auto& l:L)if(l.moe) {
        for(int e=0;e<g.n_exp;++e)if(l.host_blob[e])cpu_records.push_back({&l,e});
        for(int e:l.res_ids)if(primary_records.size()<64)primary_records.push_back({&l,e});
    }
    const auto all_helper=helper->records();std::vector<NativeHelper::Identity> hr;
    for(int li=0;li<g.n_layers;++li) {
        std::vector<NativeHelper::Identity> layer;
        for(auto id:all_helper)if(id.layer==li)layer.push_back(id);
        if(layer.size()>hr.size())hr=std::move(layer);
    }
    if(cpu_records.size()<64 || primary_records.size()<64 || hr.size()<4) {err="insufficient real records for service probe";return false;}
    const size_t blob=cpu_records[0].first->blob;
    if(batch_calibration_host) {err="unretired calibration buffer";return false;}
    if(!ck(cudaSetDevice(d.id),"probe primary device") ||
       !ck(cudaHostAlloc(&batch_calibration_host,blob,cudaHostAllocPortable),"probe portable buffer"))return false;
    // Timers are always retired at their own joined sample. On any uncertain
    // failure the caller quarantines the model and retains the buffer.
    cudaEvent_t start=nullptr,split=nullptr,finish=nullptr;
    if(!ck(cudaEventCreate(&start),"probe timer") || !ck(cudaEventCreate(&split),"probe split timer") ||
       !ck(cudaEventCreate(&finish),"probe finish timer"))return false;
    std::vector<float> input(size_t(16)*g.dim),output(input.size());
    std::vector<uint16_t> helper_input(input.size());
    for(size_t i=0;i<input.size();++i) {
        input[i]=.01f*float(int((i*37)%101)-50);uint32_t bits;std::memcpy(&bits,&input[i],4);
        helper_input[i]=uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
    }
    if(!ck(cudaMemcpy(d.xn,input.data(),input.size()*4,cudaMemcpyHostToDevice),"probe input") ||
       !ck(cudaMemset(d.sh,0,input.size()*4),"probe shared output"))return false;
    std::ostringstream out;out.precision(17);
    out<<"{\"schema\":1,\"units\":\"ms\",\"method\":\"independent real-kernel service probe, synthetic activations, five samples after warmup\""
       <<",\"record_bytes\":"<<blob<<",\"cpu_rotating_records\":"<<cpu_records.size()<<",\"working_set_bytes\":"<<cpu_records.size()*blob
       <<",\"cpu_working_set\":\"full_bank\""
       <<",\"cpu_participants\":"<<lane->threads()
       <<",\"worker_measurement\":\"one CPU pool exists at a time\""
       <<",\"arithmetic\":"<<capture::quote(native_primary_current)<<",\"measurements\":[";
    bool comma=false;size_t cursor=0;
    const auto emit=[&](const char* kind,int rows,const std::vector<int>& groups,const std::vector<double>& samples,
                        int workers=0,size_t bytes=0,const char* working_set="resident") {
        if(comma)out<<',';comma=true;
        out<<"{\"kind\":"<<capture::quote(kind)<<",\"rows\":"<<rows<<",\"groups\":[";
        for(size_t i=0;i<groups.size();++i){if(i)out<<',';out<<groups[i];}
        out<<"],\"workers\":"<<(workers?std::to_string(workers):"\"none\"")<<",\"working_set\":"<<capture::quote(working_set)
           <<",\"bytes\":"<<bytes<<",\"samples_ms\":[";
        for(size_t i=0;i<samples.size();++i){if(i)out<<',';out<<samples[i];}
        out<<"],\"evidence\":[\"native_replay_services\"],\"relative_uncertainty\":0.05}";
    };
    const auto mapping=[](int rows,const std::vector<int>& groups) {
        std::vector<std::vector<std::pair<int,float>>> users(groups.size());std::vector<int> load(rows,0);
        for(size_t e=0;e<groups.size();++e) {
            std::vector<int> order(rows);for(int t=0;t<rows;++t)order[t]=t;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return load[a]<load[b];});
            for(int i=0;i<groups[e];++i) {const int t=order[i];users[e].push_back({t,.25f});++load[t];}
        }
        return users;
    };
    const size_t off_w=size_t(kWin)*8*8,off_gp=off_w+size_t(kWin)*8*4,
        off_gs=off_gp+size_t(kWin)*8*8,off_ge=off_gs+(size_t(kWin)*8+1)*4,off_ng=off_ge+size_t(kWin)*8*4;
    const int original_workers=lane->threads()-1;
    struct RestoreLane {
        std::unique_ptr<CpuLane>& lane;int workers,dim,ff;float alpha,limit;std::atomic<bool>& uncertain;
        ~RestoreLane(){try {if(!lane || lane->threads()!=workers+1) {lane.reset();lane=std::make_unique<CpuLane>(workers,dim,ff,alpha,limit);}}
            catch(...) {uncertain.store(true);}}
    } restore_lane{lane,original_workers,g.dim,g.ff,g.alpha,g.limit,retirement_uncertain};
    std::vector<uint8_t> representative_plan;
    x3::WinArgs representative_args;
    for(int rows:{1,2,3,4,5,6,7,8,9,16})for(int count:{0,1,2,4,8,12,16,20,32,64})for(int pattern=0;pattern<3;++pattern) {
        if(count>rows*4)continue;
        if((count==0 || rows==1) && pattern)continue;
        std::vector<int> groups(count,0);
        const int full=std::min(count*rows,rows*4);
        const int total=pattern==0?full:pattern==1?count:std::max(count,std::min(count*rows,rows*2));
        if((pattern==1 && total==full) || (pattern==2 && (total==full || total==count)))continue;
        for(int i=0;i<total;++i)++groups[size_t(i%std::max(1,count))];
        const auto users=mapping(rows,groups);
        std::vector<std::pair<int,int>> order;
        for(int t=0;t<rows;++t)for(int e=0;e<count;++e)
            for(size_t i=0;i<users[e].size();++i)if(users[e][i].first==t)order.push_back({e,int(i)});
        if(count)for(int workers:{original_workers+1,8}) {
            if(workers==8 && original_workers==7)continue;
            // A second idle pool still spins on the same pinned cores. Destroy
            // the old pool before measuring another configuration; allocation
            // and thread startup are outside the samples.
            if(lane->threads()!=workers) {lane.reset();lane=std::make_unique<CpuLane>(workers-1,g.dim,g.ff,g.alpha,g.limit);}
            if(lane->threads()!=workers) {err="requested probe CPU configuration is unavailable";return false;}
            CpuLane& service=*lane;
            std::vector<double> samples;
            for(int rep=-1;rep<5;++rep) {
                std::vector<NativeCpuExpert> jobs;
                for(int e=0;e<count;++e) {
                    const auto& r=cpu_records[cursor++%cpu_records.size()];NativeCpuExpert job;job.blob=r.first->host_blob[r.second];jobs.push_back(job);
                }
                const double began=now();
                if(!native_cpu_run_multi_batch(service,g.dim,g.ff,jobs,users,input.data(),rows,output.data(),&order,native_cpu_workspace,native_cpu_bytes)) {
                    err="probe CPU batch failed";return false;
                }
                if(rep>=0)samples.push_back(1000*(now()-began));
            }
            emit("cpu",rows,groups,samples,workers,0,"full_bank");
        }
        if(lane->threads()!=original_workers+1) {lane.reset();lane=std::make_unique<CpuLane>(original_workers,g.dim,g.ff,g.alpha,g.limit);}
        if(!ck(cudaSetDevice(d.id),"probe primary restore"))return false;
        std::memset(d.win_plan_h,0,win_plan_bytes());
        auto* pp=reinterpret_cast<unsigned long long*>(d.win_plan_h);auto* pw=reinterpret_cast<float*>(d.win_plan_h+off_w);
        auto* gp=reinterpret_cast<unsigned long long*>(d.win_plan_h+off_gp);auto* gs=reinterpret_cast<int*>(d.win_plan_h+off_gs);
        auto* ge=reinterpret_cast<int*>(d.win_plan_h+off_ge);int entry=0;std::array<int,16> slots{};
        for(int e=0;e<count;++e) {
            const auto& r=primary_records[size_t(e)];gp[e]=uintptr_t(resident_address(*r.first,r.first->res_slot[r.second]));gs[e]=entry;
            for(auto u:users[e]) {const int at=u.first*8+slots[u.first]++;ge[entry++]=at;pp[at]=gp[e];pw[at]=.25f;}
        }
        gs[count]=entry;*reinterpret_cast<int*>(d.win_plan_h+off_ng)=count;
        if(!ck(cudaMemcpy(d.win_plan_d,d.win_plan_h,win_plan_bytes(),cudaMemcpyHostToDevice),"probe primary plan"))return false;
        x3::WinArgs a;a.grp_ptr=reinterpret_cast<unsigned long long*>(d.win_plan_d+off_gp);
        a.grp_start=reinterpret_cast<int*>(d.win_plan_d+off_gs);a.grp_ent=reinterpret_cast<int*>(d.win_plan_d+off_ge);
        a.n_groups=reinterpret_cast<int*>(d.win_plan_d+off_ng);a.plan_ptr=reinterpret_cast<unsigned long long*>(d.win_plan_d);
        a.plan_w=reinterpret_cast<float*>(d.win_plan_d+off_w);a.x=d.xn;a.x_stride=g.dim;a.h=d.win_h;a.part=d.win_part;
        a.sh_out=d.sh;a.out=d.out;a.out_stride=g.dim;a.lay=primary_records[0].first->lay;a.T=rows;a.k=g.k_exp;a.n_embd=g.dim;a.n_ff=g.ff;a.limit=g.limit;
        if(rows==5 && count==8) {representative_plan.assign(d.win_plan_h,d.win_plan_h+win_plan_bytes());representative_args=a;}
        std::vector<double> compute,reduce,submit;
        for(int rep=-1;rep<5;++rep) {
            const double began=now();
            if(!ck(cudaEventRecord(start,d.s),"probe primary start") || !ne::window_scatter(a,rows*g.k_exp,d.s) ||
               !ck(cudaEventRecord(split,d.s),"probe primary split") || !ne::window_combine(a,d.s) ||
               !ck(cudaEventRecord(finish,d.s),"probe primary finish")) {err="probe primary submission failed";return false;}
            const double host_ms=1000*(now()-began);
            if(!ck(cudaEventSynchronize(finish),"probe primary join"))return false;
            float c=0,r=0;if(!ck(cudaEventElapsedTime(&c,start,split),"probe primary compute") || !ck(cudaEventElapsedTime(&r,split,finish),"probe primary reduce"))return false;
            if(rep>=0){compute.push_back(c);reduce.push_back(r);submit.push_back(host_ms);}
        }
        emit("primary",rows,groups,compute);emit("primary_reduce",rows,{},reduce);emit("primary_submit",rows,groups,submit);
        if(count==0) {
            std::vector<double> bookkeeping;
            auto* flags=reinterpret_cast<int*>(d.win_cpu+size_t(kWin)*g.dim);
            for(int rep=0;rep<5;++rep) {
                const double began=now();
                for(int iteration=0;iteration<256;++iteration) {
                    for(int t=0;t<rows;++t)reinterpret_cast<volatile int*>(flags)[t]=0;
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                }
                bookkeeping.push_back(1000*(now()-began)/256);
            }
            emit("host_reduce",rows,{},bookkeeping);
        }
        if(count && count<=int(hr.size())) {
            std::vector<NativeHelper::Job> jobs;
            for(int e=0;e<count;++e)for(int first=0;first<groups[e];first+=8) {
                NativeHelper::Job j;j.layer=hr[e].layer;j.expert=hr[e].expert;j.columns=std::min(8,groups[e]-first);
                for(int c=0;c<j.columns;++c){j.token[c]=users[e][first+c].first;j.weight[c]=.25f;}jobs.push_back(j);
            }
            std::vector<double> prep,activation,service,result,reduction,queue,total;
            for(int rep=-1;rep<5;++rep) {
                const double began=now();const auto seq=helper->submit(rows,helper_input.data(),jobs,err);const double prepared=now();
                NativeHelper::PassTiming t;
                if(!seq || !helper->collect(seq,output.data(),size_t(rows)*g.dim,err,&t))return false;
                if(!t.gpu_measured){err="replay services require MM3_REPLAY_CAPTURE=1 for independent helper GPU timers";return false;}
                if(rep>=0) {prep.push_back(1000*(prepared-began));activation.push_back(t.activation_gpu_ms);service.push_back(t.compute_gpu_ms);
                    result.push_back(t.result_gpu_ms);reduction.push_back((t.collect_done_ns-t.collect_wait_end_ns)*1e-6);
                    queue.push_back((t.worker_start_ns-t.enqueue_ns)*1e-6);total.push_back(1000*(now()-began));}
            }
            emit("helper_prepare",rows,groups,prep);emit("h_activation",rows,groups,activation);emit("helper",rows,groups,service);
            emit("h_result",rows,groups,result);emit("helper_reduce",rows,groups,reduction);emit("helper_dispatch",rows,groups,queue);emit("helper_total",rows,groups,total);
        }
    }
    // Direction/device-specific weight transfers, queued one record at a time.
    // Helper reads and writes the same idle resident slot with unchanged bytes.
    for(int device:{d.id,o.native_helper}) {
        if(!ck(cudaSetDevice(device),"probe copy device"))return false;
        cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
        if(!ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"probe copy stream") ||
           !ck(cudaEventCreate(&begin),"probe copy timer") || !ck(cudaEventCreate(&end),"probe copy timer"))return false;
        const bool primary=device==d.id;const size_t bytes=primary?blob:blob-sizeof(ne::Header);
        void* gpu=primary?static_cast<void*>(staging_address(d,0)):reinterpret_cast<void*>(helper->pool_identity()+size_t(hr[0].slot)*helper->record_stride());
        if(!primary && !ck(cudaMemcpy(batch_calibration_host,gpu,bytes,cudaMemcpyDeviceToHost),"probe preserve helper record"))return false;
        for(bool upload:{true,false}) {
            std::vector<double> samples;
            for(int rep=-1;rep<5;++rep) {
                const void* src=upload?(primary?static_cast<const void*>(cpu_records[size_t(rep+1)%64].first->host_blob[cpu_records[size_t(rep+1)%64].second]):batch_calibration_host):gpu;
                void* dst=upload?gpu:batch_calibration_host;
                if(!ck(cudaEventRecord(begin,stream),"probe transfer begin") ||
                   !ck(cudaMemcpyAsync(dst,src,bytes,upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,stream),"probe transfer") ||
                   !ck(cudaEventRecord(end,stream),"probe transfer end") || !ck(cudaEventSynchronize(end),"probe transfer join"))return false;
                float ms=0;if(!ck(cudaEventElapsedTime(&ms,begin,end),"probe transfer elapsed"))return false;if(rep>=0)samples.push_back(ms);
            }
            emit(primary?(upload?"p_weight_h2d":"p_weight_d2h"):(upload?"h_weight_h2d":"h_weight_d2h"),1,{},samples,0,bytes);
        }
        cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(stream);
    }
    if(!ck(cudaSetDevice(d.id),"probe restore device"))return false;
    std::vector<double> host_copy;
    for(int rep=-1;rep<5;++rep) {
        const auto& r=cpu_records[cursor++%cpu_records.size()];const double start=now();
        std::memcpy(batch_calibration_host,r.first->host_blob[r.second]+sizeof(ne::Header),blob-sizeof(ne::Header));
        if(rep>=0)host_copy.push_back(1000*(now()-start));
    }
    emit("host_weight_copy",1,{},host_copy,0,blob-sizeof(ne::Header));
    // CPU plus copies: keep the working set and grouping fixed, record both
    // branches, and retain the actual overlap duration for independent fitting.
    out<<"],\"contention_observations\":[";bool cont_comma=false;
    const int cr=5;const std::vector<int> cg{3,3,3,3,2,2,2,2};const auto cu=mapping(cr,cg);
    std::vector<std::pair<int,int>> co;
    for(int t=0;t<cr;++t)for(int e=0;e<int(cg.size());++e)
        for(size_t i=0;i<cu[e].size();++i)if(cu[e][i].first==t)co.push_back({e,int(i)});
    const auto cpu_sample=[&](double& ms) {
        std::vector<NativeCpuExpert> jobs;
        for(size_t e=0;e<cg.size();++e) {const auto& r=cpu_records[cursor++%cpu_records.size()];NativeCpuExpert job;job.blob=r.first->host_blob[r.second];jobs.push_back(job);}
        const double began=now();
        const bool ok=native_cpu_run_multi_batch(*lane,g.dim,g.ff,jobs,cu,input.data(),cr,output.data(),&co,native_cpu_workspace,native_cpu_bytes);
        ms=1000*(now()-began);return ok;
    };
    for(int device:{d.id,o.native_helper})for(bool upload:{true,false}) {
        if(!ck(cudaSetDevice(device),"contention copy device"))return false;
        cudaStream_t stream=nullptr;cudaEvent_t begin=nullptr,end=nullptr;
        if(!ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"contention stream") ||
           !ck(cudaEventCreate(&begin),"contention timer") || !ck(cudaEventCreate(&end),"contention timer"))return false;
        const bool primary=device==d.id;const size_t bytes=primary?blob:blob-sizeof(ne::Header);
        void* gpu=primary?static_cast<void*>(staging_address(d,0)):reinterpret_cast<void*>(helper->pool_identity()+size_t(hr[0].slot)*helper->record_stride());
        if(!primary && !ck(cudaMemcpy(batch_calibration_host,gpu,bytes,cudaMemcpyDeviceToHost),"contention preserve helper"))return false;
        const auto launch=[&]() {
            if(!ck(cudaEventRecord(begin,stream),"contention begin"))return false;
            for(int n=0;n<4;++n) {
                const auto& r=cpu_records[cursor++%64];
                const void* src=upload?(primary?static_cast<const void*>(r.first->host_blob[r.second]):batch_calibration_host):gpu;
                if(!ck(cudaMemcpyAsync(upload?gpu:batch_calibration_host,src,bytes,
                    upload?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,stream),"contention copy"))return false;
            }
            return ck(cudaEventRecord(end,stream),"contention end");
        };
        for(int rep=0;rep<5;++rep) {
            double isolated_cpu=0,concurrent_cpu=0;float isolated_copy=0,concurrent_copy=0;
            if(!cpu_sample(isolated_cpu) || !launch() || !ck(cudaEventSynchronize(end),"isolated copy join") ||
               !ck(cudaEventElapsedTime(&isolated_copy,begin,end),"isolated copy time") || !launch() || !cpu_sample(concurrent_cpu) ||
               !ck(cudaEventSynchronize(end),"concurrent copy join") || !ck(cudaEventElapsedTime(&concurrent_copy,begin,end),"concurrent copy time")) {
                if(err.empty())err="CPU/copy contention probe failed";return false;
            }
            if(cont_comma)out<<',';cont_comma=true;
            out<<"{\"a\":\"cpu\",\"b\":"<<capture::quote(primary?(upload?"p_weight_h2d":"p_weight_d2h"):(upload?"h_weight_h2d":"h_weight_d2h"))
               <<",\"rows\":5,\"groups\":[3,3,3,3,2,2,2,2],\"workers\":"<<lane->threads()
               <<",\"queued_bytes\":"<<4*bytes<<",\"isolated_a_ms\":"<<isolated_cpu<<",\"isolated_b_ms\":"<<isolated_copy
               <<",\"concurrent_a_ms\":"<<concurrent_cpu<<",\"concurrent_b_ms\":"<<concurrent_copy
               <<",\"evidence\":[\"native_replay_services\"]}";
        }
        cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(stream);
    }
    if(!ck(cudaSetDevice(d.id),"probe restore primary"))return false;
    if(representative_plan.empty() || !ck(cudaMemcpy(d.win_plan_d,representative_plan.data(),win_plan_bytes(),cudaMemcpyHostToDevice),"contention grouped plan"))return false;
    cudaEvent_t copy_begin=nullptr,copy_end=nullptr;
    if(!ck(cudaEventCreate(&copy_begin),"contention primary copy timer") || !ck(cudaEventCreate(&copy_end),"contention primary copy timer"))return false;
    const auto primary_copy=[&]() {
        if(!ck(cudaEventRecord(copy_begin,d.cs),"contention primary copy start"))return false;
        for(int i=0;i<4;++i) {const auto& r=cpu_records[cursor++%64];
            if(!ck(cudaMemcpyAsync(staging_address(d,0),r.first->host_blob[r.second],blob,cudaMemcpyHostToDevice,d.cs),"contention primary H2D"))return false;}
        return ck(cudaEventRecord(copy_end,d.cs),"contention primary copy end");
    };
    const auto primary_compute=[&]() {
        return ck(cudaEventRecord(start,d.s),"contention primary start") &&
            ne::window_scatter(representative_args,20,d.s) && ne::window_combine(representative_args,d.s) &&
            ck(cudaEventRecord(finish,d.s),"contention primary finish");
    };
    const auto pair_row=[&](const char* a,const char* b,float ia,float ib,float ca,float cb) {
        if(cont_comma)out<<',';cont_comma=true;
        out<<"{\"a\":"<<capture::quote(a)<<",\"b\":"<<capture::quote(b)
           <<",\"rows\":5,\"groups\":[3,3,3,3,2,2,2,2],\"queued_bytes\":"<<4*blob
           <<",\"isolated_a_ms\":"<<ia<<",\"isolated_b_ms\":"<<ib
           <<",\"concurrent_a_ms\":"<<ca<<",\"concurrent_b_ms\":"<<cb<<",\"evidence\":[\"native_replay_services\"]}";
    };
    for(int rep=0;rep<5;++rep) {
        float ia=0,ib=0,ca=0,cb=0;
        if(!primary_compute() || !ck(cudaEventSynchronize(finish),"isolated GPU join") || !ck(cudaEventElapsedTime(&ia,start,finish),"isolated GPU time") ||
           !primary_copy() || !ck(cudaEventSynchronize(copy_end),"isolated copy join") || !ck(cudaEventElapsedTime(&ib,copy_begin,copy_end),"isolated copy time") ||
           !primary_copy() || !primary_compute() || !ck(cudaEventSynchronize(finish),"concurrent GPU join") ||
           !ck(cudaEventSynchronize(copy_end),"concurrent copy join") || !ck(cudaEventElapsedTime(&ca,start,finish),"concurrent GPU time") ||
           !ck(cudaEventElapsedTime(&cb,copy_begin,copy_end),"concurrent copy time"))return false;
        pair_row("primary","p_weight_h2d",ia,ib,ca,cb);
    }
    // Both device links active. Helper receives its own unchanged record.
    if(!ck(cudaSetDevice(o.native_helper),"cross-link helper device"))return false;
    cudaStream_t hs=nullptr;cudaEvent_t hb=nullptr,he=nullptr;
    if(!ck(cudaStreamCreateWithFlags(&hs,cudaStreamNonBlocking),"cross-link helper stream") ||
       !ck(cudaEventCreate(&hb),"cross-link timer") || !ck(cudaEventCreate(&he),"cross-link timer"))return false;
    void* helper_gpu=reinterpret_cast<void*>(helper->pool_identity()+size_t(hr[0].slot)*helper->record_stride());
    if(!ck(cudaMemcpy(batch_calibration_host,helper_gpu,blob-sizeof(ne::Header),cudaMemcpyDeviceToHost),"cross-link preserve helper"))return false;
    const auto helper_copy=[&]() {
        if(!ck(cudaSetDevice(o.native_helper),"cross-link helper") || !ck(cudaEventRecord(hb,hs),"cross-link start"))return false;
        for(int i=0;i<4;++i)if(!ck(cudaMemcpyAsync(helper_gpu,batch_calibration_host,blob-sizeof(ne::Header),cudaMemcpyHostToDevice,hs),"cross-link H2D"))return false;
        return ck(cudaEventRecord(he,hs),"cross-link end");
    };
    for(int rep=0;rep<5;++rep) {
        float ia=0,ib=0,ca=0,cb=0;
        if(!ck(cudaSetDevice(d.id),"cross-link primary") || !primary_copy() || !ck(cudaEventSynchronize(copy_end),"cross-link isolated primary join") ||
           !ck(cudaEventElapsedTime(&ia,copy_begin,copy_end),"cross-link isolated primary time") || !helper_copy() ||
           !ck(cudaEventSynchronize(he),"cross-link isolated helper join") || !ck(cudaEventElapsedTime(&ib,hb,he),"cross-link isolated helper time") ||
           !ck(cudaSetDevice(d.id),"cross-link primary") || !primary_copy() || !helper_copy() || !ck(cudaEventSynchronize(he),"cross-link concurrent helper join") ||
           !ck(cudaEventElapsedTime(&cb,hb,he),"cross-link concurrent helper time") || !ck(cudaSetDevice(d.id),"cross-link primary") ||
           !ck(cudaEventSynchronize(copy_end),"cross-link concurrent primary join") || !ck(cudaEventElapsedTime(&ca,copy_begin,copy_end),"cross-link concurrent primary time"))return false;
        pair_row("p_weight_h2d","h_weight_h2d",ia,ib,ca,cb);
    }
    if(!ck(cudaSetDevice(o.native_helper),"cross-link release helper"))return false;
    cudaEventDestroy(hb);cudaEventDestroy(he);cudaStreamDestroy(hs);
    if(!ck(cudaSetDevice(d.id),"probe restore primary"))return false;
    cudaEventDestroy(copy_begin);cudaEventDestroy(copy_end);cudaEventDestroy(start);cudaEventDestroy(split);cudaEventDestroy(finish);
    if(!ck(cudaFreeHost(batch_calibration_host),"probe buffer release"))return false;batch_calibration_host=nullptr;
    std::fill(d.slot_owner.begin(),d.slot_owner.end(),-1);
    out<<"],\"contention\":[],\"limitations\":[\"raw overlap observations require separate fitting and held-out validation\",\"other directions, shapes and multiway contention remain unqualified\",\"synthetic activations and finite working set\",\"unmeasured shapes and worker configurations are not qualified\"]}";
    report=out.str();return true;
}
