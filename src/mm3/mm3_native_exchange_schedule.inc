// Included in Model::Impl after mm3_native_helper_live.inc. exchange_schedule, off unless a request asks (item 3.1 of the
// plan's order of 2026-11-10; design: runs/exchange-schedule-20261110/design.md). The exchanges' copies are cut into
// pieces and placed in idle time; what is exchanged, and every ownership transition and its checks, stay as they are.
//   0: today's behavior.
//   1: the 3090's x4 pieces only while it has no pass in flight (windows after each collect, in passes with no 3090 job,
//      and from the forward end to early in the next forward's dense prefix); its host copies only from the forward end
//      (after the 5090's copies issued there) to early in the next prefix; the 5090's victim downloads on their own
//      stream, in passes with no upload and no CPU job and at the forward end.
//   2: as 1, every copy compared with its source (the 3090's in the helper, the 5090's before publication).
int exchange_schedule=0;
static constexpr size_t kDownloadPiece=size_t(4)<<20;    // ~0.08 ms of the 5090's 53 GB/s download
static constexpr int kDownloadPiecesPerPass=2;           // ~0.17 ms: ends before the next join in ~95% of passes
static constexpr uint64_t kX4AfterCollectNs=200000;      // under the 0.23 ms 5th percentile from collect to the next join
static constexpr uint64_t kX4NoJobNs=300000;             // a pass whose plan gives the 3090 no job
static constexpr uint64_t kPrefixWindowNs=800000;        // the first MoE join comes ~1.3 ms into a forward
uint64_t exchange_host_open_ns=0;
long exchange_download_pass_pieces=0, exchange_download_flush_pieces=0, exchange_download_windows=0;
long exchange_download_verified=0, exchange_upload_verified=0;

bool exchange_schedule_supported() const {
    return native && o.native_cooperative && D.size()==1 && !o.native_stage_prefix && helper_dispatch_current=="worker";
}
bool exchange_schedule_prepare(std::string& err) {
    for(auto& d:D) {
        if(cudaSetDevice(d.id)!=cudaSuccess) {err="exchange schedule set device";return false;}
        if(!d.xs && cudaStreamCreateWithFlags(&d.xs,cudaStreamNonBlocking)!=cudaSuccess) {
            d.xs=nullptr;err="exchange schedule download stream";return false;
        }
        for(auto& x:d.adaptive)if(!x.h2d_event && cudaEventCreateWithFlags(&x.h2d_event,cudaEventDisableTiming)!=cudaSuccess) {
            x.h2d_event=nullptr;err="exchange schedule upload event";return false;
        }
    }
    return true;
}
// The exchange's copies. Mode 0 issues exactly today's calls: the upload and the victim's download on the copy stream,
// then the completion event. Scheduled: the upload as today with its own event; the download is owed in pieces.
cudaError_t adaptive_submit_copies(Dev& d,Dev::AdaptiveSwap& x,bool reuse) {
    auto status=cudaSuccess;
    if(!exchange_schedule) {
        if(!reuse)status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess && !primary_backing_bank)status=cudaMemcpyAsync(x.target_host,x.source_gpu,adaptive_blob,cudaMemcpyDeviceToHost,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.event,d.cs);
        return status;
    }
    x.scheduled=true;x.h2d_issued=!reuse;x.h2d_done=false;x.d2h_done=0;x.d2h_total=primary_backing_bank?0:adaptive_blob;
    if(!reuse) {
        status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.h2d_event,d.cs);
    }
    if(status==cudaSuccess && !x.d2h_total)status=cudaEventRecord(x.event,d.cs);   // no download owed: as today
    return status;
}
bool exchange_downloads_owed(const Dev& d) const {
    for(const auto& x:d.adaptive)if(x.incoming>=0 && x.d2h_done<x.d2h_total)return true;
    return false;
}
// Exchanges whose upload may still be on the copy stream: what the early copy's guard protects in scheduled mode. One
// issued before scheduling began has both its copies there until publication, as before.
int adaptive_copy_pending(Dev& d) {
    int n=0;
    for(auto& x:d.adaptive)if(x.incoming>=0 && !x.scheduled)++n;
    for(auto& x:d.adaptive)if(x.incoming>=0 && x.scheduled && x.h2d_issued && !x.h2d_done) {
        if(cudaEventQuery(x.h2d_event)==cudaSuccess)x.h2d_done=true;
        else ++n;   // (not ready, or an error that publication will report: either way not idle)
    }
    return n;
}
// Up to max_pieces owed pieces (all if negative) on the download stream. A transaction's completion event is recorded
// after its last piece and behind its upload's event, so it still marks both copies done.
bool exchange_issue_downloads(Dev& d,int max_pieces,long& counter,std::string& err) {
    for(auto& x:d.adaptive) {
        while(x.incoming>=0 && x.submitted && x.d2h_done<x.d2h_total && max_pieces!=0) {
            const size_t n=std::min(kDownloadPiece,x.d2h_total-x.d2h_done);
            auto status=cudaMemcpyAsync(x.target_host+x.d2h_done,x.source_gpu+x.d2h_done,n,cudaMemcpyDeviceToHost,d.xs);
            if(status!=cudaSuccess)return adaptive_fail("exchange download piece",status,err);
            x.d2h_done+=n;++counter;if(max_pieces>0)--max_pieces;
            if(x.d2h_done==x.d2h_total) {
                if(x.h2d_issued)status=cudaStreamWaitEvent(d.xs,x.h2d_event,0);
                if(status==cudaSuccess)status=cudaEventRecord(x.event,d.xs);
                if(status!=cudaSuccess)return adaptive_fail("exchange download completion",status,err);
            }
        }
    }
    return true;
}
// Every owed piece, then the download stream joined: before any join of the copy stream that expects publication.
bool exchange_downloads_join(Dev& d,std::string& err) {
    if(!d.xs)return true;
    if(cudaSetDevice(d.id)!=cudaSuccess) {err="exchange download join device";retirement_uncertain.store(true);return false;}
    if(!exchange_issue_downloads(d,-1,exchange_download_flush_pieces,err))return false;
    const auto status=cudaStreamSynchronize(d.xs);
    if(status!=cudaSuccess)return adaptive_fail("exchange download join",status,err);
    return true;
}
// At the end of a layer's host work. The 5090's link is idle until the next join if this pass staged no demand copy and
// issued no early copy, and no early copy or exchange upload is still in flight; a CPU job reads host memory beside it.
bool exchange_pass_end(Dev& d,bool pass_uploaded,bool cpu_job,std::string& err) {
    if(!exchange_schedule || pass_uploaded || cpu_job || d.n_pending || !d.promo_q.empty() || !exchange_downloads_owed(d))return true;
    if(adaptive_copy_pending(d))return true;
    for(int s=0;s<d.n_slots;++s)if(slot_early_busy(d,s)) {
        const cudaError_t status=cudaEventQuery(d.slot_ev[size_t(s)]);
        if(status==cudaErrorNotReady)return true;
        if(status!=cudaSuccess) {
            err=std::string("early copy completion failed: ")+cudaGetErrorString(status);
            placement_failed=true;retirement_uncertain.store(true);return false;
        }
    }
    ++exchange_download_windows;
    return exchange_issue_downloads(d,kDownloadPiecesPerPass,exchange_download_pass_pieces,err);
}
// The 3090's x4 window, opened by the host at a point where its next pass is at least until_ns away.
void exchange_x4_window(uint64_t span_ns) {
    if(exchange_schedule && helper && helper_live_a>=0)helper->live_x4_window(NativeHelper::clock_ns()+span_ns);
}
// The forward end: no pass runs until the next forward's first MoE join. Every owed download goes out, the x4 window
// opens, and the host window opens once the 5090's copies issued at this end are expected done.
bool exchange_forward_end(std::string& err) {
    if(!exchange_schedule)return true;
    double busy_ms=0;
    for(auto& d:D) {
        int uploads=0;size_t owed=0;
        for(const auto& x:d.adaptive)if(x.incoming>=0) {uploads+=x.h2d_issued && !x.h2d_done;owed+=x.d2h_total-x.d2h_done;}
        if(owed && !exchange_issue_downloads(d,-1,exchange_download_flush_pieces,err))return false;
        if(uploads || owed)busy_ms=std::max(busy_ms,0.7*(uploads*adaptive_h2d_ms()+
            double(owed)/double(std::max<size_t>(adaptive_blob,1))*adaptive_d2h_ms()));
    }
    if(helper) {
        const uint64_t now=NativeHelper::clock_ns();
        exchange_host_open_ns=now+uint64_t(busy_ms*1e6);
        helper->live_x4_window(UINT64_MAX);
        helper->live_host_window(exchange_host_open_ns,UINT64_MAX);
    }
    return true;
}
// The forward start: both 3090 windows close early in the dense prefix, before its first MoE join.
void exchange_forward_begin() {
    if(!exchange_schedule || !helper)return;
    const uint64_t until=NativeHelper::clock_ns()+kPrefixWindowNs;
    helper->live_x4_window(until);
    helper->live_host_window(exchange_host_open_ns,until);
}
// Outside generation, in maintenance and when scheduling stops: no pass runs, every window is open.
void exchange_windows_open() {
    if(helper) {helper->live_x4_window(UINT64_MAX);helper->live_host_window(0,UINT64_MAX);}
}
// exchange_schedule=2: before publication, the victim's host copy against its frame and the incoming's landing frame
// against its host record, both read back through the download stream.
bool exchange_verify_publication(Dev& d,const Dev::AdaptiveSwap& x,std::string& err) {
    if(!d.xs)return true;
    std::vector<uint8_t> back(adaptive_blob);
    const auto read=[&](const uint8_t* frame)->bool {
        auto status=cudaMemcpyAsync(back.data(),frame,adaptive_blob,cudaMemcpyDeviceToHost,d.xs);
        if(status==cudaSuccess)status=cudaStreamSynchronize(d.xs);
        if(status!=cudaSuccess) {adaptive_fail("exchange verify read",status,err);return false;}
        return true;
    };
    if(x.d2h_total) {
        if(!read(x.source_gpu))return false;
        if(std::memcmp(back.data(),x.target_host,adaptive_blob)!=0) {err="exchange verify: the victim's host copy differs from its frame";placement_failed=true;return false;}
        ++exchange_download_verified;
    }
    if(x.h2d_issued) {
        if(!read(x.landing_gpu))return false;
        if(std::memcmp(back.data(),x.source_host,adaptive_blob)!=0) {err="exchange verify: the landing frame differs from its host record";placement_failed=true;return false;}
        ++exchange_upload_verified;
    }
    return true;
}
bool exchange_schedule_set(int mode,std::string& err) {
    if(mode<0 || mode>2) {err="exchange schedule must be 0, 1 or 2 (2: verify)";return false;}
    if(mode==exchange_schedule) {if(helper)helper->set_live_schedule(mode);return true;}
    if(mode && !exchange_schedule_supported()) {err="exchange schedule requires single-GPU native cooperative execution with worker helper dispatch";return false;}
    if(mode && !exchange_schedule_prepare(err))return false;
    if(!mode)for(auto& d:D)if(d.xs && !exchange_downloads_join(d,err))return false;   // (nothing stays owed)
    exchange_schedule=mode;
    if(helper)helper->set_live_schedule(mode);
    exchange_windows_open();
    return true;
}
