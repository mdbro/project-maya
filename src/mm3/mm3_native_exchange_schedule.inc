// Included in Model::Impl after mm3_native_helper_live.inc. exchange_schedule, off unless a request asks (item 3.1 of the
// plan's order of 2026-11-10; design: runs/exchange-schedule-20261110/design.md). The exchanges' copies are cut into
// pieces and placed in idle time; what is exchanged, and every ownership transition and its checks, stay as they are.
//   0: today's behavior.
//   1: the 3090's x4 pieces only while it has no pass in flight (windows after each collect, in passes with no 3090 job,
//      and from the forward end to early in the next forward's dense prefix); its host copies only from the forward end
//      to early in the next prefix, unless one has waited exchange_host_wait forward ends (then it runs at once, as
//      before, so the 3090's exchange rate is kept); the 5090's victim downloads on their own stream, in passes with no
//      upload and no CPU job and at the forward end.
//      (v2, 2026-11-10: with the GPU nucleus a token has ~0.03 ms between forwards, so v1's host window, opened only once
//       the 5090's forward-end copies were expected done and closed 0.8 ms into the prefix, starved the 3090's leases.)
//   2: as 1, every copy compared with its source (the 3090's in the helper, the 5090's before publication).
int exchange_schedule=0;
// exchange_quiet (kernel overhaul follow-up, plan step 6 "scheduled swaps in drafting"; a request's setting, 0 by default,
// effective with exchange_schedule): the exchanges' copies only in quiet moments read from the engine's live state, as the
// user's free-time decision of 2026-11-13 sets them (CLAUDE.md, Placement: the CPU lane idle, the card's link idle, no
// 5090 upload in flight). The rate, the cards and what is exchanged stay as they are; no fixed cadence.
//  - the 5090's exchange upload is owed in pieces like its victim's download (today it goes out at once on the copy
//    stream, beside the pass's demand uploads on the same link: the user's "keep refills off demand copies"); both go
//    out in passes with no demand upload and no CPU job, and at the forward end - in drafting the draft call follows,
//    a few milliseconds of idle host memory and link;
//  - the 3090's x4 pieces: its windows open from each collect (and in passes without a 3090 job) until its next pass is
//    submitted, instead of fixed spans calibrated at one row, and a piece starts only while no 5090 demand upload of the
//    pass is in flight and the CPU lane is idle.
int exchange_quiet=0;
static constexpr size_t kDownloadPiece=size_t(4)<<20;    // ~0.08 ms of the 5090's 53 GB/s download
static constexpr int kDownloadPiecesPerPass=2;           // ~0.17 ms: ends before the next join in ~95% of passes
static constexpr uint64_t kX4AfterCollectNs=200000;      // under the 0.23 ms 5th percentile from collect to the next join
static constexpr uint64_t kX4NoJobNs=300000;             // a pass whose plan gives the 3090 no job
static constexpr uint64_t kPrefixWindowNs=1200000;       // the first MoE join comes 1.55 ms (5th percentile) into a forward
int exchange_host_wait=2;   // (per request) a lease's host copy that waited this many forward ends runs at once; 0: from the first
uint64_t exchange_host_open_ns=0;
int exchange_host_waits=0;bool exchange_host_urgent=false;
long exchange_host_urgent_forwards=0;
long exchange_download_pass_pieces=0, exchange_download_flush_pieces=0, exchange_download_windows=0;
long exchange_download_verified=0, exchange_upload_verified=0;
long exchange_upload_pieces=0;   // (exchange_quiet: the 5090's exchange uploads sent in pieces)

bool exchange_quiet_active() const {return helper && ((exchange_schedule && exchange_quiet) || free_active());}
// The foreground producer revokes before submitting any new weight upload.
// An already-admitted helper chunk is allowed to retire; no foreground wait is
// introduced. Its admission and actual CUDA-submitted overlap have separate receipts.
void exchange_quiet_begin(Dev& d,unsigned streams=15) {
    if(!exchange_quiet_active())return;
    if(!d.quiet_pending) {
        d.quiet_generation=helper->live_x4_revoke();d.quiet_pending=true;
        if(free_active()) {
            third_quiet_generation=third->live_x4_revoke();
            if(host_resorter)resort_quiet_generation=host_resorter->guard.revoke().token;
        }
    }
    d.quiet_dirty|=streams;
}
bool exchange_quiet_failed(const char* operation,cudaError_t status,std::string& err) {
    (void)adaptive_fail(operation,status,err);
    if(helper)helper->live_x4_guard_failed(err);
    if(free_active()) {
        background_traffic.poison();if(third)third->live_x4_guard_failed(err);
        if(host_resorter)host_resorter->guard.revoke();
    }
    return false;
}
// Record only dirty streams independently. Existing retained markers preserve
// earlier outstanding uploads on the others. There is deliberately no wait edge
// added to d.s: ready expert compute still runs beside demand uploads on d.us.
// Markers cover each stream's queued prefix, so d.s/d.xs compute/download work
// ahead of a marker can conservatively delay optional progress. Measure that
// lost opportunity separately from correctness; do not call it copy-only time.
bool exchange_quiet_uploads(Dev& d,std::string& err) {
    if(!exchange_quiet_active())return true;
    if(!d.quiet_pending)return true; // a forward/no-copy boundary adds no markers
    auto status=cudaSuccess;
    if(d.quiet_dirty) {
        status=cudaSetDevice(d.id);
        if(status!=cudaSuccess)return exchange_quiet_failed("quiet upload guard device",status,err);
    }
    const std::array<decltype(d.s),4> streams{{d.s,d.us,d.cs,d.xs}};
    std::array<void*,4> events{};
    for(size_t i=0;i<streams.size();++i) {
        if(!streams[i])continue;
        if((d.quiet_dirty&(1u<<i)) && !d.quiet_uploads[i]) {
            status=cudaEventCreateWithFlags(&d.quiet_uploads[i],cudaEventDisableTiming);
            if(status!=cudaSuccess)return exchange_quiet_failed("quiet upload guard create",status,err);
        }
        if(d.quiet_dirty&(1u<<i)) {
            status=cudaEventRecord(d.quiet_uploads[i],streams[i]);
            if(status!=cudaSuccess)return exchange_quiet_failed("quiet upload guard record",status,err);
        }
        events[i]=d.quiet_uploads[i];
    }
    if(!helper->live_x4_publish(d.quiet_generation,events,err)) {
        placement_failed=true;retirement_uncertain.store(true);
        helper->live_x4_guard_failed(err);return false;
    }
    if(free_active()) {
        if(!third->live_x4_publish(third_quiet_generation,events,err) ||
           (host_resorter && !host_resorter->guard.publish(resort_quiet_generation,events))) {
            err="shared mover upload publication failed";placement_failed=true;retirement_uncertain.store(true);
            background_traffic.poison();return false;
        }
    }
    d.quiet_pending=false;d.quiet_dirty=0;
    return true;
}

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
    x.submitted_h2d=x.submitted_d2h=0;
    if(!exchange_schedule) {
        if(!reuse)status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess && !reuse)x.submitted_h2d=adaptive_blob;
        if(status==cudaSuccess && !primary_backing_bank) {
            status=cudaMemcpyAsync(x.target_host,x.source_gpu,adaptive_blob,cudaMemcpyDeviceToHost,d.cs);
            if(status==cudaSuccess)x.submitted_d2h=adaptive_blob;
        }
        if(status==cudaSuccess)status=cudaEventRecord(x.event,d.cs);
        return status;
    }
    x.scheduled=true;x.h2d_issued=!reuse;x.h2d_done=false;x.d2h_done=0;x.d2h_total=primary_backing_bank?0:adaptive_blob;
    x.h2d_sent=0;x.h2d_total=0;
    if(exchange_quiet && !reuse) {   // (the upload owed in pieces too; the event after the last piece of either)
        x.h2d_issued=false;x.h2d_total=adaptive_blob;
        return status;
    }
    if(!reuse) {
        status=cudaMemcpyAsync(x.landing_gpu,x.source_host,adaptive_blob,cudaMemcpyHostToDevice,d.cs);
        if(status==cudaSuccess)status=cudaEventRecord(x.h2d_event,d.cs);
    }
    if(status==cudaSuccess && !x.d2h_total)status=cudaEventRecord(x.event,d.cs);   // no download owed: as today
    return status;
}
bool exchange_downloads_owed(const Dev& d) const {   // (and exchange_quiet's owed uploads)
    for(const auto& x:d.adaptive)if(x.incoming>=0 && (x.d2h_done<x.d2h_total || x.h2d_sent<x.h2d_total))return true;
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
    const bool issued_guard=max_pieces!=0 && exchange_downloads_owed(d);
    if(issued_guard)exchange_quiet_begin(d,8); // includes outstanding uploads/downloads on d.xs
    for(auto& x:d.adaptive) {
        // exchange_quiet: the owed upload's pieces first, on the same stream (so the download's completion follows them)
        while(x.incoming>=0 && x.submitted && x.h2d_sent<x.h2d_total && max_pieces!=0) {
            const size_t n=std::min(kDownloadPiece,x.h2d_total-x.h2d_sent);
            auto status=cudaMemcpyAsync(x.landing_gpu+x.h2d_sent,x.source_host+x.h2d_sent,n,cudaMemcpyHostToDevice,d.xs);
            if(status!=cudaSuccess)return adaptive_fail("exchange upload piece",status,err);
            x.h2d_sent+=n;++counter;++exchange_upload_pieces;if(max_pieces>0)--max_pieces;
            if(x.h2d_sent==x.h2d_total) {
                status=cudaEventRecord(x.h2d_event,d.xs);
                if(status!=cudaSuccess)return adaptive_fail("exchange upload completion",status,err);
                x.h2d_issued=true;
                if(!x.d2h_total) {   // (no download owed: the transaction ends with its upload)
                    status=cudaEventRecord(x.event,d.xs);
                    if(status!=cudaSuccess)return adaptive_fail("exchange completion",status,err);
                }
            }
        }
        if(x.h2d_sent<x.h2d_total)continue;   // (the download waits for its upload's pieces)
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
    return !issued_guard || exchange_quiet_uploads(d,err);
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
// The 3090's x4 window, opened by the host at a point where its next pass is at least until_ns away (exchange_quiet:
// until the next submission closes it).
void exchange_x4_window(uint64_t span_ns) {
    if(free_active()) {
        helper->live_x4_window(UINT64_MAX);third->live_x4_window(UINT64_MAX);return;
    }
    if(exchange_schedule && helper && helper_live_a>=0)
        helper->live_x4_window(exchange_quiet ? UINT64_MAX : NativeHelper::clock_ns()+span_ns);
}
void exchange_quiet_cpu(bool busy) {
    if(free_active()) {
        helper->live_cpu_busy(busy);third->live_cpu_busy(busy);
        if(host_resorter)host_resorter->cpu_busy.store(busy,std::memory_order_release);
        if(busy)background_traffic.foreground();return;
    }
    if(exchange_schedule && exchange_quiet && helper && helper_live_a>=0)helper->live_cpu_busy(busy);
}
// The forward end: no pass runs until the next forward's first MoE join. Every owed download goes out, the x4 window
// opens, and the host window opens at once: the tail and the dense prefix are the token's idle host-memory time, which
// the 5090's forward-end refills share (nothing waits on them before the first MoE join). A lease whose host copy has
// waited exchange_host_wait forward ends gets an open host window until that copy is done.
bool exchange_forward_end(std::string& err) {
    if(free_active()) {
        for(auto& d:D)if(!exchange_quiet_uploads(d,err))return false;
        exchange_x4_window(0);
    }
    if(!exchange_schedule)return true;
    for(auto& d:D)if(exchange_downloads_owed(d) && !exchange_issue_downloads(d,-1,exchange_download_flush_pieces,err))return false;
    for(auto& d:D)if(!exchange_quiet_uploads(d,err))return false;
    if(helper) {
        using P=NativeHelper::LivePhase;
        const auto phase=helper_live_a>=0 ? helper->live_poll().phase : P::idle;
        if(phase==P::preparing || phase==P::recycling) ++exchange_host_waits; else exchange_host_waits=0;
        exchange_host_urgent=exchange_host_waits>exchange_host_wait;
        exchange_host_urgent_forwards+=exchange_host_urgent;
        exchange_host_open_ns=exchange_host_urgent ? 0 : NativeHelper::clock_ns();
        helper->live_x4_window(UINT64_MAX);
        helper->live_host_window(exchange_host_open_ns,UINT64_MAX);
    }
    return true;
}
// The forward start: both 3090 windows close early in the dense prefix, before its first MoE join (an urgent host copy
// keeps its window).
void exchange_forward_begin() {
    if(free_active()){exchange_x4_window(0);return;}
    if(!exchange_schedule || !helper)return;
    const uint64_t until=NativeHelper::clock_ns()+kPrefixWindowNs;
    helper->live_x4_window(until);
    if(!exchange_host_urgent)helper->live_host_window(exchange_host_open_ns,until);
}
// Outside generation, in maintenance and when scheduling stops: no pass runs, every window is open.
void exchange_windows_open(bool joined=false) {
    if(helper) {helper->live_x4_window(UINT64_MAX);helper->live_host_window(0,UINT64_MAX);}
    if(free_active()) {
        third->live_x4_window(UINT64_MAX);
        if(joined) {
            helper->live_x4_guard(nullptr);third->live_x4_guard(nullptr);
            helper->live_cpu_busy(false);third->live_cpu_busy(false);
            if(host_resorter){host_resorter->guard.disable();host_resorter->cpu_busy.store(false);}
            for(auto& d:D){d.quiet_pending=false;d.quiet_dirty=0;}
        }
    }
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
    if(mode && exchange_quiet && (compressed_host() || o.routing_prefetch)) {
        err="exchange_quiet requires raw HOST records and routing_prefetch off";return false;
    }
    if(mode && exchange_quiet && helper)for(auto& d:D) {
        d.quiet_generation=helper->live_x4_revoke();d.quiet_pending=true;d.quiet_dirty=15;
    }
    if(mode && !exchange_schedule_prepare(err)) {
        if(exchange_quiet && helper) {
            placement_failed=true;retirement_uncertain.store(true);helper->live_x4_guard_failed(err);
        }
        return false;
    }
    if(!mode)for(auto& d:D)if(d.xs && !exchange_downloads_join(d,err))return false;   // (nothing stays owed)
    if(!mode && helper && !free_active()) {
        helper->live_x4_guard(nullptr);helper->live_cpu_busy(false);
        for(auto& d:D) {d.quiet_pending=false;d.quiet_dirty=0;}
    }
    exchange_schedule=mode;
    if(helper)helper->set_live_schedule(mode);
    if(mode)for(auto& d:D)if(!exchange_quiet_uploads(d,err))return false;
    exchange_windows_open();
    return true;
}
