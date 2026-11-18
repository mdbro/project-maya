// Included inside Model::Impl. Canonical shared HOST storage and optional work
// admitted from live CPU/link/upload state. No fixed per-step swap cadence.
int shared_mover_mode=0;
uint64_t shared_decisions=0;
BackgroundTraffic background_traffic;
std::unique_ptr<HostResorter> host_resorter;
uint64_t third_quiet_generation=0,resort_quiet_generation=0;
struct FreeMove {
    long long incoming=-1,victim=-1;
    const uint8_t* source=nullptr;
    SharedFrames::Lease source_lease{},landing_lease{};
    uint64_t started_rows=0,started_ns=0;
    NativeHelper::LiveScheduleStats initial_stats{};
    uint64_t initial_cpu_overlaps=0;
};
std::array<FreeMove,2> free_moves{};
struct FreeSort {
    long long incoming=-1,victim=-1;
    const uint8_t *source=nullptr,*victim_source=nullptr;
    uint8_t* private_landing=nullptr;
    SharedFrames::Lease source_lease{},landing_lease{};
} free_sort;
using FreePrice=FreeTimePrice;
FreeTimeState free_learning;
uint64_t free_started[2]{},free_completed[2]{},free_churn_blocked=0,free_no_region=0;
uint64_t free_resorts_started=0,free_resorts_completed=0;
uint64_t free_policy_ns=0;

bool free_pending()const {
    return free_moves[0].incoming>=0 || free_moves[1].incoming>=0 || free_sort.incoming>=0;
}
bool free_active()const {return native_shared_pool && free_pending();}
bool free_start_guards(Dev& d,std::string& err) {
    // First publish current stream tails before a newly queued worker can copy.
    // With no outstanding job, the mode alone records no per-layer markers.
    exchange_quiet_begin(d);
    if(!exchange_quiet_uploads(d,err))return false;
    exchange_x4_window(0);return true;
}
void free_disable_guards() {
    if(third){third->live_x4_guard(nullptr);third->live_cpu_busy(false);}
    if(host_resorter){host_resorter->guard.disable();host_resorter->cpu_busy.store(false);}
    if(!(exchange_schedule && exchange_quiet) && helper) {
        helper->live_x4_guard(nullptr);helper->live_cpu_busy(false);
        for(auto& d:D){d.quiet_pending=false;d.quiet_dirty=0;}
    }
}
bool free_key_leased(long long key)const {
    if(key<0)return false;
    for(const auto& x:free_moves)if(x.incoming==key || x.victim==key)return true;
    return free_sort.incoming==key || free_sort.victim==key;
}
bool shared_pool_contains(uintptr_t a,size_t bytes)const {
    if(!native_shared_pool)return false;
    const auto begin=uintptr_t(native_shared_pool->data());const size_t size=native_shared_pool->size();
    return a>=begin && bytes<=size && a-begin<=size-bytes;
}
bool shared_pool_check(std::string& err) {
    // The separate 4 GiB gate checked every word. This bounded check verifies
    // the actual combined model allocation's direct DMA on all three contexts.
    const size_t bytes=native_shared_pool->size();const auto* base=reinterpret_cast<const uint8_t*>(native_shared_pool->data());
    for(int device:{D[size_t(head_dev)].id,o.native_helper,o.native_third}) {
        CK(cudaSetDevice(device));uint8_t* scratch=nullptr;CK(cudaMalloc(&scratch,128));
        for(size_t offset:{size_t(0),(bytes/2/128)*128,((bytes-128)/128)*128}) {
            uint8_t actual[128];
            auto status=cudaMemcpy(scratch,base+offset,sizeof actual,cudaMemcpyHostToDevice);
            if(status==cudaSuccess)status=cudaMemcpy(actual,scratch,sizeof actual,cudaMemcpyDeviceToHost);
            if(status!=cudaSuccess) {err="shared HOST model DMA failed";retirement_uncertain.store(true);return false;}
            if(std::memcmp(actual,base+offset,sizeof actual)) {cudaFree(scratch);err="shared HOST model DMA differs";return false;}
        }
        CK(cudaFree(scratch));
    }
    CK(cudaSetDevice(D[size_t(head_dev)].id));return true;
}
bool shared_reconcile(std::string& err) {
    if(!native_shared_pool)return true;
    std::vector<long long> owners(shared_frames.size(),-1);
    for(size_t li=0;li<L.size();++li)if(L[li].moe)
        for(int e=0;e<g.n_exp;++e)if(!L[li].host_blob.empty()) {
            const int frame=shared_frames.index(L[li].host_blob[size_t(e)]);
            if(frame<0)continue;
            if(owners[size_t(frame)]!=-1){err="shared HOST frame has two canonical owners";return false;}
            owners[size_t(frame)]=static_cast<long long>(li)*1024+e;
        }
    const auto external=[&](const uint8_t* p) {
        const int frame=shared_frames.index(p);if(frame<0)return true;
        if(owners[size_t(frame)]>=0)return false;owners[size_t(frame)]=-2;return true;
    };
    for(const auto* p:adaptive_host_spare)if(p && !external(p)){err="shared frame aliases a primary spare";return false;}
    for(const auto& d:D)for(const auto& x:d.adaptive)if(x.incoming>=0 && x.target_host && !external(x.target_host)) {
        err="shared frame aliases a primary landing";return false;
    }
    if(resort_private_free && !external(resort_private_free)){err="shared frame aliases the HOST free landing";return false;}
    for(size_t i=0;i<owners.size();++i) {
        const auto& frame=shared_frames.at(int(i));
        if(frame.leased) {
            if(frame.key!=owners[i]) {err="canonical ownership changed under a shared lease";return false;}
        } else if(!shared_frames.assign(int(i),owners[i],err))return false;
    }
    return true;
}
void free_learning_ready() {
    if(free_learning.layer.size()!=L.size())free_learning.layer.resize(L.size());
    if(free_learning.last_move.size()!=L.size()*1024)free_learning.last_move.resize(L.size()*1024,0);
}
void free_clock_advance() {
    const uint64_t rows=uint64_t(native_demand_rows);
    const uint64_t delta=rows>=free_learning.rate_rows?rows-free_learning.rate_rows:0;
    free_learning.rate_rows=rows;free_learning.clock_rows+=delta;
    for(double& rate:free_learning.rate)rate*=std::exp(-double(delta)/100.0);
}
void free_observe(const pass_model::Load& after,int layer,int columns,bool copied) {
    if(!shared_mover_mode || !generation_active || columns<1)return;
    free_learning_ready();const int mode=draft_running?1:0;
    const double before=pass_model::predict(after,pass_rates).passed;
    for(int device=0;device<3;++device) {
        auto x=after;
        if(copied) {
            if(!x.copies || !x.late_experts)return;
            --x.copies;--x.late_experts;x.late_picks-=columns;
            if(device) {--x.gpu_experts;x.gpu_picks-=columns;}
        } else {
            if(!x.cpu_experts)return;
            --x.cpu_experts;x.cpu_columns-=columns;
            if(!device){++x.gpu_experts;x.gpu_picks+=columns;}
        }
        if(device==1){++x.helper_experts;x.helper_columns+=columns;}
        if(device==2){++x.third_experts;x.third_columns+=columns;}
        const double saving=(before-pass_model::predict(x,pass_rates).passed)/columns;
        if(!std::isfinite(saving))continue;
        for(FreePrice* p:{&free_learning.layer[size_t(layer)],&free_learning.global}) {
            auto& n=p->samples[mode][device];auto& mean=p->value[mode][device];
            const double alpha=1.0/double(std::min<uint64_t>(++n,64));mean+=alpha*(saving-mean);
        }
    }
}
double free_value(long long key,int device)const {
    if(key<0 || size_t(key/1024)>=free_learning.layer.size())return 0;
    const int mode=draft_running?1:0;const auto& p=free_learning.layer[size_t(key/1024)];
    if(p.samples[mode][device]>=4)return std::max(0.0,p.value[mode][device]);
    const auto& global=free_learning.global;
    return global.samples[mode][device]>=32?std::max(0.0,global.value[mode][device]):0;
}
double free_horizon()const {return 5.0/-std::log(demand_decay);}
double free_worth(long long key,int device) {return score(key,false)*free_horizon()/100.0*free_value(key,device);}
bool free_stayed(long long key,unsigned actor) {
    free_learning_ready();const auto last=free_learning.last_move[size_t(key)];
    const uint64_t stay=16+uint64_t(256.0*free_learning.rate[actor]);
    if(last && (last>free_learning.clock_rows || free_learning.clock_rows-last<stay)) {++free_churn_blocked;return false;}return true;
}
double free_copy_price(unsigned actor)const {
    return free_learning.price_ms[actor];
}
ResidencyChoice free_choose_primary(const std::vector<ResidencyCandidate>& incoming,
                                    const std::vector<ResidencyCandidate>& victims) {
    ResidencyChoice out;const ResidencyCandidate* cold=nullptr;double price=1e30;
    for(const auto& c:victims)if(free_stayed(c.key,0)) {
        const double v=free_worth(c.key,0);if(v<price){price=v;cold=&c;}
    }
    if(!cold)return out;
    const double margin=1.15+std::min(0.85,4.0*free_learning.rate[0]);
    for(const auto& c:incoming)if(free_stayed(c.key,0)) {
        const double value=free_worth(c.key,0)-margin*price-adaptive_d2h_ms()-
            (c.ready_slot<0?adaptive_h2d_ms():0)-adaptive_policy.interference_ms-adaptive_policy.margin_ms;
        if(value>out.value_ms)out={c.key,cold->key,c.ready_slot,value};
    }
    return out;
}
bool free_source_busy(const Dev& d,long long key,const uint8_t* source)const {
    return helper_live_leased(key) || helper_live_source_busy(d,key,source);
}
void free_bit(long long key,bool on,bool third_card) {
    auto& l=L[size_t(key/1024)];auto& bits=third_card?l.third_ready:l.helper_ready;
    const int e=int(key%1024);const auto mask=uint32_t(1)<<(e%32);
    if(on)bits[size_t(e/32)]|=mask;else bits[size_t(e/32)]&=~mask;
}
bool free_progress(Dev& d,std::string& err) {
    if(!native_shared_pool)return true;
    free_clock_advance();
    if(!background_traffic.healthy()){err="shared background traffic failed";placement_failed=true;return false;}
    bool changed=false;
    for(unsigned actor=0;actor<2;++actor) {
        auto& x=free_moves[actor];if(x.incoming<0)continue;
        NativeHelper& card=*(actor?third:helper);const auto status=card.live_poll();
        if(status.phase==NativeHelper::LivePhase::failed) {err=status.error;placement_failed=true;return false;}
        if(status.phase!=NativeHelper::LivePhase::prepared)continue;
        if(helper_queue_pending || handoff_inflight)return true;
        if(helper_live_source_busy(d,x.incoming,x.source))continue;
        auto& a=L[size_t(x.incoming/1024)];const int ae=int(x.incoming%1024);
        if(a.host_blob[size_t(ae)]!=x.source || !shared_frames.current(x.source_lease) ||
           (!actor && !shared_frames.current(x.landing_lease))) {err="direct HOST lease changed before publication";placement_failed=true;return false;}
        if(!card.direct_publish(err)){placement_failed=true;return false;}
        if(!actor) {
            auto& b=L[size_t(x.victim/1024)];
            b.host_blob[size_t(x.victim%1024)]=shared_frames.address(x.landing_lease.frame);
            a.host_blob[size_t(ae)]=nullptr;
            if(!shared_frames.finish(x.landing_lease,x.victim,err) || !shared_frames.finish(x.source_lease,-1,err))return false;
            free_bit(x.victim,false,false);free_bit(x.incoming,true,false);
            for(int s=0;s<d.n_slots;++s)if(d.slot_owner[size_t(s)]==x.incoming) {prefetch_forget(d,s);d.slot_owner[size_t(s)]=-1;}
        } else {
            if(!shared_frames.finish(x.source_lease,x.incoming,err))return false;
            free_bit(x.incoming,true,true);
            const int victim=int(x.victim/1024)*g.n_exp+int(x.victim%1024);
            const int incoming=int(x.incoming/1024)*g.n_exp+int(x.incoming%1024);
            const auto at=std::find(third_ids.begin(),third_ids.end(),victim);
            if(at==third_ids.end()){err="third copy-cache roster lost its direct victim";placement_failed=true;return false;}
            *at=incoming;
        }
        bool safe=false;if(!card.live_retire(safe,err) || !safe){placement_failed=true;return false;}
        free_learning_ready();free_learning.last_move[size_t(x.incoming)]=free_learning.clock_rows;
        free_learning.last_move[size_t(x.victim)]=free_learning.clock_rows;
        const double delay=double(native_demand_rows-x.started_rows);
        const auto stats=card.live_schedule_stats();const auto& initial=x.initial_stats;
        const uint64_t chunks=stats.direct_chunks-initial.direct_chunks;
        if(chunks) {
            const double span=double(stats.direct_span_ns-initial.direct_span_ns)*1e-6/double(chunks);
            const double overlaps=double(stats.guard_already_inflight-initial.guard_already_inflight+
                stats.job_overlaps-initial.job_overlaps+background_traffic.overlaps(actor)-x.initial_cpu_overlaps);
            // Conservative upper price from boundary observations, not a claim
            // that all overlap delayed a pass. Keep it in the restorable policy.
            const double price=span*(1.0+std::min(double(chunks),overlaps));
            const double alpha=free_learning.price_samples[actor]++?0.25:1.0;
            free_learning.price_ms[actor]+=alpha*(price-free_learning.price_ms[actor]);
            free_learning.delay_rows[actor]+=alpha*(delay-free_learning.delay_rows[actor]);
        }
        if(capture::detailed()) {
            std::ostringstream out;out<<"\"actor\":"<<actor<<",\"incoming\":"<<x.incoming<<",\"victim\":"<<x.victim
                <<",\"elapsed_rows\":"<<delay<<",\"elapsed_ns\":"<<(NativeHelper::clock_ns()-x.started_ns)
                <<",\"h2d_bytes\":"<<(stats.direct_h2d_bytes-initial.direct_h2d_bytes)
                <<",\"d2h_bytes\":"<<(stats.direct_d2h_bytes-initial.direct_d2h_bytes)<<",\"chunks\":"<<chunks;
            capture::detail_event("shared_move_publish",out.str());
        }
        capture::light::shared_event(actor,capture::light::kSharedPublish,x.incoming,x.victim);
        ++free_completed[actor];x=FreeMove{};changed=true;
    }
    if(free_sort.incoming>=0 && host_resorter) {
        if(host_resorter->state()==HostResorter::failed){err=host_resorter->error();placement_failed=true;return false;}
        if(host_resorter->state()==HostResorter::complete) {
            auto& x=free_sort;
            if(helper_live_source_busy(d,x.incoming,x.source) || helper_live_source_busy(d,x.victim,x.victim_source))return true;
            auto& a=L[size_t(x.incoming/1024)];auto& b=L[size_t(x.victim/1024)];
            if(a.host_blob[size_t(x.incoming%1024)]!=x.source || b.host_blob[size_t(x.victim%1024)]!=x.victim_source ||
               !shared_frames.current(x.source_lease) || !shared_frames.current(x.landing_lease)) {
                err="HOST re-sort source lease changed before publication";placement_failed=true;return false;
            }
            a.host_blob[size_t(x.incoming%1024)]=shared_frames.address(x.landing_lease.frame);
            b.host_blob[size_t(x.victim%1024)]=x.private_landing;
            resort_private_free=const_cast<uint8_t*>(x.source);
            if(!shared_frames.finish(x.landing_lease,x.incoming,err) || !shared_frames.finish(x.source_lease,-1,err))return false;
            if(capture::detailed()) {
                std::ostringstream out;out<<"\"incoming\":"<<x.incoming<<",\"victim\":"<<x.victim<<",\"bytes\":"<<shared_blob;
                capture::detail_event("shared_region_publish",out.str());
            }
            capture::light::shared_event(2,capture::light::kSharedPublish,x.incoming,x.victim);
            x=FreeSort{};host_resorter->published();++free_resorts_completed;changed=true;
        }
    }
    if(changed && !free_pending())free_disable_guards();
    return !changed || shared_reconcile(err);
}
bool free_begin_move(Dev& d,unsigned actor,long long incoming,long long victim,std::string& err) {
    auto& a=L[size_t(incoming/1024)];const auto* source=a.host_blob[size_t(incoming%1024)];
    auto source_lease=shared_frames.take(shared_frames.index(source),incoming);if(!source_lease)return true;
    SharedFrames::Lease landing;
    if(!actor) {landing=shared_frames.take_free();if(!landing){shared_frames.finish(source_lease,incoming,err);++free_no_region;return true;}}
    const auto* ah=reinterpret_cast<const ne::Header*>(source);
    auto* target=actor?nullptr:shared_frames.address(landing.frame);
    auto* bh=reinterpret_cast<ne::Header*>(target);float old_weight[3]{},old_input[3]{};
    auto& card=*(actor?third:helper);
    const auto initial_stats=card.live_schedule_stats();const auto initial_cpu_overlaps=background_traffic.overlaps(actor);
    const bool first=!free_pending();
    free_moves[actor]={incoming,victim,source,source_lease,landing,uint64_t(native_demand_rows),NativeHelper::clock_ns()};
    if(first && !free_start_guards(d,err))return false;
    if(!card.direct_begin(int(victim/1024),int(victim%1024),int(incoming/1024),int(incoming%1024),
        source+sizeof(ne::Header),actor?nullptr:target+sizeof(ne::Header),shared_blob-sizeof(ne::Header),
        ah->weight_global,ah->input_global,old_weight,old_input,&background_traffic,actor,!actor,shared_mover_mode==2,err)) {
        shared_frames.finish(source_lease,incoming,err);if(landing)shared_frames.finish(landing,-1,err);
        free_moves[actor]=FreeMove{};
        placement_failed=true;return false;
    }
    if(!actor) {std::memset(bh,0,sizeof(*bh));std::copy_n(old_weight,3,bh->weight_global);std::copy_n(old_input,3,bh->input_global);}
    else free_bit(victim,false,true);
    free_moves[actor].initial_stats=initial_stats;free_moves[actor].initial_cpu_overlaps=initial_cpu_overlaps;
    ++free_started[actor];free_learning.rate[actor]+=0.01;
    capture::light::shared_event(actor,capture::light::kSharedBegin,incoming,victim);
    if(capture::detailed()) {
        std::ostringstream out;out<<"\"actor\":"<<actor<<",\"incoming\":"<<incoming<<",\"victim\":"<<victim
            <<",\"source_frame\":"<<source_lease.frame<<",\"source_generation\":"<<source_lease.generation
            <<",\"landing_frame\":"<<landing.frame<<",\"payload_bytes\":"<<(shared_blob-sizeof(ne::Header));
        capture::detail_event("shared_move_begin",out.str());
    }
    return true;
}
bool free_issue(Dev& d,std::string& err) {
    if(!shared_mover_mode || !generation_active || &d!=&D[size_t(head_dev)] || !helper_live_enabled || helper_admissions_held)return true;
    struct PolicyTime {uint64_t& total;uint64_t begin=NativeHelper::clock_ns();~PolicyTime(){total+=NativeHelper::clock_ns()-begin;}} timer{free_policy_ns};
    if(!shared_reconcile(err)){placement_failed=true;return false;}
    free_learning_ready();
    free_clock_advance();
    const bool audit=shared_mover_mode==2 && ((shared_decisions++ & 15u)==0);
    struct Candidate {long long key;double score;};
    std::vector<Candidate> ranking;
    for(size_t li=0;li<L.size();++li)if(L[li].moe)for(int e=0;e<g.n_exp;++e)if(L[li].res_slot[size_t(e)]<0) {
        const long long key=static_cast<long long>(li)*1024+e;ranking.push_back({key,score(key,false)});
    }
    std::stable_sort(ranking.begin(),ranking.end(),[](const auto& a,const auto& b){return a.score>b.score || (a.score==b.score && a.key<b.key);});
    std::vector<uint8_t> third_tier(L.size()*1024,0);
    const size_t hcap=helper_live_capacity?helper_live_capacity:helper->records().size();
    for(size_t i=hcap;i<std::min(ranking.size(),hcap+size_t(third_capacity));++i)third_tier[size_t(ranking[i].key)]=1;
    double primary_victim=1e30;
    for(size_t li=0;li<L.size();++li)for(int e:L[li].res_ids)primary_victim=std::min(primary_victim,free_worth(static_cast<long long>(li)*1024+e,0));
    for(unsigned actor=0;actor<2;++actor) {
        if(free_moves[actor].incoming>=0 || (!actor && helper_live_a>=0) || (actor && !third_on)){
            if(audit)std::fprintf(stderr,"MOVER_DECISION rows=%llu actor=%u pending=1\n",(unsigned long long)free_learning.clock_rows,actor);
            continue;
        }
        const auto records=(actor?third:helper)->records();long long victim=-1;double cold=1e30;
        for(const auto& id:records) {
            const long long key=static_cast<long long>(id.layer)*1024+id.expert;
            if(free_key_leased(key) || !free_stayed(key,actor))continue;
            const double value=free_worth(key,int(actor)+1);
            if(value<cold){cold=value;victim=key;}
        }
        if(victim<0){
            if(audit)std::fprintf(stderr,"MOVER_DECISION rows=%llu actor=%u no_victim=1\n",(unsigned long long)free_learning.clock_rows,actor);
            continue;
        }
        long long incoming=-1;double best=0;
        unsigned filter[9]{};double best_net=-1e30;long long near_key=-1;
        const double margin=1.15+std::min(0.85,4.0*free_learning.rate[actor]);
        const double discount=std::exp(-free_learning.delay_rows[actor]/free_horizon());
        for(const auto& candidate:ranking) {
            const auto key=candidate.key;auto& l=L[size_t(key/1024)];const int e=int(key%1024);const auto* source=l.host_blob[size_t(e)];
            if(!source){if(audit)++filter[0];continue;}
            if(helper_owns(l,e)||third_holds(l,e)){if(audit)++filter[1];continue;}
            if(free_source_busy(d,key,source)){if(audit)++filter[2];continue;}
            if(shared_frames.index(source)<0){if(audit)++filter[3];continue;}
            if(!free_stayed(key,actor)){if(audit)++filter[4];continue;}
            if(actor && !third_tier[size_t(key)]){if(audit)++filter[5];continue;}
            const double value=discount*(free_worth(key,int(actor)+1)-margin*cold)-free_copy_price(actor);
            const double primary=free_worth(key,0)-primary_victim-adaptive_h2d_ms()-adaptive_d2h_ms();
            if(audit){if(value>best_net){best_net=value;near_key=key;}++filter[6];if(value<=0)++filter[7];}
            if(!actor && primary>value){if(audit)++filter[8];continue;} // the current primary destination is worth more
            if(value>best){best=value;incoming=key;}
        }
        if(audit)std::fprintf(stderr,"MOVER_DECISION rows=%llu actor=%u ranking=%zu victim=%lld cold=%.6g margin=%.6g discount=%.6g price=%.6g no_source=%u owned=%u busy=%u outside=%u stay=%u tier=%u eligible=%u nonpositive=%u primary_wins=%u near=%lld net=%.6g selected=%lld\n",
            (unsigned long long)free_learning.clock_rows,actor,ranking.size(),victim,cold,margin,discount,free_copy_price(actor),
            filter[0],filter[1],filter[2],filter[3],filter[4],filter[5],filter[6],filter[7],filter[8],near_key,best_net,incoming);
        if(incoming>=0 && !free_begin_move(d,actor,incoming,victim,err))return false;
    }
    // Prepare future candidates independently of any outstanding GPU swap.
    if(free_sort.incoming<0 && host_resorter && host_resorter->state()==HostResorter::idle) {
        double card_floor[2]{1e30,1e30};
        for(unsigned actor=0;actor<2;++actor)for(const auto& id:(actor?third:helper)->records())
            card_floor[actor]=std::min(card_floor[actor],free_worth(static_cast<long long>(id.layer)*1024+id.expert,int(actor)+1));
        const auto delivery_value=[&](long long key) {
            const auto& l=L[size_t(key/1024)];const int e=int(key%1024);
            if(helper_owns(l,e) || third_holds(l,e))return 0.0;
            const unsigned actor=third_tier[size_t(key)]?1:0;
            const double margin=1.15+std::min(0.85,4.0*free_learning.rate[actor]);
            const double discount=std::exp(-free_learning.delay_rows[actor]/free_horizon());
            const double value=discount*(free_worth(key,int(actor)+1)-margin*card_floor[actor])-free_copy_price(actor);
            const double primary=free_worth(key,0)-primary_victim-adaptive_h2d_ms()-adaptive_d2h_ms();
            return !actor && primary>value?0.0:std::max(0.0,value);
        };
        long long hot=-1,cold=-1;double hot_value=0,cold_value=1e30;
        for(const auto& candidate:ranking) {
            const auto key=candidate.key;auto& l=L[size_t(key/1024)];const int e=int(key%1024);const auto* source=l.host_blob[size_t(e)];
            if(!source || free_source_busy(d,key,source))continue;
            const double value=delivery_value(key);
            if(shared_frames.index(source)>=0) {if(value<cold_value){cold_value=value;cold=key;}}
            else if(!helper_owns(l,e) && !third_holds(l,e) && value>hot_value){hot_value=value;hot=key;}
        }
        const double sort_margin=1.25+std::min(1.0,4.0*free_learning.rate[2]);
        if(hot>=0 && cold>=0 && hot_value>sort_margin*cold_value+0.1) {
            auto* source=const_cast<uint8_t*>(L[size_t(hot/1024)].host_blob[size_t(hot%1024)]);
            const auto* victim=L[size_t(cold/1024)].host_blob[size_t(cold%1024)];
            auto old=shared_frames.take(shared_frames.index(victim),cold),landing=shared_frames.take_free();
            if(old && landing) {
                const bool first=!free_pending();
                free_sort={hot,cold,source,victim,resort_private_free,old,landing};
                if(first && !free_start_guards(d,err))return false;
                if(!host_resorter->begin(shared_frames.address(landing.frame),source,resort_private_free,victim,shared_blob,shared_mover_mode==2)) {
                    err="HOST re-sort worker did not accept its reserved job";placement_failed=true;return false;
                }
                ++free_resorts_started;free_learning.rate[2]+=0.01;
                capture::light::shared_event(2,capture::light::kSharedBegin,hot,cold);
            } else {
                if(old)shared_frames.finish(old,cold,err);if(landing)shared_frames.finish(landing,-1,err);++free_no_region;
            }
        }
    }
    return true;
}
bool free_cancel_retired() {
    if(host_resorter && !host_resorter->stop_and_retire())return false;
    bool safe=false;std::string error;
    for(unsigned a=0;a<2;++a)if(free_moves[a].incoming>=0) {
        auto& card=*(a?third:helper);const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(1);
        do {if(card.live_cancel(safe,error) && safe)break;std::this_thread::sleep_for(std::chrono::microseconds(100));}
        while(std::chrono::steady_clock::now()<until);
        if(!safe)return false;
    }
    return true;
}

std::vector<int> shared_membership()const {
    std::vector<int> ids;
    if(native_shared_pool)for(size_t li=0;li<L.size();++li)if(L[li].moe)
        for(int e=0;e<g.n_exp;++e)if(!L[li].host_blob.empty() && shared_frames.index(L[li].host_blob[size_t(e)])>=0)
            ids.push_back(int(li)*g.n_exp+e);
    return ids;
}
bool restore_shared_membership(const std::vector<int>& desired,std::string& err) {
    // Explicit joined benchmark maintenance only. Ordinary turns never call
    // this, and background preparation never waits for this CPU permutation.
    if(!native_shared_pool)return desired.empty();
    if(free_pending() || shared_frames.busy()){err="shared restore still has a leased frame";return false;}
    auto wanted=desired;std::sort(wanted.begin(),wanted.end());
    if(std::adjacent_find(wanted.begin(),wanted.end())!=wanted.end() || wanted.size()>shared_frames.size()-2) {
        err="invalid shared HOST reference membership";return false;
    }
    for(int id:wanted)if(id<0 || id>=g.n_layers*g.n_exp || !L[size_t(id/g.n_exp)].moe ||
                         !L[size_t(id/g.n_exp)].host_blob[size_t(id%g.n_exp)]) {
        err="shared HOST reference names an expert not owned by HOST";return false;
    }
    const auto current=shared_membership();std::vector<int> incoming,outgoing;
    std::set_difference(wanted.begin(),wanted.end(),current.begin(),current.end(),std::back_inserter(incoming));
    std::set_difference(current.begin(),current.end(),wanted.begin(),wanted.end(),std::back_inserter(outgoing));
    const auto host=[&](int id)->const uint8_t*& {return L[size_t(id/g.n_exp)].host_blob[size_t(id%g.n_exp)];};
    std::vector<uint8_t**> external;
    for(auto& p:adaptive_host_spare)if(p)external.push_back(&p);
    external.push_back(&resort_private_free);
    const size_t paired=std::min(incoming.size(),outgoing.size());
    size_t shared_external=0,private_external=0;
    for(auto** p:external)if(*p) {(shared_frames.index(*p)>=0?shared_external:private_external)++;}
    if(incoming.size()-paired>shared_external || outgoing.size()-paired>private_external || !resort_private_free) {
        err="shared HOST reference does not fit canonical/free ownership";return false;
    }
    for(size_t i=0;i<paired;++i) {
        auto* a=const_cast<uint8_t*>(host(incoming[i]));auto* b=const_cast<uint8_t*>(host(outgoing[i]));
        std::memcpy(resort_private_free,a,shared_blob);std::memcpy(a,b,shared_blob);std::memcpy(b,resort_private_free,shared_blob);
        host(incoming[i])=b;host(outgoing[i])=a;
    }
    for(size_t i=paired;i<incoming.size();++i) {
        auto where=std::find_if(external.begin(),external.end(),[&](auto** p){return shared_frames.index(*p)>=0;});
        auto* source=const_cast<uint8_t*>(host(incoming[i]));auto* destination=**where;
        std::memcpy(destination,source,shared_blob);host(incoming[i])=destination;**where=source;
    }
    for(size_t i=paired;i<outgoing.size();++i) {
        auto where=std::find_if(external.begin(),external.end(),[&](auto** p){return *p && shared_frames.index(*p)<0;});
        auto* source=const_cast<uint8_t*>(host(outgoing[i]));auto* destination=**where;
        std::memcpy(destination,source,shared_blob);host(outgoing[i])=destination;**where=source;
    }
    if(shared_membership()!=wanted || !shared_reconcile(err)) {placement_failed=true;err="shared HOST reference restore differs";return false;}
    return true;
}
std::string free_state_json(const FreeTimeState& state)const {
    std::ostringstream out;out.precision(17);
    const auto price=[&](const FreeTimePrice& p) {
        out<<"{\"value\":[";
        for(int m=0;m<2;++m){if(m)out<<',';out<<'['<<p.value[m][0]<<','<<p.value[m][1]<<','<<p.value[m][2]<<']';}
        out<<"],\"samples\":[";
        for(int m=0;m<2;++m){if(m)out<<',';out<<'['<<p.samples[m][0]<<','<<p.samples[m][1]<<','<<p.samples[m][2]<<']';}
        out<<"]}";
    };
    out<<"{\"clock_rows\":"<<state.clock_rows<<",\"rate\":["<<state.rate[0]<<','<<state.rate[1]<<','<<state.rate[2]
       <<"],\"delay_rows\":["<<state.delay_rows[0]<<','<<state.delay_rows[1]<<"],\"price_ms\":["<<state.price_ms[0]<<','<<state.price_ms[1]
       <<"],\"price_samples\":["<<state.price_samples[0]<<','<<state.price_samples[1]<<"],\"layers\":[";
    for(size_t i=0;i<state.layer.size();++i){if(i)out<<',';price(state.layer[i]);}
    out<<"],\"global\":";price(state.global);out<<",\"last_move\":[";
    bool comma=false;
    for(size_t i=0;i<state.last_move.size();++i)if(state.last_move[i]) {
        if(comma)out<<',';comma=true;out<<'['<<i/1024<<','<<i%1024<<','<<state.last_move[i]<<']';
    }
    out<<"]}";return out.str();
}
