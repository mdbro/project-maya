// The third card's fixed pool (--native-third DEVICE, --native-third-gb GB; off unless both are given).
//
// Copies of HOST experts on one more CUDA device, held in a second NativeHelper. The HOST keeps owning every such
// expert: its record stays in the bank and no ownership rule changes, so a request without third=1 runs the engine
// exactly as it was. With third=1:
//   - a window's pick of a held expert that would otherwise go to the CPU lane or to a staging frame is served by the
//     third card, as one more submission beside the helper's, and its result is added on the host like the helper's;
//   - routing counts held experts as ready (the helper's bits of the ready mask);
//   - the exchange, staging and early-copy policies leave held experts where they are.
// The pool is set by a control command between requests and never changes during generation: no exchange touches it
// (its link carries about one record in 17 ms). PLACEMENT third <profile> gives it the profile's hottest experts that
// the HOST owns at that moment. PLACEMENT three <profile> <order> places all three cards from the profile: `order` is
// a permutation of h (the helper), p (the primary), t (this pool) and says which card takes the hottest experts, which
// the next and which the rest; hpt is helper-first placement followed by PLACEMENT third. A second command rewrites
// only the records that differ. A held expert that the primary or the helper also holds is simply served there; the
// copy on the third card stays valid, because records never change.
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

std::vector<int> third_ids;   // what the pool holds now: layer * experts + expert, in slot order of arrival

// Makes the pool hold exactly `target` (ids the HOST owns now, as many as the pool's capacity): the first call copies
// every record and seals the pool; a later call replaces only the records that differ. Every record written is read
// back and compared with the HOST's bytes and scales, and eight held experts are computed on the CPU lane and on the
// third card and compared. On success `fields` gets the reply's third_* members (no braces).
bool third_set(const std::vector<int>& target, std::string& fields, std::string& err) {
    if(!third) {err="this load has no third pool (--native-third, --native-third-gb)";return false;}
    if(third_failed) {err="an earlier fill of the third pool failed; it is unused for this load";return false;}
    if(placement_failed || retirement_uncertain.load()) {err="third: the engine's placement state is not healthy";return false;}
    if(int(target.size())!=third_capacity) {err="third: the pool takes exactly "+std::to_string(third_capacity)+" experts";return false;}
    const size_t stride=nf::record_stride(g.dim,g.ff), blob=sizeof(ne::Header)+stride;
    std::vector<uint8_t> wanted(size_t(g.n_layers)*size_t(g.n_exp),0);
    for(int id:target) {
        if(id<0 || id>=g.n_layers*g.n_exp || wanted[size_t(id)]) {err="third: an invalid or repeated expert";return false;}
        const Layer& l=L[size_t(id/g.n_exp)];const int e=id%g.n_exp;
        if(!l.moe || l.hf || l.dev!=head_dev || !pooled(l,e) || l.blob!=blob) {err="third: an expert the HOST does not own in the shared record layout";return false;}
        wanted[size_t(id)]=1;
    }
    const double began=now();
    uint8_t* stage=nullptr;
    if(cudaSetDevice(o.native_third)!=cudaSuccess || cudaHostAlloc(&stage,blob,cudaHostAllocPortable)!=cudaSuccess) {
        cudaSetDevice(D[size_t(head_dev)].id);err="third: no portable staging record";return false;
    }
    bool good=true;
    std::vector<int> written;                    // the records copied by this call
    const auto bit=[&](int id,bool on) {
        auto& word=L[size_t(id/g.n_exp)].third_ready[size_t((id%g.n_exp)/32)];
        const uint32_t mask=1u<<((id%g.n_exp)%32);
        if(on)word|=mask; else word&=~mask;
    };
    const auto load=[&](int id) {                // the HOST record into the staging buffer
        const Layer& l=L[size_t(id/g.n_exp)];
        std::memcpy(stage,l.host_blob[size_t(id%g.n_exp)],blob);
        return reinterpret_cast<const ne::Header*>(stage);
    };
    if(!third_filled) {
        for(int id:target) {
            const auto* h=load(id);
            good=third->add_expert(id/g.n_exp,id%g.n_exp,stage+sizeof(ne::Header),stride,h->weight_global,h->input_global,err);
            if(!good)break;
            bit(id,true);written.push_back(id);
        }
        good=good && third->seal(err);
        if(good)third_ids=target;
    } else {
        std::vector<uint8_t> held(wanted.size(),0);
        for(int id:third_ids)held[size_t(id)]=1;
        std::vector<int> incoming;
        for(int id:target)if(!held[size_t(id)])incoming.push_back(id);
        size_t next=0;
        for(int& old:third_ids) {
            if(wanted[size_t(old)])continue;
            const int id=incoming[next++];
            const auto* h=load(id);bool unsafe=false;
            good=third->replace_expert(old/g.n_exp,old%g.n_exp,id/g.n_exp,id%g.n_exp,stage+sizeof(ne::Header),stride,
                                       h->weight_global,h->input_global,unsafe,err);
            if(!good) {if(unsafe)stage=nullptr;break;}
            bit(old,false);bit(id,true);old=id;written.push_back(id);
        }
    }
    // Read back what was written and compare it with the HOST's: bytes and scales.
    long verified=0,mismatched=0;
    for(size_t i=0;i<written.size() && good;++i) {
        const int id=written[i];const Layer& l=L[size_t(id/g.n_exp)];const int e=id%g.n_exp;
        const auto* h=reinterpret_cast<const ne::Header*>(l.host_blob[size_t(e)]);
        float weight[3]={},input[3]={};bool unsafe=false;
        good=third->export_expert(id/g.n_exp,e,stage,stride,weight,input,unsafe,err);
        if(!good) {if(unsafe)stage=nullptr;break;}   // (an uncertain reader: the staging record is left allocated)
        const bool same=std::memcmp(stage,l.host_blob[size_t(e)]+sizeof(ne::Header),stride)==0 &&
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
            const int id=third_ids[size_t(k)*size_t(third_capacity)/8];Layer& l=L[size_t(id/g.n_exp)];const int e=id%g.n_exp;
            std::vector<CpuExpert> jobs{cpu_expert(l,e,1.0f)};
            std::vector<std::vector<std::pair<int,float>>> users(1);std::vector<std::pair<int,int>> order;
            for(int t=0;t<rows;++t) {users[0].push_back({t,1.0f});order.push_back({0,t});}
            std::fill(out_cpu.begin(),out_cpu.end(),0.0f);
            lane_run_multi(jobs,users,input.data(),rows,out_cpu.data(),&order);
            if(native_cpu_failed) {err="third: the CPU lane failed in the cross-check";good=false;break;}
            NativeHelper::Job job;job.layer=id/g.n_exp;job.expert=e;job.columns=rows;
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
        third_ids.clear();third_failed=true;third_on=false;
        if(err.empty())err="third: the pool could not be set";
        return false;
    }
    third_filled=true;
    const auto bytes=third->bytes();
    std::ostringstream out;out.precision(6);
    out<<"\"third_device\":"<<o.native_third<<",\"third_capacity\":"<<third_capacity<<",\"third_pool_bytes\":"<<bytes.pool
       <<",\"third_scratch_bytes\":"<<bytes.device_scratch<<",\"third_pinned_bytes\":"<<bytes.pinned
       <<",\"third_fill_ms\":"<<1000*(now()-began)<<",\"third_written\":"<<written.size()
       <<",\"third_check_experts\":"<<checked<<",\"third_check_max_diff\":"<<check_diff<<",\"third_check_peak\":"<<check_peak
       <<",\"third_verified\":"<<verified<<",\"third_mismatched\":"<<mismatched<<",\"third_ids\":[";
    for(size_t i=0;i<third_ids.size();++i)out<<(i?",":"")<<'['<<third_ids[i]/g.n_exp<<','<<third_ids[i]%g.n_exp<<']';
    out<<']';
    fields=out.str();
    return true;
}

// (the server checks this digest against the bytes it sent: FNV-1a over the profile's bytes, as placement reports it)
std::string third_digest(const std::vector<uint32_t>& counts) const {
    uint64_t digest=14695981039346656037ull;
    for(uint32_t n:counts)for(int b=0;b<4;++b){digest^=(n>>(8*b))&255u;digest*=1099511628211ull;}
    char hex[17];std::snprintf(hex,sizeof hex,"%016llx",(unsigned long long)digest);
    return hex;
}

// The MoE experts of the head device's layers, hottest first by the profile (ties in the order helper-first uses).
std::vector<int> third_ranked(const std::vector<uint32_t>& counts) const {
    std::vector<int> ids;
    for(int e=0;e<g.n_exp;++e)for(int li=0;li<g.n_layers;++li)
        if(L[size_t(li)].moe && !L[size_t(li)].hf && L[size_t(li)].dev==head_dev)ids.push_back(li*g.n_exp+e);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return counts[size_t(a)]>counts[size_t(b)];});
    return ids;
}

// PLACEMENT third: the profile's hottest experts that the HOST owns now.
bool third_fill(const std::vector<uint32_t>& counts, std::string& report, std::string& err) {
    if(!third) {err="this load has no third pool (--native-third, --native-third-gb)";return false;}
    if(counts.size()!=size_t(g.n_layers)*size_t(g.n_exp)) {err="third: the profile's size does not match the model";return false;}
    std::vector<int> target;
    for(int id:third_ranked(counts)) {
        const Layer& l=L[size_t(id/g.n_exp)];const int e=id%g.n_exp;
        if(int(target.size())<third_capacity && l.res_slot[size_t(e)]<0 && !helper_owns(l,e) && pooled(l,e))target.push_back(id);
    }
    if(int(target.size())<third_capacity) {err="third: fewer HOST experts than the pool holds";return false;}
    std::string fields;
    if(!third_set(target,fields,err))return false;
    report="{\"profile_fnv64\":\""+third_digest(counts)+"\","+fields+",\"ownership_complete\":true}";
    return true;
}

// PLACEMENT three: all three cards from the profile. `order` is a sequence of segments read from the hottest expert
// down: a card's letter (h the helper, p the primary, t this pool) and optionally how many experts that segment
// takes; without a number it takes what is left of that card's capacity. "hpt": the helper takes the hottest, the
// primary the next, the pool the rest. "p200htp": the primary takes the hottest 200, then the helper and the pool
// fill, and the primary's other slots take what follows. Every card must end exactly full. The primary and the
// helper are placed as helper-first placement places them (same routine, same reset of what was learned).
bool placement_three(const std::vector<uint32_t>& counts, const std::string& order, std::string& report, std::string& err) {
    if(!third || !helper) {err="three-card placement needs the helper and the third pool";return false;}
    if(counts.size()!=size_t(g.n_layers)*size_t(g.n_exp) || std::none_of(counts.begin(),counts.end(),[](uint32_t n){return n!=0;})) {
        err="three-card placement requires a nonzero complete demand profile";return false;
    }
    if(order.empty() || order.size()>64 || order.find_first_not_of("hpt0123456789")!=std::string::npos || !std::isalpha((unsigned char)order[0])) {
        err="three-card placement order: segments of h, p or t, each optionally followed by a count";return false;
    }
    if(D.size()!=1 || head_dev!=0 || (o.resident_cap>=0 && o.resident_cap<g.n_exp)) {
        err="three-card placement requires one primary device and no per-layer resident cap";return false;
    }
    size_t primary_capacity=0;
    for(const auto& l:L)if(l.moe)primary_capacity+=l.res_ids.size();
    const size_t helper_capacity=helper_live_capacity ? helper_live_capacity : helper->records().size();
    const auto ranked=third_ranked(counts);
    if(ranked.size()<primary_capacity+helper_capacity+size_t(third_capacity)) {err="three-card placement: fewer experts than the cards hold";return false;}
    std::vector<int> ps,hs,ts;size_t at=0;
    for(size_t i=0;i<order.size();) {
        const char who=order[i++];
        size_t n=0;bool counted=false;
        while(i<order.size() && std::isdigit((unsigned char)order[i])) {n=n*10+size_t(order[i++]-'0');counted=true;if(n>100000)break;}
        auto& roster=who=='h'?hs:who=='p'?ps:ts;
        const size_t capacity=who=='h'?helper_capacity:who=='p'?primary_capacity:size_t(third_capacity);
        if(roster.size()>capacity) {err="three-card placement order overfills a card";return false;}
        if(!counted)n=capacity-roster.size();
        if(roster.size()+n>capacity || at+n>ranked.size()) {err="three-card placement order overfills a card";return false;}
        roster.insert(roster.end(),ranked.begin()+long(at),ranked.begin()+long(at+n));at+=n;
    }
    if(ps.size()!=primary_capacity || hs.size()!=helper_capacity || ts.size()!=size_t(third_capacity)) {
        err="three-card placement order must fill the primary ("+std::to_string(primary_capacity)+"), the helper ("+
            std::to_string(helper_capacity)+") and the pool ("+std::to_string(third_capacity)+") exactly";return false;
    }
    std::string placed;
    if(!placement(counts,true,placed,err,&ps,&hs,false,true))return false;
    std::string fields;
    if(!third_set(ts,fields,err))return false;
    // the placement's own report, with the pool's fields and the order added
    if(placed.empty() || placed.back()!='}') {err="three-card placement: unexpected placement report";return false;}
    placed.pop_back();
    report=placed+",\"three_order\":\""+order+"\","+fields+"}";
    return true;
}
