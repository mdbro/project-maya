// Exercise the actual phase API and stage_window with deterministic queued CUDA
// streams. No model weights, GPU runtime, or duplicated scheduling policy is used.
#include "../src/mm3/mm3_demand_snapshot.hpp"
#include "../src/mm3/mm3_native_stage_schedule.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
using mm3::native_stage_schedule;

constexpr int cudaSuccess=0, cudaErrorNotReady=2;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2, cudaMemcpyDeviceToDevice=3;
using cudaError_t=int;
struct Event { bool ready=false; };
using cudaEvent_t=Event*;
struct FakeCuda {
    std::array<std::vector<std::function<void()>>,5> q;
    std::vector<std::string> operations;
    int fail_at=-1, calls=0, wrapper_errors=0;
    bool record_device_selection=false;
    bool fail(const std::string& name) { operations.push_back(name); return calls++==fail_at; }
    void submit(int s,std::function<void()> f) { q.at(s).push_back(std::move(f)); }
    void drain(int s) { auto& v=q.at(s); while(!v.empty()) { auto f=std::move(v.front());v.erase(v.begin());f(); } }
};
static FakeCuda* gpu;
static int cudaSetDevice(int id) {
    if(gpu->record_device_selection && gpu->fail("device:"+std::to_string(id)))return 1;
    return cudaSuccess;
}
static int cudaMemcpyAsync(void* dst,const void* src,size_t n,int kind,int s) {
    const std::string name="copy:"+std::to_string(s)+":"+std::to_string(kind);
    if(gpu->fail(name))return 1;
    gpu->submit(s,[=]{std::memcpy(dst,src,n);}); return cudaSuccess;
}
static int cudaStreamSynchronize(int s) {
    if(gpu->fail("join:"+std::to_string(s)))return 1;
    gpu->drain(s);return cudaSuccess;
}
static int cudaStreamQuery(int s) {
    if(gpu->fail("stream-query:"+std::to_string(s)))return 1;
    return gpu->q.at(s).empty() ? cudaSuccess : cudaErrorNotReady;
}
static int cudaEventRecord(Event* e,int s) {
    if(gpu->fail("record:"+std::to_string(s)))return 1;
    e->ready=false;gpu->submit(s,[=]{e->ready=true;});return cudaSuccess;
}
static int cudaEventSynchronize(Event* e) {
    if(gpu->fail("wait"))return 1;
    // Events in this fixture are recorded on the front stream. Drain only up
    // to the event, leaving a subsequently submitted B prefix pending.
    auto& q=gpu->q.at(1);
    while(!e->ready && !q.empty()) {auto f=std::move(q.front());q.erase(q.begin());f();}
    assert(e->ready);return cudaSuccess;
}
static int cudaEventQuery(Event* e) {
    if(gpu->fail("query"))return 1;
    return e->ready ? cudaSuccess : cudaErrorNotReady;
}
static int cudaGetLastError(){return 0;}
static const char* cudaGetErrorString(int){return "injected";}
#define CK(expr) do {if((expr)!=cudaSuccess) {err="injected CUDA failure";return false;}} while(0)
namespace k {
static void submit(int) {}
static void rms(float* h,float*,int rows,int dim,float,float* xn,void*,int s) {
    gpu->submit(s,[=]{std::memcpy(xn,h,size_t(rows)*dim*sizeof(float));});
}}
namespace x3 {
struct MvJob {void* m=nullptr;float* xf=nullptr;float* y=nullptr;int ncol=0,xf_cs=0,y_cs=0;};
static int launch_errors() {return gpu->wrapper_errors;}
static int mv_cols(int) {return 4;}
static bool mv(MvJob* j,int,int s) {
    const MvJob a=*j;gpu->submit(s,[=]{for(int i=0;i<a.ncol;++i) a.y[i*a.y_cs]=a.xf[i*a.xf_cs];});return true;
}}
struct Dev {
    int id=0,s=0,s2=0;
    std::array<float,4> emb{},hidden{},norm{},logits{},logits_host{};
    std::array<int,4> positions{},positions_host{};
    float *emb_h=emb.data(),*h=hidden.data(),*xn=norm.data(),*lg=logits.data(),*lg_h=logits_host.data();
    void* xn16=nullptr;int *pos=positions.data(),*pos_h=positions_host.data();
    unsigned snapshots[2]{};unsigned* hf_stage_h[2]={&snapshots[0],&snapshots[1]};
    std::vector<int> promo_q;
};
struct Stats {
    long stage_segments=0,stage_rows=0,stage_b_pending=0,stage_b_ready_before_a_end=0;
    long stage_b_prefix_rows=0,stage_b_prefix_discarded_rows=0,stage_b_suffix_skipped_rows=0,stage_b_suffix_completed_rows=0,stage_windows=0;
    double stage_b_wait_ms=0,stage_prefix_b_submit_ms=0,stage_suffix_a_ms=0;
};
struct Capture {int pos,rows;std::array<std::vector<float>,5> taps;};
struct Draft {
    std::vector<Capture> calls;
    int attempts=0,fail_at=-1;
    bool context(float* taps,int rows,int pos,std::string& err) {
        if(attempts++==fail_at){err="injected draft context failure";return false;}
        Capture c{pos,rows,{}};
        for(int i=0;i<5;++i)c.taps[i].assign(taps+i*rows,taps+(i+1)*rows);
        calls.push_back(std::move(c));return true;
    }
};
struct Model {
    struct {int native_stage_prefix=3,max_ctx=1024;} o;
    struct {int dim=1,n_layers=60,vocab=1;float eps=0;} g;
    bool native=true,draft_running=true,hf_demand_poisoned=false,hf_demand_active=false,stage_poisoned=false;
    std::atomic<bool> retirement_uncertain{false};
    int head_dev=1,cur_pos=0,head=0;long long fwd=0;float* norm1=nullptr;
    std::vector<int> rseq,tap_of=std::vector<int>(60,-1);
    std::vector<Dev> D=std::vector<Dev>(2);Stats st;Draft draft,*draftm=&draft;
    std::array<std::array<float,4>,2> hidden{};float* stage_hidden[2]={hidden[0].data(),hidden[1].data()};
    std::array<float,4> bridge{};float* serial_bridge=bridge.data();size_t serial_bridge_bytes=sizeof(bridge);
    Event events[2];Event* stage_ready[2]={&events[0],&events[1]};
    std::array<float,20> taps_storage{};float* taps=taps_storage.data();
    struct Row {int layer,pos;float input;};std::vector<Row> layers;
    std::vector<std::pair<int,long long>> consumed;
    int fail_layer=-1,fail_layer_end=-1,fail_progress_at=-1,progress_calls=0;
    bool fail_issue=false,fail_promotion=false,fail_reset=false;
    Model() {
        D[0].id=0;D[0].s=1;D[0].s2=2;D[1].id=1;D[1].s=3;D[1].s2=4;
        for(int i=0;i<5;++i)tap_of[3+14*i]=i;
    }
    static double now(){return 0;}
    bool embed(const int* ids,int n,float* out,std::string&) {for(int i=0;i<n;++i)out[i]=float(ids[i]);return true;}
    bool layer(int li,int n,int end,std::string& err) {
        if(li==fail_layer && end==fail_layer_end){err="injected layer failure";return false;}
        Dev& d=D[li<3?0:1];float* h=d.h;int* p=d.pos;const int host_pos=cur_pos;
        gpu->submit(d.s,[this,li,n,h,p,end,host_pos]{
            assert(end==p[n-1] && host_pos==p[n-1]);
            for(int i=0;i<n;++i){layers.push_back({li,p[i],h[i]});h[i]+=float(li+1);}
        });return true;
    }
    bool reset_hf_route_errors(std::string& err){if(fail_reset){err="injected reset failure";return false;}return true;}
    bool queue_hf_demand(Dev&,unsigned*,std::string&){hf_demand_active=true;return true;}
    bool queue_hf_route_errors(Dev&,std::string&){return true;}
    bool check_hf_route_errors(Dev&,std::string&){return true;}
    bool consume_hf_demand(Dev& d,unsigned*,long long tick,std::string&){consumed.emplace_back(d.id,tick);return true;}
    bool refresh_slots(Dev&,std::string&){return true;}
    bool helper_live_progress(Dev&,std::string& err){
        if(progress_calls++==fail_progress_at){err="injected helper progress failure";return false;}return true;
    }
    bool helper_live_issue(Dev&,std::string& err){if(fail_issue){err="injected helper issue failure";return false;}return true;}
    bool issue_promotions(Dev&,std::string& err){if(fail_promotion){err="injected publication failure";return false;}return true;}
#include "../src/mm3/mm3_native_stage_hop.inc"
#include "../src/mm3/mm3_native_stage.inc"
};
static float after_layer(int id,int layer) {return float(id+(layer+1)*(layer+2)/2);}
static void check_capture(const Capture& c,const int* ids,int base,int start,int rows) {
    assert(c.pos==base+start && c.rows==rows);
    for(int i=0;i<5;++i)for(int j=0;j<rows;++j)
        assert(c.taps[i][j]==after_layer(ids[start+j],3+14*i));
}
static void check_layers(const Model& m,const int* ids,int base,int start,int rows,int first,int last) {
    for(int l=first;l<=last;++l)for(int j=0;j<rows;++j) {
        int found=0;for(const auto& x:m.layers) if(x.layer==l && x.pos==base+start+j) {
            assert(x.input==(l?after_layer(ids[start+j],l-1):float(ids[start+j])));++found;
        } assert(found==1);
    }
}
static bool run(Model& m,FakeCuda& c,const int* ids,int n,int base,bool overlap,bool accept,int& completed,std::array<float,8>& out) {
    gpu=&c;std::string err;out.fill(-999);
    return m.stage_window(ids,n,base,overlap,[accept](int a){assert(a==4);return accept;},completed,out.data(),err);
}
static void happy_and_reuse() {
    const int ids[8]={10,11,12,13,14,15,16,17};FakeCuda c;Model m;std::array<float,8> out;int done=-1;
    assert(run(m,c,ids,8,100,true,true,done,out));assert(done==8);
    assert(m.st.stage_b_pending==1 && m.st.stage_b_suffix_completed_rows==4);
    assert(m.draft.calls.size()==2);check_capture(m.draft.calls[0],ids,100,0,4);check_capture(m.draft.calls[1],ids,100,4,4);
    check_layers(m,ids,100,0,8,0,59);
    for(int i=0;i<8;++i)assert(out[i]==after_layer(ids[i],59));
    // A later call can rewind the speculative frontier and must use fresh positions.
    const int next[8]={20,21,22,23,24,25,26,27};
    assert(run(m,c,next,7,102,true,true,done,out));assert(done==7);
    check_capture(m.draft.calls[2],next,102,0,4);check_capture(m.draft.calls[3],next,102,4,3);
    for(int i=0;i<7;++i)assert(out[i]==after_layer(next[i],59));assert(out[7]==-999);
}
static void all_segment_sizes() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    for(int n=1;n<=8;++n) for(bool overlap:{false,true}) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;
        assert(run(m,c,ids,n,80,overlap,true,done,out));assert(done==n);
        const int a=std::min(n,4),b=n-a;
        assert(m.draft.calls.size()==size_t(b?2:1));
        check_capture(m.draft.calls[0],ids,80,0,a);
        if(b)check_capture(m.draft.calls[1],ids,80,a,b);
        check_layers(m,ids,80,0,n,0,59);
        for(int i=0;i<n;++i)assert(out[i]==after_layer(ids[i],59));
        for(int i=n;i<8;++i)assert(out[i]==-999);
    }
}
static void stop_after_a() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    // Caller stop reasons share this callback boundary; caller decisions have
    // their own state tests.
    for(bool overlap:{false,true}) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;
        assert(run(m,c,ids,7,40,overlap,false,done,out));assert(done==4);
        assert(m.draft.calls.size()==1);check_capture(m.draft.calls[0],ids,40,0,4);
        check_layers(m,ids,40,0,4,0,59);
        for(const auto& row:m.layers)if(row.pos>=44)assert(overlap && row.layer<3);
        for(int i=0;i<4;++i)assert(out[i]==after_layer(ids[i],59));
        for(int i=4;i<8;++i)assert(out[i]==-999);
        assert(m.st.stage_b_suffix_skipped_rows==3);
        assert(m.st.stage_b_prefix_discarded_rows==(overlap?3:0));
        const int continuation[8]={30,31,32,33,34,35,36,37};
        assert(run(m,c,continuation,5,42,overlap,true,done,out));assert(done==5);
        assert(m.draft.calls.size()==3);
        check_capture(m.draft.calls[1],continuation,42,0,4);
        check_capture(m.draft.calls[2],continuation,42,4,1);
        for(int i=0;i<5;++i)assert(out[i]==after_layer(continuation[i],59));
        for(int i=5;i<8;++i)assert(out[i]==-999);
    }
}
static void failures() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    FakeCuda baseline;baseline.record_device_selection=true;Model good;std::array<float,8> out;int done;
    assert(run(good,baseline,ids,8,100,true,true,done,out));
    // Inject once at every actual device selection, copy, event and join.
    for(size_t fail=0;fail<baseline.operations.size();++fail) {
        FakeCuda c;c.record_device_selection=true;c.fail_at=int(fail);Model m;const auto hidden0=m.stage_hidden[0],hidden1=m.stage_hidden[1];
        done=-1;assert(!run(m,c,ids,8,100,true,true,done,out));
        assert(m.stage_poisoned && m.retirement_uncertain.load());
        assert(m.stage_hidden[0]==hidden0 && m.stage_hidden[1]==hidden1);
        assert(done==0 || done==4 || done==8);
        assert(m.draft.calls.size()==size_t(done/4));
        for(int i=0;i<done;++i)assert(out[i]==after_layer(ids[i],59));
        for(int i=done;i<8;++i)assert(out[i]==-999);
        for(size_t i=0;i<m.draft.calls.size();++i)check_capture(m.draft.calls[i],ids,100,int(i)*4,4);
        assert(m.consumed.empty());
        const int before=c.calls;int again=-1;std::string err;
        assert(!m.stage_window(ids,8,100,true,{},again,out.data(),err));assert(c.calls==before);
        // A later successful drain cannot unquarantine a failed handoff.
        assert(cudaStreamSynchronize(1)==cudaSuccess);
        assert(cudaStreamSynchronize(3)==cudaSuccess);
        assert(m.stage_poisoned && m.retirement_uncertain.load());
    }
}
static void assert_quarantined(Model& m,FakeCuda& c) {
    assert(m.stage_poisoned && m.retirement_uncertain.load() && m.consumed.empty());
    float* const a=m.stage_hidden[0];float* const b=m.stage_hidden[1];
    assert(cudaStreamSynchronize(1)==cudaSuccess);
    assert(cudaStreamSynchronize(3)==cudaSuccess);
    assert(m.stage_poisoned && m.retirement_uncertain.load());
    assert(m.stage_hidden[0]==a && m.stage_hidden[1]==b);
    const int before=c.calls;int completed=-1;float out[8]{};std::string err;const int ids[8]{};
    assert(!m.stage_window(ids,8,100,true,{},completed,out,err));
    assert(c.calls==before);
}
static void check_output(const std::array<float,8>& out,const int* ids,int written) {
    for(int i=0;i<written;++i)assert(out[i]==after_layer(ids[i],59));
    for(int i=written;i<8;++i)assert(out[i]==-999);
}
static void model_failures() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    // Failure in each prefix/suffix segment, including B submitted during A.
    struct LayerFailure {int layer,end,done;};
    for(const LayerFailure f: {LayerFailure{1,103,0},{1,107,0},{17,103,0},{17,107,4}}) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;
        m.fail_layer=f.layer;m.fail_layer_end=f.end;
        assert(!run(m,c,ids,8,100,true,true,done,out));assert(done==f.done);
        assert(m.draft.calls.size()==size_t(done/4));check_output(out,ids,done);
        for(const auto& row:m.layers)assert(!(row.layer==f.layer && row.pos>=f.end-3 && row.pos<=f.end));
        assert_quarantined(m,c);
    }
    // The head output is copied before draft context: failed context leaves
    // those logits written, but does not advance the completed frontier.
    for(int fail=0;fail<2;++fail) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;m.draft.fail_at=fail;
        assert(!run(m,c,ids,8,100,true,true,done,out));assert(done==fail*4);
        assert(m.draft.attempts==fail+1 && m.draft.calls.size()==size_t(fail));
        check_output(out,ids,(fail+1)*4);assert_quarantined(m,c);
    }
    // Progress is called at suffix A, suffix B and the post-window boundary.
    for(int fail=0;fail<3;++fail) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;m.fail_progress_at=fail;
        assert(!run(m,c,ids,8,100,true,true,done,out));assert(done==fail*4);
        assert(m.progress_calls==fail+1 && m.draft.calls.size()==size_t(done/4));
        check_output(out,ids,done);assert_quarantined(m,c);
    }
    for(bool publication:{false,true}) {
        FakeCuda c;Model m;std::array<float,8> out;int done=-1;
        if(publication){m.D[1].promo_q.push_back(1);m.fail_promotion=true;}
        else m.fail_issue=true;
        assert(!run(m,c,ids,8,100,true,true,done,out));assert(done==8);
        assert(m.draft.calls.size()==2);check_output(out,ids,8);assert_quarantined(m,c);
    }
    {
        FakeCuda c;gpu=&c;Model m;std::array<float,8> out;out.fill(-999);
        int done=-1;std::string err;bool threw=false;
        try {
            (void)m.stage_window(ids,8,100,true,[](int)->bool {throw std::runtime_error("injected callback");},done,out.data(),err);
        } catch(const std::runtime_error&) {threw=true;}
        assert(threw && done==4 && m.draft.calls.size()==1);
        check_output(out,ids,4);assert_quarantined(m,c);
    }
}
static void wrapper_failure() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    FakeCuda c;c.wrapper_errors=1;Model m;std::array<float,8> out;int done=-1;
    assert(!run(m,c,ids,8,100,true,true,done,out));
    assert(done==8 && m.draft.calls.size()==2);
    for(int i=0;i<8;++i)assert(out[i]==after_layer(ids[i],59));
    assert(m.consumed.empty());
    assert(m.stage_poisoned && m.retirement_uncertain.load());
    c.wrapper_errors=0;const int before=c.calls;std::string err;
    assert(!m.stage_window(ids,8,100,true,{},done,out.data(),err));
    assert(c.calls==before && m.consumed.empty());
}
static void explicit_phases() {
    int ids[8]={10,11,12,13,14,15,16,17};
    const int original[8]={10,11,12,13,14,15,16,17};
    FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0,other=0;bool ready=false;int done=-1;float out[8]{};
    assert(m.stage_begin(ids,8,100,t,err) && t);
    assert(!m.stage_begin(ids,8,100,other,err));
    assert(!m.stage_prefix_submit(t+1,0,err));
    assert(!m.stage_prefix_submit(t,1,err));
    assert(!m.stage_suffix_execute(t,0,out,4,err));
    assert(m.stage_prefix_submit(t,0,err));
    assert(!m.stage_prefix_submit(t,0,err));
    assert(m.stage_prefix_ready(t,0,false,ready,err) && !ready);
    assert(!m.stage_prefix_submit(t,1,err));
    assert(m.stage_prefix_ready(t,0,true,ready,err) && ready);
    for(int& id:ids)id=-1; // begin copied IDs, including the later B submission
    assert(m.stage_prefix_submit(t,1,err));
    assert(!m.stage_suffix_execute(t,1,out+4,4,err));
    assert(!m.stage_suffix_execute(t,0,out,3,err));
    assert(m.stage_suffix_execute(t,0,out,4,err));
    assert(!m.stage_suffix_execute(t,0,out,4,err));
    assert(m.stage_finish(t,done,err) && done==4);
    assert(m.st.stage_windows==1 && m.st.stage_segments==1);
    assert(m.st.stage_b_prefix_rows==4 && m.st.stage_b_prefix_discarded_rows==4 && m.st.stage_b_suffix_skipped_rows==4);
    assert(m.consumed.size()==3); // A prefix, A suffix, discarded B prefix
    check_layers(m,original,100,0,8,0,2);
    check_layers(m,original,100,0,4,3,59);
    assert(m.draft.calls.size()==1);check_capture(m.draft.calls[0],original,100,0,4);
    assert(!m.stage_finish(t,done,err));
    assert(m.stage_begin(original,5,200,other,err) && other!=t);
    assert(!m.stage_prefix_submit(t,0,err));
    assert(m.stage_finish(other,done,err) && done==0); // cancel before any submission
    assert(!m.stage_poisoned && !m.retirement_uncertain.load());
    assert(m.st.stage_windows==2 && m.st.stage_b_suffix_skipped_rows==4);
    assert(m.stage_begin(original,8,205,other,err));
    assert(m.stage_prefix_submit(other,0,err));
    assert(m.stage_prefix_ready(other,0,true,ready,err) && ready);
    assert(m.stage_prefix_submit(other,1,err));
    assert(m.stage_finish(other,done,err) && done==0);
    assert(m.st.stage_b_suffix_skipped_rows==4 && m.st.stage_b_prefix_discarded_rows==8);
    assert(m.stage_begin(original,5,210,other,err));
    assert(m.stage_prefix_submit(other,0,err));
    assert(m.stage_finish(other,done,err) && done==0); // pending A drains on cancellation
    assert(m.stage_begin(original,1,210,other,err));
    assert(m.stage_prefix_submit(other,0,err));
    assert(m.stage_suffix_execute(other,0,out,1,err));
    assert(m.stage_finish(other,done,err) && done==1);
    assert(out[0]==after_layer(original[0],59));
}
static void incremental_suffix() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;bool ready=false,all=false;int done=-1;float out[8]{};
    assert(m.stage_begin(ids,8,124,t,err));
    assert(m.stage_prefix_submit(t,0,err));
    assert(m.stage_prefix_ready(t,0,true,ready,err) && ready);
    assert(m.stage_suffix_launch(t,0,err));
    assert(!m.stage_suffix_launch(t,1,err));
    assert(!m.stage_suffix_finish(t,0,out,4,err));
    assert(m.stage_suffix_ready(t,0,false,ready,err) && !ready);
    assert(m.stage_suffix_service(t,0,all,err) && !all);
    // A ends at 127, B starts at 128 and changes the host attention position.
    assert(m.stage_prefix_submit(t,1,err));
    assert(!m.stage_suffix_service(t,1,all,err));
    for(int layer=4;layer<60;++layer) {
        assert(m.stage_suffix_service(t,0,all,err));
        assert(all==(layer==59));
    }
    assert(!m.stage_suffix_service(t,0,all,err));
    assert(m.stage_suffix_ready(t,0,false,ready,err) && !ready);
    assert(!m.stage_suffix_finish(t,0,out,4,err));
    c.drain(3); // simulated GPU progress, not a host wait
    assert(m.stage_suffix_ready(t,0,false,ready,err) && ready);
    assert(m.stage_suffix_finish(t,0,out,4,err));
    assert(!m.stage_suffix_finish(t,0,out,4,err));
    assert(m.stage_suffix_launch(t,1,err));
    for(int layer=3;layer<60;++layer)assert(m.stage_suffix_service(t,1,all,err) && all==(layer==59));
    assert(m.stage_suffix_ready(t,1,true,ready,err) && ready);
    assert(m.stage_suffix_finish(t,1,out+4,4,err));
    assert(m.stage_finish(t,done,err) && done==8);
    check_layers(m,ids,124,0,8,0,59);
    assert(m.draft.calls.size()==2);check_capture(m.draft.calls[0],ids,124,0,4);check_capture(m.draft.calls[1],ids,124,4,4);
    for(int i=0;i<8;++i)assert(out[i]==after_layer(ids[i],59));
}
static void incremental_cancel_and_failures() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    {
        FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;bool all=false;int done=-1;
        assert(m.stage_begin(ids,8,100,t,err));assert(m.stage_prefix_submit(t,0,err));
        assert(m.stage_suffix_launch(t,0,err));
        for(int i=0;i<4;++i)assert(m.stage_suffix_service(t,0,all,err) && !all);
        assert(m.stage_finish(t,done,err) && done==0);
        assert(m.consumed.size()==2); // partial suffix demand, and A prefix
        assert(m.draft.calls.empty() && m.st.stage_segments==0 && !m.stage_poisoned);
        check_layers(m,ids,100,0,4,0,6);
        assert(m.stage_begin(ids,1,100,t,err));assert(m.stage_prefix_submit(t,0,err));
        float out=0;assert(m.stage_suffix_execute(t,0,&out,1,err));
        assert(m.stage_finish(t,done,err) && done==1 && out==after_layer(ids[0],59));
    }
    {
        FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;bool ready=false,all=false;int done=-1;float out[4]{};
        assert(m.stage_begin(ids,8,100,t,err));assert(m.stage_prefix_submit(t,0,err));
        assert(m.stage_prefix_ready(t,0,true,ready,err) && ready);
        assert(m.stage_prefix_submit(t,1,err));
        assert(m.stage_suffix_execute(t,0,out,4,err));
        assert(m.stage_suffix_launch(t,1,err));
        assert(m.stage_suffix_service(t,1,all,err) && !all);
        assert(m.stage_finish(t,done,err) && done==4);
        assert(m.st.stage_b_suffix_skipped_rows==0 && m.st.stage_b_prefix_discarded_rows==4);
        assert(m.draft.calls.size()==1 && m.st.stage_segments==1);
        assert(!m.stage_poisoned && !m.retirement_uncertain.load());
    }
    {
        FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;bool all=false;int done=-1;
        assert(m.stage_begin(ids,8,100,t,err));assert(m.stage_prefix_submit(t,0,err));
        assert(m.stage_suffix_launch(t,0,err));m.fail_layer=4;m.fail_layer_end=103;
        assert(m.stage_suffix_service(t,0,all,err) && !all);
        assert(!m.stage_suffix_service(t,0,all,err));
        assert(m.stage_poisoned && m.retirement_uncertain.load());
        assert(!m.stage_finish(t,done,err));
        assert(cudaStreamSynchronize(3)==cudaSuccess);
        assert(!m.stage_begin(ids,1,100,t,err));
    }
    {
        FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;bool all=false,ready=false;
        assert(m.stage_begin(ids,1,100,t,err));assert(m.stage_prefix_submit(t,0,err));
        assert(m.stage_suffix_launch(t,0,err));
        while(!all)assert(m.stage_suffix_service(t,0,all,err));
        c.fail_at=c.calls;assert(!m.stage_suffix_ready(t,0,false,ready,err));
        assert(m.stage_poisoned && m.retirement_uncertain.load());
    }
}
static void phase_failed_begin() {
    const int ids[1]={10};FakeCuda c;gpu=&c;Model m;std::string err;uint64_t ticket=99;
    m.fail_reset=true;
    assert(!m.stage_begin(ids,1,100,ticket,err) && ticket==0);
    assert(m.stage_poisoned && m.retirement_uncertain.load());
    assert(cudaStreamSynchronize(1)==cudaSuccess && cudaStreamSynchronize(3)==cudaSuccess);
    m.fail_reset=false;
    assert(!m.stage_begin(ids,1,100,ticket,err) && ticket==0);
    assert(m.stage_poisoned && m.retirement_uncertain.load());
}
static void phase_failed_retirement() {
    const int ids[8]={10,11,12,13,14,15,16,17};
    FakeCuda c;gpu=&c;Model m;std::string err;uint64_t t=0;int done=-1;
    assert(m.stage_begin(ids,8,100,t,err));
    assert(m.stage_prefix_submit(t,0,err));
    c.fail_at=c.calls; // first retirement join fails, even if a later drain succeeds
    assert(!m.stage_finish(t,done,err));
    assert(m.stage_poisoned && m.retirement_uncertain.load());
    assert(cudaStreamSynchronize(1)==cudaSuccess);
    assert(!m.stage_begin(ids,1,100,t,err));
    assert(m.consumed.empty());
}
static void serial_hop() {
    // One-row ordinary decode and a four-row split-prefill transfer both use
    // this same production bridge helper. Its copies really remain queued
    // until the individual producer/consumer stream joins.
    for(int rows:{1,4}) {
        FakeCuda c;c.record_device_selection=true;gpu=&c;Model m;std::string err;
        for(int i=0;i<4;++i){m.D[0].h[i]=float(100+i);m.D[1].h[i]=-1;}
        assert(m.stage_serial_hop(m.D[0],m.D[1],rows*sizeof(float),m.D[1].h,err));
        assert(!m.stage_poisoned && !m.retirement_uncertain.load());
        for(int i=0;i<4;++i)assert(m.D[1].h[i]==(i<rows?float(100+i):-1));
        assert((c.operations==std::vector<std::string>{"device:0","copy:1:2","join:1","device:1","copy:3:1","join:3"}));
    }
    // Independently fail each device selection, copy submission and join.
    // A later successful join cannot clear poison or release the bridge.
    for(int fail=0;fail<6;++fail) {
        FakeCuda c;c.record_device_selection=true;c.fail_at=fail;gpu=&c;Model m;std::string err;
        for(int i=0;i<4;++i){m.D[0].h[i]=float(100+i);m.D[1].h[i]=-1;}
        float* const bridge=m.serial_bridge;
        assert(!m.stage_serial_hop(m.D[0],m.D[1],sizeof(m.bridge),m.D[1].h,err));
        assert(m.stage_poisoned && m.retirement_uncertain.load());
        assert(m.serial_bridge==bridge && m.serial_bridge_bytes==sizeof(m.bridge));
        for(int i=0;i<4;++i)assert(m.D[1].h[i]==-1);
        if(fail==2)assert(!c.q[1].empty()); // accepted D2H, failed source join
        if(fail==5)assert(!c.q[3].empty()); // accepted H2D, failed destination join
        assert(cudaStreamSynchronize(1)==cudaSuccess);
        assert(cudaStreamSynchronize(3)==cudaSuccess);
        assert(m.stage_poisoned && m.retirement_uncertain.load());
        assert(m.serial_bridge==bridge);
    }
    FakeCuda c;gpu=&c;Model m;std::string err;
    assert(!m.stage_serial_hop(m.D[0],m.D[1],sizeof(m.bridge)+1,m.D[1].h,err));
    assert(c.operations.empty());
}
int main(){happy_and_reuse();all_segment_sizes();stop_after_a();failures();model_failures();wrapper_failure();explicit_phases();incremental_suffix();incremental_cancel_and_failures();phase_failed_begin();phase_failed_retirement();serial_hop();}
