#pragma once
// A stored reference state: the learned checkpoint of mm3_native_placement.inc made durable, with the third card's
// roster. Benchmarks force every engine start into one such state (PLACEMENT reference FILE), so a run on one start can
// be compared with a control stored from another (the user, 2026-11-14: "force engine to start from the same place for
// each benchmark run"). Serving never reads one.
//
// What a state holds is what decides, between two starts of one binary and one set of start options, which lane computes
// an expert: the three cards' rosters (the host bank is the rest), the demand scores, the admission credits, and the lane
// costs a start otherwise measures for itself - the two CPU-lane costs and the copy cost, and the grouped staging table -
// with the flag that says they were measured after a real prefill (so the first request does not measure them again).
//
// The file is text, written by tools/mm3/native_reference_state.py from a PLACEMENT snapshot reply. Parsing and every
// check a file can fail on its own happen here, before any model state changes.
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "mm3_batch_staging.hpp"

namespace mm3 {
struct ReferenceState {
    std::string id;                     // the state's identity as the tool computed it (SHA-256 of its canonical content)
    int layers=0,experts=0,dim=0,ff=0,resident_cap=0;
    std::string arithmetic,calibration_arithmetic,batch_arithmetic,batch_calibration;
    bool calibrated=false,primary_adaptive=false;
    double primary_credit=0,helper_credit=0;
    bool policy_legacy=false;int policy_horizon=0,policy_max_exchanges=0;
    double policy_service_saved_ms=0,policy_h2d_ms=0,policy_d2h_ms=0,policy_interference_ms=0,policy_margin_ms=0;
    double cpu1=0,cpu_add=0,copy=0,cpu_scale=1;
    std::array<double,12> retime{};     // LaneRetime, in the order the snapshot reports it
    BatchStagingCosts costs;
    std::vector<int> primary,helper,third;   // layer * experts + expert; the primary's and the helper's in slot order
    std::vector<float> scores;          // layers * 1024, as the checkpoint keeps them
    std::vector<int> adaptive_main;
};

// Reads FILE. `moe` says which layers hold experts (one entry a layer). False with `err` set on anything missing, out of
// range, repeated, or held by two cards.
inline bool read_reference_state(const std::string& path,const std::vector<uint8_t>& moe,int experts,
                                 ReferenceState& out,std::string& err) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()>std::streamoff(16*1024*1024)) {err="reference state: missing or oversized file";return false;}
    f.seekg(0);
    const auto fail=[&](const char* what) {err=std::string("reference state: ")+what;return false;};
    const auto word=[&](const char* name) {std::string w;return bool(f>>w) && w==name;};
    const auto finite=[](double x) {return std::isfinite(x);};
    int version=0;
    if(!word("MM3_REFERENCE_STATE") || !(f>>version) || version!=1) return fail("not a version 1 file");
    if(!word("ID") || !(f>>out.id) || out.id.size()!=64 || out.id.find_first_not_of("0123456789abcdef")!=std::string::npos)
        return fail("invalid identity");
    if(!word("GEOMETRY") || !(f>>out.layers>>out.experts>>out.dim>>out.ff>>out.resident_cap) ||
       out.layers!=int(moe.size()) || out.experts!=experts || experts<1 || experts>1024) return fail("another model geometry");
    const auto name=[&](const char* key,std::string& value) {
        if(!word(key) || !(f>>value) || value.size()>64) return false;
        if(value=="-") value.clear();
        return true;
    };
    int flag=0;
    if(!name("ARITHMETIC",out.arithmetic) || !name("CALIBRATION_ARITHMETIC",out.calibration_arithmetic)) return fail("invalid arithmetic");
    if(!word("CALIBRATED") || !(f>>flag) || (flag!=0 && flag!=1)) return fail("invalid calibrated flag");
    out.calibrated=flag;
    if(!word("PRIMARY_ADAPTIVE") || !(f>>flag) || (flag!=0 && flag!=1)) return fail("invalid primary_adaptive flag");
    out.primary_adaptive=flag;
    if(!word("CREDITS") || !(f>>out.primary_credit>>out.helper_credit) || !finite(out.primary_credit) || !finite(out.helper_credit) ||
       out.primary_credit<0 || out.helper_credit<0) return fail("invalid admission credits");
    if(!word("POLICY") || !(f>>flag>>out.policy_horizon>>out.policy_max_exchanges>>out.policy_service_saved_ms>>out.policy_h2d_ms>>
       out.policy_d2h_ms>>out.policy_interference_ms>>out.policy_margin_ms) || (flag!=0 && flag!=1) || out.policy_horizon<1 ||
       out.policy_max_exchanges<0 || !finite(out.policy_service_saved_ms) || !finite(out.policy_h2d_ms) || !finite(out.policy_d2h_ms) ||
       !finite(out.policy_interference_ms) || !finite(out.policy_margin_ms)) return fail("invalid residency policy");
    out.policy_legacy=flag;
    if(!word("LANE") || !(f>>out.cpu1>>out.cpu_add>>out.copy>>out.cpu_scale) || !finite(out.cpu1) || !finite(out.cpu_add) ||
       !finite(out.copy) || !finite(out.cpu_scale) || out.cpu1<0 || out.cpu_add<0 || out.copy<0 || out.cpu_scale<=0) return fail("invalid lane costs");
    if(!word("RETIME")) return fail("missing lane retime");
    for(double& x:out.retime) if(!(f>>x) || !finite(x)) return fail("invalid lane retime");
    if(!word("COSTS") || !(f>>flag>>out.costs.cpu_base_ms>>out.costs.helper_base_ms>>out.costs.h2d_ms>>out.costs.d2h_ms) || (flag!=0 && flag!=1))
        return fail("invalid grouped staging costs");
    out.costs.valid=flag;
    const auto row=[&](const char* key,std::array<double,17>& values) {
        if(!word(key)) return false;
        for(double& x:values) if(!(f>>x) || !finite(x) || x<0) return false;
        return true;
    };
    if(!row("COSTS_CPU_EXTRA",out.costs.cpu_extra_ms) || !row("COSTS_GPU_EXTRA",out.costs.gpu_extra_ms) ||
       !row("COSTS_HELPER_EXTRA",out.costs.helper_extra_ms) || !row("COSTS_GPU_BASE",out.costs.gpu_base_ms) ||
       !finite(out.costs.cpu_base_ms) || !finite(out.costs.helper_base_ms) || !finite(out.costs.h2d_ms) || !finite(out.costs.d2h_ms) ||
       out.costs.cpu_base_ms<0 || out.costs.helper_base_ms<0 || out.costs.h2d_ms<0 || out.costs.d2h_ms<0) return fail("invalid grouped staging costs");
    if(!name("BATCH_ARITHMETIC",out.batch_arithmetic)) return fail("invalid grouped staging arithmetic");
    // The calibration's own report, kept byte for byte: the snapshot and restore replies print it, and the tools' state
    // signature reads it there.
    long bytes=-1;
    if(!word("BATCH_CALIBRATION") || !(f>>bytes) || bytes<0 || bytes>65536 || f.get()!='\n') return fail("invalid grouped staging report");
    out.batch_calibration.resize(size_t(bytes));
    if(bytes && !f.read(out.batch_calibration.data(),bytes)) return fail("short grouped staging report");
    if(out.batch_calibration.find('\n')!=std::string::npos) return fail("invalid grouped staging report");
    const int total=out.layers*experts;
    std::vector<uint8_t> owner(size_t(total),0);
    const auto roster=[&](const char* key,std::vector<int>& ids,uint8_t who) {
        long n=-1;
        if(!word(key) || !(f>>n) || n<0 || n>total) return false;
        ids.resize(size_t(n));
        for(int& id:ids) {
            if(!(f>>id) || id<0 || id>=total || !moe[size_t(id/experts)] || owner[size_t(id)]) return false;
            owner[size_t(id)]=who;
        }
        return true;
    };
    if(!roster("PRIMARY",out.primary,1)) return fail("invalid, repeated or shared primary roster");
    if(!roster("HELPER",out.helper,2)) return fail("invalid, repeated or shared helper roster");
    if(!roster("THIRD",out.third,3)) return fail("invalid, repeated or shared third-card roster");
    long n=-1;
    if(!word("SCORES") || !(f>>n) || n<0 || n>total) return fail("invalid demand scores");
    out.scores.assign(size_t(out.layers)*1024,0.0f);
    for(long i=0;i<n;++i) {
        int layer=-1,expert=-1;double value=0;
        if(!(f>>layer>>expert>>value) || layer<0 || layer>=out.layers || expert<0 || expert>=experts || !moe[size_t(layer)] ||
           !finite(value) || value<=0 || out.scores[size_t(layer)*1024+size_t(expert)]!=0.0f) return fail("invalid demand scores");
        out.scores[size_t(layer)*1024+size_t(expert)]=float(value);
        if(out.scores[size_t(layer)*1024+size_t(expert)]==0.0f) return fail("invalid demand scores");
    }
    std::vector<uint8_t> marked(size_t(total),0);
    if(!word("ADAPTIVE_MAIN") || !(f>>n) || n<0 || n>total) return fail("invalid adaptive marks");
    out.adaptive_main.resize(size_t(n));
    for(int& id:out.adaptive_main) {
        if(!(f>>id) || id<0 || id>=total || marked[size_t(id)]) return fail("invalid adaptive marks");
        marked[size_t(id)]=1;
    }
    std::string rest;
    if(!word("END") || (f>>rest)) return fail("incomplete file or trailing input");
    return true;
}
}
