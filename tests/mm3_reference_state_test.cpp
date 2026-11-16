// The reference state file's parser (mm3_reference_state.hpp): what a good file gives, and what a bad one is refused for
// before any model state could change.
#include "mm3_reference_state.hpp"
#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

// mm3_reference_state_test FILE MOE EXPERTS: reads FILE as the engine would (MOE: one 0 or 1 a layer) and prints what it
// read, one field a line at 17 digits - how tools/mm3/test_native_reference_state.py checks the file the tool writes.
static int dump(const char* path,const std::string& layers,int experts) {
    std::vector<uint8_t> moe;
    for(char c:layers)moe.push_back(c=='1');
    mm3::ReferenceState s;std::string error;
    if(!mm3::read_reference_state(path,moe,experts,s,error)) {std::printf("ERROR %s\n",error.c_str());return 1;}
    std::printf("id %s\ngeometry %d %d %d %d %d\narithmetic %s|%s|%s\nflags %d %d\n",s.id.c_str(),s.layers,s.experts,s.dim,s.ff,s.resident_cap,
                s.arithmetic.c_str(),s.calibration_arithmetic.c_str(),s.batch_arithmetic.c_str(),int(s.calibrated),int(s.primary_adaptive));
    std::printf("credits %.17g %.17g\npolicy %d %d %d %.17g %.17g %.17g %.17g %.17g\n",s.primary_credit,s.helper_credit,int(s.policy_legacy),s.policy_horizon,
                s.policy_max_exchanges,s.policy_service_saved_ms,s.policy_h2d_ms,s.policy_d2h_ms,s.policy_interference_ms,s.policy_margin_ms);
    std::printf("lane %.17g %.17g %.17g %.17g\nretime",s.cpu1,s.cpu_add,s.copy,s.cpu_scale);
    for(double x:s.retime)std::printf(" %.17g",x);
    std::printf("\ncosts %d %.17g %.17g %.17g %.17g\n",int(s.costs.valid),s.costs.cpu_base_ms,s.costs.helper_base_ms,s.costs.h2d_ms,s.costs.d2h_ms);
    const std::array<double,17>* rows[4]={&s.costs.cpu_extra_ms,&s.costs.gpu_extra_ms,&s.costs.helper_extra_ms,&s.costs.gpu_base_ms};
    for(const auto* row:rows) {std::printf("row");for(double x:*row)std::printf(" %.17g",x);std::printf("\n");}
    std::printf("calibration %s\n",s.batch_calibration.c_str());
    const std::vector<int>* rosters[4]={&s.primary,&s.helper,&s.third,&s.adaptive_main};
    for(const auto* ids:rosters) {std::printf("ids");for(int id:*ids)std::printf(" %d",id);std::printf("\n");}
    std::printf("scores");
    for(size_t i=0;i<s.scores.size();++i)if(s.scores[i]!=0.0f)std::printf(" %zu %zu %.17g",i/1024,i%1024,double(s.scores[i]));
    std::printf("\n");
    if(s.version==2) {
        std::printf("shared %llu",(unsigned long long)s.shared_bytes);for(int id:s.shared_keys)std::printf(" %d",id);
        std::printf("\nfree_clock %llu\nfree_global %.17g %llu\nfree_last",(unsigned long long)s.free_time.clock_rows,
            s.free_time.global.value[1][2],(unsigned long long)s.free_time.global.samples[1][2]);
        for(size_t i=0;i<s.free_time.last_move.size();++i)if(s.free_time.last_move[i])
            std::printf(" %zu %zu %llu",i/1024,i%1024,(unsigned long long)s.free_time.last_move[i]);
        std::printf("\n");
    }
    return 0;
}

int main(int argc,char** argv) {
    if(argc==4)return dump(argv[1],argv[2],std::atoi(argv[3]));
    const auto path=std::filesystem::temp_directory_path()/
        ("mm3-reference-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::string id(64,'a');
    const std::string report="{\"units\":\"host wall milliseconds\",\"h2d_ms\":0.59000701003242284}";
    std::string row;
    for(int i=0;i<17;++i)row+=" 0."+std::to_string(i+1);
    // three layers, the middle one without experts; four experts a layer
    const auto file=[&](const std::string& primary,const std::string& helper,const std::string& third,const std::string& scores,
                        const std::string& tail="END\n",const std::string& head="MM3_REFERENCE_STATE 1\n") {
        return head+"ID "+id+"\nGEOMETRY 3 4 6144 3072 -1\nARITHMETIC fp4\nCALIBRATION_ARITHMETIC -\nCALIBRATED 1\nPRIMARY_ADAPTIVE 1\n"
            "CREDITS 0.5 1\nPOLICY 0 40 8 0 0 0 0 0\nLANE 0.4635 0.5560000000000001 0.755 1.5\nRETIME 0 0 0 0 0 0 0 0 0 0 0 0\n"
            "COSTS 1 0.31 0.22 0.59000701003242284 0.6\nCOSTS_CPU_EXTRA"+row+"\nCOSTS_GPU_EXTRA"+row+"\nCOSTS_HELPER_EXTRA"+row+
            "\nCOSTS_GPU_BASE"+row+"\nBATCH_ARITHMETIC fp4\nBATCH_CALIBRATION "+std::to_string(report.size())+"\n"+report+"\n"+
            "PRIMARY "+primary+"\nHELPER "+helper+"\nTHIRD "+third+"\nSCORES "+scores+"\nADAPTIVE_MAIN 1 9\n"+tail;
    };
    const std::vector<uint8_t> moe{1,0,1};
    const auto check=[&](const std::string& text,bool expected,mm3::ReferenceState* keep=nullptr) {
        {std::ofstream f(path,std::ios::binary);f<<text;}
        mm3::ReferenceState out;std::string error;
        const bool ok=mm3::read_reference_state(path.string(),moe,4,out,error);
        assert(ok==expected);
        assert(ok || error.rfind("reference state: ",0)==0);
        if(keep)*keep=out;
    };
    mm3::ReferenceState s;
    check(file("2 0 9","2 1 8","1 2","2 0 0 12.5 2 1 0.1"),true,&s);
    assert(s.id==id && s.layers==3 && s.experts==4 && s.dim==6144 && s.ff==3072 && s.resident_cap==-1);
    assert(s.arithmetic=="fp4" && s.calibration_arithmetic.empty() && s.batch_arithmetic=="fp4" && s.calibrated && s.primary_adaptive);
    assert(s.primary_credit==0.5 && s.helper_credit==1 && !s.policy_legacy && s.policy_horizon==40 && s.policy_max_exchanges==8);
    assert(s.cpu1==0.4635 && s.cpu_add==0.5560000000000001 && s.copy==0.755 && s.cpu_scale==1.5);   // (17 digits come back exactly)
    assert(s.costs.valid && s.costs.h2d_ms==0.59000701003242284 && s.costs.cpu_extra_ms[0]==0.1 && s.costs.gpu_base_ms[16]==0.17);
    assert(s.batch_calibration==report);
    assert((s.primary==std::vector<int>{0,9}) && (s.helper==std::vector<int>{1,8}) && (s.third==std::vector<int>{2}));
    assert(s.scores.size()==3*1024 && s.scores[0]==12.5f && s.scores[2*1024+1]==0.1f && s.scores[1]==0.0f);
    assert(s.adaptive_main==std::vector<int>{9});
    check(file("2 0 9","2 1 8","0",""" 0"),true,&s);                       // (no third pool, no scores)
    assert(s.third.empty());
    check(file("2 0 9","2 1 8","1 2","0","END\nmore"),false);               // trailing input
    check(file("2 0 9","2 1 8","1 2","0",""),false);                        // no END
    check(file("2 0 9","2 1 8","1 2","0","END\n","MM3_REFERENCE_STATE 2\n"),false);
    check(file("2 0 0","2 1 8","1 2","0"),false);                           // an expert twice on the primary
    check(file("2 0 9","2 0 8","1 2","0"),false);                           // an expert on the primary and the helper
    check(file("2 0 9","2 1 8","1 9","0"),false);                           // an expert on the primary and the third card
    check(file("2 0 5","2 1 8","1 2","0"),false);                           // a layer without experts
    check(file("2 0 12","2 1 8","1 2","0"),false);                          // outside the model
    check(file("3 0 9","2 1 8","1 2","0"),false);                           // a count that the ids do not fill
    check(file("2 0 9","2 1 8","1 2","1 1 0 3.0"),false);                   // a score for a layer without experts
    check(file("2 0 9","2 1 8","1 2","2 0 0 1.0 0 0 2.0"),false);           // a score twice
    check(file("2 0 9","2 1 8","1 2","1 0 0 nan"),false);
    {   // another geometry than the model's
        {std::ofstream f(path,std::ios::binary);f<<file("2 0 9","2 1 8","1 2","0");}
        mm3::ReferenceState out;std::string error;
        assert(!mm3::read_reference_state(path.string(),moe,8,out,error));
        assert(!mm3::read_reference_state(path.string(),{1,1},4,out,error));
        assert(!mm3::read_reference_state((path.string()+".missing"),moe,4,out,error));
    }
    {   // a calibration report shorter than its stated length
        std::string text=file("2 0 9","2 1 8","1 2","0");
        const auto at=text.find("BATCH_CALIBRATION ");
        text.replace(at,std::string("BATCH_CALIBRATION ").size()+std::to_string(report.size()).size(),"BATCH_CALIBRATION 70000");
        check(text,false);
    }
    std::filesystem::remove(path);
    return 0;
}
