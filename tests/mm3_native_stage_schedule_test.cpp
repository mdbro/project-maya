#include "mm3_native_stage_schedule.hpp"
#include <cassert>
#include <string>
#include <vector>

// The production scheduler calls these fake handoff operations. A failure at
// each submission boundary must prevent later consumers from seeing that slot.
struct Fake {
    std::vector<std::string> trace;
    int pos=100, completed=0, fail_at=-1, calls=0, skipped=0, discarded=0;
    bool keep=true, ready[2]{};
    bool step(std::string v) {trace.push_back(v); return calls++!=fail_at;}
    bool prefix(int slot,int start,int n) {
        if(!step("prefix:"+std::to_string(slot)+":"+std::to_string(pos+start)+":"+std::to_string(n)))return false;
        ready[slot]=true;return true;
    }
    bool await_a() {return step("await-a") && ready[0];}
    void begin_prefix_b() {}
    void submitted_prefix_b(int) {}
    bool observe_b_start() {return step("observe-start");}
    void begin_suffix_a() {}
    bool suffix(int slot,int start,int n) {
        if(!ready[slot] || !step("suffix:"+std::to_string(slot)+":"+std::to_string(pos+start)+":"+std::to_string(n)))return false;
        completed+=n; return true;
    }
    void end_suffix_a() {}
    bool observe_b_end() {return step("observe-end");}
    bool continue_b(int a) {assert(a==4);trace.push_back("decision");return keep;}
    void serial_prefix_b(int) {}
    void completed_b(int) {}
    void skipped_b(int n,bool overlap) {skipped+=n;if(overlap)discarded+=n;}
};
int main() {
    for(int chunk=-1;chunk<=3;++chunk)assert(!mm3::native_stage_chunk_supported(chunk));
    for(int chunk=4;chunk<=16;++chunk)assert(mm3::native_stage_chunk_supported(chunk));
    assert(!mm3::native_stage_chunk_supported(17));
    Fake overlap;
    assert(mm3::native_stage_schedule(8,true,overlap));
    assert((overlap.trace==std::vector<std::string>{"prefix:0:100:4","await-a","prefix:1:104:4","observe-start","suffix:0:100:4","observe-end","decision","suffix:1:104:4"}));
    assert(overlap.completed==8);
    Fake reject;reject.keep=false;
    assert(mm3::native_stage_schedule(7,true,reject));
    assert(reject.completed==4 && reject.skipped==3 && reject.discarded==3);
    assert(reject.trace.back()=="decision");
    Fake serial;serial.keep=false;
    assert(mm3::native_stage_schedule(7,false,serial));
    assert(serial.completed==4 && serial.skipped==3 && serial.discarded==0 && !serial.ready[1]);
    Fake one;assert(mm3::native_stage_schedule(1,true,one));assert(one.completed==1);
    for(int failure=0;failure<7;++failure) {
        Fake bad;bad.fail_at=failure;
        assert(!mm3::native_stage_schedule(8,true,bad));
        if(failure<4)assert(bad.completed==0);
        if(failure<=5)assert(bad.completed<=4);
    }
    Fake bad_size;assert(!mm3::native_stage_schedule(9,true,bad_size));assert(bad_size.trace.empty());
}
