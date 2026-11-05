#include "mm3_replay_workload.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    auto path=std::filesystem::temp_directory_path()/
        ("mm3-replay-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto check=[&](const std::string& text,bool expected) {
        {std::ofstream f(path);f<<text;}
        mm3::ReplayWorkload out;std::string error;
        assert(mm3::read_replay_workload(path.string(),{3,4},128,8192,out,error)==expected);
        if(expected) {assert(out.batches.size()==1);assert(out.batches[0].layers[0].original[0]==0);}
        else assert(!error.empty());
    };
    const std::string header="MM3_REPLAY 1 0 2 1.3 0\nF 1 0 1\n";
    const std::string layers="L 3 0 1 2 3 4 5 6 7\nL 4 0 1 2 3 4 5 6 7\n";
    check(header+layers+"END\n",true);
    check("MM3_REPLAY 2 0 2 1.3 0 1\nF 1 0 1\n"+layers+"END\n",true);
    check("MM3_REPLAY 2 0 2 1.3 0 2\nF 1 0 1\n"+layers+"END\n",false);
    check(header+layers,false);
    check(header+layers+"END\nextra",false);
    check(header+"L 3 0 0 2 3 4 5 6 7\n"+"L 4 0 1 2 3 4 5 6 7\nEND",false);
    check("MM3_REPLAY 1 8 4 1.3 0\nF 1 12 1\n"+layers+"END",false);
    check("MM3_REPLAY 1 8 8 1.3 0\nF 1 0 1\n"+layers+"END",false);
    check(header+"L 4 0 1 2 3 4 5 6 7\nEND",false);
    std::filesystem::remove(path);
}
