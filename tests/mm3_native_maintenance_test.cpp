#include <atomic>
#include <cassert>
#include <string>
using cudaError_t=int;
constexpr int cudaSuccess=0;
static int fail_operation=0,calls=0;
static int call(){return ++calls==fail_operation ? 1 : 0;}
static int cudaSetDevice(int){return call();}
static int cudaStreamSynchronize(int){return call();}
static const char* cudaGetErrorString(int){return "injected";}
struct Dev {int id=0,s=1,s2=2,cs=3;};
struct Model {
    bool compressed=true,placement_failed=false,drained=false,reset=false,promotion=false;
    std::atomic<bool> retirement_uncertain{false};
    bool compressed_host()const{return compressed;}
    bool scale_drain(std::string&){drained=true;return true;}
    bool scale_joined(Dev&,std::string&){reset=true;return true;}
    bool issue_promotions(Dev&,std::string&){promotion=true;return true;}
#include "../src/mm3/mm3_native_maintenance.inc"
};
int main(){
    for(int fail=1;fail<=4;++fail){Model m;Dev d;std::string err;calls=0;fail_operation=fail;
        assert(!m.scale_maintenance_join(d,err));assert(m.drained && !m.reset && m.placement_failed && m.retirement_uncertain && !err.empty());}
    {Model m;Dev d;std::string err;calls=0;fail_operation=0;
        assert(m.scale_maintenance_join(d,err));assert(m.drained && m.reset && !m.promotion && !m.retirement_uncertain);}
    {Model m;Dev d;std::string err;m.compressed=false;calls=0;fail_operation=0;
        assert(m.scale_maintenance_join(d,err));assert(m.promotion && m.reset);}
}
