#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include "../src/mm3/mm3_resident_loader.hpp"
#include <atomic>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <string>
using cudaError_t=int;
constexpr int cudaSuccess=0;
static int fail_operation=0,calls=0;
static int call(){return ++calls==fail_operation ? 1 : 0;}
static int cudaSetDevice(int){return call();}
static int cudaStreamSynchronize(int){return call();}
static const char* cudaGetErrorString(int){return "injected";}
struct Dev {
    struct Promo {int slot;};
    int id=0,s=1,s2=2,cs=3,xs=0;   // (xs: exchange_schedule's download stream, absent unless scheduling was used)
    std::vector<Promo> promo_q;
    std::vector<int> slot_iss=std::vector<int>(1,0);
};
struct Model {
    struct {int native_stage_prefix=0;} o;
    bool native=true,compressed=true,placement_failed=false,drained=false,reset=false,promotion=false;
    std::atomic<bool> retirement_uncertain{false};
    int fail_promotion=0;bool copy_submitted=false,event_attempted=false;
    std::string promotion_error;
    bool compressed_host()const{return compressed;}
    bool scale_drain(std::string&){drained=true;return true;}
    bool scale_joined(Dev&,std::string&){reset=true;return true;}
    bool exchange_downloads_join(Dev&,std::string&){return true;}
    bool issue_promotions(Dev& d,std::string& err){
        promotion=true;
        // Production issue transition; inject successful H2D followed by an
        // event-record failure or exception before copy-stream retirement.
        return mm3::r2::issue(d.promo_q,d.slot_iss,promotion_error,err,
            [&](const Dev::Promo&,std::string& error){
                copy_submitted=true;event_attempted=true;
                if(fail_promotion==2)throw std::runtime_error("event record threw");
                if(fail_promotion==1){error="event record failed";return false;}
                return true;
            });
    }
#include "../src/mm3/mm3_native_maintenance.inc"
};
int main(){
    for(int fail=1;fail<=4;++fail){Model m;Dev d;std::string err;m.compressed=false;m.o.native_stage_prefix=3;calls=0;fail_operation=fail;
        assert(!m.scale_maintenance_join(d,err));assert(m.drained && !m.reset && m.placement_failed && m.retirement_uncertain && !err.empty());
        fail_operation=0;assert(m.scale_maintenance_join(d,err));assert(m.placement_failed && m.retirement_uncertain);}
    for(int fail=1;fail<=4;++fail){Model m;Dev d;std::string err;m.compressed=false;calls=0;fail_operation=fail;
        assert(!m.scale_maintenance_join(d,err));assert(m.drained && !m.reset && m.placement_failed && m.retirement_uncertain && !err.empty());
        fail_operation=0;assert(m.scale_maintenance_join(d,err));assert(m.placement_failed && m.retirement_uncertain);}
    for(int fail=1;fail<=4;++fail){Model m;Dev d;std::string err;calls=0;fail_operation=fail;
        assert(!m.scale_maintenance_join(d,err));assert(m.drained && !m.reset && m.placement_failed && m.retirement_uncertain && !err.empty());}
    for(int fault:{1,2}) {
        auto* m=new Model;Dev d;std::string err;m->compressed=false;m->fail_promotion=fault;
        d.promo_q.push_back({0});calls=0;fail_operation=0;bool threw=false;
        try {assert(!m->scale_maintenance_join(d,err));}
        catch(const std::runtime_error&){threw=true;}
        assert(threw==(fault==2) && m->copy_submitted && m->event_attempted);
        assert(m->retirement_uncertain.load() && !m->reset && calls==3);
        // Even an independently successful later copy-stream join cannot
        // clear the sticky owner flag or permit destruction.
        assert(cudaStreamSynchronize(d.cs)==cudaSuccess);
        bool disposed=false,retained=false;int retirement_joins=0;
        mm3::r2::destroy_after_retire(m,[&]{
            if(!mm3::r2::retirement_may_begin(m->retirement_uncertain))return false;
            ++retirement_joins;return true;
        },[&](Model* p){disposed=true;delete p;},[&]{retained=true;});
        assert(retained && !disposed && retirement_joins==0 && m->retirement_uncertain.load());
        delete m; // fake fixture has no GPU owner
    }
    {Model m;Dev d;std::string err;calls=0;fail_operation=0;
        assert(m.scale_maintenance_join(d,err));assert(m.drained && m.reset && !m.promotion && !m.retirement_uncertain);}
    {Model m;Dev d;std::string err;m.compressed=false;calls=0;fail_operation=0;
        assert(m.scale_maintenance_join(d,err));assert(m.promotion && m.reset);}
}
