#include "../src/mm3/mm3_batch_staging.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

using namespace mm3;

static BatchStagingCosts costs() {
    BatchStagingCosts c; c.valid=true; c.h2d_ms=0.5;
    for(int n=1;n<=16;++n) {c.cpu_extra_ms[n]=n; c.gpu_extra_ms[n]=0.1*n; c.helper_extra_ms[n]=n;}
    return c;
}
static bool near(double a,double b) {return std::abs(a-b)<1e-9;}

int main() {
    std::array<int,128> uses{};uses[3]=1;uses[7]=4;uses[9]=1;uses[20]=2;
    auto c=costs();
    auto a=plan_batch_staging(uses,{3,7},{9},{20},{3,7},2,4,c);
    auto b=plan_batch_staging(uses,{7,3},{9},{20},{7,3},2,4,c);
    assert(a.valid && b.valid && a.selected==b.selected && a.selected==std::vector<int>{7});
    assert(near(a.critical_before_ms,5) && near(a.critical_after_ms,2)); // helper floor

    // Same distinct candidate count, different window reuse changes quantity.
    std::array<int,128> low{},high{};low[1]=1;low[2]=1;high[1]=4;high[2]=4;
    auto x=plan_batch_staging(low,{1,2},{},{},{1,2},2,4,c);
    auto y=plan_batch_staging(high,{1,2},{},{},{1,2},2,4,c);
    assert(x.valid && y.valid && x.selected.size()==1 && y.selected.size()==2);

    // Shared CPU weight reuse can make the lower-use expert more expensive.
    c=costs(); c.cpu_extra_ms[1]=5; c.cpu_extra_ms[4]=2;
    uses={};uses[1]=1;uses[2]=4;
    auto shared=plan_batch_staging(uses,{2,1},{},{},{2,1},1,4,c);
    assert(shared.valid && shared.selected==std::vector<int>{1});

    // One upload per distinct expert, independent of the number of columns.
    c=costs();c.h2d_ms=1.25;uses={};uses[4]=9;
    auto wide=plan_batch_staging(uses,{4},{},{},{4},1,16,c);
    assert(wide.valid && wide.selected==std::vector<int>{4});
    assert(near(wide.predicted_gpu_ms,1.25+c.gpu_extra_ms[9]));
    uses[4]=16;
    auto widest=plan_batch_staging(uses,{4},{},{},{4},1,16,c);
    assert(widest.valid && widest.selected==std::vector<int>{4});
    assert(near(widest.predicted_gpu_ms,1.25+c.gpu_extra_ms[16]));

    // Existing GPU and helper service floors make uploads useless; no tie upload.
    c=costs();uses={};uses[1]=1;uses[2]=1;uses[3]=1;
    c.gpu_extra_ms[1]=10;
    auto gpu_floor=plan_batch_staging(uses,{1},{2},{3},{1},1,1,c);
    assert(gpu_floor.valid && gpu_floor.selected.empty());
    c=costs();c.helper_base_ms=10;
    auto helper_floor=plan_batch_staging(uses,{1},{2},{3},{1},1,1,c);
    assert(helper_floor.valid && helper_floor.selected.empty());
    auto no_slot=plan_batch_staging(uses,{1},{2},{3},{1},0,1,c);
    assert(no_slot.valid && no_slot.selected.empty());

    // Identical marginal improvements choose the lower ID, regardless of input order.
    c=costs();uses={};uses[8]=2;uses[5]=2;
    auto tie=plan_batch_staging(uses,{8,5},{},{},{8,5},1,2,c);
    assert(tie.valid && tie.selected==std::vector<int>{5});

    // A fixed CPU batch overhead can make both uploads useful together while
    // either upload alone ties the current objective. Greedy intentionally stops.
    c=costs();c.cpu_base_ms=5;c.cpu_extra_ms[2]=0;uses={};uses[5]=2;uses[8]=2;
    auto local=plan_batch_staging(uses,{5,8},{},{},{5,8},2,2,c);
    assert(local.valid && local.selected.empty() && near(local.critical_after_ms,5));
    assert(2*(c.h2d_ms+c.gpu_extra_ms[2])<local.critical_after_ms);

    c=costs();c.valid=false;
    assert(!plan_batch_staging(uses,{8,5},{},{},{8,5},1,2,c).valid);
    c=costs();c.h2d_ms=std::numeric_limits<double>::quiet_NaN();
    assert(!plan_batch_staging(uses,{8,5},{},{},{8,5},1,2,c).valid);
    c=costs();c.cpu_extra_ms[2]=-1;
    assert(!plan_batch_staging(uses,{8,5},{},{},{8,5},1,2,c).valid);
    c=costs();
    assert(!plan_batch_staging(uses,{8,8},{},{},{8},1,2,c).valid);
    assert(!plan_batch_staging(uses,{8},{5},{},{5},1,2,c).valid);
    assert(!plan_batch_staging(uses,{8},{},{},{8},1,2,c).valid); // missing ID 5
    assert(!plan_batch_staging(uses,{8,5},{},{},{8,8},1,2,c).valid);
    assert(!plan_batch_staging(uses,{8,5},{},{},{8},-1,2,c).valid);
    assert(!plan_batch_staging(uses,{8,5},{},{},{8},1,1,c).valid);
    assert(!plan_batch_staging(uses,{8,5},{},{},{8},1,17,c).valid);
}
