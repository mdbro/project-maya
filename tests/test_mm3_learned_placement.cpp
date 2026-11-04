#include "../src/mm3/mm3_learned_placement.hpp"
#include <cassert>
#include <vector>
int main() {
    std::vector<int> p,h;
    // Global helper takes the two hottest records, including the one the old
    // primary-first selector would take. Primary gets the remainder.
    assert(mm3::helper_first_rosters({10,1,8,9},2,{0,0},{1,1},0,{1},2,2,p,h));
    assert((h==std::vector<int>{0,3}));
    assert((p==std::vector<int>{2}));
    // All equal: helper expert-major; primary layer-major among the rest.
    assert(mm3::helper_first_rosters({1,1,1,1},2,{0,0},{1,1},0,{1},2,2,p,h));
    assert((h==std::vector<int>{0,2}));
    assert((p==std::vector<int>{1}));
    // A hard per-layer cap can make the complement infeasible.
    assert(!mm3::helper_first_rosters({10,9,1,1},2,{0,0},{1,1},0,{2},2,1,p,h));
}
