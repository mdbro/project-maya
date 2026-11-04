#include "../src/mm3/mm3_stage_reuse.hpp"
#include <cassert>
#include <vector>

int main() {
    // 9 is in the routes but is ineligible, so ranking must not add it.
    const int routes[] = {3, 9, 2, 1, 2, 9, 1, 4, 9, 4, 4};
    std::vector<int> eligible{3, 2, 1, 4};
    mm3::rank_stage_reuse(eligible, routes, sizeof(routes)/sizeof(routes[0]));
    assert((eligible == std::vector<int>{4, 2, 1, 3})); // 2 and 1 tie in first-seen order
    const size_t selected = 2; // caller retains its existing budget
    assert(eligible.size() == 4 && eligible[0] == 4 && eligible[1] == 2 && selected == 2);
    std::vector<int> empty;
    mm3::rank_stage_reuse(empty, routes, sizeof(routes)/sizeof(routes[0]));
    assert(empty.empty());
}
