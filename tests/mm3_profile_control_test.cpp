#include "../src/mm3/mm3_profile_control.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
int main() {
    std::vector<uint32_t> a{0x12345678,0,0xffffffffu}, decoded;
    assert(mm3::profile_hex(a)=="7856341200000000ffffffff");
    assert(mm3::parse_profile_hex(mm3::profile_hex(a),decoded) && a==decoded);
    assert(!mm3::parse_profile_hex("12345",decoded));
    assert(!mm3::parse_profile_hex("0000000z",decoded));
    std::vector<uint32_t> total;
    assert(mm3::generation_profile_accumulate({100,0xfffffffeu},{104,1},total));
    assert((total==std::vector<uint32_t>{4,3}));
    assert(mm3::generation_profile_accumulate({700,21},{701,23},total));
    assert((total==std::vector<uint32_t>{5,5}));
    auto before=total;
    assert(!mm3::generation_profile_accumulate({0,0},{UINT32_MAX,0},total));
    assert(total==before);
    assert(!mm3::generation_profile_accumulate({1},{2,3},total));
    assert(total==before);
}
