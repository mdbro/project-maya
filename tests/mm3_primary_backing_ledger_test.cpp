#include "../src/mm3/mm3_primary_backing_ledger.hpp"
#include <cassert>
#include <cstdint>
#include <vector>
using namespace mm3::primary_backing_ledger;
int main() {
    uint8_t h[4]={1,2,3,4}, b[4]={5,6,7,8};
    const uint8_t* incoming=h;const uint8_t* victim=nullptr;
    uint8_t* in_back=nullptr;uint8_t* victim_back=b;
    assert(rotate(incoming,in_back,victim,victim_back));
    assert(!incoming && in_back==h && victim==b && !victim_back);
    assert(!rotate(incoming,in_back,victim,victim_back));
    assert(rotate(victim,victim_back,incoming,in_back));
    assert(incoming==h && !in_back && !victim && victim_back==b);
    // Disable after two exchanges: HOST owners in backing bank return to the
    // outside frames currently owned by primary backing, with bytes preserved.
    uint8_t bank[8]={9,8,7,6,5,4,3,2}, outside[8]={};
    const uint8_t *x=bank,*y=bank+4;
    assert(!repatriate({&x,&y},{outside},4));
    assert(x==bank && y==bank+4);
    assert(!repatriate({&x,&y},{outside,outside},4));
    assert(x==bank && y==bank+4);
    assert(repatriate({&x,&y},{outside,outside+4},4));
    assert(x==outside && y==outside+4);
    for(int i=0;i<8;++i)assert(bank[i]==outside[i]);
}
