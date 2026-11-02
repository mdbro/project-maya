#include "../src/mm3/mm3_draft_ring.hpp"
#include <cassert>
#include <cstdio>

int main() {
    mm3::DraftRing ring;
    ring.reset(12, 8);
    assert(ring.available(0));
    assert(!ring.available(1));
    ring.context(0, 8);
    assert(ring.available(8));
    // A proposal writes all block rows, then verification replaces only the
    // committed prefix. The next proposal must see the actual target rows.
    ring.proposal(8, 3);
    ring.context(8, 3);
    assert(ring.available(9));
    ring.proposal(9, 3);
    ring.context(9, 1); // rejection at the first proposed token
    assert(ring.available(10));
    ring.context(10, 10); // wrap the ring while continuing
    assert(ring.available(20));
    assert(!ring.available(5)); // matching target prefix is no longer enough
    ring.context(0, 6); // target replay rebuilds the draft taps
    assert(ring.available(6));
    // Both older and newer logical rows cannot share a physical slot.
    assert(!ring.available(20));
    ring.context(6, 14);
    assert(ring.available(20));
    ring.proposal(20, 6); // block can evict context even before target writes
    assert(!ring.available(20));
    assert(!ring.available(-1));
    std::puts("draft ring logical positions: ok");
}
