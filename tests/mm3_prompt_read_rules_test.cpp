// CPU-only: the 3090's part of prompt_layerwise's refusal (src/mm3/mm3_prompt_read_rules.hpp). A 3090 swap in flight
// does not refuse the layer-by-layer read (the user's decision of 2026-11-13, built in the port onto minimax-wsl/local);
// the queued dispatch and a pending queued reader still do. The read itself beside a swap in flight is a GPU check
// (mm3_native_prompt_test / mm3_native_helper_live_cuda_test with a swap left in flight; the plan's test list).
//     g++ -std=c++17 -Isrc/mm3 tests/mm3_prompt_read_rules_test.cpp -o mm3_prompt_read_rules_test && ./mm3_prompt_read_rules_test
#include "mm3_prompt_read_rules.hpp"
#undef NDEBUG
#include <cassert>
#include <cstdio>

int main() {
    using mm3::prompt_read_helper_ok;
    assert(prompt_read_helper_ok(true, false, false));    // the worker dispatch, nothing in flight
    assert(prompt_read_helper_ok(true, false, true));     // ... and a swap in flight: read layer by layer (was refused)
    assert(!prompt_read_helper_ok(false, false, false));  // the queued dispatch: the chunks
    assert(!prompt_read_helper_ok(false, false, true));
    assert(!prompt_read_helper_ok(true, true, false));    // a queued 3090 reader still pending: the chunks
    assert(!prompt_read_helper_ok(true, true, true));
    std::printf("mm3_prompt_read_rules_test: ok\n");
    return 0;
}
