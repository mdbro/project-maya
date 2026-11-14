// src/mm3/mm3_prompt_read_rules.hpp - the 3090's part of prompt_layerwise's refusal (mm3_native_prompt.inc,
// prompt_read_refusal): the layer-by-layer read feeds the 3090 through the worker dispatch, and not while a queued 3090
// reader is still pending. A 3090 swap in flight does not refuse it (the user's decision of 2026-11-13, CLAUDE.md's
// decision table, Placement: "Reader runs alongside, in the port"): the read moves the swap forward at its start and at
// its end (helper_live_progress), it starts none (helper_live_issue waits for generation), and nothing changes the
// 3090's experts or their host records inside the read, so its uploads and 3090 columns see one consistent state - as
// a decode window does between two progress calls. Refused before: 3.6 times slower reads of a follow-up prompt that
// arrived while the last request's final swap was still in flight (swap-policy.md, section 3, bug 1).
#pragma once

namespace mm3 {

/// Whether the 3090 lets a prompt be read layer by layer: the worker dispatch, and no queued 3090 reader pending.
/// swap_in_flight is taken and does not refuse (the decision above; kept as an argument so the rule says it).
inline bool prompt_read_helper_ok(bool worker_dispatch, bool queue_pending, bool swap_in_flight) {
    (void) swap_in_flight;
    return worker_dispatch && !queue_pending;
}

}  // namespace mm3
