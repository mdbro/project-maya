// MiniMax cache-conditioned router. Snapshot and buffers are owned by the caller.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace mm3::resident {
// Snapshot has one bit per expert, bit (e % 32) in word (e / 32). Set only
// after publication is complete and keep it and the corresponding allocations
// immutable/live until the routing pass and its consumers retire. If both bits
// are set, primary owns the hit for statistics. Missing/loading bits stay clear.
struct Ready { uint32_t primary[4], helper[4]; };
enum class Policy : int { Off = 0, Substitute = 1, Prior = 2 };
// Four cumulative device uint64 counters, caller initializes/resets and reads
// after stream completion. Replacements count executed IDs absent from the
// original top-four set, not rank permutations. Hits count executed selections; primary takes
// precedence when both tiers are ready. original_misses counts unready original
// top-four selections, including protected ranks.
struct Stats { unsigned long long replacements, primary_hits, helper_hits, original_misses; };
// logits [T,128], bias [128], ids/original [T,4], weights [T,4], all device.
// logits and bias must be finite. Selection ranks sigmoid(logit)+bias (ties by
// smaller ID); weights are unbiased sigmoid, normalized over the executed set by default.
// With dampen and Substitute or Prior, divide by the original top-four sigmoid sum;
// there is no second normalization or ratio cap (bias can yield a ratio > 1).
// The caller applies routed_scaling_factor separately. Policy Off leaves the
// original top four exactly selected. Substitute may replace ranks 2 and 3,
// only if the original is unready, with an unselected ready expert in the
// original top eight. Search follows original biased rank. The candidate's
// unbiased sigmoid must be >= min_ratio * the original's unbiased sigmoid;
// this positive score domain stays well defined for negative corrected scores.
// A zero sigmoid original cannot be replaced (ratios are undefined).
// Prior selects all four ranks from all 128 experts by sigmoid(logit)+bias plus
// parameter for ready primary/helper experts. It does not apply substitution;
// any rank may change. Parameter is the additive score bonus in [0,1], not a
// probability or quality-qualified default. A zero bonus and an empty ready
// snapshot each produce the same IDs and weights as Off. In both policies the
// weights use the selected experts' original sigmoid probabilities. The shared
// parameter is the minimum ratio for Substitute, the additive bonus for Prior.
// stats/ready may be null when policy is Off; Prior and Substitute require a
// ready snapshot. original and stats may be null in any mode.
// Returns cudaErrorInvalidValue for invalid host arguments; otherwise launch status.
// Sanitize invalid executed IDs before any device plan indexing, retaining a
// sticky flag for the caller to check after the existing stream join.
cudaError_t guard_expert_ids(int* ids, int count, int* bad, cudaStream_t stream);
cudaError_t count_original(const int* ids, int count, unsigned* histogram, cudaStream_t stream);
cudaError_t route(const float* logits, const float* bias, int T, Policy policy, float parameter,
                  const Ready* ready, int* ids, float* weights, int* original, Stats* stats, cudaStream_t stream, bool dampen = false,
                  float* capture_scores = nullptr);
}
