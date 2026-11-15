// src/mm3/mm3_dspark.hpp - maya-mm3's DSpark draft (MiniMax-M3.1-Flash's dspark/: vLLM's MiniMaxM3DSparkForCausalLM).
//
// Context: the target's residual streams after layers 3, 17, 31, 45 and 59 of every token, concatenated, through fc
// and the hidden norm; each of the 5 draft layers keeps the K / V of that (its k / v projections, k normed and
// rotated at the token's position) for the last `window` positions, a ring.  A proposal: the block [anchor, mask x 6]
// at the positions after the context through the 5 layers (Gemma norms, per-head QK norm, partial RoPE, attention
// over the ring's window and the whole block - not causal -, a dense swigluoai MLP), the final norm, the target's
// head; then left to right each position's logits plus the Markov bias of the token before it (markov_w2 .
// markov_w1[prev], prev the anchor first), greedy, and its confidence sigmoid(proj . [hidden ; markov_w1[prev]] + b).
//
// The weights: the layers' MXFP8 matrices as stored (E4M3 + a UE8M0 scale every 32 - mx_rows reads them), fc / the
// Markov tables as stored (BF16; fc in FP16 on GPUs without BF16 tensor cores), the norms and the confidence head in
// F32, on one GPU with the target's head. DFlash2's projections may instead be NVFP4 (its pack's dflash2.weights; kernel
// item 6: mm3_draft_fp4.cuh on the 5090's FP4 tensor cores), and a request picks the projections' arithmetic (set_proj).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "mm3_exl3.hpp"
#include "mm3_head_lowp.cuh"
#include "mm3_kernels.cuh"
#include "mm3_pack.hpp"

namespace mm3 {

struct DraftGeo {
    int dim = 0, n_layers = 0, n_head = 0, n_kv = 0, hd = 0, n_rot = 0, ff = 0, vocab = 0, rank = 0;
    int block = 7, window = 4096, mask_id = 0;   // block: the most drafts a proposal makes
    std::vector<int> taps;   // the target layers whose output streams the context takes
    float eps = 1e-6f, alpha = 1.702f, limit = 7.0f, theta = 1e7f;
    // DFlash2 (load_dflash2) instead of DSpark: its block is the anchor and `block` mask columns; a dynamic grouped
    // convolution around each layer's attention and MLP (conv_g groups of D / conv_g channels); the candidate selector
    // over the target head's top sel_k (rank: the selector's rank) instead of the Markov and confidence heads
    bool dflash2 = false;
    int conv_g = 0, sel_k = 16;
};

class Draft {
public:
    Draft();
    ~Draft();
    Draft(const Draft&) = delete;
    Draft& operator=(const Draft&) = delete;

    /// Loads the draft (the pack's dspark.* tensors) onto device dev, sharing the target's head (on dev) and its
    /// embeddings (emb: the mapped BF16 table).  max_rows: the most context rows one call takes.
    bool load(const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas, const strata::mm3::exl3::Mat* head,
              const uint16_t* emb, int max_rows, std::string& err);
    /// Loads RadixArk's DFlash2 draft instead (step 6): dp, its own pack (tools/mm3/pack_dflash2.py: the projections in
    /// MXFP8, the rest BF16), checked against the target's pack pk; the same head, embeddings and context taps as
    /// DSpark's, and the same calls after. A proposal: the block [anchor, mask x block] through the 5 layers with each
    /// branch's dynamic convolution, the final norm, the target's head over the mask columns in one pass, each column's
    /// top sel_k, and the selector's conditional walk; conf: the selector's probability of each draft (temperature 1),
    /// q (sampled): its distribution over the column's candidates.
    bool load_dflash2(const Pack& dp, const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas,
                      const strata::mm3::exl3::Mat* head, const uint16_t* emb, int max_rows, std::string& err);
    /// Kernel item 10: the target head's low-precision copy (mm3_head_lowp.cuh) for this draft's head pass, or nullptr
    /// (the BF16 head, the built-in). Between proposals.
    void set_head_lowp(const mm3::head_lowp::Head* h);
    /// (DFlash2, kernel item 6) the other weight format's projections from a second DFlash2 pack of the same geometry
    /// (pack_dflash2.py with the other --weights), held beside the main pack's so that each request can choose its
    /// arithmetic (set_proj) in one engine start - for A/B tests; it costs that format's projection bytes. After
    /// load_dflash2.
    bool load_alt(const Pack& ap, std::string& err);
    /// The projections' arithmetic (kernel item 6), per request: 0 MXFP8 weights x F32 activations (mx_rows: today's),
    /// 1 MXFP8 weights x MXFP8 activations on the 5090's tensor cores (mm3_dense_fp8.cuh, as the target's dense
    /// matrices), 2 NVFP4 weights x NVFP4 activations on its FP4 tensor cores, 3 NVFP4 weights x two-term NVFP4
    /// activations (mm3_draft_fp4.cuh); -1 the built-in. Each changes the drafts (not the verify rule). False (err): the
    /// mode's weights are not loaded or its kernel does not run on this GPU (nothing changed). A change drops the
    /// retained context (its K / V were the old mode's), so the next request reads its prompt again.
    bool set_proj(int mode, std::string& err);
    int proj() const;
    /// The built-in mode: 0 when the drafter's main pack is MXFP8 (DSpark's always is), 3 when it is NVFP4.
    int proj_builtin() const;
    /// The modes this load can run, as bits (1 << mode).
    int proj_modes() const;
    /// (DFlash2) 1: a proposal's block through the layers, the head and the top-k replayed as one CUDA graph per draft
    /// count (captured on its second use), and the drafts returned by a kernel through mapped memory instead of the copy
    /// engine (which the expert cache's promotions keep busy); 0: today's launches. The drafts are the same bits either
    /// way. False (err): DSpark (no graph form).
    /// 2: the same, and every replay's candidates checked against the launches' (diagnostic: counts()).
    bool set_graph(int mode, std::string& err);
    int graph() const;
    /// (diagnostic, kernel item 6) 1: after each proposal in a mode other than 0, the same proposal again in mode 0 with
    /// the same draws, its drafts compared position by position (counts(); the drafts returned are the first's); 0 off.
    /// The second proposal attends to the same retained context, whose K / V the request's own mode projected: it
    /// measures the proposal block's arithmetic, not a whole mode-0 drafter. It doubles the draft time. False (err): no
    /// MXFP8 projections on this load.
    bool set_shadow(int on, std::string& err);
    /// Since reset_counts(): the draft graph's replays, draft_graph 2's checks and how many differed; draft_shadow's
    /// comparisons by position (offered[i]: proposals whose drafts before i agreed; agreed[i]: and draft i too).
    void counts(long long& replays, long long& checks, long long& mismatches, std::vector<long long>& offered,
                std::vector<long long>& agreed) const;
    void reset_counts();
    /// (DFlash2) bytes the drafter's device room changes by against an MXFP8 pack of dp's geometry: minus the saving of
    /// dp's NVFP4 projections (0 for an MXFP8 pack), plus the projections of the alternative pack alt (may be null).
    static long long d2_room_delta(const Pack& dp, const Pack* alt);
    /// (before load) the target's token rows from f(id, dst: D BF16 values) instead of emb - a GGUF's table, in its
    /// block type: the mask row once, each proposal's anchor row
    void set_emb_rows(std::function<void(int, uint16_t*)> f);
    const DraftGeo& geo() const;
    bool context_available(int pos) const;
    /// The context of positions pos0 .. pos0 + T - 1 from the taps (device-readable, [taps][stride][dim] F32: tap j's
    /// rows `stride` rows after tap j - 1's; 0: T).
    bool context(const float* taps, int T, int pos0, std::string& err, size_t stride = 0);
    /// The same in steps, T <= max_rows: tap j's rows (device [T][dim] F32) into the context - the taps in order, 0
    /// first - then the rest; on the draft's stream (a forward's taps can go in as its layers pass them).
    bool context_tap(int j, const float* rows, int T, std::string& err);
    bool context_finish(int T, int pos0, std::string& err);
    /// Proposes n (<= block) tokens after `anchor` at position pos (the context holds pos - 1 and before): drafts
    /// and their confidences (host).  logits (if given): each position's biased logits [n][vocab] (host), the
    /// distribution the drafts were taken from.
    bool propose(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, float* logits,
                 std::string& err);
    /// The same with each draft drawn from the draft's own distribution at its position - the biased logits through
    /// sa's temperature, top_k (<= k::kQMax), min_p and top_p, u[i] the draw (uniform in [0, 1)) - instead of its
    /// argmax; q[i]: that distribution ((id, probability) pairs, the most likely first), for the target's rejection
    /// sampling of the drafts.
    bool propose_sampled(int anchor, int pos, int n, const k::SampleArgs& sa, const std::vector<float>& u,
                         std::vector<int>& drafts, std::vector<float>& conf,
                         std::vector<std::vector<std::pair<int, float>>>& q, std::string& err);
    /// Device bytes the draft holds.
    size_t bytes() const;
    /// (DFlash2, diagnostic) parts of a proposal switched off, to find a fault: 1 the convolutions (identity), 4 the
    /// selector (each column's top candidate, q one-hot); 2 is spare (it chose the head path while a BF16 GEMM fed with
    /// rms's FP16 output was the fault, 2026-11-09). 0: none.
    void set_debug(int bits);
    /// The device bytes load() will hold for this pack and max_rows (its allocations, from the pack's geometry - the
    /// placement's room for the draft before it loads; 0: the pack has no draft).
    static size_t plan_bytes(const Pack& pk, int max_rows);

private:
    struct Impl;
    Impl* p;
};

}  // namespace mm3
