# Porting DFlash2 to Maya at MXFP8: a study of Infernix's implementation

2026-11-09, step 6 (tasks 56-57). A sub-agent's read-only study of Infernix's DFlash2 code and records, compared with
Maya's DSpark path, recorded here for the port. Nothing was built or run on a GPU. Its working notes and a read-only
Nsight query script are in `/mnt/wsl/transfer/minimax-wsl/research/dflash2-port-20261109/`. **[inferred]** marks
conclusions derived rather than read in code or records.

Path prefixes:
- **IM** = `/home/mdwbr/Infernix-minimax`, the most complete DFlash2 code; its key files are identical to
  `Infernix-glm5-local`.
- **IP** = `/home/mdwbr/Infernix-dflash-placement`, the isolated "P51" trial that ran the draft layers on the 3090.
- **IX** = `/home/mdwbr/Infernix`, which holds P51's records.
- **MAYA** = this repository.

## Summary

1. **Which variant to port.** Infernix runs DFlash2 in two engines.
   - `qwen3_5` serves the 27B/35B/122B targets and captures a whole round as one CUDA graph: context, proposal and target
     verify.
   - `qwen4_exp` serves Flash-Next, an MoE with offloaded experts, and is **the closest analog to Maya**. It runs a
     proposal graph, a host wait, the verify, then a catch-up graph that writes the committed columns' context.
   - `glm5_next` has no DFlash2. No Infernix checkout has DFlash2 for MiniMax; IM's `src/models/minimax` only wraps
     Maya's DSpark.
2. **Maya already contains a DFlash2, for GLM** (`src/core/glm_dflash.cu`, 2,050 lines). It uses F32 activations, eager
   launches, a greedy selector walk, a conv kernel, top-k and a confidence cut. It is an oracle and kernel source, not the
   fast design.
3. **Infernix's recorded speed.** Flash-Next's DFlash2 with 7 drafts on the 5090, medians over 110 rounds:
   - the proposal graph takes **0.677 ms** over 85 kernels: five layers 0.475, embedding 0.016, the final norm, head and
     selector 0.187;
   - context materialization takes 0.077 ms;
   - about 0.67 GB is read per proposal.
4. **Estimate for MiniMax's DFlash2 at MXFP8 on the 5090 [inferred].** A proposal reads about 3.1-4.4 GB: the five
   layers are 1.82-1.91 GB, and the head is either a private MXFP8 copy (1.27 GB) or the target's shared BF16 head
   (2.46 GB).
   - That is about **2.2-2.6 ms** with an MXFP8 head copy, or **3.0-3.4 ms** with the shared BF16 head.
   - The context catch-up adds 0.15-0.3 ms a round.
5. **Why DSpark is slow [inferred from code].**
   - Maya's native BF16 head kernel `bf16_mv` launches a grid of (rows/4, columns). It re-reads all 2.46 GB of the head
     **once per draft column** (`mm3_native_dense.cu:29-38,76`): about 5.5 of the 9.7 ms at four drafts.
   - DSpark's Markov step adds, sequentially per position, a full-vocabulary bias (a 102 MB read) and an argmax or
     sample (`mm3_dspark.cu:436-456`).
   - Every kernel is launched eagerly, and the proposal ends in a stream sync.
   - **The target's own verify window uses the same head path**, about 1.37 ms per row (`mm3_model.cu:4457-4472`).
   - One run with `MAYA_MM3_PROF=2` would confirm the split: it prints layers, head and Markov ms per proposal.
6. **What makes Infernix fast.**
   - Each weight is read once per step for all 8 columns.
   - Q/K/V is one GEMM, and SwiGLU runs in the gate/up GEMM's epilogue.
   - The conv's output side and the residual add follow each projection in one small kernel.
   - The head GEMM is fused with top-16 selection, so no logits are written.
   - The selector's 16×16 lattice is computed in parallel, then walked by one warp in 5.8 µs.
   - One fused kernel writes all five layers' context K/V.
   - There is one CUDA graph per batch size, with programmatic dependent launch (PDL) on sm_90 and newer.
7. **Verification.**
   - Chain by default. Greedy rows take the longest prefix matching the target.
   - Sampled rows use exact rejection sampling with the selector's 16-candidate q: accept with min(1, p/q), and on
     rejection draw from max(p−q, 0).
   - Trees are optional.
8. **Acceptance prediction exists in Infernix.**
   - Expected tokens = 1 + Σⱼ Πᵢ≤ⱼ a(pᵢ), where pᵢ is the selector's probability of draft i and a(·) a per-request
     calibration over 10 probability bins. It sets the verify length or width.
   - The DFlash2 checkpoint has **no confidence head**. See also `acceptance-prediction-research.md`.
9. **Weight formats.**
   - Infernix stores the draft's projections as int8 with an FP16 scale every 32 (`Q8_G32_FP16`).
   - The conv projections, selector and codebooks stay BF16.
   - **Infernix has no MXFP8 draft path.** The Q8 kernels map directly onto MXFP8, because both feed exact codes as BF16
     into the matrix units and apply the scale in FP32 per 32-group.
10. **The 3090 can run the draft, but should not come first.**
    - In P51 the draft layers ran 1.7-1.8× slower on the 3090. It has no FP8 tensor-core MMA, and its link here is PCIe
      4.0 ×4.
    - Drafting sits on the critical path between verify steps.
11. **Economics in Maya.** On 2026-11-07 an extra verified row cost about 20-50 ms, against 3-10 ms of drafting. So
    pricing the width per position dominates. Infernix's offloaded-MoE engines show the same:
    - Flash-Next: verifying all 7 drafts was 18-24% slower than its MTP baseline; the calibrated length was 8-11% faster.
    - Qwen3.5-122B: DFlash2 never paid there, so automatic mode drafts nothing.

## The draft model

Config read from Hugging Face:

| Field | Value |
|---|---|
| Heads | 64 query, 4 KV, `head_dim` 128 |
| RoPE | `rotary_dim` 64 (partial 0.5), θ 1e7 |
| Attention | `sliding_window` 4,096 on all five layers |
| Norms | `use_gemma_norm: true` (1+w), `qk_norm_type: per_head` |
| MLP | `swigluoai`, α 1.702, limit 7 |
| Block | `block_size` 8, `mask_token_id` 200058 |
| Selector | rank 256, top-16 |
| Conv | group 16, kernel 2 |
| Target layers | [3, 17, 31, 45, 59] |
| Vocabulary | 200,064 |

**Tensor inventory [inferred; it matches the card's 2,054,677,760 parameters exactly].** Each layer has 15 tensors:

| Tensor | Shape |
|---|---|
| q | 8192×6144 |
| k, v | 512×6144 each |
| o | 6144×8192 |
| gate, up | 12288×6144 each |
| down | 6144×12288 |
| attention and MLP conv projections | 1536×6144 each |
| two conv base kernels | 6144×2×2 each |
| input and post-attention norms | 6144 each |
| q and k head norms | 128 each |

That is 352.38 M parameters a layer. The other six tensors:
- fc, 6144×30720;
- the hidden (context) norm and the final norm;
- the selector's hidden projection, 256×6144;
- two codebooks, 200064×256 each.

There is no embedding and no head: the draft uses the target's. The exact tensor names need the safetensors header.

**What one step reads at MXFP8.** The linears are MXFP8; the conv projections, selector and codebooks are BF16.

| Item | When read | MB |
|---|---|---:|
| Layer linears | every proposal | 1,719 |
| Conv projections, BF16 | every proposal | 189 |
| Head: MXFP8 copy / shared BF16 | every proposal | 1,268 / 2,459 |
| Ring reads at a full window | every proposal | 42 |
| fc, MXFP8 | once a round, for the context | 195 |
| Context K/V | once a round | 32 |

## Infernix's draft forward (IM `src/models/qwen4_exp/execution/dflash2.cpp:161-253`; math in `docs/maintainer/dflash.md:135-212`)

```
prepare_masked_block (ids = [anchor, MASK×7], positions F..F+7, on the device)
embedding (the target's table)
per layer, ×5:
  conv prepare: input norm → BF16 split-K delta projection [4G, H] → reduce (conv side 0 + delta for side 1)
  Q/K/V: one Q8 GEMM over the fused [q;k;v] matrix, output split three ways
  rmsnorm_rope: per-head q/k norm and RoPE
  sliding_window_attention: split-KV over the ring + the block, then a reduce
  linear_dynamic_grouped_conv_add: o-projection GEMM → finish kernel (conv side 1 + residual add)
  conv prepare for the MLP
  linear_swiglu: Q8 gate/up GEMM with SwiGLU in the epilogue
  linear_dynamic_grouped_conv_add: down GEMM → conv side 1 + residual add
rmsnorm_pack_tail (final norm of the 7 mask columns)
linear_topk (head GEMM fused with stable top-16)
selector hidden projection (H → 256, BF16 GEMV)
candidate_selector_path (lattice + walk → drafts, q[16], confidence)
```

- **Fixed shapes.** Every fused kernel is registered for fixed shapes (H 5120/2560/3072, 32/8 heads, a 1,024-row
  context K/V, a capacity-2048 ring, plain norms, full-head RoPE). For MiniMax they are templates to adapt: H 6144,
  64/4 heads (group 16), window 4,096, gemma norms, RoPE on the first 64 dims.
- **Positions in parallel.** The block is always 8 columns. Attention is non-causal and the conv mixes neighbours, so
  each layer is one weight pass over all columns. The only sequential step is the selector's one-warp walk.
- **Attention.**
  - Each draft layer keeps a per-request cyclic ring (BF16 K, FP16 V; position p at slot p mod S).
  - The block's own K/V are a temporary segment, not written into the ring.
  - The kernel is split-KV flash-decoding: up to 32 splits, key blocks of 32, then a reduce.
- **Context** (`dflash.md:43-83`):
  - s is the concatenated output residuals of the target layers, and c = rmsnorm(W_fc · s);
  - per draft layer, K = rope(head_norm(W_k · c)) and V = W_v · c, reusing the block's k/v weights.
  - **Maya's DSpark taps the outputs of layers 3, 17, 31, 45 and 59, as SGLang's MiniMax M3 captures them**, so
    DFlash2 can reuse those taps [inferred]. The first acceptance number confirms it, since an off-by-one would collapse
    acceptance.
- **The conv (group 16, kernel 2)**, with G = H/16 = 384 for MiniMax:
  - d[s,t,g] = W_delta[(2s+t)G+g] · n, where W_delta is BF16 [1536, 6144] and n is the normed input.
  - Input side: prepared[h,i] = (base[h,0,0] + d[0,0,g,i])·n[h,i] + 1{i>0}·(base[h,1,0] + d[0,1,g,i])·n[h,i−1].
  - Output side: x[h,i] += (base[h,0,1] + d[1,0,g,i])·z[h,i] + 1{i>0}·(base[h,1,1] + d[1,1,g,i])·z[h,i−1], where z is the
    attention or MLP output.
- **The selector (rank 256, top-16)**, for mask column i:
  - u = the head's top-16 logits, and g = W_hidden · h (W_hidden is [256, H]).
  - E[p,c] = u[c] + Σᵣ W_pred[pred, r] · g[r] · W_succ[C[c], r].
  - The walk goes left to right along the chosen predecessor: greedy argmax, or sampled from softmax(E/T).
  - Outputs: the draft, q[16] (kept for verification) and the confidence, the temperature-1 softmax at the chosen rank.

## What makes it fast: the recorded breakdown

Flash-Next's DFlash2 (H 2560, int8 weights, a 131k-row 4-bit proposal head) on the 5090:

| Part | µs |
|---|---:|
| Proposal graph (85 kernels) | **677** |
| 20 int8 GEMMs (per layer: QKV 9.8, o 14.1, gate/up 28.8, down 20.3) | 343.5 |
| Head GEMM + top-16 merges | 146.2 + 25.6 |
| Attention, 5 × (partial + reduce) | 56.0 + 22.6 |
| Conv prepare, 10 × (partial + reduce) | 40.8 + 21.2 |
| Selector: projection, lattice, walk | 6.4 + 2.6 + 5.8 |
| Context (separate graph) | 76.5 |

On Flash-Next, plain rounds take 8.4 ms. With 2-7 verified drafts a round takes 14.2-27.7 ms: about **2.6 ms per draft
column** plus 0.6-1 ms of proposal.

Against Maya's DSpark (`mm3_dspark.cu:385-472`; drafting measured 9.6-9.7 ms a round at 4 drafts):

| | Infernix DFlash2 | Maya DSpark |
|---|---|---|
| Head | one fused GEMM + top-16 for all columns | the BF16 head re-read per column (≈ 1.37 ms per column [inferred]) |
| Between positions | a 16×16 lattice + a 5.8 µs walk | a full-vocabulary Markov bias + argmax per position, sequential |
| Projections | fused Q/K/V and gate/up; the conv in the epilogue | separate `mx_rows` calls; F32 activations |
| Launches | one graph with PDL | about 60 eager launches + a stream sync |
| Context | fc + one fused K/V kernel in a graph | 5 cuBLAS fc GEMMs per tap + a k and a v `mx_rows` per layer |
| Draft probability | the selector's softmax, free | a learned confidence head |

## Verification, and acceptance prediction

- **Chain.** The verify input is [anchor, d₁..d_P].
  - Greedy rows take the prefix matching the target's argmax.
  - Sampled rows accept with min(1, p/q) using the stored 16-slot q, draw from max(p−q, 0) on rejection, and draw a
    bonus if every draft is accepted.
  - Every verify column's tapped residuals are captured, and only committed columns enter the draft's ring.
- **Trees** are optional: best-first from the lattice, at most 32 columns and 8 paths. On a 27B with 7 drafts they gained
  6-11%. In Maya each tree column brings its own expert misses, so trees come last [inferred].
- **Infernix's acceptance prediction.**
  - `DraftCalibration` (IM `src/runtime/speculative/draft_length.h`) keeps 10 bins of conditional acceptance, starting
    at each bin's centre. After each verify it updates drafts 0..accepted with weight 0.1, separately per request.
  - Expected tokens = 1 + Σⱼ Πᵢ≤ⱼ a(pᵢ).
  - Flash-Next verifies the k ≤ 7 that maximizes expected tokens / (1 + 0.1 + 0.31k), plus one extra draft every 8
    rounds to keep measuring longer lengths: +8-11% over its MTP baseline.
  - The tree-width controller uses only history and makes no per-step prediction.
  - The 122B's MTP confidence policy masks columns past the chosen extent in the MoE (`MoeColumnMask`): a masked column
    triggers no dispatch, job, miss or prefetch.
- **What a Maya planner can use before the verify forward:**
  - pᵢ for each draft;
  - q over the 16 candidates and its ids (entropy and margin follow);
  - the calibrated aᵢ, and the reach Rⱼ = Πᵢ≤ⱼ aᵢ.

  An expert needed by a set of columns matters with probability R at the earliest of them. Two exact uses follow:
  - choose the width by expected tokens over predicted completion time;
  - drop tail columns at a layer boundary when their expected tokens are worth less than the expert work only they need.
    Causal attention and per-column routing leave earlier columns unchanged [inferred].

## Weight formats

- **Infernix.** Flash-Next requires `Q8_G32_FP16` for fc, Q/K/V, context K/V, o, gate/up and down. The conv projections,
  selector, codebooks, norms and base kernels are BF16. There is no FP8 or MXFP8 draft-weight path.
- **MXFP8 is a small change [inferred].** E4M3 codes are exact in BF16, and the E8M0 scale is a power of two applied per
  32-group, as the int8 kernels apply their FP16 scale. Maya's `mx_rows` already decodes E4M3.
- **Quantizing.** Use scale = 2^ceil(log2(amax/448)) and round to nearest even [inferred choice]. BF16 against MXFP8 is an
  acceptance question only, under exact verification.

## The 3090

- P51 built 13 draft kernel sources for sm_86, leaving the head and selector on the 5090. The draft layers took 0.80-0.88
  ms there against 0.47-0.50 ms on the 5090, and cross-device round trips took 0.155-0.278 ms.
- Two precision defects were fixed in IP only:
  - the conv prepare's normalized input is kept as BF16 high + low parts;
  - the conv-add keeps its projection in FP32.
- P51 is held and unfinished.
- **For MiniMax [inferred].** 1.8-1.9 GB of layers at 936 GB/s is at least 2 ms, realistically 2.5-3 ms, adding about
  1.5-2 ms a round on the critical path. It would free about 1.9 GB of the 5090 (about 60 experts), but take the same
  from the 3090's helper pool. That is a later trade for the planner.

## Porting plan

| Piece | Reuse from Maya | Port or adapt from Infernix |
|---|---|---|
| Target capture | DSpark's taps (`mm3_model.cu:4415-4423`); context after each forward | — |
| Context | DSpark's `context_tap` / `context_finish`, `DraftRing`, prefix reuse | later: the fused `context_kv_materialize` in a catch-up graph |
| Layer kernels | `k::rms` (1+w preloaded), `mx_rows` (MXFP8, ≤ 16 rows), `qk_norm_rope` (RoPE on 64 dims), `window_attn`, `swigluoai`, `draft_block` | later: fused MXFP8 Q/K/V and gate/up after Infernix's `q8_dflash2_*`, the conv in epilogues |
| Conv | GLM's `conv_kernel` as reference | `dynamic_grouped_conv.h` math; IP's precision choices, or F32 intermediates |
| Head and top-16 | `topk_rows`; GLM's top-k kernels | `linear_topk` for 200064×6144 |
| Selector | GLM's `walk_kernel` as an oracle | lattice + walk + `draw_rank` + `write_confidence`, almost verbatim |
| Acceptance | `mm3_main.cpp:1359-1430` already takes per-position (id, q) pairs and residual draws; windows up to 16 rows | the sparse-acceptance semantics already match |
| Length policy | DSpark's floor and survival cut | `DraftCalibration` + `calibrated_draft_length`, with the planner's column cost |
| Packing | `tools/mm3/pack_native.py`'s draft tensors | new: a BF16 → MXFP8 converter |

**Draft-step time and VRAM [inferred].**
- The draft layers take about 1.5 ms with today's kernels plus a multi-column head pass, or 1.25-1.35 ms fused.
- The head takes 0.85-0.9 ms as an MXFP8 copy, or 1.6-1.7 ms shared in BF16.
- A proposal is thus about 2.5-3.3 ms now, and about 2.2-2.4 ms fused with an MXFP8 head copy.
- **A multi-column head pass is mandatory.** Reading the head once per column at 7 columns would alone cost about 9.6 ms.
- The weights take about 2.31 GB, about 2.4 GB in total with the ring and scratch, the same as DSpark's recorded 2.37 GB.
  An MXFP8 head copy adds 1.27 GB, about 40 experts. Load one draft model at a time.

**Why width choice comes first in Maya.** Verify forwards on 2026-11-07, by row count:

| Rows | ms |
|---:|---:|
| 1 | 74-81 |
| 2 | 90-114 |
| 3 | 107-149 |
| 4 | 130-170 |
| 5 | 165-224 |

Each extra row cost about 20-50 ms. With a constant per-position acceptance fitted to the card (3.6, 4.3, 5.2 tokens a
step correspond to a ≈ 0.75, 0.81, 0.87) and a round of 80 + 30k + 3 ms, the best k at a = 0.81 is 2-3, about 17.3
tok/s, against 14.7 tok/s verifying all 7. So DFlash2's gain comes from four places: higher acceptance per position,
about 7 ms less drafting a round, calibrated per-position probabilities to set the width, and the planner's pricing of
each column's expert work.

**Milestones.**
- **M0 (optional, no engine change).** A CPU reference, `tools/mm3/ref_dflash2.py`, run on `MM3_DUMP_HS` dumps. It checks
  the capture convention and the gemma norms, gives an FP64 oracle for the CUDA code, and gives a teacher-forced
  acceptance estimate.
- **M1, the first acceptance number in Maya.** A `Draft2` class with DSpark's interface (load, context, propose and
  propose_sampled, returning drafts, confidences and q pairs):
  - pack the MXFP8 linears and fc, keeping the conv projections, selector, codebooks and norms in BF16 (+1 preloaded);
  - context through DSpark's path;
  - per layer: rms → conv projection → **a new conv side-0 kernel** → q/k/v → `qk_norm_rope` → `window_attn` →
    o-projection → **a new conv side-1 + residual kernel** → the same for the MLP with `swigluoai`;
  - after the layers: the final rms → **a multi-column head pass** → top-16 → the selector projection → **a new walk
    kernel** (greedy and sampled, writing q and the confidence) → one copy to the host;
  - a flag beside `--dspark` at the same call sites;
  - measured with today's DSpark accounting, on the same prompts at fixed widths 4 and 7, exact and lenient.
- **M2, speed.** Graphs, fused MXFP8 Q/K/V and gate/up, the conv in epilogues, a fused head + top-16, the lattice
  selector, the fused context kernel, PDL on the 5090.
- **M3, planner.** Port the calibration, and give the planner pᵢ, aᵢ and Rⱼ per column: first the width, then tail
  columns dropped at layer boundaries.
- **M4, later.** Trees, a reduced-vocabulary head, draft layers on the 3090, and the multi-column head for DSpark too, a
  cheap fix for the baseline.

**Risks.**
1. The head kernel must be multi-column. The target's verify head has the same per-row re-read, and fixing it must keep
   identical bits per row.
2. Verifying all 7 drafts can be slower than DSpark at 4: judge DFlash2 with per-position data, not one fixed width.
3. Infernix's fused kernels have fixed shapes: H, head counts, window, RoPE and norm style all differ.
4. Numerics: MXFP8 costs only acceptance; any BF16-activation fused port needs IP's two precision fixes.
5. Unknowns to resolve before coding:
   - the tensor names and the base/projection layouts;
   - whether every norm, q/k included, is gemma-style;
   - whether the head has padding rows to exclude from top-16;
   - the capture convention.
6. Comparability: the card used SGLang on TP4 GB300 with an NVFP4 target and a 16K context; Maya runs an 8K context.
7. VRAM (above).
8. Graph re-capture already cost about 0.33 ms a token with drafting.

## Open checks

- ~~DSpark's head share of its 9.7 ms.~~ *Settled by f18bfdd's multi-column head pass: DSpark's 7-draft call fell from
  about 15 ms to 7.3 ms ([evidence](evidence.md#drafter-acceptance)).*
- ~~The capture convention and the gemma norms.~~ *Settled by M1: DFlash2 drafts at DSpark's level of acceptance, so the
  taps and norms are right.*
- ~~The safetensors header.~~ *Read for the pack (`tools/mm3/pack_dflash2.py`).*
- MXFP8 against BF16 acceptance: **not measured.** M1 has no BF16 pack, and the DSpark comparison differs by drafter, not
  precision.
- `mx_rows` bandwidth at 8 rows, measured so far only at one token: M2's drafting profile (`dflash2-prof-20261109`).

## Status, 2026-11-09

- **M1 is done** (c8c9eba, fixed in f18bfdd: `rms` writes FP16, which the first form's head and selector read as BF16).
  The first acceptance numbers, side by side with DSpark, the calibration of the selector's confidence, and the first
  offline model are in [evidence](evidence.md#dflash2-m1).
- **M2 starts from a kernel profile** of the drafting pass (`dflash2-prof-20261109`). M1's draft call takes 5.6-5.7 ms
  against this study's estimate of 3.0-3.4 ms with the shared BF16 head.
