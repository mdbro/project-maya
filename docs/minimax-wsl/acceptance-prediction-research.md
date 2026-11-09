# Predicting accepted tokens per step for DFlash2 on MiniMax-M3.1-Flash

Research note, 2026-11-09. Written by a Fable research agent at the user's direction (task 59), from a minimal open-ended
prompt pointing at the literature, and recorded here for step 6. It is the agent's report, lightly edited for layout; file
references are as the agent gave them.

## Short answer

DFlash2 has no confidence head, but it does not need one. Its candidate selector already produces a per-position
probability for every drafted token, conditional on the path before it, and that is the quantity every training-free
method in the literature uses. The prediction for the planner:

- `c_k`: the selector's softmax probability of the chosen token at position k (over its 16 candidates, given the selected
  predecessor);
- `a(k, c)`: a learned acceptance table per position and confidence bin, its prior the identity (a calibrated drafter);
- the reach probability `r_k = prod_{j<=k} a(j, c_j)`; expected accepted tokens `E[A] = sum_k r_k`; expected committed
  tokens `1 + E[A]`.

`r_k` is the per-position pricing weight. The verify width is `argmax_e (1 + sum_{k<=e} r_k) / T_pred(e)`. Maya already has
this machinery for DSpark (`DraftPolicy::accept_rate`, `chain`, `tokens`). Infernix already emits the DFlash2
per-position probability and runs exactly this extent rule on another model. So the work is a port and a calibration
measurement, not an invention. A learned head (SpecDec++ or DISCO style) is a second stage, justified only if the
measured reliability diagram shows the bin table cannot fix the bias.

## What DFlash2 exposes (model card, config, SGLang, Infernix)

- **Card and config** (`RadixArk/Minimax-M3.1-Flash-Dflash2`):
  - 5 sliding-window layers; block_size 8 (an anchor and 7 proposals);
  - `selector_rank 256`, `selector_top_k 16`; conv kernel 2, group 16;
  - target layers [3, 17, 31, 45, 59]; mask token 200058; the full 200,064 draft vocabulary.
  - Reported `acc_len` at temperature 1.0, top_p 0.95 with 8 verified columns: GSM8K 5.16, AIME26 4.31, MBPP 4.38,
    MT-Bench 3.58.
- **Mechanism** (SGLang `python/sglang/srt/models/dflash.py`, vLLM speculators docs):
  - The trunk returns hidden states, and `compute_candidates` takes the top 16 unary logits per position against the
    target's `lm_head`.
  - `CandidateSelector.build_lattice` scores each transition as
    `unary(c) + <predecessor_codebook[prev] * hidden_projection(h), successor_codebook[c]>`.
  - `sample_path` walks greedily (argmax per step) or samples each transition from its own softmax. It returns
    `(tokens, q_rows)`, with `q_rows` shaped `(b, E+1, K)`: per-position probabilities over the 16 candidates. It is not
    Viterbi and has no path score.
  - `MiniMaxDFlash2DraftModel` is a bare alias of `DFlash2DraftModel`.
- **Infernix** (`/home/mdwbr/Infernix-minimax/include/infernix/ops/candidate_selector.h`):
  - `candidate_selector_path` has an optional `confidence` output, FP32 [K, B]. It is documented as "each draft's
    temperature-1 softmax over that position's 16 edges at the selected rank ... the drafter's own probability of draft i
    given the path before it, a draft-length signal that changes neither drafts nor proposal_q". That is the signal to port.
  - `candidate_selector_tree` builds trees best-first by "the probability that a column is reached and accepted". That is
    the product of the lattice law values along the path: the same reach-probability estimate.
- **Infernix's measured depth distribution** for DFlash2 at K=7 on a 27B code suite (`RESEARCH_NOTES.md`, about line
  305):
  - 26% of rounds end at the first proposal, 69% at depths 2-7, so about 5% accept the whole block.
  - Rejections are spread over every depth. Acceptance depends strongly on position, which is why the table must be per
    position.
- **The DFlash paper** (arXiv 2602.06036, sections 4.2 and 5.4.4):
  - It trains with an exponentially decaying position weight `w_k = exp(-(k-1)/gamma)`, so tail positions are less trained
    and should be expected to be less well calibrated.
  - A block-16-trained model generalizes to block 8 at inference, but not the reverse. The authors name "dynamic
    block-size scheduling" as feasible future work.
  - On Math500 the block-8 Qwen3-8B drafter fully accepts 35.7% of blocks.

## What the literature says a drafter can know before verification

Theory first, since it fixes what is predictable:

- **The acceptance law.** Under speculative sampling a token is accepted with probability `min(1, p(x)/q(x))`. Its
  expectation over the draft's draw is `1 - TV(p, q)` (SVIP, AdaEDL, the acceptance-theory paper arXiv 2606.30265).
- **What that means here.** For a greedy drafted path (q one-hot, as in Infernix's greedy rows and DFlash's default),
  acceptance at temperature 1 is exactly `p_target(x_k | accepted prefix)`. So the ideal predictor is the target's own
  probability of the drafted token, and the drafter's confidence is an estimate of it.
- **Sampled paths.** For a sampled 16-candidate path acceptance is `1 - TV(p, q_16)`. The same confidence is a proxy with
  a different bias, so the table should be keyed per spec policy or learned per session.
- **Target confidence.** The theory paper shows acceptance certificates depend on the target's top-token margin, and on
  the draft only through `KL(p||q)`. Target models are usually confident (median top-1 of 0.97-0.99 on UltraChat), so most
  steps are "certifiable", and a calibrated drafter's confidence tracks the target's margin.

Training-free predictors (all use the drafter's probability of its own token):

- **EAGLE-2** (arXiv 2406.16858, Fig. 6):
  - Draft confidence against measured acceptance is close to the identity: below 0.05 accepts about 0.04, above 0.95
    about 0.98.
  - A node's value, the product of the path's confidences, approximates the path's acceptance.
  - Ablation, ranking by value against raw confidence: 3.62x against 3.21x, tau 4.98 against 4.39.
- **AdaSpec** (arXiv 2503.05096):
  - It uses `AR_j ~= prod_{k<=j} c_k` and expected accepted `sum_j AR_j`, with a fitted linear step-time model, re-decided
    every step.
  - It reports the acceptance estimate's error as "minimal" (Fig. 8) without quantifying it.
  - 7-18% over a fixed length, 9-58% over threshold rules.
- **LibraSpec** (arXiv 2608.08721), built for diffusion drafters including DFlash:
  - It assumes calibrated confidences, `E[tau_{i:j}] = sum_k prod_l q_l`, and extends only while the marginal gain per unit
    of verification cost beats the current average.
  - On DFlash: Qwen3-4B 4.52x / tau 6.27 to 5.01x / 7.14; Qwen3-8B 4.43x / 6.29 to 5.12x / 7.20.
  - It uses no history and one correction per step.
- **FailFast** (arXiv 2512.20573), a diffusion drafter:
  - Confidence is the max probability; it stops when any position's `C_i < tau` (tau 0.4-0.5).
  - **An important negative finding.** An earlier version used previous rounds' acceptance as an easiness signal. The
    authors dropped it because "easiness appears to be highly local and not well correlated across rounds". DISCO likewise
    found only a weak correlation between the oracle length and the iteration's position.
- **AdaEDL** (arXiv 2410.18351) and **SVIP** (arXiv 2411.18462):
  - An entropy bound `beta >~ 1 - sqrt(gamma * H_q)`, via Pinsker plus a cross-entropy approximation. They stop when the
    bound (or `sqrt(H_q)`) crosses a threshold, and both need adaptive or tuned thresholds.
  - SVIP's evidence: mean draft entropy accepted against rejected is 0.25 against 1.30 (AIME), 2.18 against 3.99
    (MT-Bench).
  - AdaEDL beats max-confidence thresholding by a few percent; both beat static lengths by 10-57%.

Learned predictors:

- **SpecDec++** (arXiv 2405.19715):
  - A (D+1)-layer ResNet head on the draft's last hidden state, labels `min(1, p/q)`, weighted BCE.
  - Evaluation: binary KL 0.38-0.77, with no calibration curves.
  - It stops when `1 - prod sigmoid > h` (h = 0.7): +7.2, +9.4 and +11.1% over the best fixed K.
  - It argues draft entropy conflates uncertainty with misalignment. That critique is weaker for a greedy path at
    temperature 1, where acceptance is `p_target(x)`.
- **DISCO** (arXiv 2405.04304):
  - A 2-layer FFN on the draft's top-10 probabilities, entropy and position.
  - F1 95 / 88 / 77 against static 85 / 76 / 68 (MBPP / CNN-DM / Alpaca).
  - +10.3% over the optimal static length, +31% over the history heuristic; it transfers across tasks.
- **Draft-length predictors.** AdaEAGLE (arXiv 2412.18910) is a lightweight draft-length predictor. BanditSpec (arXiv
  2505.15141) runs a bandit over whole configurations, not per step.
- **Drafter training.** Acceptance-aware draft training (arXiv 2609.24150) and D-PACE (arXiv 2605.18810) train drafters on
  expected-accepted-length surrogates. That matters later, if the drafter is retrained, not for prediction now.

MoE-specific:

- **Cascade** (arXiv 2506.20675):
  - Verifying k tokens raises distinct expert activations. For Mixtral at k = 7, balls-in-bins gives about 3.5x the data
    movement, and verification time measured 2.3-3x.
  - Utility is tokens gained over the iteration-time overhead. It uses only measured history (test and set phases of 16
    iterations), since n-gram drafters have no probabilities.
  - Maya's planner already prices experts per position, a finer version of this. The per-position reach weights are what
    Cascade lacks.

Net reading:
- The per-step prediction comes from the current block's confidences, not from history. History's job is calibration and
  the cost model.
- Thresholding confidence directly is dominated by expected-tokens-per-time rules. Infernix measured a fixed 0.9
  threshold slower than its extent rule on the 122B.

## What Maya and Infernix already have

Maya (this worktree):

- `src/spec/draft_policy.cpp`:
  - `accept_rate(i, conf)` is the decayed held/seen count per position and confidence bin, with prior weight 4 at the
    confidence itself.
  - `chain(conf, n, q)` is the running product: reach probabilities. `tokens(q, k)` is `1 + sum q`.
  - `observe_drafts(conf, offered, accepted)` updates positions `i <= accepted` (reached); a position is kept iff
    `i < accepted`.
  - This is the per-position pricing and width input, built for DSpark's sigmoid confidence.
- `src/spec/controller.cpp`:
  - a per-position acceptance EMA `mtp_p_` (prior 0.86), and `expected(p, k)`;
  - `choose()` by tokens per `step_ms(k)`, where `CostModel::step_ms` already grows the CPU-miss cost with the
    distinct-expert ratio by width.
- `src/ds4/ds4_model.hpp`: `draft_reached[8]` and `draft_kept[8]` per-position counters; `draft()` takes `conf_min` and
  `cum_min` cuts; `dspark_conf` defaults to 0.5.
- `serve/server.py` carries `spec_policy` (existing, sampled, fast_hsd, lenience), `spec_min_p`, `spec_cum` and
  `spec_n_min`.

Infernix (`/home/mdwbr/Infernix-minimax`):

- `src/models/qwen3_5/program/speculative/mtp_confidence_policy.h`:
  - The extent `e` maximizes `(1 + sum_{j<=e} prod_{i<=j} a(q_i)) / (1 + c * e)`.
  - `a(q)` is learned in 10 buckets, decaying 0.995 per draft, with prior weight 4 at the bucket's midpoint.
  - The column cost `c` is fitted per batch size from rounds' start-to-start times.
  - It reads only the drafter's probabilities and earlier rounds, so output is unchanged.
  - On the 122B with offloaded experts: +5.5% over plain decode, and 10% over ungated 2-draft rounds.
- `src/models/qwen3_5/program/speculative/tree_width_controller.cpp`:
  - It measures same-text gain per width from each accepted path (`row_tokens`), and gain per second per batch size and
    context bucket.
  - It times widest first and explores every 32 rounds.
  - This is the tree-width controller the plan names. It measures history; it does not predict per step.
- The DFlash hand-off publishes `confidence()` to the host (`program/graphs.cpp:418`, `decode.cpp:1117`).

## What I would do

1. **Port the signal.** Add the `confidence` output of DFlash2's selector path to Maya's drafter: the 16-way softmax at
   the selected rank, conditional on the selected predecessor. For greedy rows, compute it at temperature 1 as Infernix
   does. Keep the 16-way entropy and top-1 margin per position too; they come free from the same softmax.
2. **Reuse `DraftPolicy`'s table as the calibrator.**
   - The table is `a(k, c_k)` per position and bin, with the identity as prior, decayed. Then `r_k = chain()` and
     `E[A] = tokens() - 1`.
   - Keep separate tables per spec policy (greedy path, sampled path, lenience), since the acceptance law differs.
   - Feed `r_k` to the per-position pricing, and `1 + sum r_k` over `T_pred(e)` to the width choice, as the plan's two uses
     say.
   - Exactness holds: the prediction reads only the drafter's probabilities and past rounds, never the round's own draws.
3. **Measure before trusting.**
   - Log per step: `k, c_k, entropy_k, margin_k, reached_k, kept_k`, the policy, the prompt class and the anchor token's
     class.
   - Report:
     - a reliability diagram per position (bins of `c_k` against the kept rate);
     - the Brier score per position;
     - the MAE and the distribution of predicted against actual accepted length per step;
     - how often the predicted width matched the oracle width.
   - This is the "acceptance per position, by prompt type and from any per-step prediction" input the plan's offline model
     lists.
   - Expect two kinds of error:
     - tail positions less well calibrated, from the decayed training weight;
     - the raw product optimistic where errors are correlated: a misread anchor fails every position.
     The per-position table absorbs the first; the diagram shows the second.
4. **Only if the diagram shows systematic error that the bin table cannot remove**, add a DISCO-style 2-layer MLP.
   - Its inputs: the 16-way probabilities (or the top 10), entropy, margin, position, the anchor token's class and the
     previous step's accepted count. It is trained offline from the logs in step 3, with labels `kept | reached`.
   - It is cheaper than SpecDec++'s head: no hidden states, and no retraining of the 2B trunk.
   - Keep SpecDec++'s head on the drafter's final hidden state as the last resort.
5. **What not to do.**
   - Don't use acceptance history or the round index as the per-step predictor (FailFast, DISCO); use history for
     calibration and for `T_pred`.
   - Don't gate on a fixed confidence threshold: Infernix and AdaEDL both found it inferior to the
     expected-tokens-per-time rule.
   - Don't sharpen the proposal softmax hoping for higher acceptance. Infernix found s = 0.5 lowered acceptance from 33.2%
     to 31.7%.
6. **Later, tied to step 6's offline optimizer.** LibraSpec's marginal-gain extension rule is the same objective in
   incremental form. It could set a width above 8 only if a block-16-trained DFlash2 existed. With the shipped block-8
   model the width decision is a cut at 1-7 drafts, which the extent rule above covers.

## Files

- Plan context: `docs/minimax-wsl/README.md`, step 6.
- Maya's calibrator and controller:
  - `src/spec/draft_policy.cpp` (`accept_rate` 146, `chain` 152, `tokens` 157, `observe_drafts` 262);
  - `src/spec/controller.cpp`;
  - `src/ds4/ds4_model.hpp`, lines 147-149 and 203-220.
- Infernix:
  - `/home/mdwbr/Infernix-minimax/include/infernix/ops/candidate_selector.h`;
  - `/home/mdwbr/Infernix-minimax/src/models/qwen3_5/program/speculative/mtp_confidence_policy.h`;
  - `/home/mdwbr/Infernix-minimax/src/models/qwen3_5/program/speculative/tree_width_controller.cpp`;
  - `/home/mdwbr/Infernix-minimax/docs/maintainer/tree-verification.md`;
  - `/home/mdwbr/Infernix-minimax/RESEARCH_NOTES.md`, lines 179-211 and about 305-315.

## Sources

- DFlash2 and its implementations:
  - DFlash2 card and config: https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2
  - SGLang's DFlash2 source: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/models/dflash.py
  - vLLM speculators, DFlash2: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/dflash2/
  - NeMo's DFlash2 trainer: https://docs.nvidia.com/nemo/automodel/nemo-automodel/nemo_automodel/components/speculative/dflash/dflash2_core
  - DFlash paper: https://arxiv.org/abs/2602.06036
  - LMSYS on DFlash and Spec V2: https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/
- Training-free predictors:
  - EAGLE-2: https://arxiv.org/html/2406.16858v1
  - AdaSpec: https://arxiv.org/html/2503.05096v2
  - LibraSpec: https://arxiv.org/html/2608.08721v1
  - FailFast: https://arxiv.org/html/2512.20573v2
  - AdaEDL: https://arxiv.org/html/2410.18351
  - SVIP: https://arxiv.org/html/2411.18462v2
- Learned predictors:
  - SpecDec++: https://arxiv.org/html/2405.19715v3
  - DISCO: https://arxiv.org/html/2405.04304v5
  - AdaEAGLE: https://arxiv.org/abs/2412.18910
  - BanditSpec: https://arxiv.org/abs/2505.15141
- Theory and drafter training:
  - Acceptance theory: https://arxiv.org/html/2606.30265v1
  - Acceptance-aware draft training: https://arxiv.org/html/2609.24150v1
  - D-PACE: https://arxiv.org/abs/2605.18810
- MoE-specific:
  - Cascade (MoE utility): https://arxiv.org/html/2506.20675v1
