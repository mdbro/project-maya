# Evidence for the MiniMax WSL proposals

Current requirements, assigned bases, shared weight location and work status are
owned by the [central plan](README.md). This file is supporting evidence; original
artifact paths and source snapshots below retain their historical meaning. See the
[base creation record](base-provenance.json) for the dedicated forks established
on 2026-11-01.

Inspected 2026-10-31. Local results below are retained measurements from their
original protocols; no new inference, GPU benchmark, model download or build was
run to write these proposals. Remote source was read through the GitHub connector;
Strata's oversized generator was read directly after the connector returned an
empty body. These proposals do not assume identical results on different models,
quants, operating systems or request histories.

On 2026-11-01 the user replaced the Unsloth choice with the official
[MiniMaxAI/MiniMax-M3.1-Flash repository](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)
for all three proposals, including its official DSpark weights. The central plan
owns the current download location/readiness. GGUF and EXL3 measurements below
remain historical evidence; neither is a required new comparison arm.

The [pinned official inventory](official-checkpoint.json) records revision
`2f33a8b9ae49ce7b51c994df97166a7fd059bc2e` and 252,446,157,672 repository bytes.
There are 48 target safetensors shards plus one `dspark/` weight shard.
The [official configuration](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e/config.json)
declares `nvfp4-pack-quantized` experts with group size 16 and MXFP8 dense weights
with `[1, 32]` blocks. Range reads of two target headers and the draft header
confirmed packed U8 weights/scales, F8_E4M3 tensors and BF16/F32 tensors as
applicable. Expert records include block/global weight scales and input-scale
calibration. The official DSpark shard contains MXFP8/BF16 tensors, not the prior
Q8_0 GGUF representation. These are remote metadata observations, not a complete
local download, integrity check or full parameter-weighted precision inventory.

## E1 Local artifacts and hardware

The live environment inspection identified RTX 5090 32 GB, RTX 3090 24 GB, Ryzen
9950X3D, CUDA 13.4.92, driver 617.42 and WSL kernel 6.18.33.2. `.wslconfig` allocates
244 GB and 30 logical CPUs, with swap disabled. The [prior topology record](../../../Infernix-glm5-local/docs/maintainer/multi-gpu.md)
identifies the primary as PCIe 5 x16 and the 3090 as PCIe 4 x4 behind chipset
bridges; peer access was unavailable in both directions. Those topology/transfer
tests are historical measurements, not a newly repeated link test.

| Local artifact | Inspected facts | Planning implication |
|---|---|---|
| `/mnt/e/models_ablit2/MiniMax-M3.1-Flash-EXL3-3.5bpw` | All 25 language shards present, 179.377 GiB; vision and DSpark companions also present. Config records EXL3 average 3.55 bits. Routed tensor storage sums to 168.845 GiB. | Historical local execution evidence; not the selected checkpoint. |
| `/mnt/e/models_ablit2/unsloth/MiniMax-M3.1-Flash-GGUF/UD-Q4_K_XL` | All 7 shards present, 232.274 GiB. Headers contain 1,359 unique tensors: 171 NVFP4/type-40 expert tensors, 766 F32, 420 Q8_0 and 2 BF16. Indexers are present in layers 0-2 as well. | Previous selection, retained untouched as a historical reference. Its interrupted native-WSL copy was deleted. |
| Native expert payload calculation | `57 * 128 * 3 * 6144 * 3072 * (36/64)` bytes = 216.421875 GiB; one expert = 30.375 MiB. The inspected GGUF's additional expert scale tensors total 87,552 bytes. | The group-16 NVFP4 geometry remains useful, but inventory official auxiliary scales/records separately. A GLM-sized host bank cannot be assumed. |

Header/index presence does not establish full shard integrity. No large file hash
pass was necessary for proposal selection. The [GGML type definition](../../../llama.cpp/ggml/include/ggml.h)
and block layout confirm type 40 as NVFP4. The current Maya
[GGUF packer](../../tools/mm3/pack_gguf.py) rejects it, and
[MiniMax GGUF kernels](../../src/mm3/mm3_gguf.cu) lack the route. This describes
the earlier GGUF investigation. The selected official path instead requires
safetensors loading and native NVFP4/MXFP8 scale/layout support in each base;
loader convenience does not choose the architecture.

The EXL3 [model card](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw)
describes the mixed 3/4-bit expert allocation and higher-precision dense components.
Any additional candidate needs its own parameter-weighted inventory. Maya-S and
Maya-XS fall below this project's roughly 3.4-3.5-bpw lower boundary and are not
candidate quantizations, even where their reported speeds are informative about
memory pressure.

## E2 Results already recorded on this computer

The owner clarifies that the Windows MiniMax helper was a rough port of upstream
Maya's DeepSeek V4.1 helper. The following results characterize that derivative
and its tested changes. They neither select its architecture for WSL nor establish
the performance of the original DeepSeek helper or a clean MiniMax adaptation.

| Observation | Scope and source | Consequence |
|---|---|---|
| Queued MiniMax helper: 7.904 to 8.948 tok/s | Windows, EXL3, two prompts, three trials, 128 outputs after warming; all 768 compared tokens match. [Helper account](/mnt/c/Users/mdwbr/project-maya/docs/MINIMAX_HELPER.md) and `logs/mm3-helper-validation/worker-model-comparison/result.json`. | Retain the complete-service observation; caller-submission microseconds alone did not predict the winner. It does not select the derivative wrapper over either original implementation. |
| Trained placement: median 9.068 to 12.762 tok/s; independent set 9.399 to 13.091 | Four prompts per set, 128 outputs, 256K configured context, INT8 KV, CPU16, DSpark off; all requests improve but most continuations differ. [Calibration report](/mnt/c/Users/mdwbr/project-maya/logs/mm3-calibration/20261031-014925-28028/report.json). | Placement is a large observed lever. Reuse the saved profile; do not claim these are WSL or identical-output rates. |
| Remaining CPU service: median about 46-48 ms/token on selected-profile sets | Same report; exposed synchronization counters also remain. | Avoided host work and fast CPU kernels merit attention. Counters cannot simply be added as independent wall-time savings. |
| Ordinary copies / direct GPU fetch / buffered GPU fetch: 11.886 / 10.687 / 8.399 tok/s | Windows fixed-staging comparisons, two prompts/two trials/64 outputs, matching tokens. [Fetch result](/mnt/c/Users/mdwbr/project-maya/logs/mm3-helper-validation/fetch-buffered-20261031-024526/result.json). | Preserve these losses for the experimental port. They do not reject the original DeepSeek fetch mechanism or a clean adaptation, and are not directly comparable with the placement protocol. |
| Single host allocation retained 76.416 GiB; chunked allocations retained 135.869 GiB | Windows application load, 1-GB or 4-GB chunks; the larger requests failed in single-allocation mode. [Helper account](/mnt/c/Users/mdwbr/project-maya/docs/MINIMAX_HELPER.md). | Preserve chunking and allocation diagnostics; a Windows result does not validate WSL. |
| Idle transfer estimate near 8 ms versus about 0.6 ms after activity | Recorded startup-copy diagnostic; warmed automatic-policy inference remains unmeasured. Same helper account. | Calibrate actual sustained service; never infer useful dispatch from an idle link measurement. |
| WSL direct allocation: 189 GiB raw, 182.25 GiB retained/mapped/transferred/verified; separate selective HOST VMM about 162 GiB | [Allocation study](../../../Infernix-glm5-local/profiles/bench/glm5-allocation-study-20261024/summary.json), including API/context/access conditions. | A universal 1-GB WSL pinning ceiling is contradicted. The MiniMax application's chosen bank layout still needs its own load check. |
| GLM fixed MTP1 24.5882 versus automatic policy 24.0747 tok/s | Frozen executable and five-prompt/two-pass history; automatic policy lost eight requests and all continuations differ. [GLM plan](../../../Infernix-glm5-local/docs/maintainer/glm5_3-flash-implementation-plan.md). | Complete-round policies can reverse a short-screen result. Reuse method/negative evidence, not this model's rate or its old strict numerical requirements. |
| Earlier Qwen contiguous-stage prototypes lost 6.2-35.3% on their common set | [Local multi-GPU record](../../../Infernix-glm5-local/docs/maintainer/multi-gpu.md); whole-stage/partial-stage controls and later all-hit analysis. | A serial split is not automatically beneficial. This does not reject MiniMax speculation/overlap or changed routing. |

The helper-off MiniMax logit checks also differed: the small INT8 probe agreed on
30/33 top tokens, the BF16 probe on 31/33, and the formatted chat probe on 180/192.
These remain numerical observations, not proof of a quality defect or equivalence.
The new project permits approximate execution and will use short practical screens.

## E3 Other users' measurements and what transfers

| Report | Relevant observation | Limits for this computer |
|---|---|---|
| [Maya MiniMax PR 477](https://github.com/mw00/project-maya/pull/477), contributor's own system | EXL3 3.5 bpw; 3090 + eight 5060 Ti, 152 GB aggregate VRAM, 183 GB RAM. Plain layer split about 16 tok/s; DSpark 27.7 code/about 19 prose; TP4 plus DSpark at 8K 38.2 code/32.4 prose. 300-token greedy outputs. | Same model and eligible precision, but far more VRAM, nine GPUs and different links. Shows DSpark/placement can matter; supplies no local speed target. |
| [Xolotl MiniMax issue 62](https://github.com/Indras-Mirror/Xolotl/issues/62), contributor's prototype | 5090 + EPYC 9654/624 GB, UD-IQ3_S; 27.5 tok/s at 4K versus llama.cpp 23.3; prefill 1,391 versus 425 tok/s. DSpark did not help because the expert union grew. | Host service reportedly reached about 238 GB/s; this is not the 9950X3D memory system. The quant's compliance with our precision boundary is not established. Mechanism evidence only. |
| [Strata community PR 2012](https://github.com/Niko1221/Strata/pull/2012) | 5090 + 3090, Ryzen 7900X, 191 GiB RAM, Windows, Qwen Flash-Next IQ3_S, engine 0.1.43, spec4/MTP/INT8 KV: 79.1 tok/s short chat and 115.8 agent-session decode; three-run medians. | Different model, router granularity, expert sizes and drafter; 131K is configured context. No correctness checks; storage/background/power details omitted. A topology-relevant report, not a MiniMax forecast. |
| [Strata split history](https://github.com/Niko1221/Strata/blob/main/docs/MULTI_GPU.md) | On a 9950X3D with 5080 + 3090 and the Coder model, adding the 3090 left story decode roughly flat while code reached about 110 tok/s; other weak-CPU machines benefited more. | Different model/quant. Demonstrates that CPU speed and split choice can change the value of the second GPU. |
| [Basalt author's seven-engine comparison](https://www.reddit.com/r/LocalLLaMA/comments/1xdlunp/i_ran_all_the_popular_qwen_38_fn_engines_on_a/) | Qwen Flash-Next/NVIDIA NVFP4 conversions, 5090 at 400 W, Linux/128 GB: short conversation/prose Basalt 188/180, Infernix 170/163, Strata 146/123 tok/s. | One cold process per request, one run, different drafters and some representation/KV differences. Useful source candidates, not a clean architecture ranking or warmed MiniMax evidence. |
| [FOMOE](https://github.com/pmerolla/fomoe) | Qwen3.5-397B Q4_K_M, two 16-GB RX 9060 XT, 32 GB RAM/NVMe: 5.1 to 8.8 tok/s with cache-aware substitution, reported +3.5% WikiText perplexity. | SSD-heavy AMD system and another model. Supports reviewing actual substitution/backfill code, not importing its threshold or speedup. |

Low-bit Maya-S reports, vague social-media rates, multi-request throughput and
hardware-calculator estimates were not used to predict a compliant MiniMax rate.
There is no evidence here for a numerical local speed ceiling.

## E4 Codebase assessment

| Codebase inspected | Snapshot / material source findings | Role |
|---|---|---|
| WSL Maya | `6ecbd3c`, v1.1.45, initially clean. Native MiniMax, DSpark, CPU/EXL3/GGUF execution and streaming frontend; original DeepSeek helper source also present. | Assigned base for Proposal 2 and a shared control/model donor. Adapt the helper from original sources. |
| Original Maya DeepSeek helper | `src/ds4/ds4_model.cu` / `ds4_kernels.cu`: whole-window ordered helper jobs, mapped sequence handoff, grouped expert execution and separate recent-slot refill/publication. | Primary helper-design reference alongside local WSL Infernix; MiniMax geometry, codecs and service costs still need adaptation. |
| Windows Maya | `df90e59`, v1.1.42-based branch with local changes. Ignoring line endings, ten tracked files have substantive changes plus six new untracked files. Owner identifies its MiniMax helper as a rough derivative port. | Preserve source and measurements; inspect individual useful observations. Do not select its wrapper/scheduling architecture merely because it already connects to MiniMax. |
| Local Infernix | `bae2bf11`, `feat/glm5-cooperative-c1`, plus working-tree changes. CPU/5090 host bank, helper worker/refill, graph and resource ownership. GLM FFN has H<=4096 and different activation. | Donor for the assigned independent Proposal 1 base at `/home/mdwbr/Infernix-minimax`; MiniMax needs a model port and real geometry/math adaptation. |
| Upstream Infernix checkout | Local snapshot `7dbe99a3`; single-GPU Blackwell runtime with Qwen Flash-Next and DFlash2. No MiniMax model found in inspected local source. | Technical reference. The user subsequently assigned the isolated local Infernix fork as Proposal 1's base. This is a local snapshot, not a claim about newest remote HEAD. |
| NInfer checkout | Local snapshot `68c54356`; one-GPU resident Model/Program/Engine organization. | Ownership/serving references; no demonstrated advantage for this heterogeneous expert-heavy task. |
| Basalt checkout | `b9f9b125`; Blackwell target, peer expert tier, resident cache plus temporary staging, multi-column NVFP4 CPU dots and draft-cost policy. | Kernel/window donor for the assigned bases. |
| Strata source | Read `route_prior.cu`, `native_router.cu`, `verify.cpp`, `generate.cpp`, multi-GPU docs and frontend; dedicated fork cloned at `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`. | Assigned stage/runtime base for Proposal 3 at `/home/mdwbr/strata`; concrete routing and recovery donor for Proposal 2. |
| FOMOE remote source | Read `car.c` and active calls/backfill in `gpu_kernels.hip`. | Source-derived substitution, original-demand recovery and budget reference. |
| BigMoeOnEdge remote source | `cb78bfe0f1218ab350a1b7bcd3945c45eaf94595`; inspected `core/src/moe/router_hook.cpp`, its session call and `docs/cache-aware-substitution.md`. | Additional source for score-range-scaled resident reranking with unchanged expert count. Its cache is RAM backed by flash; use the mechanism as a reference, not its runtime or speed as a MiniMax result. |
| Bloomery remote source | Read README, MiniMax body/tier and qdot CPU code. Supports a hybrid MiniMax NVFP4 path; current mixed-GPU prebuilt selection favors its sm86 archive. | CPU/layout/model donor. Per-device Blackwell/Ampere dispatch needs review before attributing native-FP4 benefit on this pair; it does not replace the assigned base. |

Remote source anchors: Strata route-prior blob `79a7581`, native-router `6611bce`,
verifier `e8fdbd4`, multi-GPU docs `51a6547`, frontend `4955193`; FOMOE CAR
`8ec84fc`, GPU implementation `02cd789`; Bloomery qdot `7f8a9b7`, MiniMax body
`695dba8`. These are file blob identities, not repository commit IDs. Resolve a
specific implementation revision before porting; do not silently mix moving HEADs.

The original DeepSeek `helper_window` queues the helper's participating layers
for a whole window in sequence; `helper_take_kernel` waits for each mapped request
and builds its resident plan. Infernix instead offers a persistent worker with
direct device publication and GLM-specific protocol dimensions. The Windows
MiniMax wrapper has a single pending `WinArgs` job and a separately queued
next-request gate. These are material scheduling differences, not interchangeable
names for an already selected implementation. Source inspection identifies the
comparison; it does not establish which is faster at MiniMax shapes in WSL.

## E5 Approximation and speculation lessons

Strata's [routing implementation](https://github.com/Niko1221/Strata/blob/main/src/core/verify.cpp)
records a cache-prior setting that improved short speed tests but was withdrawn as
a default after repetition and weaker long-agent results. Its bounded resident
substitution also records why a serial scan originally consumed almost its entire
saved transfer time. Reuse the warp-wide implementation, and count routing cost.

FOMOE's backfill addresses the cache starving the original router's desired
experts. The [PSR study](https://arxiv.org/html/2610.34721v1) explains another
practical issue: cache-conditioned choices can change with asynchronous transfer
timing. This project does not require deterministic routing, so a separate logical
cache is not an initial prerequisite. The short quality check must nevertheless
run the live chosen policy, rather than pretending an offline exact-routing replay
evaluated the deployed approximation.

BigMoeOnEdge's [substitution measurements](https://github.com/Helldez/BigMoeOnEdge/blob/cb78bfe0f1218ab350a1b7bcd3945c45eaf94595/docs/cache-aware-substitution.md)
provide another implementation example and a useful quality observation: aggressive
reranking could produce fluent prose despite substantially worse measured behavior.
Its 50-problem Qwen3.6 HumanEval comparison at the milder setting retained 42/50
passes while reporting 2.36 to 5.53 tok/s, with warm-cache flash traffic reduction.
These are the author's small, greedy, RAM/flash-cache experiments, not measurements
on our model or hardware. Reuse the existing brief checkable coding/tool outputs;
do not add a broad benchmark campaign or treat fluency alone as sufficient.

Strata's inspected pipeline code assumes an MTP drafter and has route/placement
restrictions; favorable automatic defaults are primarily from other GPU families.
MiniMax DSpark instead consumes five target-layer taps. Reuse the scheduling and
state machinery while adapting this dependency explicitly. Neither high draft
acceptance nor high cache hit rate alone establishes lower time per emitted token.

## E6 Selected Fast-HSD acceptance

The user selected [Fast-HSD](https://github.com/ZhouYuxuanYX/Fast-HSD) for relaxed
speculative acceptance while fixing the main model's temperature at **1** and
top-p at **0.95**. The initial policy is its SpecCascade target-relative cutoff;
the cutoff value remains an experimental parameter. This source selection changes
neither the assigned proposal bases nor the shared checkpoint selected in the
central plan.

Inspected revision: `66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17`.
[`truncation_verification.py`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/fast_hsd/core/truncation_verification.py)
implements `min_p_allowed_set` / `speccascade_accepts` as
`p(x) >= p_base * max(p)`. The active
[`_speculative_sampling`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/transformers/generation/utils.py)
implementation uses `min_p_spd` for this acceptance test, commits the accepted
prefix, and samples from target `p` at the first rejection or after full acceptance.
Its cutoff path does not draw the replacement from the lossless residual.

At the initial source inspection, Maya's
[MiniMax serving loop](../../src/mm3/mm3_main.cpp) performed the sampled `p/q`
acceptance test and called `draw_residual` on rejection. Its
[sampler](../../src/mm3/mm3_sample.hpp) already applied temperature and top-p.
Those were the integration points for retaining the target sampler and DSpark's
draft/context machinery while adapting Fast-HSD's acceptance/replacement semantics.
The implemented Fast-HSD route requires membership in the existing top-p support
before relaxed acceptance. Its recorded serving evidence and remaining state
verification are in the [central plan](README.md#r3-speculative-state-verification).

The induced output distribution is intentionally allowed to differ from the
target's ordinary sampled distribution. Code inspection establishes an algorithm
and integration reference, not a measured MiniMax speed or quality result.

## E7 Implementation review (2026-11-01)

The [central review follow-up](README.md#review-follow-up-2026-11-01) owns current
status, required repairs and decisions. This section preserves review evidence.

- All 64 files in the implementation's final source/binary manifest still matched.
  Twenty-four saved requests across nine result files matched raw SSE content,
  reasoning and usage; suffix counts and intervals reproduced the reported rates.
- Recomputed delivery rates were 11.5936 selected and 7.0229 ordinary tok/s.
  The architecture pair reproduced Maya 14.6715, repaired eight-layer Strata
  12.9776, and its 25-record helper composition 12.1498 tok/s. These are the
  recorded short suffixes, with the documented output lengths and histories.
- The five main API requests totaled 179.6256 s selected versus 155.8602 s ordinary.
  Selected first prefill was 93.646 s versus 33.032 s after different resident
  histories. These totals do not establish a matched warmed workflow comparison.
  The user clarified that prefill and whole API latency are diagnostic only for
  this phase; they do not enter the generation-speed ranking or reject a candidate.
  The single reasoning-case failure and simulated tool response remain explicit.
- The independent offline server/parser run passed 161 tests with one optional
  renderer skip. CPU Fast-HSD and full-vocabulary sampler checks passed. No model
  inference or GPU benchmark was run during review.
- P3 executes real stages, all five target taps and useful overlap. Only the
  eight-layer split and its small helper allocation were established. No trial
  or sufficient bound closed the smaller-prefix/larger-helper alternative.
- Source inspection found unchecked promotion/load submissions and failed-join
  handling. [CUDA events capture stream work](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html);
  later event success does not establish success of an ignored copy submission.
  This is a failure-path finding, not evidence that the recorded answers used
  corrupted weights. The older resource reconciliation's no-lifetime-defect
  conclusion is superseded; its measured allocation counts remain unchanged.
- The Fast-HSD unit test covers cutoff/support and replacement draws. It does
  not establish target/DSpark state after forced rejection and prefix reuse.
  Review found an evidence gap, not a demonstrated rollback defect.
- At the live review check (19:43 UTC), recorded PIDs 1426726/1426728 were absent,
  no process listened on 8097 and health requests failed. The reason the service
  stopped was not established; the review did not restart it.

Sources: [comparison summary](/mnt/wsl/transfer/minimax-wsl/runs/proposal-2/final-comparison-summary.json),
[source manifest](/mnt/wsl/transfer/minimax-wsl/runs/proposal-2/final-source-manifest.json),
[ordinary delivery](/mnt/wsl/transfer/minimax-wsl/runs/proposal-2/delivery-ordinary.json),
[selected delivery](/mnt/wsl/transfer/minimax-wsl/runs/proposal-2/delivery-selected.json),
and the source/test references in R1-R3. Original measurements and raw files were
preserved; review did not change implementation source.

## E8 Generation timing and helper calibration audit (2026-11-01)

The [central protocol](README.md#generation-timing-and-expert-cache-conditioning)
owns the updated measurement requirements. This read-only source/capture audit
did not run a model or change runtime behavior.

- `native_screen.summarize_stats` subtracts the engine time at emitted token 32
  from its final generation time. `serve` starts that clock after prefill and
  post-prefill service calibration. The reported pooled generation rates exclude
  both. Token 32 was a fixed boundary, not measured cache convergence.
- `NativeHelper::add_expert` rejects additions after `seal`; the model selects
  helper records from the input profile during loading. Saved learned profiles
  influence a later placement, not the current fixed 3090 set. The post-prefill
  calibration measures CPU/primary service; it does not choose new helper records.
- The final config applies the frozen 4,275,000-count native profile and writes
  later demand elsewhere. Thus earlier responses did inform the final helper.
  Its verified predecessor has 3,007,320 counts; the difference is 1,267,680.
  The five contributing requests record 4,774 processed prompt positions and
  640 emitted tokens. Each prompt position records four original choices in each
  of 57 MoE layers: `4774 * 57 * 4 = 1,088,472` prefill counts, or 85.9% of
  that increment. The remaining 179,208 counts include generation/verification
  work. This is a reconstruction from phase lengths and production counting,
  not separately captured per-expert phase matrices or a hit-rate measurement.
- The old rates remain valid observations under the recorded mixed profile.
  They do not establish representative generation-calibrated helper performance.
  Existing short-block variation and ongoing primary promotions do not establish
  convergence at token 32 or identify a different universally adequate cutoff.
- The inspected ordinary delivery requests had 31 target forwards at the token-32
  sample; selected Fast-HSD delivery and composition requests had 4-8. Their
  promotion opportunities therefore differed despite the same emitted-token
  cutoff. This does not prove every run was cold, but warmed-cache qualification
  is not established. The central
  [reassessment rule](README.md#qualification-of-existing-token-32-results) requires
  adequate archived conditioning evidence or a corrected candidate/control run,
  including when implementation source is unchanged.
- The later handoff check found a separate timing-boundary issue: `emit()` emits
  the token-32 stats inside the acceptance loop, after the entire target window
  has executed. It can therefore exclude work for later tokens counted in the
  suffix. This is a source-level attribution issue; its effect on old rates was
  not quantified. Corrective measurements must use completed generation-round
  boundaries and actual cumulative counts, without adding synchronization that
  changes execution behavior.

The [machine-readable audit](generation-conditioning-audit.json) preserves source
hashes, request phase counts and calculations. Implementation references:
[`serve` timing](../../src/mm3/mm3_main.cpp),
[`observe_routes`, helper placement and profile saving](../../src/mm3/mm3_model.cu),
[`NativeHelper` contract](../../src/mm3/mm3_native_helper.hpp), and
[frozen-profile provenance](native-conditioning-provenance.json).

## E9 Inherited primary residency and local comparison evidence (2026-11-01)

[R4](README.md#r4-evidence-based-architecture-comparison-and-adaptive-primary-residency)
owns the resulting architecture comparison. These findings came from read-only
Git/source and saved-report inspection; no model or GPU work was run.

The fixed primary bank originates in upstream MiniMax introduction commit
`ff5a55718eba2ac86c5092c451a0f2ead65b385b` and is present at the pinned starting
revision `6ecbd3c`. Both versions select `res_slot`/`res_ids` at load and direct
`consider_promotion` to the separate staging slots. The native port retains that
structure. Its measured 438 primary residents are not an adaptive main bank;
its eight staging slots supply the limited inference-time adaptation. No recorded
MiniMax comparison selected this structure over efficient main-bank replacement.

This differs from Maya's GLM `LayerPool` in
[`glm_fast_state.hpp`](../../src/core/glm_fast_state.hpp), which represents resident,
spare, draining and landing slots, and the assigned Infernix base's
[`ExpertResidency`](../../../Infernix-minimax/src/runtime/expert_offload/expert_residency.cpp).
Those are concrete reusable mechanisms. The local GLM evidence also measures
the interaction between retained demand fills and paid background promotions:

| Local GLM configuration | Recorded warmed generation tok/s |
|---|---:|
| No demand landing, normal paid primary promotions | 22.6908 |
| No demand landing, paid primary promotions disabled | 21.2840 |
| Two conditional landing frames per layer, normal promotions | 23.4209 |
| Same landing, paid primary promotions disabled | 23.6136 |

The [promotion-off-only report](../../../Infernix-glm5-local/profiles/bench/glm5-speed-campaign-20261028/w9/w9-promotion-off-only-vs-control-full.md)
records a 6.200% loss across the complete ten-request history, with matching
outputs. The [combined-versus-landing report](../../../Infernix-glm5-local/profiles/bench/glm5-speed-campaign-20261028/w9/w9-combined-vs-landing-only-full.md)
records a 0.823% gain with one differing request; that difference limits exact
attribution. The [local plan](../../../Infernix-glm5-local/docs/maintainer/glm5_3-flash-implementation-plan.md#landing-promotion-separation)
preserves all four cells, their conditions and the separate near-neutral physical
unified-pool result. Paid-promotion disabling is not the same as removing all
dynamic demand service or fixing the resident bank.

These results support testing a credible retention/refill combination on MiniMax.
They do not predict a transferable percentage gain, select every dynamic policy,
or impose GLM's arithmetic/output rules on this project. Valid MiniMax kernel,
serving and scoped speed evidence remains retained. Its broader architecture
ranking remains open because the efficient adaptive alternative was not represented.

## E10 Donor fidelity and executed-path audit (2026-11-01)

This audit compared local Git baselines, current production callers, linked source
and pinned donor files. It did not change runtime source or run models/GPU tests.
The central R2/R3 repairs and R4 comparisons own follow-up; the mandatory restart
order remains repair, corrected existing measurements, then new architectures.

| Mechanism | What is actually reused or adapted | Assessment |
|---|---|---|
| Cooperative expert service | Maya directly compiles the new assigned-P1 `minimax_nvfp4_ffn/{record.cpp,fast_cpu.cpp,fast_cuda.cu}`. The selected CPU AVX-512/VNNI and GPU INT8/DP4A service is new to this effort, with RG16 packed-decode concepts and an attributed Basalt scale table. A directly adapted GLM BF16 kernel also exists, but is not the selected linked service. | MiniMax geometry, OAI SwiGLU and native formats justify adaptation. Real complete C1/C4 service and contention were measured. Calling this shared code an unchanged upstream executor or kernel would be inaccurate. |
| Cooperative scheduling | `NativeHelper` is a new single-active-pass host worker. Host routing/readback and BF16 packing precede submission; CPU, primary and helper FFNs overlap; helper output is copied back and scattered on the host before publication. | It preserves useful concurrency and explicit sequence ownership, but does not port the original DeepSeek whole-window queue/device handoff or Infernix GPU-persistent direct dispatch. The source explains a compatibility obstacle, not a measured win over those architectures. |
| Residency-aware routing | Strata warp ranking/prior and FOMOE ratio/no-duplicate ideas are adapted to MiniMax's sigmoid-plus-bias ranking. Final mixture weights use the selected experts' unbiased sigmoid values; model scale remains outside selection. | The numerical domains and distinct-ID rules are coherent. Fixed lower-two-rank substitution among the top eight is the planned initial policy, not the entirety of FOMOE's quality-sorted budgets, protected layers, warmup and optional damping. RAM was deliberately not treated as a free GPU hit. |
| Original-demand recovery | The supported native grouped path captures original choices and calls decayed-frequency promotion; saved original counts remain distinct from executed choices. | Its recovery reaches only the eight staging slots, not the fixed main bank or sealed helper. It also lacks a useful-backfill counter. This is materially narrower than full adaptive donor recovery and remains part of R4. |
| DSpark | `mm3_dspark.cu/.hpp` and `mm3_kernels.cu/.cuh` are unchanged from Maya baseline `6ecbd3c`; native loading binds the official five-layer draft, MXFP8 projections, BF16 auxiliaries and target taps 3/17/31/45/59. | This is inherited model/drafter machinery, not a new recreation. Its state assumptions still need verification and the inherited ring/prefix defect below needs repair. |
| Fast-HSD | The selected cutoff and target replacement/bonus draw are adapted from revision `66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17` into Maya's existing sampler/verification loop. | The chosen algorithm matches the donor at source level: accept the contiguous prefix satisfying the target-relative cutoff within top-p support; draw from target p at the first rejection or after full acceptance. It does not claim to implement every Fast-HSD method or preserve the ordinary sampling distribution. |

Three concrete existing-path issues were independently checked:

- **Missing bounded helper progress.** `NativeHelper::collect` waits without a
  deadline, and its worker waits for a CUDA event. DeepSeek's `helper_answer`
  bounds its wait; Infernix's `PeerExpertExecutor::finish` raises a progress error
  on timeout. Add safe bounded failure/retirement under R2. A timeout cannot
  justify releasing an unknown writer's buffers or falling back to CPU when the
  exclusive host bank does not contain the helper's weights. No actual hang was
  observed in this audit.
- **Guaranteed inactive primary work.** The Maya adapter launches `T * k * 2`
  descriptors, four at a time, with capacity for eight columns. At C1/top-4 there
  are eight descriptor slots and at most four actual jobs. Inactive slots return
  early but still incur launch/grid work. The existing Strata compact adapter is
  restricted to its host-free prefix and is not a ready-made fix for Maya's
  host-served path. Remove the guaranteed empty work in the phase-1 repair while
  preserving row/group mapping, split jobs above eight columns, arithmetic and
  CPU/helper publication. The resulting full-model speed effect is unmeasured.
- **Inherited DSpark stale prefix context.** DSpark allocates `window + 64 = 4160`
  ring positions, writes `position % ring`, and reads attention history by the
  same modulo address without a logical-position tag. After a 6000-token prompt
  with drafting active and one emitted token, a second request sharing the first
  1000 tokens plus one new token reuses that target prefix. It rewrites position
  1000 but leaves ring slots 0-999 holding old positions 4160-5159. If the first
  new sample is non-EOS and more output is requested, the next draft at 1001
  attends to those wrong historical rows. No settings-change reset applies.
  This establishes a source-level state defect, not a measured generated-answer
  failure. It is present in the baseline and must be repaired under R3 before
  affected prefix-reuse measurements.

The router's absent recovery call in the larger slow path was also examined.
That is not a current native-serving defect: native loading enforces chunk <=16
and grouped dispatch, `forward`/`forward_cands` honor the limit, and larger
`forward_rows` calls are broken into supported rows. The selected path reaches
the window implementation where recovery exists. Revisit it only if those
restrictions are lifted; no T=17 model test is justified for this dormant path.

The remaining donor-coverage question is scoped: the 5090 packed-kernel screen
compared packed versus earlier decoding within the same new INT8/DP4A arithmetic,
and the six-record service screen used the packed route. They did not compare
against a native-FP4 tensor-core contender at MiniMax geometry. The assigned base's
`wide_expert.cuh` is a real RG16/TMA donor, but its 64-column tiles and registered
model geometries require adaptation; they do not establish a narrow-window win.
Keep the existing service evidence and address only that credible alternative in
R4 after the restart repair/remeasurement gate.

Local source references: [shared build sources](../../CMakeLists.txt),
[helper ownership and waits](../../src/mm3/mm3_native_helper.cu),
[primary adapter](../../src/mm3/mm3_native_experts.cu),
[routing and backfill integration](../../src/mm3/mm3_model.cu),
[router](../../src/mm3/mm3_resident_router.cu),
[request prefix reuse](../../src/mm3/mm3_main.cpp),
[draft context](../../src/mm3/mm3_dspark.cu),
[ring access](../../src/mm3/mm3_kernels.cu),
[DeepSeek helper](../../src/ds4/ds4_model.cu),
[Infernix executor](../../../Infernix-minimax/src/ops/offloaded_sparse_moe/cpu/peer_executor.cpp),
and [tensor-core donor](../../../Infernix-minimax/src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh).

Externally verified donor files:
[Fast-HSD cutoff](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/fast_hsd/core/truncation_verification.py)
(blob `5759095ac68ae723d3e79eeaec4af7f9217b2600`),
[Fast-HSD verification](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/transformers/generation/utils.py)
(blob `80b22686ac4c2fe2aaedf025a93fb7a666108c34`),
[FOMOE CAR](https://github.com/pmerolla/fomoe/blob/main/src/car.c)
(verified blob `8ec84fc5276e0e65bf216d6b87cdb3985bcfa745`) and its
[active recovery caller](https://github.com/pmerolla/fomoe/blob/main/src/gpu_kernels.hip)
(verified blob `02cd78920f50e24de9877fbbe050f5eb9423fbf8`). Local Strata routing
files match blobs `6611bceb0ffecad9a808bee0bf931776b71bc942` and
`79a7581d45441b4b73056e85685eacbd4f582e1b`.
