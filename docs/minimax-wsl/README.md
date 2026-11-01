# MiniMax M3.1 Flash on WSL2: central plan

Status: dedicated bases established; shared official download in progress, 2026-11-01. Use
MiniMaxAI/MiniMax-M3.1-Flash, including its official DSpark weights. The interrupted
Unsloth copy was stopped and its destination deleted at the user's request.
Fast-HSD is selected as the source for relaxed speculative acceptance. Each proposal
has its own fixed base below. Final strategy/composition selection remains open;
base setup does not establish a working MiniMax NVFP4 implementation.

**This file is the single current authority** for requirements, base assignments,
decisions, work status, experiments and [completion](#execution-and-completion)
across all three repositories:
`/home/mdwbr/project-maya/docs/minimax-wsl/README.md`.
The three proposals contain subordinate technical designs and exit evidence.
The evidence note and provenance records are supporting material. Each repository's
`AGENTS.md` is an entry point to this plan, not a separate work order. Keep current
decisions and status here; do not maintain parallel active plans in the other bases.
User instructions govern changes to this authority. Superseded instructions and
donor-project contracts do not override the MiniMax scope.

## Starting the full effort

Start the primary agent in **`/home/mdwbr/project-maya`**, branch
`minimax-wsl/proposal-2`. A sufficient task is:
**"Complete the full three-proposal MiniMax WSL effort defined by AGENTS.md and the
central plan, starting with Proposal 2 integration."** The documents define the
work; the launch prompt does not need to repeat them.

The primary agent owns the full effort across all three assigned bases.
**Proposal 2 is the first integration target; the smaller Proposal 1 and 3
comparisons remain required.** Expand the required-work accounting here before
implementation, resolve those comparisons at their defined scopes, and finish
the selected composition through the completion contract. A Proposal 2 milestone
does not close the overall task. Perform each implementation in its assigned
repository, use the Terra delegation policy when delegation is worthwhile, and
coordinate shared changes and model runs through this central record.

Source/accounting and build work can begin while the shared download finishes;
model loading waits for its verified readiness below.

## Dedicated bases

| Proposal | Working repository | Branch | Starting source | GitHub fork |
|---|---|---|---|---|
| 1: cooperative experts | `/home/mdwbr/Infernix-minimax` | `minimax-wsl/proposal-1` | Local GLM implementation `bae2bf11231283f178e3bbc93cdafc763e9535d9`, plus its captured working changes | [mdbro/Infernix](https://github.com/mdbro/Infernix) |
| 2: residency-aware routing | `/home/mdwbr/project-maya` | `minimax-wsl/proposal-2` | This Maya checkout at `6ecbd3cbc5330b57f70cd5cf8a5c42eb8ef950cc` | [mdbro/project-maya](https://github.com/mdbro/project-maya) |
| 3: speculative stage pipeline | `/home/mdwbr/strata` | `minimax-wsl/proposal-3` | Strata `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`, plus pinned Maya MiniMax/DSpark donor source | [mdbro/Strata](https://github.com/mdbro/Strata) |

These are independent Git repositories, not linked worktrees. Proposal 1 was cloned
without shared object storage and includes the donor's 16 modified tracked files
and two untracked files as copied working changes. Its
[source-state manifest](../../../Infernix-minimax/docs/minimax-wsl/source-state/infernix-local-manifest.json)
and patch preserve that starting state. The original `/home/mdwbr/Infernix-glm5-local`
and other existing Infernix worktrees are read-only references for this project.
Do not edit their code, plans, `AGENTS.md`, branches or running processes.

Proposal 3 carries a [Maya source snapshot](../../../strata/reference/maya-minimax/README.md)
at the same Maya commit as Proposal 2, including the MiniMax/DSpark module, packing
tools, parser reference, tests and licenses. It is donor code awaiting a port into
Strata's stage/window runtime, not a completed build integration. Preserve source
attribution and compare against the recorded revision when adapting it.

`origin` points to the owner's GitHub fork in each base. `upstream` is a fetch-only
reference with pushing disabled. Keep work on the assigned proposal branch.
The [base provenance record](base-provenance.json) describes creation, not current
requirements or runtime qualification. Existing upstream README/maintainer guidance
describes inherited products; it does not add requirements to these custom bases.

## Sub-agent delegation

Carried over from the [GLM delegation guidance](/home/mdwbr/Infernix-glm5-local/AGENTS.md#sub-agent-delegation). This section owns the policy for all three MiniMax bases.

Use only **GPT-6.1 Terra** sub-agents. Create every sub-agent with a fresh context:
`model="gpt-6.1-terra"`, `fork_turns="none"`, and an explicit `reasoning_effort`.
Supply the task and relevant context yourself. Do not inherit the full conversation: that also
inherits the parent's model and effort instead of the required Terra selection. Never use Ultra.

These effort choices are starting recommendations, not measured Terra results on MiniMax:

| Effort | Decision difficulty and appropriate assignments |
|---|---|
| low | Small, fully defined, read-only work whose instructions determine nearly every decision: find call sites or extract specified log fields. Low must not edit code or documentation. |
| medium | Default for well-defined, routine work with clear boundaries and an established approach: fix a Python reporting tool, implement a specified CLI change, complete a bounded app workflow, apply exact documentation edits, summarize a supplied diff, or make straightforward visual/text corrections. Use medium or higher for edits. |
| high | Default for substantive Terra assignments involving interacting constraints and implementation judgment: scoped C++ changes across callers/tests, loader/runtime bug tracing, resource-ownership review, benchmark reconciliation, documentation updates from verified evidence, or reproducible local bug diagnosis. |
| xhigh | Rare, carefully bounded assistance with subtle numerical or concurrency questions: CUDA stream/event ordering, host/GPU lifetimes, quantization and cast boundaries, independent-oracle design review, or explaining a performance regression from traces and code. The primary agent must independently review the reasoning and evidence. |

Choose effort by the difficulty of the decisions and the consequences of missing something,
not task length or diff size. A fixed protocol may need medium; a twenty-line CUDA synchronization
change may justify xhigh. More effort cannot replace missing traces, representative measurements
or an independent oracle, and subtle Terra conclusions must not be trusted blindly.

Delegate sparingly, weighing the expected benefit against the primary agent's context-preparation,
review and repair time. Give each assignment a clear outcome, relevant context, constraints and
appropriate acceptance checks. Keep a lower effort when it already produces a complete, correct
result. If Terra repeatedly struggles despite adequate context, do the work in the primary agent
rather than automatically escalating to xhigh or max. Do not delegate tasks with lengthy waits,
such as compiling code or running models.

## Shared weights and data

All three proposals use **one shared copy of the official
[MiniMaxAI/MiniMax-M3.1-Flash weights](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)**:

| Item | Shared path |
|---|---|
| Official repository revision | `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e` |
| Canonical native-WSL checkpoint root | `/mnt/wsl/transfer/minimax-wsl/checkpoints/MiniMax-M3.1-Flash` (download in progress; not ready to load) |
| Main target weights, all proposals | All 48 official `model-*-of-00048.safetensors` shards and their index/config/tokenizer/template files |
| Matching shared draft companion | Official `dspark/` directory, including `model-dspark-00001.safetensors`, its index and configuration |
| Previous Unsloth source, retained untouched | `/mnt/e/models_ablit2/unsloth/MiniMax-M3.1-Flash-GGUF`; historical reference, not selected for execution |
| Generated layouts, when needed and space permits | `/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-1`, `proposal-2`, `proposal-3` |
| Large raw benchmark/profiler data | `/mnt/wsl/transfer/minimax-wsl/runs/proposal-1`, `proposal-2`, `proposal-3` |

The [official repository inventory](official-checkpoint.json) records the pinned
revision, file sizes and published LFS hashes. The complete repository is
252,446,157,672 bytes (252.45 GB), including the official DSpark companion. It uses
native packed NVFP4 routed experts, MXFP8 dense weights and supplied BF16/F32
tensors; the drafter also uses MXFP8/BF16. These replace the previous GGUF/Q8
assumptions. Preserve the supplied codes and scale semantics through layout
changes, including expert block/global scales and the native dense/draft formats.
Record any chosen activation-scale treatment as execution arithmetic.

The download uses `hf download MiniMaxAI/MiniMax-M3.1-Flash --revision
2f33a8b9ae49ce7b51c994df97166a7fd059bc2e --local-dir
/mnt/wsl/transfer/minimax-wsl/checkpoints/MiniMax-M3.1-Flash`.
Two disjoint file groups currently download into that same directory. The
[placement record](/home/mdwbr/minimax-wsl-data/downloads/official-20261101/placement.json)
records their filenames. **All checkpoint files are stored directly on transfer**;
there are no root-backed shards or cross-volume symlinks. The temporary root
portion was stopped, its completed/partial files moved into the canonical
directory, and its old directory removed after the user freed over 400 GB on E:.
An ordinary full-repository command at this canonical path can resume missing
files after the active download jobs exit.
Do not load until the download is complete and its inventory is verified. Vision
weights remain outside initial execution scope even though they are included in
the repository. Reuse common records and packs where practical; do not make a
full weight copy for each fork or choose a separate quantization per proposal.

The user supplied `E:\wsl\transfer.vhdx`, mounted read-write at `/mnt/wsl/transfer`,
as the large-data volume. Check filesystem free space, the VHDX's existing physical
allocation and Windows backing-drive headroom before allocating large data.
Deleted Linux files can leave reusable space inside an already allocated VHDX;
Windows free space alone does not determine the room available for new files.
Before the download, transfer had 549.18 GB available internally and its VHDX
occupied 586.94 GB on E:. E: initially had 135.42 GB free; after the user's cleanup,
585.36 GB was available, sufficient for the entire official repository even
without reusing existing VHDX allocation. Root's temporary model files have been
removed, restoring about 218.49 GB free inside its 2-TiB virtual disk.
A dynamic VHDX grows physically
up to its configured virtual capacity; it does not automatically raise that limit.
Recheck these changing figures for subsequent large allocations. The existing
transfer VHDX can grow into E: as required; no virtual-capacity resize, new volume
or WSL restart was needed. At the user's subsequent explicit cleanup request,
the disposable `infernix/glm5-fp4-local/build` tree was removed, reclaiming another
61,504,446,464 bytes inside transfer. Its model pack and experiment evidence remain.

The earlier copy to
`/home/mdwbr/minimax-wsl-data/checkpoints/MiniMax-M3.1-Flash-GGUF` was stopped and
the entire incomplete destination (108.85 GB) removed. Source sizes and mtimes
were unchanged. Do not resume that superseded copy. Keep weights, packs and raw
captures outside Git repositories, and use the transfer volume for large data
when its combined internal/backing capacity permits.

## Current work and status

This is the live work record for all bases. Update it as work progresses; supporting
files may contain measurements and source details, but not a competing task status.

| Work | Status | Required next result |
|---|---|---|
| Dedicated bases and agent entry points | Complete | Branches, fork remotes, independent Git storage, 18 copied Infernix delta files, 52 pinned Maya donor files and central-plan links verified. |
| Shared official native-WSL weights | Download in progress, entirely on transfer | Finish both file groups and verify all 73 repository files plus the pinned download metadata before loading. The superseded Unsloth copy is stopped and deleted. |
| Proposal 1 | Required, open | MiniMax model binding and efficient expert service in the isolated Infernix base; follow its exit evidence. |
| Proposal 2 | Required, open | Official safetensors/NVFP4/MXFP8 loading and kernels, adapted helper and live residency-aware routing in the Maya base; follow its exit evidence. |
| Proposal 3 | Required, open | Port the pinned Maya model/drafter into Strata and obtain a credible stage/window decision; the donor snapshot alone does not satisfy this. |
| Selected composition and delivery | Required, open | Complete the retained runtime, Fast-HSD, serving/streaming and the shared short comparison after the strategy decisions. |

No MiniMax build or inference result is claimed by this setup work. Base snapshots
and setup documentation are recorded on their assigned proposal branches. No
changes have been pushed.

The active download has a [completion check](/home/mdwbr/minimax-wsl-data/downloads/official-20261101/finish_download.py)
with a [log](/home/mdwbr/minimax-wsl-data/downloads/official-20261101/finish.log).
After the remaining download process exits, it checks every file's presence,
size, filesystem placement and pinned local download metadata, including LFS ETags.
It publishes `weight-download-verification.json` here and updates this plan's
download row on success. This checks the download inventory; it does not
independently rehash every weight file or establish model execution.

## Objective and proposal comparison

The objective is the highest achievable **warmed, emitted-token rate for one
interactive coding/agent request** on this computer. There is no fixed tokens/s
target. The three proposals attack different costs:

| Proposal | Main mechanism | Assigned base | Main uncertainty |
|---|---|---|---|
| [1. Cooperative expert execution](proposal-1-cooperative-experts.md) | Make the requested experts cheaper and balance their execution across CPU, 5090 and 3090. | The dedicated Infernix fork, with Maya model code and Basalt/Bloomery kernel donors. | Whether faster arithmetic and dispatch compensate for the native NVFP4 bank's larger records and smaller cache coverage. |
| [2. Cache-aware expert routing](proposal-2-cache-aware-routing.md) | Avoid expensive expert misses by substituting sufficiently plausible resident experts. | Maya's MiniMax engine, a helper adapted from original Maya DeepSeek or local WSL Infernix code, and actual Strata/FOMOE routing implementations. | How much exposed work disappears before reasoning/tool behavior becomes unacceptable. |
| [3. Speculative layer pipeline](proposal-3-speculative-pipeline.md) | Amortize expert reads over verification columns and overlap successive portions of one speculative block across GPUs. | Strata's stage/window runtime, with Maya's MiniMax and DSpark implementations. | Whether accepted tokens and useful overlap outweigh expert-union growth, 3090 service and discarded work. |

My preferred first investment is **Proposal 2**, alongside Proposal 1's small
real-record kernel/service comparison. Proposal 2 directly addresses the large
host-expert cost already observed here, and it exploits the newly permitted
routing approximation. Proposal 1 supplies a strong alternative if substitution
has too little useful freedom. Proposal 3 deserves an early, bounded discriminator,
but a full pipeline rewrite should follow evidence of useful block reuse and
overlap. This ordering is an engineering judgment, not a measured ranking.

The final engine may combine winning parts. Three complete independent engines
are not prerequisites to choosing a direction. A shared Maya control and a few
partial-layer implementations can expose the important differences first.

VRAM-aware expert selection remains a central part of Proposal 2 alongside the
selected Fast-HSD verifier. Router scores and ready 5090/3090 residency jointly
determine which experts execute; Fast-HSD then decides which draft tokens to
accept from the resulting target probabilities at temperature 1/top-p 0.95.
The two mechanisms can operate together. Verification uses that approximate
target's output, without an additional pass through the original expert choices.

Helper selection starts from the original Maya DeepSeek V4.1 implementation and
the local Infernix WSL implementation. The owner identifies the Windows MiniMax
helper as a rough derivative port of the former. Preserve its measurements and
useful observations, but do not make porting that derivative a prerequisite or
treat its implementation as the preferred architecture. All proposed execution
is inside WSL2.

## Requirements shared by all three

- Build specifically for this PC: Ryzen 9950X3D, RTX 5090 and RTX 3090 under WSL2.
  Specialize kernels, layouts, memory ownership and scheduling for these devices,
  their actual links and the selected MiniMax checkpoint. Hardware portability
  and generic compatibility fallback implementations are outside scope.
- Use the **shared official MiniMaxAI/MiniMax-M3.1-Flash checkpoint** in the
  [shared weight location](#shared-weights-and-data) for all three proposals.
  Preserve its expert codes/scales through layout changes;
  use its MXFP8 dense and supplied BF16/F32 tensors, and the official `dspark/`
  companion. CPU and 3090 kernels may use
  different activation/arithmetic paths without requiring another weight quant.
  Unsloth GGUF and EXL3 are historical comparison evidence, not required benchmark arms. Reopen
  checkpoint selection only for a concrete problem that materially changes this
  decision; there is no initial quant-shopping or requantization campaign.
- Efficient packed-weight execution on the CPU and 3090 is required work in any
  proposal that assigns expert computation to them. Reuse and adapt optimized
  implementations, with short real-record tests at MiniMax's dimensions for
  single-token decode and a relevant verification width. Include unpacking,
  scales, activation conversion, dispatch, transfers and concurrent service in
  the decision. Select the architecture using efficient implementations for this
  hardware and complete-service timing.
- Main text-model weights should average roughly **3.5 bpw or higher**;
  approximately **3.4 bpw is acceptable**. Use a parameter-weighted inventory of
  the actual weight precisions, with scales, metadata, vision and draft weights
  reported separately. A high-precision backup must not disguise lower-precision
  weights actually used for inference. Mixed precision is allowed.
- Speed first. Choose expert arithmetic independently for the 5090, 3090 and CPU
  using whichever is fastest in complete execution. The same expert may use
  different activation formats, accumulation precision, rounding, reductions and
  mathematical approximations on each device. Neither exact nor approximate
  cross-device expert-output equivalence is required; do not add agreement
  tolerances or arithmetic reconciliation. Keep the selected stored weights and
  weight-precision floor, and assess practical coherence/tool use with the brief
  shared quality screen. Reduced-precision KV is also allowed.
  Cache-conditioned routing is explicitly in scope. Keep all model layers and
  stored experts initially; expert substitution is not permanent model pruning.
- Keep target **temperature = 1 and top-p = 0.95** unchanged across
  speculative-decoding comparisons. Do not lower them to inflate draft acceptance.
- Use **Fast-HSD** for relaxed speculative acceptance with the existing DSpark
  drafter. Start with its SpecCascade target-relative cutoff: accept a draft token
  within the target's top-p support when `p(draft) >= rho * max(p)`. Compute `p`
  using the fixed target sampling settings; choose `rho` through the short
  resident tests. The changed output distribution is intentional. See the
  [acceptance design](proposal-3-speculative-pipeline.md#fast-hsd-acceptance)
  and [inspected source](evidence.md#e6-selected-fast-hsd-acceptance).
- Initial functionality: text, reasoning, useful tool calling, local serving and
  streaming. Tolerant parsing and bounded tool-call repair are allowed. Vision,
  video and long-context testing are later work.
- Use short occupied contexts initially: an 8K configuration and approximately
  0.5-4K-token inputs are proposed test settings, not a final context limit.
- Keep weights resident in DRAM/VRAM across compatible sequential tests. Startup
  cost is recorded separately and does not determine the inference winner.
- Review and reuse existing implementations before introducing new mechanisms.
  Adapting geometry, MiniMax mathematics and ownership interfaces is expected.
- These are new MiniMax requirements. GLM's exact-arithmetic promises, fixed
  30-tok/s objective, ten-request benchmark protocol and particular host-bank
  contract are not inherited. Existing GLM source, evidence and execution holds,
  including the separate P51 hold, remain untouched.

## Common evidence and resource model

The [evidence note](evidence.md) records inspected source, local results and online
reports. Important facts for choosing experiments:

- The 3090's recorded link is PCIe 4 x4, without CUDA P2P to the 5090. Use small
  activation/result handoffs and useful persistent weights; price every weight
  refill. Do not assume an NVLink-style tensor-parallel interconnect.
- The local EXL3 text model is 179.4 GiB; its routed-expert tensors total about
  168.8 GiB. The previous GGUF is 232.3 GiB with a 216.4-GiB NVFP4 expert payload,
  about 28% larger than the EXL3 bank. These are historical artifact measurements.
  The official repository is selected and totals 235.11 GiB including its draft,
  vision and metadata. Inventory its target/expert/dense components separately
  before using those figures for a memory ledger. Maya-S/XS are below the
  precision boundary and excluded as implementation candidates.
- The NVFP4 geometry estimate for one MiniMax expert is 30.375 MiB before
  auxiliary record metadata; verify the imported official record layout.
  With no expert cache hits, top-4 execution reads 6.763 GiB of expert payload per
  token across 57 MoE layers. This is a payload calculation, not measured PCIe or
  DRAM traffic. Real routing, caching and speculative reuse determine service.
- Existing Windows EXL3/helper results reach about 12.8-13.1 tok/s on their short
  test sets, with DSpark off. The profile change altered outputs. They are useful
  evidence about the experimental port, not a WSL control, a ceiling for the
  original helper design or a prediction for another quant.

Each implementation must produce a small physical-memory ledger before a full
load: dense/shared/head weights, expert records, KV, draft, scratch, staging and
host allocations. GPU cache capacity is the remaining capacity after the selected
execution buffers, not the card's advertised VRAM. Count host aliases once and
physical duplicate banks twice. Preserve Windows physical/commit headroom using
the existing local measurement approach; the GLM-specific 200-GiB application cap
is a reference budget, not an unexplained MiniMax prohibition.

Load the verified native-WSL copy at the shared weight location. Give each engine
its own pack metadata/configuration; the old pack contains an absolute Windows
path. Once an arm requires a different layout, convert it once, preserve the same
weights and reuse it. Check physical storage before writing a full pack. Do not
make a second full quant bank resident merely to make A/B switching convenient.
Prefer no steady-state SSD expert reads, and record any that remain. Native file
placement supports WSL loading; its warmed decode effect still needs measurement.

## Short experiments with resident weights

### Keep the expensive state alive

Within each base, build policy/kernel alternatives into one executable where practical. A
benchmark session owns the immutable weight storage and persistent device pools;
request state, cache policy and execution settings are separate. Change allowed
policy settings only at a joined request/round boundary. Existing read-once
environment switches need a narrow runtime parameter interface for this purpose.

Retain allocations and weights while resetting request state and applying the
declared cache-conditioning protocol. Cache-policy differences are outcomes to
measure, not something to erase by forcing identical final cache contents.
Record the initial profile and request history so later arms do not unknowingly
inherit more favorable conditioning.

Different executables, artifact representations and ownership layouts may still
require serial reloads. Keep file-backed pages or a reusable host bank when the
existing owner can safely outlive the execution state and memory permits it. Do
not promise VRAM persistence across process exit or build a general hot-reload
service first. Partial-layer tests should load a small representative collection
of real records once and reuse it through their candidate comparisons.

### Spend measurements on decisions

| Stage | Default scope | Decision |
|---|---|---|
| Source/accounting | Existing captures, real record sizes and selected source | Is the proposed saving available, and does the storage fit? |
| Partial-layer service | One or two informative miss/width cases using real records; add competing CPU/helper work when it matters | Is the mechanism worth full-model integration, or does a concrete implementation defect need repair? |
| Early full-model screen | Two short coding/agent prompts; about 128 emitted tokens each, after shared warm conditioning | Does the complete candidate improve warmed speed and remain coherent? |
| Close decision | One reversed-order pair or one named anomaly check | Is the observed difference sufficient to choose? |
| Retained candidate | Four short representative requests, including a changed-topic/return case, up to 256 emitted tokens each | Does the combined implementation retain its advantage and usable serving behavior? |

These are starting sizes and evidence levels, not a sequence to repeat after every
edit. Reuse valid results and choose fewer cases when they settle the question.
An inconclusive screen leaves its decision open; use the smallest useful repair or
additional case, not an arbitrary test-count limit as a negative result. Before a
run, record the decision it can change, why existing evidence is insufficient, its
candidate/control and conditions, and the action each likely outcome would justify.
Omit a run if it cannot change a decision or establish an unmet requirement.

Warming must be observed through actual service and clock/link conditions; one
arbitrary token count does not prove a warmed cache. A useful early
timing interval is the complete suffix after 32 emitted tokens through the request's
declared endpoint, while retaining whole-request time. Never trim an unfavorable
continuation to its matching prefix. A longer interval needs a named unresolved
decision; long occupied contexts are not part of this initial investigation.

Report pooled committed/emitted tokens divided by elapsed time, individual request
rates, and the worst result. Drafted/rejected tokens do not count as emitted work.
Keep prefill, API time to first token, whole-request latency and startup separate.
Report cache/CPU/helper work, missed-expert bytes, speculative acceptance and retries
only to the extent needed to explain the decision. Profile only an unresolved
cost; do not attach a full profiler campaign to every comparison.

### Minimal quality screening and repair

Reuse the answers generated for speed testing: inspect coherence/repetition, run
the tiny generated function's checks where practical, and exercise a small tool
round trip. After an aggressive change, add at most a few short coding/reasoning/
tool examples if the speed outputs do not cover its likely failure. No early
perplexity corpus, benchmark suite, long-context suite or cross-device bitwise gate.
Finite values, valid indices, completion and buffer lifetime still need focused
engineering checks; permissive numerical requirements do not justify broken code.

Start with the fast candidate and roll back the offending precision/routing/window
choice when these checks expose a material problem. Repeated gibberish, runaway
repetition or no usable tool calls cannot win on speed alone. An occasional task
failure is recorded rather than automatically triggering a large quality study.

Reuse Maya's MiniMax streaming parser and Strata's declared-tool recovery and
lenient JSON handling. Preserve string arguments and declared tool names. Try local
repair first, then at most one corrective model turn for the initial experiments.
Report raw success, repaired success and failure separately, and charge retries to
the tool workflow's latency. A repaired envelope does not establish that the tool
choice or arguments were semantically correct.

## Execution and completion

The GLM [execution lessons](../../../Infernix-glm5-local/docs/maintainer/archive/glm5_3-flash-helper-plan-2026-10-27.md#execution-completion-contract)
document agents declaring narrower milestones complete and weakening requirements
to fit the implementation. Carry over the completion discipline, using MiniMax's
requirements above: independent arithmetic on each device, deliberate routing and
acceptance approximations, brief quality checks and no fixed tokens/s threshold.
GLM's exact-output rules and five-prompt/two-pass benchmark do not apply.

1. **Account for the full required work before implementation.** Expand this central
   work record with each requirement or required decision, affected production
   routes/callers, owners and allocations, evidence needed to close it, current
   status and remaining work. Cover actual MiniMax dimensions and the ordinary
   and speculative paths that the selected design uses. A small table here is
   sufficient; do not create a second planning authority or a new
   experiment-management system.
2. **Keep the agreed requirements fixed.** Only the user can remove or weaken a
   requirement. A supported negative may close a named investigation under this
   plan, but cannot silently remove selected functionality such as DSpark/Fast-HSD,
   change the checkpoint/weight floor, lower temperature/top-p, or replace required
   efficient execution with a generic compatibility path. Ordinary implementation
   choices and necessary redesign within scope remain the implementer's work.
3. **Finish the selected production path.** Integrate loading, kernels, dispatch,
   resource planning, state, actual server/streaming callers, effective options,
   diagnostics and active documentation. Establish the requested, selected and
   actually executed behavior. An available kernel, unused option, private harness,
   successful allocation probe or plausible answer does not complete integration.
4. **Keep resource efficiency and useful concurrency in the deliverable.** Account
   for actual allocated bytes and cache displacement, selected arithmetic scratch,
   record lifetimes and the overlap claimed by the design. Unused duplicate banks
   or workspaces, disabled intended overlap, unfinished CPU/3090 kernels and unbuilt
   final changes remain open work. A final reply's limitations paragraph cannot
   waive them. Permissive arithmetic does not waive functional state, indexing,
   ownership or resource correctness.
5. **Follow evidence through to a defensible decision.** Develop promising screens
   through material improvements and integration. Repair defects or prototype
   limitations that could reverse a negative decision. Missing data, a failed
   control or a first slow implementation does not reject a still-credible design.
   Assess named complementary interactions: an isolated neutral/loss result can
   still justify a combined trial, and untested combinations are not winners.
   Resolve credible in-scope opportunities without an exhaustive search over
   hypothetical variants.
6. **Use evidence at the scope of the claim.** A component result establishes that
   component; allocation checks establish storage/access; runtime observations
   establish actual routing, service and overlap. The final combined server needs
   the shared short warmed comparison and brief coding/tool screen. Hit rate,
   accepted draft length, simulated savings and separate kernel wins cannot stand
   in for complete emitted-token throughput. Build/check affected behavior before
   dependent timings, reuse valid evidence, and stop measuring once the decision
   is resolved; then finish remaining integration. Do not expand this into GLM's
   qualification matrix or impose cross-device output-agreement checks.
7. **Continue through intermediate milestones.** Once implementation is authorized,
   continue required work after builds, screens and documentation checkpoints.
   Context changes, implementation difficulty and an interim speedup do not reduce
   scope. Honor explicit stops and preserve exact remaining work across handoffs;
   continue independent authorized work when one item is blocked. A temporary
   blocker is not a supported negative or permission to mark the task complete.

Record implementation, verification and performance outcomes separately:

| Status | Meaning |
|---|---|
| Required, open | The named implementation or decision still needs its closure evidence. |
| Implemented | Present in the selected production source; actual execution and benefit may remain unverified. |
| Verified | Evidence supports the stated behavior for the cited route and conditions; this alone need not establish speed. |
| Supported negative | Adequate evidence closes the named opportunity, including material limitations that could change it. |
| Blocked / unverified | The corresponding requirement remains unfinished; record what is missing and the next useful action. |
| Complete | Every applicable implementation, interface, state, resource and evidence obligation for that item is satisfied. |

Before declaring delivery, reconcile every required row with the selected source
and actual serving behavior. Measure the final composition because residency-aware
routing, device arithmetic, cache capacity and Fast-HSD can interact. Keep matched
controls and all per-request gains/losses, full declared intervals, resource costs
and failed attempts; do not change criteria or select a weaker control after a loss.
The retained-candidate check remains the short protocol above, not a large quality
or long-context campaign. Report the best demonstrated warmed rate and its scope,
not a global optimum or an invented hardware ceiling. An unresolved required row
makes the handoff partial; a completed investigation with no demonstrated speed
benefit is a negative performance outcome, not a faster-inference delivery.

## How to select a proposal

Start from the same short WSL control on the selected official checkpoint and
compare the cheapest decision-bearing piece of each strategy in its assigned base.
Select the retained strategy/composition from complete service, capacity and
implementation evidence. The bases above are user-selected; loader convenience
does not authorize replacing them. All routes need actual NVFP4 expert and
scale support before their full-model comparisons.
Run coherent groups of complementary changes; isolate a contributor only when it
changes selection, explains a regression or is needed for maintenance.

If Proposal 2 removes enough host work, repeat only the affected Proposal 1/3 cost
comparison: faster CPU kernels or larger speculative windows may then have less
value, or new overlap may become useful. A failing first prototype does not reject
an architecture when a named repair could change the result. Record the agreed
scope of architectural comparisons before implementation. Every required
alternative needs a defensible selection or supported negative at its stated scope;
this may require more implementation than its first screen. Mentioning an optional
idea does not require building it, and selection does not require three complete
engines or every conceivable variant. Conditional deferral leaves a required
decision open; an interim winner does not close it. Complete the selected
combination's production work after those decisions, including DSpark/Fast-HSD.

The selected deliverable includes its actual server/streaming path, stable resident
weight ownership, short-context speed result, light quality observations and
disposition of material costs. Report accepted exclusions separately from unfinished
required work; the latter prevents completion. Long-context qualification and
broader quality restoration remain explicit later work. No proposal currently
establishes a MiniMax speed ceiling or a guaranteed tokens/s result on this machine.
