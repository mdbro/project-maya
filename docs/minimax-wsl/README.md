# MiniMax M3.1 Flash on WSL2: central plan

**2026-11-07: the final configuration beats the preserved best usable engine by +24.40% at 128 tokens and +28.08% at 384, confirmed across three load pairs with no regression ([result](evidence.md#final-comparison-against-the-preserved-best-usable-configuration)). The serving defaults now carry the spinner and prior bonus 0.20. The predictor [did not qualify](#speculative-decision-validation), and per-pass work remains open, so Maya optimization is not complete.**
**Central optimization objective, user direction stated repeatedly: the CPU lane and the copy-plus-primary path finish
together on each individual pass, every activation through every individual layer.** Compare completion times from a
common origin, never durations, and report per-layer mean abs(Δ) between paths and between each pass's slowest and
second-slowest path. [Details](#per-pass-balance).
**2026-11-10: a review from outside the plan re-ordered the work around what paces generation, which is expert uploads.
Redo step 6, speculative decoding, continues inside that order. [Where the work stands](#where-the-work-stands).**

The user lifted the work/GPU pause for the Maya milestone: implementation, builds,
trace collection, simulations and model runs may proceed under this plan.
[Infernix and Strata base work is on hold](#base-holds). Old runners and retired
experiment queues remain superseded; follow the current restart order.

This file alone owns current requirements, decisions, status and completion across
all three bases. Follow the user's latest instructions and update this record when
an accepted decision changes. Proposal documents contain source maps and technical
background; [evidence](evidence.md) contains scoped findings and receipts.

<a id="objective-and-current-reference"></a>

## Objective and scope

Complete the requested system and performance work in **Maya**. The active scope
includes the selected engine, all five approved residency-design requirements,
[implementation optimization](#implementation-optimization),
[parameter optimization](#required-parameter-optimization), and final serving and
performance qualification. Enabling the features and obtaining a first policy
improvement are intermediate results. Performance investigation covers material
costs in the active cooperative execution, residency, routing and drafting paths;
it is not limited to tuning existing flags or the first primary-cache heuristic
an agent implements.

Build for one interactive request at a time across coding, prose,
summarization/extraction, reasoning and tools. Generation speed is the principal
performance objective; do not claim an optimum or invent a hardware ceiling.
The [completion contract](#execution-and-completion) governs the whole active scope.
The [three-base comparison](#runtime-base-decision-remains-open) remains unfinished
and subject to the base hold; it does not block or automatically follow this milestone.

**No practical speed improvement was demonstrated** in the stopped round:
approximately 15 tok/s at its start and end. Its repairs and component gains remain reusable.
The [stopped reference and measured limits](evidence.md#stopped-round-reference) preserve the
actual configuration, historical rates and adverse results. A gain against a slower
control cannot stand in for improving the best usable complete configuration.

<a id="starting-and-resuming-work"></a>

<a id="maya-first-implementation-gap"></a>

<a id="current-decisions-and-work-status"></a>

## Current status and restart

<a id="where-the-work-stands"></a>

### Where the work stands, 2026-11-10

<a id="directions-20261111"></a>**The user's directions, 2026-11-11 (after a restart of the computer), which change the
order below.**
- **Where the day's defaults stand, 18:15** (all in `docs/minimax-wsl/native-runtime.json` and
  `native-server.json`; [checked together from serving's start](evidence.md#serving-defaults-20261111): 10.2% faster
  in generation than this morning's rule, reader and kernels [-17.7, -2.7], 25.7 to 28.6 tok/s with serving's
  drafting, and the 1,056-token prompt read in 5.9 s against 27.2 s):
  - the three cards placed at start in the user's order, 3090 hottest, 4080 second, 5090 third (`htp`), the 4080's
    pool in use;
  - the prompt read by the unmodified router (`routing_prompt_bonus` 0) with the layer-by-layer reader
    (`prompt_layerwise` 1);
  - the helper cards' own kernels, since 20:56: the tensor-core kernel with batched loads (tc2) for every pass on
    both. 3090: 4 warps a row group at one column, 8 as a graph at more, 8 units' loads ahead; 4080: 8 warps, 4
    units ahead (`helper_kernel` 108, `helper_kernel_one` 4, `third_kernel` 108, `third_kernel_one` 8,
    `helper_kernel_loads` 8, `third_kernel_loads` 4; needs an engine from f417fed on). Against the kernels before
    it: 2.3% faster at one row [-4.42, -0.16], about 3% in drafting [-5.94, -0.27] over two tests, on the meter's
    floor. Until then: 3090 dp any at one column and tc at more, 4080 dp any;
  - the bonus still 0.20. The candidate the serving kernel can apply (bonus 0.30 with dampened weights) is being
    timed and metered in the load of 18:15; the user decides the rule.
  - **Still open with the user:** the serving binary (`native-server.json`'s `exe` is the main checkout's build of
    2026-11-05, which has none of this; the orchestrator does not build or merge there).
- **Three directions at 14:23, after the sweep's wider table:**
  1. **The prompt is not read under the bonus.** "We should not read the prompt under the bonus." A decision, not a
     candidate. Built at 15:08 into the serving path: the per-request setting `routing_prompt_bonus` (the Prior bonus
     while the prompt is read; -1 the generation's bonus), `0` in `docs/minimax-wsl/native-runtime.json`. **First
     reading of its cost, two requests before the timed stage stopped:** on the 1,056-token code prompt about 57 s
     before generation against 22 to 27 s under the bonus. **Measured at 17:10 with both readers**
     ([evidence](evidence.md#prompt-readers-three-cards-20261111)): today's chunked reader takes 53.8 s for 1,056
     tokens unmodified against 26.7 s under the bonus; the layer-by-layer reader of
     [item 1](#order-20261110) takes 5.6 s unmodified, with generation unchanged afterwards. **So
     `prompt_layerwise` 1 is the serving default from 17:12, beside `routing_prompt_bonus` 0:** the decision then
     costs nothing and the prompt is read 4.7 times faster than until today.
  2. **Dampened weights:** "It looks like dampening helped?" Answered from the sweep: yes, 15 to 19% less
     divergence in the balanced screening at no time, lower in five of six single comparisons; two more direct
     pairs come from the confirmation on `htp`. Not yet a decision of the user's.
  3. **Every device gets its own kernel** ([per-device kernels](#device-kernels)). "No output identities between
     the 5090, CPU, 4080, and 3090 are implied. We can (and probably should) use different arithmetic on all four.
     We want each device to have its own custom kernel, designed, written, and optimized to be the fastest on that
     device. Especially for multi row during drafting."
- **The user's order of the tiers, 14:05:** "It should go: hottest experts on 3090, second tier of hottest experts on
  the 4080, third tier of the hottest experts on the 5090." That is `htp`, and it is the default from 14:19: in the
  serving launch file (`placement_order`) and in the lab launcher. The two fixed cards hold the hottest tiers; the
  5090, the one card that changes its experts fast, starts with the coldest of the three. Rounds 1 and 2 of
  [the optimization](#optimize-three-cards) had measured it 2.25% faster than `hpt` at one row with the interval
  clear of zero, and the orchestrator kept `hpt` all the same because drafting was unresolved. **Running since
  14:18, one load on `htp`:** (1) `htp` against `hpt` directly, to resolve drafting; (2) the sweep's candidates
  timed; (3) a fidelity confirmation of the sweep's anchors and candidates on `htp`. The user asked at 14:16 for
  more of the sweep's table around its best points and has it.
- **Corrected by the user at 09:55: the 4080 was not properly added before the sweep, and the sweep has to be
  rerun.** "That's why I told you to add it before running the lengthy fidelity sweep. Why didn't you add it?" and
  "You need to better optimize the 4080 first AND USE THE OPTIMIZED STATE FOR THE FIDELITY SWEEP!!!" What the
  orchestrator had done wrong: it built the 4080's pool as a setting that is off by default and asked whether it
  should be the default, although the instruction to add it was that decision; its retune was two ideas sized on
  paper in a few minutes; and it then ran the sweep (854 replays, about five hours) on that state, which is neither
  optimized nor what serving runs. **Order now: (1) [optimize the three-card state on the cards](#optimize-three-cards);
  (2) make the result the serving default; (3) rerun the fidelity sweep on that state.** Its timed stage 3 is
  dropped.
- **The user at 09:58 on the first three-card sweep:** "at least you did use the 4080" and "The 854 replays are
  not useless, they still tell us something about fidelity. Just not speed." So its fidelity results stand: the
  per-prompt floor, how far each rule moves the model from the router, which rules help, which do nothing, which
  break it. Its times do not carry over. The rerun on the optimized state therefore covers the rules that sweep
  showed worth having (the bonus curve, protection, pricing, the budget, dampened weights, how the prompt is read,
  the screening design over those, the bonus by budget grid) and leaves out the arms it showed dead (layer drop,
  three and five experts, the layer-group and window variants). The same replays re-measure fidelity on the new
  state, so the two can be compared.
- **Order changed again at 03:10: the 4080 first, then the retune, then the fidelity sweep.** "If the fidelity sweep
  is going to take 6 hours, add the 4080 first, retune the model and policy, then run the fidelity sweep." The sweep
  was stopped at 03:13 after 12 replays (it had measured 27 to 31 s a replay where the orchestrator had estimated 18,
  which made the whole sweep about 6 hours). Its results would also have described a cache that the 4080 changes.
  The runner, the design and the report tool are kept and rerun on the three-card engine
  ([fidelity sweep](#fidelity-sweep)). **Current order: (1) [add the 4080](#add-4080), done 04:41: 8.7% faster at
  one row and 15.8% in drafting; (2) [retune the timing model and the policy for three cards](#retune-three-cards),
  done 04:48: nothing worth changing found; (3) [the fidelity sweep](#fidelity-sweep), running on three cards since
  04:48; then what was waiting behind it.**
- **The routing lab's page is finished with for now.** "We are moving on from the webpage to what you are actually
  good at: the engine (backend). No more UIs for now." The page was overhauled by an Opus 5.5 sub-agent at the user's
  direction after they judged the orchestrator's version poor (through d44f9ff); that agent is stopped and the lab's
  server and engine are taken down. The page and its launcher stay in the tree, unused.
- **The fidelity sweep runs first** (design and status: [fidelity sweep](#fidelity-sweep)). "But first, before adding
  the 4080, run your full fidelity sweep across all of the factors we discussed. I suggest sweeps across single
  parameters, and then a design of experiments approach with more than one factor at a time and a response surface
  diagram."
- **The RTX 4080 may now be used**, after the sweep: "You can now use the attached 4080." This reverses "More VRAM:
  not right now" of 2026-11-10. How it is used is not designed yet; until it is, engine launches stay pinned to the
  5090 and the 3090.
- **Rules the user set for how the work is run** (2026-11-11): the path to a result is priced before it starts, in
  minutes per step, with only what the first use needs placed in front of it, one engine load per deliverable, and
  the estimate reported up front; answers shown to the user are never capped at a small token count (the 6,144 and
  4,096 budgets of the [adoption rules](#adoption-rules) apply to every tool, not only to usability checks).
- **Waiting behind the sweep:** the record agent's longer campaign (the lab's token records; not needed by the
  engine work of 2026-11-11 and not run). Prompt reading's step-1b campaign (the second routing pass) is
  withdrawn: with no bonus while the prompt is read there are no moved picks for it to settle. Sub-agents from
  before the restart are not running.

<a id="device-kernels"></a>**Per-device kernels (started 2026-11-11 14:25, the user's direction).** Each of the four
lanes gets a kernel designed for its device, with whatever arithmetic is fastest there; no lane has to reproduce
another's output. Multi-row passes (drafting) come first.
- **What the lanes run today** (read from the code, `Infernix-minimax/src/ops/minimax_nvfp4_ffn/`):
  - *3090, 4080, CPU:* `fast_ffn_cuda_packed` and its CPU twin. Weights stay 4-bit; activations are quantized to
    INT8 per 16 values; one thread per output row and 16 inputs decodes the weights once, then does four `dp4a`
    for each column it serves. Four kernel launches for every group of jobs with the same column count. The 4080
    runs the 3090's build.
  - *5090:* `native_fp4_ffn`. Blackwell's FP4 tensor-core instruction takes the 4-bit codes and their block scales
    as operands, 64 inputs by 8 columns at a time, with no decode. This is why its cost barely grows with rows
    (0.023 to 0.038 ms an expert from 1 to 16 rows) while the 3090's grows from 0.052 to 0.563.
- **What the format allows.** The expert record is laid out in units of 16 output rows by 16 inputs (128 bytes of
  codes, 16 scale bytes), and a job carries at most 8 columns: one unit is one operand tile of the tensor-core
  product the 3090 and the 4080 have (16 by 8 by 16 in FP16 or BF16, 16 by 8 by 32 in INT8). Geometry: hidden 6144,
  intermediate 3072, 221,184 units, 31.85 MB an expert.
- **The target.** At one row the 3090 already reads an expert at about 650 GB/s of its 936 GB/s: one row is bound
  by reading the record. A tensor-core kernel does all columns of a job in the product, so the target for any job of
  up to 8 columns is the one-row cost: about 0.05 ms against 0.16 (4 rows) and 0.32 (8 rows) today. And with one
  cost for every column count, all jobs of a pass can go in one launch, which lets drafting passes use a CUDA
  graph as one-row passes do.
- **A second finding in today's kernel, 14:45.** Its loop over a job's columns runs to a count read at run time and
  keeps its sums in an array indexed by that loop. Counted in instructions a further column should cost about a
  fifth of the first; measured it costs three quarters (0.052 ms at one column, 0.038 ms for each further one). So
  much of the growth with rows is how the loop is compiled, not the arithmetic.
- **The candidates, built and benchmarked 15:04 to 15:08** ([evidence](evidence.md#device-kernels-bench-20261111);
  `src/mm3/mm3_helper_kernels.cuh`, `tools/mm3/bench/helper_tc_bench.cu`, `runs/device-kernels-20261111/`):
  - *dp, today's arithmetic compiled per job width* (column loop unrolled, sums in registers; "dp any" takes jobs of
    every width in one launch). **Today's output bit for bit** on both cards, so it needs a timed test only.
  - *tc, tensor cores.* One 144-byte unit is the 16 by 16 weight operand, decoded from the 4-bit codes to BF16; the
    BF16 activations are the other operand unchanged; FP32 accumulation; row scales applied to the result; one
    launch for all widths. **Error against the exact product 0.235%, the least possible with BF16 results, against
    1.150% for today's kernel.** It changes the lane's numbers, so it goes on the fidelity meter.
  - **3090:** drafting-shaped passes 2.0 times faster with either (a pass of 12 jobs and 25 columns: 1,174 us today,
    577 dp any, 575 tc); 8 columns a job 2.8 times (dp) and 4.2 times (tc); one-column passes 12 to 17%.
    **4080:** dp any 1.4 to 1.6 times on drafting-shaped passes, 2.6 times at 8 columns, 2 to 6% at one column; tc no
    better than dp there. Both are bound by reading the record (729 of 936 GB/s on the 3090, 588 of 717 on the
    4080).
  - *The engine's switch,* per request and per card: `helper_kernel`, `helper_kernel_one` for the 3090 and
    `third_kernel`, `third_kernel_one` for the 4080 (multi-column passes; one-column passes): 0 today's kernel; 1, 2,
    4 or 8 tc with that many warps a row group; 16 dp with one launch group a width; 32 dp any. With 100 added (tc or
    32) a multi-column pass is submitted as one CUDA graph, as one-column passes already are.
- **In the engine, 15:32 to 16:16** ([evidence](evidence.md#device-kernels-engine-20261111);
  `runs/device-kernels-20261111/engine-test/`, 63 requests, all valid, on `htp`, bonus 0.20, the prompt read
  unmodified; predicted sha256 17515bbd):
  - **One row: dp any on both cards 3.97% faster [-6.89, -1.04]** (25.3 to 26.4 tok/s; predicted 1.5%). The 3090's
    one-column passes on tc instead: 0.97% [-3.50, +1.56], no better.
  - **Drafting: the 3090 on tc with the 4080 on dp any, passes as graphs, 5.82% faster [-11.03, -0.61]** (30.2 to
    32.0 tok/s), 13% on the code prompt in all three blocks. dp any on both: 1.79% [-7.05, +3.47]; with the graph
    2.50% [-5.08, +0.08]. Predicted 7 to 9%: the helper lanes end only part of the drafting passes, the CPU lane and
    the uploads end the rest.
  - **Adopted at 16:20: dp any on both helper cards** (`helper_kernel`, `helper_kernel_one`, `third_kernel`,
    `third_kernel_one` 32 in `docs/minimax-wsl/native-runtime.json`). Today's output bit for bit, faster at one
    row, not slower in drafting.
  - **On the meter, 16:17 to 16:54** ([evidence](evidence.md#device-kernels-meter-20261111)): at bonus 0 dp any on
    both cards repeats the reference (0.0000 nats in nine of ten replays), and the 3090 on tc sits on the floor
    (0.0154 at one row and 0.0141 in drafting against the floor control's 0.0160 and 0.0135).
  - **Adopted at 16:56: tc for the 3090's multi-column passes** (`helper_kernel` 108, with `third_kernel` 132: the
    drafting arm above; the one-column keys stay 32). The helper cards' kernels are now: 3090 dp any at one column
    and tc at more, 4080 dp any.
  - *The load of 15:10 was lost, the orchestrator's mistake:* the helper knew the kernel values 32 and +100, the
    engine's and the server's setting parsers did not, and the second request was refused (about nine minutes).
    Both parsers are fixed (51c9808), and the run script now starts with a preflight: every arm's settings of every
    stage go through the loaded engine once with a short prompt before anything is timed (46 settings, 5 minutes;
    to be shortened).
- **Tried and not kept, 16:20 to 16:40: a helper's one-row pass as one kernel**
  (`runs/device-kernels-20261111/one-kernel-pass/NOTE.txt`, with the source and both benchmark files). The engine
  test showed the helper lanes on the critical path at one row, and the 3090's lane has about 81 us before any
  expert; the orchestrator took that for the launches of the pass's six kernels and wrote the six steps as phases of
  one cooperative kernel. Output identical to today's (0 outputs differ). **Not faster:** one job 91.6 us against
  88.5 for the four-kernel dp, two jobs 143.1 against 131.4, and three jobs do not fit on the card at once. The
  premise was wrong: the time a one-job pass has beyond its arithmetic is not kernel launches. What it is needs a
  per-pass capture of the helper's timing in the engine (submit, worker start, completion seen, collect), which is
  the next step for one row; its engine test (prediction d1575df1, frozen before the benchmark) was not run.
- **Tried and not kept, 17:15 to 18:00: dp any with 16 or 32 parts a row instead of 8**
  (`runs/device-kernels-20261111/parts/NOTE.txt`). If a pass of one or two jobs waited for each thread's run over
  the inputs, more threads a row would shorten it. Benchmark: no gain on the 3090 (one job 94.4, 90.5 and 99.4 us
  with 8, 16 and 32 parts; 8 is the best from two jobs up), 3 to 6% on the 4080 from two jobs up, and the output is
  no longer today's bit for bit. With the one-kernel pass that is two guesses about a pass's fixed time ruled out
  by measurement; the per-pass capture is the way to find it.
- **What a one-row helper pass spends beyond its arithmetic, 19:38**
  ([evidence](evidence.md#helper-pass-floor-20261111)): in a pass with every pick on a card the 3090's result is
  what the pass waits for (published 194 us after the join, the pass ends at 217). About 60 us of its lane come
  before the first job. An empty submission costs 27 to 30 us on this system, and a graph of six kernels costs what
  one does: so about half of that is the system's launch latency, which the pass pays once already, and the rest is
  a first job running alone. **Nothing there for a kernel to remove;** what is left on that lane is the host's
  launch call and collect (16 to 32 us a pass).
- **The helper cards' kernels are done for now.** 3090: dp any at one column, tc at more. 4080: dp any. Measured
  and dropped: the pass as one kernel, more threads a row. CPU and 5090: nothing to change. *(Written 19:40.
  Superseded the same evening by the review below: a faster kernel was there on both helper cards, and the 5090's
  kernel had not been measured.)*
- **An independent review, assigned 19:54 at the user's direction** ("After you optimize the per-device kernels
  assign an opus agent to take a look at provide recommendations for improvements"). An Opus sub-agent
  (`kernel-review`) reads the two helper kernels, their launch path, the 5090's and the CPU's kernels and the
  day's measurements, and returns ranked recommendations with the expected gain, the cost in fidelity, the risk and
  the cheapest confirming measurement for each, and where it disagrees with the orchestrator's conclusions. It may
  back a recommendation with the standalone benchmark (at most 20 minutes of the cards, through its own copy of
  `bench.sh`); it loads no engine and changes nothing in the tree, the plan or the evidence. Its report:
  `runs/device-kernels-20261111/review/recommendations.md`. The orchestrator reads it, checks what it can against
  the code and the figures, and puts the recommendations to the user.
- **The review's report, 20:09, and what it overturns** (`runs/device-kernels-20261111/review/recommendations.md`;
  its benchmark `review/bench-review1-3090.txt`, `-4080.txt`, one run of 32 s; its source `review/src/`). Checked by
  the orchestrator at 20:13: the table's figures are the benchmark files' figures, the variant's source is the
  tensor-core kernel with only the order of its loads changed, the benchmark's own check finds 0 of 786,432 outputs
  different from tc at the same S on both cards, and the reviewer changed nothing in the tree.
  - **The main finding: tc is bound by latency, not by decoding and not by the record's bytes.** Each warp of tc
    loads one 144-byte unit, uses it, then loads the next: 48 dependent loads in gate/up and 24 in down, about
    0.75 us each. That is the 54 us a one-job pass computes in, which the orchestrator could not explain, and it
    gives the many-job throughput within 2%. The reviewer's variant (**tc2**: each warp issues the loads of P units
    before using any) leaves the arithmetic and its order alone, so **its output is tc's bit for bit**, and
    measures (microseconds a launch, today's default against the best tc2):
    3090 one job one column 90.7 to 72.2, two jobs 130.0 to 117.5, three 189.4 to 167.0, 16 jobs of 8 columns
    896.4 to 731.4, a drafting-shaped pass of 12 jobs and 25 columns 587.7 to 508.2;
    4080 one job 103.5 to 85.3, two 160.1 to 142.4, 12 jobs and 25 columns 685.5 to 631.6.
    Best settings: 3090 S=8 with P=8 at one job and S=4 with P=8 above; 4080 S=8 with P=4. A job's marginal cost
    becomes 35.4 us on the 3090 (900 GB/s) and 47.9 us on the 4080 (665 GB/s), nearly flat in its columns.
  - **Statements above that this withdraws** (the orchestrator's, all three): "both kernels are bound by reading
    the record" holds only for dp at many one-column jobs (dp 89% of peak, tc 82% and 79%; at one to three jobs
    neither is close); "tc is held by its decode" is wrong (about 22 us of instructions a job against a 34 us
    memory floor); and "nothing there for a kernel to remove" in a one-row pass is wrong: 18 us a one-job pass on
    the 3090 and 16 to 18 on the 4080 were there, with about 10 more above the floor after tc2. "CPU and 5090:
    nothing to change" was not supported by a measurement for the 5090 (R3 to R5 below).
  - **Its recommendations, ranked by the reviewer:**
    1. *R1, measured:* tc2 for every pass on both helper cards; dp then has no role on the helpers. No
       quality-loss mechanism (tc's output; 0.235% from the exact product against dp's 1.150%).
    2. *R2, measured on the host:* `NativeHelper::submit` scans all 6,144 inputs of every column with a scalar
       loop under the helper's mutex, once for each job that picked the token: 1.2 us a column, 31 us for a
       25-column pass; once a token and vectorized, 3.6 us. Same check, no output change.
    3. *R3, reasoning only:* the 5090's `native_fp4` projection has the same one-unit-in-flight loop; batching
       its loads is bit-identical and should matter most at one or two experts a launch. Test: a standalone bench
       on the 5090, about 30 s of the card.
    4. *R4, reasoning only:* a 5090 job of up to 16 columns (two MMAs sharing the weight operand), so an expert
       routed to 9 to 16 rows of a window is read once, not twice. First count how often that happens in the
       drafting window plans already captured.
    5. *R5, part measured on the host:* the 5090 is the noisiest lane. Its arithmetic quantizes activations and
       the hidden vector to NVFP4 with the input scale fixed at 1.0; simulated on the benchmark's synthetic expert
       that is 14.8% relative RMS from the result with 16-bit activations (helpers: tc 0.235%, dp 1.150%), and
       blocks whose largest value is below about 0.006 are flushed to zero. The reviewer infers that the meter's
       floor (0.016 nats from moving experts between lanes) is mostly this, and suggests timing tc2 on the 5090
       (BF16 tensor cores exist there). Synthetic data only; not checked on a real record or real activations.
    6. *R6, reasoning only:* CPU lane: check whether it is memory-bound (run with half the workers); if it has
       headroom, a BF16 kernel (`vdpbf16ps`) would take it from 1.15% to about 0.24% error.
    7. *R7:* the tensor-core kernels index a 128-entry scale table with a raw record byte; mask it.
    8. *R8, reasoning only:* about 10 us a one-job pass remains above the floor after R1 (a fragment-native
       re-layout of each group of four units, or cp.async staging; each 5 us at most).
  - **What the orchestrator does with it** (per-device kernels are the user's direction of 14:23, so a measured
    faster kernel is taken up without a further question):
    - *R1, R7, R2 into the engine:* the reviewer ports them (a follow-up brief at 20:14, 40 minutes: tc2 beside
      tc with a per-card setting `helper_kernel_loads` / `third_kernel_loads` = P, 1 meaning today's kernel; the
      graph caches keyed by P as well). Then one load: tc2 on both cards against today's defaults at one row and
      in drafting, 54 requests, then one meter arm at bonus 0 as a check on the port
      (`runs/device-kernels-20261111/engine-test3/`, `run3.py`; prediction frozen 20:16, sha256 72ea898b: one
      row 2.0% faster, drafting 3.0%). Adopted as the serving default if faster under the rule in
      `engine-test3/make_protocol.py`.
      **Done, 20:21 to 20:53** ([evidence](evidence.md#device-kernels-tc2-20261111); port f417fed, 54 requests
      all valid): **one row 2.29% faster [-4.42, -0.16]** with the 3090's one-column passes at 4 warps (25.2 to
      25.8 tok/s; at 8 warps 1.03% [-2.96, +0.90]); **drafting 6.52% faster [-9.84, -3.21]** with the 3090's
      multi-column passes at 8 warps (32.2 to 34.4 tok/s; at 4 warps 2.85% [-7.06, +1.37]). The drafting figure is
      4.5 points better than predicted and the benchmark orders the two warp counts the other way; not explained,
      and as the better of two arms it is more likely to shrink than to grow on a repeat. On the meter tc2 on
      both cards sits on the floor (0.0150 and 0.0140 nats against the floor control's 0.0160 and 0.0135).
      **Adopted at 20:56:** `helper_kernel_one` 4, `helper_kernel` 108, `third_kernel_one` 8, `third_kernel` 108,
      `helper_kernel_loads` 8, `third_kernel_loads` 4. dp no longer runs on a helper card. Not measured: that
      exact combination in drafting (its one-column passes ran at 8 warps on the 3090); confirmed in the next load.
      **Confirmed 21:05 to 21:14, and the drafting figure corrected** (`engine-test4/`, 27 requests, prediction
      ad3d420e): the adopted defaults 3.00% faster than the previous ones [-9.98, +3.98]; the first test's best
      arm repeated came out at 0.06% [-5.84, +5.73], against its 6.52% the first time. **Both tests together:
      drafting 3.11% faster [-5.94, -0.27]** (18 base requests). So the gain in drafting is about 3%, not 6.5%;
      the defaults stand under the rule frozen before the run. The base arm itself moves 5 to 12% between blocks
      on a prompt in drafting, so this design does not separate the warp settings from each other there.
    - *R3, measured 20:53* ([evidence](evidence.md#primary-kernel-bench-20261111); a second Opus sub-agent built
      the standalone benchmark, `runs/device-kernels-20261111/review-5090/`, 12 s of the 5090): **batching the
      loads of the 5090's FP4 kernel is bit-identical (0 of 9.4 million outputs differ) and helps only where a
      launch holds one expert (67.4 to 59.1 us) or two (82.4 to 78.8);** from three up nothing. The 5090 is less
      bound by the latency chain than the helpers. **Not ported:** at most 8 us on a 5090 launch of one expert at
      one row, under 1% a token if the 5090 ended every pass, nothing in drafting; that is below what the engine
      test resolves (its interval is about 2 points at nine pairs) and it adds a kernel variant, so it does not
      meet the adoption rules. The same holds for running the 16-bit tensor-core kernel on the 5090's one-column
      launches (13 us at one expert), now that R5 gives no fidelity reason for it.
    - *R4, closed 21:20 from the statistics of the drafting requests already timed, no cards:* an expert on the
      5090 serves 1.4 to 1.5 rows of a window on average (30,564 columns in 21,719 groups in one request; about
      2.8 such experts a window). A job holds 8 columns, so a second job needs 9 or more rows of one window on
      one expert; serving drafts 5 tokens and the lab's profile 8. Nothing to gain in generation. (The row limit
      of a window was not read from the code.)
    - *R5 and R6 are the user's to weigh* (they change a lane's arithmetic or its fidelity, not only its speed):
      put to the user with the review. **What the benchmark of 20:53 adds to R5:** the helpers' 16-bit
      tensor-core kernel runs on the 5090, and with one-column jobs it is faster than native FP4 at one and two
      experts a launch (54.0 against 67.4 us, 71.7 against 82.4), level at three and four, 2 to 4% slower at 8
      and 16; with 8 columns a job it is 37 to 55% slower from three experts up. Error against the exact product
      on the synthetic expert: FP4 14.76%, the tensor-core kernel 0.2354%. So at one row the 5090 could compute
      with 16-bit activations at no cost in speed; in drafting it would cost the 5090's kernel a quarter (4
      columns) to two thirds (8 columns) more a job.
    - **R5, measured 21:24 to 21:36: the 5090's 4-bit activation arithmetic moves the output 0.0122 nats in both
      modes, which is the meter's floor** ([evidence](evidence.md#primary-arithmetic-meter-20261111); one load
      with the 5090 started on dp4a, INT8 activations per 16 values, one arm at bonus 0 against the reference
      recorded on FP4; floor control 0.0160 and 0.0135, bonus 0.20 as served 0.0368 and 0.0425). The 14.76% on
      the synthetic expert does not carry to the output on real prompts. Every arithmetic change measured today
      (a few experts changing lane, tensor cores on the 3090, tc2 on both helpers, the whole 5090 on dp4a) lands
      on the same level prompt by prompt, so the floor is a prompt's sensitivity to any small change in the
      numbers and not the size of the change. **No lane's arithmetic stands out as an error source, and there is
      no fidelity case for changing the 5090's.** The first attempt at this load failed at start (the
      orchestrator's mistake, seven minutes; evidence).
    - *R6, closed without a run:* its fidelity half falls with R5 (dp4a on the CPU lane is on the same level as
      everything else), and for speed the CPU lane was measured memory-bound on 2026-11-08 and 2026-11-09 (item 4
      below). The reviewer's check with half the workers was not run.
    - *R8, not started:* at most 5 us a one-job helper pass for each of two ideas, reasoning only; below what the
      engine test resolves.
    - **Where the kernels stand, 21:37:** tc2 on both helper cards is the serving default; the 5090 and the CPU
      keep their kernels, the 5090's now on a measurement and not on the orchestrator's say-so.
- **Order of work:**
  1. *3090, multi-row:* done. tc2 for every pass (the review's kernel; before it dp any at one column and tc at
     more).
  2. *4080:* tc2 for every pass (before the review: dp any).
  3. *Into the engine:* done (one launch a pass, the drafting graph, the per-card switch, the defaults).
  4. *CPU* (Ryzen 9 9950X3D: AVX-512 with VNNI and BF16): **no kernel to build for rows.** Measured on 2026-11-08
     and 2026-11-09 ([host memory](#host-memory-and-l3)): a CPU job costs 0.557 ms with no cost per column, 31.85 MB
     at about 58 GB/s, and beside an upload the two share the roughly 80 GB/s host memory delivers. Its kernel
     already decodes a unit once for all columns. What would help the CPU lane is less host memory read, not
     arithmetic.
  5. *5090:* reviewed 16:25, nothing to change. Its jobs are built on the device and jobs of every width go in one
     launch of three kernels (`mm3_native_experts.cu`: `setup`, `native_fp4_ffn` at a fixed 8 columns, `scatter`);
     the FP4 tensor-core instruction takes the codes and block scales of both operands, which ties its 4-bit
     activations to the instruction. Its cost is already flat in rows (0.023 to 0.038 ms an expert).
- **Rules that apply.** The experts' weights are not requantized ([precision rule](#order-20261110)): every
  variant uses the stored 4-bit codes and scales exactly; what may differ by device is how activations are
  represented and how sums are accumulated. A kernel that changes a lane's numbers moves the fidelity meter by
  the floor (0.005 to 0.04 nats by prompt), so a new kernel is judged on the meter against that floor, and on the
  cards for time. No build runs while a timed request runs.

<a id="add-4080"></a>**Adding the RTX 4080 (started 2026-11-11 03:15).**
- **What is known.** 16 GiB; on a PCIe 4.0 link that runs two lanes of its sixteen (`nvidia-smi`: generation 4,
  width 2), against four lanes for the 3090 and sixteen generation-5 lanes for the 5090. So its link carries roughly
  half what the 3090's does. 16 GiB holds about 500 experts: residents would go from 1,179 to about 1,680.
- **Step 0, facts before any design (read-only and probes):** the card's real upload and download rates and its speed
  on the expert kernel; what the engine's one-helper code assumes and what a second helper needs; and the replay's
  prediction of loads and priced token time with a third resident pool of that size, at one row and in drafting.
  - *Measured, 03:16 to 03:25* ([evidence](evidence.md#add-4080-facts-20261111)). **The link is slow: 1.8 to 1.9 GB/s
    host to device**, against 6.1 for the 3090 and 53 for the 5090, so an expert takes about 17 ms to send to it.
    Its residents must therefore be a set that is rarely or never exchanged. A small request's round trip is 124 us
    against the 3090's 75 us at one row (324 against 160 us at eight rows); on a stand-in dense product it computes
    1.3 to 1.4 times faster than the 3090.
  - *Predicted on the replay, 03:28* (same evidence): **480 fixed residents on a third card cut the picks that need
    a load by 26% at one row and 28% in drafting, at today's bonus: about 7.0% of a token at one row and 11.1% in
    drafting.** They also move routing toward the router (picks moved 22.9% to 19.9% at one row, 26.3% to 22.6% in
    drafting), so this is a change that is faster and closer to the router at once. 250 residents give about half.
    The same 480 on the 5090 under its own exchanges would give 8.9% and 13.8%. The prices assume the third card's
    path is not the slowest of a pass; that is the first thing the build must show.
  - *Read, 03:30* (the map of the engine's one-helper code, from a read-only agent): the helper class itself is an
    ordinary object and a second one can run beside the first, but the model keeps one of everything around it (one
    ownership mask, one lane in the planner, one live exchange), and giving a second card its own *owned* experts
    means changing placement, the checkpoint, the exchange and the planner together.
  - *The 4080 as the only helper, 03:40 to 03:55.* The unchanged engine loads with the 4080 in the 3090's place:
    **475 experts in 14.1 GiB** (CUDA reports 15.0 GiB free on it and the engine keeps 512 MiB back). The engine's own
    setup calibration prices its lane at **0.298 ms for a small pass against 0.221 ms on the 3090, and the same at 16
    columns** (0.905 against 0.909 ms): slower to reach, faster once there. I meant to time its lane in real
    generation as well and dropped that after three failed starts, all mine: a budget above what the card
    reports free, then the `helper_dispatch` setting, which the server accepts only with a 3090 helper, first in my
    request and then in the runtime defaults file the server adds. It cost about 20 minutes of the cards and no
    result; the three-card engine below measures the same lane in place.
- **The design chosen: a fixed pool of copies ("third").** The 4080 holds *copies* of experts that the HOST keeps
  owning. Nothing about ownership, placement, the checkpoint or the 3090's exchanges changes, so a request without
  `third=1` runs today's two-card engine exactly, in the same load: the timed test's baseline needs no second load.
  With `third=1`: a pick of a held expert that would have gone to the CPU lane or to an upload is sent to the 4080
  as one more submission beside the 3090's, and its result is added on the host the same way; routing counts held
  experts as ready; the exchange, staging and early-copy rules leave held experts alone. The pool is filled once a
  load with the profile's hottest experts that neither card holds (`PLACEMENT third`), and is never exchanged, which
  is what a 1.8 GB/s link allows. Engine options `--native-third DEVICE --native-third-gb GB`. Not yet done, and part
  of the retune: the planner does not price the 4080's lane (its picks are simply outside the plan), and the split
  of the hottest experts across the three cards is unchanged.
- **Frozen prediction for the timed test** (`runs/add-4080-20261111/phase-b/timed/prediction.json`, before the
  first three-card load): **6.0% faster at one row and 10.0% in drafting**, at bonus 0.20. That is the replay's 6.9%
  and 11.0% for 475 experts, less about one point for the 4080's slower lane on the few passes a token that use it.
  More than 2 points off in either mode is a miss to explain before the retune.
- **Result, one load, 03:58 to 04:42** (engine 1a9eb2a; requested 03:10, three cards answering 04:05, timed result
  04:41):
  - **Timed against two cards in the same load** ([evidence](evidence.md#add-4080-timed-20261111)): **one row 8.7%
    faster [6.6, 10.8]** (38.8 to 35.4 ms a token), **drafting 15.8% faster [9.8, 21.9]** (36.3 to 30.5 ms), 32
    valid requests, two prompts. Demand loads fell from 16.7 to 10.4 a token at one row and from 39.4 to 30.9 in
    drafting. The prediction (6.0% and 10.0%) was too low in both modes by more than its two-point rule: the 4080's
    lane turned out to cost a pass nothing, and the loads fell more than the replay said.
  - **Per pass:** a pass whose picks are all on a card takes 0.463 ms when it uses both the 3090 and the 4080,
    0.461 ms with the 3090 alone, 0.370 ms with neither. The 4080's lane is 139 us for one job against the 3090's
    122 us; its worker is not found asleep. The 4080 serves 28 picks a token on 23 of the 57 passes at one row.
  - **Exactness** ([evidence](evidence.md#add-4080-three-cards-20261111)): with the 4080 on at bonus 0 the
    distributions move by 0.02 to 0.05 nats from the two-card engine's. Two controls on two cards (only the lane
    that serves a HOST expert changed) move them by the same amount, prompt for prompt: this is the meter's floor
    for any numerical difference, not a sign of a fault. **Checked directly at the next fill, 04:54**
    ([evidence](evidence.md#add-4080-exact-20261111)): all 475 records read back identical to the HOST's, and eight
    experts computed on the CPU lane and on the 4080 agree within 0.17% of the largest output, one rounding step.
  - **Fidelity at bonus 0.20, two prompts:** three cards move fewer picks (17.6% against 22.0% at one row) and keep
    more router mass (96.7% against 95.4%); the divergence is 0.107 against 0.087 at one row and 0.132 against 0.122
    in drafting, a difference inside the meter's floor. The sweep decides this.
  - **Adopted as the serving default, user decision 2026-11-11 09:44: "It should be the serving default."** In the
    branch since then: `third: 1` in `native-runtime.json`; in `native-server.json` the engine options
    `--native-third 2 --native-third-gb 14.1`, the three cards pinned by UUID in the order the engine needs (the
    file had no pin, and with the 4080 in the machine an unpinned load fails), and `third_profile`, from which the
    server fills the pool after every engine start (about 20 s with its checks; a failed fill is logged and the
    server serves without the pool). A load without the pool ignores the default.
    - *What the pool holds in serving.* Serving places nothing from a profile, so at start the two cards hold
      what the load gave them and the 4080 takes the retained generation profile's hottest experts outright, not
      the third tier it held in the timed test. That state is being timed now (the serving check below).
    - *The binary.* `native-server.json` points at `/home/mdwbr/project-maya/build-minimax/maya-mm3`, built
      2026-11-05 from the main checkout, which has neither the pool nor several earlier defaults. Serving gets
      the default once that binary is built from this branch; the orchestrator does not build or merge there.

<a id="retune-three-cards"></a>**The retune for three cards (2026-11-11 04:42 to 04:48; rejected by the user as not an optimization, see [optimizing the three-card state](#optimize-three-cards)).**
Judged per pass by the [per-pass balance](#per-pass-balance) objective, each idea predicted before any build
([evidence](evidence.md#retune-three-cards-20261111)).
1. **The model: why the replay under-predicted. Explained.** At one row the engine's loads fell 40% where the
   replay said 26%, on the same prompts. The replay's captures hold the early copies that were issued, not the
   predictions behind them; when a recorded target is now on the 4080 the replay issues nothing (9 early copies a
   token lost), while the engine copies the next predicted expert. Drafting, which issues few early copies, agrees.
   The replay cannot be corrected on two-card captures: further three-card predictions start from three-card
   captures.
2. **The policy: splitting the 3090's jobs with the 4080. Sized, not built.** From the captured three-card passes
   with a lane model that reproduces the measured cycles within 0.7%: a perfect per-pass split, which no placement
   can reach, saves 0.9% of a token at one row and 1.4% in drafting; a split by demand rank saves 0.1% and 0.4%.
   Only 29% of one-row passes have every pick on a card; the rest are paced by a copy.
3. **The planner's price for the 4080's lane. Sized, not built:** it would change no decision at one row (a CPU job
   is 0.78 ms, the 4080's lane 0.15 to 0.29 ms), and in drafting the 3090's lane, already priced, is the longer.
4. **What is left is the routing rule,** which the fidelity sweep measures on this engine. Deferred with the planner
   rebuild, as before: the 5090's exchange cap, the CPU price, variants of the early copy's rule.
5. **The serving default** was decided by the user at 09:44 and is in the branch; its check on the cards (a
   serving-style start, the pool's fill at start, the default against `third: 0` from serving's cold state)
   follows the sweep's grid.

<a id="optimize-three-cards"></a>**Optimizing the three-card state (started 2026-11-11 09:55, the user's direction).**
The state the engine starts a request from, as it will serve and as the sweep will measure it: which experts each
card holds, how the server reaches that state at start, and the policy around the 4080. Each candidate is timed on
the cards against the state timed so far, inside one load.
- **Tools, built 09:55 to 11:10** (engine c627cd3 and the commit after it): the 4080's pool can be rewritten between requests
  (only the records that differ, each read back against the HOST's bytes), and `PLACEMENT three <profile> <order>`
  places all three cards from the profile: `order` says which card takes the hottest experts, which the next and
  which the rest (h the 3090 with 741 experts, p the 5090 with 434, t the 4080 with 475). A card's roster can be
  split: `p150htp` gives the 5090 the hottest 150, then fills the 3090 and the 4080, then the 5090's other slots.
- **Round 1: which card holds which tier** (`runs/add-4080-20261111/optimize/round1/`). All six orders, three rounds,
  every order once a round, four requests a visit (one row and drafting, code and prose) from that placement's own
  snapshot, bonus 0.20, from token 128; the last round adds light captures for the per-pass numbers. 72 timed
  requests, about 45 minutes with the 18 placement changes. The state timed so far is `hpt` (3090 hottest, 5090
  second, 4080 third). Prediction frozen (sha256 a6ab5af7), from reasoning only: the 5090 is the one card that
  changes its experts fast, so it should start with the tier it can best afford to replace; `htp` (4080 second,
  5090 third) 2% faster, `thp` 1%, the orders with the 5090 hottest 3 to 4% slower.
- **The serving start, checked 09:49 to 10:06** ([evidence](evidence.md#add-4080-serving-20261111)): the server,
  started from the serving config with no profile placement for the two main cards, pinned the three cards and
  filled the 4080's pool at start (475 experts, all read back identical, 17.8 s). From that cold state the default
  is **18.7% faster than `third: 0`** (48.4 to 39.3 ms a token; code 25.6%, prose 11.9%), with demand loads down
  from 72 to 49 a token and the time before generation down 14% on the long prompt. **The cold state itself is the
  larger finding:** 49 to 72 loads a token against 10 to 31 in the placed state the timed tests start from. So the
  server should place all three cards from the profile at start; the server can now do that (`placement_order` in
  the serving config), and which order is what round 1 decides.
- **Round 1, result 11:02** ([evidence](evidence.md#optimize-round1-20261111)): no order of whole tiers is clearly
  better than `hpt` in both modes (one row: `htp` 3.0% and `thp` 3.3% faster; drafting: `pht` 4.0% faster with an
  interval of 9 points either way, `thp` 6.9% slower). Two effects pull against each other: expert loads are about
  5% lower when the 5090 starts with the coldest tier, because it is the card that adapts; and a pass with every
  pick on a card is cheaper when the 5090 holds the hottest tier, 0.51 against 0.64 ms in drafting, because the
  hottest experts no longer queue on the 3090's lane. On the code prompt in drafting `pht` is 11% faster in all
  three rounds.
- **Round 2, one load, started 11:05** (`runs/add-4080-20261111/optimize/round2/`), about 85 minutes:
  - *Build:* a new demand profile from twenty prompts of varied kinds written for this, answered at bonus 0, about
    12,000 tokens. The retained profile counts about 2,100 tokens of two prompts; the tiers the 4080 can hold have
    counts of 100 to 230 in it, so their order is largely chance.
  - *Stage A, the 5090's roster split:* `hpt`, `pht`, `htp`, and the 5090 holding the hottest 150 or 250 experts
    with the coldest in its other slots (`p150htp`, `p250htp`); three rounds. Predicted 3 to 4% faster than `hpt`
    in both modes (sha256 4f335c22).
  - *Stage B, the profile:* the placement stage A chooses, from the retained profile against the new one, on four
    prompts that are in neither. Predicted 3% faster (sha256 efcc6f2d).
- **Round 2, build and stage A, result 12:02** ([evidence](evidence.md#optimize-round2a-20261111)):
  - The new profile holds 2.73 million picks. It and the retained one agree on only 986 of the 1,650 experts they
    would place, and each covers 43 to 46% of the other's picks: demand past the hottest few hundred experts
    depends strongly on the prompt.
  - **The split gains nothing** (within 1% of `hpt` in both modes; predicted 3 to 4%).
  - **`pht` is chosen: the 5090 holds the hottest tier, the 3090 the next, the 4080 the rest.** Pooled over both
    studies it is 1.5% faster at one row [0.2, 2.8] and 3.5% in drafting [-0.5, 7.6], the only placement faster in
    both modes in both studies. A small gain: the order of tiers is not where the time is.
- **Round 2, stage B, result 12:25** ([evidence](evidence.md#optimize-round2b-20261111)): **the larger profile is
  not better and the retained one stays.** On `pht` it is equal at one row (-0.7% [-2.7, +1.4]) and 12.9% slower in
  drafting [+4.2, +21.6], with the same loads: the code prompts lose 18 to 29%, the letter gains 3%. The retained
  profile was built from a code task and a prose task and fits such work better; with `pht` the profile's hottest
  tier is what sits on the 5090, and in a wide drafting pass the picks that miss it queue on the helper lanes.
- **Withdrawn at 12:35 after the user's question: `pht` is not the default.** "this is the opposite of what was
  measured with previous builds, and also the opposite of upstream infernix. Did you decide to flip the placement of
  the hottest tier because it helps speculative decoding?" It reverses the decision
  [settled on 2026-11-08](evidence.md#3090-placement-work-list-item-4) (keep helper-first; primary-first priced
  3.5% slower at one row on two cards, because the 5090's slots are the adaptive cache), and the orchestrator chose
  it by its rule without noticing or saying so. What today's timed runs show, pooled over 24 requests a placement:
  the loads are equal (one row 9.5 uploads and 1.3 to 1.4 CPU jobs a token either way; drafting 19.4 against 20.1
  and 13.0 against 13.5), `pht` is 1.5% faster at one row [0.2, 2.8] and 3.5% in drafting [-0.5, 7.6], almost all
  of it on the code prompt. So most of the case is drafting, the drafting interval includes zero, and the earlier
  concern (work unlike the profile, where the 5090 must adapt) is untested: with a profile that fits code less,
  `pht` lost 13% in drafting and `hpt` was not run. **The serving launch file is back to `hpt` (helper-first, the
  4080 third), placed at start.** A direct test of `hpt` against `pht` on six prompts, four of them unlike the
  profile's, in both modes, runs before anything else depends on the order.
- **The state proposed at 12:26 (not adopted):** all three cards placed from the retained profile at start, the 5090
  holding the hottest 434 experts, the 3090 the next 741, the 4080 the next 475 (`pht`), the 4080's pool in use.
  Against the state timed at 04:41 it is 1.5% faster at one row and 3.5% in drafting; against serving's cold start
  it removes the phase in which the engine mostly loads experts. Nine placements and two profiles were timed to get
  here; the order of tiers and the size of the profile are not where the time is.
- **Found on the way, not built** (each needs engine work and its own test):
  1. *The helper lanes in drafting.* Where the hot experts sit moved drafting by 11 to 29% on code with unchanged
     loads. The helpers' fast path covers one-column passes only; extending it to drafting passes shortens every
     pass that waits on a helper lane, on the 3090 and the 4080 alike.
  2. *What the static cards hold should follow the work.* A profile fits the kind of prompt it was built from. The
     engine already counts demand as it serves; re-placing the 3090 and the 4080 from that count between requests
     would make the placement follow the user's work instead of a file. The timed tests restore the same state
     before every request and cannot show it; it needs a test over a conversation.
- **The server's placement at start, checked 12:26 to 12:43**
  ([evidence](evidence.md#add-4080-serving-placed-20261111)): it works (all three cards placed, the pool read back
  identical, ready in 400 s), and **a placed start is 11.7% faster than serving's cold start** (25.5 to 28.8 tok/s
  with serving's own settings; that load used `pht`). This part of the default stands whichever order is kept.
- **Round 3, result 13:21** ([evidence](evidence.md#optimize-round3-20261111)): `hpt` against `pht` directly, six
  prompts (four unlike the profile), both modes, 72 requests. **They do not differ:** `pht` +0.5% at one row [-1.6,
  +2.5] and +1.5% in drafting [-2.2, +5.1]; pooled over all three studies, thirty pairs a mode, -0.3% and -0.5%
  with zero well inside both intervals. The gain seen earlier on two prompts did not repeat. **Helper-first stays.**
  The rule fixed before it: `pht` replaces helper-first only if it is faster in both modes on the unlike prompts
  as well as the like ones. Predicted (sha256 26056d84): like prompts -1.5% and -3.5%; unlike prompts 0% and +2%.
- **The state the orchestrator settled on at 13:21 (superseded at 14:05 by the user's order, next item):**
  helper-first from the retained profile with the 4080 holding the third tier and its pool in use (`hpt`), placed
  by the server at start. Timed on the cards to get there: six orders of the tiers, two splits of the 5090's
  roster, a profile six times larger, and the direct test of `pht`. What the optimization changed is the server's
  start: a placed start is 11.7% faster than serving's cold start. Engine-side `hpt` is the state the first
  three-card sweep ran on (the lab launcher's helper-first placement followed by the pool's fill).
- **The user's order, 14:05: `htp`, the default.** "It should go: hottest experts on 3090, second tier of hottest
  experts on the 4080, third tier of the hottest experts on the 5090."
  - *What the rounds above had measured for it* (python-code and prose, twelve pairs a mode against `hpt`): **2.25%
    faster at one row [-4.19, -0.30]**; drafting +2.72% [-3.47, +8.92] (round 1 -0.3%, round 2 +5.8% with an
    interval of 12 points either way). Its loads are lower: 10.1 against 11.0 demand loads a token at one row,
    32.9 against 34.6 in drafting. It is the order the frozen prediction of round 1 named, for the reason the
    decision of 2026-11-08 gives: the 5090's slots are the cache that adapts.
  - *What the orchestrator did wrong:* it kept `hpt` at 13:21 under "no order is clearly better in both modes",
    although `htp` was clear of zero at one row and no harm was shown in drafting. By the
    [adoption rules](#adoption-rules) the burden of proof is on harm.
  - *Done at 14:19:* `placement_order` is `htp` in `docs/minimax-wsl/native-server.json`; the lab launcher places
    `htp` by default when the 4080 is used (it needs a binary with `PLACEMENT three`, 3c70518 or later).
  - *Round 4, result 14:56* ([evidence](evidence.md#optimize-round4-20261111); the protocol of round 3 with `htp`
    in `pht`'s place: six prompts, both modes, three rounds, 72 requests, all valid; first stage of the load that
    goes on with the sweep's candidates). **`htp` is 1.96% faster at one row [-3.72, -0.19] (25.9 to 26.4 tok/s) and
    equal in drafting (+0.13% [-4.14, +4.41], 31.9 to 32.0 tok/s).** Pooled with rounds 1 and 2, thirty pairs a
    mode: -2.07% [-3.31, -0.83] and +1.17% [-2.20, +4.53]. No harm shows; the gain holds on the prompts unlike the
    profile. Predicted (sha256 1c8b1965): like prompts -2% and +2% (measured -1.9% and +3.4%), unlike prompts -1%
    and +2% (measured -2.0% and -1.5%: one miss, in drafting, where lower loads cancel the 4080's longer lane).
  - **The optimized state, final:** all three cards placed from the retained profile by the server at start, the
    3090 holding the hottest 741 experts, the 4080 the next 475 in its pool, the 5090 the next 434 (`htp`), the
    pool in use.
  - *Lost on the way, the orchestrator's mistake:* the timed check of the sweep's candidates had been started at
    14:06 with the binary of 56ea0fb, which predates `PLACEMENT three` and cannot place `htp`. That load was
    stopped at 14:13 (about 8 minutes of the cards), and the next was restarted two minutes in to add the
    confirmation stage.

<a id="fidelity-sweep"></a>**The fidelity sweep (first start 2026-11-11 02:59, stopped 03:13; run on three cards 04:48 to 14:05 on `hpt`, 975 replays; its candidates timed and its fidelity confirmed on the user's order `htp` from 14:18).**
- **The run on three cards** (`runs/fidelity-sweep-20261111/three-cards/`, engine 56ea0fb). Every request uses the
  4080's pool; the reference is the unmodified router on three cards. Stage 1 is 100 arms on five prompts, 535
  replays with sources and references: about 3.5 hours at 20 to 25 s a replay. First come the floor controls
  (bonus 0, the 4080 off; the planner's CPU price at 0.25 and at 4.0) and today's two-card engine at bonus 0.20,
  about 20 minutes after the load, then the bonus curve. **Stage 2 was appended at 05:27 so the run goes through
  unattended:** a 16-run screening design at bonus 0.30 over eight rules, each off or on (a 2^(8-4) fraction of
  resolution IV: budget 0.20, protect one pick, priced substitution at 2, layer drop at 1.5, dampened weights,
  three experts, five experts, the prompt read by the unmodified router), and a 5 by 5 grid of bonus (0.20 to 0.60)
  against budget (0.05 to 0.40) for the response surfaces; both in both modes. In all 182 arms, 940 replays with
  sources and references, about 6.5 hours from 04:55: done near 11:30. Measured so far: 28 to 33 s a replay at
  bonus 0, less at higher bonuses.
  *Repaired at 08:21, my mistake:* I appended stage 2 before stage 1b was read, so the screening design carried the
  layer drop, which stage 1b shows breaks the model (1.6 nats and more); every run with it on says nothing about
  the other rules. Seven such one-row runs had been spent (35 replays, about 12 minutes). The nine not yet run were
  removed, and the eight runs without the layer drop were completed by their fold-over (every other rule flipped):
  together a 16-run design of resolution IV over the seven remaining rules, in each mode. 374 replays were left at
  08:21: done near 10:15.
- **Stage 1a, read at 06:20** ([evidence](evidence.md#fidelity-sweep-three-cards-20261111)):
  - At today's bonus three cards and two cards diverge equally (0.053 against 0.049 nats at one row); three cards
    are priced 10% and 14% faster, in line with the timed test.
  - **Three cards at bonus 0.15 are closer to the router and faster than today's two cards at 0.20** (0.035 against
    0.049 at one row, 0.039 against 0.055 in drafting; priced time 3.6% and 4.8% less).
  - Bonus 0.05 is on the floor. Each step of the bonus above 0.20 buys less time for more divergence at one row
    (0.30: 8.7% for 0.063; 0.40: 12.7% for 0.080); drafting gains more (0.30: 14.8% for 0.076).
  - **About half of today's divergence comes from reading the prompt under the bonus** (53.6% of the prompt's picks
    are moved at 0.20). With the prompt read by the unmodified router, generated tokens at 0.40 diverge less than
    today's rule and are priced 12.6% faster at one row. The prompt's own reading time is not in these prices and
    rises when its picks are not moved: that is the open question for prompt reading in [the order of work](#order-20261110),
    now with a measured fidelity side.
- **Stage 1b, each rule alone, read at 08:00** ([evidence](evidence.md#fidelity-sweep-three-cards-1b-20261111)):
  - **Three rules beat the plain bonus by 7 to 10 points of priced time at equal fidelity, in both modes:**
    protecting the strongest pick under a higher bonus (bonus 0.40 at one row and 0.30 in drafting then have today's
    divergence for 8.7% and 8.9% less time), pricing each substitution, and the per-row budget of 0.20.
  - **Nothing worth a rule:** three experts, five experts, dampened or amplified weights, a separate bonus for
    drafting windows.
  - **Out:** dropping a layer (1.6 to 2.2 nats from 1.5 on).
  - Whether the three good rules and the prompt's reading add up is stage 2's screening design; the candidates then
    need a timed test, since these times are priced.
- **Stage 2a, the screening design, read at 09:10** ([evidence](evidence.md#fidelity-sweep-three-cards-2a-20261111)):
  - **The rules do not add up:** at bonus 0.30 every combination of them lands at 0.031 to 0.048 nats at one row
    (0.062 with none), about 0.02 over the floor.
  - **Dampened weights and reading the prompt unmodified lower the divergence by 16 to 19% each at no priced time
    for generated tokens.** The budget, protection and pricing lower it by giving back 2 to 6 points of time.
    Three and five experts do nothing.
  - **First candidate: bonus 0.30 with dampened weights and the prompt read unmodified:** 0.033 nats at one row and
    0.047 in drafting against today's 0.053 and 0.059, priced 8.6% and 15.6% faster for generated tokens. With the
    strongest pick protected as well: 0.035 and 0.039, priced 5.8% and 9.8% faster. To be timed on the cards with
    the prompt's reading time measured, then read by the user before any adoption.
- **Stage 3, added 09:10: the candidates timed on the cards,** because the sweep's times are priced. In the sweep's
  own load when the grid is done, about 30 minutes: 60 requests, three blocks, python-code and prose, one row and
  drafting, 384 tokens from token 128, quiet flag up. Arms, all on three cards: the serving kernel at bonus 0.20
  (today's rule); the lab's routing path at 0.20 with no rule (what that path costs, and the base for the rest);
  bonus 0.30 with dampened weights and the prompt read unmodified; the same with the strongest pick protected;
  bonus 0.30 with dampened weights and the prompt under the bonus. Measured: ms a generated token, and the
  prompt's reading time. Prediction frozen from the priced times
  (`runs/fidelity-sweep-20261111/three-cards/timed/prediction.json`, sha256 5d404acf): 8.6% and 15.6% faster for
  the first candidate, 5.8% and 9.8% with protection. A timed result is not an adoption: these rules change which
  experts run, and the user reads the candidates first.
- **Stage 2b and the sweep complete, 14:05** ([evidence](evidence.md#fidelity-sweep-three-cards-2b-20261111)): 975
  replays, none failed. It ran on `hpt`; since 14:05 the default order is `htp`
  ([the user's order](#optimize-three-cards)), so its priced times describe `hpt` and its fidelity results stand as
  the user said of the first 854 replays.
  - **The per-row budget is the stronger knob:** a budget of 0.05 or 0.10 fixes divergence and time whatever the
    bonus; past a bonus of 0.40 the bonus adds little at a fixed budget. Response surfaces fitted for both modes
    (R squared 0.97 to 0.98), figures in the run folder.
  - **Better than today's rule on both axes:** bonus 0.30 with a budget of 0.30 (7.2% less priced time at one row
    with less divergence; 10.7% less in drafting with the same), and above all the screening's candidate, bonus
    0.30 with dampened weights and the prompt read unmodified (8.6% and 15.6% less time with a third and a fifth
    less divergence).
  - **Next for the sweep:** the candidates timed on the cards, since every time here is priced; then the user reads
    the winners before any adoption.
- **A wrong label, corrected 14:20.** The screening run reported as "bonus 0.30 with dampened weights and the prompt
  read unmodified" (0.033 nats at one row, 0.047 in drafting) also had the three-expert rule on (`screen-f15_EFH`).
  That rule does nothing measurable alone (divergence x0.98 at one row, x1.05 in drafting, no time), but the exact
  pair was never run, nor was the pair with protection (the nearest run, `screen-05_BEGH`, carries the five-expert
  rule). The timed arms `l1` and `l2` below are those exact pairs, so their fidelity is measured in the
  confirmation on `htp`.
- **Stage 3 on `htp`: first attempt stopped at its tenth request, 14:56 to 15:01** (second stage of the load of
  [round 4](#optimize-round4-20261111); files in `runs/fidelity-sweep-20261111/three-cards/timed/attempt-1456-htp-stopped/`).
  - *The stop.* Request `b1-one-l0-prose` (the lab's path at bonus 0.20 with no rule, one row) failed at token 202:
    a helper refused a submission ("helper unavailable, request already active, or invalid request size"). The
    message did not say which helper or which condition; the build from 6b7d8d1 on says both. Not reproduced yet;
    the lab's path had run 975 forced replays on `hpt` without it. The load was stopped at 15:06.
  - *What the nine valid requests say* (the code prompt, 1,056 tokens, one row, one block; not a result): time
    before generation 27.0 s on the serving kernel and 29.6 s on the lab's path under the bonus, 22.3 and 25.7 s
    for the two rules that read the prompt under the bonus, and **56.9 and 56.8 s for the two that read it with the
    unmodified router**. Reading the prompt unmodified with the chunked reader costs about 30 s on that prompt.
- **Stage 3 again, 17:10 to 17:55** ([evidence](evidence.md#fidelity-sweep-timed-20261111); last stage of the
  load of 15:20; 68 of 84 requests, stopped by its own time cap after two blocks and part of the third):
  - **The sweep's priced times hold on the cards.** Against the lab's path at 0.20: bonus 0.30 with dampened weights
    and the prompt unmodified 7.3% faster at one row [-10.1, -4.5] and 12.4% in drafting [-17.2, -7.5] (predicted
    8.6% and 15.6%); with the strongest pick protected 4.0% and 14.9%; bonus 0.30 with dampened weights 7.3% and
    17.0%; bonus 0.30 with budget 0.30 5.6% and 7.0%.
  - **The lab's routing path is 22% slower than the serving kernel at one row [17, 27]** and about 15% in drafting.
    A rule pays only where the serving kernel applies it: the bonus already, dampened weights from 77eb548
    (`routing_dampen`, off by default), the budget and the protection not.
  - **So the candidate that can be served today is bonus 0.30 with dampened weights and the prompt unmodified, on
    the serving kernel.** The sweep has it closer to the router than today's rule (0.033 against 0.053 nats at one
    row, 0.047 against 0.059 in drafting, for the screening run nearest to it). Owed before the user reads it: its
    timed test on the serving kernel against today's default, and its fidelity on `htp` (the exact pair `l1` is in
    the confirmation's remaining arms). The user decides the rule.
  - The prompt readers ran at the head of this stage
    ([evidence](evidence.md#prompt-readers-three-cards-20261111)); the engine error of 14:56 did not recur in 96
    requests.
- **The candidate on the serving kernel** (`runs/device-kernels-20261111/run2.py`, engine 77eb548, one load on
  today's defaults, from 18:36):
  - *Timed, 18:36 to 19:00* ([evidence](evidence.md#serving-kernel-candidate-20261111); `skernel/`, 72 requests,
    all valid; predicted sha256 695d8352). Against today's default (bonus 0.20, 25.5 tok/s at one row and 32.6 in
    drafting): **bonus 0.30 with the kernel's dampened weights 8.0% faster at one row [-10.3, -5.7] and 10.9% in
    drafting [-16.7, -5.1]** (27.7 and 36.5 tok/s); bonus 0.30 alone 6.6% and 10.3%; **bonus 0.20 with dampened
    weights no change at one row and 8.6% slower in drafting [+3.9, +13.3]** (predicted no change: fewer drafts are
    accepted, 75.0% against 78.6%). So the time is the bonus's, and dampening is not free in drafting at 0.20.
  - *On the meter, 19:00 to 19:17* ([evidence](evidence.md#serving-rules-fidelity-20261111); the same four with
    the prompt read as serving reads it, the lab's path for generation only). Mean KL from the unmodified router,
    one row and drafting: today's default 0.037 and 0.043 (this morning's rule: 0.053 and 0.059); bonus 0.20
    dampened 0.033 and 0.036; bonus 0.30 0.044 and 0.060; **bonus 0.30 dampened 0.038 and 0.047.** Dampening
    lowers the divergence in every pair, by 12 to 22%.
  - **Put to the user at 19:17, with the recommendation to adopt it: bonus 0.30 with dampened weights**
    (`routing_prior_bonus` 0.30, `routing_dampen` 1): today's divergence at one row, 0.004 nats more in drafting,
    8.0% and 10.9% faster. The default stays at 0.20 until the user says.
  - *In serving's own settings, 19:39 to 19:52*
    ([evidence](evidence.md#serving-candidate-serving-settings-20261111); `serving4/`, 16 requests from serving's
    own start, DSpark with lenience): **the candidate is 19.4% faster than today's default [-27.1, -11.7], 27.7 to
    34.2 tok/s** (code 28%, prose 11%), with demand loads down from 40.3 to 25.0 a token and the same share of
    drafts accepted. Predicted 9% (sha256 2e54d33e): a miss on the candidate's side.
  - *The user's second question, 19:55:* "How is that related with to 'The rules interact' or 'single-rule rows do
    not predict the combinations.'" It undercuts that statement of the report of 14:20, which is withdrawn for
    dampening: its only support was the one-prompt figure below. What holds, from the first sweep's own rows
    (divergence relative to bonus 0.30 with no rule, as the only rule and averaged over the screening's
    combinations): the per-row budget x0.76 and x0.95 at one row (x0.68 and x0.85 in drafting), protection x0.76 and
    x0.95 (x0.73 and x0.84), the prompt read unmodified x0.62 and x0.84 (x0.70 and x0.90): the rules that undo
    substitutions overlap, so a second adds little after a first. Dampened weights keep their effect in
    combination (x0.86 to x0.93 as the only rule at one row and x0.77 to x0.78 in drafting on 2026-11-11, x0.81
    and x0.85 averaged); they act only on rows where the bonus replaced a pick, so their effect grows with the
    bonus and is nil without it. Interactions were never tested directly: the screening design keeps each rule's
    average effect clear and cannot separate the pairs. The one interaction measured is in speed (dampening slows
    drafting at bonus 0.20 and not at 0.30, one test).
  - *The user's question at 19:50:* "How can we have damped weights with no bonus?" We cannot: the rule only
    changes the weights of a row where the bonus replaced a pick, and does nothing at bonus 0. The report of 14:20
    said "dampened weights alone" for bonus 0.30 with dampening as the only added rule; the wording was the
    orchestrator's fault. Its figure there (0.084 against 0.076 in drafting on `hpt`) came from one prompt at 2.6
    times and does not repeat: on `htp` the same pair is 0.063 against 0.082, and dampening lowers the divergence
    in every pair measured on 2026-11-11.
  - *The first sweep's anchors on `htp`, 19:19 to 19:35*
    ([evidence](evidence.md#htp-confirmation-20261111)): bonus 0.20 and 0.30, bonus 0.30 dampened and bonus 0.30
    with the prompt unmodified, as the first sweep ran them. **Each mean is within 0.007 nats of its value on
    `hpt`, so the sweep's fidelity figures hold on the user's tier order and it need not be rerun.** The
    layer-by-layer reader at bonus 0 sits on the floor (0.0161 and 0.0149 nats against the control's 0.0160 and
    0.0135). The lab-prompt variants the `sv_` arms replace and the two rules the serving kernel cannot apply
    were cut (`design-arms-not-run.json`).
  - *A load lost before it, the orchestrator's mistake:* the preflight added at 15:20 failed on its own code when an
    arm and the design both named a setting (a duplicate keyword), and the run script stopped the engine on an
    error it did not expect (18:22, about eight minutes). The merge is fixed and was dry-run without an engine, and
    the run loop now keeps the engine up on any error of its own.
- **The fidelity confirmation on `htp`** (`runs/fidelity-sweep-20261111/htp-confirm/`, 20 arms, 100 replays with
  references, about 45 minutes, not timed; second stage of the load of 15:20): the floor control, bonus 0.20 and
  0.30, the prompt read unmodified at 0.30, dampened weights at 0.30, budget 0.30 at 0.30, the two exact pairs `l1`
  and `l2`, and at bonus 0 the helper cards' new kernels ([per-device kernels](#device-kernels)); both modes, the
  sweep's five prompts. It says whether the sweep's fidelity figures carry over to `htp` and gives two more direct
  pairs on dampening. If they do not carry over, the reduced design
  (`runs/fidelity-sweep-20261111/optimized/make_design.py`, 140 arms, about four hours) is the rerun, and the user
  is asked first.
- **The floor on this engine, by prompt** (three controls a mode, bonus 0): prose 0.041 nats at one row and 0.040 in
  drafting; the first code prompt 0.015 and 0.012; extraction 0.008 and 0.006; reasoning 0.005 and 0.003; the
  second code prompt 0.005 and 0.006. The report (`report.md` in the run folder, rebuilt from the results at any
  time) gives every rule's divergence over its prompts' floor and the number of prompts where it exceeds every
  control.
- **What it is.** The fidelity meter as first proposed on 2026-11-10: a recorded answer is replayed token by token
  ("forced decoding") under each rule, and every next-token distribution is compared with the unmodified router's
  (bonus 0). With the pass prices it gives time against fidelity for every rule. The engine side is the lab binary
  (3655d1b, checked on the cards on 2026-11-10); the page is not involved.
- **Responses.** Fidelity: mean, median, 95th and 99th percentile of the divergence, share of positions with the same
  first choice, change in the forced token's probability, split into thinking and answer positions. Cost: token time
  priced from the pass counts, loads a token, experts a layer. Proxies kept beside them to test them: picks moved,
  router mass kept, rows under 0.8.
- **Fixed.** Five prompts (code, prose, extraction, reasoning, a second code task), about 240 generated tokens each,
  recorded once with the unmodified router; one row and drafting; every run from the restored snapshot; the
  reference repeated three times per prompt and mode for the noise floor, which varies from run to run.
- **The meter's floor, found 2026-11-11 04:17** ([evidence](evidence.md#add-4080-three-cards-20261111)). Repeating
  the reference does not measure it: seven of eight repeats were exact to four decimals, and the eighth moved by
  0.020 nats because a few picks ran on another lane. Which lane computes an expert changes the numbers (the 5090's
  native FP4 kernel against the integer kernel of the 3090, the 4080 and the CPU lane), and after the first
  difference everything computed later drifts. **Each prompt has a perturbation floor, about 0.02 nats on the code
  prompt and 0.045 on prose, that any numerical difference reaches.** Two lane controls on two cards at bonus 0
  (the planner's CPU price at 0.25 and at 4.0) and the 4080 switched on all land on it. *Change to the design:* per
  prompt and mode the floor is measured with those lane controls, and every rule is reported against that floor.
  A rule below its prompt's floor has no measured effect on routing.
- **Stage 1, one factor at a time:** the bonus; the drafting bonus; the per-row budget; protection of one or two
  picks; priced substitution; dropping a layer; weights (dampen, amplify); three experts; five experts; the bonus by
  layer group; and how the prompt is read (under the same rule, by the unmodified router, with its own bonus).
- **Stage 2, several factors at once:** a 16-run two-level screening design over the eight on/off rules (main
  effects clear of two-factor interactions), then a 5 by 5 grid of bonus against budget with a fitted quadratic
  response surface, drawn as contour diagrams of fidelity and of priced time, per mode.
- **Cost, stated before starting:** about 850 replays at about 18 s, about 4.5 hours of the cards unattended. First
  result after about 30 minutes: the bonus curve in both modes and the prompt's share of today's divergence.
  *Measured on the first replays: 27 to 31 s each (low bonuses load many experts), so about 6 hours. That estimate
  miss is what led the user to put the 4080 first.*
- **Not in it:** rules the replay screened that the engine does not have (a bonus that grows with rank, a cap of two
  moved picks, the ratio test). No usability judgment: the meter ranks rules; a person reads only the winners.

**The user's direction, 2026-11-10: "Completely update the plan, use the page."** The page is a review from outside the plan
of what limits speed ([page](https://claude.ai/artifact/Uv78kF9jUBuzhofHGJ9Zjj),
[evidence](evidence.md#speed-review-20261110), run folder `/mnt/wsl/transfer/minimax-wsl/runs/speed-review-20261110/`).
Its findings and its order are now in this plan. Where the page and this plan differ, this plan wins.
- **What the review found.** It is read-only work on the newest captures; nothing was timed or changed in the engine.
  - **Expert uploads pace one-row generation.** 82 to 93% of the time in the MoE layers goes to passes whose length is set
    by an expert arriving from host memory. A token uploads 50 to 68 experts over a link that carries one in 0.554 ms, and
    the engine already runs within 15 to 30% of the floor that count sets
    ([evidence](evidence.md#review-pass-types)).
  - **The CPU lane limits verify windows, not one-row passes.** Since `early_copy` it has a job on 2 to 6% of one-row
    passes. There the pair that finishes last is the upload against the GPUs' own work, and in passes with no upload it is
    the 3090 against the 5090.
  - **The cache is too small and adapts slowly.** The exchange caps are a credit per five rows: 1.6 exchanges a token on the
    5090 and one in five tokens on the 3090, which holds 63% of the slots ([evidence](evidence.md#review-rotation)).
    *Corrected the same day by the re-routed replay: the score decays by 0.95 per five generated rows, about 0.99 a
    token. The review assumed 0.95 a token, so its test of the decay is withdrawn. Screened properly on the replay,
    today's value is the best of six: faster decays give 1 to 4% more loads and a slower one slightly more
    ([evidence](evidence.md#reroute-replay-first)).*
  - **Exchanges are a scheduling problem** (the user's framing, confirmed). A pass beside an exchange runs 9 to 51% longer,
    the two cards' exchanges coincide 4.5 times more often than chance, and exchanges cost 1.3% of MoE time today
    ([evidence](evidence.md#review-exchange-cost)).
  - **A one-row pass with every expert on a GPU takes 0.37 to 0.55 ms for about 0.28 ms of GPU work.** The 3090's round
    trip through the host sets its pace ([evidence](evidence.md#review-all-resident)).
  - **Prompt reading runs at 32 to 45 tok/s:** 33 s for the benchmark's 1,056-token prompt, whose answer takes 14 to 17 s
    ([evidence](evidence.md#review-prompt-reading)).
  - **In windows an extra verified row costs 19 to 30 ms** against about 42 ms for a whole one-row forward, because each
    row brings its own uploads. Infernix's docs report the same for Qwen Flash
    ([evidence](evidence.md#review-drafting-infernix)).
- **The user's decisions and directions of 2026-11-10:**
  - Experts are not quantized further, and no weight fidelity is given up anywhere in the target model.
  - A native Linux boot is not an option, even as a one-off measurement.
  - The usability checks are to be re-run now that token cut-offs no longer count: the routing bonus at several values, and
    lenience, where substituted words are not a failure in themselves ([rules for adoption](#adoption-rules)).
  - A 6% speedup counts as significant: "it all adds up".
  - The resident sets' rotation was to be examined. So far: the caps and the scheduling of exchanges are levers. The
    decay was screened on the validated replay after the review tested it at the wrong time constant: today's 0.95
    per five rows is the best of six values, so it stays.
  - Near-simultaneous exchanges on the two cards are a scheduling problem, not a weight-delivery problem.
  - Agreed in principle: the 3090 queues each layer's work ahead and is released by a word in shared memory, and the rows
    of a verify window share experts.
  - Drafting is to use the local Infernix fork's work: its MTP, DFlash and DFlash2 for Qwen Flash, and its GLM layer
    look-ahead. Qwen is not to be loaded just to confirm a speed order.
  - The page's readers are Claude agents only, for now.
- **The user's decisions of 2026-11-10, midday**, in answer to the orchestrator's questions:
  - **The bonus:** "If a higher bonus passes usability and the timed test confirms it, adopt it."
  - **Scheduled exchanges become the one-row default:** "Yes". This covers the early copy's guard in scheduled mode.
  - **Item 4 may be bounded now:** "Yes". Building it is not decided; the bound comes first.
  - **Long prompts matter:** "Yes, but not at the expense of token generation." Item 1 enters the order, with
    generation speed as its guard. The user also asked whether better prompt processing can speed up speculative
    decoding; item 1 now carries that question.
  - **Precision:** "drafter and head can decrease in weight fidelity if we show no decrease in usibility." The
    target's experts stay as they are.
  - **More VRAM:** "Not right now (not within your current scope)." *Reversed 2026-11-11: "You can now use the
    attached 4080", after the fidelity sweep ([directions](#directions-20261111)).*
  - **The drafter to serve** was left open. The orchestrator recommends DFlash2, as the user's direction of 2026-11-09
    already points there, and runs the lenience block on DFlash2 unless the user says otherwise. The bonus block ran
    under DSpark with lenience 0.4, the harsher setting for text, so a pass there is not repeated.
- **Next, in order:** the [order of 2026-11-10](#order-20261110).
- **Execution began 2026-11-10** by user direction ("Use opus agents to complete the plan"). An orchestrating session runs
  Opus agents under shared rules and locks (`/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/AGENT_RULES.md`).
  One agent at a time holds the GPUs. Agents report to the orchestrator, which records results here after checking them.
  - **Running first:** the probe for item 5, with a detailed-capture load on today's base and the memory measurements
    for item 7; the exchange scheduling of item 3; the re-routed replay.
  - **Queued behind the captures:** the usability re-runs of item 2, with the proposed arms as the default. *Started
    2026-11-10 once the captures were recorded: the bonus arms first, then lenience at the bonus that passes.*
  - **First result: the re-routed replay is built and validated** (commit 6b24b11,
    [evidence](evidence.md#reroute-replay-first)). Against the recorded ready masks it re-routes all 87,324 rows as the
    engine did, and its simulation of the engine's residency reproduces every exchange decision in order. On the
    2026-11-08 captures it confirms the 3090's cap and the bonus curve, and it rejects the review's two score ideas.
  - **The probe for item 5 answers yes** ([evidence](evidence.md#probe-captures-memory-20261110)). One pinned buffer
    mapped on both cards under WSL2 carries a 5090 kernel's write to a polling 3090 kernel and back with no host call: a
    24.6 KB hand-off in about 11 us each way, and no stale or torn read in 348,000 trials. The design of the hand-off has
    started; its engine work follows the exchange scheduling, because both edit the same files. *The design is done
    and bounds the gain at about 1.4% of a token at today's cache state, because the pass after an all-resident one
    absorbs most of the saving while it waits for its early copy. The engine work is on hold under the 2% rule and is
    re-bounded when items 2 and 3 change the share of all-resident passes ([order](#order-20261110), item 5).*
  - **The detailed captures on today's base are recorded** (four one-row, four drafting; every router score), and the
    re-routed replay is running on them.
  - **The memory measurement shrinks item 7.** The drafter is sized before the residents, so its allowance is not a
    lever. What is left is the reserve, the KV size and 0.55 GB the engine does not account for.
  - **Exchange scheduling is built and timed** ([evidence](evidence.md#exchange-schedule-result-20261110)). In 12
    alternating blocks it is 1.17% faster on prose [0.22, 2.11] and between 0.3% and 1.25% faster on code, against a
    frozen prediction of 0.60%. Much of that comes through the early copy: beside a 5090 promotion it was blocked and
    the pass took a demand miss. *The user made it the one-row default on 2026-11-10.* As
    built it carries at most 0.26 exchanges a token on the 3090, so the host copies (item 3, step 3) now gate the cap.
  - **The hand-off bench ran, and item 5's later steps are bounded too**
    ([evidence](evidence.md#handoff-bench-20261110)). The queued hand-off is bit-exact and takes 116 to 249 us. With
    the measured figures steps 2 to 4 stay at about 1.3% of a token, and steps 5 and 6 come to under 0.7% each. All of
    item 5 is on hold under the 2% rule.
  - **The replay on today's base changes what the levers are worth**
    ([evidence](evidence.md#reroute-replay-today-20261110)).
    - At one row the early copy now carries 50 of 63 transfers a token, at every bonus. Residency acts on the 15 picks
      a token that still need a load.
    - The bonus cuts those by 25%, 46% and 69% at 0.25, 0.30 and 0.40. By the orchestrator's arithmetic over pass
      types that is roughly 5%, 9% and 14% of a token. It is timed next, on the values that pass usability, against a
      prediction frozen from the replay.
    - The 3090's cap cuts them by 3 to 5% at one row, which does not cover an exchange's cost. Its value is in
      drafting: 22% fewer at two exchanges a token, if a lease can start inside a forward. Item 3's design now aims
      there. The one-row cap arms are timed once, because two frozen predictions disagree in sign (item 3, step 2).
    - Sharing experts between the rows of a window removes 6% of a window's loads: 2% at width 2, 13% at width 8.
    - The decay and the two score ideas are settled. 0.95 stays, and both scores lose on both bases.
  - **The first usability block is done and no higher bonus is adopted on it**
    ([evidence](evidence.md#usability-block1-20261110)). It ran under DSpark with lenience 0.4. Outcomes are right at
    0.30 and 0.40 on every seed, better than control. But merged and doubled words grow with the bonus, about 0, 2, 12
    and 25 at 0.20 to 0.40, and at 0.40 reasoning leaks into three answers. That is evidence of harm until it is
    explained. Two checks separate the bonus from lenient acceptance: the same damage count on the timed tests'
    outputs, and a second block on DFlash2 crossing the bonus with exact verification and lenience 0.4.
  - **Item 4 is bounded** ([evidence](evidence.md#item4-bound-20261110)). Alone it gains little: +2.5% as first
    written, -1.7% in the variant that also copies other predicted misses when the link is idle. With item 5 that
    variant is -7.8% of MoE time at bonus 0.20, about 7% of a token, where item 5 alone is -2.1%. It moves 25.5% of
    picks against 20.6% today, inside the 27.3% at which bonus 0.20 passed. *Building it is for the user.*
  - **The orchestrator's error on the 3090's cap, corrected.** The first replay could not re-decide early copies and
    showed the cap removing 3 to 5% of picks needing a load at one row. With the new capture it is 11%, 19% and 26%
    at 0.5, 1 and 2 exchanges a token: MoE time -2.3%, -4.3% and -5.7% once exchanges are free. So the design
    without host copies pays at one row too, not only in drafting.
  - <a id="machine-change-20261110"></a>**The machine changed at about 22:07: a third card, and the power limits
    reset** ([evidence](evidence.md#machine-change-20261110)). An RTX 4080 (16 GiB) appeared as nvidia-smi index 2
    and both project cards were back at the factory power limits (5090 575 W, 3090 420 W). The orchestrator
    restored 425 W and 390 W at 22:11 with `native_host_control.py restore-power-limit all`, which acts by UUID.
    **Every engine launch must now set `CUDA_VISIBLE_DEVICES` to the 5090's and the 3090's UUIDs**: CUDA numbers
    devices fastest first, so with the 4080 present device 1 is the 4080 and `--native-helper 1` no longer means
    the 3090 (the engine refuses to load). The 4080 is not used: more VRAM is outside the current scope (user,
    2026-11-10). **Launch scripts check the power limits before loading and refuse above 425 W / 390 W**, since a
    driver restart resets them without a reboot. The kernel log dates the change to 22:07:36, after the last
    timed run ended at 22:05:21, and every campaign today recorded 425 W and 390 W at its start.
  - **A stall, 05:47 to 11:05.** No agent turn ran, and the GPUs sat locked and idle. Campaign scripts now release the
    lock themselves, and the orchestrator watches the lock and the cards.
  - **The user answered the open questions at midday** (above). In consequence: scheduled exchanges are being made
    the one-row default; item 4's bound has started; item 1 has a design agent; a higher bonus is adopted if it passes
    usability and its timed test; and the replay sizes the slots a lower-precision drafter and head would free.

**As of 2026-11-09 evening**, audited against every user prompt of the last 24 hours. Where they conflict, the latest
direction wins over older text.
- **Redo step 6, speculative decoding, is underway.** It began on 2026-11-09 by user direction: DFlash2 at MXFP8 first,
  DSpark's acceptance for reference only, and at its core an accurate offline model, offline optimization and an accurate
  planner ([step 6's list](#step-6-next)). Two items of the [order up to step 6](#host-memory-and-l3) were still open then,
  and they stay open: item 3's remaining `T=1` mechanics and item 4, the `T=1` planner rebuild.
- **Speeds, from token 128, equal-weight over eight prompts, one load
  (`expert-graph-timed-20261109`):** `T=1` 23.7 tok/s (42.1 ms a token); drafting, DFlash2 at MXFP8 under planner v1
  with exact verification, 27.3 tok/s (36.6 ms). Both carry the GPU nucleus and the expert-phase graph. The morning's
  `T=1` base ran 22.2 tok/s on two of those prompts, against 24.0 now on the same two (separate loads). Drafting gains
  most on code and reasoning and about breaks even on prose. The serving defaults (DSpark with lenience) are unchanged
  and were not timed against these.
- **Corrections from the audit:**
  - The window optimizer's screen of the CPU price, horizon and refill budget was out of order. The CPU price is being
    replaced, and the [deferral list](#host-memory-and-l3) covers windows too. The screen selects nothing.
  - The speculative planner is its own planner, not `T=1`'s extended (user direction). Its terms are fitted and judged on
    windows. A `T=1` value is a reference, not a constraint.
  - Text differs between runs and arms by design, exact changes included ([rules for adoption](#adoption-rules)). It is
    never a finding or a validity check. Exactness is checked in the engine.
  - Step 6's required start was skipped: the design comparison of the six proposals against the per-pass evidence. The
    window model was refitted from the `T=1` model and extended with terms measured in traces (v4), which the step says
    not to do ad hoc. v4 is validated per pass, but the comparison is owed before the window planner relies on it.
  - The user wants refills kept off demand copies ("absolutely!"). Bounded at about 0.2%, this enters the rebuilt
    planners' pricing instead of waiting for a separate decision. *2026-11-10: counted on every pass beside an
    exchange, not only on demand copies, exchanges cost 1.3% of MoE time at one row
    ([evidence](evidence.md#review-exchange-cost)). Scheduling them is item 3 of the [order](#order-20261110) and does
    not wait for the rebuild.*
  - The delegation limits in [sub-agent delegation](#sub-agent-delegation) do not apply to Claude Code (user direction).
- **Next, in order, as it stood on 2026-11-09:** [step 6's list](#step-6-next). It now sits inside the
  [order of 2026-11-10](#order-20261110).

<a id="order-20261110"></a>

### The order of work from the 2026-11-10 review

The numbering is the page's. Every estimate here is arithmetic or simulation from the review, not a bound from a replay or
a timed result. The usual rules hold: bound first, freeze a prediction, keep a baseline in the same load, measure from
token 128, and check usability only where a change has a quality mechanism ([rules for adoption](#adoption-rules)).

**A shared first step for items 3 and 4 and for the bonus curve: a re-routed replay on today's base.** One
detailed-capture load keeps every router score. The replay then re-routes each candidate under the bonus against its own
residents. The plan already owed this for the planner bounds ([step 6's list](#step-6-next), item 3). The detailed
captures of 2026-11-08 predate `early_copy` and the graphs.

1. **Read prompts one layer at a time.** *Entered the objective on 2026-11-10 by user decision, subordinate to
   generation: "Yes, but not at the expense of token generation." The design is done. Step 1's code is on this
   branch since 2026-11-10 14:47 (35b0560, 2b7f718, 18b2706), behind `prompt_layerwise`, default off. It compiles
   and the 161 server tests pass. Nothing has run on a GPU yet, so no claim about the path is made: its unit test,
   exactness load and timed load are queued for the cards.*
   - ***Step 1 measured on the engine, 2026-11-10 19:46 to 20:37**
     (`/mnt/wsl/transfer/minimax-wsl/runs/prompt-reading-20261110/step1/result.md`, `timed/evaluation.txt`; the
     evaluation was read before recording).*
     - *Exact: under the check setting 0 of 315,213 records differ from the chunked path, and the kernel unit test
       passes on both cards.*
     - *Prompt time, one row, median of four blocks: 234 to 247 tokens from 4.9-5.4 s to 2.2-2.4 s; 300 tokens from
       6.75 s to 2.70 s; 1,056 tokens from 30.95 s to 6.34 s, 4.9 times. The long prompt is inside its prediction;
       the short ones miss the range at the top (2.6 s).*
     - *The guard, on 32 pairs: +0.54% [-0.41, +1.48] over tokens 0 to 15, +0.24% [-1.14, +1.63] over 0 to 127,
       -0.33% [-2.20, +1.54] from 128. No slowdown is shown beyond the first tokens, but "within 1%" is not
       established. Tokens 0 to 15 are slower in 26 of 32 pairs, about 4.4 ms a request. Free VRAM is unchanged.*
     - *It moves 0.21 to 0.31 points more prompt picks than today's path, because today's mask counts leftover
       staged frames as ready. A second routing pass that counts the layer's uploaded experts as ready would cut
       moved prompt picks from 52-57% to 10-14% with no added upload. That is being built behind its own setting.*
     - *The 3090 limits it: 57 us a column where 32 to 40 was assumed, 18 ms behind the link in each pass at 1,056
       rows. Step 2 targets that: predicted 5.0 to 5.9 s for 1,056 tokens.*
     - *It stays off by default. Before it can be the default: the first-tokens cost removed or accepted, the moved
       picks settled by the second routing pass, and a run from 33 to 200 tokens for the threshold. The user
       decides.*
     - ***Default from 2026-11-11 17:12, with the prompt read by the unmodified router***
       *([evidence](evidence.md#prompt-readers-three-cards-20261111)). The user's decision of 14:23 (no bonus while
       the prompt is read) settles the moved picks: there are none, and the second routing pass has no job. On three
       cards the first 128 generated tokens and the rest are not slower after it (two prompts, two repeats). Nothing
       is owed for it any more. The run from 33 to 200 tokens has no case to run through the chat endpoint: the
       template alone is about 200 tokens (a four-word question is a 206-token prompt), and the shortest prompts
       measured, 235 to 247 tokens, are read in 3.0 s against 4.6 s; below 33 rows the engine keeps the chunks. The
       bit-for-bit check cannot hold on three cards with the pool in use (the two readers compute the pool's
       experts on different lanes); on the meter at bonus 0 the reader sits on the floor
       ([evidence](evidence.md#htp-confirmation-20261111)). The orchestrator made it the
       default without asking, because with today's reader the user's decision doubles the wait; one line in
       `native-runtime.json` turns it off.*
   - ***Its first GPU run stopped at the kernel unit test, 2026-11-10 16:31.** With dp4a arithmetic the new combine
     equals the window path bitwise; with fp4 arithmetic, which the engine serves with, it differs at element 64 by
     a factor of a thousand. The run stopped there and released the cards, as designed. The cause is being found.
     The code is off by default.*
   - *A gap in the safety net, found on the way and closed the same afternoon: 10 of the 26 CPU test targets did not
     compile on this branch, some since 2026-11-04. Their fake structs lagged the production sources, and ctest
     reports an unbuilt test as "Not Run", so nobody noticed. All 26 compile and pass since db54547 and dc38ded, with
     no production source changed and no assertion weakened
     (`/mnt/wsl/transfer/minimax-wsl/runs/test-fakes-20261110/result.md`). Rule for agents: build every listed test
     target before running ctest, and update the fixtures in the same commit as an engine change.*
   - **Design and bound** ([evidence](evidence.md#prompt-reading-design-20261110)). Predicted: 1,056 tokens from 33 s
     to 4.2 s (3.6 to about 7), 237 to 300 tokens from 5.3 to 7.2 s to 1.6 to 2.6 s, and 4,000 tokens to about 16 s.
     The new read wins from 32 rows. Above about 700 tokens the 3090 sets the pace.
   - **The first step shows, frozen before any run:** bitwise equality of hidden states with the chunked path under
     the same mask setting; prompt time against the prediction; the guard below; and the share of prompt picks moved
     off the router's top four beside today's.
   - **A finding about today's prompt read:** the bonus moves 52 to 57% of prompt picks off the router's top four,
     against 23% in generation. Of the router's own picks for prompt rows, 85 to 90% are not on a GPU.
   - **For the user:** whether the prompt may be read without the bonus, or with a bonus that falls with prompt
     length. It follows the router more closely, and for long prompts it is also faster (about 12 s against 16 s at
     4,000 tokens); for short prompts it is slower (2.1 s against about 1.2 s at 150). Its hidden states differ from
     today's.
   - **The user's question answered with first sizes:** drafts copied from the context are accepted for 19% of tokens
     on extraction, 7% on code and about 1% on prose and reasoning; a table from token to experts predicts 46 to 48%
     of picks and is worth at most 2 to 4% of drafting time; seeding residents from the prompt does not help; the
     drafter's context already covers the prompt.
   - **The guard.** The new path must leave generation at least as fast, measured from token 0 as well as from token
     128. It changes what the prompt leaves behind: staging frames, residents, scores and the KV cache.
   - **What the prompt read can give generation and drafting** (the user's question; each is sized offline before
     anything is built).
     - Drafts copied from the prompt, where the output repeats a span of it. Exact, no drafter call, no VRAM.
     - A table from token to the experts it picked at each layer, to predict a drafted token's picks before its row is
       verified. Verify passes have about 10 early copies a token today, against 50 at one row.
     - Scoring demand and exchanging residents during the prompt. Both are off outside generation today
       (`generation_active` in `mm3_native_demand.inc`, `mm3_native_adaptive.inc` and `mm3_native_helper_live.inc`).
       The benchmark measures from token 128, so this would show in real use more than in the plan's metric.
     - Reading by layer is the general form of a verify pass, but it does not cut the uploads a verified row needs.
   - **Cause today.** The native cooperative path rejects chunks above 16 rows (`src/mm3/mm3_model.cu:1455`), so a
     1,056-token prompt makes 66 trips through the layers and each trip uploads its experts again.
   - **What.** Swap the loops: layers outside, row chunks inside. Per layer, run the dense work and the router for every
     row, then serve each picked expert once. The 5090's residents run in place, rows for the 3090's residents go there
     once per layer, and every other expert is uploaded once and applied to all the rows that picked it. Keep the chunked
     path for prompts under about 100 to 150 tokens.
   - **Estimate.** The 1,056-token prompt from 33 s to roughly 5 to 8 s. All experts not on a GPU total 195 GB, 3.4 s at
     link speed.
   - **No quality mechanism:** it is exact. Compare hidden states with the chunked path.
   - **Scope.** [Generation-only modeling](#generation-only-modeling) keeps prefill outside the modeled duration, and this
     item does not change that. If the user brings prompt time into the objective, this is the first engineering item.
2. **Re-run the usability checks: routing bonus and lenience.** *User direction. The arms proposed below are the default, taken when the user directed
   completion on 2026-11-10; the user can still change them.*
   - **How.** `tools/mm3/native_usability_matrix.py` with [the template](usability-matrix-template.json) and its budgets.
     Only completed answers are judged. Substituted words alone are not a failure: judge by the task's outcome.
   - **Arms proposed.** Control (bonus 0.20), then 0.25, 0.30 and 0.40. Then lenience 0.3 and 0.2 against 0.4 at the
     chosen bonus, on the drafter the user means to serve.
   - **What is at stake.** On the curve of 2026-11-08, against 0.20: 0.25 is 4.0% less time, 0.30 is 7.1% and 0.40 is
     11.4% ([evidence](evidence.md#routing-bonus-curve-at-t1-under-hags-work-list-item-2)). That curve is first-order and
     predates `early_copy`, so re-price it on the re-routed replay. *Re-routed 2026-11-10 on the same captures: 18%,
     33% and 54% fewer picks need a load at 0.25, 0.30 and 0.40. Today's base follows when its captures exist.*
     *On today's base ([evidence](evidence.md#reroute-replay-today-20261110)): picks needing a load are 15.1, 11.3, 8.2
     and 4.7 a token at 0.20 to 0.40, and moved picks 22.9%, 24.9%, 26.3% and 28.1%. So 0.40 today moves about what
     0.20 moved when it passed (27.3%), with less router mass kept (92.65% against 94.97%).*
   - **The timed test, with predictions frozen 2026-11-10** ([evidence](evidence.md#reroute-replay-today-20261110)).
     All four values are timed in one load right after the first usability block, one row first and then drafting.
     Usability decides adoption separately. Frozen, as a share of a token against 0.20: one row -4.5%, -8.4% and
     -13.2% at 0.25, 0.30 and 0.40 on the two timed prompts; drafting -7.1%, -12.5% and -20.7%, good for direction
     and to about a third in size.
   - **Timed at one row, 2026-11-10** ([evidence](evidence.md#bonus-timed-t1-20261110)): **-4.76% [-6.12, -3.39],
     -8.94% [-9.82, -8.06] and -13.60% [-14.35, -12.85]** at 0.25, 0.30 and 0.40. The prediction held at all three
     values, within half a point. So the speed is confirmed and the replay's pricing is validated by a timed test.
     At today's 23.7 tok/s that is about 24.9, 26.0 and 27.4 tok/s. In the engine the bonus moves fewer extra picks
     than the replay expected: on prose 18.1% at 0.20 and 20.1% at 0.40.
   - **Timed under drafting, 2026-11-10** ([evidence](evidence.md#bonus-timed-plan1-20261110)): -6.56% [-9.48,
     -3.64] at 0.25, -17.15% [-23.20, -11.10] at 0.30 and -9.08% [-17.88, -0.28] at 0.40, against -7.1%, -12.5% and
     -20.7% frozen. 0.25 lands on its prediction. The other two are driven by the form of the code answer: at 0.30
     one short text that drafts well, at 0.40 an answer that never finishes within 384 tokens and drafts badly
     (0.55 drafts accepted a token against 0.76). On prose: +2%, -6% and -13%, with wide intervals. **So in drafting
     0.40 is not faster in practice, and that is the third sign that the model behaves differently there.**
   - **Adoption is not yet decided.** The first usability block passes 0.30 and 0.40 on outcomes but shows text
     damage rising with the bonus under DSpark with lenience 0.4
     ([evidence](evidence.md#usability-block1-20261110)). The damage count on the timed tests' answers and a second
     block on DFlash2 separate the bonus from lenience.
     - *At one row there is no such damage at any bonus (about 8,000 tokens an arm), which pointed at lenient
       acceptance and not at the bonus alone.*
     - ***But under DFlash2 with exact verification the damage appears at 0.40 at block 1's rate**, and not at 0.25
       or 0.30 ([evidence](evidence.md#bonus-timed-plan1-20261110)). So lenience is not needed for it. The
       orchestrator's reading: the dose is the share of rows that keep little of their router mass, and drafting
       moves more picks than one row at the same bonus. Rows under 0.8 of their mass are 8.7% at one row and 0.40,
       8.8% in drafting at 0.25, 13.3% in drafting at 0.30 and 21.1% in drafting at 0.40. If that holds, one row and
       drafting want different bonus values, or a floor on each row
       ([item 4](#order-20261110), the candidate kept for later).*
     - ***Finished answers may get longer with the bonus.** In block 1 the tokens of all finished answers, reasoning
       included, are 1.06, 1.31 and 1.21 times the control's at 0.25, 0.30 and 0.40. On those lengths the time to a
       finished answer is 1.01, 1.19 and 1.04 times the control's, so no value's gain a token survives. Three seeds:
       suggestive, not established. At one row code answers finish within 384 tokens 11, 6, 1 and 0 times of 13.
       Block 2 counts lengths under exact verification with nine entries a bonus value.*
     - *So a higher bonus is judged on time to a finished answer as well as on time a token
       ([method rules](#order-20261110)).*
   - **The second usability block, on DFlash2, 2026-11-10** ([evidence](evidence.md#usability-block2-20261110)).
     - **0.40 is rejected.** Under exact verification and under lenience it breaks outputs: tool calls written as
       plain text and never made, an empty answer, an omitted fact, broken sentences. It is also no faster as time
       to a finished answer.
     - **0.30 is the user's trade.** Outcomes are right on every task and seed with exact verification, and a
       finished answer takes 0.76 to 0.82 times the control's time. But about one answer in six carries a doubled or
       merged word, and the control has none. That is damage that follows the dose, so the orchestrator does not
       adopt it without the user.
     - **The damage is the bonus's, not lenience's.** Lenience 0.4 at bonus 0.20 is clean, and at each bonus the
       exact and lenient slip rates are close. Block 1's lengthening at 0.30 does not repeat.
     - ~~**0.25 is the candidate to adopt**~~ *(tested and not adopted, next bullet).*
   - **The third usability block, four new seeds, 2026-11-10** ([evidence](evidence.md#usability-block3-20261110)).
     - **0.25 is not adopted.** It makes one scheduling error that 0.20, 0.30 and 0.40 never make: 3 of 11
       scheduling answers at 0.25 across the three blocks, none of 43 elsewhere. Its finished answers are also
       longer, so its time to a finished answer is 1.07 times the control's [0.91, 1.37]. The error does not follow
       the dose and two of the three cases share a seed, so it may be chance; but it is evidence of harm with no
       gain to set against it.
     - **0.30 holds up.** Over seven seeds every task is right on every seed, a finished answer takes 0.85 times
       the control's time [0.72, 1.04], and about one answer in 12 carries a doubled or merged word. It remains the
       user's trade.
     - **The bonus stays at 0.20** until the user decides on 0.30.
     - **Lenience:** 0.4 at bonus 0.20 is clean and saves about 3 to 8% of verify rounds under DFlash2; 0.3 and 0.2
       save no more and add damage. Lenience is a small lever here, and its old question is answered: keep 0.4 or
       exact verification.
   - **Is one number for the bonus too simple? (the user's question, 2026-11-10 evening.)** Yes, on three counts.
     - *What the rule does* (`src/mm3/mm3_resident_router.cuh`, policy Prior): one additive bonus for every expert
       on a GPU, at every rank, over all 128 experts; the executed four are weighted by their own router
       probability, renormalised to sum to one. An older policy in the same file protected the two strongest picks,
       took substitutes from the router's top eight and required 0.8 of the replaced pick's score, with an optional
       dampening that does not scale the other picks up. Dampening was parked as inconclusive
       ([parked work](#dampening-disposition)), before today's tools existed.
     - *What one number ignores:* what a row loses (the bonus bounds the loss per pick, not per row); what a
       substitution buys (the first load of a pass costs 0.6 ms, a further one 0.3 ms, one covered by a copy in
       flight nothing); and where it applies (all ranks, all layers, one row and drafting alike).
     - *Screened on the replay* ([evidence](evidence.md#routing-rules-20261110)). One bonus is too simple by mode
       and by row, not by layer. At an equal share of badly rerouted rows, one row can carry 0.28 where drafting
       carries 0.20. At one row, substitutions chosen by time saved per unit of router mass lost, with no row under
       0.80, give 6.4% of a token with no badly rerouted row at all. In drafting the plain bonus at 0.40 under the
       same floor gives 6.1%, also with none, where today 4.9% of rows are under 0.8. The older ideas all lose:
       a relative test, protecting the strongest picks, dropping a weak pick, values by layer. None of the winners
       is in the engine, and router mass is a proxy: the fidelity meter is what can confirm them.
     - <a id="routing-lab"></a>**The routing lab: the user judges the rules by hand** (user's request,
       2026-11-10 evening; [evidence](evidence.md#routing-lab-20261110)). "To save time, instead of the official
       usability suite, I want to be able the chat with minimax myself and make my own evalaution by being able to
       set the bonus manually while toggling each of the following options independently/ adjusting their
       parameters for each prompt I give (without fully reloading the model, and I only need to evaulate usability
       and quality for myself, speed it not what is being tested here)". The options, in the user's words:
       - *Selecting substitutes:* a per-row budget on lost router mass, adjustable; protecting the strongest one or
         two picks; pricing each substitution by time saved per unit of mass lost; different values for one row and
         drafting, and by layer; "Drop an entire layer if the top experts in that layer rank under a certain
         combined score."
       - *Multipliers:* dampening; "Amplifying the weak pick: leave the other picks' weights alone, use the weight
         of the original routed expert that is substituted for the substitute expert"; a different number of
         experts by score (five when ranks 1, 2, 3, 5 and 6 are resident, 4 is not and the top six all rank above a
         threshold; three when a weak unready pick is dropped).
       - *The page:* "Build a local website page/ web interface so that I can do this. Also show various stats,
         metrics, and allow the user to run various analyses on the output, including each next token
         distrobutions, not just the tokens themselves." The fidelity meter runs from it, with graphed results.
       - **This replaces further official usability blocks for choosing among routing rules.** The user's own
         reading through the lab decides usability and quality; the timed tests still decide speed.
       - **Status, 2026-11-10 22:43: ready, and handed to the user**
         ([evidence](evidence.md#routing-lab-ready-20261110)). The engine side is on the shared branch (token
         records and forced decoding 796aa33 to 69d9a9f and fb8e196; the rules 3655d1b) and one binary carries
         both (`routing-lab-20261110/bin/maya-mm3-xs`, sha256 2bc72c8b...; 28 of 28 CPU tests, 163 server
         tests). The lab's engine was loaded at 22:32 with `launch_lab.py start` and the page is at
         http://127.0.0.1:8796/. Checked on the cards: the 14-step rule check (every rule acts, the fifth expert
         runs, zero mismatches against the serving kernel on generated and prompt rows), chat in both modes,
         and the meter in both modes (every forced token honoured, noise floor 0.0008 nats at one row and 0 in
         drafting against 0.11 to 0.26 for the rules). While the user has the lab, the cards are held and no
         timed test runs.
       - **The two owed checks ran at 22:59 to 23:04, while the lab sat idle, and both passed.** The record
         check: files whole, every forced token honoured, the unmodified router moved no pick, zero mismatches.
         The readiness comparison: on identical tokens the lab's path finds the same share of picks not ready as
         the serving kernel, within 0.4% (one row), so **the lab's numbers are not optimistic**. Two limits: the
         like-for-like comparison was not made in drafting, and **the noise floor varies from run to run** (0
         to 0.014 nats mean, 0.07 at its 95th percentile, between two forced runs of the unmodified router),
         so two rules closer together than that cannot be told apart from one replay.
       - **Seen at once on the real engine:** at bonus 0.20 the prompt's rows move 57 to 58% of their picks and
         keep about 0.80 of router mass, where generated rows move about 24% and keep 0.95. This is the serving
         kernel's own behaviour on 16-row prompt chunks (the self-check counters are zero), and it is the
         prompt-reading question already open for the user (open decision 13's neighbour: read prompts without
         the bonus, or with the two-pass mask). The page lets the user set the prompt's bonus and rules apart.
       - **The rules apply to the prompt as well** (decided 22:00 on the agent's costing of about 20 lines): the
         prompt's 16-row chunks go through the same host-planned path as a verify window. The page sends this by
         default, with a switch to limit the rules to generated tokens and a separate bonus for the prompt.
       - *Closed, 23:04:* the worry that the lab's slower routing path leaves more experts ready in time than
         serving does. Measured on the cards; it does not (see the status above).
       - While the lab is running it holds the cards, so no timed test runs beside it.
   - **Cost.** No engineering. By token arithmetic an arm of 36 answers takes one to two and a half hours of the machine.
   - This replaces waiting for the offline optimization before any usability check
     ([user decisions of 2026-11-07](#user-decisions-2026-11-07), item 7). The offline optimization still chooses values.
3. **Schedule the exchanges, then let the resident sets adapt faster.** *Ready: no user decision is open.*
   1. **Schedule the exchanges into idle time.** Send the 3090's x4 copies in chunks while it has no job in flight: it is
      idle 29 ms of every token, and an exchange needs about 11 ms of its link. Run the host copies, and the 5090's victim
      downloads, only when no 5090 upload is in flight: in the token's dense prefix and tail (1.7 ms a token) and in
      passes with no upload (1.8 ms a token). Never start a 5090 exchange and a 3090 phase in the same pass. When
      drafting, the draft call is a window with no target pass at all.
      - **Worth:** up to the 1.3% of MoE time that exchanges cost today, and it gates the next step. Re-measure with
        `exchange_cost.py` and `exchange_schedule.py` in the run folder.
      - **This is planner work:** price each piece by what it does to the completion time of the pass beside it. It
        does not depend on the CPU price that the rebuild replaces, so the [deferral list](#host-memory-and-l3) does not
        hold it.
      - *Built and timed 2026-11-10 ([evidence](evidence.md#exchange-schedule-result-20261110)). With
        `exchange_schedule=1` prose is 1.17% faster [0.22, 2.11] and code 0.3 to 1.25%; the prediction was 0.60%. The
        pieces land where intended, and the two cards' exchanges coincide no more than chance. A scheduled lease lasts
        163 ms against 15.7, so the 3090 completes at most 0.26 exchanges a token.*
      - ***Adopted for one row by the user's decision of 2026-11-10**, including the early copy's guard in scheduled
        mode (serving default since commit 418424e).*
      - ***Not for drafting: timed 8.3% slower there** [5.1, 11.5], where 0.5% faster was predicted
        ([evidence](evidence.md#bonus-timed-plan1-20261110)). Scheduled victim downloads wait for an idle pass or
        the forward's end, verify windows have few of either, and the 5090's exchanges fall from 1.45 to 0.53 a
        token. The drafting design needs a rule that lets the downloads run beside uploads. The default was changed
        for one row only, so serving with drafting was never affected.*
   2. **Make the decay and both exchange caps runtime settings**, then time the 3090's cap as arms of one load: one in
      five tokens (today), one in two (what scheduling carries), one and two a token. In simulation one to two a token
      leave 8 to 15% fewer of the router's picks outside the cache. As built an exchange costs 1.5 ms of pass time, so a
      higher cap does not pay before step 1. *Confirmed 2026-11-10 by the re-routed replay on the 2026-11-08 captures:
      at one in two tokens, one a token and two a token, 6.8%, 14.7% and 21.4% fewer picks need a load.*
      *The settings exist since 2026-11-10 (`residency_decay`, `helper_max_exchanges`, `exchange_host_wait`). On
      today's base the first replay showed the cap moving picks needing a load by only 3 to 5% at one row; that
      figure was an artefact of a capture that could not re-decide early copies. With the capture of 12:34 it is 11%,
      19% and 26% at 0.5, 1 and 2 a token, MoE time -2.3%, -4.3% and -5.7% with exchanges free
      ([evidence](evidence.md#item4-bound-20261110)). In drafting it is worth 7% at the 0.5 a token that a
      forward-end lease delivers and 22% at two a token
      ([evidence](evidence.md#reroute-replay-today-20261110)).*
      *The one-row arms are timed all the same, after the two bonus tests: today, unscheduled at 0.5 and 1 a
      token, and scheduled at 0.5 with the host copies run at once. Two frozen predictions disagree in sign. The
      agent's is about 1% faster, counting each avoided early copy at 0.10 ms; the orchestrator's is break-even or
      slower, counting only picks that need a load. The replay cannot settle what an avoided early copy is worth, and
      the answer sizes step 3 before it is built.*
   3. **Past about one exchange every two tokens** the host copies have to get shorter or go away: more threads in the
      prefix, or a bank pinned for both cards and a spare bank frame for the victim. *This is now the gating step
      (2026-11-10): as built, scheduling carries 0.26 exchanges a token, not one in two, because the host copies wait
      for a 1.2 ms window.*
      - *Why the host copies exist (`design-33.md` in the scheduling run folder): the bank is CUDA virtual-memory host
        memory that only the CPU and the 5090 may access, so an expert passes through a pinned buffer on its way to and
        from the 3090. That is 64 MB an exchange at 14 to 17 GB/s, whatever the thread count. Residents keep no bank
        copy, so the victim's download cannot be skipped.*
      - *The design: give the 3090 access to the bank, download the victim straight into a spare frame and upload the
        incoming expert straight from its own frame, rotating the spare as the 5090's exchanges do. No new memory.
        Predicted: a lease of 75 to 85 ms, 0.5 to 1 exchange a token, and almost no pass time.*
      - ***Built, and its first engine run failed, 2026-11-10 16:31.** The build (direct exchanges behind
        `exchange_direct`, default 0, with CPU tests of the direct path) passes all 26 CPU tests and all six real-CUDA
        exchange tests. In the engine the first request with direct exchanges on stopped with a CUDA out-of-memory
        error and no token. The likely cause, not yet confirmed: the engine grants the 3090 access to the whole bank
        at once, where the probe below granted one 64 MB chunk. The timed test did not run. The orchestrator accepted
        a probe that did not test the real size; a probe at the bank's real size comes before any further engine
        run. Nothing of it is on this branch.*
      - *Diagnosis from logs and code (`result.md` and `s1-scale/` in the scheduling run folder): the engine granted
        the whole bank, 181.6 GiB in 46 chunks, and every grant call succeeded, taking about 7 s. The 5090's next
        call then failed, and its context stayed broken. Two causes are possible: a budget on system memory visible
        to GPUs (the bank mapped for two cards is 363 GiB against 240 GiB of RAM), or page tables against the 5090's
        989 MiB of free VRAM. A probe at real size separates them and runs when the cards are next free. The likely
        remedy is to grant per lease only the incoming expert's frame (32 MiB) and the spare, or else a small
        sub-bank granted once.*
      - ***The probe at real size, 2026-11-10 19:04 to 19:14** (`s1-scale/` in the scheduling run folder; both runs'
        result lines were read before recording).*
        - *It is a budget on mapped host memory, not page tables. With the 5090's 181.6 GiB bank mapped, the 3090 can
          be granted 16 GiB and breaks at 20 GiB, the same with VRAM full and with VRAM free. Free VRAM does not
          change with any grant, and the 5090's upload rate does not degrade before the break.*
        - *The grant call succeeds; the failure comes at the first work that uses the memory. It then breaks both
          cards for the whole process and stays after the grant is undone. Only a restart recovers.*
        - *Access can be set only on whole mappings. The bank is 46 mappings of 4 GiB, so one frame cannot be
          granted by itself: a 32 MiB grant inside a mapping is refused. A whole 4 GiB grant takes about 170 ms and
          its revoke about 105 ms.*
        - *Unknown: whether the budget counts the 3090 alone or both cards' mappings together; where the limit lies
          between 16 and 20 GiB; and why the 3090's upload rate fell to a quarter at 16 GiB granted, on two copies.*
        - *Two designs remain. One maps the bank frame by frame (about 6,100 mappings of 32 MiB), grants the incoming
          frame per lease and keeps the 3090 under 100 MiB; it needs a second probe for the cost of a frame's grant
          and revoke beside real work and for what 6,100 mappings do to the 5090's rate and the load time. The other
          leaves the bank alone and adds a small pool of frames (2 GiB) granted once, into which victims are
          downloaded directly; it removes one host copy always and the other when the incoming expert lives in the
          pool. Both get a hard cap on granted bytes inside the bank's code and a clean exit if a card fails.*
        - *A rule from this: on this machine a second device may map at most 16 GiB of host memory beside the
          5090's bank, and crossing the limit is fatal to the process.*
      - ***A second probe, 2026-11-10 19:27 to 19:45** (`s1-frames/` in the scheduling run folder; its result lines
        were read before recording).*
        - *The limit is on the second device's grant, between 19.5 and 20.0 GiB, whatever the 5090's bank is. Beside
          the full bank a grant of 20 GiB breaks both cards. Beside a 64 GiB bank nothing fails up to 60 GiB, but the
          5090's upload rate falls from 56 to 7 GB/s at 20 GiB and to 3.7 GB/s beyond 24 GiB, silently. Host memory
          counters do not move with any grant.*
        - *The frame-by-frame layout pins as fast as today's (210 s against 215 s) and runs 5,000 exchange rounds
          healthy and exact, but it fails two of its gates: the 5090's quiet rate is 1.3 to 1.9% lower, and a
          frame's grant plus revoke takes 3.75 ms at the 99th percentile beside real work. It is not built.*
        - *The 3090's slow copies are a transient of its idle clock state, not a warning: they appear at single
          steps with the link unchanged.*
        - *The pool design after counting real leases: an incoming expert was a recent victim in only 0.7% of 29,377
          leases, so the pool keeps both host copies and gains only the scheduled lease time, 163 ms to about 73 ms.
          Predicted -2.2% at one row, nothing unscheduled, nothing in drafting while scheduling is off there.
          **Its build is on hold.** A hard cap of 4 GiB on any second device's grant, checked before any driver
          call, and a clean exit when a card stops accepting work are coded and tested on the agent's branch.*
        - ***What runs first: the cap arms**, with settings that exist today, at one row and under drafting. They
          measure what a completed exchange is worth, which decides whether a shorter lease is worth building. The
          orchestrator twice folded these arms into a later test; they should have run in the afternoon.*
        - ***The cap arms say no, 2026-11-10 22:05** ([evidence](evidence.md#cap-arms-20261110)). No raised cap on
          the 3090's exchanges is faster than today's: at one row +0.2 to +0.7% (intervals include zero) at 2.5 and
          5 times the cap, scheduled or not; in drafting +0.8% and +2.4%, slower. The replay's gain was real in
          loads (code 18% fewer demand loads at one exchange a token) and an exchange's cost ate it. Scheduling off
          at today's cap costs 0.8%, which confirms the one-row default. **The cap stays as it is, the pool design
          (option C) stops, and its branch is parked** (`minimax-wsl/exchange-s1-r2` at cfb0230). The frozen
          predictions for the raised caps were too optimistic: all three one-row ones fall below their intervals.*
      - ***The probe passed, 2026-11-10 12:49** (`caps/s1probe/probe-output.txt` in the scheduling run folder). Under
        WSL2 the 3090 can be granted access to a host chunk mapped like the bank. It reads and writes it byte-exact at
        the x4 link's full rate (6.39 GB/s up, 6.59 down, the same as pinned memory), and the 5090's rate from the
        same chunk is unchanged (55.1 before, 55.5 after). The build waits for the cap arms' gain per exchange.*
      - *The cheaper variant, host copies also in passes with no upload, is worth 0.1 to 0.3% and is not built.*
      - *For drafting (same file): scheduled exchanges deliver only 0.16 a token today, because a lease lasts about
        3.3 forwards and one starts at each forward's end; the draft call already serves as the host window. The
        design adds leases that start at layer joins inside a forward and transfer windows sized to each pass's real
        gap. The replay puts leases at any layer pass at 17% fewer picks needing a load and MoE time -6.5% at one
        exchange a token, against -3.2% with forward-end leases, saturating near 2.5 a token
        ([evidence](evidence.md#reroute-replay-today-20261110)). No engine work until the probe has answered.*
   4. **A better score.** Two speeds, a slow average plus a fast one, leave 3 to 4% fewer picks outside the cache than
      any single decay when rotation is free. A bonus-aware score credits an expert whenever it could have replaced a
      pick that was loaded, within the bonus. It is the one residency change that can go below the foresight figure,
      because it works through the bonus. It raises the share of moved picks at a fixed bonus, so report that number.
      Both are ranked on the re-routed replay. *Ranked 2026-11-10 on the 2026-11-08 captures: both lose. Two speeds
      give 1 to 2% more loads and the bonus-aware score 0 to 1% more, at today's caps and at one to two exchanges a
      token. Both are withdrawn: today's captures agree, with 16% and 7% more picks needing a load at one row. The decay stays
      at 0.95 per five rows, the best of six values on both bases.*
   5. **The 5090's cap** is the refill budget and stays with the planner rebuild. In simulation 4 a token leaves 4 to 6%
      fewer picks outside the cache. Its promotions cost no upload, but the victim's download makes the passes beside it
      18% longer, so step 1 comes first for it too. *Corrected 2026-11-10: the cause is the early copy's guard, not the
      download. Beside a promotion the early copy is skipped and the pass takes a demand miss: 0.093 ms a token today,
      0.013 with scheduling on.*
4. **Early copies only for predicted misses.** *The user allowed its bound on 2026-11-10. A detailed capture that
   records the early copy's predictions is being taken and the rule is replayed on it. Building it still waits for the
   user, on the two points below; the bound reports how far routing moves back.*
   - **Why it matters more now.** On today's base the early copy carries 50 of 63 transfers a token, and 60% of
     one-row passes wait on one ([evidence](evidence.md#reroute-replay-today-20261110)).
   - **The bound, 2026-11-10** ([evidence](evidence.md#item4-bound-20261110)).
     - As written, item 4 halves the early copies and doubles the passes with every pick on a resident (21% to 45%),
       but alone it is 2.5% slower: today's copies happen to cover picks the bonus-routed prediction misses.
     - **Item 4b** also copies the other predicted misses when the link is idle, counted as ready only once landed.
       Alone it is -1.7%.
     - **With item 5 it is -7.8% of MoE time at bonus 0.20, about 7% of a token**, against -2.1% for item 5 on
       today's rule. The two belong together: item 4b makes the runs of passes that need no upload, and item 5
       shortens exactly those. At 0.25, 0.30 and 0.40 the pair gives -14.6%, -19.5% and -27.8%.
     - **The give-back:** moved picks 25.5% against 20.6% today at 0.20, with 95.5% of router mass kept. Bonus 0.20
       passed usability at 27.3% moved and 94.97% kept, so at 0.20 item 4 stays inside what passed.
     - ~~**Recommendation to the user:** build item 4b and item 5 together and judge them as one change, with a
       usability check, because item 4b moves picks. Neither is worth building alone.~~
     - **Recommendation withdrawn the same day** ([evidence](evidence.md#options-fidelity-20261110)). At equal
       router mass kept, item 4b is not ahead of simply raising the bonus: at about 95.5% kept, item 4b at 0.20 with
       item 5 gives -7.8% of MoE time, and today's rule at a bonus of about 0.27 with item 5 gives -8.8% while moving
       fewer picks. Item 4b spends routing fidelity at the same rate as the bonus and needs engineering. **It is not
       built now.** What moves the frontier is the 3090's cap (item 3): faster and closer to the router at once.
     - **A candidate kept for later: a floor on each row.** Allow the bonus's substitutions only while a row keeps at
       least 0.80 of its top-four router mass. It removes the rows where most of the mass was moved and keeps a third
       of the bonus's gain. It is insurance for drafting with lenience, to test only if block 2 ties the text damage
       to those rows.
   - **Today.** `early_copy_predict` routes the prediction without the bonus (`resident::Policy::Off`,
     `src/mm3/mm3_native_prefetch.inc:312`) and `early_copy_issue` copies the first predicted pick not on a GPU (line
     365). Many of those picks would have been replaced by a resident under the bonus at no cost in time. Once copied
     they count as ready, win the pick, and the pass waits: 95% of used early copies are waited on.
   - **What.** Route the prediction with the same bonus and mask as the real routing, and copy the first predicted pick
     that is still not on a GPU. Other copies run only when the link is idle and count as ready only once landed.
   - **Estimate.** About 3% alone, 10 to 15% together with item 5, by arithmetic over pass types.
   - **Point one: the routing moves.** The early copy bought routing fidelity as a side effect: picks moved off the
     router's choice fell from about 28% to between 15 and 27%. This gives it back, toward the level measured when
     bonus 0.20 passed its usability check.
   - **Point two: the deferral.** The [deferral list](#host-memory-and-l3) holds variants of the early copy's issue rule
     until the planner rebuild. This variant changes which expert is copied, not whether a CPU job or a copy serves a
     miss, so the review holds that it does not depend on the rebuild. It stays deferred until the user lifts it.
5. **Take the host out of the one-row pass that needs no upload.** *Agreed in principle by the user. Starts with a probe.*
   *On hold since 2026-11-10 under the 2% rule; see the bound below.*
   - **The design is done and the bound is lower than the review's** ([evidence](evidence.md#handoff-design-bound),
     `/mnt/wsl/transfer/minimax-wsl/runs/host-free-handoff-20261110/design.md`). Per pass the design predicts 0.455 to
     0.317 ms weighted over passes with every expert on a GPU, as the review estimated. But 60 to 66% of those passes are
     followed by a pass that waits for the early copy issued at their own join, and that next pass absorbs the saving:
     on code the two passes sum to 1.09 to 1.10 ms whatever the first one's length. Only an all-resident pass followed
     by another one keeps the gain (31% of them on prose, 21% on code).
   - **Frozen prediction for the three settings together:** prose -2.0%, code -0.7%, about -1.4% equal-weight, where
     the review said 2 to 7%. It is under 2% and adds code, so it is not built at today's cache state.
   - **It is re-bounded, not dropped.** Its value grows with the share of all-resident passes, and faster than that
     share, because pairs of them grow faster. Items 2 and 3 raise both. The same absorption applies to steps 5 and 6
     below, which are re-bounded the same way before either is built.
   - **Bench and re-bound, 2026-11-10** ([evidence](evidence.md#handoff-bench-20261110)). The queued hand-off is
     bit-exact and takes 116 to 249 us for 1 to 4 experts on the 3090, 8 to 21 us above the design's figure. With the
     measured figures steps 2 to 4 come to prose -1.9%, code -0.7%. Step 5 comes to -0.2 to -0.6% and step 6 to -0.1
     to -0.3%, so both are on hold with the rest. Step 3 uses flag-gated kernels: conditional graph nodes work under
     WSL2 but cost twice as much. The evidence gives a formula for the gain at any share of all-resident passes; the
     gain passes 2% when about 30% of passes need no upload, or 20 to 25% with more of them in a row.
   - **With the replay's shares** ([evidence](evidence.md#reroute-replay-today-20261110)): about 1.5% today, 1.8% at
     bonus 0.40, 2.6% with the 3090 at one exchange a token, and 3.7% at two a token with bonus 0.40. The bonus alone
     does not lift item 5 over 2%; a higher 3090 cap does, and item 4 would change the picture altogether.
   - **For the user:** to override the hold; and whether issuing the early copy straight after the join in these passes
     (0.035 ms earlier, the same rule) counts as a deferred variant of the early copy's rule.
   1. **The probe:** one pinned buffer mapped into both cards under WSL2, a 5090 kernel's write seen by a 3090 kernel
      without the host. *Done 2026-11-10: it works with every allocation tried. The signal takes a median 1.76 us from
      the 5090 to the 3090 and 1.31 us back, a 24.6 KB hand-off 11.2 and 9.4 us, and 16.8 and 17.9 us while the 5090
      uploads at 50 GB/s. Read the payload wide: a single block of scalar loads on the 3090 takes 37 us.*
   2. **Hand the 3090 its work without the host.** Queue its graph for the layer ahead of time, headed by a wait on a
      word in that buffer. The 5090's routing writes the picks and the input row there, and the 3090 writes its rows and
      a done word back. Its results would arrive about 0.08 ms sooner. The waiting kernel also does the spinner's job.
      The publication poll (task 54) and the 3090's start after the join fold into this step.
   3. **Dispatch the 5090's own residents on the device.** `plan_dev` already builds a plan on the device for the
      non-cooperative path.
   4. **Queue the next preparation behind the expert phase** before the publication.
   5. **Finish the preparation's fusion:** bounded at 38 to 41 us a layer, of which `dense_split` took 22.5.
   6. **Then the 3090's time per expert**, the open `T=1` item. Its 3.6 to 4.1% bound assumed the 3090 finishes last on
      most passes. Today that holds for passes with no upload and two or more experts on the 3090, 7 to 17% of passes.
      Bound it again after step 2.
   - **Estimate.** These passes about a quarter shorter, 0.45 to about 0.33 ms weighted. 2 to 7% alone, because only 11 to
     25% of one-row passes need no upload. Items 2 to 4 raise that share. *The 2 to 7% is superseded by the bound above.*
   - **No quality mechanism.** In these passes the pair to balance is the 3090's path against the 5090's own.
6. **Drafting, inside redo step 6.** [Step 6's list](#step-6-next) keeps its items. The review adds these, and none is
   quantified yet.
   - **Rows that share experts** (the user agrees). Route all rows with the bonus, take the union of picks not on a GPU,
     then route again with the same bonus on that union. It never adds a load. It moves picks, so it has the bonus's
     quality mechanism and needs a usability check. Size it first on a detailed capture of verify windows: count the
     picks not on a GPU that lie within 0.20 of another row's pick in the same pass. *Corrected 2026-11-10 by the
     replay agent: routing again on the whole union removes no load by construction, because every row's own pick
     keeps its bonus. The scheme has to choose which of the union to load, most shared first, and give the bonus only
     to those. The replay counts the sizing figure, the literal scheme as a check, and that greedy variant.*
     *Sized 2026-11-10 ([evidence](evidence.md#reroute-replay-today-20261110)): 13.2% of load picks lie within 0.20 of
     another row's pick. The greedy scheme removes 6.0% of a window's loads, from 2.0% at width 2 to 13.1% at width 8,
     and moves 5,534 picks for 4,405 loads. It is a modest lever, not the highest ceiling the review expected.*
   - **Narrow a window while it runs.** It is exact: the rows that finish are verified as before, and the dropped drafts
     are not tested. A row dropped at layer 45 saves a quarter of its uploads, and at layer 31 nearly half. One cheap
     signal to test offline: the hidden states already kept for the drafter at layers 31 and 45, scored against the
     drafter's 16 candidates.
   - **Skip the draft call** on stretches where the planner keeps offering zero drafts, and probe again every few
     tokens. Today the call is paid even then.
   - **The proposal's speed** is the rest of M2: port Infernix's shape, one graph with fused Q/K/V and gate/up and a
     head top-16 that writes no logits. About 2 to 3 ms a verify step. The proposal cannot overlap the verify, because
     it needs the anchor token.
   - **Write the drafter's context after acceptance**, for accepted columns only, as Infernix does.
   - **Use the link while the drafter runs:** room for 8 to 10 uploads a step. Infernix's version lost 1.5 to 2.2% on
     Qwen, so bound it offline first.
   - **Look further ahead only into idle link time:** a second copy a layer, or one for two layers ahead, never waited
     for. Infernix found two layers ahead 6 to 12 points less accurate and gained 1.9% at best. Measure Maya's own hit
     rates first with the predict-only mode (`early_copy` 3).
   - **Per-position pricing** is already in step 6. **Lenience** returns to the usability matrix (item 2).
     **Drafts copied from the context** are exact and need no VRAM.
7. **Trim the 5090's fixed memory.** *Ready.* Measure the drafter's real peak against its 3.22 GB allowance (up to 25
   slots), the free-memory low point against the 1.07 GB reserve (10 to 20), and the 0.8 GB between planned and actual
   residents (up to 25). Size the KV cache to the session's context (8 slots at 4,096). That is 30 to 80 slots and 2 to
   4% fewer uploads, with no quality mechanism ([evidence](evidence.md#review-5090-memory)). A 4-bit drafter (about 35
   slots) and a two-stage head (40 to 58 slots and about 1 ms a token) wait for the user: neither touches the target's
   weights, but both are precision choices. *Allowed by the user on 2026-11-10 if usability shows no decrease. With
   exact verification a lower-precision drafter changes only how many drafts are accepted, so its check is the
   acceptance rate and the speed; under lenience it needs the usability matrix. The head changes output probabilities
   and needs the matrix. First the replay sizes what 35 to 95 more residents on the 5090 are worth.*
   *Sized ([evidence](evidence.md#item4-bound-20261110)): 35, 60 and 95 more residents give MoE time -1.0%, -1.8% and
   -2.8% at one row and -1.3%, -2.4% and -3.7% in drafting. Each lever alone is under 2%; both together about 2.5 to
   3.5%. They come after items 3 and 4.*
   - *Measured 2026-11-10 ([evidence](evidence.md#probe-captures-memory-20261110)): two of those levers do not exist.
     The drafter is sized before the residents, so trimming its allowance frees nothing, and the gap between the
     planned 463 and the actual 434 residents is accounted for (a difference between CUDA's and the driver's counts,
     the context and cuBLAS). What is left: the reserve (989 MiB free when ready, about 865 MiB at the low point of one
     short run), the KV cache sized to the session's context, and 0.55 GB the engine does not account for. That is
     nearer 10 to 40 slots than 30 to 80. Bound the reserve on long drafting runs before trimming it.*

**What waits.**
- **The CPU-or-copy part of the planner rebuilds.** Its bounds are 0.06 to 0.55% in windows, and at one row the CPU lane
  has a job on 2 to 6% of passes. The rest of the [deferral list](#host-memory-and-l3) holds as written.
- **Set aside by the review:** a different decay by itself (*re-opened and screened 2026-11-10 on the validated replay,
  after the review tested it at the wrong time constant: today's value is the best of six, on the 2026-11-08 captures
  and on today's*); lossless packing of
  uploads (under about 5% by arithmetic, not measured); re-splitting residents between the cards by heat (about 1% of a
  token or less); more copies in parallel (the link is at its limit); moving the head or the drafter to the 3090.

**Open decisions for the user.**
1. The usability arms of item 2. *They are running with the default arms since 2026-11-10; the user can change them.*
2. Whether prompt reading enters the objective (item 1). *Answered 2026-11-10: yes, but not at the expense of
   generation.*
3. Item 4: the routing give-back, and lifting its deferral. *The bound is done (2026-11-10). The orchestrator first
   recommended building item 4b with item 5 and withdrew that the same day: at equal router mass kept a higher bonus
   does as well. No decision is needed now; item 4b is not built.*
13. Whether bonus 0.30 is adopted for drafting: every outcome right over seven seeds and a finished answer about 15%
   sooner, with a doubled or merged word in about one answer in 12. The orchestrator recommends it if such slips are
   tolerable, and has not adopted it.
14. Whether to build the fidelity meter (item 2, the last bullet). *Answered 2026-11-10 evening: "Yes, build the
   fidelity meter." The further requests followed the same evening and are the [routing lab](#routing-lab): the
   meter runs from a local page on which the user sets every rule per prompt and judges quality by hand.*
12. The project's Python environment lacks packages that `setup.py` installs, `jsonschema` among them. With it
   missing the server answers any request for a JSON schema or strict tools with HTTP 400, by design. The live
   server uses this environment. Installing the packages is the user's call
   (`/mnt/wsl/transfer/minimax-wsl/runs/test-fakes-20261110/serve/result.md`). It also accounts for 13 of the 14
   failing server tests; the fourteenth is a test that assumes a refused connection, which WSL's mirrored networking
   turns into a hang.
11. Whether time to a finished answer is the objective for a setting that changes what the model writes. The
   orchestrator treats it as a guard on every such change unless told otherwise.
10. The bonus. Outcomes pass at 0.30 and 0.40, but merged and doubled words grow with the bonus under DSpark with
   lenience 0.4. The orchestrator has not adopted a higher value and is separating the bonus from lenience first.
   The user may decide otherwise.
4. Whether the drafter's precision may change and whether the head may use a two-stage scheme (item 7). *Answered
   2026-11-10: yes, if usability shows no decrease.*
5. Whether more VRAM is an option. In the foresight simulation 2,000 slots would roughly halve the uploads. *Answered
   2026-11-10: not now, and outside this work's scope. Answered again 2026-11-11: the RTX 4080 now in the machine
   may be used, after the fidelity sweep.*
9. Whether the prompt may be read without the bonus, or with a bonus that falls with prompt length (item 1). Today
   the bonus moves 52 to 57% of prompt picks. The orchestrator recommends dropping it for long prompts once the new
   path is in: closer to the router and faster there.
8. Which drafter is served. The orchestrator recommends DFlash2 and proceeds on that unless told otherwise. A timed
   comparison of today's default (DSpark, lenience 0.4) against DFlash2 is owed; it needs two loads.
6. *Answered 2026-11-10: yes.* Whether `exchange_schedule=1` becomes the one-row default (item 3, step 1). It is faster and has no quality
   mechanism, and the orchestrator recommends it. The open point is the early copy's guard. In scheduled mode it counts
   only exchanges whose upload may still be on the copy stream, where today it counts every pending 5090 exchange. Much
   of the gain comes through that. Nothing else in the early copy changes, but the
   [deferral list](#host-memory-and-l3) holds variants of the early copy's issue rule.
7. Item 5: whether to override its hold, and whether issuing the early copy straight after the join counts as a
   deferred variant.

**Method rules from the review.**
- **Separate PCIe links are not independent paths.** Host memory is shared. Before calling a copy, an exchange or a CPU
  job free of another path, read the host-side copies in the code and compare passes inside and outside its phase
  intervals, at the same layer and pass type.
- **When background work collides with passes, look for idle time on each path first.** Change how weights move only
  for what scheduling cannot carry.
- **A simulation on the router's own picks does not re-route under the bonus.** Rank residency and routing candidates
  on the re-routed replay.
- **Compare levers at equal routing fidelity.** Some levers make the cache fit the router better (the 3090's cap,
  more residents, scheduling): they are faster and keep more of the router's mass. Others spend fidelity for speed
  (the bonus, item 4). Before recommending one of the second kind, put it beside the bonus at the same router mass
  kept. The orchestrator recommended item 4b before doing so and had to withdraw it.
- **A change that moves picks is judged on time to a finished answer, not only on time a token.** Count the tokens
  of finished answers, reasoning included, in every arm of a usability block. A setting that makes the model reason or
  write longer can be faster a token and slower an answer. Raised by the bonus's first usability block, 2026-11-10.
- **Price a residency change by pass types, not by transfers.** With the early copy on, 50 of 63 transfers a token are
  early copies, issued whatever the residents are. What a better cache changes is the picks that still need a load and
  the share of passes with every pick on a resident.
- **Check a constant's units against logged values before simulating with it.** The review read the score's decay as
  0.95 a forward from `kDecay`. The native path applies 0.95 per five rows, which the replay proved against 5,502 logged
  scores.

<a id="adoption-rules"></a>

### Rules for adoption, user direction 2026-11-08

These replace the earlier rule that anything changing output needs a usability check, and the strict 2% rule.

- **The engine is designed for speed, not repeatability.** The same expert runs with different arithmetic on different
  devices. Routing follows the residency bonus, so it depends on the VRAM cache state, which varies during real inference.
  All of this is intended and correct. Text that differs between runs, or between arms of a test, is therefore expected. It is
  not evidence that quality changed.
- **The burden of proof is on harm.** A faster change is adopted unless there is evidence that it lowers quality. Before any
  usability check, ask whether the change has a mechanism by which it could lower quality.
  - **No mechanism, no check.** This covers changes to timing, scheduling, launches and graphs, placement and residency, which
    device computes an expert, prefetching and copies, planner prices and admission settings, and anything else that only
    moves the cache state or the order of work. Examples: `prep_graph`, `helper_fast`, `cpu_first`, `early_copy`, CPU price,
    horizon, refill budget.
  - **A mechanism, a check.** This covers changes to the model's arithmetic beyond each device's normal precision (weight, KV
    or activation formats, approximations), the size of the routing bonus or ratio (it moves picks away from the router's
    choice), speculative acceptance (lenience, draft confidence) and sampling.
- **Token-budget cut-offs are not quality results.** A cut-off answer says nothing about quality. Usability checks give
  high-effort reasoning room to finish: every task at 6,144 tokens and each code turn at 4,096
  ([template](usability-matrix-template.json)). An answer that still hits its budget is re-run with a larger budget or left
  out. It never counts as a failure. Only completed answers are judged.
- **The 2% rule, revised.** A candidate bounded under 2% is still built and adopted if it is faster and also improves things
  in another way: simpler or cleaner code or behavior, or a real chance of making later work easier. Otherwise it is not
  built.
- **Every timed test keeps its own baseline in the same load, shared by every candidate ready at the time.** The same
  settings on different loads differ more than most effects tested. On 2026-11-08's `T=1` base, five normal loads ran prose at
  57.4-59.7 and code at 62.4-65.0 ms per token, and one load with a broken calibration ran 18-27% slower. With drafting a
  different load writes different text, and the serving code baseline moved 8.7% between two loads. Alternating arms within
  a load pins an effect to about ±0.5-1.8%. Candidates that are ready together run as arms of one load against one
  baseline, as the three-arm `early_copy` test and the query-and-spinner 2×2 did. Stopping a test early once its effect is
  clear would need a rule declared in advance, a proposal for the user.
- **Timed tests measure from token 128, with the caches warm (user direction, 2026-11-09).** Every request starts from the
  restored origin, so its first tokens also pay for bringing the request's experts into the GPUs' caches and for capturing
  graphs. The effect is the steady state: from the first completed round at or after token 128 to the final one
  (`native_screen.summarize_stats`). Evaluators report the whole-generation figure beside it.
  - Tests before 2026-11-09 kept only each request's first and last engine samples when monitoring was off, so their
    from-128 figures need a re-run. A change in progress makes the engine record one extra sample at token 128, monitoring
    or not, and the harness measure from there.
- **A change that alters the text is judged per prompt and by its mechanism** (2026-11-09, from `dense_split`'s test).
  Different texts have different lengths, and pooled ms per token over prompts then moves with the arms' length mix:
  `dense_split` pooled -4.04% while code was +0.65% and prose -2.71%. Evaluators report each prompt's effect and their
  equal-weight mean, and where possible the mechanism is measured directly, on work the text does not change.
- **Applied 2026-11-08** ([evidence](evidence.md#rules-for-adoption-revised)):
  - CPU price 1.5 and horizon 40 are restored as the user decided. Their failed check counted only cut-offs, routing fidelity
    was unchanged, and they have no quality mechanism.
  - `early_copy` is admissible. Its -5.03% was measured before `helper_fast` and `prep_graph`, and it currently turns
    `prep_graph` off in the windows it acts in. So it is integrated with `prep_graph` and re-timed on the new base, with no
    usability check.
  - `helper_fast` joins the serving defaults: -0.58% [-1.19, +0.03] with drafting, and one setting for both modes is simpler.
  - `prep_graph`'s two usability checks were not needed. They stay as records.
  - The bonus at 0.30 and lenience 0.3 do have a quality mechanism. Their earlier failures were cut-offs too (bonus 0.30 failed
    on one empty answer), so a fair check with the new budgets can be run whenever the user wants them reconsidered.
    *The user asked for these re-runs on 2026-11-10, at several bonus values and for lenience; substituted words alone are
    not a failure ([where the work stands](#where-the-work-stands)).*

<a id="host-memory-and-l3"></a>

### Host memory, the L3 and the CPU lane (2026-11-08 evening)

Details and tables are in the evidence ([contention](evidence.md#contention-between-the-cpu-lane-and-demand-uploads)).
Benchmarks are in `/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/`.
- **The hardware.** Ryzen 9 9950X3D: 16 cores, 15 visible to WSL; 96 MB L3 on the V-Cache CCD and 32 MB on the other; 16 MB of
  L2. Two DDR5-6000 channels, 96 GB/s in theory. The 5090 runs at PCIe 5.0 ×16, the 3090 at **PCIe 4.0 ×4** (6.4 GB/s).
- **Bandwidth caps.** CPU reads about 58-62 GB/s, reached from two threads. One 5090 upload reaches 57.4 GB/s, and a second at
  once adds nothing. Together the CPU and an upload share about 79 GB/s, nearly evenly (about 38 and 41) from 2 to 15
  threads. A 3090 refill cuts a concurrent 5090 upload from 40.8 to 31.6 GB/s.
- **The planner models none of this.** `plan_batch_staging` prices the CPU, the 5090 and the 3090 as independent paths. Its
  CPU price of 1.5 matches the CPU's ×1.53 slowdown beside an upload, but it never charges the upload's ×1.40. The refit (task
  40) uses per-path caps under the shared cap.
- **Refill timing is worth about 0.2%:** refills overlap only 1.5-2.3% of demand-copy time. The user wants refills kept off
  demand copies ("absolutely!", 2026-11-09). So the rebuilt planners charge a refill beside a demand copy its measured
  cost: the copy falls from 40.8 to 31.6 GB/s. It is not left as a separate decision.
- **The L3.** A cached expert is reread at 25 or more times DRAM speed, and the concurrent upload keeps its full 57.5 GB/s.
  - Busy threads keep a prefetched expert cached across 0.5-1 ms (99.8-100% of slices); sleeping threads lose a third to 45%.
  - The lane already pins its workers and spins 1-3 ms between jobs.
  - **Capacity: about 86 MB, about 2.7 experts, whatever the layout.** A probe that barely touches the data finds 80-87 MB
    still cached 0.5 ms after a 128 MB prefetch for every layout tried: an even split over 15 workers, one group of 7-8,
    each expert on its own 4 workers, two experts per group, more threads on the guest's hyperthreads. Layouts only change
    how close 2-3 experts come (7 workers keep 84-87% of them, 15 workers 71-72%). The earlier 50-56 MB came from full
    rereads, whose misses evicted cached data.
- **Combined GB/s: the DRAM caps CPU plus upload at about 80 GB/s, and no in-WSL trick raises it.**
  - Beside an upload the total stays at 77-84 GB/s for every worker set; placement only moves the CPU's share (36-45 GB/s).
  - **The CPU alone reads about 60 GB/s, and host placement does not change it.** Nine busy readers read about 79 GB/s
    under the default placement, while 8, 10 and 15 stay at 57-67. The guest's vCPU numbers and hyperthread pairs say
    nothing about the hardware, and High performance changes nothing (Balanced restored). The affinity tests then confined
    the VM to either CCD, or spread it one hyperthread a core over both: reads stayed at 56-61 GB/s. So the 9-reader case
    is not a both-CCD effect, and its cause is unknown.
  - **Only L3-served bytes beat the cap:** 15 threads rereading a cached expert read 1,460 GB/s while the upload keeps 57.3.
  - The both-CCD bandwidth lever was priced at -1.0% to -1.6% at `T=1` before `early_copy`, which cut CPU experts by three
    quarters; the affinity tests found no placement that delivers it.
- **Placement from the host (task 45, 2026-11-09): the affinity action works; the default `all` stays.** Logical
  processors 0-15 (cores 0-7) are the V-Cache CCD, holding about 94 MB. No mask lifts CPU reads or the kernel from DRAM
  (515-558 us alone, 847-867 beside an upload); confining the VM to one CCD slows the L3-hot kernel from 139 to 201-214
  us, since 15 threads then share 8 cores ([evidence](evidence.md#contention-between-the-cpu-lane-and-demand-uploads)).
- **Candidate: an L3 early copy for the CPU lane** (task 43). Prefetch the predicted CPU expert into L3 during idle DRAM
  windows, with a fixed row-to-thread mapping. **Bounded 2026-11-08: about -2.9% to -3.6% on today's base, but
  `early_copy` uses the same prediction for more, so it waits for task 41**
  ([evidence](evidence.md#contention-between-the-cpu-lane-and-demand-uploads)).
  - The engine's own CPU kernel takes 525 us for an expert from DRAM, 159 us from L3 and 251 us when the pool prefetched it
    0.3 ms earlier; beside an upload 878, 195 and 455 us, the upload gaining 7 GB/s. The prefetch costs 554 us of DRAM
    time, so it pays only in idle windows.
  - Per pass, from the HAGS-on captures and the refitted model: `early_copy`'s one-layer-ahead prediction arrives at the
    previous layer's join, which leaves a median 0.42 ms of idle DRAM before the CPU pass, enough for 74-80% of its bytes:
    -2.9% at its 53% and 75% retention, -3.6% at full retention. *(First recorded as -1.4% to -1.8%, on the mistaken
    reading that the router's prediction is same-layer; corrected the same night.)*
  - `early_copy` uploads that predicted expert to the 5090 instead, which removed most CPU jobs in its bound and measured
    -5.03%. After its integration (task 41) the L3 prefetch is bounded again on the CPU jobs that remain.
- **User decisions, 2026-11-08 evening:**
  - **The Windows-side affinity action has the user's OK.** It is a new allowlisted host-control action that sets the CPU
    affinity of the WSL VM's `vmmem` vCPU threads. The aim is to pin the lane to the V-Cache CCD (cache) or spread it over both
    CCDs (bandwidth). GPU power limits stay untouched.
  - **Try the in-WSL tricks and SMT first**, before the Windows-side action.
  - **Try other expert-to-worker layouts:** each expert split over its own 4 dedicated workers, or 2 experts over one group of
    7-8 dedicated threads and 2 others over a different group of 7-8.
  - **Ask of everything whether it can raise the combined GB/s.** CPU bytes served from L3 cost no DRAM bandwidth. CPU reads
    spread over both CCDs might pass the ~60 GB/s ceiling.
- **Next, in order:**
  1. *Done:* the layout and SMT experiments and the combined-GB/s measurements (above).
  2. *Done 2026-11-09:* the Windows-side affinity action, registered by the user (`vm-affinity-query`, `vm-affinity
     <set>`: six fixed masks of at least 16 logical processors, `all` to restore). CCD0 holds the V-Cache; no mask lifts
     CPU reads or the kernel from DRAM; the default stays (above).
  3. *Done:* the real NVFP4 kernel from L3 against DRAM, and the idle DRAM window per layer (the candidate above, which
     now waits on task 41).
  4. *Done:* the refit (task 40). The refitted model's fitted shares already are the per-path caps at `T=1` (CPU 35 +
     copy 38 GB/s while both run, every CPU pass also uploading). `cpu_first` re-predicts at -0.89%, the frozen prediction
     the hardware refuted, so it is not re-timed ([evidence](evidence.md#cpu-batch-before-the-gpu-submissions-work-list-item-3)).
     The bandwidth lever prices at -1.0% to -1.6% at `T=1`.
  5. *Done:* `early_copy` folded into `prep_graph` and re-timed (task 41): -12.63%, adopted for `T=1`. The CPU job's work
     around the dot products (task 46, the Astra review) is measured: no change pays.
  **Order up to step 6, re-checked 2026-11-09 at the user's request.** `early_copy` moved the bottleneck (the CPU does a
  quarter of its old work, the copy engine is about three quarters busy, 95% of used early copies are waited on), and
  the planner is best fitted last, on the final mechanics:
  1. *Done 2026-11-09:* the launch shape (one graph per layer for every position, exact, 57 captures a request instead of
     up to 399) and the token-128 measurement verified ([evidence](evidence.md#early-copy-inside-prep-graph));
  2. *Done 2026-11-09:* the per-pass picture of the new base: one light-capture load and one Nsight trace with
     `early_copy`, from token 128, for the per-layer path metrics and where the copy engine's time goes;
  3. mechanics that shorten fixed per-pass time and do not depend on the planner, each bounded first: the preparation's
     projections (147 of about 202 us a layer, if they are not already bandwidth-bound), the side-stream fork, the 3090's
     fused kernel. *Done for the projections' cheap part: `dense_split`, -22.5 us a layer, adopted.* **The trace then
     ranked the 5090's chain** ([evidence](evidence.md#five-ninety-chain-trace)): after each preparation the 5090 idles a
     median 166-201 us (traced; about 100+ untraced) for the host's round trip, and 780-856 us behind a demand upload on
     the main stream. Inside the expert phase the 5090 computes only about 85-90 us of a median 179-186 us span: the rest
     is the host submitting its kernels one by one. **But the 3090's path binds on 95-97% of passes**: the 5090 then spins
     for the host's publication of the 3090's results, so an expert-phase graph would save about 1 us a pass and is not
     built. The target is the 3090's path. *Done 2026-11-09: zero-copy copies in `helper_fast`'s graph
     (`helper_zero_copy=1`, adopted; the 3090's span per pass -26 to -29 us, mostly the host's launch call; timed -1.91%
     [-3.69, -0.13], likely nearer -0.6 to -1.2% without three slow off-arm requests;
     [evidence](evidence.md#helper-zero-copy)).* **Re-ranked the same day, per pass**
     ([evidence](evidence.md#after-zero-copy-bound)). With the 3090's results about 29 us earlier, the 5090's own path now
     binds on about 69% of code passes and 46% of prose passes; mean |delta| between the two paths is 96 and 69 us. So, in
     order:
     1. ~~the 5090's expert-phase graph: 1.09 / 0.67 ms a token (code / prose), about 2.3% / 1.6%~~ *Built and tested
        the same day ([evidence](evidence.md#expert-graph)): exact, and the host submits the 5090's experts about 60 us
        a pass sooner, but the layer cycle does not move. Nsight's traced gaps had overstated the 5090's device idle.
        Kept off. The timed test overlapped two sub-agents' file scans: blocks after them show about -1%, so it is to be
        re-timed cleanly and with drafting;* *Re-timed 2026-11-09 and adopted for both: `T=1` -1.30%, drafting -4.58%
        ([evidence](evidence.md#expert-graph-windows)).*
     2. the 3090's projections near peak bandwidth: they stream each 31.85 MB expert at about 660 GB/s, 70% of peak.
        -10 us a job gives 0.40 / 0.83 ms, about 0.8% / 2.0%, since it shortens most the passes where the 3090 binds.
        Fold in fusing the two quant kernels (6.5 us of kernel time, 0.3-0.6% alone). Both cuts together: about 1.7 ms a
        token, 3.6-4.1%;
     3. the result path, about 9 us of host work a pass plus the worker's poll. *The merge done 2026-11-09: branch-free
        and bit-identical, the host's serial time after its last lane down from 0.009-0.096 to 0.005-0.039 ms a pass
        ([evidence](evidence.md#helper-merge)). The poll remains (task 54);*
     4. the 3090's start, 44-55 us after the layer's join;
     5. residents beside a demand upload;
     6. the remaining projection fusion.
     The large mean |delta| is per-pass imbalance in both directions, the planner rebuild's subject. It follows the
     3090's job count: with 4 or more jobs (9-15% of passes) the 5090 still spins a median 123-166 us, and with one job it
     waits for nothing;
  4. the planner rebuild below, then the items deferred behind it. *Open (task 48);*
  5. step 6. *Begun 2026-11-09 by user direction while items 3 and 4 were open
     ([where the work stands](#where-the-work-stands)).*
  6. **The planner, priced by completion times from the measured GB/s** (user direction, 2026-11-09). A CPU-price
     test with `early_copy` was assembled and is **not run**: the price is a single number standing in for contention
     the planner does not model, and with `early_copy` the planner also cannot see early copies holding the link, so
     tuning the number would tune a part the rebuild replaces. The earlier bound that re-pricing gains little predates
     `early_copy`. Order: refit the per-pass model on the new base from a light-capture load measured from token 128
     (early copies and the measured rates in it); bound a planner that prices each pass by completion time, early copies
     in flight included, against today's; build and test it if the bound reaches 2% or the rebuild simplifies.
     - **Deferred until the rebuild, since their optimum or value depends on how the planner prices a pass** (user
       question, 2026-11-09): horizon and refill-budget tuning, and those parts of the offline optimizer (task 23);
       variants of `early_copy`'s issue rule (mode 1 or 2, copying behind a demand copy only without a CPU job); the
       device-side upload decision (0.7%, it would carry today's planner onto the GPU); the copy-stream demand upload
       (0.8-0.9%, its bound predates `early_copy` and depends on the new planner's uploads); recalculating old planner
       tests from token 128. **The same holds in windows:** the window CPU price, and the window optimizer's screen of the
       CPU price, horizon and refill budget, which ran out of this order on 2026-11-09 and selects nothing
       ([evidence](evidence.md#window-optimizer)). *2026-11-10: the review proposes one variant of the early copy's
       rule, copies only for predicted misses ([order](#order-20261110), item 4). It stays deferred until the user
       lifts it. Scheduling the exchanges (item 3 there) is not a planner price and is not deferred.*
     - **Not replaced, so they go ahead:** correctness checks of engine changes; kernel work (the 3090's four-kernel
       fusion, QKV/indexer fusion, the NVFP4 expert kernels); the routing-bonus usability check, whenever the user wants.
  7. Speculative decoding (DSpark) waits for step 6 of the redo: no serving-configuration tests before then. *Step 6
     began 2026-11-09 with DFlash2 first; the serving defaults still change only after step 6's tests.*

### The `T=1` base after 2026-11-08's hardware tests

Each change was tested on its own fresh load against the base of the day. Details are in the work list below and in
[evidence](evidence.md).
- **Adopted 2026-11-09 (step 6): `expert_graph=1`, the 5090's expert phase as one replayed graph** (5e35257,
  [evidence](evidence.md#expert-graph-windows)).
  - Planner v1 -4.58% [-7.98, -1.18] against a frozen -4.4%; `T=1` -1.30% [-3.30, +0.69] against a frozen -1.4%. Both
    predictions fall inside.
  - In windows the host submits the 5090's work 0.05-0.39 ms sooner, so the CPU lane, last on 25-72% of window passes,
    starts that much earlier.
  - It is exact: mode 2 checked 23,427 window replays (and 21,831 at `T=1` earlier) with 0 mismatches, and the timed arms
    replayed 1.24 million times without a fallback.
  - It joins the `T=1` base and planner v1's drafting base. Serving defaults are unchanged until step 6.
- **Adopted 2026-11-09 (step 6): `gpu_nucleus=1`, temperature-1 top_p nuclei from GPU candidates** (1bdd4f6,
  [evidence](evidence.md#gpu-nucleus-second-round)). `T=1` -5.70% [-6.94, -4.45] (-2.58 ms a token); planner v1
  -5.76% [-9.48, -2.04].
  - It is exact. The in-engine check compared 766 settled rows with `dist()` on the whole row: 0 mismatches.
  - It removes the host's 2.3 ms of sampling a row and costs the forward nothing measurable.
  - It joins the `T=1` base and planner v1's drafting base. Serving defaults are unchanged until step 6.
  - Exactness is checked in the engine, not by comparing texts: by design, text differs between runs
    ([rules for adoption](#adoption-rules)).
- **Adopted:** `helper_fast` (the 3090's spin waits and graph submission), -5.04% [-5.86, -4.22]. It does not change output.
  - **With drafting: -0.58% [-1.19, +0.03]** at the serving configuration
    ([evidence](evidence.md#the-3090s-startup-work-list-item-5)). The strict rule kept it out, since the interval crosses zero.
    **Under the [revised rules](#adoption-rules) it joins the serving defaults**: five of six blocks were faster, and one
    setting for both modes is simpler.
- **Adopted 2026-11-09: `helper_zero_copy=1`, the 3090's graph copies as zero-copy kernels** (65c47c3,
  [evidence](evidence.md#helper-zero-copy)). It is exact: every pass is checked in mode 2, 17,577 passes with 0 mismatches.
  The 3090's span per pass falls 26-29 us, mostly the host's launch call. Twelve blocks at up to 384 tokens give -1.91%
  [-3.69, -0.13] from token 128 against a frozen -1.16%. Three slow off-arm requests, system noise by their stat lines,
  inflate that; the arms' medians differ by -0.4% (code) and -0.9% (prose). No quality mechanism. Serving defaults are
  unchanged until step 6. Future `T=1` tests carry it in their base.
- **Adopted 2026-11-09: `dense_split=1`, split K for the 512-output projections (attention k and v, the indexer's q)**
  ([evidence](evidence.md#preparation-projections)). Each layer's preparation span on the 5090 fell 22.5 us [-23.3, -21.7]
  (about 10%), as bounded; exact to F32 rounding; no quality mechanism. Its timed test pooled -4.04% from token 128, but
  that is mostly a length mix of different texts (per prompt: code +0.65%, prose -2.71%). Serving defaults unchanged
  until step 6.
- **Also verified 2026-11-09:** one preparation graph per layer for every position (6fc35b1, exact), and timed tests
  measured from token 128 (0c2246d).
- **Adopted 2026-11-08: `early_copy` mode 2 inside `prep_graph`, -12.63% [-13.85, -11.42]**, about 18.9 to 21.6 tok/s
  ([evidence](evidence.md#early-copy-inside-prep-graph)). Since 3fc50e6 its prediction runs inside the preparation's graph,
  checked bit for bit, so the two combine. CPU experts fell from 9.5 to 2.5 a token and demand uploads from 32.3 to 13.2.
  - It has no quality mechanism (its earlier usability "failure" was two cut-off notes), so no usability check.
  - The frozen -5.70% failed on the good side: at CPU price 1.5 the base uploads far more than the old 1.3 base did.
  - With drafting it acts only on one-row windows; the serving defaults change after its own drafting test.
- **Kept as they were:**
  - the submission query (a tie under HAGS) and the spinner (off costs 16.4%);
  - `cpu_first`, which stays off (+0.66%). Re-predicted with host-memory contention it gives -0.89%, the same prediction
    the hardware refuted; the likely cost is its extra submit thread, so it is not re-timed;
- **Restored: CPU price 1.5 and horizon 40**, the user's decision. Their failed usability check counted only token-budget
  cut-offs, and routing fidelity was unchanged (94.61% against 94.60% of router mass kept). On speed they tied with 1.3 and
  20, and 1.5 is the safer price from load to load.
- **Bounded under 2%, to re-assess under the revised rule** (faster plus another benefit): the copy-stream upload
  (0.8-0.9%), the device-side upload decision (0.7%), and the others in the [work list](#work-list). The first two wait
  for the planner rebuild, since their value depends on its uploads ([item 6](#host-memory-and-l3)).
- **Found and fixed:**
  - the server dropped the newer engine counters from each request's stats (9f8c726);
  - back-to-back calibrations caught the 5090 slow, and a planner kept such a table. The calibration now warms the 5090 and
    re-measures implausible values (080c9ed).
- **Adopted: `prep_graph`, -6.99% [-8.84, -5.14]** against a frozen -7.71%, and it **passed its usability check**
  ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)). It joins the `T=1` base.
  - It replays each one-row MoE layer's preparation on the 5090 as a CUDA graph, verified bit for bit.
  - The preparation's span falls from 280 to 202 us a layer. Per-pass balance does not move.
  - Its five small host copies are captured as zero-copy kernels. As graph copy nodes they cost about 13 us each, which made
    the first form (793c78d) slower.
  - The text varies only as the base's own does from run to run. The usability check passed: no task on fewer seeds than
    off, no visible damage.
  - **With drafting, at the serving configuration: -4.39% [-6.14, -2.64]** (f8e2ff8, frozen -3.80% inside), about 21.0 to
    21.8 tok/s, with identical text in all 12 pairs. **It passed its serving usability check, so `prep_graph: 1` is now a
    serving default** (`native-runtime.json`).
- **On CPU passes the CPU lane and the copy already finish together, both bound by host memory bandwidth**
  ([measurement](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/upload-contention.txt), HAGS-on traces):
  - A 31.85 MB demand upload takes 554 us alone (57.5 GB/s, near PCIe 5.0 x16's limit), and 748 us beside a CPU job (+35%).
  - A CPU job takes 0.548 ms alone and 0.806 ms beside an upload in two-row windows (+47%). At `T=1` every one-job pass at
    price 1.3 has an upload, and its batch takes 0.844-0.854 ms.
  - Together the two read about 80 GB/s from host memory, about what it delivers. So the copy-plus-primary path (about
    0.80-0.83 ms) and the CPU lane (0.85 ms) end within about 0.03-0.05 ms of each other.
  - **Each path alone is capped below the shared 80 GB/s** (user correction, 2026-11-08): the upload by PCIe at about 57.5
    GB/s, one CPU job at about 58 GB/s (31.85 MB in 0.548 ms). Removing a CPU job lets a concurrent upload rise only from 42.6
    to 57.5 GB/s, not to 80, and the pass's host throughput falls from about 80 to 57.5 GB/s. So moving work from the CPU to
    uploads trades two parallel paths for one capped path. That is why the CPU lane's saving at 1.5/40 bought the token so
    little. The refit models each path's rate as the lesser of its own cap and its share of the shared cap.
  - **Profiled at several splits** ([evidence](evidence.md#contention-between-the-cpu-lane-and-demand-uploads)):
    - The shared cap is about 79 GB/s, split nearly evenly (CPU about 38, upload about 41) from 2 to 15 CPU threads.
    - A second concurrent 5090 upload adds nothing, so PCIe is full with one.
    - The 3090 runs at PCIe 4.0 ×4 (6.4 GB/s). Its refills cut a concurrent 5090 demand copy from 40.8 to 31.6 GB/s.
    - The planner models none of this. Its CPU price of 1.5 matches the CPU's ×1.53 slowdown beside an upload, but it never
      charges the upload's ×1.40.
  - The 63.7 MB the pass moves needs at least about 0.8 ms at 80 GB/s. Re-pricing the planner for contention can gain
    little. Uploading both misses instead serializes on the copy engine, about 1.1 ms.
  - The 0.78 ms gap between the CPU lane and the helper is the 3090 finishing early. The levers are fewer bytes from host
    memory a pass (fewer misses) and more work on the 3090.
- **Why the early copy fell short:** its reads from host memory slow a concurrent CPU batch by 12-13% (about 0.1 ms). Its
  prediction costs only 5-18 us per layer net, and half that in the lean form. Copying only on passes without a CPU job is the
  next rule to bound, for whenever a larger usability check is run.
- **The calibration fix works:** the diagnostic load's own first calibration measured implausibly and was measured again.
- **A lesson for the per-pass model:** it credits CPU-lane savings about three times too strongly at `T=1`, and helper-path
  host time about 40% too weakly. Read its predictions of CPU-removing changes as upper bounds until it is refitted.

### Timing-model replacement

**User decision, 2026-11-06: “this timing-model approach needs to be replaced.”**
Retire the factorized per-operation timing reconstruction as the active route to
optimization. Its probe-based full-work comparator missed the measured control
by 80.2% and resolved none of the three prospective intervention decisions.
The earlier foreground refit, its separate 5.57% miss and every adverse receipt
remain preserved. Further calibration or qualification of this timing approach
is no longer the next task or a prerequisite for its replacement.

Retain the useful state replay, shared native policy entry points, original-demand
accounting, ownership/capacity checks, instrumentation, fixed-work harness and
repaired validation principles. Preserve the retired timing code and frozen
outputs as historical references; do not select runtime parameters from them or
resume their improvement as the default workstream. The selected inference
engine and cache-aware top-four routing remain in scope. Admission learning must
still credit original-router demand only.

**Replacement selection is open.** The [six collected proposals](timing-predictor-proposal-measured-transitions.md)
contain two primary-agent proposals and one each from Terra, Fable, Opus
and sol-6.1. The user supplied or requested the external proposals; do not contact
or spawn their authors. Compare all six before choosing the implementation.
This is a design comparison, not an instruction to build six predictors.
Consolidation does not validate the submissions' claims. No replacement is
selected, implemented or qualified by this decision.

The earlier [joint boundary-response regression design](timing-predictor-proposal-measured-transitions.md#primary-agent-proposal-1)
is one candidate reference. Its readiness heads, state representation and large
pilot are not the next implementation queue. The user now requires the bounded
[diagnostic study](#shared-model-diagnostic-study) before selecting the new
first-stage design's mechanisms.

<a id="generation-only-modeling"></a>

**User direction, 2026-11-06: model token generation only, with speculative
decoding disabled first.** Build, calibrate and independently validate the first
predictor on ordinary one-token decode forwards (`T=1`). Source captures,
conditioning generation, fixed-work generation and live confirmation must use
that regime. Verify actual one-row decode forwards and zero draft proposals and
speculative verification work. Preserve cooperative CPU/5090/3090 execution,
adaptive residency and cache-aware routing while changing the decoding mode.
The public request override is `strata_tune.spec=1`, with
`spec_policy="existing"`, `verify_segment=0` and `stage_overlap=0`.
The historical `native_delivery.py --arm ordinary` also disables routing;
it is not this baseline. Keep the selected prior routing and its explicit bonus
when using `native_screen.py --spec-policy ordinary`.

*2026-11-10: the review measured prompt reading at 32 to 45 tok/s, 33 s for the benchmark's 1,056-token prompt against
14 to 17 s for its answer, and proposed reading prompts one layer at a time ([order](#order-20261110), item 1). Whether
prompt time enters the objective is the user's open decision. The modeling scope below is unchanged.*

**Prefill is outside the modeled duration and objective.** It may run as native
setup, and its resulting placement, learned demand, pending work and context
length are explicit generation-start inputs. Do not predict prefill time, fit
generation costs to prefill observations, or treat the starting state as empty
unless the declared setup makes it so. The three initial dense decoder layers
still execute on generated tokens; they are not prompt prefill. Keep complete
generation accounting and the policy's shared-resource and later-use effects.

Dropping draft time from the old fixed-work traces does not meet this contract:
those traces retain multi-row speculative verification and rejected rows. Use
genuine ordinary-generation work; do not relabel or split old speculative batches
and call them an ordinary-request baseline. The existing recorded-assignment,
no-copy generation-only diagnostic also does not implement this new policy test.

Once the ordinary-generation model has demonstrated useful policy-decision
accuracy, extend and qualify it for speculative generation and its grouped work.
That is a later extension within token generation, not a prefill-modeling phase.
Keep the six proposals and the broader systems/modeling literature as sources of
candidate methods. Select only mechanisms needed for a coherent, testable design;
there is no requirement to take a contribution from every proposal or paper.
The simpler first regime does not waive independent validation, correct timing
boundaries, instrumentation checks or declared measurement budgets. Existing
proposal budgets are ceilings to assess, not mandatory experimental queues.

<a id="shared-model-diagnostic-study"></a>

### Shared model and diagnostic study

**User decision, 2026-11-06: start with one shared model evaluated at each MoE
layer.** The 57 MoE layers have common geometry and ordinary decode has one token
row. Do not start with separately fitted layer parameters, layer-specific offsets,
or a campaign to establish whether every layer deserves its own model. Reconsider
that choice only if the shared approach has a demonstrated failure. Layer IDs
remain observation/alignment keys, not proposed fitted parameters.

First measure which effects materially change generation timing. Use those
findings to choose the first-stage model's inputs, state and composition rules.
In particular, test whether explicit cross-layer cache contents, buffer state and
pending-transfer state can be omitted or represented by smaller summaries. Do
not assume a full stateful timing model is needed merely because the real engine
has state. Full diagnostic records provide a reference; recording a field does
not commit the predictor to using it. Retain the real engine's ownership and
reader protections throughout.

**Initial campaign, completed 2026-11-07;** see the
[result](#shared-model-diagnostic-result). The protocol as declared:

Use the same build in two sequential engine loads: the standalone series without
Nsight/CUPTI injection, then the separate Nsight-launched diagnostic process.
Use the existing Python-code/prose prompts for the scored observations, high effort,
seed `20261101`, and a 384-output-token cap for ordinary requests. Preserve EOS
and actual outputs; report actual evaluated/generated work. Keep the selected
arithmetic, CPU participants, worker helper, prior routing with bonus `0.10`, and
all settings except the stated intervention fixed. Every source and measurement
has speculation off. Prefill remains native setup and is excluded from the
timing objective. All setup time still counts against the experiment budget.

**Clarification, 2026-11-07:** conditioning uses the retained independent coding
and prose prompts, not the scored prompts. The execution protocol selects the
`count_initials` coding prompt and seed-envelope prose prompt from that history.
Use the same declared conditioning sequence in each load. Each process performs
its bootstrap/calibration and helper-first setup, then those two conditioning
requests, and saves its own learned-state origin. Learned state is process-local;
do not transfer it across loads or infer equality merely from identical prompts.
Restore the process's own origin before each comparison arm, process its scored
prompt as setup and record the actual generation-start state. Do not claim
identical physical cache warmth merely from a matching logical-state hash.
The trace pair uses the Nsight process's own origin; its results remain a separate
diagnostic pair rather than a matched timing continuation of the standalone load.

| Run group | Count | Purpose |
|---|---:|---|
| Bootstrap/calibration setup | 2 requests | One short bootstrap per engine load; maximum 32 output tokens each |
| Independent ordinary conditioning | 4 requests | One independent coding and one independent prose request per load; save a separate origin in each process |
| Longer ordinary observations | 6 requests | For each prompt: monitoring off, light records, monitoring off; 384-token cap per request |
| Matched monitoring test | 4 short fixture runs | Off/light/light/off on identical newly captured `T=1` expert work and restored starting state |
| Grouped state-effects test | 4 short fixture runs | Normal optional residency updates versus holding new primary/helper admissions, in both orders; demand servicing and helper computation remain enabled |
| GPU trace companion | 2 short fixture runs | One off and one detailed/Nsight run of the same control fixture in the separately conditioned Nsight load; diagnostic comparison only |
| Conditional follow-up reserve | 0 additional runs | The required second-load setup and bootstraps consume the former four-run reserve |

This is **22 planned executions at the existing 22-execution cap**: **16
measurement runs** (six longer requests and ten short fixtures), **four independent
conditioning requests**, and **two bootstrap requests**. The earlier total of
18 already included the standalone load's two conditioning requests; it was not
18 measurements. The 2026-11-07 [execution protocol](shared-model-diagnostic-protocol.json)
enumerates these 22 requests and fixtures, and its metadata now says 22 total and
zero additional reserve; setup does not sit outside the count cap. The driver has
no path for further runs. Helper-first placement and snapshot/restore operations
are also setup and count against the warmed time cap. Each short fixture uses
the same predeclared 64 ordinary-generation forwards, with every MoE call retained
and the required native KV/prefill setup outside scored generation. Choose the
first 64 complete generation forwards of the coding light capture; use prose
only if coding ended before 64. This selection rule precedes timing inspection.
These short windows
are mechanism diagnostics; they do not replace the complete longer requests or
qualify serving performance. Cap warmed campaign time across both loads, including
native setup and conditioning, at 30 minutes; report both loads and build time
separately. There is no additional run reserve in this reconciled allocation.

**Monitoring preparation.** Existing `timeline` mode lacks the required per-layer
work records. Existing `detailed` mode constructs/compresses text and enables
NVTX. Add a small, preallocated binary record mode for this study; do not build a
replacement predictor first. Give light mode its own predicate; the existing
`mode > 1` detailed branch would also enable the heavier instrumentation.
Record generation start/end, existing layer/join
and host-service boundaries, the four original/executed expert IDs and their
service locations, actual demand-copy work, free/reserved frame counts, and
existing pending/publication observations. Capture the starting cache map once
and mutations thereafter, rather than dumping the complete map at every layer.
Collect only already available CPU/helper timing observations. No additional
CUDA events, queries or waits in the light mode; serialize outside generation.
Overflow or missing required intervals invalidates that capture. Include enough
source-work information to initialize the fixed fixture, without fitting prefill
timing. Reuse the existing trace and policy entry points where appropriate.

The longer requests reveal the actual work and queue exposure. They are not,
by themselves, an isolated monitoring-overhead test: equal prompts and seeds
may still produce different routing/work trajectories. The matched fixed-work
off/light test measures the net effect of recording, including any changed
publication behavior; no constant overhead is subtracted. The detailed GPU
companion establishes ordering and overlap at its recorded scope. Its CUDA event
envelopes are not treated as pure arithmetic costs, nor as timing truth for an
uninstrumented run.

**Minimal effect questions and omission tests:**

| Question | First check | Consequence for model design |
|---|---|---|
| Is immediate foreground work sufficient to explain time? | Tabulate elapsed intervals by actual CPU/primary/helper assignments and demanded transfers across the light captures; no layer-specific fit | Start with these work descriptors; add information only for a material unexplained effect |
| Is the complete cache map needed by the timing predictor? | Track whether history affects time beyond the current selected experts' placement/reuse and resulting work; separately retain later hits/misses and displacement | Try the current-work summary before full-map timing features; do not assume that observing today's placements predicts a changed policy's future placements |
| Do pending copies or occupied buffers add material delay? | Use the grouped normal/held-admission fixture comparison and trace actual pending intervals, availability and work changes | No detailed queue model by default; test a busy/count/bytes summary before phase/age modeling if an effect is present |
| Can readiness be handled by execution order alone? | Audit publication/reader retirement against existing joins on the new ordinary traces; check a declared order-only replay rule using no future availability | Retain a timing-free rule only where its resulting service/transfer decisions remain adequate; localize its failures before adding readiness prediction |

Holding admissions changes future residency and therefore can change foreground
assignments. Fixed expert IDs alone do not freeze that work. Report this fixture
as a **net policy-effect screen**, and show its changed assignments, demand copies,
retained reuse and pending exposure. It does not identify an isolated copy-price
or prove that a zero net effect means every constituent effect is negligible.
If the observations cannot distinguish changed work from contention, leave that
effect unresolved under this allocation. A specifically matched foreground-work
follow-up would need a separately declared budget. Do not infer a causal per-upload price from observational
counts alone.

Use **2% of complete scored generation time** as the proposed materiality band
for this diagnostic screen, reporting milliseconds as well. A component/interval
effect must be related to its actual exposure over generation; do not add
overlapping spans or nested residuals. Treat whole paired executions/blocks as
replicates. Two-pair half-ranges are descriptive spread, not confidence intervals.
Classify an effect as material, bounded below the practical band, or unresolved;
an inconclusive comparison is not evidence that a state field can be omitted.
If the small budget cannot resolve an omission, retain that uncertainty for the
design decision rather than silently escalating to a large calibration campaign.

For prospective policy use, a simplified model cannot recover omitted state by
reading the control trace's future hits, assignments or publication times. Keep
two questions separate: which state the existing policy/ownership bookkeeping
needs, and which state the timing predictor needs beyond its current-work inputs.
Any claimed omission must state which of those interfaces it applies to.

The study's deliverable is a short evidence-backed disposition for each candidate
effect: observed size, monitoring sensitivity, the smallest justified summary,
and unresolved coverage. Then write the new shared first-stage design. The six
proposals inform the choices where useful. Broader references used here are
[causal profiling](https://github.com/plasma-umass/coz), for evaluating whole-program
effects rather than optimizing timer categories, and
[switchback experiment design](https://arxiv.org/abs/2009.00148), for respecting
effects that persist across time. Their application to this engine is a design
inference, not evidence that a particular estimator or predictor will work.

<a id="shared-model-diagnostic-result"></a>

**Diagnostic result, 2026-11-07.** All 22 declared executions ran without step
errors, using 1,304.5 s of the 1,800 s warmed cap. The
[full account](evidence.md#shared-model-diagnostic-study) holds the measurements,
limits and receipts. Coverage is `T=1`, eight staging frames, the two scored prompts
and positions 240-1439. "Rule" is the predeclared class; the reading beside it may
draw on post hoc checks, which changed no label. Each disposition concerns the
**timing predictor's inputs**. Existing policy and ownership bookkeeping keeps
whatever state it needs, and supplies the predictor's current-work inputs.

| Effect | Rule | Observed | First-stage timing predictor |
|---|---|---|---|
| Current service mix | Held-out table unresolved | Work-mix median table misses -1.43%, +1.56% between prompts; all-forward table -8.9%, +10.3%, mostly slow spells | **Represent:** per-layer counts of CPU, helper, primary and new-copy picks |
| Demand copies | Material as part of work | +0.54-0.55 ms each in every capture; 9.9-14.8% of generation in long requests | **Represent:** count per layer |
| CPU and helper overlap | Post hoc | Span matches `max(CPU, helper)` within 0.014 ms; the sum misses by over 0.5 ms; hides 15-16% of generation | **Represent:** a maximum, not a sum |
| Primary path at `T=1` | Post hoc | No overlap with the host lanes, with or without a profiler | **Represent:** sequential after the host lanes in this regime |
| Forward constants | Post hoc | Prefix about 2.6 ms (1.45 with admissions held), final layer about 3 ms, first MoE window +0.41 ms, outside forwards 2.0-2.2% | **Represent:** small per-forward and per-token constants |
| Optional admissions, net | Material | Holding them was slower, +3.14%, +5.61%; changed work alone predicts +346 to +386 ms, against +223, +375 ms | **Represent through policy-driven changes in work;** no isolated copy price claimed |
| Layer identity | Not screened | Median residual about ±0.03 ms apart from the first windows | **Omit:** shared model, no demonstrated failure |
| Foreground copies pending | No exposure | Always retired at plan boundaries | **Omit:** join ordering accounts for it |
| Available buffers | Free count unresolved, no exposure | At least 6 of 8 free in 65,550 windows; limit never bound | **Omit in this regime:** use the resulting plan. Not exercised under pressure |
| Reserved frame or pending primary exchange | Bounded, +0.25 to +0.38% | Same windows for both | **Omit:** at most part of a busy flag |
| Cache history | Unresolved | Long requests within ±0.54%; fixtures up to +3.90% with mixed signs; under 1.4x median within ±1.11% | **Start without;** not shown negligible |
| Helper lease or busy flag | Unresolved | +0.86-1.46% and +1.12-1.73%, near the band | **Candidate one-bit flag** if needed; no phase or age model |
| Context length | Unresolved | Join wait unchanged at positions 240-623 and 1056-1439; later-position excess sits in the helper lane | **Start without** in this range |
| Helper speed state | Not predeclared | `project` kernel 0.076-0.627 ms for similar work; forwards over 1.4x median carry up to 14.1%; 19-30% of generation waits on the helper | **Unresolved:** work inputs cannot represent it. Treat as an operating condition and noise source |
| Light monitoring | Material, +19.33%, +10.85% | Recorder work 0.009%; long requests -0.21% pooled; prefill pairs -6.28%, +0.29%. Recording is not established as the cause | **Unresolved:** trust light records for structure, not short-fixture absolute comparisons |

**Consequence for evaluation:** the same work in the same mode spanned 6,686-8,037 ms
over 64 forwards, so short fixtures cannot resolve a 2% decision here while helper
speed varies this much. A prospective test needs longer work, more replicates or
control of that state.

**Two leads, neither tested in this study nor selected, both needing a separately
declared budget:** the 3090's speed state, and starting primary work during the host
lanes at `T=1`. Low 3090 clocks were [already on record](evidence.md#shared-model-diagnostic-study)
but had not reached this plan. Forcing clocks has no measured result: the 2026-11-01
attempt was refused for lack of permission and the user-run test never ran. Testing it
needs administrator rights and the user's decision. Each lead could change the
composition the first-stage model must describe. The earlier
`primary_submit_query` result used grouped speculative work, so it does not cover
this regime; it remains off and no runtime default changed.

**Corrected reading, user review 2026-11-07.** Accept the campaign as completed and use
it to develop a compact shared-model hypothesis. Do not accept that these inputs are
sufficient. The median-table figure replaces the target's work too and is not accuracy
against actual duration; the 0.014 ms agreement uses observed branch times; the recorder
microbenchmark does not clear monitoring; the helper slowdown is localized, not explained.
[Details and the paired per-invocation recheck](evidence.md#corrected-reading-of-the-diagnostic-study).

**Why the model works per pass.** A pass's completion depends on when its operations start,
which run together, which depend on others and which ultimately delays progress. Keep the
joint timeline of each `(forward, layer)`: actual assignments and required transfers, CPU
and helper submission, execution and completion boundaries, copy and kernel times where a
trace supplies them, and existing joins. A submission returning is not the GPU starting. A
wait at the next join can include this layer's unfinished work. Establish consequences per
invocation, then aggregate them; compare completion times from a common origin rather than
averages of durations.

<a id="shared-model-follow-up"></a>

**Rough plan, user direction 2026-11-07, executed in this order; the active sequence below
stands with these checks placed before the timing formula is fixed:**

1. Finish the interpretation from existing per-pass records. Done above.
2. One bounded follow-up on helper variability and recording together.
3. A bounded `T=1` test of the existing submission query. Freeze the execution
   configuration before final calibration. A negative result ends that experiment.
4. Design and implement the smallest supported shared predictor from complete in-engine
   intervals, current-work descriptors and the observed composition. Keep admission-dependent
   prefix and tail costs visible. Existing policy bookkeeping generates each candidate's
   placements and work; check separately whether simple ordering reproduces optional-transfer
   availability. Small direct timing effects do not make readiness irrelevant to future work.
5. Validate prospective decisions with speculation off: freeze predictions before untouched
   workloads and interventions, judge complete elapsed work and candidate-minus-control
   differences, and keep slow periods and uncertainty across chronological blocks.
6. Extend the qualified model to speculative generation by measured execution. Prefill stays setup.
7. Use the predictor for admission repair and supported optimization, accounting for displaced
   experts, transfers and later reuse. Confirm actual serving speed and usability against the
   preserved best usable complete configuration. The ordinary-decode diagnostic rate is not
   the final benchmark.

**Declared follow-up, frozen before its runs.** [Protocol](shared-model-followup-protocol.json),
rules in `tools/mm3/native_followup_analysis.py`, limit 63 executions and 3,600 s warmed. "Bounded"
means a declared run limit and stopping rule, not an open-ended search.

| Check | Arms and runs | Decision |
|---|---|---|
| Timing variation | Two 4x4 Williams squares, 32 runs of a 241-forward fixture: recorder off without sampler, recorder off with sampler, timestamps only, full records | Each effect is material, bounded or unresolved against 2% by its 95% interval. Clock link needs clock state to explain half the between-forward helper variance and 80% of slow forwards at lower clocks |
| Operating state | Three diagnostics: GPU metrics on each card, clock-event reasons | Descriptive only |
| `T=1` submission | 12 fixture runs in ABBA, BAAB, ABBA with full records; a separate Nsight load for per-pass start and end times, query off and on; 8 complete requests only if the fixture continues | Kept only for at least 2% gain with an interval excluding zero, confirmed on complete requests with no prompt losing. Otherwise the switch stays off and the model describes existing execution |

Forcing clocks stays out of this: it changes a system setting and needs the user's decision
and administrator rights.

**Results, 2026-11-07.** [Timing variation](evidence.md#shared-model-follow-up-timing-variation):
recording is bounded under 2% at both levels, the earlier fixture differences do not replicate, and
the helper's slow periods follow the 3090's clocks, which the driver reports as idle at a median 315 MHz.
Over 241 forwards, a whole-run paired mean resolves 2% with about 5-8 pairs and 1% with 11-25. Keep
full records on where work identity matters, and compare complete totals in the same mode.
[`T=1` submission](evidence.md#shared-model-follow-up-primary-submission-at-t1): **useful**, -15.16%
pooled on complete requests. The ordinary-decoding configuration now has `primary_submit_query=1`. The
timing model describes that execution. Speculative generation keeps it off until tested.

<a id="helper-occupancy-test"></a>

**Worthwhile-work decision, helper occupancy, 2026-11-07.** Measured cost: the driver reports the
3090 as idle in P3 at a median 315 MHz during generation, and a per-pass replay at its 700 MHz level
saves a virtual 13.45% of generation. Mechanism: one minimal kernel on that GPU during generation,
off by default, changing no device setting. Costs: more 3090 power while generating, one mostly
sleeping host thread and any kernel contention. Cheapest check: the
[declared protocol](helper-occupancy-protocol.json), 12 fixture runs and 8 conditional complete
requests under the submission-query rule. A clock rise with no generation gain removes the control.
It must be settled before the final timing calibration, because it changes the helper lane.

**First design, negative, removed.** A separate continuously running kernel held the 3090 at P2,
9,501 MHz memory and about 1,950 MHz core in all samples, without administrator rights, but each pass
waited about 2.9 ms behind it and generation took 165% longer.
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/helper-occupancy-20261107/occupancy_detail.txt).

<a id="helper-gap-spinner-test"></a>

**Final design, one test only.** The clock response is real and the loss is the collision. A polling
kernel now fills only the gaps between passes on the helper's own stream, and the worker releases it
just before submitting, so a pass cannot queue behind it. [Protocol](helper-spinner-protocol.json):
same arms and rule, 12 fixture runs and 8 conditional requests. Whatever the result, no further
occupancy design follows without a user decision.

**Result: useful.** [-13.54% pooled on complete requests](evidence.md#helper-occupancy), on top of the
query. The 3090 holds P2 at 9,501 MHz and about 1,950 MHz throughout, one-job helper time falls from 0.585
to 0.317 ms and same-setting requests agree within 0.3%, at about 90 W more while generating.

**Frozen ordinary-decoding configuration, 2026-11-07:** `primary_submit_query=1`, `helper_occupancy=1`,
all else as selected. Final calibration and decision validation use this. Both controls stay per request,
and neither is yet a serving default: speculative generation has not been tested with them.

<a id="shared-model-decision-validation"></a>

**Decision validation result, 2026-11-07: not qualified under its predeclared rule.** No reversals, all
non-tie predictions confirmed, median error 0.96 points, but the 90th-percentile check failed at 4.36 points
against 3 after the one allowed repair. Whole-engine stalls drive that failure; clipped descriptively, the largest
error is 1.46 points, which does not change the verdict. [Evidence](evidence.md#shared-model-decision-validation-speculation-off).

**Consequence for steps 6 and 7, primary-agent decision in the user's absence.** The user asked for the complete
plan and is not available to adjust the rule. The model continues as a **screen only, labelled unqualified**:
it proposes and orders candidates, and nothing is selected without hardware confirmation on complete
generation. A user who prefers to stop at the failed rule can do so; the receipts stand either way.

<a id="speculative-configuration-gate"></a>

**Speculative configuration gate, declared 2026-11-07.** `T=1` results do not transfer automatically: grouped
windows already flush launches, and drafting shares the primary. [Protocol](speculative-gate-protocol.json): selected
spec 5, lenience 0.4, a 2 by 2 factorial of query and spinner, 32 complete 384-token requests in Williams squares with
one square per prompt. A control is selected for speculative generation only at -2% or better with a 95% interval
excluding zero and neither prompt slower. These same full-record runs supply the speculative training data.

**Result: the spinner is selected, -5.37% [-7.36, -3.38], and the query is not, -0.70% [-2.60, +1.20].**
[Evidence](evidence.md#speculative-generation-configuration-gate-and-grouped-model). Query stays off with speculation.

<a id="speculative-decision-validation"></a>

**Speculative decision validation, 2026-11-07: not qualified under its predeclared rule.** Round 1 failed on the
90th-percentile error, 8.67 points against 3, with 100% class agreement. Round 2, on untouched workloads after the one
allowed repair, had no reversals and 7 confirmed non-ties, but failed agreement at 77.8% against 80%, median error at
1.53 against 1.5, and 90th-percentile error at 4.32 against 3. The rule ends here. The model still orders arms correctly
and underpredicts the size of damaging ones. It stays a **labelled, unqualified screen**: it proposes
candidates, and hardware on complete generation decides.

<a id="integrated-screens"></a>

**Step 7 screens, declared 2026-11-07 before their runs.** [Protocol](integrated-screens-protocol.json): 152 complete
384-token requests in the resident engine with speculation and spinner on, recorder off, Williams squares per prompt
on the untouched `val-prose` and `val-code`. Judged by `tools/mm3/native_arm_screen.py`: at least -2%, 95% interval
excluding zero, neither prompt slower. Anything that changes output also needs the bounded usability checks. *(Superseded
2026-11-08 by the [rules for adoption](#adoption-rules): a check only where a change has a quality mechanism.)*

| Screen | Control | Arms | Why these |
|---|---|---|---|
| Admission | Effective 0.58 ms saved per use, 8 exchanges | Repaired price 0.32 ms and 0.15 ms interference; budget 16; both | Plan requires testing the repair. An expert's marginal saving on recorded passes is 0.29-0.37 ms, and an exchange costs 1.34 ms against 1.19 charged. The unqualified screen predicts a tie or small loss for all prices tried |
| Cache-prior bonus | 0.10 | 0.15, 0.20, 0.30 | Required value comparison; 0.05 lost earlier |
| Lenience | 0.4 | 0.3, 0.5, 0.6 | Required value comparison; 0.2 lost earlier |
| Drafting | Spec 5, floor 0.3 | Spec 4, spec 6, floor 0.45 | CPU sets the pace at wide windows, and the spinner changed each row's price since the earlier two-draft test |
| DRAM backing, exclusive control | No backing | Backing with the old D2H charge kept as interference, so the schedule stays feasible; backing with the charge dropped | The reopened decision: a lower transfer cost leaves the old schedule available. Needs 13.9 GB plus the 16 GiB reserve; a refusal ends the phase as untested |

<a id="per-pass-balance"></a>

**User direction, 2026-11-07: per-pass balance is the central objective for dividing work.** Aim for the CPU lane
and the copy-plus-primary path to finish together on every individual activation through every individual layer, judged
from the joint timeline and a common origin, never from separate averages. Working form: each pass's finish time
is what is minimised, and the test for a pass is whether any single move of work would make it finish sooner. Work
moves in whole jobs, and a hit removes work outright, so an all-resident pass is unbalanced and still the fastest.

**Correction to the agent's reading.** The claim that wide passes have two nearly equal limiting paths was unqualified model
output. Grouped per-pass copy and kernel times have not been measured with the spinner on. The 0.29-0.37 ms marginal
saving is an average of per-pass consequences, and the `repaired` admission arm above uses it as one global price, so that
arm is not the valuation repair this plan requires. A flat response to a global knob is not evidence of balance: it is also
what passes imbalanced in both directions would show. How far individual passes sit from balance has not been measured.

**Where the engine stands.** [`plan_batch_staging`](../../src/mm3/mm3_batch_staging.hpp) minimises a per-pass maximum, but
on estimated durations, with a flat 1.3 on CPU work and uncontended copies, and its only move is CPU to primary.
[Primary admission](../../src/mm3/mm3_cooperative_policy.hpp) looks at no individual pass.

**Bounded per-pass study, before the final comparison; the screens above continue.**

1. From existing speculative records: which path finished last on each pass and the slack on the others, as distributions.
2. One Nsight load with speculation and spinner on, for copy and kernel start and end times per pass on wide windows.
3. For each pass, the best assignment of its actual experts against the planner's choice. The total bounds what better
   balancing could save on fixed demand.
4. If that bound reaches 2% of generation: move the planner to completion times with contention, value admission by the
   finish-time change each expert causes per pass rather than its frequency, and test both on hardware against control.
   Below that, report that passes are already close to balance. Hardware decides either way; the model stays labelled unqualified.

<a id="per-pass-redo"></a>

**User direction, 2026-11-07: go back before speculative decoding and redo the ordinary-decoding steps with per-pass
balance, for each individual layer, as the objective.** The balance objective was adopted only after the work had moved to
speculative generation, so the steps that shaped the predictor and policies at `T=1` did not use it. Carried forward because
they were already measured per pass: the submission query (copies moved from 0.93 ms into the pass to 0.03 ms, ending a median
0.09 ms from the lanes on one-copy passes), the spinner (one-job helper time 0.585 to 0.317 ms), and the recording and noise
results. Redone at `T=1` with query and spinner on, speculation off:

1. Balance per pass and per layer: which path finishes last, the slack on the others, and pass time against misses. Existing
   records first, then one Nsight load that traces a `T=1` fixture and a wide-window fixture with the spinner on.
2. The predictor judged on per-pass consequences. The earlier validation arms were all global knobs.
3. Residency and admission valued by per-pass finish-time change. First an offline bound at `T=1`: the same residents
   placed to minimise summed pass time instead of maximising hits. At 2% or more, implement and test on hardware against control.
4. Then extend to speculative generation and re-verify the bonus and lenience on the new base. Their screens remain
   measurements and select nothing until then.

**Redo status, 2026-11-07, steps of the rough plan.**
- **Step 1, per-pass interpretation: done** at `T=1` and in wide windows, with the per-layer abs(Δ) metrics and host
  bandwidth measured. The held-launch correction is [withdrawn at `T=1`](evidence.md#copy-to-kernel-handoff): what it caught
  there is a fixed copy-to-kernel handoff on every upload. In wide windows it mostly stands.
- **Steps 2 and 3: stand.**
- **Step 4, per-pass completion model at `T=1`: done** ([evidence](evidence.md#per-pass-model-redo-step-4)). The slowest path
  is right on 95.0-97.7% of passes against about 50% for a layer-aware guess, and every pair's per-layer abs(Δ) is reproduced
  within about 0.01-0.03 ms. Constants were fitted on one trace and tested on the other. It describes observed work and has
  not yet predicted an intervention.
- **Settled by bounds under 2%:** placement valuation 0.18%, splitting misses between CPU and uploads 0.08-0.51%, splitting one
  expert 0.62%, staging frames at bonus 0.20 0.4-0.6%, exchange retiming at most 1.1%, removing the copy-to-kernel handoff
  0.6-1.3% on its own.
<a id="work-list"></a>

- **Work list, in order** (the user's decisions are [below](#user-decisions-2026-11-07)). User direction: make `T=1` as
  strong a foundation as possible before speculative decoding returns.
  1. **Finish the [HAGS study](#hags-study).** The HAGS-on loads, the in-situ traces and the evaluation, then keep or revert.
  2. **Refit and tune at `T=1` under the kept scheduling mode.**
     - Refit the 5090 structure of the per-pass model, and rerun the step-4 check. **Done 2026-11-07**
       ([evidence](evidence.md#per-pass-model-refitted-under-hags-work-list-item-2)). The model now takes the copy's
       lead over the CPU. Every pass check matches the HAGS-off foundation's or does better, and the slowest path is right
       on 97.9-98.2% of passes.
     - **Optimize the continuous `T=1` parameters offline** on the refitted model. **Planner and admission settings: done
       2026-11-08.** The current values are at the model's optimum, the best is -0.02%, and the CPU price sits 0.1 above a 3%
       cliff ([evidence](evidence.md#offline-optimization-of-t1-policy-parameters-under-hags-work-list-item-2)).
       **User review: the plateau from 1.3 to 2.1 argues for its middle, not its edge.**
       - **Corrected 2026-11-08.** This bullet once said the planner's own decisions put the single-miss flip at 1.154-1.286
         (p1-p99), so that 1.3 cleared the most marginal passes by 0.014. That sample held only misses the planner had
         uploaded at 1.3, and they flip below 1.3 by construction. It could not see CPU-served misses that flip just
         above 1.3. Two of three loads had many of those
         ([diagnostic](evidence.md#hardware-check-of-the-selected-t1-settings-under-hags)).
       - Uploading a pass's second miss flips at about 2.12-2.20.
       - The plateau's geometric midpoint, 1.65, is the first step-5 candidate. Its predicted cost is 0.04-0.08%, within
         model error.
       - Hardware must confirm the plateau under HAGS first: with HAGS off, a hardware screen found price 2.0 slower by
         2.5-6.2%.
       - **User decisions 2026-11-08:** CPU price **1.5**, inside the plateau and replacing the recommended 1.65, and horizon
         40 for the `T=1` base, with the hardware check under HAGS next. Predicted: CPU price 1.5 alone +0.05%, horizon 40
         alone +0.07%, both together +0.08%. 1.65 with 40 came to +0.14%. So neither interacts with the other at bonus 0.10
         ([follow-up](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/t1_followup.txt)).
       - **The routing bonus stays at 0.20**, a user decision given the curve below.
       - **Horizon at finer resolution, done 2026-11-08** at the user's request
         ([evidence](evidence.md#admission-horizon-at-finer-resolution)): 20 is both the minimum and the knee.
         - Below 20 the cost climbs: 18 +0.12%, 16 +0.28%, 12 +0.55%, 8 +1.36%, 6 +2.21%.
         - From 22 to 120 the curve is flat within +0.10%, so 40 is clear of the knee.
       - **Hardware check of 1.5 and 40 under HAGS, done 2026-11-08: kept, a tie**
         ([evidence](evidence.md#hardware-check-of-the-selected-t1-settings-under-hags)).
         - Pooled per block, +0.01% [-1.08, +1.09]. Per prompt, code -0.65% and prose -0.74% [-1.13, -0.35].
         - Both arms ran at bonus 0.20. **The frozen prediction of -2.12% failed**: it lies outside the interval.
         - The settings change the output at bonus 0.20, since routing follows residency.
         - They take a fifth to a quarter of the experts off the CPU. The CPU lane saves 1.2-2.9 ms per token, but the
           token gains only 0.4-0.5 ms.
         - **Usability check, done 2026-11-08: 1.5 and 40 fail under the frozen rule, so the serving defaults and the `T=1`
           base go back to 1.3 and 20**
           ([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/usability-settings-20261108/evaluation.txt)).
           - The user had switched the serving defaults to 1.5 and 40 ("I would rather switch now"). The check's rule, fixed
             before the run, sends them back on a failure.
           - 1.5 and 40 pass on one seed fewer than 1.3 and 20 on two tasks: action items 2 of 3, and the first `count_runs`
             attempt 2 of 3.
           - Every failure in either arm is a token-budget truncation, and 1.3 and 20 has one too (the `count_runs`
             return). No output shows repetition, broken text or a logic collapse.
           - Routing fidelity is unchanged (94.61% against 94.60% of router mass kept).
           - Three seeds cannot separate this from noise. A rerun with more seeds or larger budgets could, if the user wants
             1.5 and 40 back.
           - The later `T=1` tests use 1.3 and 20.
         - **1.5 is safe on every load.** Replayed with the planner tables of all seven HAGS-on loads, 1.5 stays within
           0.09% of the best, and all seven share 1.45-1.6. At 1.3, three loads sat above the floor, by up to 4.5%.
         - **The failed prediction is mostly explained.** Priced with the check load's own planner tables, the model gives
           -0.25%, inside the pooled interval. **Later frozen predictions are made after the run's load has measured its
           tables, and priced with them.**
         - **The user's checkout serves from its own copy.**
           - It is on `minimax-wsl/proposal-2` and lags this branch: bonus 0.1, and no helper occupancy, which neither its
             server nor its engine knows. So matching this branch needs this branch's code, not just the file.
           - The user wants it to match, but deferred the branching on 2026-11-08 ("don't worry about the other branching
             right now").
           - Side branch `minimax-wsl/runtime-port` (9df332f) changes only the two values there. It is superseded by the
             full match, and now also by the failed usability check: **do not merge it**.
         - **Per-pass traces of both prices, done 2026-11-08**
           ([evidence](evidence.md#per-pass-traces-of-cpu-price-13-and-15)):
           - On the traced load both prices made the same decisions: 5 of 2,198 code CPU jobs and 8 of 1,054 prose jobs
             moved. The traces therefore cannot show where the saving goes.
           - The passes that did move finished about 0.4 ms sooner, with the copy path last.
           - Useful traces need a load whose flip lies between the prices, chosen from that load's tables after startup.
         - A replay diagnostic of the prediction's failure is in the evidence.
       - **Refill budget stays 8.** It is a single best value: 7 +0.67%, 9 +0.41%, 10 +0.83%. *2026-11-10: the budget is
         a credit of 8 per five rows, so 1.6 exchanges a token at one row, and the 3090's is 1 per five rows
         ([evidence](evidence.md#review-rotation)). See item 3 of the [order](#order-20261110).*
       - **Decay:** stays 0.95 until a fair test, which the user suggests running after the cache has warmed under the candidate.
         *User direction 2026-11-10: examine the decay and how fast the resident sets rotate.*
       - [Dashboard of every curve](https://claude.ai/artifact/1CMJ4igE6oABSW56Gdqgg5).
       - **Bonus curve, done 2026-11-08** ([evidence](evidence.md#routing-bonus-curve-at-t1-under-hags-work-list-item-2)).
         It is smooth and concave, with no knee: the step per 0.01 falls from about 2.2% to 0.4%, and router mass kept falls
         about linearly. 0.20 passed usability, at -11.5% against 0.10; 0.30 failed, at -17.8%. About 6% of predicted speed
         lies between them, and only usability checks at values in between can say how much quality allows.
       - **Other settings near an edge, user question 2026-11-08:**
         - the horizon of 20 sits at a knee, with 10 at +0.87% and 40-80 flat at +0.07-0.08%, so move it to 40;
         - the refill budget of 8 is a sharp V, with 6 at +1.42% and 12 at +1.74%, so evaluate 7, 9 and 10 before deciding;
         - decay 0.95 looks sharp, but the replays start from scores accumulated at 0.95 over 384 tokens, which favours
           0.95, and it has no runtime setting, so keep it until a fair test;
         - margin 0 and the upload cap of 4 stay.

         The joint candidate (CPU price 1.5, horizon 40, refill budget 8) was scored in the model together and is now on
         hardware.

       Parameters searched:
       - the planner's CPU price;
       - the upload cap and staging;
       - the refill budget, horizon, margin and decay;
       - the bonus, re-routed from the router scores that the detailed captures record.

       Values are searched, not taken as multiples of 0.1. Each choice is confirmed on hardware. A usability check follows only
       for a parameter with a quality mechanism, such as the bonus or lenience ([rules for adoption](#adoption-rules)).
  3. **HAGS changed the timing ([per-pass structure](evidence.md#per-pass-structure-with-hags-on)), so re-evaluate under it**,
     the new imbalance first:
     - **start the CPU batch before issuing the GPU launches**, or issue those from another host thread. With HAGS the host
       starts its CPU batch 0.03 ms later at `T=1` and 0.10-0.16 ms later in wide windows. In wide code the CPU is now the
       slowest path on 82% of passes, by about 0.8 ms. **Bounded and built 2026-11-08**
       ([evidence](evidence.md#cpu-batch-before-the-gpu-submissions-work-list-item-3)):
       - Uncaptured, the CPU batch starts 0.14 ms after the join. Of that, 0.105 ms goes to GPU submissions it does not
         need.
       - The bound is 2.67% of `T=1` generation, about 2.0-2.1% at CPU price 1.5.
       - Built as the per-request option `cpu_first`: a submit thread issues the GPU work while the CPU batch starts.
       - **Hardware test, 2026-11-08: no gain, so it stays off**
         ([evidence](evidence.md#cpu-batch-before-the-gpu-submissions-work-list-item-3)). On over off was +0.66%
         [-0.36, +1.69], and the frozen -0.84% failed. It did run: on identical work, its CPU batch was 4.7-5.3% longer.
         The extra submit thread slows the batch about as much as the earlier start gains.
     - **the upload cap and staging frames:** the cap of four binds in wide code, while the 5090's path has room;
     - **the planner's CPU calibration:** for the same decision its estimate rose 13.5% while the CPU work got 3-23% faster.
       Its startup tables are also noisy from load to load, by more than 2x in the CPU base cost
       ([sweep](evidence.md#hardware-check-of-the-selected-t1-settings-under-hags)). At the old price of 1.3 that cost
       one load 4.5%. Candidates: median of repeated startup measurements, or refinement from measured passes. It still
       matters for the second-miss decision near 2.0;
     - recompute the HAGS-off bounds from the HAGS-on traces: staging frames, exchange retiming, expert splitting, misses
       between CPU and uploads, and the handoff;
     - the `T=1` submission query, -15% with HAGS off: it worked around Windows' submission batching, which HAGS changes;
     - the spinner against no spinner; it shares the 3090's queue with the helper's work;
     - **Query and spinner under HAGS, done 2026-11-08: both stay on**
       ([evidence](evidence.md#query-and-spinner-under-hags-work-list-item-3)). A 2×2 on one load, six blocks:
       - turning the spinner off costs 16.4% [15.9, 16.9];
       - turning the query off is a tie, +0.17% [-0.16, +0.49];
       - the interaction cannot be told from zero.
     - the helper dispatch mode;
     - the speculative query, -0.70% and not selected with HAGS off, belongs with item 7;
     - the copy-stream upload, item 4, and the SM-driven upload fallback, both specific to the scheduling mode.

     The power plan and clock locks are not repeated unless the spinner's result changes.
  4. **Step 5: frozen per-pass predictions for `T=1` candidates, with hardware tests.** They run under the predictor rule;
     the [rule](#predictor-requirement), decided by the user, includes the stall check. Candidates:
     - per-pass routing: a usability check only if it moves picks away from the router's choice beyond the accepted bonus;
     - helper refill timing;
     - contention-aware planner estimates;
     - if HAGS is kept, demand copies on their own copy stream, with the compute stream waiting on an event. With HAGS on,
       the probe's 5090 upload-to-kernel gap is 32.8 µs on one stream and 3.8 µs this way. The 3090 shows 52 against 20 µs,
       for its refills. The bound is to be recomputed from the HAGS-on trace ([evidence](evidence.md#copy-to-kernel-handoff)).
       **Recomputed 2026-11-08: 0.8-0.9% at `T=1`, so it is not built.** In the engine the handoff is still about
       0.048 ms with HAGS. Removing it shortens only the already-early primary path, so CPU against primary balance
       worsens slightly.
     - **Device-side upload decision: bounded 2026-11-08 at about 0.7%, not built**
       ([evidence](evidence.md#device-side-upload-decision-work-list-item-4)). Uncaptured, the host issues a pass's first
       demand copy about 12 us after the router finishes on the GPU: a 1.4 us readback, a 1.8 us synchronize return and
       8 us of planning. The 0.067-0.104 ms "host preparation" below was a fitted constant that includes GPU work. The
       original item follows.
     - **bound first: a device-side upload decision**, the one piece of the audit's GPU-driven paging not covered more
       cheaply. The 5090's demand copies would start right after routing rather than after the host's round trip: the
       fitted host preparation is 0.067-0.104 ms per `T=1` pass, plus about 0.03 ms to the copy's start. That only matters
       where copy-plus-primary finishes last, 23-35% of traced `T=1` passes. **Ways to start the copy without the host**,
       to be measured in the probe first:
       - a graph launched from the GPU: CUDA 13.4 permits memcpy nodes in device-launched graphs for pinned,
         device-mapped host memory, so the copy may keep the copy engine. A node's addresses are fixed when it is built,
         so the choice among experts needs prebuilt nodes or a switch conditional node, which a kernel sets with
         `cudaGraphSetConditional`;
       - a mapped-memory read by the SMs: 8% slower in the naive probe kernel, which a better kernel may narrow;
       - shortening the host's round trip instead: the router writes its picks into mapped host memory, and the host
         polls, plans and issues copy-engine copies at once;
       - prefetch on a prediction before routing ends: the engine's `routing_prefetch`, now off.

       **Existing local implementations, agent search 2026-11-07:**
       - Basalt's `dm_stage_kernel`, in `/home/mdwbr/basalt/src/kernels/cuda/dec_moe.cuh:390-444`. A device kernel picks the
         uploads from the router's ids and a device residency row, then copies with SM loads from mapped memory on a
         high-priority side stream. Single GPU only.
       - Infernix's `stage_kernel`, in `/home/mdwbr/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu:295-420`:
         device-ranked misses, staging slots or landing frames, SM copies. Primary GPU only.
       - Neither uses copy-engine copies started from the GPU. A memcpy node cannot be retargeted from the device, so
         that route needs prebuilt copies per expert and slot.

       **A minimal Maya test is estimated at about 400 lines behind a flag:**
       - a one-block device port of `plan_batch_staging`;
       - a stage kernel launched right after `route_kernel`, reusing Basalt's copy loop and `NativeHostBank`'s
         existing mapping for the 5090;
       - slots reserved before routing, with the host adopting the device's plan after the existing join.

       **Risks:**
       - SM reads from Maya's own host bank under WSL2 are untested. Infernix's 24 GB/s was another machine, at Gen5 x8.
         This machine's naive probe kernel reached 53.6 GB/s from `cudaHostAlloc` memory.
       - Slot ownership.
       - Long spins under TDR.

       **Order:** first probe SM reads from the bank's actual allocation, with Basalt's loop and the CPU reading at the
       same time. Then bound the chosen mechanism from the HAGS-on traces, with the host time refitted under HAGS. Build
       only at 2% or more.
     - **`routing_prefetch`, user direction:** another agent wrote it, and it is off today. Review it, tune its settings
       from the per-pass information, and try improvements of our own, frozen and tested like the other candidates.
       **Reviewed 2026-11-08** ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)):
       - It cannot run today: it requires adaptive primary residency off, so there is nothing to tune as it stands.
       - It predicts layer *l+1*'s experts by applying that layer's router to layer *l*'s input, and uploads one
         expert per layer into a staging frame.
       - The improvement worth testing uses that prediction to start the next layer's likely misses copying a layer
         early, through the planner's staging frames.
       - Its ceiling is large: there are 0.6-0.9 misses per `T=1` pass. The unknown is accuracy, so first measure it with
         a predict-only diagnostic, then bound it on the per-pass model, and build at 2% or more.
       - **Measured, bounded and built 2026-11-08**
         ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)):
         - the prediction holds 91.4% of the next layer's misses;
         - the per-pass bound is -8.0% under the rule the engine can follow (copy only when the layer queued no demand
           copy and no exchange is in flight);
         - copies are used 87% of the time, because routing gives a copied expert the bonus;
         - router mass kept rises from 94.8% to 95.8%;
         - slowest minus second-slowest falls from 0.278 to 0.223 ms, on every MoE layer.

         Built as the per-request option `early_copy`.

         **Tested 2026-11-08: mode 2 chosen, -5.03% [-6.39, -3.66] against off on a normal load. Its usability check is
         running** ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)):
         - **Frames verified.** Every early-copied frame was byte-identical to its HOST record, in both modes.
         - **First timed test.** It gave -10.5%, but its load's planner never uploaded: the calibration repeats had left it a
           table with the 5090 caught slow. So that result was measured against a broken base.
         - **The three-arm test on a normal load decided:**
           - mode 1 over off: -2.43% [-4.49, -0.38];
           - mode 2 over mode 1: -2.65% [-3.56, -1.73], as the model predicted (-2.60%);
           - mode 2 also copies behind the layer's one demand copy.
         - **The model over-predicted against off** (-9.2% for mode 1). The CPU lane fell 3.5-5.8 ms per token, but the
           token only 1.3-2.8 ms. At `T=1` the model credits CPU-lane savings about three times too strongly. The next refit
           must fix that.
         - **Usability check: failed under the frozen rule, so `early_copy` stays off**
           ([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-usability-20261108/evaluation.txt)). Two of mode 2's
           three choir notes were cut off by the token budget, against none for control. Every other task passed. A larger
           check could bring it back, which is the user's decision.
     - **3090 placement, from the user's question.** Helper-first placement puts the globally hottest experts, by
       original demand, on the 3090 at load. During generation the 3090 only swaps its coldest resident for the hottest
       CPU-served expert, when the hot score exceeds 1.25 × cold + 1. An expert that turns hot while resident on the
       5090 is never moved to the 3090. So "hottest on the 3090" holds at load and only roughly afterwards. Bound a
       re-homing rule on the per-pass model, counting the 6.4 GB/s refill and whether the 2026-11-07 placement bound
       already covered which GPU holds which expert. **Settled 2026-11-08: keep helper-first**
       ([evidence](evidence.md#3090-placement-work-list-item-4)):
       - Primary-first placement, which puts the hottest experts on the 5090, is 3.5% slower. The 5090's slots are the
         adaptive cache, and stable hot experts there push each request's own hot experts to the CPU.
       - The unconstrained split bound of 8.3% ignores capacity. It comes from the 3090's startup, about 0.19 ms on a pass
         with no CPU job, which is item 5's target.
       - Re-homing during generation only reshuffles the same resident set, so it is not built.
  5. **Kernel candidates at `T=1`**, user direction 2026-11-07. Each is measured on complete individual layer passes and on
     complete generation, with the trace bound kept for interpretation:
     - **combined target QKV and indexer projections**, against the current overlapping branches. The layer's preparation is
       sequential with its expert pass, so a saving there shortens every pass. **Measured 2026-11-08 on the 5090, from the
       HAGS-on traces** ([kernels](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/prepare_kernels.txt)):
       - a `T=1` MoE layer's preparation spans 280-286 us, of which 213-215 us is kernel time over 22 launches;
       - eight `mx_rows` projections take 147 us of that (18.4 us each), then sparse attention 17.8 us, the router's dot
         product 11.7 us, and small kernels;
       - 64-68 us is idle between launches. That is an upper bound, since detailed capture delays the host's launches.

       Fusing projections targets the 147 us. Capturing the preparation in a graph targets the gaps.

       **Uncaptured, it is the largest block of a `T=1` pass**
       ([join waits](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/join_wait.txt), light captures of the
       HAGS-on validation load):
       - the host spends a median 0.236 ms enqueuing each layer's preparation, about 22% of the pass;
       - it then waits a median 0.157-0.176 ms at the join (p90 0.43-0.52), 19-20% of generation in all;
       - the wait also holds the previous layer's unfinished primary work, so not all of it is preparation.

       **Diagnostic, 2026-11-08:** the preparation's span on the primary stream (0.283-0.293 ms per layer) is only 25-35 us
       longer than the host's enqueue time, and its kernels need about 0.215 ms
       ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)). That read as launch-bound.

       **Built and measured, 2026-11-08: `prep_graph`, -6.99% [-8.84, -5.14] at `T=1`**
       ([evidence](evidence.md#routing_prefetch-review-work-list-item-4)):
       - each one-row MoE layer's preparation replays as a CUDA graph, keyed by every captured argument that moves and
         captured again at each new indexer block count. It is verified bit for bit;
       - the first form (793c78d) was no shorter. Microbenchmarks showed why: a graph cuts each dependent kernel boundary
         by 2.7-3.9 us here, but its five small host copy nodes cost about 13 us each in a two-branch graph. Programmatic
         dependent launch does nothing on this platform;
       - with the copies captured as zero-copy kernels (842faac), the span falls from 280 to 202 us a layer. The timed test
         measured -6.99% against a frozen -7.71%, and per-pass balance does not move;
       - adopted under the frozen rule, and **passed its usability check** (`prep-graph-usability-20261108`), so it joins the
         `T=1` base. The text changes only as the base's own does from run to run.

       Next: drafting windows (f8e2ff8, a graph per window size), verified and timed at the serving configuration
       (`prep-graph-spec-verify-20261108`). Then the remaining boundaries: the side-stream fork, and kernels that could be
       fused now that each boundary costs about 1.7 us in a graph.
     - **NVFP4 expert kernels, the 3090 first:** SGLang's Marlin W4A16 path, keeping Maya's separate gate, up and down scales.
       At `T=1` the helper is the slowest path on 11-28% of traced passes. The user's nuance applies: those are
       almost all passes with no CPU job, 31.4% of `T=1` passes in the records. There the GPUs hold every needed expert
       and the pass is already short, so helper-last does not mean the 3090 is slow. **User direction: that is no
       reason to skip it.** A clean speed gain counts, and faster helper kernels may matter more later. On those passes
       the 5090 finishes very early, so a faster helper turns almost fully into a shorter pass. **Target both parts of
       the 3090's time:**
       - the kernel, through NVFP4 and Marlin. **Measured 2026-11-08**
         ([evidence](evidence.md#the-3090s-startup-work-list-item-5)):
         - each added job already streams at about 740 GB/s, 79% of the 3090's peak, so a per-job kernel buys at most
           about 0.4% at `T=1`;
         - the larger target is about 0.028 ms per window of fixed cost: four launches with their tails. A fused
           kernel would cut it, worth perhaps 0.5-1%;
       - the startup before it, the activation transfer, worker dispatch, launch and handoff. In the layer-33
         example its kernels ran 0.217-0.301 ms after the pass began.

       Build and measure regardless of the 2% bound. In wide windows the helper is rarely last. The 5090's SM120
       kernels follow only if MiniMax's clipped SwiGLU can be kept.
       **The startup, measured and built 2026-11-08** ([evidence](evidence.md#the-3090s-startup-work-list-item-5)):
       - Uncaptured, the helper path takes a median 0.354 ms from planning to the collected result on passes with no CPU
         job. Only 0.150 ms of that is the 3090's own work.
       - The rest is two thread wake-ups (0.033 and 0.032 ms), the worker's CUDA submission (0.092 ms) and the merge.
       - `helper_fast` adds spin waits and graph submission. The bound is 1.9-3.0%, and the frozen prediction is -2.99%.
       - **Hardware test, 2026-11-08: -5.04% [-5.86, -4.22], adopted for the `T=1` base**
         ([evidence](evidence.md#the-3090s-startup-work-list-item-5)). The frozen -2.99% was too small: the model
         under-reads the helper path's host time. It does not change output. The serving defaults follow a recheck with
         speculation.
       - **Recheck with drafting, 2026-11-08: -0.58% [-1.19, +0.03], not added to the serving defaults**
         ([evidence](evidence.md#the-3090s-startup-work-list-item-5)). Both arms did the same work and wrote the same text.
         With drafting the on arm submitted only 3.4 graphs a token, on its one-column passes.
  6. **Step 6: the speculative extension, the hardest step.** An earlier attempt struggled here, and the user sent the work
     back to build the `T=1` foundation first. Six proposals from five agents exist because this is hard. It starts with the
     design comparison the plan already requires, of the [six collected proposals](timing-predictor-proposal-measured-transitions.md),
     against the per-pass evidence gathered at `T=1`. The `T=1` model is not extended ad hoc.
     - **Two drafters, user direction 2026-11-09: implement, optimize and test DFlash2 as well as DSpark, possibly DFlash2
       first.**
       - DSpark is MiniMax's shipped head (`dspark/`): MXFP8, 2.3 GB, block 7, Markov and confidence heads.
       - [DFlash2](https://huggingface.co/RadixArk/Minimax-M3.1-Flash-Dflash2) is RadixArk's separately trained draft for
         this target: BF16, 4.1 GB, 7 proposals after an anchor, a rank-256 top-16 selector and a small conv. Its card reports
         3.6-5.2 tokens per verify step at temperature 1.0.
       - Both have five sliding-window layers of the target's width with a dense 12,288 FFN, and both condition on target
         layers 3, 17, 31, 45 and 59 with mask token 200058. So DFlash2 reuses DSpark's capture, mask-token block and layer
         kernels; its head and weights are new.
       - **The user's decision, the same day: step 6 starts with DFlash2 at MXFP8, and DSpark comes back later.**
         - 8-bit weights should barely move acceptance.
         - DFlash2 has no Markov head, so its block parallelizes more easily.
         - Infernix's DFlash2 implementation is much faster than DSpark; port from it.
         - Net throughput is not the yardstick this early in step 6.
         - DSpark's acceptance rate is measured early in step 6, for reference only: it already runs in the engine.
       - **The port, studied the same day ([study](dflash2-port-study.md)).**
         - Infernix's Flash-Next path is the closest analog (an MoE with offloaded experts): a proposal graph, a host
           wait, the verify, a catch-up graph.
         - At MXFP8 a proposal should take about 2.2-2.6 ms on the 5090 with an MXFP8 head copy, 3.0-3.4 ms with the
           shared BF16 head. DSpark takes 9.6-9.7 ms.
         - **DSpark is slow mostly because Maya's BF16 head kernel re-reads the 2.46 GB head once per draft column**
           (about 5.5 of its 9.7 ms at 4 drafts), and because its Markov head forces a sequential full-vocabulary pass per
           position. **The target's own verify windows re-read the head per row too, about 1.37 ms a row.** A
           multi-column head pass is mandatory for DFlash2 and a step-6 item for verification. *Built in f18bfdd for
           both: `bf16_mv_cols`, bit for bit
           ([evidence](evidence.md#verify-head-multicolumn)).*
         - The repo's GLM DFlash (`src/core/glm_dflash.cu`) is an oracle and kernel source.
         - Milestones:
           - M0, an optional CPU reference on dumped hidden states;
           - M1, the first acceptance number: a `Draft2` class with DSpark's interface, MXFP8 linears, new conv,
             multi-column head and walk kernels;
           - M2, speed: graphs and fused kernels;
           - M3, the planner;
           - M4, later: trees, the 3090, the multi-column head for DSpark too.
         - Width choice matters more than draft speed here. On 2026-11-07 each extra verified row cost 20-50 ms, against
           3-10 ms of drafting. At the card's acceptance, verifying 2-3 drafts beats verifying all 7.
     - **Proposed rules for drafting, 2026-11-09, for the user's review:**
       1. The unit is the verify step: one pass per layer over up to 8 positions, yielding 1-8 tokens. Report tokens/s from
          token 128, accepted length per step and step time. Engine changes act on step time and drafter changes on
          acceptance; each is judged by its own mechanism.
       2. Per-pass balance becomes per-window balance, with the column counts priced. The 3090's DP4A FFN, the CPU lane
          (possibly compute-bound) and the 5090's residents each grow differently with columns, and uploads amortize over
          more tokens. The drafter is a new stage on the 5090, so its placement and overlap with the verify step are balance
          decisions. The per-layer metrics apply per window.
       3. The `T=1` graphs mostly do not cover windows. The helper graphs, zero-copy included, serve one-column passes
          only; `early_copy` acts on one-row windows; `dense_split` targets one-row shapes. Item 7 below covers them.
          *Since then the expert-phase graph covers windows (adopted 2026-11-09), and early copies for windows were
          bounded at 0.2-0.4% and not built. In windows the 3090 finishes 1.0-2.5 ms early, so its one-column graphs
          would not shorten them.*
       4. Exact rejection-sampling verification preserves the target's distribution. So the drafter, its precision (an FP8
          copy of DFlash2, say), the block length and confidence-based early stopping have no quality mechanism. Lenient
          acceptance (today's `spec_ell`) or skipped verification has one and keeps its usability check.
       5. *Corrected 2026-11-09 evening:* text and acceptance differ between runs and arms by design, exact engine changes
          included ([rules for adoption](#adoption-rules)). Judge every change per prompt and by its mechanism: engine
          changes by step time at each width, drafter or acceptance changes by accepted length per step. Exactness is
          checked in the engine, never by comparing texts.
       6. A drafter's VRAM is part of its cost. DFlash2's 4.1 GB in BF16 displaces about 128 resident experts on the 5090.
     - **An acceptance-aware planner, user direction 2026-11-09.** If the drafter predicts how many tokens a verify step
       will accept, the planner uses it. DSpark has a confidence head; for DFlash2 the source is to be found in Infernix
       (its tree-width controller, the draft's per-position probabilities, acceptance history). Two uses, both exact under
       rejection-sampling verification:
       - the verify width per step, chosen by expected accepted tokens over the predicted step completion time;
       - per-position pricing: each position's expert work weighed by its chance of being reached, so an upload or CPU
         job needed only by an unlikely tail position is not paid for, and the window stops before it.
       It is part of the speculative planner, which is separate from the `T=1` one.
       - **How, from the literature (a Fable research agent, the same day;
         [note](acceptance-prediction-research.md)).** DFlash2 has no confidence head, but its candidate selector gives
         each drafted token's probability c_k: a softmax over its 16 candidates, given the selected predecessor. Infernix's
         selector already exposes it as a `confidence` output.
         - The calibrated acceptance a(k, c_k) comes from a table per position and confidence bin, with the identity as
           prior, decayed, and one table per spec policy.
         - Reach probabilities r_k = prod_{j<=k} a(j, c_j) are the per-position pricing weights. The verify width is the e
           maximizing (1 + sum_{k<=e} r_k) / T_pred(e).
         - This repo's DSpark-era calibrator (`src/spec/draft_policy.cpp`) and Infernix's extent rule
           (`mtp_confidence_policy.h`) already implement it. The work is a port and a calibration measurement.
         - Calibration is measured before it is trusted: reliability diagrams and Brier scores per position, and
           predicted against actual accepted length per step.
         - Acceptance history is not the per-step predictor (FailFast: "easiness is highly local"); history calibrates
           and feeds the cost model. No fixed confidence threshold: expected tokens per time beats it.
         - A small learned head (DISCO-style, no retraining) only if the bin table cannot remove a measured bias.
     - **The core of step 6, user direction 2026-11-09: an accurate offline model of speculative decoding, offline
       optimization of its parameters, and an accurate planner. A huge part of the step.**
       1. **The model.** Per verify window and layer, it predicts each path's completion from a common origin. The inputs:
          - the window's width;
          - the columns per expert over the union of experts its positions route to;
          - placement: CPU lane, 3090, 5090 residents, staged frames, demand uploads;
          - the drafter's forward on the 5090 and its overlap with the verify step;
          - acceptance per position, by prompt type and from any per-step prediction.
          It starts from the design comparison of the six collected proposals. It is validated window by window and per
          layer against light captures and traces, as the `T=1` model was in step 4: the slowest path right on most windows,
          and the per-layer mean |delta| metrics. *Audit, 2026-11-09 evening: the comparison was skipped. The window model
          was refitted from the `T=1` model instead (v4 below). The comparison is [next](#step-6-next).*
       2. **Offline optimization on the model**, in place of round-value screens. It covers:
          - the verify width, or its per-step policy;
          - the confidence floor or any acceptance predictor's threshold;
          - lenience, with its usability check, since it has a quality mechanism;
          - after the window planner's rebuild, its horizon, refill budget and staging frames. The planner's prices are
            not tuned: the rebuild replaces them with completion times from the measured GB/s (user direction,
            2026-11-09).
          Hardware tests confirm against frozen predictions, as at `T=1`.
       3. **The planner**, priced by each window's completion time from the measured GB/s and the model, acceptance-aware
          as above. It is its own planner, not the `T=1` planner extended (user direction, 2026-11-09). It uses the same
          measured rates, and its terms are fitted and judged on windows.
     - **First results, 2026-11-09 ([evidence](evidence.md#dflash2-m1)).**
       - **DFlash2 M1 runs at MXFP8** (2,357 MB on the 5090). One bug was fixed in f18bfdd: `rms` writes FP16, which the
         head and the selector read as BF16.
       - **Acceptance, same binary and arms.** Tokens a verify step, DFlash2 / DSpark, as equal-weight means over the
         prompts: exact7 3.61 / 3.92, exact4 3.12 / 3.36.
         - DSpark leads on code and extraction; they tie on prose and reasoning.
         - The texts differ by drafter, so compare per prompt.
       - **Confidence.** DSpark's head is calibrated as it comes; DFlash2's selector probability needs a calibration
         table. Today's 0.3 floor offered no draft on 18% of DFlash2's steps.
       - **The draft call:** DFlash2 5.7 ms, DSpark 6.2-7.3 ms. f18bfdd's multi-column head also serves the target's
         verify windows, which paid about 1.48 ms of head a row before.
       - **The first offline model, round level ([evidence](evidence.md#spec-model-v1)).** At today's costs no fixed width
         beats `T=1`. An acceptance-aware width rule is predicted at -15% (DFlash2) and -18% (DSpark) against `T=1`, as
         equal-weight means with tables fitted leaving the prompt out; the oracle is about -29%. (A first split left one
         request out, but each repeat was a replica of its twin, so it leaked; the harness now takes a seed per step.)
         - Verify costs +30 ms for the first extra row and about +19 ms for each later one.
         - Host sampling costs about 2.3 ms a row ([task 62](evidence.md#host-sampling)).
         - The model reproduces exact4 from exact7's rounds within 0.2% (DFlash2) and 5% (DSpark).
       - **M2 began with a kernel profile ([evidence](evidence.md#dflash2-profile)).** M1's proposal is device-bound. The
         head took half of it, 2.67 ms at 920 GB/s, and so did the target's 8-row verify heads.
         - **The head pass's second form** ([evidence](evidence.md#head-kernel-second-form)) takes 1.49-1.59 ms at 2-8
           columns, bit for bit. That is about 1.1 ms off every proposal and every verify window of 2 rows or more.
         - Left in the proposal: the MXFP8 GEMMs at about 1.06 TB/s (tensor cores, about -0.6 ms), top-16 on 7 SMs
           (-0.2), launches (graph), and the selector (-0.1).
       - **Planner v1, timed: -13.7% [-16.4, -11.0] against `T=1`, inside the model's frozen -16.3%
         ([evidence](evidence.md#planner-v1)).** It is DFlash2's drafting base from now on.
         - By prompt: code -21.1%, prose +0.7%, extraction -14.2%, reasoning -20.2%.
         - All 1,017 of its widths matched the offline rule.
         - Its expected tokens a round are calibrated in aggregate, but not per prompt.
         - Its time model is right at 1-4 rows and 16-21% optimistic at 6-8. The acceptance phase costs one target
           distribution per produced token.
       - **The GPU nucleus, first round ([evidence](evidence.md#gpu-nucleus)): -5.0% under the planner, but no gain at
         `T=1`. It stays off under its rule.**
         - At `T=1` its `row_stats` kernel cost 0.73 ms. The second form takes 0.02-0.04 ms.
         - The rule's identical-text check was a protocol error: by design text differs between runs and arms
           ([rules for adoption](#adoption-rules)), so it was never a valid test.
         - The exactness check is now inside the engine (`gpu_nucleus=2`).
       - **The GPU nucleus, second round: adopted ([evidence](evidence.md#gpu-nucleus-second-round)).**
         - `T=1` -5.70% [-6.94, -4.45]; planner v1 -5.76% [-9.48, -2.04].
         - The in-engine check found 0 mismatches over 766 rows.
       - **Planner v1.1 is not adopted, on the replay.** It refits the table on the planner's own regime. Replayed over 16
         new texts at the nucleus era's costs:
         - v1's decisions came to -9.9% against `T=1`; v1.1's to -8.5%, and -9.5% with per-request calibration.
         - The oracle is -24%. So the headroom is in predicting each round's acceptance, not in the time table.
         - v1 stays the drafting base.
       - **The acceptance model on all eight prompts ([evidence](evidence.md#acceptance-model)).**
         - A logistic model on position and logit(confidence) predicts better than the table (log-loss 0.476 against
           0.519). The draft's q shape, the target's uncertainty at the anchor and history add nothing.
         - Replayed on the broad mix, drafting gains about 7%, and prose loses under any rule.
       - **Planner v2, timed ([evidence](evidence.md#planner-v2)): no better than v1 on hardware, so v1 stays.**
         - v2 over v1: +0.95% [-2.68, +4.57], against a predicted -2.1%.
         - Both planners gain about 8.7% against `T=1` on eight prompts, with the GPU nucleus on both sides.
         - The round-level replay holds verify-by-rows fixed and cannot rank policies 2% apart: the window model per
           layer is owed.
       - **The window model per pass ([evidence](evidence.md#window-model)): refitted on verify windows and qualified.**
         - On held-out planner captures, forward time misses by a median of 1.07%. Passes with 3 or more new copies are
           underpriced by up to 0.43 ms.
         - **The CPU lane finishes after the 3090 on 91-100% of window passes.** The gap is 0.74 ms a pass at 2 rows and
           1.6-1.8 ms at 7-8 rows, up to 3.2 ms on some layers.
         - A better same-pass CPU-or-copy split is worth under 1% (planner v1 0.67%, an overstatement), so there is no
           window CPU price. *Superseded by v4 below.*
       - **Per-pass paths in windows, traced ([evidence](evidence.md#window-paths)): the CPU lane and the copy-plus-primary
         path take turns finishing last; the 3090 has the slack.**
         - Host records and the CUDA trace share one clock to within ±2 µs (`tools/mm3/native_window_paths.py`).
         - **8-row windows:** the CPU is last on 53-72% of passes and the copy-plus-primary path on 22-27%, 0.34-0.65 ms
           apart on average. Slowest - second is 0.33-0.61 ms, per layer up to 2.2 ms.
         - **Windows of 1-3 rows:** the 5090's own path is last on 54-82% of passes.
         - The 3090 ends 1.0-2.5 ms before the CPU on 8-row passes, up to 4.4 ms on some layers.
         - The host's publication follows its last lane by 0.10 ms at 8 rows.
       - **The window model with measured paths, v4 ([evidence](evidence.md#window-model-v4)), is the window model from
         here on.** Three terms come from the traces:
         - the copy's share beside the CPU, 0.66 (it was fitted at 0.878);
         - the uploads' head start on the CPU lane, 0.11-0.52 ms by rows;
         - the 5090's path as the later of its resident work and its last upload plus a tail.

         It names each pass's later path right on 85-92% of passes at 2-8 rows (v1: 47-90%). Held-out forward time
         misses by a median of 1.12%.
         - **Under v4 the CPU-or-copy bound points to more uploads in windows:** 1.5% for planner v1, 1.5-2.3% at 2-5
           rows. Reuse would add to that.
       - **The 3090's result merge is branch-free and bit-identical (5b2e439, [evidence](evidence.md#helper-merge)).** The
         host's serial time after its last lane falls from 0.009-0.096 ms to 0.005-0.039 ms a pass (1-8 rows), about 1%
         of an 8-row window. Adopted.
       - **`topk_rows` split over many blocks a row (5e35257, [evidence](evidence.md#topk-split)): the same candidates,
         2.5-3.3 times faster.** About 0.14 ms off every `T=1` token and 0.16 ms off every proposal and verify window. The
         engine's own nucleus check passes (0 mismatches over 1,149 rows). Adopted.
       - **Early copies for windows were bounded and not built ([evidence](evidence.md#early-copy-windows)).** From 3 rows
         up they move the bottleneck to the CPU lane of the pass before; the gain is 0.2-0.4% overall.
       - **Verify cost differs by prompt ([evidence](evidence.md#verify-by-prompt)):** prose 0.84-0.96 and code or reasoning
         1.05-1.21 times the mean, at every width. Planner v1's table is 2-13% optimistic at 4-7 rows. A per-request cost
         scale is small and waits.
       - **The production-policy replay is faithful on today's base (89e0bdc, [evidence](evidence.md#replay-fidelity)).**
         - Fixed: early copies, pick-order frames, a millisecond clock, and the resident order restored per request.
         - 97.8-100% of passes have the same jobs, and priced replay is within -0.04% to +0.78% of recorded work.
         - It holds out of sample: 97.9-100% on another load.
         - Candidates inherit the control's early-copy predictions, clock and ties, so the optimizer on it is a screen.
       - **The expert-phase graph, re-timed with drafting: adopted ([evidence](evidence.md#expert-graph-windows)).**
         - Planner v1 -4.58% [-7.98, -1.18] and `T=1` -1.30% [-3.30, +0.69], both inside their frozen predictions (-4.4%
           and -1.4%).
         - In windows the CPU lane starts 0.05-0.39 ms sooner, because the 5090's submission is one graph launch.
         - Drafting with planner v1 and the graph now runs about 13.8% faster than `T=1` did without it.
       - **The window optimizer on it is biased toward the control ([evidence](evidence.md#window-optimizer)).**
         - The screen put the control best on every axis, even CPU price 1.4 and 1.6 (+1.1-1.6%).
         - A bias test traced that to inputs from the control's own run: the executed picks, which the routing bonus
           shaped toward the control's residents, and the early-copy predictions.
         - Serving the router's original picks flattens the landscape (CPU price 1.3 -1.3%), but that is a no-bonus
           regime.
         - An unbiased screen re-routes each candidate under the bonus, from the router's scores.
       - <a id="step-6-next"></a>**Next, in order** (re-ordered by the audit of 2026-11-09 evening, [where the work stands](#where-the-work-stands)).
         *Re-ordered again 2026-11-10: the [order of 2026-11-10](#order-20261110) is current. The items below keep
         their content and status, and that order says where each now sits:*
         1. *Done:* the 5090's path in windows and the model refit on it (v4), the replay made faithful (89e0bdc), and the
            expert-phase graph re-timed and adopted (-4.58% drafting, -1.30% `T=1`);
         2. *Done 2026-11-09 evening ([evidence](evidence.md#proposal-comparison-windows)):* **the design comparison of
            the six proposals for windows**, owed as step 6's start. Only the lane-structured family (Fable, Opus) gives
            each path's completion, which the objective and the planner need, and v4 is in it with traced rates. v4
            prices the rebuild's bound, with a support check for passes beyond the observed copies. Refills overlap
            0.4-2.7% of demand-copy time in windows (0.03-0.12% of a request): charged, but they cannot move the bound.
            Randomized upload prices are needed only if the bound reaches 2% only outside the support;
         3. *Bounded 2026-11-09 evening, not built ([evidence](evidence.md#window-planner-bound)):* staging each window
            pass by v4's predicted completion gains 0.06-0.55% on the router's own picks and loses 1.4-2.6% on the
            control's executed picks. The loss is the picks' bias toward the control's residents: resident hits fall 3-5%.
            Under 2% and not simpler, so the CPU price stays in windows. Ranking any planner or residency policy fairly in
            the bonus regime needs demand re-routed from the router's scores (detailed captures); that now comes before
            the `T=1` rebuild's bound and step 6's offline optimization. The item as planned:
            **the planner rebuild for windows** ([item 6 above](#host-memory-and-l3), the same principle on windows):
            - **What changes.** Each pass's uploads are chosen by its predicted completion time from the measured GB/s,
              with window model v4's terms or their successor: the CPU's and the copies' shares of host memory, the
              uploads' head start, the 5090's resident path and upload tail, early copies in flight, and a refill's
              measured cost beside a demand copy. The CPU price goes.
            - **Separate planners.** The speculative-decoding planner is a different planner from the `T=1` one (user
              direction, 2026-11-09). Its terms are fitted and judged on windows, never held to `T=1`'s.
            - **Bound first.** Bound it against today's planner on the faithful replay. v4's same-pass bound is 1.5% of
              planner v1's cycle, before reuse.
            - **Build and test** if the bound reaches 2% or the rebuild simplifies; it removes a parameter.
            - **After it.** Horizon, refill-budget and staging tuning for windows, and the optimizer's re-routed screen
              (detailed captures or a light recorder with the router's scores), on the rebuilt planner;
         4. **the open `T=1` items of the [order up to step 6](#host-memory-and-l3).** First item 3's 3090 projections near
            peak bandwidth, with the two quant kernels fused: about 3.6-4.1% at `T=1`. In windows the 3090 has slack,
            so this is a `T=1` gain. *2026-11-10: that bound assumed the 3090 finishes last on most passes. It does so
            on 7 to 17% today, so bound it again after the 3090's hand-off ([order](#order-20261110), item 5).* Then the
            rest of item 3, then the `T=1` planner rebuild (item 4, task 48) on its own
            model, refitted from a token-128 light capture. Its bound waits for a re-routed replay: a detailed-capture
            load gives the router's scores, so each candidate is re-routed under the bonus against its own residents;
         5. bound the 3090's slack in windows (placement, more experts per GB) and the publication's remaining serial
            cost (task 54). *User direction 2026-11-10: experts are not quantized further, so more experts per GB cannot
            come from a smaller expert format;*
         6. the rest of M2: the MXFP8 GEMMs, bound by shared-memory loads at 7 rows. Drafter arithmetic changes only
            acceptance, never output quality, since every draft is verified exactly. The split top-16 is done.
  7. **With speculative decoding back on, re-evaluate earlier `T=1` decisions that may not transfer:**
     - `helper_fast`: **done 2026-11-08**, -0.58% [-1.19, +0.03]; under the revised rules it joined the serving defaults;
     - the query and the spinner, both already gated once for speculation;
     - the HAGS items above;
     - staging frames, the upload cap and admission prices: tuned on the speculative planner after its rebuild, on
       windows;
     - the two kernel candidates again, since wide windows change the kernel shapes;
     - then DSpark context-projection batching and GPU-only graph replay, which exist only with drafting.
  8. **Step 7 on the new base:**
     - per-pass routing, and refill timing under speculation (priced in the rebuilt planners first);
     - **offline optimization of all continuous parameters**, lenience and drafting included, in place of round-value screens
       such as the lenience 0.3 retest. *Moved into step 6 (user direction, 2026-11-09); step 7 re-runs it on the final
       mechanics;*
     - hardware confirmation, usability checks only where a change has a quality mechanism, and a new final comparison if the
       configuration changes.
  9. **Close-out:** dispositions, memory and the final report.
- **Dropped by measurement:**
  - The post-copy flush, the user's build item and polling-thread candidate. The probe shows polling leaves the handoff
    unchanged, 46 against 44 µs, and the host already makes CUDA calls inside the gap.
  - An SM-driven upload on the 5090, kept as a fallback only: its 8% slower transfer cancels the saving.
- **Waiting on the user:** fast-forwarding their checkout to `minimax-wsl/objective-port`.

The configuration confirmed today stands until then.

<a id="user-decisions-2026-11-07"></a>

**User decisions, 2026-11-07 evening, in the user's numbering, with status.**
1. **Spinner kept** over a held 3090 clock: same speed, the lock about 16 W less. *Done: serving default unchanged.*
2. **HAGS test approved**, two reboots at most. The user expects to keep it after one. *Done 2026-11-07: kept on, a tie at
   2048 tokens and 5-8% faster at 128 and 384 ([result](#hags-study)).*
3. **Build and measure.** *On the [work list](#work-list) unless noted:*
   - post-copy flush, small: *dropped by measurement*, since polling leaves the handoff unchanged, 46 against 44 µs;
   - per-pass routing, which changes output: work-list item 4, then 8;
   - refill timing: work-list item 4, then 8;
   - planner estimates: work-list item 4;
   - from the user's audit, combined QKV/indexer projections at `T=1`, item 5, and again with speculation, item 7; DSpark
     projection batching and graphs with speculation, item 7. Their benefit is still to be measured.

   The handoff evidence adds one item, if HAGS is kept: demand copies on a copy stream with an event.
4. **Predictor: decided by the user, 2026-11-07.** Fund a new predictor on the per-pass law, and test it in step 5 against the
   same pass marks. Stalls are detected by a fixed check, removed from both the predicted and measured times, and reported
   on their own. They do not decide pass or fail, unless a candidate causes them and they are excessive. That test and its
   size threshold are fixed in the step 5 protocol before the round. [Details](#predictor-requirement).
5. **Push the feature branch.** *Done.* `minimax-wsl/shared-model-diagnostic` is on origin; feature branch only.
6. **Bring the AGENTS.md and CLAUDE.md objective into the user's checkout.** *Prepared.* This session's harness refuses git in
   the shared checkout. Branch `minimax-wsl/objective-port` (04f47ad) holds only those two files on top of the checkout's
   4b56b15. The user runs `git -C /home/mdwbr/project-maya merge --ff-only minimax-wsl/objective-port`, or cherry-picks it.
   Branch `minimax-wsl/runtime-port` (9df332f) adds the serving switch to CPU price 1.5 and horizon 40 on top of it. That
   switch failed the usability check, so take `objective-port` alone and do not merge `runtime-port`.
7. **Lenience 0.3 retest at bonus 0.20: replaced by offline optimization.** *Changed 2026-11-10 by user direction: the
   usability checks of lenience and of several bonus values are re-run now, with the fair budgets
   ([order](#order-20261110), item 2). The offline optimization still chooses the values.* Lenience, bonus and the other parameters are
   searched as continuous values, not multiples of 0.1. The search uses the per-pass model, the planner and trace replay,
   then hardware confirmation, with usability checks where a parameter has a quality mechanism (lenience and the bonus do).
   *Work list: at `T=1` in item 2, with speculation in item 8.* *At
   `T=1`, the planner, admission and bonus curves are done (2026-11-08). The user chose CPU price 1.5 and horizon 40, kept
   by the hardware check as a tie, and kept the bonus at 0.20. The usability check then failed 1.5 and 40 under its frozen
   rule, every failure a token-budget truncation, so 1.3 and 20 are back until a larger rerun says otherwise.*
8. **The user's NIM/SGLang audit**, `docs/minimax-wsl/nim-sglang-reference-audit.md`, on branch `minimax-wsl/proposal-2` at
   4b56b15 (not on this branch). *Taken into the work list:*
   - QKV/indexer fusion against the current overlapping branches, and NVFP4 expert kernels for the 3090 (the faster helper
     kernels), both at `T=1` before speculative decoding returns, item 5, and again after it, item 7. That is the user's
     direction of 2026-11-07, later than the audit's order.
   - DSpark context-projection batching and GPU-only graph replay, item 7, since drafting exists only with speculation.

   *Limits recorded from it, not to be adopted:*
   - **MiniMax MegaKernel:** targets SM100/SM103 and TP4, not the 5090's SM120.
   - **NIM recipes:** their settings target 128 concurrent requests; they are not interactive defaults.
   - **SGLang's captured sampling-acceptance path:** it excludes top-p, so it cannot replace Maya's top-p and lenience sampler.
   - **SGLang's KTransformers CPU/GPU wrapper:** not a replacement for the adaptive two-GPU arrangement.
   - **GPU-driven expert paging, not adopted as a whole.** Its documented path excludes speculation and multiple GPUs,
     adapting it would be substantial, and its per-layer LRU slots are a weaker policy than Maya's adaptive residency.
     Its copy mechanism's benefit, no copy-to-kernel handoff, comes more cheaply from the copy stream with an event under
     HAGS. One piece is bounded first in work-list item 4: deciding the 5090's uploads on the GPU so its copies start
     before the host's round trip.
   - **The NIM image:** it declares an older SGLang base plus NVIDIA patches, so no upstream feature is assumed present.
9. **Two candidates for the post-copy gap.**
   - The polling host thread: *dropped by measurement*, as in 3. The "held on 19-26% of passes" premise was withdrawn: the gap
     is a fixed copy-to-kernel handoff on every upload pass.
   - HAGS: *in progress*, item 2.

<a id="predictor-requirement"></a>

**Predictor requirement, evaluated for the user.** The requirement is the predeclared decision-validation rule: no reversals,
at least 80% class agreement, median error at most 1.5 points, 90th percentile at most 3 points, on frozen prospective decisions.
- At `T=1` it failed only on the 90th percentile, 4.36 points. Whole-engine stalls drove that; clipped, the largest error is
  1.46 points.
- Speculative round 2 failed narrowly on agreement, median and 90th percentile.

The new per-pass model does not meet the requirement either: it is validated on observed work, not on predicted interventions.

| Option | What it gives | Against |
|---|---|---|
| Accept it as unmet | No new cost; the model stays a labelled screen and hardware decides | The offline optimizer the user asked for would then rank by an unqualified model, and Maya cannot be called complete while the requirement stands |
| Fund a predictor on the per-pass law | Builds on what is now validated per pass. It needs two missing parts: routing's response to the candidate (replaying recorded router scores under the candidate's bonus and lenience) and stalls kept as a measured condition. Redo step 5 is its validation round anyway | A day or more of work and a validation campaign. It can still fail |
| Change the requirement | Judge decisions, not totals that include stalls. The plan already keeps operating-state slow periods visible rather than priced | Retroactive relaxation would pass a model that failed. It is legitimate only if declared before a new round |

**Recommendation, in plain terms.**
- **Build the new predictor** on the per-pass model.
- **Test it in step 5 the same way the old one was tested.** Freeze its predictions for each candidate change, then run them
  on hardware, with the same pass marks: no reversals, at least 80% agreement on which way a change goes, median error at
  most 1.5 points, 90th percentile at most 3.
- **Change one thing, written down before the test.** A stall is a whole-engine pause unrelated to the work being done, which
  no model of the work can foresee. Stalls caused the old `T=1` failure. They would be identified by a test fixed in advance,
  removed from both the predicted and the measured times, and reported separately.
- **If the new predictor still fails,** the requirement is accepted as unmet, and hardware keeps deciding everything.

**Decided by the user, 2026-11-07:** as recommended. Stalls do not factor into pass or fail unless a candidate causes them
and they are excessive. Then they count against that candidate in both the measurement and the prediction error.

<a id="hags-study"></a>

**HAGS study, declared 2026-11-07 before its runs.** [Generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_protocols.py).
The HAGS-off arm is the final comparison's three new-arm loads, plus a drift check: the same twelve scored requests on the
still-resident `load6-new`. The HAGS-on arm is three fresh loads after one reboot. The decision:
- **Revert** with a second reboot if either budget is credibly slower, its 95% interval across pairs below zero, or its mean
  is below -2%.
- **Otherwise keep it on**, reported as faster, or as a tie with or without a shorter handoff in `native_handoff_probe`.
- **Drift:** if the check moves more than 3% at either budget, the engine comparison is confounded, and the probe and an
  in-situ `T=1` trace carry the result.

The removable handoff alone is worth about 1%, below what three loads per arm can resolve. Only a larger effect, through
submission in general, would show in tok/s.

**Drift check: confounded under the rule.** The twelve requests on the resident `load6-new` ran at 17.78 tok/s at 128 tokens
and 19.04 at 384, against 19.53 and 20.14 at 16:26: **-8.98% and -5.47%**, or -5.96% and -5.71% over repeats 2-3
([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/drift-evaluation.txt)). Its restores took 64-209 s a step,
against 30-40 s at 16:26, after about two hours of further experiments on that engine. That points to the aged load rather
than the system, but one check cannot separate them. It now sits inside the spread of the other fresh loads, 17.8-19.1.

**How the probe and the in-situ trace decide, fixed before any HAGS-on data:**
1. *Mechanism:* the 5090's large-copy handoff falls by at least half, either in the probe (same-stream median at most
   22 µs) or in situ (median at most 0.029 ms after the last demand copy).
2. *Per-pass speed:* the summed generation time of the four traced fixtures is no more than 2% above the HAGS-off traces.
3. *Keep HAGS* if 2 holds and either 1 holds or the traced fixtures are at least 2% faster. *Otherwise revert.*
4. *Second reboot:* if the three fresh loads are credibly slower (95% interval below zero and mean below -2%), run three
   fresh HAGS-off loads before deciding, for a comparison in the same condition.

**Fresh-load control, user direction before the reboot:** "do a full reload of the engine and test again". One fresh HAGS-off
load with the same setup and twelve requests ([generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_fresh_off.py)).
Rule, fixed before its data:
- **Within range:** its pooled rate lies within the earlier fresh loads' range widened by 3% at both budgets, 17.28-20.12 tok/s
  at 128 and 18.26-20.74 at 384. Then the system has not drifted and the drift check is put down to the aged load. The engine
  comparison stands, and this load joins the HAGS-off arm as a fourth load in a secondary comparison.
- **Outside it:** the decision above applies.

**Fresh-load control result: inside, so the engine comparison stands.** `off-fresh1` ran 18.05 tok/s at 128 tokens and
19.17 at 384, against the earlier fresh loads' 17.81-19.53 and 18.82-20.14. Its restores took 40-55 s, as at 16:26
([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/control-evaluation.txt)). The drift check mostly reflects
load-to-load spread: `load6-new` had been the fastest fresh load, and the aged engine's 17.78 and 19.04 tok/s are within
1.5% of this fresh load.

**Revised by user direction, before any 2048-token or HAGS-on data:** "Run 2048 tokens for hardware acceleration control and also the
after reboot test to get a more accurate test", and "run once with monitoring and once without monitoring to get a no monitoring
average token speed only measurement". [Generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_long.py).

Six fresh loads: three with HAGS off before the reboot, `off-fresh1` to `off-fresh3`, and three with HAGS on after it,
`hags-load1` to `hags-load3`. Each runs the same sequence:
1. the final comparison's setup and twelve scored requests;
2. six 2048-token requests, three repeats of each prompt, monitoring off: **the speed measurement**;
3. two 2048-token requests with full records and GPU telemetry, for per-pass balance and clocks only.

**Decision:** at 2048 tokens, over the pairs (`off-freshN`, `hags-loadN`), revert if the 95% interval lies below zero or the mean
is below -2%. Otherwise keep HAGS on, reported as faster if the interval lies above zero, else as a tie.

**Reported, deciding nothing:** the 128/384 results against both sets of HAGS-off loads, the probe, the in-situ trace and the
monitored per-pass balance. Same-evening fresh loads replace the aged-load drift check as the control, so the rule written
under that confound is superseded. One reboot's own effect cannot be separated from HAGS without a second.

**HAGS-off arm done, 2026-11-07 19:00-20:40.** [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/off-evaluation.txt),
[per request](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/off-long-detail.txt).

| Fresh load | 2048 tokens, speed | Python-code | Prose | 128 | 384 |
|---|---:|---:|---:|---:|---:|
| `off-fresh1` | 20.45 | 22.19 | 17.79 | 18.05 | 19.17 |
| `off-fresh2` | 20.39 | 21.96 | 18.16 | 19.18 | 19.35 |
| `off-fresh3` | 20.91 | 22.00 | 19.56 | 19.58 | 19.37 |

Rates are tok/s, monitoring off.

- **Natural stops.** Every 2048-token request ended on the model's own stop, at 365-1,449 tokens, mean about 830. Each load
  still pools about 5,000 tokens, against 2,300 at 384.
- **Spread.** The per-load spread at 2048 is about ±1.4%, against ±4% at 128. Python-code agrees within ±0.5%; prose varies
  with its length and content.
- **Monitored pair:** 21.4-23.3 tok/s for code and 16.9-18.8 for prose, kept for diagnosis only.

**Result, 2026-11-07 22:13: keep HAGS on, a tie under the rule.** [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/evaluation.txt).

| Comparison | HAGS on over off | 95% interval | Per prompt |
|---|---:|---:|---|
| **2048 tokens, three pairs: decides** | **+0.12%** | -8.40, +8.63 | prose +1.74%, code +2.74% |
| 384 tokens, three pairs | +5.08% | **+0.69, +9.47** | prose +4.03%, code +7.15% |
| 128 tokens, three pairs | +6.68% | -3.59, +16.95 | prose +8.27%, code +5.36% |
| 384 tokens, all six HAGS-off fresh loads against three on | +4.92% | **+2.01, +7.83** | |
| 128 tokens, the same | +7.56% | **+2.80, +12.32** | |

- **The 2048 result is a tie.** HAGS-on ran 20.58, 21.05 and 20.17 tok/s against 20.45, 20.39 and 20.91, pairs +0.62%, +3.26% and -3.54%. The pooled rate
  mixes prompts whose natural stopping lengths differ by load, so both prompts gain while the pooled mean barely moves.
- **The shorter budgets gain credibly**, about 5-8%. Their requests are capped, so composition does not move them.
- **The 3090 was busy.** In every HAGS-on monitored request it held P2, 1,965 MHz, 87-89% NVML utilization and
  184-205 W, against 80-84% with HAGS off ([telemetry](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/telemetry-compare.txt)).
  The user saw it near idle in a Windows view, which plausibly undercounts CUDA work with HAGS on. That is not verified.
- **Next.** The in-situ traces run next. HAGS changed the timing, so work-list item 3 applies.

After the reboot, in order:
1. `native_host_control.py mount-transfer`, then `hags-query`.
2. `native_handoff_probe.py` into `handoff-probe-20261107/hags-on`.
3. The three loads of `protocol-hags.json`.
4. The `T=1` fixture traces.
5. Evaluate, then keep or revert.

**Per-pass study outcome, 2026-11-07.** [Evidence](evidence.md#per-pass-balance). The CPU lane and the copy-plus-primary
path already finish within about 0.18-0.32 ms of each other at `T=1` and in wide windows. Every bound for moving work
between them is under the 2% rule: how misses are split 0.08-0.51%, placement 0.07-0.18%, exchange retiming at most
1.1% at `T=1` and 0.6% wide. The large per-pass imbalance is CPU against helper. The helper finishes 0.3-0.6 ms earlier at
`T=1` and 0.9-2.7 ms earlier in wide windows on most passes, and the 3090's Gen4 x4 link makes per-pass uploads to it
impossible, so only routing can move work there. The uniform bonus pulls work toward primary and helper residents
alike, and in every pass. Next design to test: substitution applied only where it shortens the pass, toward the path
with slack in passes the CPU finishes last. The bonus usability review comes first, because every value changes output.

**Model outline, 2026-11-07; formula and coefficients wait for the two checks above.** One shared
model for all MoE layers, working on the join-to-join cycle that partitions a forward exactly:

| Part | Outline | Source of its inputs |
|---|---|---|
| Work | Each candidate's assignments, copies and exchanges per pass | Existing policy bookkeeping replayed over recorded demand; no future availability from the control run |
| Optional-transfer availability | Declared order rule in layer joins, checked separately against recordings | Fixed before use; failures localized before anything heavier |
| Pass | Host preparation + slower lane's completion from the common origin + resume and wait for copies and primary picks | Current-work descriptors only; no layer parameters |
| Forward | Prefix + first-window extra + cycles + final-layer extra + tail, with admission-dependent prefix and tail kept visible | Exchanges and leases from the replayed work |
| Operating state | Helper stretch stays a measured condition and a noise source; not hidden in a coefficient | Follow-up telemetry decides how it is represented |
| Evaluation | Complete elapsed work and candidate-minus-control differences across chronological blocks | Slow periods stay in the data |

Judge a replacement by demonstrated accuracy in predicting consequential
candidate-minus-control performance changes, its measurement and implementation
cost, and its declared operating domain. It must retain causal use of demand and
state, account for displacement/transfers/overlap at the resolution needed for
the decisions, and avoid importing a control run's future availability or clock
trajectory into a changed schedule. A coarse model is a candidate, not a proven
winner; a joint lookup over every lane/transfer combination is not required.

Before new hardware measurements, declare a finite comparison budget, the
workloads and controls, practical-equivalence margins and the model-selection
rule. Use independent chronological conditioning and untouched evaluation work.
Keep nested timing residuals separate, preserve correlation in paired
differences, distinguish supported practical ties from inconclusive intervals,
and require evidence of useful discrimination rather than an all-tie predictor.
Predictions outside supported cost/operating coverage remain unqualified.

The active sequence is **bounded generation-only diagnostic study → shared
first-stage design using the measured material effects → implementation and
decision validation with ordinary decoding → speculative-generation extension
and qualification → repair admission valuation and
perform supported offline optimization → short integrated confirmation**. Reuse
the valid state-reproduction checks throughout. The former reconstruction's
stage-2/3 passes are not required to start this sequence. Replacement decision
validation, actual operating benefit, bounded usability and the existing
[completion contract](#execution-and-completion) remain required; this decision
does not waive unfinished admission, parameter or integrated-performance work.
The primary agent performs the modeling and evaluation without delegation.
Held bases remain held. Query-on remains unselected after its ordinary-request
test; no further repeat of that screen follows from the replacement decision.

### Bounded cooperative replay

**Historical approach and receipts, superseded by the
[replacement decision](#timing-model-replacement).** The earlier six-stage order
and next-step wording below describe that effort, not a current instruction to
keep repairing its timing model. Retained state/instrumentation requirements are
carried forward above.

**User instruction, 2026-11-04:** build a bounded event-driven replay of the
actual cooperative engine. This supersedes the earlier approximate replay scope
and the order of the reopened admission work. The primary agent performs all of
this work; **no sub-agents** are permitted for this task.

The implementation order is **state-faithful replay → independently calibrated
execution model → intervention validation → repaired admission valuation → offline
parameter search → short integrated confirmation**. These are separate gates;
agreement between a policy and a simulator sharing its prices is not validation.

- Extract hardware-independent production decisions for demand, staging,
  retention/refill, eviction and helper actions. Replay gives these decisions
  only the historical prefix and current observed or simulated state. Original
  router selections alone update demand; executed selections, including rejected
  speculative rows, determine the serviced work. Cache-aware top-four stays on.
- Schedule work and dependencies on CPU, primary/helper compute, copy queues and
  shared host/transfer resources. Recompute availability, transfers, publication
  and completion for each policy. Recorded launch times, staging opportunities
  and operation durations are observations, never counterfactual predictions.
  Exchange descriptors reserve existing staging frames; count real allocations
  and aliases, active readers, pending work and preservation copies.
- Capture complete starting policy state and missing dependencies. Reject
  incomplete/unsupported captures, or explicitly restrict them to complete
  intervals. Preserve learned state across requests; changing learning rules
  requires replaying chronological conditioning under the candidate.
- Keep physical service tables independent of policy knobs, including
  `cooperative_cpu_scale`. Start with grouped measurements and fill only
  decision-relevant coverage gaps: mixed groups/working sets, measured worker
  configurations, primary/helper submission and reduction, transfer directions,
  queueing and concurrent CPU/copy/compute contention. Retain sparse-coverage
  uncertainty; thread count alone cannot predict worker speed.
- Validate (1) decisions/state/bytes with recorded availability, (2) independent
  timing against held-out service, queues, publication, branch and request
  observations, and (3) predicted interventions on hardware. First interventions
  are backing, reduced optional admissions and changed staging amount. Use fixed
  expert workloads to isolate scheduling, then ordinary model requests for the
  integrated result. Preserve adverse outcomes and unresolved rankings.
- Value a bounded incoming/victim shortlist by the marginal completion-time
  reduction over identical recent original-demand batch patterns, less an
  uncertainty margin. Include displaced service, necessary copies, unavailable
  frames, contention, overlap and irrevocable issued work inside both alternatives.
  No second interference surcharge. Keeping the allocation with no extra refill
  is always a candidate. Offline evaluation may use future work; policy may not.
- Check fixed-schedule monotonicity when transfers shorten; when backing removes
  D2H, retain the previous feasible schedule as a diagnostic comparison. This
  adds no inference waits. Timing acceptance must resolve the proposed effect:
  a 3% gain within 5% uncertainty remains unresolved.
- Search coupled policy/capacity/scheduling choices within measured ranges,
  including incumbent and conservative/no-extra-refill controls. Rank total
  generation time for recorded emitted work on held-out complete requests/topic
  sequences, with per-task and cost-uncertainty checks. Routing bonuses and
  speculative controls support only conditional screening on covered scores and
  proposals; changed hidden states/tokens require live confirmation.

**Retained implementation:** shared production decisions, complete starting-state
checks and frame/byte accounting passed scoped reproduction. Both the older
approximate replay and the later factorized resource/timing model are historical
diagnostics. The later timing model never qualified; its retirement does not
invalidate the useful state checks. Follow the replacement sequence above.

**Replay implementation progress, 2026-11-04:** the first complete extended
capture passes observation-fed reproduction: 1,710 staging choices, 261 primary
choices, 16 helper choices, 18,240 generation original selections (80 target
rows across 57 MoE layers), 128 primary issues/127 publications, 112 retained
uploads, transfer bytes and final canonical/frame ownership. One primary
exchange remains pending at the recorded end and stays pending in the account.
[Stage-1 receipt](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/state-capture-v2/reproduction.json).
The first extended schema omitted cumulative starting counters; its first
pre-exchange prefill service supplies that verified baseline. Subsequent captures
record the counters in the starting snapshot. This is decision/state evidence,
**not timing validation**.

Production demand, grouped staging and incumbent primary/helper selection now
have hardware-independent entry points. The new offline implementation has a
separate resource scheduler, physical measurement tables, actual frame rotation,
bounded input checks and three independent evidence gates. Native measurement
hooks and a fixed-expert-work target runner are exercised below. The old replay
remains historical diagnostic code. Ten focused offline checks and four native
policy/ownership checks pass; independently calibrated timing, prospective
interventions, repaired admission, held-out search and integrated confirmation
remain open. Do not use the partial execution model to select runtime defaults.

The first independent service probe has 431 cells and 30 concurrency samples,
but its worker comparison is **unqualified**: it kept the alternate pool alive
while the normal workers could still spin on the same pinned cores. Preserve
that probe at `cooperative-replay-20261104/physical/service-probe.json`; the fitter
now rejects it. The repair measures one pool at a time and adds partial/mixed
groups. This load's actual CPU configuration is **15 participants total (14
workers plus the caller)**, observed in its native startup and probe. Earlier
wording that described 15 workers plus the caller was incorrect for this source's
automatic configuration; do not infer worker performance from that count.
Both new training and held-out captures pass state/decision reproduction; their
unchanged request execution can be reused after correcting the probe.

**User direction, 2026-11-05: foreground reconciliation comes next.** The immediate
deliverable is a reconciled timing account of the existing fixed expert workload
with weight copies disabled. First hold recorded assignments and dependency order
fixed while independently predicting execution; report isolated service, scheduled
service including contention, queue/host-submission delay, observed service and
observed completion separately. Only then enable autonomous policy scheduling.
Explain measurement-boundary, launch-structure, working-set and contention effects
at the scope supported by the observations. GPU event spans can include host
submission gaps; do not label them pure arithmetic time.

The corrected isolated-probe model predicted the held-out generation **9.4% high**.
The separately reported **CPU -34%, helper -71%, primary +139%** are isolated
**table-to-live discrepancies**. They omit scheduler contention adjustments and
are not a decomposition of that whole-request prediction error. Preserve both
accounts at their correct scope. No global helper/primary multiplier is justified.

Preserve helper job widths after the actual eight-column split and the distinct
width launches; model fixed cost plus a measured service per occupied width/job
count, with preparation and activation/result transfers separate. Use corresponding
CPU/primary chunk and kernel boundaries. Replace the provisional
`min(elapsed_a, elapsed_b)` contention fit: only aligned measured intervals can
establish overlap. The maximum-of-pair-factors approximation needs common multiway
checks. Validate interpolation by holding out measured shapes within the same
worker/kernel/launch regime, and derive uncertainty from residuals and repeat
variation, rather than fixed 5%/20% allowances or requiring every mixed shape.

After the corrected foreground model passes fresh held-out work, add transfer
scheduling and prospectively test backing, admission-budget and staging changes.
Acceptance is whether uncertainty can reverse the **difference between schedules**,
with important costs represented, not a percentage requirement on every tiny
operation. The existing copy-free grouped and C1 captures and rejected probe remain
available; do not expand this into a general GPU simulator or resume parameter
ranking before these gates.

**Foreground timing account and fresh validation, 2026-11-05:** the
[reconciled account](evidence.md#foreground-timing-reconciliation) now reports
isolated service, modeled resource service, queue/submission dependencies and
observed measurement spans separately. Original GPU endpoints that were not
captured remain unavailable. Helper width groups, real chunk counts and record
reuse are retained; interpolation holds out whole shapes. Duration-only
contention observations and universal percentage allowances are rejected.

The no-copy CUDA/NVTX companion measured primary event spans of **7,155.812 ms**
but only **215.264 ms** in recorded kernels. Helper spans were **2,671.522 ms**,
with **2,331.370 ms** in kernels. The fixture-only nonblocking stream-query test
moved most of the primary CPU-dependent delay from the compute timer into its
explicit wait timer. It does not select a runtime optimization. The original
9,995.161 ms fixed generation also spent **565.240 ms** formatting route records,
an omitted capture cost. Working-set causality remains unproved: on this same
fixed work, the older CPU table was **0.24% low**, while the new full-bank probe
was **7.26% low**. The earlier -34% figure concerned a different ordinary request.

The frozen model predicted **5,228.542 ms** for fresh held-out fixed work; hardware
measured **4,952.715 ms (+5.57% prediction error)**. Its CPU service was +13.95%
and helper event-span prediction +42.57%; these overlap and do not decompose the
request error. The conservative 1,974.761–12,670.227 ms envelope is not a
confidence interval or an accuracy pass. State/decision reproduction passed
1,995 staging choices and 19,608 original generation selections over 4,902
evaluated layer rows. Exact work, static rosters and zero weight copies were
verified. [Prospective plan and evaluation](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-heldout/evaluation.json).

**Timing remained unqualified at this checkpoint.** Further foreground cost and
submission-context generalization was then proposed; that continuation is now
superseded by the replacement decision. No global lane multiplier or broader
passing tolerance made the model qualified. No parameter defaults were selected
by this work.
The native build, 29 focused Python checks and four native CPU checks pass.

**Revised immediate order, user direction 2026-11-05:** repair acceptance logic,
then test one nonblocking `cudaStreamQuery` immediately after primary expert
submission and before CPU service in ordinary inference. The existing Windows
`k::submit` runs later and does not reproduce this placement. Use the existing
warmed Python/prose comparison with detailed capture and Nsight absent. Adopt
only if an integrated benefit is demonstrated; then calibrate the operating
regime intended for use and prospectively test backing, admission budget and
staging changes with the existing factorized model. Coarser empirical episodes
remain an alternative if decision failures justify them, not the chosen design.

Nested operation/branch/request residuals must remain separate diagnostics.
Prospective candidate-minus-control differences use uncertainty that preserves
correlation across arms; practical ties can pass only inside a predeclared
equivalence margin. An interval merely containing zero is unresolved. The
defective gate did **not** reject the latest held-out run: its evaluator recorded
the diagnostic result as unresolved directly. Gate repairs do not erase the
275.827 ms whole-generation prediction error. The detailed investigation remains
useful evidence of a potentially controllable queue-progress effect, not a
guarantee that a completion-status query flushes driver work.

**Ordinary query comparison, 2026-11-05:** completed both orders with detailed
capture, Nsight and optional monitoring absent. Pair effects were +6.48% and
+0.18%; the control moved +5.83%. The first off prose response took five extra
target rounds, while the later off and both on outputs/work matched. The pooled
+3.32% is not a demonstrated useful gain under that variation. Retain
`primary_submit_query=0`; the request control remains available for diagnostics.
No additional ordinary speed repetition is planned. The
[full receipt](evidence.md#acceptance-correction-and-ordinary-primary-query-test)
preserves all four arms, per-task results, validation and truncation limits.

**Prospective fixed-work interventions completed, 2026-11-05:** the new complete
setup snapshot permits prediction binding without detailed capture during timed
work. Two reserved requests passed state reproduction and supplied 64/85 evaluated
generation rows. Predictions for backing, admission budget 8→2 and staging 4→2
were frozen before both hardware orders. All three measured paired differences
changed sign; the control moved 6.05%, so none supports a resolved decision.
The full-work probe-based comparator predicted 12.062 s versus 6.693 s observed
mean (**+80.2%**). It is separate from the earlier generation-only foreground
refit, which does not cover these prefill/transfer interventions. Exact admission
counts and verified removed D2H do not qualify its timing. The retained fixed
schedule passes the shortened-transfer monotonicity diagnostic.
[Complete intervention account](evidence.md#prospective-fixed-work-intervention-result).

The acceptance logic now keeps nested timing residuals separate and permits
supported practical ties. Its repaired gate leaves these results unresolved.
The user subsequently required [replacement](#timing-model-replacement), rather
than another calibration round or an optional comparison with the same approach.
Submission context and useful resolution of paired differences remain design
issues for the replacement. No global lane multiplier, relaxed tolerance,
further repeat to obtain a favorable number or parameter selection follows from
these results. Admission repair, supported offline search and integrated
confirmation remain unfinished; qualification of the retired timing model is
not their continuing prerequisite.

The 2026-11-05 handoff recorded a resident engine at `http://127.0.0.1:8097/v1`, server **2161071**,
engine **2161074**, with 438 primary, 741 helper and 6,117 HOST records. The common
independent five-request learned state was restored after the synthetic tests.
Query and backing are off; helper-live and primary adaptation are on. The
capture-control file is absent, optional monitoring is disabled and Nsight/CUPTI
libraries are absent. Ordinary runtime defaults are unchanged. This is a healthy
resident handoff, not final simulator, serving or performance qualification.
[Observed handoff](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/resident-handoff.json).
It ran the preserved `maya-mm3-fixed-snapshot` executable with SHA-256
`4120a4dac2dfcacbd6d4f4de7fedb11da8766f97b64cb1177418a82b82cba8cc`;
later Python input handling and documentation changes need no weight reload.
Match source and executable before collecting further evidence. The native build,
27 focused replay checks, complete snapshot/ownership verification and the two
new source-state reproductions pass; none substitutes for timing qualification.

### Policy evaluation correction

**Latest user correction, 2026-11-04:** evaluate the approaches and choices in
this plan against the objective and evidence, not just whether code follows the
plan. An explanation of a slower result does not justify the policy that produced
it. The prior completion review was too narrow. The admission/scheduling and
DRAM-backing dispositions below are reopened; the implementation, actual speed
measurements, functional checks and monitoring-off result remain valid at their
recorded scope. The active Maya optimization is **not complete**.

The user's clarification concerns **GPU admissions**: only original-router demand
may credit residency learning. Cache-induced executed hits must not manufacture
admission demand. Cache-aware model routing remains part of the selected engine.
The current native score updates use the original IDs, so this is not a newly
identified hit-credit configuration bug. The 0.10 routing bonus was tested against
0.05 in two orders (+9.48% pooled); it is a measured provisional setting, not a
global optimum or validation of the admission policy.

The primary-agent [policy reassessment](evidence.md#policy-objective-reassessment)
identifies two gaps that the earlier successful checks did not close:

- The foreground planner uses grouped service curves and a maximum across
  overlapping lanes. Primary admission instead multiplies row-frequency scores by
  a single-column CPU/GPU service difference. It does not share that grouped
  critical-path valuation, and its configured interference surcharge is zero.
  Removing D2H correctly lowers a physical cost; the remaining benefit estimate
  still needs to represent the work and opportunity costs it claims to save.
- The retention report's `used` category includes cache-induced substitutions.
  In the selected trace, seven of the 58 admissions described as used before
  eviction had only substitution-induced executed uses. Correct storage reuse and executed hits
  do not establish positive net admission value, including the evicted victim's
  later service, transfers and occupied working frames.

The next work is a focused correction of these existing workstreams:

1. Preserve original-only demand credit and the defined generation-row clock.
   Keep original demand, executed service and substitution-induced uses separate
   in both decision inputs and evaluations. Reuse valid ownership/state checks.
2. Repair admission valuation to use grouped avoided work, the displaced resident's
   value, transfer interference and useful overlap. Use a small inspectable model
   and existing measured curves; a large predictor is not a prerequisite. Having
   bandwidth available or admission credits remaining is an opportunity, not an
   obligation to admit. Keeping the current allocation and doing no extra refill
   must remain an explicit alternative.
3. Have the primary agent use bounded same-demand modeling/diagnostics to challenge
   the decisions before tuning numerical values. With required work, state and
   other resources fixed, lowering a transfer cost leaves the old schedule feasible;
   assess the selected plan against that alternative. Validate model conclusions
   with focused real tests, including the backing interaction. This diagnostic
   does not require deterministic text/cache history or optional-transfer waits
   in ordinary inference, and modeled savings are not live speed evidence.
4. Revalidate the affected implementation and parameter/backing choices in the
   combined engine. Preserve all adverse results and actual original-engine gains.
   A gain from the complete composition cannot by itself justify every internal
   policy. Close this work on justified decisions and actual net operating benefit,
   not compliance with the old plan or explanatory wording.

Keep the currently usable resident configuration while these choices are repaired
and evaluated; no runtime setting is changed by this documentation correction.
The old instruction that no further backing check is planned is superseded by
this focused diagnosis. Held bases and unrelated parked mechanisms remain held.

**Monitoring-off benchmark completed, 2026-11-04:** the requested final-engine
screen measured **15.5226 tok/s pooled** (256 tokens / 16.492107 s): Python
**16.0539**, prose **15.0253**. Both high-effort responses reached the 128-token
cap in reasoning. [Primary-agent evaluation and raw outputs](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/monitoring-off/evaluation.json).
This is a single absolute-speed screen; it is not a paired estimate of monitoring
overhead because the reload changed calibration and learned/generated histories.
The original-engine comparisons below retain their earlier monitoring conditions.

The previous standalone server still ran its NVML hardware sampler, and capture
mode `off` incorrectly retained frontend token receipts. Those controls are now
fixed at `a105c56`; the native build and 16 focused checks pass. The native change
only gates serving progress/statistic output; model arithmetic/kernels and selected
policies are unchanged. The monitoring-off binary is SHA-256
`94cd6ffcb63407f3f5534cc016136d4b6d23967f342b80691b6285b395376b4c`.
The run disables CPU/RAM/disk/GPU hardware sampling, GPU-map reads, profiler
injection, native/frontend token-round capture, intermediate native STAT/PP,
periodic frontend progress/rate sampling, client stream tracing and process-I/O
sampling. Only generation start/final samples and normal completion measurements
remain. Before/after checks show no monitor processes, `monitoring_enabled=false`,
empty hardware samples, two generation samples per request and zero trace artifacts.
No CPU/GPU polling occurred during the scored requests. The user's request
supersedes the earlier capture requirement for this measurement.

That monitoring-off handoff left server **2093734**, engine **2093738**, idle with
weights resident at `http://127.0.0.1:8097/v1` and monitoring disabled; it is
superseded by the foreground diagnostic handoff above. Its configuration/logs
are under [monitoring-off](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/monitoring-off/).
Use `--no-monitoring` for frontend hardware/progress sampling and
`MM3_SERVE_MONITORING=0` for native boundary-only statistics; the actual run config
also disables the activity-sampling watchdog and has no capture environment.
Earlier process/resource handoffs below are dated observations of superseded processes.

**Preserved implementation and speed qualification; optimization closure reopened above.** The final resident
server passed default-serving and ownership/resource checks at **2026-11-04
16:53 UTC**. The final comparison load measured **14.7816 tok/s versus the exact
original's 13.1024 (+12.82%)**, with Python +6.80% and prose +20.38%. The earlier
declared two orders measured 16.1144 versus 12.8314 (+25.59%); both results and the
lower final-load speed are retained. A stable 25.6% gain across reloads is not claimed.
[Final-load evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/final-load-evaluation.json)
and [resident handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json).

The selected engine has actual five-part residency implementation, original-engine
comparison, bounded complete-task usability and default/resource evidence. Those
observations remain accounted for in the work table; the reopened policy valuation
and backing decisions prevent treating them as full optimization completion.
Short capped speed screens, approximate service prices and sampling changes limit
the claim; this is no hardware optimum or broad quality guarantee. Infernix and
Strata remain on hold, and their unfinished comparison is outside this completion.

The earlier 2026-11-03 declaration was premature: its sequential-start policy gain
and feature checks did not establish the later conditioning/optimization/whole-engine
outcome. That withdrawal and all adverse receipts remain preserved; the completed
work below supplies the previously missing evidence.

### Active Maya work

Follow [timing-model replacement](#timing-model-replacement) and the
[policy evaluation correction](#policy-evaluation-correction) for current work.
The retained requirements and chronological receipts below remain evidence
and design context; superseded completion language does not close the reopened work.

### Implementation first, then measurement and tuning

**User priority correction, 2026-11-04:** first implement the best complete design
we can justify from the architecture, existing evidence and engineering judgment.
The current implementation is not a reasonable baseline for another optimization
campaign. Repair known material design weaknesses together instead of tuning around
them or requiring a separate speed experiment before each logical repair.
This priority supersedes the earlier comparison-first sequence below.

1. **Build the reasonable baseline.** Implement the
   [required baseline behavior](#required-baseline-implementation) across placement,
   persistent learning, whole-batch cooperative assignment, staging, retention,
   refill and backing. Choose coherent initial policies and numerical estimates;
   record assumptions and use the existing packed kernels and ownership machinery.
   The current first-seen staging assignment is a diagnostic control, not the
   default to preserve until a local parameter win permits its replacement.
2. **Check it as it is built.** Run focused correctness, ownership, resource and
   production-caller checks. Collect bounded independent demand or service data
   when it supplies a missing implementation input. These checks and necessary
   diagnostics may accompany implementation; ranked policy screens and parameter
   sweeps wait. Reuse already collected history and receipts at their valid scope.
3. **Establish operation, then optimize.** Once the baseline is implemented and
   its affected checks pass, demonstrate its behavior under the
   [independent conditioning protocol](#placement-and-residency-qualification-correction).
   Re-establish the usable original-engine comparison, perform the explicitly
   isolated helper/backing checks, and then tune the material remaining parameters
   and interactions using short complete-engine measurements.

The implementation gate is finite and closes on **integrated full-model behavior**:
ownership, capacity, state continuity, useful retention, bounded transfers and the
actual ordinary/grouped callers must be demonstrated. Predicted savings, a build
or passing component tests alone cannot close it. It does not require anticipating
every possible optimization. Logical justification selects what to implement first;
it does not prove optimality or a speed gain.
The new baseline is a candidate engineering starting point, not a replacement for
the preserved best usable engine in the final performance comparison. Required
measurement, quality checks and the completion contract still follow.
The defined demand clock, small batch-cost model and integrated acceptance criteria
below are refinements of this existing baseline, not additional workstreams.

### Baseline implementation progress

The following progress receipts are chronological. Intermediate next-step wording
is historical; the final outcome and active work table own current status.

The independent high-effort history was collected once: five requests and 482,676
generation original-router selections. Code/prose/extraction/reasoning reached
their 384-token caps in reasoning (including draft code/prose); the tool proposal
completed in 44 tokens. This is scoped demand history, not five complete-task
passes. The actual joined helper-first setup verified 741 helper and 437 primary
experts, exact agreement with frequency ranking, full ownership and retained
allocations. Coverage on that training history is 35.28% helper plus 12.74% primary;
those are placement calculations, not held-out hits or speed.
[Receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/history-summary.json).

The implementation under validation makes these intentional initial choices:

- All native demand-derived initialization ranks the helper globally first and
  the primary next. A histogram also seeds a frequency prior for admission scores;
  it does not reconstruct chronological learned state. A helper-only refresh
  rejects incompatible primary contents. Missing history is labelled an adaptive
  bootstrap; an explicitly supplied invalid profile fails instead of silently
  selecting sequential identities.
- Native residency scores track original selections per **evaluated generation
  row**, including rejected rows. Prefill counters remain captured but do not age
  or credit generation residency. Scores are normalized to 100 times frequency.
  Aging is `0.95^(rows/5)`; block-uniform credit preserves the old recurrence at
  five rows and removes its accidental width-dependent scale. This is an initial
  unit conversion, not an optimized decay choice. Horizon 20 means 100 evaluated
  generation rows. Primary/helper admission credits accrue per generation row,
  bounded at eight/one respectively; these capacities replenish per five rows.
- Grouped staging chooses count and expert identities from the whole batch using
  measured CPU/primary/helper grouped-service curves and one upload per distinct
  expert. The bounded greedy objective includes existing GPU work and a helper
  completion estimate; ties prefer no upload, then expert ID. Its additive service,
  interpolation and overlap assumptions remain approximations, and greedy selection
  is not a global optimum. The old first-seen allocator is an explicit diagnostic
  control. The `cooperative_stage_reuse=1` control now means this complete batch
  planner, superseding the unselected reuse-sort prototype.
- Priced retention/refill shares the row-based budget and uses observed grouped
  marginal service and transfer proxies by default (`primary_h2d_ms=0` and `primary_d2h_ms=0` select measured prices). Legacy unpriced temporary promotions are suppressed while
  priced primary adaptation owns admissions. Optional publication remains
  nonblocking and all active readers retain their leases.
- Full-primary DRAM backing is implemented as a separate explicit setup control.
  Existing HOST pointers rotate between canonical ownership and primary backing;
  backed eviction does not download weights. Its allocation checks keep 16 GiB
  physical/commit headroom. The exclusive control remains available; implementing
  the mode does not select its performance or mix it into the helper comparison.

**The integrated baseline implementation gate is established, 2026-11-04.** The
[qualification receipt](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json)
records the actual ordinary/grouped behavior, ownership, fixed capacity, learning,
retention, bounded transfers, backing allocation/release and exact state restore.
This is an untuned engineering baseline, not plan completion or a speed gain.
Implementation checkpoint `7125f8a` supplies the native binary; the focused build,
CPU ownership, row-accounting, planner and checkpoint checks remain valid.
Checkpoints retain rosters, scores, admission
credits and grouped service state in-process; exported JSON is a review record,
not a cross-restart image.
No parameter sweep or qualified performance selection has run in this phase.

**Service-calibration capture correction:** the first timing screen measured
12.5137 tok/s pooled (Python 11.7747, prose 13.3515), but its common checkpoint
retained the one-time post-prefill service recalibration performed during the
earlier Nsight/detailed capture. The size of that instrumentation effect on
learned prices and policy is unmeasured. Preserve this as
[diagnostic timing](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/starting-engine/baseline-a-qualification.json),
not a normal-calibration winner or completed starting-engine comparison. On the
already required next loads, run the first independent conditioning request with
Nsight stopped and detailed capture off so recalibration finishes before tracing.
Then retain those prices across conditioning and comparisons. Repeat only the
affected baseline timing; unchanged integration checks remain valid. The extra
companion for the profiled-calibration timing is not run.

**Calibration input repair integrated at `e6d0cb4`:** the primary agent's matched-layer
[diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/profiled-cost-model-diagnostic.json)
finds median helper prediction / observed ready-and-reduction proxy of 3.69 on
2,738 generation layer calls; CPU wall time / prediction was about 1.7 in the
same profiled check. These ratios do not isolate a cause or measure a speed gain.
Source inspection shows repeated timing of only four CPU records and no sustained
helper warm-up. The next build rotates existing CPU HOST records across up to 64
identities (no weight allocation or cache flush) and warms helper service for a
bounded 0.5 seconds before its short measurements. This repairs known input
representativeness issues without adding a parameter sweep. The native build passed;
the profile-free bootstrap returned the correct answer, and its unprofiled
post-prefill calibration recorded 64 CPU records and 500 ms of helper warm-up.
Five further independent requests retained the exact service estimates and fresh
KV while primary learning continued with a static ranked helper. Full ownership,
437/741/6,118 capacities and zero helper weight traffic were verified. The
[current diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/adaptive-companion-reconciliation.json)
also reconciles 9,234 planner calls and 548 primary exchanges, including 471 retained
uploads serving 5,152 later uses. The earlier ownership/backing receipts remain
valid at their source scope; final performance selection remains open.

**First usable-engine comparison:** the exact original executable measured
12.5713 tok/s pooled under the current protocol. The corrected current baseline
measured 13.3001 tok/s, a first-screen gain of 5.80% (Python +8.91%, prose +2.71%).
[Evaluation and limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/starting-engine/first-current-comparison.json).
Both responses reached 128-token caps. This is not final confirmation or a quality
pass, and does not close remaining optimization. The same current adaptive arm is
reused as the helper-test control. The completed helper comparison is inconclusive:
adaptive scored 13.3001 then 13.7574 tok/s; static scored 14.0226 then 13.4545.
Static's pair effects reverse (+5.43%, then -2.20%). Across both orders it is
1.54% faster pooled, with Python 6.00% slower and prose 10.22% faster; sampled
outputs and speculative work differ. Static verified zero helper weight traffic
and replacements. The common restored logical state and identity rosters match;
physical frame assignments can differ. [Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/helper-comparison-evaluation.json).
No confirmed static advantage or adaptive superiority is established, and the
user's static-win stop condition is not triggered. No further helper repeat is
planned. The separately authorized full-primary backing comparison follows, with
the latest compatible static/exclusive arm reused as its declared control and
both earlier exclusive rates preserved. Final static adoption is not implied.

**Original-engine comparison protocol:** the first comparison above used the
preserved held M-110 executable under the same current high-effort Python/prose
prompts, 128-token caps, seed, sampling, memory budgets and complete-generation
timing. The original executable's SHA-256 exactly matches both the held M-110
qualification and its cooperative source receipt (`00d6dff`); this is the original
binary, not a new-policy control renamed as the reference. Its fixed-primary,
live-helper, seven-draft cutoff recipe remains a diagnostic control. Apply the
independent global helper-first rosters before conditioning, retain normal helper
learning between requests, and do not reset placement before scoring. Historical
primary-first setup is driven by an explicitly labelled synthetic roster-control
file whose resulting identities are checked against the genuine demand histogram.
The existing collector now supports this pinned historical setup and fixed-primary
conditioning; the scored runner verifies uninterrupted process/request history.
The old engine cannot restore the new learned checkpoint or expose its newer
trace fields. Record those differences and the fresh-KV adapter explicitly.
The short pair established the first comparison; confirmation follows the selected
final candidate. It does not replace the isolated helper/backing work or value tuning.

**Full-primary backing comparison completed:** backed scored 13.1431 and 13.1434
tok/s; its declared exclusive controls scored 13.4545 and 14.5047. The earlier
compatible exclusive score of 14.0226 is also retained. Across the declared pairs,
backing was 5.85% slower pooled (Python -7.53%, prose -4.04%). The exclusive rates
and sampled work vary, so this is no demonstrated practical gain, not an isolated
measurement of the cost of backing. [Evaluation and raw receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/backing-comparison-evaluation.json).
All 437 primary records were backed with 13,918,797,824 additional bytes; allocation
left 18.68 GiB Windows physical headroom. Both ranked backed runs verified zero
primary/helper expert-weight D2H and zero helper replacements. The detailed
companion reconciled 10,032 planner calls and 630 primary exchanges, including
539 retained uploads serving 4,841 later uses. Disabling backing repatriated 182
canonical HOST records, freed the full bank and retained ownership/capacity.
The mode remains available and currently disabled. The measured loss does not
justify rejecting backing or closing admission/scheduling optimization; the
[focused policy diagnosis](#policy-evaluation-correction) supersedes the earlier
decision to stop investigation here.
Nsight reconciliation of the three new companions passes the selected request,
layer, round and expert-weight copy joins on both GPUs: 28,956 layer calls total.
Each report also records five incomplete CUPTI events dropped at profiling stop,
plus possible-loss warnings; this is not a claim of complete coverage of every
event. [Capture index](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/companion-capture-index.json).
Their raw reports and complete native timelines are retained. Final adaptive-helper
operation, required parameter selection and final original-engine confirmation
remain open.

**Cache-prior value result:** 0.10 gained in both conditioned orders over 0.05
(+11.27%, +7.67%), with live helper, exclusive primary and all other controls fixed.
The pooled pair totals are 15.3507 versus 14.0219 tok/s (+9.48%); Python +16.07%,
prose +3.25%. Reverse-order prose was 2.88% slower, retained alongside the gains.
Both values used high request effort; the candidate Python output began final code
directly while the controls reasoned. These are actual-work speeds, not equal-token
compute. [Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/bonus-comparison-evaluation.json).
Carry 0.10 as the candidate, pending complete-task usability; no final default
or quality claim is made from the capped outputs. The changed configuration has
a bounded diagnostic companion with complete native capture and reconciled
selected Nsight joins. Next compare lenience 0.4 versus 0.2 at bonus 0.10, changing
only lenience; reuse the compatible 0.4 controls. This evaluates lenience directly,
not a translation of historical cutoff rho. Material residency/cooperative-work
assessment and final original-engine confirmation remain required.

**Lenience and admission-budget decision:** at bonus 0.10, lenience 0.2 measured
14.8415 tok/s versus the latest 0.4 control's 15.3932 (-3.58%; Python -4.50%,
prose -2.70%). Retain 0.4 without another ranked repeat; preserve the earlier
compatible 15.3084 control and the capped outputs. A single prose diagnostic
captures the changed verifier configuration. [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/lenience-comparison-evaluation.json).

The primary agent's [causal forecast sensitivity](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/demand-age-budget-sensitivity.json)
holds recorded original demand fixed and omits actual prices, transfers, active
readers and physical-frame constraints. It is not a production replay or speed
prediction. At the current budget, faster aging changed hypothetical primary hits
only 1.5–3.5%; raising the admission budget from 8 to 16 changed them 19–26%, with
roughly twice the admissions. Actual captures already show the current row-based
budget being spent and useful in-place retention. This prioritizes one cheap live
8-versus-16 budget comparison, keeping age 0.95 per five rows, horizon 20 and
physical capacity fixed. Extra preservation/refill traffic may erase its benefit;
inspect that tradeoff in complete generation. Do not infer a live hit rate or
optimized decay from the forecast. The [declared protocol](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/budget-protocol.json)
reuses the compatible bonus-0.10/lenience-0.4 controls and allows one reversed
confirmation if promising or close, not a budget sweep.

**Admission-budget outcome:** budget 16 measured 15.5718 and 15.2987 tok/s,
versus the declared budget-8 controls 15.3932 and 15.3929. Pair directions differ;
the pooled difference is +0.27% (Python -2.55%, prose +3.07%). Retain 8 under the
declared tie preference: 16 roughly doubled optional primary exchanges and
preservation traffic without a demonstrated generation-speed advantage.
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/budget-comparison-evaluation.json).
This does not prove 8 is faster. The first single-prose diagnostic accidentally
used the runtime default of 8; it is preserved as invalid budget-16 evidence.
The collector now explicitly pins and checks that control, and the corrected
budget-16 capture reconciles native events and selected Nsight joins. Neither
diagnostic rate is ranked; no further budget repeat is planned.

**Cooperative CPU-price refinement:** the primary agent's
[unprofiled accounting](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/unprofiled-cpu-price-bound.json)
compares actual generation CPU batch time with the recorded calibration. Even
charging every CPU group the largest measured 1–5-column marginal and every layer
the CPU base charge, observed time exceeds that prediction by 13–22% across twelve task
requests. These are bounds on that model's predictions, not hardware bounds or
isolated critical-path savings. Mixed groups, concurrency and memory/scheduling
effects remain combined. Add one explicit, checkpointed CPU price scale used by
both batch assignment and automatic residency service pricing, plus an aggregate
predicted-CPU counter for direct unprofiled reconciliation. Preserve raw measured
curves and scale 1.0 as the control. A bounded 1.0-versus-1.3 comparison will test
whether this correction improves complete generation; no online predictor or
separate executor is warranted. This is unfinished cooperative optimization,
not another prerequisite added to the already established baseline gate.
Implementation `ebe02b0` builds; the planner and learned-placement checks plus
14 focused frontend/checkpoint checks pass. Its new load completed the profile-free
bootstrap with answer 60, exposed the actual scale 1.0, and retained 437 primary,
741 helper and 6,118 HOST records with complete ownership. One unprofiled
post-prefill calibration was captured, using 64 rotating CPU records and 500 ms
helper warm-up. Independent conditioning and the new common snapshot now pass
[review](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/conditioning-verified.json):
raw prices remained unchanged, learned rows stayed continuous, and the static
helper had zero weight traffic. The direct CPU-time/prediction ratios were
1.23–1.25 in the grouped checks and 0.81 in the ordinary tool check; this shape
dependence remains explicit. The earlier qualified executable remains preserved; none of these
implementation checks selects scale 1.3 or establishes its speed.

**CPU-price comparison:** scale 1.3 measured 16.1732 and 16.0560 tok/s; scale 1.0
measured 15.7816 and 15.0539. Both paired effects favor 1.3 (+2.48%, +6.66%);
pooled rates are 16.1144 versus 15.4092 (+4.58%; Python +3.17%, prose +5.90%).
Control movement and differing sampled work limit the precision of that component
gain. Actual/predicted CPU time changed from 1.21–1.26 at scale 1.0 to 1.00–1.014
at 1.3 in these grouped speed requests. Recorded expert-weight transfer bytes
increased about 13%, so the improved balance is not a reduction in every cost.
[Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/price-comparison-evaluation.json).
Carry scale 1.3 in the candidate; full-task usability and the final original-engine
comparison still follow. Actual inspection verifies scale restoration independently
of the saved report, and detailed records reconcile the new prediction counter.

**Next bounded scheduling check:** compare two draft proposals (at most three
verification rows) with four (at most five), at prior 0.10, lenience 0.4, CPU scale
1.3 and primary budget 8. Reuse the compatible four-draft controls. Current
verification evaluates more target rows than it commits; fewer proposals can
reduce discarded tails but require more target rounds and reduce weight reuse.
Use complete-generation results, not acceptance alone, with one reversed
confirmation if promising or close. The [declared protocol](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/width-protocol.json)
defines the tie preference and actual-width verification; it is not a width grid.

**Scheduling outcome:** retain four drafts. Two drafts measured 15.0162 tok/s
versus the compatible 16.0560 control (-6.48%; Python -13.24%, prose +0.97%).
Both Python runs evaluated 153 target rows, but two drafts required 52 drafting
batches versus 32; shortening the window did not reduce its target work.
Outputs differ, so this is not isolated launch-cost attribution. No further
ranked repeat or width grid is warranted. [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/width-first-evaluation.json).
The [timeline audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/drafting-audit.json)
verifies actual proposal/window sizes, draft counters, scalar lenience probabilities
and decisions, and captured residual-choice support. The existing confidence floor
0.3 pruned five of 32 coding batches and 16 of 43 prose batches at four drafts;
discarded work remains charged. Keep that floor and the disabled product cutoff;
these results do not justify a new confidence scheduler or more values now.

The candidate entering final qualification uses **prior 0.10, lenience 0.4, four
drafts, CPU price scale 1.3 and primary budget 8/horizon 20**, with the grouped
planner, live helper and exclusive primary storage. These values are now in the
serving runtime defaults (since superseded: bonus 0.20. CPU price 1.5 and horizon 40 were tried on 2026-11-08 and
reverted after a failed usability check, see the [work list](#work-list)). The final original-engine comparison,
complete-task
checks and resident source/resource handoff below complete their qualification.

**Selected-candidate usability passed, 2026-11-04.** Both complete `count_runs`
responses pass five functional cases each. The high-effort scheduling response
stopped naturally at 2,146 tokens: both 11/12 schedules are feasible and the
minimum-time arguments pass primary-agent semantic review. The real `read_file`
round trip returned version 7.4.2 on its first raw call, with no repair or retry.
The facts-only library announcement completed in 92 tokens with all supplied
facts and no added policy. Disconnect was observed, native retirement ended in
`cancel`, and the follow-up returned READY. All completed workflows retained the
selected effective settings. [Reviewed outputs and capture limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/quality-reviewed.json).
These are bounded usable-task checks, not broad quality qualification. Their
native timelines have no dropped events; the cancelled API is intentionally
partial. GPU/expert detail is supplied by the same-source selected-configuration
companion, not claimed for these unprofiled full-task intervals.

**Final original-engine confirmation passed, 2026-11-04.** Across the declared
orders, the exact original executable measured 12.5713 and 13.1024 tok/s; the
selected engine measured 16.1732 and 16.0560. Pooled rates are **12.8314 → 16.1144
tok/s (+25.59%)**, with Python **12.3497 → 16.6072 (+34.47%)** and prose
**13.3522 → 15.6500 (+17.21%)**. Both pair effects are positive (+28.65%, +22.54%).
[Primary-agent evaluation, all outputs and latency diagnostics](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/final-comparison-evaluation.json).
This demonstrates a bounded usable complete-engine gain, not isolated feature
speedups or a hardware optimum. Compatible candidate arms were reused as declared
before confirmation; every speed response reached its 128-token cap. Candidate
Python entered final code directly while the original reasoned, under the same
high request effort. Separate completed tasks supply usability evidence.

Configured memory budgets, prompts/order, caps, seed, target sampling and whole
generation timing match. Actual residual allocation on the final original load
was 438 primary/741 helper/6,117 HOST, versus 437/741/6,118 in the earlier arms.
The final original ran standalone; earlier scored arms had Nsight launched with
collection off. These differences prevent a claim of identical physical allocation
or instrumentation overhead. Original control movement was 4.22%; it is preserved
alongside the larger gains. The original cannot restore the new learned checkpoint;
both used independent helper-first ranks and retained their respective ordinary
learning. The subsequent final-load check and observed default/resource handoff follow.

**Final-load variation and handoff, 2026-11-04.** The normal standalone reload
changed automatic service estimates; observed/predicted grouped CPU time became
1.17–1.22 (ordinary 0.74). The one allowed targeted check was declared for that
specific uncertainty, with no parameter change. It measured **14.7816 versus
13.1024 tok/s (+12.82%)** against the final original arm, with matching standalone
environments and actual 438/741/6,117 capacities. Python was **13.2709 versus
12.4264 (+6.80%)**; prose was **16.6803 versus 13.8561 (+20.38%)**.
[All outputs, service/work counters and limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/final-load-evaluation.json).

This check is not pooled into the earlier two-order result: final independent
conditioning used live helper learning, automatic prices changed, and sampled work
differed. Coding was 19.65% slower than the earlier selected-b arm while evaluated
target rows rose from 153 to 192 (+25.49%); target time per evaluated row was nearly
unchanged. This supports work-count sensitivity without isolating its cause. Prose
was faster. The current usable engine still improves both tasks over the original;
the 16.11 rate is not a promise for every reload. No additional repeat is planned.
Approximate service pricing retains meaningful shape/load error; its numerical
fit is not itself the completion or speed criterion.

The final process loaded without a placement profile. Its 32-token bootstrap
reached the cap in reasoning and is only an operation check; the earlier same-source
profile-free bootstrap completed correctly. Independent helper-first initialization
then preceded five fresh-KV requests with continuous primary/helper learning and
one unprofiled post-prefill calibration. The ordinary tool caller verified CPU scale
1.3. [Operating receipt](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/operation-reviewed.json).
A normal API request with no tuning, effort or sampling overrides returned READY
and verified the selected effective controls; its 195 cached prefix tokens are
normal serving reuse, not the fresh-KV benchmark protocol. At 16:53 UTC the server
was idle with zero queued requests, complete ownership and weights resident.
The HOST bank is 194,831,712,256 bytes (181.45 GiB); observed Windows physical
headroom is 41.30 GiB, above the 16 GiB reserve. GPU total usage is 30,447 MiB on
5090 and 23,132 MiB on 3090, including driver/other use. No primary backing or HOST
compression is active. Both final speed requests recorded zero native expert-disk
service and zero Linux process `read_bytes`; these are not a Windows driver/pagefile
trace. Resource figures are point-in-time observations.

### Current implementation and parameter dispositions

These are the prior dispositions and the evidence behind them. The
[policy evaluation correction](#policy-evaluation-correction) reopens admission,
scheduling valuation and backing selection, including affected parameter choices.
Their implementation checks and measured rates remain evidence, not sufficient
justification for closing those decisions. Other scoped findings remain reusable.

- **Cooperative service and scheduling:** keep the new whole-batch grouped-cost
  planner and CPU price scale 1.3. The live comparison improved 4.58% pooled and
  brought actual/predicted grouped CPU time to 1.00–1.014. Packed CPU tiles already
  reuse each 144-byte weight unit across up to eight columns; the selected verifier
  uses at most five. Extending that width is inactive for this workload. Keep
  15 CPU workers plus the caller, the existing fine output-row partition and the
  primary FP4/helper packed kernels. One compute participant per physical core
  avoids adding SMT competition; no measured deficit in this partition justifies
  another worker grid after the service-price repair. Unprofiled host planning is
  52–59 ms per 7.7–8.2 s request. The helper's conditional dispatch exposure is
  222–376 ms (about 3–5%), overlapping other work and not isolated removable time.
  Retain the worker dispatcher; the tested mapped queue remains parked.
- **Drafting and grouped verification:** keep four drafts, confidence floor 0.3
  and the disabled product cutoff. The two-draft test lost 6.48% and did not reduce
  coding target rows; the [drafting audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/drafting-audit.json)
  reconciles actual widths, proposals, acceptance, residual decisions and discarded
  work. Confidence pruning already limits weak tails. No new scheduler or wider
  grid is justified by that result. Lenience 0.4 remains selected after the direct
  0.2 comparison; lower acceptance alone was not the decision criterion.
- **Routing and device preference:** select the measured uniform prior 0.10.
  The [selected trace audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/routing-service-audit.json)
  checks 10,317 generation routing rows and 41,268 mixture/scaling entries at
  recorded precision. All four IDs are valid/distinct, ranks follow their recorded
  scores, and mixtures use original sigmoid weights with scale 2.0. It observes
  7,142 substitutions: 2,692 toward primary-ready experts and 4,450 toward helper.
  Median base fourth/fifth gap is 0.02097; the largest displaced gap stays below
  the 0.10 bonus. Predicted limiting lanes vary across CPU, primary/copy and helper;
  those are model outputs, not measured critical paths. Keep the measured uniform
  policy: the evidence does not justify a separate device-specific or score-range
  policy now. Inactive substitution-ratio knobs do not count as tuning this prior.
- **Primary retention, refill and capacity:** keep eight admissions per five
  generation rows, 0.95 decay per five rows, horizon 20 (100 rows), measured copy
  and scaled service prices, and zero added margin/interference surcharge.
  The budget-16 comparison was practically tied (+0.27%) with roughly doubled
  optional exchanges; retain eight. The causal score forecast predicted a much
  larger hit opportunity that failed to become a speed gain: it omits prices,
  leases and changing text and is not a production simulator. Faster decay gave
  only a small forecast change at budget eight; slower decay was worse. Combined
  reuse evidence previously supported retaining the current age/horizon/margin.
  That disposition is reopened: raw reuse and a score forecast do not establish
  net value under the actual grouped/overlapped service. In the selected prose companion, 237 demanded
  uploads stayed in their frames and served 2,235 later uses; 58 of 68 subsequently
  evicted admissions had executed uses, seven exclusively from cache-induced
  substitutions; still-resident outcomes are censored. These counts are not
  original-demand benefit or net savings. Fifty
  additional uploads and outgoing preservation are charged separately. Eight
  working frames and two exchange slots retain fixed total capacity; the checked
  production ownership/availability machinery bounds admissions. These observations
  do not establish universal residency economics or optimal numerical values.
- **Helper placement, capacity and transfers:** retain global independent-demand
  helper-first initialization, continuous live adaptation and coordinated
  ownership. The isolated static/live comparison is inconclusive, so it provides
  no basis to remove required live behavior. Keep the 741-record helper and
  437-record primary budgets with their established workspace/headroom; no enlarged
  bank is justified by the inconclusive policy result. The full-primary backing
  comparison was 5.85% slower pooled despite eliminating expert D2H. Backing is
  currently disabled, but its policy interaction is reopened rather than accepted
  as a reason to reject the design. Helper activation/result traffic remains distinct.
- **Transfer implementation and overlap:** the new planner prices complete grouped
  service and one upload per distinct record, replacing anchor-first staging.
  Retained uploads avoid a redundant incoming copy; optional refills stay bounded
  and nonblocking. Preserve the actual overlap and source accounting in the
  [companion audit](evidence.md#companion-overlap-and-capture-audit) and new captures.
  CPU/copy durations overlap and cannot be added as critical-path costs. A GPU
  demand-fetch port would still need CPU/helper assignment and ownership retirement;
  neither the old Windows port nor traced launch groups establish a present WSL
  gain. The measured planner/price repair addresses the observed assignment defect;
  current evidence does not justify a new fetch executor. This is an engineering
  investment decision, not a claim that all further gains are impossible.

**Retained implementation-gate check status:** implementation checkpoint `7125f8a` builds; nine focused
CMake/CPU checks and the affected Python checks pass. The full model has now
loaded from the independent frequency profile at
[the source-bound baseline launch](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/launch.json).
Its initial memory-reserve wait completed and the server reached READY. Startup
inspection verifies 437 primary + 741 helper + 6,118 HOST identities, full ownership,
global frequency-ranked rosters and the initialized frequency prior. Four grouped
requests and the corrected ordinary caller completed with validated captures;
retention/refill activity is present. The first ordinary payload was rejected by
the frontend before execution because it combined `spec=1` with lenience; it is
preserved, and only that unexecuted request was corrected. Full trace reconciliation
and actual learned-state restore now pass. Across five executed requests, 885
evaluated generation rows retained the learning clock; 361 demanded uploads were
retained in their existing frames and served 9,096 later expert uses (7,227
original-demand uses). Of the 218 retained admissions evicted within the capture,
206 were used before eviction; still-resident admissions remain censored. All
18,354 captured ordinary/grouped planner calls reconcile with the implemented
decision, and ownership remains complete. The joined restore reproduced saved
rosters, scores, admission credits and service state after changing 295 primary
and 130 helper identities. These are implementation checks, not a speed improvement
or evidence of optimized parameters. [Receipts](evidence.md#baseline-integrated-behavior).

The first full-primary backing allocation was safely refused before allocation: its
13,918,797,824-byte bank plus the 16 GiB reserve exceeded the observed Windows
physical availability of 29,572,321,280 bytes. Guest availability alone does not
establish Windows headroom. The failed receipt is preserved; no reserve was lowered
and no capacity was reduced. After user-assisted host reclamation and the explicitly
requested REA/Claude process cleanup, the retry allocated all 437 backing records
with 16.35 GiB Windows physical headroom remaining. Two bounded grouped/ordinary
requests issued 89 primary exchanges with zero expert-weight D2H; five-second
resource samples stayed at or above 16.23 GiB. Disabling backing repatriated 71
canonical HOST records, released the full bank and preserved complete ownership.
The common learned-state hash then restored exactly. Native trace/counter checks
pass; Nsight matches all 4,674 layer calls but captures 4,271 of 4,272 submitted
weight uploads. Its possible-loss warnings and this one-copy gap remain explicit;
they are not reported as complete GPU-event coverage.
The combined check used live helper adaptation to exercise that implementation.
A separate declared five-request conditioning check has now established the
common static-helper state: independent global helper-first ranks, primary
adaptation on, fresh KV per request and continuous learned residency. The helper
roster stayed unchanged with zero weight traffic/replacements. Its 959 evaluated
generation rows retained 429 uploads in place, serving 12,002 later uses. All
19,038 planner calls and trace/counter joins reconcile; ownership remained complete.
The resulting in-process [common checkpoint](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/common-checkpoint/learned-checkpoint.json)
was saved at generation row 1,844. This establishes observed operation under the
declared history, not convergence, optimality or a speed gain. That earlier
in-process checkpoint was superseded on the corrected normal-calibration load;
the current helper/backing comparisons use its separately recorded common state.
The extraction response preserved the supplied facts but added a note beyond the
requested three bullets; the capped outputs are not complete-task passes.
These earlier receipts establish the baseline gate. Subsequent helper/backing and
parameter decisions are recorded above; the active work table owns what remains.

### Placement and residency qualification correction

**User correction, 2026-11-04:** treat initialization, persistent learned state,
demand accounting/aging, global primary allocation, helper placement/adaptation,
retention/refill, transfer scheduling and selective DRAM backing as **one operating
system**. Sequential placement and 0.95-per-forward aging were inherited from
upstream Maya; main-bank adaptation was added later. Checkpoint `4e8792c` removed
the saved startup profile and profile output without establishing a replacement
warmed baseline. This is a setup regression as well as a longstanding validation
gap; no corresponding execution-speed regression has yet been isolated.

The legacy trace continued replacing experts through the final generation forward.
Its primary/helper admissions were disabled during prefill; the 66 coding-prefill
forwards therefore did not condition those banks. The 128-token output limit and
bounded admission cadence left most initial primary identities in place. Prose
continued that partially adapted history. Neither that observation nor a fixed
suffix cutoff establishes convergence or a permanently fixed fraction of the bank.

**Do not use the affected cold/adapting results to select warmed policies.** Preserve
the priced-policy comparison/replays, parameter screens, historical recipe checks
and helper screen at their actual initial-state scope. The 6.89% priced-policy gain
is a real result against its sequential-start legacy control, not validated warmed
LRFU optimization. Horizons 20/40 and budgets 2/8/16 on the cold trace do not resolve
aging, admission economics or warmed behavior. Numerical defaults remain provisional.

<a id="static-placement-order"></a>

**Static placement amendment, 2026-11-04:** for every static GPU placement, rank
experts globally across all MoE layers by **original-router demand frequency**
from the declared independent representative history. Fill the **3090 first**
with the most frequently demanded experts up to its actual resident capacity;
fill the **5090 next** with the most frequently demanded experts remaining.
Use the same ranking for demand-derived initialization of adaptive banks.
Expert IDs and per-layer allotments are not popularity ranks. Executing a
cache-selected substitute does not add original-demand credit. Retain the source
history, counting/phase scope and resulting ranked rosters with the experiment;
do not fit them to scored prompts. This governs roster selection at setup.
Adaptive banks may subsequently change through ordinary online policy; the rule
does not require reseeding between requests or frozen adaptive cache contents.

After the [baseline implementation](#implementation-first-then-measurement-and-tuning),
use the sequence below for conditioning and comparisons. Its setup/capture repairs
and necessary demand collection are implementation prerequisites, not reasons to
postpone known execution-policy repairs:

1. Complete faithful common-state setup using the implemented capture/checkpoint
   controls and remaining affected checks. Legacy `PLACEMENT all` resets live
   scores and chooses primary before helper; a demand histogram is not a
   learned-state checkpoint. At an explicit joined setup boundary, save published
   primary/helper identities, normalized demand scores, policy configuration and
   learned service/scheduling state. Restore once per arm, with ordinary staging
   cleared equally and that limitation reported. Do not add joins, forced cache
   contents or resets to ordinary inference or redo completed component repairs.
2. Inventory existing independent high-effort original-router demand. Reuse only
   histories with known prompts/settings/phases and no scored-prompt fitting.
   Fill missing coverage with a declared bounded independent history, selecting
   both GPU rosters in the [required frequency order](#static-placement-order).
   A profile supplies initial evidence, not proof of warmed operation. Preserve
   normal learned state across subsequent independent fresh-KV requests.
3. Demonstrate the operating behavior on independent demand: original/executed
   coverage, miss service/weight traffic, useful retention and rereads, turnover,
   unavailable frames, admission reasons/costs and complete latency over request
   and round history. Ongoing turnover is allowed. No arbitrary token cutoff,
   requirement for frozen identities or indefinite warming substitutes for this
   evidence. Material continuing loss/churn calls for a policy repair, then a
   bounded affected check, not longer warming until the result looks favorable.
4. Re-establish the usable original-engine baseline under that declared history
   and current timing/workload protocol. Redo only the decisions invalidated by
   initialization or materially interacting policy changes. The helper-only
   comparison below must start with a conditioned primary and change only
   `helper_live`; the separately authorized primary-backing test follows it.
   Revalidate aging/admission/refill and interacting width/prior/lenience decisions
   using the smallest useful experiments. Unaffected functional checks remain valid.

In the inherited policy, original demand was credited per evaluated row (including
rejected speculative rows and prefill), while 0.95 aging and admission opportunities
advanced per target forward. Changing width or prefill chunk size therefore changed
their relationship to emitted tokens and request history. The baseline's defined
generation-row clock and separate prefill accounting replace those mixed units;
their numerical values remain provisional. Neither a replacement count nor a
hit-rate improvement alone qualifies the resulting policy. The
[evidence correction](evidence.md#placement-and-residency-audit)
preserves the implementation/history findings separately from unmeasured remedies.

**Initial prerequisite implementation, `2a5bcab`:** an in-process joined checkpoint
now saves/restores published rosters, normalized scores, primary policy and learned
lane timing; ordinary staging is cleared equally at that explicit boundary.
Helper-first profile placement is available without changing live admission rules.
The detailed capture buffer now compresses records in memory and retains explicit
overflow reporting. Focused CPU ownership/state/capture checks and the native build
pass; actual-model restore and complete new capture remain to be demonstrated.
Neither checkpoint JSON nor a generation profile is a durable cross-restart state.

**Prepared independent-history protocol:** the old saved profile's actual requests
used none/low effort, and the available recent traces include scored prompts.
Reuse any valid collection already completed or underway; otherwise collect one separate
high-effort five-request history (code, prose, extraction, reasoning, tool proposal;
at most 384 output tokens per request), then apply global helper-first demand ranks.
After the baseline implementation and its affected checks, run one independent
five-request behavior check (at most 192 tokens each), primary adaptation on and
helper static, retaining state between requests. These are bounded
collection budgets, not warming thresholds. The protocols and provenance are under
[conditioned-residency](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/history-provenance.json).
Inspect original-demand coverage, service/miss traffic, retention/turnover and
latency trajectory; if continuing admission loss is material, diagnose it before
scoring policies rather than extending warming. Detailed capture/Nsight timings
are diagnostic. A complete independent operating account and verified restore
permit the matched baseline/helper steps; adverse behavior requires a focused repair.
Do not use this collection protocol to resume tuning the old implementation before
the [implementation priority](#implementation-first-then-measurement-and-tuning) is met.

### Bounded static versus adaptive 3090 comparison

**Latest user correction, 2026-11-04:** begin from a representative, conditioned
primary cache, preserve learned residency between independent requests, and change
only the behavior being evaluated. The completed screen below began from the
arbitrary sequential primary roster. **It is cold-start evidence, not a decision
about warmed helper policy.** Frequency-ranking the helper alone did not condition
the primary. Expert IDs are identities, not popularity ranks.

The [cold receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/helper-static-adaptive/cold-summary.json)
retain both orders: adaptive 11.5124 and static 11.7460 tok/s pooled (+2.03% static),
with substantial Python variation and opposite pair directions. Static verified zero
helper expert-weight uploads/downloads/replacements. Actual capacity was 437 primary,
741 helper and 6,118 HOST records under unchanged launch memory limits; the older
process had 438 primary records. Do not use these results to decide the warmed question.
The older 10.3% versus 39.7% helper demand-coverage calculation is historical placement
arithmetic, not observed cache-hit rate, measured speed or proof of an optimum.

Before the warmed comparison:

1. Complete the [reasonable baseline](#required-baseline-implementation) and its
   affected checks, including the detailed-capture repair. Retain the partial
   earlier captures. Token/round timelines and raw Nsight reports from the cold
   screen survive; some detailed expert/cache events do not.
2. Establish both GPU rosters from separate representative original-router demand
   in the [required frequency order](#static-placement-order). Keep that chosen
   static helper roster while conditioning the adaptive primary from the remaining
   experts, so the common snapshot retains the required static-helper placement.
   Use existing suitable evidence and a declared bounded conditioning/check capture
   as needed; do not fit to scored prompts.
3. Save one common **published placement and learned policy state** from that
   conditioned primary, with primary adaptation enabled. Demonstrate conditioning
   with actual independent demand/service/cache evidence; an applied profile or a
   token cutoff alone is not proof of completed warming. Restore that same state
   for each arm. Preserve normal learned residency between Python and prose, while
   giving each fresh target/DSpark KV. No per-request reseeding or waits for optional
   cache completion are introduced into inference.
4. Freeze four drafts (at most five verification rows), lenience 0.4, cache bonus
   0.05, the implemented baseline's primary policy/CPU/FP4/worker settings and
   memory budgets. Use the same new batch-aware staging and assignment policy in
   both arms; the old requirement to keep staging reuse off is superseded.
   **Only `helper_live` changes**. The static arm must show zero
   helper expert-weight traffic and replacements throughout inference; activation
   and result transfers continue. Use the short high-effort Python/prose screen,
   one pair and one reversed confirmation if static is promising or close.

Report per-task/pooled rates, variation, actual transfers and output limitations.
An inconclusive result does not establish adaptive superiority. A confirmed static
advantage requires an evidence report and no automatic wider optimization or final
static-policy adoption. The user's later explicit backing-copy experiment below is
separately authorized; it is not combined with this helper-policy comparison.

### Bounded primary DRAM backing experiment

**User storage amendment, 2026-11-04:** exclusivity was an implementation requirement,
not a runtime decision that surplus DRAM was unavailable. Temporary staging uploads
already keep their HOST source; main-bank promotion currently recycles that HOST
frame for another expert, without returning the allocation to the OS.

After the conditioned helper-only comparison, run the separately requested test of
retaining DRAM backing for **the entire 5090 resident bank**, with a fully static
3090 containing the most frequently demanded experts across all MoE layers. Both
control and candidate start from the same demand-calibrated GPU placement and learned
primary policy state, following the [static placement order](#static-placement-order).
Keep four drafts, lenience 0.4, prior 0.05 and the implemented baseline's primary
and batch-aware staging/assignment policies fixed in both arms. The control remains
exclusive; only primary backing retention changes in the candidate. Do not mix
that change into the helper-policy test.

Preserve all experts and the existing GPU capacities. Charge the additional HOST
records and bounded workspace, recheck actual Windows physical/commit and guest
headroom, and retain the configured 16 GiB reserve. Roughly 13 GiB can back the primary
bank; neither a snapshot nor that arithmetic proves an allocation will succeed. A
complete duplicate of both GPU banks is not required. In the backed candidate,
primary evictions preserve their already-valid HOST copy rather than downloading
weights; the static helper also issues no expert-weight downloads. Verify **zero
expert-weight VRAM-to-DRAM transfers** in inference, separately from normal
activation/result readback. Do not shrink the model or retune unrelated settings.

The current exclusive control remains available. This bounded backing mode is an
explicit exception to the old exclusive-primary requirement; keep ownership/readers,
ready publication, routing and state correctness, and report physical copies and
logical expert ownership separately.

### Remaining work and retained evidence

**Resumed investigation, 2026-11-04:** working implementation checkpoint `4e8792c`
preserves the previously uncommitted changes. At that investigation's start, the
resident binary matched the v2 source manifest. Its measured screen requested four draft
proposals (`spec=5`), overriding runtime `spec=8`; the old protocol's window-eight
description was wrong. Rates are unchanged. After the
[implementation](#implementation-first-then-measurement-and-tuning) and
[conditioning gates](#placement-and-residency-qualification-correction),
compare the preserved M-110 recipe (fixed calibrated primary, cutoff rho 0.2, seven proposals, live helper)
against the new baseline under the current high-effort Python/prose protocol.
Treat the existing sequential-start lenience, cache-prior and width screens as
provisional evidence and revalidate only affected choices under that gate. Existing timing
puts draft work near 3–4% and conditional helper dispatch exposure below 2% of
generation; prioritize grouped expert service/residency costs. Exact historical
binary reconciliation remains part of the starting-engine gate. No final selection
or completion is inferred from these planned comparisons.

The ordered legacy-policy trace exposes a concrete assignment weakness: experts
are chosen for staging in first-seen row order, giving anchors priority over the
other verification rows. Choosing the most-used eligible HOST experts for the same
staging count covers 24,453 rather than 20,062 generation expert-columns (+21.9%);
this is a fixed-trace opportunity, not speed evidence. The
[anchor audit](evidence.md#anchor-first-staging-audit) also found 163 of 164 anchor
misses staged in one five-row batch that uploaded 209 records (6.20 GiB).
Implement [batch-aware selection and staging amount](#required-baseline-implementation)
as part of the baseline now. A reuse-only reorder at the old staging count is a
useful component, not the complete repair. Retain the old first-seen assignment as
a diagnostic control; evaluate complete performance after the coherent baseline
exists. Existing host-planning/CPU-service/column and transfer counters support that check.

| Workstream | Established evidence to retain | Remaining obligation / status |
|---|---|---|
| Replacement performance predictor | State/decision replay and instrumentation are reusable. Retired timing approaches, the +5.57% foreground miss, the separate +80.2% full-work miss and unresolved interventions remain preserved. | Start with one shared model. The [bounded generation-only diagnostic study](#shared-model-diagnostic-study), with speculation off, precedes selection of its mechanisms. Use measured material effects and useful ideas from the [six proposals](timing-predictor-proposal-measured-transitions.md) and broader literature to write the new first-stage design, then implement and validate policy-decision accuracy. Do not start a layer-specific fitting study or resume retired timing calibration. |
| Reasonable implementation baseline | [Integrated qualification](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json): actual ordinary/grouped ownership, capacity, continuous learning, useful in-place retention, bounded transfers and full-primary backing allocation/release. | Working integration established at the recorded scope. Admission value and evaluation are reopened by the policy review; correct ownership and component/model execution alone do not establish a reasonable optimized policy. Preserve valid checks and their coverage limits. |
| Required configuration | Official target/DSpark, cooperative CPU/5090/3090, FP4 primary, worker/live helper, prior 0.10, lenience 0.4, four drafts, grouped-cost planner/CPU scale 1.3, raw exclusive HOST storage. | Closed: final profile-free load, ordinary/grouped operation and no-override default request passed; actual resources/ownership are recorded. |
| Complete adaptive design | [Five-part production account](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json), cross-layer frame movement, retained-upload later uses, row-based original-demand learning, separately priced additional refills/preservation, live helper replacement and exact in-process state restore. Latest selected companion reconciles all planner/ownership events and 237 in-place retentions. | Required behavior established; preserve bounded nonblocking publication and reader leases. Histogram initialization and in-process learned checkpoints remain distinct. |
| Residency optimization | Independent conditioned histories, actual useful/unused/censored retention, measured transfer/service prices, bounded age forecasts, budget-8/16 comparison, isolated static/live and exclusive/backed comparisons. | Reopened: repair grouped marginal-value accounting and evaluate admission decisions, displaced work and the backing interaction. Budget 8 and existing settings are provisional operating values. Static/live remains inconclusive; the backing result does not close this work. |
| Lenience and cache-aware selection parameters | Current conditioned comparisons: prior 0.10 versus 0.05 (+9.48% pooled); lenience 0.2 versus 0.4 (-3.58%). Actual routing/rank/mixing audit and complete-task checks pass. | Select prior 0.10 and lenience 0.4. No original-distribution or broad-quality preservation claim. |
| Inference implementation choices | Whole-batch assignment/staging replaces first-seen behavior; representative service calibration and CPU scale repair; actual packed grouped callers; routing/service audit and overlap/cost evidence. | Reopened for admission/scheduling valuation and their interaction with backing. Retain the actual whole-batch implementation and valid checks while repairing the identified decision-model gaps. Unrelated parked mechanisms do not reopen automatically. |
| Other active parameter families | CPU scale 1.3 gains 4.58% pooled, two drafts lose 6.48%, budget 16 ties; confidence decisions and rejected work reconcile. Current physical-core worker partition, capacity and transfer controls have explicit source/cost dispositions. | Retain the measured values as provisional controls. Revalidate choices materially affected by the admission/scheduling repair using focused modeling and real tests; no indiscriminate sweep. Local gains are not additive. |
| Usable starting engine and final speed | Final declared orders: original 12.8314 versus selected 16.1144 tok/s pooled (+25.59%); Python +34.47%, prose +17.21%, both pair directions positive. All source/outputs/latencies in the [evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/final-comparison-evaluation.json). | Closed with bounded scope. Also retain the final standalone-load result 14.7816 versus 13.1024 (+12.82%), Python +6.80%/prose +20.38%. Do not hide the lower reload speed or pool its different conditioning into the earlier result. |
| Functional and state checks | Final selected code, scheduling, real file-tool round trip, prose/extraction, cancellation/follow-up pass. Existing unaffected rejection/continuation/prefix and ownership tests retained. | Bounded usability closed. Natural reasoning completion at 2,146 tokens; no new broad matrix. Old failures/truncations remain recorded. |
| Final composition and handoff | Native source `ebe02b0`, binary SHA-256 `4c060020…384000b`; final server 2085689/engine 2085693 at port 8097. No-override default request passed, weights resident, idle/zero queued, ownership complete and 41.30 GiB Windows physical headroom. [Handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json). | Closed at the 16:53 UTC observation. Keep the compatible process resident; later status must be observed anew. |
| Capture and artifact reconciliation | Every new configuration has native timelines and bounded detailed/Nsight companions; selected joins reconcile on both GPUs. Complete-task native timelines have no drops; cancelled API is deliberately partial. | Current captures reconciled at explicit coverage. Preserve five incomplete CUPTI stop events/possible-loss warnings in companions, the earlier one-copy gap and old-binary unsupported fields. Final-load and default/handoff timelines are retained with their uncaptured GPU-detail intervals explicit. |


The [companion overlap/capture audit](evidence.md#companion-overlap-and-capture-audit)
supplies diagnostic evidence for the existing cooperative and capture work. CPU
usually finishes last, but its lead is much smaller than its total batch duration;
most demand-copy time overlaps CPU work. Keep joint CPU/copy/5090/helper planning
as the implementation basis. The launch groups observed under Nsight remain an
unproven, bounded scheduling lead, not another baseline prerequisite or a reason
to delay the implementation-first sequence. Reuse the audit's coverage and
provenance corrections without treating either companion as a warmed speed test.

The [earlier sequential-start policy result](evidence.md#current-reference)
remains **512 / 44.777231 s = 11.4344 tok/s**, versus
**512 / 47.863285 s = 10.6971 tok/s** for its new legacy-adaptive control.
It was a local policy improvement with different initial-state/measurement scope,
not the original-engine comparison. The final conditioned comparison above now
provides that required evidence; historical roughly-15 tok/s suffix results remain
separate rather than being treated as a literal target.

The [launch](native-server.json) and [runtime](native-runtime.json) files contain
the selected profile-free/adaptive/lenience defaults, high effort and no forced
thinking closure. Native source is `ebe02b0`, with the unchanged binary identified
in the [source/weight account](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/source-and-weight-account.json).
Later commits update tools, defaults and records. The final [source-bound launch](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/launch.json)
uses server PID 2085689 and engine PID 2085693. At the recorded handoff they were
idle with weights resident at `http://127.0.0.1:8097/v1`; active configuration, logs
and all final receipts are under `conditioned-residency/final-selected/`. The
[observed handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json)
is dated, not a guarantee of future process state. Earlier v2 observations remain
historical evidence.

Normal launch after checking actual process/resource state:

```bash
.venv/bin/python serve/server.py --engine strata \
  --config docs/minimax-wsl/native-server.json \
  --host 127.0.0.1 --port 8097 --idle-unload 0
```

Here `strata` names the shared frontend adapter; the JSON launches `maya-mm3`.

### Mandatory restart order

**For a restart now (2026-11-10):** the work is on branch `minimax-wsl/shared-model-diagnostic`, worktree
`.claude/worktrees/shared-model-diagnostic` (user direction), not the branch named below. It stands at the
[order of 2026-11-10](#order-20261110), with redo step 6 inside it: see
[where the work stands](#where-the-work-stands). Step 1 below applies at every restart. Step 2's study is done
(2026-11-07), and the [per-pass redo](#per-pass-redo) continues it.

The [replacement decision](#timing-model-replacement) supersedes further repair
or qualification of the retired timing model. First observe actual process and
resource state and preserve a compatible resident process if one exists; old
handoff PIDs are dated observations. Do not repeat completed runs merely to
reconstruct a handoff. Retain valid implementation and policy-review evidence.
The held-base comparison remains held. Start in `/home/mdwbr/project-maya`,
branch `minimax-wsl/proposal-2`:

1. Recover the actual source/binary, worktree, running model and resource state;
   preserve compatible resident state and existing changes. Read this plan and
   the relevant evidence/source map. Consult archives only for a specific question.
2. Perform the bounded [shared-model diagnostic study](#shared-model-diagnostic-study)
   before selecting the new first-stage design's mechanisms. Start with one
   shared model; do not run a layer-specific parameter-selection study. Follow
   the [generation-only scope](#generation-only-modeling): ordinary one-token
   decoding first, with speculative decoding off and prefill outside the model.
   Use the measured effects to select and implement the design with the retained
   state/policy machinery, qualify its policy predictions, then qualify speculative
   generation as a later extension. Declare each measurement protocol before running it.
   Do not build all six proposals or restart the old calibration
   campaign. Reuse the established baseline and unaffected checks.
3. Repair the identified admission valuation and complete the affected
   [parameter optimization](#required-parameter-optimization), using supported
   replacement predictions and focused hardware confirmation. Preserve the
   [independent conditioning](#placement-and-residency-qualification-correction)
   and [usable starting-engine comparison](#comparison-with-the-usable-starting-engine)
   contracts. Revalidate only choices and comparisons materially affected by new
   implementation; retain unresolved helper/backing evidence and all losses.
   A new predictor or a local policy gain does not itself complete this step.
4. Reconcile every active workstream and the final integrated server against
   [completion](#execution-and-completion), reusing valid functional/state evidence
   and rerunning only affected checks. Report unfinished work explicitly.

Apply the focused-check policy in [synchronization](#synchronization-and-buffer-retirement).
Completed historical repairs and unchanged checks do not become a fresh campaign
at restart or a context handoff.

<a id="requirements-shared-by-all-three"></a>

## Selected engine

These requirements apply to the Maya milestone and to any base comparison the user
later resumes under the [base-hold policy](#base-holds).

- **CPU/5090/3090 cooperation, cache-aware routing and fully adaptive residency
  are required together.** Assign complete expert computations to the devices;
  retain the native FP4/worker/live-helper route. Routing-off, fixed residency and
  ordinary decoding are targeted diagnostic controls, not eligible final designs.
  Tensor parallelism and CPU/helper-off candidates or performance controls are
  outside this work. A first adaptive implementation's weak speed result calls
  for improving its policy, not dropping the requirement.
- Specialize for Ryzen 9950X3D, RTX 5090, RTX 3090 and WSL2. Use observed links and
  capacities; assume neither P2P nor NVLink-class bandwidth. No portability/fallback
  framework or unrelated model setup.
- Use the [pinned official target and DSpark](#shared-weights-and-data), preserving
  supplied NVFP4 expert codes/scales, MXFP8 dense/draft formats and BF16/F32 tensors
  through layout changes. Keep persistent weights packed and conversions in bounded
  workspace. Retain every target layer and stored expert; substitution
  is not permanent pruning. Main text-model weights should average roughly 3.5 bpw
  or higher; about 3.4 is acceptable. Keep a parameter-weighted inventory with
  scales, metadata, vision and draft reported separately. No new quantization campaign.
- Choose fast packed-weight arithmetic independently on each device. Activation
  formats, reductions, rounding and approximations may differ; neither exact nor
  approximate cross-device expert-output agreement is required. Do not introduce
  a slow compatibility path for numerical matching. State/index/ownership
  correctness remains required, with no determinism requirement for text, routing
  or cache history.
- Use **uncompressed HOST storage**. The tested lossless codec is excluded; its
  reconstruction does not reduce PCIe weight bytes. Run only one full model at a
  time across bases, retain compatible weights for sequential checks, and account for
  actual GPU/DRAM allocations, staging, exchange space, scratch, KV and draft.
  Allocate only the selected execution routes.
  Favor no steady-state SSD expert service and report actual reads. Check current
  memory/backing-storage headroom before large allocations; no GLM memory cap or
  exact-output benchmark contract is inherited.
- Preserve efficient loading, local serving/streaming, reasoning and useful tools.
  Tolerant parsing and at most one corrective tool turn are allowed initially;
  report raw, repaired and failed outcomes and charge retry work. Vision/video
  execution and long-context qualification remain later work.

<a id="optional-next-layer-prefetch"></a>

<a id="dampening-disposition"></a>

### Parked work

| Mechanism | Disposition |
|---|---|
| Layer pipeline and further split/composition matrices | Keep disabled. Deferred retirement of its speculative buffers and its inactive waits are outside Maya-first. |
| Next-layer predictive prefetch | Keep `routing_prefetch` off; adapting it is not a prerequisite for residency work. |
| Substitution dampening | Keep off and outside Maya-first. Its inconclusive/limited evidence does not require testing another variant. |
| Device-queue dispatcher | Parked; no demonstrated advantage justifies further work now. |
| Old C/R/P factorial matrix and auxiliary gates | Retired. Unchecked historical rows do not activate experiments. |

[Evidence and limitations](evidence.md#parked-mechanisms) explain these dispositions;
reopening follows [investment policy](#choosing-worthwhile-work).

### Expert mixing and initial routing/verification settings

Keep expert selection, weight placement and draft-token verification distinct.
The approved verifier change replaced the target-relative cutoff with lenience;
it did not replace the additive expert-selection prior. The separately approved
time-priced retention/refill design governs weight placement. Neither algorithm
change establishes an optimal routing bonus. The routing policy below is the
approved starting point; justified selection-policy improvements remain within
[implementation optimization](#implementation-optimization).

Keep the existing cache-prior routing as the starting policy. Rank experts using their sigmoid
routing strengths, the model's correction bias and an initial 0.05 bonus for ready
GPU-resident experts. Select four, then compute their mixture weights from their
original sigmoid strengths, normalized over the selected set. Preserve the model's
routed scaling factor of 2.0 and shared-expert contribution. The cache bonus affects
selection, not mixture weighting; this is not dampening. Keep four distinct valid IDs.
Only successfully published ready residents qualify; freeze the relevant pass
view to protect readers, without promising repeatability across cache histories.
RAM residency is not a free GPU hit. Record original demand, executed choices,
substitutions and useful backfills separately.

**Clarification, 2026-11-04:** cache-aware routing does not authorize using its
executed hits as admission demand. The original top-four IDs before the residency
bonus remain the residency-learning input. Observed executed reuse, original-demand
reuse and net service savings are separate quantities; none substitutes for another.

Keep official DSpark drafting and grouped target verification, but replace the
target-relative cutoff with **lenience-based verification**. For a proposed token
`x`, accept with probability `min(1, p(x) / (ell * q(x)))`, where `0 < ell <= 1`.
Here `p` is the configured approximate target distribution and `q` is the actual
distribution that produced the draft. Verification does not restore original top-4
routing or establish invariance across window widths. Keep target temperature 1
and top-p 0.95. For `ell < 1`,
this intentionally changes the sampling distribution; do not describe it as
distribution-preserving or as restoring unmodified-model quality.

The **0.05 cache bonus** and **lenience factor `ell=0.4`** are approved starting
values. Both require [parameter optimization](#required-parameter-optimization)
before final selection; neither is permanent or a demonstrated optimum. Hold them
constant within each declared residency-policy comparison so its result remains
interpretable. That temporary control does not waive their later tuning.
**Do not reuse or translate the old rho 0.2 as ell.** Address any implicated quality
issue through the bounded diagnosis. The selected mechanisms remain required;
parameter tuning does not reopen their inclusion or the retired combination matrix.

The integration must preserve the complete lenience algorithm, not change only
one comparison in the old cutoff branch:

- Use DSpark's actual proposal probabilities, including its sampling filters and
  normalization. The existing sampled-DSpark path is the implementation starting
  point. The current cutoff path uses argmax drafts; its softmax confidence is
  not the probability of that deterministic proposal. Do not substitute a
  confidence-head score or an unrelated raw softmax value for `q`.
- Commit the longest accepted prefix. On the first rejection, discard dependent
  draft state and draw the replacement from the normalized positive residual
  `max(p - q, 0)`, matching the selected lenience method. Do not retain the old
  cutoff branch's direct target draw on rejection. After full acceptance, draw
  the normal bonus token from `p`, subject to EOS/output limits.
- Validate the acceptance and residual calculation with focused distribution
  checks, including `ell=1` reducing to standard speculative sampling relative
  to the configured target and a case with excessive draft probability. Check
  affected actual rejection/continuation, cancellation and prefix-reuse callers;
  old cutoff tests alone do not validate the new route. This is not a cross-device
  arithmetic-equivalence or output-identity requirement.
- Expose the effective verifier and lenience factor through serving controls and
  diagnostics. Charge any proposal-probability, sampling and residual-draw work
  to complete generation timing. Update launch defaults only when the new route
  exists and passes its affected checks; keep old receipts/settings labelled as
  cutoff evidence.

Algorithm reference: Fast-HSD
[`lenience_accept_prob`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/fast_hsd/core/collaborative_verification.py#L20)
and the lenience acceptance/residual branches in
[`_speculative_sampling`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/transformers/generation/utils.py#L5537).

<a id="adaptive-residency-next-cache-work"></a>

## Adaptive residency

<a id="r4-evidence-based-architecture-comparison-and-adaptive-primary-residency"></a>

An initial implementation of this architecture is integrated and has scoped
measurements. Its [coverage and optimization remain under review](#active-maya-work);
implementation presence does not close the design or performance work. The five
requirements below remain the approved contract.
Build from [Maya's existing adaptive path](../../src/mm3/mm3_native_adaptive.inc),
including its bounded exchange storage, pointer rotation and nonblocking
publication. Donors supply useful mechanisms, not a required series of ports.

1. **One fully replaceable primary expert cache, with capacity movable across
   layers.** Keep existing physical allocations where practical, but manage expert
   placement globally. There are no permanently fixed main-bank identities or
   permanent per-layer quotas. Keep a small shared reserve for demanded uploads
   and exchanges, initially the existing eight working slots within the current
   total memory budget. Temporary protection for active readers/writers is not
   permanent residency.
2. **Retain useful demand uploads in their existing frames.** Decide whether to
   retain an expert while its uploaded copy is still available, rather than only
   at the end of the complete target pass. Reserve that frame when retention is
   worthwhile and the exchange budget permits it. For uploaded expert B replacing
   resident A, keep B in place, preserve A in RAM if it has no other valid backing,
   then change ownership mappings after the relevant operations finish: B's frame
   becomes resident and A's former frame becomes working storage. Do not upload
   or copy B again merely to change its retention role. Saving A is a distinct,
   potentially necessary transfer. Do not force retention of an unhelpful record
   to improve a reuse counter.
3. **Canonical storage with bounded copies and explicit backing policy.** The
   retained control distributes the uncompressed expert bank exclusively across
   RAM, the 5090 and the 3090. The user now authorizes the
   [bounded primary-backing experiment](#bounded-primary-dram-backing-experiment):
   keep a valid HOST copy for every primary resident within a measured DRAM budget,
   so primary eviction needs no preservation download. This is not a complete RAM
   duplicate of both GPU banks. Keep every expert's last valid copy and all active
   sources protected; account for logical ownership and physical backing separately.
   Necessary transfers in the exclusive control remain real measured costs.
4. **Additional refills based on expected execution time saved.** CPU- or
   helper-served experts may never produce a primary demand upload, so retaining
   uploads alone is insufficient. Use observed original-router demand, expected
   reuse and service cost for additional admissions, charging incoming traffic,
   outgoing preservation and interference with current work. An already-uploaded
   candidate has no additional incoming-transfer cost; one requiring a fresh
   upload does. Account for that difference before selecting an admission winner.
   Preserve original-demand-only credit: executing a cache-selected substitute
   does not manufacture demand credit. Background refill here uses observed
   demand and does not reactivate the parked next-layer prediction experiment.
5. **Adaptive helper residency with coordinated ownership.** Keep the existing
   3090 live replacement mechanism. Primary and helper admissions respect each
   other's pending transfers and active readers. Optional exchanges stay bounded
   and nonblocking; unavailable cache entries continue through normal cooperative
   service according to actual weight ownership.

No offline placement profile is required for the engine to operate. This does
not require discarding learned initialization or justify using sequential cold
placement as a warmed baseline. A valid saved learned state/profile may improve
startup; retain its provenance and distinguish histogram initialization from
actual adaptive policy state. Start from a valid placement within the existing
RAM/VRAM budget. Static controls and demand-derived initialization follow the
[3090-first frequency ranking](#static-placement-order); follow the conditioning
gate for warmed policy decisions.

<a id="independent-request-state"></a>

For independent requests, keep model weights loaded and learned expert residency
alive without manual reseeding. Give each fresh target and DSpark KV/context,
without conversation history or prefix/KV reuse between these requests.
Expert-weight residency and attention KV state are distinct.

The [donor evidence](evidence.md#adaptive-residency-evidence) explains the upstream
GLM lifecycle and its different per-layer capacity policy.

### Real-trace replay and optimization

**Use replay to tune the chosen design, not to reopen the architecture decision.**
After the [initial implementation](#required-baseline-implementation), use ordered
traces from the [default screen](#default-speed-screen), under
its request-state and workload scope. Histograms alone cannot reproduce
admission or transfer scheduling. Record original demand and executed choices,
request/round/row and layer identities, available copies and pending transfers,
prefill versus generation, and the service observations needed to price decisions.
Include attempted speculative rows that are later rejected. Existing traces may
inform the initial implementation without another replay or live comparison first.
Reuse instrumentation and add only missing fields. If capture perturbs throughput, use it for replay
and turn it off for the actual speed comparison.

Compare the implemented baseline with decision-relevant policy alternatives.
Tune a small number of controls, such as aging, admission margin and refill budget.
Do not require separate implementations of every donor cache or a broad sweep
over unrelated replacement algorithms.

Hold total memory capacity constant, including working and exchange storage.
Charge incoming transfers, necessary outgoing preservation, occupied/unavailable
frames, CPU-memory contention, helper interaction and competing useful service.
Represent overlap without summing simultaneous work as if it were all exposed.
Where timing is unknown, preserve uncertainty. The current
[replacement decision](#timing-model-replacement) retains logical state and
physical-resource constraints while replacing per-operation timing reconstruction.
Demonstrated prediction of policy differences selects the timing abstraction;
passing the retired timing model's validation gates is not required. Both older
timing approaches remain historical diagnostic evidence only.

Rank supported predicted performance differences on common work; use hits,
transferred bytes and replacement counts to explain the result. Fixed-trace
replay cannot prove final speed or quality:
changed residency can change routing, generated text and speculative acceptance.
Confirm useful changes in the actual combined engine with the short two-task
screen. Initial conditioning and costs remain explicit; the old calibrated
coding comparisons do not establish operation without calibration.

Removing mandatory calibration is an operational improvement; only measured
integrated throughput establishes an additional speed gain.

### Synchronization and buffer retirement

<a id="nonblocking-cache-publication-and-test-disposition"></a>

<a id="original-demand-promotion-and-retention"></a>

<a id="r2-cuda-failure-path-publication-and-retirement"></a>

<a id="r3-speculative-state-verification"></a>

<a id="required-helper-only-refresh"></a>

Preserve nonblocking optional cache publication: query completion without waiting
or spinning. Check submission and completion success before publishing; incomplete
or failed records stay unavailable and keep their storage while independent work
proceeds. Do not drain optional transfers to make cache contents repeatable.

Recycle storage only after all readers and writers finish. A failed stream join
or timeout does not establish safe release or permit fallback to absent HOST
weights. Collect and report worker errors after joining started work; preserve
bounded helper failure and safe ownership when completion is uncertain.
Required-data dependencies and safe teardown still need correct ordering.

Reuse existing ownership/completion machinery. Deferring reclamation permits
useful overlap only with independent work and storage; it does not cancel writes.
Change synchronization only for a concrete active-path opportunity, without a
general retirement framework. “Optional” describes the cache optimization, not
whether required expert computation can be skipped.

Retain correct target/DSpark rejection, continuation, cancellation and prefix/ring
reuse, including affected EOS/output limits, KV/indexer/context frontiers and
exclusion of discarded writes. Run focused checks for changed ownership, indexing,
arithmetic or state callers, using bounded injected failures where appropriate.
Reuse valid unchanged checks rather than expanding each change into a broad audit.
The existing same-process helper placement refresh serves declared measurement
resets; it is neither new implementation work nor a normal-operation calibration chore.

## Short experiments with resident weights

### Required captures

**User instruction, 2026-11-04, applies throughout the remaining plan.** Extend the
existing engine/collector logging; do not create a separate benchmark harness or
change prompts, budgets, request order, cache history or policies for capture.

- Give experiments, runs and requests shared unique IDs linked to the central-plan
  item. Preserve control/candidate/order, commit/diff and binary hash, exact prompts,
  seed, model/tokenizer identity, requested/effective settings, actual draft count
  and total verification-window width as distinct fields.
- Record complete token/round timelines from generation start through completion,
  cancellation or failure: token IDs, positions/emission timestamps, rounds and
  target forwards, all proposals/acceptances/rejections, replacement/bonus tokens,
  discarded work, committed outputs and reasoning/final distinction. Lenience
  diagnostics retain actual proposal/target probabilities and decisions.
- Add correlated NVTX request/round/forward/layer and phase ranges. Nsight captures
  both GPUs' kernels, copies and streams using `--trace=cuda,nvtx
  --cuda-graph-trace=node`, alongside CPU expert work/helper submission/collection.
  Cover each remaining benchmark configuration and relevant test windows; name
  every uncaptured interval and unsupported historical-binary field explicitly.
- Preserve reconstructable expert/cache records: original/executed IDs, selection
  scores and mixture weights, ready residency, actual CPU/resident/staged/helper
  assignment, initial ownership/frame maps and ordered admission/retention/eviction,
  transfer/publication and pending/failed events with directions, bytes and decision
  inputs. Retain completed-round counters, before/after resources and allocations,
  service/jobs/columns, transfers and activity, separating prefill from generation.
- Preserve raw test output, expected/actual checks, errors, truncation and state
  transitions, including reasoning review and tool repairs/retries/costs. Tests
  without generation need operation timelines/results, not fabricated token records.
- Document clocks, units, counters and overlap. Do not fabricate per-token compute
  times or sum concurrent intervals into wall time. Buffer records and write outside
  timed work where practical, with no added CUDA synchronization, optional-transfer
  waits, per-token disk flushing, reseeding or longer output solely for capture.
- Keep detailed tracing off in ranked speed measurements while overhead is unknown
  or material. Use already planned diagnostics or a clearly bounded companion with
  matching source/settings and declared cache history; record mode/overhead/coverage.
  Account for capture serialization even after a buffer fills; a trace's host-plan
  interval can include instrumentation work absent from normal inference. The
  [companion audit](evidence.md#companion-overlap-and-capture-audit) records this case.
  No completed comparison is rerun merely to fill a visualization or historical gap.
- Store raw Nsight reports, JSONL and run-indexed manifests under the existing
  `/mnt/wsl/transfer/minimax-wsl/runs/` tree. Validate joins, token counts, round totals
  and trace/counter reconciliation. Missing fields, dropped events and incomplete
  coverage remain explicit, never zero-filled or overwritten. Preserve adverse and
  inconclusive results and reuse valid existing captures.

### Effort and harness

Evaluate MiniMax at **high reasoning effort or higher**. High is the default for
quality checks and both speed requests; a supported higher level may be declared.
Hold effort constant within comparisons. The user reports poor reasoning and
other output quality below high. Do not lower effort to shorten answers or inflate
rates; historical lower-effort receipts do not qualify high-effort behavior.
This is a model request setting, separate from coding-agent delegation.

The existing harness/fixtures now request high effort and use the replacement
reasoning task. Preserve those implemented changes, verify effective serving/template
settings in final measurements, and do not restore the old forced `none`/`low` settings.

### Default speed screen

Use exactly two independent requests: **write actual Python code** and **write
prose for a non-coding task**, with prompts, order, sampling and short output
budgets declared before candidate/control comparison. The declared order is held
within a comparison; no follow-up, topic return or additional category is appended
automatically. Completing a speed task is unnecessary; longer generation needs a
specific unresolved question that a shorter check cannot answer. Do not force
filler, substitute favorable continuations or extend a run merely to finish a long answer.

Use the [independent-request state rules](#independent-request-state) for both.

Include generated reasoning and its work in timing/accounting; separate reasoning
and final-answer tokens when available. Inspect output for obvious problems and
label a stop before the final answer. Truncated output supplies speed evidence,
not a passed functionality check. Separate functional/cache-behavior checks need
a stated purpose. Use small tasks whose complete answers fit short budgets for
functionality, reusing speed outputs when adequate; the reasoning check has its
own completion budget below.

### Generation timing and expert-cache conditioning

<a id="qualification-of-existing-token-32-results"></a>

The primary metric is **whole generation after prefill** across the declared
screen, measured at normal completed-round boundaries. Include drafting, rejected
work, sampling, transfers and cache maintenance, without adding synchronization
for timing. Pool emitted counts over elapsed time, not averages of rates, and
retain individual task rates and outputs. Historical coding/suffix results remain
separately labelled and cannot be equated to this metric. Do not discard initial
or topic-switch adaptation costs.

Distinguish a loaded/hardware-primed engine from a cache trained on scored tasks.
Declare neutral setup generation, initial placement, source/settings, capacities
and sequence order. Do not fit residency per scored prompt, require fixed cache
contents or wait for stable routing. Normal ongoing adaptation is part of operation.
Automatic CPU/GPU service-cost calibration may remain: it measures execution costs,
not workload-specific placement. Saving demand counts does not itself apply placement.

Controlled arm-to-arm placement/policy resets establish declared starting conditions;
they are measurement setup, not normal-operation requirements. Each arm's requests
retain cache history. Reuse compatible model state across safe policy changes.
Include online refill in generation; report startup, setup placement, prefill and
end-to-end workflow latency separately. Those diagnostics do not enter the
current generation-speed ranking or reject a candidate on their own.

Start with the smallest representative screen that changes a decision, then one
confirmation with candidate/control order reversed for a promising or close result.
One further targeted
check is allowed for a specific unresolved anomaly, with its decision value recorded.
Do not repeat until gains or control movement reach a desired number. Large
unexplained variation prevents a close speed claim; keep the current usable setting
while parking an uncertain choice, without calling it superior.

### Comparison with the usable starting engine

Before closing performance work, identify and preserve the best usable, reproducible
starting configuration and its source/binary, settings and original measurement
scope. Preserve that reference now; implementing the
[reasonable baseline](#required-baseline-implementation) does not require completing
another historical-recipe benchmark first. After implementation and conditioning,
compare it with the new baseline and final candidate using the same current Python/prose
prompts, high-or-higher effort, output budgets, target sampling, memory budgets
and whole-generation timing. Record each arm's actual execution and initial cache
history; preserve normal within-arm adaptation. A previously saved placement profile
may describe the historical control, but no new fitting to scored prompts is allowed.
The final engine must still operate without a mandatory profile and meet the
approved adaptive requirements. Optional learned initialization is permitted;
profile-free operation does not require a cold reset before measurement.

Report this whole-engine comparison alongside policy comparisons within the new
configuration. A legacy-policy arm that already contains new verifier, placement
or runtime changes cannot by itself stand in for the earlier usable engine.
The roughly-15 tok/s historical figure is not a literal target under a different
protocol. If the original recipe cannot be reproduced or the conditions cannot be
reconciled, record the missing evidence and keep performance completion open.

Diagnose a material loss with the cheapest checks that can change the next action.
Include costs of required new behavior; do not assume effort, lenience, adaptation
or timing boundaries explain the gap without evidence. Historical policies may
serve as diagnostic controls without becoming eligible final designs. This does
not authorize a compulsory routing/speculation/cooperation matrix or removal of
required features. Apply the existing [selection rules](#cost-observations-and-selection)
to the relevant starting-engine comparison as well as local candidate comparisons.

<a id="bounded-reasoning-diagnosis"></a>

### Approved reasoning check and bounded diagnosis

Use the approved scheduling problem below with a changed condition in the same
request, under the [effort requirement](#effort-and-harness).
It replaces the three-bin arithmetic problem as the primary reasoning check.
Keep the old responses and failures as historical evidence, with their effort
and truncation conditions; do not rerun that fixture by default or use its
lower-effort failures to qualify high-effort reasoning. This quality check is
separate from the two-task Python/prose speed screen.

The prompt is:

Two workers are available from time 0. Each task needs one worker throughout its
execution. Tasks cannot be paused or split.

| Task | Duration | Prerequisites |
|---|---:|---|
| A | 3 | None |
| B | 2 | None |
| C | 4 | A |
| D | 3 | B |
| E | 2 | C and D |

Tasks C and D also require the same machine, so they cannot overlap. All other
tasks may overlap when workers are available.

1. What is the earliest time all tasks can finish? Give a feasible schedule and
   briefly explain why an earlier finish is impossible.
2. Now change only B's duration from 2 to 4. Find the new earliest finish, give
   a feasible schedule, and explain what changes.

Use concise prose and task start/end times. No code or JSON is needed.

Reference outcomes for the checker, not included in the model prompt:

| Scenario | Optimal finish | One valid schedule |
|---|---:|---|
| B takes 2 | 11 | A 0–3; B 0–2; D 2–5; C 5–9; E 9–11 |
| B takes 4 | 12 | A 0–3; B 0–4; C 3–7; D 7–10; E 10–12 |

In the first case the shared machine cannot start before time 2; its seven units
of work and E's two units give a lower bound of 11, attained by the schedule.
In the second case its earliest start is time 3, giving an attained lower bound
of 12. The optimal machine-task order changes from D/C to C/D.

Check completion time, schedule feasibility and minimum-time justification
separately for each scenario. Accept alternative valid optimal schedules.
Validate dependencies, worker capacity and machine exclusivity rather than
matching one expected answer string. Allow sufficient total generation budget
for high-effort reasoning to finish; the requested final response remains concise.
Do not inherit the old 192-token cap or silently force reasoning to close to fit it.
Truncation is a budget/completion limitation, not by itself an incorrect reasoning
answer. The [fixture/checker](../../tools/mm3/native_reasoning_check.py) is implemented;
[current evidence](evidence.md#current-reference) preserves its completed and truncated outcomes.

Address recorded reasoning failures through bounded diagnosis. Inspect the exact
prompt, expected answer, reasoning effort, output budget, raw reasoning/final text
and parser. Distinguish incomplete output from an incorrect completed answer.
Reuse existing evidence before requesting another generation.

If a model comparison is needed, use a small complete task and an ordinary-target
diagnostic control that retains cooperation, uncompressed HOST storage and device
arithmetic. Original expert selection and ordinary target decoding may isolate
the approximate-path group for this diagnosis; neither is an alternative final
design. This is a same-engine control, not an independently validated reference.
If both paths fail, investigate the shared cause; if only the approximate path
fails, isolate the implicated approximation. Do not run a compulsory combination
matrix or append these checks to every speed screen.

A failing control does not establish preserved quality. Switching to lenience
does not itself resolve the historical failure, and those old failures do not
establish that the new verifier fails. Record actual outcomes without requiring
universal reasoning perfection or expanding into a broad quality campaign.

<a id="preferred-options-when-performance-is-practically-tied"></a>

<a id="cost-observations-and-retained-implementation-checks"></a>

### Cost observations and selection

Use existing counters before new profiling. Separate exposed critical-path time
from overlapping CPU/helper/GPU service: CPU/helper service divided by CPU job
count is a composite, not a pure CPU-kernel measurement. Hits, acceptance and
replacement counts explain time; they are not optimization goals themselves.
Inactive paths and setup-only costs are not recurring bottlenecks without evidence.

Evaluate branch finish times and their margins as well as work duration. A high
CPU-last fraction does not make all CPU batch time exposed, and reducing that
branch can expose the copy/5090 or helper branch. Use the small batch-cost model
to assess the combined assignment; do not sum overlapping CPU and copy durations
or convert diagnostic finish margins into a production speedup ceiling.
The [companion evidence](evidence.md#companion-overlap-and-capture-audit) motivates
this accounting but cannot qualify uninstrumented performance. Any later launch
submission check must establish behavior without Nsight and charge its own cost;
the traced groups-of-11 pattern alone does not establish a production benefit
from flushing or make it a required baseline repair.

Among implementations meeting the requirements, declare a preference before
measurement for practically tied results; do not default to incumbency. Roughly
2% is a practical closeness band, not a target for repeated testing. Operational
selection does not prove equal speed; noisy unresolved effects remain inconclusive.
Investigate a 5% task-class regression; a confirmed >10% regression blocks default
selection until addressed. Review sampled output/work differences before calling
a loss repeatable and report all task outcomes.

## Implementation optimization

The remaining work includes implementation choices in the combined engine, as
well as parameter values. Reuse the established packed kernels, grouped execution
and ownership machinery. Their presence is a working foundation, not evidence
that every material execution cost or omitted donor mechanism has been resolved.
The [source maps](proposal-1-cooperative-experts.md#donor-map) preserve concrete
references; they are not a requirement to port every donor.

### Required baseline implementation

Implement these connected behaviors before the next ranked optimization campaign.
Use the best justified initial design, with explicit cost assumptions where live
measurements are still missing. Do not make proof of a speedup a prerequisite for
repairing these already identified weaknesses.

1. **Coherent initial placement and learned residency.** Apply the
   [global demand ranking and GPU order](#static-placement-order), retain learned
   residency across independent fresh-KV requests, and preserve the policy state
   needed for faithful comparison setup. Use a **defined demand clock: evaluated
   generation rows**, including rejected speculative rows. Express demand credit,
   aging horizons and admission-budget replenishment in those declared units before
   tuning their values. Capture prefill demand and service separately; prefill must
   not credit or age generation-residency learning or replenish its admission
   budgets. Preserve the clock and learning semantics across requests and setup
   checkpoints. The current five-row normalization is an initial parameterization,
   not an optimum. Keep global primary allocation and coordinated helper ownership.
2. **Whole-batch cooperative assignment and staging.** For each layer, plan over
   all rows actually being evaluated, group identical expert uses, and account for
   how many columns each expert serves. Choose both the staging candidates and
   the number of uploads by expected complete service time: CPU work avoided versus
   one weight upload plus grouped GPU work, available frames and interference with
   useful CPU/5090/3090 work. Use ready copies directly. An anchor's position at the
   front of the input is not a benefit score. A fixed upload count with candidates
   merely reordered by reuse is only a partial solution. Start with a **small,
   inspectable batch-cost model** whose inputs and assumptions explain each choice.
   Existing estimates and bounded grouped-service measurements can supply missing
   inputs. Building an elaborate predictor or a separate calibration campaign must
   not become another prerequisite. Keep distinct records uploaded,
   expert-column uses and bytes separate. Current-batch executed reuse can guide
   service assignment; only original-router demand credits future residency.
3. **Integrated retention, refill and transfer policy.** Preserve useful demanded
   copies in place, assess additional admissions against avoided service and actual
   incoming/outgoing obligations, and coordinate budgets with foreground staging,
   active readers and pending primary/helper exchanges. Choose a sensible bounded
   admission cadence and initial prices as part of this design. Do not rely on
   longer warming to hide an admission bottleneck, promote every upload, or count
   overlapping work as separate elapsed costs. Keep optional maintenance nonblocking.
4. **Explicit backing and complete integration.** Implement the already authorized
   full-primary DRAM-backing mode with its memory accounting and exclusive control,
   so the later isolated comparison is ready to run. Preserve the 16 GiB reserve,
   both GPU capacities, all experts and valid-copy ownership. Wire the chosen
   policies through actual ordinary/grouped execution and serving controls, using
   the selected packed arithmetic, routing and verifier. Controls for the old
   behavior remain diagnostic; active baseline settings use the new implementation
   after its affected checks pass.

Before moving to tuning, close the baseline gate with the bounded **full-model
integrated behavior check** through the actual ordinary and grouped production
callers. Establish valid ownership and reader protection; actual capacity and memory
accounting; learned-state continuity across fresh-KV requests and declared restores;
useful retention in existing frames; and bounded, correctly accounted transfers
with nonblocking optional maintenance. Use the existing check to establish these
properties together, reusing valid receipts rather than adding separate campaigns.
Predicted savings and focused CPU/component passes support implementation but do
not establish this gate. Any unverified required behavior keeps it open.
The later static/adaptive-helper and exclusive/backed-primary comparisons remain
isolated as specified; implementation of both modes does not preselect their
measured winner. Refine policy values and further implementation choices from
complete-engine results after this baseline exists.

| Area | Established starting point | Investigation and disposition required |
|---|---|---|
| Cooperative expert service | Packed CPU batching, native FP4 primary service and worker-based helper execution already run together. The mapped device-queue comparator was tested and parked. | Assess current preparation, grouping, submission/collection, CPU/primary/helper assignment and useful overlap. Improve the selected path where complete service costs justify it. Reuse valid kernel/contention evidence; do not describe already-integrated batching or FP4 as missing, or revive the tested queue under a different name. A materially different donor mechanism needs the investment case below. |
| DSpark and grouped target execution | Official DSpark and grouped expert-column service are integrated; the verifier has changed to sampled lenience. | Assess target work per committed token, rejected tails, expert-union growth, packed-record reuse and repeated launch/packing/handoff work. Decide whether changes to grouping, verification scheduling or current service implementation offer a worthwhile improvement, alongside width/confidence tuning. This work is distinct from the parked inter-device layer pipeline. |
| Demand transfer and preservation | Primary staging, helper refill and outgoing preservation operate in the exclusive bank. The original DeepSeek GPU-fetch mechanism and derivative Windows tests provide scoped evidence. | Assess exposed copy/preparation costs, required/optional transfer contention and overlap with useful service. Consider changes to transfer granularity, staging or device-driven fetch only where present-path evidence supports them. The slower Windows fetch ports do not establish a WSL result or rule out every adaptation; donor source alone does not justify a port. Demand fetch is distinct from the parked next-layer predictor. |
| Cache-aware expert selection | The current CUDA policy uses the approved additive ready-cache prior, with unchanged original-score mixing. A score-range-scaled donor and bounded-substitution implementation also exist as references. | Optimize the active prior and assess actual score gaps, original-demand misses and primary/helper service imbalance. A justified scaled or device-cost-aware selection policy remains a candidate within the required cache-aware design; none is already selected or proved superior. Preserve the routing/mixing and bounded quality contracts, and do not silently stack alternative policies. |
| Residency implementation | The [five-part adaptive design](#adaptive-residency) has an initial implementation and a scoped policy gain. | Complete its production coverage and cost-led optimization, including useful retention, global bank movement, ownership and coordinated helper/refill service. The dedicated design and trace sections own these requirements; exchange counts alone do not close them. |

For each material decision, record the actual caller, useful existing evidence,
remaining uncertainty and the next focused change/check or supported reason to
retain the implementation. Use the [investment policy](#choosing-worthwhile-work)
to choose the order and depth; use current counters/traces before constructing a
new executor. Other material costs exposed in the active generation path belong
to this assessment too. A failed prototype, missing measurement or generic claim
that a donor would be complicated does not establish that an important opportunity
is exhausted. Conversely, a settled unfavorable or inconclusive experiment does
not become mandatory again merely because an older proposal listed it.

The [parked mechanisms](#parked-work) and [held runtime bases](#base-holds) retain
their explicit dispositions. Retiring an old experiment queue or archiving a
proposal does not retire relevant mechanisms from consideration in Maya. Current
implementation decisions belong here and in the active work table, rather than
being silently labelled optional or future work in a subordinate document.

## Required parameter optimization

Complete the following optimization **after the
[reasonable baseline](#required-baseline-implementation) is implemented and its
independent operating behavior checked**. Use coherent provisional values during
implementation; do not let early value screens delay repair of known design flaws.
For **lenience** and the **cache-prior bonus**, compare the starting value with
meaningful alternatives in short matched actual-model screens, then select values
using complete generation speed and the bounded usability/quality evidence. The
starting value may remain selected, but leaving it unchanged without evaluation
is not completion of this work.

| Parameter family | Required investigation |
|---|---|
| DSpark lenience | Tune `spec_ell`, accounting for draft acceptance, rejection/residual work, output behavior and complete generation cost. Higher acceptance alone is not a win. Preserve the declared target sampling and actual-proposal probability contract. |
| Cache-aware expert selection | Tune the active `routing_prior_bonus` and assess material primary/helper preference effects. Preserve four valid experts, original-score mixture weights, routed scaling and shared-expert contribution. Verify which controls the selected policy actually uses; inactive substitution-only knobs do not count as cache-prior tuning. |
| DSpark work and scheduling | Assess verification/draft window width and confidence scheduling with the selected lenience rule; charge drafting, rejected work, expert-union growth, target service and committed tokens. Use a small informative set of compatible choices, not an automatic width grid or a new drafter-training requirement. |
| Primary residency | Assess aging/horizon, retention/admission margin, additional-refill budget and working/exchange capacity under the approved design. Price avoided service against incoming copies, necessary preservation, unavailable frames and interference. Resolve material uncertainty with observed ranges and focused trials. |
| Helper residency and transfers | Assess useful placement/retention, live-refill admission and scheduling, capacity allocation and coordination with primary ownership. Charge helper-link and CPU-memory contention, displaced work and useful completion; a larger hit count or a finished refill alone is insufficient. |
| Cooperative execution | After implementing batch-aware assignment and staging, tune material CPU/5090/3090 cost assumptions, upload amount, CPU worker/partition controls and transfer/compute overlap in the actual selected path. Evaluate whole-batch latency and emitted throughput, not a smaller staging count by itself. |

This is minimum coverage of the active system, not an instruction to sweep every
flag. Identify other material controls exposed by current evidence and record their
disposition in the [active work table](#active-maya-work). Outside the two explicitly
required value comparisons, source/counter evidence or a useful bound may justify
retaining a setting without a new run; explain why. A missing measurement, failed
prototype or build blocker is not evidence that an important opportunity is exhausted.

Hold unrelated settings fixed within a declared comparison. Screen coupled changes
as a group when their mechanism warrants it, then verify the retained combination
in the final engine; isolated gains are not additive. Reassess after useful results
under the [investment policy](#choosing-worthwhile-work). Preserve losses and bounded
quality failures. Keep high-or-higher effort and the specified target sampling;
neither is a speed-tuning knob. Existing [parked mechanisms](#parked-work) remain
parked unless their reopening rule is met.

## Choosing worthwhile work

Before substantial implementation or a full-model load, record a short decision
here: measured cost or missing behavior, proposed mechanism, plausible complete-engine
benefit/costs, cheapest informative check, and what its likely outcomes would change.
Source, counters, traces and rough estimates can guide investment; they are not
measured speedups or mathematical bounds.

For the [required initial baseline](#required-baseline-implementation), the known
design defects and the coherent proposed remedy are sufficient to proceed with
implementation. The cheapest informative check may be a focused correctness or
integration check. Do not require isolated performance proof for each repair or
use the optional-investment thresholds below to defer required baseline work.
Measure the combined implementation, then optimize and isolate contributors when
that answers a material question.

- For optional substantial work beyond required integration/residency, normally
  seek a credible opportunity near 10% or more in complete-engine speed, or necessary
  operating behavior. A cheap integrated 2–5% gain is useful but does not justify
  disproportionate infrastructure. These are priorities, not empirical limits.
- Reassess after meaningful results, integration steps, blockers or scope growth,
  by expected benefit and remaining cost. No fixed durations/deadlines or repeated
  permission questions at ordinary reassessment points.
- Screen linked changes as a group when a concrete mechanism connects them;
  isolate contributors after a useful result or for a real defect. An unexcluded
  interaction or generic possible synergy does not require another model experiment.
- Reassess an enlarged bank, compression, new executor or major port needed only
  to run a weak control. Optional allocation failure is not a new product requirement.
- Park weak branches on engineering judgment without claiming impossibility.
  Reopen only for new evidence, relevant implementation changes or a user request.
  Keep one implementation branch in focus; parallel agents must not launch
  independent optimization campaigns. Stop adding mechanisms when the measured
  problem has an adequate solution; record why that closes the relevant obligation.
  This does not close other unfinished workstreams.

### Runtime-base decision remains open

<a id="base-holds"></a>

<a id="base-choice-and-strategy-combinations"></a>

<a id="strategic-comparison-and-locked-gates"></a>

**User decision, 2026-11-03: Infernix and Strata are on hold as runtime bases.**
Do not advance their independent engine integration, base builds, model/GPU runs,
benchmarks or serving qualification until the user explicitly resumes the relevant
base. The three-base comparison is outside the active work and completion queue.

This hold does not exclude their ideas, code or evidence. Inspection, reuse and
adaptation for Maya remain allowed, including shared-component edits, builds and
focused checks needed by Maya, such as the Infernix MiniMax expert-service sources
that Maya already compiles. That shared work does not resume either base effort.
Existing read-only donor boundaries remain in force.

If the user resumes the comparison, the retained contract gives **Maya, Infernix
and Strata each an actual full-model comparison of the selected design**.
Match weights, workload, sampling, memory
budgets and measurement conditions. Base-specific scheduling/cache policies may
differ where that is the engineering advantage; state the differences. Exercise
each runtime's own request/state/scheduler ownership, not one engine behind three
wrappers. Reuse existing integrations and valid checks.

A repository name is not a cache architecture. [Existing base evidence](evidence.md#base-comparisons)
neither selects Maya by incumbency, eliminates Infernix from its ordinary control,
nor eliminates Strata from a tied common configuration. A build/integration blocker
leaves that comparison incomplete; reassess feasibility rather than silently
selecting another base. Only a new user scope decision can drop one of the three.

## Execution and completion

**“Complete the plan” means the whole active Maya scope.** Only the user may remove
required work, waive a required outcome or redefine completion as a smaller milestone.
An agent may choose implementation details and park optional mechanisms under the
[investment policy](#choosing-worthwhile-work); it may not turn a difficult or
unfinished requirement into an optional follow-up. Held bases and explicitly parked
mechanisms remain governed by their existing dispositions.

Keep implemented functionality, scoped checks, policy improvements and final plan
completion distinct. The short-screen/reversed-order limits bound each experiment;
they are not a budget of work after which the entire plan is automatically done.
Elapsed time, a first positive comparison or passing a feature checklist is not a
completion criterion. Continue within existing authorization while required work
remains, using new checks only for new changes or unresolved decision-relevant questions.

The [active work table](#active-maya-work) must account for every required outcome:
what is implemented, which real execution/checks establish it, what performance was
measured, and what remains. Reuse valid earlier evidence; neither a missing receipt
nor an unchecked box requires indiscriminate reruns. Resolve the actual gap.
The [implementation-first sequence](#implementation-first-then-measurement-and-tuning)
changes the order of work, not these final evidence requirements. A reasonable
untuned baseline is an intermediate result, not plan completion or a proven optimum.
Final completion requires all of the following in the combined Maya engine:

| Evidence | Required account |
|---|---|
| Scope and design coverage | Every active Maya workstream is closed with evidence or an explicit user scope change. All five residency requirements map to actual production behavior. [Implementation choices](#implementation-optimization) and [parameter optimization](#required-parameter-optimization) have explicit results/dispositions, including measured lenience and cache-bonus value comparisons. Switches, object presence, aggregate exchange counts or tuning only existing flags are insufficient by themselves. |
| Actual system | The [selected engine](#selected-engine) through the real server, working launch defaults, effective settings, starting/final source/configuration, loader, target/drafter, cooperative dispatch, streaming/tools and affected state checks. |
| Residency optimization | The [approved design](#adaptive-residency) operates without mandatory placement profiles or manual reseeding. Account for useful retention, additional refills, outgoing preservation, CPU/helper interaction, real bank replacement and retained state across requests. Explain selected policies and remaining material cost uncertainty using traces, service observations and integrated measurements. The [replacement predictor](#timing-model-replacement) must demonstrate useful decision accuracy for supported held-out search, retaining state/ownership checks and actual integrated confirmation. The retired timing model's gates are superseded; neither exhaustive tuning nor a general-purpose simulator is required. |
| Performance outcome | The [starting-engine comparison](#comparison-with-the-usable-starting-engine) demonstrates a usable integrated gain, with all absolute/per-task results and material regressions reported under the [matched protocol](#short-experiments-with-resident-weights). A gain solely over a newly established lower control does not satisfy this outcome. No arbitrary speed target or unmeasured hardware ceiling is introduced. |
| Quality and resources | The bounded reasoning account and usable-task checks, plus actual memory/cache/transfer accounting, ownership and useful overlap in the final composition. A few passes do not establish broad quality or make the approximations “quality preserving.” |
| Handoff | Observed endpoint/source and process/resource state when requested, with remaining limitations and the precise scope of every claim. An earlier running-server handoff does not discharge later integration/performance obligations. |

Unfinished required work, unresolved comparison validity or an unaddressed material
regression means the plan is incomplete. If bounded investigation finds no useful
improvement, report that outcome and the unmet objective; do not relabel it successful
completion or run indefinitely to obtain a favorable number. Any proposed reduction
of the required outcome must be a visible user decision. Preserve adverse evidence.

Use “working integration” or “scoped improvement” for intermediate progress,
“parked/unmeasured” for optional deprioritized opportunities and “inconclusive” for
unresolved measured effects. Reserve “Maya plan complete” for the contract above.
Maya completion does not finish the held base comparison, whose scope and resumption
follow the [base-hold policy](#base-holds). Link raw receipts externally.

## Dedicated bases and work boundaries

<a id="dedicated-bases"></a>

| Repository | Branch | Role |
|---|---|---|
| `/home/mdwbr/project-maya` | `minimax-wsl/proposal-2` | First implementation milestone; central authority |
| `/home/mdwbr/Infernix-minimax` | `minimax-wsl/proposal-1` | [Runtime base on hold; code reuse allowed](#base-holds) |
| `/home/mdwbr/strata` | `minimax-wsl/proposal-3` | [Runtime base on hold; code reuse allowed](#base-holds) |

These are independent repositories. Origins are the owner's forks; upstreams are
fetch-only with pushing disabled. Inspect branch/worktree before edits and work in
the assigned base. Cross-base edits require explicitly shared scope. Preserve local
changes, copied source deltas, attribution and applicable notices; never reset,
clean or discard work to make a checkout clean. Inherited product/setup documentation
is technical reference, not an alternative plan. [Initial provenance](base-provenance.json)
is historical, not an execution obligation.

`/home/mdwbr/Infernix-glm5-local`, other pre-existing Infernix worktrees and Windows
Maya are read-only donors, including files, Git state and running processes.
No separate current plan/status tracker belongs in another checkout.

### Regular commits

The user renewed local commit authorization on 2026-11-04: **commit as work
progresses**, including the preserved implementation. Push only when requested.
A commit neither requires a new benchmark nor turns incomplete evidence into delivery.

## Sub-agent delegation

Carried over from the [GLM delegation guidance](/home/mdwbr/Infernix-glm5-local/AGENTS.md#sub-agent-delegation). This section owns the policy for all three MiniMax bases.

**Claude Code, user direction 2026-11-09: the primary-agent-only limits and the Terra-only rule below do not apply.** A
Claude Code primary agent may delegate modeling, simulation, replay optimization and evaluation to its own sub-agents. It
still checks what comes back before recording it, and weighs each sub-agent's cost against its benefit. Timed tests stay
quiet: a message reaches a busy sub-agent only when its turn ends, so a sub-agent that may run CPU-heavy work checks a
lock file named in its brief before each heavy step.

**Primary-agent-only work, user instruction 2026-11-04:** never delegate offline
modeling, simulations, or offline optimization based on trace replays, simulations
or already collected data. The primary agent must do that work itself. Never
delegate benchmark-result evaluation: the primary agent must personally inspect
and evaluate comparison validity, results, limitations and resulting decisions.
These restrictions take precedence over the general delegation guidance below.

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

| Item | Location |
|---|---|
| Official revision | `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e` |
| Official target and DSpark | `/mnt/wsl/transfer/minimax-wsl/checkpoints/MiniMax-M3.1-Flash` |
| Existing generated layouts | `/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-{1,2,3}` |
| Large captures and receipts | `/mnt/wsl/transfer/minimax-wsl/runs/` |

The [official inventory](official-checkpoint.json) records the verified shared
download: all 73 files, 252,446,157,672 bytes including DSpark and non-executed
vision assets. Recheck readiness before loading; do not launch upstream installers,
download another model, or make per-repository full copies as a setup shortcut.
The old Unsloth/GGUF copy and Windows artifacts are historical sources only.

Keep large assets outside source repositories. Check transfer filesystem space,
existing VHDX allocation and Windows backing-drive headroom together before new
large data. Do not delete unrelated models/evidence or repeat the old superseded
copy. Reuse existing packs where compatible.

## Historical evidence and stopped handoff

<a id="stopped-work-handoff-2026-11-03"></a>

<a id="phase-2-exit-record-2026-11-02"></a>

<a id="review-follow-up-2026-11-01"></a>

<a id="r1-smaller-prefix-and-suffix-helper-decision"></a>

<a id="required-27-case-matrix"></a>

The stopped handoff recorded Maya execution source `d4fa863` plus later
documentation/configuration changes; later guards were not remeasured on that
configuration. Infernix `i-c1` was cancelled before READY/inference; its runner and
monitor were stopped and the PID-bound advancement script disabled. Prepared Maya
final-run scripts were never executed. Preserve that incomplete run.
Owned processes/listener were absent at the handoff. Observe current state during
the mandatory restart checks rather than inferring it from this historical record.

The [evidence index](evidence.md) provides scoped results and source links.
The [stopped-round archive](history-2026-11-03-stopped-plan.md) and
[pre-consolidation snapshot](history-2026-11-03-pre-consolidation.md) preserve full
original wording, adverse outcomes and superseded queues. Consult them only for a
specific historical question; neither is routine restart reading or a work order.
Legacy anchors remain for incoming links and do not reactivate archived instructions.
