# CLAUDE.md

## Central optimization objective: read first, keep in mind throughout

**Getting the CPU lane and the copy-plus-primary path to finish together on each individual pass — every activation through
every individual layer — is the central objective that drives optimization.** The user has stated this at least three times,
and agents have repeatedly lost it. What is minimised is each pass's finish time, summed over passes. A pass's completion
depends on when its operations start, which run together, which depend on others and which one ultimately delays progress;
separate averages lose those relationships. Judge every change by what it does to individual passes, then aggregate.
The user confirmed it on 2026-11-12: judge every engine change per pass first, then confirm with token time.

**If start times differ, compare completion times measured from a common origin, never durations.**

Quantify it per pass, then report by layer:

- Mean abs(Δ) between completion times of the CPU path and the primary GPU transfer path.
- Mean abs(Δ) between the helper GPU transfer path and the CPU path.
- Mean abs(Δ) between the primary GPU transfer path and the helper GPU transfer path.
- Each of these against the helper GPU resident compute path as well.
- Mean abs(Δ) between the slowest and the second-slowest path of each individual pass.

The plan's [per-pass balance section](docs/minimax-wsl/README.md#per-pass-balance) records the evidence and the open checks.

## Decisions that stand

The user's standing decisions, in their words. **This table is the one decision authority** (the user, 2026-11-12). It is
read at every restart; the plan links to it and keeps the history and evidence, and no other file keeps a decision list.
Decisions have been lost and reversed before ("It is interesting how we keep regressing in this project", 2026-11-11).
**Before any test, default or design that would reopen one of these, stop and say so.**

- Quotes are the user's typed words, dated in Pacific time, as the plan is. A few older records date a late-evening
  message by its UTC day, one day later.
- "Chose" marks an option the user picked from a multiple-choice question on 2026-11-12: its label in quotes, then what
  the option said, quoted or in short form. Where the user typed their own words into an answer, the row says "typed"
  before them.
- "Carried by" names what carries the decision today, or "pending:" and the step that will. Phases and numbered items are
  those of the plan's [current order](docs/minimax-wsl/README.md#order-20261112). Engine files are in `src/mm3/`;
  `native-server.json` is `docs/minimax-wsl/native-server.json`; `README.md` is the plan, `docs/minimax-wsl/README.md`.

A decision is carried out only when the behaviour without flags carries it: the engine's and the server's built-in values.
`native-server.json` should hold only what depends on this computer (paths and device numbers). Today it also sets the
sampling (equal to the built-ins), the reasoning level, the reasoning budget, the loop guard and the 3090's budget; Phase 0
builds the last four in. `native-runtime.json` is empty. The test launcher and the test profiles (`tools/mm3/routing_lab/`)
are not serving.

| Area | The user's words (date) | Carried by |
|---|---|---|
| **Objective** | "getting the CPU lane and the copy-plus-primary path to finish together on each individual pass is the central objective. I have said this at least three times" (2026-11-07). Chose "Yes, keep it central": judge each engine change per pass first, then confirm with token time (2026-11-12) | The section above; `AGENTS.md:3-21`; pending: the 4080, placement and kernel results re-read per pass from existing captures (Phase 2) |
| | Prompt-reading speed (time to first token): chose "Yes, an objective": "Measure it on every serving check, with short, 4K-token and 30K-token prompts, and give it its own optimization work after the kernels" (2026-11-12) | pending: Phase 0 (the shared evaluator's serving check), Phase 3 (long prompts), item 8 |
| **Testing** | "We really should be benchmarking starting from token 128 to ensure cache is warm" (2026-11-08); "do not run sensitive benchmarks until it finishes" (a download, 2026-11-09) | `README.md:1930`; the quiet rule, `CLAUDE.md:177` |
| | Chose "Seeds + prompts + A/A": a seed per block shared by its arms, results per prompt with n = prompts × seeds, at least 5–6 prompts for drafting effects of 3–5%, an identical-arms check in each run. Chose "One shared evaluator": arms paired within a run, no pairing across runs unless engine starts alternate, "Also fix the three scripts" (2026-11-12) | pending: Phase 0 (the shared evaluator) |
| | Across engine starts, chose "Alternate + per-pass": starts A, B, A, B, each with a one-row anchor arm, judged mainly per pass and confirmed with token time against the measured start-to-start spread (2026-11-12) | pending: Phase 0 (the shared evaluator) |
| | "I do not want to go back to doing that lengthy fidelity sweep. Lets try to interpret what we have, a few isolated runs are ok here too." (2026-11-12). Chose "Only where a decision rides": re-measure inside the runs already planned, nothing re-run for its own sake, and the grades recorded in the plan (2026-11-12) | `README.md:143` (the grades); the order, `README.md:67`, has no sweep |
| | "If a change is faster, the burden of proof is to prove that it decreases usability … If there is no mechanism for loss of quality then no usability check is needed at all for integration into default." "If it's 0-2% faster and it improves in other ways (simpler/cleaner, chance it will make future work easier, etc) then it should be adopted." (2026-11-08) | `README.md:1902-1917`; `CLAUDE.md:147` |
| | For settings that change what the model writes (the bonus, lenience, the budget): chose "Tokens per second only", with no guard on time to a finished answer (2026-11-13; the plan's open item 11) | pending: Phase 0 (the shared evaluator's adoption rule) |
| | "Token budget cut-offs should not be counted as quality regressions!" (2026-11-08); "1,024-token cap is way too low. we already discussed this issue" (2026-11-11) | `docs/minimax-wsl/usability-matrix-template.json:41-53` (6,144 and 4,096 tokens); `README.md:1911` |
| **Serving** | "Please make sure that nothing else wrong or legacy is enabled by default or runs by default. Default is the behavior without flags" (2026-11-12) | The built-in values in `mm3_model.hpp`, `mm3_sample.hpp` and `serve/server.py` (200031d); pending: Phase 0 engine change for the four still in `native-server.json:11-12,34-36` |
| | "One of the rules of this project has always been to minimize loading time by keeping weights resisdent when possible and reloading only what we need to reload."; remember "run" and "engine start" (2026-11-11). More work on engine-start time: chose "Not now" (2026-11-12) | `serve/server.py:2655` (never unloads); `tools/mm3/native_resident.py`; `CLAUDE.md:155`. The lab launcher and older run scripts still start their own engine |
| | "It should be 32768 by default. I have never asked to decrease the context window." (2026-11-12); chose "Batch it" (2026-11-12) | pending: Phase 0 engine change (`mm3_model.hpp:34` is still 8192) |
| | "Reasoning must be evaluated at high effort or higher." (2026-11-03); chose "High, no cap": built in, with no reasoning cap (2026-11-12) | pending: Phase 0 engine change (today only `native-server.json:34-35`; the server's own budget default is 32,768, `serve/server.py:6923`) |
| | Loop guard: chose "Off in speed tests only": "On for serving; timed tests turn it off per run." (2026-11-12) | pending: Phase 0 engine change (`native-server.json:36` still turns it off) |
| | "We do need to avoid modifying recommended sampling parameters (like lowering top_p or temperature) just to artificially inflate token acceptance rates"; "keep its default top p=0.95 and temperature=1" (2026-10-31). Top-k and min-p: chose "Model card's values", looked up and brought to the user before anything changes (2026-11-12) | `mm3_sample.hpp:25-26` (temperature 1.0, top-p 0.95, top-k off, min-p 0). The model card gives only temperature 1.0 and top-p 0.95 (looked up 2026-11-13, `generation_config.json` of the checkpoint), so top-k off and min-p 0 match it and nothing changes |
| | Image encoder: chose "Keep it off": "Text only, as now. Record it as your decision." (2026-11-12) | `mm3_model.hpp:56` (`vision` false) |
| | "Why would we build in fallbacks? This will be custom built just for this pc" (2026-10-31). Chose "Refuse or warn": a planner table that does not match is refused at start, a missing one gives a warning and a measured fixed width, every other quiet fallback a warning line at start and in the request's stats (2026-11-12) | pending: Phase 0 engine change (today's quiet fallbacks: `mm3_main.cpp:984-997`) |
| | Environment switches: chose "Also refuse unknowns": every set `MM3_`, `MAYA_` or `STRATA_` switch printed at engine start and saved in each run's record, each read once at start, "plus refuse to start on an unknown switch with those prefixes" (2026-11-12) | pending: Phase 0 engine change |
| | Progress lines: chose "Measure in next run": "Time serving with it on and off in the same run. Turn it off by default if it costs anything measurable." (2026-11-12) | pending: Phase 3 |
| | "Include instructions to commit regularly in the plan." (2026-11-02); "Just stay on this branch and resume work" (2026-11-08); chose "Stay on the branch": "Serving keeps using the branch's build, and the main checkout is left as it is. Merging is your call later." (2026-11-12) | `native-server.json:2` (the branch's build); `CLAUDE.md:123` and `CLAUDE.md:190` |
| | The server's missing Python packages (`jsonschema` among them; any request with a JSON schema or strict tools got HTTP 400): chose "Install them": into the project's environment, in effect at the next server start (2026-11-13; the plan's open item 12) | pending: Phase 0 (the install), Phase 3 (the server start) |
| **Drafting** | "we replaced DSpark with DFlash2 awhile ago, then worked to further optimize Dflash2 (quantization, predictive draft length, etc)" (2026-11-11); first "I would start with dflash2 at MXFP8 and loop back around to DSpark later." (2026-11-09) | `native-server.json:13-16` (the drafter and its planner table; without them the engine drafts nothing) |
| | Chose "Yes, full fair test" and typed: "Fit a DSpark planner table or equation, then alternate DSpark and DFlash2, each with its own table or equation at lenience 0.4. Evaluate not just performance, but also consider future optimization potential, and theroetical future bottlenecks with each. for example, marakov head, and experimental things"; chose "Before drafter kernels" (2026-11-12) | pending: item 5 |
| | "We can probably quantize dflash2 to 8 bit (like fp8 or MXFP8 or something) without impacting acceptance significantly" (2026-11-09); "drafter and head can decrease in weight fidelity if we show no decrease in usibility." (2026-11-10); "mxfp8 vs nvfp4 with native fp4 on 5090" (2026-11-12) | `tools/mm3/pack_dflash2.py:2` (MXFP8, the pack serving loads); pending: kernel items 6 (the drafter, NVFP4) and 10 (the head) |
| | "switch to lenience-based verification that limits excessive draft confidence" (2026-11-03); "Why did you go from lenient to exact?" (2026-11-12). Chose "Lenient 0.4, measure it": keep it, time it against exact verification in the width sweep, and run a short usability check before any refit planner widens windows (2026-11-12; this replaced the same day's "Switch to exact", given when "exact" was read as bit-exact). On that check the user typed: "I do not want to spend 2 hours of GPU time on this check." | `mm3_sample.hpp:43,45` (lenience, 0.4); pending: Phase 1 (against exact), Phase 3 (the short check). The drafting test profile still verifies exactly (`tools/mm3/routing_lab/profiles.json`) |
| | "If we can predict # of accepted tokens for a step this should be integrated into the planner."; "we need accurate speculative decoding offline modeling, offline optimization of parameters, and an accurate planner." (2026-11-09); "we definately do need to optimize number of drafts." (2026-11-12). Chose "Sweep, refit, A/B" and typed: "see if you can move from a table to a fitted equation with variables. This is more robust to future changes, and can be used to predict optimize parameters mathematically." (2026-11-12) | `mm3_sample.hpp:77` (the planner on) with `native-server.json:15-16` (table v1, fitted 2026-11-09 from exact runs on two cards); pending: Phases 0 to 3 |
| | "The planner for speculative decoding is going to be different from the planner for T=1" (2026-11-09); "Once speculative decoding is added back in you also need to consider what previously evaluated and decided ideas/proposals/items make sense to repeat and re-evaluate with speculative decoding turned on." (2026-11-07) | `CLAUDE.md:140`; `README.md:3215` |
| **Kernels** | "We do **not** need equivalent arithmetic for the same experts if they are executed on the 5090, 3090 or cpu. We use whatever is fastest" (2026-10-31); "We want each device to have its own custom kernel, designed, written, and optimized to be the fastest on that device." (2026-11-11) | `mm3_sample.hpp:67,70` (each helper card's kernel) |
| | "We optimized the speed on the 5090 by deliberately moving from the initial slow bf16 arithmetic to native fp4. Then we established rules about speed over consistancy" (2026-11-11); "the non deterministic and system state dependent performance that is the intended design of this project" (2026-11-02) | `mm3_model.hpp:110` (FP4); the cuBLAS pin only when asked (`mm3_main.cpp:3052-3054`); `CLAUDE.md:137` |
| | Other arithmetic on the 5090: chose "For speed only": "The kernel overhaul may try other 5090 arithmetic only to make it faster, for example per job width. Never for fidelity or repeatability." (2026-11-13) | pending: kernel item 7 (the 5090's FP4 expert kernel) |
| | "Quantifying experts is not permitted for this project." "No decrease in weight fidelity." (2026-11-10) | A rule; no setting |
| | "all kernels that were changed need an overhaul, all kernels if changed or not need a review to look of improvements that Fable missed" (2026-11-12). Chose "This session + Opus agents" (one Opus agent per device, standalone and in-engine timing, frozen predictions) and "Time share × headroom" for the order (2026-11-12) | pending: item 6 (`README.md:113`) |
| | "I do agree that it would be a good idea in general for the 3090 to queue each layer's work ahead and release it with a word in shared memory" (2026-11-10); chose "Yes, build it first" (2026-11-12). Rows of a verify window sharing uploads: "agreed" (2026-11-10) | pending: kernel item 1; shared uploads: not built, placed with the bandwidth-aware planner (item 7) |
| | CPU lane: chose "Move it to #2": the lane's worker count (9 against today's) tested in the engine with the cause looked into, then the L3 prefetch re-bounded on drafting's CPU jobs and built if it pays (2026-11-12) | pending: kernel item 2 |
| | "keep the spinner." "I don't care about idle power use. Pick whatever is faster"; the GPU scheduling test "Worth two reboots … (we are likely to keep it)" (2026-11-07). "This does NOT need a usability check. Prep_graph should be added."; for the early copy, "If there is no mechanism for loss of quality then no usability check is needed at all for integration into default." (2026-11-08) | `mm3_sample.hpp:59` (spinner), `:72` (`early_copy`), `:74` (`prep_graph`); GPU scheduling kept on in Windows (`README.md:3401`) |
| | "You can now use the attached 4080."; "It should be the serving default." (2026-11-11) | `mm3_model.hpp:99-100` (the 4080 found automatically, 14.1 GB), `mm3_sample.hpp:71` (`third` 1) |
| | The 4080's link: chose "Measure first": a two-minute bandwidth test in the gap of the next batched engine restart, then a report of where the loss is and what would fix it (2026-11-12) | pending: Phase 3 |
| | "GPU power limit, lower or restore -> NEVER above my current 425 W (5090) and 390 W (3090)"; "do not revert those to the default full power" (2026-11-07). The 4080: chose "Keep 320 W" (2026-11-12) | `CLAUDE.md:184`; `tools/mm3/windows/maya-host-control.ps1:28-31` (5090, 3090); pending: the 4080 in that script and in `restore-power-limit` (`tools/mm3/native_host_control.py:36`) |
| **Placement** | "It should go: hottest experts on 3090, second tier of hottest experts on the 4080, third tier of the hottest experts on the 5090." (2026-11-11); first "The most freqently demanded experts first go on the 3090, then on the 5090." (2026-11-04) | `serve/server.py:2603` (`htp`), applied after each engine start only when `third_profile` is set (`native-server.json:22`) |
| | "I thought we decided the 3090 is static. … The reason it is static is that its PCIe is only PCIe 4 x4"; "It is important to evaluate static/ not static experts on the 3090." (2026-11-03). Chose "Time it warmed": "Adopt the winner as the built-in." (2026-11-12). Then chose "Back to swapping": swapping stays the built-in until a multi-turn test without snapshot restores settles it (2026-11-13) | `mm3_sample.hpp:61` (`helper_live` 1: the 3090 swaps experts). Timed warmed 2026-11-13, static against swapping, tokens per second (measured): drafting +0.1% [-3.9, +4.0], equal; one row +1.4% [+0.1, +2.6] (n 18; over the 3 snapshot pairs [-0.3, +3.1]); run `/mnt/wsl/transfer/minimax-wsl/runs/catchup-phase1-20261113/helper-static`. Static was built in by the agent (fb225ef) and reverted: every request restored the same warmed snapshot and the prompts matched the start-up profile, so the test could not measure what swapping is for, keeping the 3090's cache warm for the user's work across requests. pending: the multi-turn test (item 8, adaptive placement) |
| | "I want to just get CPU/5090/3090 cooperation, cache-aware routing, optimized fully adaptive residency, DSpark/Fast-HSD and uncompressed HOST storage working with Maya first."; "cooperation is a given. It was always a given." (2026-11-03); "Compressed storage decreased speed, that is Counterproductive." (2026-11-02) | `mm3_model.hpp:81` (cooperative; compressed storage only without it, `mm3_model.cu:618`), `:102,105` (cache-aware routing), `mm3_sample.hpp:97` (the 5090 adapts) |
| | "Admissions must not use cache aware hit demand. They must use original router demand." (2026-11-04) | `mm3_native_adaptive.inc:199`, `mm3_native_helper_live.inc:2` (code comments; not checked by a run) |
| | "CPU price scale we want 1.5"; "Change the horizon from 20 to 40."; "Decay, 0.95: keep it for now"; refill budget 8: "check before deciding"; "Margin 0 and upload cap 4: stable, no change … Revisit the cap with speculation, where it does bind." (2026-11-08) | `mm3_model.hpp:85` (1.5); `mm3_residency_policy.hpp:9` (horizon 40; refill budget 8, which beat 7, 9 and 10); `mm3_sample.hpp:90` (0.95); pending: the cap with drafting (item 7) |
| | "I thought "CPU price" parameter was going to be replaced by a bandwidth aware model?" (2026-11-09). Chose "Next, after kernels" and typed: "Read the results from the L2 cache work on the CPU. I think we found a different reason than "wsl2" for the lower CPU bandwidth than theoretical, and even a possible way to overcome it" (2026-11-12) | `CLAUDE.md:143`; pending: item 7 |
| | Values agents set without the user (chunk 16, staging 8, prefill tuning off, routing ratio 0.8, the helper exchange credit and wait, the 3090's budget): chose "Optimize them later": "Keep today's values until the bandwidth-aware planner and the fitted models exist, then optimize them together as continuous values. Make the 22 GB the built-in now" (2026-11-12); earlier "You keep using "round-ish" values" (2026-11-07) | pending: Phase 0 engine change (22 GB; `mm3_model.hpp:94` says 21), then item 7 |
| | "'…keep refills off demand copies.' - absolutely!" (2026-11-08); "May scheduled exchanges become the one-row default? Yes" (2026-11-10) | `mm3_main.cpp:989` (scheduled at one row, not yet in drafting); pending: drafting and the planner's charge for refills (item 7) |
| | Swap cadence: agreed with the agent's suggestion (scheduled exchanges in drafting first, then more frequent swaps) and changed the frequency: "At the opportune time: 1. Make scheduled exchanges work in drafting first. That cuts the cost at any cadence. 2. Then raise the 3090 to about one swap per 2 steps, and give the 4080 scheduled swaps at about every 4 steps. Maybe sure they do not overlap (3090 and 4080 should not swap during the same step). Note that these are per step, not per token during drafting." (2026-11-13). Asked what "not during the same step" means, typed: "2 steps at drafting and also 2 at one row for the 3090. Four steps at drafting and also 4 at one row for the 4080." Then chose "One card per step": each 3090 swap is copied within one step, steps go 3090, 4080, 3090, no swap; no step has both cards copying. The 4080's choice of experts: chose "Keep the tiers": it takes experts ranked just below the 3090's set, so the placement order holds while the work changes (a replay before building). On learning that a scheduled one-row swap lasts about five steps today: "That changes things" (2026-11-13), and asked what rate the two helper cards can take without slowing anything; the rate is open again until the user answers | pending, in this order: (1) scheduled exchanges in drafting (`mm3_main.cpp:1200` turns them off when drafting); (2) the credit counted per forward step instead of per five generated rows (`mm3_sample.hpp:108`): the 3090 about 0.5 a step, the 4080 about 0.25 a step and never on a step the 3090 swaps. The 4080's pool is filled once and never exchanged today, so its swaps are new engine work, and as built scheduled exchanges carry at most 0.26 a token on the 3090 until the host copies (item 3, step 3). Once built, these are the built-ins |
| | A swap in flight when the next prompt arrives turns off the layer-by-layer prompt reader (3.6 times slower reads, measured): chose "Reader runs alongside, in the port": the layer-by-layer reader runs while a swap is in flight (it already moves swaps forward), built in minimax-wsl/local before Phase 3 (2026-11-13) | pending: `mm3_native_prompt.inc:85` refuses the reader on `helper_live_a >= 0`; the 4080's swaps get the same rule |
| | Where a newly popular expert goes, the 3090 or the 5090 (today the 5090 takes it first, so "hottest on the 3090" holds only at start): chose "Decide with the planner": the bandwidth-aware planner (item 7) decides by the per-pass balance; today's behavior until then (2026-11-13) | `mm3_native_helper_live.inc:168` (the 3090 takes no expert the 5090 holds); pending: item 7 |
| | Adaptive placement: chose "After kernels, drafting": designed and tested after the kernel and drafting work, over multi-turn conversations (2026-11-12) | pending: item 8 |
| | "We can likely use one shared model, evaluated for each layer." (2026-11-06) | `README.md:2272` |
| **Routing** | "And you agree with how we are mixing the experts?" … "Approved." (2026-11-03): experts picked under the bonus, mixed by the router's own strengths | `mm3_model.hpp:102,105`; `README.md:4770-4779` |
| | Chose "0.30 dampened, timed first" and typed over it: "0.30 dampened with very tolerant budget to catch outliers (0.6+ or something), timed first … Ceiling: the old served rule (with modification) … Confirm it does not significantly decrease speed from 0.30 dampened alone with no decrease in divergence. Confirm the speed on today's serving … and adopt if it is faster." Typed next: "very tolerant budget. not very tolerant bonus". Chose "Yes, try several": budgets 0.5, 0.6 and 0.7 metered, the best timed (2026-11-12) | Today bonus 0.20 (`mm3_model.hpp:106`), no dampening (`:104`); pending: Phase 0 engine change (the per-row budget in the serving kernel), Phase 1 (0.20 against 0.30 dampened), Phase 3 (the budgets) |
| | One bonus for one-row decoding and another for drafting: chose "Yes, add it" and typed: "Make the engine change in the batched build, and probe one row at 0.40 dampened + very high bonus (0.6 or more to catch outliers) in the meter run" ("very high bonus" meant the per-row budget, confirmed the same evening) (2026-11-12) | pending: Phase 0 engine change; Phase 3 |
| | "We should not read the prompt under the bonus." (2026-11-11) | `mm3_sample.hpp:62` (`prompt_bonus` 0) |
| | Layer-by-layer prompt reader: chose "Keep, time long prompts" (2026-11-12, after being told wrongly that its fidelity sat at the meter's floor); then chose "Measure at batched start": "The reader stays the default meanwhile." (2026-11-12) | `mm3_sample.hpp:85` (`prompt_layerwise` 1); pending: Phase 0 (the meter adapter), Phase 3 (the meter run, long prompts) |
| | The meter's cuBLAS pin: chose "Test once, then decide": one arm repeated without it; "If repeats stay exact, leave it off; if not, the meter asks for the pin explicitly." (2026-11-12) | pending: Phase 3 |
| **Records** | "And we need one current and central authority" (2026-11-01). Chose "CLAUDE.md table": "One short table in your words, read at every restart, each row with where it is carried. The plan links to it, and keeps the history and evidence but no separate decision list." Chose "Rebuild from your words", with the diff shown before committing (2026-11-12) | This table; `README.md:20`; `AGENTS.md:23` |
| | "my requirements changed through the transcript: the most recent requirement over rule conflicting older ones" (2026-11-09) | `CLAUDE.md:128` |
| | "You left the y axis label off of every single plot"; "Are you even trying to make these readable to a human?" (2026-11-11); "I'm so tired of your incoherent cryptic phrases, and your drifting to differnt concepts and subjects as you go down each single row of a table" (2026-11-12) | `CLAUDE.md:163` |
| | "Use Opus subagents to speed this up" (2026-11-03); "That agents.md line about subagents does not apply to claude code" (2026-11-08) | `CLAUDE.md:171`; `AGENTS.md:56`; `README.md:5518` |
| | "Take ownership." (2026-11-04); "avoid agents drastically decreasing scope and declaring done before finishing the plan properly" (2026-11-03) | `CLAUDE.md:153`; `README.md:5442` |
| | "Native Linux boot is not an option." (2026-11-10); "No more UIs for now." (2026-11-11); "Put a hold on the infernix base and strata base (but not ideas or code that come out of those)." (2026-11-03) | Rules; `README.md:5415` (the base hold) |

Not decisions of the user's: values agents chose that are on by default with nothing from the user behind them. Among them
are the 8,192 context (being changed), chunk 16, staging 8, prefill tuning off, 8 drafts and no
drafting confidence floor (copied from a test profile on 2026-11-12), routing ratio 0.8 and the quiet fallbacks. Each stays as it is until a row above changes it, and none is cited as the user's choice.

When the user decides something new, add its row here the same day, with what carries it; if nothing carries it yet,
write "pending:" and the step that will. `tools/mm3/check_serving_drift.py` lists what serving's files and the test
launcher and profiles differ in; run it before quoting a test result as a statement about serving.

## At every restart

1. Read the decision table above, then [AGENTS.md](AGENTS.md), then the plan's
   [current order](docs/minimax-wsl/README.md#order-20261112), its
   [reliability grades](docs/minimax-wsl/README.md#speed-reliability-20261112), the
   [mandatory restart order](docs/minimax-wsl/README.md#mandatory-restart-order) and the
   [rules for adoption](docs/minimax-wsl/README.md#adoption-rules). The table above alone holds the user's decisions;
   the plan holds the order of work, status, history and evidence. Keep no other tracker.
2. Work on branch `minimax-wsl/shared-model-diagnostic` in this worktree (user direction). AGENTS.md's "This checkout"
   names the main checkout and its branch; that does not apply to this work.
3. Before running a test, a screen or a tuning, find it in the plan's current order. If the order defers it or puts
   something else first, do not run it. Most of the user's corrections on 2026-11-09 came from skipping this check.

## The user's latest direction wins

The user's requirements change during the work. When an instruction conflicts with older plan text or an older
instruction, the most recent one wins. Change its row in the table above, and record the change in the plan at the place
the older text sits.

## Rules agents have lost before

Each of these is a user correction.
- **Output varies by design.** The same expert runs with different arithmetic on different devices, and routing follows
  the cache state. Text that differs between runs or between arms is expected. It is never a finding and never a
  validity check. Exactness is checked inside the engine.
- **The speculative-decoding planner is its own planner.** "The planner for speculative decoding is going to be different
  from the planner for T=1" (2026-11-09). Its terms are fitted and judged on verify windows, never held to `T=1`'s values.
  A `T=1` result is a reference, not a constraint, and a window-specific parameter is not "added complexity".
- **The CPU price is being replaced.** A planner priced by each pass's completion time from the measured GB/s takes its
  place, at `T=1` and in windows; it comes right after the kernel overhaul (the user, 2026-11-12). Until that rebuild, do
  not tune or screen the CPU price, the horizon, the refill budget or the staging frames. The plan's deferral list names
  the rest.
- **Adoption.** "If a change is faster, the burden of proof is to prove that it decreases usability" (2026-11-08). A
  change with no quality mechanism needs no usability check. Token-budget cut-offs are never failures. "If it's 0-2%
  faster and it improves in other ways (simpler/cleaner, chance it will make future work easier, etc) then it should be
  adopted." Timed tests measure from token 128 (2026-11-08). Each block gets its own seed, shared by its arms; results
  are reported per prompt with n = prompts × seeds; arms are paired within one run; engine starts are compared only
  when they alternate A, B, A, B (the user's answers of 2026-11-12).
- **Own your mistakes.** When the user corrects something you did, say that you did it. Do not attribute it to "the
  plan" or to what "places assumed".
- **Keep the weights resident; reload only what has to be reloaded.** "One of the rules of this project has always been
  to minimize loading time by keeping weights resident when possible and reloading only what we need to reload" (the
  user, 2026-11-11, after an agent started the engine about fourteen times in a day at 350 s each). An engine that is
  up with the right binary and start options is used as it is: settings go in the request, placements are rewritten in
  place. It is stopped only when the next run needs another binary or start option, and runs are ordered so that those
  changes are few. An "engine start" is the launch until ready; a "run" is everything for one configuration. Say which.
  The tool: `tools/mm3/native_resident.py` (`status`, `start`, `needs-restart CONFIG`, `run RUN_DIR`, `stop`). Before a
  run, ask it whether the engine that is up is the one the run needs; a run script does not start an engine of its own.
- **Results shown to the user are readable.** "You left the y axis label off of every single plot"; "Are you even trying
  to make these readable to a human?" (2026-11-11); "I'm so tired of your incoherent cryptic phrases, and your drifting
  to differnt concepts and subjects as you go down each single row of a table" (2026-11-12). Label every axis with its
  quantity and unit. Keep one kind of thing in each table row and one kind of value in each column. Use plain words,
  not internal names, run ids or setting names (add a setting name in code font only where the reader needs to find
  it). Say whether each number is measured or modeled, and keep a comparison's baseline in view. Look at every chart as
  its reader before showing it.

## Sub-agents in Claude Code

AGENTS.md's limits on delegating modeling, simulation and evaluation, and the plan's Terra-only rule, do not apply to
Claude Code. The user: "That agents.md line about subagents does not apply to claude code" (2026-11-08), and "Use Opus
subagents to speed this up" (2026-11-03). Delegate when it helps, and check what comes back before recording it.

Keep timed tests quiet. No CPU-heavy work (builds, replays, sweeps, file scans) runs while a timed request runs. A
message reaches a busy sub-agent only when its turn ends. So the brief of any sub-agent that may run heavy work names a
lock file, such as `$CLAUDE_JOB_DIR/tmp/quiet.lock`, that it checks before each heavy step. Create that file for the
campaign's duration.

## Hardware safety

- GPU power limits: never above 425 W (5090), 390 W (3090) or 320 W (4080). The user: "NEVER above my current 425 W
  (5090) and 390 W (3090)" (2026-11-07); the 4080 stays at its default 320 W (2026-11-12). Never restore the default
  full power of the 5090 or the 3090. After every reboot, run `native_host_control.py restore-power-limit all`. It
  covers the 5090 and the 3090 today; the 4080 is to be added to it and to the host-control script (pending).
- Restore `vm-affinity all` after any affinity experiment.

## Git

- Commit regularly and push this branch. Never push to main or master, never force-push, never merge.
- Commit messages end with the session's `Co-Authored-By` and `Claude-Session` lines.
- Push with `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -q origin minimax-wsl/shared-model-diagnostic`.
