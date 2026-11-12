# CLAUDE.md

## Central optimization objective: read first, keep in mind throughout

**Getting the CPU lane and the copy-plus-primary path to finish together on each individual pass — every activation through
every individual layer — is the central objective that drives optimization.** The user has stated this at least three times,
and agents have repeatedly lost it. What is minimised is each pass's finish time, summed over passes. A pass's completion
depends on when its operations start, which run together, which depend on others and which one ultimately delays progress;
separate averages lose those relationships. Judge every change by what it does to individual passes, then aggregate.

**If start times differ, compare completion times measured from a common origin, never durations.**

Quantify it per pass, then report by layer:

- Mean abs(Δ) between completion times of the CPU path and the primary GPU transfer path.
- Mean abs(Δ) between the helper GPU transfer path and the CPU path.
- Mean abs(Δ) between the primary GPU transfer path and the helper GPU transfer path.
- Each of these against the helper GPU resident compute path as well.
- Mean abs(Δ) between the slowest and the second-slowest path of each individual pass.

The plan's [per-pass balance section](docs/minimax-wsl/README.md#per-pass-balance) records the evidence and the open checks.

## Decisions that stand

The user's decisions, in their words. They have been lost and reversed before ("It is interesting how we keep
regressing in this project", 2026-11-11). **Before any test, default or design that would reopen one of these, stop and
say so.** A decision is carried out only when serving's own files carry it: `docs/minimax-wsl/native-server.json` (how
the engine is started) and `docs/minimax-wsl/native-runtime.json` (what a request gets when it sets nothing). The test
launcher and the test profiles (`tools/mm3/routing_lab/`) are not serving.

| Decision | The user's words | Carried by |
|---|---|---|
| Weights stay resident; reload only what must be reloaded | "One of the rules of this project has always been to minimize loading time by keeping weights resident when possible and reloading only what we need to reload" (2026-11-11) | The rule under "Rules agents have lost before"; `tools/mm3/native_resident.py` |
| The drafter is DFlash2 | "we replaced DSpark with DFlash2 awhile ago, then worked to further optimize Dflash2 (quantization, predictive draft length, etc)" (2026-11-11; first directed 2026-11-09) | `native-server.json` starts `--dflash2` with planner v1's table (since 2026-11-12); the test launcher |
| The 5090 computes experts with native FP4; speed over consistency | "We optimized the speed on the 5090 by deliberately moving from the initial slow bf16 arithmetic to native fp4. Then we established rules about speed over consistancy" (2026-11-11) | `--native-primary-arithmetic fp4`, `primary_arithmetic` fp4. Do not test other arithmetic on the 5090 again; no work on bitwise repeatability |
| Hottest experts on the 3090, the next tier on the 4080, the third on the 5090 | "It should go: hottest experts on 3090, second tier of hottest experts on the 4080, third tier of the hottest experts on the 5090." (2026-11-11; hottest on the 3090 since 2026-11-08) | `placement_order` `htp` in `native-server.json`; the launcher's default |
| The prompt is read by the unmodified router | "We should not read the prompt under the bonus." (2026-11-11) | `routing_prompt_bonus` 0, `prompt_layerwise` 1 |
| Each device has its own kernel and arithmetic; no output identity between lanes | "We want each device to have its own custom kernel, designed, written, and optimized to be the fastest on that device." (2026-11-11) | `helper_kernel*`, `third_kernel*` in `native-runtime.json` |
| Experts are not quantized further | "Quantifying experts is not permitted for this project." (2026-11-10) | No setting; a rule |
| No native Linux boot | "Native Linux boot is not an option." (2026-11-10) | A rule |
| No new UIs for now | "No more UIs for now." (2026-11-11) | A rule |
| The default is the behaviour without flags, and nothing wrong or legacy is on by default | "Please make sure that nothing else wrong or legacy is enabled by default or runs by default. Default is the behavior without flags" (2026-11-12) | The engine's and the server's built-in values (`src/mm3/mm3_model.hpp`, `mm3_sample.hpp`, `serve/server.py`). **In progress:** the built-in values were older than serving's files on 2026-11-12; see the plan |

When the user decides something new, add its row here the same day, with where it is carried. If serving's files
do not carry it yet, say so in the row. `tools/mm3/check_serving_drift.py` lists what serving's files and the test
launcher and profiles differ in; run it before quoting a test result as a statement about serving.

## At every restart

1. Read [AGENTS.md](AGENTS.md), then the [central plan](docs/minimax-wsl/README.md): first
   [where the work stands](docs/minimax-wsl/README.md#where-the-work-stands), then the
   [mandatory restart order](docs/minimax-wsl/README.md#mandatory-restart-order) and the
   [rules for adoption](docs/minimax-wsl/README.md#adoption-rules). The plan alone owns requirements, decisions and
   status. Update it when a decision changes, and keep no parallel tracker.
2. Work on branch `minimax-wsl/shared-model-diagnostic` in this worktree (user direction). AGENTS.md's "This checkout"
   names the main checkout and its branch; that does not apply to this work.
3. Before running a test, a screen or a tuning, find it in the plan's current order. If the plan defers it or puts
   something else first, do not run it. Most of the user's corrections on 2026-11-09 came from skipping this check.

## The user's latest direction wins

The user's requirements change during the work. When an instruction conflicts with older plan text or an older
instruction, the most recent one wins. Record the change in the plan, at the place the older text sits.

## Rules agents have lost before

Each of these is a user correction.
- **Output varies by design.** The same expert runs with different arithmetic on different devices, and routing follows
  the cache state. Text that differs between runs or between arms is expected. It is never a finding and never a
  validity check. Exactness is checked inside the engine.
- **The speculative-decoding planner is its own planner.** Its terms are fitted and judged on verify windows, never held
  to `T=1`'s values. A `T=1` result is a reference, not a constraint, and a window-specific parameter is not "added
  complexity".
- **The CPU price is being replaced.** A planner priced by each pass's completion time from the measured GB/s takes its
  place, at `T=1` and in windows. Until that rebuild, do not tune or screen the CPU price, the horizon, the refill budget
  or the staging frames. The plan's deferral list names the rest.
- **Steps start where the plan says.** Step 6 starts with the design comparison of the six timing proposals. A model is
  not extended ad hoc from another regime.
- **Adoption.** The burden of proof is on harm. A change with no quality mechanism needs no usability check.
  Token-budget cut-offs are never failures. A sub-2% gain is adopted if it also simplifies. Every timed test keeps its
  own baseline in the same load, measures from token 128, and is judged per prompt and by its mechanism.
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

## Sub-agents in Claude Code

AGENTS.md's limits on delegating modeling, simulation and evaluation, and the plan's Terra-only rule, do not apply to
Claude Code (user direction, 2026-11-09). Delegate when it helps, and check what comes back before recording it.

Keep timed tests quiet. No CPU-heavy work (builds, replays, sweeps, file scans) runs while a timed request runs. A
message reaches a busy sub-agent only when its turn ends. So the brief of any sub-agent that may run heavy work names a
lock file, such as `$CLAUDE_JOB_DIR/tmp/quiet.lock`, that it checks before each heavy step. Create that file for the
campaign's duration.

## Hardware safety

- GPU power limits: never above 425 W (5090) or 390 W (3090). Never restore the default full power. After every reboot,
  run `native_host_control.py restore-power-limit all`.
- Restore `vm-affinity all` after any affinity experiment.

## Git

- Commit regularly and push this branch. Never push to main or master, never force-push, never merge.
- Commit messages end with the session's `Co-Authored-By` and `Claude-Session` lines.
- Push with `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push -q origin minimax-wsl/shared-model-diagnostic`.
