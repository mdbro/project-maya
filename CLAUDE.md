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
