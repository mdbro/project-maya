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

## Everything else

Follow [AGENTS.md](AGENTS.md): it points to the [central MiniMax plan](docs/minimax-wsl/README.md), which owns requirements,
decisions and status, and it states the work the primary agent must not delegate.
