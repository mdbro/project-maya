# AGENTS.md

Read the [central MiniMax plan](docs/minimax-wsl/README.md) before work and follow its
[mandatory restart order](docs/minimax-wsl/README.md#mandatory-restart-order). It alone owns current
requirements, decisions, status and completion across the three bases. The user's
latest instructions take precedence; update that record when an accepted decision changes.

## This checkout

- Path: `/home/mdwbr/project-maya`; branch: `minimax-wsl/proposal-2`.
- Role: Maya implementation and home of the central MiniMax record.
- Source map: `src/mm3/` model/drafter; `src/ds4/` original helper; `tools/mm3/` packing; `serve/` frontend.
- `origin` is the owner's fork; `upstream` is fetch-only with pushing disabled.

Work in this assigned base; cross-base edits belong only to explicitly shared work.
The central [work boundaries](docs/minimax-wsl/README.md#dedicated-bases-and-work-boundaries) govern
preserving changes and read-only donors. Inherited engine/setup/GLM documentation
is technical reference, not another MiniMax plan or work queue.

## Work the primary agent must do

- **Never delegate offline modeling, simulations, or offline optimization based
  on trace replays, simulations, or already collected data to sub-agents.** The
  primary agent must perform this work itself.
- **Never delegate evaluation of benchmark results to sub-agents.** The primary
  agent must inspect and evaluate the results itself, including comparison
  validity, interpretation, limitations and resulting decisions.

These restrictions override general delegation guidance. The
[central delegation policy](docs/minimax-wsl/README.md#sub-agent-delegation)
records the same boundaries for all three MiniMax bases.

## Navigation

- [Technical source map](docs/minimax-wsl/proposal-2-cache-aware-routing.md)
- [Active Maya work and remaining scope](docs/minimax-wsl/README.md#active-maya-work)
- [Base holds and permitted code reuse](docs/minimax-wsl/README.md#base-holds)
- [Shared weights and data](docs/minimax-wsl/README.md#shared-weights-and-data)
- [Terra delegation policy](docs/minimax-wsl/README.md#sub-agent-delegation)
- [Commit/push policy](docs/minimax-wsl/README.md#regular-commits)
- [Scope and completion contract](docs/minimax-wsl/README.md#execution-and-completion)

Keep shared requirements in the central plan; do not create a parallel status tracker.
