# Upstream sync: synced through c5842c5

`minimax-wsl/local` is our MiniMax fork with upstream's work rebuilt on top of it. It was made on 2026-11-13 from
`e325603` (the head of `minimax-wsl/shared-model-diagnostic`). Upstream's net change `6ecbd3c..c5842c5` came in as
commits by area: diffs applied, no `git merge`.

| | Commit | What it is |
|---|---|---|
| Fork point | `6ecbd3c` | upstream v1.1.45 |
| Upstream taken | `c5842c5` | v1.1.61 (`f66a913`) plus a pull request template and a code of conduct: 764 files, +112,058 / −4,340 lines |
| Ours before | `e325603` | 494 commits after the fork point |

Upstream had moved on to `8b681ad` (v1.1.62) by the time of the sync; that is not in this branch.

The plan this follows is `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/upstream-map-merged.md`, and the
user's decisions are in `CLAUDE.md`'s decision table (the Upstream rows and the rows above them). The report of the
port, with build and test results, is `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/port-report.md`.

## What came in, commit by commit

| # | Commit | Upstream files | What ours adds in it |
|---:|---|---:|---|
| 1 | `c8f7cd2` The shared code `maya-mm3` compiles | 31 + `CMakeLists.txt` hunks 1, 22, 23, 28 | Five switch names |
| 2 | `d000856` The MiniMax engine (`src/mm3`) and its tests | 14 + hunks 4 (its `mm3_ablate_test` lines), 20, 21 | Nine conflicts resolved; refusal ablation built into the native path; the lookup's request checks and `lookup_control`; the drafter's room; two switch names |
| – | `4df1cf3` The owner-window CPU test's stub | none | `ablate()` stubbed in the test fixture (commit 2 broke its build) |
| 3 | `14c8bcc` The other engines and platforms | 431 + the remaining hunks | `cmake/MiniMaxBackend.cmake` builds the GLM four-row lane; one switch name |
| 4 | `fa80bf5` Tools, launcher and setup | 151 | Nothing |
| 5 | `e87d6a8` The server and web | 84 (one deleted) | 31 conflicts resolved (26 in `server.py`); the behaviour fixes below; 24 switch names |
| 6 | `8c64c8d` Docs and repository files | 52 | One conflict: our README banner first |
| 7 | `acd09bd` Ours: the prompt reader beside a swap in flight; exchange credit below 1 | none | `mm3_native_prompt.inc`, `mm3_native_demand_units.hpp` |
| 8 | `49a8517` Ours: the prompt lookup under lenient drafting | none | `mm3_main.cpp`, `mm3_sample.hpp`, token records |
| – | `332063d` The GPU check for ablation's default path; then this file | none | `tools/mm3/ablate_identity_check.py`; `docs/minimax-wsl/upstream-sync.md` |

Every commit message names what it keeps of ours over upstream and the `CLAUDE.md` row behind it.

## What was resolved, and how

Ours wins a conflict unless the plan names upstream's change as a fix we want. Section 1 of the plan gives every
conflict hunk by hunk. In short:

- **`src/mm3/mm3_main.cpp`.** The resolution keeps our request built-ins, token records, capture events, GPU nucleus,
  staged verify and planner. Upstream's prompt lookup is added around them, with upstream's `td` renamed `td_ms`. The
  keys `suffix_draft`, `lookup_trust` and `ablate` take only 0 or 1. The planner's adaptive offset skips lookup rounds.
- **`serve/server.py`.** It has 26 conflicts, and most are upstream's `STRATA_X` to `MAYA_X` rename meeting our
  read-once switches: upstream's line with `env_switches.get`. `env_switches.get` answers for both names.
- **The reasoning level and budget.** `checked_config_effort` gives a config without a level the model's own before
  the load ("auto" for MiniMax). After the load, `builtin_effort` replaces it with the engine's built-in, high.
  `builtin_think_budget` gives the budget: the config's, else the built-in 0 for mm3. `a.engine == "strata"` became
  `"maya"`, the value upstream renames it to.
- **Switch lists.** 32 names were added to both lists (`serve/env_switches.py`, `src/mm3/mm3_env_switches.hpp`), each
  in the commit whose code reads it.

Upstream changes that merged without a conflict but would have changed a standing decision, and what holds it:

| Upstream change | `CLAUDE.md` row | Held by |
|---|---|---|
| A config without a reasoning level gets "auto" before the load | Serving: high, no cap | `builtin_effort`, `builtin_think_budget` (`serve/server.py`) |
| Model management on by default | Serving: nothing else on; weights resident | `model_management_setting` defaults to off |
| `loop_guard.py` reads `os.environ`, which ours no longer imports | Serving: loop guard on | `env_switches.get` |
| New per-call `os.environ` reads; the prompt memo and turn trace copied in after the freeze | Serving: switches read once | `env_switches.get` everywhere; both names in `SERVER_ENV` |
| The drafter's room planned from DSpark's geometry | Placement: values agents chose stay | A flat 3 GB with `--dflash2` (`mm3_model.cu`) |
| Refusal ablation would miss our four native residual adds | Upstream: ablation built in | The projection in the expert graph, its verify, the owner window, the prompt reader; graphs keyed by the setting |
| The prompt lookup is skipped without a word under sampled drafts | Serving: refuse or warn; Upstream: the lookup's rule | A STAT warning; commit 8 runs it under lenience |
| Web answers run to their end (`c6884c3`) | Upstream: keep the 5-minute stop | `StreamJob.ORPHAN_S`, and `jobs_put` stops an unread running answer when the next starts |
| Turbo rewrites the serving config | Upstream: Turbo off for MiniMax | `Service.turbo_minimax`: hidden, and `POST /api/turbo` refused |
| Prompt preparation by increment, on | Upstream: on, timed in Phase 3 | Kept on; `strata_tune.prompt_memo` 0 / 1 per request |
| Answers saved in `answers/` beside the config | Upstream: take the saving | `.gitignore`: `docs/minimax-wsl/answers/` |

## How to repeat the sync

1. Fetch upstream (`git fetch upstream`). Pick the new upstream commit `U`. The last one taken is `c5842c5`.
2. List what upstream changed since: `git diff --name-status c5842c5 U`. Then list what both sides changed:
   `git diff --name-only <merge-base> HEAD` against that list.
3. Make the merge result to read conflicts from, without merging:
   `git merge-tree --write-tree HEAD U`, then `git show <tree>:<path>`.
4. Bring the files in by area, in the order above: the shared C++ the MiniMax engine builds; the MiniMax engine; the
   other engines; tools; server; docs.
   - A file only upstream changed: `git checkout U -- <paths>`.
   - A file both changed: take the merge result and resolve its conflicts. Ours wins unless upstream's change is a fix
     we want.
   - `CMakeLists.txt`: hunk by hunk. The tool is
     `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/merged-map-tools/apply_cmake_hunks.py`; its hunk numbers
     are those of `git diff -U0 <old upstream> U -- CMakeLists.txt`.
5. After each area, check four things:
   - The engine builds (`cmake --build <dir> --target maya-mm3 -j 26`; the quiet lock first).
   - The switch scan finds no unlisted name: `python3 -I merged-map-tools/scan_switches.py . .` (0 names).
   - `serve.test_env_switches` and `serve.test_server` pass, but for the failures known before.
   - Every clean merge that changes behaviour without flags is checked against `CLAUDE.md`'s table. The places to
     look first are the server's start (`main`), `MODEL_DEFAULTS`, `model_management_setting`, any `os.environ` read,
     and `mm3_model.cu`'s `plan_memory`.
6. Record the new sync point in the commit messages and update this file's first line and tables.
