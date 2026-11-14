# Upstream sync: synced through 6651b5b

`minimax-wsl/local` is our MiniMax fork with upstream's work rebuilt on top of it. It was made on 2026-11-13 from
`e325603` (the head of `minimax-wsl/shared-model-diagnostic`). Upstream's work comes in as commits by area, never
with `git merge`: the first sync took the net change `6ecbd3c..c5842c5`; the second took `c5842c5..6651b5b`
(v1.1.62 and v1.1.63) the same way.

| | Commit | What it is |
|---|---|---|
| Fork point | `6ecbd3c` | upstream v1.1.45 |
| Upstream taken, first sync | `c5842c5` | v1.1.61 (`f66a913`) plus a pull request template and a code of conduct: 764 files, +112,058 / −4,340 lines |
| Ours before the first sync | `e325603` | 494 commits after the fork point |
| Upstream taken, second sync | `6651b5b` | v1.1.63 (`upstream/main` on 2026-11-13): 58 commits since `c5842c5`, 44 files, +6,798 / −136 lines |
| Ours before the second sync | `31af1f7` | the first sync's commits and three `CLAUDE.md` commits |

The second sync is described in its own section below. What follows first is the first sync.

The plan the first sync follows is `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/upstream-map-merged.md`, and the
user's decisions are in `CLAUDE.md`'s decision table (the Upstream rows and the rows above them). The report of the
port, with build and test results, is `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/port-report.md`.

## The first sync: what came in, commit by commit

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

## The first sync: what was resolved, and how

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

## The second sync: `c5842c5..6651b5b` (v1.1.63)

The user asked for v1.1.63 to be reviewed by two agents and ported before the first engine start (`CLAUDE.md`,
Upstream rows). The two maps are
`/mnt/wsl/transfer/minimax-wsl/runs/local-sync-v1163-20261113/upstream-map-opus.md` (whose commit plan this follows)
and `upstream-map-fable.md` beside it. The report, with build and test results, is `port-report.md` in the same
folder.

| # | Commit | Upstream files | What ours adds in it |
|---:|---|---:|---|
| 1 | `1c99991` Ours: our CUDA- and WSL-only native files exempt from upstream's portability tests | none | `tools/fork_cuda_only.py` (12 files, each with its reason), read by `tools/check_hip_compat.py` and `tools/test_port_portability.py` |
| 2 | `a256558` The other engines and platforms (GLM, DeepSeek, AMD) and their build | 29 + `CMakeLists.txt`, `.gitignore` | Nothing |
| 3 | `025b1fc` The server (two-GPU modes for AMD + NVIDIA PCs, the engine-restart fix) | 6 | The import conflict resolved; ten switch names in both lists; a test that a MiniMax start takes no two-GPU path |
| 4 | `32bb793` Launcher and tools | 3 | Nothing |
| 5 | `0e55860` Docs and version (1.1.63) | 4 | Our README banner first |
| 6 | this file's commit | none | `docs/minimax-wsl/upstream-sync.md` |

What was resolved, and how:

- **One conflict: the imports of `serve/server.py`.** Our `env_switches` stays in the first line; upstream's
  `from serve import attach, mixed_gpus` and `from serve import glm_link` are taken. The other four files both sides
  changed (`CMakeLists.txt`, `.gitignore`, `README.md`, `serve/test_server.py`) merged cleanly. For all five, the result
  is our file plus exactly upstream's change.
- **The portability tests** (Upstream: "Exclude our native files"). `tools/fork_cuda_only.py` lists our 12 files that
  the checks flag, none of which upstream has. The exemption is in `tools/check_hip_compat.py` itself, because
  upstream's GitHub workflow runs that script on every push to the fork. Every other file, ours or upstream's, is still
  checked.
- **Switch lists** (Serving: switches read once). Ten names that `serve/glm_link.py` and `serve/mixed_gpus.py` name or
  set for other engines joined both lists. With the first sync's 32, 42 upstream names are listed. `maya-mm3` changed
  only through the header: sha256 `19d534a82a914e189187f8f9968cccd3ea948306bca94f625bddcd44f22fc45b`.
- **Nothing new turns on for MiniMax** (Serving: nothing else on by default). Upstream's two-GPU modes need an AMD
  (`"backend": "hip"`) DeepSeek or GLM config. `UpstreamSyncDecisions.test_a_minimax_start_takes_no_two_gpu_path` pins
  this.
- **The engine-restart fix, taken.** A replaced engine's late output no longer marks the new engine ended, so no
  needless engine start follows a restart (Serving: weights resident).

## How to repeat the sync

1. Fetch upstream (`git fetch upstream`). Pick the new upstream commit `U`. The last one taken is `6651b5b`.
2. List what upstream changed since: `git diff --name-status 6651b5b U`. The files both sides changed are those of
   that list for which `git diff --quiet 6651b5b HEAD -- <file>` reports a difference.
3. Make the merge result to read conflicts from, without merging, against the last upstream commit taken:
   `git merge-tree --write-tree --merge-base=6651b5b HEAD U`, then `git show <tree>:<path>`.
   - The `--merge-base` is needed. The syncs bring upstream in as diffs, never as a merge, so git's own merge base
     stays at the fork point `6ecbd3c`. Without the option, every file an earlier sync resolved shows as a conflict
     again (33 files for the second sync, of which one was real).
4. Bring the files in by area: the shared C++ the MiniMax engine builds; the MiniMax engine; the other engines; then
   the server, tools and launcher; docs.
   - A file only upstream changed: `git checkout U -- <paths>`.
   - A file both changed: take the merge result and resolve its conflicts. Ours wins unless upstream's change is a fix
     we want. Use the three-way merge for `serve/server.py`, not a plain patch: a plain patch rejects hunks whose
     context our lines changed.
   - The order of the Python commits: look at which way the imports between `serve/` and `maya.py` or `tools/` run in
     the new release. The first sync put tools before the server. The second put the server first, because
     `maya.py` imports `serve.glm_link`.
   - `CMakeLists.txt`: take the three-way merge result. Only when it conflicts, go hunk by hunk with
     `/mnt/wsl/transfer/minimax-wsl/runs/local-port-20261113/merged-map-tools/apply_cmake_hunks.py`; its hunk numbers
     are those of `git diff -U0 <old upstream> U -- CMakeLists.txt`.
5. After each area, check five things:
   - The engine builds (`cmake --build <dir> --target maya-mm3 -j 26`; the quiet lock first).
   - The switch scan finds no unlisted name: `python3 -I merged-map-tools/scan_switches.py . .` (0 names).
   - `serve.test_env_switches` and `serve.test_server` pass, but for the failures known before.
   - `tools.test_hip_compat` and `tools.test_port_portability` pass, and `python tools/check_hip_compat.py` exits 0. A
     path listed in `tools/fork_cuda_only.py` must be ours: `git cat-file -e U:<path>` must fail for each.
   - Every clean merge that changes behaviour without flags is checked against `CLAUDE.md`'s table. The places to
     look first are the server's start (`main`), `MODEL_DEFAULTS`, `model_management_setting`, any `os.environ` read,
     and `mm3_model.cu`'s `plan_memory`.
6. Record the new sync point in the commit messages and update this file's first line and tables.
