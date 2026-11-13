#!/usr/bin/env python3
"""What serving runs against what the tests run: every difference, in one listing.

    python3 tools/mm3/check_serving_drift.py            # prints the differences; exit 1 if there are any
    python3 tools/mm3/check_serving_drift.py --quiet    # exit code only

Serving is docs/minimax-wsl/native-server.json (how the engine is started) and native-runtime.json (what a request
gets when it sets nothing). The tests are the routing lab's launcher (tools/mm3/routing_lab/launch_lab.py,
engine_args) and its two profiles (profiles.json: one_row, drafting), which every timed test of 2026-11-11 used.
Written that day, after the user pointed out that decisions kept being reversed: the tests loaded DFlash2 and turned
six engine switches on while serving started DSpark with them off, and nothing showed it.

A difference is not always a fault (the profiles choose a mode: one row, or drafting with exact verification), but
every one of them is a way a measured result can fail to describe serving. Standard library only; no engine.

Since 2026-11-12 serving's files hold almost nothing (the engine's built-in values are serving's), so both sides are
compared as EFFECTIVE values: the engine's built-ins (tools/mm3/engine_builtins.py, read from the sources) overlaid
with serving's files on one side and with the launcher's arguments and a profile's keys on the other. A request setting
is compared in the profile's own mode (one row: spec 1; drafting: spec above 1), as the engine resolves it.

The server's own settings are compared the same way (since the second change of 2026-11-12: native-server.json no longer
names the reasoning level, the thinking budget, the loop guard or the sampling - the server's built-ins for this engine,
engine_builtins.server_builtins): serving's built-ins overlaid with its file against the launcher's run config
(launch_lab.py's `config = {...}`), and the environment switches serving's file sets against the launcher's."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[2]
DOCS = W / "docs/minimax-wsl"
LAB = W / "tools/mm3/routing_lab"


def options(args: list) -> dict:
    """["--a", "1", "--b", "--c", "x"] -> {"--a": "1", "--b": True, "--c": "x"}"""
    out, i = {}, 0
    while i < len(args):
        if i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
            out[args[i]] = str(args[i + 1]); i += 2
        else:
            out[args[i]] = True; i += 1
    return out


def launcher_config() -> dict:
    """The run config launch_lab.py writes (its `config = {...}` dict): each key's constant value, "(computed)" for one
    that is not a constant, and the same for its "env" block's keys."""
    import ast
    tree = ast.parse((LAB / "launch_lab.py").read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and
                any(getattr(x, "id", "") == "config" for x in node.targets)):
            out = {}
            for k, v in zip(node.value.keys, node.value.values):
                if not isinstance(k, ast.Constant):
                    continue
                if k.value == "env" and isinstance(v, ast.Dict):
                    env = {}
                    for ek, ev in zip(v.keys, v.values):
                        if isinstance(ek, ast.Constant):
                            try:
                                env[ek.value] = ast.literal_eval(ev)
                            except ValueError:
                                env[ek.value] = "(computed)"
                    out["env"] = env
                    continue
                try:
                    out[k.value] = ast.literal_eval(v)
                except ValueError:
                    out[k.value] = "(computed)"
            if "args" in out:
                return out
    raise ValueError("tools/mm3/routing_lab/launch_lab.py: its run config (config = {...}) is no longer found")


def main() -> int:
    quiet = "--quiet" in sys.argv
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import engine_builtins                                  # (this folder: the engine's built-in values)
    server = json.loads((DOCS / "native-server.json").read_text())
    runtime = json.loads((DOCS / "native-runtime.json").read_text())
    profiles = json.loads((LAB / "profiles.json").read_text())
    spec = importlib.util.spec_from_file_location("launch_lab", LAB / "launch_lab.py")
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    start = engine_builtins.start_builtins()
    serving_start = {**start, **options(server["args"])}
    test_start = {**start, **options(lab.engine_args())}
    lines = []
    for key in sorted(set(serving_start) | set(test_start)):
        a, b = serving_start.get(key, "(not given)"), test_start.get(key, "(not given)")
        if a != b:
            lines.append(f"  start option {key}: serving {a}; tests {b}")
    order_s, order_t = server.get("placement_order", "htp"), "htp (the launcher's default)"   # (the server's built-in is htp)
    if not str(order_t).startswith(str(order_s)):
        lines.append(f"  placement order: serving {order_s}; tests {order_t}")
    for name in ("one_row", "drafting"):
        prof = {k: v for k, v in profiles[name].items() if not isinstance(v, (list, dict))}
        drafts = int(prof.get("spec", runtime.get("spec", 8))) > 1
        builtin = engine_builtins.request_builtins(drafts)
        serving = {**builtin, **runtime}
        if not drafts:                                       # (spec 1: the engine's one-row resolution of a built-in)
            serving.update({k: builtin[k] for k in ("spec_policy", "spec_plan", "exchange_schedule") if k not in runtime})
        for key in sorted(prof):
            if key == "fresh_context":
                continue                                     # (the profiles' own isolation of each prompt, not a setting)
            a, b = serving.get(key, "(not set: the engine's own default)"), prof[key]
            if a != b:
                lines.append(f"  request setting {key}, profile {name}: serving {a}; tests {b}")
    # the server's own settings: its built-ins for this engine overlaid with serving's file, against the launcher's
    server_side = {**engine_builtins.server_builtins(), "monitoring": True,
                   **{k: server[k] for k in ("reasoning_effort", "thinking_budget", "loop_guard", "sampling", "monitoring")
                      if k in server}}
    launcher = launcher_config()
    for key in ("reasoning_effort", "thinking_budget", "loop_guard", "monitoring", "sampling"):
        a, b = server_side.get(key, "(not set)"), launcher.get(key, server_side.get(key, "(not set)"))
        if isinstance(a, dict) and isinstance(b, dict):
            same = set(a) == set(b) and all(float(a[x]) == float(b[x]) for x in a)
        else:
            same = a == b
        if not same:
            lines.append(f"  server setting {key}: serving {a}; tests {b}")
    serving_env, test_env = server.get("env") or {}, launcher.get("env") or {}
    for key in sorted(set(serving_env) | set(test_env)):
        if key == "CUDA_VISIBLE_DEVICES":
            continue                                         # (the cards: the launcher adds the 4080's only for its runs)
        a, b = serving_env.get(key, "(not set)"), test_env.get(key, "(not set)")
        if a != b:
            lines.append(f"  environment switch {key}: serving {a}; tests {b}")
    if not quiet:
        if lines:
            print(f"{len(lines)} differences between serving and the tests:")
            print("\n".join(lines))
        else:
            print("serving and the tests agree")
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
