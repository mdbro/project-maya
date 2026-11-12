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
is compared in the profile's own mode (one row: spec 1; drafting: spec above 1), as the engine resolves it."""
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
    if not quiet:
        if lines:
            print(f"{len(lines)} differences between serving and the tests:")
            print("\n".join(lines))
        else:
            print("serving and the tests agree")
    return 1 if lines else 0


if __name__ == "__main__":
    raise SystemExit(main())
