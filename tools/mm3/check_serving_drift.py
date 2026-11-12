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
every one of them is a way a measured result can fail to describe serving. Standard library only; no engine."""
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
    server = json.loads((DOCS / "native-server.json").read_text())
    runtime = json.loads((DOCS / "native-runtime.json").read_text())
    profiles = json.loads((LAB / "profiles.json").read_text())
    spec = importlib.util.spec_from_file_location("launch_lab", LAB / "launch_lab.py")
    lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
    serving_start, test_start = options(server["args"]), options(lab.engine_args())
    lines = []
    for key in sorted(set(serving_start) | set(test_start)):
        a, b = serving_start.get(key, "(not given)"), test_start.get(key, "(not given)")
        if a != b:
            lines.append(f"  start option {key}: serving {a}; tests {b}")
    order_s, order_t = server.get("placement_order", "(none)"), "htp (the launcher's default)"
    if not str(order_t).startswith(str(order_s)):
        lines.append(f"  placement order: serving {order_s}; tests {order_t}")
    for name in ("one_row", "drafting"):
        prof = {k: v for k, v in profiles[name].items() if not isinstance(v, (list, dict))}
        for key in sorted(set(prof) | set(runtime)):
            a, b = runtime.get(key, "(not set: the engine's own default)"), prof.get(key, "(not set: serving's value applies)")
            if key not in prof:
                continue                       # serving's value reaches the test unchanged
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
