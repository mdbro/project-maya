#!/usr/bin/env python3
"""The native MiniMax engine's built-in values, read from its sources: what an engine started without optional flags
and a request that sets nothing get (since 2026-11-12 these are serving's values; docs/minimax-wsl/native-runtime.json
is empty and native-server.json gives only paths and device choices). Standard library only; no engine.

    start_builtins()            {"--chunk": "16", "--vision": "off", ...}        the start options serving and the tests use
    request_builtins(drafts)    {"spec_policy": "lenience", "exchange_schedule": 0, ...} for a request that drafts or not
    server_builtins()           {"reasoning_effort": "high", "thinking_budget": 0, "loop_guard": True, "sampling": {...}}
                                the server's own for this engine, what a run config without those keys gets

Values are read with one pattern each from src/mm3/mm3_model.hpp, mm3_sample.hpp, mm3_residency_policy.hpp and
mm3_main.cpp, and the server's from serve/server.py (MM3_SERVER_DEFAULTS, the loop guard's default) and tools/maya_quants.py
(MiniMax's sampling); a pattern that no longer matches raises, so a renamed field cannot pass silently. Where a built-in depends
on the request's mode (spec_policy, spec_plan, exchange_schedule) the engine's own rule (mm3_main.cpp) is applied here."""
from __future__ import annotations

import re
from pathlib import Path

W = Path(__file__).resolve().parents[2]
SRC = W / "src/mm3"


def _read(name: str, pattern: str) -> str:
    m = re.search(pattern, (SRC / name).read_text())
    if not m:
        raise ValueError(f"{name}: the built-in value's pattern {pattern!r} no longer matches")
    return m.group(1)


def _num(text: str):
    text = text.rstrip("f")
    return float(text) if "." in text else int(text)


def start_builtins() -> dict:
    o = lambda field: _read("mm3_model.hpp", r"\b" + field + r"\s*=\s*([^;]+);")
    third = _num(o("int native_third"))
    return {
        "--max-context": str(_num(o("int max_ctx"))),
        "--chunk": str(_num(o("int chunk"))),
        "--staging": str(_num(o("int staging"))),
        "--native-helper": str(_num(o("int native_helper"))),
        "--native-helper-gb": f"{_num(o('double native_helper_gb')):g}",
        "--native-third": "auto" if third == -2 else str(third),
        "--native-third-gb": f"{_num(o('double native_third_gb')):g}",
        "--vision": "on" if o("bool vision") == "true" else "off",
        "--no-prefill-tune": o("bool tune") == "false",
        "--native-primary-arithmetic": o("std::string native_primary_arithmetic").strip('"'),
        "--native-helper-dispatch": o("std::string native_helper_dispatch").strip('"'),
        "--native-cooperative": "1" if o("bool native_cooperative") == "true" else "0",
        "--routing-floor": f"{_num(o('float routing_floor')):g}",                       # (0: the per-row budget off)
        "--routing-prior-bonus-one": f"{_num(o('float routing_prior_bonus_one')):g}",   # (-1: routing_prior_bonus)
    }


def request_builtins(drafts: bool) -> dict:
    s = lambda pattern: _read("mm3_sample.hpp", pattern)
    o = lambda field: _read("mm3_model.hpp", r"\b" + field + r"\s*=\s*([^;]+);")
    pol = lambda field: _read("mm3_residency_policy.hpp", r"\b" + field + r"\s*=\s*([^;,]+)[;,]")
    routing = "off" if o("bool routing") != "true" else ("prior" if o("bool routing_prior") == "true" else "substitute")
    exchange = _num(s(r"int exchange_schedule = (-?\d+);"))
    policy = s(r'std::string spec_policy = "(\w+)";')
    b = {
        "spec": 8 if drafts else 1,   # (server form: the verify window; DFlash2's block 8 when a drafter is loaded)
        "spec_policy": policy if drafts else "existing",
        "spec_ell": _num(s(r"float spec_ell = ([\d.]+f?);")),
        "spec_min_p": _num(_read("mm3_main.cpp", r"float dspark_conf = ([\d.]+f?);")),
        "spec_plan": _num(s(r"int spec_plan = (\d+);")) if drafts else 0,
        "spec_fixed_width": _num(s(r"int spec_fixed_width = (\d+);")),   # (the planner without a table)
        "serve_monitoring": int(_read("mm3_main.cpp", r'const bool monitoring_start = !monitoring_env \|\| std::string\(monitoring_env\) != "(0)";') == "0"),
        "verify_segment": _num(s(r"int verify_segment = (\d+);")),
        "routing": routing,
        "routing_ratio": _num(o("float routing_ratio")),
        "routing_prior_bonus": _num(o("float routing_prior_bonus")),
        "routing_prior_bonus_one": _num(o("float routing_prior_bonus_one")),   # (-1: routing_prior_bonus)
        "routing_floor": _num(o("float routing_floor")),                       # (0: the per-row budget off)
        "routing_prompt_bonus": _num(s(r"float prompt_bonus = (-?[\d.]+f?);")),
        "routing_dampen": int(o("bool routing_dampen") == "true"),
        "routing_prefetch": int(o("bool routing_prefetch") == "true"),
        "primary_arithmetic": o("std::string native_primary_arithmetic").strip('"'),
        "primary_adaptive": _num(s(r"int primary_adaptive = (-?\d+);")),
        "primary_policy": "legacy" if pol("bool legacy") == "true" else "cost",
        "primary_horizon": _num(pol("int horizon")),
        "primary_max_exchanges": _num(_read("mm3_residency_policy.hpp", r"max_exchanges = (\d+);")),
        "primary_service_ms": _num(_read("mm3_residency_policy.hpp", r"service_saved_ms = ([\d.]+);")),
        **{k: _num(_read("mm3_residency_policy.hpp", r"\b" + k[8:] + r" = ([\d.]+)")) for k in
           ("primary_h2d_ms", "primary_d2h_ms", "primary_interference_ms", "primary_margin_ms")},
        "helper_live": _num(s(r"int helper_live = (-?\d+);")),
        "helper_dispatch": o("std::string native_helper_dispatch").strip('"'),
        "cooperative_stage_reuse": int(o("bool cooperative_stage_reuse") == "true"),
        "cooperative_cpu_scale": _num(o("double cooperative_cpu_scale")),
        "primary_submit_query": int(o("bool primary_submit_query") == "true"),
        "helper_occupancy": _num(s(r"int helper_occupancy = (\d+);")),
        "helper_fast": _num(s(r"int helper_fast = (\d+);")),
        "helper_zero_copy": _num(s(r"int helper_zero_copy = (\d+);")),
        "third": _num(s(r"int third = (\d+);")),
        "early_copy": _num(s(r"int early_copy = (\d+);")),
        "prep_graph": _num(s(r"int prep_graph = (\d+);")),
        "expert_graph": _num(s(r"int expert_graph = (\d+);")),
        "dense_split": _num(s(r"int dense_split = (\d+);")),
        "gpu_nucleus": _num(s(r"int gpu_nucleus = (\d+);")),
        "prompt_layerwise": _num(s(r"int prompt_layerwise = (\d+);")),
        "exchange_schedule": exchange if exchange >= 0 else (0 if drafts else 1),
        "helper_max_exchanges": 1.0,   # (-1 in the request: 1.0 where it is applied, mm3_main.cpp)
        "exchange_host_wait": 2,       # (-1 in the request: 2 where it is applied)
        "stage_overlap": 0,            # (-1: the start setting, off without the stage prefix)
    }
    for key, pattern in (("helper_kernel", r"int helper_kernel = (\d+)"), ("helper_kernel_one", r"helper_kernel_one = (\d+)"),
                         ("third_kernel", r"third_kernel = (\d+)"), ("third_kernel_one", r"third_kernel_one = (\d+)"),
                         ("helper_kernel_loads", r"int helper_kernel_loads = (\d+)"), ("third_kernel_loads", r"third_kernel_loads = (\d+)")):
        b[key] = _num(s(pattern))
    return b


def server_builtins() -> dict:
    """The server's built-in values for the MiniMax engine (serve/server.py MM3_SERVER_DEFAULTS; the loop guard on unless a
    config or the environment turns it off; MiniMax's sampling from tools/maya_quants.py, with the engine's own top_k and
    min_p, which the server leaves to it) - what a run config without these keys runs with."""
    import ast
    server = (W / "serve/server.py").read_text()
    tree = ast.parse(server)
    found = [n.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(x, "id", "") == "MM3_SERVER_DEFAULTS" for x in n.targets)]
    if not found:
        raise ValueError("serve/server.py: MM3_SERVER_DEFAULTS no longer found")
    out = dict(ast.literal_eval(found[0]))
    if not re.search(r"\bself\.loop_guard = True\b", server):
        raise ValueError("serve/server.py: the loop guard's default (self.loop_guard = True) no longer found")
    out["loop_guard"] = True
    quants = ast.parse((W / "tools/maya_quants.py").read_text())
    md = [n.value for n in quants.body if isinstance(n, ast.Assign) and any(getattr(x, "id", "") == "MODEL_DEFAULTS" for x in n.targets)]
    if not md:
        raise ValueError("tools/maya_quants.py: MODEL_DEFAULTS no longer found")
    sampling = dict(ast.literal_eval(md[0])["minimax"][0])
    s = lambda pattern: _read("mm3_sample.hpp", pattern)
    sampling.setdefault("top_k", _num(s(r"int top_k = (\d+);")))
    sampling.setdefault("min_p", float(_num(s(r"min_p = ([\d.]+f?);"))))
    out["sampling"] = sampling
    return out


if __name__ == "__main__":
    import json
    print(json.dumps({"start": start_builtins(), "request_one_row": request_builtins(False), "request_drafting": request_builtins(True),
                      "server": server_builtins()}, indent=1))
