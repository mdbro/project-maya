#!/usr/bin/env python3
"""Fidelity sweep: replay recorded answers under many routing rules and measure how far each rule moves the model.

For every prompt one answer is recorded with the unmodified router (bonus 0). That answer's tokens are then forced
through the engine under each rule ("forced decoding"), from the same restored cache state, and every next-token
distribution is compared with the unmodified router's at the same position. Beside the divergence the run keeps the
priced token time and the routing counters, so each rule gets a point on time against fidelity.

    python3 tools/mm3/native_fidelity_sweep.py run RUN_DIR            # loads one engine, works through the design, stops it
    python3 tools/mm3/native_fidelity_sweep.py run RUN_DIR --mock     # no engine: the mock, to test the runner itself
    python3 tools/mm3/native_fidelity_sweep.py status RUN_DIR

RUN_DIR holds:
    design.json      {"prompts": {id: text}, "tokens": N, "effort": "low", "seed": S,
                      "arms": [{"id", "stage", "mode": "one_row"|"drafting", "settings": {...}}, ...]}
                     Arms may be appended while the runner works: it re-reads the file when it runs out of work and
                     stops the engine only after `idle_minutes` with nothing new.
    sources/         the recorded answers (token ids and text) and their reference replays
    results.jsonl    one line per finished (arm, prompt): summary numbers against the reference
    series/          per-position divergence for each finished run
    run.log          progress, one line a run

Untimed: nothing here reads a clock for a result. It needs the GPU mutex (taken by the launcher), not the quiet flag.
The runner stops the engine and releases the cards itself, on success and on any failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = HERE / "routing_lab"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LAB))          # the lab's modules win: tools/mm3 has another module named compare

import compare                      # noqa: E402
import engines                      # noqa: E402
import settings_schema as schema    # noqa: E402

RUNS = Path("/mnt/wsl/transfer/minimax-wsl/runs")
LOCKS = RUNS / "plan-completion-20261110/locks"
RECEIPT = RUNS / "routing-lab-20261110/engine/launch.json"
RECORDS = RUNS / "routing-lab-20261110/token-records"
REFERENCE = {"routing_prior_bonus": 0.0}          # the unmodified router, on the prompt and on generated tokens
N_REFERENCE = 3                                   # a, b, c: a is the base; b and c give the noise floor


def say(log: Path, msg: str) -> None:
    line = f"{dt.datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with log.open("a") as f:
        f.write(line + "\n")


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class Sweep:
    def __init__(self, run_dir: Path, mock: bool):
        self.dir, self.mock = run_dir, mock
        for sub in ("sources", "series"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.log = run_dir / "run.log"
        self.results = run_dir / "results.jsonl"
        self.done = load_done(self.results)          # {(arm@mode, prompt)} already recorded without an error
        self.engine = None
        self.reader = None
        self.refs: dict[tuple[str, str], list[dict]] = {}       # (prompt, mode) -> reference a's records

    # ---- the engine -----------------------------------------------------------------------------------------------
    def connect(self) -> None:
        if self.mock:
            self.engine = engines.MockEngine()
            return
        profiles = json.loads((LAB / "profiles.json").read_text())
        self.engine = engines.MayaEngine("http://127.0.0.1:8097", profiles, records_dir=RECORDS, locks=LOCKS, receipt=RECEIPT)
        info = self.engine.info()
        if not info.get("connected"):
            raise SystemExit(f"the engine is not usable: {info.get('error')}")
        missing = [k for k in ("route_lab_control", "token_records_control", "force_tokens_control") if info.get(k) != 1]
        if missing:
            raise SystemExit(f"the engine lacks {missing}")
        self.reader = self.engine.reader()

    def one(self, messages, settings, sampling, drafting, force):
        """One request from the restored snapshot. -> (records, done summary, error)."""
        got = self.engine.restore()
        if not got.get("ok"):
            return [], None, f"restore failed: {got.get('error')}"
        records, summary, error, text = [], None, None, []
        clean = schema.validate(settings)
        samp = schema.validate_sampling(sampling)
        for ev in self.engine.generate(messages, clean, samp, drafting, force, threading.Event()):
            if ev["type"] == "token":
                records.append(ev["record"])
            elif ev["type"] == "delta" and ev.get("channel") != "reasoning":
                text.append(ev["text"])
            elif ev["type"] == "done":
                summary = ev["summary"]
            elif ev["type"] == "error":
                error = ev["message"]
        if summary is not None:
            summary["answer_text"] = "".join(text)
        return records, summary, error

    def price(self, summary) -> dict | None:
        path = (summary or {}).get("capture")
        if not path or self.reader is None or not hasattr(self.reader, "price"):
            return None
        try:
            got = self.reader.price(self.reader.read(path))
        except Exception as exc:                      # a price is an extra; its failure must not lose the run
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {k: got.get(k) for k in ("ms_per_token", "tokens_per_s", "exact", "tokens")} if got else None

    def discard(self, summary) -> None:
        """The engine's per-token file for a rule's run is megabytes; its numbers are kept, the file is not."""
        path = (summary or {}).get("capture")
        if path and not self.mock:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ---- sources and references --------------------------------------------------------------------------------------
    def source(self, design, pid: str) -> dict:
        path = self.dir / "sources" / f"{pid}.json"
        if path.exists():
            return json.loads(path.read_text())
        messages = [{"role": "user", "content": design["prompts"][pid]}]
        sampling = {"max_tokens": int(design.get("tokens", 240)), "reasoning_effort": design.get("effort", "low"),
                    "seed": int(design.get("seed", 20261101))}
        records, summary, error = self.one(messages, REFERENCE, sampling, False, None)
        if error or not records:
            raise SystemExit(f"could not record the source answer for {pid}: {error}")
        src = {"prompt": pid, "messages": messages, "sampling": sampling, "ids": [int(r["id"]) for r in records],
               "text": "".join(r.get("text", "") for r in records), "thinking_tokens": compare.thinking_span(records),
               "finish": (summary or {}).get("finish"), "prompt_tokens": ((summary or {}).get("begin") or {}).get("prompt_tokens")}
        path.write_text(json.dumps(src))
        self.discard(summary)
        say(self.log, f"source {pid}: {len(src['ids'])} tokens ({src['thinking_tokens']} thinking), finish {src['finish']}")
        return src

    def reference(self, design, pid: str, mode: str) -> list[dict]:
        key = (pid, mode)
        if key in self.refs:
            return self.refs[key]
        src = self.source(design, pid)
        sampling = dict(src["sampling"], max_tokens=0)
        base = None
        for letter in "abc"[:N_REFERENCE]:
            path = self.dir / "sources" / f"{pid}__{mode}__ref_{letter}.json"
            if path.exists():
                records = json.loads(path.read_text())
            else:
                records, summary, error = self.one(src["messages"], REFERENCE, sampling, mode == "drafting", src["ids"])
                if error or len(records) != len(src["ids"]) or any(int(r["id"]) != i for r, i in zip(records, src["ids"])):
                    raise SystemExit(f"reference {letter} for {pid} in {mode} failed: {error or 'the forced tokens were not honoured'}")
                slim = [{k: r.get(k) for k in ("i", "id", "text", "p", "p_raw", "top", "rest", "entropy", "source", "rows", "route")} for r in records]
                for r in slim:                                   # a reference keeps distributions, not per-layer arrays
                    if isinstance(r.get("route"), dict):
                        r["route"] = {k: v for k, v in r["route"].items() if k != "layers"}
                path.write_text(json.dumps(slim))
                self.record(design, {"id": f"ref_{letter}", "stage": "reference", "mode": mode, "settings": REFERENCE}, pid,
                            records, summary, base, src)
                self.discard(summary)
                records = slim
            if base is None:
                base = records
        self.refs[key] = base
        return base

    # ---- one result ------------------------------------------------------------------------------------------------
    def record(self, design, arm, pid, records, summary, ref, src) -> None:
        out = {"arm": arm["id"], "stage": arm.get("stage"), "mode": arm["mode"], "prompt": pid, "settings": arm["settings"],
               "tokens": len(records), "at": time.time()}
        stats = compare.summarize(records)
        out["route"] = stats.get("route")
        out["experts_per_layer"] = (stats.get("route") or {}).get("experts_per_layer")
        done_route = (summary or {}).get("route") or {}
        out["prompt_route"] = None
        if isinstance(done_route.get("prompt"), dict) and done_route["prompt"].get("picks"):
            pr = done_route["prompt"]
            out["prompt_route"] = {"moved_share": pr.get("moved", 0) / pr["picks"], "mass_kept": pr.get("mass_kept"),
                                   "min_kept": pr.get("min_kept"), "loads": pr.get("loads"), "rows": pr.get("rows")}
        out["changed"] = done_route.get("changed")
        out["mismatch"] = [done_route.get("kernel_mismatch_rows"), done_route.get("identity_mismatch_rows")]
        out["finish"] = (summary or {}).get("finish")
        out["price"] = self.price(summary)
        if not (out["price"] or {}).get("ms_per_token") and (stats.get("route") or {}).get("est_ms"):
            # No priced token time (the mock, or a reader fault): the engine's own load-cost estimate stands in, marked.
            out["price"] = {"ms_per_token": stats["route"]["est_ms"], "exact": False, "tokens": len(records),
                            "source": "engine load cost, not a token time"}
        if ref is not None:
            c = compare.compare(ref, records)
            think = src["thinking_tokens"]
            parts = {"all": list(range(len(c["positions"]))),
                     "thinking": [k for k, p in enumerate(c["positions"]) if p < think],
                     "answer": [k for k, p in enumerate(c["positions"]) if p >= think]}
            out["compared"], out["mismatched_tokens"], out["unreadable_rows"] = c["compared"], c["mismatched_tokens"], c.get("unreadable_rows", 0)
            for name, idx in parts.items():
                kl = [c["kl"][k] for k in idx]
                lr = [c["logratio"][k] for k in idx]
                t1 = [c["top1"][k] for k in idx]
                out[name] = {"n": len(idx), "kl_mean": sum(kl) / len(kl) if kl else None, "kl_median": quantile(kl, 0.5),
                             "kl_p95": quantile(kl, 0.95), "kl_p99": quantile(kl, 0.99), "kl_max": max(kl, default=0.0),
                             "kl_over_0.1": sum(1 for x in kl if x > 0.1) / len(kl) if kl else None,
                             "kl_over_1": sum(1 for x in kl if x > 1.0) / len(kl) if kl else None,
                             "top1": sum(t1) / len(t1) if t1 else None,
                             "logratio_mean": sum(lr) / len(lr) if lr else None,
                             "abs_logratio_mean": sum(abs(x) for x in lr) / len(lr) if lr else None,
                             "halved": sum(1 for x in lr if x < math.log(0.5)) / len(lr) if lr else None}
            series = {"positions": c["positions"], "kl": [round(x, 6) for x in c["kl"]], "top1": c["top1"],
                      "logratio": [round(x, 5) for x in c["logratio"]],
                      "min_kept": [(r.get("route") or {}).get("min_kept") for r in records],
                      "moved": [(r.get("route") or {}).get("moved") for r in records]}
            (self.dir / "series" / f"{arm['id']}__{arm['mode']}__{pid}.json").write_text(json.dumps(series))
        with self.results.open("a") as f:
            f.write(json.dumps(out) + "\n")
        self.done.add((arm["id"] + "@" + arm["mode"], pid))

    def run_arm(self, design, arm, pid) -> None:
        src = self.source(design, pid)
        ref = self.reference(design, pid, arm["mode"])
        sampling = dict(src["sampling"], max_tokens=0)
        began = time.time()
        records, summary, error = self.one(src["messages"], arm["settings"], sampling, arm["mode"] == "drafting", src["ids"])
        honoured = len(records) == len(src["ids"]) and all(int(r["id"]) == i for r, i in zip(records, src["ids"]))
        if error or not honoured:
            with self.results.open("a") as f:
                f.write(json.dumps({"arm": arm["id"], "stage": arm.get("stage"), "mode": arm["mode"], "prompt": pid,
                                    "settings": arm["settings"], "error": error or f"forced tokens not honoured ({len(records)} of {len(src['ids'])})"}) + "\n")
            say(self.log, f"FAILED {arm['id']} {arm['mode']} {pid}: {error or 'forced tokens not honoured'}")
            self.discard(summary)
            return
        self.record(design, arm, pid, records, summary, ref, src)
        self.discard(summary)
        last = json.loads(self.results.read_text().splitlines()[-1])
        price = (last.get("price") or {}).get("ms_per_token")
        say(self.log, f"{arm['id']} {arm['mode']} {pid}: KL {last['all']['kl_mean']:.4f} p95 {last['all']['kl_p95']:.3f} "
                      f"top1 {100 * last['all']['top1']:.1f}% moved {100 * (last['route'] or {}).get('moved_share', 0):.1f}% "
                      f"ms/tok {price if price is None else round(price, 1)} mismatch {last['mismatch']} ({time.time() - began:.0f} s)")

    # ---- the loop --------------------------------------------------------------------------------------------------
    def pending(self, design):
        out = []
        for arm in design["arms"]:
            for pid in design["prompts"]:
                if (arm["id"] + "@" + arm["mode"], pid) not in self.done:
                    out.append((arm, pid))
        return out

    def loop(self, idle_minutes: float) -> None:
        design_path = self.dir / "design.json"
        idle_since = None
        n = 0
        while True:
            design = json.loads(design_path.read_text())
            todo = self.pending(design)
            if not todo:
                if idle_since is None:
                    idle_since = time.time()
                    say(self.log, f"STAGE DONE: nothing left in the design ({n} runs so far); waiting up to {idle_minutes:g} min for more arms")
                if time.time() - idle_since > idle_minutes * 60:
                    say(self.log, "no new arms: finishing")
                    return
                time.sleep(15)
                continue
            idle_since = None
            arm, pid = todo[0]
            self.run_arm(design, arm, pid)
            n += 1
            if n % 25 == 0:
                say(self.log, f"progress: {n} runs done, {len(todo) - 1} left in the design")


def load_done(results: Path) -> set:
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("error"):
                done.add((r["arm"] + "@" + r["mode"], r["prompt"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=Path)
    r.add_argument("--mock", action="store_true", help="the mock engine, to test the runner")
    r.add_argument("--idle-minutes", type=float, default=20.0, help="how long to keep the engine up waiting for more arms")
    r.add_argument("--keep-engine", action="store_true", help="do not stop the engine at the end")
    s = sub.add_parser("status")
    s.add_argument("run_dir", type=Path)
    a = ap.parse_args()

    if a.cmd == "status":
        design = json.loads((a.run_dir / "design.json").read_text())
        done = load_done(a.run_dir / "results.jsonl")
        total = len(design["arms"]) * len(design["prompts"])
        left = sum(1 for arm in design["arms"] for pid in design["prompts"] if (arm["id"] + "@" + arm["mode"], pid) not in done)
        print(f"{total - left} of {total} runs done; last log line: {(a.run_dir / 'run.log').read_text().splitlines()[-1] if (a.run_dir / 'run.log').exists() else '-'}")
        return

    sweep = Sweep(a.run_dir, a.mock)
    launcher = [sys.executable, "-I", str(LAB / "launch_lab.py")]
    started = False
    try:
        if not a.mock:
            say(sweep.log, "loading the engine (no page)")
            subprocess.run(launcher + ["start", "--no-page"], check=True)
            started = True
        sweep.connect()
        say(sweep.log, f"engine ready; {len(sweep.done)} runs already recorded")
        sweep.loop(a.idle_minutes)
        say(sweep.log, "ALL DONE")
    except BaseException as exc:
        say(sweep.log, f"STOPPED: {type(exc).__name__}: {exc}")
        raise
    finally:
        if started and not a.keep_engine:
            subprocess.run(launcher + ["stop"])
            say(sweep.log, "engine stopped and the cards released")


if __name__ == "__main__":
    main()
