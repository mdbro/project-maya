#!/usr/bin/env python3
"""Routing lab: a local page for chatting with MiniMax while switching routing rules per prompt.

    python3 tools/mm3/routing_lab/lab_server.py                 # the Maya server at http://127.0.0.1:8097
    python3 tools/mm3/routing_lab/lab_server.py --mock          # no engine: synthetic tokens, to try the page

Then open http://127.0.0.1:8796/ . Standard library only.

The backend keeps no conversation state: the page sends the whole message list each turn. Every answer is stored as
a run (its messages, settings, text and per-token records) so it can be inspected, compared and replayed by the
fidelity meter. One generation runs at a time, because the engine serves one request at a time.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare            # noqa: E402
import engines            # noqa: E402
import settings_schema as schema   # noqa: E402

DEFAULT_DATA = Path("/mnt/wsl/transfer/minimax-wsl/runs/routing-lab-20261110/data")
DEFAULT_LOCKS = Path("/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/locks")
STATIC = HERE / "static"
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".svg": "image/svg+xml", ".json": "application/json"}


class Refused(Exception):
    """The lab will not send this request (for example, the engine at the address belongs to a timed campaign)."""


class Run:
    """One generation: its inputs, its events as they arrive, and its stored result."""

    def __init__(self, kind, messages, settings, sampling, drafting, label="", force_tokens=None):
        self.id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.kind, self.messages, self.settings, self.sampling = kind, messages, settings, sampling
        self.drafting, self.label, self.force_tokens = bool(drafting), label, force_tokens
        self.created = time.time()
        self.events: list[dict] = []
        self.cond = threading.Condition()
        self.cancel = threading.Event()
        self.status, self.text, self.reasoning, self.records, self.summary, self.error = "running", "", "", [], None, None
        self.notes: list[str] = []

    def stats(self) -> dict:
        return compare.summarize(self.records, bool(self.reasoning))

    def push(self, event: dict) -> None:
        with self.cond:
            self.events.append(event)
            self.cond.notify_all()

    def to_json(self, with_records=True) -> dict:
        out = {"id": self.id, "kind": self.kind, "created": self.created, "label": self.label, "status": self.status,
               "messages": self.messages, "settings": self.settings, "sampling": self.sampling, "drafting": self.drafting,
               "text": self.text, "reasoning": self.reasoning, "summary": self.summary, "error": self.error,
               "describe": schema.describe(self.settings), "stats": self.stats(), "notes": self.notes,
               "forced": self.force_tokens is not None}
        if with_records:
            out["records"] = self.records
        return out


class Lab:
    def __init__(self, engine, data: Path):
        self.engine, self.data = engine, data
        for sub in ("runs", "meters", "captures"):
            (data / sub).mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, Run] = {}
        self.meters: dict[str, dict] = {}
        self.busy = threading.Lock()          # the engine serves one request at a time
        self.lock = threading.Lock()

    # ---- generations ------------------------------------------------------------------------------------------
    def start(self, kind, messages, settings, sampling, drafting, label="", force_tokens=None, wait=False) -> Run:
        settings = schema.validate(settings or {})
        samp = schema.validate_sampling(sampling)
        if not isinstance(messages, list) or not messages:
            raise ValueError("no messages")
        run = Run(kind, messages, settings, samp, drafting, label, force_tokens)
        with self.lock:
            self.runs[run.id] = run
        if wait:
            self._work(run, block=True)
        else:
            if not self.busy.acquire(blocking=False):
                run.status, run.error = "error", "the engine is busy with another request"
                run.push({"type": "error", "message": run.error})
                return run
            threading.Thread(target=self._work, args=(run, False), daemon=True).start()
        return run

    def _work(self, run: Run, block: bool) -> None:
        if block:
            self.busy.acquire()
        try:
            why = getattr(self.engine, "blocked", lambda: None)()
            if why:
                raise Refused(why)
            if run.sampling.get("same_start") or run.force_tokens is not None:
                got = self.engine.restore()
                note = (f"cache state restored in {got.get('seconds', 0):g} s" if got.get("ok") else
                        f"the starting cache state could not be restored ({got.get('error')}); "
                        "this run began from the state the last one left")
                run.notes.append(note)
                run.push({"type": "note", "text": note})
            for event in self.engine.generate(run.messages, run.settings, run.sampling, run.drafting, run.force_tokens,
                                              run.cancel):
                if event["type"] == "delta":
                    if event.get("channel") == "reasoning":
                        run.reasoning += event["text"]
                    else:
                        run.text += event["text"]
                elif event["type"] == "token":
                    run.records.append(event["record"])
                elif event["type"] == "done":
                    run.summary = event["summary"]
                elif event["type"] == "error":
                    run.error = event["message"]
                run.push(event)
            run.status = "error" if run.error else "done"
        except engines.Cancelled:
            run.status = "stopped"
            run.push({"type": "stopped"})
        except Refused as exc:                                      # nothing was sent to the engine
            run.status, run.error = "error", str(exc)
            run.push({"type": "error", "message": run.error})
        except Exception as exc:                                    # shown on the page; the trace goes to the log
            traceback.print_exc()
            run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
            run.push({"type": "error", "message": run.error})
        finally:
            self.busy.release()
            run.push({"type": "end", "status": run.status, "stats": run.stats()})
            self._save(run)

    def _save(self, run: Run) -> None:
        try:
            (self.data / "runs" / f"{run.id}.json").write_text(json.dumps(run.to_json()))
        except OSError as exc:
            print(f"[lab] could not save run {run.id}: {exc}", file=sys.stderr)

    def get_run(self, run_id: str) -> dict | None:
        run = self.runs.get(run_id)
        if run is not None:
            return run.to_json()
        path = self.data / "runs" / f"{Path(run_id).name}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def list_runs(self) -> list[dict]:
        seen, out = set(), []
        for run in self.runs.values():
            seen.add(run.id)
            out.append(_brief(run.to_json(with_records=False)))
        for path in (self.data / "runs").glob("*.json"):
            if path.stem in seen:
                continue
            try:
                out.append(_brief(json.loads(path.read_text())))
            except (ValueError, OSError):
                continue
        return sorted(out, key=lambda r: -r["created"])[:400]

    # ---- presets ----------------------------------------------------------------------------------------------
    def presets(self) -> dict:
        path = self.data / "presets.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        base = schema.defaults()
        built_in = {
            "Today (bonus 0.20)": base,
            "Unmodified router (bonus 0)": dict(base, routing_prior_bonus=0.0),
            "Bonus 0.30": dict(base, routing_prior_bonus=0.30),
            "Bonus 0.40, budget 0.20": dict(base, routing_prior_bonus=0.40, route_floor=0.80),
            "Priced, as in the replay (budget 0.20)": dict(base, routing_prior_bonus=1.0, route_floor=0.80, route_priced=2.0),
            "Bonus 0.30, dampened": dict(base, routing_prior_bonus=0.30, route_weights=1),
        }
        return {"built_in": built_in, "saved": saved}

    def save_preset(self, name: str, settings) -> None:
        path = self.data / "presets.json"
        saved = json.loads(path.read_text()) if path.exists() else {}
        if settings is None:
            saved.pop(name, None)
        else:
            saved[name] = schema.validate(settings)
        path.write_text(json.dumps(saved, indent=1))

    # ---- the fidelity meter -----------------------------------------------------------------------------------
    def start_meter(self, source_id: str, arms: list, drafting: bool, noise_floor: bool, label: str) -> dict:
        source = self.get_run(source_id)
        if source is None:
            raise ValueError("unknown source answer")
        tokens = [int(r["id"]) for r in source.get("records", []) if r.get("type", "token") == "token"]
        if len(tokens) < 2:
            raise ValueError("the source answer has no recorded tokens to replay")
        if not arms:
            raise ValueError("no rules to measure")
        clean = []
        for arm in arms:
            clean.append({"name": str(arm.get("name") or schema.describe(arm.get("settings", {})))[:80],
                          "settings": schema.validate(arm.get("settings", {}))})
        job = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6], "created": time.time(), "label": label,
               "source": source_id, "source_text": source.get("text", "")[:4000], "tokens": len(tokens),
               "drafting": bool(drafting), "status": "running", "progress": "waiting for the engine", "done": 0,
               "total": len(clean) + 1 + (1 if noise_floor else 0), "arms": [], "error": None,
               "reference": {"name": "Unmodified router (bonus 0)", "settings": dict(schema.defaults(), routing_prior_bonus=0.0)}}
        with self.lock:
            self.meters[job["id"]] = job
        threading.Thread(target=self._meter, args=(job, source["messages"], source.get("sampling", {}), tokens, clean,
                                                   noise_floor), daemon=True).start()
        return job

    def _meter(self, job, messages, sampling, tokens, arms, noise_floor) -> None:
        def forced(settings, what):
            job["progress"] = what
            run = self.start("meter", messages, settings, sampling, job["drafting"], label=f"meter {job['id']}: {what}",
                             force_tokens=tokens, wait=True)
            if run.status != "done":
                raise RuntimeError(f"{what}: {run.error or run.status}")
            job["done"] += 1
            return run
        try:
            ref = forced(job["reference"]["settings"], "the reference (unmodified router)")
            job["reference"].update({"run": ref.id, "stats": ref.stats()})
            job["notes"] = list(ref.notes)
            if noise_floor:
                again = forced(job["reference"]["settings"], "the reference again, for the noise floor")
                c = compare.compare(ref.records, again.records)
                job["arms"].append({"name": "Noise floor (two reference runs)", "settings": job["reference"]["settings"],
                                    "run": again.id, "noise_floor": True, "stats": again.stats(), **c})
            for arm in arms:
                run = forced(arm["settings"], arm["name"])
                c = compare.compare(ref.records, run.records)
                job["arms"].append({"name": arm["name"], "settings": arm["settings"], "run": run.id,
                                    "describe": schema.describe(arm["settings"]), "stats": run.stats(), **c})
            job["status"], job["progress"] = "done", "done"
        except Exception as exc:
            traceback.print_exc()
            job["status"], job["error"], job["progress"] = "error", f"{type(exc).__name__}: {exc}", "stopped on an error"
        try:
            (self.data / "meters" / f"{job['id']}.json").write_text(json.dumps(job))
        except OSError as exc:
            print(f"[lab] could not save meter {job['id']}: {exc}", file=sys.stderr)

    def get_meter(self, job_id: str) -> dict | None:
        if job_id in self.meters:
            return self.meters[job_id]
        path = self.data / "meters" / f"{Path(job_id).name}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def list_meters(self) -> list[dict]:
        seen, out = set(), []
        for job in list(self.meters.values()):
            seen.add(job["id"])
            out.append(_brief_meter(job))
        for path in (self.data / "meters").glob("*.json"):
            if path.stem not in seen:
                try:
                    out.append(_brief_meter(json.loads(path.read_text())))
                except (ValueError, OSError):
                    continue
        return sorted(out, key=lambda j: -j["created"])[:200]


def _brief(run: dict) -> dict:
    user = next((m.get("content", "") for m in reversed(run.get("messages", [])) if m.get("role") == "user"), "")
    return {"id": run["id"], "kind": run.get("kind"), "created": run.get("created", 0), "label": run.get("label", ""),
            "status": run.get("status"), "describe": run.get("describe", ""), "drafting": run.get("drafting", False),
            "prompt": user[:160], "answer": (run.get("text") or "")[:160], "tokens": (run.get("stats") or {}).get("tokens", 0),
            "forced": run.get("forced", False)}


def _brief_meter(job: dict) -> dict:
    return {"id": job["id"], "created": job["created"], "label": job.get("label", ""), "status": job["status"],
            "tokens": job.get("tokens", 0), "arms": [a["name"] for a in job.get("arms", [])], "drafting": job.get("drafting"),
            "progress": job.get("progress"), "done": job.get("done", 0), "total": job.get("total", 0)}


class Handler(BaseHTTPRequestHandler):
    lab: Lab = None            # set in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):           # one line per request is noise for a local tool
        return

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 8 << 20:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        lab = self.lab
        try:
            if url.path == "/api/info":
                self._json({"engine": lab.engine.info(), "schema": schema.SCHEMA, "sampling": schema.SAMPLING,
                            "defaults": schema.defaults(), "sampling_defaults": schema.sampling_defaults(),
                            "n_layers": schema.N_LAYERS, "busy": lab.busy.locked()})
            elif url.path == "/api/runs":
                self._json(lab.list_runs())
            elif url.path == "/api/run":
                run = lab.get_run(q.get("id", ""))
                self._json(run if run else {"error": "unknown run"}, 200 if run else 404)
            elif url.path == "/api/presets":
                self._json(lab.presets())
            elif url.path == "/api/compare":
                a, b = lab.get_run(q.get("a", "")), lab.get_run(q.get("b", ""))
                if not a or not b:
                    self._json({"error": "unknown answer"}, 404)
                else:
                    self._json(compare.shared_prefix(a.get("records", []), b.get("records", [])))
            elif url.path == "/api/meters":
                self._json(lab.list_meters())
            elif url.path == "/api/meter":
                job = lab.get_meter(q.get("id", ""))
                self._json(job if job else {"error": "unknown meter run"}, 200 if job else 404)
            elif url.path == "/api/events":
                self._events(q.get("run", ""), int(q.get("from", 0)))
            else:
                self._static(url.path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        lab = self.lab
        try:
            body = self._body()
            if url.path == "/api/chat":
                run = lab.start("chat", body.get("messages"), body.get("settings"), body.get("sampling"),
                                body.get("drafting", False), label=str(body.get("label", ""))[:120])
                self._json({"run": run.id, "status": run.status, "error": run.error}, 409 if run.status == "error" else 200)
            elif url.path == "/api/stop":
                run = lab.runs.get(body.get("run", ""))
                if run:
                    run.cancel.set()
                self._json({"ok": bool(run)})
            elif url.path == "/api/presets":
                lab.save_preset(str(body.get("name", ""))[:80], body.get("settings"))
                self._json(lab.presets())
            elif url.path == "/api/meter":
                job = lab.start_meter(body.get("source", ""), body.get("arms") or [], body.get("drafting", False),
                                      body.get("noise_floor", True), str(body.get("label", ""))[:120])
                self._json({"id": job["id"]})
            else:
                self._json({"error": "unknown endpoint"}, 404)
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _events(self, run_id: str, start: int):
        run = self.lab.runs.get(run_id)
        if run is None:
            self._json({"error": "unknown run"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        i = start
        while True:
            with run.cond:
                while i >= len(run.events):
                    if not run.cond.wait(timeout=15):
                        break
                batch = run.events[i:]
            if not batch:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                continue
            for event in batch:
                self.wfile.write(f"id: {i}\ndata: {json.dumps(event)}\n\n".encode())
                i += 1
            self.wfile.flush()
            if batch[-1].get("type") == "end":
                return

    def _static(self, path: str):
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        if name.startswith("static/"):
            name = name[7:]
        target = (STATIC / name).resolve()
        if STATIC not in target.parents and target != STATIC or not target.is_file():
            self._json({"error": "not found"}, 404)
            return
        data = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(target.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser(description="Routing lab: chat with MiniMax while switching routing rules per prompt.")
    ap.add_argument("--port", type=int, default=8796)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 by default: the page is for this machine only")
    ap.add_argument("--maya", default="http://127.0.0.1:8097", help="the Maya server's address")
    ap.add_argument("--mock", action="store_true", help="synthetic tokens instead of the engine, to try the page")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="where answers, meter runs and presets are kept")
    ap.add_argument("--records", type=Path, default=None,
                    help="the engine's token-record directory (default: the one the engine reports in /v1/status)")
    ap.add_argument("--locks", type=Path, default=DEFAULT_LOCKS,
                    help="the campaigns' lock directory: requests go to the engine only while the lab's launcher "
                         "holds the GPU mutex there")
    ap.add_argument("--any-engine", action="store_true",
                    help="skip that check and talk to whatever engine is at --maya. Never use this while a timed "
                         "campaign may be running: a foreign request spoils its measurements")
    ap.add_argument("--tokenizer", type=Path, default=None,
                    help="the pack's tokenizer directory, for token text (default: the one profiles.json names)")
    ap.add_argument("--profiles", type=Path, default=HERE / "profiles.json",
                    help="the tune keys of the one-row and drafting modes")
    a = ap.parse_args()
    if a.mock:
        engine = engines.MockEngine()
        a.data = a.data.parent / (a.data.name + "-mock")
    else:
        profiles = json.loads(a.profiles.read_text()) if a.profiles.exists() else {"one_row": {"spec": 1}, "drafting": {}}
        engine = engines.MayaEngine(a.maya, profiles, records_dir=a.records, tokenizer_dir=a.tokenizer,
                                    locks=None if a.any_engine else a.locks)
    Handler.lab = Lab(engine, a.data)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    print(f"routing lab: http://{a.host}:{a.port}/   engine: {engine.name}   data: {a.data}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
