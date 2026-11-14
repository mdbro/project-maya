#!/usr/bin/env python3
"""The GPU check that refusal ablation, built into the native serving path by the upstream sync (minimax-wsl/local,
synced through c5842c5, 2026-11-13), leaves the default path as it was: with no direction file, no extra kernel or graph
node and the same output (CLAUDE.md's decision table, Upstream: "Build it in"). Not run by the port: it needs the cards.

    ablate_identity_check.py --old-exe OLD --new-exe NEW --out DIR [--config FILE] [--direction FILE] [--tokens N]

Each engine run below is one engine start (about 350 s), from the serving config's arguments and environment (default:
docs/minimax-wsl/native-server.json; its exe is replaced by OLD / NEW). The engine is driven on its own stdin protocol
(GEN / T / STAT / DONE), no server, so the run sees only the engine. Every request reads its prompt from position 0
(fresh_context=1); MM3_PROMPT_HASH writes every layer's output row hashes of each prompt read, from the layer-by-layer
reader (prompt_layerwise, the built-in) or the chunks.

  1. OLD (the binary before the sync), no direction: the reference.
  2. NEW, no direction: the same generated token ids for every request, the same per-layer prompt hashes (bit for
     bit), and the same graph counts in each request's STAT (prep_graph_captures / _launches, expert_graph_captures /
     _launches). With ablation's projection absent from every captured graph and from the reader's combine
     (prompt_combine as before), equal bits with equal graph counts are what "nothing extra" leaves to see; the node
     count of the projection itself is tests/mm3_native_ablate_test.cu's part 4.
  3. With --direction: NEW started with --ablate FILE --ablate-default off - the direction loaded, every request off:
     the same as 2 again (the graphs are keyed by the setting, so none captured with the projection replays). Then,
     on the same engine, the requests with ablate=1 and prep_graph=2 expert_graph=2 (each replay checked bit for bit
     against its launches, the projection in both): prep_graph_mismatches and expert_graph_mismatches 0, the prompt
     hashes differ from 2's at every layer (the projection reached the reader and the window paths), and then ablate=0
     once more: 2's tokens and hashes again.

Exit 0 when every comparison holds; the report is DIR/report.json. Take the GPU mutex as the plan's tools do
(LOCKS/gpu.lock.d), with the serving engine stopped: the script refuses to start an engine while the mutex is held or a
card holds memory. Standard library only."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

W = Path(__file__).resolve().parents[2]
LOCKS = Path("/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/locks")
MUTEX, NAME = LOCKS / "gpu.lock.d", "ablate-identity-check"
GRAPH_KEYS = ("prep_graph_captures", "prep_graph_launches", "expert_graph_captures", "expert_graph_launches")
# a prompt long enough for the layer-by-layer reader (more than 32 rows), and short and long answers
PROMPTS = [
    "Explain, step by step, how a CPU cache hierarchy reduces memory latency, and give one example where it fails to help. " * 4,
    "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with three tests. " * 3,
]


class Engine:
    """One engine start: its stdin protocol, and the lines it printed."""

    def __init__(self, exe: str, cfg: dict, extra: list[str], env_extra: dict, log: Path):
        args = [str(x) for x in cfg.get("args") or []]
        env = {**os.environ, **{k: str(v) for k, v in (cfg.get("env") or {}).items()}, **env_extra}
        self.log = log.open("w")
        self.p = subprocess.Popen([exe, *args, *extra], cwd=cfg.get("cwd") or None, env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        self.info: dict = {}
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit(f"the engine ended before READY (log {log})")
            if line.startswith("INFO "):
                self.info.update(kv(line[5:]))
            if line.startswith("READY"):
                break

    def gen(self, ids: list[int], max_new: int, keys: dict) -> dict:
        head = "GEN " + str(max_new) + "".join(f" {k}={v}" for k, v in keys.items())
        self.p.stdin.write(f"{head} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        out, stat, err = [], {}, None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("the engine ended in a request")
            if line.startswith("T "):
                out.append(int(line.split()[1]))
            elif line.startswith("STAT "):
                stat = kv(line[5:])
            elif line.startswith("ERR "):
                err = line[4:].strip()
            elif line.startswith("DONE"):
                return {"tokens": out, "stat": stat, "error": err}

    def quit(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()
        self.log.close()


def kv(text: str) -> dict:
    out = {}
    for part in text.split():
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def hashes(d: Path) -> list[list[str]]:
    """Each prompt file's lines (kind, layer, position, hash), in the order the engine wrote the files."""
    files = sorted(d.glob("prompt-*.txt"), key=lambda p: int(p.stem.split("-")[1]))
    return [p.read_text().splitlines() for p in files]


def run(exe: str, cfg: dict, extra: list[str], name: str, out: Path, requests: list[tuple[list[int], dict]], tokens: int):
    hd = out / f"{name}-hash"
    hd.mkdir(parents=True, exist_ok=True)
    eng = Engine(exe, cfg, extra, {"MM3_PROMPT_HASH": str(hd)}, out / f"{name}.log")
    try:
        results = [eng.gen(ids, tokens, {"fresh_context": 1, "seed": 1234 + i, **keys})
                   for i, (ids, keys) in enumerate(requests)]
    finally:
        eng.quit()
    return {"info": eng.info, "results": results, "hashes": hashes(hd)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-exe", required=True)
    ap.add_argument("--new-exe", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=W / "docs/minimax-wsl/native-server.json")
    ap.add_argument("--direction", help="a control vector GGUF of the model's hidden size (step 3)")
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    if MUTEX.exists():
        raise SystemExit(f"REFUSED: the GPU mutex is held ({(MUTEX / 'owner').read_text().strip()})")
    used = [int(x) for x in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                    text=True).split()]
    if max(used) > 2000:
        raise SystemExit(f"REFUSED: a card already holds memory ({used} MiB)")
    MUTEX.mkdir()
    (MUTEX / "owner").write_text(NAME + "\n")
    try:
        cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
        sys.path.insert(0, str(W / "tools"))
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(Path(cfg["tokenizer"]))
        prompts = [tok.encode(p) for p in PROMPTS]
        a.out.mkdir(parents=True, exist_ok=True)
        base = [(ids, {}) for ids in prompts]
        report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": []}

        def check(name, ok, detail=""):
            report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
            print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""), flush=True)

        old = run(a.old_exe, cfg, [], "old", a.out, base, a.tokens)
        new = run(a.new_exe, cfg, [], "new", a.out, base, a.tokens)
        for i, (r0, r1) in enumerate(zip(old["results"], new["results"])):
            check(f"request {i}: no error", r0["error"] is None and r1["error"] is None, f"{r0['error']} / {r1['error']}")
            check(f"request {i}: the same tokens", r0["tokens"] == r1["tokens"])
            for k in GRAPH_KEYS:
                check(f"request {i}: the same {k}", r0["stat"].get(k) == r1["stat"].get(k),
                      f"{r0['stat'].get(k)} / {r1['stat'].get(k)}")
        check("the same per-layer prompt hashes", old["hashes"] == new["hashes"] and len(new["hashes"]) == len(base),
              f"{len(old['hashes'])} / {len(new['hashes'])} files")
        check("no ablate in the new engine's INFO", "ablate" not in new["info"])
        if a.direction:
            reqs = base + [(ids, {"ablate": 1, "prep_graph": 2, "expert_graph": 2}) for ids in prompts] + \
                [(ids, {"ablate": 0}) for ids in prompts]
            dirn = run(a.new_exe, cfg, ["--ablate", a.direction, "--ablate-default", "off"], "direction", a.out, reqs,
                       a.tokens)
            n = len(prompts)
            check("direction loaded: INFO ablate=1 ablate_default=0",
                  dirn["info"].get("ablate") == 1 and dirn["info"].get("ablate_default") == 0)
            for i in range(n):
                off, on, back = dirn["results"][i], dirn["results"][n + i], dirn["results"][2 * n + i]
                check(f"loaded, off, request {i}: the tokens of no direction", off["tokens"] == new["results"][i]["tokens"])
                check(f"loaded, off again, request {i}: the same tokens", back["tokens"] == new["results"][i]["tokens"])
                for k in GRAPH_KEYS:
                    check(f"loaded, off, request {i}: the same {k}", off["stat"].get(k) == new["results"][i]["stat"].get(k))
                check(f"on, request {i}: no error", on["error"] is None, str(on["error"]))
                for k in ("prep_graph_mismatches", "expert_graph_mismatches"):
                    check(f"on, request {i}: {k} 0", on["stat"].get(k, 0) == 0, str(on["stat"].get(k)))
            h = dirn["hashes"]
            check("loaded, off: the hashes of no direction", h[:n] == new["hashes"])
            check("loaded, off again: the hashes of no direction", h[2 * n:3 * n] == new["hashes"])
            for i in range(n):
                on, ref = h[n + i], new["hashes"][i]
                layers_on = {ln.split()[1] for ln, lr in zip(on, ref) if ln.startswith("h ") and ln != lr}
                layers = {ln.split()[1] for ln in ref if ln.startswith("h ")}
                check(f"on, request {i}: every layer's rows differ from no direction", layers and layers_on == layers,
                      f"{len(layers_on)} of {len(layers)} layers")
        report["ok"] = all(c["ok"] for c in report["checks"])
        (a.out / "report.json").write_text(json.dumps(report, indent=1))
        print("OK" if report["ok"] else "FAILED", "-", a.out / "report.json")
        return 0 if report["ok"] else 1
    finally:
        (MUTEX / "owner").unlink(missing_ok=True)
        MUTEX.rmdir()


if __name__ == "__main__":
    sys.exit(main())
