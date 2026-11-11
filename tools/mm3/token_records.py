#!/usr/bin/env python3
"""The routing lab's token records (runs/routing-lab-20261110/API.md, section 1): read a capture, follow one while the
engine writes it, compare two position by position, and price a capture's token time from its route counters.

A capture is the engine's JSON-lines file <MM3_TOKEN_RECORDS_DIR>/<token_log>.jsonl: one `begin` line, one `token` line
per committed token, one `done` line.  Each token line has two distributions of the row that produced it:
  raw  (top, rest, entropy, p_raw, logp_raw): softmax at temperature 1 after penalties, before temperature, top_k,
       top_p and min_p; exact over the whole vocabulary (rest is the mass outside top);
  samp (samp, samp_n, samp_rest, p, logp): the sampler's distribution, the one the token was drawn from.
The comparisons use raw only.

Standard library only (the tokenizer for token text is tools/strata_tokenizer.py, which needs `regex`).

    python3 tools/mm3/token_records.py self-test CAPTURE            # a capture against itself: exactly zero
    python3 tools/mm3/token_records.py compare REF OTHER [--json]    # per-position comparison and summaries
    python3 tools/mm3/token_records.py price CAPTURE [--json]        # the token time the route counters price
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from typing import Callable, Iterator

SCHEMA = 1
KL_THRESHOLDS = (0.01, 0.1, 1.0)          # nats
RATIO_THRESHOLDS = (-0.1, -1.0)           # log p_other(token) - log p_ref(token)
LAYERS = 57                                # MiniMax's MoE layers
MOE_SHARE = {"one_row": 0.906, "window": 0.875}   # MoE's share of a token's time (pricing/token_times.txt)

# The pass prices of the re-routed replay (reroute-replay-20261110/pricing/pass_prices.json, measured from the detailed
# captures of 2026-11-10; timed checks in bonus-timed-prediction.md): ms per MoE layer pass.
#   one row: A all resident, B a staged or early-copied pick and no load, C loads (the type average); by loads 1..4+;
#   window (drafting) by its rows: a + b * loads, and a table by loads where the capture had 30 or more passes.
PASS_PRICES = {
    "one_row_type": {"A": 0.4498133938194966, "B": 0.5522062866052168, "C": 1.2238270360441317},
    "one_row_loads": {"1": 1.1451458828868577, "2": 1.4630937707197385, "3": 1.9607581494736843,
                      "4": 2.1196458695652174},
}


# ---------------------------------------------------------------------------------------------------------- reading
class Capture:
    """One request's records: `begin` (dict or None), `tokens` (list of dicts, by output index), `done` (dict or None:
    a capture without one is still running, or its engine died)."""

    def __init__(self, path=None):
        self.path = pathlib.Path(path) if path is not None else None
        self.begin: dict | None = None
        self.tokens: list[dict] = []
        self.done: dict | None = None
        self.bad_lines = 0

    @property
    def finished(self) -> bool:
        return self.done is not None

    @property
    def ids(self) -> list[int]:
        return [t["id"] for t in self.tokens]

    def add(self, rec: dict):
        kind = rec.get("type")
        if kind == "begin":
            self.begin = rec
        elif kind == "token":
            self.tokens.append(rec)
        elif kind == "done":
            self.done = rec


class TokenText:
    """Token ids to text with the server's own tokenizer (tools/strata_tokenizer.py and the pack's vocab.json
    directory), loaded on first use; a piece split inside a UTF-8 character reads with U+FFFD, as the server's
    logprobs do."""

    def __init__(self, tokenizer_dir):
        self.dir = pathlib.Path(tokenizer_dir)
        self._tok = None
        self._cache: dict[int, str] = {}

    def _load(self):
        if self._tok is None:
            tools = pathlib.Path(__file__).resolve().parents[1]
            if str(tools) not in sys.path:
                sys.path.insert(0, str(tools))
            import strata_tokenizer
            self._tok = strata_tokenizer.Tokenizer.from_dir(self.dir)
        return self._tok

    def __call__(self, i: int) -> str:
        s = self._cache.get(i)
        if s is None:
            tok = self._load()
            try:
                s = tok.token_bytes(int(i)).decode("utf-8", "replace")
            except (IndexError, KeyError, ValueError):
                s = ""
            self._cache[i] = s
        return s


def add_text(rec: dict, text: Callable[[int], str]) -> dict:
    """A token record in the contract's shape: `text`, and [id, p, text] triples in top and samp."""
    if rec.get("type") != "token":
        return rec
    rec["text"] = text(rec["id"])
    for key in ("top", "samp"):
        rec[key] = [[e[0], e[1], text(e[0])] for e in rec.get(key) or []]
    return rec


def _text_of(tokenizer_dir, text):
    if text is not None:
        return text
    return TokenText(tokenizer_dir) if tokenizer_dir is not None else None


def read(path, tokenizer_dir=None, text: Callable[[int], str] | None = None) -> Capture:
    """The capture at path, as far as it is written (a last line without its newline yet is left out)."""
    cap = Capture(path)
    text = _text_of(tokenizer_dir, text)
    with open(path, "rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                cap.bad_lines += 1
                continue
            cap.add(add_text(rec, text) if text else rec)
    return cap


def follow(path, tokenizer_dir=None, text: Callable[[int], str] | None = None, poll_s: float = 0.05,
           timeout_s: float | None = None, wait_for_file_s: float = 60.0) -> Iterator[dict]:
    """The capture's records as the engine writes them, ending after `done`.  Waits up to wait_for_file_s for the file
    to appear (the request names it before the engine creates it); timeout_s bounds the wait between two records
    (None: no bound).  TimeoutError when either runs out."""
    path = pathlib.Path(path)
    text = _text_of(tokenizer_dir, text)
    start = time.monotonic()
    while not path.exists():
        if time.monotonic() - start > wait_for_file_s:
            raise TimeoutError(f"{path} did not appear in {wait_for_file_s} s")
        time.sleep(poll_s)
    buf = b""
    heard = time.monotonic()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if chunk:
                buf += chunk
                heard = time.monotonic()
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    yield add_text(rec, text) if text else rec
                    if rec.get("type") == "done":
                        return
                continue
            if timeout_s is not None and time.monotonic() - heard > timeout_s:
                raise TimeoutError(f"{path}: no record for {timeout_s} s")
            time.sleep(poll_s)


# ------------------------------------------------------------------------------------------------------ comparison
def partition_kl(ref: dict, other: dict) -> float:
    """KL(ref || other) of two token records' raw distributions over the partition {each id in both tops} plus one
    bucket holding everything else (for each side: its rest plus its top ids the other's top lacks).  A coarsening of
    both distributions, so a lower bound on the true KL (the log-sum inequality); exactly 0 for identical records."""
    p = {e[0]: e[1] for e in ref["top"]}
    q = {e[0]: e[1] for e in other["top"]}
    vals = list(p.values()) + list(q.values()) + [ref.get("rest"), other.get("rest")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in vals):
        return math.nan   # (a record the engine could not read, or a damaged one: no comparison, never a silent 0)
    both = [i for i in p if i in q]
    kl = 0.0
    for i in both:
        pi, qi = p[i], q[i]
        if pi > 0:
            kl += pi * math.log(pi / qi) if qi > 0 else math.inf
    pr = ref["rest"] + sum(v for i, v in p.items() if i not in q)
    qr = other["rest"] + sum(v for i, v in q.items() if i not in p)
    if pr > 0:
        kl += pr * math.log(pr / qr) if qr > 0 else math.inf
    return max(kl, 0.0) if kl == kl else math.nan   # (rounding can leave -1e-17 for near-equal records)


def _quantile(sorted_vals: list[float], q: float) -> float:
    """numpy's default (linear) quantile of an ascending list."""
    if not sorted_vals:
        return math.nan
    x = q * (len(sorted_vals) - 1)
    lo = int(math.floor(x))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (x - lo)


def summary(vals: list[float]) -> dict:
    """mean, median, p95, p99, max of the finite values (n: how many; inf and nan: how many were not)."""
    finite = sorted(v for v in vals if v is not None and math.isfinite(v))
    out = {"n": len(finite), "inf": sum(1 for v in vals if v is not None and math.isinf(v)),
           "nan": sum(1 for v in vals if v is None or v != v)}
    if finite:
        out.update(mean=sum(finite) / len(finite), median=_quantile(finite, 0.5), p95=_quantile(finite, 0.95),
                   p99=_quantile(finite, 0.99), max=finite[-1], min=finite[0])
    return out


def compare(ref: Capture, other: Capture) -> dict:
    """Position by position (output index), while the committed ids agree: per-position series for plotting and their
    summaries.  Under forced decoding every position compares; between free runs only the shared prefix does
    (`aligned.diverged_at`: the first index whose ids differ, None when they never do)."""
    a, b = ref.tokens, other.tokens
    n = min(len(a), len(b))
    diverged = None
    series = {"i": [], "kl": [], "top1": [], "log_ratio": [], "d_entropy": [], "source": [], "rows": []}
    for k in range(n):
        x, y = a[k], b[k]
        if x["i"] != k or y["i"] != k:
            raise ValueError(f"position {k}: records out of order")
        if x["id"] != y["id"]:
            diverged = k
            break
        if x.get("invalid_row") or y.get("invalid_row"):
            continue
        series["i"].append(k)
        series["kl"].append(partition_kl(x, y))
        series["top1"].append(1 if x["top"][0][0] == y["top"][0][0] else 0)
        lr = None if x.get("logp_raw") is None or y.get("logp_raw") is None else y["logp_raw"] - x["logp_raw"]
        series["log_ratio"].append(lr)
        series["d_entropy"].append(y["entropy"] - x["entropy"])
        series["source"].append(y.get("source"))
        series["rows"].append(y.get("rows"))
    m = len(series["i"])
    kl, lr = series["kl"], [v for v in series["log_ratio"] if v is not None]
    shares = {}
    if m:
        for t in KL_THRESHOLDS:
            shares[f"kl_over_{t:g}"] = sum(1 for v in kl if v > t) / m
        shares["top1_disagree"] = 1 - sum(series["top1"]) / m
        for t in RATIO_THRESHOLDS:
            shares[f"log_ratio_under_{t:g}"] = sum(1 for v in lr if v < t) / len(lr) if lr else math.nan
    return {
        "aligned": {"compared": m, "ref_tokens": len(a), "other_tokens": len(b), "diverged_at": diverged},
        "series": series,
        "summary": {"kl": summary(kl), "log_ratio": summary(lr), "d_entropy": summary(series["d_entropy"]),
                    "abs_log_ratio": summary([abs(v) for v in lr]), "shares": shares,
                    "top1_agreement": sum(series["top1"]) / m if m else math.nan},
    }


def self_test(cap: Capture) -> list[str]:
    """A capture against itself: every KL, log ratio and entropy change exactly 0, every top-1 the same.  The failures
    (empty: passed)."""
    r = compare(cap, cap)
    s = r["series"]
    bad = []
    if r["aligned"]["diverged_at"] is not None:
        bad.append(f"diverged at {r['aligned']['diverged_at']}")
    if r["aligned"]["compared"] == 0:
        bad.append("no positions compared")
    bad += [f"kl {v!r} at {i}" for i, v in zip(s["i"], s["kl"]) if v != 0.0]
    bad += [f"log_ratio {v!r} at {i}" for i, v in zip(s["i"], s["log_ratio"]) if v not in (0.0, None)]
    bad += [f"d_entropy {v!r} at {i}" for i, v in zip(s["i"], s["d_entropy"]) if v != 0.0]
    bad += [f"top1 differs at {i}" for i, v in zip(s["i"], s["top1"]) if v != 1]
    return bad


def check_forced(cap: Capture, force: list[int]) -> list[str]:
    """Forced decoding honoured: every committed id is the list's, in order, marked forced, and the run ended with the
    list (finish "forced") or at an end token inside it ("stop").  The failures (empty: passed)."""
    bad = []
    ids = cap.ids
    if ids != force[:len(ids)]:
        k = next(k for k, (u, v) in enumerate(zip(ids, force)) if u != v) if any(
            u != v for u, v in zip(ids, force)) else min(len(ids), len(force))
        bad.append(f"committed ids differ from the list at {k}")
    if any(not t.get("forced") for t in cap.tokens):
        bad.append("a token not marked forced")
    fin = (cap.done or {}).get("finish")
    if fin == "forced" and len(ids) != len(force):
        bad.append(f"finish forced after {len(ids)} of {len(force)}")
    if fin not in ("forced", "stop"):
        bad.append(f"finish {fin!r}")
    return bad


# ---------------------------------------------------------------------------------------------------------- pricing
def _window_price(rows: int, loads: int, prices: dict) -> float:
    w = prices.get("window", {}).get(str(min(max(rows, 1), 8)))
    if not w:
        raise KeyError(f"no window price for {rows} rows")
    table = w.get("by_loads") or {}
    if w.get("n", 0) >= 30 and str(loads) in table:
        return table[str(loads)]
    return w["a"] + w["b"] * loads


def _one_row_price(loads: int, staged: int, prices: dict) -> float:
    if loads <= 0:
        return prices["one_row_type"]["B" if staged > 0 else "A"]
    return prices["one_row_loads"][str(min(loads, 4))]


def load_prices(path=None) -> dict:
    """The pass prices: the built-in one-row table, with the window table from pass_prices.json (path, or the replay's
    run folder)."""
    prices = json.loads(json.dumps(PASS_PRICES))
    p = pathlib.Path(path) if path else pathlib.Path(
        "/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/pricing/pass_prices.json")
    if p.exists():
        prices.update(json.loads(p.read_text()))
    return prices


def price(cap: Capture, prices: dict | None = None) -> dict:
    """The decode's MoE time priced from the route objects, forward by forward (the prompt's last call, fwd 0, left
    out): with per-layer loads (route_stats 2: route.layers.loads, and route.layers.staged when given) each layer's pass
    is priced as the replay prices it; with only the token's totals (route.loads, route.staged) the loads are taken as
    spread one a layer - an approximation, flagged.  A forward's time is shared by the tokens it committed.  None when
    no token carries a route object."""
    prices = prices or load_prices()
    fwds: dict[int, dict] = {}
    for t in cap.tokens:
        f = t.get("fwd", -1)
        if f < 1:
            continue
        g = fwds.setdefault(f, {"rows": t.get("rows", 1), "tokens": 0, "route": t.get("route")})
        g["tokens"] += 1
        if g["route"] is None and t.get("route") is not None:
            g["route"] = t["route"]
    if not fwds or all(g["route"] is None for g in fwds.values()):
        return None
    moe = {"one_row": 0.0, "window": 0.0}
    toks = {"one_row": 0, "window": 0}
    exact = True
    unpriced = 0
    for g in fwds.values():
        route = g["route"]
        if route is None:
            unpriced += g["tokens"]
            continue
        kind = "one_row" if g["rows"] == 1 else "window"
        layers = route.get("layers") or {}
        loads = layers.get("loads")
        if isinstance(loads, list) and len(loads) > 0:
            staged = layers.get("staged")
            if not isinstance(staged, list):
                staged = [0] * len(loads)
                exact = exact and kind == "window"   # (one row: a staged pass costs more than a resident one)
            ms = sum(_one_row_price(l, s, prices) if kind == "one_row" else _window_price(g["rows"], l, prices)
                     for l, s in zip(loads, staged))
        else:
            exact = False
            total = int(route.get("loads", 0))
            spread = min(total, LAYERS)
            if kind == "one_row":
                ms = (LAYERS - spread) * prices["one_row_type"]["A"] + spread * prices["one_row_loads"]["1"]
                ms += (total - spread) * (prices["one_row_loads"]["2"] - prices["one_row_loads"]["1"])
            else:
                ms = (LAYERS - spread) * _window_price(g["rows"], 0, prices) + spread * _window_price(g["rows"], 1, prices)
                ms += (total - spread) * prices["window"][str(min(g["rows"], 8))]["b"]
        moe[kind] += ms
        toks[kind] += g["tokens"]
    n = toks["one_row"] + toks["window"]
    token_ms = sum(moe[k] / MOE_SHARE[k] for k in moe)
    return {"tokens": n, "unpriced_tokens": unpriced, "exact": exact, "moe_ms": moe["one_row"] + moe["window"],
            "token_ms": token_ms, "ms_per_token": token_ms / n if n else math.nan,
            "tokens_per_s": 1000.0 * n / token_ms if token_ms > 0 else math.nan, "by_mode": {
                k: {"tokens": toks[k], "moe_ms": moe[k]} for k in moe}}


# -------------------------------------------------------------------------------------------------------------- CLI
def _jsonable(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("self-test")
    st.add_argument("capture")
    cp = sub.add_parser("compare")
    cp.add_argument("ref")
    cp.add_argument("other")
    cp.add_argument("--json", action="store_true")
    pr = sub.add_parser("price")
    pr.add_argument("capture")
    pr.add_argument("--prices")
    pr.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "self-test":
        cap = read(a.capture)
        bad = self_test(cap)
        print(f"{a.capture}: {len(cap.tokens)} tokens, finish {(cap.done or {}).get('finish')}: "
              f"{'ok, exactly zero' if not bad else 'FAILED ' + '; '.join(bad[:10])}")
        return 1 if bad else 0
    if a.cmd == "compare":
        r = compare(read(a.ref), read(a.other))
        if a.json:
            print(json.dumps(_jsonable(r)))
        else:
            print(json.dumps(_jsonable({"aligned": r["aligned"], "summary": r["summary"]}), indent=1))
        return 0
    if a.cmd == "price":
        r = price(read(a.capture), load_prices(a.prices))
        print(json.dumps(_jsonable(r), indent=None if a.json else 1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
