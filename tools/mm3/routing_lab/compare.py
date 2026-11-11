"""Comparing two lab captures position by position, and summarising one capture.

A capture is the list of token records of CONTRACT.md. Each record gives the next-token distribution as the top
candidates [[id, prob, text], ...] plus `rest`, the probability mass outside them.

The fidelity meter compares a rule's capture with a reference capture of the same forced tokens. If the engine
agents' reader module is importable it is preferred (it is the one validated on the engine); this file is the
fallback and the place for the page's own summaries.
"""
from __future__ import annotations

import math

EPS = 1e-12


def _dist(rec: dict) -> tuple[dict, float]:
    top = {int(t[0]): float(t[1]) for t in rec.get("top", [])}
    rest = rec.get("rest")
    rest = max(0.0, 1.0 - sum(top.values())) if rest is None else float(rest)
    return top, max(rest, 0.0)


def unreadable(rec: dict) -> bool:
    """A row the engine could not read (NaN or infinite logits): it records no candidates and null numbers."""
    return bool(rec.get("invalid_row")) or not rec.get("top") or rec.get("entropy") is None


def kl_with_rest(p_rec: dict, q_rec: dict) -> float:
    """KL(P || Q) in nats over a partition both records can fill: each candidate listed by both, plus one bucket
    for everything else.

    Merging outcomes can only lower a KL divergence, so this is a lower bound on the divergence over the whole
    vocabulary, and it is exactly 0 for identical records. With 64 candidates a side the bucket is a few percent of
    the mass, so the bound is close. (The engine agents' reader uses the same definition.)
    """
    p, _ = _dist(p_rec)
    q, _ = _dist(q_rec)
    common = [i for i in p if i in q]
    kl = 0.0
    for i in common:
        if p[i] > 0.0:
            kl += p[i] * math.log(p[i] / max(q[i], EPS))
    p_other = max(1.0 - sum(p[i] for i in common), 0.0)
    q_other = max(1.0 - sum(q[i] for i in common), 0.0)
    if p_other > 0.0:
        kl += p_other * math.log(p_other / max(q_other, EPS))
    return max(kl, 0.0)


def compare(reference: list[dict], other: list[dict]) -> dict:
    """Per-position series and summaries for `other` against `reference` (both lists of token records)."""
    n = min(len(reference), len(other))
    kl, top1, logratio, dent, pos = [], [], [], [], []
    mismatch = invalid = 0
    for k in range(n):
        a, b = reference[k], other[k]
        if a.get("id") != b.get("id"):
            mismatch += 1          # forced decoding should make these equal; a mismatch voids the position
            continue
        if unreadable(a) or unreadable(b):
            invalid += 1           # never a silent zero: the position is left out and counted
            continue
        pos.append(k)
        kl.append(kl_with_rest(a, b))
        ta = max(a.get("top", [[None, 0]]), key=lambda t: t[1])[0] if a.get("top") else None
        tb = max(b.get("top", [[None, 0]]), key=lambda t: t[1])[0] if b.get("top") else None
        top1.append(1 if ta == tb else 0)
        pa, pb = _praw(a), _praw(b)      # the raw distribution: filtered ones differ in support
        logratio.append(math.log(max(pb, EPS) / max(pa, EPS)))
        dent.append(float(b.get("entropy", 0.0)) - float(a.get("entropy", 0.0)))
    return {
        "positions": pos, "kl": kl, "top1": top1, "logratio": logratio, "entropy_change": dent,
        "compared": len(pos), "mismatched_tokens": mismatch, "unreadable_rows": invalid,
        "summary": {
            "kl_mean": _mean(kl), "kl_median": _quantile(kl, 0.5), "kl_p95": _quantile(kl, 0.95), "kl_max": max(kl, default=0.0),
            "kl_over_0.1": _share(kl, lambda x: x > 0.1), "kl_over_1": _share(kl, lambda x: x > 1.0),
            "top1_agreement": _mean(top1),
            "logratio_mean": _mean(logratio), "logratio_p05": _quantile(logratio, 0.05),
            "forced_prob_halved": _share(logratio, lambda x: x < math.log(0.5)),
            "entropy_change_mean": _mean(dent),
        },
    }


def shared_prefix(a: list[dict], b: list[dict]) -> dict:
    """Two free-running answers: how many leading tokens they share, the divergence of their distributions over
    those tokens, and both records at the first position where they chose differently."""
    n = 0
    while n < min(len(a), len(b)) and a[n].get("id") == b[n].get("id"):
        n += 1
    out = {"shared": n, "len_a": len(a), "len_b": len(b), "identical": n == len(a) == len(b)}
    if n:
        out.update(compare(a[:n], b[:n]))
    if n < min(len(a), len(b)):
        out["first_a"], out["first_b"] = a[n], b[n]
        out["kl_at_split"] = kl_with_rest(a[n], b[n])
    return out


THINK_BEGIN = ("<mm:think>", "<think>")
THINK_END = ("</mm:think>", "</think>")


def thinking_span(records: list[dict], saw_reasoning: bool = False) -> int:
    """How many leading tokens are the model's thinking (the closing tag included); 0 if it did not think.

    MiniMax closes its thinking with a tag of its own. When thinking is switched on the opening tag is part of the
    prompt, so an answer cut off while still thinking has no tag at all: `saw_reasoning` (the server streamed
    reasoning text) then says the whole of it is thinking.
    """
    last_end = -1
    begun = False
    for k, r in enumerate(records):
        t = (r.get("text") or "").strip()
        if t in THINK_END:
            last_end = k
        elif t in THINK_BEGIN:
            begun = True
    if last_end >= 0:
        return last_end + 1
    return len(records) if (begun or saw_reasoning) else 0


def summarize(records: list[dict], saw_reasoning: bool = False) -> dict:
    """What one answer's records say: confidence, surprise and routing."""
    toks = [r for r in records if r.get("type", "token") == "token"]
    if not toks:
        return {"tokens": 0}
    good = [r for r in toks if not unreadable(r)]            # the statistics leave unreadable rows out, and say so
    logp = [math.log(max(_praw(r), EPS)) for r in good] or [0.0]
    ent = [float(r["entropy"]) for r in good]
    ranks = []
    for r in good:
        order = sorted(r.get("top", []), key=lambda t: -t[1])
        rank = next((k + 1 for k, t in enumerate(order) if t[0] == r.get("id")), len(order) + 1)
        ranks.append(rank)
    route = [r.get("route") for r in toks if isinstance(r.get("route"), dict)]
    out = {
        "tokens": len(toks),
        "mean_prob": _mean([math.exp(x) for x in logp]),
        "perplexity": math.exp(-_mean(logp)),
        "surprisal_mean": -_mean(logp), "surprisal_p95": _quantile([-x for x in logp], 0.95),
        "entropy_mean": _mean(ent),
        "chosen_top1": _share(ranks, lambda x: x == 1), "chosen_beyond_5": _share(ranks, lambda x: x > 5),
        "drafted": _share(toks, lambda r: r.get("source") == "draft"),
        "rows_mean": _mean([float(r.get("rows", 1)) for r in toks]),
        "thinking_tokens": thinking_span(toks, saw_reasoning),
        "unreadable_rows": len(toks) - len(good),
    }
    if route:
        picks = sum(float(x.get("picks", 0)) for x in route) or 1.0
        layers = sum(float(x.get("layer_count") or 57) for x in route)
        out["route"] = {
            "experts_per_layer": sum(float(x.get("picks", 0)) for x in route) / layers,
            "moved_share": sum(float(x.get("moved", 0)) for x in route) / picks,
            "mass_kept_mean": _mean([float(x.get("mass_kept", 1.0)) for x in route]),
            "min_kept": min(float(x.get("min_kept", 1.0)) for x in route),
            "rows_under_08": _mean([float(x.get("rows_under_08", 0)) for x in route]),
            "loads": _mean([float(x.get("loads", 0)) for x in route]),
            "staged": _mean([float(x.get("staged", 0)) for x in route]),
            "dropped": _mean([float(x.get("dropped", 0)) for x in route]),
            "widened": _mean([float(x.get("widened", 0)) for x in route]),
            "layers_dropped": _mean([float(x.get("layers_dropped", 0)) for x in route]),
            "est_ms": _mean([float(x.get("est_ms", 0.0)) for x in route]),
        }
    return out


def _praw(rec: dict) -> float:
    """The committed token's probability in the raw distribution (temperature 1, before filters), if recorded."""
    v = rec.get("p_raw")
    return float(v if v is not None else (rec.get("p") or 0.0))


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _share(xs, pred) -> float:
    xs = list(xs)
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else 0.0


def _quantile(xs, q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def self_test() -> None:
    rec = {"id": 5, "p": 0.5, "top": [[5, 0.5, "a"], [7, 0.3, "b"]], "rest": 0.2, "entropy": 1.03}
    assert kl_with_rest(rec, rec) == 0.0
    c = compare([rec, rec], [rec, rec])
    assert c["summary"]["kl_mean"] == 0.0 and c["summary"]["top1_agreement"] == 1.0 and c["compared"] == 2
    other = {"id": 5, "p": 0.25, "top": [[7, 0.6, "b"], [5, 0.25, "a"]], "rest": 0.15, "entropy": 0.9}
    d = compare([rec], [other])
    assert d["kl"][0] > 0.0 and d["top1"][0] == 0 and abs(d["logratio"][0] - math.log(0.5)) < 1e-9
    s = summarize([dict(rec, type="token", source="target", rows=1,
                        route={"picks": 228, "moved": 57, "mass_kept": 0.95, "min_kept": 0.7})])
    assert abs(s["route"]["moved_share"] - 0.25) < 1e-12 and abs(s["perplexity"] - 2.0) < 1e-9
    assert s["route"]["experts_per_layer"] == 4.0
    # Lists of different lengths: the shared candidate agrees, so the coarse partition sees no difference.
    q2 = {"id": 5, "p": 0.5, "top": [[5, 0.5, "a"]], "rest": 0.5}
    assert kl_with_rest(rec, q2) == 0.0
    # By hand: common {5, 7}, P = (0.5, 0.3, 0.2), Q = (0.25, 0.6, 0.15).
    want = 0.5 * math.log(0.5 / 0.25) + 0.3 * math.log(0.3 / 0.6) + 0.2 * math.log(0.2 / 0.15)
    assert abs(kl_with_rest(rec, other) - want) < 1e-12
    think = [{"text": "plan"}, {"text": "</mm:think>"}, {"text": "Hello"}]
    assert thinking_span(think) == 2 and thinking_span(think[2:]) == 0 and thinking_span(think[:1], True) == 1


if __name__ == "__main__":
    self_test()
    print("compare.py self-test: ok")
