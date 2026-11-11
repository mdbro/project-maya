#!/usr/bin/env python3
"""Read a fidelity sweep's results and write its tables and diagrams.

    python3 tools/mm3/native_fidelity_report.py RUN_DIR            # writes RUN_DIR/report.md and RUN_DIR/figures/*.svg

Per rule and mode, over the prompts: the divergence from the unmodified router (mean, tails, same first choice), the
priced token time against today's rule (bonus 0.20 in the same mode), and the routing counters. A rule is then placed
against the plain bonus's own curve: how much faster or slower it is than the plain bonus at the same divergence.

Standard library only. The diagrams are SVG files.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

TODAY = "bonus-0.20"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e4dd"


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------------------------------------------------------------------------------------------- loading
def load(run: Path):
    design = json.loads((run / "design.json").read_text())
    rows = []
    for line in (run / "results.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return design, rows


def aggregate(run: Path, rows):
    """{(arm, mode): {...}} pooled over prompts, position-weighted; the per-position series give the pooled tails."""
    groups: dict[tuple, list] = {}
    for r in rows:
        if r.get("error") or not r.get("all"):
            continue
        groups.setdefault((r["arm"], r["mode"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        arm, mode = key
        kl, top1, lr = [], [], []
        per_prompt = {}
        for r in rs:
            path = run / "series" / f"{arm}__{mode}__{r['prompt']}.json"
            if path.exists():
                s = json.loads(path.read_text())
                kl += s["kl"]; top1 += s["top1"]; lr += s["logratio"]
            per_prompt[r["prompt"]] = r["all"]["kl_mean"]
        n_tok = sum(r["tokens"] for r in rs)
        seg = {}
        for name in ("thinking", "answer"):
            n = sum((r.get(name) or {}).get("n") or 0 for r in rs)
            seg[name] = (sum((r[name]["kl_mean"] or 0) * r[name]["n"] for r in rs if (r.get(name) or {}).get("n")) / n) if n else None
        priced = [r for r in rs if (r.get("price") or {}).get("ms_per_token")]
        ms = (sum(r["price"]["ms_per_token"] * r["price"]["tokens"] for r in priced) / sum(r["price"]["tokens"] for r in priced)) if priced else None
        route = [r["route"] for r in rs if r.get("route")]
        wr = lambda k: (sum(x[k] * r["tokens"] for x, r in zip(route, [r for r in rs if r.get("route")]) if x.get(k) is not None) / n_tok) if route else None
        prompt_route = [r["prompt_route"] for r in rs if r.get("prompt_route")]
        out[key] = {
            "arm": arm, "mode": mode, "stage": rs[0].get("stage"), "settings": rs[0].get("settings"), "prompts": len(rs), "n": len(kl),
            "kl_mean": sum(kl) / len(kl) if kl else float("nan"), "kl_median": quantile(kl, 0.5), "kl_p95": quantile(kl, 0.95),
            "kl_p99": quantile(kl, 0.99), "kl_over_0.1": sum(1 for x in kl if x > 0.1) / len(kl) if kl else float("nan"),
            "kl_over_1": sum(1 for x in kl if x > 1.0) / len(kl) if kl else float("nan"),
            "top1": sum(top1) / len(top1) if top1 else float("nan"),
            "logratio": sum(lr) / len(lr) if lr else float("nan"),
            "halved": sum(1 for x in lr if x < math.log(0.5)) / len(lr) if lr else float("nan"),
            "kl_thinking": seg["thinking"], "kl_answer": seg["answer"],
            "kl_by_prompt": per_prompt, "kl_prompt_min": min(per_prompt.values()), "kl_prompt_max": max(per_prompt.values()),
            "ms": ms, "exact": all((r.get("price") or {}).get("exact") for r in priced) if priced else None,
            "moved": wr("moved_share"), "mass_kept": wr("mass_kept_mean"), "under_08": wr("rows_under_08"), "loads": wr("loads"),
            "experts": wr("experts_per_layer"),
            "prompt_moved": (sum(x["moved_share"] for x in prompt_route) / len(prompt_route)) if prompt_route else None,
            "prompt_kept": (sum(x["mass_kept"] for x in prompt_route) / len(prompt_route)) if prompt_route else None,
            "mismatch": max((max([m or 0 for m in (r.get("mismatch") or [0, 0])]) for r in rs), default=0),
        }
    for key, a in out.items():                                  # time against today's rule in the same mode
        base = out.get((TODAY, key[1]))
        a["time"] = (a["ms"] / base["ms"] - 1.0) if (a["ms"] and base and base["ms"]) else None
    return out


def plain_curve(agg, mode):
    """The plain bonus's own curve in a mode: [(kl_mean, time)] sorted by kl, from bonus 0.05 to 0.50."""
    pts = []
    for (arm, m), a in agg.items():
        if m == mode and arm.startswith("bonus-") and a["settings"] and set(a["settings"]) == {"routing_prior_bonus"} and a["time"] is not None:
            pts.append((a["kl_mean"], a["time"], a["settings"]["routing_prior_bonus"]))
    return sorted(pts)


def against_plain(curve, kl, time):
    """Time of a rule minus the plain bonus's time at the same divergence (negative: the rule is faster at equal
    fidelity). None outside the curve's range."""
    if time is None or len(curve) < 2 or not (curve[0][0] <= kl <= curve[-1][0]):
        return None
    for (k0, t0, _), (k1, t1, _) in zip(curve, curve[1:]):
        if k0 <= kl <= k1:
            w = 0.0 if k1 == k0 else (kl - k0) / (k1 - k0)
            return time - (t0 + w * (t1 - t0))
    return None


# ---------------------------------------------------------------------------------------------------- SVG
def svg_open(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="system-ui, Segoe UI, Arial, sans-serif" font-size="12">',
            f'<rect width="{w}" height="{h}" fill="#fcfcfb"/>', f'<text x="16" y="24" font-size="15" font-weight="600" fill="{INK}">{esc(title)}</text>']


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ticks(lo, hi, n=5):
    if not hi > lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s for s in (1 * mag, 2 * mag, 2.5 * mag, 5 * mag, 10 * mag) if s >= raw)
    v = math.ceil(lo / step) * step
    out = []
    while v <= hi + step * 1e-9:
        out.append(round(v, 10)); v += step
    return out


def fmt(v):
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}" if a >= 1 else f"{v:.3f}".rstrip("0").rstrip(".") if a >= 0.001 else ("0" if v == 0 else f"{v:.0e}")


def axes(svg, box, xr, yr, xlabel, ylabel, xlog=False, ylog=False, xfmt=fmt, yfmt=fmt):
    x0, y0, w, h = box
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: x0 + w * (tx(v) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0]))
    Y = lambda v: y0 + h - h * (ty(v) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0]))
    yt = [10 ** k for k in range(math.ceil(math.log10(yr[0])), math.floor(math.log10(yr[1])) + 1)] if ylog else ticks(*yr)
    xt = [10 ** k for k in range(math.ceil(math.log10(xr[0])), math.floor(math.log10(xr[1])) + 1)] if xlog else ticks(*xr)
    for v in yt:
        svg.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        svg.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}">{yfmt(v)}</text>')
    for v in xt:
        svg.append(f'<text x="{X(v):.1f}" y="{y0 + h + 16}" text-anchor="middle" fill="{MUTED}">{xfmt(v)}</text>')
    svg.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="#bbb"/>')
    svg.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 34}" text-anchor="middle" fill="{MUTED}">{esc(xlabel)}</text>')
    svg.append(f'<text x="14" y="{y0 + h / 2}" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 14 {y0 + h / 2})">{esc(ylabel)}</text>')
    return X, Y


def figure_bonus(agg, floors, path):
    """Divergence and priced time against the bonus, both modes."""
    svg = svg_open(980, 420, "The plain bonus: divergence from the unmodified router, and priced token time")
    for col, (title, key, ylog) in enumerate((("Mean divergence, nats (log scale)", "kl_mean", True), ("Token time against today's 0.20", "time", False))):
        box = (70 + col * 480, 60, 380, 290)
        series = {}
        for mode in ("one_row", "drafting"):
            pts = sorted((a["settings"]["routing_prior_bonus"], a[key]) for (arm, m), a in agg.items()
                         if m == mode and arm.startswith("bonus-") and set(a["settings"]) == {"routing_prior_bonus"} and a[key] is not None)
            series[mode] = pts
        ys = [y for pts in series.values() for _, y in pts]
        if not ys:
            continue
        if ylog:
            lo = min(min(ys), min(floors.values(), default=1e-3) or 1e-3) / 1.5
            yr = (max(lo, 1e-5), max(ys) * 1.5)
        else:
            pad = (max(ys) - min(ys)) * 0.1 or 0.01
            yr = (min(ys) - pad, max(ys) + pad)
        X, Y = axes(svg, box, (0.0, 0.55), yr, "bonus", title, ylog=ylog, yfmt=(fmt if ylog else (lambda v: f"{100 * v:+.0f}%")))
        for i, (mode, pts) in enumerate(series.items()):
            if not pts:
                continue
            d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y) in enumerate(pts))
            svg.append(f'<path d="{d}" fill="none" stroke="{PALETTE[i]}" stroke-width="2"/>')
            for x, y in pts:
                svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3.5" fill="{PALETTE[i]}" stroke="#fcfcfb" stroke-width="1.5"/>')
            lx, ly = pts[-1]
            svg.append(f'<text x="{X(lx) + 8:.1f}" y="{Y(ly) + 4:.1f}" fill="{INK}">{"one row" if mode == "one_row" else "drafting"}</text>')
            if ylog and floors.get(mode):
                fy = Y(max(floors[mode], yr[0]))
                svg.append(f'<line x1="{box[0]}" x2="{box[0] + box[2]}" y1="{fy:.1f}" y2="{fy:.1f}" stroke="{PALETTE[i]}" stroke-dasharray="4 4"/>')
                svg.append(f'<text x="{box[0] + 4}" y="{fy - 4:.1f}" fill="{MUTED}">noise floor, {"one row" if mode == "one_row" else "drafting"}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def figure_frontier(agg, mode, path, floor):
    """Priced time against divergence for every rule in a mode; the plain bonus's curve is the line to beat."""
    pts = [a for (arm, m), a in agg.items() if m == mode and a["time"] is not None and not arm.startswith("ref_")]
    if not pts:
        return
    svg = svg_open(980, 620, f"Time against fidelity, {'one row' if mode == 'one_row' else 'drafting'}: down and to the left is better")
    kls = [max(a["kl_mean"], 1e-4) for a in pts]
    ts = [a["time"] for a in pts]
    xr = (min(kls) / 1.3, max(kls) * 1.3)
    pad = (max(ts) - min(ts)) * 0.08 or 0.01
    X, Y = axes(svg, (80, 60, 860, 480), xr, (min(ts) - pad, max(ts) + pad), "mean divergence from the unmodified router, nats (log scale)",
                "token time against today's 0.20", xlog=True, yfmt=lambda v: f"{100 * v:+.0f}%")
    curve = plain_curve(agg, mode)
    if len(curve) > 1:
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(max(kl, xr[0])):.1f},{Y(t):.1f}" for k, (kl, t, _) in enumerate(curve))
        svg.append(f'<path d="{d}" fill="none" stroke="{PALETTE[0]}" stroke-width="2"/>')
    if floor:
        fx = X(max(floor, xr[0]))
        svg.append(f'<line x1="{fx:.1f}" x2="{fx:.1f}" y1="60" y2="540" stroke="#999" stroke-dasharray="4 4"/><text x="{fx + 4:.1f}" y="74" fill="{MUTED}">noise floor</text>')
    for a in sorted(pts, key=lambda a: a["kl_mean"]):
        plain = a["arm"].startswith("bonus-") and set(a["settings"]) == {"routing_prior_bonus"}
        x, y = X(max(a["kl_mean"], xr[0])), Y(a["time"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if plain else 4.5}" fill="{PALETTE[0] if plain else PALETTE[1]}" stroke="#fcfcfb" stroke-width="1.5"/>')
        label = a["arm"].replace("bonus-", "b ").replace("_budget-", " budget ").replace("_", " ")
        svg.append(f'<text x="{x + 7:.1f}" y="{y + 4:.1f}" fill="{INK}" font-size="10.5">{esc(label)}</text>')
    svg.append(f'<rect x="700" y="68" width="10" height="10" fill="{PALETTE[0]}"/><text x="716" y="77" fill="{MUTED}">the plain bonus (its curve)</text>')
    svg.append(f'<rect x="700" y="86" width="10" height="10" fill="{PALETTE[1]}"/><text x="716" y="95" fill="{MUTED}">other rules</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def solve(a, b):
    """Least squares by the normal equations and Gaussian elimination (small systems)."""
    n = len(a[0])
    m = [[sum(r[i] * r[j] for r in a) for j in range(n)] + [sum(r[i] * y for r, y in zip(a, b))] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda k: abs(m[k][i]))
        m[i], m[p] = m[p], m[i]
        if abs(m[i][i]) < 1e-12:
            continue
        for k in range(n):
            if k != i:
                f = m[k][i] / m[i][i]
                m[k] = [x - f * y for x, y in zip(m[k], m[i])]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def quad_fit(points):
    """z = c0 + c1 x + c2 y + c3 x^2 + c4 y^2 + c5 x y. points: [(x, y, z)]. -> (coefficients, R^2)."""
    a = [[1, x, y, x * x, y * y, x * y] for x, y, _ in points]
    z = [p[2] for p in points]
    c = solve(a, z)
    pred = [sum(ci * v for ci, v in zip(c, row)) for row in a]
    mean = sum(z) / len(z)
    ss_tot = sum((v - mean) ** 2 for v in z) or 1e-12
    return c, 1 - sum((v - p) ** 2 for v, p in zip(z, pred)) / ss_tot


def figure_surface(points, path, title, xlabel, ylabel, zlabel, zfmt, log_z=False, levels=6):
    """A fitted quadratic response surface as a filled contour map with the measured points on it."""
    if len(points) < 6:
        return None
    pts = [(x, y, math.log(max(z, 1e-9)) if log_z else z) for x, y, z in points]
    c, r2 = quad_fit(pts)
    f = lambda x, y: c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * y * y + c[5] * x * y
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    xr, yr = (min(xs), max(xs)), (min(ys), max(ys))
    n = 48
    grid = [[f(xr[0] + (xr[1] - xr[0]) * i / n, yr[0] + (yr[1] - yr[0]) * j / n) for i in range(n + 1)] for j in range(n + 1)]
    zmin, zmax = min(min(r) for r in grid), max(max(r) for r in grid)
    svg = svg_open(640, 520, title)
    x0, y0, w, h = 80, 56, 440, 380
    X = lambda v: x0 + w * (v - xr[0]) / (xr[1] - xr[0])
    Y = lambda v: y0 + h - h * (v - yr[0]) / (yr[1] - yr[0])
    cw, ch = w / n, h / n
    for j in range(n):
        for i in range(n):
            v = (grid[j][i] + grid[j][i + 1] + grid[j + 1][i] + grid[j + 1][i + 1]) / 4
            k = min(7, max(0, int((v - zmin) / ((zmax - zmin) or 1) * 7.999)))
            svg.append(f'<rect x="{x0 + i * cw:.1f}" y="{y0 + h - (j + 1) * ch:.1f}" width="{cw + 0.5:.1f}" height="{ch + 0.5:.1f}" fill="{SEQ[k]}"/>')
    for li in range(1, levels):                                   # iso-lines by marching squares
        lv = zmin + (zmax - zmin) * li / levels
        segs = []
        for j in range(n):
            for i in range(n):
                corners = [(i, j, grid[j][i]), (i + 1, j, grid[j][i + 1]), (i + 1, j + 1, grid[j + 1][i + 1]), (i, j + 1, grid[j + 1][i])]
                cross = []
                for (ax, ay, av), (bx, by, bv) in zip(corners, corners[1:] + corners[:1]):
                    if (av - lv) * (bv - lv) < 0:
                        t = (lv - av) / (bv - av)
                        cross.append((ax + t * (bx - ax), ay + t * (by - ay)))
                if len(cross) >= 2:
                    segs.append((cross[0], cross[1]))
        for (ax, ay), (bx, by) in segs:
            svg.append(f'<line x1="{x0 + ax * cw:.1f}" y1="{y0 + h - ay * ch:.1f}" x2="{x0 + bx * cw:.1f}" y2="{y0 + h - by * ch:.1f}" stroke="#ffffff" stroke-width="1" opacity="0.8"/>')
        if segs:
            (ax, ay), _ = segs[len(segs) // 2]
            val = math.exp(lv) if log_z else lv
            svg.append(f'<text x="{x0 + ax * cw + 3:.1f}" y="{y0 + h - ay * ch - 3:.1f}" fill="{INK}" font-size="10.5" stroke="#fcfcfb" stroke-width="2.5" paint-order="stroke">{zfmt(val)}</text>')
    for x, y, z in points:
        svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{INK}" stroke="#fcfcfb" stroke-width="1"/>')
    for v in ticks(*xr):
        svg.append(f'<text x="{X(v):.1f}" y="{y0 + h + 16}" text-anchor="middle" fill="{MUTED}">{fmt(v)}</text>')
    for v in ticks(*yr):
        svg.append(f'<text x="{x0 - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTED}">{fmt(v)}</text>')
    svg.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 36}" text-anchor="middle" fill="{MUTED}">{esc(xlabel)}</text>')
    svg.append(f'<text x="18" y="{y0 + h / 2}" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 18 {y0 + h / 2})">{esc(ylabel)}</text>')
    for k in range(8):                                             # the scale
        svg.append(f'<rect x="548" y="{y0 + h - (k + 1) * h / 8:.1f}" width="16" height="{h / 8 + 0.5:.1f}" fill="{SEQ[k]}"/>')
    for k in (0, 4, 8):
        v = zmin + (zmax - zmin) * k / 8
        svg.append(f'<text x="570" y="{y0 + h - k * h / 8 + 4:.1f}" fill="{MUTED}">{zfmt(math.exp(v) if log_z else v)}</text>')
    svg.append(f'<text x="548" y="{y0 - 8}" fill="{MUTED}">{esc(zlabel)}</text>')
    svg.append(f'<text x="{x0}" y="{y0 + h + 58}" fill="{MUTED}">Quadratic surface fitted to the {len(points)} measured points (dots); R squared {r2:.2f}{" on the log" if log_z else ""}.</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))
    return r2


# ---------------------------------------------------------------------------------------------------- report
def pct(v, d=1):
    return "–" if v is None or v != v else f"{100 * v:.{d}f}%"


def signed(v, d=1):
    return "–" if v is None or v != v else f"{100 * v:+.{d}f}%"


def num(v, d=4):
    return "–" if v is None or v != v else f"{v:.{d}f}"


FACTOR_NAMES = {"A": "per-row budget 0.20", "B": "protect the strongest pick", "C": "priced substitution at 2", "D": "layer drop at 1.5",
                "E": "dampened weights", "F": "three experts below 0.15", "G": "five experts above 0.05", "H": "prompt read by the unmodified router"}


def screening(design, agg, mode, prompt_floor):
    """Main effects of a two-level screening design (arms with a `factors` dict): the mean response of the runs with
    the factor on less the mean of the runs with it off. Divergence as a ratio (log scale) and net of the floor;
    priced time in points of today's rule."""
    arms = [a for a in design["arms"] if a.get("factors") and a["mode"] == mode and (a["id"], mode) in agg]
    if len(arms) < 8:
        return ""
    def net(a):
        g = agg[(a["id"], mode)]
        shared = [q for q in g["kl_by_prompt"] if prompt_floor and q in prompt_floor]
        return (sum(g["kl_by_prompt"][q] - prompt_floor[q][0] for q in shared) / len(shared)) if shared else g["kl_mean"]
    lines = [f"**Screening design, main effects ({len(arms)} runs; each factor on against off, the other seven balanced)**", "",
             "| Factor | Divergence, on / off | Divergence over the floor, on less off | Time against 0.20, on less off | Picks moved, on less off |",
             "|---|---|---|---|---|"]
    for f in sorted({k for a in arms for k in a["factors"]}):
        on = [a for a in arms if a["factors"][f]]; off = [a for a in arms if not a["factors"][f]]
        if not on or not off:
            continue
        m = lambda xs, fn: sum(fn(a) for a in xs) / len(xs)
        lk = lambda a: math.log(max(agg[(a["id"], mode)]["kl_mean"], 1e-9))
        tm = lambda a: agg[(a["id"], mode)]["time"] or 0.0
        mv = lambda a: agg[(a["id"], mode)]["moved"] or 0.0
        lines.append(f"| {f}: {FACTOR_NAMES.get(f, f)} | x{math.exp(m(on, lk) - m(off, lk)):.2f} | {m(on, net) - m(off, net):+.4f} | "
                     f"{100 * (m(on, tm) - m(off, tm)):+.1f} pt | {100 * (m(on, mv) - m(off, mv)):+.1f} pt |")
    return "\n".join(lines) + "\n"


def table(agg, keys, mode, curve, title, prompt_floor=None):
    """prompt_floor: {prompt: (mean, highest)} of the floor controls in this mode, if the design has them."""
    lines = [f"**{title}**", "",
             "| Rule | Mean divergence | Over the floor | Prompts above their floor | Across prompts | 95th pct | Same first choice | Thinking / answer | Time against 0.20 | Against plain bonus at equal fidelity | Picks moved | Mass kept | Experts a layer |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in keys:
        a = agg.get((arm, mode))
        if not a:
            continue
        vs = against_plain(curve, a["kl_mean"], a["time"])
        plain = arm.startswith("bonus-") and set(a["settings"]) == {"routing_prior_bonus"}
        over, above = "–", "–"
        if prompt_floor:
            shared = [q for q in a["kl_by_prompt"] if q in prompt_floor]
            if shared:
                net = sum(a["kl_by_prompt"][q] - prompt_floor[q][0] for q in shared) / len(shared)
                over = f"{net:+.4f}"
                above = f"{sum(1 for q in shared if a['kl_by_prompt'][q] > prompt_floor[q][1])} of {len(shared)}"
        lines.append(f"| {arm} | {num(a['kl_mean'])} | {over} | {above} | {num(a['kl_prompt_min'], 3)} to {num(a['kl_prompt_max'], 3)} | {num(a['kl_p95'], 3)} | {pct(a['top1'])} | "
                     f"{num(a['kl_thinking'], 3)} / {num(a['kl_answer'], 3)} | {signed(a['time'])} | {'(the curve)' if plain else ('–' if vs is None else f'{100 * vs:+.1f} pt')} | "
                     f"{pct(a['moved'])} | {pct(a['mass_kept'])} | {num(a['experts'], 2)} |")
    return "\n".join(lines) + "\n"


def main():
    run = Path(sys.argv[1])
    design, rows = load(run)
    agg = aggregate(run, rows)
    figs = run / "figures"
    figs.mkdir(exist_ok=True)
    failed = [r for r in rows if r.get("error")]
    floors, prompt_floors, out = {}, {}, ["# Fidelity sweep: results", ""]
    out.append(f"{len([r for r in rows if not r.get('error')])} replays recorded, {len(failed)} failed. Every number is against the unmodified router (bonus 0) on the same "
               "forced tokens, from the same restored cache state. Time is priced from pass counts, not measured.\n")
    for mode in ("one_row", "drafting"):
        fl = [agg[k] for k in agg if k[1] == mode and k[0] in ("ref_b", "ref_c")]
        if fl:
            n = sum(a["n"] for a in fl)
            floors[mode] = sum(a["kl_mean"] * a["n"] for a in fl) / n
            out.append(f"- **Repeat of the reference, {'one row' if mode == 'one_row' else 'drafting'}:** mean {floors[mode]:.4f} nats over {n} positions "
                       f"(reference against itself, two repeats a prompt); 95th percentile {max(a['kl_p95'] for a in fl):.3f}; per prompt "
                       f"{min(v for a in fl for v in a['kl_by_prompt'].values()):.4f} to {max(v for a in fl for v in a['kl_by_prompt'].values()):.4f}. "
                       "A repeat with nothing changed is nearly exact and is not the meter's floor.")
        ctl = [agg[k] for k in agg if k[1] == mode and k[0].startswith("floor_")]
        if ctl:
            per = {}
            for a in ctl:
                for q, v in a["kl_by_prompt"].items():
                    per.setdefault(q, []).append(v)
            prompt_floors[mode] = {q: (sum(v) / len(v), max(v)) for q, v in per.items()}
            n = sum(a["n"] for a in ctl)
            floors[mode] = sum(a["kl_mean"] * a["n"] for a in ctl) / n
            out.append(f"- **The floor, {'one row' if mode == 'one_row' else 'drafting'}:** mean {floors[mode]:.4f} nats over {len(ctl)} controls "
                       "(bonus 0, the same experts, only the lane that computes some of them changed); by prompt "
                       + ", ".join(f"{q} {m:.4f} (highest {hi:.4f})" for q, (m, hi) in sorted(prompt_floors[mode].items())) + ". "
                       "\"Over the floor\" is a rule's mean divergence less its prompts' floor; \"above their floor\" counts the prompts "
                       "where the rule exceeds every control.")
    bad = [k for k, a in agg.items() if a["mismatch"]]
    out.append(f"- **Engine self-check:** {'no mismatch against the serving kernel in any run' if not bad else 'MISMATCH in ' + str(bad)}.\n")
    figure_bonus(agg, floors, figs / "bonus.svg")
    for mode in ("one_row", "drafting"):
        curve = plain_curve(agg, mode)
        name = "One row" if mode == "one_row" else "Drafting"
        arms = [a["id"] for a in design["arms"] if a["mode"] == mode]
        seen, order = set(), []
        for x in arms:
            if x not in seen:
                seen.add(x); order.append(x)
        stages = {}
        for x in order:
            stages.setdefault(next(a["stage"] for a in design["arms"] if a["id"] == x and a["mode"] == mode), []).append(x)
        out.append(f"## {name}\n")
        for stage, keys in stages.items():
            out.append(table(agg, keys, mode, curve, f"Stage {stage}", prompt_floors.get(mode)))
        scr = screening(design, agg, mode, prompt_floors.get(mode))
        if scr:
            out.append(scr)
        figure_frontier(agg, mode, figs / f"frontier-{mode}.svg", floors.get(mode))
        grid = [(a["settings"]["routing_prior_bonus"], 1 - a["settings"]["route_floor"], a) for (arm, m), a in agg.items()
                if m == mode and a["stage"] == "2b" and a["time"] is not None]
        if len(grid) >= 6:
            r1 = figure_surface([(b, g, a["kl_mean"]) for b, g, a in grid], figs / f"surface-divergence-{mode}.svg",
                                f"Response surface, {name.lower()}: divergence over bonus and budget", "bonus", "per-row budget on lost router mass",
                                "nats", lambda v: fmt(v), log_z=True)
            r2 = figure_surface([(b, g, a["time"]) for b, g, a in grid], figs / f"surface-time-{mode}.svg",
                                f"Response surface, {name.lower()}: token time over bonus and budget", "bonus", "per-row budget on lost router mass",
                                "against 0.20", lambda v: f"{100 * v:+.0f}%")
            out.append(f"Response surfaces written for {len(grid)} grid points (R squared {r1:.2f} for log divergence, {r2:.2f} for time).\n")
    if failed:
        out.append("## Failed runs\n")
        for r in failed[:40]:
            out.append(f"- {r['arm']} {r['mode']} {r['prompt']}: {r['error'][:200]}")
    (run / "report.md").write_text("\n".join(out) + "\n")
    (run / "aggregate.json").write_text(json.dumps({f"{k[0]}@{k[1]}": v for k, v in agg.items()}, indent=1))
    print(f"report: {run / 'report.md'}; figures in {figs}; {len(agg)} rule-mode groups")


if __name__ == "__main__":
    main()
