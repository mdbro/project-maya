"""Per-pass rows for the bandwidth-aware pass model (mm3_pass_model.hpp), from format-2 light captures.

One row per layer pass of a generation forward at token 128 and later: what each lane was given (the planner's view at
its decision, the plan's end) and when each lane finished, in ms after the plan's end. Read only.

Lanes and their measured finish (ms after the plan's end; None where the lane had no job or no usable stamp):
  cpu       cpu_end (the host's CPU batch)
  3090      helper_worker_published (the 3090's worker published its result)
  4080      the third pass's worker_published (the 4080's worker published its result)
  upload    gpu_upload_end (the 5090 passed its uploads; with no upload, the expert graph's start)
  5090      gpu_expert_end (the 5090's own experts done)
  pass      gpu_wait_end (the 5090's wait for the host's publication ended: the pass's experts are all in)
The 5090 stamps count only where correct_gpu_stamps found the forward's clock consistent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import native_light as nl  # noqa: E402

MS = 1e-6
PRIMARY = ('primary_resident', 'staged_ready', 'staged_new')


@dataclass
class PassRow:
    source: str
    forward: int
    layer: int
    rows: int
    # what each lane was given (distinct experts and columns = picks)
    cpu_experts: int
    cpu_columns: int
    helper_experts: int
    helper_columns: int
    third_experts: int
    third_columns: int
    g_experts: int          # the 5090's distinct experts, resident, ready and new
    g_picks: int
    g_new: int              # distinct experts uploaded in this pass (staged_new)
    g_ready: int            # distinct experts already in staging frames (staged_ready)
    uploads: int            # the 5090's upload count (gpu_uploads)
    exchanges_pending: int
    helper_lease: int
    stage_max: int          # the staging planner's limit this pass (new uploads at most)
    cpu_uses: list          # columns of each CPU-lane expert, by expert id
    new_uses: list          # columns of each expert uploaded this pass, by expert id
    gpu_uses: list          # columns of each other 5090 expert (resident or ready), by expert id
    helper_uses: list       # columns of each 3090 expert, by expert id
    # measured finish, ms after the plan's end
    cpu: float | None
    helper: float | None
    third: float | None
    upload: float | None
    g: float | None
    wait: float | None
    collect: float | None
    # host stamps, ms after the plan's end (inputs to the start offsets)
    helper_submit: float | None
    primary_submit: float | None
    first_copy_submit: float | None
    third_enqueue: float | None
    gpu_ok: bool


def uses_of(svc: list, kinds: tuple) -> list[int]:
    count: dict[int, int] = {}
    for e, s in svc:
        if s in kinds:
            count[e] = count.get(e, 0) + 1
    return [count[e] for e in sorted(count)]


def rows_of(path: str | pathlib.Path, from_token: int = 128) -> list[PassRow]:
    cap = nl.read(path)
    nl.correct_gpu_stamps(cap)
    gen = sorted(cap.generation_forwards(), key=lambda f: f['position'])
    if not gen:
        return []
    first = gen[0]['position']
    keep = {f['forward'] for f in gen if f['position'] - first >= from_token}
    third = {(t['forward'], t['layer']): t for t in cap.of('third_pass')}
    copies: dict[tuple[int, int], int] = {}
    for c in cap.of('demand_copy'):
        k = (c['forward'], c['layer'])
        t = c['t_ns'] + c['submit_end']
        copies[k] = min(copies.get(k, t), t)
    out = []
    for r in cap.of('layer'):
        if r['forward'] not in keep or not r['service']:
            continue
        base = r['t_ns'] + r['plan_end']
        rel = lambda name: (r['t_ns'] + r[name] - base) * MS if r.get(name) else None
        svc = list(zip(r['executed'], r['service']))
        lane = lambda kinds: {e for e, s in svc if s in kinds}
        cols = lambda kinds: sum(1 for e, s in svc if s in kinds)
        t = third.get((r['forward'], r['layer']))
        ok = bool(r.get('gpu_clock_consistent', True)) and bool(r.get('gpu_expert_end'))
        k = (r['forward'], r['layer'])
        out.append(PassRow(
            source=str(path), forward=r['forward'], layer=r['layer'], rows=r['rows'],
            cpu_experts=len(lane(('cpu',))), cpu_columns=cols(('cpu',)),
            helper_experts=len(lane(('helper',))), helper_columns=cols(('helper',)),
            third_experts=len(lane(('third',))), third_columns=cols(('third',)),
            g_experts=len(lane(PRIMARY)), g_picks=cols(PRIMARY),
            g_new=len(lane(('staged_new',))), g_ready=len(lane(('staged_ready',))),
            uploads=int(r.get('gpu_uploads', 0)),
            exchanges_pending=int(r.get('exchanges_pending', 0)), helper_lease=int(r.get('helper_lease', 0)),
            stage_max=int(r.get('stage_max', 0)),
            cpu_uses=uses_of(svc, ('cpu',)), new_uses=uses_of(svc, ('staged_new',)),
            gpu_uses=uses_of(svc, ('primary_resident', 'staged_ready')), helper_uses=uses_of(svc, ('helper',)),
            cpu=rel('cpu_end'), helper=rel('helper_worker_published'),
            third=((t['t_ns'] + t['worker_published'] - base) * MS) if t and t.get('worker_published') else None,
            upload=rel('gpu_upload_end') if ok else None, g=rel('gpu_expert_end') if ok else None,
            wait=rel('gpu_wait_end') if ok else None, collect=rel('collect_end'),
            helper_submit=rel('helper_submit_end'), primary_submit=rel('primary_submit_end'),
            first_copy_submit=((copies[k] - base) * MS) if k in copies else None,
            third_enqueue=((t['t_ns'] + t['enqueue'] - base) * MS) if t and t.get('enqueue') else None,
            gpu_ok=ok))
    return out


def reuse_counts(path: str | pathlib.Path, window: int = 8, from_token: int = 128) -> dict[int, list[int]]:
    """Uploads and their later uses from the frame, by path (0: windows of 2+ rows, 1: one row): each staged_new expert,
    and the passes of the same layer within `window` generation forwards that read it as staged_ready. {path: [copies,
    reuses]}"""
    cap = nl.read(path)
    gen = sorted(cap.generation_forwards(), key=lambda f: f['position'])
    if not gen:
        return {0: [0, 0], 1: [0, 0]}
    first = gen[0]['position']
    order = {f['forward']: i for i, f in enumerate(gen)}
    keep = {f['forward'] for f in gen if f['position'] - first >= from_token}
    last: dict[tuple[int, int], tuple[int, int]] = {}   # (layer, expert) -> (forward index of its copy, its path)
    out = {0: [0, 0], 1: [0, 0]}
    for r in sorted((r for r in cap.of('layer') if r['forward'] in order), key=lambda r: (order[r['forward']], r['layer'])):
        i = order[r['forward']]
        seen = set()
        for e, s in zip(r['executed'], r['service']):
            if (e, s) in seen:
                continue
            seen.add((e, s))
            if s == 'staged_new':
                p = 1 if r['rows'] == 1 else 0
                last[(r['layer'], e)] = (i, p)
                if r['forward'] in keep:
                    out[p][0] += 1
            elif s == 'staged_ready':
                j = last.get((r['layer'], e))
                if j is not None and i - j[0] <= window and gen[j[0]]['forward'] in keep:
                    out[j[1]][1] += 1
    return out


def forward_span(path: str | pathlib.Path, from_token: int = 128) -> float:
    """ms from the first generation forward at token `from_token` or later to the last one's end."""
    gen = sorted(nl.read(path).generation_forwards(), key=lambda f: f['position'])
    if not gen:
        return 0.0
    kept = [f for f in gen if f['position'] - gen[0]['position'] >= from_token]
    return (kept[-1]['t_ns'] + kept[-1]['end'] - kept[0]['t_ns']) * MS if kept else 0.0


def as_dicts(rows: list[PassRow]) -> list[dict]:
    return [asdict(r) for r in rows]
