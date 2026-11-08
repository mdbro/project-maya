"""Pass-by-pass validation of the shared timing model: every path's completion, the slowest path, and per-layer balance.

Earlier validation judged whole-request totals and candidate-minus-control differences. That cannot show whether the model
gets individual passes right, which is where balance is decided. Here each pass's predicted completion of each path, from the
pass's common origin, is compared with what that pass recorded:

  records  CPU end and helper end per pass are direct observations; the cycle is join to join
  trace    adds the 5090's copy end and its own end, and the helper's copy and kernel ends

and the per-layer mean abs(delta) between paths, the user's balance measure, is compared predicted against observed.

The 5090's own path follows the trace's structure (fit_structure): demand copies start shortly after the lanes, run at the shared
bandwidth rates, and the first dependent kernel starts a fixed handoff after the last copy ends, whether or not the CPU is still
working. The host has issued those kernels about 0.9 ms earlier. Passes with no upload end when the resident kernels end.

An earlier rule held copy-dependent kernels until the host's next CUDA call at the CPU end. It is withdrawn: the handoff stays
at about 0.057 ms when the CPU ended before the copy, and the host makes CUDA calls inside the gap.
"""
from __future__ import annotations
from collections import defaultdict
import json
import statistics

from .timing import SharedTiming


def helper_mean(model: SharedTiming, jobs: int) -> float | None:
    if not jobs or not model.helper:
        return None
    draws = model.helper[str(min(jobs, max(int(k) for k in model.helper)))]
    return statistics.fmean(draws)


def detailed_work(path) -> dict:
    """Work per (forward, layer) from a detailed capture: distinct experts per service, and uploads from the staging decision.

    A fixture replay re-decides staging, so traced passes take their work from the replay's own capture, not the source's."""
    work: dict = {}
    uploads: dict = {}
    with open(path) as f:
        for line in f:
            if '"expert_assignment"' not in line and '"batch_staging_decision"' not in line:
                continue
            d = json.loads(line)
            key = (d['forward'], d['layer'])
            if d['type'] == 'batch_staging_decision':
                uploads[key] = int(d.get('distinct_uploads', 0))
                continue
            services: dict[str, set] = {}
            for a in d['assignments']:
                services.setdefault(a['service'], set()).add(a['expert'])
            work[key] = dict(rows=1 + max(a['row'] for a in d['assignments']), cpu=len(services.get('cpu', ())), helper=len(services.get('helper', ())),
                             primary=sum(len(v) for k, v in services.items() if k.startswith('primary')))
    for key, w in work.items():
        w['new'] = uploads.get(key, 0)
    return work


def _median_by(groups: dict) -> dict:
    return {str(k): statistics.median(v) for k, v in groups.items() if len(v) >= 10}


def fit_structure(windows: dict, phases: dict, work: dict) -> dict:
    """Medians of the 5090's own path from one trace: copy start after the lanes begin, handoff from the last copy to the first
    dependent kernel (CPU still working or not), span of the dependent kernels by 5090 experts, and the resident-only end."""
    start, handoff, span, resident, helper_tail = (defaultdict(list) for _ in range(5))
    for (forward, layer), w in windows.items():
        cpu = phases.get((forward, layer, 'cpu_expert_batch')); helper = phases.get((forward, layer, 'helper_collect'))
        lanes = [x for x in (cpu, helper) if x]
        wk = work.get((forward, layer))
        if not lanes or not wk or wk['rows'] != 1:
            continue
        begin = min(x[0] for x in lanes)
        if w['helper'] and w['helper_copies']:
            helper_tail[wk['helper']].append((max(b for _, b in w['helper_copies']) - max(b for _, b in w['helper'])) * 1e-6)
        if w['copies']:
            last = max(b for _, b in w['copies'])
            after = sorted(x for x in w['expert'] if x[0] >= last)
            if not after:
                continue
            start[min(wk['new'], 2)].append((min(a for a, _ in w['copies']) - begin) * 1e-6)
            handoff[bool(cpu and cpu[1] > last)].append((after[0][0] - last) * 1e-6)
            span[wk['primary']].append((max(b for _, b in after) - after[0][0]) * 1e-6)
        elif w['expert']:
            resident[wk['primary']].append((max(b for _, b in w['expert']) - begin) * 1e-6)
    return dict(copy_start=_median_by(start), handoff=_median_by(handoff), span=_median_by(span), resident_end=_median_by(resident),
                helper_tail=_median_by(helper_tail))


def _lookup(table: dict, key) -> float:
    """The median for this key, else the nearest numeric key's, else the mean of all."""
    if str(key) in table:
        return table[str(key)]
    numeric = [int(k) for k in table if k.lstrip('-').isdigit()]
    if numeric and not isinstance(key, bool):
        return table[str(min(numeric, key=lambda k: abs(k - int(key))))]
    return statistics.fmean(table.values())


def predict_pass(model: SharedTiming, cpu: int, helper: int, primary: int, new: int, *, structure: dict | None = None,
                 exchange: bool = False, lease: bool = False) -> dict:
    """Each path's predicted completion from the common origin, and the cycle.

    structure: the 5090's own path as traced (fit_structure). Without it, the fitted gpu_base term is used.
    """
    cpu_end, copy_end = model.overlap_ends(cpu, new)
    help_end = helper_mean(model, helper)
    transfer = compute = None
    if structure and new:
        copy_end = _lookup(structure['copy_start'], min(new, 2)) + copy_end
        transfer = copy_end + _lookup(structure['handoff'], bool(cpu and cpu_end > copy_end)) + _lookup(structure['span'], max(primary, 1))
    elif structure and primary:
        compute = _lookup(structure['resident_end'], primary)
    elif primary or new:
        transfer = model.gpu_base + model.gpu_per_primary * primary + copy_end
    primary_end = transfer if transfer is not None else compute
    key = '1' if helper else '0'
    lanes = model.lanes_ms(cpu, helper, new) + model.finish.get(key, 0.0)
    unhidden = max(0.0, (primary_end or 0.0) - lanes - model.resume)
    cycle = (model.prepare[key] + lanes + model.resume + model.wait_base + unhidden + model.level +
             (model.exchange_excess if exchange else 0.0) + (model.lease_excess if lease else 0.0))
    helper_compute = None
    if help_end is not None and structure and structure.get('helper_tail'):
        helper_compute = help_end - _lookup(structure['helper_tail'], helper)
    ends = dict(cpu=cpu_end if cpu else None, primary_transfer=transfer, primary_compute=compute, helper_transfer=help_end, helper_compute=helper_compute)
    lanes_end = dict(cpu=ends['cpu'], primary=primary_end, helper=help_end)
    present = {k: v for k, v in lanes_end.items() if v is not None}
    return dict(ends=ends, lanes=lanes_end, copy_end=copy_end if new else None, cycle=cycle,
                slowest=max(present, key=present.get) if len(present) >= 2 else None)


def records_comparison(model: SharedTiming, rows: list, *, structure: dict | None = None) -> dict:
    """Pass by pass against full records: CPU end, helper end, cycle, which of CPU and helper is last, per-layer balance."""
    errors = defaultdict(list)
    agree = total = 0
    layer_obs, layer_pred = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.final_layer or r.index == 0 or r.rows != 1:
            continue
        p = predict_pass(model, r.cpu_jobs, r.helper_jobs, r.primary_jobs, r.new, structure=structure,
                         exchange=bool(r.exchanges_pending), lease=bool(r.helper_lease))
        errors['cycle'].append(p['cycle'] - r.cycle_ms)
        cpu = r.cpu_end - r.lane_begin if r.cpu_end is not None else None
        helper = r.helper_published - r.lane_begin if r.helper_published is not None else None
        if cpu is not None and p['ends']['cpu'] is not None:
            errors['cpu_end'].append(p['ends']['cpu'] - cpu)
        if helper is not None and p['ends']['helper_transfer'] is not None:
            errors['helper_end'].append(p['ends']['helper_transfer'] - helper)
        if cpu is not None and helper is not None and p['ends']['cpu'] is not None and p['ends']['helper_transfer'] is not None:
            total += 1
            agree += (cpu >= helper) == (p['ends']['cpu'] >= p['ends']['helper_transfer'])
            layer_obs[r.layer].append(abs(cpu - helper))
            layer_pred[r.layer].append(abs(p['ends']['cpu'] - p['ends']['helper_transfer']))
    layers = {k: (statistics.fmean(layer_obs[k]), statistics.fmean(layer_pred[k])) for k in layer_obs if len(layer_obs[k]) >= 10}
    gap = [abs(a - b) for a, b in layers.values()]
    return dict(passes=len(errors['cycle']), errors={k: summary(v) for k, v in errors.items()},
                cpu_helper_order_agreement=agree / total if total else None,
                layer_cpu_helper_abs_delta=dict(observed_mean=statistics.fmean(a for a, _ in layers.values()) if layers else None,
                                                predicted_mean=statistics.fmean(b for _, b in layers.values()) if layers else None,
                                                per_layer_abs_error=summary(gap)))


def trace_comparison(model: SharedTiming, trace_rows: list[dict], work: dict, *, structure: dict | None = None) -> dict:
    """Pass by pass against a trace's raw path ends: every path's end, the slowest lane, and predicted rows for per-layer balance.

    Trace durations include instrumentation. Only single-row passes the replay's capture describes are judged."""
    errors = defaultdict(list)
    agree = total = 0
    confusion = defaultdict(int)
    near_ties = []
    predicted_rows, observed_rows = [], []
    for t in trace_rows:
        w = work.get((t['forward'], t['layer']))
        if not w or w['rows'] != 1:
            continue
        p = predict_pass(model, w['cpu'], w['helper'], w['primary'], w['new'], structure=structure)
        for path, value in p['ends'].items():
            if value is not None and t['ends'].get(path) is not None:
                errors[path].append(value - t['ends'][path])
        predicted_rows.append(dict(forward=t['forward'], layer=t['layer'], ends=p['ends'], lanes=p['lanes']))
        observed_rows.append(t)
        present = {k: v for k, v in t['lanes'].items() if v is not None}
        if len(present) >= 2 and p['slowest']:
            total += 1
            observed = max(present, key=present.get)
            agree += observed == p['slowest']
            confusion[f'{observed}->{p["slowest"]}'] += 1
            if observed != p['slowest']:
                ordered = sorted(present.values(), reverse=True)
                near_ties.append(ordered[0] - ordered[1])
    return dict(passes=total, errors={k: summary(v) for k, v in errors.items()}, slowest_agreement=agree / total if total else None,
                confusion=dict(confusion), disagreement_margin=summary(near_ties), predicted_rows=predicted_rows, observed_rows=observed_rows)


def summary(values: list[float]) -> dict:
    if not values:
        return dict(n=0)
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.fmean(values), mean_abs=statistics.fmean(abs(v) for v in values),
                p10=ordered[len(ordered) // 10], p50=ordered[len(ordered) // 2], p90=ordered[9 * len(ordered) // 10])
