"""Separate evidence gates. A shared-policy check never grants timing validity."""
from __future__ import annotations
import hashlib
import json
import math


def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def timing_validation(prediction, observations, *, required_effect_fraction):
    """Check held-out component AND request residuals at decision-sized precision."""
    if not 0 < required_effect_fraction < 1: raise ValueError('declare a positive decision-sized effect')
    if prediction['physical_sha256']!=observations['physical_sha256']:
        raise ValueError('observations refer to a different preregistered model')
    calibration=set(observations['calibration_request_ids'])
    held_out=set(observations['held_out_request_ids'])
    if not held_out or calibration&held_out: raise ValueError('timing holdout leaks into calibration')
    # Categories cannot be silently waived by reporting only whole-request fit.
    required={'grouped_service','queue_delay','publication','branch_finish','request_duration'}
    rows=observations['comparisons'];categories={r['category'] for r in rows}
    if not required<=categories: raise ValueError('missing held-out timing categories: '+str(required-categories))
    comparisons=[]
    for row in rows:
        if row['request_id'] not in held_out: raise ValueError('comparison outside held-out execution')
        ref=row['prediction_ref']
        if ref['scope']=='request':
            expected=next(r for r in prediction['requests'] if r['request_id']==ref['request_id'])[ref['metric']]
        elif ref['scope']=='operation':
            op=prediction['operations'][ref['id']]
            if op['id']!=ref['id']:raise ValueError('prediction operation identity differs')
            if ref['metric']=='duration_ms':expected=op['finish_ms']-op['start_ms']
            else:expected=op[ref['metric']]
        elif ref['scope']=='branch':
            begin=prediction['operations'][ref['begin_id']];end=prediction['operations'][ref['end_id']]
            expected=end['finish_ms']-begin['start_ms']
        elif ref['scope']=='publication':
            action=prediction['actions'][ref['index']]
            if action['kind']!='primary_publish':raise ValueError('not a publication prediction')
            expected=action['time_ms']-ref.get('origin_ms',0)
        else:raise ValueError('unsupported prediction reference')
        if 'predicted_ms' in row and not math.isclose(row['predicted_ms'],expected,rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('comparison substituted a different prediction')
        actual=row['observed_ms']
        uncertainty=row['uncertainty_ms']
        if not all(math.isfinite(v) and v>=0 for v in (expected,actual,uncertainty)) or actual==0:
            raise ValueError('invalid held-out time')
        relative=abs(expected-actual)/actual
        comparisons.append({**row,'predicted_ms':expected,'relative_error':relative,'relative_uncertainty':uncertainty/actual,
                            'resolved':max(relative,uncertainty/actual)<required_effect_fraction})
    qualified=all(r['resolved'] for r in comparisons) and not prediction.get('coverage_gaps')
    return dict(stage=2,status='pass' if qualified else 'unresolved',required_effect_fraction=required_effect_fraction,
        physical_sha256=prediction['physical_sha256'],prediction_sha256=digest(prediction),
        comparisons=comparisons,coverage_gaps=prediction.get('coverage_gaps',[]),
        scope='independent held-out execution prediction, including local and whole-request timing')


def intervention_validation(plan, hardware, *, timing_gate):
    """Match predictions committed BEFORE hardware results; retain every arm."""
    required={'backing','reduced_admissions','staging_amount'}
    if timing_gate.get('stage')!=2 or timing_gate['physical_sha256']!=plan['physical_sha256']:
        raise ValueError('intervention model differs from independently tested model')
    if hardware.get('prediction_sha256')!=digest(plan): raise ValueError('hardware not bound to prior predictions')
    if not required<={a['kind'] for a in plan['interventions']}: raise ValueError('required interventions missing')
    if hardware['started_unix_ns']<=plan['created_unix_ns']: raise ValueError('prediction was not prospective')
    evidence={a['id']:a for a in hardware['interventions']};results=[]
    for proposed in plan['interventions']:
        measured=evidence.get(proposed['id'])
        if measured is None: raise ValueError('missing hardware intervention')
        if (measured['workload_sha256']!=plan['workload_sha256'] or
            measured['initial_state_sha256']!=plan['initial_state_sha256']):
            raise ValueError('unmatched fixed expert work or initial state')
        predicted=proposed['delta_ms'];actual=measured['delta_ms']
        uncertainty=proposed['uncertainty_ms']+measured['uncertainty_ms']
        agreement=abs(actual-predicted)<=uncertainty
        resolved=abs(predicted)>uncertainty and agreement and predicted*actual>0
        results.append(dict(id=proposed['id'],kind=proposed['kind'],predicted_delta_ms=predicted,
            observed_delta_ms=actual,combined_uncertainty_ms=uncertainty,agreement=agreement,resolved=resolved))
    passed=timing_gate['status']=='pass' and all(r['resolved'] for r in results)
    return dict(stage=3,status='pass' if passed else 'unresolved',physical_sha256=plan['physical_sha256'],
                prediction_sha256=digest(plan),results=results,scope='prospective fixed-work hardware interventions')


def require_search_gates(gates,physical_sha256):
    by_stage={g['stage']:g for g in gates}
    for stage in (1,2,3):
        g=by_stage.get(stage)
        if not g or g.get('status')!='pass': raise ValueError(f'offline parameter ranking requires gate {stage}')
        if stage>1 and g.get('physical_sha256')!=physical_sha256:
            raise ValueError('validation gates belong to different physical measurements')


def rank_held_out(results, incumbent):
    """Recorded emitted work is fixed; hits/bytes never enter the objective."""
    baseline=next(r for r in results if r['name']==incumbent)
    base={r['request_id']:r for r in baseline['requests']}
    ranked=[]
    for candidate in results:
        tasks=[]
        if {r['request_id'] for r in candidate['requests']}!=set(base):raise ValueError('held-out requests differ')
        for task in candidate['requests']:
            control=base[task['request_id']]
            if task['emitted']!=control['emitted']:raise ValueError('recorded emitted work differs')
            gain=control['generation_ms']-task['generation_ms']
            uncertainty=task['uncertainty_ms']+control['uncertainty_ms']
            tasks.append(dict(request_id=task['request_id'],gain_ms=gain,uncertainty_ms=uncertainty,
                              robust_gain=gain>uncertainty))
        total=sum(t['generation_ms'] for t in candidate['requests'])
        total_gain=sum(t['gain_ms'] for t in tasks);uncertainty=sum(t['uncertainty_ms'] for t in tasks)
        ranked.append(dict(name=candidate['name'],generation_ms=total,gain_ms=total_gain,uncertainty_ms=uncertainty,
            retained=total_gain>uncertainty and all(t['gain_ms']>=-t['uncertainty_ms'] for t in tasks),tasks=tasks))
    return sorted(ranked,key=lambda r:r['generation_ms'])
