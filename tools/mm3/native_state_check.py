#!/usr/bin/env python3
"""Public-server forced-rejection and matching-prefix diagnostic; not a speed test."""
import argparse
import datetime as dt
import json
import math
import pathlib
import sys

import native_calibrate as calibration
import native_screen as screen

ROOT=pathlib.Path(__file__).resolve().parents[2]
PROMPT=('Explain in several short numbered steps how to make a simple vegetable soup. '
        'Include preparation, cooking, and serving.')


def check_verifier(item, policy, factor):
    stats=item.get('engine_stats') or {}
    if stats.get('join')!='one_new_sequence_and_matching_completion_count':
        raise ValueError('verifier engine/API count attribution failed')
    final=stats.get('final') or {}
    key='spec_ell' if policy=='lenience' else 'spec_rho'
    source='history'
    if final.get('spec_policy') is None or final.get(key) is None:
        final=((item.get('metrics_after') or {}).get('tiers') or {}).get('now') or {}
        source='metrics_after.tiers.now'
        if (final.get('request_seq') != stats.get('request_seq') or
            final.get('generation_final') != 1 or
            final.get('emitted') != (item.get('usage') or {}).get('completion_tokens')):
            raise ValueError('verifier fallback is not the matching completed request')
    actual=final.get(key)
    if final.get('spec_policy') != policy or isinstance(actual,bool) or not isinstance(actual,(int,float)) or not math.isclose(actual,factor,rel_tol=0,abs_tol=1e-6):
        raise ValueError('effective verifier or factor does not match the request')
    return {'policy':policy, key:actual, 'source':source}


def check_forced(item):
    stats=item.get('engine_stats') or {}
    if stats.get('join')!='one_new_sequence_and_matching_completion_count':
        raise ValueError('forced request engine/API count attribution failed')
    final=stats.get('final') or {}
    if final.get('forced_rejections')!=1:
        raise ValueError('exactly one forced rejection was not observed')
    emitted=final.get('forced_rejection_emitted');frontier=final.get('forced_rejection_frontier')
    rejection_round=final.get('forced_rejection_round')
    if any(not isinstance(v,(int,float)) or v<0 for v in (emitted,frontier,rejection_round)):
        raise ValueError('rejection frontier or round missing')
    if final.get('emitted',0)<=emitted or final.get('target_rounds',0)<=rejection_round:
        raise ValueError('continuation after rejection was not observed')
    return {'forced_rejections':1,'emitted_at_rejection':emitted,'frontier_at_rejection':frontier,
            'target_round_at_rejection':rejection_round,'final_emitted':final['emitted'],
            'final_target_rounds':final['target_rounds'],'natural_accept':final.get('forced_natural_accept')}


def check_reuse(item):
    stats=item.get('engine_stats') or {}
    if stats.get('join')!='one_new_sequence_and_matching_completion_count':
        raise ValueError('follow-up engine/API count attribution failed')
    details=(item.get('usage') or {}).get('prompt_tokens_details') or {}
    timing=item.get('timings') or {}
    cached=timing.get('cache_n',details.get('cached_tokens'))
    if not isinstance(cached,(int,float)) or cached<=0:
        raise ValueError('matching prompt did not report prefix reuse')
    final=stats.get('final') or {}
    if final.get('forced_rejections')!=0:
        raise ValueError('follow-up unexpectedly forced a rejection')
    return {'reused_prompt_tokens':cached,'final_emitted':final.get('emitted')}


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',default='http://127.0.0.1:8097')
    ap.add_argument('--output-dir',type=pathlib.Path,required=True)
    ap.add_argument('--timeout',type=float,default=900)
    ap.add_argument('--seed',type=int,default=20261102)
    ap.add_argument('--spec-policy',choices=('lenience','fast_hsd'),default='lenience')
    ap.add_argument('--ell',type=float,default=0.4)
    ap.add_argument('--rho',type=float,default=0.2)
    ap.add_argument('--verify-segment',type=int,default=0)
    ap.add_argument('--stage-overlap',type=int,choices=(0,1))
    ap.add_argument('--routing',choices=('off','substitute','prior'))
    ap.add_argument('--routing-prefetch', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing-dampen',type=int,choices=(0,1))
    ap.add_argument('--check-prefetch-invalidation', action='store_true',
                    help='check prefetch policy reuse and startup restoration; requires startup routing_prefetch=0 and no runtime default override')
    ap.add_argument('--check-routing-invalidation', action='store_true',
                    help='check reuse and dampening invalidation; requires startup routing_dampen=0 and no runtime default override')
    ap.add_argument('--primary-adaptive',type=int,choices=(0,1))
    ap.add_argument('--helper-live',type=int,choices=(0,1))
    ap.add_argument('--helper-dispatch',choices=('worker','queue'))
    ap.add_argument('--primary-arithmetic',choices=('dp4a','fp4'))
    ap.add_argument('--note',default='')
    a=ap.parse_args(argv)
    if not 0<a.ell<=1 or not 0<a.rho<=1 or a.timeout<=0 or not 0<=a.verify_segment<=7 or (a.stage_overlap==1 and a.verify_segment!=4):
        ap.error('invalid timeout, verification segment or stage overlap')
    out=a.output_dir.expanduser().resolve()
    if out.is_relative_to(ROOT):ap.error('output must be outside source repository')
    out.mkdir(parents=False,exist_ok=False)
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'note':a.note,
            'purpose':'controlled speculative-state diagnostic; not a throughput measurement',
            'base_url':a.base_url,'events':[],'requests':[],'complete':False}
    path=out/'result.json'
    def save():path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    tune={'spec':8,'spec_policy':a.spec_policy,'spec_min_p':0.0,
          'verify_segment':a.verify_segment}
    if a.spec_policy=='lenience':tune['spec_ell']=a.ell
    else:tune['spec_rho']=a.rho
    if a.stage_overlap is not None:tune['stage_overlap']=a.stage_overlap
    if a.routing is not None:tune['routing']=a.routing
    if a.routing_prefetch is not None: tune['routing_prefetch'] = a.routing_prefetch
    if a.routing_dampen is not None:tune['routing_dampen']=a.routing_dampen
    if a.primary_adaptive is not None:tune['primary_adaptive']=a.primary_adaptive
    if a.helper_live is not None:tune['helper_live']=a.helper_live
    if a.helper_dispatch is not None:tune['helper_dispatch']=a.helper_dispatch
    if a.primary_arithmetic is not None:tune['primary_arithmetic']=a.primary_arithmetic
    try:
        base=a.base_url.rstrip('/')
        with (out/'requests.sse.jsonl').open('x',encoding='utf-8') as raw:
            before=calibration.inspect(base,result['events'],a.timeout);result['initial_inspection']=before;save()
            first=calibration.payload(PROMPT,{**tune,'state_test_reject_first':1},64,a.seed)
            calibration.run_generation(base,'forced-rejection',first,a.timeout,raw,result);save()
            result['rejection_check']=check_forced(result['requests'][-1])
            result['rejection_verifier']=check_verifier(result['requests'][-1],a.spec_policy,a.ell if a.spec_policy=='lenience' else a.rho);save()
            second=calibration.payload(PROMPT,tune,16,a.seed)
            calibration.run_generation(base,'matching-prefix',second,a.timeout,raw,result);save()
            result['reuse_check']=check_reuse(result['requests'][-1])
            result['reuse_verifier']=check_verifier(result['requests'][-1],a.spec_policy,a.ell if a.spec_policy=='lenience' else a.rho)
            if a.check_routing_invalidation:
                # Explicitly establish the baseline, then change only the mixture.
                # Matching requests must reuse; each mixture change must start fresh.
                engine_info=screen.metrics(base, min(a.timeout, 30)).get('engine', {})
                if engine_info.get('routing_dampen') != 0:
                    raise ValueError('routing invalidation check requires startup routing_dampen=0')
                records=[]
                for label, value, expect_reuse in (('dampen-baseline',0,None), ('dampen-reuse-off',0,True),
                                                   ('dampen-change-on',1,False), ('dampen-reuse-on',1,True),
                                                   ('dampen-omitted-startup',None,False)):
                    policy={**tune,'routing':'substitute'}
                    policy.pop('routing_dampen', None)
                    if value is not None: policy['routing_dampen']=value
                    calibration.run_generation(base,label,calibration.payload(PROMPT,policy,16,a.seed),a.timeout,raw,result)
                    item=result['requests'][-1]
                    final=((item.get('engine_stats') or {}).get('final') or {})
                    expected = 0 if value is None else value
                    if final.get('effective_routing_dampen') != expected:
                        raise ValueError(f'{label}: effective routing dampen differs from {expected}')
                    details=(item.get('usage') or {}).get('prompt_tokens_details') or {}
                    cached=(item.get('timings') or {}).get('cache_n',details.get('cached_tokens'))
                    if not isinstance(cached,(int,float)) or cached<0:
                        raise ValueError(f'{label}: missing cached prompt count')
                    if expect_reuse is not None and (cached>0)!=expect_reuse:
                        raise ValueError(f'{label}: unexpected cached prompt count {cached}')
                    records.append({'label':label,'cached_prompt_tokens':cached})
                    save()
                result['routing_invalidation_check']=records
            if a.check_prefetch_invalidation:
                engine_info=screen.metrics(base, min(a.timeout, 30)).get('engine', {})
                if engine_info.get('routing_prefetch') != 0 or engine_info.get('routing_prefetch_control') != 1:
                    raise ValueError('prefetch invalidation check requires capable engine with startup routing_prefetch=0')
                records=[]
                for label, value, expect_reuse in (('prefetch-baseline',0,None), ('prefetch-reuse-off',0,True),
                                                   ('prefetch-change-on',1,False), ('prefetch-reuse-on',1,True),
                                                   ('prefetch-omitted-startup',None,False)):
                    policy={**tune,'routing':'prior'}
                    policy.pop('routing_prefetch', None)
                    if value is not None: policy['routing_prefetch']=value
                    calibration.run_generation(base,label,calibration.payload(PROMPT,policy,16,a.seed),a.timeout,raw,result)
                    item=result['requests'][-1]
                    final=((item.get('engine_stats') or {}).get('final') or {})
                    expected=0 if value is None else value
                    if final.get('routing_prefetch') != expected or final.get('effective_routing_prefetch') != expected:
                        raise ValueError(f'{label}: configured or effective prefetch differs from {expected}')
                    details=(item.get('usage') or {}).get('prompt_tokens_details') or {}
                    cached=(item.get('timings') or {}).get('cache_n',details.get('cached_tokens'))
                    if not isinstance(cached,(int,float)) or cached<0:
                        raise ValueError(f'{label}: missing cached prompt count')
                    if expect_reuse is not None and (cached>0)!=expect_reuse:
                        raise ValueError(f'{label}: unexpected cached prompt count {cached}')
                    records.append({'label':label,'cached_prompt_tokens':cached})
                    save()
                result['prefetch_invalidation_check']=records
            after=calibration.inspect(base,result['events'],a.timeout);calibration.same_process(before,after)
            result['final_inspection']=after;result['complete']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:save()
    print(path)
    if result.get('error'):print(result['error'],file=sys.stderr)
    return 0 if result['complete'] else 1

if __name__=='__main__':sys.exit(main())
