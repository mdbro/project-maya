#!/usr/bin/env python3
"""Public-server forced-rejection and matching-prefix diagnostic; not a speed test."""
import argparse
import datetime as dt
import json
import pathlib
import sys

import native_calibrate as calibration
import native_screen as screen

ROOT=pathlib.Path(__file__).resolve().parents[2]
PROMPT=('Explain in several short numbered steps how to make a simple vegetable soup. '
        'Include preparation, cooking, and serving.')


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
    ap.add_argument('--verify-segment',type=int,default=0)
    ap.add_argument('--stage-overlap',type=int,choices=(0,1))
    ap.add_argument('--primary-adaptive',type=int,choices=(0,1))
    ap.add_argument('--helper-live',type=int,choices=(0,1))
    ap.add_argument('--primary-arithmetic',choices=('dp4a','fp4'))
    ap.add_argument('--note',default='')
    a=ap.parse_args(argv)
    if a.timeout<=0 or not 0<=a.verify_segment<=7 or (a.stage_overlap==1 and a.verify_segment!=4):
        ap.error('invalid timeout, verification segment or stage overlap')
    out=a.output_dir.expanduser().resolve()
    if out.is_relative_to(ROOT):ap.error('output must be outside source repository')
    out.mkdir(parents=False,exist_ok=False)
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'note':a.note,
            'purpose':'controlled speculative-state diagnostic; not a throughput measurement',
            'base_url':a.base_url,'events':[],'requests':[],'complete':False}
    path=out/'result.json'
    def save():path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    tune={'spec':8,'spec_policy':'fast_hsd','spec_rho':.2,'spec_min_p':0.0,
          'verify_segment':a.verify_segment}
    if a.stage_overlap is not None:tune['stage_overlap']=a.stage_overlap
    if a.primary_adaptive is not None:tune['primary_adaptive']=a.primary_adaptive
    if a.helper_live is not None:tune['helper_live']=a.helper_live
    if a.primary_arithmetic is not None:tune['primary_arithmetic']=a.primary_arithmetic
    try:
        base=a.base_url.rstrip('/')
        with (out/'requests.sse.jsonl').open('x',encoding='utf-8') as raw:
            before=calibration.inspect(base,result['events'],a.timeout);result['initial_inspection']=before;save()
            first=calibration.payload(PROMPT,{**tune,'state_test_reject_first':1},64,a.seed)
            calibration.run_generation(base,'forced-rejection',first,a.timeout,raw,result);save()
            result['rejection_check']=check_forced(result['requests'][-1]);save()
            second=calibration.payload(PROMPT,tune,16,a.seed)
            calibration.run_generation(base,'matching-prefix',second,a.timeout,raw,result);save()
            result['reuse_check']=check_reuse(result['requests'][-1])
            after=calibration.inspect(base,result['events'],a.timeout);calibration.same_process(before,after)
            result['final_inspection']=after;result['complete']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally:save()
    print(path)
    if result.get('error'):print(result['error'],file=sys.stderr)
    return 0 if result['complete'] else 1

if __name__=='__main__':sys.exit(main())
