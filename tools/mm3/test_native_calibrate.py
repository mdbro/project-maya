import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import native_calibrate as cal


def profile(changes):
    c=[0]*7680
    for (li,e),v in changes.items():c[li*128+e]=v
    return struct.pack('<7680I',*c)


def report():
    return {'engine_pid':71,'ownership_complete':True,'experts_per_layer':128,'resident_cap':1,
            'helper_eligible_cuda_device':0,'layer_devices':[1]*8+[0]*52,
            'layer_moe':[False]*3+[True]*57,
            'primary_ids':[[3,0,1,100],[8,0,0,101]],'helper_ids':[[8,1,0]],
            'allocations':[1,2,3],'target_object':4,'draft_object':5,'host_address':6,
            'host_bytes':7,'host_records':7294,'helper_pool':8}


class SelectionTests(unittest.TestCase):
    def test_primary_device_capacity_and_helper_exclusion(self):
        r=report(); p=profile({(4,2):10,(9,5):9,(8,4):8})
        primary,helper=cal.selection(p,r,'all')
        self.assertEqual(primary,{(4,2),(9,5)})
        self.assertEqual(helper,{(8,4)})
        self.assertFalse(primary & helper)
    def test_tie_order_helper_and_primary(self):
        r=report(); p=profile({(3,2):5,(4,0):5,(8,2):4,(9,1):4})
        primary,helper=cal.selection(p,r,'all')
        self.assertEqual(primary,{(3,2),(8,2)})
        self.assertEqual(helper,{(9,1)})
        _,helper=cal.selection(profile({(9,2):3,(8,3):3}),r,'helper')
        self.assertEqual(helper,{(9,2)})
    def test_missing_geometry_and_wrong_device_rejected(self):
        r=report();del r['layer_moe']
        with self.assertRaises(ValueError):cal.selection(profile({(8,2):1}),r,'all')
        r=report();r['primary_ids'][0][2]=0
        with self.assertRaises(ValueError):cal.selection(profile({(8,2):1}),r,'all')
    def test_helper_changed_identities_and_allocations(self):
        before=report();p=profile({(9,2):100})
        after={**before,'helper_ids':[[9,2,0]],'primary_ids':[r[:] for r in before['primary_ids']]}
        checked=cal.verify_placement(before,after,p,'helper')
        self.assertTrue(checked['helper_ids_match'])
        for key,val in [('draft_object',999),('ownership_complete',False),('helper_ids',[[9,2,9]])]:
            bad={**after,key:val}
            with self.assertRaises(ValueError):cal.verify_placement(before,bad,p,'helper')
    def test_all_verification_checks_device_assignment(self):
        before=report();p=profile({(4,2):10,(9,5):9,(8,4):8})
        after={**before,'primary_ids':[[4,2,1,100],[9,5,0,101]],'helper_ids':[[8,4,0]]}
        self.assertTrue(cal.verify_placement(before,after,p,'all')['primary_ids_match'])
        after['primary_ids'][0][2]=0
        with self.assertRaises(ValueError):cal.verify_placement(before,after,p,'all')

class CollectionTests(unittest.TestCase):
    def test_tool_detection_handles_streamed_function_name(self):
        item={'events':[{'data':{'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':part}}]}}]}}
                        for part in ('lookup_', 'status')]}
        self.assertTrue(cal.tool_calls_observed(item))
        self.assertFalse(cal.tool_calls_observed({'events':[]}))

    def test_collect_resets_freezes_and_records_each_generation_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);source=root/'startup.bin';source.write_bytes(profile({(3,1):99}))
            out=root/'run';current=[0]*7680
            def post(base,endpoint,body,events,timeout):
                events.append({'endpoint':endpoint,'request':body})
                if endpoint.endswith('placement'):return report()
                if body['action']=='reset':
                    current[:]=[0]*7680
                    return {'action':'reset','engine_pid':71}
                raw=struct.pack('<7680I',*current)
                return {'engine_pid':71,'profile_hex':raw.hex(),'profile_sha256':hashlib.sha256(raw).hexdigest(),
                        'source':'generation-only original target expert demand'}
            def run_case(base,case,request,timeout,raw,result,*args):
                # Synthetic test double: only generation increments these counters.
                current[3*128]+=10
                result['requests'].append({'case':case['id'],'request':request,'complete':True,
                    'usage':{'completion_tokens':4},'engine_stats':{'join':'one_new_sequence_and_matching_completion_count'},
                    'metrics_before':{},'metrics_after':{},'events':[{'data':{'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'lookup_status'}}]}}]}}] if case['id']=='calibration-tool' else []})
                raw.write('sample\n')
            with patch.object(cal,'post',side_effect=post),patch.object(cal.screen,'run_case',side_effect=run_case):
                rc=cal.main(['collect','--output-dir',str(out),'--source-profile',str(source)])
            self.assertEqual(rc,0)
            result=json.loads((out/'result.json').read_text()); data=(out/'generation-profile.bin').read_bytes()
            self.assertEqual([p['generation_demand_delta'] for p in result['phases']],[10,10,10])
            self.assertEqual(result['profile']['total_generation_demand'],30)
            self.assertEqual(result['profile']['sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(cal.counts_from_bytes(data)[3*128],30)
            self.assertEqual(len((out/'requests.sse.jsonl').read_text().splitlines()),3)
            self.assertIn('not independently confirmed',result['declared_source_profile']['verification'])
    def test_process_change_fails_and_does_not_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);source=root/'startup.bin';source.write_bytes(profile({(3,1):99}));out=root/'run'
            def post(base,endpoint,body,events,timeout):
                if endpoint.endswith('placement'):return report()
                return {'action':'reset','engine_pid':72}
            with patch.object(cal,'post',side_effect=post):
                rc=cal.main(['collect','--output-dir',str(out),'--source-profile',str(source)])
            self.assertEqual(rc,1)
            self.assertFalse((out/'generation-profile.bin').exists())
            self.assertIn('process changed',json.loads((out/'result.json').read_text())['error'])

if __name__=='__main__':unittest.main()
