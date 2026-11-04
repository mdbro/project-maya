import hashlib
import copy
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
    return {'engine_pid':71,'ownership_complete':True,'experts_per_layer':128,'resident_cap':-1,
            'helper_eligible_cuda_device':0,'layer_devices':[0]*60,
            'layer_moe':[False]*3+[True]*57,
            'primary_ids':[[3,0,0,100],[8,0,0,101]],'helper_ids':[[8,1,0]],
            'allocations':[1,2,3],'target_object':4,'draft_object':5,'host_address':6,
            'host_bytes':7,'host_records':7294,'helper_pool':8}


class SelectionTests(unittest.TestCase):
    def test_helper_first_global_rank_and_primary_complement(self):
        r=report();p=profile({(4,2):10,(9,5):90,(8,4):8})
        primary,helper=cal.selection(p,r,'helper-first')
        self.assertEqual(helper,{(9,5)})
        self.assertEqual(primary,{(4,2),(8,4)})

    def test_checkpoint_signature_ignores_addresses_but_covers_learning(self):
        r={'scope':'in-process only','staging':'cleared at joined boundary','placement':report(),
           'geometry':[60,128,6144,3072,-1],'arithmetic':'fp4','primary_adaptive':True,
           'policy':{'horizon':20},'lane':{'cpu1':2},'scores':[[3,2,8.5]],'adaptive_main':[[3,2]]}
        original=cal.learned_state_signature(r)
        same=copy.deepcopy(r);same['placement']['primary_ids'][0][3]=999
        same['snapshot_forward']=500;same['scores_normalized_at_forward']=600
        self.assertEqual(cal.learned_state_signature(same),original)
        for key,value in [('scores',[[3,2,0]]),('lane',{'cpu1':3}),('policy',{'horizon':40})]:
            self.assertNotEqual(cal.learned_state_signature({**r,key:value}),original)
        bad=copy.deepcopy(r);bad['placement']['helper_ids'][0][1]=4
        self.assertNotEqual(cal.learned_state_signature(bad),original)
        bad=copy.deepcopy(r);bad['placement']['ownership_complete']=False
        with self.assertRaises(ValueError):cal.learned_state_signature(bad)

    def test_primary_device_capacity_and_helper_exclusion(self):
        r=report(); p=profile({(4,2):10,(9,5):9,(8,4):8})
        primary,helper=cal.selection(p,r,'all')
        self.assertEqual(primary,{(9,5),(8,4)})
        self.assertEqual(helper,{(4,2)})
        self.assertFalse(primary & helper)
    def test_tie_order_helper_and_primary(self):
        r=report(); p=profile({(3,2):5,(4,0):5,(8,2):4,(9,1):4})
        primary,helper=cal.selection(p,r,'all')
        self.assertEqual(primary,{(3,2),(8,2)})
        self.assertEqual(helper,{(4,0)})
        with self.assertRaises(ValueError):
            cal.selection(profile({(9,2):3,(8,3):3}),r,'helper')
    def test_missing_geometry_and_wrong_device_rejected(self):
        r=report();del r['layer_moe']
        with self.assertRaises(ValueError):cal.selection(profile({(8,2):1}),r,'all')
        r=report();r['primary_ids'][0][2]=1
        with self.assertRaises(ValueError):cal.selection(profile({(8,2):1}),r,'all')
    def test_helper_changed_identities_and_allocations(self):
        before=report();p=profile({(9,2):100,(3,0):90,(8,0):80})
        after={**before,'helper_ids':[[9,2,0]],'primary_ids':[r[:] for r in before['primary_ids']]}
        checked=cal.verify_placement(before,after,p,'helper')
        self.assertTrue(checked['helper_ids_match'])
        for key,val in [('draft_object',999),('ownership_complete',False),('helper_ids',[[9,2,9]])]:
            bad={**after,key:val}
            with self.assertRaises(ValueError):cal.verify_placement(before,bad,p,'helper')
    def test_all_verification_checks_device_assignment(self):
        before=report();p=profile({(4,2):10,(9,5):9,(8,4):8})
        after={**before,'primary_ids':[[9,5,0,100],[8,4,0,101]],'helper_ids':[[4,2,0]]}
        self.assertTrue(cal.verify_placement(before,after,p,'all')['primary_ids_match'])
        after['primary_ids'][0][2]=1
        with self.assertRaises(ValueError):cal.verify_placement(before,after,p,'all')

    def test_primary_staging_rotation_preserves_complete_physical_inventory(self):
        before=report();before['staging_frames']=[[0,0,201,-1,1],[0,1,202,-1,0]]
        before['adaptive_host_scratch_bytes']=64
        p=profile({(4,2):10,(9,5):9,(8,4):8})
        after={**before,'primary_ids':[[9,5,0,100],[8,4,0,201]],'helper_ids':[[4,2,0]],
               'staging_frames':[[0,0,101,-1,0],[0,1,202,-1,0]]}
        checked=cal.verify_placement(before,after,p,'all')
        self.assertTrue(checked['primary_staging_frame_rotation'])
        for changed in ([[0,0,999,-1,0],[0,1,202,-1,0]],
                        [[0,0,201,-1,0],[0,1,202,-1,0]],
                        [[1,0,101,-1,0],[0,1,202,-1,0]]):
            with self.assertRaises(ValueError):
                cal.verify_placement(before,{**after,'staging_frames':changed},p,'all')
        with self.assertRaises(ValueError):
            cal.verify_placement(before,{**after,'adaptive_host_scratch_bytes':128},p,'all')

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
