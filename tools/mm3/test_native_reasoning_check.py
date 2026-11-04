import unittest
import json
import tempfile
from pathlib import Path
import native_delivery as delivery
import native_reasoning_check as check
import native_screen as screen

class ReasoningTest(unittest.TestCase):
    def test_semantics_and_distinct_justification(self):
        first = {'finish': 11, 'schedule': {'A':[0,3], 'B':[0,2], 'C':[5,9], 'D':[2,5], 'E':[9,11]},
                 'lower_bound_explanation':'machine cannot begin before 2; seven units plus E', 'lower_bound_valid':True}
        second = {'finish':12, 'schedule': {'A':[0,3], 'B':[0,4], 'C':[3,7], 'D':[7,10], 'E':[10,12]},
                  'lower_bound_explanation':'machine cannot begin before 3; seven units plus E', 'lower_bound_valid':True}
        out=check.check({'complete':True,'content':'answer','finish_reason':'stop'}, {'scenarios':[first,second]})
        for scenario in out['scenarios']:
            self.assertTrue(scenario['finish_correct']); self.assertTrue(scenario['schedule_feasible']); self.assertTrue(scenario['lower_bound_valid'])
        first['schedule']['C']=[3,7]
        self.assertFalse(check.check_scenario(first,2)['schedule_feasible'])
        second['lower_bound_valid']=False
        self.assertFalse(check.check_scenario(second,4)['lower_bound_valid'])
    def test_truncation_and_missing_not_marked_wrong(self):
        out=check.check({'complete':True,'content':'','finish_reason':'length'}, {})
        self.assertEqual(out['completion'],'truncated')
        self.assertIsNone(out['scenarios'][0]['finish_correct'])
        self.assertIsNone(out['scenarios'][0]['schedule_feasible'])
    def test_pooled_throughput_and_incomplete(self):
        def item(name, t, ms):
            return {'case':name,'complete':True,'engine_stats':{'generation_interval':{'tokens':t,'interval_ms':ms,'tokens_per_second':1000*t/ms}}}
        out=screen.aggregate_screen([item('a',10,1000),item('b',90,3000)],2,0)
        self.assertEqual(out['pooled_tokens_per_second'],25)
        self.assertTrue(out['whole_post_prefill'])
        self.assertIsNone(screen.aggregate_screen([item('a',10,1000)],2,0)['pooled_tokens_per_second'])

    def test_fresh_context_requires_explicit_consistent_zero(self):
        self.assertTrue(screen.fresh_context_check({'timings':{'cache_n':0}, 'usage':{'prompt_tokens_details':{'cached_tokens':0}}})['verified'])
        for item in ({}, {'timings':{'cache_n':1}}, {'timings':{'cache_n':False}},
                     {'timings':{'cache_n':0}, 'usage':{'prompt_tokens_details':{'cached_tokens':1}}}):
            self.assertFalse(screen.fresh_context_check(item)['verified'])

    def test_real_fixture_read_confined_to_declared_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); (folder/'notes.txt').write_text('real fixture', encoding='utf-8')
            call={'function':{'name':'read_file','arguments':json.dumps({'path':'notes.txt'})}}
            self.assertEqual(delivery.read_fixture_call(call,folder),'real fixture')
            call['function']['arguments']=json.dumps({'path':'../notes.txt'})
            with self.assertRaises(ValueError): delivery.read_fixture_call(call,folder)
            (folder/'notes.txt').unlink(); (folder/'notes.txt').symlink_to('/etc/passwd')
            call['function']['arguments']=json.dumps({'path':'notes.txt'})
            with self.assertRaises(ValueError): delivery.read_fixture_call(call,folder)

if __name__ == '__main__': unittest.main()
