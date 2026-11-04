import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import native_state_check as diagnostic

class StateCheck(unittest.TestCase):
    def forced(self, **updates):
        final={'forced_rejections':1,'forced_rejection_emitted':3,'forced_rejection_frontier':10,
               'forced_rejection_round':1,'forced_natural_accept':1,'emitted':7,'target_rounds':3}
        final.update(updates)
        return {'engine_stats':{'join':'one_new_sequence_and_matching_completion_count','final':final}}
    def test_forced_then_later_round(self):
        r=diagnostic.check_forced(self.forced())
        self.assertEqual((r['emitted_at_rejection'],r['frontier_at_rejection'],r['final_target_rounds']),(3,10,3))
    def test_rejection_without_continuation_rejected(self):
        for changes in ({'forced_rejections':0},{'target_rounds':1},{'emitted':3},{'forced_rejection_frontier':-1}):
            with self.assertRaises(ValueError):diagnostic.check_forced(self.forced(**changes))
    def test_effective_verifier_and_matching_fallback(self):
        item={'engine_stats': {'join':'one_new_sequence_and_matching_completion_count', 'request_seq':3,
                              'final':{'request_seq':3,'emitted':12,'generation_final':1,
                                       'spec_policy':'lenience','spec_ell':.4}},
              'usage':{'completion_tokens':12}}
        self.assertEqual(diagnostic.check_verifier(item,'lenience',.4)['source'],'history')
        now=dict(item['engine_stats']['final'])
        item['engine_stats']['final'].pop('spec_policy')
        item['metrics_after']={'tiers':{'now':now}}
        self.assertEqual(diagnostic.check_verifier(item,'lenience',.4)['source'],'metrics_after.tiers.now')
        for key,value in (('request_seq',4),('generation_final',0),('emitted',11),('spec_policy','fast_hsd'),('spec_ell',float('nan'))):
            saved=now[key]; now[key]=value
            with self.assertRaises(ValueError): diagnostic.check_verifier(item,'lenience',.4)
            now[key]=saved
        now['spec_policy']='fast_hsd'; now['spec_rho']=.2
        self.assertEqual(diagnostic.check_verifier(item,'fast_hsd',.2)['spec_rho'],.2)

    def test_followup_requires_reuse_and_no_forcing(self):
        item={'engine_stats':{'join':'one_new_sequence_and_matching_completion_count','final':{'forced_rejections':0,'emitted':12}},
              'timings':{'cache_n':10}}
        self.assertEqual(diagnostic.check_reuse(item)['reused_prompt_tokens'],10)
        item['timings']['cache_n']=0
        with self.assertRaises(ValueError):diagnostic.check_reuse(item)
        item['timings']['cache_n']=10;item['engine_stats']['final']['forced_rejections']=1
        with self.assertRaises(ValueError):diagnostic.check_reuse(item)

if __name__=='__main__':unittest.main()
