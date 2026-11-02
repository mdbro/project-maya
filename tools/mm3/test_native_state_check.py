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
    def test_followup_requires_reuse_and_no_forcing(self):
        item={'engine_stats':{'join':'one_new_sequence_and_matching_completion_count','final':{'forced_rejections':0,'emitted':12}},
              'timings':{'cache_n':10}}
        self.assertEqual(diagnostic.check_reuse(item)['reused_prompt_tokens'],10)
        item['timings']['cache_n']=0
        with self.assertRaises(ValueError):diagnostic.check_reuse(item)
        item['timings']['cache_n']=10;item['engine_stats']['final']['forced_rejections']=1
        with self.assertRaises(ValueError):diagnostic.check_reuse(item)

if __name__=='__main__':unittest.main()
