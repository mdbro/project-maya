import copy
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import native_compare as compare


class ConditionedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.state={'engine_pid':71,'ownership_complete':True,'target_object':1,'draft_object':2,
                    'host_address':3,'host_bytes':4,'host_records':5,'helper_pool':6,
                    'primary_ids':[[3,0,0,100]],'helper_ids':[[3,1,0]]}
        self.prior={'complete':True,'historical_fixed_primary':True,'final_inspection':self.state,
                    'required_binary':{'observed':'abc','expected':'abc'},
                    'requests':[{'process_io_after':{'pid':71,'start_ticks':'123'},
                                 'engine_stats':{'request_seq':8}}]}
        self.io={'pid':71,'start_ticks':'123'}
        self.metrics={'tiers':{'history':{'request_seq':[7,8,8]}}}

    def check(self, prior=None, state=None, io=None, metrics=None, sha='abc'):
        return compare.verify_conditioned_history(prior or self.prior,state or self.state,
                                                   io or self.io,metrics or self.metrics,sha)

    def test_uninterrupted_history_is_bound_without_reset(self):
        self.assertEqual(self.check()['request_seq'],8)

    def test_intervening_request_or_pid_reuse_rejects_arm(self):
        with self.assertRaises(ValueError):self.check(metrics={'tiers':{'history':{'request_seq':[8,9]}}})
        with self.assertRaises(ValueError):self.check(io={'pid':71,'start_ticks':'456'})
        with self.assertRaises(ValueError):self.check(sha='different')

    def test_changed_roster_or_unqualified_history_rejects_arm(self):
        changed=copy.deepcopy(self.state);changed['helper_ids']=[[3,2,0]]
        with self.assertRaises(ValueError):self.check(state=changed)
        with self.assertRaises(ValueError):self.check(prior={**self.prior,'complete':False})
        with self.assertRaises(ValueError):self.check(prior={**self.prior,'historical_fixed_primary':False})


if __name__=='__main__':unittest.main()
