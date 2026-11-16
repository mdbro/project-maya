"""Regression cases for the measured input semantics, without running a model."""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass_data
import pass_model as pm


class Capture:
    def generation_forwards(self):
        return [dict(forward=i,position=100+i) for i in range(3)]

    def of(self, kind):
        assert kind == 'layer'
        return [dict(forward=0,layer=3,rows=3,executed=[7,7,7],service=['staged_new','staged_ready','staged_ready']),
                dict(forward=1,layer=3,rows=2,executed=[7,7],service=['staged_ready','staged_ready']),
                dict(forward=2,layer=3,rows=1,executed=[9],service=['primary_resident'])]


class Inputs(unittest.TestCase):
    def test_repeated_picks_in_upload_pass_are_not_future_reuse(self):
        with patch.object(pass_data.nl,'read',return_value=Capture()):
            self.assertEqual(pass_data.reuse_counts('fake',from_token=0),{0:[1,1],1:[0,0]})
            self.assertEqual(pass_data.reuse_counts('fake',window=0,from_token=0),{0:[1,0],1:[0,0]})

    def test_capture_failure_is_refused_before_metadata_or_fitting(self):
        with patch.object(pass_data.nl,'read',return_value=Capture()), patch.object(pass_data.nl,'validate',
                return_value=dict(status='invalid',errors=['dropped records'])):
            with self.assertRaisesRegex(ValueError,'dropped records'):
                pass_data.rows_of('fake')

    def test_grouped_service_sample_removes_the_whole_expert(self):
        r=pm.Rates()
        load=pm.Load(rows=4,cpu_experts=1,cpu_columns=4,gpu_experts=1,gpu_picks=1,
                     copies=1,late_experts=1,late_picks=1,overlap=True)
        moved=pm.Load(**load.__dict__)
        moved.cpu_experts=0;moved.cpu_columns=0;moved.gpu_experts+=1;moved.gpu_picks+=4
        expected=max(0,pm.predict(load,r).passed-pm.predict(moved,r).passed)
        self.assertAlmostEqual(pm.use_samples(load,r,[4],[1])[0],expected)

    def test_enabled_early_policy_is_distinct_from_observed_issue(self):
        row=dict(rows=1,cpu_experts=1,cpu_columns=1,uploads=0,exchanges_pending=0,g_ready=0,g_experts=1,g_picks=1,
                 helper_experts=0,helper_columns=0,helper_lease=0,third_experts=0,third_columns=0,gpu_rounds_of=32,
                 overlap=True,late_experts=0,late_picks=0,early_active=1,early_issued=0,runtime={'early_copy':2})
        self.assertEqual(pm.load_of(row).early,1)


if __name__ == '__main__':
    unittest.main()
