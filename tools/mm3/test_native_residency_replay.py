import json
import tempfile
import unittest
from pathlib import Path
import native_residency_replay as r


def records():
    request={'type':'request','schema':1,'request':1,'record_bytes':100,
             'forward':0,'layers':[{'layer':0,'primary':[0],'helper':[1],'host':[2,3,4]}]}
    def event(i, forward, original, executed, staging=None):
        route={'type':'routes','request':1,'event':i,'forward':forward,'generation':1,'layer':0,
               'rows':1,'positions':[i], 'original':original,'executed':executed, 'staging':staging or [], 'host':[2,3,4], 'helper':[1]}
        service={'type':'service','request':1,'event':i,'forward':forward,'host_service_ms':1., 'cpu_helper_lane_ms':1.,'cpu_jobs':1,'helper_jobs':0}
        return [route, service]
    return [request]+event(1,1,[2,2,2,2],[2,2,2,2])+event(2,2,[2,2,2,2],[2,2,2,2])+event(3,3,[2,2,2,2],[2,2,2,2])+[{'type':'end','request':1,'forward':3}]

def with_other_layer(rows):
    rows[0]['layers'].append({'layer':1,'primary':[],'helper':[1],'host':[]})
    out=[rows[0]]
    event=0
    for i in range(1,len(rows)-1,2):
        route,service=rows[i:i+2]; event+=1; route['event']=event; service['event']=event
        out.extend([route,service]);event+=1
        other={**route,'event':event,'layer':1,'original':[1,1,1,1],'executed':[1,1,1,1],
               'host':[],'helper':[1],'primary':[],'staging':[]}
        other_service={**service,'event':event,'cpu_jobs':0,'helper_jobs':1,'cpu_helper_lane_ms':0,'host_service_ms':0}
        if 'time_s' in route: other['time_s']=service['time_s']+.001;other_service['time_s']=other['time_s']
        out.extend([other,other_service])
    return out+[rows[-1]]

class ReplayTest(unittest.TestCase):
    def load(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace';p.write_text(''.join(json.dumps(x)+'\n' for x in rows));return r.read_trace(p)
    def test_original_demand_drives_admission_not_executed(self):
        req,events=self.load(with_other_layer(records()))
        arm=r.replay(req,events,'priced',h2d_ms=.01,d2h_ms=.01,interference_ms=0,service_saved_ms=10)
        self.assertGreater(arm['counters']['replacements'],0)
        self.assertEqual(arm['counters']['original_rows'],24)
        self.assertEqual(arm['primary_capacity'],1)
        self.assertEqual(arm['helper_capacity'],2)
        self.assertGreater(arm['counters']['primary_hits'],0)
    def test_pending_deadline_and_zero_budget(self):
        req,events=self.load(records())
        arm=r.replay(req,events,'priced',h2d_ms=1500,d2h_ms=1500,interference_ms=0,service_saved_ms=1000,horizon=100)
        self.assertEqual(arm['counters'].get('replacements',0),0)
        self.assertGreater(arm['counters']['pending_at_end'],0)
        arm=r.replay(req,events,'priced',refill_budget=0,service_saved_ms=1000)
        self.assertEqual(arm['counters'].get('issued',0),0)
    def test_early_retains_ready_copy_without_another_h2d(self):
        rows=records()[:3]
        rows[0]['layers'].append({'layer':1,'primary':[],'helper':[],'host':[2]})
        rows[1]['layer']=1; rows[1]['host']=[2]; rows[1]['helper']=[]
        rows[1]['original']=[2,2,2,2]; rows[1]['executed']=[2,2,2,2]
        route={'type':'routes','request':1,'event':2,'forward':1,'generation':1,'layer':0,
               'rows':1,'positions':[2],'original':[1,1,1,1],'executed':[1,1,1,1],
               'staging':[[1026,-1,0,0,0,0]], 'host':[2,3,4], 'helper':[1]}
        rows.extend([route,{'type':'service','request':1,'event':2,'forward':1,'host_service_ms':1.,'cpu_helper_lane_ms':1.,'cpu_jobs':0,'helper_jobs':1},{'type':'end','request':1,'forward':1}])
        req,events=self.load(rows)
        arm=r.replay(req,events,'priced',h2d_ms=.01,d2h_ms=.01,interference_ms=0,service_saved_ms=100)
        self.assertEqual(arm['counters']['early_retained'],1)
        self.assertEqual(arm['counters']['h2d_bytes'],0)
        self.assertEqual(arm['counters']['d2h_bytes'],100)

    def test_service_staging_used_at_forward_end(self):
        rows=records()
        rows[2]['staging']=[[2,-1,0,0,0,0]]
        req,events=self.load(rows)
        arm=r.replay(req,events,'legacy',h2d_ms=.01,d2h_ms=.01,interference_ms=0,service_saved_ms=10)
        self.assertGreater(arm['counters']['reused_staging'],0)

    def test_counterfactual_service_and_transfer_change_estimate(self):
        req,events=self.load(with_other_layer(records()))
        cold=r.replay(req,events,'priced',refill_budget=0,service_saved_ms=1)
        hot=r.replay(req,events,'priced',h2d_ms=.01,d2h_ms=.01,interference_ms=0,service_saved_ms=10)
        self.assertLess(hot['counters']['generation_estimated_layer_ms'],cold['counters']['generation_estimated_layer_ms'])
        costly=r.replay(req,events,'priced',h2d_ms=.01,d2h_ms=.01,interference_ms=.5,service_saved_ms=10)
        self.assertGreater(costly['counters']['transfer_interference_ms'],0)
        self.assertGreater(costly['counters']['generation_estimated_layer_ms'],hot['counters']['generation_estimated_layer_ms'])

    def test_footer_required_and_must_match(self):
        with self.assertRaises(ValueError): self.load(records()[:-1])
        bad=records();bad[-1]['request']=2
        with self.assertRaises(ValueError): self.load(bad)
        bad=records();bad[-1]['forward']=1
        with self.assertRaises(ValueError): self.load(bad)

    def test_gap_advances_pending_deadline_before_next_forward(self):
        rows=records()
        for index in (1,3,5):
            rows[index]['time_s'] = (index // 2) * .020
            rows[index+1]['time_s'] = rows[index]['time_s'] + .001
        req,events=self.load(with_other_layer(rows))
        arm=r.replay(req,events,'priced',h2d_ms=2,d2h_ms=2,interference_ms=0,service_saved_ms=100)
        self.assertGreater(arm['counters']['replacements'],0)
        self.assertGreater(arm['counters']['observed_inter_event_gap_ms'],0)

    def test_observed_nonprimary_includes_staged_even_without_cpu_job(self):
        rows=records();rows[2]['cpu_jobs']=0;rows[2]['cpu_helper_lane_ms']=2;rows[2]['host_service_ms']=3
        req,events=self.load(rows)
        arm=r.replay(req,events,'priced',refill_budget=0,service_saved_ms=10)
        self.assertEqual(arm['events'][0]['observed_host_jobs'],1)
        self.assertEqual(arm['events'][0]['estimated_lane_ms'],2)
        self.assertEqual(arm['events'][0]['estimated_layer_ms'],3)

    def test_frozen_helper_is_exclusive_when_trace_helper_changes(self):
        rows=records(); rows[1]['helper']=[2]; rows[1]['host']=[1,3,4]
        req,events=self.load(rows)
        arm=r.replay(req,events,'priced',refill_budget=0,service_saved_ms=1)
        self.assertEqual(arm['events'][0]['helper_hits'],0)
        self.assertEqual(arm['events'][0]['host_jobs'],1)

    def test_early_reserves_one_admission_and_current_layer_is_protected(self):
        rows=records()[:3]
        rows[0]['layers'].append({'layer':1,'primary':[],'helper':[],'host':[2]})
        rows[1]['layer']=1;rows[1]['host']=[2];rows[1]['helper']=[]
        route={'type':'routes','request':1,'event':2,'forward':1,'generation':1,'layer':0,
               'rows':1,'positions':[2],'original':[1,1,1,1],'executed':[1,1,1,1],
               'staging':[[1026,-1,0,0,0,0]],'host':[2,3,4],'helper':[1]}
        service={'type':'service','request':1,'event':2,'forward':1,'host_service_ms':1.,
                 'cpu_helper_lane_ms':1.,'cpu_jobs':0,'helper_jobs':1}
        req,events=self.load(rows+[route,service,{'type':'end','request':1,'forward':1}])
        arm=r.replay(req,events,'priced',refill_budget=1,h2d_ms=.01,d2h_ms=.01,interference_ms=0,service_saved_ms=100)
        self.assertEqual(arm['counters'].get('early_retained',0),0)

    def test_validation_and_missing_service(self):
        with self.assertRaises(ValueError): self.load(records()[:-1])
        bad=records(); bad[1]['executed']=[0]
        with self.assertRaises(ValueError): self.load(bad)
    def test_wall_is_max_not_sum(self):
        req,events=self.load(records())
        arm=r.replay(req,events,'priced',refill_budget=0,service_saved_ms=2)
        self.assertEqual(arm['counters']['estimated_layer_ms'],3)

if __name__ == '__main__': unittest.main()
