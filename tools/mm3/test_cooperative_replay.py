import copy
import unittest
from cooperative_replay.execution import Estimate, Scheduler, fixed_schedule_witness
from cooperative_replay.costs import CoverageError, ServiceTable
from cooperative_replay.policy import NativePolicy, Demand


def measured(kind, value, **kwargs):
    return dict(kind=kind,samples_ms=[value*.99,value,value*1.01],evidence=['independent-fixture'],**kwargs)


class ExecutionTest(unittest.TestCase):
    def test_dependencies_and_copy_queues(self):
        s=Scheduler();copy=s.submit('copy',Estimate(4,4,4),('h2d',),queue='main')
        gpu=s.submit('gpu',Estimate(2,2,2),('gpu',),(copy,),queue='main')
        cpu=s.submit('cpu',Estimate(5,5,5),('cpu',))
        joined=s.join([gpu,cpu])
        self.assertEqual(s.ops[joined].finish,6)
        self.assertEqual(s.ops[cpu].start,0)
        # An issued copy remains an obligation, even when no future demand uses it.
        extra=s.submit('copy',Estimate(3,3,3),('h2d',),queue='optional')
        s.drain();self.assertEqual(s.ops[extra].finish,9)

    def test_contention_only_during_overlap(self):
        s=Scheduler({('cpu','copy'):Estimate(2,2,2)})
        cpu=s.submit('cpu',Estimate(4,4,4),('cpu',))
        cp=s.submit('copy',Estimate(2,2,2),('h2d',))
        s.drain()
        self.assertEqual(s.ops[cp].finish,2)
        self.assertEqual(s.ops[cpu].finish,5)
        self.assertEqual(s.ops[cpu].pressure_ms,1)

    def test_same_direction_serializes_and_opposite_overlaps(self):
        s=Scheduler()
        a=s.submit('h2d',Estimate(3,3,3),('h2d',))
        b=s.submit('h2d',Estimate(2,2,2),('h2d',))
        c=s.submit('d2h',Estimate(4,4,4),('d2h',))
        s.drain()
        self.assertEqual([s.ops[x].finish for x in (a,b,c)],[3,5,4])

    def test_fixed_schedule_monotonic_and_retained_witness(self):
        s=Scheduler()
        a=s.submit('h2d',Estimate(4,4,4),('copy',))
        b=s.submit('d2h',Estimate(3,3,3),('copy',),(a,))
        c=s.submit('gpu',Estimate(2,2,2),('gpu',),(b,))
        s.drain()
        self.assertLessEqual(fixed_schedule_witness(s.ops,{a:1,b:0}),s.now)
        with self.assertRaises(ValueError):fixed_schedule_witness(s.ops,{a:5})

    def test_illegal_dependencies_and_bounded_work(self):
        s=Scheduler(max_operations=1)
        with self.assertRaises(ValueError):s.submit('x',Estimate(1,1,1),dependencies=[0])
        s.submit('x',Estimate(1,1,1))
        with self.assertRaises(ValueError):s.submit('x',Estimate(1,1,1))


class PhysicalTest(unittest.TestCase):
    def table(self):
        return dict(schema=1,units='ms',measurements=[
            measured('cpu',2,rows=1,groups=[1],workers=16,working_set='rotating64'),
            measured('cpu',6,rows=5,groups=[5,5,5,5],workers=16,working_set='rotating64')])

    def test_worker_count_is_measured_not_scaled(self):
        t=ServiceTable(self.table())
        with self.assertRaises(CoverageError):t.estimate('cpu',rows=1,groups=[1],workers=8,working_set='rotating64')
        with self.assertRaises(CoverageError):t.estimate('cpu',rows=5,groups=[5]*8,workers=16,working_set='rotating64')
        v=t.estimate('cpu',rows=3,groups=[3,1],workers=16,working_set='rotating64')
        self.assertFalse(v.qualified);self.assertGreater(v.high-v.low,v.ms*.3)

    def test_rejects_timing_leakage_and_policy_prices(self):
        doc=self.table();doc['measurements'][0]['request_id']='held-out'
        with self.assertRaises(ValueError):ServiceTable(doc,held_out=['held-out'])
        doc=self.table();doc['measurements'][0]['cooperative_cpu_scale']=1.3
        with self.assertRaises(ValueError):ServiceTable(doc)


class SharedPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bridge=NativePolicy()

    def test_original_demand_and_prefill_do_not_confuse_credit(self):
        d=Demand(self.bridge,row=5)
        d.observe(3,[0,1,2,3]*5,5,True)
        self.assertGreater(d.score(3*1024),0)
        self.assertEqual(d.score(3*1024+4),0) # Executed substitute is never an input.
        before=copy.deepcopy(d.scores)
        d.observe(3,[4,5,6,7],1,False)
        self.assertEqual(before,d.scores)

    def test_cpu_price_changes_actions_but_not_physical_cost(self):
        prices={k:[0]+[1]*16 for k in ('cpu_extra_ms','gpu_extra_ms','helper_extra_ms','gpu_base_ms')}
        prices.update(cpu_base_ms=0,helper_base_ms=0,h2d_ms=1,d2h_ms=1)
        uses=[1,1,1,1]+[0]*124
        a,_=self.bridge.stage(uses,[0,1,2,3],[],[],[0,1,2,3],2,1,prices,.1)
        b,_=self.bridge.stage(uses,[0,1,2,3],[],[],[0,1,2,3],2,1,prices,10)
        self.assertNotEqual(a,b)
        physical=ServiceTable(dict(schema=1,units='ms',measurements=[measured('cpu',5,rows=1,groups=[1]*4,workers=16)]))
        x=physical.estimate('cpu',rows=1,groups=[1]*4,workers=16)
        self.assertEqual(x.ms,5)

    def test_no_refill_and_victim_cost_are_explicit(self):
        self.assertEqual(self.bridge.choose('primary',[(1,10,-1)],[(2,10,-1)],[1,2,2,0,0])[:2],[-1,-1])
        self.assertEqual(self.bridge.choose('primary',[(1,10,0)],[(2,0,-1)],[1,2,2,0,0])[:3],[1,2,0])


if __name__=='__main__':unittest.main()
