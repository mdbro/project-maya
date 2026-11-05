import copy
import unittest
from types import SimpleNamespace
from cooperative_replay.execution import Estimate, Scheduler, fixed_schedule_witness
from cooperative_replay.costs import CoverageError, ServiceTable
from cooperative_replay.policy import NativePolicy, Demand
from cooperative_replay.machine import Machine
from cooperative_replay.validation import require_search_gates, rank_held_out, timing_validation
from cooperative_replay.structure import helper_launches,width_histogram,execution_shape
from cooperative_replay.overlap import fit_observation
from cooperative_replay.diagnostics import compare


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
        self.assertFalse(v.qualified);self.assertLess(v.high-v.low,v.ms*.1) # Only observed repeats; no invented 20% margin.

    def test_rejects_timing_leakage_and_policy_prices(self):
        doc=self.table();doc['measurements'][0]['request_id']='held-out'
        with self.assertRaises(ValueError):ServiceTable(doc,held_out=['held-out'])
        doc=self.table();doc['measurements'][0]['cooperative_cpu_scale']=1.3
        with self.assertRaises(ValueError):ServiceTable(doc)

    def test_helper_preserves_distinct_launches_despite_same_old_summary(self):
        measurements=[measured('helper',1,rows=4,groups=[w]*n) for w in range(1,5) for n in (1,2)]
        table=ServiceTable(dict(schema=1,units='ms',measurements=measurements))
        a,b=[1,1,4,4],[1,2,3,4]
        self.assertEqual((len(a),sum(a),max(a)),(len(b),sum(b),max(b)))
        self.assertNotEqual(width_histogram(a),width_histogram(b))
        self.assertEqual(table.estimate('helper',rows=4,groups=a).ms,2)
        self.assertEqual(table.estimate('helper',rows=4,groups=b).ms,4)

    def test_chunking_and_record_reuse_survive_shape_encoding(self):
        launches=helper_launches([16,9,1])
        self.assertEqual([(x['width'],x['jobs']) for x in launches],[(1,2),(8,3)])
        self.assertEqual(launches[1]['reuse'],(1,2))
        table=ServiceTable(dict(schema=1,units='ms',measurements=[
            measured('helper',2.5,rows=16,groups=[16]),measured('helper',1.5,rows=8,groups=[8,8])]))
        self.assertEqual(table.estimate('helper',rows=16,groups=[16]).ms,2.5)
        self.assertEqual(table.estimate('helper',rows=8,groups=[8,8]).ms,1.5)
        self.assertEqual(execution_shape('primary',5,[1,2])['launch_batches'],5)

    def test_interpolation_can_be_supported_by_whole_shape_holdouts(self):
        table=ServiceTable(dict(schema=1,units='ms',measurements=[
            measured('helper',n*.1,rows=4,groups=[1]*n) for n in (1,3,5,7)]))
        value=table.estimate('helper',rows=4,groups=[1]*4)
        self.assertAlmostEqual(value.ms,.4);self.assertTrue(value.qualified)
        checks=next(iter(table.interpolation_checks.values()))
        self.assertEqual(len(checks['checks']),2)

    def test_primary_mixed_columns_preserve_fixed_launch_geometry(self):
        table=ServiceTable(dict(schema=1,units='ms',measurements=[
            measured('primary',.1,rows=4,groups=[]),measured('primary',.2,rows=4,groups=[4,4]),
            measured('primary',.3,rows=4,groups=[1]*6)]))
        # Nine useful columns exceed either sampled total (8 and 6), but the
        # five active <=8-column chunks are inside the measured 0..6 jobs.
        value=table.estimate('primary',rows=4,groups=[3,2,2,1,1])
        self.assertFalse(value.qualified)
        with self.assertRaises(CoverageError):table.estimate('primary',rows=3,groups=[3,2,2,1,1])


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


class ReconciliationTest(unittest.TestCase):
    def test_diagnostic_uses_scheduled_service_without_erasing_isolated_error(self):
        route=dict(forward=1,layer=3,generation=True,rows=1)
        capture=SimpleNamespace(request_id='heldout',routes=[route],events=[
            dict(type='expert_assignment',forward=1,layer=3,assignments=[dict(service='cpu',expert=7)]),
            dict(type='service',forward=1,layer=3,steady_ns=10_000_000,coop_cpu_run_ms=5,host_service_ms=8),
            dict(type='generation_begin',steady_ns=0),dict(type='generation_end',steady_ns=10_000_000)])
        operation=dict(id=0,isolated_ms=4,predicted_service_ms=5,pressure_ms=1,queue_ms=1,
                       dependency_delay_ms=2,start_ms=3,finish_ms=8)
        prediction=dict(physical_sha256='independent',layers=[dict(forward=1,layer=3,rows=1,
            service_ready_ms=0,cpu=[7],cpu_op=0,host_reduce_op=0)],operations=[operation],counts={},
            requests=[dict(request_id='heldout',generation_ms=10,emitted=0)],assignment_mode='recorded')
        result=compare(capture,prediction,SimpleNamespace(sha256='independent'))
        row=result['service']['cpu']
        self.assertAlmostEqual(row['isolated_to_live_relative_error'],-.2)
        self.assertEqual(row['scheduled_to_live_relative_error'],0)
        self.assertIsNone(result['service_rows'][0]['observed_completion_from_route_ms'])

    def test_overlap_fit_uses_aligned_execution_not_shorter_elapsed_span(self):
        row=dict(a='cpu',b='copy',isolated_a_ms=8,isolated_b_ms=4,isolated_boundary='active_service',
            execution_intervals=dict(scope='execution',complete=True,clock_id='aligned',
                a_clock_id='aligned',b_clock_id='aligned',a_ns=[[0,10_000_000]],b_ns=[[5_000_000,9_000_000]]))
        result=fit_observation(row)
        self.assertEqual(result[0]['factor'],2)
        self.assertEqual(result[0]['overlap_ms'],4)
        row['execution_intervals']['b_ns']=[[11_000_000,15_000_000]]
        with self.assertRaisesRegex(ValueError,'no measured overlap'):fit_observation(row)
        row['execution_intervals']['b_clock_id']='other'
        with self.assertRaisesRegex(ValueError,'common measured clock'):fit_observation(row)


class OwnershipTest(unittest.TestCase):
    def machine(self, backing=False):
        class Physical:
            contention={};sha256='unit-fixture'
            document={'cpu_participants':16}
            def estimate(self,kind,**kwargs):return Estimate(2,2,2,('fixture',))
        prices={k:[0]+[1]*16 for k in ('cpu_extra_ms','gpu_extra_ms','helper_extra_ms','gpu_base_ms')}
        prices.update(cpu_base_ms=0,helper_base_ms=0,h2d_ms=1,d2h_ms=1)
        state=dict(row_clock=5,block_rows=5,scores=[[2,100,5]],primary_credit=8,primary_credit_row=5,
                   helper_credit=0,helper_credit_row=5,host_spares=[900,901],helper_buffers=[902,903])
        snap=dict(forward=1,record_bytes=1000,dimension=4,
            policy=dict(primary_backing=False,service_saved_ms=1,configured_service_saved_ms=1,
                        configured_h2d_ms=0,configured_d2h_ms=0,interference_ms=0),
            layers=[dict(layer=0,primary=[0],primary_frames=[[0,100]],host_frames=[[2,200],[3,201]],
                         helper=[1],primary_backing_frames=[])],helper_frames=[[0,1,0]],
            staging=[[2,-1,0,0,0,0,101],[-1,-1,0,0,0,0,102],[-1,-1,0,0,0,0,103],[-1,-1,0,0,0,0,104]],
            replay_state=state,batch_staging_calibration=prices)
        m=Machine(snap,Physical(),backing=backing);m.generation=True
        return m

    def test_retention_reserves_existing_frame_and_publication_boundary(self):
        m=self.machine();frames=set(m.gpu_frames)
        m.admission(9)
        self.assertEqual(len(m.exchanges),1);self.assertTrue(m.slots[0].reserved)
        self.assertEqual(m.primary,{0:100});self.assertEqual(m.host[2],200)
        self.assertEqual(m.counts['optional_h2d_bytes'],0)
        m.scheduler.drain();m.publish(0)
        self.assertEqual(len(m.exchanges),1) # Current routing snapshot protects both identities.
        m.publish()
        self.assertEqual(m.primary,{2:101});self.assertEqual(m.slots[0].address,100)
        self.assertEqual(m.host[0],900);self.assertEqual(m.spares[0],200)
        self.assertEqual(m.gpu_frames,frames);m.assert_ownership()

    def test_active_reader_cannot_be_retained_and_backing_has_no_d2h(self):
        m=self.machine()
        reader=m.scheduler.submit('primary',Estimate(4,4,4),('gpu',))
        m.slots[0].reader=reader;m.admission(9)
        self.assertEqual(len(m.exchanges),0)
        backed=self.machine(backing=True);backed.admission(9);backed.scheduler.drain();backed.publish()
        self.assertEqual(backed.counts['optional_d2h_bytes'],0)
        self.assertEqual(set(backed.backing_frames),set(backed.primary))
        self.assertEqual(len(backed.gpu_frames),5)


class GateTest(unittest.TestCase):
    def test_tiny_operation_percentage_does_not_set_the_decision_budget(self):
        prediction=dict(physical_sha256='frozen',requests=[dict(request_id='test',generation_ms=1000,tiny_ms=1.5,unit_ms=1)])
        comparisons=[]
        for kind in ('grouped_service','queue_delay','publication','branch_finish','request_duration'):
            metric='generation_ms' if kind=='request_duration' else 'tiny_ms' if kind=='grouped_service' else 'unit_ms'
            comparisons.append(dict(request_id='test',category=kind,observed_ms=1000 if kind=='request_duration' else 1,
                uncertainty_ms=0,prediction_ref=dict(scope='request',request_id='test',metric=metric)))
        observations=dict(physical_sha256='frozen',calibration_request_ids=['train'],held_out_request_ids=['test'],comparisons=comparisons)
        self.assertEqual(timing_validation(prediction,observations,required_effect_fraction=.03)['status'],'pass')
        # The same exact whole-request prediction cannot hide large opposing
        # local errors, even though they cancel arithmetically.
        prediction['requests'][0].update(tiny_ms=51,opposite_ms=1)
        comparisons[0]['observed_ms']=26
        comparisons[1]['observed_ms']=26
        comparisons[1]['prediction_ref']['metric']='opposite_ms'
        self.assertEqual(timing_validation(prediction,observations,required_effect_fraction=.03)['status'],'unresolved')

    def test_decision_reproduction_does_not_authorize_ranking(self):
        with self.assertRaises(ValueError):require_search_gates([dict(stage=1,status='pass')],'physical')
        gates=[dict(stage=1,status='pass'),dict(stage=2,status='pass',physical_sha256='physical'),
               dict(stage=3,status='unresolved',physical_sha256='physical')]
        with self.assertRaises(ValueError):require_search_gates(gates,'physical')

    def test_three_percent_inside_five_percent_is_unresolved(self):
        rows=[dict(name='incumbent',requests=[dict(request_id='x',generation_ms=100,emitted=10,uncertainty_ms=2.5)]),
              dict(name='candidate',requests=[dict(request_id='x',generation_ms=97,emitted=10,uncertainty_ms=2.5)])]
        ranked=rank_held_out(rows,'incumbent')
        self.assertEqual(ranked[0]['name'],'candidate');self.assertFalse(ranked[0]['retained'])


if __name__=='__main__':unittest.main()
