#!/usr/bin/env python3
"""Rules of the bounded follow-up, checked on constructed numbers before any run exists."""
import unittest

import native_followup_analysis as analysis
import native_passes as passes


def row(cpu=0, helper=0, new=0, ordinal=0, layer=3, lane_begin=0.1, cpu_end=None, enqueue=None, record_end=None, ready=None, published=None,
        collect_end=None, end=1.0, next_pre=None, cycle=2.0):
    return passes.Pass(forward=ordinal + 100, ordinal=ordinal, layer=layer, index=layer - 3, position=0, final_layer=next_pre is None,
                       cpu=cpu, helper=helper, resident=4 - cpu - helper - new, ready=0, new=new, boundary=.01, plan=.02, helper_submit=None,
                       primary_submit=lane_begin, cpu_end=cpu_end, collect_end=collect_end, end=end, helper_enqueue=enqueue, helper_worker_start=None,
                       helper_record_end=record_end, helper_ready=ready, helper_published=published, next_pre_end=next_pre, cycle_ms=cycle)


class EffectRule(unittest.TestCase):
    def test_interval_inside_band_is_bounded(self):
        rows = [dict(a=100, b=100 + d) for d in (0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'bounded')

    def test_interval_wholly_outside_band_is_material(self):
        rows = [dict(a=100, b=110 + d) for d in (0.1, -0.1, 0.2, -0.2)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'material')

    def test_a_broad_interval_is_not_evidence_of_no_effect(self):
        rows = [dict(a=100, b=100 + d) for d in (9, -9, 8, -8, 7, -7)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'unresolved')

    def test_single_block_never_resolves(self):
        self.assertEqual(analysis.block_effect([dict(a=100, b=100)], 'b', 'a')['classification'], 'unresolved')


class Resolution(unittest.TestCase):
    def test_block_effects_do_not_count_as_noise(self):
        rows = [dict(x=100 * f, y=110 * f) for f in (1, 2, 3, 4)]
        self.assertAlmostEqual(analysis.two_way_residual_sd(rows, ['x', 'y'])['residual_sd'], 0, places=9)

    def test_more_noise_needs_more_pairs(self):
        self.assertLess(analysis.pairs_needed(.01, .01), analysis.pairs_needed(.03, .01))
        self.assertIsNone(analysis.pairs_needed(None, .01))


class QueryRule(unittest.TestCase):
    def test_gain_goes_on_to_complete_requests(self):
        effect = dict(mean=-.05, low=-.07, high=-.03)
        self.assertEqual(analysis.query_decision(effect, None)['decision'], 'continue_to_requests')

    def test_useful_needs_complete_requests(self):
        effect = dict(mean=-.05, low=-.07, high=-.03)
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.03, per_prompt=dict(a=-.05, b=-.01)))['decision'], 'useful')
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.03, per_prompt=dict(a=-.07, b=.01)))['decision'], 'inconclusive')
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.01, per_prompt=dict(a=-.01, b=-.01)))['decision'], 'inconclusive')

    def test_no_gain_is_negative(self):
        self.assertEqual(analysis.query_decision(dict(mean=0, low=-.01, high=.01), None)['decision'], 'negative')
        self.assertEqual(analysis.query_decision(dict(mean=.05, low=.01, high=.09), None)['decision'], 'negative')

    def test_uncertain_stays_off(self):
        self.assertEqual(analysis.query_decision(dict(mean=-.03, low=-.08, high=.02), None)['decision'], 'inconclusive')


class JointTimeline(unittest.TestCase):
    """The user's hypothetical executions: equal durations, different completion."""

    def test_maximum_is_taken_per_pass_from_a_common_origin(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=4, enqueue=0, published=1, collect_end=4)
        b = row(cpu=1, helper=1, lane_begin=0, cpu_end=1, enqueue=0, published=4, collect_end=4, ordinal=1)
        summary = passes.mean_of_max_vs_max_of_means([a, b])
        self.assertEqual(summary['max_of_means'], 2.5)
        self.assertEqual(summary['mean_of_max'], 4)

    def test_shortening_the_lane_that_is_not_last_saves_nothing(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=1, enqueue=0, published=4, collect_end=4)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(cpu_factor=.5)), 0)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(helper_factor=.5)), 2)

    def test_shortening_one_lane_can_expose_the_other(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=3, enqueue=0, published=4, collect_end=4)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(helper_factor=.5)), 1)

    def test_overlap_opportunity_is_limited_by_the_pass_lane(self):
        short = row(new=1, lane_begin=0, cpu=1, cpu_end=.2, collect_end=None)
        long = row(new=1, lane_begin=0, cpu=1, cpu_end=5, collect_end=None, ordinal=1)
        result = passes.overlap_opportunity([short, long], copy_ms=.5, expert_ms=.1)
        self.assertAlmostEqual(result['upper_bound_ms'], .2 + .6)
        self.assertAlmostEqual(result['copies_only_ms'], .2 + .5)

    def test_paired_passes_require_identical_work(self):
        a = row(cpu=1, new=1, next_pre=1.1, cycle=2.0)
        same = row(cpu=1, new=1, next_pre=1.1, cycle=1.5)
        different = row(cpu=2, new=1, next_pre=1.1, cycle=1.5)
        self.assertEqual(analysis.paired_passes([a], [same])['same_work_windows'], 1)
        self.assertAlmostEqual(analysis.paired_passes([a], [same])['by_new_copies']['1']['cycle']['mean'], -.5)
        self.assertEqual(analysis.paired_passes([a], [different])['same_work_windows'], 0)


class ClockLink(unittest.TestCase):
    def test_explained_variance(self):
        self.assertAlmostEqual(analysis.explained_variance([1, 1, 2, 2], ['a', 'a', 'b', 'b']), 1)
        self.assertAlmostEqual(analysis.explained_variance([1, 2, 1, 2], ['a', 'a', 'b', 'b']), 0)
        self.assertIsNone(analysis.explained_variance([1, 1], ['a', 'b']))


class BackingStep(unittest.TestCase):
    def step(self, action, enabled):
        import pathlib, tempfile, types
        from unittest import mock
        import native_followup as followup
        replies = {action: dict(engine_pid=7, pre_allocation_windows_physical=9), 'inspect': dict(primary_backing_enabled=enabled, ownership_complete=True, primary_backing_valid_records=437)}
        with tempfile.TemporaryDirectory() as d:
            fake = types.SimpleNamespace(base='x', a=types.SimpleNamespace(timeout=1), directory=pathlib.Path(d))
            with mock.patch.object(followup, 'post', lambda base, endpoint, body, timeout: replies[body['action']]):
                return followup.FollowUp.run_backing(fake, dict(step='s', action=action))

    def test_confirms_the_state_it_asked_for(self):
        self.assertEqual(self.step('backing-on', True)['errors'], [])
        self.assertEqual(self.step('backing-off', False)['errors'], [])

    def test_reports_a_state_that_did_not_change(self):
        self.assertTrue(self.step('backing-on', False)['errors'])
        self.assertTrue(self.step('backing-off', True)['errors'])


class HostStep(unittest.TestCase):
    def run_step(self, reply):
        import pathlib, tempfile, types
        from unittest import mock
        import native_followup as followup
        import native_host_control as host
        with tempfile.TemporaryDirectory() as d:
            fake = types.SimpleNamespace(directory=pathlib.Path(d))
            with mock.patch.object(host, 'request', lambda actions: reply):
                return followup.FollowUp.run_host(fake, dict(step='plan', actions=[dict(op='power-plan', scheme='balanced')]))

    def test_a_successful_operation_has_no_errors(self):
        self.assertEqual(self.run_step(dict(ok=True, actions=[dict(exit=0)]))['errors'], [])

    def test_a_failed_operation_stops_the_phase(self):
        self.assertTrue(self.run_step(dict(ok=False, error='unknown op', actions=[]))['errors'])


class PrefixReuse(unittest.TestCase):
    """A conversation's next turn (expect_reuse) passes only if the engine kept the earlier turn's prefix."""
    def check(self, cache_n=None, cached=None):
        import native_followup as followup
        item = dict(timings={} if cache_n is None else dict(cache_n=cache_n),
                    usage={} if cached is None else dict(prompt_tokens_details=dict(cached_tokens=cached)))
        return followup.prefix_reuse_check(item)['verified']

    def test_kept_prefix_passes(self):
        self.assertTrue(self.check(cache_n=231, cached=231))

    def test_a_fresh_read_fails(self):
        self.assertFalse(self.check(cache_n=0, cached=0))

    def test_no_count_fails(self):
        self.assertFalse(self.check())

    def test_conflicting_counts_fail(self):
        self.assertFalse(self.check(cache_n=231, cached=0))


class ConversationPayload(unittest.TestCase):
    """A conditioning case may carry a re-sent conversation (messages: user and assistant turns) instead of one prompt."""
    def payload(self, case, tune=None):
        import native_followup as followup
        campaign = followup.FollowUp.__new__(followup.FollowUp)   # (payload reads only the protocol)
        campaign.protocol = dict(model_name='m', seed=7, reasoning_effort='high', tune={'fresh_context': 1})
        return campaign.payload(case, 64, tune)

    def test_messages_are_sent_as_a_deep_copy(self):
        turns = [dict(role='user', content='a'), dict(role='assistant', content='b'), dict(role='user', content='c')]
        p = self.payload(dict(id='conversation', messages=turns), dict(prompt_memo=1))
        self.assertEqual(p['messages'], turns)
        p['messages'][0]['content'] = 'changed'
        self.assertEqual(turns[0]['content'], 'a')
        self.assertEqual(p['strata_tune'], {'fresh_context': 1, 'prompt_memo': 1})

    def test_without_messages_the_prompt_is_one_user_message(self):
        one = [dict(role='user', content='hi')]
        self.assertEqual(self.payload(dict(id='p', prompt='hi'))['messages'], one)
        self.assertEqual(self.payload(dict(id='p', prompt='hi', messages=[]))['messages'], one)

class FinalStat(unittest.TestCase):
    """The ledger's final statistics carry the keys only the latest STAT line has, when that line is the request's."""
    def item(self, final, now):
        return dict(engine_stats=dict(final=final), metrics_after=dict(tiers=dict(now=now)))

    def test_the_latest_line_of_the_same_request_adds_its_keys(self):
        import native_followup as followup
        got = followup.final_stat(self.item(dict(request_seq=5.0, emitted=384.0, routing_prior_bonus=0.2),
                                            dict(request_seq=5.0, emitted=380.0, routing_floor=0.4, serve_monitoring=0.0,
                                                 prompt_wall_ms=5972.5, prompt_path='layer', warnings='none')))
        self.assertEqual(got['routing_floor'], 0.4)
        self.assertEqual(got['prompt_path'], 'layer')
        self.assertEqual(got['emitted'], 384.0)   # (the history's value wins)

    def test_another_requests_line_adds_nothing(self):
        import native_followup as followup
        final = dict(request_seq=5.0, emitted=384.0)
        self.assertEqual(followup.final_stat(self.item(final, dict(request_seq=6.0, routing_floor=0.4))), final)
        self.assertEqual(followup.final_stat(self.item(final, dict(routing_floor=0.4))), final)
        self.assertEqual(followup.final_stat(dict(engine_stats=dict(final=final))), final)
        self.assertEqual(followup.final_stat(self.item({}, dict(request_seq=5.0, routing_floor=0.4))), {})


class FinalComparison(unittest.TestCase):
    def rows(self, new_rates):
        rows = []
        for i, gain in enumerate(new_rates):
            for arm, phase, rate in (('old', f'old{i}', 15.0), ('new', f'new{i}', 15.0 * (1 + gain))):
                for prompt, factor in (('code', 1.0), ('prose', .9)):
                    rows.append(dict(phase=phase, arm=arm, budget=128, prompt=prompt, emitted=128, generation_ms=128000 / (rate * factor)))
        return rows, [dict(old=f'old{i}', new=f'new{i}') for i in range(len(new_rates))]

    def test_rates_pool_tokens_over_time(self):
        import native_final_comparison as final
        self.assertAlmostEqual(final.pooled([dict(emitted=100, generation_ms=1000), dict(emitted=100, generation_ms=3000)])['tokens_per_second'], 50)

    def test_consistent_gain_is_confirmed(self):
        import native_final_comparison as final
        rows, pairs = self.rows([.06, .07, .065])
        result = final.evaluate(rows, pairs)
        self.assertTrue(result['budgets'][128]['confirmed'])
        self.assertEqual(result['regressions'], [])

    def test_a_losing_load_stays_listed_and_blocks_confirmation(self):
        import native_final_comparison as final
        rows, pairs = self.rows([.10, -.02, .05])
        result = final.evaluate(rows, pairs)
        self.assertFalse(result['budgets'][128]['confirmed'])
        self.assertEqual(len(result['regressions']), 2)

    def test_a_missing_request_is_an_error_not_a_gap(self):
        import native_final_comparison as final
        with self.assertRaises(ValueError):
            final.collect(dict(study=dict(requests=[dict(step='a')])), dict(steps=[]))


class UsabilityChecks(unittest.TestCase):
    GOOD = '''```python
import heapq
def order_jobs(deps):
    names = set(deps)
    for values in deps.values():
        names.update(values)
    needs = {n: set() for n in names}
    for job, values in deps.items():
        needs[job].update(values)
    heap = sorted(n for n in names if not needs[n])
    heapq.heapify(heap)
    order = []
    while heap:
        n = heapq.heappop(heap)
        order.append(n)
        for m in names:
            if n in needs[m]:
                needs[m].discard(n)
                if not needs[m]:
                    heapq.heappush(heap, m)
    if len(order) != len(names):
        raise ValueError('cycle')
    return order
```'''

    def test_a_correct_function_passes(self):
        import native_usability_matrix as matrix
        self.assertEqual(matrix.check_order_jobs(self.GOOD)['status'], 'pass')

    def test_a_wrong_order_fails(self):
        import native_usability_matrix as matrix
        wrong = self.GOOD.replace('n = heapq.heappop(heap)', 'n = heap.pop()')
        self.assertEqual(matrix.check_order_jobs(wrong)['status'], 'fail')

    def test_escape_constructs_are_rejected_before_anything_runs(self):
        import native_usability_matrix as matrix
        # These strings are only parsed. The test's point is that each is refused and never executed.
        attempts = ['import os', 'from os import system', 'import typing\ntyping.sys.exit(0)', 'from typing import sys', '().__class__', "x = type('a', (), {})", 'class A: pass',
                    "getattr(1, 'real')", 'def g(): yield\ng().gi_frame', "'{0.__class__}'.format(1)", 'eval("1")', '__builtins__', 'from collections import _sys']
        for attempt in attempts:
            self.assertIsNotNone(matrix.generated_code_problem(attempt), attempt)
            self.assertEqual(matrix.check_order_jobs(f'```python\n{attempt}\ndef order_jobs(deps): return []\n```')['status'], 'rejected', attempt)

    def test_ordinary_code_is_accepted_by_the_rules(self):
        import native_usability_matrix as matrix
        self.assertIsNone(matrix.generated_code_problem(self.GOOD.strip('`').replace('python\n', '', 1)))
        self.assertIsNone(matrix.generated_code_problem('from __future__ import annotations\nfrom typing import Dict\nimport collections.abc\nif __name__ == "__main__":\n    print(1)'))

    def test_the_sandbox_has_no_network_home_or_writable_system(self):
        import json
        import native_usability_matrix as matrix
        probe = ("import json,socket,os\nr={}\ntry:\n socket.create_connection(('1.1.1.1',53),1);r['net']=True\nexcept OSError:r['net']=False\n"
                 "r['home']=os.path.exists('/home');r['env']=dict(os.environ)\ntry:\n open('/usr/x','w');r['write']=True\nexcept OSError:r['write']=False\nprint(json.dumps(r))")
        out = json.loads(matrix.sandboxed(['-c', probe], '').stdout)
        self.assertEqual((out['net'], out['home'], out['write']), (False, False, False))
        self.assertNotIn('HOME', out['env'])
        self.assertNotIn('PATH', out['env'])

    def test_without_the_sandbox_nothing_runs(self):
        from unittest import mock
        import native_usability_matrix as matrix
        with mock.patch.object(matrix.shutil, 'which', lambda name: None):
            self.assertEqual(matrix.check_order_jobs(self.GOOD)['status'], 'unchecked')

    def test_runaway_code_is_stopped(self):
        import native_usability_matrix as matrix
        self.assertEqual(matrix.check_order_jobs('```python\ndef order_jobs(deps):\n    while True:\n        pass\n```')['status'], 'execution_error')

    def test_muffin_answer_is_found(self):
        import native_usability_matrix as matrix
        out = matrix.check_muffins('There are 3 combinations: 7 boxes of 4 and 3 boxes of 6; 4 boxes of 4 and 5 boxes of 6; 1 boxes of 4 and 7 boxes of 6.')
        self.assertTrue(out['states_three'])
        self.assertEqual(out['pairs_found'], [(1, 7), (4, 5), (7, 3)])


class UsabilityBuiltins(unittest.TestCase):
    """A usability protocol with "builtins" runs serving's settings and its planner: only the arm's keys are sent."""
    PROTOCOL = dict(builtins=True, tune={}, arms=dict(lenient={'spec_plan_file': 'refit.txt'},
                                                      exact={'spec_plan_file': 'refit.txt', 'spec_policy': 'sampled'}))

    def test_a_request_sends_the_arms_keys_and_nothing_else(self):
        import native_usability_matrix as matrix
        self.assertEqual(matrix.request_tune(self.PROTOCOL, 'exact'), {'fresh_context': 1, 'spec_plan_file': 'refit.txt', 'spec_policy': 'sampled'})
        self.assertNotIn('spec', matrix.request_tune(self.PROTOCOL, 'lenient'))   # (no fixed width: the planner decides)

    def test_the_command_lines_values_are_the_engines_built_ins(self):
        import engine_builtins
        import native_usability_matrix as matrix
        b = engine_builtins.request_builtins(True)
        v = matrix.command_values(self.PROTOCOL, matrix.request_tune(self.PROTOCOL, 'exact'))
        self.assertEqual((v['routing_prior_bonus'], v['spec_ell'], v['spec'], v['spec_policy']), (b['routing_prior_bonus'], b['spec_ell'], b['spec'], 'sampled'))

    def test_an_older_protocol_still_sends_its_full_tune(self):
        import native_usability_matrix as matrix
        old = dict(tune={'spec': 5, 'routing_prior_bonus': 0.1}, arms=dict(control={}))
        self.assertEqual(matrix.request_tune(old, 'control'), {'spec': 5, 'routing_prior_bonus': 0.1})

    def test_delivery_tune_only_sends_no_width(self):
        import argparse
        import native_delivery as delivery
        a = argparse.Namespace(tune_only=True, tune_json={'spec_plan_file': 'refit.txt'}, arm='selected', width=7)
        self.assertEqual(delivery.build_tune(a), {'fresh_context': 1, 'spec_plan_file': 'refit.txt'})

    def test_the_settings_a_request_reported_are_kept(self):
        import native_usability_matrix as matrix
        item = dict(engine_stats=dict(final=dict(request_seq=3.0, spec_policy='sampled', spec_rounds=10.0, draft_off=35.0)),
                    metrics_after=dict(tiers=dict(now=dict(request_seq=3.0, spec_plan_file='refit.txt'))))
        s = matrix.settings_seen(item)
        self.assertEqual((s['spec_policy'], s['spec_plan_file'], s['drafts_per_round']), ('sampled', 'refit.txt', 3.5))


class ArmScreen(unittest.TestCase):
    def rows(self, prose, code):
        return [dict(prompt=p, arms=dict(control=dict(rate=100.0), candidate=dict(rate=100.0 * (1 + v))))
                for p, values in (('prose', prose), ('code', code)) for v in values]

    def test_consistent_gain_qualifies(self):
        import native_arm_screen as arm_screen
        self.assertTrue(arm_screen.effects(self.rows([-.05, -.04, -.06, -.05], [-.03, -.04, -.03, -.02]), 'control')['candidate']['qualifies'])

    def test_one_prompt_slower_does_not(self):
        import native_arm_screen as arm_screen
        self.assertFalse(arm_screen.effects(self.rows([-.10, -.11, -.09, -.10], [.01, .01, .02, .01]), 'control')['candidate']['qualifies'])

    def test_gain_below_the_band_does_not(self):
        import native_arm_screen as arm_screen
        self.assertFalse(arm_screen.effects(self.rows([-.01, -.012, -.011, -.01], [-.012, -.01, -.011, -.01]), 'control')['candidate']['qualifies'])


if __name__ == '__main__':
    unittest.main()
