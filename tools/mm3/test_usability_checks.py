#!/usr/bin/env python3
"""Tests of the short usability check (2026-11-14, from the audit in runs/usability-audit-20261114): the answer checks against
real answers with hand labels, the decision rule's arithmetic, and the evaluator's gates.

    python3 -m unittest test_usability_checks      (from tools/mm3)

Real answers: testdata/muffins-real-and-probes-20261114.json (every distinct finished muffins answer of 2026-11-07..14, all
right by reading, and the audit's constructed probes), testdata/muffins-answers-all-20261114.json (the 22 of 2026-11-14) and
testdata/order-jobs-action-items-20261114.json (the 22 order-jobs answers of 2026-11-14, each read in full and re-run on 23
independent cases, and 25 action-items answers)."""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import usability_checks as uc  # noqa: E402

DATA = json.loads((HERE / 'testdata' / 'muffins-real-and-probes-20261114.json').read_text())


class Muffins(unittest.TestCase):
    def test_no_real_answer_is_scored_wrong(self):
        # every distinct finished answer on disk (2026-11-07..14) is right by reading; the checker may defer to a reader, never fail one
        verdicts = {a['source'] + ' ' + a['sha1']: uc.check_muffins_v2(a['content'])['verdict'] for a in DATA['answers']}
        wrong = [k for k, v in verdicts.items() if v == 'wrong']
        self.assertEqual(wrong, [])
        read = [k for k, v in verdicts.items() if v == 'read']
        self.assertLessEqual(len(read), 3, read)          # (the audit measured 2 of 111)

    def test_the_four_answers_the_previous_checker_failed(self):
        for a in DATA['answers']:
            if any(s in a['content'] for s in ('3 boxes for 6', '3 boxes by 6', '5 boxes for 6', 'needs 11 boxes', 'violates the 10-box limit')):
                self.assertNotEqual(uc.check_muffins_v2(a['content'])['verdict'], 'wrong', a['source'])

    def test_constructed_probes_are_never_unsafe(self):
        # a right answer never comes out wrong; a wrong answer never comes out right
        for p in DATA['probes']:
            finish = 'length' if p['name'].startswith('cut off') else 'stop'
            v = uc.check_muffins_v2(p['text'], finish)['verdict']
            if p['should_be_right']:
                self.assertNotEqual(v, 'wrong', p['name'])
            else:
                self.assertNotEqual(v, 'right', p['name'])

    def test_a_cut_off_answer_is_not_judged(self):
        self.assertEqual(uc.check_muffins_v2('There are 3 comb', 'length')['verdict'], 'cut off')

    def test_a_plain_wrong_count(self):
        text = '## Final answer\n\nThere are 4 combinations: (10,1), (7,3), (4,5), (1,7).'
        self.assertEqual(uc.check_muffins_v2(text)['verdict'], 'wrong')


class Fences(unittest.TestCase):
    CODE = 'def order_jobs(deps):\n    return []\n'

    def test_fence_styles(self):
        for text in (f'```python\n{self.CODE}```', f'```python3\n{self.CODE}```', f'``` python\n{self.CODE}```', f'~~~python\n{self.CODE}~~~',
                     f'```python\n{self.CODE}'):
            self.assertTrue(uc.fenced_code_v2(text).lstrip().startswith('def order_jobs'), text)

    def test_the_defining_block_not_a_later_usage_block(self):
        text = f'```python\n{self.CODE}```\n\nExample:\n\n```python\nprint(order_jobs({{}}))\n```'
        self.assertTrue(uc.fenced_code_v2(text).lstrip().startswith('def order_jobs'))


class DecisionRule(unittest.TestCase):
    def test_two_seeds_can_never_show_harm(self):
        d = uc.decide(1, 2, 0, 2, 0.18)
        self.assertEqual(d['outcome'], "can't tell")
        self.assertFalse(d['any_outcome_can_show_harm'])
        self.assertEqual(uc.decide(2, 2, 0, 2, 0.18)['outcome'], "can't tell")     # (p = 0.17)

    def test_four_of_four_against_none_shows_harm(self):
        self.assertEqual(uc.decide(4, 4, 0, 4, 0.18)['outcome'], 'harm shown')

    def test_equal_counts_are_not_harm(self):
        self.assertNotEqual(uc.decide(1, 4, 1, 4, 0.10)['outcome'], 'harm shown')

    def test_fisher_matches_a_hand_value(self):
        # 2 of 2 against 0 of 2: P = C(2,2)C(2,0)/C(4,2) = 1/6
        self.assertAlmostEqual(uc.fisher_one_sided(2, 2, 0, 2), 1 / 6, places=9)

    def test_detectable_sizes(self):
        # the smallest tested-arm failure rate shown 80% of the time, control at 10% (rule_stats.txt)
        self.assertEqual(uc.detectable(16, 16, 0.10), 0.55)
        self.assertIsNone(uc.detectable(2, 2, 0.10))

    def test_no_harm_needs_enough_answers(self):
        # 0 of 48 against 0 of 48 at a 10% base rate: the tested arm's 95% bound sits below what the run could show
        self.assertEqual(uc.decide(0, 48, 0, 48, 0.10)['outcome'], 'no harm of the stated size')


class OrderJobsHarness(unittest.TestCase):
    def test_labels(self):
        import native_usability_matrix as matrix
        self.assertEqual(matrix.order_jobs_label({'status': 'pass'}), 'right')
        self.assertEqual(matrix.order_jobs_label({'status': 'fail'}), 'wrong')
        for s in ('rejected', 'execution_error', 'unchecked'):
            self.assertEqual(matrix.order_jobs_label({'status': s}), 'not scored', s)

    def test_float_and_indexerror_are_available(self):
        import native_usability_matrix as matrix, shutil
        if not shutil.which('bwrap'):
            self.skipTest('no sandbox')
        code = ('```python\nimport heapq\ndef order_jobs(deps):\n    best = float("inf")\n    try:\n        [][0]\n    except IndexError:\n        pass\n'
                '    nodes = set(deps)\n    for v in deps.values():\n        nodes.update(v)\n    indeg = {n: 0 for n in nodes}\n    out = {n: [] for n in nodes}\n'
                '    for k, v in deps.items():\n        for p in set(v):\n            indeg[k] += 1\n            out[p].append(k)\n'
                '    h = [n for n in nodes if indeg[n] == 0]\n    heapq.heapify(h)\n    r = []\n    while h:\n        n = heapq.heappop(h)\n        r.append(n)\n'
                '        for m in out[n]:\n            indeg[m] -= 1\n            if indeg[m] == 0:\n                heapq.heappush(h, m)\n'
                '    if len(r) != len(nodes):\n        raise ValueError("cycle")\n    return r\n```')
        self.assertEqual(matrix.check_order_jobs(code)['status'], 'pass')


# ---- the labels, the harness, the rule's outcomes and the evaluator (added with usability_eval.py) ----------------------------
import shutil      # noqa: E402
import tempfile    # noqa: E402

import native_usability_matrix as matrix   # noqa: E402
import usability_checks as checks          # noqa: E402
import usability_eval as ev                # noqa: E402
import usability_rule as rule              # noqa: E402

MORE = json.loads((HERE / 'testdata' / 'order-jobs-action-items-20261114.json').read_text())
TONIGHT = json.loads((HERE / 'testdata' / 'muffins-answers-all-20261114.json').read_text())['answers']
THREE = [(1, 7), (4, 5), (7, 3)]
RIGHT_MUFFINS = TONIGHT[0]['content']


class MuffinsLabels(unittest.TestCase):
    def test_tonights_22_real_answers_are_right(self):
        self.assertEqual(len(TONIGHT), 22)
        for a in TONIGHT:
            out = checks.check_muffins(a['content'])
            self.assertEqual((out['label'], out['final_count'], out['pairs']), ('right', 3, THREE), a['source'])

    def test_no_real_answer_is_marked_wrong_and_the_four_of_3adf4455_are_right(self):
        # (111 distinct answers of 2026-11-07..14, all right by reading; the checker of 3adf4455 marked 4 of the earlier ones wrong)
        answers = DATA['answers']
        self.assertEqual(len(answers), 111)
        labels = [checks.check_muffins(a['content'])['label'] for a in answers]
        self.assertNotIn('wrong', labels)
        self.assertGreaterEqual(labels.count('right'), 109)         # (the rest go to a reader, never to "wrong")
        failed_then = [a for a in answers if a['sha1'][:10] in ('e361c8bc58', '2d4235d9d6', 'bc195850cc', '863c8590b6')]
        self.assertEqual(len(failed_then), 4)
        for a in failed_then:
            self.assertEqual(checks.check_muffins(a['content'])['label'], 'right', a['source'])

    def test_a_box_size_is_not_a_count(self):
        # (usability-matrix-20261107 bonus30 seed 20261102: read as a final count of 6 by the checker of 3adf4455)
        text = ('## Answer\n\n**3 different combinations** work: (7 boxes of 4, 3 boxes of 6), (4, 5), and (1, 7).\n\nThe only solution that fails the box '
                'constraint is the 10-box-of-4 plus 1-box-of-6 combination, which needs 11 boxes.')
        self.assertEqual(checks.check_muffins(text)['final_count'], 3)
        self.assertEqual(checks.check_muffins(text)['label'], 'right')

    def test_substituted_prepositions_still_name_the_pairs(self):
        # (block1 bonus30 / bonus40, block2 b40l: "boxes for 6", "boxes by 6"; substituted words alone are not a failure - the user, 2026-11-10)
        for prep in ('for', 'by', 'on'):
            text = f'## Answer\n\n**3 combinations** work:\n- 7 boxes of 4 + 3 boxes {prep} 6\n- 4 boxes {prep} 4 + 5 boxes {prep} 6\n- 1 box {prep} 4 + 7 boxes {prep} 6\n'
            self.assertEqual(checks.check_muffins(text)['label'], 'right', prep)

    def test_a_note_after_the_answer_does_not_change_the_count(self):
        base = '## Answer\n\n**3 combinations** work:\n- 7 boxes of 4 + 3 boxes of 6\n- 4 boxes of 4 + 5 boxes of 6\n- 1 box of 4 + 7 boxes of 6\n\n'
        for note in ('The one other way to make 46 muffins, 10 boxes of 4 and 1 box of 6, uses 11 boxes.',
                     'Ignoring the box limit there are 4 combinations; the limit removes one.',
                     'There is one other option, 10 boxes of 4 and 1 box of 6, but it needs 11 boxes.'):
            out = checks.check_muffins(base + note)
            self.assertEqual((out['label'], out['final_count']), ('right', 3), note)

    def test_a_cut_off_probe_is_cut_off_and_both_keys_agree(self):
        for p in DATA['probes']:
            out = checks.check_muffins(p['text'], 'length' if p['name'].startswith('cut off') else 'stop')
            self.assertEqual(out['label'], out['verdict'], p['name'])        # ("verdict" is the key of 38cf0ab5; "label" the same value)
            if p['name'].startswith('cut off'):
                self.assertEqual(out['label'], 'cut off')

    def test_wrong_needs_a_stated_count_other_than_three(self):
        self.assertEqual(checks.check_muffins('## Answer\n\n**4 combinations** work: (10, 1), (7, 3), (4, 5), (1, 7).')['label'], 'wrong')
        self.assertEqual(checks.check_muffins('## Answer\n\n**2 combinations** work: (7, 3) and (4, 5).')['label'], 'wrong')
        self.assertEqual(checks.check_muffins('Use 7 boxes of 4 and 3 boxes of 6, or 4 boxes of 4 and 5 boxes of 6.')['label'], 'read')      # (no count)
        self.assertEqual(checks.check_muffins('## Answer\n\n**3 combinations** work: (10, 1), (7, 3), (4, 5). The pair (1, 7) is ruled out.')['label'], 'read')

    def test_the_matrix_keeps_the_old_fields_and_read_is_not_correct(self):
        out = matrix.check_muffins(RIGHT_MUFFINS)
        self.assertEqual((out['label'], out['correct'], out['states_three'], out['pairs_found']), ('right', True, True, THREE))
        self.assertFalse(matrix.check_muffins('Seven and three; four and five; one and seven.')['correct'])
        self.assertEqual(matrix.check_muffins(RIGHT_MUFFINS, 'length')['label'], 'cut off')


@unittest.skipUnless(shutil.which('bwrap'), 'the sandbox (bwrap) is not installed')
class OrderJobs(unittest.TestCase):
    GOOD = 'def order_jobs(deps):\n    names = set(deps)\n    for v in deps.values():\n        names.update(v)\n    done, order = set(), []\n    while len(order) < len(names):\n' \
           '        ready = sorted(n for n in names if n not in done and set(deps.get(n, ())) <= done)\n        if not ready:\n            raise ValueError("cycle")\n' \
           '        order.append(ready[0]); done.add(ready[0])\n    return order\n'

    def block(self, code, fence='```python'):
        return f'{fence}\n{code}{fence[:3]}'

    def test_tonights_22_real_answers_get_their_hand_label(self):
        self.assertEqual(len(MORE['order_jobs']), 22)
        self.assertEqual(sum(a['label'] == 'wrong' for a in MORE['order_jobs']), 4)
        for a in MORE['order_jobs']:
            out = matrix.check_order_jobs(a['content'])
            self.assertEqual(out['label'], a['label'], a['source'])
            self.assertEqual(out['fault'], 'model' if a['label'] == 'wrong' else None, a['source'])
            self.assertEqual(out['status'], a['stored_status'], a['source'])        # (the status of then is unchanged)

    def test_the_answers_failure_and_the_harnesss_are_told_apart(self):
        cases = {
            'no code at all': ('', 'wrong', 'model'),
            'code that does not parse': (self.block('def order_jobs(deps)\n    return []\n'), 'wrong', 'model'),
            'no function of that name': (self.block('def schedule(deps):\n    return []\n'), 'wrong', 'model'),
            'a function that never returns': (self.block('def order_jobs(deps):\n    while True:\n        pass\n'), 'wrong', 'model'),
            'a class, which the rules refuse to run': (self.block('class Graph:\n    pass\n' + self.GOOD), 'not scored', 'harness'),
            'type(), which the rules refuse to run': (self.block(self.GOOD.replace('names = set(deps)', 'names = set(deps); kind = type(deps)')), 'not scored', 'harness'),
            'a builtin the runner withholds': (self.block(self.GOOD.replace('names = set(deps)', 'names = set(deps); memo = bytearray()')), 'not scored', 'harness'),
        }
        for why, (text, label, fault) in cases.items():
            out = matrix.check_order_jobs(text)
            self.assertEqual((out['label'], out['fault']), (label, fault), f"{why}: {out['why']}")

    def test_ordinary_builtins_are_available(self):
        code = self.GOOD.replace('names = set(deps)', 'names = set(deps); best = float("inf"); n = round(1.5)')
        self.assertEqual(matrix.check_order_jobs(self.block(code))['label'], 'right')

    def test_a_failing_top_level_statement_does_not_hide_a_right_function(self):
        out = matrix.check_order_jobs(self.block(self.GOOD + '\nassert order_jobs({"a": ["b"]}) == ["wrong"]\n'))
        self.assertEqual(out['label'], 'right')

    def test_every_fence_style_is_read(self):
        for fence in ('```python', '```python3', '``` python', '```py', '~~~python', '```'):
            self.assertEqual(matrix.check_order_jobs(self.block(self.GOOD, fence))['label'], 'right', fence)
        with_usage = self.block(self.GOOD) + '\n\nExample:\n\n```python\nprint(order_jobs({"b": ["a"]}))\n```'
        self.assertEqual(matrix.check_order_jobs(with_usage)['label'], 'right')      # (the block that defines it, not the one that calls it)
        self.assertTrue(checks.fenced_code('```python\n' + self.GOOD).startswith('def order_jobs'))      # (an unclosed last fence)

    def test_a_wall_clock_timeout_is_the_harnesss(self):
        from unittest import mock
        import subprocess
        with mock.patch.object(matrix, 'sandboxed', side_effect=subprocess.TimeoutExpired('bwrap', 8, output='{"ready": true}\n{"top": null, "function": true}\n{"start": "{}"}\n')):
            out = matrix.check_order_jobs(self.block(self.GOOD))
        self.assertEqual((out['label'], out['fault']), ('not scored', 'harness'))

    def test_the_label_of_a_check_stored_before_this_change(self):
        L = checks.order_jobs_label
        self.assertEqual(L({'status': 'pass'})['label'], 'right')
        self.assertEqual(L({'status': 'fail', 'checks': [{'input': '{}', 'pass': False}]})['label'], 'wrong')
        self.assertEqual(L({'status': 'rejected', 'reason': 'no code or code too long'})['label'], 'wrong')
        self.assertEqual(L({'status': 'rejected', 'reason': 'class definition'})['label'], 'not scored')
        self.assertEqual(L({'status': 'execution_error', 'stderr': 'x'})['label'], 'not scored')
        self.assertEqual(L({'status': 'unchecked'})['label'], 'not scored')
        self.assertEqual(L({'status': 'fail'}, 'length')['label'], 'cut off')
        self.assertEqual(L(None)['label'], 'not scored')


class EngineFacts(unittest.TestCase):
    def test_the_engine_that_serves_an_entry_is_recorded(self):
        import os
        from unittest import mock
        with mock.patch.object(matrix.delivery.screen, 'metrics', lambda base, timeout: dict(engine=dict(kv='fp4', max_context=32768, experts=[1, 2]))):
            facts = matrix.engine_facts(os.getpid(), 'http://127.0.0.1:1')
        self.assertEqual(facts['pid'], os.getpid())
        self.assertEqual(len(facts['exe_sha256']), 64)
        self.assertTrue(facts['exe'] and facts['started'] > 0 and isinstance(facts['args'], list))
        self.assertEqual(facts['info'], dict(kv='fp4', max_context=32768))          # (scalars only)

    def test_a_fact_that_cannot_be_read_does_not_stop_the_run(self):
        facts = matrix.engine_facts(2 ** 30, 'http://127.0.0.1:1', timeout=1)        # (no such process, no such server)
        self.assertIn('process_error', facts)
        self.assertIn('info_error', facts)

    def test_the_same_bits_settings_are_kept(self):
        item = dict(engine_stats=dict(final=dict(request_seq=5.0)), metrics_after=dict(tiers=dict(now=dict(request_seq=5.0, expert_fast=2, fp4_kernel=4, expert_overlap=1))))
        s = matrix.settings_seen(item)
        self.assertEqual((s['expert_fast'], s['fp4_kernel'], s['expert_overlap']), (2, 4, 1))


class ActionItems(unittest.TestCase):
    def test_a_fourth_item_is_wrong_and_nothing_else_is(self):
        self.assertEqual(len(MORE['action_items']), 25)
        for a in MORE['action_items']:
            out = checks.check_action_items(a['content'])
            self.assertEqual(out['label'] == 'wrong', a['structure'] == 'wrong', f"{a['source']}: {out['why']}")
        self.assertEqual(sum(a['structure'] == 'wrong' for a in MORE['action_items']), 4)

    def test_calendar_dates_go_to_the_reader(self):
        good = '[{"owner": "Tomas", "due": "Friday"}, {"owner": "Lena", "due": "9 March"}, {"owner": "Priya", "due": "next Tuesday"}]'
        self.assertEqual(checks.check_action_items('Summary. Two.\n' + good)['label'], 'right')
        self.assertEqual(checks.check_action_items(good.replace('Friday', '2026-03-06'))['label'], 'read')
        self.assertEqual(checks.check_action_items(good.replace('Friday', 'Friday, 10 March'))['label'], 'read')      # (kbatch1 hl0 seed 20261101: a date the notes do not give)
        self.assertEqual(checks.check_action_items(good.replace('next Tuesday', 'before the 14 March delivery'))['label'], 'right')      # (the notes' own dates)
        self.assertEqual(checks.invented_dates('due 9 March; delivery March 14; then 2026-03-06, 13 March 2026 and March 9, 2026'), ['13 March 2026', '2026-03-06', 'March 9, 2026'])
        self.assertEqual(checks.check_action_items('no list here')['label'], 'read')
        self.assertEqual(checks.check_action_items(good, 'length')['label'], 'cut off')


class TextDamage(unittest.TestCase):
    def test_the_fixed_rule_of_2026_11_10(self):
        kinds = lambda text, where='content': [k for k, _ in checks.slips(text, where)]
        self.assertEqual(kinds('## Solving the the Muffin Equation'), ['doubled'])
        self.assertEqual(kinds('a flask ofof something'), ['merged'])
        self.assertEqual(kinds('exceeds the limit of10.'), ['merged'])
        self.assertEqual(kinds('I had had enough; that that is fine'), [])
        self.assertEqual(kinds('Done.</think>```python'), ['think_tag'])
        self.assertEqual(kinds('Wait - I need to check'), ['deliberation'])
        self.assertEqual(kinds('Wait - I need to check', 'reasoning'), [])

    def test_a_loop_is_seen(self):
        self.assertTrue(checks.repetition('the bus leaves at seven from the old hall every week ' * 12)['looping'])
        self.assertFalse(checks.repetition(RIGHT_MUFFINS)['looping'])


class Rule(unittest.TestCase):
    def test_exact_values(self):
        self.assertAlmostEqual(rule.fisher_one_sided(2, 2, 0, 2), 1 / 6)
        self.assertAlmostEqual(rule.fisher_one_sided(4, 4, 0, 4), 1 / 70)
        self.assertAlmostEqual(rule.fisher_one_sided(1, 2, 0, 2), 0.5)
        self.assertAlmostEqual(rule.upper95(2, 0), 1 - 0.05 ** 0.5, places=6)
        self.assertAlmostEqual(rule.stratified_p([(1, 4, 1, 4)]), rule.fisher_one_sided(1, 4, 1, 4))
        self.assertAlmostEqual(rule.stratified_p([(0, 2, 0, 2), (0, 2, 0, 2)]), 1.0)
        self.assertLess(rule.stratified_p([(4, 4, 0, 4), (4, 4, 0, 4)]), 0.001)

    def test_two_answers_an_arm_can_show_nothing(self):
        self.assertFalse(rule.can_show_harm(2, 2))
        self.assertTrue(rule.can_show_harm(4, 4))
        for tw in (0, 1, 2):
            self.assertEqual(rule.decide(tw, 2, 0, 2)['outcome'], "can't tell")
        self.assertIn('no outcome can show harm', rule.what_it_can_show(2, 2, 0.10))

    def test_the_three_outcomes(self):
        self.assertEqual(rule.decide(4, 4, 0, 4)['outcome'], 'harm shown')
        self.assertEqual(rule.decide(5, 8, 0, 8)['outcome'], 'harm shown')
        self.assertEqual(rule.decide(1, 8, 1, 8)['outcome'], 'no harm of the stated size')      # (1 wrong of 8: below 47%)
        self.assertEqual(rule.decide(2, 8, 1, 8)['outcome'], "can't tell")                       # (2 wrong of 8: up to 60%)
        self.assertEqual(rule.decide(1, 4, 1, 4)['outcome'], "can't tell")                       # (the paired run of 2026-11-14)
        self.assertEqual(rule.decide(1, 2, 0, 2)['outcome'], "can't tell")                       # (the template's rule printed FAILS)
        self.assertEqual(rule.decide(0, 0, 0, 2)['outcome'], "can't tell")

    def test_the_sizes_a_run_needs(self):
        self.assertIn(rule.size_for(0.50, 0.10), range(17, 22))
        self.assertIn(rule.size_for(0.50, 0.18), range(29, 36))
        self.assertLess(rule.power(8, 8, 0.50, 0.18), 0.25)       # (eight answers an arm: a 50% failure rate is shown less than a quarter of the time)
        self.assertLess(rule.power(8, 8, 0.18, 0.18), 0.05)       # (and with no difference harm is shown under 5% of the time)


def task_row(task, content='', finish='stop', settings=None, check=None, request=None):
    return dict(id=task, complete=True, finish=finish, emitted=500, settings=settings or {}, content=content, reasoning='', check=check,
                request=request or dict(max_tokens=6144, temperature=1, top_p=0.95, seed=1, reasoning_effort='high'))


class Evaluator(unittest.TestCase):
    TASKS = [dict(id='order-jobs', prompt='Write order_jobs.', max_tokens=6144), dict(id='muffins', prompt='How many combinations?', max_tokens=6144),
             dict(id='choir-note', prompt='Write a note.', max_tokens=6144)]

    ENGINE = dict(pid=777, started=100, exe='/artifacts/serving/maya-mm3-aaaa', exe_sha256='a' * 64, args=['--serve', '--kv-cache', 'fp4', '--index-keys', 'fp4'],
                  info=dict(kv='fp4', max_context=32768, expert_slots=1155))

    def make(self, seeds=(1, 2), wrong=(), expect=None, pids=None, control_settings=None, muffins=None, engines=None, **protocol):
        """A run folder: arms old (control) and new, the given seeds; wrong = {(arm, seed)} whose order-jobs check failed."""
        d = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        expect = expect or dict(old=dict(dense_fp8=1), new=dict(dense_fp8=2))
        seq = [dict(arm=a, seed=s) for i, s in enumerate(seeds) for a in (('old', 'new') if i % 2 == 0 else ('new', 'old'))]
        p = dict(experiment_id='test', arms=dict(old={'dense_fp8': 1}, new={}), control='old', seeds=list(seeds), sequence=seq, expect=expect, tasks=self.TASKS, **protocol)
        (d / 'protocol.json').write_text(json.dumps(p))
        for e in seq:
            a, s = e['arm'], e['seed']
            settings = dict(control_settings or expect['old']) if a == 'old' else dict(expect['new'])
            rows = [task_row('order-jobs', check=dict(status='fail', checks=[dict(input='{}', **{'pass': False})]) if (a, s) in wrong else dict(status='pass'), settings=settings),
                    task_row('muffins', (muffins or {}).get((a, s), RIGHT_MUFFINS), settings=settings), task_row('choir-note', f'Dear choir, the hall moves. ({a} {s})', settings=settings)]
            engine = dict(self.ENGINE, **(dict(pid=pids[a]) if pids and a in pids else {}), **((engines or {}).get(a) or {}))
            (d / f'tasks-{a}-s{s}.json').write_text(json.dumps(dict(arm=a, seed=s, engine_pid=engine['pid'], engine=engine, tasks=rows)))
        return d

    def test_a_clean_run_passes_the_gates_and_says_what_two_seeds_can_show(self):
        lines, r = ev.verdict(self.make(wrong={('new', 1)}))
        self.assertTrue(r['can_judge'])
        self.assertEqual(r['outcomes']['new|order-jobs']['outcome'], "can't tell")        # (1 of 2 against 0 of 2: the template's rule printed FAILS)
        self.assertEqual(r['outcomes']['new|choir-note']['outcome'], 'open')              # (a judged task waits for its reader)
        text = '\n'.join(lines)
        self.assertIn('no outcome can show harm', text)
        self.assertNotIn('FAILS', text)

    def test_harm_is_shown_when_the_counts_support_it(self):
        _, r = ev.verdict(self.make(seeds=range(1, 9), wrong={('new', s) for s in (1, 2, 3, 4, 5)}))
        self.assertEqual(r['outcomes']['new|order-jobs']['outcome'], 'harm shown')
        self.assertEqual(r['outcomes']['new|muffins']['outcome'], 'no harm of the stated size')
        self.assertEqual(r['outcomes']['new|automatic tasks together']['outcome'], 'harm shown')

    def test_two_engine_starts_give_no_verdict(self):
        lines, r = ev.verdict(self.make(pids=dict(old=111, new=222)))
        self.assertFalse(r['can_judge'])
        self.assertIn('NO VERDICT', '\n'.join(lines))
        self.assertTrue(any('2 engine starts served the answers (old pid 111; new pid 222)' in c for c in r['confounds']), r['confounds'])

    def test_two_binaries_or_start_options_are_named(self):
        # (2026-11-14: B ran 48e6d92 and A 77071b8; kbatch1's start had --head-lowp fp8,nvfp4 and kbatch2's did not)
        other = dict(pid=778, started=200, exe='/artifacts/serving/maya-mm3-bbbb', exe_sha256='b' * 64, args=['--serve', '--kv-cache', 'fp4', '--index-keys', 'fp4', '--head-lowp', 'fp8,nvfp4'])
        lines, r = ev.verdict(self.make(engines=dict(old=other)))
        self.assertFalse(r['can_judge'])
        text = '\n'.join(r['confounds'])
        self.assertIn('the arms ran different binaries: old maya-mm3-bbbb (sha256 bbbbbbbbbbbb); new maya-mm3-aaaa (sha256 aaaaaaaaaaaa)', text)
        self.assertIn('the start options differ between the arms: old --head-lowp fp8,nvfp4; new (nothing more)', text)

    def test_the_kv_format_is_checked_and_a_start_option_as_the_tested_setting_gives_no_verdict(self):
        int8 = dict(pid=778, started=200, args=['--serve'], info=dict(kv='int8', max_context=32768, expert_slots=1115))
        d = self.make(engines=dict(old=int8), expect_engine=dict(old=dict(kv='int8'), new=dict(kv='fp4', args=['--kv-cache', 'fp4'])))
        lines, r = ev.verdict(d)
        self.assertFalse(r['can_judge'])
        text = '\n'.join(r['confounds'])
        self.assertIn('kv (old int8; new fp4)', text)
        self.assertIn('the tested setting is a start option', text)
        d = self.make(expect_engine=dict(kv='int8'))                    # (one start, but not the engine the protocol expects)
        r = ev.verdict(d)[1]
        self.assertFalse(r['can_judge'])
        self.assertIn("the engine reports kv 'fp4' (expected 'int8')", '\n'.join(r['confounds']))
        self.assertTrue(ev.verdict(self.make(expect_engine=dict(kv='fp4', args=['--index-keys', 'fp4'])))[1]['can_judge'])

    def test_a_setting_outside_the_protocol_that_differs_is_named(self):
        # (kbatch2 against kbatch1's hl0: third_sm89, helper_q_rows and three more differed and no table showed it)
        d = self.make()
        for f in d.glob('tasks-new-*.json'):
            j = json.loads(f.read_text())
            for t in j['tasks']:
                t['settings'].update(third_sm89=1, cpu_kernel=2)
            f.write_text(json.dumps(j))
        for f in d.glob('tasks-old-*.json'):
            j = json.loads(f.read_text())
            for t in j['tasks']:
                t['settings'].update(third_sm89=0)
            f.write_text(json.dumps(j))
        r = ev.verdict(d)[1]
        self.assertFalse(r['can_judge'])
        text = '\n'.join(r['confounds'])
        self.assertIn('settings the protocol does not name differ between new and old: third_sm89 (old 0.0; new 1.0)', text)
        self.assertIn("old's answers do not record cpu_kernel: its side of those settings cannot be verified", text)

    def test_the_gates_are_one_function_over_the_protocol_and_the_records(self):
        g = ev.gates(ev.read_run(self.make()))
        self.assertTrue(g['ok'])
        self.assertEqual(g['confounds'], [])
        self.assertEqual([name for name, _, _ in g['lines']][:4], ['every answer from one engine start', 'one binary', 'the same start options', 'the K/V cache format'])
        self.assertIn('fp4 on every answer', dict((n, t) for n, _, t in g['lines'])['the K/V cache format'])

    def test_an_answer_that_did_not_report_its_arms_settings_gives_no_verdict(self):
        lines, r = ev.verdict(self.make(control_settings=dict(dense_fp8=2)))
        self.assertFalse(r['gates']["every answer reported its arm's settings"])
        self.assertIn('dense_fp8 2 (expected 1)', '\n'.join(lines))

    def test_arms_whose_recorded_settings_do_not_differ_give_no_verdict(self):
        # (a start option, like the K/V format of 2026-11-14's B against A: not in any answer's record)
        d = self.make(expect=dict(old=dict(routing_floor=0.5), new=dict(routing_floor=0.5)))
        self.assertFalse(ev.verdict(d)[1]['can_judge'])
        lines, ok = ev.preflight_lines(json.loads((d / 'protocol.json').read_text()))
        self.assertFalse(ok)
        self.assertIn('the tested setting is not in each answer\'s record', '\n'.join(lines))

    def test_a_control_from_another_run_folder_gives_no_verdict(self):
        control = self.make()
        d = self.make()
        p = json.loads((d / 'protocol.json').read_text())
        p.update(control=None, control_from=dict(run=str(control), arm='old'), arms=dict(new={}), sequence=[e for e in p['sequence'] if e['arm'] == 'new'])
        (d / 'protocol.json').write_text(json.dumps(p))
        lines, r = ev.verdict(d)
        self.assertFalse(r['can_judge'])
        self.assertIn('another engine start', '\n'.join(lines))
        self.assertFalse(ev.preflight_lines(p)[1])

    def test_different_sampling_in_the_arms_gives_no_verdict(self):
        d = self.make()
        f = d / 'tasks-new-s1.json'
        j = json.loads(f.read_text())
        j['tasks'][0]['request']['temperature'] = 0.7
        f.write_text(json.dumps(j))
        self.assertFalse(ev.verdict(d)[1]['gates']['the same sampling and budget in every arm'])

    def test_an_answer_the_check_cannot_decide_waits_for_a_reader_and_is_never_wrong(self):
        unread = 'Seven boxes and three; four and five; one and seven. So three ways.'
        d = self.make(muffins={('new', 1): unread})
        _, r = ev.verdict(d)
        self.assertEqual(r['outcomes']['new|muffins']['outcome'], 'open')
        self.assertEqual(r['counts']['muffins|new']['wrong'], 0)
        text, template = ev.blind(d)
        key = next(k for k in template['labels'] if k.startswith('muffins/1/'))
        labels = d / 'labels.json'
        labels.write_text(json.dumps(dict(reader='test', labels={key: dict(label='right', note='three, stated in words')})))
        _, r = ev.verdict(d, labels)
        self.assertNotEqual(r['outcomes']['new|muffins']['outcome'], 'open')
        self.assertEqual(r['counts']['muffins|new']['right'], 2)

    def test_the_blind_packet_names_no_arm(self):
        d = self.make(seeds=range(1, 7))
        text, template = ev.blind(d)
        self.assertEqual(sum(k.startswith('choir-note/') for k in template['labels']), 12)
        answers = text.split('### ')[1:]
        self.assertTrue(all('(old ' not in a.split('\n', 1)[0] and '(new ' not in a.split('\n', 1)[0] for a in answers))
        first = [next(a for a in answers if a.startswith(f'choir-note/{s}/P')) for s in range(1, 7)]
        self.assertEqual(len({'(old' in a for a in first}), 2)            # (P is not always the same arm)

    def test_a_reader_cannot_overturn_an_executed_check(self):
        d = self.make(wrong={('new', 1)})
        names = ev.blind_names(json.loads((d / 'protocol.json').read_text()), 'order-jobs', 1, ['old', 'new'])
        labels = d / 'labels.json'
        labels.write_text(json.dumps(dict(labels={f"order-jobs/1/{names['new']}": dict(label='right')})))
        self.assertEqual(ev.verdict(d, labels)[1]['counts']['order-jobs|new']['wrong'], 1)

    def test_a_cut_off_answer_is_not_a_failure_but_a_loop_is_flagged(self):
        d = self.make()
        f = d / 'tasks-new-s1.json'
        j = json.loads(f.read_text())
        j['tasks'][2].update(finish='length', content='the bus leaves at seven from the old hall every week ' * 40)
        f.write_text(json.dumps(j))
        lines, r = ev.verdict(d)
        self.assertEqual(r['counts']['choir-note|new']['cut_off'], 1)
        self.assertEqual(r['counts']['choir-note|new']['wrong'], 0)
        self.assertIn('cut off while repeating', '\n'.join(lines))

    def test_a_sequence_flips_its_order_from_seed_to_seed(self):
        seq = ev.interleaved_sequence('old', ['new'], [1, 2, 3, 4])
        self.assertEqual([(e['arm'], e['seed']) for e in seq], [('old', 1), ('new', 1), ('new', 2), ('old', 2), ('old', 3), ('new', 3), ('new', 4), ('old', 4)])
        p = dict(arms=dict(old={}, new={}), control='old', sequence=seq, expect=dict(old=dict(k=1), new=dict(k=2)), tasks=self.TASKS)
        self.assertTrue(ev.preflight_lines(p)[1])
        p['sequence'] = [dict(arm=a, seed=s) for s in (1, 2, 3, 4) for a in ('old', 'new')]
        self.assertIn('alternate the order', '\n'.join(ev.preflight_lines(p)[0]))

    def test_the_preflight_states_the_size(self):
        d = self.make(seeds=range(1, 9))
        lines, ok = ev.preflight_lines(json.loads((d / 'protocol.json').read_text()))
        self.assertTrue(ok)
        self.assertIn('8 against 8 answers', '\n'.join(lines))


if __name__ == '__main__':
    unittest.main()
