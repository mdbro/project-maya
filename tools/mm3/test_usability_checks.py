#!/usr/bin/env python3
"""Tests for usability_checks (the audited automatic checks and decision rule, 2026-11-14).

    python3 -m unittest test_usability_checks      (from tools/mm3)
"""
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


if __name__ == '__main__':
    unittest.main()
