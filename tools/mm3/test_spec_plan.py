"""The fitter's warm boundary follows emitted tokens, including variable acceptance."""
import json
from pathlib import Path
import tempfile
import unittest

import spec_plan


class WarmBoundary(unittest.TestCase):
    def run_fixture(self, emitted=True):
        root = Path(self.directory.name)
        capture = root / 'request.json.capture'
        capture.mkdir(exist_ok=True)
        (root / 'protocol.json').write_text(json.dumps(dict(study=dict(arms={'fit': {}}, timeline=[dict(step='request', arm='fit')]))))
        (root / 'ledger.json').write_text(json.dumps(dict(steps=[dict(step='request', summary=dict(final_stat=dict(spec_policy='sampled')))])))
        rows = []
        # Round count and prompt position cannot identify the cache-warm boundary.
        for number, (start, count) in enumerate([(0, 1), (1, 4), (5, 4), (9, 2), (127, 3), (130, 4), (134, 2), (136, 1)]):
            rows.append(dict(type='round_begin', state_position=12000 + start))
            if emitted:
                rows.append(dict(type='token_output', output_index=start))
            rows.extend([dict(type='draft_begin', steady_ns=number * 1000000),
                         dict(type='draft_end', steady_ns=number * 1000000 + 100),
                         dict(type='target_forward_begin', steady_ns=number * 1000000 + 200, rows=4),
                         dict(type='target_forward_end', steady_ns=number * 1000000 + 500)])
            if emitted:
                rows.extend(dict(type='token_output', output_index=i) for i in range(start + 1, start + count))
            rows.append(dict(type='round_end', steady_ns=number * 1000000 + 600, evaluated=3, committed_tokens=12000 + start + count))
        (capture / 'events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return root, capture

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_crossing_round_is_excluded_and_later_rounds_keep_real_indices(self):
        root, _ = self.run_fixture()
        rounds = spec_plan.load([root])
        self.assertEqual([r['start_emitted'] for r in rounds], [130, 134, 136])
        self.assertEqual([r['end_emitted'] for r in rounds], [134, 136, 137])
        self.assertTrue(all(r['exact'] for r in rounds))  # resolved rule from actual statistics

    def test_missing_token_counters_are_refused(self):
        root, _ = self.run_fixture(emitted=False)
        with self.assertRaisesRegex(ValueError, 'no emitted-token boundary'):
            spec_plan.load([root])

    def test_bad_within_round_indices_are_refused(self):
        _, capture = self.run_fixture()
        path = capture / 'events.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in rows if r.get('output_index') == 132)['output_index'] = 135
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'non-contiguous'):
            spec_plan.rounds_of(capture)


if __name__ == '__main__':
    unittest.main()
