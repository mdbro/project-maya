#!/usr/bin/env python3
"""Validate a reviewer transcription of the scheduling answer, preserving source prose.

Input JSON: {"scenarios": [{"finish": 11, "schedule": {"A": [0, 3], ...},
"lower_bound_explanation": "verbatim excerpt", "lower_bound_valid": true}, ...]}.
The reviewer judges the explanation; the checker validates the schedule independently.
Missing or ambiguous transcription remains unassessed, not an incorrect answer.
"""
import argparse
import json
import math
import pathlib

PROMPT = """Two workers are available from time 0. Each task needs one worker throughout its execution. Tasks cannot be paused or split.

Task A: duration 3, no prerequisites
Task B: duration 2, no prerequisites
Task C: duration 4, prerequisite A
Task D: duration 3, prerequisite B
Task E: duration 2, prerequisites C and D

Tasks C and D also require the same machine, so they cannot overlap. All other tasks may overlap when workers are available.

1. What is the earliest time all tasks can finish? Give a feasible schedule and briefly explain why an earlier finish is impossible.
2. Now change only B's duration from 2 to 4. Find the new earliest finish, give a feasible schedule, and explain what changes.

Use concise prose and task start/end times. No code or JSON is needed."""


def check_scenario(data, b_duration):
    expected = 11 if b_duration == 2 else 12
    finish = data.get('finish')
    finish_ok = None if finish is None else type(finish) in (int, float) and math.isfinite(finish) and finish == expected
    schedule = data.get('schedule')
    errors = []
    if schedule is not None:
        durations = {'A': 3, 'B': b_duration, 'C': 4, 'D': 3, 'E': 2}
        if not isinstance(schedule, dict) or set(schedule) != set(durations):
            errors.append('schedule must contain exactly A through E')
        else:
            for name, duration in durations.items():
                interval = schedule[name]
                if (not isinstance(interval, list) or len(interval) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in interval)):
                    errors.append(f'{name}: invalid interval'); continue
                start, end = interval
                if start < 0 or end - start != duration: errors.append(f'{name}: start or duration invalid')
            if not errors:
                for task, prereqs in {'C':['A'], 'D':['B'], 'E':['C','D']}.items():
                    for prereq in prereqs:
                        if schedule[task][0] < schedule[prereq][1]: errors.append(f'{task} precedes {prereq}')
                points = sorted({x for interval in schedule.values() for x in interval})
                if any(sum(a <= t < b for a, b in schedule.values()) > 2 for t in points): errors.append('worker capacity exceeded')
                c, d = schedule['C'], schedule['D']
                if max(c[0], d[0]) < min(c[1], d[1]): errors.append('machine overlap')
                if finish is not None and finish != max(x[1] for x in schedule.values()): errors.append('schedule does not match stated finish')
    explanation = data.get('lower_bound_explanation')
    judgment = data.get('lower_bound_valid')
    if not isinstance(explanation, str) or not explanation.strip() or type(judgment) is not bool:
        judgment = None
    return {'optimal_finish': expected, 'finish_correct': finish_ok,
            'schedule_feasible': None if schedule is None else not errors, 'schedule_errors': errors,
            'lower_bound_valid': judgment, 'lower_bound_explanation': explanation,
            'lower_bound_review': 'reviewer semantic judgment of the transcribed explanation'}


def check(item, transcription):
    truncated = item.get('finish_reason') == 'length'
    complete = bool(item.get('complete') and not truncated and item.get('content', '').strip())
    scenarios = transcription.get('scenarios', [])
    return {'completion': 'truncated' if truncated else 'complete' if complete else 'incomplete',
            'scenarios': [check_scenario(scenarios[i] if i < len(scenarios) else {}, b)
                          for i, b in enumerate((2, 4))],
            'note': 'Transcription and lower-bound judgments require review against the raw final answer.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('result', type=pathlib.Path, help='native_delivery result JSON')
    ap.add_argument('transcription', type=pathlib.Path, help='reviewer transcription JSON')
    a = ap.parse_args()
    result = json.loads(a.result.read_text())
    item = next(x for x in result['requests'] if x['case'] == 'reasoning')
    print(json.dumps(check(item, json.loads(a.transcription.read_text())), indent=2))


if __name__ == '__main__': main()
