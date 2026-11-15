#!/usr/bin/env python3
"""The short usability check's answer checks (2026-11-14, from the audit in runs/usability-audit-20261114: the user asked for
usability verdicts that are accurate from now on; first installed in 38cf0ab5, completed here). No engine, no GPU, no imports
beyond the standard library. The decision rule is usability_rule.py; the evaluator is usability_eval.py.

Every check gives one of four labels, and says why (the key "label"; "verdict" holds the same value for the callers of 38cf0ab5):
  right        the answer meets the task
  wrong        the MODEL's answer fails the task (positive evidence only)
  read         the check cannot decide: a reader labels it (a notation it does not know, an inconsistency)
  not scored   the HARNESS could not score it (the checker's rules refuse a construct, no sandbox, the runner failed before the
               code ran, a withheld builtin, a wall-clock timeout under load)
and, from the request's finish reason, "cut off" (the token budget ended the answer: never a failure - the user, 2026-11-08).

  check_muffins(text, finish)        the muffin count. wrong only when the final section states a count other than 3 and never 3.
  order_jobs_label(check, finish)    the label of an executed order-jobs check (native_usability_matrix.check_order_jobs), old or new
  fenced_code(text)                  the answer's code block: the last block that DEFINES order_jobs; any fence style
  check_action_items(text, finish)   structure only: exactly the three owners Tomas, Lena, Priya; calendar dates the notes do not
                                     give send it to the reader
  label_row(row)                     the label of one row of a tasks-ARM-sSEED.json file, whatever checker wrote it
  slips(text, where)                 small text damage by one fixed rule (runs/usability-rerun-20261110/slip_counter.py, unchanged):
                                     reported per 1,000 tokens, never a failure by itself (the user, 2026-11-10)
  repetition(text)                   whether a text loops (a cut-off answer that loops is damage, not a budget matter)
"""
from __future__ import annotations

import collections
import json
import re

RIGHT, WRONG, READ, NOT_SCORED, CUT_OFF = 'right', 'wrong', 'read', 'not scored', 'cut off'

# ---- muffins: 46 muffins in boxes of 4 and 6, at most 10 boxes: 3 combinations, (7,3), (4,5), (1,7); (10,1) needs 11 boxes ----
MUFFIN_VALID = ((1, 7), (4, 5), (7, 3))
_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
_NUM = r'(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)'
_NOUN = r'(?:combinations?|ways?|options?|possibilit(?:y|ies)|choices?|solutions?|pairs?)'
_FINAL = re.compile(r'^[ \t]*(?:#{1,6}[ \t]*|\*\*)[ \t]*(?:the[ \t]+)?(?:final[ \t]+)?(?:answer|conclusion|result|summary)\b', re.I | re.M)
# a stated count: N, then up to 30 characters holding no digit and no sentence end, then combinations / ways / ...
_COUNT = re.compile(r'(?<![\w.\-])' + _NUM + r'\**(?=\s)((?:(?![\d.;:\n]).){0,30}?)\b' + _NOUN + r'\b', re.I)
_OF_M = re.compile(r'(?<![\w.\-])' + _NUM + r'\**\s+(?:of|out of)\s+(?:the\s+)?' + _NUM + r'\s+(?:[A-Za-z-]+\s+){0,2}?' + _NOUN + r'\b', re.I)
_ANSWER = re.compile(r'\banswer\**\s*(?:is|:)[\s*]*' + _NUM + r'\b|\\boxed\{\s*' + _NUM + r'\s*\}'
                     r'|\b(?:count|total|number)(?:\s+of\s+[a-z ]{0,30}?)?\s+is\s+\**' + _NUM + r'\b', re.I)
_BARE = re.compile(r'^[ \t]*\**' + _NUM + r'\**\.?[ \t]*$', re.I | re.M)
_COUNTERFACTUAL = re.compile(r'\b(?:would|without|ignoring|were|if|unless|other|remaining|extra|additional|excluded?|rejected|fails?|fourth)\b', re.I)
_BOX_BEFORE = re.compile(r'(?:box(?:es)?|packs?|\bof|\bfor|\bby|\bon)\s*$', re.I)   # "1 box of 6 combination": 6 is a box size
_REJECTS = re.compile(r"\b11\b|exclud|reject|too many|exceed|violat|fail|\bnot\b|n't|ruled out|[✗✘❌×]|\bover\b|more than|invalid|only other|"
                      r"near-miss|breaks|doesn|cannot|can't|disqualif|drop|remov|discard", re.I)


def _n(s: str) -> int:
    s = s.lower()
    return _WORDS[s] if s in _WORDS else int(s)


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence of the line around text[start:end] (bounded by the line and by '.', '!', '?' or ';' followed by a space)."""
    a = text.rfind('\n', 0, start) + 1
    b = text.find('\n', end)
    b = len(text) if b < 0 else b
    for m in re.finditer(r'[.!?;]\s', text[a:start]):
        a2 = a + m.end()
        if a2 <= start:
            a = a2
    m = re.search(r'[.!?;](?:\s|$)', text[end:b])
    return text[a:end + m.end() if m else b]


def muffin_final_section(text: str) -> str:
    """After the last heading (markdown or a bold line) named Answer, Final answer, Conclusion, Result or Summary; the last
    700 characters when the answer has no such heading."""
    heads = list(_FINAL.finditer(text))
    return text[heads[-1].start():] if heads else text[-700:]


def muffin_claims(section: str) -> list[tuple[int, int, bool]]:
    """Every stated count in the section, in order: (position, number, in a counterfactual or "the other one" sentence?)."""
    out, taken = [], []
    for m in _OF_M.finditer(section):                      # "3 of the 4 solutions fit": the count is 3
        out.append((m.start(), _n(m.group(1)), bool(_COUNTERFACTUAL.search(_sentence(section, m.start(), m.end())))))
        taken.append((m.start(), m.end()))
    for m in _COUNT.finditer(section):
        if any(a <= m.start() < b for a, b in taken) or _BOX_BEFORE.search(section[max(0, m.start() - 9):m.start()]):
            continue
        out.append((m.start(), _n(m.group(1)), bool(_COUNTERFACTUAL.search(_sentence(section, m.start(), m.end())))))
    for m in _ANSWER.finditer(section):
        out.append((m.start(), _n(m.group(1) or m.group(2) or m.group(3)), bool(_COUNTERFACTUAL.search(_sentence(section, m.start(), m.end())))))
    for m in _BARE.finditer(section):
        out.append((m.start(), _n(m.group(1)), False))
    return sorted(out)


def _plain(text: str) -> str:
    t = text.replace('\\cdot', '×').replace('\\times', '×').replace('‑', '-').replace('‐', '-').replace('−', '-')
    return re.sub(r'[$*`\\]', '', t)


def muffin_valid_pairs(text: str) -> list[tuple[int, int]]:
    """Which of the three valid pairs (boxes of 4, boxes of 6) the answer names, anywhere, in any of its notations:
    "7 boxes of 4 + 3 boxes of 6" in either order, with any short preposition (a substituted "for", "by" or "on" is not a
    failure: the user, 2026-11-10) and with number words; "4-packs", "four-packs"; "(7, 3)"; "7(4) + 3(6)", "4(7) + 6(3)",
    "7 x 4 + 3 x 6"; "a = 7, b = 3" (also "a=1: b = 7"); a table row holding both numbers with their box total or both
    products, whatever its header says."""
    t = _plain(text)
    def num(n): return r'(?:' + str(n) + '|' + next(k for k, v in _WORDS.items() if v == n) + r')'
    prep = r'(?:of|for|by|on|with)'
    four = r'(?:(?:box(?:es)?|packs?|cartons?)\s+' + prep + r'\s+(?:4|four)|(?:4|four)-?\s?(?:box(?:es)?|packs?)|small(?:\s+box(?:es)?)?)\b'
    six = r'(?:(?:box(?:es)?|packs?|cartons?)\s+' + prep + r'\s+(?:6|six)|(?:6|six)-?\s?(?:box(?:es)?|packs?)|large(?:\s+box(?:es)?)?)\b'
    times = r'\s*(?:\(\s*{n}\s*\)|[×x·*]\s*{n})'
    found = set()
    for a, b in MUFFIN_VALID:
        pats = (rf'(?<![\d\w]){num(a)}\s*{four}[^\n]{{0,40}}?(?<![\d\w]){num(b)}\s*{six}',
                rf'(?<![\d\w]){num(b)}\s*{six}[^\n]{{0,40}}?(?<![\d\w]){num(a)}\s*{four}',
                rf'\(\s*{a}\s*,\s*{b}\s*\)',
                rf'(?<![\d.]){a}' + times.format(n=4) + rf'\s*\+\s*{b}' + times.format(n=6) + r'(?![\d.])',
                r'(?<![\d.])4' + times.format(n=a) + r'\s*\+\s*6' + times.format(n=b) + r'(?![\d.])',
                rf'\b[a-z]\s*=\s*{a}\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*{b}(?!\d)',
                rf'\b[a-z]\s*=\s*{b}\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*{a}(?!\d)(?=[^\n]*\b{a + b}\b)')
        if any(re.search(q, t, re.I) for q in pats):
            found.add((a, b))
        for line in t.splitlines():                        # a table row: both numbers, and the box total or both products
            if not line.lstrip().startswith('|'):
                continue
            ints = [int(x) for x in re.findall(r'(?<![\d.])\d+(?![\d.])', line)]
            if a in ints and b in ints and (a + b in ints or (4 * a in ints and 6 * b in ints)) and not (a == 1 and 10 in ints and 11 in ints):
                found.add((a, b))
    lines = t.splitlines()
    for i, line in enumerate(lines[:-1]):                  # a table whose header names both box sizes: one pair a row
        if not line.lstrip().startswith('|') or not re.match(r'^\s*\|?\s*:?-{3,}', lines[i + 1]):
            continue
        head = [c.strip().lower() for c in line.strip().strip('|').split('|')]
        f = [k for k, c in enumerate(head) if re.search(four, c)]
        x = [k for k, c in enumerate(head) if re.search(six, c)]
        if len(f) != 1 or len(x) != 1:
            continue
        for row in lines[i + 2:]:
            if not row.lstrip().startswith('|'):
                break
            cells = [c.strip() for c in row.strip().strip('|').split('|')]
            if len(cells) > max(f[0], x[0]) and cells[f[0]].isdigit() and cells[x[0]].isdigit() and (int(cells[f[0]]), int(cells[x[0]])) in MUFFIN_VALID:
                found.add((int(cells[f[0]]), int(cells[x[0]])))
    return sorted(found)


def muffin_accepts_wrong_set(text: str) -> str | None:
    """A reason to send the answer to a reader, or None: (10, 1) named on a line with no sign of rejection, or a valid pair
    named in a sentence of the final section that rejects it."""
    t = _plain(text)
    ten = (r'(?<![\d\w])(?:10|ten)\s*(?:box(?:es)?|packs?)\s+(?:of|for|by|on)\s+(?:4|four)[^\n]{0,40}?(?<![\d\w])(?:1|one)\s*(?:box(?:es)?|packs?)\s+(?:of|for|by|on)\s+(?:6|six)',
           r'\(\s*10\s*,\s*1\s*\)', r'\b[a-z]\s*=\s*10\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*1(?!\d)', r'\b[a-z]\s*=\s*1\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*10(?!\d)')
    for line in t.splitlines():
        for q in ten:
            if re.search(q, line, re.I) and not _REJECTS.search(line):
                return f'(10, 1) is named with no sign of rejection: "{line.strip()[:90]}"'
    sec = _plain(muffin_final_section(text))
    for a, b in MUFFIN_VALID:
        for m in re.finditer(rf'\(\s*{a}\s*,\s*{b}\s*\)', sec):
            sent = _sentence(sec, m.start(), m.end())
            if re.search(r"ruled out|exclud|reject|invalid|does not work|doesn't work|fails", sent, re.I) and not re.search(r'\b11\b|10\s*,\s*1|ten', sent):
                return f'the valid pair ({a}, {b}) is named in a sentence that rejects it: "{sent.strip()[:90]}"'
    return None


def _result(label: str, why: str, **more) -> dict:
    return dict(label=label, verdict=label, why=why, **more)


def check_muffins(text: str, finish: str | None = 'stop') -> dict:
    """label right | wrong | read | cut off, with final_count (the first count the final section states outside counterfactual
    sentences), pairs (the valid pairs named) and why.
      right   the final section states 3 first, the three valid pairs are named, and nothing suggests another set
      wrong   the final section states a count other than 3 and never 3
      read    anything else (a count the patterns do not find, a notation they do not read, 3 beside another count,
              (10, 1) named without rejection): a reader decides; the evaluator never counts it as wrong."""
    text = text or ''
    if finish not in (None, 'stop'):
        return _result(CUT_OFF, f'finish {finish}: cut off, never a failure', final_count=None, pairs=[])
    claims = [n for _, n, counterfactual in muffin_claims(muffin_final_section(text)) if not counterfactual]
    first = claims[0] if claims else None
    pairs = muffin_valid_pairs(text)
    odd = muffin_accepts_wrong_set(text)
    if first == 3 and pairs == sorted(MUFFIN_VALID) and not odd:
        label, why = RIGHT, 'the final section states 3 first and the three valid pairs are named'
    elif first == 3 and pairs == sorted(MUFFIN_VALID):
        label, why = READ, 'states 3 and names the three valid pairs, but ' + odd
    elif first == 3:
        label, why = READ, f'states 3; valid pairs recognized: {pairs} (a notation the check does not read, or a pair is missing)'
    elif first is None:
        label, why = READ, 'no stated count found in the final section'
    elif 3 in claims:
        label, why = READ, f'the final section states {first} first and 3 later'
    else:
        label, why = WRONG, f'the final section states {first} and never 3'
    return _result(label, why, final_count=first, pairs=pairs)


check_muffins_v2 = check_muffins      # (the name of 38cf0ab5)


# ---- order-jobs ---------------------------------------------------------------------------------------------------------------
def fenced_code(text: str) -> str:
    """The answer's code: the last fenced block that defines order_jobs (not one that only calls it), else the last block,
    else the whole text. Reads ```python, ```python3, ``` python, ~~~ fences and an unclosed last fence."""
    blocks = re.findall(r'(?:```|~~~)[ \t]*[\w+.-]*[ \t]*\n(.*?)(?:```|~~~)', text, re.S)
    if not blocks:
        m = re.search(r'(?:```|~~~)[ \t]*[\w+.-]*[ \t]*\n(.*)\Z', text, re.S)
        blocks = [m.group(1)] if m else []
    for block in reversed(blocks):
        if re.search(r'^\s*def\s+order_jobs\s*\(', block, re.M):
            return block
    return blocks[-1] if blocks else text.strip()


fenced_code_v2 = fenced_code          # (the name of 38cf0ab5)


def order_jobs_label(check: dict | None, finish: str | None = 'stop') -> dict:
    """The label of a stored order-jobs check. A check written since 2026-11-14 carries its own label and fault; an older one
    has only its status: pass = right, fail = wrong (an executed check failed), rejected = wrong when the answer had no code or
    its code did not parse and not scored when the checker's own rules refused it, unchecked / execution_error = not scored."""
    if finish not in (None, 'stop'):
        return _result(CUT_OFF, f'finish {finish}: cut off, never a failure')
    if not check:
        return _result(NOT_SCORED, 'no check stored')
    if check.get('label'):
        return _result(check['label'], check.get('why') or check.get('status'))
    status, reason = check.get('status'), str(check.get('reason') or '')
    if status == 'pass':
        return _result(RIGHT, 'the eight executed checks pass')
    if status == 'fail':
        return _result(WRONG, 'executed checks fail: ' + '; '.join(c.get('input', '?') for c in check.get('checks', []) if not c.get('pass')))
    if status == 'rejected' and (reason.startswith('no code') or reason.startswith('SyntaxError')):
        return _result(WRONG, 'the answer has no runnable function: ' + reason)
    if status == 'rejected':
        return _result(NOT_SCORED, 'the checker\'s rules refused to run it: ' + reason)
    return _result(NOT_SCORED, f'{status}: ' + (reason or str(check.get('stderr') or '')[-200:]))


# ---- action-items: three items in the notes (Tomas by Friday, Lena before the 9 March audit, Priya by next Tuesday) ----------
ACTION_OWNERS = ['Lena', 'Priya', 'Tomas']


def action_items_list(text: str) -> list[dict] | None:
    """The first JSON array of objects in the answer (fenced or bare), or None."""
    for m in re.finditer(r'\[', text):
        depth = 0
        for i in range(m.start(), len(text)):
            depth += text[i] == '['
            if text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[m.start():i + 1])
                    except ValueError:
                        break
                    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                        return v
                    break
    return None


def invented_dates(text: str) -> list[str]:
    """Calendar dates in the text that the notes do not give: the notes name only 9 March (the audit) and 14 March (the
    delivery); "Friday" and "next Tuesday" have no date. A year, an ISO date, a numeric date or any other day of a month."""
    found = set(re.findall(r'\b20\d\d-\d\d-\d\d\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b', text))
    months = r'(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)'
    for m in re.finditer(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({months})\.?(\s+20\d\d)?\b|\b({months})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(,?\s+20\d\d)?\b', text):
        day, month, year = (m.group(1), m.group(2), m.group(3)) if m.group(1) else (m.group(5), m.group(4), m.group(6))
        if year or not (month.lower().startswith('mar') and int(day) in (9, 14)):
            found.add(m.group(0).strip())
    return sorted(found)


def check_action_items(text: str, finish: str | None = 'stop') -> dict:
    """Structure only. wrong: the JSON list does not have exactly the three owners (an invented fourth item, a missing one).
    read: no JSON list found, or calendar dates the notes do not give (a year, an ISO date) - whether a resolved date is an
    error is the reader's call. right: three items, the three owners, the notes' own dates. The summary, the wording of the
    actions and the dates are the reader's."""
    text = text or ''
    if finish not in (None, 'stop'):
        return _result(CUT_OFF, f'finish {finish}: cut off, never a failure')
    items = action_items_list(text)
    if items is None:
        return _result(READ, 'no JSON list of objects found')
    owners = sorted(str(next((v for k, v in d.items() if 'owner' in k.lower() or k.lower() in ('who', 'assignee', 'responsible')), '?')) for d in items)
    dates = invented_dates(json.dumps(items, ensure_ascii=False))
    if owners != ACTION_OWNERS:
        return _result(WRONG, f'{len(items)} items with owners {owners}: the notes give three (Lena, Priya, Tomas)', owners=owners, calendar_dates=dates)
    if dates:
        return _result(READ, f'three items, the three owners; calendar dates the notes do not give: {dates}', owners=owners, calendar_dates=dates)
    return _result(RIGHT, 'three items, the three owners, no invented calendar date (structure only)', owners=owners, calendar_dates=[])


# ---- one row of a tasks file --------------------------------------------------------------------------------------------------
AUTOMATIC = ('order-jobs', 'muffins', 'action-items')     # tasks with an automatic label; the rest are read (docs/minimax-wsl/usability-check.md)


def label_row(row: dict) -> dict:
    """The label of one task row of a tasks-ARM-sSEED.json file. muffins and action-items are re-checked from the answer's
    text with this module (so an old file is scored by today's check); order-jobs keeps its executed check."""
    task, finish, content = row.get('id'), row.get('finish'), row.get('content') or ''
    if task == 'muffins':
        return check_muffins(content, finish)
    if task == 'action-items':
        return check_action_items(content, finish)
    if task == 'order-jobs':
        return order_jobs_label(row.get('check'), finish)
    return _result(CUT_OFF if finish not in (None, 'stop') else READ, 'a judged task: the reader labels it')


# ---- small text damage: the fixed rule of 2026-11-10 (runs/usability-rerun-20261110/slip_counter.py), unchanged ---------------
_SLIP_PATTERNS = {
    'doubled': re.compile(r'(?<![A-Za-z])([A-Za-z]+)[ \t]+\1(?![A-Za-z])', re.I),
    'merged': re.compile(r'(?<![A-Za-z])(?:(of|to|by|for|and|the)\1(?![A-Za-z])|(?:of|to|by|for)(?=\d|[a-z]?\$)|of(?!f|ten)[a-z]{2,}(?![A-Za-z])|\*[A-Za-z]+_(?![A-Za-z]))'),
}
_SLIP_ALLOWED_DOUBLES = {'that', 'had'}
_THINK = re.compile(r'</?think>')
_DELIBERATION = re.compile(r'^\s*(?:Wait\b|Hmm\b|Hold on\b|Oops\b|Actually,|Let me (?:double|re|check|verify|fix|clean|finalize|write|rewrite|present)|'
                           r'I need to (?:check|fix|double|verify)|But (?:wait|let me)|I\'ll present the final|I should double)', re.M)


def _outside_code(text: str) -> str:
    out, inside = [], False
    for line in text.split('\n'):
        if line.strip().startswith('```'):
            inside = not inside
            continue
        if not inside:
            out.append(line)
    return '\n'.join(out)


def slips(text: str, where: str = 'content') -> list[tuple[str, str]]:
    """(kind, context) for each slip: doubled ("the the"), merged ("ofof", "of10", "ofb$", "*will_"), and in an answer
    (where='content') think_tag (a literal <think> or </think>) and deliberation (a line outside code that starts as thinking
    aloud). Substituted words ("boxes for 6") have no fixed pattern and stay with the reader."""
    text = text or ''
    found = []
    for m in _SLIP_PATTERNS['doubled'].finditer(text):
        if m.group(1).lower() not in _SLIP_ALLOWED_DOUBLES:
            found.append(('doubled', m.start()))
    for m in _SLIP_PATTERNS['merged'].finditer(text):
        found.append(('merged', m.start()))
    out = [(kind, text[max(0, at - 35):at + 25].replace('\n', ' ')) for kind, at in found]
    if where == 'content':
        out += [('think_tag', text[max(0, m.start() - 35):m.start() + 25].replace('\n', ' ')) for m in _THINK.finditer(text)]
        prose = _outside_code(text)
        out += [('deliberation', prose[m.start():m.start() + 40].strip()) for m in _DELIBERATION.finditer(prose)]
    return out


def repetition(text: str, run: int = 8) -> dict:
    """Whether a text loops: the most often repeated run of `run` words and the share of the last 200 runs that repeat an
    earlier one. looping: one run occurs 4 times or more, or over half of the last 200 runs are repeats."""
    words = re.findall(r'\S+', text or '')
    grams = [' '.join(words[i:i + run]) for i in range(max(0, len(words) - run + 1))]
    if not grams:
        return dict(most_repeated=0, tail_repeat_share=0.0, looping=False)
    counts = collections.Counter(grams)
    top = counts.most_common(1)[0][1]
    seen, tail = set(), []
    for g in grams:
        tail.append(g in seen)
        seen.add(g)
    last = tail[-200:]
    share = sum(last) / len(last)
    return dict(most_repeated=top, tail_repeat_share=round(share, 3), looping=top >= 4 or (len(last) >= 50 and share > 0.5))


# ---- the decision rule lives in usability_rule.py; the names of 38cf0ab5 stay importable from here -------------------------------
import usability_rule as _rule   # noqa: E402

fisher_one_sided, upper95, power, detectable = _rule.fisher_one_sided, _rule.upper95, _rule.power, _rule.detectable


def decide(tw: int, tn: int, cw: int, cn: int, base_rate: float, alpha: float = _rule.ALPHA) -> dict:
    """usability_rule.decide with the argument order of 38cf0ab5 (the control's base rate fifth). The outcome "no harm of the
    stated size" now means that a failure rate of usability_rule.MEANINGFUL (50%) or more is excluded - see usability_rule.py."""
    return _rule.decide(tw, tn, cw, cn, alpha=alpha, base_rate=base_rate)
