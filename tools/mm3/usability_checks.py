#!/usr/bin/env python3
"""The short usability check's automatic checks and its decision rule (installed 2026-11-14 from the usability audit,
runs/usability-audit-20261114/: the user asked for usability verdicts that are accurate from now on).

  check_muffins_v2(text)      'right' | 'wrong' | 'read' (a reader decides) | 'cut off'. 'wrong' only on positive evidence: the
                              final section states a count other than 3 and never 3. Anything the patterns do not understand
                              goes to the reader instead of being scored wrong. On the 111 distinct real answers on disk
                              (2026-11-07..14, all right by reading): 109 right, 2 read, 0 wrong.
  fenced_code_v2(text)        the answer's code block: the last block that DEFINES order_jobs; accepts ```python3, ``` python,
                              ~~~ fences and an unclosed last fence.
  check_action_items(text)    structure only: the JSON list parses and has exactly the three owners Tomas, Lena, Priya; calendar
                              dates the notes do not give go to the reader, never scored.
  decide(...)                 the decision rule for one automatic task, both arms on ONE engine start: 'harm shown' (one-sided
                              Fisher exact p <= 0.05), 'no harm of the stated size' (the tested arm's 95% upper bound is below
                              the smallest failure rate the run could show 80% of the time), or "can't tell". It replaces the
                              template's "fails when wrong on more seeds than the control", which at 2-8 seeds prints FAILS
                              23-37% of the time when nothing differs (runs/usability-audit-20261114/rule_stats.txt).
  detectable(...)             size a run BEFORE it runs: the smallest tested-arm failure rate shown 80% of the time.
"""
from __future__ import annotations
import json, re
from math import comb

VALID = ((1, 7), (4, 5), (7, 3))
WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
NUM = r'(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)'
NOUN = r'(?:combinations?|ways?|options?|possibilit(?:y|ies)|choices?|solutions?|pairs?)'
_FINAL = re.compile(r'^[ \t]*(?:#{1,6}[ \t]*|\*\*)[ \t]*(?:the[ \t]+)?(?:final[ \t]+)?(?:answer|conclusion|result|summary)\b', re.I | re.M)
# a stated count: N <up to 30 characters with no digit, no sentence end> combinations/ways/...; N not a box size ("box of 6 combination")
_COUNT = re.compile(r'(?<![\w.\-])' + NUM + r'\**(?=\s)((?:(?![\d.;:\n]).){0,30}?)\b' + NOUN + r'\b', re.I)
_OF_M = re.compile(r'(?<![\w.\-])' + NUM + r'\**\s+(?:of|out of)\s+(?:the\s+)?' + NUM + r'\s+(?:[A-Za-z-]+\s+){0,2}?' + NOUN + r'\b', re.I)
_ANSWER = re.compile(r'\banswer\**\s*(?:is|:)[\s*]*' + NUM + r'\b|\\boxed\{\s*' + NUM + r'\s*\}|\b(?:count|total|number)(?:\s+of\s+[a-z ]{0,30}?)?\s+is\s+\**' + NUM + r'\b', re.I)
_BARE = re.compile(r'^[ \t]*\**' + NUM + r'\**\.?[ \t]*$', re.I | re.M)
_COUNTERFACTUAL = re.compile(r'\b(?:would|without|ignoring|were|if|unless|other|remaining|extra|additional|excluded?|rejected|fails?|fourth)\b', re.I)
_BOX_BEFORE = re.compile(r'(?:box(?:es)?|packs?|\bof|\bfor|\bby|\bon)\s*$', re.I)


def _n(s: str) -> int:
    s = s.lower()
    return WORDS[s] if s in WORDS else int(s)


def _sentence(text: str, start: int, end: int) -> str:
    a = text.rfind('\n', 0, start) + 1
    b = text.find('\n', end)
    b = len(text) if b < 0 else b
    for m in re.finditer(r'[.!?;]\s', text[a:start]):
        a2 = a + m.end()
        if a2 <= start:
            a = a2
    m = re.search(r'[.!?;](?:\s|$)', text[end:b])
    return text[a:end + m.end() if m else b]


def final_section(text: str) -> str:
    heads = list(_FINAL.finditer(text))
    return text[heads[-1].start():] if heads else text[-700:]


def muffin_claims(section: str) -> list[tuple[int, int, bool]]:
    """Every stated count in the section, in order: (position, number, counterfactual sentence?)."""
    out = []
    taken = []
    for m in _OF_M.finditer(section):                     # "3 of the 4 solutions fit": the count is 3
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


def muffin_pairs_v2(text: str) -> list[tuple[int, int]]:
    """Which of the three valid pairs the answer names. Notations: "7 boxes of 4 + 3 boxes of 6" with any short preposition (a
    substituted "for", "by", "on" is not a failure: the user, 2026-11-10) and number words; "4-packs", "four-packs"; "(7, 3)";
    "7(4) + 3(6)", "4(7) + 6(3)", "7 x 4 + 3 x 6"; "a = 7, b = 3" (also "a=1: b = 7"); a table row holding both numbers with their
    box total or both products (whatever the header says)."""
    t = text.replace('\\cdot', '×').replace('\\times', '×').replace('‑', '-').replace('‐', '-').replace('−', '-')
    t = re.sub(r'[$*`\\]', '', t)
    def num(n): return r'(?:' + str(n) + '|' + [k for k, v in WORDS.items() if v == n][0] + r')'
    prep = r'(?:of|for|by|on|with)'
    four = r'(?:(?:box(?:es)?|packs?|cartons?)\s+' + prep + r'\s+(?:4|four)|(?:4|four)-?\s?(?:box(?:es)?|packs?)|small(?:\s+box(?:es)?)?)\b'
    six = r'(?:(?:box(?:es)?|packs?|cartons?)\s+' + prep + r'\s+(?:6|six)|(?:6|six)-?\s?(?:box(?:es)?|packs?)|large(?:\s+box(?:es)?)?)\b'
    times = r'\s*(?:\(\s*{n}\s*\)|[×x·*]\s*{n})'
    found = set()
    for a, b in VALID:
        pats = (rf'(?<![\d\w]){num(a)}\s*{four}[^\n]{{0,40}}?(?<![\d\w]){num(b)}\s*{six}',
                rf'(?<![\d\w]){num(b)}\s*{six}[^\n]{{0,40}}?(?<![\d\w]){num(a)}\s*{four}',
                rf'\(\s*{a}\s*,\s*{b}\s*\)',
                rf'(?<![\d.]){a}' + times.format(n=4) + rf'\s*\+\s*{b}' + times.format(n=6) + r'(?![\d.])',
                r'(?<![\d.])4' + times.format(n=a) + r'\s*\+\s*6' + times.format(n=b) + r'(?![\d.])',
                rf'\b[a-z]\s*=\s*{a}\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*{b}(?!\d)',
                rf'\b[a-z]\s*=\s*{b}\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*{a}(?!\d)(?=[^\n]*\b{a + b}\b)')
        if any(re.search(q, t, re.I) for q in pats):
            found.add((a, b))
        for line in t.splitlines():                       # a table row: both numbers, and the box total or both products
            if not line.lstrip().startswith('|'):
                continue
            ints = [int(x) for x in re.findall(r'(?<![\d.])\d+(?![\d.])', line)]
            if a in ints and b in ints and (a + b in ints or (4 * a in ints and 6 * b in ints)) and not (a == 1 and 10 in ints and 11 in ints):
                found.add((a, b))
    return sorted(found)


_REJECTS = re.compile(r"\b11\b|exclud|reject|too many|exceed|violat|fail|\bnot\b|n't|ruled out|[✗✘❌×]|\bover\b|more than|invalid|only other|near-miss|breaks|doesn|cannot|can't|disqualif|drop|remov|discard", re.I)


def muffin_accepts_wrong_set(text: str) -> str | None:
    """A reason to send the answer to a reader: (10, 1) named on a line with no sign of rejection, or a valid pair named in a
    sentence of the final section that rejects it. None when neither shows."""
    t = re.sub(r'[$*`\\]', '', text.replace('\\cdot', '×').replace('\\times', '×'))
    ten = (r'(?<![\d\w])(?:10|ten)\s*(?:box(?:es)?|packs?)\s+(?:of|for|by|on)\s+(?:4|four)[^\n]{0,40}?(?<![\d\w])(?:1|one)\s*(?:box(?:es)?|packs?)\s+(?:of|for|by|on)\s+(?:6|six)',
           r'\(\s*10\s*,\s*1\s*\)', r'\b[a-z]\s*=\s*10\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*1(?!\d)', r'\b[a-z]\s*=\s*1\s*(?:,|;|:|and)?\s*[a-z]\s*=\s*10(?!\d)')
    for line in t.splitlines():
        for q in ten:
            for m in re.finditer(q, line, re.I):
                if not _REJECTS.search(_sentence(line, m.start(), m.end())) and not _REJECTS.search(line):
                    return f'(10, 1) named with no sign of rejection: "{line.strip()[:90]}"'
    sec = re.sub(r'[$*`\\]', '', final_section(text))
    for a, b in VALID:
        for m in re.finditer(rf'\(\s*{a}\s*,\s*{b}\s*\)', sec):
            sent = _sentence(sec, m.start(), m.end())
            if re.search(r'ruled out|exclud|reject|invalid|does not work|doesn\'t work|fails', sent, re.I) and not re.search(r'\b11\b|10\s*,\s*1|ten', sent):
                return f'the valid pair ({a}, {b}) named in a sentence that rejects it: "{sent.strip()[:90]}"'
    return None


def check_muffins_v2(text: str, finish: str | None = 'stop') -> dict:
    if finish not in (None, 'stop'):
        return dict(verdict='cut off', final_count=None, pairs=[], why=f'finish {finish}: not judged')
    sec = final_section(text)
    claims = [(p, n) for p, n, cf in muffin_claims(sec) if not cf]
    first = claims[0][1] if claims else None
    any3 = any(n == 3 for _, n in claims)
    pairs = muffin_pairs_v2(text)
    all3 = pairs == sorted(VALID)
    odd = muffin_accepts_wrong_set(text)
    if first == 3 and all3 and odd:
        v, why = 'read', 'states 3 and names the three valid pairs, but ' + odd
    elif first == 3 and all3:
        v, why = 'right', 'the final section states 3 first and the three valid pairs are named'
    elif first == 3:
        v, why = 'read', f'states 3; valid pairs recognized: {pairs} (a notation the checker does not read, or a pair missing)'
    elif first is None:
        v, why = 'read', 'no stated count found in the final section'
    elif any3:
        v, why = 'read', f'the final section states {first} first and 3 later'
    else:
        v, why = 'wrong', f'the final section states {first} and never 3'
    return dict(verdict=v, final_count=first, pairs=pairs, why=why)


def fenced_code_v2(text: str) -> str:
    blocks = re.findall(r'(?:```|~~~)[ \t]*[\w+.-]*[ \t]*\n(.*?)(?:```|~~~)', text, re.S)
    if not blocks:                                         # an unclosed last fence (a cut-off answer)
        m = re.search(r'(?:```|~~~)[ \t]*[\w+.-]*[ \t]*\n(.*)\Z', text, re.S)
        blocks = [m.group(1)] if m else []
    for block in reversed(blocks):
        if re.search(r'^\s*def\s+order_jobs\s*\(', block, re.M):      # the block that DEFINES it, not one that calls it
            return block
    return blocks[-1] if blocks else text.strip()


def check_action_items(text: str) -> dict:
    items = None
    for m in re.finditer(r'\[', text):
        depth = 0
        for i in range(m.start(), len(text)):
            depth += text[i] == '['
            if text[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[m.start():i + 1])
                        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                            items = v
                    except Exception:
                        pass
                    break
        if items:
            break
    if items is None:
        return dict(verdict='read', why='no JSON list of objects found')
    owners = sorted(str(next((v for k, v in d.items() if 'owner' in k.lower() or k.lower() in ('who', 'assignee')), '?')) for d in items)
    dates = sorted(set(re.findall(r'20\d\d-\d\d-\d\d|\b\d{1,2} March 20\d\d|\b\d{1,2}/\d{1,2}/\d{2,4}', json.dumps(items))))
    if owners != ['Lena', 'Priya', 'Tomas']:
        return dict(verdict='wrong', why=f'{len(items)} items with owners {owners}: the notes give three (Lena, Priya, Tomas)', owners=owners, calendar_dates=dates)
    return dict(verdict='right structure' if not dates else 'read', why='three items, the three owners' + (f'; calendar dates the notes do not give: {dates}' if dates else ''),
                owners=owners, calendar_dates=dates)


# --- the decision rule -------------------------------------------------------------------------------------------------------
def _pmf(n, p): return [comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n + 1)]

def fisher_one_sided(tw: int, tn: int, cw: int, cn: int) -> float:
    """P(the tested arm has at least tw wrong | the margins) when both arms share one failure rate."""
    tot = tw + cw
    return sum(comb(tn, x) * comb(cn, tot - x) for x in range(tw, min(tn, tot) + 1) if 0 <= tot - x <= cn) / comb(tn + cn, tot)

def upper95(n: int, k: int) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(_pmf(n, mid)[:k + 1]) > 0.05: lo = mid
        else: hi = mid
    return hi

def power(tn: int, cn: int, pt: float, pc: float, alpha: float = 0.05) -> float:
    a, b = _pmf(tn, pt), _pmf(cn, pc)
    return sum(a[i] * b[j] for i in range(tn + 1) for j in range(cn + 1) if fisher_one_sided(i, tn, j, cn) <= alpha)

def detectable(tn: int, cn: int, pc: float, want: float = 0.8) -> float | None:
    """The smallest tested-arm failure rate the run shows as harm with probability `want` (None: no rate reaches it)."""
    for i in range(1, 100):
        if power(tn, cn, i / 100, pc) >= want:
            return i / 100
    return None

def decide(tw: int, tn: int, cw: int, cn: int, base_rate: float, alpha: float = 0.05) -> dict:
    """tw of tn tested answers wrong, cw of cn control answers wrong (finished answers of one automatic task, both arms on one
    engine start). base_rate: the task's failure rate in earlier runs (to size the run, not to judge it)."""
    p = fisher_one_sided(tw, tn, cw, cn)
    det = detectable(tn, cn, base_rate)
    can_show = any(fisher_one_sided(i, tn, 0, cn) <= alpha for i in range(tn + 1))
    if p <= alpha:
        outcome = 'harm shown'
    elif det is not None and upper95(tn, tw) < det + 1e-9:
        outcome = 'no harm of the stated size'
    else:
        outcome = "can't tell"
    return dict(outcome=outcome, p=p, tested_upper95=upper95(tn, tw), smallest_rate_shown_80pct=det, any_outcome_can_show_harm=can_show)
