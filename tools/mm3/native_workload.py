"""Immutable request inputs used when importing a stored benchmark control."""
from __future__ import annotations

import hashlib
import json

REQUEST_FIELDS = ('model', 'messages', 'tools', 'max_tokens', 'temperature', 'top_p', 'top_k', 'min_p',
                  'seed', 'reasoning_effort', 'loop_guard')


def identity(payload: dict, protocol: dict) -> str:
    content = {key: payload.get(key) for key in REQUEST_FIELDS}
    content.update(model_revision=protocol.get('model_revision'), tokenizer_revision=protocol.get('tokenizer_revision'))
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def declared(protocol: dict, step: dict) -> str:
    # Same input construction as FollowUp.payload; tuning differences are checked
    # separately as the arm's declared intervention and effective engine settings.
    import native_screen
    cases = {c['id']: c for c in json.loads(native_screen.FIXTURE.read_text())['cases']}
    cases.update({c['id']: c for c in protocol.get('conditioning', [])})
    case = cases[step['case']]
    payload = dict(model=protocol['model_name'], messages=case.get('messages') or [{'role': 'user', 'content': case['prompt']}],
                   max_tokens=step.get('max_tokens', protocol['max_tokens']), temperature=1, top_p=0.95, top_k=0, min_p=0,
                   seed=step.get('seed', protocol['seed']), reasoning_effort=protocol['reasoning_effort'])
    if case.get('tools'):
        payload['tools'] = case['tools']
    if protocol.get('loop_guard', False) != 'serving':
        payload['loop_guard'] = protocol.get('loop_guard', False)
    return identity(payload, protocol)
