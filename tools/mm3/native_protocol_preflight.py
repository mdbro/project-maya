#!/usr/bin/env python3
"""Pre-flight for a native_followup protocol: every request step through the server's own request checks, on the CPU,
before any engine load. A protocol the server would refuse fails here in seconds instead of after a seven-minute load
(2026-11-10: a check load at --staging 48 stopped on an HTTP 400 for routing_prefetch=0, a key the server refuses when
the engine reports routing_prefetch_control 0).

For each request step: the payload as native_followup builds it (the protocol's tune with the step's merged over it, the
step's seed, the case's prompt), then Service.check_request of this tree's serve/server.py, which also merges the runtime
defaults file the load will use, against a mock engine reporting the controls the binary reports (ENGINE_INFO: the
engine's info as a request's result records it, for the --staging of the load), then StrataEngine.sampling_keys, checking
that every prompt_* key the step asks for is forwarded to the engine (--all-keys: every strata_tune key; a key the
server takes and leaves off the engine's line would otherwise change nothing without a word).
--server-root DIR checks against the server code in DIR instead of this tree's: a resident server runs the code it
loaded at its start (for example `git archive <commit> serve tools | tar -x -C DIR` for the commit it started from).
--config FILE is the start's server config (default: this tree's docs/minimax-wsl/native-server.json): a request's
spec_plan_file names a planner table in the folder of its first --spec-plan table, which a server sets from its config
at its start (set_spec_plan_dir; a server from before 2026-11-13 has none to set).
Usage: native_protocol_preflight.py PROTOCOL ENGINE_INFO DEFAULTS_JSON [--drop-default KEY ...] [--server-root DIR]
       [--config FILE] [--all-keys]
(exit 0: all accepted)"""
import argparse, copy, json, pathlib, sys, tempfile
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/mm3'))
import native_screen as screen                                                # noqa: E402


def forwarded(key: str, value, keys: list[str]) -> bool:
    """Whether a strata_tune key reached the engine's line (spec goes as spec= and dspark_width=)."""
    if key == 'spec':
        return any(k.startswith('spec=') for k in keys) and (value == 1 or any(k.startswith('dspark_width=') for k in keys))
    return any(k.startswith(key + '=') for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('protocol'); ap.add_argument('engine_info'); ap.add_argument('defaults')
    ap.add_argument('--drop-default', action='append', default=[], help='a key the load leaves out of the runtime defaults')
    ap.add_argument('--server-root', type=pathlib.Path, default=ROOT, help='the tree whose serve/server.py checks the requests')
    ap.add_argument('--config', type=pathlib.Path, default=ROOT / 'docs/minimax-wsl/native-server.json',
                    help="the start's server config: a request's spec_plan_file names a table in its first --spec-plan table's folder")
    ap.add_argument('--all-keys', action='store_true', help='every strata_tune key must reach the engine line, not only prompt_*')
    a = ap.parse_args()
    sys.path.insert(0, str(a.server_root.resolve())); sys.path.insert(1, str(a.server_root.resolve() / 'tools'))
    from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine   # noqa: E402
    from serve.frontend import ChatTemplate                                       # noqa: E402
    import serve.server as server                                                 # noqa: E402
    if hasattr(server, 'set_spec_plan_dir'):   # (as the server's main does at its start: the planner tables folder)
        server.set_spec_plan_dir(json.loads(a.config.read_text(encoding='utf-8-sig')))
    # (the keys the mm3 engine takes: a key outside them - one of the other engines' that the server also forwards -
    #  can still reach the line, and the mm3 engine ignores it without a word. cpu_threads is in them since 2026-11-13:
    #  the mm3 engine's CPU-lane threads for the request, 1 .. its lane's (INFO cpu_threads), sent only to an engine
    #  that reports cpu_threads_control 1; check_request refuses it otherwise, so an older binary never gets it)
    mm3_keys = set(server._MM3_RUNTIME_KEYS) | set(getattr(server, '_ROUTE_LAB_KEYS', ())) | {'token_log', 'token_topk', 'force_tokens'}
    proto = json.loads(pathlib.Path(a.protocol).read_text())
    info = json.loads(pathlib.Path(a.engine_info).read_text())
    defaults = json.loads(pathlib.Path(a.defaults).read_text())
    for k in a.drop_default:
        defaults.pop(k, None)
    tmp = pathlib.Path(tempfile.mkdtemp()) / 'native-runtime.json'
    tmp.write_text(json.dumps(defaults))
    tok = ByteTokenizer()
    engine = MockEngine(tok, 'x', max_context=int(info.get('max_context', 8192)))
    engine.info = info
    svc = Service(engine, tok, ChatTemplate(a.server_root / 'serve/chat_template.jinja'), runtime_defaults_file=str(tmp))
    cases = {c['id']: c for c in json.loads(screen.FIXTURE.read_text())['cases']}
    cases.update({c['id']: c for c in proto.get('conditioning', [])})
    bad = n = 0
    for step in proto['sequence']:
        if step.get('kind') != 'request':
            continue
        n += 1
        case = cases[step['case']]
        payload = {'model': proto['model_name'], 'messages': [{'role': 'user', 'content': case['prompt']}], 'stream': True,
                   'return_progress': True, 'max_tokens': step.get('max_tokens', proto['max_tokens']), 'temperature': 1,
                   'top_p': 0.95, 'top_k': 0, 'min_p': 0, 'seed': proto['seed'], 'reasoning_effort': proto['reasoning_effort'],
                   'loop_guard': False, 'strata_tune': {**proto['tune'], **(step.get('tune') or {})}}
        if step.get('seed') is not None:
            payload['seed'] = int(step['seed'])
        asked = {k: v for k, v in payload['strata_tune'].items() if k.startswith('prompt_')}
        req = copy.deepcopy(payload)
        try:
            svc.check_request(req)
            keys = StrataEngine.sampling_keys(req, 'mm3', info).split()
            missing = [f'{k}={v}' for k, v in asked.items() if f'{k}={v}' not in keys]
            if a.all_keys:
                missing += [f'{k}={v}' for k, v in req['strata_tune'].items() if not k.startswith('prompt_') and not forwarded(k, v, keys)]
                ignored = [k for k in req['strata_tune'] if k not in mm3_keys]
                if ignored:
                    raise ValueError(f'not keys the mm3 engine takes (it would ignore them): {ignored}')
            if missing:
                raise ValueError(f'not forwarded to the engine: {missing}')
            print(f"ok      {step['step']}")
        except Exception as e:   # (every refusal is reported, the run continues to the next step)
            bad += 1
            print(f"REFUSED {step['step']}: {type(e).__name__}: {e}")
    print(f'{n} requests, {bad} refused')
    return 1 if bad or not n else 0


if __name__ == '__main__':
    sys.exit(main())
