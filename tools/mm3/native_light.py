"""Reader and validation for the native light recorder's binary captures.

The layouts mirror src/mm3/mm3_light_recorder.hpp. A capture that overflowed,
lost required intervals or fails state reconstruction is invalid; callers must
not use it as a timing observation.

Format 2 (2026-11-13) adds the 5090's GPU stamps to every layer record, on the
same clock and origin as its other intervals (ns after the record's t_ns):
gpu_prepare_end, gpu_upload_end, gpu_expert_end, gpu_wait_end, the forward's
reference width gpu_clock_span and the pass's upload count gpu_uploads (their
meanings: LayerFixed in the header). Format 1 is still read; its layer records
get those fields as 0, "no stamp".
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import pathlib
import struct

MAGIC = b'MM3LITE1'
HEAD = struct.Struct('<HHIQ')
KIND = {1: 'map', 2: 'generation_begin', 3: 'generation_end', 4: 'forward', 5: 'layer', 6: 'demand_copy',
        7: 'foreground_retire', 8: 'primary_issue', 9: 'primary_publish', 10: 'helper_event', 11: 'third_pass'}
SERVICE = {1: 'helper', 2: 'cpu', 3: 'primary_resident', 4: 'staged_ready', 5: 'staged_new', 6: 'third'}
LOCATION_HOST, LOCATION_PRIMARY, LOCATION_HELPER = 1, 2, 4
HELPER_PHASE = {1: 'lease', 2: 'withdrawn', 3: 'published', 4: 'retired'}
# Same order as the counters array in Model::Impl::chunk_.
FORWARD_COUNTERS = ('staged', 'foreground_queries', 'foreground_not_ready', 'adaptive_queries', 'adaptive_not_ready',
                    'adaptive_source_busy', 'adaptive_issued', 'adaptive_published', 'adaptive_reused',
                    'adaptive_no_gain', 'adaptive_no_slot', 'adaptive_h2d_bytes', 'adaptive_d2h_bytes',
                    'helper_started', 'helper_withdrawals', 'helper_publications', 'helper_finished',
                    'primary_submit_queries')
VERSIONS = (1, 2)
LAYER_FIXED_V1 = struct.Struct('<q h B B 4B 5B 3x 9I 9I')
LAYER_FIXED = struct.Struct('<q h B B 4B 5B 3x 9I 9I 5I H 2x')   # (format 2: the GPU stamps after the helper's)
LAYER_FIELDS_V1 = ('forward', 'layer', 'rows', 'k', 'free_slots', 'reserved_slots', 'foreground_slots', 'pending_promotions',
                   'exchanges_pending', 'helper_lease', 'stage_max', 'stage_selected', 'total_slots',
                   'pre_end', 'join_end', 'boundary_end', 'plan_end', 'helper_submit_end', 'primary_submit_end',
                   'cpu_end', 'collect_end', 'end',
                   'helper_enqueue', 'helper_worker_start', 'helper_record_begin', 'helper_record_end',
                   'helper_last_not_ready', 'helper_ready_observed', 'helper_worker_published',
                   'helper_wait_begin', 'helper_wait_end')
GPU_FIELDS = ('gpu_prepare_end', 'gpu_upload_end', 'gpu_expert_end', 'gpu_wait_end', 'gpu_clock_span', 'gpu_uploads')
LAYER_FIELDS = LAYER_FIELDS_V1 + GPU_FIELDS
FORWARD_FIXED = struct.Struct('<qq iiiI 4I 18q')
GENERATION = struct.Struct('<8q d II')
DEMAND = struct.Struct('<qq iiiII4x')
RETIRE = struct.Struct('<qq i4x')
ISSUE = struct.Struct('<qqqQQ iiII')
PUBLISH = struct.Struct('<qqq ii')
HELPER = struct.Struct('<qqq iI')
THIRD = struct.Struct('<q hHHH 6I')   # (third: the third card's lane in one pass)
MAP_FIXED = struct.Struct('<qqqq HHHH II')
SLOT = struct.Struct('<qq BBB5x')
EXCHANGE = struct.Struct('<qq ii')


@dataclass
class LightCapture:
    path: pathlib.Path
    sha256: str
    meta: dict
    records: list[dict] = field(default_factory=list)
    format_version: int = 2   # (read() sets the file's; 1 has no GPU stamps)

    def of(self, kind: str) -> list[dict]:
        return [r for r in self.records if r['type'] == kind]

    def generation_forwards(self) -> list[dict]:
        return [r for r in self.records if r['type'] == 'forward' and r['generation']]

    def scored_ms(self) -> float:
        end = self.of('generation_end')
        if len(end) != 1:
            raise ValueError('capture lacks exactly one generation end')
        return end[0]['elapsed_ms']


def _parse_map(body: bytes) -> dict:
    forward, row_clock, helper_incoming, helper_victim, layers, experts, slots, exchanges, phase, flags = MAP_FIXED.unpack_from(body, 0)
    at = MAP_FIXED.size
    location = body[at:at + layers * experts]
    at += layers * experts
    slot_rows = []
    for _ in range(slots):
        owner, pending, issued, foreground, reserved = SLOT.unpack_from(body, at)
        at += SLOT.size
        slot_rows.append(dict(owner=owner, pending=pending, issued=issued, foreground=foreground, reserved=reserved))
    exchange_rows = []
    for _ in range(exchanges):
        incoming, victim, landing, submitted = EXCHANGE.unpack_from(body, at)
        at += EXCHANGE.size
        exchange_rows.append(dict(incoming=incoming, victim=victim, landing=landing, submitted=submitted))
    return dict(forward=forward, row_clock=row_clock, helper_incoming=helper_incoming, helper_victim=helper_victim,
                layers=layers, experts=experts, phase=phase, helper_flags=flags, location=location,
                slots=slot_rows, exchanges=exchange_rows)


def read(path) -> LightCapture:
    path = pathlib.Path(path)
    raw = path.read_bytes()
    if raw[:8] != MAGIC:
        raise ValueError(f'{path}: not a light capture')
    version, json_bytes, record_bytes = struct.unpack_from('<IIQ', raw, 8)
    if version not in VERSIONS:
        raise ValueError(f'{path}: unsupported format version {version}')
    layer_fixed, layer_fields = (LAYER_FIXED, LAYER_FIELDS) if version >= 2 else (LAYER_FIXED_V1, LAYER_FIELDS_V1)
    meta = json.loads(raw[24:24 + json_bytes])
    at = 24 + json_bytes + ((8 - json_bytes % 8) % 8)
    if len(raw) != at + record_bytes:
        raise ValueError(f'{path}: truncated records')
    capture = LightCapture(path, hashlib.sha256(raw).hexdigest(), meta, format_version=version)
    timing_only = meta.get('mode') == 'timestamps'
    end = at + record_bytes
    while at < end:
        kind, bytes_, seq, t_ns = HEAD.unpack_from(raw, at)
        if bytes_ < HEAD.size or at + bytes_ > end or kind not in KIND:
            raise ValueError(f'{path}: malformed record at offset {at}')
        body = raw[at + HEAD.size:at + bytes_]
        name = KIND[kind]
        row = dict(type=name, seq=seq, t_ns=t_ns)
        if name == 'layer':
            values = layer_fixed.unpack_from(body, 0)
            row.update(zip(layer_fields, values))
            if version < 2:
                row.update(dict.fromkeys(GPU_FIELDS, 0))   # (format 1: no GPU stamps)
            picks = 0 if timing_only else row['rows'] * row['k']
            tail = layer_fixed.size
            row['original'] = list(body[tail:tail + picks])
            row['executed'] = list(body[tail + picks:tail + 2 * picks])
            row['service'] = [SERVICE.get(v, 'unknown') for v in body[tail + 2 * picks:tail + 3 * picks]]
        elif name == 'forward':
            v = FORWARD_FIXED.unpack_from(body, 0)
            row.update(forward=v[0], row_clock=v[1], position=v[2], rows=v[3], want=v[4], generation=bool(v[5]),
                       prepare_end=v[6], body_end=v[7], sync_end=v[8], end=v[9])
            row['counters'] = dict(zip(FORWARD_COUNTERS, v[10:]))
        elif name in ('generation_begin', 'generation_end'):
            v = GENERATION.unpack_from(body, 0)
            row.update(forward=v[0], row_clock=v[1], position=v[2], emitted=v[3], drafts_offered=v[4],
                       drafts_accepted=v[5], target_rounds=v[6], spec_rounds=v[7], elapsed_ms=v[8],
                       scope='fixture' if v[9] else 'request', ok=bool(v[10]))
        elif name == 'demand_copy':
            v = DEMAND.unpack_from(body, 0)
            row.update(forward=v[0], displaced=v[1], layer=v[2], expert=v[3], slot=v[4], bytes=v[5], submit_end=v[6])
        elif name == 'foreground_retire':
            v = RETIRE.unpack_from(body, 0)
            row.update(forward=v[0], owner=v[1], slot=v[2])
        elif name == 'primary_issue':
            v = ISSUE.unpack_from(body, 0)
            row.update(forward=v[0], incoming=v[1], victim=v[2], h2d_bytes=v[3], d2h_bytes=v[4], landing=v[5],
                       early_layer=v[6], reuse=bool(v[7]), submit_end=v[8])
        elif name == 'primary_publish':
            v = PUBLISH.unpack_from(body, 0)
            row.update(forward=v[0], incoming=v[1], victim=v[2], landing=v[3], protected_layer=v[4])
        elif name == 'helper_event':
            v = HELPER.unpack_from(body, 0)
            row.update(forward=v[0], incoming=v[1], victim=v[2], phase=HELPER_PHASE.get(v[3], 'unknown'), submit_end=v[4])
        elif name == 'third_pass':
            v = THIRD.unpack_from(body, 0)
            row.update(forward=v[0], layer=v[1], jobs=v[2], columns=v[3], enqueue=v[5], worker_start=v[6],
                       worker_published=v[7], wait_begin=v[8], wait_end=v[9], merge_end=v[10])
        elif name == 'map':
            row.update(_parse_map(body))
        capture.records.append(row)
        at += bytes_
    return capture


def _replay_locations(start: dict, capture: LightCapture) -> bytearray:
    """Apply recorded mutations; returns the reconstructed canonical location map."""
    experts = start['experts']
    state = bytearray(start['location'])
    index = lambda key: (key // 1024) * experts + key % 1024
    in_map = False
    for r in capture.records:
        if r['type'] == 'map' and r['phase'] == 0:
            in_map = True
            continue
        if not in_map:
            continue
        if r['type'] == 'primary_publish':
            state[index(r['incoming'])] = LOCATION_PRIMARY
            state[index(r['victim'])] = LOCATION_HOST
        elif r['type'] == 'helper_event':
            if r['phase'] == 'withdrawn':
                state[index(r['victim'])] = LOCATION_HOST
            elif r['phase'] == 'published':
                state[index(r['incoming'])] = LOCATION_HELPER
    return state


def validate(capture: LightCapture, expected_layers: int = 57) -> dict:
    """Coverage and consistency checks. Any error invalidates the capture."""
    errors: list[str] = []
    meta = capture.meta
    timing_only = meta.get('mode') == 'timestamps'
    if meta.get('dropped_records'):
        errors.append(f"recorder dropped {meta['dropped_records']} records")
    if meta.get('attempted_records') != len(capture.records):
        errors.append('recorded count differs from attempted count')
    if [r['seq'] for r in capture.records] != list(range(1, len(capture.records) + 1)):
        errors.append('record sequence gap')
    maps = capture.of('map')
    begin, end = capture.of('generation_begin'), capture.of('generation_end')
    if timing_only:
        if maps or any(r['type'] not in ('generation_begin', 'generation_end', 'forward', 'layer') for r in capture.records):
            errors.append('timestamps-only capture holds full records')
    elif [m['phase'] for m in maps] != [0, 1]:
        errors.append('expected one starting and one ending state map')
    if len(begin) != 1 or len(end) != 1:
        errors.append('expected one generation begin and end')
    elif not end[0]['ok']:
        errors.append('generation did not finish successfully')
    forwards = capture.of('forward')
    layers = Counter(r['forward'] for r in capture.of('layer'))
    incomplete = [f['forward'] for f in forwards if layers.get(f['forward']) != expected_layers]
    if incomplete:
        errors.append(f'{len(incomplete)} forwards lack {expected_layers} MoE layer records')
    if set(layers) - {f['forward'] for f in forwards}:
        errors.append('layer records without a completed forward')
    missing = 0
    for r in capture.of('layer'):
        if not (r['pre_end'] and r['join_end'] and r['boundary_end'] and r['plan_end'] and r['primary_submit_end'] and r['end']):
            missing += 1
        elif not r['pre_end'] <= r['join_end'] <= r['boundary_end'] <= r['plan_end'] <= r['primary_submit_end'] <= r['end']:
            missing += 1
        if 'unknown' in r['service']:
            missing += 1
    if missing:
        errors.append(f'{missing} layer records lack required ordered intervals or service locations')
    generation = capture.generation_forwards()
    non_ordinary = sum(f['rows'] != 1 for f in generation)
    summary = dict(status='invalid' if errors else 'validated', errors=errors, sha256=capture.sha256,
                   records=len(capture.records), record_bytes=meta.get('record_bytes'),
                   capacity_bytes=meta.get('capacity_bytes'), forwards=len(forwards),
                   generation_forwards=len(generation), generation_rows_other_than_one=non_ordinary,
                   layer_records=sum(layers.values()))
    summary.update(gpu_stamps(capture))
    if len(maps) == 2 and not errors:
        rebuilt = _replay_locations(maps[0], capture)
        final = maps[1]['location']
        differing = sum(1 for a, b in zip(rebuilt, final) if a != b)
        # A helper lease still open at the end can leave a transitional owner.
        summary['state_reconstruction'] = dict(differing_locations=differing,
            open_exchanges=len(maps[1]['exchanges']), open_helper_lease=maps[1]['helper_incoming'] >= 0)
        if differing and not (maps[1]['helper_incoming'] >= 0 and differing <= 2):
            errors.append(f'{differing} locations differ after applying recorded mutations')
            summary['status'] = 'invalid'
    if end and len(end) == 1:
        e = end[0]
        summary.update(scored_ms=e['elapsed_ms'], emitted=e['emitted'], drafts_offered=e['drafts_offered'],
                       drafts_accepted=e['drafts_accepted'], target_rounds=e['target_rounds'], spec_rounds=e['spec_rounds'],
                       scope=e['scope'])
    return summary


def gpu_stamps(capture: LightCapture) -> dict:
    """The 5090's GPU stamps over the layer records (format 2; information, never a reason to invalidate): how many
    records carry the expert-compute stamp, how many break the stream's order (preparation, then the arrival at the expert
    phase, its own compute, its wait - each where present), and the widest clock reference. A format 1 capture, or
    records built without the fields, count as unstamped."""
    layers = capture.of('layer')
    stamped = [r for r in layers if r.get('gpu_expert_end')]
    order = 0
    for r in stamped:
        seen = [r.get(k, 0) for k in ('gpu_prepare_end', 'gpu_upload_end', 'gpu_expert_end', 'gpu_wait_end')]
        seen = [v for v in seen if v]
        order += any(a > b for a, b in zip(seen, seen[1:]))
    return dict(format_version=capture.format_version, gpu_stamped_layers=len(stamped),
                gpu_unstamped_layers=len(layers) - len(stamped), gpu_stamp_order_breaks=order,
                gpu_clock_span_max=max((r.get('gpu_clock_span', 0) for r in stamped), default=0))


def state_digest(capture: LightCapture, phase: int = 0) -> str:
    """Digest of one logical location map. Equal digests do not prove equal physical warmth."""
    entry = [m for m in capture.of('map') if m['phase'] == phase]
    if len(entry) != 1:
        raise ValueError('state map missing')
    return hashlib.sha256(entry[0]['location']).hexdigest()


def export_fixture(capture: LightCapture, output, generation_forwards: int = 64, *, budget: int = 8, stage_limit: int = 4,
                   cpu_scale: float = 1.3, margin_ms: float = 0, hold_primary: bool = False, hold_helper: bool = False,
                   primary_query: bool = False, helper_occupancy: bool = False, ordinary: bool = True) -> dict:
    """Write a version 4 fixed workload: complete prefill setup, then the first N generation forwards.

    Selects by order only; timing fields are never read here. Fails unless every selected
    forward is complete, contiguous and, for generation, exactly one row.
    """
    layers: dict[int, list[dict]] = {}
    for r in capture.of('layer'):
        layers.setdefault(r['forward'], []).append(r)
    forwards = capture.of('forward')
    prefill = [f for f in forwards if not f['generation']]
    generation = [f for f in forwards if f['generation']]
    if len(generation) < generation_forwards:
        raise ValueError(f'source has {len(generation)} generation forwards; {generation_forwards} required')
    selected = prefill + generation[:generation_forwards]
    if len(selected) > 256:
        raise ValueError('fixed workload exceeds the native 256-batch limit')
    lines = [f'MM3_REPLAY 4 {budget} {stage_limit} {cpu_scale} {margin_ms} {int(primary_query)} {int(hold_primary)} {int(hold_helper)} {int(helper_occupancy)}']
    position = 0
    for f in selected:
        if f['position'] > position or (ordinary and f['position'] != position):
            raise ValueError('fixture requires contiguous positions from zero')
        if ordinary and f['generation'] and f['rows'] != 1:
            raise ValueError('ordinary fixture generation forwards must have one row')
        rows = sorted(layers.get(f['forward'], []), key=lambda r: r['layer'])
        if not rows or any(r['rows'] != f['rows'] for r in rows):
            raise ValueError('forward layer records incomplete')
        lines.append(f"F {f['rows']} {f['position']} {int(f['generation'])}")
        for r in rows:
            lines.append(' '.join(map(str, ['L', r['layer'], *r['original'], *r['executed']])))
        # A rejected speculative row is evaluated again at the same position.
        position = f['position'] + f['rows']
    lines.append('END')
    text = '\n'.join(lines) + '\n'
    with open(output, 'x') as f:
        f.write(text)
    return dict(path=str(output), sha256=hashlib.sha256(text.encode()).hexdigest(), source_sha256=capture.sha256,
                prefill_forwards=len(prefill), generation_forwards=generation_forwards,
                moe_layers=len(rows), hold_primary=hold_primary, hold_helper=hold_helper, primary_query=primary_query,
                helper_occupancy=helper_occupancy,
                budget=budget, stage_limit=stage_limit, cpu_scale=cpu_scale, margin_ms=margin_ms)
