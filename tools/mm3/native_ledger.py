#!/usr/bin/env python3
"""Point-in-time pre-load ledger for the fixed native MiniMax WSL scenario.

Reads metadata and memory counters only. No CUDA allocation, checkpoint tensor reads,
page-cache manipulation, or inference. Output is a planning estimate, not a load test.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-2/native-pack')
GIB = 1 << 30
H, I, L, E, T, CTX, K = 6144, 3072, 60, 128, 16, 8192, 4
REC = 3 * H * I * 9 // 16
BLOB = REC + 256
ALL = 57 * E

def up(x, n=256): return (x + n - 1) // n * n

def read_tensors():
    tensors = {}
    for line in (ROOT / 'tensors.txt').open():
        p = line.split()
        if p and p[0] == 't':
            tensors[p[1]] = {'dtype': p[2], 'bytes': int(p[5]), 'shape': [int(x) for x in p[6:]]}
    return tensors

def command(*args):
    return subprocess.check_output(args, text=True, timeout=15).strip()

def main():
    ts = read_tensors()
    p = 'language_model.model.'
    layers = {n: v for n, v in ts.items() if n.startswith(p + 'layers.') and '.experts.' not in n}
    # Matches dense_bytes() used by plan_memory, including BF16 vectors expanded to F32.
    dense_stored = sum(v['bytes'] for v in layers.values())
    dense_expansion = sum(v['bytes'] for v in layers.values() if v['dtype'] in ('BF16','F16') and len(v['shape']) == 1)
    dense = dense_stored + dense_expansion
    head = ts['language_model.lm_head.weight']['bytes']
    embedding = ts[p + 'embed_tokens.weight']['bytes']
    draft = {n: v for n, v in ts.items() if n.startswith('dspark.')}
    draft_stored = sum(v['bytes'] for v in draft.values())
    # vec() expands these BF16 vectors to F32 on device; confidence weight is [1,6400].
    draft_vec_extra = sum(v['bytes'] for n,v in draft.items() if v['dtype'] == 'BF16' and
                          (len(v['shape']) == 1 or '.confidence_head.proj.weight' in n))
    draft_weights_device = draft_stored + draft_vec_extra
    gpu = []
    for line in command('nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free',
                        '--format=csv,noheader,nounits').splitlines():
        idx,name,total,used,free = [x.strip() for x in line.split(',')]
        gpu.append({'index':int(idx),'name':name,'total_bytes_reported_mib':int(total)*1048576,
                    'used_bytes_reported_mib':int(used)*1048576,'free_bytes_reported_mib':int(free)*1048576})
    byid = {x['index']:x for x in gpu}
    if '5090' not in byid[0]['name'] or '3090' not in byid[1]['name']: raise RuntimeError('device ordinals differ from fixed scenario')
    fields = dict(x.split('=',1) for x in command('/tmp/mm3-host-memory').split())
    snapshot = {k: (v if k == 'cgroup' else int(v)) for k,v in fields.items()}
    # Plan's chunk scratch expression; the model subsequently allocates individual arrays.
    ffmax=12288; heads=64; hd=128; vocab=200064
    chunk_plan = T*H*4*9 + T*heads*hd*6 + T*ffmax*8 + T*K*H*4 + T*K*I*2
    # Native dense uses two streams by default, expert B=4,C=8, sizeof(FastJob)=48 on x86_64.
    nk=12288; dense_one=16*256*max(16,(nk+511)//512)*4  # selected MV-only context
    fast8=up(8*(H//16)*20)+up(8*(I//16)*20)+up(8*I*2)
    expert_scratch=up(4*48)+2*up(4*8*H*2)+up(8*H*4)+4*fast8
    kv=CTX*4*(132+132)*L
    index=CTX*128*2*L
    tap=5*T*H*4
    logits=32*vocab*4
    stage=8*BLOB
    # Match plan_memory fixed_of, with its explicit 3 GiB draft and 0.2 GiB window allowance.
    # Also charge draft taps: actual code allocates them but fixed_of omits them.
    fixed = dense + head + logits + chunk_plan + stage + int(0.2*GIB) + 3*GIB + 2*dense_one + expert_scratch + kv + index + tap
    reserve=GIB
    free=byid[0]['free_bytes_reported_mib']
    resid=max(0,(free-fixed-reserve)//BLOB)
    resid=min(resid,ALL)
    helper_budget=22*GIB
    helper=min(helper_budget//REC,ALL-resid)
    nonresident=ALL-resid-helper
    bank=nonresident*BLOB
    helper_pool=(helper_budget//REC)*REC  # allocated to requested capacity even if fewer picks
    maxjobs=64; io=maxjobs*H*2; desc=maxjobs*48
    helper_workspace=max((maxjobs//cols)*(up(cols*(H//16)*20)+up(cols*(I//16)*20)+up(cols*I*2)) for cols in range(1,9))
    helper_device_scratch=2*io+desc+helper_workspace
    helper_host_pinned=2*io+desc
    cpu_entries=64
    fast1=up((H//16)*20)+up((I//16)*20)+up(I*2)
    cpu_scratch=2*up(cpu_entries*H*2)+cpu_entries*fast1
    host_total=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:')))*1024
    host_reserve=int(min(16,max(4,host_total/10/GIB))*GIB)
    admission=bank+embedding+host_reserve <= snapshot['effective_physical'] and bank+embedding+host_reserve <= snapshot['effective_commit']
    inventory=json.loads((ROOT/'precision-inventory.json').read_text())
    out={
      'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'method':'Read-only metadata/OS counters. Values are allocation formulas or planning estimates; no model loaded or allocation verified.',
      'scope':{'primary_cuda_ordinal':0,'helper_cuda_ordinal':1,'ctx':CTX,'chunk':T,'staging_slots':8,'helper_budget_gib':22,'sequences':1,'dspark':True,'kv':'int8 both, rotated','side_stream':True,'vision':False},
      'pack':{'metadata_dir':str(ROOT),'inventory_file':'precision-inventory.json','checkpoint_root':inventory['layout']['model_dir'],'weight_copy':inventory['layout']['weight_copy'],'tensor_metadata_count':len(ts),'source_text_parameters':inventory['main_text_parameters'],'source_expert_codes_bytes':inventory['groups']['expert']['NVFP4']['bytes'],'source_expert_block_scale_bytes':inventory['groups']['expert_scales']['F8_E4M3']['bytes'],'source_expert_global_scalar_bytes':inventory['groups']['expert_scales']['F32']['bytes'],'draft_source_weight_and_scale_bytes':draft_stored},
      'gpu_snapshot':gpu,'host_snapshot_bytes':snapshot,
      'expert_geometry':{'count':ALL,'payload_bytes_per_expert':REC,'runtime_header_bytes_per_host_or_primary_record':256,'host_or_primary_record_bytes':BLOB,'all_payload_bytes':ALL*REC,'all_runtime_blob_bytes':ALL*BLOB},
      'primary_bytes':{'layer_dense_stored':dense_stored,'bf16_vector_f32_expansion':dense_expansion,'layer_dense_planner_total':dense,'head_bf16':head,'logits':logits,'chunk_scratch_planner_estimate':chunk_plan,'native_dense_scratch_two_streams':2*dense_one,'native_expert_scratch':expert_scratch,'kv_int8':kv,'indexer_bf16':index,'staging_slots':stage,'draft_planner_estimate':3*GIB,'draft_source_weights_expanded_vectors':draft_weights_device,'draft_kv_bf16_ring_exact':5*(4096+64)*512*2*2,'draft_taps':tap,'planner_window_allowance':int(.2*GIB),'runtime_free_reserve':reserve,'fixed_estimate_including_taps':fixed,'estimated_resident_capacity_experts':resid,'estimated_resident_payload_plus_headers':resid*BLOB},
      'helper_bytes':{'requested_budget':helper_budget,'allocated_record_capacity':helper_budget//REC,'exclusive_record_count_estimate':helper,'pool_payload_allocation':helper_pool,'device_scratch':helper_device_scratch,'pinned_io_and_jobs':helper_host_pinned,'minimum_unallocated_headroom_check':512*1048576},
      'host_bytes':{'estimated_exclusive_nonresident_experts':nonresident,'bank_payload_and_headers_before_vmm_rounding':bank,'bank_vmm_rounding':'unknown until CUDA granularity query','required_os_reserve':host_reserve,'admission_using_current_snapshot_before_vmm_rounding':admission,'resident_embedding_host_vmm_before_rounding':embedding,'embedding_shared_with_draft':True,'cpu_scratch_64_entries':cpu_scratch,'helper_upload_bounded_pinned_record':BLOB,'primary_resident_upload_peak_two_pinned_records':2*BLOB,'primary_persistent_pinned_staging_record':0,'primary_logits_pinned':logits,'checkpoint_source_expert_pages_touched_upper_bound_for_host_experts':nonresident*REC,'checkpoint_pages':'Mapped source pages are OS cache, not an additional persistent allocated bank; transient overlap with bank may approach touched source payload. drop_src advises dropping pages after each assembled record by default, but bank admission occurs before filling the bank. These hints do not establish that initial cached pages have been reclaimed.'},
      'limitations':['Primary residual count uses nvidia-smi MiB-rounded free memory and a planner estimate, not cudaMemGetInfo after actual target/draft allocations; actual runtime can choose a different count.',
       'The 3 GiB DSpark allowance is a planner estimate; raw weights are reported separately and actual draft KV/scratch and driver allocations are not measured.',
       'Helper count assumes 22 GiB is accepted and excludes primary resident IDs. Runtime checks helper free budget plus 512 MiB; helper CUDA and pinned allocations still require measured headroom.',
       'HOST VMM allocation granularity, physical resident pages, file-cache retention, host metadata, pinned buffers, CUDA context, cuBLAS workspace, graphs and runtime thread overhead remain to be measured. Source pages and current file cache are not guaranteed all reclaimable.',
       'Pre-load snapshot is time-sensitive. Reserve check is point-in-time and does not reserve OS resources. No model execution, correctness, startup or throughput claim.']}
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
