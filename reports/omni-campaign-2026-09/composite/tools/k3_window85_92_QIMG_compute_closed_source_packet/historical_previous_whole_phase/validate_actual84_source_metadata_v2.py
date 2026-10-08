"""Concrete actual84 metadata/source/PID/cap validator; source-only unexecuted.

The predecessor module argument must be fullSHA-pinned to the final reviewed normal source binder.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,re,uuid
PRIOR_NATIVE='21e322039fa65334a49e222997f92a76e640cf68d75e5388244b1e7719baae56'
POSITIVE_HEADER_DELTA=4110417920
def require(ok,message):
    if ok is not True:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def raw_ref(ref):
    require(isinstance(ref,dict) and set(ref)=={'path','sha256'} and isinstance(ref['path'],str)
        and re.fullmatch('[0-9a-f]{64}',ref.get('sha256') or '') is not None,'actual raw fullSHA reference remains unresolved')
    path=Path(ref['path']);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=64<<20,'bounded actual metadata required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and sha(raw)==ref['sha256'],'actual full raw pin changed')
    return raw

def decode(raw):
    def unique(pairs):
        out={}
        for key,value in pairs:require(key not in out,'duplicate metadata key');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda value:require(False,'nonfinite metadata'))

def validate_actual_84(spec,pins,previous_binder):
    """Require real closed84 and complete recursively authenticated predecessor.

    previous_binder must be loaded by the final normal source binder from its
    exact independently reviewed fullSHA. This function does no model/remote IO.
    """
    roles={'closed84_continuity','terminal_receipt','full_ffn_validation','snapshot_manifest','fitter_execution',
        'controller_terminal_result','resources_jsonl','controller_events_jsonl','binding_receipt'}
    prior_roles={'source_closure','source_binding','identity','catalog','observation','driver_config','source_freeze','allocation','recovery_plan'}
    require(set(spec['inputs'])==roles and set(spec['previous73']['inputs'])==prior_roles,
        'exact actual84 raw9 and full previous73 provenance required')
    raw={role:raw_ref(ref) for role,ref in spec['inputs'].items()}
    previous_raw={role:raw_ref(ref) for role,ref in spec['previous73']['inputs'].items()}
    values={role:decode(value) for role,value in raw.items() if not role.endswith('_jsonl')}
    prior={role:decode(value) for role,value in previous_raw.items()}
    bundle=Path(spec['previous73']['source_bundle']).resolve()
    previous_binder.RUN=prior['identity']['run_uuid']
    previous_binder.verify_bound_bundle(bundle,prior['source_closure'])
    require(prior['source_binding']==prior['source_closure']['source_binding'],
        'whole previous73 source closure/binding differs')
    proof_pin=prior['source_closure']['sources']['completed61_phase_source']
    proof=decode(raw_ref({'path':str(bundle/'production_control'/proof_pin['name']),'sha256':proof_pin['sha256']}))
    earlier_spec={'inputs':proof['actual72_nine_role_pins'],'previous61':proof['previous61_complete_provenance'],'operating_cap_proposal':proof.get('retained_operating_cap_proposal',{'memory_bytes':proof['next_operating_cap_proposal_bytes'],'CPU_threads':22,'swap_bytes':0})}
    earlier=previous_binder.validate_actual72(earlier_spec,previous_binder.selected())
    earlier_values=earlier[4];earlier_raw=earlier[3]
    previous=prior['identity'];cfg=prior['driver_config'];freeze=prior['source_freeze'];allocation=prior['allocation']
    prior_run=previous['run_uuid'];require(isinstance(prior_run,str) and str(uuid.UUID(prior_run))==prior_run,
        'actual preceding source UUID required; no invented future UUID')
    require(previous['new_layers']==[73,84] and cfg['identity']==previous
        and cfg['schema']=='neural_forge.k3_nas_closed72_worker_config.v1' and cfg['window']==[73,84]
        and cfg['native_tool_sha256']==PRIOR_NATIVE and cfg['budget']['cpus']==22,
        'actual previous73 native/context/config required')
    require(sha(previous_raw['identity'])==prior['source_closure']['sources']['identity']['sha256']
        and sha(previous_raw['catalog'])==cfg['resume_source_manifest']['sha256']
        and sha(previous_raw['observation'])==cfg['resume_source_observation']['sha256'],
        'complete preceding identity/catalog/observation actual byte joins differ')
    require(freeze['identity']==previous and freeze['native_source_sha256']==PRIOR_NATIVE
        and freeze['config_sha256']==sha(previous_raw['driver_config'])
        and freeze['allocation_id']==allocation['allocation_id'] and freeze['container_id']==allocation['container_id']
        and freeze['schema']=='neural_forge.k3_NAS73_84_execution_source_freeze.v1'
        and freeze['immutable_source_ro_mounts'] is True and freeze['old_source_unchanged'] is True,
        'whole real preceding source freeze/config/A/container lineage differs')
    expected_code={name[len('production_code/'):]:row['sha256'] for name,row in prior['source_binding']['production_file_pins'].items() if name.startswith('production_code/')}
    require(freeze['code_files']==expected_code,'complete actual prior execution code roster differs')
    for name,row in prior['source_binding']['production_file_pins'].items():
        if name.startswith('production_control/'):
            require(freeze['control_files'][name[len('production_control/'):]]==row['sha256'],
                'complete preceding bound control source hash differs')
    for name,expected in cfg['source_pins'].items():require(freeze['control_files'][name]==expected,'actual predecessor config source roster differs')
    plan=prior['recovery_plan']
    require(sha(previous_raw['recovery_plan'])==cfg['recovery_plan']['sha256']
        and freeze['control_files'][Path(cfg['recovery_plan']['path']).name]==sha(previous_raw['recovery_plan'])
        and plan['identity']==previous and plan['next_layers']=='73-84' and plan['native_source_sha256']==PRIOR_NATIVE
        and plan['mounts']==cfg['current_mounts']==allocation['details']['mounts'],
        'whole preceding recovery plan/argv/physical mount hash joins differ')
    closed=values['closed84_continuity'];terminal=values['terminal_receipt'];scanner=values['full_ffn_validation'];manifest=values['snapshot_manifest']
    previous_binder.observation_metadata(prior['catalog'],sha(previous_raw['catalog']),prior['observation'],
        previous,cfg['current_mounts'],earlier_values['closed72_continuity'])
    require(closed['receipt_sha256']==sha(raw['terminal_receipt']) and closed['ffn_validation_sha256']==sha(raw['full_ffn_validation'])
        and closed['snapshot_manifest_sha256']==sha(raw['snapshot_manifest'])
        and closed['prior_window_continuity_sha256']==sha(earlier_raw['closed72_continuity']),
        'current raw closure/native/scanner/manifest and exact earlier closure joins differ')
    catalog_rows={row.get('recorded_logical_path') or '/run-output/'+row['relative_path']:row for row in prior['catalog']['input_catalog']}
    for row in closed['prefix'][:72]:
        logical=f"/run-output/fold/dense/L{row['layer']:02d}/"
        require(row['ffn_sha256']==catalog_rows[logical+'ffn.safetensors']['sha256']
            and row['done_sha256']==catalog_rows[logical+'done.json']['sha256'],
            'every original earlier prefix artifact SHA differs')
    old={}
    for row in closed['prefix']:
        logical=f"/run-output/fold/dense/L{row['layer']:02d}/"
        old[logical+'ffn.safetensors']={'sha256':row['ffn_sha256']};old[logical+'done.json']={'sha256':row['done_sha256']}
    for logical,row in manifest['files'].items():old[logical]={'sha256':row['sha256']}
    context=copy.deepcopy(earlier_values['closed72_continuity']['resume_context']);context['software']['tool_sha256']=PRIOR_NATIVE
    runtime=raw_ref({'path':pins['runtime']['path'],'sha256':pins['runtime']['sha256']});tree=ast.parse(runtime)
    names={'context_equal','validate_closed84_metadata'};functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(functions)==2,'exact independently reviewed pure latest prefix validator required')
    env={'require':require,'json':json};exec(compile(ast.Module(body=functions,type_ignores=[]),'<reviewed actual84 metadata validator>','exec'),env)
    env['validate_closed84_metadata'](closed,scanner,terminal,manifest,SimpleNamespace(old=old),context,sha(earlier_raw['closed72_continuity']))
    status=values['controller_terminal_result'];execution=values['fitter_execution']
    require(status['schema']=='neural_forge.k3_actual_continuation_worker_status.v1' and status['identity']==previous
        and status['stage']=='bounded_window_complete' and status['window_scope']=='73-84'
        and status['full_prefix_layers']==84 and status['later_windows_admitted'] is False
        and status['final_continuity_sha256']==sha(raw['closed84_continuity'])
        and status['config_sha256']==sha(previous_raw['driver_config'])
        and status['allocation']['allocation_id']==allocation['allocation_id'] and status['allocation']['container_id']==allocation['container_id']
        and len(status['completed_windows'])==1 and status['completed_windows'][0]['layers']=='73-84'
        and status['completed_windows'][0]['continuity_sha256']==sha(raw['closed84_continuity'])
        and status['completed_windows'][0]['terminal_receipt_sha256']==sha(raw['terminal_receipt']),
        'whole actual controller terminal and physical fully scanned prefix required')
    pid=execution['pid'];require(type(pid) is int and pid>0 and execution['phase']=='native_fitter' and execution['exit_code']==0
        and execution['ended_at'] and execution['argv']==plan['command_argv'],'derive genuine successful native Popen namespace PID/argv')
    events=[decode(line) for line in raw['controller_events_jsonl'].splitlines() if line]
    starts=[r for r in events if r['stage']=='native_fitter_started'];running=[r for r in events if r['stage']=='native_fitter_running'];ends=[r for r in events if r['stage']=='native_fitter_terminal']
    require(len(starts)==len(ends)==1 and running and starts[0]['command_argv']==execution['argv']
        and all(r['child_pid']==pid for r in running) and ends[0]['actual_command_result']==execution,
        'real native namespace PID/start/running/terminal event joins differ')
    completed=[r for r in events if r['stage']=='bounded_window_complete']
    require(len(completed)==1 and completed[0]['full_prefix_layers']==84 and completed[0]['later_windows_admitted'] is False
        and completed[0]['final_continuity_sha256']==sha(raw['closed84_continuity'])
        and completed[0]['final_continuity_path']==status['final_continuity_path']==status['completed_windows'][0]['continuity_path']
        and completed[0]['ended_at']==status['ended_at'],'single whole-controller completion event/terminal source receipt differs')
    frames=[decode(line) for line in raw['resources_jsonl'].splitlines() if line]
    sealed=[r for r in frames if r['stage']=='window_validated'];require(len(sealed)==1,'one actual after-all-readers aggregate frame required')
    peaks=[]
    for frame in [execution['resources_after'],sealed[0]]:
        cg=frame['cgroup'];require(cg['memory.max']==cfg['budget']['memory_limit_bytes'] and cg['memory.swap.max']==cg['memory.swap.current']==0
            and cg['memory.events'].get('oom_kill',0)==0,'real preceding caps/noSwap/non-OOM frame differs')
        peaks.append(cg['memory.peak'])
    peak=max(peaks);require(type(peak) is int and 0<peak<=cfg['budget']['memory_limit_bytes'],'actual full preceding cgroup peak required')
    cap=spec['operating_cap_proposal'];require(set(cap)=={'memory_bytes','CPU_threads','swap_bytes'} and type(cap['memory_bytes']) is int
        and cap['memory_bytes']>=peak+POSITIVE_HEADER_DELTA and cap['CPU_threads']==22 and cap['swap_bytes']==0,
        'actual future selected cap must cover measured predecessor plus conservative positive header delta; not a whole RSS proof')
    receipt=values['binding_receipt'];require(receipt['exit_code']==0 and receipt['response']['ok'] is True and receipt['argv'][-1]=='connect',
        'genuine current own normal supported reconnect receipt required')
    data=receipt['response']['data'];binding={k:data[k] for k in ('harness_id','thread_id','session_id','session_generation','context_sha256')}
    require(binding['harness_id']=='01a11108-009c-72f3-b2ba-e0d48fca1435' and binding['thread_id']=='95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
        and type(binding['session_generation']) is int and binding['session_generation']>=previous['native_binding']['session_generation']
        and str(uuid.UUID(binding['session_id']))==binding['session_id'] and re.fullmatch('[0-9a-f]{64}',binding['context_sha256']) is not None,
        'actual latest own context/binding lineage differs')
    parse=lambda stamp:__import__('datetime').datetime.fromisoformat(stamp.replace('Z','+00:00'))
    require(parse(status['ended_at'])<=parse(receipt['at']),'genuine current reconnect must follow whole preceding terminal')
    return previous_binder,bundle,prior,raw,values,binding,peak,pid
