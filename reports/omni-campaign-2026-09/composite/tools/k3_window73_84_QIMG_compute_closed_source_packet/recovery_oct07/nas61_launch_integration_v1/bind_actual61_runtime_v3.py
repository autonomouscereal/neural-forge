"""Local-only61-72 source binder; actual60 nine raws and previous49 source are mandatory.

No connection, payload read/copy, allocator, provider or runtime operation.
Selected eleven source templates are normal-tested D commit6700 and immutable
packet d426. Actual60 evidence, new UUID/filesystem, three-origin metadata, fresh
whole admission and normal allocation/config/freeze/caps remain mandatory.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib.util, json, os, re, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
LOCAL_OUTPUTS=Path('D:/codex-owned-source-receipts/k3-window61-72-source-only-9d98292e804f/launch_integration_v1')
RECOVERY=HERE.parent
CANDIDATE=RECOVERY/'nas61_72_integration_proposal_v1/candidate_v3'
PREVIOUS_BINDER=RECOVERY/'recovery_resource_review/actual49_60_binding_proposal_v1/bind_actual49_60_current_peer_v4.py'
PREVIOUS_BINDER_SHA='cffe9c1ef2113a05c4b066331aa44b0c886b53752b594be7383bbb25fcad1ae2'
PREVIOUS_RUN='03603908-570d-4c90-96cb-ee3a63a9a139'
PREVIOUS_NATIVE='5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'
NATIVE='01a11108-009c-72f3-b2ba-e0d48fca1435'
THREAD='95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
GIB=1<<30
CAP_PROPOSAL=104*GIB
WRITER=166556705792
ARRAY_DELTA=8220835840
COMMON_BASE=692026605568
WRITER_PATH=RECOVERY/'remaining_parent_header_accounting_v1/ACTUAL_HEADER_61_72_WRITE_AND_ARRAY_COMPONENTS_V1.json'
WRITER_SHA='d85a82c01d844daf2b89ab2b4071f5d6b4f7b616ae581e65ade55767d96ad78e'
ROLES={'closed60_continuity','terminal_receipt','full_ffn_validation','snapshot_manifest','fitter_execution',
    'controller_terminal_result','resources_jsonl','controller_events_jsonl','binding_receipt'}
PRIOR_ROLES={'source_closure','source_binding','identity','catalog','observation','driver_config','source_freeze','allocation','recovery_plan'}



PACKET_PATH=Path('D:\\codex-owned-source-receipts\\k3-window61-72-source-only-9d98292e804f\\source_promotion_v3\\immutable_packet_v1\\K3_WINDOW61_72_REVIEWED_UNBOUND_IMMUTABLE_SOURCE_PACKET_V1.zip')
PACKET_SHA='d426b2fdfacf0969fa3f63fa669d35063be0812813ebfa26304d3387b4ba8cbe'
PACKET_MANIFEST_SHA='1c82e7b6a289812ad01389165e96fec26a308faf1c0d2cb11b8aa07f343af7fb'
PROMOTED_COMMIT='6700c7dee88003983f8ef107bd1a7806863653ee'
SOURCE_PINS_SHA='8de56667a46076344393d49358c6bbd83246f9f3934acc6aead63ccddda6870f'
RUN=None
CAP_BYTES,CAP_CPUS=CAP_PROPOSAL,22
WRITER_BYTES=WRITER
COMMON_BYTES=None
NATIVE_SHA='078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'
RESOLVER_SHA='219fd84678e3cf4c17401fc367c4f803e4046432a34241cff0cdc3a859e880c4'
RUNTIME_SHA='1bd169f1c215337475fc250e2e1ae6c19a37f35bc761e0fc48566d85acf614c7'
GUARD_SHA='88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41'
SEALED36='/resume-prefix/ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b'
SEALED48_LOGICAL='/run-output/ancestry/window-37-48-764524b1-96a9-4636-be74-942f9c54754c'
SEALED60_LOGICAL='/run-output/ancestry/window-49-60-'+PREVIOUS_RUN
SEALED60_MOUNTED='/resume-window49-60/ancestry/window-49-60-'+PREVIOUS_RUN

def require(ok,message):
    if ok is not True:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def encode(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

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

def pin(path,raw):return {'path':str(path.resolve()),'name':path.name,'sha256':sha(raw),'bytes':len(raw)}

def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    require(path.read_bytes()==raw,'exclusive local byte readback differs')

def previous_binder():
    raw=raw_ref({'path':str(PREVIOUS_BINDER),'sha256':PREVIOUS_BINDER_SHA})
    spec=importlib.util.spec_from_file_location('actual49_source_binder',PREVIOUS_BINDER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def validate_actual60(spec,pins):
    require(set(spec['inputs'])==ROLES and set(spec['previous49']['inputs'])==PRIOR_ROLES,
        'exact nine current60 raws and complete actual49 provenance required')
    raw={role:raw_ref(ref) for role,ref in spec['inputs'].items()}
    previous_raw={role:raw_ref(ref) for role,ref in spec['previous49']['inputs'].items()}
    values={role:decode(value) for role,value in raw.items() if not role.endswith('_jsonl')}
    prior={role:decode(value) for role,value in previous_raw.items()}
    bundle=Path(spec['previous49']['source_bundle']).resolve()
    b=previous_binder();b.verify_bound_bundle(bundle,prior['source_closure'])
    require(prior['source_binding']==prior['source_closure']['source_binding']
        and prior['source_binding']['immutable_packet_sha256']=='f28b4ef8acf58d89eca2956baf27b1a7bbb1390c54fa7a68d6a22f435485cc43',
        'authentic selected49 source-binding provenance differs')
    i=prior['identity'];cfg=prior['driver_config'];freeze=prior['source_freeze'];allocation=prior['allocation']
    require(i['run_uuid']==PREVIOUS_RUN and i['new_layers']==[49,60] and cfg['identity']==i
        and cfg['schema']=='neural_forge.k3_nas_closed48_worker_config.v1' and cfg['window']==[49,60]
        and cfg['native_tool_sha256']==PREVIOUS_NATIVE and cfg['budget']['cpus']==22,
        'actual selected49 native/identity/config required')
    require(sha(previous_raw['identity'])==prior['source_closure']['sources']['identity']['sha256']
        and sha(previous_raw['catalog'])==cfg['resume_source_manifest']['sha256']
        and sha(previous_raw['observation'])==cfg['resume_source_observation']['sha256'],
        'actual49 identity/catalog/observation byte joins differ')
    require(freeze['identity']==i and freeze['native_source_sha256']==PREVIOUS_NATIVE
        and freeze['config_sha256']==sha(previous_raw['driver_config'])
        and freeze['allocation_id']==allocation['allocation_id'] and freeze['container_id']==allocation['container_id'],
        'authentic actual49 source freeze/allocation joins differ')
    require(freeze['schema']=='neural_forge.k3_NAS49_60_execution_source_freeze.v1'
        and freeze['immutable_source_ro_mounts'] is True and freeze['old_source_unchanged'] is True,
        'genuine previous49 immutable source freeze required')
    expected_code={name[len('production_code/'):]:row['sha256']
        for name,row in prior['source_binding']['production_file_pins'].items() if name.startswith('production_code/')}
    require(freeze['code_files']==expected_code,'whole authentic previous49 execution code roster differs')
    for name,row in prior['source_binding']['production_file_pins'].items():
        if name.startswith('production_control/'):
            require(freeze['control_files'][name[len('production_control/'):]]==row['sha256'],
                'whole actual prior49 bound control bytes differ from frozen execution')
    for name,expected in cfg['source_pins'].items():require(freeze['control_files'][name]==expected,'previous49 complete source/config pin differs')
    plan=prior['recovery_plan']
    require(sha(previous_raw['recovery_plan'])==cfg['recovery_plan']['sha256']
        and freeze['control_files'][Path(cfg['recovery_plan']['path']).name]==sha(previous_raw['recovery_plan'])
        and plan['identity']==i and plan['next_layers']=='49-60' and plan['native_source_sha256']==PREVIOUS_NATIVE
        and plan['mounts']==cfg['current_mounts']==allocation['details']['mounts'],
        'actual prior49 plan/argv/source/physical mount joins differ')
    closed=values['closed60_continuity'];terminal=values['terminal_receipt'];scanner=values['full_ffn_validation'];manifest=values['snapshot_manifest']
    old={}
    for row in closed['prefix']:
        logical=f"/run-output/fold/dense/L{row['layer']:02d}/"
        old[logical+'ffn.safetensors']={'sha256':row['ffn_sha256']};old[logical+'done.json']={'sha256':row['done_sha256']}
    for logical,row in manifest['files'].items():old[logical]={'sha256':row['sha256']}
    closed48=decode(raw_ref({'path':str(bundle/'production_control'/prior['source_closure']['sources']['closed48_continuity']['name']),
        'sha256':i['closed48_continuity_sha256']}))
    b.observation_metadata(prior['catalog'],sha(previous_raw['catalog']),prior['observation'],i,cfg['current_mounts'],closed48)
    require(closed['receipt_sha256']==sha(raw['terminal_receipt'])
        and closed['ffn_validation_sha256']==sha(raw['full_ffn_validation'])
        and closed['snapshot_manifest_sha256']==sha(raw['snapshot_manifest']),
        'actual complete closed60 raw receipt/scanner/manifest pins differ')
    catalog_rows={row.get('recorded_logical_path') or '/run-output/'+row['relative_path']:row for row in prior['catalog']['input_catalog']}
    for row in closed['prefix'][:48]:
        logical=f"/run-output/fold/dense/L{row['layer']:02d}/"
        require(row['ffn_sha256']==catalog_rows[logical+'ffn.safetensors']['sha256']
            and row['done_sha256']==catalog_rows[logical+'done.json']['sha256'],'actual inherited48 prefix bytes differ')
    context=copy.deepcopy(closed48['resume_context'])
    context['software']['tool_sha256']=PREVIOUS_NATIVE
    runtime=raw_ref({'path':pins['runtime']['path'],'sha256':pins['runtime']['sha256']})
    tree=ast.parse(runtime);names={'context_equal','validate_closed60_metadata'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    require(len(functions)==2,'exact reviewed pure closed60 validator required')
    env={'require':require,'json':json}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<reviewed actual60 metadata validator>','exec'),env)
    env['validate_closed60_metadata'](closed,scanner,terminal,manifest,SimpleNamespace(old=old),context,i['closed48_continuity_sha256'])
    status=values['controller_terminal_result'];execution=values['fitter_execution']
    require(status['schema']=='neural_forge.k3_actual_continuation_worker_status.v1' and status['identity']==i
        and status['stage']=='bounded_window_complete' and status['window_scope']=='49-60'
        and status['full_prefix_layers']==60 and status['later_windows_admitted'] is False
        and status['final_continuity_sha256']==sha(raw['closed60_continuity'])
        and status['config_sha256']==sha(previous_raw['driver_config']) and status['allocation']['allocation_id']==allocation['allocation_id']
        and status['allocation']['container_id']==allocation['container_id'] and len(status['completed_windows'])==1
        and status['completed_windows'][0]['layers']=='49-60'
        and status['completed_windows'][0]['continuity_sha256']==sha(raw['closed60_continuity'])
        and status['completed_windows'][0]['terminal_receipt_sha256']==sha(raw['terminal_receipt']),
        'authentic49-60 whole controller terminal closure required')
    namespace_pid=execution['pid']
    require(type(namespace_pid) is int and namespace_pid>0 and execution['phase']=='native_fitter'
        and execution['exit_code']==0 and execution['ended_at'] and execution['argv']==plan['command_argv'],'actual successful Popen namespace PID/end required')
    events=[decode(line) for line in raw['controller_events_jsonl'].splitlines() if line]
    starts=[r for r in events if r['stage']=='native_fitter_started'];running=[r for r in events if r['stage']=='native_fitter_running'];ends=[r for r in events if r['stage']=='native_fitter_terminal']
    require(len(starts)==len(ends)==1 and running and starts[0]['command_argv']==execution['argv']
        and all(r['child_pid']==namespace_pid for r in running) and ends[0]['actual_command_result']==execution,
        'actual execution namespace PID/argv/running/terminal event joins differ')
    completed=[r for r in events if r['stage']=='bounded_window_complete']
    require(len(completed)==1 and completed[0]['full_prefix_layers']==60
        and completed[0]['later_windows_admitted'] is False
        and completed[0]['final_continuity_sha256']==sha(raw['closed60_continuity'])
        and completed[0]['final_continuity_path']==status['final_continuity_path']
        ==status['completed_windows'][0]['continuity_path']
        and completed[0]['ended_at']==status['ended_at'],
        'actual whole-controller bounded completion event/result joins differ')
    resources=[decode(line) for line in raw['resources_jsonl'].splitlines() if line]
    frames=[r for r in resources if r['stage']=='window_validated']
    require(len(frames)==1,'actual after-all-readers sealed60 aggregate frame required')
    peaks=[]
    for frame in [execution['resources_after'],frames[0]]:
        cg=frame['cgroup'];require(cg['memory.max']==cfg['budget']['memory_limit_bytes']
            and cg['memory.swap.max']==cg['memory.swap.current']==0 and cg['memory.events'].get('oom_kill',0)==0,
            'actual prior operating cap/noSwap/non-OOM frames required')
        peaks.append(cg['memory.peak'])
    peak=max(peaks)
    require(type(peak) is int and 0<peak<=cfg['budget']['memory_limit_bytes'] and peak+ARRAY_DELTA<=CAP_PROPOSAL,
        'actual final60 peak plus qualified next array delta exceeds104GiB operating proposal')
    receipt=values['binding_receipt'];require(receipt['exit_code']==0 and receipt['response']['ok'] is True
        and receipt['argv'][-1]=='connect','actual normal current native connect receipt required')
    data=receipt['response']['data'];binding={k:data[k] for k in ('harness_id','thread_id','session_id','session_generation','context_sha256')}
    require(binding['harness_id']==NATIVE and binding['thread_id']==THREAD and type(binding['session_generation']) is int
        and binding['session_generation']>=i['native_binding']['session_generation']
        and str(uuid.UUID(binding['session_id']))==binding['session_id']
        and re.fullmatch('[0-9a-f]{64}',binding['context_sha256']) is not None,'genuine future own native context required')
    return b,bundle,prior,raw,values,binding,peak,namespace_pid

def read_raw(path,expected):return raw_ref({'path':str(path),'sha256':expected})
def read_ref(ref):
    raw=raw_ref(ref);return decode(raw),raw
def full_sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'full SHA required');return value
def leaf(row):return {key:row[key] for key in ('name','sha256')}
def write_raw(path,raw):
    write(path,raw);return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}

def selected():
    packet_raw=PACKET_PATH.read_bytes();require(sha(packet_raw)==PACKET_SHA,'exact promoted D packet differs')
    manifest=decode(raw_ref({'path':str(PACKET_PATH.parent/'IMMUTABLE_PACKET_MANIFEST_V1.json'),'sha256':PACKET_MANIFEST_SHA}))
    require(manifest['normal_tested_promoted_commit']==PROMOTED_COMMIT and manifest['source_only'] is True
        and manifest['git_source_member_count']==30 and manifest['selected_source_count']==11,'normal-tested exact source template packet required')
    with zipfile.ZipFile(PACKET_PATH) as archive:
        require(len(archive.infolist())==32 and set(archive.namelist())==set(manifest['members']),'whole exact32 source packet required')
        for name,row in manifest['members'].items():
            raw=archive.read(name);require(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'full promoted source packet member differs')
    pins=decode(raw_ref({'path':str(HERE/'SELECTED61_PACKET_SOURCE_PINS_V1.json'),'sha256':SOURCE_PINS_SHA}))
    require(len(pins)==11,'all eleven promoted source roles required')
    for row in pins.values():
        raw=raw_ref({'path':row['path'],'sha256':row['sha256']});require(len(raw)==row['bytes'],'selected promoted source byte extent differs')
    return pins

def bind(args):
    spec=decode(raw_ref({'path':str(args.inputs),'sha256':args.inputs_sha256}))
    require(spec['schema']=='neural_forge.k3_actual61_72_local_source_binding_inputs.v1','exact future binder input schema required')
    run=spec['run_uuid'];require(isinstance(run,str) and str(uuid.UUID(run))==run and run!=PREVIOUS_RUN,'new actual61 UUID remains unresolved')
    storage={'NAS':'/mnt/ai_data','Lexar':'/mnt/lexar'}.get(spec['new_output_filesystem'])
    require(storage is not None,'fresh NAS/Lexar output selection remains unresolved')
    require(spec['operating_cap_proposal']=={'memory_bytes':CAP_PROPOSAL,'CPU_threads':22,'swap_bytes':0},'exact conditional104GiB/22/noSwap proposal required')
    pins=selected();b,bundle,prior,raw,values,binding,peak,pid=validate_actual60(spec,pins)
    out=args.output_directory.resolve();require(out.parent==LOCAL_OUTPUTS and not out.exists() and not out.is_symlink(),'exclusive own local future output required')
    i=copy.deepcopy(prior['identity']);prefix='k3-nas-window61-72-oct07-'+run
    i.update(run_uuid=run,recovery_uuid=run,name='k3-nas61-72-'+run.replace('-','')[:16],new_layers=[61,72],
        runtime_root='/home/cereal/omni-campaign/runtime/'+prefix,
        output_root='/home/cereal/omni-campaign/outputs/'+prefix+'/control-output',
        controller_output_root='/home/cereal/omni-campaign/outputs/'+prefix+'/control-output',
        native_working_output_root=storage+'/omni-campaign/'+prefix+'/model',new_output_filesystem=spec['new_output_filesystem'],
        closed60_continuity_sha256=sha(raw['closed60_continuity']),parent_catalog_sha256=sha(raw_ref(spec['previous49']['inputs']['catalog'])),
        parent_catalog_row_count=len(prior['catalog']['input_catalog']),
        parent_catalog_rows_sha256=sha(json.dumps(prior['catalog']['input_catalog'],sort_keys=True,separators=(',',':')).encode()),
        native_binding=binding,binding_receipt=spec['inputs']['binding_receipt']['path'],actual_runtime_admitted=False,
        identity_role='separate61-72 source proposal; no actual new allocation/caps/admission claimed')
    i['readonly_resume_source_roots']['closed49_60']=prior['identity']['native_working_output_root']
    identity_raw=encode(i);binding_raw=encode(binding)
    proof={'schema':'neural_forge.k3_completed49_source_to61_binding.v1','previous_run_uuid':PREVIOUS_RUN,
        'closed60_continuity_sha256':sha(raw['closed60_continuity']),'actual_namespace_pid':pid,
        'actual60_final_aggregate_peak_bytes':peak,'qualified_next_array_delta_bytes_not_RSS':ARRAY_DELTA,
        'next_operating_cap_proposal_bytes':CAP_PROPOSAL,'whole_RSS_upper_bound_claimed':False,
        'previous49_complete_provenance':spec['previous49'],'actual60_nine_role_pins':spec['inputs'],
        'source_candidate_only_not_normal_promoted':False,'normal_template_promoted_commit':PROMOTED_COMMIT,'normal_template_packet_sha256':PACKET_SHA,'fresh_current_peer_and_FS_admission_required':True}
    proof_raw=encode(proof);writer_raw=raw_ref({'path':str(WRITER_PATH),'sha256':WRITER_SHA})
    require(decode(writer_raw)['writer']['conservative_complete_source_scope_extent_bytes']==WRITER,
        'actual header-qualified whole61 writer bound differs')
    closure=copy.deepcopy(prior['source_closure']);roster=closure['sources']
    roles_raw={role:raw_ref({'path':str(bundle/'production_control'/row['name']),'sha256':row['sha256']}) for role,row in roster.items()}
    roles_raw.update(identity=identity_raw,binding=binding_raw,closed60_continuity=raw['closed60_continuity'],
        own61_72_writer_extent=writer_raw,completed49_phase_source=proof_raw)
    roster['closed60_continuity']={'name':'closed60_continuity.json'}
    roster['own61_72_writer_extent']={'name':'ACTUAL_HEADER_61_72_WRITE_AND_ARRAY_COMPONENTS_V1.json'}
    roster['completed49_phase_source']={'name':'COMPLETED49_SOURCE_TO61_BINDING_V1.json'}
    scope=decode(roles_raw['NAS_current_scope']);scope.update(schema='neural_forge.k3_61_72.source_qualified_resource_scope.v1',
        run_uuid=run,identity_sha256=sha(identity_raw),layers=[61,72],new_output_filesystem=spec['new_output_filesystem'],
        NAS_current_common_guard_bytes_once=COMMON_BASE+(WRITER if storage=='/mnt/ai_data' else 0),
        NAS_common_guard_current_K3_writer_bytes_once=WRITER if storage=='/mnt/ai_data' else 0,
        selected_output_writer_bytes_once=WRITER,completed49_source_proof_sha256=sha(proof_raw),
        source_proposal_only=False,fresh_whole_current_FS_and_peer_admission_required=True)
    for role in ('identity','binding','closed60_continuity','own61_72_writer_extent','completed49_phase_source'):
        scope['source_pins'][role]={'name':roster[role]['name'],'sha256':sha(roles_raw[role])}
    roles_raw['NAS_current_scope']=encode(scope);roster['NAS_current_scope']['name']='BOUND61_72_RESOURCE_SCOPE_SOURCE_PROPOSAL_V1.json'
    common={'EXPECTED_RUN':run,'IDENTITY_SHA':sha(identity_raw),'BINDING_SHA':sha(binding_raw),
        'CLOSED60_SHA':sha(raw['closed60_continuity']),'FUTURE_SESSION_ID':binding['session_id'],
        'FUTURE_GENERATION':binding['session_generation'],'FUTURE_CONTEXT_SHA':binding['context_sha256'],
        'OWN_RAM_BYTES':CAP_PROPOSAL,'OWN_CPU_THREADS':22,'OUTPUT_STORAGE_ROOT':storage}
    generated={}
    for role in ('facts','host'):
        generated[role]=b.bind_assignments(raw_ref({'path':pins[role]['path'],'sha256':pins[role]['sha256']}),common)
    generated['preflight']=b.bind_assignments(raw_ref({'path':pins['preflight']['path'],'sha256':pins['preflight']['sha256']}),
        dict(common,FACTS_SHA=sha(generated['facts']),CURRENT_SCOPE_SHA=sha(roles_raw['NAS_current_scope'])))
    generated['driver']=b.bind_assignments(raw_ref({'path':pins['driver']['path'],'sha256':pins['driver']['sha256']}),
        {'RUN':run,'IDENTITY_SHA':sha(identity_raw),'CAP_BYTES':CAP_PROPOSAL,'CAP_CPUS':22,
            'SESSION_GENERATION':binding['session_generation'],'SESSION_ID':binding['session_id'],'OUTPUT_STORAGE_ROOT':storage})
    generated['collector']=b.bind_assignments(raw_ref({'path':pins['collector']['path'],'sha256':pins['collector']['sha256']}),{'RUN':run})
    generated['qualification']=b.bind_assignments(raw_ref({'path':pins['qualification']['path'],'sha256':pins['qualification']['sha256']}),{'EXPECTED_RUN':run})
    files={name:raw_ref({'path':str(bundle/name),'sha256':row['sha256']}) for name,row in prior['source_binding']['production_file_pins'].items()}
    tool='reports/omni-campaign-2026-09/composite/tools/'
    for role,name in [('native','k3_dense_pass.py'),('resolver','ro_resume_paths.py'),('runtime','ro_resume_runtime.py'),('ancestry','k3_recovery_ancestry.py')]:
        content=raw_ref({'path':pins[role]['path'],'sha256':pins[role]['sha256']})
        files['production_code/'+tool+name]=content
        if role!='native':files['production_control/'+name]=content
    files['production_control/verify_later_window.py']=raw_ref({'path':pins['verifier']['path'],'sha256':pins['verifier']['sha256']})
    mapping={'facts':'k3_owned_readonly_facts_NAS61_72_v1.py','host':'k3_capacity_host_loop_v1.py',
        'preflight':'k3_owned_capacity_preflight_NAS61_72_v1.py','driver':'continuation_worker.py','qualification':'native_nas_io_qualification.py'}
    for role,name in mapping.items():files['production_control/'+name]=generated[role]
    files['production_control/identity.json']=identity_raw
    observer=files['production_control/observe_native.py']
    for role,old in [('native',PREVIOUS_NATIVE),
        ('resolver','be4f3aede6f94d242cad9c4037656c3de3eff8f2268f5f2b17d23cf69025a804'),
        ('runtime','4e58891969a6e83b6cf2247b8837edeb53d7a37d03de2aba07a4a14e7b150e6b')]:
        require(observer.count(old.encode())==1,'exact dated observer pin slot required')
        observer=observer.replace(old.encode(),pins[role]['sha256'].encode())
    files['production_control/observe_native.py']=observer
    boot=files['production_control/recovery_boot.py']
    require(boot.count(PREVIOUS_RUN.encode())==1,'exact prior boot UUID slot required')
    files['production_control/recovery_boot.py']=boot.replace(PREVIOUS_RUN.encode(),run.encode())
    files['production_control/source_revisions/k3_dense_pass_5e.py']=raw_ref({'path':str(bundle/'production_code'/tool/'k3_dense_pass.py'),'sha256':PREVIOUS_NATIVE})
    for role,content in roles_raw.items():
        name=roster[role]['name'];files['production_control/'+name]=content;roster[role]=pin(out/'production_control'/name,content)
    scripts=closure['scripts']
    for role,own_role in [('facts','facts'),('preflight','preflight'),('host_caller','host')]:
        name=mapping[own_role];scripts[role]=pin(out/'production_control'/name,generated[own_role])
    for role in ('service','math'):
        row=scripts[role];scripts[role]=pin(out/'production_control'/row['name'],files['production_control/'+row['name']])
    mounts=copy.deepcopy(prior['driver_config']['current_mounts'])
    changes={'/code/neural-forge':i['runtime_root']+'/production_code','/control':i['runtime_root']+'/production_control',
        '/run-control-output':i['output_root'],'/run-output':i['native_working_output_root']}
    for row in mounts:
        if row['target'] in changes:row['source']=changes[row['target']]
    mounts.append({'source':prior['identity']['native_working_output_root'],'target':'/resume-window49-60','read_only':True});mounts.sort(key=lambda row:row['target'])
    require(len(mounts)==11 and len({row['target'] for row in mounts})==11,'exact eleven disjoint three-origin mount rows required')
    require(set(roster)==set(prior['source_closure']['sources'])|{'closed60_continuity','own61_72_writer_extent','completed49_phase_source'}
        and len(roster)==66 and len(scripts)==5,'exact previous63 plus three new phase roles/five scripts required')
    for name,content in files.items():
        if name.endswith('.py'):compile(content,name,'exec')
    closure.update(selected_run_uuid=run,actual_runtime_admission_claimed=False,source_proposal_only=False,
        source_binding={'schema':'neural_forge.k3_actual61_72_local_source_binding_proposal.v1','identity_sha256':sha(identity_raw),
            'closed60_sha256':sha(raw['closed60_continuity']),'actual60_nine_role_pins':spec['inputs'],
            'previous49_provenance':spec['previous49'],'current_candidate_source_pins':pins,
            'actual_namespace_pid':pid,'actual60_final_peak_bytes':peak,'array_delta_bytes_not_RSS':ARRAY_DELTA,
            'operating_cap_proposal_bytes':CAP_PROPOSAL,'writer_bytes_once':WRITER,'chosen_output_filesystem':spec['new_output_filesystem'],
            'source_proposal_only_not_deployable':False,'fresh_peer_source_and_FS_capacity_admission_required':True,
            'production_file_pins':{name:{'sha256':sha(content),'bytes':len(content)} for name,content in files.items()}})
    closure['source_binding'].update(mounts_sha256=sha(encode(mounts)),normal_template_promoted_commit=PROMOTED_COMMIT,immutable_template_packet_sha256=PACKET_SHA,common_NAS_floor_compare_bytes=COMMON_BASE+(WRITER if storage=='/mnt/ai_data' else 0),observer_source_pin=pin(out/'observer_source/collect_actual61_72_multi_origin.py',generated['collector']))
    # The actual three-origin observer, plan, configs, A/CID/physicalcaps and
    # normal source publication are still required. No guessed placeholders
    # are turned into an executable config or barrier by this local binder.
    extras={'SOURCE_BINDING_RESULT.json':encode(closure['source_binding']),'SOURCE_CLOSURE61_72.json':encode(closure),'mounts.json':encode(mounts),
        'observer_source/collect_actual61_72_multi_origin.py':generated['collector'],
        'NEXT_BINDING_UNRESOLVED.json':encode({'three_origin_catalog':None,'three_origin_observation':None,
            'actual_new_allocation':None,'new_config':None,'native_runtime_observation':None,'fresh_capacity':None,
            'source_promoted_commit':None,'new_immutable_packet':None,'can_arm':False})}
    out.mkdir(exist_ok=False)
    for name,content in sorted({**files,**extras}.items()):write(out/name,content)
    print(json.dumps({'path':str(out),'source_closure_sha256':sha(extras['SOURCE_CLOSURE61_72.json']),
        'metadata_count':len(roster),'script_count':len(scripts),'normal_template_promoted_commit':PROMOTED_COMMIT,'actual_runtime_admitted':False,'runtime_actions':False}))
def verify_bound_bundle(bundle, closure):
    require(closure['schema'] == 'neural_forge.k3_owned_capacity_source_closure.v15'
            and closure['selected_run_uuid'] == RUN and len(closure['sources']) == 66
            and len(closure['scripts']) == 5, 'complete source-bound66 metadata/five-script closure required')
    peer = closure['sources'].get('current_peer_holder_source')
    require(isinstance(peer, dict) and peer.get('name') == 'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json'
            and peer.get('sha256') == '1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9'
            and peer.get('bytes') == 1797866, 'complete exact current-peer proof role required')
    read_raw(bundle/'mounts.json', closure['source_binding']['mounts_sha256'])
    expected = closure['source_binding']['production_file_pins']
    actual = {}
    for root in ('production_code', 'production_control'):
        for path in (bundle/root).rglob('*'):
            require(not path.is_symlink(), 'bound production source symlink refused')
            if path.is_file(): actual[path.relative_to(bundle).as_posix()] = path
    require(set(actual) == set(expected), 'whole source-bound production file roster changed')
    for name, path in actual.items():
        raw = read_raw(path, expected[name]['sha256'])
        require(len(raw) == expected[name]['bytes'], 'source-bound production file extent changed')
    require(closure['source_binding']['normal_template_promoted_commit']==PROMOTED_COMMIT and closure['source_binding']['immutable_template_packet_sha256']==PACKET_SHA,'actual normal-tested source template provenance required')

def observation_metadata(catalog,catalog_sha,observation,identity,mounts,closed):
    require(catalog['schema']=='chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
        and observation['schema']=='chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
        and observation['source_catalog_sha256']==catalog_sha
        and observation['first_new_layer']==61 and observation['last_new_layer']==72
        and observation['current_mounts']==mounts,'actual three-origin catalog/mount metadata required')
    require(observation['acting_identity']=={'owner':identity['owner'],'native_context':identity['native_thread_id'],
        'run_uuid':RUN,'image':identity['image'],'control_binding':identity['native_binding']},'actual acting identity differs')
    roots=observation['source_roots'];aliases={'legacy_closed36':'/resume-prefix','closed37_48':'/resume-window37-48','closed49_60':'/resume-window49-60'}
    require(set(roots)==set(aliases)==set(identity['readonly_resume_source_roots']),'exact three RO origins required')
    for role,alias in aliases.items():
        require(roots[role]['path']==roots[role]['alias']==alias and roots[role]['read_only'] is True
            and roots[role]['host_path']==identity['readonly_resume_source_roots'][role],'exact per-origin host/alias/RO mapping differs')
    out=observation['output_root'];require(out['path']==out['alias']=='/run-output' and out['read_only'] is False
        and out['host_path']==identity['native_working_output_root'],'exclusive new chosen-filesystem output differs')
    all_roots=[*roots.values(),out]
    for idx,left in enumerate(all_roots):
        for right in all_roots[idx+1:]:
            for key in ('path','host_path'):
                a,b=PurePosixPath(left[key]),PurePosixPath(right[key])
                require(a!=b and not a.is_relative_to(b) and not b.is_relative_to(a),'disjoint RO/RW roots required')
            require((left['stat']['dev'],left['stat']['ino'])!=(right['stat']['dev'],right['stat']['ino']),'distinct physical roots required')
    parent_count=identity['parent_catalog_row_count'];rows=catalog['input_catalog'];observed=observation['source_files']
    require(type(parent_count) is int and parent_count>127 and len(rows)>parent_count
        and catalog['parent_catalog_sha256']==observation['parent_catalog_sha256']==identity['parent_catalog_sha256']
        and observation['parent_catalog_row_count']==parent_count
        and observation['parent_catalog_rows_sha256']==identity['parent_catalog_rows_sha256']
        and sha(json.dumps(rows[:parent_count],sort_keys=True,separators=(',',':')).encode())==identity['parent_catalog_rows_sha256'],
        'unchanged entire actual49 source catalog and complete new49-60 rows required')
    require(observation['closed60_continuity']=={'path':SEALED60_LOGICAL+'/window_continuity.json','sha256':identity['closed60_continuity_sha256']}
        and observation['closed48_continuity']=={'path':SEALED48_LOGICAL+'/window_continuity.json','sha256':identity['closed48_continuity_sha256']},
        'actual latest sealed60 and preserved closed48 joins required')
    old={}
    for idx,row in enumerate(rows):
        logical=row.get('recorded_logical_path') or '/run-output/'+row['relative_path']
        rel=PurePosixPath(logical).relative_to('/run-output');origin=row['source_origin'];root=roots[origin]
        require((origin=='closed49_60')==(idx>=parent_count) and logical not in old
            and row['prospective_RO_input_path']==root['alias']+'/'+str(rel)
            and row['unchanged_host_input_path']==str(PurePosixPath(root['host_path'])/rel),'exact unique source-origin row mapping required')
        fact=observed[logical]
        require(fact['source_origin']==origin and fact['source_sha256']==full_sha(row['sha256'])
            and fact['path']==root['alias']+'/'+str(rel) and fact['stat']['dev']==root['stat']['dev']
            and ('bytes' not in row or fact['stat']['size']==row['bytes']),'actual expectedSHA/stat/extent join differs')
        old[logical]=row
    require(set(old)==set(observed),'complete observed input roster required')
    for row in closed['prefix']:
        logical=f"/run-output/fold/dense/L{row['layer']:02d}/"
        require(old[logical+'ffn.safetensors']['sha256']==row['ffn_sha256'] and old[logical+'done.json']['sha256']==row['done_sha256'],
            'actual full60 prefix/catalog join differs')
    require(all(f'/run-output/fold/resid/L{n:02d}/held_rows.safetensors' in old for n in range(37,61)),
        'all24 authentic held-row source extents required')
    require(observation['source_pins']=={'ro_resume_paths.py':RESOLVER_SHA,'ro_resume_runtime.py':RUNTIME_SHA,
        'k3_recovery_ancestry.py':GUARD_SHA,'k3_dense_pass.py':NATIVE_SHA},'selected exact source pins differ')
    require(observation['previous_recovery_packet']=={'path':'/control/source_revisions/recovery_ancestry_capacity_wait.json','sha256':'5b9559b8ac20428de7e4ac09201f71839f4d03113a985715ca81d3aedd01246c'}
        and observation['previous_native_source']=={'path':'/control/source_revisions/k3_dense_pass_9086.py','sha256':'9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69'}
        and observation['completed_native_source']=={'path':'/control/source_revisions/k3_dense_pass_720.py','sha256':'720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'}
        and observation['completed49_native_source']=={'path':'/control/source_revisions/k3_dense_pass_5e.py','sha256':PREVIOUS_NATIVE},
        'authentic historical native/context source lineage required')
    require(observation['new_write_paths'] and len(set(observation['new_write_paths']))==len(observation['new_write_paths'])
        and not(set(observation['new_write_paths'])&set(old)),'finite disjoint new61-72 writes required')

def plans_bind(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual61_72_local_plan_binding_inputs.v1'
            and spec['run_uuid'] == RUN and set(spec['inputs']) == {'catalog', 'observation'},
            'actual metadata-only three-origin plan input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == LOCAL_OUTPUTS and bundle.is_dir() and not bundle.is_symlink(), 'own bound source bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE61_72.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    control = bundle/'production_control'
    identity = decode(read_raw(control/'identity.json', closure['sources']['identity']['sha256']))
    closed = decode(read_raw(control/closure['sources']['closed60_continuity']['name'], identity['closed60_continuity_sha256']))
    mounts = decode((bundle/'mounts.json').read_bytes())
    catalog, catalog_raw = read_ref(spec['inputs']['catalog'])
    observation, observation_raw = read_ref(spec['inputs']['observation'])
    observation_metadata(catalog, sha(catalog_raw), observation, identity, mounts, closed)
    completed=decode(read_raw(control/closure['sources']['completed49_phase_source']['name'],closure['sources']['completed49_phase_source']['sha256']))
    prior=decode(raw_ref(completed['previous49_complete_provenance']['inputs']['recovery_plan']))
    # Use the reviewed Driver's exact pure original-argv transformer. Old
    # assertions remain inside dated_original_plan_provenance only.
    driver_raw = (control/'continuation_worker.py').read_bytes()
    tree = ast.parse(driver_raw)
    transform = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'operational_argv']
    require(len(transform) == 1, 'exact selected pure operational argv function required')
    env = {'require': require}
    exec(compile(ast.Module(body=transform, type_ignores=[]), '<reviewed original argv transform>', 'exec'), env)
    argv = env['operational_argv'](prior['dated_original_plan_provenance']['plan']['command_argv'], '61-72')
    require(argv[:4] == ['env', 'OMP_NUM_THREADS=22', 'MKL_NUM_THREADS=22', 'OPENBLAS_NUM_THREADS=22'],
            'source selected original CPU22 environment required')
    refs = {'manifest': {'path': '/control/ro_multi_origin_catalog.json', 'sha256': sha(catalog_raw)},
            'observation': {'path': '/control/ro_multi_origin_observation.json', 'sha256': sha(observation_raw)},
            'identity': {'path': '/control/identity.json', 'sha256': closure['sources']['identity']['sha256']}}
    flags = []
    for role in ('manifest', 'observation', 'identity'):
        flags += ['--resume-source-'+role, refs[role]['path'], '--resume-source-'+role+'-sha256', refs[role]['sha256']]
    flags += ['--resume-source-root', '/resume-prefix']
    plan = {'schema': 'neural_forge.k3_actual_NAS_closed60_record_plan.v1', 'identity': identity,
            'command_argv': argv+flags, 'next_layers': '61-72', 'mounts': mounts, 'image_id': identity['image'],
            'required_gpu_uuid': identity['gpu_uuid'], 'native_source_sha256': NATIVE_SHA,
            'closed60_continuity_sha256': identity['closed60_continuity_sha256'],
            'original_record_plan_sha256': prior['original_record_plan_sha256'],
            'dated_original_plan_provenance': prior['dated_original_plan_provenance'],
            'RO_resume_source_refs': refs, 'RO_resume_source_flags': flags,
            'current_native_software_and_inputs_require_actual_runtime_observation': True,
            'actual_runtime_admitted': False, 'actual_fitter_started': False, 'later_windows_admitted': False,
            'capability_accepted': False, 'training_eligible': False, 'numerical_targets_enforced': False,
            'artifact_scope':'unique chosen-filesystem61-72 outputs; three exact old originsRO',
            'created_utc': datetime.now(timezone.utc).isoformat()}
    plan_raw = encode(plan)
    runtime = {'schema': 'neural_forge.k3_owned_NAS61_72_immutable_runtime_plan.v1', 'identity': identity,
               'plan_path': '/control/recovery_plan61_72.json', 'plan_sha256': sha(plan_raw), 'current_mounts': mounts,
               'source_pins': {p.relative_to(control).as_posix(): sha(p.read_bytes()) for p in control.rglob('*') if p.is_file()},
               'budget': {'memory_bytes': CAP_BYTES, 'cpus': CAP_CPUS, 'swap_bytes': 0, 'extra_GPUs': 0,
                          'selected_new_output_write_bytes': WRITER_BYTES, 'selected_new_NAS_write_bytes': WRITER_BYTES if identity['new_output_filesystem']=='NAS' else 0, 'NAS_qualification_inside_writer_bytes': 536870912 if identity['new_output_filesystem']=='NAS' else 0, 'selected_output_IO_qualification_inside_writer_bytes': 536870912,
                          'ROOT_metadata_bound_bytes': 2*GIB, 'serial_NAS_model_writers': 1 if identity['new_output_filesystem']=='NAS' else 0, 'serial_selected_output_model_writers': 1},
               'NAS_common_floor_comparison_bytes': COMMON_BYTES, 'inherited_common_padding_is_not_new_phase_jobs': True,
               'old_inputs_copied_bytes': 0, 'no_capability_claim': True, 'no_later_window_admission': True,
               'fresh_actual_native_runtime_observation_required_before_fit': True,
               'source_pinned_current_admission_required_before_allocation': True,
               'restore_original_exact_provider_configuration': True}
    out = args.output_directory.resolve()
    require(out.parent == LOCAL_OUTPUTS and not out.exists() and not out.is_symlink(), 'exclusive own plan output leaf required')
    raw_outputs = {'recovery_plan61_72.json': plan_raw, 'runtime_plan61_72.json': encode(runtime),
                   'ro_multi_origin_catalog.json': catalog_raw, 'ro_multi_origin_observation.json': observation_raw}
    out.mkdir(exist_ok=False)
    result = {name: write_raw(out/name, raw) for name, raw in raw_outputs.items()}
    print(json.dumps({'run_uuid': RUN, 'files': result, 'actual_runtime_admitted': False}, sort_keys=True))
def final_configs(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual61_72_local_final_config_inputs.v1'
            and spec['run_uuid'] == RUN, 'exact post-allocation config input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == LOCAL_OUTPUTS and bundle.is_dir() and not bundle.is_symlink(), 'own source-bound local bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE61_72.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    require(closure['selected_run_uuid'] == RUN and len(closure['sources']) == 66 and len(closure['scripts']) == 5,
            'whole exact66 metadata/five-script source closure required')
    control = bundle/'production_control'; source_pins = {}
    for file in control.rglob('*'):
        require(not file.is_symlink(), 'source bundle symlink refused')
        if file.is_file(): source_pins[file.relative_to(control).as_posix()] = sha(file.read_bytes())
    for roster in (closure['sources'], closure['scripts']):
        for row in roster.values(): read_raw(control/row['name'], row['sha256'])
    identity = decode((control/'identity.json').read_bytes())
    require(sha((control/'identity.json').read_bytes()) == closure['sources']['identity']['sha256'],
            'byte-identical mounted and capacity identity required')
    required = {'allocation', 'recovery_plan', 'runtime_plan', 'catalog', 'observation'}
    require(set(spec['inputs']) == required, 'complete exact post-allocation current plan/observation inputs required')
    values, raws = {}, {}
    for role, ref in spec['inputs'].items(): values[role], raws[role] = read_ref(ref)
    allocation, plan, runtime = values['allocation'], values['recovery_plan'], values['runtime_plan']
    catalog, observation = values['catalog'], values['observation']
    mounts = decode((bundle/'mounts.json').read_bytes())
    closed = decode(read_raw(control/closure['sources']['closed60_continuity']['name'], identity['closed60_continuity_sha256']))
    observation_metadata(catalog, sha(raws['catalog']), observation, identity, mounts, closed)
    require(catalog['schema'] == 'chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
            and observation['schema'] == 'chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
            and plan['identity'] == runtime['identity'] == identity and plan['next_layers'] == '61-72'
            and plan['later_windows_admitted'] is False and plan['actual_fitter_started'] is False
            and plan['native_source_sha256'] == NATIVE_SHA and runtime['plan_sha256'] == sha(raws['recovery_plan'])
            and runtime['no_later_window_admission'] is True
            and plan['mounts'] == runtime['current_mounts'] == observation['current_mounts'] == mounts,
            'complete actual three-origin plan/catalog/observation/source joins differ')
    require(runtime['budget']['memory_bytes'] == CAP_BYTES and runtime['budget']['cpus'] == CAP_CPUS
            and runtime['budget']['swap_bytes'] == 0 and runtime['budget']['selected_new_output_write_bytes'] == WRITER_BYTES and runtime['budget']['selected_new_NAS_write_bytes'] == (WRITER_BYTES if identity['new_output_filesystem']=='NAS' else 0),
            '104GiB/22CPU/noSwap and whole166556705792 writer must be bound in current runtime plan')
    require(allocation['state'] == 'running' and allocation['owner'] == identity['owner']
            and allocation['name'] == identity['name'] and allocation['gpu_uuid'] == identity['gpu_uuid']
            and allocation['gpu_index'] == 6 and allocation['details']['image'] == identity['image']
            and allocation['details']['mounts'] == mounts and
            re.fullmatch(r'[0-9a-f]{64}', allocation['container_id']) is not None,
            'genuine normal new allocation/image/exact11mounts required')
    require(str(uuid.UUID(allocation['allocation_id'])) == allocation['allocation_id'], 'canonical actual allocation UUID required')
    flags = plan['RO_resume_source_flags']; refs = plan['RO_resume_source_refs']
    expected_names = {'catalog': 'ro_multi_origin_catalog.json', 'observation': 'ro_multi_origin_observation.json',
                      'recovery_plan': 'recovery_plan61_72.json', 'runtime_plan': 'runtime_plan61_72.json',
                      'allocation': 'actual_allocation.json'}
    require(len(flags) == 14 and plan['command_argv'][-14:] == flags and dict(zip(flags[::2], flags[1::2])) == {
        '--resume-source-manifest': '/control/'+expected_names['catalog'], '--resume-source-manifest-sha256': sha(raws['catalog']),
        '--resume-source-observation': '/control/'+expected_names['observation'], '--resume-source-observation-sha256': sha(raws['observation']),
        '--resume-source-identity': '/control/identity.json', '--resume-source-identity-sha256': source_pins['identity.json'],
        '--resume-source-root': '/resume-prefix'}, 'exact existing seven-flag/14-token caller ABI required')
    require(refs == {'manifest': {'path': '/control/'+expected_names['catalog'], 'sha256': sha(raws['catalog'])},
                     'observation': {'path': '/control/'+expected_names['observation'], 'sha256': sha(raws['observation'])},
                     'identity': {'path': '/control/identity.json', 'sha256': source_pins['identity.json']}},
            'complete current source references must match exact mounted byte pins')
    for role, name in expected_names.items(): source_pins[name] = sha(raws[role])
    sources = {role: leaf(row) for role, row in closure['sources'].items()}
    sources['allocation'] = {'name': expected_names['allocation'], 'sha256': sha(raws['allocation'])}
    scripts = {role: leaf(closure['scripts'][role]) for role in ('facts', 'preflight', 'service', 'math')}
    host_pin = leaf(closure['scripts']['host_caller'])
    host = {'schema': 'neural_forge.k3_capacity_host_loop_config.v1', 'run_uuid': RUN,
            'output_root': identity['output_root'], 'sources': sources, 'scripts': scripts,
            'host_caller_source_pin': host_pin, 'driver_config_name': 'continuation_config.json',
            'driver_sha256': source_pins['continuation_worker.py'], 'allocation_id': allocation['allocation_id'],
            'container_id': allocation['container_id']}
    host_raw = encode(host)
    driver = {'schema': 'neural_forge.k3_nas_closed60_worker_config.v1', 'identity': identity,
              'expected_allocator_name': allocation['name'], 'continuation_uuid': RUN,
              'window': [61,72], 'later_windows_admitted': False,
              'first_plan': {'path': SEALED36+'/inputs/original_record_plan.json', 'sha256': '17b4e172ac84c82dea494c7bef7e393f926a58e5fca67ece20358fb4e9a29a6e'},
              'baseline_receipt': {'path': '/recovery-source/ancestry/original_receipts/k3_dense_pass_receipt.json', 'sha256': 'c54e2fd0ade6fb8bbc9a111cb76fc5469abc1c197001b137e70f1c1156e976ff'},
              'prep_receipt': {'path': '/recovery-source/prep_result.json', 'sha256': 'bcc3876d5f79e84a5acaa90586ef9d03a84cee0f1781ac9f12cff17e902790ae'},
              'prior_continuity': {'path': SEALED60_MOUNTED+'/window_continuity.json', 'sha256': identity['closed60_continuity_sha256']},
              'closed36_disk_projection': {'path': '/control/closed36_next_resource_admission.json', 'sha256': '1e9004bc11daca688ab836e460517759d8b7d36c7f174d27748f1a38b987443a'},
              'source_pins': source_pins, 'native_tool_sha256': NATIVE_SHA,
              'budget': {'memory_limit_bytes': CAP_BYTES, 'cpus': CAP_CPUS, 'host_headroom_bytes': 16*GIB},
              'native_binding': identity['native_binding'], 'current_mounts': mounts, 'resume_argv': flags,
              'resume_source_manifest': refs['manifest'], 'resume_source_observation': refs['observation'],
              'resume_source_identity': refs['identity'], 'recovery_packet': refs['manifest'],
              'recovery_plan': {'path': '/control/'+expected_names['recovery_plan'], 'sha256': sha(raws['recovery_plan'])},
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'capacity_caller': {'host_caller_config_sha256': sha(host_raw), 'source_scripts': scripts, 'source_pins': sources}}
    out = args.output_directory.resolve()
    require(out.parent == LOCAL_OUTPUTS and not out.exists() and not out.is_symlink(), 'exclusive own final-config output leaf required')
    raw_outputs = {expected_names[role]: raw for role, raw in raws.items()}
    raw_outputs.update({'hostcapacity_caller_config.json': host_raw, 'continuation_config.json': encode(driver)})
    result = {'schema': 'neural_forge.k3_actual61_72_local_final_config_build.v1', 'run_uuid': RUN,
              'source_closure_sha256': spec['source_closure_sha256'], 'allocation_id': allocation['allocation_id'],
              'container_id': allocation['container_id'], 'files': {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in raw_outputs.items()},
              'actual_cgroup104GiB22CPU_noSwap_attestation_required_before_fit': True,
              'complete_source_freeze_and_fresh_owner_capacity_required': True, 'remote_or_runtime_actions': False}
    raw_outputs['FINAL_CONFIG_BUILD_RESULT.json'] = encode(result)
    out.mkdir(exist_ok=False)
    for name, raw in raw_outputs.items(): write_raw(out/name, raw)
    print(json.dumps(result, sort_keys=True))

def main():
    global RUN,COMMON_BYTES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('sources','plans','configs'))
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();spec=decode(raw_ref({'path':str(args.inputs),'sha256':args.inputs_sha256}))
    RUN=spec['run_uuid'];require(isinstance(RUN,str) and str(uuid.UUID(RUN))==RUN and RUN!=PREVIOUS_RUN,'new actual61 UUID remains unresolved')
    selected()
    if args.phase=='sources':return bind(args)
    bundle=Path(spec['source_bundle']);closure=decode(read_raw(bundle/'SOURCE_CLOSURE61_72.json',spec['source_closure_sha256']))
    COMMON_BYTES=closure['source_binding']['common_NAS_floor_compare_bytes']
    return {'plans':plans_bind,'configs':final_configs}[args.phase](args)
if __name__=='__main__':main()
