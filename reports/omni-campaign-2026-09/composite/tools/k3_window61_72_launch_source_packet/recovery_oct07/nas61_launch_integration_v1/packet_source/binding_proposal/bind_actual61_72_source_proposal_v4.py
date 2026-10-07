"""Local-only61-72 source binder; actual60 nine raws and previous49 source are mandatory.

No connection, payload read/copy, allocator, provider or runtime operation.
This is an unpromoted source proposal. Bound output remains undeployable until
independent review, tested source promotion, immutable packaging, actual
three-origin metadata and fresh ordinary allocation/config/admission.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib.util, json, os, re, uuid
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
RECOVERY=HERE.parents[1]
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


def selected():
    pins=decode(raw_ref({'path':str(HERE/'FUTURE_SOURCE_PINS_V3.json'),'sha256':SOURCE_PINS_SHA}))
    for row in pins.values():
        raw=raw_ref({'path':row['path'],'sha256':row['sha256']})
        require(len(raw)==row['bytes'],'future source bytes changed')
    return pins


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


def bind(args):
    spec=decode(raw_ref({'path':str(args.inputs),'sha256':args.inputs_sha256}))
    require(spec['schema']=='neural_forge.k3_actual61_72_local_source_binding_inputs.v1','exact future binder input schema required')
    run=spec['run_uuid'];require(isinstance(run,str) and str(uuid.UUID(run))==run and run!=PREVIOUS_RUN,'new actual61 UUID remains unresolved')
    storage={'NAS':'/mnt/ai_data','Lexar':'/mnt/lexar'}.get(spec['new_output_filesystem'])
    require(storage is not None,'fresh NAS/Lexar output selection remains unresolved')
    require(spec['operating_cap_proposal']=={'memory_bytes':CAP_PROPOSAL,'CPU_threads':22,'swap_bytes':0},'exact conditional104GiB/22/noSwap proposal required')
    pins=selected();b,bundle,prior,raw,values,binding,peak,pid=validate_actual60(spec,pins)
    out=args.output_directory.resolve();require(out.parent==HERE and not out.exists() and not out.is_symlink(),'exclusive own local future output required')
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
        'source_candidate_only_not_normal_promoted':True,'fresh_current_peer_and_FS_admission_required':True}
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
        source_proposal_only=True,fresh_whole_current_FS_and_peer_admission_required=True)
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
    closure.update(selected_run_uuid=run,actual_runtime_admission_claimed=False,source_proposal_only=True,
        source_binding={'schema':'neural_forge.k3_actual61_72_local_source_binding_proposal.v1','identity_sha256':sha(identity_raw),
            'closed60_sha256':sha(raw['closed60_continuity']),'actual60_nine_role_pins':spec['inputs'],
            'previous49_provenance':spec['previous49'],'current_candidate_source_pins':pins,
            'actual_namespace_pid':pid,'actual60_final_peak_bytes':peak,'array_delta_bytes_not_RSS':ARRAY_DELTA,
            'operating_cap_proposal_bytes':CAP_PROPOSAL,'writer_bytes_once':WRITER,'chosen_output_filesystem':spec['new_output_filesystem'],
            'source_proposal_only_not_deployable':True,'fresh_peer_source_and_FS_capacity_admission_required':True,
            'production_file_pins':{name:{'sha256':sha(content),'bytes':len(content)} for name,content in files.items()}})
    # The actual three-origin observer, plan, configs, A/CID/physicalcaps and
    # normal source publication are still required. No guessed placeholders
    # are turned into an executable config or barrier by this local binder.
    extras={'SOURCE_CLOSURE61_72_SOURCE_PROPOSAL.json':encode(closure),'mounts_proposal.json':encode(mounts),
        'observer_source/collect_actual61_72_multi_origin.py':generated['collector'],
        'NEXT_BINDING_UNRESOLVED.json':encode({'three_origin_catalog':None,'three_origin_observation':None,
            'actual_new_allocation':None,'new_config':None,'native_runtime_observation':None,'fresh_capacity':None,
            'source_promoted_commit':None,'new_immutable_packet':None,'can_arm':False})}
    out.mkdir(exist_ok=False)
    for name,content in sorted({**files,**extras}.items()):write(out/name,content)
    print(json.dumps({'path':str(out),'source_closure_sha256':sha(extras['SOURCE_CLOSURE61_72_SOURCE_PROPOSAL.json']),
        'metadata_count':len(roster),'script_count':len(scripts),'source_proposal_only':True,'deployable':False,'runtime_actions':False}))


SOURCE_PINS_SHA='86f03668a61eeaa4a906304b0cf5cbabc929f279876bf10203316944ffb8c07f'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();bind(args)


if __name__=='__main__':main()
