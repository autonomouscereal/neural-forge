"""Local-only73-84 source binder; actual72 nine raws and previous61 source are mandatory.

No connection, payload read/copy, allocator, provider or runtime operation.
Selected sources use the full current pinned roster and authenticated promoted phase component packet;
 actual preceding raw evidence, new UUID/filesystem, four-origin metadata, fresh
whole admission and normal allocation/config/freeze/caps remain mandatory.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib.util, json, os, re, uuid, zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
LOCAL_OUTPUTS=Path('D:/codex-owned-source-receipts/k3-later73-92-QIMG-compute-closed-source-only/phase73_84')
RECOVERY='C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07' if False else Path('C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07')
CANDIDATE='C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\later73_92_integration_source_proposals_v1\\window73_84_candidate_v3' if False else Path('C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\later73_92_integration_source_proposals_v1\\window73_84_candidate_v3')
PREVIOUS_BINDER=Path('C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\nas61_launch_integration_v1\\current_QIMG_compute_closed_v1\\bind_actual61_QIMG_compute_closed_v1.py')
PREVIOUS_BINDER_SHA='9aeee414367f1123d08f20a7978026a5acc394117662529c06c38fbe72f909e4'
PREVIOUS_RUN=None
PREVIOUS_NATIVE='078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'
NATIVE='01a11108-009c-72f3-b2ba-e0d48fca1435'
THREAD='95f6c2bd-7139-46f6-b5d6-228d1b9753f7'
GIB=1<<30
CAP_PROPOSAL=None
WRITER=180943168512
ARRAY_DELTA=8220835840
COMMON_BASE=692026605568
WRITER_PATH=Path('C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\remaining_parent_header_accounting_v1\\ACTUAL_HEADER_73_84_WRITE_AND_ARRAY_COMPONENTS_V1.json')
WRITER_SHA='f183d67fc519f4896415b5084c1be069a5574bfbb1cc534b7b4969ebed9577fa'
ROLES={'closed72_continuity','terminal_receipt','full_ffn_validation','snapshot_manifest','fitter_execution',
    'controller_terminal_result','resources_jsonl','controller_events_jsonl','binding_receipt'}
PRIOR_ROLES={'source_closure','source_binding','identity','catalog','observation','driver_config','source_freeze','allocation','recovery_plan'}



PACKET_PATH=Path('D:\\codex-owned-source-packets\\k3-later73-92-546b512c-052a-4f77-ace3-cf2fc4ab9e64\\k3_window73_84_source_components_v1.zip')
PACKET_SHA='ae66ce7f5db00cb053d9f8078fd38290ad83f318c03ee44415e81b9631eb9ace'
PACKET_MANIFEST_SHA='cd1a6bc059869fd7fae24cc807d530e234e2bb3f5ada4f9fea52fd3b759ce90a'
PROMOTED_COMMIT='69e1e3c5e546de20f1807bbeed1c1f9c7f864e97'
SOURCE_PINS_SHA='ff759776801d428282cb9aab966e75705d62a36a98fb6af83eba148ec8d02334'
QIMG_CLOSED_PREFLIGHT_SHA='115c5edb6cdcff614a9db58115069759794148aa3a1ecd8872275471ed8ee366'
QIMG_CLOSURE_PROOF_ROLES={'QIMG_native_compute_closure': {'bytes': 46772, 'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\recovery_resource_review\\NEXT_QIMG_compute_closure_v1\\QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json', 'sha256': '0f287f18050665afd92ee5d35cbf5f3cee516efff185af2e9d643f0721d54b20', 'name': 'QIMG_EXACT_NATIVE_COMPUTE_CLOSURE_PUBLIC_PROOF_V1.json'}, 'QIMG_native_terminal_raw_part_1': {'bytes': 9338811, 'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\recovery_resource_review\\NEXT_QIMG_compute_closure_v1\\QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json', 'sha256': '8e2ada9351e913b593c1b0d851e40725aabdbf11ce1e14b6e9cd3355eec68e28', 'name': 'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_1_V1.json'}, 'QIMG_native_terminal_raw_part_2': {'bytes': 9393937, 'path': 'C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07\\recovery_resource_review\\NEXT_QIMG_compute_closure_v1\\QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json', 'sha256': 'db9fd521dc10e5524f7cd8260d9e079a65647b9cb64ad33426cb1def51fae4ad', 'name': 'QIMG_EXACT_NATIVE_TERMINAL_RAW_PART_2_V1.json'}}
RUN=None
COMPONENT_MEMBER_COUNT=19
PRIOR_VALIDATOR_SHA='3ac5d3f4b7de46fbb3685e3e8e17a15ba86ed0f2cf6218223e71914052a89cde'
CAP_BYTES,CAP_CPUS=None,22
WRITER_BYTES=WRITER
COMMON_BYTES=None
NATIVE_SHA='21e322039fa65334a49e222997f92a76e640cf68d75e5388244b1e7719baae56'
RESOLVER_SHA='146665bbce576e52462d73384d96fafd7ec07a07a2f7a9f95bd52c9655f84ada'
RUNTIME_SHA='288bf5f3cde564b70932097d28a049fc1c4be500b83d7a43e33cb1273fa5e260'
GUARD_SHA='88e118847efb589d9cc48fc578996c3ca48cd0a0b9a4b491aab573fed54a9d41'
SEALED36='/resume-prefix/ancestry/window-25-36-b873d51f-b251-4944-aa84-5d91df96e87b'
SEALED48_LOGICAL='/run-output/ancestry/window-37-48-764524b1-96a9-4636-be74-942f9c54754c'
SEALED72_LOGICAL=None
SEALED72_MOUNTED=None

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

def validate_actual72(spec,pins):
    global PREVIOUS_RUN,SEALED72_LOGICAL,SEALED72_MOUNTED
    b=previous_binder()
    module_raw=raw_ref({'path':str(HERE/'validate_actual72_source_metadata_v2.py'),'sha256':PRIOR_VALIDATOR_SHA})
    namespace={'__name__':'reviewed_later_prior_validator','__file__':str(HERE/'validate_actual72_source_metadata_v2.py')}
    exec(compile(module_raw,'<exact local preceding metadata validator>','exec'),namespace)
    result=namespace['validate_actual_72'](spec,pins,b)
    PREVIOUS_RUN=result[2]['identity']['run_uuid']
    SEALED72_LOGICAL='/run-output/ancestry/window-61-72-'+PREVIOUS_RUN
    SEALED72_MOUNTED='/resume-window61-72/ancestry/window-61-72-'+PREVIOUS_RUN
    return result

def read_raw(path,expected):return raw_ref({'path':str(path),'sha256':expected})
def read_ref(ref):
    raw=raw_ref(ref);return decode(raw),raw
def full_sha(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None,'full SHA required');return value
def leaf(row):return {key:row[key] for key in ('name','sha256')}
def write_raw(path,raw):
    write(path,raw);return {'path':str(path.resolve()),'sha256':sha(raw),'bytes':len(raw)}

def selected():
    packet_raw=PACKET_PATH.read_bytes();require(sha(packet_raw)==PACKET_SHA,'exact promoted component packet differs')
    manifest=decode(raw_ref({'path':str(PACKET_PATH.with_suffix('.json')),'sha256':PACKET_MANIFEST_SHA}))
    require(manifest['git_commit']==PROMOTED_COMMIT and manifest['source_only'] is True
        and manifest['deployable'] is False and manifest['actual72_or84'] is None,'exact normal-promoted unbound component provenance required')
    with zipfile.ZipFile(PACKET_PATH) as archive:
        require(len(archive.infolist())==COMPONENT_MEMBER_COUNT and set(archive.namelist())==set(manifest['members']),'complete component packet required')
        for name,row in manifest['members'].items():
            raw=archive.read(name);require(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'full component member differs')
    pins=decode(raw_ref({'path':str(HERE/'SELECTED_PHASE_QIMG_COMPUTE_CLOSED_SOURCE_PINS_V1.json'),'sha256':SOURCE_PINS_SHA}))
    require(set(pins)=={'native','resolver','runtime','ancestry','verifier','driver','collector','qualification','facts','host','preflight'},'exact eleven concrete source roles required')
    with zipfile.ZipFile(PACKET_PATH) as archive:
        for role,row in pins.items():
            raw=raw_ref({'path':row['path'],'sha256':row['sha256']});require(len(raw)==row['bytes'],'selected template extent differs')
            if role not in {'facts','host','preflight'}:
                require(archive.read(Path(row['path']).name)==raw,'exact promoted mathematical/IO component differs')
    return pins

def bind(args):
    spec=decode(raw_ref({'path':str(args.inputs),'sha256':args.inputs_sha256}))
    require(spec['schema']=='neural_forge.k3_actual73_84_local_source_binding_inputs.v1','exact future binder input schema required')
    run=spec['run_uuid'];require(isinstance(run,str) and str(uuid.UUID(run))==run and run!=PREVIOUS_RUN,'new actual61 UUID remains unresolved')
    storage={'NAS':'/mnt/ai_data','Lexar':'/mnt/lexar'}.get(spec['new_output_filesystem'])
    require(storage is not None,'fresh NAS/Lexar output selection remains unresolved')
    global CAP_PROPOSAL,CAP_BYTES
    cap=spec['operating_cap_proposal'];require(set(cap)=={'memory_bytes','CPU_threads','swap_bytes'} and type(cap['memory_bytes']) is int and cap['memory_bytes']>0 and cap['CPU_threads']==22 and cap['swap_bytes']==0,'actual phase cap/CPU22/noSwap remains unresolved')
    CAP_PROPOSAL=CAP_BYTES=cap['memory_bytes']
    pins=selected();b,bundle,prior,raw,values,binding,peak,pid=validate_actual72(spec,pins)
    out=args.output_directory.resolve();require(out.parent==LOCAL_OUTPUTS and not out.exists() and not out.is_symlink(),'exclusive own local future output required')
    i=copy.deepcopy(prior['identity']);prefix='k3-nas-window73-84-oct07-'+run
    i.update(run_uuid=run,recovery_uuid=run,name='k3-nas73-84-'+run.replace('-','')[:16],new_layers=[73,84],
        runtime_root='/home/cereal/omni-campaign/runtime/'+prefix,
        output_root='/home/cereal/omni-campaign/outputs/'+prefix+'/control-output',
        controller_output_root='/home/cereal/omni-campaign/outputs/'+prefix+'/control-output',
        native_working_output_root=storage+'/omni-campaign/'+prefix+'/model',new_output_filesystem=spec['new_output_filesystem'],
        closed72_continuity_sha256=sha(raw['closed72_continuity']),parent_catalog_sha256=sha(raw_ref(spec['previous61']['inputs']['catalog'])),
        parent_catalog_row_count=len(prior['catalog']['input_catalog']),
        parent_catalog_rows_sha256=sha(json.dumps(prior['catalog']['input_catalog'],sort_keys=True,separators=(',',':')).encode()),
        native_binding=binding,binding_receipt=spec['inputs']['binding_receipt']['path'],actual_runtime_admitted=False,
        identity_role='separate73-84 source proposal; no actual new allocation/caps/admission claimed')
    require(run!=prior['identity']['run_uuid'],'fresh phase UUID must differ from exact predecessor')
    i['readonly_resume_source_roots']['closed61_72']=prior['identity']['native_working_output_root']
    i['resume_window_producer_uuids']=copy.deepcopy(prior['identity'].get('resume_window_producer_uuids',{'closed37_48':'764524b1-96a9-4636-be74-942f9c54754c','closed49_60':'03603908-570d-4c90-96cb-ee3a63a9a139'}))
    i['resume_window_producer_uuids']['closed61_72']=prior['identity']['run_uuid']
    i['operating_cap_proposal']=copy.deepcopy(cap)
    require(run not in set(i['resume_window_producer_uuids'].values()),'new genuine phase UUID must be unique across every old origin')
    identity_raw=encode(i);binding_raw=encode(binding)
    proof={'schema':'neural_forge.k3_completed61_source_to73_binding.v1','previous_run_uuid':prior['identity']['run_uuid'],
        'closed72_continuity_sha256':sha(raw['closed72_continuity']),'actual_namespace_pid':pid,
        'actual72_final_aggregate_peak_bytes':peak,'qualified_next_array_delta_bytes_not_RSS':ARRAY_DELTA,
        'next_operating_cap_proposal_bytes':CAP_PROPOSAL,'retained_operating_cap_proposal':copy.deepcopy(cap),'whole_RSS_upper_bound_claimed':False,
        'previous61_complete_provenance':spec['previous61'],'actual72_nine_role_pins':spec['inputs'],
        'source_candidate_only_not_normal_promoted':False,'normal_template_promoted_commit':PROMOTED_COMMIT,'normal_template_packet_sha256':PACKET_SHA,'fresh_current_peer_and_FS_admission_required':True}
    proof_raw=encode(proof);writer_raw=raw_ref({'path':str(WRITER_PATH),'sha256':WRITER_SHA})
    require(decode(writer_raw)['writer']['conservative_complete_source_scope_extent_bytes']==WRITER,
        'actual header-qualified whole61 writer bound differs')
    closure=copy.deepcopy(prior['source_closure']);roster=closure['sources']
    roles_raw={role:raw_ref({'path':str(bundle/'production_control'/row['name']),'sha256':row['sha256']}) for role,row in roster.items()}
    roles_raw.update(identity=identity_raw,binding=binding_raw,closed72_continuity=raw['closed72_continuity'],
        own73_84_writer_extent=writer_raw,completed61_phase_source=proof_raw)
    roster['closed72_continuity']={'name':'closed72_continuity.json'}
    roster['own73_84_writer_extent']={'name':'ACTUAL_HEADER_73_84_WRITE_AND_ARRAY_COMPONENTS_V1.json'}
    roster['completed61_phase_source']={'name':'COMPLETED61_SOURCE_TO73_BINDING_V1.json'}
    scope=decode(roles_raw['NAS_current_scope']);scope.update(schema='neural_forge.k3_73_84.source_qualified_resource_scope.v1',
        run_uuid=run,identity_sha256=sha(identity_raw),layers=[73,84],new_output_filesystem=spec['new_output_filesystem'],
        NAS_current_common_guard_bytes_once=COMMON_BASE+(WRITER if storage=='/mnt/ai_data' else 0),
        NAS_common_guard_current_K3_writer_bytes_once=WRITER if storage=='/mnt/ai_data' else 0,
        selected_output_writer_bytes_once=WRITER,completed61_source_proof_sha256=sha(proof_raw),
        source_proposal_only=False,fresh_whole_current_FS_and_peer_admission_required=True)
    for role in ('identity','binding','closed72_continuity','own73_84_writer_extent','completed61_phase_source'):
        scope['source_pins'][role]={'name':roster[role]['name'],'sha256':sha(roles_raw[role])}
    roles_raw['NAS_current_scope']=encode(scope);roster['NAS_current_scope']['name']='BOUND73_84_RESOURCE_SCOPE_SOURCE_PROPOSAL_V1.json'
    generated,assignment_values=generated_templates(pins,i,binding,CAP_PROPOSAL,storage,sha(raw['closed72_continuity']),sha(binding_raw),sha(roles_raw['NAS_current_scope']),b)
    files={name:raw_ref({'path':str(bundle/name),'sha256':row['sha256']}) for name,row in prior['source_binding']['production_file_pins'].items()}
    tool='reports/omni-campaign-2026-09/composite/tools/'
    for role,name in [('native','k3_dense_pass.py'),('resolver','ro_resume_paths.py'),('runtime','ro_resume_runtime.py'),('ancestry','k3_recovery_ancestry.py')]:
        content=generated['runtime'] if role=='runtime' else raw_ref({'path':pins[role]['path'],'sha256':pins[role]['sha256']})
        files['production_code/'+tool+name]=content
        if role!='native':files['production_control/'+name]=content
    files['production_control/verify_later_window.py']=raw_ref({'path':pins['verifier']['path'],'sha256':pins['verifier']['sha256']})
    mapping={'facts':'k3_owned_readonly_facts_NAS73_84_v1.py','host':'k3_capacity_host_loop_v1.py',
        'preflight':'k3_owned_capacity_preflight_NAS73_84_v1.py','driver':'continuation_worker.py','qualification':'native_nas_io_qualification.py'}
    for role,name in mapping.items():files['production_control/'+name]=generated[role]
    files['production_control/identity.json']=identity_raw
    observer=files['production_control/observe_native.py']
    prior_tool='production_code/'+tool
    old_pins={role:prior['source_binding']['production_file_pins'][prior_tool+name]['sha256'] for role,name in [('native','k3_dense_pass.py'),('resolver','ro_resume_paths.py'),('runtime','ro_resume_runtime.py')]}
    new_pins={'native':pins['native']['sha256'],'resolver':pins['resolver']['sha256'],'runtime':sha(generated['runtime'])}
    for role,old in old_pins.items():
        require(observer.count(old.encode())==1,'exact actual prior observer pin slot required')
        observer=observer.replace(old.encode(),new_pins[role].encode())
    files['production_control/observe_native.py']=observer
    boot=files['production_control/recovery_boot.py']
    require(boot.count(PREVIOUS_RUN.encode())==1,'exact prior boot UUID slot required')
    files['production_control/recovery_boot.py']=boot.replace(PREVIOUS_RUN.encode(),run.encode())
    files['production_control/source_revisions/k3_dense_pass_61.py']=raw_ref({'path':str(bundle/'production_code'/tool/'k3_dense_pass.py'),'sha256':PREVIOUS_NATIVE})
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
    mounts.append({'source':prior['identity']['native_working_output_root'],'target':'/resume-window61-72','read_only':True});mounts.sort(key=lambda row:row['target'])
    require(len(mounts)==12 and len({row['target'] for row in mounts})==12,'exact 12 disjoint four-origin mount rows required')
    require(set(roster)==set(prior['source_closure']['sources'])|{'closed72_continuity','own73_84_writer_extent','completed61_phase_source'}
        and len(roster)==73 and len(scripts)==5,'exact previous70 plus three new phase roles/five scripts required')
    for name,content in files.items():
        if name.endswith('.py'):compile(content,name,'exec')
    closure.update(selected_run_uuid=run,actual_runtime_admission_claimed=False,source_proposal_only=False,
        source_binding={'schema':'neural_forge.k3_actual73_84_local_source_binding_proposal.v1','identity_sha256':sha(identity_raw),
            'closed72_sha256':sha(raw['closed72_continuity']),'actual72_nine_role_pins':spec['inputs'],
            'previous61_provenance':spec['previous61'],'current_candidate_source_pins':pins,'generated_template_assignment_values':assignment_values,
            'bound_template_source_pins':{role:{'sha256':sha(value),'bytes':len(value)} for role,value in generated.items()},
            'actual_namespace_pid':pid,'actual72_final_peak_bytes':peak,'array_delta_bytes_not_RSS':ARRAY_DELTA,
            'operating_cap_proposal_bytes':CAP_PROPOSAL,'writer_bytes_once':WRITER,'chosen_output_filesystem':spec['new_output_filesystem'],
            'ROOT_serial_scope_source_sha256':'3f12e995d8715d56700d1b32954876808f544ab5244917a033b147fb2e2152ab','ROOT_serial_preflight_basis_sha256':'f4726d11d3edaee3a48357100ff931d22c337889a19c23e274c282e2bc476faa','ROOT_serial_current_preflight_template_sha256':pins['preflight']['sha256'],
            'current_ROOT_serial_RAM_bytes_once':16*GIB,'current_ROOT_serial_CPU_threads_once':4,'current_ROOT_serial_future_write_bytes_once':10*GIB,
            'source_proposal_only_not_deployable':False,'fresh_peer_source_and_FS_capacity_admission_required':True,
            'production_file_pins':{name:{'sha256':sha(content),'bytes':len(content)} for name,content in files.items()}})
    closure['source_binding'].update(mounts_sha256=sha(encode(mounts)),normal_template_promoted_commit=PROMOTED_COMMIT,immutable_template_packet_sha256=PACKET_SHA,common_NAS_floor_compare_bytes=COMMON_BASE+(WRITER if storage=='/mnt/ai_data' else 0),observer_source_pin=pin(out/'observer_source/collect_actual73_84_multi_origin.py',generated['collector']))
    # The actual four-origin observer, plan, configs, A/CID/physicalcaps and
    # normal source publication are still required. No guessed placeholders
    # are turned into an executable config or barrier by this local binder.
    closure['source_binding'].update(QIMG_closed_compute_preflight_template_sha256=QIMG_CLOSED_PREFLIGHT_SHA,
        QIMG_closed_compute_preflight_normal_promoted_commit=None,QIMG_closed_compute_source_only_revision=True,
        QIMG_closed_compute_public_proof_roles=QIMG_CLOSURE_PROOF_ROLES)
    extras={'SOURCE_BINDING_RESULT.json':encode(closure['source_binding']),'SOURCE_CLOSURE73_84.json':encode(closure),'mounts.json':encode(mounts),
        'observer_source/collect_actual73_84_multi_origin.py':generated['collector'],
        'NEXT_BINDING_UNRESOLVED.json':encode({'complete_current_origin_catalog':None,'complete_current_origin_observation':None,
            'actual_new_allocation':None,'new_config':None,'native_runtime_observation':None,'fresh_capacity':None,
            'source_promoted_commit':None,'new_immutable_packet':None,'can_arm':False})}
    out.mkdir(exist_ok=False)
    for name,content in sorted({**files,**extras}.items()):write(out/name,content)
    print(json.dumps({'path':str(out),'source_closure_sha256':sha(extras['SOURCE_CLOSURE73_84.json']),
        'metadata_count':len(roster),'script_count':len(scripts),'normal_template_promoted_commit':PROMOTED_COMMIT,'actual_runtime_admitted':False,'runtime_actions':False}))
def template_assignment_values(identity,binding,cap,storage,closed_sha,binding_sha,scope_sha,pins):
    roots=identity['readonly_resume_source_roots'];prior_run=identity['resume_window_producer_uuids']['closed61_72']
    common={'EXPECTED_RUN':identity['run_uuid'],'IDENTITY_SHA':sha(encode(identity)),'BINDING_SHA':binding_sha,
        'CLOSED72_SHA':closed_sha,'FUTURE_SESSION_ID':binding['session_id'],'FUTURE_GENERATION':binding['session_generation'],
        'FUTURE_CONTEXT_SHA':binding['context_sha256'],'OWN_RAM_BYTES':cap,'OWN_CPU_THREADS':22,'OUTPUT_STORAGE_ROOT':storage,
        'RO_ROOTS':roots,'PREVIOUS_RUN_UUID':prior_run}
    runtime={'ROOT61_72':roots['closed61_72']}
    driver={'RUN':identity['run_uuid'],'IDENTITY_SHA':sha(encode(identity)),'CAP_BYTES':cap,'CAP_CPUS':22,
        'SESSION_GENERATION':binding['session_generation'],'SESSION_ID':binding['session_id'],'OUTPUT_STORAGE_ROOT':storage,
        'LATEST_NAS':roots['closed61_72']}
    return {'facts':dict(common),'host':dict(common),'runtime':runtime,'driver':driver,
        'qualification':{'EXPECTED_RUN':identity['run_uuid']},
        'collector':{'RUN':identity['run_uuid'],'PREVIOUS_RUN':prior_run,'HOST_ORIGINS':roots}}

def bind_assignments(raw, values):
    """Change only one exact top-level literal assignment for each named slot."""
    text = raw.decode(); tree = ast.parse(text); spans = {}
    lines = text.splitlines(keepends=True); offsets = [0]
    for line in lines: offsets.append(offsets[-1] + len(line))
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                require(name not in spans, 'duplicate literal binding slot')
                ast.literal_eval(node.value)
                spans[name] = (offsets[node.lineno-1] + node.col_offset,
                               offsets[node.end_lineno-1] + node.end_col_offset)
    require(set(spans) == set(values), 'missing exact top-level binding slot')
    for name, (start, end) in sorted(spans.items(), key=lambda row: row[1][0], reverse=True):
        text = text[:start] + name + ' = ' + repr(values[name]) + text[end:]
    new = ast.parse(text)
    funcs = lambda tr: [ast.dump(n, include_attributes=False) for n in tr.body
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    require(funcs(tree) == funcs(new), 'binding changed reviewed function/class AST')
    compile(text, '<bound source>', 'exec')
    return text.encode()

def bind_collector_assignments(raw,values):
    # The reviewed collector's HOST_ORIGINS uses fixed historical string
    # concatenations. Bind that one dictionary explicitly, then use the exact
    # original literal-slot helper for every other allowed top-level value.
    require(set(values['HOST_ORIGINS'])==set(['legacy_closed36', 'closed37_48', 'closed49_60', 'closed61_72']),'exact authenticated old-origin roster required')
    text=raw.decode();tree=ast.parse(text)
    slots=[n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='HOST_ORIGINS']
    require(len(slots)==1,'one exact reviewed collector physical map required')
    old=ast.get_source_segment(text,slots[0]);require(text.count(old)==1,'unique physical map span required')
    new=text.replace(old,'HOST_ORIGINS = '+repr(values['HOST_ORIGINS']))
    funcs=lambda tr:[ast.dump(n,include_attributes=False) for n in tr.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    require(funcs(tree)==funcs(ast.parse(new)),'collector function/class AST must remain byte-semantic unchanged')
    remainder={name:value for name,value in values.items() if name!='HOST_ORIGINS'}
    return bind_assignments(new.encode(),remainder)

def generated_templates(pins,identity,binding,cap,storage,closed_sha,binding_sha,scope_sha,b):
    values=template_assignment_values(identity,binding,cap,storage,closed_sha,binding_sha,scope_sha,pins)
    generated={role:bind_assignments(raw_ref({'path':pins[role]['path'],'sha256':pins[role]['sha256']}),slots) for role,slots in values.items() if role!='collector'}
    values['preflight']=dict(values['facts'],FACTS_SHA=sha(generated['facts']),CURRENT_SCOPE_SHA=scope_sha)
    generated['preflight']=bind_assignments(raw_ref({'path':pins['preflight']['path'],'sha256':pins['preflight']['sha256']}),values['preflight'])
    values['collector']['SOURCES']={'k3_dense_pass.py':pins['native']['sha256'],'ro_resume_paths.py':pins['resolver']['sha256'],
        'ro_resume_runtime.py':sha(generated['runtime']),'k3_recovery_ancestry.py':pins['ancestry']['sha256']}
    generated['collector']=bind_collector_assignments(raw_ref({'path':pins['collector']['path'],'sha256':pins['collector']['sha256']}),values['collector'])
    return generated,values

def verify_bound_bundle(bundle, closure):
    require(closure['schema'] == 'neural_forge.k3_owned_capacity_source_closure.v15'
            and closure['selected_run_uuid'] == RUN and len(closure['sources']) == 73
            and len(closure['scripts']) == 5, 'complete source-bound73 metadata/five-script closure required')
    peer = closure['sources'].get('current_peer_holder_source')
    require(isinstance(peer, dict) and peer.get('name') == 'CURRENT_QIMG_NATIVE_AND_ROOT1075_PUBLIC_PROOF_V3.json'
            and peer.get('sha256') == '1a49ab6d7ae2c5cff10a93df3e6ef63a0f239feb9646ec246c3a9b5cc5d242b9'
            and peer.get('bytes') == 1797866, 'complete exact current-peer proof role required')
    serial=closure['sources'].get('ROOT_current_serial_scope')
    require(isinstance(serial,dict) and serial['name']=='ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json'
        and serial['sha256']=='3f12e995d8715d56700d1b32954876808f544ab5244917a033b147fb2e2152ab' and serial['bytes']==191797,'exact authentic same-owner serial ROOT16/4/10 source role required')
    require(read_raw(bundle/'production_control'/serial['name'],serial['sha256'])==raw_ref({'path':str(HERE/'ROOT_CURRENT_SERIAL_H3_AUDIO_ACE_TEXT_PUBLIC_PROOF_V2.json'),'sha256':serial['sha256']}),'complete retained ROOT serial proof bytes differ')
    for role,ref in QIMG_CLOSURE_PROOF_ROLES.items():
        row=closure['sources'].get(role)
        require(isinstance(row,dict) and {key:row.get(key) for key in ('name','sha256','bytes')}=={key:ref[key] for key in ('name','sha256','bytes')},'complete exact inherited QIMG native compute proof roles required')
        require(read_raw(bundle/'production_control'/row['name'],row['sha256'])==raw_ref({'path':str(HERE/ref['name']),'sha256':ref['sha256']}),'retained QIMG native compute proof full bytes differ')
    require(closure['source_binding']['QIMG_closed_compute_preflight_template_sha256']==QIMG_CLOSED_PREFLIGHT_SHA
        and closure['source_binding']['QIMG_closed_compute_preflight_normal_promoted_commit'] is None,'honest NEXT-only unpromoted current QIMG source provenance required')
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
    identity=decode(read_raw(bundle/'production_control/identity.json',closure['sources']['identity']['sha256']))
    require(identity['run_uuid']==RUN and identity['new_layers']==[73,84],'current exact phase identity required')
    cap=identity['operating_cap_proposal'];require(type(cap['memory_bytes']) is int and cap['memory_bytes']>0 and cap['CPU_threads']==22 and cap['swap_bytes']==0,'bound actual cap is unresolved')
    global CAP_PROPOSAL,CAP_BYTES,RUNTIME_SHA,PREVIOUS_RUN,SEALED72_LOGICAL,SEALED72_MOUNTED
    CAP_PROPOSAL=CAP_BYTES=cap['memory_bytes'];PREVIOUS_RUN=identity['resume_window_producer_uuids']['closed61_72']
    SEALED72_LOGICAL='/run-output/ancestry/window-61-72-'+PREVIOUS_RUN
    SEALED72_MOUNTED='/resume-window61-72/ancestry/window-61-72-'+PREVIOUS_RUN
    pins=selected();previous=previous_binder()
    proof=decode(read_raw(bundle/'production_control'/closure['sources']['completed61_phase_source']['name'],closure['sources']['completed61_phase_source']['sha256']))
    previous.RUN=proof['previous_run_uuid']
    generated,slots=generated_templates(pins,identity,identity['native_binding'],CAP_BYTES,{'NAS':'/mnt/ai_data','Lexar':'/mnt/lexar'}[identity['new_output_filesystem']],
        identity['closed72_continuity_sha256'],closure['sources']['binding']['sha256'],closure['sources']['NAS_current_scope']['sha256'],previous)
    require(slots==closure['source_binding']['generated_template_assignment_values'],'exact permitted IO assignment values differ')
    mapping={'runtime':'ro_resume_runtime.py','driver':'continuation_worker.py','qualification':'native_nas_io_qualification.py','facts':'k3_owned_readonly_facts_NAS73_84_v1.py','host':'k3_capacity_host_loop_v1.py','preflight':'k3_owned_capacity_preflight_NAS73_84_v1.py'}
    for role,name in mapping.items():require(read_raw(bundle/'production_control'/name,sha(generated[role]))==generated[role],'exact re-derived template bytes differ')
    for role,name in [('native','k3_dense_pass.py'),('resolver','ro_resume_paths.py'),('runtime','ro_resume_runtime.py'),('ancestry','k3_recovery_ancestry.py')]:
        require(read_raw(bundle/'production_code/reports/omni-campaign-2026-09/composite/tools'/name,sha(generated['runtime']) if role=='runtime' else pins[role]['sha256'])==(generated['runtime'] if role=='runtime' else raw_ref({'path':pins[role]['path'],'sha256':pins[role]['sha256']})),'exact native/helper code derivative differs')
    require(read_raw(bundle/'production_control/verify_later_window.py',pins['verifier']['sha256'])==raw_ref({'path':pins['verifier']['path'],'sha256':pins['verifier']['sha256']}),'exact serial scanner/terminal publication verifier differs')
    require(read_raw(bundle/'observer_source/collect_actual73_84_multi_origin.py',sha(generated['collector']))==generated['collector'],'exact bounded collector template differs')
    RUNTIME_SHA=sha(generated['runtime'])

def observation_metadata(catalog,catalog_sha,observation,identity,mounts,closed):
    require(catalog['schema']=='chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
        and observation['schema']=='chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
        and observation['source_catalog_sha256']==catalog_sha
        and observation['first_new_layer']==73 and observation['last_new_layer']==84
        and observation['current_mounts']==mounts,'actual four-origin catalog/mount metadata required')
    require(observation['acting_identity']=={'owner':identity['owner'],'native_context':identity['native_thread_id'],
        'run_uuid':RUN,'image':identity['image'],'control_binding':identity['native_binding']},'actual acting identity differs')
    roots=observation['source_roots'];aliases={'legacy_closed36': '/resume-prefix', 'closed37_48': '/resume-window37-48', 'closed49_60': '/resume-window49-60', 'closed61_72': '/resume-window61-72'}
    require(set(roots)==set(aliases)==set(identity['readonly_resume_source_roots']),'exact 4 RO origins required')
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
    require(observation['closed72_continuity']=={'path':SEALED72_LOGICAL+'/window_continuity.json','sha256':identity['closed72_continuity_sha256']}
        and observation['closed48_continuity']=={'path':SEALED48_LOGICAL+'/window_continuity.json','sha256':identity['closed48_continuity_sha256']},
        'actual latest sealed60 and preserved closed48 joins required')
    for origin,producer in identity['resume_window_producer_uuids'].items():
        lo_s,hi_s=origin[len('closed'):].split('_');key='closed'+hi_s+'_continuity'
        require(observation[key]=={'path':'/run-output/ancestry/window-'+lo_s+'-'+hi_s+'-'+producer+'/window_continuity.json','sha256':identity[key+'_sha256']},
            'every retained old origin closure/producer/rawSHA join required')
    old={}
    for idx,row in enumerate(rows):
        logical=row.get('recorded_logical_path') or '/run-output/'+row['relative_path']
        rel=PurePosixPath(logical).relative_to('/run-output');origin=row['source_origin'];root=roots[origin]
        require((origin=='closed61_72')==(idx>=parent_count) and logical not in old
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
    require(all(f'/run-output/fold/resid/L{n:02d}/held_rows.safetensors' in old for n in range(37,73)),
        'all24 authentic held-row source extents required')
    require(observation['source_pins']=={'ro_resume_paths.py':RESOLVER_SHA,'ro_resume_runtime.py':RUNTIME_SHA,
        'k3_recovery_ancestry.py':GUARD_SHA,'k3_dense_pass.py':NATIVE_SHA},'selected exact source pins differ')
    require(observation['previous_recovery_packet']=={'path':'/control/source_revisions/recovery_ancestry_capacity_wait.json','sha256':'5b9559b8ac20428de7e4ac09201f71839f4d03113a985715ca81d3aedd01246c'}
        and observation['previous_native_source']=={'path':'/control/source_revisions/k3_dense_pass_9086.py','sha256':'9086d7c0a5ec2bc7cfd1b823aecdc12e2973b3c81b064776ad3670df9d6b7b69'}
        and observation['completed_native_source']=={'path':'/control/source_revisions/k3_dense_pass_720.py','sha256':'720465874722e1548c08f82a019be7dcb477efcc5828d6edd10610c1ea076243'}
        and observation['completed49_native_source']=={'path':'/control/source_revisions/k3_dense_pass_5e.py','sha256':'5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'}
        and observation['completed61_native_source']=={'path':'/control/source_revisions/k3_dense_pass_61.py','sha256':'078187a4cfa2f5f7810f0c0a8fa85f7fcc8304085684d45dfda5b895854881d0'},
        'authentic historical native/context source lineage required')
    require(observation['new_write_paths'] and len(set(observation['new_write_paths']))==len(observation['new_write_paths'])
        and not(set(observation['new_write_paths'])&set(old)),'finite disjoint new73-84 writes required')

def plans_bind(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual73_84_local_plan_binding_inputs.v1'
            and spec['run_uuid'] == RUN and set(spec['inputs']) == {'catalog', 'observation'},
            'actual metadata-only four-origin plan input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == LOCAL_OUTPUTS and bundle.is_dir() and not bundle.is_symlink(), 'own bound source bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE73_84.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    control = bundle/'production_control'
    identity = decode(read_raw(control/'identity.json', closure['sources']['identity']['sha256']))
    closed = decode(read_raw(control/closure['sources']['closed72_continuity']['name'], identity['closed72_continuity_sha256']))
    mounts = decode((bundle/'mounts.json').read_bytes())
    catalog, catalog_raw = read_ref(spec['inputs']['catalog'])
    observation, observation_raw = read_ref(spec['inputs']['observation'])
    observation_metadata(catalog, sha(catalog_raw), observation, identity, mounts, closed)
    completed=decode(read_raw(control/closure['sources']['completed61_phase_source']['name'],closure['sources']['completed61_phase_source']['sha256']))
    prior=decode(raw_ref(completed['previous61_complete_provenance']['inputs']['recovery_plan']))
    # Use the reviewed Driver's exact pure original-argv transformer. Old
    # assertions remain inside dated_original_plan_provenance only.
    driver_raw = (control/'continuation_worker.py').read_bytes()
    tree = ast.parse(driver_raw)
    transform = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'operational_argv']
    require(len(transform) == 1, 'exact selected pure operational argv function required')
    env = {'require': require}
    exec(compile(ast.Module(body=transform, type_ignores=[]), '<reviewed original argv transform>', 'exec'), env)
    argv = env['operational_argv'](prior['dated_original_plan_provenance']['plan']['command_argv'], '73-84')
    require(argv[:4] == ['env', 'OMP_NUM_THREADS=22', 'MKL_NUM_THREADS=22', 'OPENBLAS_NUM_THREADS=22'],
            'source selected original CPU22 environment required')
    refs = {'manifest': {'path': '/control/ro_multi_origin_catalog.json', 'sha256': sha(catalog_raw)},
            'observation': {'path': '/control/ro_multi_origin_observation.json', 'sha256': sha(observation_raw)},
            'identity': {'path': '/control/identity.json', 'sha256': closure['sources']['identity']['sha256']}}
    flags = []
    for role in ('manifest', 'observation', 'identity'):
        flags += ['--resume-source-'+role, refs[role]['path'], '--resume-source-'+role+'-sha256', refs[role]['sha256']]
    flags += ['--resume-source-root', '/resume-prefix']
    plan = {'schema': 'neural_forge.k3_actual_NAS_closed72_record_plan.v1', 'identity': identity,
            'command_argv': argv+flags, 'next_layers': '73-84', 'mounts': mounts, 'image_id': identity['image'],
            'required_gpu_uuid': identity['gpu_uuid'], 'native_source_sha256': NATIVE_SHA,
            'closed72_continuity_sha256': identity['closed72_continuity_sha256'],
            'original_record_plan_sha256': prior['original_record_plan_sha256'],
            'dated_original_plan_provenance': prior['dated_original_plan_provenance'],
            'RO_resume_source_refs': refs, 'RO_resume_source_flags': flags,
            'current_native_software_and_inputs_require_actual_runtime_observation': True,
            'actual_runtime_admitted': False, 'actual_fitter_started': False, 'later_windows_admitted': False,
            'capability_accepted': False, 'training_eligible': False, 'numerical_targets_enforced': False,
            'artifact_scope':'unique chosen-filesystem73-84 outputs; four exact old originsRO',
            'created_utc': datetime.now(timezone.utc).isoformat()}
    plan_raw = encode(plan)
    runtime = {'schema': 'neural_forge.k3_owned_NAS73_84_immutable_runtime_plan.v1', 'identity': identity,
               'plan_path': '/control/recovery_plan73_84.json', 'plan_sha256': sha(plan_raw), 'current_mounts': mounts,
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
    raw_outputs = {'recovery_plan73_84.json': plan_raw, 'runtime_plan73_84.json': encode(runtime),
                   'ro_multi_origin_catalog.json': catalog_raw, 'ro_multi_origin_observation.json': observation_raw}
    out.mkdir(exist_ok=False)
    result = {name: write_raw(out/name, raw) for name, raw in raw_outputs.items()}
    print(json.dumps({'run_uuid': RUN, 'files': result, 'actual_runtime_admitted': False}, sort_keys=True))
def final_configs(args):
    spec, _ = read_ref({'path': str(args.inputs), 'sha256': args.inputs_sha256})
    require(spec['schema'] == 'neural_forge.k3_actual73_84_local_final_config_inputs.v1'
            and spec['run_uuid'] == RUN, 'exact post-allocation config input schema required')
    bundle = Path(spec['source_bundle']).resolve()
    require(bundle.parent == LOCAL_OUTPUTS and bundle.is_dir() and not bundle.is_symlink(), 'own source-bound local bundle required')
    closure = decode(read_raw(bundle/'SOURCE_CLOSURE73_84.json', spec['source_closure_sha256']))
    verify_bound_bundle(bundle, closure)
    require(closure['selected_run_uuid'] == RUN and len(closure['sources']) == 73 and len(closure['scripts']) == 5,
            'whole exact73 metadata/five-script source closure required')
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
    closed = decode(read_raw(control/closure['sources']['closed72_continuity']['name'], identity['closed72_continuity_sha256']))
    observation_metadata(catalog, sha(raws['catalog']), observation, identity, mounts, closed)
    require(catalog['schema'] == 'chimera.k3_RO_multi_origin_prefix_fresh_NAS_output_source_route_review.v1'
            and observation['schema'] == 'chimera.k3_RO_multi_origin_prefix_separate_output_observation.v1'
            and plan['identity'] == runtime['identity'] == identity and plan['next_layers'] == '73-84'
            and plan['later_windows_admitted'] is False and plan['actual_fitter_started'] is False
            and plan['native_source_sha256'] == NATIVE_SHA and runtime['plan_sha256'] == sha(raws['recovery_plan'])
            and runtime['no_later_window_admission'] is True
            and plan['mounts'] == runtime['current_mounts'] == observation['current_mounts'] == mounts,
            'complete actual four-origin plan/catalog/observation/source joins differ')
    require(runtime['budget']['memory_bytes'] == CAP_BYTES and runtime['budget']['cpus'] == CAP_CPUS
            and runtime['budget']['swap_bytes'] == 0 and runtime['budget']['selected_new_output_write_bytes'] == WRITER_BYTES and runtime['budget']['selected_new_NAS_write_bytes'] == (WRITER_BYTES if identity['new_output_filesystem']=='NAS' else 0),
            'actual selected phase cap/22CPU/noSwap and whole180943168512 writer must be bound in current runtime plan')
    require(allocation['state'] == 'running' and allocation['owner'] == identity['owner']
            and allocation['name'] == identity['name'] and allocation['gpu_uuid'] == identity['gpu_uuid']
            and allocation['gpu_index'] == 6 and allocation['details']['image'] == identity['image']
            and allocation['details']['mounts'] == mounts and
            re.fullmatch(r'[0-9a-f]{64}', allocation['container_id']) is not None,
            'genuine normal new allocation/image/exact12mounts required')
    require(str(uuid.UUID(allocation['allocation_id'])) == allocation['allocation_id'], 'canonical actual allocation UUID required')
    flags = plan['RO_resume_source_flags']; refs = plan['RO_resume_source_refs']
    expected_names = {'catalog': 'ro_multi_origin_catalog.json', 'observation': 'ro_multi_origin_observation.json',
                      'recovery_plan': 'recovery_plan73_84.json', 'runtime_plan': 'runtime_plan73_84.json',
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
    driver = {'schema': 'neural_forge.k3_nas_closed72_worker_config.v1', 'identity': identity,
              'expected_allocator_name': allocation['name'], 'continuation_uuid': RUN,
              'window': [73,84], 'later_windows_admitted': False,
              'first_plan': {'path': SEALED36+'/inputs/original_record_plan.json', 'sha256': '17b4e172ac84c82dea494c7bef7e393f926a58e5fca67ece20358fb4e9a29a6e'},
              'baseline_receipt': {'path': '/recovery-source/ancestry/original_receipts/k3_dense_pass_receipt.json', 'sha256': 'c54e2fd0ade6fb8bbc9a111cb76fc5469abc1c197001b137e70f1c1156e976ff'},
              'prep_receipt': {'path': '/recovery-source/prep_result.json', 'sha256': 'bcc3876d5f79e84a5acaa90586ef9d03a84cee0f1781ac9f12cff17e902790ae'},
              'prior_continuity': {'path': SEALED72_MOUNTED+'/window_continuity.json', 'sha256': identity['closed72_continuity_sha256']},
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
    result = {'schema': 'neural_forge.k3_actual73_84_local_final_config_build.v1', 'run_uuid': RUN,
              'source_closure_sha256': spec['source_closure_sha256'], 'allocation_id': allocation['allocation_id'],
              'container_id': allocation['container_id'], 'files': {name: {'sha256': sha(raw), 'bytes': len(raw)} for name, raw in raw_outputs.items()},
              'actual_cgroup_selected_cap22CPU_noSwap_attestation_required_before_fit': True,
              'complete_source_freeze_and_fresh_owner_capacity_required': True, 'remote_or_runtime_actions': False}
    raw_outputs['FINAL_CONFIG_BUILD_RESULT.json'] = encode(result)
    out.mkdir(exist_ok=False)
    for name, raw in raw_outputs.items(): write_raw(out/name, raw)
    print(json.dumps(result, sort_keys=True))

def main():
    global RUN,COMMON_BYTES,CAP_BYTES,CAP_PROPOSAL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('sources','plans','configs'))
    parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();spec=decode(raw_ref({'path':str(args.inputs),'sha256':args.inputs_sha256}))
    RUN=spec['run_uuid'];require(isinstance(RUN,str) and str(uuid.UUID(RUN))==RUN and RUN!=PREVIOUS_RUN,'new actual61 UUID remains unresolved')
    selected()
    if args.phase=='sources':return bind(args)
    bundle=Path(spec['source_bundle']);closure=decode(read_raw(bundle/'SOURCE_CLOSURE73_84.json',spec['source_closure_sha256']))
    COMMON_BYTES=closure['source_binding']['common_NAS_floor_compare_bytes']
    return {'plans':plans_bind,'configs':final_configs}[args.phase](args)
if __name__=='__main__':main()
