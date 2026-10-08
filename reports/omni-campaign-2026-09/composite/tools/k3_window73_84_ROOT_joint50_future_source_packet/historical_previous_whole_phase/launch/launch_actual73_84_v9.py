"""Explicit normal next73-84 steps; root executes after authentic closed60.

No step runs at import. Sources is local only. Every later step uses the existing
server-manager control lane, normal allocator and six reviewed controls. Failed
operations retain their receipts; allocation/freeze/arm are never auto-retried.
"""
from __future__ import annotations
import argparse, asyncio, base64, contextlib, datetime, hashlib, importlib.util, io, json
import os, pathlib, shlex, subprocess, sys, uuid, zipfile, zlib

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
BIND_ROOT = HERE.parent
RECOVERY=pathlib.Path('C:\\Users\\cereal\\Documents\\Codex\\2026-10-05\\task-4\\owner_fold_policy_oct06\\native_executor\\k3_continuation\\recovery_oct07')
K3 = RECOVERY.parent
CONTROLS = BIND_ROOT/'controls'
RUN = None  # genuine future UUID is read from full-pinned actual72 input, never invented
BINDER=BIND_ROOT/'bind_actual73_84_runtime_v9.py'
BINDER_SHA='a99d5d6cca5878a71ca96a711d02305978ec29a6aa797cdd67b75143d6bde357'
LOCAL_OUTPUTS=pathlib.Path('D:/codex-owned-source-receipts/k3-later73-92-source-only/phase73_84')
BUNDLE = LOCAL_OUTPUTS/'source_bundle_launch_v1'
PLANS = LOCAL_OUTPUTS/'plans_launch_v1'
CONFIGS = LOCAL_OUTPUTS/'configs_launch_v1'
WORK = LOCAL_OUTPUTS/'launch_receipts_v1'
PYTHON = 'C:/Python314/python.exe'
WRITER = 180943168512
COMMON_BASE = 692026605568
HELPER_MANIFEST_SHA='5566e865a732394b4dfeaa4c6c6f132d9aad6a4792092e35dcaaaa6f0826c75b'
CONTROL_PINS={'normal_caller_ready73_remote.py': '1d9718f3d69bddc1adb885ed192eead2a15806f9dc0d02bb4f6922f869f594b3', 'normal_cap_arm_bound_remote.py': '17d0a9a85d1fbacc82974e7e6f35d31ca28ffab8da13681b10a8baca30c2ad8a', 'normal_freeze73_remote.py': '41c8d6cd851d99e824c78d2367d04d639e9126470b79c4e0be6c6df592857dfb', 'normal_precheck12_remote.py': 'c8c0155f21f3a073147167940025f6e51bd2acae389cebc41d16c2c65fbeb758', 'normal_stage_assets73_remote.py': '8853e897a12d525f8a1c61ea75612179f587c852fcc4669f1240c52ff074cc65', 'stage_owned_bound_source_remote.py': '109fd4792e92f639fcd18594f70ece15dced06e08c297a9ca958c6515378ce78'}
LOCAL_DEPENDENCIES = {
    K3/'infra.py':'985468b35c10d95eacd9cd9a2e5da9691a127f7d30bf5b4bfde384964d730a26',
    K3/'read_ledger_recovery.py':'7cfc66965a6b24a1b2e86025379c47131dcdb7a2e69b4397525840c9dfca1594',
    K3/'initial_snapshot.json':'48f4c5de740a8bffaa924680472601a0add04b522478f242c3cf35681ce98f0d',
    pathlib.Path('C:/Users/cereal/hermes.py'):'2dca0c8d91be3efb4de0da5cb2e0ed04c40a23b46a6c1186a815931a2138c723',
    pathlib.Path('D:/relay/worktrees/local-hermes-collaboration/best/local-hermes/hermes.py'):'97b58088b555b4669b94c15649d370dfd4dacd28dd6e3a430ad417522424117e',
}


def require(ok, message):
    if ok is not True: raise ValueError(message)


def sha(path):
    with pathlib.Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def encode(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def load(path): return json.loads(pathlib.Path(path).read_bytes())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def ref(path): return {'path':str(pathlib.Path(path).resolve()),'sha256':sha(path)}


def write(path, value):
    raw = encode(value)
    with pathlib.Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    require(pathlib.Path(path).read_bytes()==raw,'exclusive local receipt readback differs')
    return ref(path)


def fresh(stamp):
    instant=datetime.datetime.fromisoformat(stamp.replace('Z','+00:00'))
    require(instant.tzinfo is not None and 0 <= (datetime.datetime.now(datetime.timezone.utc)-instant).total_seconds() <=150,
            'fresh capacity/facts required within150seconds')


def pinned_module(path, expected, name):
    require(path.is_file() and not path.is_symlink() and sha(path)==expected,'selected local source differs: '+path.name)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def binder(): return pinned_module(BINDER,BINDER_SHA,'selected_actual49_binder')


def bound():
    emitted=previous('sources')
    require(emitted['source_bundle']==str(BUNDLE)
        and ref(BUNDLE/'SOURCE_CLOSURE73_84.json')==emitted['source_closure']
        and ref(BUNDLE/'SOURCE_BINDING_RESULT.json')==emitted['source_binding'],
        'original source-completion closure and binding raw pins changed')
    b=binder();b.RUN=RUN; closure=load(BUNDLE/'SOURCE_CLOSURE73_84.json')
    b.verify_bound_bundle(BUNDLE,closure)
    identity=load(BUNDLE/'production_control/identity.json');mounts=load(BUNDLE/'mounts.json')
    require(identity['run_uuid']==RUN and len(mounts)==12,'actual source-bound identity/mounts differ')
    return identity,closure,mounts


def infrastructure():
    for path, expected in LOCAL_DEPENDENCIES.items():require(sha(path)==expected,'existing local infrastructure source differs')
    sys.path.insert(0,str(K3));import infra
    # One exclusive workspace receipt leaf. The root's real native context is
    # checked by existing infra.native_session when allocation is requested.
    infra.ROOT=WORK
    return infra


def packed_control(infra, path, variables, label, *, capacity=False, timeout=25):
    # Wire-only repair for Linux MAX_ARG_STRLEN. Logical variables, source bytes,
    # SHA guards, installed control connection and operation order are unchanged.
    payload=path.read_bytes()
    logical=json.dumps(variables).encode('utf-8')
    packed=zlib.compress(logical)
    require(zlib.decompress(packed)==logical,'full variable wire roundtrip differs')
    code=('import base64,json,zlib;globals().update(json.loads(zlib.decompress(base64.b64decode('
          +repr(base64.b64encode(packed).decode())+'))));'
          +'exec(compile(base64.b64decode('+repr(base64.b64encode(payload).decode())
          +'),"<owned-next73-existing-control>","exec"))')
    require(len(code.encode('utf-8'))<98304,'bounded packed control argument still exceeds budget')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        ssh=infra.ssh_connection()
        try:status,out,err=ssh.execute(shlex.join(['python3','-B','-c',code]),timeout=timeout)
        finally:ssh.close()
    receipt=infra.record(label,{'at':now(),'source_sha256':sha(path),'variables':variables,
        'exit_code':status,'stdout':out,'stderr':err,'normal_control_lane':True,
        'wire_encoding':'zlib.base64.full-json-variables.v1',
        'variable_bytes':len(logical),'variable_sha256':hashlib.sha256(logical).hexdigest(),
        'packed_variable_bytes':len(packed),'code_argument_bytes':len(code.encode('utf-8'))})
    require(status in ((0,2) if capacity else (0,)),
            'actual existing-control operation failed; receipt '+str(receipt))
    result=json.loads(out)
    if capacity:
        require((status==0)==(result['admitted'] is True),'actual preflight exit/admission differs')
    return result,receipt,status


def control(infra, name, variables, label, timeout=25):
    path=CONTROLS/name
    require(sha(path)==CONTROL_PINS[name],'reviewed normal control source changed')
    result,receipt,status=packed_control(infra,path,variables,label,timeout=timeout)
    return {'result':result,'receipt':ref(receipt)}


def own_helper(infra, name, variables, label, capacity=False):
    manifest=HERE/'LAUNCH_HELPER_SOURCE_PINS_V9.json'
    require(sha(manifest)==HELPER_MANIFEST_SHA,'exact helper source manifest changed')
    pins=load(manifest)
    path=HERE/name;require(sha(path)==pins[name]['sha256'],'selected additive helper source differs')
    result,receipt,status=packed_control(infra,path,variables,label,capacity=capacity)
    response={'result':result,'receipt':ref(receipt)}
    if capacity:response['exit_code']=status
    return response


def previous(step):
    path=WORK/(step+'-complete.json')
    require(path.is_file(),'previous exact step receipt required: '+step)
    value=load(path);require(value['run_uuid']==RUN and value['step']==step,'milestone run/step differs')
    return value['result']


def finish(step, result):
    value={'schema':'neural_forge.k3_actual73_84_normal_local_step.v1','at':now(),
           'run_uuid':RUN,'step':step,'local_entrypoint_sha256':sha(pathlib.Path(__file__)),
           'binder_sha256':BINDER_SHA,'result':result}
    name=step+'-'+result['attempt_uuid']+'-complete.json' if step=='admit' else step+'-complete.json'
    pin=write(WORK/name,value)
    print(json.dumps({'step':step,'receipt':pin,'result':result},sort_keys=True))


def local_binder(phase, inputs, output):
    # Keep the exact manifest bytes and CLI result, including ordinary refusal.
    command=[PYTHON,'-B',str(BINDER),phase,'--inputs',str(inputs),'--inputs-sha256',sha(inputs),
             '--output-directory',str(output)]
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=150)
    if WORK.exists():write(WORK/(phase+'-binder-invocation.json'),{'at':now(),'argv':command,
        'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    require(run.returncode==0,'local exact binder refused: '+run.stderr[-3000:])
    return json.loads(run.stdout)


def source_step(args):
    require(args.source_inputs is not None and args.source_inputs_sha256 is not None,'authentic nine-input manifest and fullSHA required')
    require(sha(args.source_inputs)==args.source_inputs_sha256,'authentic nine-input manifest rawSHA differs')
    require(not LOCAL_OUTPUTS.exists() and not LOCAL_OUTPUTS.is_symlink(),'exclusive owned D local output namespace already exists')
    require(LOCAL_OUTPUTS.parent.is_dir() and not LOCAL_OUTPUTS.parent.is_symlink(),'existing owned D receipts parent required')
    LOCAL_OUTPUTS.mkdir(exist_ok=False)
    WORK.mkdir(exist_ok=False)
    write(WORK/'source_inputs_pin.json',ref(args.source_inputs))
    result=local_binder('sources',args.source_inputs,BUNDLE)
    identity_raw=(BUNDLE/'production_control/identity.json').read_bytes()
    with (WORK/'identity.json').open('xb') as stream:
        stream.write(identity_raw);stream.flush();os.fsync(stream.fileno())
    require((WORK/'identity.json').read_bytes()==identity_raw,'ledger exact identity readback differs')
    finish('sources',{'source_bundle':str(BUNDLE),'source_closure':ref(BUNDLE/'SOURCE_CLOSURE73_84.json'),
                      'source_binding':ref(BUNDLE/'SOURCE_BINDING_RESULT.json'),'binder_result':result})


def packet(name, files):
    path=WORK/name;members={}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for dest,source in sorted(files.items()):
            raw=pathlib.Path(source).read_bytes();archive.writestr(dest,raw)
            members[dest]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return path,members


def stage_step():
    previous('sources');identity,closure,_=bound();infra=infrastructure()
    roots=own_helper(infra,'create_next73_roots_ROOT_serial_v9_remote.py',{'I':identity},'next73-exclusive-roots')
    expected=closure['source_binding']['production_file_pins']
    archive,members=packet('bound_source73_84.zip',{name:BUNDLE/name for name in expected})
    require(members==expected,'whole bound code/control stage roster differs')
    upload=infra.upload(archive,identity['runtime_root']+'/'+archive.name)
    stage=control(infra,'stage_owned_bound_source_remote.py',{'I':identity,'ARCHIVE_NAME':archive.name,
        'ARCHIVE_SHA':sha(archive),'MEMBERS':members},'next73-whole-bound-source-stage')
    # These observation inputs are outside the immutable source-bundle roster.
    # Upload only to the fresh, empty observer_source directory created above.
    completed=load(BUNDLE/'production_control'/closure['sources']['completed61_phase_source']['name'])
    parent_ref=completed['previous61_complete_provenance']['inputs']['catalog']
    observer_assets={
        'collect_actual73_84_multi_origin.py':BUNDLE/'observer_source/collect_actual73_84_multi_origin.py',
        'parent61_catalog.json':pathlib.Path(parent_ref['path']),
        'mounts.json':BUNDLE/'mounts.json',
    }
    require(sha(observer_assets['collect_actual73_84_multi_origin.py'])==closure['source_binding']['observer_source_pin']['sha256'],'exact UUID-bound metadata-only collector differs')
    require(sha(observer_assets['parent61_catalog.json'])==parent_ref['sha256']==identity['parent_catalog_sha256'],'actual complete parent49 catalog differs')
    uploads={name:ref(infra.upload(path,identity['runtime_root']+'/observer_source/'+name)) for name,path in observer_assets.items()}
    finish('stage',{'roots':roots,'archive':ref(archive),'archive_upload':ref(upload),'source_stage':stage,
                    'observer_assets':{n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in observer_assets.items()},'observer_uploads':uploads})


def download_exact(infra, remote, local, expected):
    require(not local.exists(),'exclusive downloaded receipt already exists')
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        ssh=infra.ssh_connection()
        try:ok=ssh.download_file(remote,str(local.parent))
        finally:ssh.close()
    require(ok and local.is_file() and sha(local)==expected,'full raw remote metadata download differs')
    return ref(local)


def observe_step():
    staged=previous('stage');identity,closure,_=bound();infra=infrastructure()
    result=own_helper(infra,'execute_metadata_collector73_ROOT_serial_v9_remote.py',{'I':identity,
        'ASSETS':staged['observer_assets'],'IDENTITY_SHA':closure['sources']['identity']['sha256'],
        'MOUNTS_SHA':closure['source_binding']['mounts_sha256'],'COLLECTOR_PIN':staged['observer_assets']['collect_actual73_84_multi_origin.py']},'next73-two-origin-metadata-only-observation')
    folder=WORK/'actual_observation';folder.mkdir(exist_ok=False)
    refs={}
    for role in ('catalog','observation','collection_receipt'):
        row=result['result'][role];local=folder/pathlib.PurePosixPath(row['path']).name
        refs[role]=download_exact(infra,row['path'],local,row['sha256'])
    finish('observe',{'collection':result,'inputs':refs})


def plans_step():
    observed=previous('observe');_,closure,_=bound()
    path=WORK/'plan_binding_inputs.json'
    write(path,{'schema':'neural_forge.k3_actual73_84_local_plan_binding_inputs.v1','run_uuid':RUN,
        'source_bundle':str(BUNDLE),'source_closure_sha256':sha(BUNDLE/'SOURCE_CLOSURE73_84.json'),
        'inputs':{role:observed['inputs'][role] for role in ('catalog','observation')}})
    result=local_binder('plans',path,PLANS)
    finish('plans',{'binder_result':result,'plan_directory':str(PLANS),
        'files':{p.name:ref(p) for p in PLANS.iterdir() if p.is_file()}})


def prospective_step():
    previous('plans');identity,closure,_=bound();infra=infrastructure();attempt=str(uuid.uuid4())
    require(not(WORK/'actual_allocation.json').exists(),'prospective refresh cannot omit an existing own allocation')
    # Source owner prospective request, not a fabricated paused/live Driver publication.
    result=own_helper(infra,'execute_prospective73_ROOT_serial_v9_remote.py',{'I':identity,'CLOSURE':closure,
        'CLOSURE_SHA':sha(BUNDLE/'SOURCE_CLOSURE73_84.json'),'ATTEMPT':attempt},'next73-source-owner-fresh-prospective',capacity=True)
    pin=result['result'];folder=WORK/('capacity_'+attempt);folder.mkdir(exist_ok=False)
    capacity=download_exact(infra,pin['path'],folder/pathlib.PurePosixPath(pin['path']).name,pin['sha256'])
    body=load(capacity['path'])
    request=download_exact(infra,identity['runtime_root']+'/capacity_calls/'+attempt+'/source-owner-prospective.json',
        folder/'source-owner-prospective.json',body['request_pin']['sha256'])
    facts=download_exact(infra,identity['runtime_root']+'/capacity_calls/'+attempt+'/'+body['facts_pin']['name'],
        folder/body['facts_pin']['name'],body['facts_pin']['sha256'])
    require(body['mode']=='prospective' and body['identity']==identity and body['attempt_uuid']==attempt
        and body['disk_components']['K3_selected_window_remaining_bytes']==WRITER
        and body['NAS_required_components']['current_common_guard_bytes_once']==COMMON_BASE+(WRITER if identity['new_output_filesystem']=='NAS' else 0),
        'actual source-owner capacity writer/identity/guard differs')
    finish('admit',{'attempt_uuid':attempt,'remote_result':result,'capacity':capacity,
        'remote_capacity_pin':pin,'request':request,'facts':facts,'admitted':body['admitted'],
        'driver_emitted_request':False,'normal_allocation_started':False})
    return 0 if body['admitted'] else 2


def baseline():
    snap=load(K3/'initial_snapshot.json')
    return {key:snap[key] for key in ('services','installed_pins')}


def pg_ledger_json_value(value):
    if isinstance(value,uuid.UUID):return str(value)
    if isinstance(value,datetime.datetime):return value.isoformat()
    raise TypeError('unsupported raw PostgreSQL ledger type: '+type(value).__name__)


def allocate_step(args):
    require(args.admission_receipt is not None and args.admission_receipt_sha256 is not None,
        'select the exact fresh immutable admission receipt and fullSHA')
    admission_path=args.admission_receipt.resolve()
    require(admission_path.parent==WORK and not admission_path.is_symlink()
        and sha(admission_path)==args.admission_receipt_sha256,'exact own immutable admission receipt differs')
    admitted_receipt=load(admission_path)
    require(admitted_receipt['schema']=='neural_forge.k3_actual73_84_normal_local_step.v1'
        and admitted_receipt['run_uuid']==RUN and admitted_receipt['step']=='admit'
        and admitted_receipt['binder_sha256']==BINDER_SHA,'actual selected next-window admission receipt schema differs')
    admitted=admitted_receipt['result'];identity,closure,mounts=bound();infra=infrastructure()
    require(admitted['admitted'] is True and not (WORK/'actual_allocation.json').exists(),'fresh true admission/no prior allocation required')
    binding,connect=infra.native_session(['connect'])
    require({key:binding[key] for key in identity['native_binding']}==identity['native_binding'],'actual same-context source binding differs')
    from read_ledger_recovery import observe
    ledger=json.loads(json.dumps(asyncio.run(observe(identity['name'])),default=pg_ledger_json_value,allow_nan=False));ledger_ref=write(WORK/'allocation_ledger.json',ledger)
    precheck=control(infra,'normal_precheck12_remote.py',{'I':identity,'BASELINE':baseline(),
        'LEDGER':ledger,'MOUNTS':mounts},'next73-normal-allocator-readonly-precheck')
    readback=own_helper(infra,'read_fresh_prospective73_ROOT_serial_v9_remote.py',{'I':identity,'CLOSURE':closure,
        'CLOSURE_SHA':sha(BUNDLE/'SOURCE_CLOSURE73_84.json'),'CAPACITY_PIN':admitted['remote_capacity_pin'],
        'ATTEMPT':admitted['attempt_uuid']},'next73-fresh-whole-capacity-readback-before-normal-allocation')
    body=load(admitted['capacity']['path']);fresh(body['at']);fresh(body['facts_at'])
    argv=[PYTHON,'-B','C:/Users/cereal/hermes.py','run','--server','ai','--gpu',identity['gpu_uuid'],
        '--owner',identity['owner'],'--name',identity['name'],'--image',identity['image']]
    for row in mounts:argv += ['--mount-ro' if row['read_only'] else '--mount-rw',row['source'],row['target']]
    for arg in ('/usr/local/bin/python','-B','/control/recovery_boot.py'):argv += ['--runtime-arg='+arg]
    write(WORK/'normal_allocator_invocation.json',{'at':now(),'argv':argv,'binding_receipt':ref(connect),
        'ledger':ledger_ref,'precheck':precheck,'actual_capacity_readback':readback,
        'selected_immutable_admission_receipt':ref(admission_path),
        'boot_waits_existing_barrier':True,'operating_caps_apply_before_barrier':True})
    process=subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=55)
    write(WORK/'normal_allocator_raw_result.json',{'at':now(),'argv':argv,'exit_code':process.returncode,
        'stdout':process.stdout,'stderr':process.stderr})
    require(process.returncode==0,'normal allocator failed; raw result preserved, do not duplicate blindly')
    allocation=json.loads(process.stdout);a_pin=write(WORK/'actual_allocation.json',allocation)
    require(allocation['owner']==identity['owner'] and allocation['name']==identity['name']
        and allocation['gpu_uuid']==identity['gpu_uuid'] and allocation['details']['mounts']==mounts
        and allocation['details']['image']==identity['image'] and allocation['state']=='running','actual normal allocation differs')
    capped=control(infra,'normal_cap_arm_bound_remote.py',{'I':identity,'A':allocation,'BASELINE':baseline(),'ARM':False},'next73-actualselected-cap22CPU-noSwap-waiting-boot')
    finish('allocate',{'allocation':a_pin,'initial_cap_attestation':capped,'native_fitter_started':False})


def configs_step():
    allocated=previous('allocate');plans=previous('plans');bound()
    files=plans['files'];path=WORK/'final_config_inputs.json'
    write(path,{'schema':'neural_forge.k3_actual73_84_local_final_config_inputs.v1','run_uuid':RUN,
        'source_bundle':str(BUNDLE),'source_closure_sha256':sha(BUNDLE/'SOURCE_CLOSURE73_84.json'),
        'inputs':{'allocation':allocated['allocation'],'recovery_plan':files['recovery_plan73_84.json'],
            'runtime_plan':files['runtime_plan73_84.json'],'catalog':files['ro_multi_origin_catalog.json'],
            'observation':files['ro_multi_origin_observation.json']}})
    result=local_binder('configs',path,CONFIGS)
    finish('configs',{'binder_result':result,'config_directory':str(CONFIGS)})


def freeze_step():
    previous('configs');allocated=previous('allocate');observed=previous('observe')
    identity,_,_=bound();infra=infrastructure();allocation=load(allocated['allocation']['path'])
    files={p.name:p for p in CONFIGS.iterdir() if p.is_file() and p.name!='FINAL_CONFIG_BUILD_RESULT.json'}
    files['recovery_allocation.json']=pathlib.Path(allocated['allocation']['path'])
    files['ro_multi_origin_collection_receipt.json']=pathlib.Path(observed['inputs']['collection_receipt']['path'])
    archive,members=packet('allocation_bound_final_configs73_84.zip',{'production_control/'+name:path for name,path in files.items()})
    uploaded=infra.upload(archive,identity['runtime_root']+'/'+archive.name)
    staged=control(infra,'normal_stage_assets73_remote.py',{'I':identity,'ARCHIVE_NAME':archive.name,
        'ARCHIVE_SHA':sha(archive),'MEMBERS':members},'next73-allocation-bound-config-stage')
    config_pins={name:sha(path) for name,path in files.items()}
    host_argv=['/usr/bin/python3','-B',identity['runtime_root']+'/production_control/k3_capacity_host_loop_v1.py',
        '--config',identity['runtime_root']+'/production_control/hostcapacity_caller_config.json',
        '--config-sha256',config_pins['hostcapacity_caller_config.json']]
    frozen=control(infra,'normal_freeze73_remote.py',{'I':identity,'A':allocation,
        'CONFIG_PINS':config_pins,'HOST_ARGV':host_argv},'next73-whole-source-freeze-and-selected-caller-start')
    ready=control(infra,'normal_caller_ready73_remote.py',{'I':identity,'FREEZE':frozen['result']},'next73-actual-caller-serving-readback',timeout=15)
    finish('freeze',{'config_archive':ref(archive),'upload':ref(uploaded),'staged':staged,'freeze':frozen,'caller_ready':ready})


def arm_step():
    frozen=previous('freeze');allocated=previous('allocate');identity,_,_=bound();infra=infrastructure()
    result=control(infra,'normal_cap_arm_bound_remote.py',{'I':identity,'A':load(allocated['allocation']['path']),
        'BASELINE':baseline(),'ARM':True,'CONTROL_FREEZE':frozen['freeze']['result']['control_files']},
        'next73-ownselected-cap-noSwap-exact-source-barrier-arm')
    finish('arm',{'physical_arm':result,'fresh_actual_Driver_runtime_and_capacity_before_fit_still_required':True,
                  'actual_native_fitter_started_claimed':False,'later_windows_admitted':False})


def main():
    global RUN
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('sources','stage','observe','plans','admit','allocate','configs','freeze','arm'))
    parser.add_argument('--source-inputs',type=pathlib.Path);parser.add_argument('--source-inputs-sha256')
    parser.add_argument('--admission-receipt',type=pathlib.Path);parser.add_argument('--admission-receipt-sha256')
    args=parser.parse_args();require(sha(BINDER)==BINDER_SHA,'exact reviewed V4 binder required')
    if args.step=='sources':
        require(args.source_inputs is not None and args.source_inputs_sha256 is not None and sha(args.source_inputs)==args.source_inputs_sha256,'actual72 complete source inputs and fullSHA required')
        RUN=load(args.source_inputs)['run_uuid'];require(isinstance(RUN,str) and str(uuid.UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139','actual new73 UUID remains unresolved')
        source_step(args);return 0
    RUN=load(WORK/'identity.json')['run_uuid'];require(str(uuid.UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139','actual bound new73 UUID required')
    require(WORK.is_dir() and not WORK.is_symlink() and (args.step=='admit' or not(WORK/(args.step+'-complete.json')).exists()),'exclusive not-yet-completed step required')
    operations={'stage':stage_step,'observe':observe_step,'plans':plans_step,'admit':prospective_step,
        'allocate':lambda:allocate_step(args),'configs':configs_step,'freeze':freeze_step,'arm':arm_step}
    try:return operations[args.step]() or 0
    except Exception as error:
        write(WORK/('failure-'+args.step+'-'+str(uuid.uuid4())+'.json'),{'at':now(),'step':args.step,
            'error_type':type(error).__name__,'error':str(error),'no_automatic_action_retry':True})
        raise


if __name__=='__main__':raise SystemExit(main())
