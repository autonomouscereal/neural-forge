"""Read fresh actual prospective request/facts/capacity immediately before normal allocation."""
import datetime, hashlib, json, uuid
from pathlib import Path

RUN=I['run_uuid']
assert str(__import__('uuid').UUID(RUN))==RUN and RUN!='03603908-570d-4c90-96cb-ee3a63a9a139'
def require(ok,message):
    if ok is not True:raise ValueError(message)
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(path,expected,size=None):
    require(path.is_file() and not path.is_symlink() and path.resolve()==path and path.stat().st_size<=64<<20,'canonical bounded actual metadata required')
    before=path.stat();raw=path.read_bytes();after=path.stat()
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'actual metadata changed while reading')
    require(hashlib.sha256(raw).hexdigest()==expected and (size is None or len(raw)==size),'full actual metadata SHA/size differs')
    return json.loads(raw)
def fresh(at):
    instant=datetime.datetime.fromisoformat(at.replace('Z','+00:00'))
    require(instant.tzinfo is not None and 0 <= (datetime.datetime.now(datetime.timezone.utc)-instant).total_seconds()<=150,'actual capacity/facts age exceeds150seconds')
require(I['run_uuid']==RUN and str(uuid.UUID(ATTEMPT))==ATTEMPT,'exact own run/attempt required')
require(hashlib.sha256(canonical(CLOSURE)).hexdigest()==CLOSURE_SHA,'whole exact source closure differs')
require(len(CLOSURE['sources'])==71 and len(CLOSURE['scripts'])==5,'whole exact71/five source roster required')
root=Path(I['runtime_root']);control=root/'capacity_calls'/ATTEMPT;base=root/'production_control'
require(root==Path('/home/cereal/omni-campaign/runtime/k3-nas-window61-72-oct07-'+RUN),'exact own runtime required')
sources={role:{key:pin[key] for key in ('name','sha256')} for role,pin in CLOSURE['sources'].items()}
for folder in (control,base):
    for pin in [*CLOSURE['sources'].values(),*CLOSURE['scripts'].values()]:
        path=folder/pin['name'];require(path.stat().st_size==pin['bytes'],'exact staged byte size differs')
        before=path.stat();raw=path.read_bytes();after=path.stat()
        require(not path.is_symlink() and path.resolve()==path and hashlib.sha256(raw).hexdigest()==pin['sha256']
            and (before.st_ino,before.st_mtime_ns,before.st_size)==(after.st_ino,after.st_mtime_ns,after.st_size),
            'current complete staged source bytes differ')
capacity_path=control/('capacity-'+ATTEMPT+'.json')
require(CAPACITY_PIN['path']==str(capacity_path),'actual capacity path differs')
capacity=read(capacity_path,CAPACITY_PIN['sha256'])
request_path=control/'source-owner-prospective.json'
require(capacity['request_pin']['name']==request_path.name,'source-owner request name differs')
request=read(request_path,capacity['request_pin']['sha256'])
require(request_path.with_suffix('.sha256').read_bytes()==(capacity['request_pin']['sha256']+'\n').encode(),'full immutable request sidecar differs')
require(request['schema']=='neural_forge.k3_owned_capacity_preflight_request.v2' and request['mode']=='prospective'
    and request['attempt_uuid']==ATTEMPT and request['sources']==sources and request['remaining_copy_bytes']==0
    and request['own_remaining_write_bytes']==166556705792 and request['driver_emitted_request'] is False
    and request['purpose']=='SOURCE_OWNER_NEXT61_72_PROSPECTIVE'
    and 'owned_host_caller' not in request and 'owned_prospective_supervisor' not in request,
    'honest actual source-owner prospective contract differs')
require(request['request_boundary']=={'purpose':'SOURCE_OWNER_NEXT61_72_PROSPECTIVE','next_window':'61-72',
    'request_boundary_id':ATTEMPT,'source_closure_sha256':CLOSURE_SHA,'driver_emitted_request':False},'whole boundary differs')
require(capacity['schema']=='neural_forge.k3_owned_launch_capacity.v2' and capacity['mode']=='prospective'
    and capacity['identity']==I and capacity['binding']==I['native_binding'] and capacity['attempt_uuid']==ATTEMPT
    and capacity['request_boundary']==request['request_boundary'] and capacity['source_pins']==sources
    and capacity['admitted'] is True and all(capacity[key] is True for key in ('memory_admitted','cpu_admitted','disk_admitted'))
    and capacity['actual_locked_admission_completed'] is False and capacity['provider_or_allocator_mutations'] is False
    and capacity['model_payload_reads']==0 and capacity['overall92_admitted'] is False,
    'actual fresh whole source capacity admission required')
require(capacity['disk_components']['remaining_copy_bytes']==0
    and capacity['disk_components']['K3_selected_window_remaining_bytes']==166556705792
    and capacity['ROOT_serial_source_scope_once']['RAM_bytes']==17179869184
    and capacity['ROOT_serial_source_scope_once']['CPU_threads']==4
    and capacity['ROOT_serial_source_scope_once']['new_ROOT_write_bytes']==10737418240
    and capacity['ROOT_serial_source_scope_once']['source_pin']==request['sources']['ROOT_current_serial_scope']
    and capacity['QIMG_pending_scope_once']['RAM_bytes']==0 and capacity['QIMG_pending_scope_once']['cpu_threads']==0
    and capacity['QIMG_native_compute_closure_proof']['fresh_holder_absence_checked_by_classify'] is True
    and capacity['QIMG_native_compute_closure_proof']['normal_release_sha256']=='6ddcd4815040a345b12e979df272633e09593ad9561dec5a90df95c4036e0b63'
    and capacity['disk_components']['QIMG_post_fold_remaining_Lexar_bytes']==25769803776
    and capacity['root_disk_components']['QIMG_once_bytes']==2147483648
    and capacity['ROOT_joint50_source_scope_once']['owner_adoption_sha256']=='e886d40173bd3a02065946f9f60a49d89356d7894334e8d5af26369cdf81930f'
    and capacity['ROOT_joint50_source_scope_once']['source_pin']==request['sources']['ROOT_current_joint50_serial_future']
    and capacity['root_disk_components']['ROOT_joint50_capture_once_bytes']==2147483648
    and capacity['disk_components']['ROOT_joint50_fold_Lexar_once_bytes']==30064771072
    and capacity['ROOT_joint50_source_scope_once']['additional_RAM_bytes']==0
    and capacity['ROOT_joint50_source_scope_once']['additional_CPU_threads']==0
    and capacity['disk_components']['K3_write_destination']==I['native_working_output_root']
    and capacity['disk_components']['K3_NAS_qualification_reserve_bytes']==0
    and capacity['disk_components']['K3_NAS_qualification_embedded_in_whole_writer_bytes']==(536870912 if I['new_output_filesystem']=='NAS' else 0)
    and capacity['disk_components']['K3_output_IO_qualification_embedded_in_whole_writer_bytes']==536870912
    and capacity['disk_components']['K3_write_filesystem']==I['new_output_filesystem']
    and capacity['NAS_required_components']['current_common_guard_bytes_once']==692026605568+(166556705792 if I['new_output_filesystem']=='NAS' else 0)
    and capacity['NAS_required_components']['current_common_guard_includes_K3_next61_72_writer_once']==(I['new_output_filesystem']=='NAS'),
    'whole writer/qualification/common floor must be funded once')
facts_pin=capacity['facts_pin'];facts=read(control/facts_pin['name'],facts_pin['sha256'])
require(facts_pin['name']=='foreign_classification_facts-'+ATTEMPT+'.json'
    and facts['schema']=='neural_forge.k3_recovery_foreign_facts.v1' and facts['run_uuid']==RUN
    and facts['binding']==I['native_binding'] and facts['observation_uuid']==ATTEMPT
    and facts['at']==capacity['facts_at'] and facts['read_only'] is True and facts['model_payload_reads']==0
    and facts['provider_or_allocator_mutations'] is False
    and facts['identity_sha256']==sources['identity']['sha256']
    and facts['binding_sha256']==sources['binding']['sha256'],'fresh full exact facts identity differs')
for stamp in (request['at'],capacity['at'],capacity['facts_at'],facts['at']):fresh(stamp)
require(capacity['own_readonly_call_chain']['source']=={'path':str(control/CLOSURE['scripts']['preflight']['name']),
    'sha256':CLOSURE['scripts']['preflight']['sha256']}
    and capacity['own_readonly_call_chain']['shared_session_or_ancestry_wildcard'] is False,
    'real source-bound SSH ancestry required')
print(json.dumps({'ok':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'capacity_pin':CAPACITY_PIN,'source_closure_sha256':CLOSURE_SHA,'request_pin':capacity['request_pin'],
    'facts_pin':facts_pin,'facts_at':capacity['facts_at'],'RAM_required_bytes':capacity['RAM_required_bytes'],
    'CPU_required_threads':capacity['CPU_required_threads'],'remaining_disk_required_bytes':capacity['remaining_disk_required_bytes'],
    'source_owner_request':True,'driver_emitted_request':False,'normal_locked_admission_not_yet_claimed':True}))
