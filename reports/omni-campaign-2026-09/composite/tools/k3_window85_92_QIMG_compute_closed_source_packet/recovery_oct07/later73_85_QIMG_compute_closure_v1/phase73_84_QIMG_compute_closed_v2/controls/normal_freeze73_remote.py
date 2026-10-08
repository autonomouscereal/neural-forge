"""Freeze actual owned code/control bytes and start the selected readonly caller."""
import datetime,hashlib,json,pathlib,subprocess,sys
P=pathlib.Path;root=P(I['runtime_root']);control=root/'production_control';out=P(I['output_root']);code=root/'production_code'
def sha(p):
    with P(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name,expected in CONFIG_PINS.items():assert sha(control/name)==expected,name
config=json.loads((control/'continuation_config.json').read_text());assert config['identity']==I and config['expected_allocator_name']==A['name']
for name,h in config['source_pins'].items():assert sha(control/name)==h,name
tool=code/'reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py';assert sha(tool)==config['native_tool_sha256']
assert sha(tool.parent/'k3_recovery_ancestry.py')==config['source_pins']['k3_recovery_ancestry.py']
for basename in ('ro_resume_paths.py','ro_resume_runtime.py'):assert sha(tool.parent/basename)==config['source_pins'][basename]
for role in ('recovery_packet','recovery_plan','resume_source_manifest','resume_source_observation','resume_source_identity'):
    ref=config[role];assert sha(control/P(ref['path']).name)==ref['sha256'],role
assert json.loads((control/'recovery_allocation.json').read_text())==A
assert not (out/'continuation.ready').exists()
for directory in (out/'capacity_requests',out/'capacity_responses',root/'capacity_calls'):directory.mkdir(exist_ok=True)
assert not any(p.is_symlink() for folder in (control,code) for p in folder.rglob('*'))
control_files={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
code_files={p.relative_to(code).as_posix():sha(p) for p in code.rglob('*') if p.is_file()}
freeze={'schema':'neural_forge.k3_NAS73_84_execution_source_freeze.v1','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':I,'allocation_id':A['allocation_id'],'container_id':A['container_id'],'control_files':control_files,'code_files':code_files,'immutable_source_ro_mounts':True,'old_source_unchanged':True,'native_source_sha256':sha(tool),'config_sha256':sha(control/'continuation_config.json'),'source_ancestry_sha256':config['recovery_packet']['sha256'],'resource_caller_config_sha256':config['capacity_caller']['host_caller_config_sha256'],'selected_runtime_plan_sha256':CONFIG_PINS['runtime_plan73_84.json'],'capability_accepted':False}
with (out/'execution_source_freeze.json').open('x') as f:json.dump(freeze,f,indent=2,sort_keys=True);f.write('\n')
with (out/'capacity_host_caller.log').open('xb') as log:
    process=subprocess.Popen(HOST_ARGV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
started={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':process.pid,'argv':HOST_ARGV,'source_freeze_sha256':sha(out/'execution_source_freeze.json'),'read_only_resource_actions':True,'purpose':'fresh own natural window capacity decisions only'}
with (out/'capacity_host_caller_launch.json').open('x') as f:json.dump(started,f,indent=2);f.write('\n')
print(json.dumps({'ok':True,'control_files':control_files,'code_files':code_files,'source_freeze_sha256':sha(out/'execution_source_freeze.json'),'host_caller_pid':process.pid,'host_caller_argv':HOST_ARGV}))
