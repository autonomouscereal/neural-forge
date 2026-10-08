import base64,hashlib,json,pathlib,subprocess,zlib
P=pathlib.Path
for path,h in BASELINE['installed_pins'].items():
    with P(path).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h,path
obj=json.loads(subprocess.check_output(['docker','inspect',A['container_id']],text=True,timeout=15))[0]
assert obj['Id']==A['container_id'] and obj['Image']==I['image']
labels=obj['Config']['Labels']
assert labels['ai.local.owner']==I['owner'] and labels['ai.local.allocation']==A['allocation_id'] and labels['ai.local.gpu-uuid']==I['gpu_uuid']
assert obj['State']['Status']=='exited' and obj['State']['Running'] is False and obj['State']['Paused'] is False
assert obj['State']['Restarting'] is False and obj['State']['Dead'] is False and obj['State']['Pid']==0
# Exit/OOM/error are preserved evidence; they do not gate metadata-only return.
assert obj['HostConfig']['RestartPolicy']['Name']=='no'
assert obj['HostConfig']['Memory']==obj['HostConfig']['MemorySwap']==96*(1<<30) and obj['HostConfig']['NanoCpus']==22*10**9
mounts=[{'source':x['Source'],'target':x['Destination'],'read_only':not x['RW']} for x in obj['Mounts']]
assert sorted(mounts,key=lambda x:x['target'])==sorted(A['details']['mounts'],key=lambda x:x['target'])
assert len(mounts)==10 and {x['target'] for x in mounts}=={'/code/neural-forge','/control','/inputs/corpus','/inputs/digest-receipts','/parent','/recovery-source','/resume-prefix','/resume-window37-48','/run-control-output','/run-output'}
assert next(x for x in mounts if x['target']=='/run-output')=={'source':I['native_working_output_root'],'target':'/run-output','read_only':False}
assert next(x for x in mounts if x['target']=='/run-control-output')=={'source':I['output_root'],'target':'/run-control-output','read_only':False}
assert next(x for x in mounts if x['target']=='/resume-prefix')=={'source':I['readonly_resume_source_roots']['legacy_closed36'],'target':'/resume-prefix','read_only':True}
assert next(x for x in mounts if x['target']=='/resume-window37-48')=={'source':I['readonly_resume_source_roots']['closed37_48'],'target':'/resume-window37-48','read_only':True}
def canonical_sha(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
assert canonical_sha(obj['Config'])=='6deaffcd6ce50778e1cf8fb925f32c7329e81a28d355751351b9e2671ed97624'
assert canonical_sha(obj['HostConfig'])=='ebab51141e54be6dbc6e607125d0ba24089f3ec68f2799279ad4f88fcd3fb94f'
assert canonical_sha(sorted(obj['Mounts'],key=lambda x:x['Destination']))=='cf8ad38b6a141b8748642cf50c7b47ce20c362d141a97ddcf1061ee67f85fc55'
assert I['run_uuid']=='03603908-570d-4c90-96cb-ee3a63a9a139' and A['allocation_id']=='aebe07fe-2905-470c-8a10-f683a770faeb'
assert obj['Id']=='f924bdf0fa81ff21e26b791623a115e6812bd579b3a1d048b69b451ddf0a7514'
control=P(I['runtime_root'])/'production_control';code=P(I['runtime_root'])/'production_code'
with (P(I['output_root'])/'execution_source_freeze.json').open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==FROZEN['source_freeze_sha256']
assert FROZEN['ok'] is True
for root,fields in ((control,FROZEN['control_files']),(code,FROZEN['code_files'])):
 for relative,digest in fields.items():
  rp=pathlib.PurePosixPath(relative);assert not rp.is_absolute() and '..' not in rp.parts
  path=root/relative;assert path.is_file() and not path.is_symlink()
  with path.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==digest,str(path)
assert FROZEN['control_files']['continuation_config.json']=='eaebf00e1506dc6f7a55f333ac05a35fab0cb3f2d4b7a96e6a10badece4f8e37'
assert FROZEN['control_files']['continuation_worker.py']=='3109e40c14c87fd9f81c957688221a5eedda13d7f8deedfc0a47336b47274fd5'
assert FROZEN['code_files']['reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py']=='5e852b63d28ba41599ca3e994851b3e72103f213384de77b9c172c7823036007'
requests=obj['HostConfig']['DeviceRequests'];assert len(requests)==1 and requests[0]['DeviceIDs']==[I['gpu_uuid']]
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=10)
assert not any(line.split(',')[0].strip()==I['gpu_uuid'] for line in procs.splitlines())
# Preserve exact terminal/failure bytes before ROOT calls release_group. No
# tensor payload is collected: these are the frozen Driver's own metadata/log
# tree and native receipt metadata. Missing terminal/error files are recorded
# as absent, including a PID1 OOM before the Driver could publish them.
meta=P(I['output_root']);native=P(I['native_working_output_root'])
driver=meta/'continuation'/I['run_uuid']
seal=native/'ancestry'/('window-49-60-'+I['run_uuid'])
closed_metadata=[seal/'window_continuity.json',seal/'fold/snapshots/snapshots_manifest.json',seal/'inputs/full_ffn_validation.json',seal/'receipts/k3_dense_pass_receipt.json']
evidence=[]
def capture(path, root):
 assert path.is_file() and not path.is_symlink() and path.suffix in ('.json','.jsonl','.log'),str(path)
 assert path.resolve().is_relative_to(root.resolve()),str(path)
 stat0=path.stat();raw=path.read_bytes();stat1=path.stat()
 assert (stat0.st_dev,stat0.st_ino,stat0.st_size,stat0.st_mtime_ns)==(stat1.st_dev,stat1.st_ino,stat1.st_size,stat1.st_mtime_ns),str(path)
 assert len(raw)==stat1.st_size,str(path)
 return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'raw_base64':base64.b64encode(raw).decode('ascii')}
for root in (driver,native/'receipts'):
 if root.exists():
  assert root.is_dir() and not root.is_symlink(),str(root)
  for path in sorted(root.rglob('*')):
   assert not path.is_symlink(),str(path)
   if path.is_file():evidence.append(capture(path,root))
for path in sorted(set([meta/'execution_source_freeze.json',meta/'continuation_status.json',*meta.glob('continuation_worker_error-*.json')])):
 if path.exists():evidence.append(capture(path,meta))
for path in closed_metadata:
 if path.exists():evidence.append(capture(path,seal))
expected=[*closed_metadata,meta/'execution_source_freeze.json',meta/'continuation_status.json',driver/'terminal_result.json',driver/'error.json',driver/'events.jsonl',driver/'resources.jsonl',driver/'window49_60_launch/fitter-49-60.execution.json',driver/'window49_60_launch/fitter-49-60.log',native/'receipts/k3_dense_pass_receipt.json',native/'receipts/k3_dense_pass_receipt.partial.json']
absence=[str(path) for path in expected if not path.exists()]
logs=subprocess.run(['docker','logs',A['container_id']],capture_output=True,timeout=15)
assert logs.returncode==0,'complete own Docker terminal log readback failed'
docker_log={name:{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'raw_base64':base64.b64encode(raw).decode('ascii')} for name,raw in (('stdout',logs.stdout),('stderr',logs.stderr))}
final=json.loads(subprocess.check_output(['docker','inspect',A['container_id']],text=True,timeout=15))[0]
assert final['Id']==obj['Id'] and final['State']==obj['State'],'terminal state changed during complete evidence collection'
assert final['State']['Status']=='exited' and final['State']['Running'] is False and final['State']['Paused'] is False and final['State']['Restarting'] is False and final['State']['Pid']==0
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,timeout=10)
assert not any(line.split(',')[0].strip()==I['gpu_uuid'] for line in procs.splitlines())
# Only actual terminal metadata is joined here; tensor payloads were checked by
# the frozen Driver's serial scanner/native verifier and are not reread.
by_path={row['path']:row for row in evidence}
assert len(by_path)==len(evidence)
assert by_path[str(meta/'execution_source_freeze.json')]['sha256']==FROZEN['source_freeze_sha256']
def body(path):
 row=by_path[str(path)];return json.loads(base64.b64decode(row['raw_base64'],validate=True))
events=[json.loads(line) for line in base64.b64decode(by_path[str(driver/'events.jsonl')]['raw_base64'],validate=True).splitlines()] if str(driver/'events.jsonl') in by_path else []
execution_path=driver/'window49_60_launch/fitter-49-60.execution.json'
execution=body(execution_path) if str(execution_path) in by_path else None
fitter_identity=None
if execution is not None:
 assert execution['phase']=='native_fitter' and type(execution['pid']) is int and execution['pid']>0
 plan=json.loads((control/'recovery_plan49_60.json').read_bytes())
 assert plan['next_layers']=='49-60' and execution['argv']==plan['command_argv']
 assert execution['log_path']=='/run-control-output/continuation/'+I['run_uuid']+'/window49_60_launch/fitter-49-60.log'
 assert execution['log_sha256']==by_path[str(driver/'window49_60_launch/fitter-49-60.log')]['sha256']
 starts=[x for x in events if x['stage']=='native_fitter_started']
 runs=[x for x in events if x['stage']=='native_fitter_running']
 ends=[x for x in events if x['stage']=='native_fitter_terminal']
 assert len(starts)==len(ends)==1 and runs
 assert starts[0]['command_argv']==execution['argv'] and starts[0]['command_started_at']==execution['started_at']
 assert all(x['child_pid']==execution['pid'] for x in runs)
 assert ends[0]['actual_command_result']==execution
 fitter_identity={'namespace_pid':execution['pid'],'source':'actual native_fitter execution/Popen and start/running/terminal event joins','host_pid_at_terminal':None,'PID1_launch_host_pid_historical':1316999,'PID1_launch_birth_ticks_historical':304100690,'execution_sha256':by_path[str(execution_path)]['sha256'],'exit_code':execution['exit_code'],'argv':execution['argv']}
closed60=None
if final['State']['ExitCode']==0 and final['State']['OOMKilled'] is False:
 status=body(driver/'terminal_result.json');current=body(meta/'continuation_status.json')
 assert status==current and status['identity']==I and status['allocation']==A
 assert status['stage']=='bounded_window_complete' and status['window_scope']=='49-60'
 assert type(status['full_prefix_layers']) is int and status['full_prefix_layers']==60
 assert status['later_windows_admitted'] is False and status['config_sha256']==FROZEN['control_files']['continuation_config.json']
 assert status['capability_accepted'] is False and status['training_eligible'] is False and status['numerical_targets_enforced'] is False
 assert len(status['completed_windows'])==1
 window=status['completed_windows'][0];logical='/run-output/ancestry/window-49-60-'+I['run_uuid']+'/window_continuity.json'
 assert window['layers']=='49-60' and window['continuity_path']==status['final_continuity_path']==logical
 closed=body(seal/'window_continuity.json');closed_sha=by_path[str(seal/'window_continuity.json')]['sha256']
 assert closed_sha==window['continuity_sha256']==status['final_continuity_sha256']
 complete=[x for x in events if x['stage']=='bounded_window_complete']
 assert len(complete)==1 and complete[0]['full_prefix_layers']==60 and complete[0]['later_windows_admitted'] is False
 assert complete[0]['final_continuity_path']==logical and complete[0]['final_continuity_sha256']==closed_sha and complete[0]['ended_at']==status['ended_at']
 assert closed['schema']=='neural_forge.k3_actual_window_continuity.v1' and closed['completed_window']=='49-60' and closed['full_prefix_layers']==60 and closed['ffn_full_scan_complete'] is True
 assert closed['prior_window_continuity_sha256']=='ab1f4b36e07c6606f48c1498624c947438dcc9ff352e31ed060fafd3955a17b9'
 assert [x['layer'] for x in closed['prefix']]==list(range(1,61))
 scan=body(seal/'inputs/full_ffn_validation.json');receipt=body(seal/'receipts/k3_dense_pass_receipt.json')
 assert by_path[str(seal/'inputs/full_ffn_validation.json')]['sha256']==closed['ffn_validation_sha256']
 assert scan['schema']=='neural_forge.k3_functional_ffn_receipt.v2' and scan['functional_artifacts_usable'] is True
 assert sorted(x['layer'] for x in scan['ffns'])==list(range(49,61))
 assert all(x['actual_library_load_verified'] is True and x['all_coefficients_finite'] is True for x in scan['ffns'])
 assert by_path[str(seal/'fold/snapshots/snapshots_manifest.json')]['sha256']==closed['snapshot_manifest_sha256']
 assert by_path[str(seal/'receipts/k3_dense_pass_receipt.json')]['sha256']==closed['receipt_sha256']==window['terminal_receipt_sha256']==by_path[str(native/'receipts/k3_dense_pass_receipt.json')]['sha256']
 assert receipt['layers_processed_this_run']==list(range(49,61)) and receipt['start']['layer']==49 and receipt['start']['installed_layers']==list(range(1,49))
 assert receipt['resume_context']==closed['resume_context'] and receipt['resume_context']['software']['tool_sha256']==FROZEN['code_files']['reports/omni-campaign-2026-09/composite/tools/k3_dense_pass.py']
 assert receipt['installed_ffn_sha256']=={str(x['layer']):x['ffn_sha256'] for x in closed['prefix']}
 assert receipt['arguments']['acceptance']=='record' and receipt['arguments']['snapshot_keep']==receipt['snapshots']['keep']==0
 assert {x['stream'] for x in closed['endpoint_readback']}=={'teacher','folded'} and all(x['layer']==61 and x['actual_library_load_verified'] is True and x['all_coordinates_finite'] is True for x in closed['endpoint_readback'])
 runtime_path=driver/'windows/49-60/runtime_readback.json'
 assert by_path[str(runtime_path)]['sha256']==closed['runtime_observation_sha256']
 assert body(runtime_path)['actual_resume_context']==closed['resume_context']
 assert closed['capability_accepted'] is False and closed['training_eligible'] is False and closed['numerical_targets_enforced'] is False and closed['native_whole_child_capability_checked'] is False
 assert fitter_identity is not None and fitter_identity['exit_code']==0
 closed60={'full_prefix_layers':60,'completed_window':'49-60','continuity_sha256':closed_sha,'ffn_validation_sha256':closed['ffn_validation_sha256'],'snapshot_manifest_sha256':closed['snapshot_manifest_sha256'],'terminal_receipt_sha256':closed['receipt_sha256'],'metadata_joins_complete':True,'tensor_payload_reads_by_this_collector':0,'capability_accepted':False,'training_eligible':False}
terminal_evidence={'complete':True,'actual_closed60_metadata_join':closed60,'actual_fitter_identity':fitter_identity,'files':evidence,'explicit_absences':absence,'docker_logs':docker_log,'docker_state':final['State'],'model_artifact_success_claimed':False,'tensor_payload_reads':0}
# Same complete V3 payload; compression changes transport only. ROOT verifies
# complete compressed and expanded bytes/SHA plus every file/log member before
# preserving local evidence and invoking unchanged metadata-only release_group.
payload=(json.dumps({'ok':True,'container_id':obj['Id'],'docker_state':obj['State'],'memory_limit':obj['HostConfig']['Memory'],'cpus':22,'allocation_swap_bytes':0,'mounts':mounts,'GPU6_no_compute_process':True,'exact_installed_pins':True,'exact49_60_production_source_closure':True,'source_freeze_sha256':FROZEN['source_freeze_sha256'],'container_preserved':True,'read_only':True,'terminal_evidence':terminal_evidence},sort_keys=True,allow_nan=False)+'\n').encode('utf-8')
packed=zlib.compress(payload)
print(json.dumps({'transport':'zlib.base64.complete-owned-terminal-metadata.v1','container_id':A['container_id'],'allocation_id':A['allocation_id'],'uncompressed_bytes':len(payload),'uncompressed_sha256':hashlib.sha256(payload).hexdigest(),'compressed_bytes':len(packed),'compressed_sha256':hashlib.sha256(packed).hexdigest(),'compressed_base64':base64.b64encode(packed).decode('ascii')}))
