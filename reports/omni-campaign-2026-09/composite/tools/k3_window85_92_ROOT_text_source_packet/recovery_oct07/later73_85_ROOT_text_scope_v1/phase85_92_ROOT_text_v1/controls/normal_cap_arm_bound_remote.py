"""Use installed shared mutation locks to cap/attest only this exact own recovery."""
import contextlib,datetime,hashlib,importlib.util,io,json,os,pathlib,subprocess,sys
CAP_BYTES=I['operating_cap_proposal']['memory_bytes']
assert type(CAP_BYTES) is int and CAP_BYTES>0 and I['operating_cap_proposal']=={'memory_bytes':CAP_BYTES,'CPU_threads':22,'swap_bytes':0}
P=pathlib.Path;i=I;a=A;out=P(i['output_root']);control=P(i['runtime_root'])/'production_control'
def sha(p):
    with P(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def inspect(cid):
    obj=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0];assert obj['Id']==cid;return obj
for path,h in BASELINE['installed_pins'].items():assert sha(path)==h,path
spec=importlib.util.spec_from_file_location('k3_recovery_installed_rt','/home/cereal/local-hermes/remote_runtime.py');rt=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rt;spec.loader.exec_module(rt)
captured=io.StringIO()
with contextlib.redirect_stdout(captured),rt.mutation():
    target=next(s for s in BASELINE['services'] if s['name']=='comfyui_qwen21_gpu6');provider=inspect(target['id'])
    fp={'image':provider['Image'],'config_sha256':digest(provider['Config']),'host_config_sha256':digest(provider['HostConfig']),'mounts_sha256':digest(sorted(provider['Mounts'],key=lambda x:x['Destination']))}
    assert not provider['State']['Running'] and all(fp[k]==target[k] for k in fp)
    obj=inspect(a['container_id']);labels=obj['Config']['Labels']
    assert obj['State']['Running'] and obj['Image']==i['image'] and obj['Name'].lstrip('/')==i['name']==a['name']
    assert labels['ai.local.owner']==i['owner']==a['owner'] and labels['ai.local.allocation']==a['allocation_id'] and labels['ai.local.gpu-controller']=='v1'
    assert [d for request in obj['HostConfig']['DeviceRequests'] for d in request['DeviceIDs']]==[i['gpu_uuid']]
    assert {(m['source'],m['target'],m['read_only']) for m in a['details']['mounts']}=={(m['Source'],m['Destination'],not m['RW']) for m in obj['Mounts']}
    assert any(m['Destination']=='/recovery-source' and m['Source']==i['historical_output_root'] and not m['RW'] for m in obj['Mounts'])
    process=subprocess.run(['docker','update','--memory',str(CAP_BYTES),'--memory-swap',str(CAP_BYTES),'--cpus','22',obj['Id']],capture_output=True,text=True);assert process.returncode==0,process.stderr
    obj=inspect(obj['Id']);assert obj['HostConfig']['Memory']==obj['HostConfig']['MemorySwap']==CAP_BYTES and obj['HostConfig']['NanoCpus']==22*10**9
    if globals().get('ARM'):
        config=json.loads((control/'continuation_config.json').read_text());assert config['identity']==i and config['expected_allocator_name']==a['name'] and config['budget']['memory_limit_bytes']==CAP_BYTES and config['budget']['cpus']==22
        actual={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
        assert actual==CONTROL_FREEZE and not any(p.is_symlink() for p in control.rglob('*'))
        assert not (out/'continuation.ready').exists() and not (out/'continuation_allocation.json').exists()
        mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in P('/proc/meminfo').read_text().splitlines())}
        assert mem['MemAvailable']>=CAP_BYTES+16*(1<<30)
        attestation={'schema':'neural_forge.k3_actual_continuation_allocation.v1','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identity':i,'config_sha256':sha(control/'continuation_config.json'),'image_id':obj['Image'],'allocation':a,'provider_id':provider['Id'],'provider_running':False,'provider_fingerprints':fp,'control_files':actual,'mounts':a['details']['mounts'],'host_memory_before':{k:mem[k] for k in ('MemAvailable','SwapFree')},'resource_budget':{'memory_limit_bytes':CAP_BYTES,'memory_swap_limit_bytes':CAP_BYTES,'cpus':22,'allocation_swap_bytes':0},'source_revision_sha256':config['native_tool_sha256'],'recovery_packet_sha256':config['recovery_packet']['sha256'],'docker_persistent_environment_modified':False,'capability_accepted':False,'fresh_window_capacity_adoption_required':True}
        path=out/'continuation_allocation.json'
        with path.open('x') as f:json.dump(attestation,f,indent=2);f.write('\n')
        barrier={'config_sha256':attestation['config_sha256'],'allocation_receipt_path':'/run-control-output/continuation_allocation.json','allocation_receipt_sha256':sha(path),'allocation_id':a['allocation_id'],'container_id':a['container_id']}
        temporary=out/'continuation.ready.writing'
        with temporary.open('x') as f:json.dump(barrier,f,indent=2);f.write('\n')
        os.rename(temporary,out/'continuation.ready')
print(json.dumps({'ok':True,'armed':bool(globals().get('ARM')),'allocation_id':a['allocation_id'],'container_id':obj['Id'],'host_pid1':obj['State']['Pid'],'provider_id':provider['Id'],'provider_fingerprints':fp,'caps':{'memory':obj['HostConfig']['Memory'],'swap':obj['HostConfig']['MemorySwap'],'nano_cpus':obj['HostConfig']['NanoCpus']},'control_stdout':captured.getvalue()}))
