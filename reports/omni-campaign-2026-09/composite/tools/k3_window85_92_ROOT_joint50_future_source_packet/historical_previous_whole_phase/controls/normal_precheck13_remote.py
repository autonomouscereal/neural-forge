import hashlib,importlib.util,json,pathlib,subprocess,sys
P=pathlib.Path;i=I
def sha(p):
    with P(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for path,h in BASELINE['installed_pins'].items():assert sha(path)==h,path
spec=importlib.util.spec_from_file_location('k3_g5_normal_rt','/home/cereal/local-hermes/remote_runtime.py');rt=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rt;spec.loader.exec_module(rt)
mounts=rt.validate_mounts({'kind':'generic','mounts':MOUNTS})
assert len(mounts)==13
rows=[]
for row in LEDGER['allocations']:
    if row['state'] not in ('starting','running'):continue
    obj=json.loads(subprocess.check_output(['docker','inspect',row['container_id']],text=True,timeout=20))[0]
    assert obj['Id']==row['container_id'] and obj['State']['Running'] and obj['State']['Pid']>0,('normal_allocator_would_reconcile_terminal_foreign_row',row['allocation_id'])
    rows.append({'allocation_id':row['allocation_id'],'container_id':obj['Id'],'running':True,'pid1':obj['State']['Pid']})
assert not any(row['gpu_uuid']==i['gpu_uuid'] for row in LEDGER['allocations'] if row['state'] in ('starting','running'))
print(json.dumps({'ok':True,'normal_allocator_foreign_active_rows_are_physically_running':rows,'installed_mount_api':'validate_mounts({kind:generic,mounts})','supported_mounts':mounts,'mutations':False}))
