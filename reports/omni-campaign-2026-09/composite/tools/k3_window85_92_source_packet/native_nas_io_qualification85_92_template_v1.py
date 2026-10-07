"""One serial native safetensors save/rename/fsync/full-readback on own new existing-filesystem output."""
EXPECTED_RUN = "__UNRESOLVED_NEXT_RUN_UUID__"
assert str(__import__("uuid").UUID(EXPECTED_RUN)) == EXPECTED_RUN
import argparse, hashlib, json, os, pathlib, stat, time
from datetime import datetime, timezone
import torch
from safetensors import safe_open
from safetensors.torch import save_file
P=pathlib.Path
def sha(path):
    with P(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--identity',type=P,required=True);parser.add_argument('--identity-sha256',required=True);parser.add_argument('--out',type=P,required=True);a=parser.parse_args()
assert sha(a.identity)==a.identity_sha256
i=json.loads(a.identity.read_text());assert i['run_uuid']==EXPECTED_RUN and i['native_thread_id']=='01a11108-009c-72f3-b2ba-e0d48fca1435'
assert a.out==P('/run-output') and a.out.is_dir() and not a.out.is_symlink()
assert i['new_output_filesystem'] in ('NAS','Lexar')
assert i['readonly_resume_source_roots']=={
    'legacy_closed36':'/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318',
    'closed37_48':'/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model',
    'closed49_60':'/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139/model',
    'closed61_72':'__UNRESOLVED_CLOSED61_72_ROOT__',
    'closed73_84':'__UNRESOLVED_CLOSED73_84_ROOT__'}
storage={'NAS':'/mnt/ai_data','Lexar':'/mnt/lexar'}[i['new_output_filesystem']]
assert i['native_working_output_root']==storage+'/omni-campaign/k3-nas-window85-92-oct07-'+EXPECTED_RUN+'/model'
for alias in ('/resume-prefix','/resume-window37-48','/resume-window49-60','/resume-window61-72','/resume-window73-84'):
    old=P(alias)
    assert old.is_dir() and not old.is_symlink() and not a.out.samefile(old)
    assert not a.out.is_relative_to(old) and not old.is_relative_to(a.out)
if i['new_output_filesystem']=='NAS':
    assert a.out.stat().st_dev!=P('/resume-prefix').stat().st_dev
# Lexar output may share the original prefix device. The root must be distinct
# and allocator/resolver separately require the exact disjoint owned mount.

root=a.out/'io_qualification';assert not root.exists();root.mkdir()
assert root.stat().st_dev==a.out.stat().st_dev
before=os.statvfs(root);at=datetime.now(timezone.utc).isoformat();start=time.monotonic()
values={'BF16':torch.full((8192,4096),0.125,dtype=torch.bfloat16),'F32':torch.full((4096,4096),0.25,dtype=torch.float32),'channels':torch.arange(4096,dtype=torch.int32)}
temp=root/'native_safetensors.writing';out=root/'native_safetensors.safetensors'
assert not temp.exists() and not out.exists()
save_file(values,str(temp),metadata={'purpose':i['new_output_filesystem']+' IO qualification only','run_uuid':i['run_uuid'],'not_model_weights':'true'})
with temp.open('rb') as f:os.fsync(f.fileno())
saved=time.monotonic();expected=sha(temp)
os.rename(temp,out);assert not temp.exists() and out.is_file();renamed=time.monotonic()
try:
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    directory_fsync='completed'
except OSError as error:
    directory_fsync={'unsupported_errno':error.errno,'file_fsync_completed':True}
loaded={}
with safe_open(str(out),framework='pt',device='cpu') as f:
    assert set(f.keys())==set(values)
    for key,want in values.items():
        got=f.get_tensor(key);assert got.shape==want.shape and got.dtype==want.dtype and torch.equal(got,want)
        assert not got.is_floating_point() or bool(torch.isfinite(got).all())
        loaded[key]={'shape':list(got.shape),'dtype':str(got.dtype),'numel':got.numel(),'full_values_equal':True,'finite':True}
        del got
assert sha(out)==expected
end=time.monotonic();v=os.statvfs(root);size=out.stat().st_size
assert size<512*(1<<20)
receipt={'schema':'neural_forge.k3_actual_native_output_IO_qualification.v1','output_filesystem':i['new_output_filesystem'],'started_utc':at,'ended_utc':datetime.now(timezone.utc).isoformat(),'identity':i,'native_python':{'executable':__import__('sys').executable,'sha256':sha(__import__('sys').executable)},'torch':torch.__version__,'safetensors_save_file':True,'path':str(out),'sha256':expected,'bytes':size,'device':out.stat().st_dev,'native_serial_save_seconds':saved-start,'hash_rename_seconds':renamed-saved,'full_load_and_hash_seconds':end-renamed,'total_seconds':end-start,'save_bytes_per_second':size/(saved-start),'serial_model_output_IO_writers':1,'concurrent_ancestry_copy_during_native_fit':False,'own_IO_qualification_bound_bytes':512*(1<<20),'fresh_free_before_bytes':before.f_bavail*before.f_frsize,'fresh_free_after_payload_bytes':v.f_bavail*v.f_frsize,'directory_fsync':directory_fsync,'full_library_readback':loaded,'model_loaded':False,'CUDA_model_compute':False,'capability_accepted':False,'existing_backup_or_foreign_paths_modified':False,'old_Lexar_payload_copy_bytes':0}
with (root/'READBACK_MANIFEST_V1.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(receipt,sort_keys=True))
