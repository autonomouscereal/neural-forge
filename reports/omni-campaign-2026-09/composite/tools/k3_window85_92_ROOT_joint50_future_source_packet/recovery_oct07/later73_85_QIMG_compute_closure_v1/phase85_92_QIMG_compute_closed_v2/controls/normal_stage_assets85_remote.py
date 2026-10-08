"""Verify complete packet bytes before exclusive owned extraction."""
import hashlib,json,pathlib,zipfile,datetime
P=pathlib.Path;root=P(I['runtime_root']);archive=root/ARCHIVE_NAME
def sha(p):
 with P(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert root.is_dir() and I['new_layers']==[85,92] and str(__import__('uuid').UUID(I['run_uuid']))==I['run_uuid']
assert sha(archive)==ARCHIVE_SHA
with zipfile.ZipFile(archive) as z:
 assert set(z.namelist())==set(MEMBERS)
 verified={}
 for name,pin in MEMBERS.items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='production_control'
  b=z.read(name);assert len(b)==pin['bytes'] and hashlib.sha256(b).hexdigest()==pin['sha256'];verified[name]=b
 for name,b in verified.items():
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);assert not p.is_symlink()
  if p.exists() and sha(p)!=MEMBERS[name]['sha256']:
   replacements=globals().get('REPLACEMENTS',{})
   assert name in replacements and sha(p)==replacements[name],name
   assert not (P(I['output_root'])/'execution_source_freeze.json').exists()
   assert not (P(I['output_root'])/'continuation_allocation.json').exists()
   old=root/'production_control/source_revisions/unselected_resource_v3'/pathlib.PurePosixPath(name).relative_to('production_control')
   old.parent.mkdir(parents=True,exist_ok=True)
   with old.open('xb') as f:f.write(p.read_bytes())
   assert sha(old)==replacements[name]
   p.write_bytes(b)
  elif p.exists():assert sha(p)==MEMBERS[name]['sha256'],name
  else:
   with p.open('xb') as f:f.write(b)
print(json.dumps({'ok':True,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':sha(archive),'identity':I,'actual_control_pins':{n:sha(root/n) for n in MEMBERS},'runtime_actions':False}))
