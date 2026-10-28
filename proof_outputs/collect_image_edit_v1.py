import hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'image_edit_cpu_v1'
program=r'''from pathlib import Path
import hashlib,json,tarfile,os
p=Path('/home/cereal/omni-campaign/distill/codex-image-edit-modality-v1')
assert (p/'exitcode').read_text().strip()=='0'
proof=json.loads((p/'proof.json').read_text());assert proof['passed'] and proof['native_batches']==32
files=[p/n for n in ('proof.json','plan.json','runner.py','run.sh','log.txt','exitcode')]
files+=sorted((p/'native-edit-v1').iterdir())
pins={str(f.relative_to(p)):{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in files if f.is_file()}
archive=p/'collection-v1.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for f in files:
  if f.is_file():t.add(f,arcname=str(f.relative_to(p)),recursive=False)
receipt={'files':pins,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size}
(p/'collection-v1.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
os.chmod(archive,0o644);os.chmod(p/'collection-v1.json',0o644)
print(json.dumps(receipt))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=15)
 if rc:raise RuntimeError(err)
 pins=json.loads(out)
finally:c.close()
folder=HERE/'downloaded';folder.mkdir(exist_ok=True)
c=SSHClient.from_config(ServerConfig().get('ai'))
if not c.connect():raise RuntimeError('connect failed')
try:
 for name in ('collection-v1.tar.gz','collection-v1.json'):
  if not c.download_file('/home/cereal/omni-campaign/distill/codex-image-edit-modality-v1/'+name,str(folder)):raise RuntimeError('download failed')
finally:c.close()
archive=folder/'collection-v1.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==pins['archive_sha256']
target=HERE/'actual';target.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as t:t.extractall(target,filter='data')
for name,pin in pins['files'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==pin['sha256']
print(json.dumps({'actual':str(target),'files':len(pins['files']),'all_sha256_verified':True,'proof':pins['files']['proof.json'],'manifest':pins['files']['native-edit-v1/manifest.json']}))
