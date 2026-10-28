import hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).parent/'image_edit_d4_cpu_v1'
REMOTE='/home/cereal/omni-campaign/distill/codex-image-edit-d4-modality-v1'
program=r'''from pathlib import Path
import hashlib,json,tarfile,os
p=Path(REMOTE);assert (p/'exitcode').read_text().strip()=='0'
names=['proof.json','plan.json','exitcode','log.txt','chain.log','run.sh','runner.py']
pins={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}
out=p/'collection-v1.tar.gz'
with tarfile.open(out,'w:gz') as t:
 for n in names:t.add(p/n,arcname=n)
r={'files':pins,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
(p/'collection-v1.json').write_text(json.dumps(r)+'\n');os.chmod(out,0o644);os.chmod(p/'collection-v1.json',0o644)
print(json.dumps(r))
'''.replace('REMOTE',repr(REMOTE))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 receipt=json.loads(out)
finally:c.close()
folder=HERE/'actual';folder.mkdir(exist_ok=True)
c=SSHClient.from_config(ServerConfig().get('ai'))
if not c.connect():raise RuntimeError('download connect failed')
try:
 for n in ('collection-v1.tar.gz','collection-v1.json'):c.download_file(REMOTE+'/'+n,str(HERE))
finally:c.close()
archive=HERE/'collection-v1.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['archive_sha256']
with tarfile.open(archive,'r:gz') as t:t.extractall(folder,filter='data')
for n,h in receipt['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h,n
print(json.dumps({'files':len(receipt['files']),'all_sha256_verified':True,'local':str(folder)}))
