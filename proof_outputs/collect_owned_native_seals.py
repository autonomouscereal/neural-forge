"""Collect exact owned preparation receipts without changing source caches."""
import hashlib,json,shlex,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent
program=r'''from pathlib import Path
import hashlib,json,tarfile,os
out=Path('/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2/collection-v3.tar.gz')
audio=out.parent;h3=Path('/home/cereal/omni-campaign/distill/codex-h3-native-modality-v3')
status={'audio_audit_exitcode':(audio/'logs-audit-repair-v3/exitcode').read_text().strip() if (audio/'logs-audit-repair-v3/exitcode').exists() else None,
 'audio_audit_log_tail':(audio/'logs-audit-repair-v3/replay-audit.log').read_text(errors='replace')[-2000:] if (audio/'logs-audit-repair-v3/replay-audit.log').exists() else '',
 'h3_exitcode':(h3/'exitcode').read_text().strip()}
audio_done=status['audio_audit_exitcode']=='0'
if not audio_done:out=h3/'collection-h3-v3.tar.gz'
files=[]
for base,names in ((audio,['replay-index.json','logs-audit-repair-v3/input_receipt.json','logs-audit-repair-v3/exitcode','logs-audit-repair-v3/replay-audit.log','logs-mountfix-v2/chain.exitcode','logs-mountfix-v2/chain.log']),
 (h3,['proof.json','exitcode','log.txt'])):
 if base==audio and not audio_done:continue
 for n in names:
  f=base/n
  if f.exists():files.append((f,base.name+'/'+n))
if audio_done:
 for name in ('native-speech-ce-v2','native-speech-flow-v2','joint-clean-cache-v2','native-audio-ce-v2'):
  for f in sorted((audio/name).glob('*.json')):files.append((f,audio.name+'/'+name+'/'+f.name))
for f in sorted((h3/'native-h3-v1').iterdir()):
 if f.is_file():files.append((f,h3.name+'/native-h3-v1/'+f.name))
pins={rel:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f,rel in files}
with tarfile.open(out,'w:gz') as t:
 for f,rel in files:t.add(f,arcname=rel,recursive=False)
receipt=out.with_suffix('.json');receipt.write_text(json.dumps({'files':pins,'archive_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive_bytes':out.stat().st_size,'remote_archive':str(out)},sort_keys=True)+'\n')
os.chmod(out,0o644);os.chmod(receipt,0o644)
status.update(archive=str(out),receipt=str(receipt),archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),files=len(files),archive_bytes=out.stat().st_size)
print(json.dumps(status))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=15)
 if rc:raise RuntimeError(err)
 status=json.loads(out);(HERE/'native_audio_consumer_v2/collection-status.json').write_text(json.dumps(status,indent=2)+'\n')
 print(json.dumps(status))
finally:c.close()
if 'archive' not in status:raise SystemExit(0)
c=SSHClient.from_config(ServerConfig().get('ai'))
if not c.connect():raise RuntimeError('download connect failed')
try:
 folder=HERE/'native_audio_consumer_v2/downloaded-seals';folder.mkdir(exist_ok=True)
 dest=folder/Path(status['archive']).name;receipt=folder/Path(status['receipt']).name
 c.download_file(status['archive'],str(folder));c.download_file(status['receipt'],str(folder))
finally:c.close()
if hashlib.sha256(dest.read_bytes()).hexdigest()!=status['archive_sha256']:raise RuntimeError('download SHA mismatch')
target=HERE/'native_seals_v3';target.mkdir(exist_ok=True)
with tarfile.open(dest,'r:gz') as t:t.extractall(target,filter='data')
for name,pin in json.loads(receipt.read_text())['files'].items():
 if hashlib.sha256((target/name).read_bytes()).hexdigest()!=pin['sha256']:raise RuntimeError('collected file SHA mismatch '+name)
print(json.dumps({'collected':len(json.loads(receipt.read_text())['files']),'local':str(target),'all_sha256_verified':True}))
