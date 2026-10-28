"""Retain failed launcher/logs and add only empty mountpoints in owned capsule."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'native_audio_consumer_v2'
REMOTE='/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2'
SPEECH='/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1'
script=(HERE/'run_chain.sh').read_text().replace('logs/','logs-mountfix-v1/').replace('-modality-v2','-modality-mountfix-v1')
target=HERE/'run_chain_mountfix_v1.sh';target.write_bytes(script.encode())
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 if not c.upload_file(str(target),REMOTE+'/'+target.name):raise RuntimeError('upload failed')
 program='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);s=Path(SPEECH)
assert (p/'logs/chain.exitcode').read_text().strip()=='125'
assert not(p/'logs-mountfix-v1').exists();(p/'logs-mountfix-v1').mkdir()
for name in ('native-speech-ce-v2','native-speech-flow-v2','joint-clean-cache-v2','native-audio-ce-v2'):assert not(p/name).exists()
created=[]
for name in ('clean-audio-cache-v1','audio-cache-inputs-v1'):
 d=s/name;assert d.parent.resolve()==s.resolve();assert not d.exists();d.mkdir();created.append(str(d))
assert hashlib.sha256((p/'run_chain_mountfix_v1.sh').read_bytes()).hexdigest()==EXPECTED
subprocess.run(['bash','-n',str(p/'run_chain_mountfix_v1.sh')],check=True)
receipt={'previous_exit_code':125,'previous_logs_preserved':True,'created_empty_owned_mountpoints':created,
 'original_encoder_manifest_sha256':hashlib.sha256((s/'clean-speech-cache-v1/manifest.json').read_bytes()).hexdigest(),
 'genuine_source_files_unchanged':True,'new_launcher_sha256':EXPECTED,'training_started':False,'gpu_exposed':False}
(p/'mountfix-receipt.json').write_text(json.dumps(receipt)+'\\n')
log=open(p/'logs-mountfix-v1/chain.log','wb');proc=subprocess.Popen(['timeout','23000','bash',str(p/'run_chain_mountfix_v1.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'receipt':receipt}))
'''.replace('REMOTE',repr(REMOTE)).replace('SPEECH',repr(SPEECH)).replace('EXPECTED',repr(hashlib.sha256(target.read_bytes()).hexdigest()))
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 result={'exit_code':rc,'stdout':out,'stderr':err};(HERE/'mountfix-launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if rc:raise RuntimeError('owned mount repair validation failed')
finally:c.close()
