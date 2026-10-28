import hashlib,json,re,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'native_audio_consumer_v2'
REMOTE='/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2'
original=(HERE/'run_chain.sh').read_text()
script=re.sub(r'(codex-audio-consumer-[a-z-]+)-modality-v2',r'\1-modality-mountfix-v2',original).replace('logs/','logs-mountfix-v2/')
assert 'cd '+shlex.quote(REMOTE) in script and script.count('--name codex-audio-consumer-')==5
target=HERE/'run_chain_mountfix_v2.sh';target.write_bytes(script.encode())
expected=hashlib.sha256(target.read_bytes()).hexdigest()
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 if not c.upload_file(str(target),REMOTE+'/'+target.name):raise RuntimeError('upload failed')
 code='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE)
assert not(p/'logs-mountfix-v2').exists();(p/'logs-mountfix-v2').mkdir()
for name in ('native-speech-ce-v2','native-speech-flow-v2','joint-clean-cache-v2','native-audio-ce-v2'):assert not(p/name).exists()
assert hashlib.sha256((p/'run_chain_mountfix_v2.sh').read_bytes()).hexdigest()==EXPECTED
subprocess.run(['bash','-n',str(p/'run_chain_mountfix_v2.sh')],check=True)
log=open(p/'logs-mountfix-v2/chain.log','wb');proc=subprocess.Popen(['timeout','23000','bash',str(p/'run_chain_mountfix_v2.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'script_sha256':EXPECTED,'corrected_only_launcher_container_names_and_log_namespace':True,'original_logs_preserved':True,'training_started':False}))
'''.replace('REMOTE',repr(REMOTE)).replace('EXPECTED',repr(expected))
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 result={'exit_code':rc,'stdout':out,'stderr':err};(HERE/'mountfix-v2-launch.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if rc:raise RuntimeError('fresh owned launcher failed')
finally:c.close()
