import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'native_audio_consumer_v2'
REMOTE='/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2'
program='''from pathlib import Path
import json
p=Path(REMOTE);phases={};logs=next((p/name for name in ('logs-mountfix-v2','logs-mountfix-v1','logs') if (p/name).exists()))
for name in ('speech-ce','speech-flow','compose','joint-ce','replay-audit','chain'):
 f=logs/(name+'.exitcode');log=logs/(name+'.log')
 phases[name]={'exit_code':f.read_text().strip() if f.exists() else None,'log_tail':log.read_text(errors='replace')[-2000:] if log.exists() else None}
manifests={}
for name in ('native-speech-ce-v2','native-speech-flow-v2','joint-clean-cache-v2','native-audio-ce-v2'):
 f=p/name/'manifest.json'
 if f.exists():
  value=json.loads(f.read_text());manifests[name]={'sealed':value.get('sealed'),'entries':len(value.get('entries',value.get('batches',[]))),'training_started':False}
print(json.dumps({'remote':str(p),'phases':phases,'manifests':manifests,'replay_index_exists':(p/'replay-index.json').exists(),'training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(REMOTE))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 value=json.loads(out);(HERE/'latest_status.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
finally:c.close()
