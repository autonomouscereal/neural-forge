import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'joint_audio_native_prepare_v1'
REMOTE='/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1'
code='''from pathlib import Path
import json,subprocess
p=Path(REMOTE)
phases={}
for name in ('chain','compose','native-ce'):
 path=p/'logs'/(name+'.exitcode')
 phases[name]={'done':path.exists(),'exit_code':path.read_text().strip() if path.exists() else None}
 log=p/'logs'/(name+'.log')
 if log.exists():phases[name]['log_tail']=log.read_text(errors='replace')[-2500:]
print(json.dumps({'remote':str(p),'exists':p.exists(),'started':(p/'started').exists(),
'inputs_verified':json.loads((p/'inputs_verified.json').read_text()) if (p/'inputs_verified.json').exists() else None,
'input_files':len(list((p/'inputs').rglob('*'))) if (p/'inputs').exists() else 0,
'files':[str(f.relative_to(p)) for f in p.iterdir()] if p.exists() else [],
'phases':phases,'clean_sealed':(p/'joint-clean-cache-v1/manifest.json').exists(),
'native_ce_sealed':(p/'native-audio-ce-v1/manifest.json').exists(),'training_started':False,'gpu_visible':False}))
'''.replace('REMOTE',repr(REMOTE))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('server-manager connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 if rc:raise RuntimeError('owned joint prep snapshot failed: '+err)
 result=json.loads(out);(HERE/'latest_status.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
finally:c.close()

