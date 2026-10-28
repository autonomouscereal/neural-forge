import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent/'speech_native_prepare_v1'
launch=json.loads((HERE/'launch.json').read_text());remote=launch['remote']
program='''from pathlib import Path
import json,subprocess
p=Path(REMOTE);root=p/'clean-speech-cache-v1'
phases={}
for name in ('encoder','native-ce','native-flow','chain'):
    path=p/'logs'/(name+'.exitcode')
    phases[name]={'done':path.exists(),'exitcode':path.read_text().strip() if path.exists() else None}
for name in ('encoder','chain','native-ce','native-flow'):
    path=p/'logs'/(name+'.log')
    if path.exists():phases[name]['log_tail']=path.read_text(errors='replace')[-2500:]
proc=subprocess.run(['docker','inspect','codex-speech-clean-modality-v1','--format','{{json .State}}'],text=True,capture_output=True)
print(json.dumps({'phases':phases,'probe_receipts_written':len(list(root.glob('[0-9]*.json'))),
'clean_sealed_manifest_exists':(root/'manifest.json').exists(),'native_ce_manifest_exists':(p/'native-speech-ce-v1/manifest.json').exists(),
'native_flow_manifest_exists':(p/'native-speech-flow-v1/manifest.json').exists(),'encoder_container':json.loads(proc.stdout) if proc.returncode==0 else proc.stderr,
'training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(remote))
client=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not client.connect():raise RuntimeError('server-manager connect failed')
try:
    rc,out,err=client.execute('python3 -c '+shlex.quote(program),timeout=10)
    if rc:raise RuntimeError('owned native preparation poll failed: '+err)
    result=json.loads(out)
    (HERE/'latest_status.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:client.close()
