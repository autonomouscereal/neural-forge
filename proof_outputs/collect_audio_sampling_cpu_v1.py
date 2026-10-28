import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
base=Path(__file__).resolve().parent
version=sys.argv[1] if len(sys.argv)>1 else 'v1'
if version not in {'v1','v2'}:raise ValueError('select explicit private proof namespace')
launch=json.loads((base/('audio_sampling_cpu_'+version+'_launch.json')).read_text());remote=launch['remote']
code='''from pathlib import Path
import hashlib,json
p=Path(REMOTE);expected=PINS
changed=[item['path'] for item in expected if hashlib.sha256((p/item['path']).read_bytes()).hexdigest()!=item['sha256']]
exit=p/'exitcode'
print(json.dumps({'finished':exit.exists(),'exit_code':exit.read_text().strip() if exit.exists() else None,
'output':(p/'output.log').read_text(errors='replace')[-12000:],'source_hashes_unchanged':not changed,'changed_sources':changed,
'training_started':False,'actual_current_model_loaded':False}))
'''.replace('REMOTE',repr(remote)).replace('PINS',repr(launch['files']))
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('server-manager connect failed')
try:
    rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
    if rc:raise RuntimeError('owned sampling proof collection failed: '+err)
    result=json.loads(out)
    (base/('audio_sampling_cpu_'+version+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
finally:c.close()
