import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent
program='''from pathlib import Path
import json
p=Path('/home/cereal/omni-campaign/distill/codex-h3-native-modality-v3')
print(json.dumps({'exitcode':(p/'exitcode').read_text().strip() if (p/'exitcode').exists() else None,
 'log_tail':(p/'log.txt').read_text(errors='replace')[-6000:] if (p/'log.txt').exists() else None,
 'chain_tail':(p/'chain.log').read_text(errors='replace')[-1000:] if (p/'chain.log').exists() else None,
 'proof':json.loads((p/'proof.json').read_text()) if (p/'proof.json').exists() else None}))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 value=json.loads(out);(HERE/'h3_native_cpu_v3/latest_status.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps(value))
finally:c.close()
