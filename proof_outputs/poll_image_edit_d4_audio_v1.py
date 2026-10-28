import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
program=r'''from pathlib import Path
import json,hashlib
r={}
for name,base,exit,log in [('image_d4','/home/cereal/omni-campaign/distill/codex-image-edit-d4-modality-v1','exitcode','log.txt'),
 ('audio_audit','/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2','logs-audit-repair-v3/exitcode','logs-audit-repair-v3/replay-audit.log')]:
 p=Path(base);v={'exitcode':(p/exit).read_text().strip() if (p/exit).exists() else None,
 'log_tail':(p/log).read_text(errors='replace')[-4500:] if (p/log).exists() else None}
 for f in ('proof.json','replay-index.json'):
  if (p/f).is_file():v[f]={'sha256':hashlib.sha256((p/f).read_bytes()).hexdigest()}
 r[name]=v
print(json.dumps(r))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 (Path(__file__).parent/'image_edit_d4_cpu_v1/latest_status.json').write_text(out+'\n');print(out)
finally:c.close()
