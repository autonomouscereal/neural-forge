import json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
HERE=Path(__file__).resolve().parent
program=r'''from pathlib import Path
import json,hashlib
values={}
for name,base,exit,log in [('image_edit','/home/cereal/omni-campaign/distill/codex-image-edit-modality-v1','exitcode','log.txt'),
 ('audio_audit','/home/cereal/omni-campaign/distill/codex-native-audio-consumer-modality-v2','logs-audit-repair-v3/exitcode','logs-audit-repair-v3/replay-audit.log')]:
 p=Path(base);v={'exitcode':(p/exit).read_text().strip() if (p/exit).exists() else None,'log_tail':(p/log).read_text(errors='replace')[-8500:] if (p/log).exists() else None}
 for f in ('proof.json','replay-index.json'):
  if (p/f).is_file():
   data=json.loads((p/f).read_text());v[f]={'sha256':hashlib.sha256((p/f).read_bytes()).hexdigest(),**{k:data[k] for k in ('sealed','passed','verified','source_admitted','native_batches','complete_source_traversal','production_training_ready') if k in data}}
 values[name]=v
print(json.dumps(values))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('sudo -n python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err)
 (HERE/'image_edit_cpu_v1/latest_status.json').write_text(out+'\n');print(out)
finally:c.close()
