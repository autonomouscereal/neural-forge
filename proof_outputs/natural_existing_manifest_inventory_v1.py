import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
program=r'''from pathlib import Path
import hashlib,json
base=Path('/home/cereal/omnicoder_2026_work');out={}
for name in ('reports','registry','exports','profiles'):
 p=base/name;rows=[]
 if p.exists():
  for f in sorted(p.iterdir()):
   rows.append({'name':f.name,'is_dir':f.is_dir(),'bytes':f.stat().st_size if f.is_file() else None})
 out[name]=rows[:200]
print(json.dumps({'path':str(base),'inventory':out}))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 if rc:raise RuntimeError(err[-500:])
 path=Path(__file__).parent/'natural_recorded_sources_oct13_v1/existing_materialization_dirs.json';path.write_text(out+'\n')
 print(out[:11000])
finally:c.close()
