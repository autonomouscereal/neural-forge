import json, shlex, sys
from pathlib import Path
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
outdir=Path(__file__).resolve().parent/'speech_native_prepare_v1'
code='''from pathlib import Path
import json,sys
p=Path('/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1')
sys.path.insert(0,str(p/'source'))
from experiments.chimera_recovery.h3_audio_chart import validate_encoder_proof
manifest=json.loads((p/'clean-speech-cache-v1/manifest.json').read_text())
bad=[]
for f in sorted((p/'clean-speech-cache-v1').glob('[0-9]*.json')):
 proof=json.loads(f.read_text())
 try:validate_encoder_proof(proof)
 except ValueError as e:
  bad.append({'path':str(f),'error':str(e),'input_contract':proof.get('input_contract'),'measurement':{k:v for k,v in proof.get('measurement',{}).items() if k!='nominal_frame_interval_seconds'},'decode':proof.get('decode'),'teacher_unchanged':proof.get('teacher_unchanged'),'probe_passed':proof.get('probe_passed')})
print(json.dumps({'manifest':manifest,'bad_count':len(bad),'bad':bad[:4]}))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 if rc:raise RuntimeError(err)
 value=json.loads(out)
 (outdir/'proof_consumer_failure_v1.json').write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({k:v for k,v in value.items() if k!='manifest'}))
finally:c.close()
