import json,hashlib,subprocess,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
assert str(p)=='/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1/cpu-canary-v1'
assert not (p/'launch.json').exists(),'canary already launched'
hashes=json.loads((p/'upload-pins.json').read_text())
for name,expected in hashes.items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
assert available>=32*1024**3,('insufficient host memory before CPU canary',available)
log=(p/'launcher.log').open('xb')
job=subprocess.Popen(['bash',str(p/'run-tts.sh')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'schema':'chimera_recovery.speech_teacher_cpu_canary_launch.v1','namespace':str(p),'pid':job.pid,'host_available_before_bytes':available,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uploaded_source_sha256':hashes,'cpu_threads':4,'memory_cap_gib':24,'task_timeout_seconds':600,'GPU':False,'training_started':False}
(p/'launch.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
