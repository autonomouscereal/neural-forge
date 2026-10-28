import json,hashlib,subprocess,datetime
from pathlib import Path
capsule=Path(__file__).resolve().parent;p=capsule.parent
assert str(capsule)=='/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1/cpu-canary-v1/capsule-asr-v2'
assert not (p/'launch-asr-v2.json').exists(),'ASR successor already launched'
pins=json.loads((capsule/'upload-pins.json').read_text())
for name,expected in pins.items():assert hashlib.sha256((capsule/name).read_bytes()).hexdigest()==expected,name
request=json.loads((capsule/'asr-request.json').read_text());source=Path(request['tts_result']['path']);assert hashlib.sha256(source.read_bytes()).hexdigest()==request['tts_result']['sha256']
result=json.loads(source.read_text());assert result['complete'] is False and result['failure']==request['partial_tts_capture']['preserved_failure']
assert (p/'tts-exitcode').read_text().strip()=='1'
available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
assert available>=32*1024**3,('insufficient host memory before CPU ASR',available)
log=(p/'launcher-asr-v2.log').open('xb')
job=subprocess.Popen(['bash',str(capsule/'run-asr.sh')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'schema':'chimera_recovery.speech_teacher_cpu_asr_launch.v2','namespace':str(p),'pid':job.pid,'host_available_before_bytes':available,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request_sha256':request['request_sha256'],'uploaded_source_sha256':pins,'cpu_threads':4,'memory_cap_gib':24,'task_timeout_seconds_per_model':600,'sequential_models':['Qwen3-ASR-1.7B','Qwen3-ASR-0.6B'],'GPU':False,'training_started':False,'tts_producer_complete':False,'original_waveforms':4,'captured_new_waveforms':3}
(p/'launch-asr-v2.json').write_text(json.dumps(r,indent=2)+chr(10));print(json.dumps(r))
