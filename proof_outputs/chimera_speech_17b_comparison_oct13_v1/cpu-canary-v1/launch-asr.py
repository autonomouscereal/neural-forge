import json,hashlib,subprocess,datetime
from pathlib import Path
p=Path(__file__).resolve().parent
assert str(p)=='/home/cereal/omni-campaign/distill/codex-speech-1p7-comparison-modality-v1/cpu-canary-v1'
assert not (p/'launch-asr.json').exists(),'ASR tasks already launched'
assert (p/'tts-exitcode').read_text().strip()=='0','TTS task did not pass'
request=json.loads((p/'capsule/request.json').read_text())
result=p/'tts-candidate/result.json'
tts=json.loads(result.read_text())
assert tts['complete'] is True and tts['request_sha256']==request['request_sha256']
assert len(tts['rows'])==4 and {r['id'] for r in tts['rows']}=={r['id'] for r in request['rows']}
assert tts['helper_sha256']==hashlib.sha256((p/'capsule/speech_teacher_canary.py').read_bytes()).hexdigest()
request['parent_request_sha256']=request.pop('request_sha256')
request['tts_result']={'path':str(result),'sha256':hashlib.sha256(result.read_bytes()).hexdigest()}
request['request_sha256']=hashlib.sha256(json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
(p/'capsule/asr-request.json').write_text(json.dumps(request,indent=2)+chr(10))
available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
assert available>=32*1024**3,('insufficient host memory before CPU ASR',available)
log=(p/'launcher-asr.log').open('xb')
job=subprocess.Popen(['bash',str(p/'run-asr.sh')],cwd=p,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r={'schema':'chimera_recovery.speech_teacher_cpu_asr_launch.v1','namespace':str(p),'pid':job.pid,'host_available_before_bytes':available,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'request_sha256':request['request_sha256'],'cpu_threads':4,'memory_cap_gib':24,'task_timeout_seconds_per_model':600,'sequential_models':['Qwen3-ASR-1.7B','Qwen3-ASR-0.6B'],'GPU':False,'training_started':False}
(p/'launch-asr.json').write_text(json.dumps(r,indent=2)+chr(10));print(json.dumps(r))
