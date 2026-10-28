"""Bounded source-read/decode only, authentic existing audio; no admission."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
OWN=Path(__file__).resolve().parents[1];HERE=OWN/'proof_outputs/natural_audio_readproof_v1';HERE.mkdir()
SOURCE=Path('C:/Users/cereal/Documents/ai training/neural-forge/proof_outputs/chimera_corpus_recovery_oct13/normalized.jsonl')
PIN='168c4dc994db5858f0dcea3854c6983625f3c692876925c7c4b880b4f327c51e'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==PIN
sys.path.insert(0,str(OWN));from experiments.chimera_recovery import natural_speech as N
rows=[];chosen=set();speech=0
for line in SOURCE.open(encoding='utf-8'):
 r=json.loads(line);origin=r.get('provenance',{}).get('registry_origin',{});d=origin.get('source_dataset')
 if d=='openslr/librispeech_asr' and r['route']=='audio_understand' and speech<2:
  if r['media'][0]['sha256'] in chosen:continue
  speech+=1;chosen.add(r['media'][0]['sha256']);rows.append(r)
 elif d in {'laion/laions_got_talent_with_voice_emotion_speed_tags_for_orpheus_tuning','amaai-lab/SonicMasterDataset'} and d not in chosen:
  chosen.add(d);rows.append(r)
 if speech==2 and len(rows)==4:break
request={'schema':'chimera_recovery.existing_audio_read_request.v1','source':{'path':str(SOURCE),'sha256':PIN},
 'rows':rows,'source_row_hashes':{r['id']:N.digest(r) for r in rows},'train_admitted':0}
(HERE/'request.json').write_bytes(N.canonical(request)+b'\n')
runner=r'''import hashlib,json,sys
from pathlib import Path
import numpy as np,soundfile as sf
root=Path('/owned');request=json.loads((root/'request.json').read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
records=[];shards={}
for row in request['rows']:
 origin=row['provenance']['registry_origin'];shard=Path(origin['shard_mirror'])
 if str(shard) not in shards:
  shards[str(shard)]={'path':str(shard),'expected_sha256':origin['shard_sha256']}
  try:
   h=sha(shard);shards[str(shard)].update(actual_sha256=h,unchanged=h==origin['shard_sha256'])
  except Exception as e:shards[str(shard)].update(unchanged=False,error=repr(e))
 for asset in row['media']:
  if not any(asset.get(k) in {'audio','audio_input','audio_target'} for k in ('type','kind','media')):continue
  p=Path(asset['path']);v={'source_id':row['id'],'source_row_sha256':request['source_row_hashes'][row['id']],
    'asset_path':str(p),'expected_asset_sha256':asset['sha256'],'dataset':origin['source_dataset'],
    'source_split':row['split'],'source_recovery_group_id':row['provenance']['recovery_group_id'],
    'original_data_date_verified':False,'independent_transcript_verified':False,'train_admitted':False}
  try:
   assert p.is_relative_to('/mnt/ai_data/omnicoder_2026/media_artifacts')
   actual=sha(p);size=p.stat().st_size
   assert actual==asset['sha256'] and size==asset.get('byte_size',asset.get('bytes'))
   info=sf.info(str(p));waveform,rate=sf.read(str(p),dtype='float32',always_2d=True)
   assert waveform.shape==(info.frames,info.channels) and rate==info.samplerate and np.isfinite(waveform).all()
   v.update(passed=True,asset_sha256=actual,bytes=size,sample_rate=rate,channels=info.channels,frames=info.frames,
    duration_seconds=info.frames/rate,format=info.format,subtype=info.subtype,
    full_decode=True,finite=True,abs_peak=float(np.abs(waveform).max()),decoded_float32_sha256=hashlib.sha256(waveform.tobytes()).hexdigest())
  except Exception as e:v.update(passed=False,error=repr(e))
  records.append(v)
proof={'schema':'chimera_recovery.existing_audio_read_proof.v1','source':request['source'],
 'request_sha256':sha(root/'request.json'),'records':records,'source_shards':list(shards.values()),
 'passed':all(v['passed'] for v in records) and all(v['unchanged'] for v in shards.values()),
 'full_decode_only':True,'train_admitted':0,'native_ready':False,'gpu_exposed':False,
 'collection_date_promoted_to_original_date':False,'teacher_transcript_generated':False}
(root/'proof.json').write_text(json.dumps(proof,sort_keys=True)+'\n');print(json.dumps(proof))
'''
(HERE/'runner.py').write_text(runner)
REMOTE='/home/cereal/omni-campaign/distill/codex-natural-audio-readproof-modality-v1'
IMAGE='sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1'
argv=['docker','run','--network','none','--cpus','2','--memory','4g','--no-healthcheck','--entrypoint','/usr/bin/timeout',
 '-e','CUDA_VISIBLE_DEVICES=','-e','OMP_NUM_THREADS=2','-e','MKL_NUM_THREADS=2','-e','PYTHONDONTWRITEBYTECODE=1',
 '-v',REMOTE+':/owned:rw','-v',REMOTE+'/runner.py:/owned/runner.py:ro','-v',REMOTE+'/request.json:/owned/request.json:ro',
 '-v','/mnt/ai_data/omnicoder_2026/media_artifacts:/mnt/ai_data/omnicoder_2026/media_artifacts:ro',
 '-v','/home/cereal/omni-campaign/distill/corpus/registry_mirror:/home/cereal/omni-campaign/distill/corpus/registry_mirror:ro',
 '--name','codex-natural-audio-readproof-modality-v1',IMAGE,'180','/opt/environments/python/comfyui/bin/python','/owned/runner.py']
script='set -euo pipefail\ncd '+shlex.quote(REMOTE)+"\ntrap 'rc=$?; printf \"%s\\n\" \"$rc\" > exitcode' EXIT\n"+shlex.join(argv)+' > log.txt 2>&1\n'
(HERE/'run.sh').write_bytes(script.encode())
plan={'image':IMAGE,'remote':REMOTE,'source':request['source'],'inputs':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('runner.py','request.json','run.sh')},'train_admitted':0,'gpu_exposed':False}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir()'),timeout=10)
 if rc:raise RuntimeError(err)
 for n in ('runner.py','request.json','run.sh','plan.json'):
  if not c.upload_file(str(HERE/n),REMOTE+'/'+n):raise RuntimeError('upload failed')
 program=r'''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);plan=json.loads((p/'plan.json').read_text())
for n,h in plan['inputs'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
subprocess.run(['bash','-n',str(p/'run.sh')],check=True)
job=subprocess.Popen(['timeout','240','bash',str(p/'run.sh')],stdout=open(p/'chain.log','wb'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({'pid':job.pid,'gpu_exposed':False,'train_admitted':0}))
'''.replace('REMOTE',repr(REMOTE))
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 (HERE/'launch.json').write_text(json.dumps({'rc':rc,'stdout':out,'stderr':err})+'\n');print(out)
 if rc:raise RuntimeError(err)
finally:c.close()
