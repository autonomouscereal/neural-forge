import hashlib,json,sys
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
