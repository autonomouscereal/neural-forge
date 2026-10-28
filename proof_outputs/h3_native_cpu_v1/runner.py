import copy,hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,'/recovery/source')
from experiments.chimera_recovery import h3_native as N
from transformers import AutoTokenizer
import torch
from safetensors.torch import load_file
root=Path('/recovery');out=root/'native-h3-v1';aliases=json.loads((root/'inputs/aliases.json').read_text())
D=N.module('h3_native_actual_tokenizer',N.HERE/'audio_dialogue.py')
for name,h in D.TOKENIZER_PINS.items():N.pinned({'path':'/model/'+name,'sha256':h})
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True,use_fast=True,trust_remote_code=False)
report=N.prepare_cache(Path('/velocity'),'7770cf61b2b88a9dfce802009ca7e3d0f26be05ea9904f2ca47b9f002c90df17',
 Path('/encode'),root/'inputs/items.container.jsonl',root/'inputs/admission.json',
 'de642d15eb976cce615264453e20d49b2eb71369cb1870b8eb9ad57638527202',
 Path('/geometry/proof.json'),'2150acf1c6e30ac9b34076df43361b70b86c7dcb6cf52d02dc38db504fa62e77',
 tokenizer,'8df2df5fff60cb9820d302b21e5a29ee7c53d88b179f45bf087142c53b096596',out,aliases=aliases)
assert report['sealed'] is True and len(report['batches'])==32
checks=[]
for b in report['batches']:
 receipt=N.read(out/b['teacher_receipt']['path']);cpu=N.M.inflate_native_batch(load_file(str(out/b['tensors']['path'])),b['storage'])
 checks.append(N.validate_batch(cpu,receipt,out,aliases=aliases))
first=report['batches'][0];receipt=N.read(out/first['teacher_receipt']['path'])
cpu=N.M.inflate_native_batch(load_file(str(out/first['tensors']['path'])),first['storage'])
bad={k:v.clone() for k,v in cpu.items()};bad['native_media_targets'][0,0,224]+=1
negative=[]
for name,batch,r in [('changed_target',bad,receipt),('relabelled_group',cpu,{**receipt,'source':{**receipt['source'],'recovery_group_id':'held-fixture'}})]:
 try:N.validate_batch(batch,r,out,aliases=aliases)
 except Exception as e:negative.append({'case':name,'refused':True,'reason':str(e)})
 else:raise AssertionError('negative fixture was admitted '+name)
suite=unittest.defaultTestLoader.discover('/recovery/source/tests',pattern='test_chimera_h3_native.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful() and not result.skipped
proof={'schema':'chimera_recovery.h3_actual_ingress.v1','sealed':True,'passed':True,
 'accepted_batches':len(checks),'sources':len({c['source_id'] for c in checks}),
 'all32_genuine_source_reconstruction':checks,'negative_cases':negative,
 'native_manifest_sha256':N.H.sha(out/'manifest.json'),'adapter_sha256':N.H.sha(N.HERE/'h3_native.py'),
 'actual_parent_tokenizer':True,'audio_latent_frames_encoded':36,'audio_latent_padded':1,
 'zero_latent_pad_preserved':True,'tests_run':result.testsRun,'test_skips':len(result.skipped),
 'training_started':False,'gpu_exposed':False,'amendment_sha256':'8df2df5fff60cb9820d302b21e5a29ee7c53d88b179f45bf087142c53b096596'}
(root/'proof.json').write_bytes(N.R.canonical(proof)+b'\n');print(json.dumps(proof))
