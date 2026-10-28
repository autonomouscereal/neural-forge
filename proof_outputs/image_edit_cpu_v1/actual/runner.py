import copy,json,sys
from pathlib import Path
import pytest,torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from experiments.chimera_recovery import image_edit as E,train_image_edit as D
root=Path('/recovery');aliases=json.loads((root/'inputs/aliases.json').read_text());out=root/'native-edit-v1'
selection=E.N.read(root/'inputs/selection.json')
assert selection['selected'] is True and selection['amendment_sha256']=='8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573'
assert E.N.H.sha(root/'inputs/selected.md')=='8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573'
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True,use_fast=True,trust_remote_code=False)
report=E.prepare(Path('/pairs/proof.json'),'2f66575259f4b1155b0c3e1224e0bb12fca67d8e2f4641d751652df7bf5dc203',
 root/'inputs/source_ready.jsonl',root/'inputs/admission.json','2d0fc404c7ce8d9d4a0f9023ec7c46dc124350d04254c4f6cc3540768b3ce806',
 tokenizer,'8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573',out,tokenizer_dir=Path('/model'),aliases=aliases)
assert report['sealed'] and len(report['batches'])==32 and len(report['excluded'])==16
D.cache_manifest(out/'manifest.json','8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573',aliases=aliases)
checks=[]
for b in report['batches']:
 cpu,compute=D.load_row(b,out,aliases=aliases)
 checks.append({'id':b['id'],'source_id':b['source_id'],'verified':True,'supervised_units':65536})
first=report['batches'][0];receipt=E.N.read(E.N.pinned(first['teacher_receipt'],out))
cpu=E.M.inflate_native_batch(load_file(str(out/first['tensors']['path'])),first['storage'])
target={k:v.clone() for k,v in cpu.items()};target['native_media_targets'][0,0,256]+=1.
reference={k:v.clone() for k,v in cpu.items()};reference['native_media_features'][0,1024,512]+=1.
negative=[]
for name,batch,r in [('changed_after_target',target,receipt),('changed_before_reference',reference,receipt),
 ('relabelled_held_group',cpu,{**receipt,'recovery_group_id':'held-fixture'}),('changed_tokenizer_pin',cpu,
 {**receipt,'tokenizer_files':{**receipt['tokenizer_files'],'tokenizer.json':{**receipt['tokenizer_files']['tokenizer.json'],'sha256':'0'*64}}})]:
 try:E.validate_batch(batch,r,out,aliases=aliases)
 except Exception as e:negative.append({'case':name,'refused':True,'reason':str(e)})
 else:raise AssertionError('mutation was admitted '+name)
rc=pytest.main(['/recovery/source/tests/test_chimera_image_edit.py','/recovery/source/tests/test_chimera_image_edit_stage.py','-q'])
assert rc==0
proof={'schema':'chimera_recovery.image_edit_actual_ingress.v1','sealed':True,'passed':True,
 'actual_parent_tokenizer':True,'all32_source_reconstruction':checks,'negative_cases':negative,
 'source_admitted':8,'excluded_held':16,'native_batches':32,'adapter_sha256':E.N.H.sha(Path(E.__file__)),
 'driver_sha256':E.N.H.sha(Path(D.__file__)),'native_manifest_sha256':E.N.H.sha(out/'manifest.json'),
 'amendment_sha256':'8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573','native_parent_full_parity':False,'training_started':False,'gpu_exposed':False}
(root/'proof.json').write_bytes(E.R.canonical(proof)+b'\n');print(json.dumps(proof))
