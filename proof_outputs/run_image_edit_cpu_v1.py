"""Owned actual CPU ingress/test proof; no teacher/model/production writes."""
import hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
OWN=Path(__file__).resolve().parents[1];ROOT=OWN.parent/'chimera-training-recovery-oct13-eae1b72e2c3f'
HERE=OWN/'proof_outputs/image_edit_cpu_v1';HERE.mkdir()
BASE=Path('C:/Users/cereal/Documents/ai training')
REMOTE='/home/cereal/omni-campaign/distill/codex-image-edit-modality-v1'
PROC='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
IMAGE='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
AMEND='8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(ROOT)).replace('\\','/'):p for p in (ROOT/'experiments/chimera_recovery').glob('*.py')}
for name in ('image_edit.py','train_image_edit.py'):files['experiments/chimera_recovery/'+name]=OWN/'experiments/chimera_recovery'/name
for name in ('test_chimera_image_edit.py','test_chimera_image_edit_stage.py'):files['tests/'+name]=OWN/'tests'/name
admit=OWN/'proof_outputs/chimera_all_media_source_admission_oct13_v2/global_admission'
for p,name in ((admit/'manifest.json','admission.json'),(admit/'admission.jsonl','admission-rows.jsonl'),
               (admit.parent/'source_ready.jsonl','source_ready.jsonl'),
               (BASE/'reviews/2026-10-13-folding/recovery-preparation/image-edit-selection-v1.json','selection.json'),
               (BASE/'reviews/2026-10-13-folding/recovery-preparation/PREREGISTRATION_IMAGE_EDIT_RECOVERY_2026-10-13.selected-v1.md','selected.md')):
 files['inputs/'+name]=p
aliases={sha(admit/'admission.jsonl'):'/recovery/inputs/admission-rows.jsonl',sha(admit.parent/'source_ready.jsonl'):'/recovery/inputs/source_ready.jsonl'}
(HERE/'aliases.json').write_text(json.dumps(aliases)+'\n');files['inputs/aliases.json']=HERE/'aliases.json'
runner=r'''import copy,json,sys
from pathlib import Path
import pytest,torch
from transformers import AutoTokenizer
from safetensors.torch import load_file
from experiments.chimera_recovery import image_edit as E,train_image_edit as D
root=Path('/recovery');aliases=json.loads((root/'inputs/aliases.json').read_text());out=root/'native-edit-v1'
selection=E.N.read(root/'inputs/selection.json')
assert selection['selected'] is True and selection['amendment_sha256']==AMEND
assert E.N.H.sha(root/'inputs/selected.md')==AMEND
tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True,use_fast=True,trust_remote_code=False)
report=E.prepare(Path('/pairs/proof.json'),'2f66575259f4b1155b0c3e1224e0bb12fca67d8e2f4641d751652df7bf5dc203',
 root/'inputs/source_ready.jsonl',root/'inputs/admission.json','2d0fc404c7ce8d9d4a0f9023ec7c46dc124350d04254c4f6cc3540768b3ce806',
 tokenizer,AMEND,out,tokenizer_dir=Path('/model'),aliases=aliases)
assert report['sealed'] and len(report['batches'])==32 and len(report['excluded'])==16
D.cache_manifest(out/'manifest.json',AMEND,aliases=aliases)
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
 'amendment_sha256':AMEND,'native_parent_full_parity':False,'training_started':False,'gpu_exposed':False}
(root/'proof.json').write_bytes(E.R.canonical(proof)+b'\n');print(json.dumps(proof))
'''.replace('AMEND',repr(AMEND))
(HERE/'runner.py').write_text(runner);files['runner.py']=HERE/'runner.py'
hashes={n:sha(p) for n,p in files.items()}
with tarfile.open(HERE/'capsule.tar','w') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=('source/'+name if name.startswith(('experiments/','tests/')) else name))
assert hashes=={n:sha(p) for n,p in files.items()}
plan={'remote':REMOTE,'image':IMAGE,'cpu_threads':4,'memory_gb':8,'source_hashes':hashes,'capsule_sha256':sha(HERE/'capsule.tar'),
 'root_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'candidate_commits':['7f5c3708','a96765d0'],'amendment_sha256':AMEND,'training_started':False,'gpu_exposed':False}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
argv=['docker','run','--network','none','--cpus','4','--memory','8g','--no-healthcheck','--entrypoint','/usr/bin/timeout',
 '-e','CUDA_VISIBLE_DEVICES=','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_OFFLINE=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
 '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/recovery/source',
 '-e','CHIMERA_OMNI_SRC=/foundation','-e','CHIMERA_STREAMED_BASELINE=/baseline/d3_hybrid_module.py',
 '-v',REMOTE+':/recovery:rw','-v',REMOTE+'/source:/recovery/source:ro','-v',REMOTE+'/inputs:/recovery/inputs:ro',
 '-v',PROC+'/qimg-edit-vae-v1/out:/pairs:ro',
 '-v','/home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro','-v','/home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro',
 '-v','/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9:/foundation:ro','-v','/home/cereal/omni-campaign/distill/tools:/baseline:ro',
 '--name','codex-image-edit-modality-v1',IMAGE,'1800','/usr/bin/python3.12','/recovery/runner.py']
script='set -euo pipefail\ncd '+shlex.quote(REMOTE)+"\ntrap 'rc=$?; printf \"%s\\n\" \"$rc\" > exitcode' EXIT\n"+shlex.join(argv)+' > log.txt 2>&1\n'
(HERE/'run.sh').write_bytes(script.encode())
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 rc,out,err=c.execute('python3 -c '+shlex.quote('from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir()'),timeout=10)
 if rc:raise RuntimeError(err)
 for name in ('capsule.tar','plan.json','run.sh'):
  if not c.upload_file(str(HERE/name),REMOTE+'/'+name):raise RuntimeError('upload failed')
 program=r'''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);plan=json.loads((p/'plan.json').read_text());assert hashlib.sha256((p/'capsule.tar').read_bytes()).hexdigest()==plan['capsule_sha256']
subprocess.run(['tar','xf',str(p/'capsule.tar'),'-C',str(p)],check=True)
for name,h in plan['source_hashes'].items():
 q=p/('source/'+name if name.startswith(('experiments/','tests/')) else name)
 assert hashlib.sha256(q.read_bytes()).hexdigest()==h
subprocess.run(['bash','-n',str(p/'run.sh')],check=True)
log=open(p/'chain.log','wb');job=subprocess.Popen(['timeout','1900','bash',str(p/'run.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':job.pid,'training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(REMOTE))
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 (HERE/'launch.json').write_text(json.dumps({'exit_code':rc,'stdout':out,'stderr':err},indent=2)+'\n');print(out)
 if rc:raise RuntimeError(err)
finally:c.close()
