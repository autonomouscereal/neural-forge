"""One private CPU ingress proof; original teacher/cache/input mounts read-only."""
import hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
OWN=Path(__file__).resolve().parents[1]
ROOT=OWN.parent/'chimera-training-recovery-oct13-eae1b72e2c3f'
BASE=Path('C:/Users/cereal/Documents/ai training')
HERE=OWN/'proof_outputs/h3_native_cpu_v3';HERE.mkdir()
REMOTE='/home/cereal/omni-campaign/distill/codex-h3-native-modality-v3'
PROC='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
IMAGE='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
AMEND='8df2df5fff60cb9820d302b21e5a29ee7c53d88b179f45bf087142c53b096596'
VELOCITY=PROC+'/h3-bf16-velocity-v1/out/cache/h3-velocity-minimax_h3_fl2va_pruned_bf16-same-mux-bf16-native-v1/c354051646c9aaf0'
ENCODE=PROC+'/h3-joint-encode-v2/out/cache/h3-vae-same-mux-native-window-v1/c354051646c9aaf0'
GEOMETRY=PROC+'/h3-native-geometry-v2/out'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(ROOT)).replace('\\','/'):p for p in (ROOT/'experiments/chimera_recovery').glob('*.py')}
files['experiments/chimera_recovery/h3_native.py']=OWN/'experiments/chimera_recovery/h3_native.py'
files['experiments/chimera_recovery/train_media.py']=OWN/'experiments/chimera_recovery/train_media.py'
files['tests/test_chimera_h3_native.py']=OWN/'tests/test_chimera_h3_native.py'
files['tests/test_chimera_recovery_train_media.py']=OWN/'tests/test_chimera_recovery_train_media.py'
g=OWN/'proof_outputs/chimera_h3_source_admission_oct13_v1/global_admission'
for p,name in [(g/'manifest.json','admission.json'),(g/'admission.jsonl','admission-rows.jsonl'),
              (g.parent/'source_ready.jsonl','source_ready.jsonl'),
              (BASE/'reviews/2026-10-13-folding/recovery-preparation/h3-joint-encode-v2/items.container.jsonl','items.container.jsonl')]:
    files['inputs/'+name]=p
aliases={sha(g/'admission.jsonl'):'/recovery/inputs/admission-rows.jsonl',
         sha(g.parent/'source_ready.jsonl'):'/recovery/inputs/source_ready.jsonl'}
(HERE/'aliases.json').write_text(json.dumps(aliases)+'\n');files['inputs/aliases.json']=HERE/'aliases.json'
runner='''import copy,hashlib,json,sys,unittest
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
 tokenizer,AMEND,out,aliases=aliases)
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
import pytest
assert pytest.main(['/recovery/source/tests/test_chimera_recovery_train_media.py','-k','h3_dispatch','-q'])==0
proof={'schema':'chimera_recovery.h3_actual_ingress.v1','sealed':True,'passed':True,
 'accepted_batches':len(checks),'sources':len({c['source_id'] for c in checks}),
 'all32_genuine_source_reconstruction':checks,'negative_cases':negative,
 'native_manifest_sha256':N.H.sha(out/'manifest.json'),'adapter_sha256':N.H.sha(N.HERE/'h3_native.py'),
 'actual_parent_tokenizer':True,'audio_latent_frames_encoded':36,'audio_latent_padded':1,
 'zero_latent_pad_preserved':True,'tests_run':result.testsRun,'test_skips':len(result.skipped),
 'training_started':False,'gpu_exposed':False,'amendment_sha256':AMEND}
(root/'proof.json').write_bytes(N.R.canonical(proof)+b'\\n');print(json.dumps(proof))
'''.replace('AMEND',repr(AMEND))
(HERE/'runner.py').write_text(runner);files['runner.py']=HERE/'runner.py'
hashes={name:sha(p) for name,p in files.items()}
with tarfile.open(HERE/'capsule.tar','w') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=('source/'+name if name.startswith(('experiments/','tests/')) else name))
assert hashes=={name:sha(p) for name,p in files.items()}
plan={'remote':REMOTE,'image':IMAGE,'cpu_threads':4,'memory_gb':8,'source_hashes':hashes,
 'capsule_sha256':sha(HERE/'capsule.tar'),'root_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'source_adapter_commit':'a6629dfd+3d4dc2dc','amendment_sha256':AMEND,'training_started':False,'gpu_exposed':False}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
argv=['docker','run','--network','none','--cpus','4','--memory','8g','--no-healthcheck','--entrypoint','/usr/bin/timeout',
 '-e','CUDA_VISIBLE_DEVICES=','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_OFFLINE=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
 '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/recovery/source',
 '-v',REMOTE+':/recovery:rw','-v',REMOTE+'/source:/recovery/source:ro','-v',REMOTE+'/inputs:/recovery/inputs:ro',
 '-v',VELOCITY+':/velocity:ro','-v',ENCODE+':/encode:ro','-v',GEOMETRY+':/geometry:ro',
 '-v','/home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro','-v','/home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro',
 '--name','codex-h3-native-modality-v3',IMAGE,'1800','/usr/bin/python3.12','/recovery/runner.py']
script='set -euo pipefail\ncd '+shlex.quote(REMOTE)+"\ntrap 'rc=$?; printf \"%s\\n\" \"$rc\" > exitcode' EXIT\n"+shlex.join(argv)+' > log.txt 2>&1\n'
(HERE/'run.sh').write_bytes(script.encode())
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('connect failed')
try:
 code='from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();p.mkdir()'
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 if rc:raise RuntimeError(err)
 for name in ('capsule.tar','plan.json','run.sh'):
  if not c.upload_file(str(HERE/name),REMOTE+'/'+name):raise RuntimeError('upload failed')
 code='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE);plan=json.loads((p/'plan.json').read_text())
assert hashlib.sha256((p/'capsule.tar').read_bytes()).hexdigest()==plan['capsule_sha256']
subprocess.run(['tar','xf',str(p/'capsule.tar'),'-C',str(p)],check=True)
for name,h in plan['source_hashes'].items():
 q=p/('source/'+name if name.startswith(('experiments/','tests/')) else name)
 assert hashlib.sha256(q.read_bytes()).hexdigest()==h
subprocess.run(['bash','-n',str(p/'run.sh')],check=True)
log=open(p/'chain.log','wb');job=subprocess.Popen(['timeout','1900','bash',str(p/'run.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':job.pid,'training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(REMOTE))
 rc,out,err=c.execute('python3 -c '+shlex.quote(code),timeout=10)
 (HERE/'launch.json').write_text(json.dumps({'exit_code':rc,'stdout':out,'stderr':err},indent=2)+'\n')
 print(out)
 if rc:raise RuntimeError(err)
finally:c.close()
