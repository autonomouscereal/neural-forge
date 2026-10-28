"""Owned canonical-env actual CPU validation; no production optimizer."""
import hashlib,json,shlex,subprocess,sys,tarfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
OWN=Path(__file__).resolve().parents[1]
HERE=OWN/'proof_outputs/image_edit_d4_cpu_v1';HERE.mkdir()
REMOTE='/home/cereal/omni-campaign/distill/codex-image-edit-d4-modality-v1'
ORIGINAL='/home/cereal/omni-campaign/distill/codex-image-edit-modality-v1'
PROC='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
IMAGE='sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={str(p.relative_to(OWN)).replace('\\','/'):p for p in (OWN/'experiments/chimera_recovery').glob('*.py')}
tests=['test_chimera_image_edit.py','test_chimera_image_edit_stage.py','test_chimera_image_edit_consolidation.py',
       'test_chimera_geometry_dispatch.py','test_chimera_consolidation_runtime.py']
for name in tests:files['tests/'+name]=OWN/'tests'/name
runner=r'''import json,sys
from pathlib import Path
import pytest
from experiments.chimera_recovery import image_edit as E,train_image_edit as D,consolidation_dispatch as C
root=Path('/recovery');aliases=json.loads((root/'inputs/aliases.json').read_text())
cache=root/'native-edit-v1/manifest.json'
data=D.cache_manifest(cache,'8feb2fdc8be34cff31f7a1c661a52f755ced49891202f0208c9f991d7e3b3573',aliases=aliases)
assert len(data['batches'])==32
assert C.LANES['image_edit']==E.OBJECTIVE and C.LANES['qimg']!=E.OBJECTIVE
tests=['test_chimera_image_edit.py','test_chimera_image_edit_stage.py',
       'test_chimera_image_edit_consolidation.py','test_chimera_geometry_dispatch.py']
rc=pytest.main(['/candidate/source/tests/'+name for name in tests]+['-q'])
assert rc==0
proof={'schema':'chimera_recovery.image_edit_d4_actual_cpu.v1','passed':True,
 'source_native_manifest_sha256':E.N.H.sha(cache),'source_authenticated_batches':32,
 'fixture_environment':{'CHIMERA_STREAMED_BASELINE':'/baseline/streamed_kd_trainer.py',
   'CHIMERA_D3_HYBRID':'/baseline/d3_hybrid_module.py','CHIMERA_OMNI_SRC':'/foundation/src'},
 'image_edit_objective':E.OBJECTIVE,'guided_qimg_objective':C.LANES['qimg'],
 'dispatcher_sha256':E.N.H.sha(Path(C.__file__)),
 'production_training_started':False,'gpu_exposed':False,'native_parent_full_parity':False}
Path('/candidate/proof.json').write_bytes(E.R.canonical(proof)+b'\n');print(json.dumps(proof))
'''
(HERE/'runner.py').write_text(runner);files['runner.py']=HERE/'runner.py'
hashes={n:sha(p) for n,p in files.items()}
with tarfile.open(HERE/'capsule.tar','w') as t:
 for name,p in sorted(files.items()):t.add(p,arcname=('source/'+name if name.startswith(('experiments/','tests/')) else name))
assert hashes=={n:sha(p) for n,p in files.items()}
plan={'remote':REMOTE,'image':IMAGE,'cpu_threads':4,'memory_gb':8,'source_hashes':hashes,
 'capsule_sha256':sha(HERE/'capsule.tar'),'candidate_baseline':subprocess.check_output(['git','rev-parse','HEAD'],cwd=OWN,text=True).strip(),
 'production_training_started':False,'gpu_exposed':False}
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
argv=['docker','run','--network','none','--cpus','4','--memory','8g','--no-healthcheck','--entrypoint','/usr/bin/timeout',
 '-e','CUDA_VISIBLE_DEVICES=','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_OFFLINE=1','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
 '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/candidate/source:/foundation/src',
 '-e','CHIMERA_OMNI_SRC=/foundation/src','-e','CHIMERA_STREAMED_BASELINE=/baseline/streamed_kd_trainer.py',
 '-e','CHIMERA_D3_HYBRID=/baseline/d3_hybrid_module.py',
 '-v',REMOTE+':/candidate:rw','-v',REMOTE+'/source:/candidate/source:ro','-v',ORIGINAL+':/recovery:ro',
 '-v',PROC+'/qimg-edit-vae-v1/out:/pairs:ro','-v','/home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro',
 '-v','/home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro',
 '-v','/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9:/foundation:ro','-v','/home/cereal/omni-campaign/distill/tools:/baseline:ro',
 '--name','codex-image-edit-d4-modality-v1',IMAGE,'1800','/usr/bin/python3.12','/candidate/runner.py']
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
print(json.dumps({'pid':job.pid,'production_training_started':False,'gpu_exposed':False}))
'''.replace('REMOTE',repr(REMOTE))
 rc,out,err=c.execute('python3 -c '+shlex.quote(program),timeout=10)
 (HERE/'launch.json').write_text(json.dumps({'exit_code':rc,'stdout':out,'stderr':err},indent=2)+'\n');print(out)
 if rc:raise RuntimeError(err)
finally:c.close()
