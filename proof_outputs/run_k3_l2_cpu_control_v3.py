import hashlib,json,shlex,sys,zipfile
from pathlib import Path
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig,SSHClient
own=Path(r'C:/Users/cereal/Documents/ai training/.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-memory-corpus-oct13-9c19be622841')
cap=own/'proof_outputs/k3_b2_l2_source_audit_v1';bundle=cap/'cpu_control_bundle.v3.zip'
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
 for p in cap.glob('*.py'):z.write(p,'sources/'+p.name)
 z.write(own/'experiments/chimera_recovery/k3_l2_diagnostic.py','experiments/chimera_recovery/k3_l2_diagnostic.py')
 z.write(own/'tests/test_chimera_k3_l2_diagnostic.py','tests/test_chimera_k3_l2_diagnostic.py')
 z.writestr('tests/__init__.py','')
sha=hashlib.sha256(bundle.read_bytes()).hexdigest()
remote='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/k3-l2-cpu-control-v3'
probe='''import hashlib,json,os,resource,subprocess,sys,time,zipfile
from pathlib import Path
root=Path('''+repr(remote)+''');bundle=root/'bundle.zip'
if hashlib.sha256(bundle.read_bytes()).hexdigest()!='''+repr(sha)+''':raise RuntimeError('bundle pin mismatch')
work=root/'work'
if work.exists():raise RuntimeError('preserve previous private run')
with zipfile.ZipFile(bundle) as z:z.extractall(work)
output=root/'output';output.mkdir()
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CHIMERA_K3_DIAGNOSTIC_SOURCES=str(work/'sources'),PYTHONPATH=str(work),PYTHONDONTWRITEBYTECODE='1')
def bounds():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(60,60))
 allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,allowed[:4])
start=time.monotonic();commands=[['/home/cereal/model-surgery-lab/venv/bin/python','-B','-m','unittest','tests.test_chimera_k3_l2_diagnostic','-v'],['/home/cereal/model-surgery-lab/venv/bin/python','-B',str(work/'experiments/chimera_recovery/k3_l2_diagnostic.py'),'cpu-pair-control','--sources',str(work/'sources'),'--out',str(output/'control.json')]]
records=[]
for command in commands:
 p=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=60,preexec_fn=bounds)
 records.append({'argv':command,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
 if p.returncode:break
receipt={'schema':'private_k3_l2_cpu_control_execution.v1','bundle_sha256':''' +repr(sha)+''','gpu_hidden':True,'real_parent_weights_loaded':False,'threads':4,'address_space_limit_bytes':8*1024**3,'cpu_limit_seconds':60,'elapsed_seconds':time.monotonic()-start,'commands':records}
(output/'execution.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps({'exit_codes':[r['exit_code'] for r in records],'output':str(output),'elapsed_seconds':receipt['elapsed_seconds']}))
'''
c=SSHClient.from_config(ServerConfig().get('ai'),control=True)
if not c.connect():raise RuntimeError('no connection')
try:
 if not c.upload_file(str(bundle),remote+'/bundle.zip'):raise RuntimeError('upload failed')
 rc,out,err=c.execute('/usr/bin/python3 -B -c '+shlex.quote(probe),timeout=20)
 print(json.dumps({'exit_code':rc,'stdout':out,'stderr':err}))
 for name in ['execution.json','control.json']:
  if not c.download_file(remote+'/output/'+name,str(cap/'cpu_control_v3')):print('not downloaded',name)
finally:c.close()
