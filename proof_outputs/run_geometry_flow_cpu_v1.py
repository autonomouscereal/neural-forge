import hashlib, json, os, shlex, sys, zipfile
from pathlib import Path
MAIN=Path('C:/Users/cereal/Documents/ai training')
OWN=Path(__file__).resolve().parents[1]
LOCAL=MAIN/'reviews/2026-10-13-folding/recovery-preparation/geometry-flow-cpu-v1'
ROOT='/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
REMOTE=ROOT+'/geometry-flow-cpu-v1'
TREE='/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9'
TOKENIZER='/home/cereal/models/Qwen3.8-27B-Uncensored'
IMAGE='sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1'
INTERSECTION=MAIN/'proof_outputs/chimera_geometry_targets_oct13_v1/manifest.json'
AMENDMENT=OWN/'experiments/chimera_recovery/GEOMETRY_FLOW_AMENDMENT_2026-10-13.v1.json'
SELECTED_AMENDMENT=MAIN/'.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f/experiments/chimera_recovery/GEOMETRY_FLOW_AMENDMENT_2026-10-13.v1.json'
SELECTION=MAIN/'reviews/2026-10-13-folding/recovery-preparation/geometry-flow-selection-v1.json'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(path,v):path.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
LOCAL.mkdir(parents=True,exist_ok=True)
if sys.argv[1]=='augment-selection':
 plan=json.loads((LOCAL/'capsule.json').read_text())
 if sha(SELECTED_AMENDMENT)!='d12d3a173ccfc13ab0de1db18f9dfc540093bcdc1cb075ebd3ae7a038db228ad' or sha(SELECTION)!='ee32d74bc3b0385cbb926568ffb889a22dc79d0e863b10c3dd8ce943bec322a7':raise ValueError('exact root owner selection bytes required')
 archive=Path(plan['capsule']['path'])
 with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for path in (SELECTED_AMENDMENT,SELECTION):
   key=sha(path);z.write(path,'files/'+key)
   plan['files'][key]={'path':str(path),'sha256':key,'bytes':path.stat().st_size}
   plan['aliases'][str(path)]=key;plan['aliases'][str(path).replace('\\','/')]=key
  z.write(SELECTED_AMENDMENT,'code/experiments/chimera_recovery/'+AMENDMENT.name)
  z.writestr('aliases.json',json.dumps(plan['aliases']))
 plan['amendment_sha256']=sha(SELECTED_AMENDMENT);plan['selection_sha256']=sha(SELECTION)
 plan['capsule'].update(sha256=sha(archive),bytes=archive.stat().st_size)
 dump(LOCAL/'capsule.json',plan)
 print(json.dumps({'selected_amendment_sha256':plan['amendment_sha256'],'selection_sha256':plan['selection_sha256'],'capsule_sha256':sha(archive)}));raise SystemExit()
if sys.argv[1]=='build':
 files, aliases, pending={}, {}, [(str(INTERSECTION),sha(INTERSECTION)),(str(AMENDMENT),sha(AMENDMENT))]
 missing=[]
 def refs(value):
  if isinstance(value,dict):
   if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str) and len(value['sha256'])==64:
    pending.append((value['path'],value['sha256']))
   for item in value.values():refs(item)
  elif isinstance(value,list):
   for item in value:refs(item)
 while pending:
  original, expected=pending.pop()
  if original in aliases:
   if aliases[original]!=expected:raise ValueError('same source path has conflicting seals')
   continue
  path=Path(original)
  if not path.is_file():
   missing.append({'path':original,'sha256':expected});continue
  if sha(path)!=expected:
   # Nested historical code references are not current inventory inputs. They
   # get no alias, never a substitute SHA. Any actually required pin will still
   # fail closed in the unchanged native inventory verifier.
   missing.append({'path':original,'sha256':expected,'reason':'nested historical reference differs; not aliased'});continue
  aliases[original]=expected
  aliases[original.replace('\\','/')]=expected
  if expected in files:continue
  files[expected]=path
  if path.suffix=='.json':refs(json.loads(path.read_text(encoding='utf-8')))
  elif path.name in {'source_ready.jsonl','prepared_targets.jsonl'}:
   for line in path.read_text(encoding='utf-8').splitlines():
    if line.strip():refs(json.loads(line))
 archive=LOCAL/'source_capsule.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for key,path in files.items():z.write(path,'files/'+key)
  names=('geometry_flow.py','geometry_targets.py','geometry_bridge_probe.py','geometry_conditioning.py','geometry_prepare.py','hy3d_target_session.py','hy3d_surface_probe.py','audio_admission.py')
  for name in names:z.write(OWN/'experiments/chimera_recovery'/name,'code/experiments/chimera_recovery/'+name)
  z.writestr('code/experiments/__init__.py','')
  z.writestr('code/experiments/chimera_recovery/__init__.py','')
  z.writestr('code/tests/__init__.py','')
  z.write(OWN/'tests/test_chimera_geometry_flow.py','code/tests/test_chimera_geometry_flow.py')
  z.write(AMENDMENT,'code/experiments/chimera_recovery/'+AMENDMENT.name)
  z.writestr('aliases.json',json.dumps(aliases))
 plan={'schema':'recovery_geometry_flow_cpu_capsule.v1','aliases':aliases,'files':{k:{'path':str(v),'sha256':k,'bytes':v.stat().st_size} for k,v in files.items()},'unneeded_container_refs_not_local':missing,'capsule':{'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size},'intersection_sha256':sha(INTERSECTION),'amendment_sha256':sha(AMENDMENT),'training_started':False}
 dump(LOCAL/'capsule.json',plan)
 print(json.dumps({'files':len(files),'aliases':len(aliases),'bytes':archive.stat().st_size,'capsule_sha256':sha(archive)}))
 raise SystemExit()
sys.path.insert(0,'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient, ServerConfig
c=SSHClient.from_config(ServerConfig().get('ai'),control=sys.argv[1]!='launch')
if not c.connect():raise SystemExit('connection failed')
try:
 if sys.argv[1]=='launch':
  plan=json.loads((LOCAL/'capsule.json').read_text())
  capsule=Path(plan['capsule']['path'])
  if sha(capsule)!=plan['capsule']['sha256']:raise ValueError('capsule changed')
  setup="from pathlib import Path;import json;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','output']];print(json.dumps({'mem_available_kib':int([x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1])}))"
  rc,out,err=c.execute('python3 -c '+shlex.quote(setup),timeout=10)
  if rc:raise SystemExit(err)
  memory=json.loads(out)
  if memory['mem_available_kib']<40*1024**2:raise ValueError('bounded32GiB CPU proof lacks headroom')
  if not c.upload_file(str(capsule),REMOTE+'/input/source_capsule.zip'):raise ValueError('capsule upload failed')
  runner=LOCAL/'runner.py'
  text=r'''import hashlib,json,os,sys,time,traceback,unittest,zipfile
from pathlib import Path
root=Path('/work/input');output=Path('/work/output');started=time.time()
def dump(name,v):(output/name).write_text(json.dumps(v,indent=2)+'\n')
archive=root/'source_capsule.zip'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=CAPSULE_SHA:raise ValueError('uploaded source capsule changed')
with zipfile.ZipFile(archive) as z:
 for name in z.namelist():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('capsule path outside namespace')
 # Isolated owned output tree, no extracted file changes original sources.
 z.extractall(output/'capsule')
capsule=output/'capsule';aliases=json.loads((capsule/'aliases.json').read_text())
workspace=output/'aliases';workspace.mkdir()
for original,sha in aliases.items():
 target=capsule/'files'/sha
 if hashlib.sha256(target.read_bytes()).hexdigest()!=sha:raise ValueError('capsule artifact seal differs')
 if not original.startswith('C:') or '..' in original.replace('\\','/').split('/'):raise ValueError('bounded workspace alias required')
 link=workspace/original
 link.parent.mkdir(parents=True,exist_ok=True)
 link.symlink_to(target)
os.chdir(workspace);sys.path.insert(0,str(capsule/'code'))
os.environ['CHIMERA_OMNI_SRC']=TREE+'/src'
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(8)
dump('status.json',{'phase':'actual_native_tests','device':'cpu','torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized()})
try:
 suite=unittest.defaultTestLoader.loadTestsFromName('tests.test_chimera_geometry_flow')
 result=unittest.TestResult();suite.run(result)
 proof={'schema':'recovery_geometry_flow_cpu_proof.v1','source_capsule_sha256':CAPSULE_SHA,'runtime':{'device':'cpu','threads':8,'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized()},'tests_run':result.testsRun,'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],'skipped':[(str(t),s) for t,s in result.skipped],'elapsed_seconds':time.time()-started,'native_train_ready':False,'training_started':False}
 dump('tests.json',proof)
 if not result.wasSuccessful() or result.skipped:raise ValueError('actual numerical/native test failure or skip')
 dump('status.json',{'phase':'native_tests_passed_preparing32','tests':result.testsRun,'elapsed_seconds':time.time()-started})
 from experiments.chimera_recovery import geometry_flow as F
 manifest=F.prepare(intersection=workspace/INTERSECTION_PATH,intersection_sha256=INTERSECTION_SHA,
  amendment=capsule/'code/experiments/chimera_recovery/GEOMETRY_FLOW_AMENDMENT_2026-10-13.v1.json',amendment_sha256=AMENDMENT_SHA,
  tokenizer_dir=Path(TOKENIZER),output=output/'native-batches',selected=capsule/'files'/SELECTION_SHA,max_length=8192)
 from safetensors.torch import load_file
 first=manifest['batches'][0]
 batch=F.inflate(load_file(first['tensors']['path']))
 compute=F.GeometryFlowAdapter(first['receipt']['path'],first['receipt']['sha256']).bind(batch)
 prediction=(-batch['native_media_targets']).requires_grad_()
 loss=compute(prediction+.25);loss.backward()
 if not torch.isfinite(loss) or not torch.isfinite(prediction.grad).all() or not bool(prediction.grad[:,1025:,320:384].abs().sum()>0):raise ValueError('actual sealed source/objective binding backward failed')
 proof.update(batch_manifest={'path':str(output/'native-batches/manifest.json'),'sha256':F.A.file_sha(output/'native-batches/manifest.json')},batches=len(manifest['batches']),held_excluded=len(manifest['excluded']),selected=True,sealed_adapter_gradient_proved=True,elapsed_seconds=time.time()-started)
 dump('proof.json',proof)
 dump('status.json',{'phase':'complete','tests':result.testsRun,'batches':len(manifest['batches']),'held_excluded':len(manifest['excluded']),'elapsed_seconds':time.time()-started,'native_train_ready':False})
except BaseException as error:
 dump('status.json',{'phase':'failed','type':type(error).__name__,'message':str(error),'elapsed_seconds':time.time()-started})
 traceback.print_exc();raise
'''
  text='CAPSULE_SHA='+repr(plan['capsule']['sha256'])+'\nTREE='+repr(TREE)+'\nINTERSECTION_PATH='+repr(str(INTERSECTION).replace('\\','/'))+'\nINTERSECTION_SHA='+repr(plan['intersection_sha256'])+'\nAMENDMENT_SHA='+repr(plan['amendment_sha256'])+'\nSELECTION_SHA='+repr(plan['selection_sha256'])+'\nTOKENIZER='+repr(TOKENIZER)+'\n'+text
  runner.write_text(text,encoding='utf-8')
  if not c.upload_file(str(runner),REMOTE+'/input/runner.py'):raise ValueError('runner upload failed')
  argv=['docker','run','--detach','--name','codex-geometry-flow-cpu-oct13-v1','--user','1000:1000','--runtime','runc','--network','none','--no-healthcheck','--cpus','8','--memory','32g','--memory-swap','32g','--read-only','--pids-limit','128','--security-opt','no-new-privileges','--cap-drop','ALL','--tmpfs','/tmp:rw,size=64m','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=8','--env','MKL_NUM_THREADS=8','--mount','type=bind,src='+REMOTE+'/input,dst=/work/input,readonly','--mount','type=bind,src='+REMOTE+'/output,dst=/work/output','--mount','type=bind,src='+TREE+',dst='+TREE+',readonly','--mount','type=bind,src='+TOKENIZER+',dst='+TOKENIZER+',readonly','--entrypoint','timeout',IMAGE,'--signal=TERM','--kill-after=15','1200','/opt/environments/python/comfyui/bin/python','-B','/work/input/runner.py']
  rc,out,err=c.execute(shlex.join(argv),timeout=15)
  receipt={'schema':'recovery_geometry_flow_cpu_launch.v1','argv':argv,'source_capsule_sha256':plan['capsule']['sha256'],'runner_sha256':sha(runner),'container':out.strip(),'exit_code':rc,'stderr':err,'memory_before':memory,'launcher_sha256':sha(__file__),'training_started':False,'cpu_only':True}
  dump(LOCAL/'launch.json',receipt);print(json.dumps({k:receipt[k] for k in ('container','exit_code','memory_before')}))
 elif sys.argv[1]=='status':
  if c.download_file(REMOTE+'/output/status.json',str(LOCAL)):print((LOCAL/'status.json').read_text())
 elif sys.argv[1]=='collect':
  if c.download_file(REMOTE+'/output/tests.json',str(LOCAL)):print((LOCAL/'tests.json').read_text())
  if c.download_file(REMOTE+'/output/proof.json',str(LOCAL)):print((LOCAL/'proof.json').read_text())
  if c.download_file(REMOTE+'/output/native-batches/manifest.json',str(LOCAL)):print('actual batch manifest SHA',sha(LOCAL/'manifest.json'))
 elif sys.argv[1]=='logs':
  receipt=json.loads((LOCAL/'launch.json').read_text());rc,out,err=c.execute('docker logs --tail 40 '+shlex.quote(receipt['container']),timeout=10);print(out);print(err)
 else:raise SystemExit('build/launch/status/collect/logs only')
finally:c.close()
