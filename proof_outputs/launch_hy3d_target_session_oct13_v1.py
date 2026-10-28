"""Launch/collect one bounded warm native CPU target session; no services/GPU."""
import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig

OWN=Path(__file__).resolve().parents[1]
MAIN=Path("C:/Users/cereal/Documents/ai training")
PREP=MAIN/"proof_outputs/chimera_geometry_prepare_oct13_v2"
PLAN=MAIN/"proof_outputs/chimera_hy3d_target_plan_oct13_v1/plan.json"
LOCAL=MAIN/"reviews/2026-10-13-folding/recovery-preparation/hy3d-target-session-v1"
ROOT="/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013"
REMOTE=ROOT+"/hy3d-target-session-v1"
SOURCE="/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1"
WEIGHTS="/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors"
IMAGE="sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")

LOCAL.mkdir(parents=True,exist_ok=True)
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("server-manager connection failed")
try:
 if sys.argv[1]=="launch":
  plan=json.loads(PLAN.read_text());pin=sha(PLAN)
  if sha(PREP/"manifest.json")!=plan["preparation"]["sha256"]:raise SystemExit("immutable preparation changed")
  code="from pathlib import Path;import json;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','input/assets','tools','output']];print(json.dumps({'mem_available_kib':int([x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1])}))"
  rc,out,err=c.execute("python3 -c "+shlex.quote(code),timeout=10)
  if rc:raise SystemExit(err)
  memory=json.loads(out)
  if memory["mem_available_kib"]<40*1024**2:raise SystemExit("bounded32GiB proof requires40GiB MemAvailable")
  files={}
  def upload(path,name):
   assert c.upload_file(str(path),REMOTE+"/"+name);files[name]=sha(path)
  upload(PLAN,"input/plan.json")
  for name in ("manifest.json","source_ready.jsonl","admission.jsonl"):upload(PREP/name,"input/"+name)
  ids=[e["original_id"] for e in plan["sources"]]
  code="from pathlib import Path;p=Path("+repr(REMOTE+"/input/assets")+");[(p/x).mkdir() for x in "+repr(ids)+"]"
  rc,out,err=c.execute("python3 -c "+shlex.quote(code),timeout=10)
  if rc:raise SystemExit(err)
  for original_id in ids:
   for name in ("physical_object_rgba.png","target_geometry.obj","target_surface.npy"):
    upload(PREP/"assets"/original_id/name,"input/assets/"+original_id+"/"+name)
  for name in plan["tool_pins"]:
   path=OWN/"experiments/chimera_recovery"/name
   if sha(path)!=plan["tool_pins"][name]:raise SystemExit("immutable target tool changed:"+name)
   upload(path,"tools/"+name)
  argv=["docker","run","--detach","--name","codex-hy3d-target-session-cpu-oct13-v1","--user","1000:1000",
   "--runtime","runc","--network","none","--no-healthcheck","--cpus","8","--memory","32g","--memory-swap","32g",
   "--read-only","--pids-limit","128","--security-opt","no-new-privileges","--cap-drop","ALL","--tmpfs","/tmp:rw,size=64m",
   "--env","NVIDIA_VISIBLE_DEVICES=void","--env","CUDA_VISIBLE_DEVICES=","--env","PYTHONDONTWRITEBYTECODE=1",
   "--env","HY3DGEN_DEBUG=0","--env","PYTHONPATH=/opt/cpu-deps","--env","OMP_NUM_THREADS=8","--env","MKL_NUM_THREADS=8",
   "--mount","type=bind,src="+REMOTE+"/input,dst=/work/input,readonly",
   "--mount","type=bind,src="+REMOTE+"/tools,dst=/work/tools,readonly",
   "--mount","type=bind,src="+REMOTE+"/output,dst=/work/output",
   "--mount","type=bind,src="+ROOT+"/hy3d-cpu-deps-v1/site,dst=/opt/cpu-deps,readonly",
   "--mount","type=bind,src="+SOURCE+",dst="+SOURCE+",readonly",
   "--mount","type=bind,src="+WEIGHTS+",dst="+WEIGHTS+",readonly",
   "--entrypoint","timeout",IMAGE,"--signal=TERM","--kill-after=15","1200",
   "/opt/environments/python/comfyui/bin/python","-B","/work/tools/hy3d_target_session.py","run",
   "--plan","/work/input/plan.json","--plan-sha256",pin,"--input-root","/work/input",
   "--source-root",SOURCE,"--weights",WEIGHTS,"--output","/work/output/proof","--threads","8"]
  rc,out,err=c.execute(shlex.join(argv),timeout=15)
  receipt={"schema":"recovery_hy3d_target_cpu_launch.v1","argv":argv,"plan_sha256":pin,"files":files,
     "container":out.strip(),"exit_code":rc,"stderr":err,"image":IMAGE,"cpu_only":True,"max_seconds":1200,
     "mem_available_before":memory,"training_started":False,"launcher_sha256":sha(__file__)}
  dump(LOCAL/"launch.json",receipt);print(json.dumps({k:receipt[k] for k in ('container','exit_code','plan_sha256','mem_available_before')}))
  if rc:raise SystemExit(rc)
 elif sys.argv[1]=="status":
  if not c.download_file(REMOTE+"/output/proof/status.json",str(LOCAL)):raise SystemExit("no completed row yet")
  status=json.loads((LOCAL/"status.json").read_text());print(json.dumps({k:v for k,v in status.items() if k!='receipts'}))
 elif sys.argv[1]=="collect":
  if not c.download_file(REMOTE+"/output/proof/manifest.json",str(LOCAL)):raise SystemExit("final manifest not available")
  m=json.loads((LOCAL/"manifest.json").read_text());files={"manifest.json":sha(LOCAL/"manifest.json")}
  for ref in m["targets"]:
   directory=LOCAL/ref["id"];directory.mkdir(exist_ok=True)
   for name in ("receipt.json","captured_parent_surface.npy","clean_target.safetensors","sample_native_mc.obj"):
    if not c.download_file(REMOTE+"/output/proof/"+ref["id"]+"/"+name,str(directory)):
     raise SystemExit("missing actual target artifact:"+ref["id"]+"/"+name)
    files[ref["id"]+"/"+name]=sha(directory/name)
  dump(LOCAL/"collected_files.json",{"files":files,"container_state_not_polled":True})
  print(json.dumps({'rows':m['rows'],'source_train_eligible':m['source_train_eligible'],
    'held_sources_retained':m['held_sources_retained'],'original_weights_unchanged':m['original_weights_unchanged'],
    'native_train_ready':m['native_train_ready'],'elapsed_seconds':m['elapsed_seconds'],'manifest_file_sha256':sha(LOCAL/'manifest.json')}))
 else:raise SystemExit("launch, status or collect only")
finally:c.close()
