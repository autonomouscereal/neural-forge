import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig
OWN=Path(__file__).resolve().parents[1];MAIN=Path("C:/Users/cereal/Documents/ai training")
LOCAL=MAIN/"reviews/2026-10-13-folding/recovery-preparation/geometry-bridge-cpu-v1"
ROOT="/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013"
REMOTE=ROOT+"/geometry-bridge-cpu-v1"
TARGET=ROOT+"/hy3d-target-session-v1/output/proof"
TREE="/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9"
CLONE="/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"
SOURCE="/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1"
WEIGHTS="/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors"
IMAGE="sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
LOCAL.mkdir(parents=True,exist_ok=True)
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("connection failed")
try:
 if sys.argv[1]=="launch":
  rows=[json.loads(x) for x in (MAIN/"proof_outputs/chimera_geometry_targets_oct13_v1/prepared_targets.jsonl").read_text().splitlines()]
  selected=[next(x for x in rows if x["family"]==family) for family in ("mesh_box","mesh_octahedron")]
  plan={"schema":"recovery_geometry_bridge_probe_plan.v1","sample_seed":20261013,"target_xyz_input_allowed":False,"sources":[]}
  for row in selected:
   item=row["id"]
   spec={k:row[k] for k in ("id","family","split","source_row_sha256")}
   spec["target"]={"path":TARGET+"/"+item+"/clean_target.safetensors","sha256":row["target_only"]["latent"]["sha256"]}
   spec["target_receipt"]={"path":TARGET+"/"+item+"/receipt.json","sha256":row["target_only"]["native_receipt"]["sha256"]}
   spec["image_ingress"]={"path":"/work/input/"+item+".npz","sha256":row["inputs"]["raw_image_ingress"]["sha256"]}
   plan["sources"].append(spec)
  dump(LOCAL/"plan.json",plan)
  code="from pathlib import Path;import json;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','tools','output']];print(json.dumps({'mem_available_kib':int([x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1])}))"
  rc,out,err=c.execute("python3 -c "+shlex.quote(code),timeout=10)
  if rc:raise SystemExit(err)
  memory=json.loads(out)
  if memory["mem_available_kib"]<40*1024**2:raise SystemExit("insufficient actual MemAvailable for bounded32GiB proof")
  files={}
  def upload(path,name):
   if not c.upload_file(str(path),REMOTE+"/"+name):raise SystemExit("upload failed:"+name)
   files[name]=sha(path)
  upload(LOCAL/"plan.json","input/plan.json")
  for row in selected:upload(row["inputs"]["raw_image_ingress"]["path"],"input/"+row["id"]+".npz")
  for name in ("geometry_bridge_probe.py","hy3d_target_session.py","hy3d_surface_probe.py","geometry_conditioning.py","geometry_prepare.py","audio_admission.py"):
   upload(OWN/"experiments/chimera_recovery"/name,"tools/"+name)
  argv=["docker","run","--detach","--name","codex-geometry-bridge-cpu-oct13-v1","--user","1000:1000","--runtime","runc",
   "--network","none","--no-healthcheck","--cpus","8","--memory","32g","--memory-swap","32g","--read-only","--pids-limit","128",
   "--security-opt","no-new-privileges","--cap-drop","ALL","--tmpfs","/tmp:rw,size=64m",
   "--env","NVIDIA_VISIBLE_DEVICES=void","--env","CUDA_VISIBLE_DEVICES=","--env","PYTHONDONTWRITEBYTECODE=1",
   "--env","HY3DGEN_DEBUG=0","--env","PYTHONPATH=/opt/cpu-deps","--env","OMP_NUM_THREADS=8","--env","MKL_NUM_THREADS=8",
   "--mount","type=bind,src="+REMOTE+"/input,dst=/work/input,readonly","--mount","type=bind,src="+REMOTE+"/tools,dst=/work/tools,readonly",
   "--mount","type=bind,src="+REMOTE+"/output,dst=/work/output","--mount","type=bind,src="+ROOT+"/hy3d-cpu-deps-v1/site,dst=/opt/cpu-deps,readonly"]
  for path in (TREE,CLONE,SOURCE,WEIGHTS,TARGET):argv+=['--mount','type=bind,src='+path+',dst='+path+',readonly']
  argv += ["--entrypoint","timeout",IMAGE,"--signal=TERM","--kill-after=15","1200","/opt/environments/python/comfyui/bin/python","-B",
   "/work/tools/geometry_bridge_probe.py","--tree",TREE,"--clone",CLONE,"--source-root",SOURCE,"--weights",WEIGHTS,
   "--plan","/work/input/plan.json","--plan-sha256",sha(LOCAL/"plan.json"),"--output","/work/output/proof"]
  rc,out,err=c.execute(shlex.join(argv),timeout=15)
  receipt={"schema":"recovery_geometry_bridge_cpu_launch.v1","argv":argv,"plan_sha256":sha(LOCAL/"plan.json"),"files":files,
   "container":out.strip(),"exit_code":rc,"stderr":err,"image":IMAGE,"mem_available_before":memory,
   "cpu_only":True,"max_seconds":1200,"training_started":False,"launcher_sha256":sha(__file__)}
  dump(LOCAL/"launch.json",receipt);print(json.dumps({k:receipt[k] for k in ('container','exit_code','mem_available_before')}))
  if rc:raise SystemExit(rc)
 elif sys.argv[1]=="status":
  if c.download_file(REMOTE+"/output/proof/status.json",str(LOCAL)):print((LOCAL/"status.json").read_text())
 elif sys.argv[1]=="collect":
  if not c.download_file(REMOTE+"/output/proof/probe.json",str(LOCAL)):raise SystemExit("final actual proof not available")
  result=json.loads((LOCAL/"probe.json").read_text());print(json.dumps({k:result[k] for k in ('schema','elapsed_seconds','native_train_ready','receipt_sha256')}))
  print('Actual proof file SHA:',sha(LOCAL/"probe.json"))
 elif sys.argv[1]=="logs":
  receipt=json.loads((LOCAL/"launch.json").read_text());rc,out,err=c.execute('docker logs --tail 35 '+shlex.quote(receipt['container']),timeout=10)
  print(out);print(err)
 else:raise SystemExit("launch/status/collect/logs only")
finally:c.close()
