import hashlib,json,shlex,sys
from pathlib import Path
sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig

MAIN=Path("C:/Users/cereal/Documents/ai training")
OWN=Path(__file__).resolve().parents[1]
PREP=MAIN/"proof_outputs/chimera_geometry_conditioning_oct13_v2"
LOCAL=MAIN/"reviews/2026-10-13-folding/recovery-preparation/geometry-conditioning-cpu-v1"
REMOTE="/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/geometry-conditioning-cpu-v1"
IMAGE="sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1"
PIN="fddc42cecfd9ebe54bbe70b2a932e7117aebaee6b3bcf83f1ac75ab426fc2a60"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")

LOCAL.mkdir(parents=True,exist_ok=True)
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("server-manager connection failed")
try:
 if sys.argv[1]=="launch":
  if sha(PREP/"manifest.json")!=PIN:raise SystemExit("immutable ingress changed")
  mkdir="from pathlib import Path;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','input/shards','input/assets','tools','output']]"
  rc,out,err=c.execute("python3 -c "+shlex.quote(mkdir),timeout=10)
  if rc:raise SystemExit(err)
  files={}
  def upload(path,name):
   assert c.upload_file(str(path),REMOTE+"/"+name);files[name]=sha(path)
  for name in ("manifest.json","receipts.jsonl"):upload(PREP/name,"input/"+name)
  for row in map(json.loads,(PREP/"receipts.jsonl").read_text().splitlines()):
   p=Path(row["raw_image_ingress"]["path"]);upload(p,"input/shards/"+p.name)
   upload(Path(row["conditioning_image"]["path"]),"input/assets/"+row["id"]+".png")
  for name in ("geometry_conditioning.py","geometry_prepare.py","audio_admission.py"):
   upload(OWN/"experiments/chimera_recovery"/name,"tools/"+name)
  upload(MAIN/"neural-forge/reports/omni-campaign-2026-09/distill/tools/media_b_teacher_cache.py","tools/media_b_teacher_cache.py")
  upload(MAIN/"neural-forge/reports/omni-campaign-2026-09/composite/tools/s7_pairs_parent.py","tools/s7_pairs_parent.py")
  runner=LOCAL/"run.py"
  runner.write_text("from pathlib import Path\nimport geometry_conditioning as C\nr=C.verify_ingress(manifest=Path('/work/input/manifest.json'),manifest_sha256="+repr(PIN)+",media_b_source=Path('/work/tools/media_b_teacher_cache.py'),s7_source=Path('/work/tools/s7_pairs_parent.py'),assets_root=Path('/work/input/assets'))\nr['source_code_sha256']=C.A.file_sha(Path(C.__file__))\nr['receipt_sha256']=C.A.digest(r)\nPath('/work/output/proof.json').write_bytes(C.A.canonical(r)+b'\\n')\nprint('actual original Torch parity passed:',len(r['checks']))\n")
  upload(runner,"tools/run.py")
  argv=["docker","run","--detach","--name","codex-geometry-conditioning-cpu-oct13-v1","--user","1000:1000",
   "--runtime","runc","--network","none","--no-healthcheck","--cpus","2","--memory","4g","--memory-swap","4g",
   "--read-only","--pids-limit","128","--security-opt","no-new-privileges","--cap-drop","ALL","--tmpfs","/tmp:rw,size=64m",
   "--env","CUDA_VISIBLE_DEVICES=","--env","NVIDIA_VISIBLE_DEVICES=void","--env","PYTHONDONTWRITEBYTECODE=1",
   "--env","OMP_NUM_THREADS=2","--env","MKL_NUM_THREADS=2",
   "--mount","type=bind,src="+REMOTE+"/input,dst=/work/input,readonly",
   "--mount","type=bind,src="+REMOTE+"/tools,dst=/work/tools,readonly",
   "--mount","type=bind,src="+REMOTE+"/output,dst=/work/output",
   "--entrypoint","timeout",IMAGE,"--signal=TERM","--kill-after=15","180",
   "/opt/environments/python/comfyui/bin/python","-B","/work/tools/run.py"]
  rc,out,err=c.execute(shlex.join(argv),timeout=15)
  receipt={"schema":"recovery_geometry_conditioning_cpu_launch.v1","argv":argv,"files":files,"container":out.strip(),
    "exit_code":rc,"stderr":err,"image":IMAGE,"cpu_only":True,"training_started":False,"launcher_sha256":sha(__file__)}
  dump(LOCAL/"launch.json",receipt);print(json.dumps(receipt))
  if rc:raise SystemExit(rc)
 elif sys.argv[1]=="collect":
  if not c.download_file(REMOTE+"/output/proof.json",str(LOCAL)):raise SystemExit("proof not available")
  proof=json.loads((LOCAL/"proof.json").read_text())
  dump(LOCAL/"collected_files.json",{"proof_sha256":sha(LOCAL/"proof.json"),"container_state_not_polled":True})
  print(json.dumps({"proof_sha256":sha(LOCAL/"proof.json"),"actual_original_patchify_executed":proof["actual_original_patchify_executed"],
     "verified_sources":len(proof["checks"]),"native_train_ready":proof["native_train_ready"]}))
 else:raise SystemExit("launch or collect only")
finally:c.close()
