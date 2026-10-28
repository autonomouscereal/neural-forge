"""Launch/collect one owned, pinned, CPU-only native VAE diagnostic."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig

WORKTREE=Path(__file__).resolve().parents[1]
MAIN=Path("C:/Users/cereal/Documents/ai training")
PREP=MAIN/"proof_outputs/chimera_geometry_prepare_oct13_v2"
LOCAL=MAIN/"reviews/2026-10-13-folding/recovery-preparation/hy3d-geometry-cpu-v4"
ROOT="/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013"
REMOTE=ROOT+"/hy3d-geometry-cpu-v4"
SOURCE="/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1"
WEIGHTS="/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors"
IMAGE="sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1"
NAME="codex-hy3d-geometry-cpu-oct13-v4"
MANIFEST_SHA="b044870d0373444bc13273e4d987d887980bb8e27a9f06940189523e3777ee6c"
ITEM="geometry-corrected-media-seed-05ed11dae684f64e9798e8af"
ASSET="media-seed-05ed11dae684f64e9798e8af"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2)+"\n",encoding="utf-8")

LOCAL.mkdir(parents=True,exist_ok=True)
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("server-manager connection failed")
try:
    mode=sys.argv[1]
    if mode=="launch":
        if sha(PREP/"manifest.json")!=MANIFEST_SHA:raise SystemExit("prepared immutable source seal changed")
        rc,out,err=c.execute("python3 -c "+shlex.quote("import json,pathlib;print(json.dumps({'mem_available_kib':int([x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')][0].split()[1])}))"),timeout=10)
        if rc:raise SystemExit(err)
        mem=json.loads(out)
        if mem["mem_available_kib"]<40*1024**2:raise SystemExit("bounded32GiB proof requires40GiB MemAvailable")
        code="from pathlib import Path;p=Path("+repr(REMOTE)+");p.mkdir(exist_ok=False);[(p/x).mkdir() for x in ['input','tools','output']]"
        rc,out,err=c.execute("python3 -c "+shlex.quote(code),timeout=10)
        if rc:raise SystemExit(err)
        files={}
        for name in ("manifest.json","source_ready.jsonl"):
            p=PREP/name;assert c.upload_file(str(p),REMOTE+"/input/"+name);files["input/"+name]=sha(p)
        for name in ("physical_object_rgba.png","target_surface.npy","target_geometry.obj"):
            p=PREP/"assets"/ASSET/name;assert c.upload_file(str(p),REMOTE+"/input/"+name);files["input/"+name]=sha(p)
        for name in ("hy3d_surface_probe.py","geometry_prepare.py","audio_admission.py"):
            p=WORKTREE/"experiments/chimera_recovery"/name
            assert c.upload_file(str(p),REMOTE+"/tools/"+name);files["tools/"+name]=sha(p)
        argv=["docker","run","--detach","--name",NAME,"--user","1000:1000","--no-healthcheck","--network","none","--runtime","runc",
              "--cpus","8","--memory","32g","--memory-swap","32g","--pids-limit","128","--read-only",
              "--security-opt","no-new-privileges","--cap-drop","ALL","--tmpfs","/tmp:rw,size=64m",
              "--env","NVIDIA_VISIBLE_DEVICES=void","--env","CUDA_VISIBLE_DEVICES=","--env","HY3DGEN_DEBUG=0",
              "--env","PYTHONPATH=/opt/cpu-deps","--env","PYTHONDONTWRITEBYTECODE=1",
              "--env","OMP_NUM_THREADS=8","--env","MKL_NUM_THREADS=8",
              "--mount","type=bind,src="+REMOTE+"/input,dst=/work/input,readonly",
              "--mount","type=bind,src="+REMOTE+"/tools,dst=/work/tools,readonly",
              "--mount","type=bind,src="+REMOTE+"/output,dst=/work/output",
              "--mount","type=bind,src="+ROOT+"/hy3d-cpu-deps-v1/site,dst=/opt/cpu-deps,readonly",
              "--mount","type=bind,src="+SOURCE+",dst="+SOURCE+",readonly",
              "--mount","type=bind,src="+WEIGHTS+",dst="+WEIGHTS+",readonly",
              "--entrypoint","timeout",IMAGE,"--signal=TERM","--kill-after=15","1200",
              "/opt/environments/python/comfyui/bin/python","-B","/work/tools/hy3d_surface_probe.py",
              "--source-root",SOURCE,"--weights",WEIGHTS,"--prepared-manifest","/work/input/manifest.json",
              "--prepared-manifest-sha256",MANIFEST_SHA,"--item-id",ITEM,"--surface","/work/input/target_surface.npy",
              "--mesh","/work/input/target_geometry.obj","--conditioning","/work/input/physical_object_rgba.png",
              "--output","/work/output/proof","--threads","8","--execute",
              "--native-surface-control","--dense-resolution","32","--compare-posterior-sample"]
        rc,out,err=c.execute(shlex.join(argv),timeout=15)
        receipt={"schema":"chimera.hy3d-clean-geometry-cpu-launch.v1","cpu_only":True,"max_seconds":1200,
                 "mem_available_before":mem,"max_memory_bytes":32*1024**3,"max_threads":8,"image":IMAGE,
                 "remote_job":REMOTE,"container":out.strip(),"exit_code":rc,"stderr":err,"argv":argv,
                 "source_files":files,"launcher_sha256":sha(__file__),"training_started":False}
        dump(LOCAL/"launch.json",receipt)
        print(json.dumps(receipt))
        if rc:raise SystemExit(rc)
    elif mode=="collect":
        launch=json.loads((LOCAL/"launch.json").read_text())
        rc,out,err=c.execute(shlex.join(["docker","inspect","--format","{{json .State}}",launch["container"]]),timeout=10)
        if rc:raise SystemExit(err)
        state=json.loads(out)
        rc,logs,err=c.execute(shlex.join(["docker","logs","--tail","80",launch["container"]]),timeout=10)
        result={"state":state,"log_tail":logs,"log_stderr":err,"training_started":False}
        (LOCAL/"log_tail.txt").write_text(logs+err,encoding="utf-8")
        if not state["Running"]:
            for name in ("probe.json","target_latents.safetensors","captured_parent_surface.npy","mode_native_mc.obj","sample_native_mc.obj"):
                if c.download_file(REMOTE+"/output/proof/"+name,str(LOCAL)):
                    result[name]={"path":str(LOCAL/name),"sha256":sha(LOCAL/name)}
            dump(LOCAL/"completion.json",result)
        print(json.dumps(result))
    else:raise SystemExit("launch or collect only")
finally:c.close()
