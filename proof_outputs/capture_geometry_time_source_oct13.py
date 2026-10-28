import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig
out=Path("C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/geometry-native-source-v1")
src="/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1/hy3dshape/hy3dshape"
paths={name:src+"/"+name for name in ("schedulers.py","models/denoisers/hunyuandit.py")}
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("connection failed")
pins={}
try:
 for name,path in paths.items():
  dest=out/"native_parent"/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not c.download_file(path,str(dest.parent)):raise SystemExit("missing actual time source:"+path)
  pins[name]={"path":path,"local_path":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"bytes":dest.stat().st_size}
finally:c.close()
(out/"time_source_pins.json").write_text(json.dumps(pins,indent=2)+"\n",encoding="utf-8")
print(json.dumps(pins,indent=2))
