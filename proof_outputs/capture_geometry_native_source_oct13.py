import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,"C:/Users/cereal/.agents/skills/server-manager")
from ssh_client import SSHClient,ServerConfig
out=Path("C:/Users/cereal/Documents/ai training/reviews/2026-10-13-folding/recovery-preparation/geometry-native-source-v1")
out.mkdir(parents=True,exist_ok=True)
src="/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9/src/chimera_omni"
clone="/mnt/lexar/omni-campaign/clones/OC02-F-Q38-27BU-OMNI1.clone-pre-D1"
paths={name:src+"/"+name for name in ("config.py","charted.py","modeling/chimera.py","modeling/charted.py","modeling/hy3d.py","modeling/layers.py")}
paths.update({"clone_config.json":clone+"/config.json","model.safetensors.index.json":clone+"/model.safetensors.index.json"})
c=SSHClient.from_config(ServerConfig().get("ai"),control=True)
if not c.connect():raise SystemExit("connection failed")
pins={}
try:
 for name,path in paths.items():
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not c.download_file(path,str(dest.parent)):raise SystemExit("missing exact current source:"+path)
  got=dest.parent/Path(path).name
  if got!=dest:got.rename(dest)
  pins[name]={"path":path,"local_path":str(dest),"sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),"bytes":dest.stat().st_size}
finally:c.close()
(out/"pins.json").write_text(json.dumps(pins,indent=2)+"\n",encoding="utf-8")
print(json.dumps(pins,indent=2))
