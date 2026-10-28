set -euo pipefail
python3 - <<'PY'
import hashlib,json,pathlib,subprocess
base=pathlib.Path('/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1')
rels=['configs/dit_config_2_1.yaml','hy3dshape/hy3dshape/surface_loaders.py','hy3dshape/hy3dshape/models/autoencoders/model.py','hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py','hy3dshape/hy3dshape/models/autoencoders/distributions.py','hy3dshape/hy3dshape/pipelines.py']
files={}
for r in rels:
 p=base/r
 files[r]={'path':str(p),'exists':p.is_file()}
 if p.is_file(): files[r].update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
w=pathlib.Path('/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors')
print(json.dumps({'source_files':files,'weight':{'path':str(w),'exists':w.is_file(),'bytes':w.stat().st_size if w.is_file() else None},'containers':subprocess.check_output(['docker','ps','--format','{{.Names}} {{.Image}}'],text=True)},indent=2))
PY
