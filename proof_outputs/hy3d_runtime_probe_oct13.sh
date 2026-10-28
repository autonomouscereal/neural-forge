set -euo pipefail
docker exec -i -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 p40-hy3d-r1-gpu2 /opt/environments/python/comfyui/bin/python -B - <<'PY'
import json,pathlib,struct,importlib,hashlib
r={'runtime':{},'cuda_visible_devices':'','training_started':False}
for name in ['torch','torch_cluster','trimesh','numpy','safetensors','yaml']:
 try:
  m=importlib.import_module(name); r['runtime'][name]={'file':str(m.__file__),'version':str(getattr(m,'__version__','unknown'))}
 except Exception as e: r['runtime'][name]={'error':type(e).__name__+': '+str(e)}
try:
 import torch,torch_cluster
 p=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 r['fps_cpu_indices']=torch_cluster.fps(p,ratio=.5,random_start=False).tolist()
except Exception as e:r['fps_cpu_error']=type(e).__name__+': '+str(e)
p=pathlib.Path('/home/cereal/comfyui/models/checkpoints/hunyuan_3d_v2.1.safetensors')
with p.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
tensors={k:v for k,v in h.items() if k.startswith('vae.')}
r['weights']={'path':str(p),'bytes':p.stat().st_size,'header_bytes':n,'vae_tensor_count':len(tensors),'vae_elements':sum(__import__('math').prod(v['shape']) for v in tensors.values()),'vae_storage_bytes':sum(v['data_offsets'][1]-v['data_offsets'][0] for v in tensors.values())}
base=pathlib.Path('/home/cereal/comfyui/custom_nodes/ComfyUI-Hunyuan3d-2-1')
rels=['configs/dit_config_2_1.yaml','hy3dshape/hy3dshape/surface_loaders.py','hy3dshape/hy3dshape/models/autoencoders/model.py','hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py','hy3dshape/hy3dshape/preprocessors.py','hy3dshape/hy3dshape/pipelines.py']
r['source_files']={x:{'path':str(base/x),'sha256':hashlib.sha256((base/x).read_bytes()).hexdigest(),'bytes':(base/x).stat().st_size} for x in rels}
print(json.dumps(r,indent=2))
PY
