from pathlib import Path
import hashlib
import json
import shutil
import sys
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient, ServerConfig
HERE = Path(__file__).resolve().parent
ROOT = '/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
plan = json.loads((HERE / 'plan.json').read_text())
inputs = HERE / 'inputs'
inputs.mkdir(exist_ok=False)
for entry in plan['files']:
    source = Path(entry['local_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('launch input SHA changed: ' + str(source))
    relative = Path(entry['private_relative_path']).relative_to('audio-cache-inputs-v1')
    target = inputs / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
client = SSHClient.from_config(ServerConfig().get('ai'))
if not client.connect():
    raise RuntimeError('server-manager connect failed')
try:
    if not client.upload_directory(str(inputs), ROOT + '/audio-cache-inputs-v1'):
        raise RuntimeError('input upload failed')
    script = '''set -euo pipefail
test ! -e "{root}/clean-audio-cache-v1"
cd "{root}"
docker run -d --name codex-clean-audio-cache-20261013-v1 --network none --cpus 4 --memory 8g \
  --entrypoint /opt/environments/python/comfyui/bin/python \
  -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/work \
  -v "{root}:/work:rw" \
  -v "{root}/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro" \
  -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro \
  sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1 \
  /work/experiments/chimera_recovery/audio_cache.py encode-source \
  --source /work/audio-cache-inputs-v1/source.jsonl \
  --source-preflight /work/audio-cache-inputs-v1/source-preflight.json \
  --source-preflight-sha256 26a0e830d863afe90fcd426242a5bdec65e4b481944c6b78af4573d609251f95 \
  --asset-root /work/audio-cache-inputs-v1/assets \
  --stage-c /work/audio-cache-inputs-v1/stage_c_media_a.py \
  --comfy-root /opt/ComfyUI \
  --audio-vae /opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors \
  --out /work/clean-audio-cache-v1
'''.format(root=ROOT)
    code, output, error = client.execute_script(script)
    result = {'script': script, 'exit_code': code, 'stdout': output, 'stderr': error,
              'source_plan_sha256': hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest(),
              'training_started': False, 'gpu_exposed': False}
    (HERE / 'launch.json').write_text(json.dumps(result, sort_keys=True) + '\n')
    print(json.dumps({'exit_code': code, 'container_id': output.strip(), 'stderr': error}))
    if code:
        raise RuntimeError('isolated CPU encode launch failed')
finally:
    client.close()
