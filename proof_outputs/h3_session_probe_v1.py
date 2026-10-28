"""Owned bounded genuine two-wave CPU session proof, no model training."""
import hashlib
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import ServerConfig, SSHClient
HERE = Path(__file__).resolve().parent
remote = '/home/cereal/omni-campaign/distill/codex-speech-tests-modality-01a1326f-v3'
previous = '/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
plan = json.loads((HERE / 'audio_native_cache_run_v1/plan.json').read_text())
assets = [item for item in plan['files'] if '/assets/' in item['private_relative_path']][:2]
cases = [{'audio': '/previous/' + x['private_relative_path'], 'audio_sha256': x['sha256'],
          'reference': '/previous/clean-audio-cache-v1/' + f'{i:06d}.json'} for i, x in enumerate(assets)]
program = '''from pathlib import Path
from types import SimpleNamespace
import json,sys,time
sys.path.insert(0,'/source/experiments/chimera_recovery')
import h3_audio_probe as P
import torch
from safetensors.torch import load_file
base=dict(stage_c=Path('/previous/audio-cache-inputs-v1/stage_c_media_a.py'),comfy_root=Path('/opt/ComfyUI'),
audio_vae=Path('/opt/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors'),audio_source_sha256=P.AUDIO_SOURCE_SHA)
started=time.monotonic()
session=P.EncoderSession(SimpleNamespace(**base))
entries=[]
for i,case in enumerate(CASES):
    args=SimpleNamespace(**base,audio=Path(case['audio']),audio_sha256=case['audio_sha256'],out=Path('/output')/f'{i:06d}.json')
    result=P.run(args,session=session)
    assert result['probe_passed'], result.get('failure')
    reference=json.loads(Path(case['reference']).read_text())
    oldpath=Path('/previous/clean-audio-cache-v1')/Path(reference['tensors']['path']).name
    assert P.sha(oldpath)==reference['tensors']['sha256']
    old=load_file(str(oldpath),device='cpu')
    new=load_file(str(args.out.with_suffix('.safetensors')),device='cpu')
    assert set(old)==set(new) and all(torch.equal(old[k],new[k]) for k in old), 'actual separate-runtime tensors differ'
    entries.append({'asset_sha256':case['audio_sha256'],'proof':str(args.out),'proof_sha256':P.sha(args.out),
    'separate_runtime_reference_sha256':P.sha(Path(case['reference'])),'all_tensors_bitwise_equal':True})
sealed=session.seal()
assert sealed['actual_waveforms']==len(entries)==2
result={'schema':'chimera_recovery.h3_reusable_cpu_actual_proof.v1','passed':True,'session':sealed,'entries':entries,
'wall_seconds':time.monotonic()-started,'adapter_sha256':P.sha(Path(P.__file__)),'training_started':False,'gpu_exposed':False}
Path('/output/result.json').write_bytes(P.canonical(result)+b'\\n')
print(json.dumps({'passed':True,'waveforms':2,'wall_seconds':result['wall_seconds'],'result_sha256':P.sha(Path('/output/result.json'))}),flush=True)
'''.replace('CASES', repr(cases))
path = HERE / 'h3_session_actual_v1.py'
path.write_bytes(program.encode('utf-8'))
client = SSHClient.from_config(ServerConfig().get('ai'), control=True)
if not client.connect():
    raise RuntimeError('server-manager connect failed')
try:
    if not client.upload_file(str(path), remote + '/session_probe.py'):
        raise RuntimeError('session probe upload failed')
    script = '''set -euo pipefail
test -f "{remote}/owner.json"
test ! -e "{remote}/session-proof"
mkdir "{remote}/session-proof"
chmod 0777 "{remote}/session-proof"
docker run -d --name codex-h3-reusable-session-modality-v1 --network none --cpus 4 --memory 8g --no-healthcheck \
 --entrypoint /opt/environments/python/comfyui/bin/python \
 -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/source \
 -v "{remote}:/source:ro" -v "{remote}/session-proof:/output:rw" \
 -v "{previous}:/previous:ro" -v /home/cereal/comfyui/models:/opt/ComfyUI/models:ro \
 sha256:5070872ac7aef716eadc60ec92b0f944521c7899d90c6899a7b1ed1e883942e1 /source/session_probe.py
'''.format(remote=remote, previous=previous)
    code, out, err = client.execute_script(script)
    (HERE / 'h3_session_actual_v1_launch.json').write_text(json.dumps({'remote': remote, 'script': script,
        'exit_code': code, 'stdout': out, 'stderr': err, 'program_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'training_started': False}, indent=2) + '\n')
    print(json.dumps({'exit_code': code, 'stdout': out, 'stderr': err}))
    if code:
        raise RuntimeError('bounded CPU session launch failed')
finally:
    client.close()
