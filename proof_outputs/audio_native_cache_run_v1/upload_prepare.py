from pathlib import Path
import hashlib
import json
import sys
import tarfile
sys.path.insert(0, 'C:/Users/cereal/.agents/skills/server-manager')
from ssh_client import SSHClient, ServerConfig
HERE = Path(__file__).resolve().parent
ROOT = '/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013'
BASE = Path('C:/Users/cereal/Documents/ai training')
PROMOTED = BASE / '.agent-worktrees/neural-forge-a495ffdd2f/workers/codex/chimera-training-recovery-oct13-eae1b72e2c3f'
bundle = HERE / 'admission-derived.tar.gz'
if bundle.exists():
    raise ValueError('preserve existing portable transfer bundle')
with tarfile.open(bundle, 'w:gz', compresslevel=1) as archive:
    for path in sorted((HERE / 'admission-derived').rglob('*')):
        if path.is_file():
            archive.add(path, arcname=str(path.relative_to(HERE / 'admission-derived')).replace('\\', '/'), recursive=False)
digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
client = SSHClient.from_config(ServerConfig().get('ai'))
if not client.connect():
    raise RuntimeError('server-manager connect failed')
try:
    for name in ('audio_cache.py', 'audio_paths.py'):
        if not client.upload_file(str(PROMOTED / 'experiments/chimera_recovery' / name), ROOT + '/experiments/chimera_recovery/' + name):
            raise RuntimeError('promoted helper upload failed')
    files = [(bundle, ROOT + '/audio-admission-derived-v1.tar.gz'),
        (BASE / 'reviews/2026-10-13-folding/recovery-preparation/audio-preregistration-v1/PREREGISTRATION_AUDIO_RECOVERY_2026-10-13.md', ROOT + '/audio-cache-inputs-v1/selected-audio-preregistration.md'),
        (BASE / 'chimera-omni/src/chimera_omni/modeling/charted.py', ROOT + '/audio-cache-inputs-v1/charted_attention.py')]
    for source, target in files:
        if not client.upload_file(str(source), target):
            raise RuntimeError('portable admission/amendment/chart upload failed')
    script = '''set -euo pipefail
cd "{root}"
test ! -e audio-admission-derived-v1
test ! -e native-audio-batches-v1
printf '%s  %s\n' '{sha}' audio-admission-derived-v1.tar.gz | sha256sum --check -
mkdir audio-admission-derived-v1
tar --extract --gzip --file audio-admission-derived-v1.tar.gz --directory audio-admission-derived-v1
docker run -d --name codex-native-audio-prepare-20261013-v1 --network none --cpus 4 --memory 8g --no-healthcheck \
  -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/work \
  -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
  -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \
  -v "{root}:/work:rw" \
  -v "{root}/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro" \
  -v "{root}/audio-admission-derived-v1:/work/audio-admission-derived-v1:ro" \
  sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 \
  /usr/bin/python3.12 /work/experiments/chimera_recovery/audio_cache.py prepare \
  --source /work/audio-cache-inputs-v1/source.jsonl \
  --clean-manifest /work/clean-audio-cache-v1/manifest.json \
  --clean-manifest-sha256 "$(sha256sum clean-audio-cache-v1/manifest.json | cut -d' ' -f1)" \
  --admission /work/audio-admission-derived-v1/manifest.json \
  --admission-sha256 "$(sha256sum audio-admission-derived-v1/manifest.json | cut -d' ' -f1)" \
  --tokenizer-dir /model \
  --charted-attention /work/audio-cache-inputs-v1/charted_attention.py \
  --amendment-sha256 61eae20c433179431342e4e6fc21679110e42b6b702609bf44c5407a072510ec \
  --out /work/native-audio-batches-v1
'''.format(root=ROOT, sha=digest)
    code, output, error = client.execute_script(script)
    result = {'script': script, 'exit_code': code, 'stdout': output, 'stderr': error,
        'bundle_sha256': digest, 'bundle_bytes': bundle.stat().st_size, 'training_started': False, 'gpu_exposed': False}
    (HERE / 'prepare_launch.json').write_text(json.dumps(result, sort_keys=True) + '\n')
    print(json.dumps({'exit_code': code, 'stdout': output, 'stderr': error, 'bundle_bytes': bundle.stat().st_size}))
    if code:
        raise RuntimeError('native CPU preparation launch failed')
finally:
    client.close()
