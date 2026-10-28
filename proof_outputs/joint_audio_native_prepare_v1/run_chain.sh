set -euo pipefail
cd "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1"
trap 'rc=$?; printf "%s\n" "$rc" > logs/chain.exitcode' EXIT
deadline=$((SECONDS+10800))
while [ ! -f "/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/clean-speech-cache-v1/manifest.json" ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'speech seal deadline exceeded\n' >&2; exit 2; fi
 running=$(docker inspect --format '{{.State.Running}}' codex-speech-clean-modality-v1)
 if [ "$running" != true ]; then printf 'speech encoder stopped without a seal\n' >&2; exit 3; fi
 sleep 10
done
# Serialize with the original native CE/speaking chain: one CPU8/32GB
# preparation phase at a time. A failed original chain grants no joint seal.
while [ ! -f "/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/logs/chain.exitcode" ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'speech native chain deadline exceeded\n' >&2; exit 4; fi
 sleep 10
done
test "$(cat '/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/logs/chain.exitcode')" = 0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1');speech=Path('/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/clean-speech-cache-v1/manifest.json')
proc=Path('/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/clean-audio-cache-v1/manifest.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(proc)=='ae08c35f02458b9e36a19431e2d9ff9ffedaf29d4d3ed9118b2d00a9b0572b47'
pins=[{'path':'/work/clean-audio-cache-v1/manifest.json','sha256':sha(proc)},
      {'path':'/work/clean-speech-cache-v1/manifest.json','sha256':sha(speech)}]
(p/'clean_pins.json').write_text(json.dumps(pins)+'\n')
PY
docker run --name codex-joint-audio-compose-modality-v1 --network none --cpus 8 --memory 32g --no-healthcheck --entrypoint /usr/bin/timeout  -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1  -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/joint/source  -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro  -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro  -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1:/joint:rw" -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/source:/joint/source:ro" -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/inputs:/joint/inputs:ro"  -v "/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1:/work:ro"  -v "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro"  -v "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro"  sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 3600 /usr/bin/python3.12 /joint/source/experiments/chimera_recovery/audio_sources.py compose-cache  --source /joint/inputs/admission/files/096270c869454e62b70b65cc97b30cf1ddc76449da7f19a513dec4021b0dd1bc__source_ready.jsonl --admission /joint/inputs/admission/manifest.json --admission-sha256 a07c0139fd073647d8482ef4af4b94c9d677f9a195534e1350adc45aa399ae8c  --clean-manifests /joint/clean_pins.json --out /joint/joint-clean-cache-v1 > logs/compose.log 2>&1
printf '0\n' > logs/compose.exitcode
clean_sha=$(sha256sum joint-clean-cache-v1/manifest.json | cut -d' ' -f1)
docker run --name codex-joint-audio-native-ce-modality-v1 --network none --cpus 8 --memory 32g --no-healthcheck --entrypoint /usr/bin/timeout  -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1  -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/joint/source  -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro  -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro  -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1:/joint:rw" -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/source:/joint/source:ro" -v "/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/inputs:/joint/inputs:ro"  -v "/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1:/work:ro"  -v "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro"  -v "/home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro"  sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 3600 /usr/bin/python3.12 /joint/source/experiments/chimera_recovery/audio_cache.py prepare  --source /joint/inputs/admission/files/096270c869454e62b70b65cc97b30cf1ddc76449da7f19a513dec4021b0dd1bc__source_ready.jsonl --clean-manifest /joint/joint-clean-cache-v1/manifest.json --clean-manifest-sha256 "$clean_sha"  --admission /joint/inputs/admission/manifest.json --admission-sha256 a07c0139fd073647d8482ef4af4b94c9d677f9a195534e1350adc45aa399ae8c  --tokenizer-dir /model --charted-attention /joint/inputs/charted_attention.py  --amendment-sha256 312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055  --out /joint/native-audio-ce-v1 > logs/native-ce.log 2>&1
printf '0\n' > logs/native-ce.exitcode
