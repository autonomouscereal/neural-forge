set -euo pipefail
cd /home/cereal/omni-campaign/distill/codex-native-replay-audit-modality-v1
trap 'rc=$?; printf "%s\n" "$rc" > logs/chain.exitcode' EXIT
deadline=$((SECONDS+10800))
while [ ! -f /home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/logs/chain.exitcode ]; do
 if [ "$SECONDS" -ge "$deadline" ]; then printf 'joint native seal deadline exceeded\n' >&2; exit 2; fi
 sleep 15
done
test "$(cat /home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/logs/chain.exitcode)" = 0
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('/home/cereal/omni-campaign/distill/codex-native-replay-audit-modality-v1');ce=Path('/home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/native-audio-ce-v1/manifest.json');flow=Path('/home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/native-speech-flow-v1/manifest.json')
def pin(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(p/'manifest_pins.json').write_text(json.dumps({'ce':pin(ce),'flow':pin(flow)})+'\n')
PY
ce_sha=$(sha256sum /home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1/native-audio-ce-v1/manifest.json | cut -d' ' -f1)
flow_sha=$(sha256sum /home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1/native-speech-flow-v1/manifest.json | cut -d' ' -f1)
docker run --name codex-native-replay-audit-modality-v1 --network none --cpus 8 --memory 32g --no-healthcheck \
 --entrypoint /usr/bin/timeout -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
 -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e PYTHONDONTWRITEBYTECODE=1 \
 -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages \
 -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
 -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \
 -v /home/cereal/omni-campaign/distill/codex-native-replay-audit-modality-v1:/audit:rw -v /home/cereal/omni-campaign/distill/codex-native-replay-audit-modality-v1/source:/audit/source:ro \
 -v /home/cereal/omni-campaign/distill/codex-joint-audio-preparation-modality-01a1326f-v1:/joint:ro -v /home/cereal/omni-campaign/distill/codex-speech-preparation-modality-01a1326f-v1:/work:ro \
 -v /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/clean-audio-cache-v1:/work/clean-audio-cache-v1:ro \
 -v /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/audio-cache-inputs-v1:/work/audio-cache-inputs-v1:ro \
 sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 7200 /usr/bin/python3.12 /audit/source/audio_replay_index.py \
 --ce-manifest /joint/native-audio-ce-v1/manifest.json --ce-sha256 "$ce_sha" \
 --ce-source-dir /joint/source/experiments/chimera_recovery \
 --flow-manifest /work/native-speech-flow-v1/manifest.json --flow-sha256 "$flow_sha" \
 --flow-source-dir /work/source/experiments/chimera_recovery \
 --amendment-sha256 312706e85efa8fbb996b67f220856adc76236bc516af1ba30f03a365d5b64055 --out /audit/replay-index.json > logs/audit.log 2>&1
printf '0\n' > logs/audit.exitcode
