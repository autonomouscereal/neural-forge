set -euo pipefail
cd /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013
docker run --rm --name codex-audio-dialogue-20261013-v1 --network none \
  -e CUDA_VISIBLE_DEVICES= -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/work \
  -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
  -v /home/cereal/models/Qwen3.8-27B-Uncensored:/model:ro \
  -v /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013:/work:rw \
  sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 \
  /usr/bin/python3.12 /work/experiments/chimera_recovery/audio_dialogue.py \
  --source /work/audio-dialogue-inspection-v1/source.jsonl \
  --source-preflight /work/audio-dialogue-inspection-v1/source-preflight.json \
  --source-preflight-sha256 26a0e830d863afe90fcd426242a5bdec65e4b481944c6b78af4573d609251f95 \
  --tokenizer-dir /model \
  --out /work/audio-dialogue-inspection-v1/inspection.json
