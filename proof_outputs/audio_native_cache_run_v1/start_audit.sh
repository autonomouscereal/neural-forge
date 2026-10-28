set -euo pipefail
cd /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013
mkdir native-audio-audit-v1
docker run -d --name codex-audio-native-audit-20261013-v1 --network none --cpus 4 --memory 8g --no-healthcheck \
 -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=4 \
 -e PYTHONPATH=/opt/teacher-runtime/lib/python3.12/site-packages:/work \
 -v /home/cereal/model-surgery-lab/venv:/opt/teacher-runtime:ro \
 -v /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013:/work:ro \
 -v /home/cereal/omni-campaign/distill/codex-preparation-01a1326f-20261013/native-audio-audit-v1:/audit:rw \
 sha256:20e01d3a0d9cd8eda1cde410870c8111918f1e6e06cadfc9016dd4069c223164 \
 /usr/bin/python3.12 /work/audio-native-audit-v1.py
