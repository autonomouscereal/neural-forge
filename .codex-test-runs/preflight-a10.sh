set -euo pipefail

runtime=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
python=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
codebook=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact
trace=/home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b

printf 'runtime=%s\n' "$(test -d "$runtime" && printf ready || printf missing)"
printf 'python=%s\n' "$(test -x "$python" && printf ready || printf missing)"
printf 'codebook=%s\n' "$(test -d "$codebook" && printf ready || printf missing)"
printf 'trace=%s\n' "$(test -d "$trace" && printf ready || printf missing)"
printf 'disk=%s\n' "$(df -h /home/cereal/model-surgery-lab | tail -n 1)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,name,used_memory --format=csv,noheader 2>/dev/null || true
