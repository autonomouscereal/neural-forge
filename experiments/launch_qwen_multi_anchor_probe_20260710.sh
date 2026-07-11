#!/usr/bin/env bash
set -euo pipefail

repo=/home/cereal/model-surgery-lab/neural-forge-pivot
python=/home/cereal/model-surgery-lab/venv/bin/python
source_model=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
calibration=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl
root=/mnt/ai_data/neural-forge/qwen3_4b_multi_anchor_probe_20260710
layer_index=${LAYER_INDEX:-18}
drop_count=${DROP_COUNT:-64}
candidate_pool_size=${CANDIDATE_POOL_SIZE:-512}
max_anchors=${MAX_ANCHORS:-4}
tag=${TAG:-layer${layer_index}_drop${drop_count}_gpu4_r1}
gpu=${GPU:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}
output="$root/$tag/report.json"
log="$root/$tag.log"
pid_file="$root/$tag.pid"
identity_file="$root/$tag.identity.json"

mkdir -p "$root/$tag"
"$python" "$repo/experiments/launch_detached.py" \
  --log "$log" \
  --pid-file "$pid_file" \
  --identity-file "$identity_file" \
  --cwd "$repo" \
  --env "CUDA_VISIBLE_DEVICES=$gpu" \
  --env "PYTHONPATH=$repo/src" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$python" "$repo/experiments/probe_qwen_multi_anchor_layer.py" \
  --source-model "$source_model" \
  --calibration-jsonl "$calibration" \
  --output "$output" \
  --device cuda:0 \
  --dtype float16 \
  --layer-index "$layer_index" \
  --drop-count "$drop_count" \
  --calibration-count 64 \
  --validation-count 64 \
  --activation-rows 256 \
  --max-length 160 \
  --protected-fraction 0.05 \
  --candidate-pool-size "$candidate_pool_size" \
  --max-anchors "$max_anchors" \
  --local-files-only
printf 'launched gpu=%s output=%s log=%s\n' "$gpu" "$output" "$log"
