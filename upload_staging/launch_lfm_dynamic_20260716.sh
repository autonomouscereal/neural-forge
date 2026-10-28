#!/usr/bin/env bash
set -euo pipefail

cd /home/cereal/model-surgery-lab/neural-forge-src

PY=/home/cereal/model-surgery-lab/venv/bin/python
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_dynamic_fold_20260716
CACHE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/circuit-cache
CAL=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_fold_20260716/calibration_parent_greedy.json
MODEL=/home/cereal/model-surgery-lab/models/LFM2.5-230M

mkdir -p "$BASE"

launch() {
  local name="$1"
  local gpu="$2"
  local width="$3"
  local layers="$4"
  local out="$BASE/$name"

  if [[ -e "$out" ]]; then
    echo "refusing existing $out" >&2
    exit 1
  fi
  mkdir -p "$out"
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONPATH=/home/cereal/model-surgery-lab/neural-forge-src/src \
    "$PY" experiments/run_lfm2_residual_attention_fold.py \
      --parent-model "$MODEL" \
      --calibration "$CAL" \
      --cache-root "$CACHE" \
      --output-root "$out" \
      --device cuda:0 \
      --dtype bfloat16 \
      --target-query-heads 16 \
      --target-key-value-heads 8 \
      --target-intermediate-size "$width" \
      --max-layers "$layers" \
      --max-selection-rows 256 \
      --sample-last-tokens 8 \
      --max-decode-steps 8 \
      --max-capture-bytes 134217728 \
      --capture-logit-rtol 0.2 \
      --capture-logit-atol 1.0 \
      --endpoint-refinement-passes 2 \
      --max-fit-rows 512 \
      --max-validation-rows 128 \
      --validation-fraction 0.2 \
      --endpoint-ridge 1e-6 \
      --max-endpoint-relative-rmse 1.5 \
      --min-endpoint-cosine-similarity -0.5 \
      --min-teacher-top1 0.05 \
      --min-free-run-match 0.05 \
      --min-residual-cosine 0.0 \
      --max-score-regression 0.0 \
      --max-layer-rmse-regression 0.0 \
      >"$out/run.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$out/run.pid"
  echo "$name pid=$pid physical_gpu=$gpu width=$width max_layers=$layers"
}

launch m2559_provisional_gpu0 0 2559 0
launch m2559_closedloop_gpu4 4 2559 14
launch m2558_provisional_gpu6 6 2558 0
