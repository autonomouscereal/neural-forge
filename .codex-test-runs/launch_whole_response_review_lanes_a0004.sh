#!/usr/bin/env bash
set -euo pipefail

SHA=02cec5c82b6820a77e273af38b78cc0b0880285a
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0004

launch_lane() {
  local lane=$1
  local gpu=$2
  local run=$BASE/$lane/terminal-frontier
  local receipt=$run/frontier-receipt.json
  local dossier=$run/whole-response-dossier.json

  test -s "$receipt"
  test ! -e "$dossier"

  local run_sha
  local run_file_sha
  run_sha=$(jq -r .run_sha256 "$receipt")
  run_file_sha=$(sha256sum "$receipt" | awk '{print $1}')

  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONPATH="$CODE/src" \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MKL_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    OMP_NUM_THREADS=16 \
    OPENBLAS_NUM_THREADS=16 \
    /home/cereal/model-surgery-lab/venv/bin/python -u \
    "$CODE/experiments/review_lfm2_semantic_frontier.py" \
    --frontier-run-receipt "$receipt" \
    --expected-frontier-run-sha256 "$run_sha" \
    --expected-frontier-run-file-sha256 "$run_file_sha" \
    --prompts-json "$CODE/research/lfm2_semantic_whole_response_prompts_v1.json" \
    --output-json "$dossier" \
    --device cuda:0 \
    --dtype bfloat16 \
    --maximum-prompt-tokens 1024 \
    --maximum-new-tokens 256 \
    --parent-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json \
    --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
    --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
    >"$run/review.log" 2>&1 </dev/null &

  local pid=$!
  printf '%s\n' "$pid" >"$run/review.pid"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s gpu=%s pid=%s\n' "$lane" "$gpu" "$pid"
  else
    cat "$run/review.log"
    return 1
  fi
}

launch_lane fit_guided 0
launch_lane full_strength_structural 4
launch_lane residual_writer 6
