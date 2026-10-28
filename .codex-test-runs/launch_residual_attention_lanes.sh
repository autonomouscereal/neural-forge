#!/usr/bin/env bash
set -euo pipefail

SHA=48e79136785da21bc497dae42e099e37b5f8b36d
SHORT=${SHA:0:7}
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
COMPANION_ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_interface_frontier_companion_20260722/$SHORT/a0001
COMPANION=$COMPANION_ROOT/frontier-companion.json
COMPANION_FILE_SHA=$(sha256sum "$COMPANION" | awk '{print $1}')
COMPANION_SHA=$(jq -r .companion_sha256 "$COMPANION")
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/$SHORT/a0003

test -d "$CODE"
test -s "$COMPANION"
mkdir -p "$BASE"

launch_lane() {
  local recipe=$1
  local gpu=$2
  local run=$BASE/$recipe
  local pidfile=$run/postfold.pid

  test ! -e "$run"
  mkdir -p "$run"

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
    "$CODE/experiments/run_lfm2_semantic_postfold.py" \
    --input-checkpoint /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/checkpoint \
    --expected-input-artifact-sha256 517515e3525c107db180953dcde6c1f5dc2ab6af3bc0f50486e442d6bf69899f \
    --output-checkpoint "$run/checkpoint" \
    --preterminal-capsule "$run/preterminal-capsule" \
    --receipt-json "$run/postfold-receipt.json" \
    --trace-cache /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
    --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
    --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --compiler-diagnostic-json /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/foundation-receipt.json \
    --expected-compiler-diagnostic-sha256 7bffddca0fc90e026a62d625cff9f14b436955dbc799f9d8cdb804dc54e07afd \
    --interface-statistics-directory /home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0002/statistics-store/42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a \
    --expected-interface-statistics-sha256 42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a \
    --interface-frontier-companion-json "$COMPANION" \
    --expected-interface-frontier-companion-file-sha256 "$COMPANION_FILE_SHA" \
    --expected-interface-frontier-companion-sha256 "$COMPANION_SHA" \
    --anchor-candidate-index 7 \
    --device cuda:0 \
    --fold-compute-device cuda:0 \
    --body-recipe "$recipe" \
    --dtype bfloat16 \
    --tied-interface-mode strong-congruence-plus-anchor-residual \
    >"$run/postfold.log" 2>&1 </dev/null &

  local pid=$!
  printf '%s\n' "$pid" >"$pidfile"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s gpu=%s pid=%s\n' "$recipe" "$gpu" "$pid"
  else
    cat "$run/postfold.log"
    return 1
  fi
}

launch_lane fit_guided 0
launch_lane full_strength_structural 4
launch_lane residual_writer 6
