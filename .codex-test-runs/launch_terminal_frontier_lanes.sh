#!/usr/bin/env bash
set -euo pipefail

SHA=48e79136785da21bc497dae42e099e37b5f8b36d
SHORT=${SHA:0:7}
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/$SHORT/a0003

launch_lane() {
  local lane=$1
  local gpu=$2
  local body=$BASE/$lane
  local postfold=$body/postfold-receipt.json
  local run=$body/terminal-frontier

  test -s "$postfold"
  test ! -e "$run"
  mkdir -p "$run"

  local capsule
  local capsule_file_sha
  local capsule_sha
  local bindings_sha
  local cache_key
  capsule=$(jq -r .preterminal_capsule.path "$postfold")
  capsule_file_sha=$(jq -r .preterminal_capsule.receipt_file_sha256 "$postfold")
  capsule_sha=$(jq -r .preterminal_capsule.receipt_sha256 "$postfold")
  bindings_sha=$(jq -r .preterminal_capsule.bindings_sha256 "$postfold")
  cache_key=$(jq -r .semantic_evidence_cache_key "$postfold")

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
    "$CODE/experiments/run_lfm2_semantic_terminal_frontier.py" \
    --preterminal-capsule "$capsule" \
    --expected-preterminal-receipt-file-sha256 "$capsule_file_sha" \
    --expected-preterminal-receipt-sha256 "$capsule_sha" \
    --expected-preterminal-bindings-sha256 "$bindings_sha" \
    --trace-cache /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --expected-cache-key "$cache_key" \
    --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
    --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
    --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
    --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
    --frontier-bank "$run/frontier-bank" \
    --materializations "$run/materializations" \
    --receipt-json "$run/frontier-receipt.json" \
    --device cuda:0 \
    --dtype bfloat16 \
    >"$run/frontier.log" 2>&1 </dev/null &

  local pid=$!
  printf '%s\n' "$pid" >"$run/frontier.pid"
  sleep 1
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s gpu=%s pid=%s\n' "$lane" "$gpu" "$pid"
  else
    cat "$run/frontier.log"
    return 1
  fi
}

launch_lane fit_guided 0
launch_lane full_strength_structural 4
launch_lane residual_writer 6
