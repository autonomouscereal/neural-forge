#!/usr/bin/env bash
set -euo pipefail

SHA=839758e3b13ba5bb0eb72ad5f4e306a37fafc6a0
CODE=/home/cereal/model-surgery-lab/runtime/neural-forge-$SHA
BASE=/home/cereal/model-surgery-lab/artifacts/lfm2_residual_attention_4x_20260722/48e7913/a0005
FOUNDATION=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/checkpoint
CACHE=/home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b
BRIDGE=/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2
DONOR=/home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M
FOUNDATION_RECEIPT=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0003/foundation-receipt.json
STATISTICS=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_statistics_20260721/02b5c6e/a0002/statistics-store/42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a
COMPANION=/home/cereal/model-surgery-lab/artifacts/lfm2_interface_frontier_companion_20260722/48e7913/a0001/frontier-companion.json
PROMPTS=$CODE/research/lfm2_semantic_whole_response_prompts_v1.json
PARENT_ENVELOPE=/home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json

test -d "$CODE"
test ! -e "$BASE"
mkdir -p "$BASE/logs"

launch_lane() {
  local label=$1
  local recipe=$2
  local recipe_sha=$3
  local anchor=$4
  local gpu=$5
  local run=$BASE/$label
  local log=$BASE/logs/$label.log
  local pidfile=$BASE/logs/$label.pid

  test ! -e "$run"
  test ! -e "$log"
  test ! -e "$pidfile"

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
    "$CODE/experiments/run_lfm2_semantic_fold_to_review.py" \
    --run-directory "$run" \
    --input-checkpoint "$FOUNDATION" \
    --expected-input-artifact-sha256 517515e3525c107db180953dcde6c1f5dc2ab6af3bc0f50486e442d6bf69899f \
    --trace-cache "$CACHE" \
    --expected-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
    --bridge-directory "$BRIDGE" \
    --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
    --donor-tokenizer-directory "$DONOR" \
    --target-tokenizer-directory "$BRIDGE" \
    --parent-model-directory "$PARENT" \
    --compiler-diagnostic-json "$FOUNDATION_RECEIPT" \
    --expected-compiler-diagnostic-sha256 7bffddca0fc90e026a62d625cff9f14b436955dbc799f9d8cdb804dc54e07afd \
    --interface-statistics-directory "$STATISTICS" \
    --expected-interface-statistics-sha256 42083aed1e86665fa99a53e023dfbd613608ac60f86cd7c85c4bb1b92908cd6a \
    --interface-frontier-companion-json "$COMPANION" \
    --expected-interface-frontier-companion-file-sha256 703d7b458f4fcbd346521d0ecfb41ffaf8e262a99cfe480adcf9b3954c4efda9 \
    --expected-interface-frontier-companion-sha256 7fa07c2c6e834f35313ef36b5a2a1d0876d8bc20be72e843a488a7ad5f6a1046 \
    --anchor-candidate-index "$anchor" \
    --body-recipe "$recipe" \
    --expected-body-recipe-sha256 "$recipe_sha" \
    --tied-interface-mode strong-congruence-plus-anchor-residual \
    --device cuda:0 \
    --fold-compute-device cuda:0 \
    --dtype bfloat16 \
    --prompts-json "$PROMPTS" \
    --expected-prompts-file-sha256 ff6fa27e08a3cdbaa647bdeec1d0630d218f98ee13b195a836b85de7438974fa \
    --maximum-prompt-tokens 1024 \
    --maximum-new-tokens 256 \
    --parent-envelope "$PARENT_ENVELOPE" \
    --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
    --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
    >"$log" 2>&1 </dev/null &

  local pid=$!
  printf '%s\n' "$pid" >"$pidfile"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    printf '%s recipe=%s anchor=%s gpu=%s pid=%s\n' \
      "$label" "$recipe" "$anchor" "$gpu" "$pid"
  else
    cat "$log"
    return 1
  fi
}

launch_lane \
  source_producer_residual_writer_anchor_7 \
  source_producer_residual_writer \
  5767c45aeb2b022c50fd084d4c7e17d9451b33b6a188c4fba4e9c705095cc4da \
  7 \
  6

launch_lane \
  source_producer_residual_writer_anchor_0 \
  source_producer_residual_writer \
  5767c45aeb2b022c50fd084d4c7e17d9451b33b6a188c4fba4e9c705095cc4da \
  0 \
  4

launch_lane \
  full_strength_structural_anchor_0 \
  full_strength_structural \
  eed15b90833fcae8b01493b02c23716999f2e4fb1cebca2af7d198e01298e7ff \
  0 \
  0
