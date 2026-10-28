#!/usr/bin/env bash
set -euo pipefail

RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260721/97ae4b9/a0002
PIDFILE="$RUN/foundation.pid"
mkdir -p "$RUN"
test ! -e "$RUN/foundation-receipt.json"

if [[ -s "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

nohup env \
  CUDA_VISIBLE_DEVICES=6 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  PYTHONPATH=/home/cereal/model-surgery-lab/python_overlays_transformers_4_57_3:/home/cereal/model-surgery-lab/runtime/neural-forge-97ae4b942852b7407b44c2983b2a26d5475e3b59/src \
  HF_HUB_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=16 \
  /home/cereal/model-surgery-lab/venv/bin/python -u \
  /home/cereal/model-surgery-lab/runtime/neural-forge-97ae4b942852b7407b44c2983b2a26d5475e3b59/experiments/materialize_lfm2_semantic_foundation.py \
  --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --compiler-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --trajectory-corpus /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json \
  --selective-catalog-store-directory /home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_frontier_20260719/8be9f891254519950e114bfd54196ce661bcf5cf/selective-catalogs \
  --selective-catalog-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_selective_catalog_20260721/97ae4b9/a0002/selective-catalog-preparation.json \
  --expected-selective-catalog-receipt-sha256 4fbe0a9750446926c2a63f70d31078af2786971048d4c7f9d45882c4fdeb8dc9 \
  --output-checkpoint-directory "$RUN/checkpoint" \
  --foundation-receipt-json "$RUN/foundation-receipt.json" \
  --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-cache-key 76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --expected-target-plan-fingerprint 638dd9860d4e0a5cf52404af4611fc56792045abcc8ad25fff5cb35222444a96 \
  --expected-selected-candidate-fingerprint bc74a6aba1e423b98545912acae55be24fcb301f3cc80538db0eaa815a58dc89 \
  --child-name lfm2-cpcf-v5-4x \
  --model-device cuda:0 \
  --dtype bfloat16 \
  >"$RUN/foundation.log" 2>&1 </dev/null &

pid=$!
printf '%s\n' "$pid" > "$PIDFILE"
sleep 1
if kill -0 "$pid" 2>/dev/null; then
  echo "launched pid=$pid"
else
  echo "process exited during launch"
  cat "$RUN/foundation.log"
  exit 1
fi
