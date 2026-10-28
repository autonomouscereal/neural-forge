#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cereal/model-surgery-lab/runtime/neural-forge-961c957
RUN=/home/cereal/model-surgery-lab/artifacts/lfm2_semantic_foundation_20260720/961c957-c61-4x
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="/home/cereal/model-surgery-lab/python_overlays_transformers_4_57_3:$ROOT/src"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16

exec "$PYTHON" -u "$ROOT/experiments/materialize_lfm2_semantic_foundation.py" \
  --parent-model-directory /home/cereal/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M/snapshots/37b30cce3446f3f2e26a0d3f8c67c9167f5079d7 \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --compiler-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --trajectory-corpus /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json \
  --selective-catalog-store-directory /home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_frontier_20260719/8be9f891254519950e114bfd54196ce661bcf5cf/selective-catalogs \
  --selective-catalog-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_frontier_20260719/8be9f891254519950e114bfd54196ce661bcf5cf/selective-catalog-preparation.json \
  --expected-selective-catalog-receipt-sha256 54841de1e3bb9a84884b2c321388c6e2d5074da0e25ebc72079a6349db544f67 \
  --output-checkpoint-directory "$RUN/checkpoint" \
  --foundation-receipt-json "$RUN/foundation-receipt.json" \
  --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-cache-key 76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --expected-target-plan-fingerprint 638dd9860d4e0a5cf52404af4611fc56792045abcc8ad25fff5cb35222444a96 \
  --expected-selected-candidate-fingerprint bc74a6aba1e423b98545912acae55be24fcb301f3cc80538db0eaa815a58dc89 \
  --child-name lfm2-cpcf-v5-4x \
  --model-device cuda:0 \
  --dtype bfloat16
