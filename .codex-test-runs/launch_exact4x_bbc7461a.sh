#!/usr/bin/env bash
set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/runtimes/neural-forge-d6f5e1c8
RUN_ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_exact4x_composite_20260724/d6f5e1c8
RUN_DIR=${RUN_ROOT}/structural-first-effectful-a0005
LOG=${RUN_DIR}.log
PIDFILE=${RUN_DIR}.pid

if [[ -e "${RUN_DIR}" || -e "${PIDFILE}" ]]; then
  echo "Run directory or PID file already exists: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${RUNTIME}/src
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

nohup /home/cereal/model-surgery-lab/venv/bin/python -u \
  "${RUNTIME}/experiments/run_lfm2_exact4x_composite_fold.py" \
  --run-directory "${RUN_DIR}" \
  --parent-model-directory /home/cereal/model-surgery-lab/models/LFM2.5-230M \
  --donor-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_donor_tokenizer_packages_20260719/eb035507f87ad49002da1bdd10aaf5e727416e2e8cd471014af1672c120d6f46 \
  --target-tokenizer-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --bridge-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridges_20260717/v4096_modelcap_v2 \
  --expected-bridge-artifact-sha256 5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384 \
  --compiler-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --expected-compiler-cache-key 76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95 \
  --semantic-cache-directory /home/cereal/model-surgery-lab/artifacts/lfm2_complete_causal_evidence_20260720_a0003/parent/dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --expected-semantic-cache-key dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b \
  --trajectory-corpus /home/cereal/model-surgery-lab/artifacts/lfm2_bridge_folding_corpora_20260717/v4096_broad256_v1/trajectories.json \
  --selective-catalog-store-directory /home/cereal/model-surgery-lab/artifacts/lfm2_cpcf_v5_frontier_20260719/8be9f891254519950e114bfd54196ce661bcf5cf/selective-catalogs \
  --selective-catalog-receipt-json /home/cereal/model-surgery-lab/artifacts/lfm2_selective_catalog_20260721/97ae4b9/a0002/selective-catalog-preparation.json \
  --expected-selective-catalog-receipt-sha256 4fbe0a9750446926c2a63f70d31078af2786971048d4c7f9d45882c4fdeb8dc9 \
  --expected-parent-identity-sha256 96a6779d927a8aca826b26147eca9f7bab2c1c207d762cd7e98a52c55e6be3eb \
  --expected-parent-unique-parameters 229693184 \
  --prompts-json "${RUNTIME}/research/lfm2_semantic_whole_response_prompts_v1.json" \
  --expected-prompts-file-sha256 ff6fa27e08a3cdbaa647bdeec1d0630d218f98ee13b195a836b85de7438974fa \
  --parent-envelope /home/cereal/model-surgery-lab/artifacts/lfm2_whole_response_parent_20260721/e739a88/a0002/parent-envelope.json \
  --expected-parent-envelope-sha256 e985b5614ba7af0f49ff0ee4eae722c6475fd653775bfc22c4c767c9fd8a2f97 \
  --expected-parent-envelope-file-sha256 c45f1cf903a25571add9e3a58a6108355d672914449e77866455e3472ca6c0e7 \
  --body-recipe nonlinear_qdeim_coupled_attention_fit_corrected \
  --expected-body-recipe-sha256 0c37f95ad809c83a1a34df61afcf080a64e228bbe19a7465d300d36a541334ce \
  --short-conv-construction full-source-structural-v1 \
  --blank-seed 0 \
  --model-device cuda:0 \
  --fold-compute-device cuda:0 \
  --analytic-solve-device cuda:0 \
  --global-residual-backend cuda \
  --global-residual-basis-rank 128 \
  --correction-scale 1.0 \
  --target-dtype-representability-level first_effectful \
  --maximum-prompt-tokens 1024 \
  --maximum-new-tokens 256 \
  >"${LOG}" 2>&1 < /dev/null &

echo $! > "${PIDFILE}"
echo "pid=$(cat "${PIDFILE}")"
echo "log=${LOG}"
echo "run=${RUN_DIR}"
