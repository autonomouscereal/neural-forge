#!/usr/bin/env bash
# W13A / R13 plan item 5 (cache-only half), second lane.
#
# Authenticated exact-4x causal folding diagnostic, downstream of the sealed
# first-divergence ledger produced by nf-r12-diagnosis-first-divergence.sh.
# Sealed cached evidence only: zero refolding, zero training, zero parent
# forwards.  Usage: nf-r12-diagnosis-causal.sh <layer-index>
set -euo pipefail

LAYER="${1:?Pass the operator layer index to diagnose}"
LABEL=$(printf 'layer-%02d' "$LAYER")

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-exact4x-9bc0d735
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12
ROOT=/home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1
FIRST_DIVERGENCE="$ROOT/first-divergence"
FIRST_DIVERGENCE_LEDGER_SHA256=a4fa6e6886a4aaf95d34b47c272597a1eab72df5da964cb0e7cbf3efe0d51bd8
OUT="$ROOT/causal-diagnostic/$LABEL"
PACK=/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5
TRACE="$PACK/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
BALANCED_BASIS=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1
BALANCED_BASIS_HMAC_KEY=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
BALANCED_BASIS_HMAC_KEY_ID=lfm2-circuit-balanced-residual-basis-v1
BASIS_PHASE_KEY_SHA256=ce0593c4762a8ed586d2c16741f1ed02df694c432a8233416585f4d16ebb49f4
CHILD="$RUN/published-child/child"
CHILD_PACKAGE_SHA256=19f5e7d26619d129520154c78fbec4d1f32b7c5a3a08254ea2a64d5788aa71db
VENV_PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

export PYTHONPATH="$RUNTIME/src:$RUNTIME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
# Same Quadro RTX 8000 the first-divergence lane used; the causal runner
# fails closed if its runtime device differs from the sealed v3 request.
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd

mkdir -p "$OUT"

nohup "$VENV_PYTHON" -u \
  "$RUNTIME/experiments/audit_lfm2_exact4x_causal_diagnostic.py" \
  --first-divergence-directory "$FIRST_DIVERGENCE" \
  --first-divergence-ledger-sha256 "$FIRST_DIVERGENCE_LEDGER_SHA256" \
  --child-directory "$CHILD" \
  --child-package-sha256 "$CHILD_PACKAGE_SHA256" \
  --trace-cache "$TRACE" \
  --basis-phase-cache-root "$BALANCED_BASIS" \
  --basis-phase-key-sha256 "$BASIS_PHASE_KEY_SHA256" \
  --basis-phase-hmac-key-file "$BALANCED_BASIS_HMAC_KEY" \
  --basis-phase-hmac-key-id "$BALANCED_BASIS_HMAC_KEY_ID" \
  --output-directory "$OUT" \
  --runtime-device cuda:0 \
  --runtime-dtype float32 \
  --operator-layer-index "$LAYER" \
  --operator-kind auto \
  --bootstrap-resample-count 4096 \
  --bootstrap-confidence-level 0.95 \
  --maximum-selected-parent-bytes 536870912 \
  --maximum-donor-rows-per-score-chunk 4096 \
  > "$OUT/job.log" 2>&1 &

pid=$!
echo "$pid" > "$OUT/job.pid"
echo "OUT=$OUT"
echo "PID=$pid"
