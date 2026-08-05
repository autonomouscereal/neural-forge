#!/usr/bin/env bash
# Layer-0 representability study launcher (R13 follow-up to NOTES-W13A).
# Closed-form only: zero parent forwards, zero child forwards, zero gradients.
set -euo pipefail

MODE="${1:?usage: nf-layer0-study.sh inventory|study [extra args...]}"
shift || true

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-layer0-3953ede4
OUT_ROOT=/home/cereal/model-surgery-lab/artifacts/neural_forge_layer0_study_v1
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12
PACK=/home/cereal/model-surgery-lab/artifacts/lfm2_tokenizer_bridge_evidence_20260717/v4096_modelcap_v2_broad256_cpcf_decision_v5
TRACE="$PACK/parent_trace_cache/76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95"
BALANCED_BASIS=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-circuit-balanced-residual-basis-v1
BASIS_KEY=ce0593c4762a8ed586d2c16741f1ed02df694c432a8233416585f4d16ebb49f4
BASIS_HMAC_KEY=/home/cereal/model-surgery-lab/keys/neural-forge-balanced-basis-phase-v1.key
BASIS_HMAC_KEY_ID=lfm2-circuit-balanced-residual-basis-v1
PARENT=/home/cereal/model-surgery-lab/models/LFM2.5-230M
CHILD="$RUN/published-child/child"
VENV_PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

export PYTHONPATH="$RUNTIME/src:$RUNTIME"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
# Every arm is dense fp64 CPU linear algebra.  No CUDA context is created, so
# no GPU is taken away from the on-policy round or anything else on the box.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

mkdir -p "$OUT_ROOT"
DEST="$OUT_ROOT/$MODE"
mkdir -p "$DEST"

nohup "$VENV_PYTHON" -u \
  "$RUNTIME/experiments/run_lfm2_layer0_representability_study.py" \
  --mode "$MODE" \
  --trace-cache "$TRACE" \
  --basis-phase-cache-root "$BALANCED_BASIS" \
  --basis-phase-key-sha256 "$BASIS_KEY" \
  --basis-phase-hmac-key-file "$BASIS_HMAC_KEY" \
  --basis-phase-hmac-key-id "$BASIS_HMAC_KEY_ID" \
  --parent-model "$PARENT" \
  --published-child-directory "$CHILD" \
  --output-directory "$DEST" \
  "$@" \
  > "$DEST/run.log" 2>&1 &

pid=$!
echo "$pid" > "$DEST/run.pid"
echo "DEST=$DEST"
echo "PID=$pid"
