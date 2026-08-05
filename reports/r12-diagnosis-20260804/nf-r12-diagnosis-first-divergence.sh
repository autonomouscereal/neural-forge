#!/usr/bin/env bash
# W13A / R13 plan item 5 (cache-only half).
#
# First-divergence diagnosis of the SEALED r12 exact-4x child using only sealed
# cached evidence: zero refolding, zero training, zero parent forward calls.
# Every path below is read-only input; the only writes go to $OUT.
#
# Difference from Codex's a0023-...-layer0-v3 probe: focus layers are all 14
# instead of just layer 0, so per-layer branch (operator.output / mlp.down)
# diagnostics exist for the whole stack.  State-boundary residual metrics are
# observed for every layer regardless of the focus set.
set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-exact4x-9bc0d735
RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_exact4x_20260730/catalog-integrated-carrier/a0023-exact4x-h816-mgv3-r12
OUT=/home/cereal/model-surgery-lab/artifacts/neural_forge_r12_diagnosis_v1/first-divergence
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
# Pin the Quadro RTX 8000 by UUID so cuda:0 is that card and nothing else is
# visible.  r12's fold owns the two RTX 3090s (nvidia-smi 0 and 4) and the P40s
# are sm_61, which this torch build cannot run.  Codex's sealed v3 probe used
# the same card, so the numeric contract matches.
export CUDA_VISIBLE_DEVICES=GPU-f5a09535-fc85-a027-50f7-734b344747dd

mkdir -p "$OUT"

"$VENV_PYTHON" - <<'PREFLIGHT' || exit 42
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("Preflight: CUDA is unavailable.")
if torch.cuda.device_count() != 1:
    sys.exit(f"Preflight: {torch.cuda.device_count()} visible devices, expected 1.")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"cuda:0 = {name} cc={capability[0]}.{capability[1]}")
if capability < (7, 5):
    sys.exit(f"Preflight: cuda:0 is {name!r} with unsupported cc {capability}.")
PREFLIGHT

nohup "$VENV_PYTHON" -u \
  "$RUNTIME/experiments/audit_lfm2_exact4x_first_divergence.py" \
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
  --source-width 1024 \
  --target-hidden-size 816 \
  --focus-layer-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --maximum-selected-parent-bytes 536870912 \
  --maximum-donor-rows-per-score-chunk 4096 \
  > "$OUT/job.log" 2>&1 &

pid=$!
echo "$pid" > "$OUT/job.pid"
echo "OUT=$OUT"
echo "PID=$pid"
