#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "${1:-}" != "--preflight" && "${1:-}" != "--execute" ) ]]; then
  echo "usage: $0 --preflight|--execute" >&2
  exit 2
fi
MODE=$1

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
PYTHON=${PYTHON:-/home/cereal/model-surgery-lab/venv/bin/python}
AUDIT_RUN_REPORT=${AUDIT_RUN_REPORT:-/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712/run_report.json}
AUDIT_PREREGISTRATION=${AUDIT_PREREGISTRATION:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_step24_output_jacobian_audit_20260712/preregistration.json}
PREREG_DIR=${PREREG_DIR:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_schemafix_v2_20260712}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_guided_qkvo_continuation_schemafix_v2_20260712}

ROOT=$(realpath -e "$ROOT")
test "$ROOT" = "/home/cereal/model-surgery-lab/neural-forge-src"
test -d "$ROOT/.git"
test -x "$PYTHON"
test -f "$ROOT/experiments/prepare_qwen_exact_target_native_guided_continuation.py"
test -f "$ROOT/experiments/run_qwen_exact_target_native_guided_continuation.py"
test -f "$AUDIT_RUN_REPORT"
test -f "$AUDIT_PREREGISTRATION"

export PYTHONPATH="$ROOT/src:$ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$PREREG_DIR/build_report.json" ]]; then
  test ! -e "$PREREG_DIR"
  "$PYTHON" \
    "$ROOT/experiments/prepare_qwen_exact_target_native_guided_continuation.py" \
    --output-dir "$PREREG_DIR" \
    --audit-run-report "$AUDIT_RUN_REPORT" \
    --audit-preregistration "$AUDIT_PREREGISTRATION"
fi

PREREGISTRATION=$PREREG_DIR/preregistration.json
test -f "$PREREGISTRATION"
PREREGISTRATION_SHA256=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$PREREGISTRATION")

RUNNER=(
  "$PYTHON"
  "$ROOT/experiments/run_qwen_exact_target_native_guided_continuation.py"
  --preregistration "$PREREGISTRATION"
  --preregistration-sha256 "$PREREGISTRATION_SHA256"
)

if [[ "$MODE" == "--preflight" ]]; then
  echo "Mode: model-free guided preflight; no model weights or GPUs will be loaded."
  exec "${RUNNER[@]}" --preflight
fi

# Re-authenticate the complete model-free contract immediately before the one
# explicit execution boundary.  Physical GPU 4 is an RTX 3090 and physical GPU
# 6 is an RTX 8000; CUDA exposes them as parent cuda:0 and child cuda:1.
"${RUNNER[@]}" --preflight >/dev/null
test ! -e "$OUTPUT_ROOT"
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,6}
echo "Mode: guided execution; physical GPUs ${CUDA_VISIBLE_DEVICES}, parent cuda:0, child cuda:1."
exec "${RUNNER[@]}" \
  --execute \
  --output-root "$OUTPUT_ROOT" \
  --dtype float16 \
  --parent-device-map cuda:0 \
  --target-device-map cuda:1
