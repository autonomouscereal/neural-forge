#!/usr/bin/env bash
set -euo pipefail

# Foreground-only canonical stage runner.  The default is authentication and
# preflight; model loading/training requires the literal --execute flag.
MODE=preflight
LEXICAL_CONTINUATION=${LEXICAL_CONTINUATION:-0}
if [[ ${1:-} == "--lexical-continuation" ]]; then
  LEXICAL_CONTINUATION=1
  shift
fi
if [[ ${1:-} == "--execute" ]]; then
  MODE=execute
  shift
elif [[ ${1:-} == "--preflight" ]]; then
  shift
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--lexical-continuation] [--preflight|--execute]" >&2
  exit 2
fi
if [[ "$LEXICAL_CONTINUATION" != 0 && "$LEXICAL_CONTINUATION" != 1 ]]; then
  echo "LEXICAL_CONTINUATION must be 0 or 1" >&2
  exit 2
fi

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
PYTHON=${PYTHON:-/home/cereal/model-surgery-lab/venv/bin/python}
PARENT=${PARENT:-/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
TARGET=${TARGET:-/mnt/ai_data/neural-forge/canonical_blank_targets/qwen3_stage1_blank_1410567168_fp16_seed0}
CONTINUATION_SOURCE_REPORT=${CONTINUATION_SOURCE_REPORT:-}
LEXICAL_CONTINUATION_SOURCE_REPORT=${LEXICAL_CONTINUATION_SOURCE_REPORT:-}
ORIGINAL_SPLIT_DIR=${ORIGINAL_SPLIT_DIR:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_fp32factors_eager_20260712}
SOURCE_SPLIT_DIR=${SOURCE_SPLIT_DIR:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_step16_residual_attention_continuation_20260712/splits}
COMPILER_CACHE_ROOT=${COMPILER_CACHE_ROOT:-/mnt/ai_data/neural-forge/compiled_child_cache}
if [[ "$LEXICAL_CONTINUATION" == 1 ]]; then
  PREREG_DIR=${PREREG_DIR:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_step24_lexical_attention_continuation_20260712}
  OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_step24_lexical_attention_continuation_20260712}
else
  PREREG_DIR=${PREREG_DIR:-/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_20260712}
  OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/runs/qwen_exact_target_native_stage1_first_run_20260712}
fi

ROOT=$(realpath -e "$ROOT")
test "$ROOT" = "/home/cereal/model-surgery-lab/neural-forge-src"
test -d "$ROOT/.git"
test -f "$ROOT/experiments/run_qwen_exact_target_native_stage.py"
test -f "$ROOT/experiments/prepare_qwen_exact_target_native_stage.py"
test -d "$PARENT"
test -d "$TARGET"

export PYTHONPATH="$ROOT/src:$ROOT/experiments${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$PREREG_DIR/build_report.json" ]]; then
  test ! -e "$PREREG_DIR"
  PREPARE_ARGS=(
    --output-dir "$PREREG_DIR"
    --parent "$PARENT"
    --target "$TARGET"
  )
  if [[ "$LEXICAL_CONTINUATION" == 1 ]]; then
    if [[ -z "$LEXICAL_CONTINUATION_SOURCE_REPORT" ]]; then
      echo "LEXICAL_CONTINUATION_SOURCE_REPORT is required in lexical mode" >&2
      exit 2
    fi
    PREPARE_ARGS+=(
      --lexical-continuation-source-report "$LEXICAL_CONTINUATION_SOURCE_REPORT"
      --source-split-dir "$SOURCE_SPLIT_DIR"
      --original-split-dir "$ORIGINAL_SPLIT_DIR"
      --compiler-cache-root "$COMPILER_CACHE_ROOT"
    )
  elif [[ -n "$CONTINUATION_SOURCE_REPORT" ]]; then
    PREPARE_ARGS+=(
      --continuation-source-report "$CONTINUATION_SOURCE_REPORT"
      --original-split-dir "$ORIGINAL_SPLIT_DIR"
      --compiler-cache-root "$COMPILER_CACHE_ROOT"
    )
  fi
  "$PYTHON" "$ROOT/experiments/prepare_qwen_exact_target_native_stage.py" "${PREPARE_ARGS[@]}"
fi

PREREGISTRATION=$PREREG_DIR/preregistration.json
PREREGISTRATION_SHA256=$("$PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$PREREGISTRATION")

if [[ "$MODE" == "preflight" ]]; then
  echo "Mode: preflight only (lexical continuation=$LEXICAL_CONTINUATION); no model weights or GPUs will be loaded."
  exec "$PYTHON" "$ROOT/experiments/run_qwen_exact_target_native_stage.py" \
    --preregistration "$PREREGISTRATION" \
    --preregistration-sha256 "$PREREGISTRATION_SHA256" \
    --preflight
fi

# Physical GPU 4 is an RTX 3090 and physical GPU 6 is an RTX 8000.  CUDA
# remaps them to logical cuda:0 (parent) and cuda:1 (child) in this process.
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,6}
test ! -e "$OUTPUT_ROOT"
echo "Mode: foreground execution; physical GPUs ${CUDA_VISIBLE_DEVICES}, parent logical cuda:0, child logical cuda:1."
exec "$PYTHON" "$ROOT/experiments/run_qwen_exact_target_native_stage.py" \
  --preregistration "$PREREGISTRATION" \
  --preregistration-sha256 "$PREREGISTRATION_SHA256" \
  --output-root "$OUTPUT_ROOT" \
  --dtype float16 \
  --parent-device-map cuda:0 \
  --target-device-map cuda:1
