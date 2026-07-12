#!/usr/bin/env bash
set -euo pipefail

# Deliberately foreground-only: invoking this file is the explicit launch.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
PYTHON=${PYTHON:-/home/cereal/model-surgery-lab/venv/bin/python}
STAGING_ROOT=${STAGING_ROOT:-/home/cereal/model-surgery-lab/neural-forge-checkpoint-staging}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/mnt/ai_data/neural-forge/canonical_blank_targets}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-Qwen/Qwen3-4B-Base}
EXISTING_STAGED=${EXISTING_STAGED:-/home/cereal/model-surgery-lab/neural-forge-checkpoint-staging/qwen3_exact_1p410567168b_blank_seed0_fp16_20260712}

resolved_root=$(realpath "$ROOT")
if [[ "$resolved_root" != "/home/cereal/model-surgery-lab/neural-forge-src" ]]; then
  printf 'refusing non-canonical source root: %s\n' "$resolved_root" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  printf 'missing Python runtime: %s\n' "$PYTHON" >&2
  exit 3
fi
mkdir -p "$STAGING_ROOT" "$ARTIFACT_ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT"
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" "$ROOT/experiments/materialize_qwen3_stage1_blank_target.py" \
  --staging-root "$STAGING_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --existing-staged "$EXISTING_STAGED" \
  --local-files-only
