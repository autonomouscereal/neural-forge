#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
GPU_UUID=${GPU_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}
MANIFEST=${MANIFEST:-$ROOT/experiments/lfm2_a9_a11_high_cap_reuse_manifest_20260730.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730}
RUN_TAG=${RUN_TAG:-a9-a11-highcap-3090-r1}
LAUNCH_ROOT=${LAUNCH_ROOT:-$OUTPUT_ROOT/_launch/$RUN_TAG}

if [[ ! -x "$PY" ]]; then
  echo "Python runtime is not executable: $PY" >&2
  exit 2
fi
for path in \
  "$ROOT/experiments/launch_detached.py" \
  "$ROOT/experiments/rerun_lfm2_sparse_semantic_capped_prompts.py" \
  "$MANIFEST"
do
  if [[ ! -f "$path" ]]; then
    echo "Required launch input is missing: $path" >&2
    exit 2
  fi
done

mkdir -p "$LAUNCH_ROOT"
if [[ -e "$LAUNCH_ROOT/job.pid" || -e "$LAUNCH_ROOT/job.identity.json" ]]; then
  echo "Launch identity already exists: $LAUNCH_ROOT" >&2
  echo "Completed prompt/cap checkpoints remain reusable; choose a new RUN_TAG to resume." >&2
  exit 3
fi

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$LAUNCH_ROOT/job.log" \
  --pid-file "$LAUNCH_ROOT/job.pid" \
  --identity-file "$LAUNCH_ROOT/job.identity.json" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU_UUID" \
  --env "PYTHONPATH=$ROOT/src:$ROOT" \
  --env "PYTHONUNBUFFERED=1" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$PY" "$ROOT/experiments/rerun_lfm2_sparse_semantic_capped_prompts.py" \
  --manifest "$MANIFEST" \
  --lane a9-k32-residual-omp \
  --lane a11-k32-split-cout \
  --device cuda:0

echo "launch_root=$LAUNCH_ROOT"
echo "gpu_uuid=$GPU_UUID"
echo "pid=$(cat "$LAUNCH_ROOT/job.pid")"
echo "log=$LAUNCH_ROOT/job.log"
