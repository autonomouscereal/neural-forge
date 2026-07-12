#!/usr/bin/env bash
set -euo pipefail

# Reconstruct only the already-evaluated unified-v8 seven-neuron transaction,
# require its immutable registry/accounting/gate evidence, then perform the
# missing save/reload and parent-free cached smoke. No discovery or behavioral
# evaluation is repeated.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-unified-20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
EVIDENCE=${EVIDENCE:-$ROOT/runs/unified-v8-checkpoint-failure-20260711}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711}
GPU=${GPU:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}
NAME=${NAME:-p35_mixed_drop7_v8_exact_replay_20260711}
OUTPUT=$OUTPUT_ROOT/$NAME
REPORT=$OUTPUT_ROOT/$NAME.replay.json
LOG=$OUTPUT_ROOT/$NAME.replay.log
PID_FILE=$OUTPUT_ROOT/$NAME.replay.pid
STARTTICKS_FILE=$OUTPUT_ROOT/$NAME.replay.startticks

test -d "$ROOT"
test -x "$PY"
test -d "$CHECKPOINT"
test -d "$EVIDENCE"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$OUTPUT"
test ! -e "$REPORT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
test ! -e "$STARTTICKS_FILE"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
test -n "$GPU_ROW"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')
test "$GPU_MEMORY_USED" -lt 1024

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT/src:$ROOT/experiments" \
TOKENIZERS_PARALLELISM=false \
nohup "$PY" \
  "$ROOT/experiments/replay_qwen_unified_v8_commit.py" \
  --input-codebook-checkpoint "$CHECKPOINT" \
  --evidence-directory "$EVIDENCE" \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --device cuda:0 \
  --reload-device cpu \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'launched pid=%s startticks=%s gpu=%s output=%s report=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$OUTPUT" "$REPORT" "$LOG" "$GPU_ROW"
