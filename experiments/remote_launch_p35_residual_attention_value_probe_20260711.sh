#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-residual-attn-20260711}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
MODEL=${MODEL:-/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
GPU=${GPU:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}
NAME=${NAME:-p35_L19G4_to_L21G6_v_o_four_control_h8}
DTYPE=${DTYPE:-float16}
LOGIT_RTOL=${LOGIT_RTOL:-0.01}
LOGIT_ATOL=${LOGIT_ATOL:-0.04}
CONTEXT_RTOL=${CONTEXT_RTOL:-0.008}
CONTEXT_ATOL=${CONTEXT_ATOL:-0.008}
OUTPUT=$OUTPUT_ROOT/$NAME.json
LOG=$OUTPUT_ROOT/$NAME.log
PID_FILE=$OUTPUT_ROOT/$NAME.pid

test -d "$ROOT"
test -x "$PY"
test -d "$MODEL"
test -d "$CHECKPOINT"
test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
mkdir -p "$OUTPUT_ROOT"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
test -n "$GPU_ROW"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')
test "$GPU_MEMORY_USED" -lt 4096

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$ROOT/src:$ROOT/experiments" \
TOKENIZERS_PARALLELISM=false \
nohup "$PY" "$ROOT/experiments/run_qwen_residual_attention_value_probe.py" \
  --codebook-checkpoint "$CHECKPOINT" \
  --tokenizer "$MODEL" \
  --calibration-prompts-json "$ROOT/experiments/p35_residual_attention_v_calibration_prompts_20260711.json" \
  --heldout-prompts-json "$ROOT/experiments/p35_residual_attention_v_heldout_prompts_20260711.json" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --dtype "$DTYPE" \
  --horizon 8 \
  --ridge 1e-6 \
  --max-prompts-per-split 2 \
  --max-prompt-tokens 256 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --max-capture-bytes 536870912 \
  --max-projection-bytes 134217728 \
  --logit-rtol "$LOGIT_RTOL" \
  --logit-atol "$LOGIT_ATOL" \
  --context-rtol "$CONTEXT_RTOL" \
  --context-atol "$CONTEXT_ATOL" \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
printf 'launched pid=%s gpu=%s output=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$GPU" "$OUTPUT" "$LOG" "$GPU_ROW"
