#!/usr/bin/env bash
set -euo pipefail

# Bounded strict-FP32 real-Q/K causal search for the exact p35 redirect.
# The P40-compatible container is selected by immutable GPU UUID; the runner
# is diagnostic-only, rolls back every edit, and has no checkpoint save path.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
GPU=${GPU:-GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_L19G4_to_L21G6_qk_anchor_search_a9_h4_p1_fp32_p40_gpu3}
CONTAINER_NAME=${CONTAINER_NAME:-nf-p35-qk-a9-h4-p1-fp32-20260711}
OUTPUT=$OUTPUT_ROOT/$NAME.json
LOG=$OUTPUT_ROOT/$NAME.log
PID_FILE=$OUTPUT_ROOT/$NAME.pid
STARTTICKS_FILE=$OUTPUT_ROOT/$NAME.startticks

test -d "$ROOT"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test -d "$TRANSFORMERS_SITE/transformers"
test -d "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info"
test -d "$TRANSFORMERS_SITE/huggingface_hub"
test -d "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info"
test -f "$ROOT/experiments/run_qwen_residual_attention_qk_anchor_search.py"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$OUTPUT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
test ! -e "$STARTTICKS_FILE"
test -z "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
test -n "$GPU_ROW"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')
test "$GPU_MEMORY_USED" -lt 4096

nohup docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus "device=$GPU" \
  --entrypoint python3 \
  -e PYTHONPATH=/vendor:/workspace/src:/workspace/experiments \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$ROOT:/workspace:ro" \
  -v "$HF_HUB:/hf-hub:ro" \
  -v "$CHECKPOINT:/codebook:ro" \
  -v "$OUTPUT_ROOT:/output" \
  -v "$TRANSFORMERS_SITE/transformers:/vendor/transformers:ro" \
  -v "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info:/vendor/transformers-5.12.1.dist-info:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub:/vendor/huggingface_hub:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info:/vendor/huggingface_hub-1.21.0.dist-info:ro" \
  "$IMAGE" \
  /workspace/experiments/run_qwen_residual_attention_qk_anchor_search.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --qk-anchor 21:6 \
  --qk-anchor 18:4 \
  --qk-anchor 20:4 \
  --qk-anchor 22:4 \
  --qk-anchor 19:3 \
  --qk-anchor 19:5 \
  --qk-anchor 18:6 \
  --qk-anchor 20:6 \
  --qk-anchor 22:6 \
  --calibration-prompt "Work carefully through this mixed reasoning task. A warehouse starts with 240 components, ships 37, receives 18, and then reserves one quarter of what remains. State the remaining unreserved count and briefly verify each arithmetic step." \
  --heldout-prompt "A driver reaches a flooded road at night and cannot judge the water depth. Give the safest immediate action and explain why attempting to cross is dangerous." \
  --output "/output/$NAME.json" \
  --device cuda:0 \
  --dtype float32 \
  --horizon 4 \
  --ridge 1e-6 \
  --skip-o-ridge \
  --max-anchors 9 \
  --max-prompts-per-split 1 \
  --max-prompt-tokens 256 \
  --max-affected-sites 16 \
  --max-affected-layers 8 \
  --max-capture-bytes 536870912 \
  --max-projection-bytes 134217728 \
  --logit-rtol 0.005 \
  --logit-atol 0.005 \
  --context-rtol 0.008 \
  --context-atol 0.008 \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'launched pid=%s startticks=%s gpu=%s output=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$OUTPUT" "$LOG" "$GPU_ROW"
