#!/usr/bin/env bash
set -euo pipefail

# Strict FP32 P40 companion for the residual-targeted value-basis probe.  The
# host venv's current Torch build lacks sm_61 kernels, so the model runs in the
# established P40-compatible image with the audited Transformers install
# mounted read-only.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
GPU=${GPU:-GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_L19G4_to_L21G6_residual_value_basis_four_control_h4_fp32_p40}
CONTAINER_NAME=${CONTAINER_NAME:-neural-forge-p35-residual-value-basis-h4-fp32-p40}
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
  /workspace/experiments/run_qwen_residual_attention_value_basis_probe.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --calibration-prompts-json /workspace/experiments/p35_residual_attention_v_calibration_prompts_20260711.json \
  --heldout-prompts-json /workspace/experiments/p35_residual_attention_v_heldout_prompts_20260711.json \
  --output "/output/$NAME.json" \
  --device cuda:0 \
  --dtype float32 \
  --horizon 4 \
  --basis-ridge 1e-4 \
  --o-ridge 1e-6 \
  --max-prompts-per-split 2 \
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
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'launched pid=%s startticks=%s gpu=%s output=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$OUTPUT" "$LOG" "$GPU_ROW"
