#!/usr/bin/env bash
set -euo pipefail

# Prepared only. This wrapper has not been uploaded or launched. It tests the
# exact eight p35->p34 redirects with three fixed uneven-rank schedules, writes
# only the strict winning factor bundle, reload-verifies it, and restores p35.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-p35-multisite-ragged-qkvo-20260711}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
ANCHOR_NAME=${ANCHOR_NAME:-p35_L19G4_to_L21G6_qkvo_selected_s16_factor_pack_fp16_p40_gpu3}
ANCHOR_ROOT=${ANCHOR_ROOT:-$OUTPUT_ROOT/$ANCHOR_NAME}
ANCHOR_PACK=${ANCHOR_PACK:-$ANCHOR_ROOT/qkvo_factors.safetensors}
ANCHOR_MANIFEST=${ANCHOR_MANIFEST:-$ANCHOR_ROOT/qkvo_factors.manifest.json}
GPU=${GPU:-GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_multisite_ragged_qkvo_r36_r40_r50_h4_s32_fp16_p40_gpu3}
CONTAINER_NAME=${CONTAINER_NAME:-neural-forge-p35-multisite-ragged-qkvo-p40-gpu3}
BUNDLE=$OUTPUT_ROOT/${NAME}_factor_bundle
REPORT=$OUTPUT_ROOT/$NAME.json
LOG=$OUTPUT_ROOT/$NAME.log
PID_FILE=$OUTPUT_ROOT/$NAME.pid
STARTTICKS_FILE=$OUTPUT_ROOT/$NAME.startticks

test -d "$ROOT"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test -f "$ANCHOR_PACK"
test -f "$ANCHOR_MANIFEST"
test -f "$ROOT/experiments/run_qwen_p35_multisite_ragged_qkvo_residual.py"
test -f "$ROOT/experiments/reproduce_pack_qwen_p35_qkvo_selected.py"
test -f "$ROOT/src/neural_forge/hf_ragged_assignment_local_qkvo_residual_adapter.py"
test -f "$ROOT/src/neural_forge/hf_ragged_qkvo_factor_bundle.py"
test -f "$ROOT/src/neural_forge/hf_qkvo_factor_pack.py"
test -d "$TRANSFORMERS_SITE/transformers"
test -d "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info"
test -d "$TRANSFORMERS_SITE/huggingface_hub"
test -d "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$BUNDLE"
test ! -e "$REPORT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
test ! -e "$STARTTICKS_FILE"
test -z "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"

GPU_ROW=$(nvidia-smi \
  --query-gpu=uuid,name,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
test -n "$GPU_ROW"
test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')" = "Tesla P40"
GPU_MEMORY_USED=$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $3}')
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
  -v "$ANCHOR_PACK:/anchor.safetensors:ro" \
  -v "$ANCHOR_MANIFEST:/anchor.manifest.json:ro" \
  -v "$OUTPUT_ROOT:/output" \
  -v "$TRANSFORMERS_SITE/transformers:/vendor/transformers:ro" \
  -v "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info:/vendor/transformers-5.12.1.dist-info:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub:/vendor/huggingface_hub:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info:/vendor/huggingface_hub-1.21.0.dist-info:ro" \
  "$IMAGE" \
  /workspace/experiments/run_qwen_p35_multisite_ragged_qkvo_residual.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --anchor-factor-pack /anchor.safetensors \
  --anchor-factor-manifest /anchor.manifest.json \
  --fit-prompts-json /workspace/experiments/p35_p34_residual_attention_fit_prompts_20260711.json \
  --fit-prompt-manifest /workspace/experiments/p35_p34_residual_attention_fit_prompt_manifest_20260711.json \
  --calibration-selection-prompts-json /workspace/experiments/p35_p34_residual_attention_calibration_selection_prompts_20260711.json \
  --heldout-evaluation-prompts-json /workspace/experiments/p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json \
  --audited-prompt-manifest /workspace/experiments/p35_p34_residual_attention_prompt_manifest_20260711.json \
  --transition-inventory /workspace/experiments/p35_p34_generation_transitions_20260711.json \
  --bundle-output "/output/${NAME}_factor_bundle" \
  --output "/output/$NAME.json" \
  --device cuda:0 \
  --dtype float16 \
  --max-prompt-tokens 256 \
  --max-vocabulary 200000 \
  --max-logit-bytes 268435456 \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'prepared multisite run launched pid=%s startticks=%s gpu=%s bundle=%s report=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$BUNDLE" "$REPORT" "$LOG" "$GPU_ROW"
