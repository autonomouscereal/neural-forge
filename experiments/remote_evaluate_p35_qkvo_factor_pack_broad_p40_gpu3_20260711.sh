#!/usr/bin/env bash
set -euo pipefail

# Prepared only. This wrapper has not been uploaded or launched. It mounts the
# exact p35 checkpoint, factor pack, manifest, broad suite, and raw parent
# report read-only. The fresh artifact directory receives JSON only.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-qkvo-broad-consumer-20260711}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
RUN_ROOT=${RUN_ROOT:-$OUTPUT_ROOT/broad_evaluations}
PACK_NAME=${PACK_NAME:-p35_L19G4_to_L21G6_qkvo_selected_s16_factor_pack_fp16_p40_gpu3_v3}
PACK_ROOT=${PACK_ROOT:-$OUTPUT_ROOT/$PACK_NAME}
FACTOR_PACK=${FACTOR_PACK:-$PACK_ROOT/qkvo_factors.safetensors}
FACTOR_MANIFEST=${FACTOR_MANIFEST:-$PACK_ROOT/qkvo_factors.manifest.json}
BROAD_ROOT=${BROAD_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}
BROAD_SUITE=${BROAD_SUITE:-$BROAD_ROOT/behavior_suite_exactgen_20260702.json}
RAW_QWEN_REPORT=${RAW_QWEN_REPORT:-$BROAD_ROOT/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json}
GPU=${GPU:-GPU-08b615f9-8dbe-3edd-d59f-f6ebe8212ca8}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_qkvo_selected_factor_pack_v3_broad_eval_fp16_p40_gpu3_v2}
CONTAINER_NAME=${CONTAINER_NAME:-neural-forge-p35-qkvo-v3-broad-eval-p40-gpu3-v2}
ARTIFACT_DIR=$RUN_ROOT/$NAME
LOG=$OUTPUT_ROOT/$NAME.log
PID_FILE=$OUTPUT_ROOT/$NAME.pid
STARTTICKS_FILE=$OUTPUT_ROOT/$NAME.startticks

test -d "$ROOT"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test -f "$FACTOR_PACK"
test -f "$FACTOR_MANIFEST"
test "$(sha256sum "$FACTOR_PACK" | awk '{print $1}')" = \
  5807fc789516ed5548a78fb4657f369a0f1c67c45a0b9e1626aa5813dd0832a8
test "$(sha256sum "$FACTOR_MANIFEST" | awk '{print $1}')" = \
  6c6e979712e67f862e4574121e500e9c823778f6774ce81647752158ccb2886d
test -f "$BROAD_SUITE"
test -f "$RAW_QWEN_REPORT"
test "$(sha256sum "$BROAD_SUITE" | awk '{print $1}')" = \
  fa6066980faf4677b194e00951076dc477cc4243d185449aa6590ca4c8768166
test "$(sha256sum "$RAW_QWEN_REPORT" | awk '{print $1}')" = \
  3333bc54e9170dc8a65e8a53179c9a20a2ab7d1931e2da5795f6bc5ecf031395
test -f "$ROOT/experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py"
test -f "$ROOT/src/neural_forge/hf_qkvo_factor_pack.py"
test -d "$TRANSFORMERS_SITE/transformers"
mkdir -p "$OUTPUT_ROOT"
mkdir -p "$RUN_ROOT"
test ! -e "$ARTIFACT_DIR"
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
  -v "$PACK_ROOT:/factor-input:ro" \
  -v "$BROAD_SUITE:/broad-suite.json:ro" \
  -v "$RAW_QWEN_REPORT:/raw-qwen-parent.json:ro" \
  -v "$RUN_ROOT:/output" \
  -v "$TRANSFORMERS_SITE/transformers:/vendor/transformers:ro" \
  -v "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info:/vendor/transformers-5.12.1.dist-info:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub:/vendor/huggingface_hub:ro" \
  -v "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info:/vendor/huggingface_hub-1.21.0.dist-info:ro" \
  "$IMAGE" \
  /workspace/experiments/run_qwen_p35_qkvo_factor_pack_broad_eval.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --factor-pack /factor-input/qkvo_factors.safetensors \
  --factor-manifest /factor-input/qkvo_factors.manifest.json \
  --broad-suite /broad-suite.json \
  --raw-qwen-parent-report /raw-qwen-parent.json \
  --fit-prompts-json /workspace/experiments/p35_p34_residual_attention_fit_prompts_20260711.json \
  --fit-prompt-manifest /workspace/experiments/p35_p34_residual_attention_fit_prompt_manifest_20260711.json \
  --calibration-selection-prompts-json /workspace/experiments/p35_p34_residual_attention_calibration_selection_prompts_20260711.json \
  --heldout-evaluation-prompts-json /workspace/experiments/p35_p34_residual_attention_heldout_evaluation_prompts_20260711.json \
  --audited-prompt-manifest /workspace/experiments/p35_p34_residual_attention_prompt_manifest_20260711.json \
  --transition-inventory /workspace/experiments/p35_p34_generation_transitions_20260711.json \
  --output-directory "/output/$NAME" \
  --device cuda:0 \
  --dtype float16 \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'prepared broad-eval launched pid=%s startticks=%s gpu=%s artifacts=%s log=%s prelaunch_gpu=%s\n' \
  "$PID" "$STARTTICKS" "$GPU" "$ARTIFACT_DIR" "$LOG" "$GPU_ROW"
