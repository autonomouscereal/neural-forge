#!/usr/bin/env bash
set -euo pipefail

# V3 continuation of the rejected static-prefix V2 run. L1 width64,
# fit/calibration prompts, 0/8/16/32 candidates, 32 updates, Pareto selector,
# exact lineage, and two-P40 topology remain unchanged. Fit-only student
# drivers refresh after 8/16/24 while manual Adam state persists. Evaluation
# uses a new V3 seal because V1/V2 inspected heldouts are development data.
# JSON is the only output; do not launch this wrapper automatically.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
PARENT_REVISION=${PARENT_REVISION:-906bfd4b4dc7f14ee4320094d8b41684abff8539}
PARENT_SNAPSHOT=$HF_HUB/models--Qwen--Qwen3-4B-Base/snapshots/$PARENT_REVISION
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_mlp_behavior_residual_20260711}
PARENT_GPU=${PARENT_GPU:-GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d}
CHILD_GPU=${CHILD_GPU:-GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_mlp_L1_behavior_residual_swiglu_w64_h4_s32_dagger_v3_fp16_p40_pair}
CONTAINER_NAME=${CONTAINER_NAME:-neural-forge-p35-mlp-residual-v3-w64-p40-pair}
REPORT=$OUTPUT_ROOT/$NAME.json
LOG=$OUTPUT_ROOT/$NAME.log
PID_FILE=$OUTPUT_ROOT/$NAME.pid
STARTTICKS_FILE=$OUTPUT_ROOT/$NAME.startticks
SEALED_HELDOUT=$ROOT/experiments/p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json
SEALED_HELDOUT_MANIFEST=$ROOT/experiments/p35_mlp_behavior_residual_v3_sealed_heldout_manifest_20260711.json

test -d "$ROOT"
test -d "$HF_HUB"
test "$PARENT_REVISION" = 906bfd4b4dc7f14ee4320094d8b41684abff8539
test -d "$PARENT_SNAPSHOT"
test -d "$CHECKPOINT"
test -f "$CHECKPOINT/manifest.json"
test "$(sha256sum "$CHECKPOINT/manifest.json" | awk '{print $1}')" = \
  28332a650a936ba3025834d0118c708aa0632484ed89a6da66c08e9daab547a9
test "$(sha256sum "$CHECKPOINT/model.safetensors" | awk '{print $1}')" = \
  43f5e77ca079c9a8de5692bd32b02ecb57e87c4bbc95260309ec33d40e81fa91
test "$(sha256sum "$CHECKPOINT/tokenizer_contract.json" | awk '{print $1}')" = \
  aabb2c58fefaba3386507654bcec6d514d84f8b70c5b4f160195abc53207afc1
test "$(sha256sum "$PARENT_SNAPSHOT/model-00001-of-00003.safetensors" | awk '{print $1}')" = \
  4c807e2503d68ae373d508689d00a41f4b33f33c2536da97ab81a20caddc1241
test "$(sha256sum "$PARENT_SNAPSHOT/model-00002-of-00003.safetensors" | awk '{print $1}')" = \
  f4707585548b2fc75a6b1d732e8465c62040a8699903c32850781beeb9b27826
test "$(sha256sum "$PARENT_SNAPSHOT/model-00003-of-00003.safetensors" | awk '{print $1}')" = \
  c7b1aa8fb672de2e00423c99876926022e50b18d4f0d140670788510a27f9965
test -f "$ROOT/experiments/run_qwen_p35_mlp_behavior_residual.py"
test -f "$ROOT/src/neural_forge/hf_behavior_residual_swiglu.py"
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_mlp_behavior_residual.py" | awk '{print $1}')" = \
  9bd6495a2320a68b5b9e3e9664f2d298756e92218012dc8b6009e9d3c8c0cdaa
test "$(sha256sum "$ROOT/src/neural_forge/hf_behavior_residual_swiglu.py" | awk '{print $1}')" = \
  c3c36c0e35bc70506adda4558e4546633749b298d47dc8747f1af826b3db51dd
test "$(sha256sum "$ROOT/experiments/run_qwen_p35_functional_residual_adapter.py" | awk '{print $1}')" = \
  d914e65c0bde4ccaf4bdeada17aee5208c08b92f1edc3cd08d1754bd3b1c0846
test "$(sha256sum "$ROOT/src/neural_forge/hf_functional_decode_trace.py" | awk '{print $1}')" = \
  e2f21b689b46d9e07ed1953f2f20e56a199bc3b0d3e0078039eb6f2209e55492
test "$(sha256sum "$ROOT/src/neural_forge/hf_functional_residual_adapter.py" | awk '{print $1}')" = \
  e157ca326c266cf2bbc145d76b6e2e126352595eb0222d8439b76389370b1a7f
grep -q 'neural-forge-qwen-p35-mlp-behavior-residual-swiglu-run-v3' \
  "$ROOT/experiments/run_qwen_p35_mlp_behavior_residual.py"
grep -q 'calibration_rollout_selection_key' \
  "$ROOT/src/neural_forge/hf_behavior_residual_swiglu.py"
test -f "$SEALED_HELDOUT"
test -f "$SEALED_HELDOUT_MANIFEST"
test "$(sha256sum "$SEALED_HELDOUT" | awk '{print $1}')" = \
  edcb437d3f331e65ab4f670a3cc20352709162dc31982adc82640e51ac509f3b
test "$(sha256sum "$SEALED_HELDOUT_MANIFEST" | awk '{print $1}')" = \
  d197bab46ed02c847ebcc5a0a93c95b1dce3b7298b8871835464b8509103754b
test -d "$TRANSFORMERS_SITE/transformers"
test -d "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info"
test -d "$TRANSFORMERS_SITE/huggingface_hub"
test -d "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$REPORT"
test ! -e "$LOG"
test ! -e "$PID_FILE"
test ! -e "$STARTTICKS_FILE"
test -z "$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$")"
test "$PARENT_GPU" != "$CHILD_GPU"

for GPU in "$PARENT_GPU" "$CHILD_GPU"; do
  GPU_ROW=$(nvidia-smi \
    --query-gpu=uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F ', ' -v uuid="$GPU" '$1 == uuid')
  test -n "$GPU_ROW"
  test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $2}')" = "Tesla P40"
  test "$(printf '%s\n' "$GPU_ROW" | awk -F ', ' '{print $3}')" -lt 4096
done

VISIBLE_GPU_ROWS=$(docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$PARENT_GPU,$CHILD_GPU" \
  --entrypoint nvidia-smi \
  "$IMAGE" \
  --query-gpu=uuid --format=csv,noheader,nounits)
test "$VISIBLE_GPU_ROWS" = "$(printf '%s\n%s' "$PARENT_GPU" "$CHILD_GPU")"

nohup docker run --rm \
  --name "$CONTAINER_NAME" \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$PARENT_GPU,$CHILD_GPU" \
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
  /workspace/experiments/run_qwen_p35_mlp_behavior_residual.py \
  --parent-model "/hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/$PARENT_REVISION" \
  --parent-revision "$PARENT_REVISION" \
  --codebook-checkpoint /codebook \
  --tokenizer "/hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/$PARENT_REVISION" \
  --fit-prompts-json /workspace/experiments/p35_p34_residual_attention_fit_prompts_20260711.json \
  --fit-prompt-manifest /workspace/experiments/p35_p34_residual_attention_fit_prompt_manifest_20260711.json \
  --calibration-selection-prompts-json /workspace/experiments/p35_p34_residual_attention_calibration_selection_prompts_20260711.json \
  --heldout-evaluation-prompts-json /workspace/experiments/p35_mlp_behavior_residual_v3_sealed_heldout_prompts_20260711.json \
  --sealed-heldout-manifest /workspace/experiments/p35_mlp_behavior_residual_v3_sealed_heldout_manifest_20260711.json \
  --audited-prompt-manifest /workspace/experiments/p35_p34_residual_attention_prompt_manifest_20260711.json \
  --transition-inventory /workspace/experiments/p35_p34_generation_transitions_20260711.json \
  --output "/output/$NAME.json" \
  --parent-device cuda:0 \
  --device cuda:1 \
  --dtype float16 \
  --residual-width 64 \
  --horizon 4 \
  --steps 32 \
  --calibration-steps 0 8 16 32 \
  --fit-refresh-steps 8 16 24 \
  --micro-batch-size 2 \
  --gate-up-learning-rate 0.001 \
  --down-learning-rate 0.002 \
  --max-prompt-tokens 256 \
  --max-vocabulary 1000000 \
  --max-logit-bytes 268435456 \
  --local-files-only \
  >"$LOG" 2>&1 </dev/null &

PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
sleep 1
kill -0 "$PID"
STARTTICKS=$(awk '{print $22}' "/proc/$PID/stat")
printf '%s\n' "$STARTTICKS" >"$STARTTICKS_FILE"
printf 'MLP behavior-residual v3 launched pid=%s startticks=%s parent_gpu=%s child_gpu=%s report=%s log=%s\n' \
  "$PID" "$STARTTICKS" "$PARENT_GPU" "$CHILD_GPU" "$REPORT" "$LOG"
