#!/usr/bin/env bash
set -euo pipefail

# Isolated from every unified-surgery checkout.  The main model-surgery venv's
# Torch 2.11 wheel no longer contains sm_61 kernels, so this smoke uses the
# existing P40-compatible Torch 2.4.1 container image read-only.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-cached-attention-smoke-20260711}
MODEL=${MODEL:-/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/cached_attention_smoke_20260711}
GPU=${GPU:-GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
DTYPE=${DTYPE:-float32}
OUTPUT_NAME=${OUTPUT_NAME:-incumbent_p35_layer35_g0_greedy_h2_p40_tf512_fp32_container.json}

test -d "$ROOT"
test -d "$MODEL"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test -d "$TRANSFORMERS_SITE/transformers"
test -d "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info"
test -d "$TRANSFORMERS_SITE/huggingface_hub"
test -d "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$OUTPUT_ROOT/$OUTPUT_NAME"

docker run --rm \
  --gpus "device=$GPU" \
  --entrypoint python3 \
  -e PYTHONPATH=/vendor:/workspace/src \
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
  /workspace/experiments/run_qwen_cached_attention_smoke.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --prompt "The capital of France is" \
  --layer-index 35 \
  --group-indices 0 \
  --horizon 2 \
  --lane greedy \
  --device cuda:0 \
  --dtype "$DTYPE" \
  --max-horizon 2 \
  --max-prompt-tokens 32 \
  --max-groups 1 \
  --max-capture-bytes 33554432 \
  --local-files-only \
  --output "/output/$OUTPUT_NAME"
