#!/usr/bin/env bash
set -euo pipefail

# Evidence-grade FP32 causal intervention on the isolated residual-attention
# checkout.  The dense parent and codebook candidate are loaded sequentially.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-residual-attn-20260711}
MODEL=${MODEL:-/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa34_gpu6_r1_20260711/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
GPU=${GPU:-GPU-c95fb5b3-f97c-bfb2-9760-79fe107ffe6d}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
LAYER=${LAYER:-19}
# GROUPS is a Bash special array containing the caller's Unix group IDs.
GROUP_INDICES=${GROUP_INDICES:-4}
HORIZON=${HORIZON:-4}
FEEDBACK=${FEEDBACK:-forced-cache-recurrent}
OUTPUT_NAME=${OUTPUT_NAME:-p34_layer19_group4_h4_forced_cache_oracle_fp32.json}

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
  /workspace/experiments/run_qwen_residual_attention_oracle.py \
  --parent-model /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --candidate-codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --prompt "Complete the Python function: def add(a, b):" \
  --layer-index "$LAYER" \
  --group-indices "$GROUP_INDICES" \
  --horizon "$HORIZON" \
  --lane greedy \
  --feedback "$FEEDBACK" \
  --device cuda:0 \
  --dtype float32 \
  --max-horizon 8 \
  --max-prompt-tokens 64 \
  --max-trace-bytes 536870912 \
  --max-replay-bytes 268435456 \
  --local-files-only \
  --output "/output/$OUTPUT_NAME"
