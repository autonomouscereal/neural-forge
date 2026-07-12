#!/usr/bin/env bash
set -euo pipefail

# Strict FP32 companion to the faster RTX/FP16 h8 probe.  The global venv no
# longer includes sm_61 kernels, so this uses the proven P40-compatible image.
ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-residual-attn-20260711}
HF_HUB=${HF_HUB:-/home/cereal/.cache/huggingface/hub}
CHECKPOINT=${CHECKPOINT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_residual_attention_20260711}
GPU=${GPU:-GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4}
IMAGE=${IMAGE:-comfyui:codex-extended-ltx-ace-20260522}
TRANSFORMERS_SITE=${TRANSFORMERS_SITE:-/home/cereal/model-surgery-lab/venv/lib/python3.12/site-packages}
NAME=${NAME:-p35_L19G4_to_L21G6_v_o_four_control_h4_fp32_p40}

test -d "$ROOT"
test -d "$HF_HUB"
test -d "$CHECKPOINT"
test -d "$TRANSFORMERS_SITE/transformers"
test -d "$TRANSFORMERS_SITE/transformers-5.12.1.dist-info"
test -d "$TRANSFORMERS_SITE/huggingface_hub"
test -d "$TRANSFORMERS_SITE/huggingface_hub-1.21.0.dist-info"
mkdir -p "$OUTPUT_ROOT"
test ! -e "$OUTPUT_ROOT/$NAME.json"

docker run --rm \
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
  /workspace/experiments/run_qwen_residual_attention_value_probe.py \
  --codebook-checkpoint /codebook \
  --tokenizer /hf-hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --calibration-prompts-json /workspace/experiments/p35_residual_attention_v_calibration_prompts_20260711.json \
  --heldout-prompts-json /workspace/experiments/p35_residual_attention_v_heldout_prompts_20260711.json \
  --output "/output/$NAME.json" \
  --device cuda:0 \
  --dtype float32 \
  --horizon 4 \
  --ridge 1e-6 \
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
  --local-files-only
