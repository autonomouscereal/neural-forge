#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-pivot}
RUN_ROOT=${RUN_ROOT:-/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710}
PY=${PY:-/home/cereal/model-surgery-lab/venv/bin/python}
GPU=${GPU:?Set GPU to a physical GPU UUID.}
P_NEURON=${P_NEURON:?Set P_NEURON to a prototype depth or logical_layers.}
P_ATTENTION=${P_ATTENTION:?Set P_ATTENTION to a prototype depth or logical_layers.}
TAG=${TAG:-qwen3_4b_codebook_pn${P_NEURON}_pa${P_ATTENTION}_r1_20260710}
CALIBRATION_COUNT=${CALIBRATION_COUNT:-128}
ACTIVATION_ROWS=${ACTIVATION_ROWS:-256}

MODEL=/home/cereal/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Base/snapshots/906bfd4b4dc7f14ee4320094d8b41684abff8539
CALIBRATION=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/calibration_mix_plus_recovery_drills_20260702.jsonl
EVAL_SUITE=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/behavior_suite_exactgen_20260702.json
PARENT_EVAL=/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627/evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json

mkdir -p "$RUN_ROOT"

"$PY" "$ROOT/experiments/launch_detached.py" \
  --log "$RUN_ROOT/$TAG.log" \
  --pid-file "$RUN_ROOT/$TAG.pid" \
  --identity-file "$RUN_ROOT/$TAG.identity.json" \
  --cwd "$ROOT" \
  --env "CUDA_VISIBLE_DEVICES=$GPU" \
  --env "PYTHONPATH=$ROOT/src" \
  --env "TOKENIZERS_PARALLELISM=false" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- \
  "$PY" "$ROOT/experiments/run_qwen_codebook_in_memory.py" \
  --source-model "$MODEL" \
  --source-tokenizer "$MODEL" \
  --target-tokenizer "$MODEL" \
  --output-dir "$RUN_ROOT/$TAG" \
  --calibration-jsonl "$CALIBRATION" \
  --eval-suite "$EVAL_SUITE" \
  --cached-parent-eval "$PARENT_EVAL" \
  --calibration-count "$CALIBRATION_COUNT" \
  --calibration-max-length 160 \
  --activation-rows "$ACTIVATION_ROWS" \
  --p-neuron "$P_NEURON" \
  --p-attention "$P_ATTENTION" \
  --dtype float16 \
  --device cuda:0 \
  --max-new-tokens 96 \
  --max-text-evals 1000 \
  --max-choice-probes 250 \
  --max-generation-prompts 1000 \
  --max-behavior-probes 1000 \
  --manual-sample-count 128 \
  --local-files-only

echo "launched tag=$TAG gpu=$GPU p_neuron=$P_NEURON p_attention=$P_ATTENTION"
