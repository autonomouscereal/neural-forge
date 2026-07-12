#!/usr/bin/env bash
set -euo pipefail

export TAG=qwen3_4b_to_blank_llama131m_headgqa_producttrust_gpu4_r1_20260710
export GPU=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
export TARGET_FAMILY=llama
export HIDDEN_SIZE=512
export INTERMEDIATE_SIZE=1024
export NUM_HIDDEN_LAYERS=24
export NUM_ATTENTION_HEADS=4
export NUM_KEY_VALUE_HEADS=1
export HEAD_DIM=128
export EXACT_PARAMETER_BUDGET=131293696
export MAX_OPERATOR_DELTA_RATIO=4

exec bash /home/cereal/model-surgery-lab/neural-forge-src/experiments/launch_qwen3_4b_to_blank_llama88m_20260710.sh
