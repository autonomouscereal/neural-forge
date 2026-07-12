#!/usr/bin/env bash
set -euo pipefail

export TAG=qwen3_4b_to_blank_qwen134m_headgqa_producttrust_gpu0_r1_20260710
export GPU=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561
export TARGET_FAMILY=qwen3
export HIDDEN_SIZE=512
export INTERMEDIATE_SIZE=1024
export NUM_HIDDEN_LAYERS=24
export NUM_ATTENTION_HEADS=4
export NUM_KEY_VALUE_HEADS=2
export HEAD_DIM=128
export EXACT_PARAMETER_BUDGET=134445568
export MAX_OPERATOR_DELTA_RATIO=4

exec bash /home/cereal/model-surgery-lab/neural-forge-src/experiments/launch_qwen3_4b_to_blank_llama88m_20260710.sh
