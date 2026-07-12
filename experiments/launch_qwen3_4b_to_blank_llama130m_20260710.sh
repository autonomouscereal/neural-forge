#!/usr/bin/env bash
set -euo pipefail

export TAG=qwen3_4b_to_blank_llama130m_structuralprior_gpu4_r2_20260710
export GPU=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582
export HIDDEN_SIZE=512
export INTERMEDIATE_SIZE=1024
export NUM_ATTENTION_HEADS=8
export EXACT_PARAMETER_BUDGET=129720832

exec bash /home/cereal/model-surgery-lab/neural-forge-src/experiments/launch_qwen3_4b_to_blank_llama88m_20260710.sh
