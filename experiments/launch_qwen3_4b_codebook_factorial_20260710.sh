#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}
launcher="$root/experiments/launch_qwen3_4b_codebook_20260710.sh"

GPU=GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561 \
P_NEURON=35 \
P_ATTENTION=36 \
TAG=qwen3_4b_codebook_pn35_pa36_gpu0_r1_20260710 \
ROOT="$root" \
bash "$launcher"

GPU=GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 \
P_NEURON=36 \
P_ATTENTION=35 \
TAG=qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710 \
ROOT="$root" \
bash "$launcher"

GPU=GPU-f5a09535-fc85-a027-50f7-734b344747dd \
P_NEURON=34 \
P_ATTENTION=36 \
TAG=qwen3_4b_codebook_pn34_pa36_gpu6_r1_20260710 \
ROOT="$root" \
bash "$launcher"
