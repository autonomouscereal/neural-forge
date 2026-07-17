#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
venv_python=/home/cereal/model-surgery-lab/venv/bin/python

cd "$source_root"

sha256sum \
  experiments/build_lfm2_cached_causal_evidence.py \
  src/neural_forge/hf_lfm2_variable_width_fold.py \
  src/neural_forge/hf_lfm2_canonical_proposal_pack.py \
  src/neural_forge/hf_lfm2_cached_causal_sharding.py \
  src/neural_forge/hf_lfm2_interaction_beam_allocator.py \
  src/neural_forge/hf_lfm2_canonical_frontier_replay.py \
  experiments/finalize_lfm2_cached_causal_merge_canonical.py \
  experiments/run_lfm2_global_causal_allocator.py \
  tests/test_hf_lfm2_cached_causal_sharding.py \
  tests/test_hf_lfm2_global_causal_allocator.py

"$venv_python" -m py_compile \
  experiments/build_lfm2_cached_causal_evidence.py \
  experiments/finalize_lfm2_cached_causal_merge_canonical.py \
  experiments/run_lfm2_global_causal_allocator.py \
  src/neural_forge/hf_lfm2_variable_width_fold.py \
  src/neural_forge/hf_lfm2_canonical_proposal_pack.py \
  src/neural_forge/hf_lfm2_cached_causal_sharding.py \
  src/neural_forge/hf_lfm2_interaction_beam_allocator.py \
  src/neural_forge/hf_lfm2_canonical_frontier_replay.py \
  tests/test_hf_lfm2_cached_causal_sharding.py

"$venv_python" -m pytest \
  tests/test_hf_lfm2_variable_width_fold.py \
  tests/test_hf_lfm2_cached_causal_sharding.py \
  tests/test_hf_lfm2_global_causal_allocator.py \
  -q
