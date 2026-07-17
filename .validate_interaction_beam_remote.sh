#!/usr/bin/env bash
set -euo pipefail

source_root=/home/cereal/model-surgery-lab/neural-forge-src
venv_python=/home/cereal/model-surgery-lab/venv/bin/python

cd "$source_root"
PYTHONPATH="$source_root/src:$source_root" \
  "$venv_python" -m pytest -q \
    tests/test_hf_lfm2_global_causal_allocator.py \
    -k "interaction_beam or canonical_multi_pack or canonical_finalizer or canonical_merge_wrapper or canonical_frontier_replay or canonical_frontier_authentication"
