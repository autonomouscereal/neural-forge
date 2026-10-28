#!/usr/bin/env bash
set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-b65c81f5
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

cd "$RUNTIME"
test "$(git rev-parse HEAD)" = b65c81f5d086830247b0a418a78d54fb12975ca6
test -z "$(git status --porcelain)"
export PYTHONPATH="$RUNTIME/src"
"$PYTHON" -m pytest -q \
  tests/test_hf_lfm2_child_replay_batch_checkpoint.py \
  tests/test_dynamic_closed_loop_fold.py \
  tests/test_run_lfm2_dynamic_catalog.py \
  tests/test_hf_lfm2_structural_residual_attention_fold.py \
  tests/test_hf_lfm2_structural_split_install.py \
  tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
  tests/test_hf_lfm2_dense_budget_model_fold_heterogeneous.py
