#!/usr/bin/env bash
set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-d296b602
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

cd "$RUNTIME"
test "$(git rev-parse HEAD)" = d296b602eec723461d476043617641ef7a2c64c4
test -z "$(git status --porcelain)"
export PYTHONPATH="$RUNTIME/src"
"$PYTHON" -m pytest -q \
  tests/test_hf_lfm2_child_replay_batch_checkpoint.py \
  tests/test_dynamic_closed_loop_fold.py \
  tests/test_run_lfm2_dynamic_catalog.py \
  tests/test_hf_lfm2_structural_residual_attention_fold.py \
  tests/test_hf_lfm2_structural_split_install.py \
  tests/test_hf_lfm2_dynamic_closed_loop_heterogeneous.py \
  tests/test_hf_lfm2_dense_budget_model_fold_heterogeneous.py \
  tests/test_hf_lfm2_dynamic_closed_loop_adapter.py \
  tests/test_hf_lfm2_tokenizer_bridge_trace_adapter.py \
  tests/test_hf_lfm2_child_closed_loop_token_bridge.py \
  tests/test_hf_lfm2_child_closed_loop_refinement.py \
  tests/test_hf_lfm2_sealed_evidence_set.py
