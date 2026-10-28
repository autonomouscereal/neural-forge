#!/usr/bin/env bash
set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-6f178f14
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

cd "$RUNTIME"
test "$(git rev-parse HEAD)" = 6f178f14add54ec882c09ad183f69a626a0cfb48
test -z "$(git status --porcelain)"
export PYTHONPATH="$RUNTIME/src"
"$PYTHON" -m pytest -q \
  tests/test_hf_lfm2_child_replay_batch_checkpoint.py \
  tests/test_dynamic_closed_loop_fold.py \
  tests/test_run_lfm2_dynamic_catalog.py
