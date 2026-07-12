#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
cd "$ROOT"
source /home/cereal/model-surgery-lab/venv/bin/activate

python -m py_compile \
  src/neural_forge/hf_xfamily.py \
  src/neural_forge/cli.py \
  src/neural_forge/lrc.py \
  tests/test_hf_xfamily.py \
  tests/test_lrc.py

PYTHONPATH=src python -m pytest tests/test_hf_xfamily.py tests/test_lrc.py -q -k \
  'lrc or donor_project_init_handler_passes_flags or hf_cli_parses_hf_commands or chain_anchor_init_stage_forwards_activation_fit_flags or chain_anchor_handler_passes_flags'
