#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
SOURCE_SUMMARY="${SOURCE_SUMMARY:?Set SOURCE_SUMMARY to a chain_anchor_summary.json from the Qwen-folded child.}"
SOURCE_MODEL="${SOURCE_MODEL:-LiquidAI/LFM2.5-230M}"
TAG="${TAG:-lfm230m_stabilizer_after_qwen_child_gpu6_20260629}"
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"

ANCHOR_MODEL="$("$PY" - <<PY
import json, pathlib
r = json.loads(pathlib.Path("$SOURCE_SUMMARY").read_text())
print(r.get("final_model_dir") or r["best_model_dir"])
PY
)"

export RUN_ROOT REPO_ROOT SOURCE_MODEL ANCHOR_MODEL TAG
exec "$REPO_ROOT/experiments/launch_lfm230m_blank_smollm_chain.sh"
