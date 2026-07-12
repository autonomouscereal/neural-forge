#!/usr/bin/env bash
set -euo pipefail

BASE=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710
TAG=qwen3_4b_pair_drop1_late33to35_fragile_v5c_freshcert_v5_20260711
ROOT="$BASE/$TAG"
UNIFIED=/home/cereal/model-surgery-lab/neural-forge-unified-20260710
SELECTION=/home/cereal/model-surgery-lab/neural-forge-pivot/runs/qwen3_compensated_fragile_selection_v5c_20260711.json

python3 - "$ROOT" "$SELECTION" "$UNIFIED" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
selection = Path(sys.argv[2])
unified = Path(sys.argv[3])

def load(name):
    path = root / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

def counts(report):
    if not isinstance(report, dict):
        return None
    return {
        "choice": {
            key: (report.get("choice_accuracy") or {}).get(key)
            for key in ("correct", "total", "accuracy")
        },
        "generation": {
            key: (report.get("generation_health") or {}).get(key)
            for key in ("correct", "total", "pass_rate")
        },
        "behavior": {
            key: (report.get("behavior_suite") or {}).get(key)
            for key in ("correct", "total", "pass_rate")
        },
        "param_count": report.get("param_count"),
    }

state = load("run_state.json") or {}
transaction = load("transaction_report.json") or {}
final_eval = load("final_eval.json")
comparison = load("parent_final_comparison.json")
certification = load("certification_verdict.json")
manual_packet = load("manual_review_packet.json")
selection_payload = json.loads(selection.read_text(encoding="utf-8"))
result = {
    "root": str(root),
    "root_exists": root.is_dir(),
    "run_state": {
        key: state.get(key) for key in ("status", "stage", "details")
    },
    "transactions": {
        key: transaction.get(key)
        for key in ("status", "stop_reason", "accepted_count", "rejected_count")
    },
    "accepted_transactions": transaction.get("accepted_transactions"),
    "final_eval": counts(final_eval),
    "parent_final_comparison": comparison.get("summary") if isinstance(comparison, dict) else None,
    "certification_verdict": certification,
    "manual_review_packet_exists": isinstance(manual_packet, dict),
    "checkpoint_directories": sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_dir() and (path / "manifest.json").is_file()
    ) if root.is_dir() else [],
    "selection_counts": {
        key: len(selection_payload.get(key, []))
        for key in ("text_evals", "choice_probes", "generation_prompts", "behavior_probes")
    },
    "unified_checkout_exists": unified.exists(),
}
print(json.dumps(result, sort_keys=True))
PY

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

pgrep -af 'qwen3_4b_pair_drop1_late33to35|run_qwen_unified_surgery' || true
