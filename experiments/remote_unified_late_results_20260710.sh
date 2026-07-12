#!/usr/bin/env bash
set -euo pipefail

base=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710

python3 - "$base" <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
tags = (
    "qwen3_4b_pair_drop1_layer34_dev_reused_eval_20260710",
    "qwen3_4b_pair_drop1_layer33_dev_reused_eval_20260710",
    "qwen3_4b_multi_anchor_drop1_layer35_dev_reused_eval_20260710",
)
files = (
    "run_state.json",
    "transaction_report.json",
    "final_eval.json",
    "parent_final_comparison.json",
    "certification_verdict.json",
    "selection_certification_split.json",
    "three_split_provenance.json",
)

for tag in tags:
    root = base / tag
    print(f"TAG {tag}")
    print(json.dumps({
        "exists": root.is_dir(),
        "files": sorted(path.name for path in root.glob("*") if path.is_file()),
        "transaction_gate_count": len(list((root / "transaction_gates").glob("*.json"))),
        "milestone_count": len(list((root / "milestones").glob("*.json"))),
    }, sort_keys=True))
    for name in files:
        path = root / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            print(name, json.dumps({"read_error": repr(error)}, sort_keys=True))
            continue
        if name == "transaction_report.json":
            payload = {
                "status": payload.get("status"),
                "stop_reason": payload.get("stop_reason"),
                "accepted_count": payload.get("accepted_count"),
                "rejected_count": payload.get("rejected_count"),
                "accepted_transactions": payload.get("accepted_transactions"),
                "rejected_transactions": payload.get("rejected_transactions"),
                "final_accounting": payload.get("final_accounting"),
            }
        elif name in {"final_eval.json", "parent_final_comparison.json"}:
            payload = payload.get("summary", payload)
        print(name, json.dumps(payload, sort_keys=True)[:30000])
PY

printf '%s\n' GPU_STATUS
nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

printf '%s\n' RELEVANT_PROCESSES
pgrep -af 'qwen3_4b_(pair|multi_anchor)|run_qwen_compensated' || true
