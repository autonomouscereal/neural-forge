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

def load(root, name):
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

def suite_counts(payload):
    if not isinstance(payload, dict):
        return None
    result = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        lowered = key.lower()
        if any(word in lowered for word in ("behavior", "choice", "generation")):
            picked = {
                name: value.get(name)
                for name in (
                    "correct", "total", "parent_correct", "candidate_correct",
                    "retained_parent_passes", "loss_count", "gain_count", "passed",
                )
                if name in value
            }
            if picked:
                result[key] = picked
    for key in ("perplexity", "nll", "loss"):
        if key in payload and isinstance(payload[key], (int, float)):
            result[key] = payload[key]
    return result or None

for tag in tags:
    root = base / tag
    state = load(root, "run_state.json") or {}
    transaction = load(root, "transaction_report.json") or {}
    final_eval = load(root, "final_eval.json") or {}
    comparison = load(root, "parent_final_comparison.json") or {}
    verdict = load(root, "certification_verdict.json") or {}
    accepted = []
    for item in transaction.get("accepted_transactions", []):
        proposal = item.get("proposal", {})
        accepted.append({
            "attempt": item.get("attempt"),
            "layer": proposal.get("layer_index"),
            "strategy": proposal.get("fold_strategy"),
            "parameter_savings": proposal.get("parameter_savings"),
            "pair": proposal.get("merges"),
        })
    rejected = []
    for item in transaction.get("rejected_transactions", []):
        proposal = item.get("proposal", {})
        small = ((item.get("small_gate") or {}).get("verdict") or {}).get("domains")
        rejected.append({
            "attempt": item.get("attempt"),
            "layer": proposal.get("layer_index"),
            "strategy": proposal.get("fold_strategy"),
            "parameter_savings": proposal.get("parameter_savings"),
            "small_gate_domains": small,
        })
    summary = {
        "tag": tag,
        "exists": root.is_dir(),
        "run_state": {key: state.get(key) for key in ("status", "stage", "details")},
        "transactions": {
            "stop_reason": transaction.get("stop_reason"),
            "accepted_count": transaction.get("accepted_count"),
            "rejected_count": transaction.get("rejected_count"),
            "accepted": accepted,
            "rejected": rejected,
        },
        "final_eval_top": suite_counts(final_eval.get("summary", final_eval)),
        "comparison_top": suite_counts(comparison.get("summary", comparison)),
        "certification": verdict,
    }
    print(json.dumps(summary, sort_keys=True))
PY
