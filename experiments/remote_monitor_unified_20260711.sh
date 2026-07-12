#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
TAG=${TAG:-p35_mixed_drop16_pareto_v7_20260711}
ROOT="$RUN_ROOT/$TAG"
PID_FILE="$RUN_ROOT/$TAG.pid"
IDENTITY_FILE="$RUN_ROOT/$TAG.identity.json"
LOG="$RUN_ROOT/$TAG.log"

python3 - "$ROOT" "$PID_FILE" "$IDENTITY_FILE" "$LOG" <<'PY'
from pathlib import Path
import json
import os
import sys

root = Path(sys.argv[1])
pid_path = Path(sys.argv[2])
identity_path = Path(sys.argv[3])
log = Path(sys.argv[4])
pid = int(pid_path.read_text(encoding="utf-8").strip())
identity = json.loads(identity_path.read_text(encoding="utf-8"))
running = Path(f"/proc/{pid}").exists()
report_path = root / "run_report.json"
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
initial_path = root / "initial_incumbent.json"
initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.is_file() else {}
accounting = (
    report.get("current_accounting")
    or report.get("initial_accounting")
    or initial.get("accounting")
    or {}
)
run_contract = report.get("run_contract") or initial.get("run_contract") or {}
result = {
    "pid": pid,
    "running": running,
    "identity_start_ticks": identity.get("start_ticks"),
    "status": report.get("status"),
    "error": report.get("error"),
    "attempts_total": report.get("attempts_total"),
    "accepted_transactions_total": report.get("accepted_transactions_total"),
    "rejected_transactions_total": report.get("rejected_transactions_total"),
    "target_reached": report.get("target_reached"),
    "candidate_marginal_claim": report.get("candidate_marginal_claim"),
    "checkpoint_artifact_bytes": accounting.get("deployable_bytes"),
    "child_unique_parameters": accounting.get("child_unique_parameters"),
    "resolved_target_checkpoint_bytes": (
        report.get("resolved_target_deployable_bytes")
        or run_contract.get("resolved_target_deployable_bytes")
    ),
    "initial_incumbent_exists": initial_path.is_file(),
    "journal_commit_count": len(list((root / "accepted_commit_journal").glob("commit-*.json"))),
    "validation_round_count": len(list((root / "candidate_validation").glob("round-*"))),
    "candidate_probe_count": len(
        list((root / "candidate_validation").glob("round-*/candidate-*.json"))
    ),
    "candidate_summary_count": len(
        list((root / "candidate_validation").glob("round-*/validation_summary.json"))
    ),
    "transaction_gate_count": len(list((root / "transaction_gates").glob("*.gate.json"))),
    "checkpoint_count": len(list((root / "checkpoints").glob("state-*"))),
    "top_level_files": sorted(path.name for path in root.glob("*") if path.is_file()),
    "log_bytes": log.stat().st_size if log.is_file() else 0,
}
print(json.dumps(result, sort_keys=True))
PY

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

printf '%s\n' LOG_TAIL
tail -n 80 "$LOG" || true
