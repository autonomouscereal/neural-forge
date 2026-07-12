#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711/p35_mixed_drop16_pareto_v4_20260711
PID_FILE=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711/p35_mixed_drop16_pareto_v4_20260711.pid

python3 - "$ROOT" "$PID_FILE" <<'PY'
from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve()
pid = int(Path(sys.argv[2]).read_text(encoding="utf-8").strip())
if Path(f"/proc/{pid}").exists():
    raise SystemExit("refusing to finalize a live process")
report_path = root / "run_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if int(report.get("accepted_transactions_total", 0)) != 0:
    raise SystemExit("refusing to finalize a run with accepted transactions")
if report.get("current_checkpoint") is not None:
    raise SystemExit("refusing to finalize a run with a current checkpoint")
if any((root / "accepted_commit_journal").glob("commit-*.json")):
    raise SystemExit("refusing to finalize a run with journal commits")
report["status"] = "operator_stopped_pre_mutation"
report["error"] = (
    "Operator stop after audit found that process-local allocator state hashes "
    "cannot be compared across the v3/v4 import boundary."
)
report["operator_stop"] = {
    "accepted_transactions_total": 0,
    "candidate_validation_rounds": 0,
    "checkpoint_count": 0,
    "journal_commit_count": 0,
    "model_mutated": False,
    "reason": "pre-import cross-process state-hash contract correction",
}
payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
temporary = report_path.with_suffix(".json.tmp")
temporary.write_text(payload, encoding="utf-8")
temporary.replace(report_path)
print(
    json.dumps(
        {
            "report": str(report_path),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "status": report["status"],
        },
        sort_keys=True,
    )
)
PY
