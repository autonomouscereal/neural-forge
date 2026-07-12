#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
TAG=p35_mixed_drop16_pareto_v4_20260711
PID_FILE="$RUN_ROOT/$TAG.pid"
IDENTITY_FILE="$RUN_ROOT/$TAG.identity.json"
OUTPUT="$RUN_ROOT/$TAG"

python3 - "$PID_FILE" "$IDENTITY_FILE" "$OUTPUT" <<'PY'
from pathlib import Path
import json
import os
import signal
import sys
import time

pid = int(Path(sys.argv[1]).read_text(encoding="utf-8").strip())
identity = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
report_path = Path(sys.argv[3]) / "run_report.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
if int(report.get("accepted_transactions_total", 0)) != 0:
    raise SystemExit("refusing to stop a run with an accepted transaction")
if report.get("current_checkpoint") is not None:
    raise SystemExit("refusing to stop a run with a current checkpoint")
stat = Path(f"/proc/{pid}/stat")
if not stat.is_file():
    print(json.dumps({"pid": pid, "already_exited": True}, sort_keys=True))
    raise SystemExit(0)
start_ticks = int(stat.read_text(encoding="utf-8").split()[21])
if start_ticks != int(identity["start_ticks"]):
    raise SystemExit("PID identity changed; refusing to signal")
os.kill(pid, signal.SIGTERM)
for _ in range(300):
    if not stat.exists():
        print(json.dumps({"pid": pid, "terminated": True}, sort_keys=True))
        raise SystemExit(0)
    time.sleep(0.1)
raise SystemExit("process did not exit after SIGTERM")
PY
