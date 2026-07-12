#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
TAG=p35_mixed_drop16_pareto_v9_checkpointdetach_20260711
PID_FILE=$RUN_ROOT/$TAG.pid
IDENTITY_FILE=$RUN_ROOT/$TAG.identity.json

test -f "$PID_FILE"
test -f "$IDENTITY_FILE"
PID=$(cat "$PID_FILE")
EXPECTED_START=$(python3 - "$IDENTITY_FILE" <<'PY'
from pathlib import Path
import json
import sys

print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["start_ticks"])
PY
)
test -r "/proc/$PID/stat"
ACTUAL_START=$(awk '{print $22}' "/proc/$PID/stat")
test "$ACTUAL_START" = "$EXPECTED_START"
kill -TERM "$PID"
printf 'stopped pid=%s startticks=%s tag=%s reason=exact-v8-replay-avoids-duplicate-evaluation\n' \
  "$PID" "$ACTUAL_START" "$TAG"
