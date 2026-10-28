set -eu

RUN=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_bridge_20260730/a9-k32-residual-omp

if [ ! -d "$RUN" ]; then
  printf '{"stage":"monitor","status":"not-started","run":"%s"}\n' "$RUN"
  exit 0
fi

pid=""
if [ -f "$RUN/driver.pid" ]; then
  pid=$(cat "$RUN/driver.pid")
fi
running=false
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  running=true
fi

complete=false
if [ -f "$RUN/run-receipt.json" ]; then
  complete=true
fi

RUN="$RUN" PID_VALUE="$pid" RUNNING_VALUE="$running" COMPLETE_VALUE="$complete" python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN"])
log_path = run / "job.log"
counts = {
    "cache_hit": 0,
    "cache_miss": 0,
    "compute_progress": 0,
    "prefix_replay": 0,
    "publication": 0,
}
last_core = None
last_stage = None
last_phase = None
last_error = None
if log_path.is_file():
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if "Traceback (most recent call last)" in raw or "CUDA out of memory" in raw:
                last_error = raw.strip()
            try:
                event = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            nested = event
            while isinstance(nested, dict):
                phase = nested.get("phase")
                if isinstance(phase, str):
                    last_phase = phase
                stage = nested.get("stage")
                if isinstance(stage, str):
                    last_stage = stage
                    if stage == "cache-hit":
                        counts["cache_hit"] += 1
                    elif stage == "cache-miss":
                        counts["cache_miss"] += 1
                    elif stage == "compute-progress":
                        counts["compute_progress"] += 1
                    elif stage == "publication-complete":
                        counts["publication"] += 1
                details = nested.get("details")
                nested = details if isinstance(details, dict) else None
            serialized = raw
            if "omp-prefix-step-replayed" in serialized:
                counts["prefix_replay"] += 1
            cursor = event
            for key in ("details", "details", "compiler_progress", "core_progress"):
                cursor = cursor.get(key, {}) if isinstance(cursor, dict) else {}
            if isinstance(cursor, dict) and cursor.get("event") in {
                "omp-prefix-step-replayed",
                "omp-step-complete",
                "chunk-complete",
            }:
                last_core = {
                    key: cursor.get(key)
                    for key in (
                        "event",
                        "donor_row_start",
                        "donor_row_stop",
                        "loaded_support_depth",
                        "support_depth",
                        "omp_step",
                        "progress_index",
                    )
                    if cursor.get(key) is not None
                }

artifacts = sorted(
    path.name
    for path in run.iterdir()
    if path.is_file() and path.name not in {"job.log", "driver.pid"}
) if run.is_dir() else []
print(json.dumps({
    "schema": "neural-forge-live-job-monitor-v1",
    "run": str(run),
    "pid": os.environ.get("PID_VALUE") or None,
    "running": os.environ.get("RUNNING_VALUE") == "true",
    "complete": os.environ.get("COMPLETE_VALUE") == "true",
    "log_bytes": log_path.stat().st_size if log_path.is_file() else 0,
    "event_counts": counts,
    "last_stage": last_stage,
    "last_phase": last_phase,
    "last_core_progress": last_core,
    "last_error_marker": last_error,
    "published_artifacts": artifacts,
}, sort_keys=True))
PY
