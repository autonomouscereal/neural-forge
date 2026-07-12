#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$SRC/runs/qwen4b_to_smollm135m_transport_random_20260627}"
EVAL_DIR="${EVAL_DIR:-$RUN_ROOT/evals}"
QUEUE_PID="${QUEUE_PID:-2331766}"
TAG="${TAG:-qwen4b_blank_best_drill_recover_gpu6_20260702}"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p "$EVAL_DIR" "$RUN_ROOT/logs"
cd "$SRC"

echo "compare_queue_start=$(date -Is) tag=$TAG upstream_pid=$QUEUE_PID revision=$(cat .deployed_revision 2>/dev/null || true)"
if [[ "$QUEUE_PID" != "0" ]]; then
  while kill -0 "$QUEUE_PID" 2>/dev/null; do
    echo "$(date -Is) waiting_for_upstream_pid=$QUEUE_PID"
    sleep "$POLL_SECONDS"
  done
fi

RANK="$EVAL_DIR/rank_blank_candidates_before_drill_exactgen_20260702.json"
PARENT_REPORT="$EVAL_DIR/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json"

for path in "$RANK"; do
  if [[ ! -f "$path" ]]; then
    echo "missing_required_file=$path" >&2
    exit 2
  fi
done

BEST_SOURCE=$(/home/cereal/model-surgery-lab/venv/bin/python - "$RANK" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1], encoding="utf-8"))
print(rank["best"]["source"])
PY
)

BEST_MODEL=$(/home/cereal/model-surgery-lab/venv/bin/python - "$RANK" <<'PY'
import json, sys
rank = json.load(open(sys.argv[1], encoding="utf-8"))
print(rank["best"]["model"])
PY
)

BEST_TAG=$(basename "$(dirname "$(dirname "$BEST_MODEL")")")
AFTER_REPORT="$EVAL_DIR/eval_${BEST_TAG}_drill_recovered_exactgen_20260702_1000x1000x250.json"
OUT_DIR="$EVAL_DIR/retention_${BEST_TAG}_drill_20260702"
mkdir -p "$OUT_DIR"

if [[ ! -f "$BEST_SOURCE" ]]; then
  echo "missing_before_report=$BEST_SOURCE" >&2
  exit 3
fi
if [[ ! -f "$AFTER_REPORT" ]]; then
  echo "missing_after_report=$AFTER_REPORT" >&2
  exit 4
fi

echo "selected tag=$BEST_TAG before=$BEST_SOURCE after=$AFTER_REPORT parent=$PARENT_REPORT"

if [[ -f "$PARENT_REPORT" ]]; then
  /home/cereal/model-surgery-lab/venv/bin/python experiments/compare_eval_retention.py \
    "$PARENT_REPORT" \
    "$BEST_SOURCE" \
    --out "$OUT_DIR/parent_vs_before.json" \
    --max-samples 12
  /home/cereal/model-surgery-lab/venv/bin/python experiments/compare_eval_retention.py \
    "$PARENT_REPORT" \
    "$AFTER_REPORT" \
    --out "$OUT_DIR/parent_vs_after.json" \
    --max-samples 12
else
  echo "missing_parent_report=$PARENT_REPORT"
fi

/home/cereal/model-surgery-lab/venv/bin/python experiments/compare_eval_retention.py \
  "$BEST_SOURCE" \
  "$AFTER_REPORT" \
  --out "$OUT_DIR/before_vs_after.json" \
  --max-samples 12

/home/cereal/model-surgery-lab/venv/bin/python - "$OUT_DIR" "$BEST_TAG" "$PARENT_REPORT" "$BEST_SOURCE" "$AFTER_REPORT" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
summary = {
    "operation": "blank_drill_retention_compare",
    "best_tag": sys.argv[2],
    "parent_report": sys.argv[3],
    "before_report": sys.argv[4],
    "after_report": sys.argv[5],
    "comparisons": {},
}
for name in ("parent_vs_before", "parent_vs_after", "before_vs_after"):
    path = out_dir / f"{name}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        summary["comparisons"][name] = {
            "path": str(path),
            "parent": data.get("parent"),
            "child": data.get("child"),
            "param_compression_ratio": data.get("param_compression_ratio"),
            "behavior": data.get("behavior", {}).get("overall"),
            "generation": data.get("generation", {}).get("overall"),
            "choice": data.get("choice", {}).get("overall"),
            "manual_behavior_samples": data.get("manual_samples", {}).get("behavior", [])[:3],
            "manual_generation_samples": data.get("manual_samples", {}).get("generation", [])[:3],
        }
summary_path = out_dir / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "compare_queue_done=$(date -Is) tag=$TAG out_dir=$OUT_DIR"
