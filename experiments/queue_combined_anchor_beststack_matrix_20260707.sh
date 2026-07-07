#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO_ROOT="${REPO_ROOT:-/home/cereal/model-surgery-lab/neural-forge-src}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/qwen4b_to_smollm135m_transport_random_20260627}"
MATRIX_EVAL_DIR="${MATRIX_EVAL_DIR:-$RUN_ROOT/evals/combined_anchor_beststack_20260707}"
MATRIX_STATE="${MATRIX_STATE:-$MATRIX_EVAL_DIR/queue_state.json}"
MATRIX_SUMMARY_JSON="${MATRIX_SUMMARY_JSON:-$MATRIX_EVAL_DIR/summary.json}"
MATRIX_SUMMARY_MD="${MATRIX_SUMMARY_MD:-$MATRIX_EVAL_DIR/summary.md}"
POLL_SECONDS="${POLL_SECONDS:-60}"
UPSTREAM_WAIT_PIDS="${WAIT_PIDS:-0}"

GPU0_UUID="${GPU0_UUID:-GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561}"
GPU4_UUID="${GPU4_UUID:-GPU-2ce6a95c-7b1f-644a-3808-d781732a0582}"
GPU6_UUID="${GPU6_UUID:-GPU-f5a09535-fc85-a027-50f7-734b344747dd}"

QUEUE_PRIOR_ANCHORS="${QUEUE_PRIOR_ANCHORS:-1}"
QUEUE_QWEN_LFM="${QUEUE_QWEN_LFM:-1}"
QUEUE_QWEN_GPT2="${QUEUE_QWEN_GPT2:-1}"
SUMMARIZE_NOW="${SUMMARIZE_NOW:-1}"

DUALALM29_TAG="${DUALALM29_TAG:-qwen4b_dualalm29_anchor_topneg_exact_gpu6_20260707}"
FFN1496_TAG="${FFN1496_TAG:-qwen4b_ffn1496_strict_anchor_topneg_gpu6_20260707}"
QWEN_LFM_TAG="${QWEN_LFM_TAG:-qwen4b_lfm230m_multidonor_beststack_seqall_qrsinkhorn_gpu0_20260707}"
QWEN_GPT2_TAG="${QWEN_GPT2_TAG:-qwen4b_gpt2_multidonor_beststack_seqall_qrsinkhorn_gpu4_20260707}"

mkdir -p "$RUN_ROOT/logs" "$MATRIX_EVAL_DIR"

launch_detached() {
  local label="$1"
  shift
  local log="$RUN_ROOT/logs/${label}.matrix.queue.log"
  local pid_file="$RUN_ROOT/logs/${label}.matrix.queue.pid"

  echo "combined_matrix_launch=$(date -Is) label=$label log=$log" | tee -a "$log"
  nohup "$@" >> "$log" 2>&1 &
  local pid="$!"
  printf '%s\n' "$pid" > "$pid_file"
  echo "combined_matrix_pid label=$label pid=$pid log=$log pid_file=$pid_file"
}

if [[ "$QUEUE_PRIOR_ANCHORS" = "1" ]]; then
  launch_detached prior_anchor_pair_20260707 \
    env \
      REPO_ROOT="$REPO_ROOT" \
      RUN_ROOT="$RUN_ROOT" \
      EVAL_DIR="$MATRIX_EVAL_DIR" \
      GPU_UUID="$GPU6_UUID" \
      WAIT_PIDS="$UPSTREAM_WAIT_PIDS" \
      POLL_SECONDS="$POLL_SECONDS" \
      DUALALM29_TAG="$DUALALM29_TAG" \
      FFN1496_TAG="$FFN1496_TAG" \
      bash "$SCRIPT_DIR/queue_qwen4b_prior_anchor_pair_20260707.sh"
fi

if [[ "$QUEUE_QWEN_LFM" = "1" ]]; then
  launch_detached "$QWEN_LFM_TAG" \
    env \
      REPO_ROOT="$REPO_ROOT" \
      RUN_ROOT="$RUN_ROOT" \
      BESTSTACK_EVAL_DIR="$MATRIX_EVAL_DIR" \
      GPU_UUID="$GPU0_UUID" \
      WAIT_PIDS="${QWEN_LFM_WAIT_PIDS:-0}" \
      POLL_SECONDS="$POLL_SECONDS" \
      TAG="$QWEN_LFM_TAG" \
      bash "$SCRIPT_DIR/queue_qwen4b_lfm230m_multidonor_beststack_20260707.sh"
fi

if [[ "$QUEUE_QWEN_GPT2" = "1" ]]; then
  launch_detached "$QWEN_GPT2_TAG" \
    env \
      REPO_ROOT="$REPO_ROOT" \
      RUN_ROOT="$RUN_ROOT" \
      BESTSTACK_EVAL_DIR="$MATRIX_EVAL_DIR" \
      VARIANT=qwen_gpt2 \
      GPU_UUID="$GPU4_UUID" \
      WAIT_PIDS="${QWEN_GPT2_WAIT_PIDS:-0}" \
      POLL_SECONDS="$POLL_SECONDS" \
      TAG="$QWEN_GPT2_TAG" \
      INIT_TRANSPORT_BACKEND="${INIT_TRANSPORT_BACKEND:-qr-sinkhorn}" \
      TRANSPORT_SUPPORT="${TRANSPORT_SUPPORT:-quantile-medoid}" \
      RECOVERY_SOURCE_MODE="${RECOVERY_SOURCE_MODE:-sequential-all}" \
      FINAL_OUTPUT_HEAD_POLICY="${FINAL_OUTPUT_HEAD_POLICY:-tie}" \
      bash "$SCRIPT_DIR/queue_beststack_after_retest_20260706.sh"
fi

python - <<'PY' "$MATRIX_STATE" "$REPO_ROOT" "$RUN_ROOT" "$MATRIX_EVAL_DIR" \
  "$DUALALM29_TAG" "$FFN1496_TAG" "$QWEN_LFM_TAG" "$QWEN_GPT2_TAG" \
  "$QUEUE_PRIOR_ANCHORS" "$QUEUE_QWEN_LFM" "$QUEUE_QWEN_GPT2"
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
run_root = Path(sys.argv[3])
eval_dir = Path(sys.argv[4])
dual_tag, ffn_tag, qwen_lfm_tag, qwen_gpt2_tag = sys.argv[5:9]
queue_prior, queue_qwen_lfm, queue_qwen_gpt2 = sys.argv[9:12]

queued = []

def add_topneg(tag: str, category: str, enabled: bool) -> None:
    if not enabled:
        return
    queued.append(
        {
            "label": tag,
            "source_label": tag,
            "category": category,
            "model_dir": str(run_root / tag),
            "report": str(eval_dir / f"eval_{tag}_exactgen_20260702_1000x1000x250.json"),
            "manual_json": str(eval_dir / f"manual_{tag}_exactgen_20260702.json"),
            "retention_dir": str(eval_dir / f"retention_{tag}_20260702"),
            "log": str(run_root / "logs" / f"{tag}.queue.log"),
        }
    )

def add_beststack(tag: str, category: str, enabled: bool) -> None:
    if not enabled:
        return
    queued.append(
        {
            "label": tag,
            "source_label": tag,
            "category": category,
            "model_dir": str(run_root / tag),
            "report": str(eval_dir / f"eval_{tag}_strict_20260706_1000x1000x250.json"),
            "manual_json": str(eval_dir / f"manual_{tag}_strict_20260706.json"),
            "retention_dir": str(eval_dir / f"retention_{tag}_strict_20260706"),
            "log": str(run_root / "logs" / f"{tag}_beststack_retention.queue.log"),
        }
    )

prior_enabled = queue_prior == "1"
add_topneg(dual_tag, "restored_anchor_topneg", prior_enabled)
add_topneg(ffn_tag, "restored_anchor_topneg", prior_enabled)
add_beststack(qwen_lfm_tag, "cross_family_multidonor_beststack", queue_qwen_lfm == "1")
add_beststack(qwen_gpt2_tag, "cross_family_multidonor_beststack", queue_qwen_gpt2 == "1")

state = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "repo_root": str(repo_root),
    "run_root": str(run_root),
    "eval_dir": str(eval_dir),
    "matrix": "combined_anchor_beststack_20260707",
    "notes": [
        "Restored high-signal prior anchors are evaluated through current exactgen/top-negative recovery.",
        "Cross-family Qwen+LFM and Qwen+GPT-2 beststack lanes use current OT/activation/ALM defaults.",
        "Reports are written to deterministic paths so retest_matrix can summarize without re-running child evals.",
    ],
    "queued": queued,
}
state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"queue_state": str(state_path), "queued": len(queued)}, indent=2))
PY

if [[ "$SUMMARIZE_NOW" = "1" ]]; then
  python "$REPO_ROOT/experiments/retest_matrix.py" summarize \
    --queue-state "$MATRIX_STATE" \
    --out-json "$MATRIX_SUMMARY_JSON" \
    --out-md "$MATRIX_SUMMARY_MD"
fi
