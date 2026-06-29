#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cereal/model-surgery-lab/neural-forge-src/runs/qwen4b_to_smollm135m_transport_random_20260627}"
PY="${PY:-/home/cereal/model-surgery-lab/venv/bin/python}"

TAGS=(
  qwen4b_blank_smollm_chain_mlpout_rank16_blockio_alm_gpu4_20260629
  qwen4b_blank_smollm_chain_all_rank8_subspace_alm_gpu6_20260629
  qwen4b_blank_smollm_chain_transport_only_blockio_alm_gpu0_20260629
  qwen4b_blank_smollm_chain_mlpout_true_lrc_rank16_blockio_alm_gpu4_20260629
  qwen4b_blank_smollm_chain_oldbest_noactfit_cka_alm_gpu0_20260629
)

for tag in "${TAGS[@]}"; do
  echo "==== $tag"
  pid_file="$ROOT/logs/$tag.pid"
  log_file="$ROOT/logs/$tag.log"
  summary="$ROOT/$tag/reports/chain_anchor_summary.json"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "running:$pid"
      ps -p "$pid" -o pid,etime,pcpu,pmem,rss --no-headers || true
    else
      echo "not_running:$pid"
    fi
  else
    echo "no_pidfile"
  fi
  echo "log_bytes:$([ -f "$log_file" ] && wc -c < "$log_file" || echo 0)"
  echo "files:$(find "$ROOT/$tag" -type f 2>/dev/null | wc -l)"
  find "$ROOT/$tag" -maxdepth 4 -type f 2>/dev/null | sed "s#$ROOT/$tag/##" | sort | tail -90 || true
  if [ -f "$summary" ]; then
    "$PY" - <<PY
import json, pathlib
r = json.loads(pathlib.Path("$summary").read_text())
print({"best_tag": r.get("best_tag"), "stage_count": r.get("stage_count"), "promoted": r.get("promoted_stage_count"), "best_metrics": r.get("best_metrics")})
for rec in r.get("stage_records", []):
    print({k: rec.get(k) for k in ["tag", "stage", "promoted", "passed", "metrics", "model_dir"]})
PY
  fi
  for rpt in anchor_eval 01_init_eval 02_block_eval 03_alm_eval; do
    f="$ROOT/$tag/reports/$rpt.json"
    if [ -f "$f" ]; then
      echo "-- $rpt"
      "$PY" - <<PY
import json, pathlib
r = json.loads(pathlib.Path("$f").read_text())
print({k: r.get(k) for k in ["behavior_score", "behavior_total", "generation_score", "generation_total", "choice_score", "choice_total", "ppl_mean"]})
print("cats", r.get("category_scores"))
PY
    fi
  done
  report="$ROOT/$tag/models/01_init/neural_forge_donor_projected_init_report.json"
  if [ -f "$report" ]; then
    echo "-- init_report"
    "$PY" - <<PY
import json, pathlib
r = json.loads(pathlib.Path("$report").read_text())
keys = [
  "activation_fit_attempt_count", "activation_fit_applied_count", "activation_fit_method",
  "activation_fit_rank", "activation_fit_rank_mode", "activation_fit_target", "activation_fit_alpha",
  "fold_mode", "donor_layer_fold_mode", "donor_layer_group_strategy",
]
print({k: r.get(k) for k in keys})
PY
  fi
done

echo "==== gpuutil"
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader | egrep '4ead|2ce6|f5a' || true
echo "==== gpuapps"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader | grep python || true
echo "==== mem"
free -h | sed -n '1,2p'
