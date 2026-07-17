set -euo pipefail
RUN_ROOT=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-rtx3090-gpu4-v1
SRC=/home/cereal/model-surgery-lab/neural-forge-src
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python
printf 'run_root_exists=%s\n' "$(test -d "$RUN_ROOT" && printf yes || printf no)"
if test -f "$RUN_ROOT/run.pid"; then
  PID=$(cat "$RUN_ROOT/run.pid")
  printf 'pid=%s\n' "$PID"
  if kill -0 "$PID" 2>/dev/null; then
    printf 'pid_live=yes\n'
    ps -p "$PID" -o pid=,etime=,stat=,%cpu=,%mem=,cmd=
  else
    printf 'pid_live=no\n'
  fi
else
  printf 'pid_file=missing\n'
fi
printf 'candidate_records=%s\n' "$(find "$RUN_ROOT/work/candidate-records" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
printf 'candidate_cache_dirs=%s\n' "$(find "$RUN_ROOT/work/candidate-caches" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
printf 'frontier_exists=%s\n' "$(test -f "$RUN_ROOT/frontier.json" && printf yes || printf no)"
printf 'producer_manifest_exists=%s\n' "$(test -f "$RUN_ROOT/work/producer-run.json" && printf yes || printf no)"
if test -f "$RUN_ROOT/run.log"; then
  printf 'log_bytes=%s\n' "$(stat -c %s "$RUN_ROOT/run.log")"
  tail -n 30 "$RUN_ROOT/run.log"
fi
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name --format=csv,noheader
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader
if test -f "$RUN_ROOT/frontier.json"; then
  cd "$SRC"
  export PYTHONPATH="$SRC/src:$SRC"
  "$PYTHON" - "$RUN_ROOT/frontier.json" <<'PY'
import json
import sys

from neural_forge.hf_lfm2_cached_causal_evidence import (
    verify_lfm2_cached_causal_frontier_document,
)

path = sys.argv[1]
document = json.load(open(path, encoding="ascii"))
verify_lfm2_cached_causal_frontier_document(document)
print(
    json.dumps(
        {
            "frontier_verified": True,
            "frontier_sha256": document["frontier_sha256"],
            "target35": document["targets"]["0.35"],
            "target50": document["targets"]["0.50"],
            "candidate_count": len(document["candidates"]),
        },
        sort_keys=True,
    )
)
PY
fi
