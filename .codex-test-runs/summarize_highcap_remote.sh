set -euo pipefail

base=/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730
launch="$base/_launch/a9-a11-highcap-3090-r2"
pid=$(cat "$launch/job.pid")
if kill -0 "$pid" 2>/dev/null; then
  ps -p "$pid" -o pid,etime,%cpu,%mem,rss,stat,args
else
  echo "PROCESS_NOT_RUNNING pid=$pid"
fi

/home/cereal/model-surgery-lab/venv/bin/python - <<'PY'
import json
from pathlib import Path

base = Path("/home/cereal/model-surgery-lab/artifacts/neural_forge_sparse_semantic_highcap_20260730")
for path in sorted(base.glob("*/attempts/*-cap*.json")):
    value = json.loads(path.read_text())
    response = value.get("response") or {}
    diagnostics = response.get("diagnostics") or {}
    text = str(response.get("response", "")).replace("\n", "\\n")
    print(json.dumps({
        "lane": value.get("lane_id"),
        "prompt": value.get("prompt_id"),
        "cap": value.get("maximum_new_tokens"),
        "completion": value.get("termination_evidence", {}).get("completion_status"),
        "eos": value.get("termination_evidence", {}).get("ended_with_eos"),
        "censored": value.get("termination_evidence", {}).get("censored_by_output_cap"),
        "donor_tokens": diagnostics.get("donor_response_tokens"),
        "target_tokens": diagnostics.get("final_target_response_tokens"),
        "response_preview": text[:320],
        "attempt_sha256": value.get("attempt_sha256"),
    }, sort_keys=True))
for path in sorted(base.glob("*/high-cap-dossier.json")):
    value = json.loads(path.read_text())
    print(json.dumps({
        "lane_complete": value.get("lane_id"),
        "prompt_count": value.get("prompt_count"),
        "result_sha256": value.get("result_sha256"),
    }, sort_keys=True))
PY

echo LOG_TAIL
tail -16 "$launch/job.log"
echo GPU4
nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | sed -n '5p'
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | grep 'GPU-2ce6a95c-7b1f-644a-3808-d781732a0582' || true
