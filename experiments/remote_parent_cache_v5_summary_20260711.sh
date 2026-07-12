#!/usr/bin/env bash
set -euo pipefail

base=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710
tag=qwen3_4b_parent_caches_fragile_v5_20260710
root="$base/$tag"

python3 - "$root" "$base" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
base = pathlib.Path(sys.argv[2])
result = {
    "root_exists": root.is_dir(),
    "files": sorted(path.name for path in root.glob("*") if path.is_file()),
}
for name in ("run_state.json", "cache_report.json", "parent_cache_report.json"):
    path = root / name
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[name] = payload
for name in (
    "qwen3_4b_base_certification_v5.json",
    "qwen3_4b_base_selection_fragile_v5.json",
):
    path = base / "parent_caches" / name
    entry = {"exists": path.is_file()}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry.update({
            "bytes": path.stat().st_size,
            "top_keys": sorted(payload),
            "model": payload.get("model"),
            "param_count": payload.get("param_count"),
            "eval_suite": payload.get("eval_suite"),
            "choice_accuracy": {
                key: (payload.get("choice_accuracy") or {}).get(key)
                for key in ("correct", "total", "accuracy")
            },
            "generation_health": {
                key: (payload.get("generation_health") or {}).get(key)
                for key in ("correct", "total", "pass_rate")
            },
            "behavior_suite": {
                key: (payload.get("behavior_suite") or {}).get(key)
                for key in ("correct", "total", "pass_rate")
            },
            "identity_binding": payload.get("identity_binding"),
        })
    result[name] = entry
print(json.dumps(result, sort_keys=True)[:50000])
PY

nvidia-smi \
  --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
