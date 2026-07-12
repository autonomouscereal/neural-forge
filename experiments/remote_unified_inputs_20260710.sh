#!/usr/bin/env bash
set -euo pipefail

base=/mnt/ai_data/neural-forge/qwen3_4b_compensated_20260710
p35=/mnt/ai_data/neural-forge/qwen3_4b_codebook_20260710/qwen3_4b_codebook_pn36_pa35_gpu4_r1_20260710/compact_child

printf '%s\n' LAYER35_RESULT
python3 - "$base/qwen3_4b_pair_drop1_layer35_dev_reused_eval_20260710" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in (
    "certification_verdict.json",
    "parent_final_comparison.json",
    "transaction_report.json",
    "selection_certification_split.json",
    "three_split_provenance.json",
):
    path = root / name
    if not path.is_file():
        continue
    payload = json.load(open(path, encoding="utf-8"))
    if name == "certification_verdict.json":
        print(name, json.dumps(payload, sort_keys=True))
    elif name == "parent_final_comparison.json":
        print(name, json.dumps(payload.get("summary", payload), sort_keys=True)[:12000])
    elif name == "transaction_report.json":
        accepted = payload.get("accepted_transactions", [])
        print(name, json.dumps({
            "accepted_count": payload.get("accepted_count"),
            "rejected_count": payload.get("rejected_count"),
            "stop_reason": payload.get("stop_reason"),
            "accepted_transactions": accepted,
        }, sort_keys=True)[:20000])
    else:
        print(name, json.dumps(payload, sort_keys=True)[:12000])
PY

printf '%s\n' P35_MANIFEST
python3 - "$p35" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ("manifest.json", "parameter_report.json", "tokenizer_contract.json"):
    path = root / name
    if path.is_file():
        payload = json.load(open(path, encoding="utf-8"))
        print(name, json.dumps(payload, sort_keys=True)[:20000])
PY

printf '%s\n' CHECKOUTS
for directory in \
  /home/cereal/model-surgery-lab/neural-forge-pivot \
  /home/cereal/model-surgery-lab/neural-forge-src; do
  if [[ -d "$directory" ]]; then
    printf 'directory=%s\n' "$directory"
    find "$directory" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort | head -80
  fi
done

printf '%s\n' SELECTION_SUITES
find /mnt/ai_data/neural-forge /home/cereal/model-surgery-lab/neural-forge-src/runs \
  -type f \( -iname '*selection*suite*.json' -o -iname '*adaptive*suite*.json' \) \
  -printf '%p\n' 2>/dev/null | sort | tail -80
