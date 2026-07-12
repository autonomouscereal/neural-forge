#!/usr/bin/env bash
set -euo pipefail

base=/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711
for run in \
  p35_mixed_drop16_pareto_v3_20260711 \
  p35_mixed_drop16_pareto_v4_20260711 \
  p35_mixed_drop16_pareto_v5_20260711 \
  p35_mixed_drop16_pareto_v6_20260711
do
  echo "RUN:${run}"
  find "${base}/${run}" -maxdepth 3 -type f -printf '%P %s\n' \
    | sort \
    | tail -80
  echo "LOGTAIL"
  tail -120 "${base}/${run}/launcher.log" 2>/dev/null || true
  tail -120 "${base}/${run}/stderr.log" 2>/dev/null || true
done

python3 - <<'PY'
import json
from pathlib import Path

base = Path("/mnt/ai_data/neural-forge/qwen3_4b_unified_20260711")
for run in (
    "p35_mixed_drop16_pareto_v3_20260711",
    "p35_mixed_drop16_pareto_v4_20260711",
    "p35_mixed_drop16_pareto_v5_20260711",
    "p35_mixed_drop16_pareto_v6_20260711",
):
    report = json.loads((base / run / "run_report.json").read_text())
    print("REPORT", run)
    for key in (
        "status",
        "stop_reason",
        "failure",
        "error",
        "accepted_transactions",
        "attempted_transactions",
        "parent_state_sha256",
        "model_content_sha256",
        "proposal_rounds",
    ):
        if key in report:
            value = report[key]
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)[:4000]
            print(key, value)
    print("top_keys", sorted(report))
    print("sensitivity_fingerprints", json.dumps(report.get("sensitivity", {}).get("fingerprints"), sort_keys=True))
    print("sensitivity_report", json.dumps(report.get("sensitivity", {}).get("report"), sort_keys=True)[:4000])
    def walk(value, path=()):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from walk(item, (*path, str(key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from walk(item, (*path, str(index)))
        elif isinstance(value, str) and any(
            token in value
            for token in (
                "181418841018f67a",
                "2809e0279aad8416",
                "b3e960b2d018e156",
                "04f9b1f253a9639a",
            )
        ):
            print("MATCH", ".".join(path), value[:1000])
    for _ in walk(report):
        pass

round_dir = base / "p35_mixed_drop16_pareto_v3_20260711/candidate_validation/round-000001"
for path in sorted(round_dir.glob("candidate-*.json")):
    payload = json.loads(path.read_text())
    print("candidate_top_keys", path.name, sorted(payload))
    candidate = payload.get("candidate", payload.get("proposal", {}).get("candidate", {}))
    audit = payload.get("audit", payload.get("identity_binding", payload))
    if path.name.startswith("candidate-0001-"):
        print("identity_binding", json.dumps(payload.get("identity_binding"), sort_keys=True)[:8000])
        print("model", json.dumps(payload.get("model"), sort_keys=True)[:8000])
    print(
        "V3CANDIDATE",
        audit.get("kind"),
        audit.get("candidate_sha256"),
        candidate.get("left_prototype"),
        candidate.get("right_prototype"),
        candidate.get("resulting_medoid"),
        candidate.get("score", {}).get("weighted_damage"),
    )
PY
