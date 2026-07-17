#!/usr/bin/env bash
set -euo pipefail

record_root=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-rtx3090-gpu4-v1/work/candidate-records
python_bin=/home/cereal/model-surgery-lab/venv/bin/python

"$python_bin" - "$record_root" <<'PY'
import json
import pathlib
import sys

weights = {
    "final_logit_margin": 0.25,
    "eos": 0.15,
    "repetition": 0.15,
    "task_anchor": 0.25,
    "residual_path": 0.10,
    "attention_path": 0.10,
}
rows = []
for path in pathlib.Path(sys.argv[1]).glob("*.json"):
    document = json.loads(path.read_text(encoding="ascii"))
    candidate = document["candidate"]
    damage = candidate["damage"]
    score = sum(damage[name] * weight for name, weight in weights.items())
    rows.append(
        {
            "option_id": candidate["option_id"],
            "layer": candidate["layer_index"],
            "width": candidate["target_units"],
            "saved": candidate["parameters_saved"],
            "score": round(score, 9),
            "task": round(damage["task_anchor"], 6),
            "residual": round(damage["residual_path"], 6),
            "attention": round(damage["attention_path"], 6),
            "margin": round(damage["final_logit_margin"], 6),
        }
    )
rows.sort(key=lambda row: (row["score"], row["option_id"]))
print(json.dumps({"count": len(rows), "best": rows[:10]}, sort_keys=True))
PY
