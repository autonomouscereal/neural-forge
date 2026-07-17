#!/usr/bin/env bash
set -euo pipefail

manifest=/home/cereal/model-surgery-lab/artifacts/lfm2_global_causal_20260717/cached-causal35-sharded-v1/canonical-proposals/ffn-l00-w1792-3a2b558c30a4d321/manifest.json
/home/cereal/model-surgery-lab/venv/bin/python - "$manifest" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="ascii") as handle:
    manifest = json.load(handle)
print(
    json.dumps(
        {
            "option_id": manifest["proposal"]["option_id"],
            "requires_grad": manifest["layer_plan"]["requires_grad"],
            "tensor_records": manifest["tensor_records"],
        },
        sort_keys=True,
    )
)
PY
