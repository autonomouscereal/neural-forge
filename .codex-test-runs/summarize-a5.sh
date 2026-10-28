set -euo pipefail

result=/home/cereal/model-surgery-lab/artifacts/neural_forge_functional_output_refit_20260730/a5-fixed-a4-support/result.json
if [ ! -f "$result" ]; then
  printf 'missing=%s\n' "$result"
  exit 0
fi

python3 - "$result" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
print("schema=", result.get("schema"))
for candidate in result.get("candidates", []):
    cached = candidate.get("cached_hidden", {})
    first = candidate.get("first_token", {})
    print(json.dumps({
        "functional_weight": candidate.get("functional_weight"),
        "coefficient_sha256": candidate.get("coefficient_sha256"),
        "file": candidate.get("coefficient_file"),
        "all": cached.get("all"),
        "fit": cached.get("fit"),
        "holdout": cached.get("holdout"),
        "parent_direct": cached.get("parent_direct"),
        "parent_multi": cached.get("parent_multi"),
        "first_token": first,
    }, sort_keys=True))
PY
