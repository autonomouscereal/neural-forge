set -euo pipefail

RUNTIME=/home/cereal/model-surgery-lab/deployments/neural-forge-bccb5b78
PYTHON=/home/cereal/model-surgery-lab/venv/bin/python

test "$(cat "$RUNTIME/SOURCE_COMMIT")" = \
  bccb5b7899d8f01f449b205fdf5a316ab9e5e7b8
test "$(sha256sum "$RUNTIME/src/neural_forge/hf_lfm2_cached_global_topology_frontier.py" | cut -d' ' -f1)" = \
  338d5b800bfac45dc8cee0017a98a850392610b34322db4cd55a0d8a0de73b12

PYTHONPATH="$RUNTIME/src:$RUNTIME" "$PYTHON" - <<'PY'
import json
import sys

from neural_forge.hf_lfm2_cached_global_topology_frontier import (
    _global_algorithm_fingerprint,
)
from neural_forge.hf_lfm2_global_topology_phase_compatibility import (
    explicit_global_topology_phase_compatibility_receipt,
)

historical = "3e4531740792b3abe7ed54c02a49ca99caabc3ae94dd181c167913cbb5dbcab1"
global_source = "338d5b800bfac45dc8cee0017a98a850392610b34322db4cd55a0d8a0de73b12"
functional_source = "3ca319c165d81c7fffb2f57a55edf3fcab49c647dd73cefb5a9b8d955a4741a7"
expected_current = "d2ed9e076927dea1bfa44e47f3a2fff19b9617f486ca079e27948d212b6212dd"
expected_receipt = "95dba15ce807ec91aa807b601a918344f36dae3c4ef6e93d4f5a45c773674fe0"

current = _global_algorithm_fingerprint()
assert current == expected_current, (current, expected_current)
receipt = explicit_global_topology_phase_compatibility_receipt(
    historical_algorithm_sha256=historical,
    current_global_frontier_source_sha256=global_source,
    current_functional_width_source_sha256=functional_source,
    current_algorithm_sha256=current,
)
assert receipt is not None
assert receipt["compatibility_receipt_sha256"] == expected_receipt
print(
    json.dumps(
        {
            "python": sys.version.split()[0],
            "current_algorithm_sha256": current,
            "compatibility_receipt_sha256": receipt[
                "compatibility_receipt_sha256"
            ],
            "policy_sha256": receipt["policy_sha256"],
        },
        sort_keys=True,
    )
)
PY
