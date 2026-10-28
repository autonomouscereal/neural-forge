set -euo pipefail

runtime=/home/cereal/model-surgery-lab/runtime/neural-forge-wip-local-action-v2
python=/home/cereal/model-surgery-lab/venv-pascal-cu118-torch260/bin/python
codebook=/home/cereal/model-surgery-lab/artifacts/neural_forge_phase_cache_v1/lfm2-sparse-semantic-codebook/5b5557b8da7aa125ada8cac305005de7b9e157933c3d4878f6be133fb5c51cc7/payload/artifact

PYTHONPATH="$runtime/src" "$python" - "$codebook" <<'PY'
import json
import sys
from pathlib import Path

from neural_forge.hf_lfm2_sparse_semantic_codebook import load_lfm2_sparse_semantic_codebook

artifact = load_lfm2_sparse_semantic_codebook(
    Path(sys.argv[1]),
    expected_artifact_sha256="4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d",
)
receipt = artifact.as_dict()
print(json.dumps(receipt, indent=2, sort_keys=True))
PY
