set -euo pipefail
LAB=/home/cereal/model-surgery-lab
SRC="$LAB/neural-forge-src"
PYTHON="$LAB/venv/bin/python"
RUN_ROOT="$LAB/artifacts/lfm2_global_causal_20260717/cached-causal35-rtx3090-gpu4-v1"

export PYTHONPATH="$SRC/src:$SRC"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SRC"
"$PYTHON" - "$RUN_ROOT/work/candidate-caches" <<'PY'
import json
import pathlib
import sys

from neural_forge.hf_lfm2_cached_causal_evidence import (
    verify_lfm2_full_prefix_child_trace_cache,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    open_lfm2_circuit_trace_cache,
)

root = pathlib.Path(sys.argv[1])
directories = sorted(path for path in root.iterdir() if path.is_dir())
assert len(directories) == 1, directories
option_root = directories[0]
manifests = sorted(option_root.glob("*/manifest.json"))
assert len(manifests) == 1, manifests
path = manifests[0].parent
cache = open_lfm2_circuit_trace_cache(path, verify=True)
verify_lfm2_full_prefix_child_trace_cache(cache)
manifest = cache.manifest
completion = json.loads((path / "complete.json").read_text(encoding="ascii"))
print(json.dumps({
    "candidate_option_id": option_root.name,
    "cache_verified": True,
    "cache_key": cache.cache_key,
    "captured_model_identity": cache.parent_fingerprint,
    "corpus_fingerprint": cache.corpus_fingerprint,
    "batch_count": len(cache),
    "capture": manifest["capture"],
    "parent": manifest["parent"],
    "manifest_sha256": manifest["manifest_sha256"],
    "completion_sha256": completion["completion_sha256"],
}, sort_keys=True))
PY
