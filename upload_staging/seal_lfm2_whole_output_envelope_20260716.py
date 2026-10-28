from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path("/home/cereal/model-surgery-lab/neural-forge-src")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from neural_forge.dynamic_lfm2_behavior import (  # noqa: E402
    load_lfm2_cached_whole_output_baseline,
    seal_lfm2_cached_whole_output_envelope,
)
from run_lfm2_dynamic_catalog import fingerprint_local_tokenizer  # noqa: E402


CACHE_KEY = "4164681e417f9dbab9ab92c2dc073649600832f764544530a724663303cbe3f1"
CACHE_ROOT = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_residual_attention_fold_20260716/circuit-cache/parent"
)
SOURCE = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_residual_attention_fold_20260716/heldout_parent_responses.json"
)
DESTINATION = Path(
    "/home/cereal/model-surgery-lab/artifacts/"
    "lfm2_residual_attention_fold_20260716/"
    "heldout_parent_responses.envelope-v1-strict.json"
)
TOKENIZER = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads((CACHE_ROOT / CACHE_KEY / "manifest.json").read_text())
parent_state = manifest["parent"]["identity_sha256"]
tokenizer_fingerprint = fingerprint_local_tokenizer(TOKENIZER)
if DESTINATION.exists():
    sealed = {
        "path": str(DESTINATION),
        "file_sha256": sha256(DESTINATION),
        "created": False,
    }
else:
    sealed = seal_lfm2_cached_whole_output_envelope(
        SOURCE,
        DESTINATION,
        parent_cache_key=CACHE_KEY,
        parent_state_fingerprint=parent_state,
        tokenizer_fingerprint=tokenizer_fingerprint,
        generation_limit=64,
    )
    sealed["created"] = True
baseline = load_lfm2_cached_whole_output_baseline(
    DESTINATION,
    expected_sha256=sealed["file_sha256"],
    expected_parent_cache_key=CACHE_KEY,
    expected_parent_state_fingerprint=parent_state,
    expected_tokenizer_fingerprint=tokenizer_fingerprint,
    expected_generation_limit=64,
    fragment_minimum_characters=12,
)
report = {
    **sealed,
    "parent_cache_key": CACHE_KEY,
    "parent_state_fingerprint": parent_state,
    "tokenizer_fingerprint": tokenizer_fingerprint,
    "case_count": len(baseline.cases),
    "semantic_contract_count": sum(
        contract is not None for _case_id, contract in baseline.contracts
    ),
    "source_rows_sha256": sha256(SOURCE),
    "parent_forward_passes": 0,
}
print(json.dumps(report, sort_keys=True))
