"""Export the SEALED A9 child's verbatim text at the same 256-token cap.

The dossier under review was produced at `maximum_new_tokens: 256` -- the sealed
A9 launch parameter -- while the published A9 quality band (8/15) was measured by
`reports/frontier-rescore-20260808/` at 1,024 with a 2,048 retry, where 14 of 16
A9 rows completed by EOS instead of 8 of 16.  Grading one child on complete
responses and the other on 256-token prefixes would measure the cap, not the
mechanism, so the sealed A9 dossier is re-graded here at its own 256 cap as an
evidence-matched control.

Source (sealed, read-only): `/home/cereal/model-surgery-lab/artifacts/
neural_forge_sparse_semantic_bridge_20260730/a9-k32-residual-omp/
whole-response-dossier.json`.  Its older schema carries termination inside
`child_semantic_step_response.diagnostics` rather than a `termination_evidence`
block; the two fields are normalized to the same shape here and nothing else is
touched.
"""

import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from _extract_responses import describe  # noqa: E402

DOSSIER_FILE_SHA256 = (
    "d04ef2de7dc9c825d069affa3f8083769dd1762f5aaa97362c7c85a79eb635ff"
)
DOSSIER_SELF_SHA256 = (
    "30486c355f3d6ef87456ba337b0a82b1f2bc1bb13008cdd9d1381506d46c6b6e"
)
NORMALIZED_PROMPT_SET_SHA256 = (
    "5615b9e4cee0727a9c35c0da08ac770541941d42077e9c50ac1d2a03fea9d741"
)
SEALED_CODEBOOK_SHA256 = (
    "4439a2b7c5629a0019cd18fcacb020ee133482ab9752df64fd38bb612787ed0d"
)


def main(source):
    with open(source, "rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == DOSSIER_FILE_SHA256, actual
    dossier = json.loads(raw.decode("utf-8"))
    assert dossier["dossier_sha256"] == DOSSIER_SELF_SHA256
    assert dossier["prompt_set_sha256"] == NORMALIZED_PROMPT_SET_SHA256
    assert dossier["sparse_semantic_codebook_artifact_sha256"] == SEALED_CODEBOOK_SHA256
    assert dossier["maximum_new_tokens"] == 256
    assert dossier["prompt_count"] == 16

    rows = []
    for entry in dossier["comparisons"]:
        text = entry["authoritative_child_target_text"]
        diagnostics = entry["child_semantic_step_response"]["diagnostics"]
        censored = bool(diagnostics["generation_limit_reached"]) and not bool(
            diagnostics["ended_with_eos"]
        )
        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "prompt": entry["prompt"],
                "child_text": text,
                "parent_cached_response": entry.get("parent_cached_response"),
                "termination_evidence": {
                    "completion_status": "censored" if censored else "complete",
                    "ended_with_eos": bool(diagnostics["ended_with_eos"]),
                    "censored_by_output_cap": censored,
                    "generation_limit_reached": bool(
                        diagnostics["generation_limit_reached"]
                    ),
                    "maximum_new_tokens": int(diagnostics["maximum_new_tokens"]),
                },
                "observation": describe(text),
            }
        )
    export = {
        "schema": "neural-forge-matrix-a9-256-control-export-v1",
        "source_dossier_file_sha256": DOSSIER_FILE_SHA256,
        "source_dossier_self_sha256": DOSSIER_SELF_SHA256,
        "prompt_set_sha256": dossier["prompt_set_sha256"],
        "maximum_new_tokens": dossier["maximum_new_tokens"],
        "child_state_sha256": dossier["child_state_sha256"],
        "routing_contract_sha256": dossier["routing_contract_sha256"],
        "sparse_semantic_codebook_artifact_sha256": SEALED_CODEBOOK_SHA256,
        "rows": rows,
    }
    out = os.path.join(BASE, "a9-256-control-responses.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=1, ensure_ascii=False, sort_keys=True)
    with open(out, "rb") as handle:
        print(
            "a9-256-control-responses.json",
            hashlib.sha256(handle.read()).hexdigest(),
        )


if __name__ == "__main__":
    main(sys.argv[1])
