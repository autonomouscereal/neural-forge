"""Export the recompiled 0b child's verbatim decoded text into one compact file.

Same instrument as `reports/frontier-rescore-20260808/_extract_responses.py`, so
this review and the sealed A9 re-score it is compared against are graded off
identically-shaped evidence.  Every string here is copied out of the sealed
route dossier under a digest assertion; nothing is retyped.

Source: `/home/cereal/model-surgery-lab/artifacts/
         neural_forge_matrix_campaign_20260810/CAL.C01.T5.X4-route-mxc2/
         whole-response-dossier.json`, fetched read-only.
"""

import collections
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

DOSSIER_FILE_SHA256 = (
    "223bfa47c658834c19edc2dbec07d6f8021fa3663bac4506a5cb31789deecae2"
)
DOSSIER_SELF_SHA256 = (
    "b3e205660d2afb7e56778de5477022d43a2eb082037ccc92cfe5be4e57c5e13e"
)
NORMALIZED_PROMPT_SET_SHA256 = (
    "5615b9e4cee0727a9c35c0da08ac770541941d42077e9c50ac1d2a03fea9d741"
)


def describe(text):
    """Navigation-aid statistics only; no decision authority (protocol s.Per-response)."""
    if not text:
        return {"characters": 0, "empty": True}
    words = text.split()
    longest_tok_run, run, tok = 1, 1, words[0] if words else ""
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run > longest_tok_run:
            longest_tok_run, tok = run, words[i]
    longest_char_run, run, ch = 1, 1, text[0]
    for i in range(1, len(text)):
        run = run + 1 if text[i] == text[i - 1] else 1
        if run > longest_char_run:
            longest_char_run, ch = run, text[i]
    return {
        "characters": len(text),
        "empty": False,
        "tokens": len(words),
        "distinct_tokens": len(set(words)),
        "non_ascii": sum(1 for c in text if ord(c) > 127),
        "longest_token_run": [longest_tok_run, tok],
        "longest_char_run": [longest_char_run, ch],
        "most_common": collections.Counter(words).most_common(3),
    }


def main(source):
    with open(source, "rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == DOSSIER_FILE_SHA256, actual
    dossier = json.loads(raw.decode("utf-8"))
    assert dossier["dossier_sha256"] == DOSSIER_SELF_SHA256
    assert dossier["prompt_set_sha256"] == NORMALIZED_PROMPT_SET_SHA256
    assert dossier["prompt_count"] == 16

    rows = []
    for entry in dossier["comparisons"]:
        text = entry["authoritative_child_target_text"]
        rows.append(
            {
                "id": entry["id"],
                "category": entry["category"],
                "prompt": entry["prompt"],
                "child_text": text,
                "child_donor_text": entry.get("diagnostic_child_donor_text"),
                "parent_cached_response": entry.get("parent_cached_response"),
                "termination_evidence": entry.get("termination_evidence"),
                "observation": describe(text),
            }
        )
    export = {
        "schema": "neural-forge-matrix-0b-mxc2-response-export-v1",
        "source_dossier_file_sha256": DOSSIER_FILE_SHA256,
        "source_dossier_self_sha256": DOSSIER_SELF_SHA256,
        "prompt_set_sha256": dossier["prompt_set_sha256"],
        "maximum_new_tokens": dossier["maximum_new_tokens"],
        "child_state_sha256": dossier["child_state_sha256"],
        "routing_contract_sha256": dossier["routing_contract_sha256"],
        "sparse_semantic_codebook_artifact_sha256": dossier[
            "sparse_semantic_codebook_artifact_sha256"
        ],
        "rows": rows,
    }
    out = os.path.join(BASE, "mxc2-responses.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(export, handle, indent=1, ensure_ascii=False, sort_keys=True)
    with open(out, "rb") as handle:
        print("mxc2-responses.json", hashlib.sha256(handle.read()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
