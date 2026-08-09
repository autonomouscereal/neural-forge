"""Export the verbatim decoded text of both re-score lanes into one compact file.

Every string here is copied out of the sealed lane dossiers; nothing is retyped.
The export is what `_build_review.py` quotes from, so the review's code blocks and
the dossiers cannot drift apart.
"""

import collections
import hashlib
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))

LANES = {
    "a9-k32-residual-omp": (
        "a9-frozen-v5-rescore-dossier.json",
        "5dbdac7ade6abb680dc57626f61e4c06bb16e081025751079a7ecae0029eb3f6",
    ),
    "a11-k32-split-cout": (
        "a11-frozen-v5-rescore-dossier.json",
        "4d3c053b94f5febfe2942f92ca8a66992935f2aee716144748c7781ab6e2ab3f",
    ),
}


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


export = {"lanes": {}}
for lane_id, (filename, expected_sha) in LANES.items():
    path = os.path.join(BASE, filename)
    with open(path, "rb") as handle:
        raw = handle.read()
    actual = hashlib.sha256(raw).hexdigest()
    assert actual == expected_sha, (lane_id, actual)
    dossier = json.loads(raw.decode("utf-8"))
    rows = []
    for row in dossier["prompt_results"]:
        attempts = []
        for attempt in row["attempts"]:
            response = attempt["response"] or {}
            attempts.append(
                {
                    "cap": attempt["maximum_new_tokens"],
                    "status": attempt["status"],
                    "text": response.get("response"),
                    "termination": attempt["termination_evidence"],
                    "diagnostics": response.get("diagnostics"),
                    "target_roundtrip_exact": response.get("target_roundtrip_exact"),
                    "parent_response": response.get("parent_response"),
                    "attempt_sha256": attempt["attempt_sha256"],
                }
            )
        authoritative = attempts[-1]
        legacy = row.get("legacy_256_cap_comparison") or {}
        legacy_child = legacy.get("child_semantic_step_response") or {}
        rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "attempts": attempts,
                "authoritative_cap": authoritative["cap"],
                "authoritative_text": authoritative["text"],
                "authoritative_termination": authoritative["termination"],
                "statistics": describe(authoritative["text"] or ""),
                "legacy_256_text": legacy_child.get("response"),
                "legacy_256_was_censored": row["legacy_256_cap_was_censored"],
                "initial_1024_text": attempts[0]["text"],
            }
        )
    export["lanes"][lane_id] = {
        "lane_label": dossier["lane_label"],
        "counted_compaction": dossier["counted_compaction"],
        "body_contraction": dossier["body_contraction"],
        "result_sha256": dossier["result_sha256"],
        "dossier_file_sha256": expected_sha,
        "child_state_sha256": dossier["child_state_sha256"],
        "routing_contract_sha256": dossier["routing_contract_sha256"],
        "codebook_artifact_sha256": dossier["codebook_artifact_sha256"],
        "physical_child_parameter_count": dossier["physical_child_parameter_count"],
        "legacy_review_tally": dossier["legacy_review_tally"],
        "frozen_protocol_binding": dossier["frozen_protocol_binding"],
        "runtime_dtype_receipt": dossier["runtime_dtype_receipt"],
        "operation_audit": dossier["operation_audit"],
        "rows": rows,
    }

out = os.path.join(BASE, "frozen-v5-responses.json")
with open(out, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(export, handle, indent=1, ensure_ascii=False, sort_keys=True)
    handle.write("\n")
print("wrote", out)
for lane_id, lane in export["lanes"].items():
    print(lane_id, len(lane["rows"]), "rows")
