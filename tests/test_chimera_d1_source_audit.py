"""Independent source isolation/counting fixtures; no model or fake real corpus."""
import copy
import json
from pathlib import Path

import pytest
from experiments.chimera_recovery import d1_source_audit as A


def files(root):
    def write(name, value, lines=False):
        path = root / name
        path.write_bytes(b"".join(A.canonical(v) + b"\n" for v in value) if lines else A.canonical(value))
        return path
    docs = [dict(uid="u1", rid="r1", doc=0, start=0, end=5, kept=5), dict(uid="u2", rid="r2", doc=1, start=5, end=12, kept=7)]
    docfile = write("documents.jsonl", docs, True)
    text = write("text.json", dict(documents_file=dict(sha256=A.sha(docfile)), sequence_length=4))
    records = [dict(id="u1", split="test", route="text_arithmetic", quality=dict(train_ready=False, blockers=["teacher_rewrite_exposes_current_answer"]),
                    provenance=dict(registry_origin=dict(rid="r1"), recovery_group_id="g1")),
               dict(id="u2", split="train", route="text_code", quality=dict(train_ready=True, blockers=[]),
                    provenance=dict(registry_origin=dict(rid="r2"), recovery_group_id="g2"))]
    # Same content UID in another RID is legal in the full quarantined registry.
    records.append({**copy.deepcopy(records[0]), "provenance": dict(registry_origin=dict(rid="r3"), recovery_group_id="g1")})
    normalized = write("normalized.jsonl", records, True)
    carve = write("carve.json", dict(assignment=dict(r1="train", r2="train", r3="duplicate")))
    excluded = write("excluded.jsonl", [dict(id="u1", rid="r1", blockers=["frozen_source_group"])], True)
    return docfile, text, normalized, carve, excluded


def test_new_split_and_quarantine_risks_overlap_but_union_tokens_count_once(tmp_path):
    paths = files(tmp_path)
    report = A.audit(*paths, tmp_path / "out", total_rows=3, fit_rows=2, width=4)
    assert report["current_incompatible_fit_tokens"] == 5
    assert report["fit_tokens_by_risk"]["current_recovery_test"] == 5
    assert report["fit_tokens_by_risk"]["global_cross_route_exclusion"] == 5
    assert report["original_frozen_carve_violation_documents"] == 0
    assert report["fit_diagnostic_shared_connected_groups"] == report["boundary_crossing_documents"] == 1
    assert report["original_artifacts_changed"] is False


def test_actual_span_ledger_hash_and_source_uid_rid_mismatch_refused(tmp_path):
    paths = files(tmp_path)
    docs = paths[0]
    original = docs.read_bytes()
    docs.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="sealed text manifest"):
        A.audit(*paths, tmp_path / "bad-hash", total_rows=3, fit_rows=2, width=4)
    docs.write_bytes(original)
    rows = list(A.read_rows(paths[2]))
    rows[0]["id"] = "another-source-uid"
    paths[2].write_bytes(b"".join(A.canonical(v) + b"\n" for v in rows))
    with pytest.raises(ValueError, match="UID/RID"):
        A.audit(*paths, tmp_path / "bad-source", total_rows=3, fit_rows=2, width=4)


def test_missing_tail_cannot_claim_exact_d1_token_budget(tmp_path):
    paths = files(tmp_path)
    with pytest.raises(ValueError, match="exact original prefix"):
        A.audit(*paths, tmp_path / "bad-tail", total_rows=4, fit_rows=2, width=4)
