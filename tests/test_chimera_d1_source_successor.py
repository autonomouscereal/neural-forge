"""Source receipt, complete-document budget and diagnostic isolation contracts."""
import copy
import json
from pathlib import Path

import pytest
from experiments.chimera_recovery import d1_source_successor as D


def fixture_source(tmp_path):
    rows = [dict(schema="recovery_corpus.v1", id="task-a", route="text_arithmetic", split="train", quality=dict(train_ready=True),
                 provenance=dict(split="train", recovery_group_id="family-a", source_verified=True)),
            dict(schema="recovery_corpus.v1", id="task-b", route="text_code", split="train", quality=dict(train_ready=True),
                 provenance=dict(split="train", recovery_group_id="family-b", source_verified=True))]
    records = tmp_path / "records.jsonl"
    records.write_bytes(b"".join(D.A.canonical(row) + b"\n" for row in rows))
    shard = tmp_path / "metadata-fixture.bin"
    shard.write_bytes(b"metadata contract fixture; not runtime tensor proof")
    receipts = [dict(row_id=row["id"], source_record_sha256=D.digest(row), template_sha256="1" * 64,
                     rendered_sha256="2" * 64, native_messages_sha256="3" * 64) for row in rows]
    value = dict(schema="chimera_recovery.assistant_supervision.v1", split="train", cross_split_groups=[],
                 input_records_sha256=D.digest(rows), records_sha256=D.digest(rows), row_receipts=receipts, rows=2,
                 template_sha256="1" * 64, shard=str(shard), sha256=D.A.sha(shard))
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(D.A.canonical(value))
    return rows, records, shard, manifest, value


def test_actual_source_receipt_and_exclusion_groups_are_preserved(tmp_path):
    rows, records, _, manifest, _ = fixture_source(tmp_path)
    _, candidates, removed, _ = D.packed_source(manifest, records, "train", set(), {"family-b"})
    assert [row["id"] for row in candidates] == ["task-a"]
    assert [row["id"] for row in removed] == ["task-b"]
    assert candidates[0]["source_record_sha256"] == D.digest(rows[0])
    with pytest.raises(ValueError, match="split contract"):
        D.packed_source(manifest, records, "validation", set(), set())


def test_changed_raw_shard_or_resealed_orphan_receipt_rejected(tmp_path):
    _, records, shard, manifest, value = fixture_source(tmp_path)
    shard.write_bytes(b"tampered raw source")
    with pytest.raises(ValueError, match="shard missing or changed"):
        D.packed_source(manifest, records, "train", set(), set())
    value["row_receipts"][0]["source_record_sha256"] = "0" * 64
    manifest.write_bytes(D.A.canonical(value))
    with pytest.raises(ValueError, match="complete admitted source"):
        D.packed_source(manifest, records, "train", set(), set())


def test_complete_documents_overshoot_real_budget_without_padding_or_repetition():
    rows = [dict(id="a", group="ga", source_index=0), dict(id="b", group="gb", source_index=1), dict(id="c", group="gc", source_index=2)]
    selected, total = D.select_complete(rows, [3, 5, 7], 6)
    assert total == 8 and [row["id"] for row in selected] == ["a", "b"]
    assert selected[-1]["start"] == 3 and selected[-1]["end"] == 8
    with pytest.raises(ValueError, match="insufficient real"):
        D.select_complete(rows, [3, 5, 7], 16)
    with pytest.raises(ValueError, match="group overlaps"):
        D.select_complete(rows, [3, 5, 7], 6, unavailable_groups={"gb"})
    with pytest.raises(ValueError, match="real length"):
        D.select_complete(rows, [3, 0, 7], 6)


def test_amendment_never_replaces_a_fictional_frozen_body():
    with pytest.raises(ValueError, match="frozen preregistration"):
        D.amended_preregistration(b"## 1. Scope\nfiction\n## Amendments\n", {}, {}, {}, {})
