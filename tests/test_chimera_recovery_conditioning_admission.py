"""Independent file-fixture failures for immutable ACE hint admission rebinding."""
import json
from pathlib import Path

import pytest

from experiments.chimera_recovery import conditioning_admission as r
from experiments.chimera_recovery.corpus import canonical, digest, file_sha256


def setup(tmp_path):
    cache, hints = tmp_path / "cache", tmp_path / "hints"
    cache.mkdir(); hints.mkdir()
    def write(path, body):
        path.write_bytes(canonical(body) + b"\n")
        return file_sha256(path)
    item = {"id": "genuine-item", "sha256": "a" * 64, "caption": "Original caption", "lyrics": "", "seed": 7, "held": False,
            "bytes": 108, "clip_seconds": 60, "rid": "original-source:1", "tier": "original"}
    corpus = {"items": [item], "n_items": 1, "corpus_sha256": digest([item])}
    items = tmp_path / "items.json"
    items_sha = write(items, corpus)
    raw = cache / "teacher-000.safetensors"
    raw.write_bytes(b"Independent authenticated teacher fixture bytes")
    sidecar = {"schema": "omni-campaign.teacher_cache_shard.v2", "item_id": item["id"],
               "item": {k: v for k, v in item.items() if k not in {"bytes", "clip_seconds", "rid", "tier"}},
               "file_sha256": file_sha256(raw), "finite": True, "corpus_sha256": corpus["corpus_sha256"],
               "config_sha256": "e" * 64, "teacher": "ace_dit_velocity"}
    sidecar_sha = write(raw.with_suffix(".json"), sidecar)
    teacher = {"files": [{"path": "/sealed/teacher.safetensors", "sha256": "f" * 64, "bytes": 192}]}
    source_manifest = {"schema": "omni-campaign.teacher_cache.v2", "teacher": "ace_dit_velocity", "finite": True,
        "sealed_at_utc": "2026-10-13T00:00:00Z", "teacher_integrity": {"equal": True, "before": [teacher], "after": teacher},
        "corpus": {"sha256": corpus["corpus_sha256"], "file_sha256": items_sha}, "config_sha256": "e" * 64,
        "shards": [{"item_id": item["id"], "name": raw.name, "sha256": file_sha256(raw), "finite": True, "meta": {"held": False}}]}
    manifest_sha = write(cache / "manifest.json", source_manifest)
    row = {"schema": "recovery_media_admission.v1", "kind": "ace", "item_id": item["id"], "split": "train",
           "train_admitted": True, "blockers": [], "recovery_group_id": "same-source-group",
           "asset_sha256": item["sha256"], "prompt_sha256": digest(item["caption"]),
           "conditioning_sha256": digest([item["caption"], item["lyrics"]]), "item_payload_sha256": digest(item)}
    sources = {"ace_items": {"sha256": items_sha}, "ace_cache_manifest": {"sha256": manifest_sha}, "eval_rows": {"sha256": "b" * 64}}
    paths = {}
    for name in ("old", "new"):
        directory = tmp_path / name; directory.mkdir()
        rows_path = directory / "admission.jsonl"; rows_sha = write(rows_path, row)
        body = {"schema": "recovery_media_admission.v1", "connected_groups_verified": True, "eval_disjoint": True,
                "sources": dict(sources), "outputs": {"admission": {"path": str(rows_path), "rows": 1, "sha256": rows_sha}}}
        if name == "new": body["sources"]["new_qvl_cache"] = {"sha256": "c" * 64}
        body["manifest_sha256"] = digest(body)
        path = directory / "manifest.json"; write(path, body); paths[name] = path
    hint = hints / "condition-000.safetensors"; hint.write_bytes(b"Immutable precomputed authentic code hint fixture")
    component = {str(i): digest([i, "independent component"]) for i in range(32)}
    conditioning = {"schema": "chimera_recovery.ace_conditioning.v1", "sealed": True, "all_finite": True,
        "complete_source_traversal": True, "source_manifest_sha256": manifest_sha, "corpus_sha256": file_sha256(paths["old"]),
        "adapter_sha256": "d" * 64, "component": {"source_pins": {"authentic.py": "3" * 64}, "teacher_sha256": "f" * 64,
            "component_tensors": component, "component_sha256": digest(component), "device": "cpu", "dtype": "float32"},
        "items": [{"item_id": item["id"], "path": hint.name, "sha256": file_sha256(hint), "frames": 12,
            "hints_sha256": "1" * 64, "codes_sha256": "2" * 64, "source_shard_sha256": file_sha256(raw),
            "source_sidecar_sha256": sidecar_sha, "recovery_group_id": row["recovery_group_id"]}]}
    cp = hints / "manifest.json"; write(cp, conditioning)
    kwargs = dict(cache=cache, items=items, old_admission=paths["old"], new_admission=paths["new"], conditioning=cp,
        output=tmp_path / "result", source_manifest_sha256=manifest_sha, old_admission_sha256=file_sha256(paths["old"]),
        new_admission_sha256=file_sha256(paths["new"]), conditioning_sha256=file_sha256(cp), conditioner_sha256="d" * 64)
    return kwargs, hint, raw


def update_admission(kwargs, mutate):
    path = kwargs["new_admission"]; body = json.loads(path.read_text())
    row_path = Path(body["outputs"]["admission"]["path"])
    row = json.loads(row_path.read_text()); mutate(row)
    row_path.write_bytes(canonical(row) + b"\n")
    body["outputs"]["admission"]["sha256"] = file_sha256(row_path)
    body.pop("manifest_sha256"); body["manifest_sha256"] = digest(body)
    path.write_bytes(canonical(body) + b"\n"); kwargs["new_admission_sha256"] = file_sha256(path)


def test_complete_rebind_preserves_every_hint_byte_and_old_artifacts(tmp_path):
    kwargs, hint, _ = setup(tmp_path)
    old_bytes = kwargs["conditioning"].read_bytes()
    result = r.rebind_ace(**kwargs)
    assert (kwargs["output"] / hint.name).read_bytes() == hint.read_bytes()
    assert kwargs["conditioning"].read_bytes() == old_bytes
    assert result["corpus_sha256"] == kwargs["new_admission_sha256"]
    assert result["admission_rebind"]["original_admission_sha256"] == kwargs["old_admission_sha256"]
    assert result["items"] == json.loads(old_bytes)["items"]
    assert result["admission_rebind"]["model_loaded"] is False


@pytest.mark.parametrize("change", ["held", "group", "asset"])
def test_resealed_new_admission_cannot_reassign_or_drop_hints(tmp_path, change):
    kwargs, _, _ = setup(tmp_path)
    def mutate(row):
        if change == "held": row.update(train_admitted=False, blockers=["teacher_held"])
        elif change == "group": row["recovery_group_id"] = "different-source-family"
        else: row["asset_sha256"] = "9" * 64
    update_admission(kwargs, mutate)
    with pytest.raises(ValueError, match="item set changed|identity changed"):
        r.rebind_ace(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("target", ["hint", "teacher"])
def test_same_size_hint_or_teacher_corruption_refuses_publish(tmp_path, target):
    kwargs, hint, raw = setup(tmp_path)
    path = hint if target == "hint" else raw
    value = path.read_bytes(); path.write_bytes(bytes([value[0] ^ 1]) + value[1:])
    with pytest.raises(ValueError, match="bytes moved|bytes changed"):
        r.rebind_ace(**kwargs)
    assert not kwargs["output"].exists()


def test_partial_original_conditioning_and_wrong_audited_helper_refuse(tmp_path):
    kwargs, _, _ = setup(tmp_path)
    kwargs["conditioner_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="complete pinned"):
        r.rebind_ace(**kwargs)
    kwargs["conditioner_sha256"] = "d" * 64
    path = kwargs["conditioning"]; body = json.loads(path.read_text()); body["complete_source_traversal"] = False
    path.write_bytes(canonical(body)); kwargs["conditioning_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="complete pinned"):
        r.rebind_ace(**kwargs)
    assert not kwargs["output"].exists()
