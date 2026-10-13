"""Evidence, tool-structure, leakage, and split-isolation tests; no model runs."""
import hashlib
import json
import struct
from pathlib import Path

import pytest

from experiments.chimera_recovery import corpus as c


def row(identifier="one", route="text_reasoning", split="train"):
    return {"schema": c.SCHEMA, "id": identifier, "route": route, "split": split,
            "messages": [{"role": "user", "content": "Explain how heat changes the state of water."},
                         {"role": "assistant", "content": "Heating can turn liquid water into vapor."}],
            "tools": [], "media": [], "supervision": {"assistant_only": True, "message_mask": [False, True]},
            "provenance": {"source": "test_fixture", "source_ids": [identifier], "group_id": identifier,
                           "source_verified": True, "source_dates": ["2026-10-13T00:39:44Z"]}}


def function(name="lookup"):
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object",
            "properties": {"key": {"type": "string"}}, "required": ["key"], "additionalProperties": False}}}


def call(identifier="call1", name="lookup", arguments='{"key":"item"}'):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_row():
    r = row(route="tool_call")
    r["tools"] = [function()]
    r["messages"][1] = {"role": "assistant", "content": None, "tool_calls": [call()]}
    return r


def write_rows(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_supervision_normalizes_exact_integer_mask_and_rejects_other_values():
    r = row()
    r["supervision"] = {"policy": "assistant_only", "message_mask": [0, 1]}
    normalized = c.normalize_record(r)
    assert normalized["supervision"]["message_mask"] == [False, True]
    assert all(type(v) is bool for v in normalized["supervision"]["message_mask"])
    assert c.validate_record(normalized)["train_ready"]
    r["supervision"]["message_mask"] = [0, 2]
    assert "assistant_only_supervision_invalid" in c.validate_record(c.normalize_record(r))["errors"]


def test_terminal_call_supervision_and_recorded_result_links():
    r = tool_row()
    a = c.validate_record(r)
    assert a["train_ready"] and "terminal_open_tool_calls" in a["diagnostics"]
    r["messages"].append({"role": "tool", "tool_call_id": "other", "content": "value"})
    r["supervision"]["message_mask"].append(False)
    assert "tool_result_call_link_missing" in c.validate_record(r)["errors"]


def test_tool_schema_is_never_inferred_and_arguments_are_checked():
    r = tool_row()
    r["tools"] = []
    assert "authentic_tool_schema_missing" in c.validate_record(r)["errors"]
    r["tools"] = [function()]
    r["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"key":3}'
    assert "tool_arguments_schema_mismatch" in c.validate_record(r)["errors"]
    r["tools"][0]["function"]["parameters"]["oneOf"] = []
    assert "tool_schema_unsupported_validation_subset" in c.validate_record(r)["errors"]


def test_unique_name_ids_are_transparent_and_ambiguous_names_stay_blocked():
    r = tool_row()
    del r["messages"][1]["tool_calls"][0]["id"]
    r["messages"].append({"role": "tool", "name": "lookup", "content": "value"})
    r["supervision"]["message_mask"].append(False)
    r = c.normalize_record(r)
    assert r["messages"][1]["tool_calls"][0]["id"].startswith("derived_")
    assert r["messages"][2]["tool_call_id"] == r["messages"][1]["tool_calls"][0]["id"]
    assert "not claimed authentic" in r["provenance"]["tool_identifier_authenticity"]
    assert c.validate_record(r)["train_ready"]
    r = tool_row()
    r["messages"][1]["tool_calls"] = [call(None), call(None)]
    r["messages"].append({"role": "tool", "name": "lookup", "content": "ambiguous result"})
    r["supervision"]["message_mask"].append(False)
    assert "tool_result_call_link_missing" in c.validate_record(c.normalize_record(r))["errors"]


def test_parallel_requires_multiple_calls_in_one_assistant_message():
    r = tool_row()
    r["tools"].append(function("other"))
    r["messages"][1]["tool_calls"].append(call("call2", "other"))
    assert "tool_parallel" in c.validate_record(r)["capabilities"]
    r["messages"][1]["tool_calls"].pop()
    r["messages"].extend([{"role": "tool", "tool_call_id": "call1", "content": "ERROR: timeout"},
                          {"role": "assistant", "tool_calls": [call("retry")]},
                          {"role": "tool", "tool_call_id": "retry", "content": "recovered"}])
    r["supervision"]["message_mask"].extend([False, True, False])
    a = c.validate_record(r)
    assert a["train_ready"]
    assert {"tool_retry", "tool_multiturn", "tool_result"}.issubset(a["capabilities"])
    assert "tool_parallel" not in a["capabilities"]


def test_bfcl_captured_schema_and_acceptable_target_recovery():
    schema = {"name": "triangle.get", "parameters": {"type": "dict", "properties": {
        "side": {"type": "integer"}, "get_area": {"type": "boolean", "optional": True}}, "required": ["side"]}}
    target = {"answer_format": "bfcl_ground_truth_json", "call_count": 1,
              "expected_tool_calls": [{"triangle.get": {"side": [3], "get_area": ["", True]}}]}
    raw = {"rid": "547:1", "route": "tool", "instruction": "User request: find a triangle.\nAvailable tool schemas:\n" + json.dumps([schema]),
           "response": json.dumps(target)}
    r = c.normalize_record(raw, source_verified=True, collection_date="2026-10-13")
    assert c.validate_record(r)["train_ready"]
    assert json.loads(r["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"side": 3}
    assert r["tools"][0]["function"]["parameters"]["type"] == "object"
    assert r["provenance"]["bfcl_schema_target_adaptation"]["original_schemas"] == [schema]
    assert r["provenance"]["bfcl_schema_target_adaptation"]["original_acceptable_targets"] == target
    assert not any(m["role"] == "tool" for m in r["messages"])


def test_frozen_evaluation_exact_and_near_duplicates_exclude_shared_catalog():
    evaluation = row("eval")
    index = c.EvalIndex([evaluation])
    r = row("train")
    assert "eval_exact_text" in index.hits(r)
    r["messages"][1]["content"] += " Eventually."
    assert "eval_13gram_near_duplicate" in index.hits(r)
    a, b = row("a"), row("b")
    catalog = "\nAvailable tool schemas:\n" + json.dumps([function()])
    a["messages"][0]["content"] = "Pick up a parcel from the depot." + catalog
    b["messages"][0]["content"] = "Schedule maintenance at the mountain warehouse." + catalog
    a["messages"][1]["content"] = "Depot pickup accepted."
    b["messages"][1]["content"] = "Maintenance scheduled tomorrow."
    assert not c.EvalIndex([a]).hits(b)


def test_existing_split_source_components_cannot_cross(tmp_path):
    a, b = row("a", split="train"), row("b", split="test")
    b["provenance"]["source_ids"] = a["provenance"]["source_ids"]
    b["messages"][1]["content"] = "A distinct answer with the same source event."
    source = tmp_path / "rows.jsonl"
    write_rows(source, [a, b])
    report = c.build_corpus(corpus_root=tmp_path / "original", record_paths=[source], output=tmp_path / "out")
    assert report["counts"]["ready"] == 0
    assert report["blockers"]["source_group_crosses_existing_splits"] == 2


def test_eligible_legacy_duplicate_is_not_lost_to_ineligible_first_copy(tmp_path):
    source = tmp_path / "rows.jsonl"
    write_rows(source, [{"rid": "outside", "route": "text", "instruction": "Question with several different words.", "response": "The source answer is retained verbatim."},
                        {"rid": "inside", "route": "text", "instruction": "Question with several different words.", "response": "The source answer is retained verbatim."}])
    allocation = tmp_path / "allocation.json"
    allocation.write_text(json.dumps({"rows_sha256": c.file_sha256(source), "text": {"docs": ["inside"]}}))
    report = c.build_corpus(corpus_root=tmp_path / "original", record_paths=[source], output=tmp_path / "out",
                            allocation=allocation, collection_date="2026-10-13")
    assert report["counts"]["ready"] == 1
    ready = json.loads((tmp_path / "out/trainable.jsonl").read_text())
    assert ready["id"] == "inside"
    assert ready["messages"][1]["content"] == "The source answer is retained verbatim."


def test_carve_seal_and_source_binding_are_verified(tmp_path):
    body = {"rows_sha256": "a" * 64, "assignment": {"one": "train"}}
    body["carve_sha256"] = hashlib.sha256(json.dumps(body, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
    path = tmp_path / "carve.json"
    path.write_text(json.dumps(body))
    assert c.pinned_carve(path, [{"sha256": "a" * 64}])["assignment"]["one"] == "train"
    body["assignment"]["one"] = "eval"
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="seal"):
        c.pinned_carve(path, [{"sha256": "a" * 64}])


def test_source_hash_and_dates_fail_closed(tmp_path):
    source = tmp_path / "rows.jsonl"
    write_rows(source, [row()])
    with pytest.raises(ValueError, match="source hash mismatch"):
        c.build_corpus(corpus_root=tmp_path, record_paths=[source], output=tmp_path / "out", expected_hashes={str(source): "0" * 64})
    r = row()
    r["provenance"]["source_dates"] = ["2026-99-30"]
    assert "source_dates_missing_or_outside_2025_2026" in c.validate_record(r)["errors"]


def tensor_file(path, shape=(2,)):
    header = json.dumps({"latent": {"dtype": "F32", "shape": list(shape), "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<ff", 1.0, 2.0))


def teacher():
    return {"model_id": "test_fixture", "checkpoint_sha256": "c" * 64, "source_verified": True}


def test_real_asset_hash_tensor_selection_and_clean_target_leak(tmp_path):
    path = tmp_path / "real.safetensors"
    tensor_file(path)
    asset = {"role": "target", "type": "audio", "representation": "latent", "path": str(path),
             "sha256": c.file_sha256(path), "tensor": "latent", "shape": [2], "teacher": teacher()}
    resolver = c.AssetResolver(tmp_path)
    assert resolver.inspect(asset)["verified"]
    assert "media_tensor_shape_mismatch" in resolver.inspect({**asset, "shape": [1]})["errors"]
    r = row(route="music_generate")
    r["media"] = [asset, {**asset, "role": "conditioning", "noise_level": 0}]
    errors = c.validate_record(r, resolver=resolver)["errors"]
    assert "target_in_input_leakage" in errors
    assert "clean_latent_conditioning_leakage" in errors
    assert "media_sha256_mismatch" in resolver.inspect({**asset, "sha256": "0" * 64})["errors"]


def test_unrelated_av_and_2d_volume_proxy_are_blocked(tmp_path):
    path = tmp_path / "target.safetensors"
    tensor_file(path)
    asset = {"path": str(path), "sha256": c.file_sha256(path), "tensor": "latent", "shape": [2], "teacher": teacher(),
             "representation": "latent", "source_media_sha256": "d" * 64, "start_seconds": 0}
    r = c.prepare_av_pair(identifier="av", prompt="Source scene", video=asset, audio={**asset, "source_media_sha256": "e" * 64},
                          provenance=row()["provenance"], duration_seconds=2, fps=24, sample_rate=16000)
    r["av_alignment"]["pairing"] = "unrelated"
    errors = c.validate_record(r, resolver=c.AssetResolver(tmp_path))["errors"]
    assert "joint_av_unrelated_pairing" in errors and "joint_av_same_origin_or_timestamp_missing" in errors
    r = row(route="volume_generate")
    r["media"] = [{**asset, "role": "target", "type": "image"}]
    assert "genuine_3d_target_missing" in c.validate_record(r, resolver=c.AssetResolver(tmp_path))["errors"]


def test_manifest_claims_and_teacher_rewrites_do_not_become_ready(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"items": 1000}))
    inventory = c.inventory_manifests(tmp_path, c.AssetResolver(tmp_path))
    assert inventory[0]["declared_counts"]["items"] == 1000 and inventory[0]["verified_references"] == 0
    r = row()
    r["messages"][0]["content"] = "You are a teacher model producing compact structured distillation targets. CURRENT_TARGET: the answer is 6."
    assert "teacher_rewrite_exposes_current_answer" in c.validate_record(r)["errors"]
