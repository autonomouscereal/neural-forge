"""Tiny synthetic CPU weights/captures only; fixtures never receive acceptance."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
import torch
import c16_qimg_development_refold as development
import c16_qimg_fold as original
import c16_qimg_fresh_fold as writer
from test_c16_qimg_development_refold import tensor_data, policy


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def assert_no_acceptance(out):
    assert not (out / "fold_receipt.json").exists()
    rejection = out.with_name(out.name + ".rejected.json")
    assert rejection.exists()
    body = json.loads(rejection.read_text())
    assert body["accepted"] is body["acceptance_enabled"] is False
    if (out / "provisional_fold_receipt.json").exists():
        assert json.loads((out / "provisional_fold_receipt.json").read_text())["accepted"] is False


def test_actual_safetensors_roundtrip_fp8_bf16_bias_and_selection(tmp_path):
    data = tensor_data(); before = {k: v.clone() for k, v in data["frozen"].items()}
    result = writer._convert_fixture_for_tests(data, policy(), tmp_path / "candidate")
    assert result["accepted"] is result["acceptance_enabled"] is False
    assert result["persisted_numeric_gates_passed"] is True
    assert result["persisted_tensors_rescored_without_refit"] is True
    for field in ("native_block_accepted", "media_installed", "child_accepted", "capability_accepted", "full_joint_block_replay_performed"):
        assert result[field] is False
    loaded, identity = writer.joint.load_candidate(tmp_path / "candidate/fold_receipt.json",
        original.file_sha(tmp_path / "candidate/fold_receipt.json"), data)
    selections, _ = original.verified_tensors(tmp_path / "candidate", result["selection"], 1 << 20)
    for stream in original.STREAMS:
        prefix = f"transformer_blocks.0.{stream}_mlp."
        assert loaded[prefix + "net.0.proj.weight"].dtype == torch.float8_e4m3fn
        assert loaded[prefix + "net.0.proj.bias"].dtype == torch.float8_e4m3fn
        assert loaded[prefix + "net.2.weight"].dtype == torch.bfloat16
        assert loaded[prefix + "net.2.bias"].dtype == torch.bfloat16
        assert selections[stream + ".selected_channels"].tolist() == [3, 7, 12]
        assert result["streams"][stream]["child_parameters"] == (data["width"]+1)*3 + data["width"]*(3+1)
        assert result["streams"][stream]["stored_update_scale"] <= 1
    assert all(writer.joint.same_bytes(v, before[k]) for k, v in data["frozen"].items())


def test_kernel_is_one_fit_no_rescore_refit_and_development_unchanged(tmp_path, monkeypatch):
    calls = []
    solve, select = development.law_abiding_solve, development.select_fit_output_weighted_columns
    def fit(*args): calls.append("solve"); return solve(*args)
    def choose(*args): calls.append("select"); return select(*args)
    monkeypatch.setattr(development, "law_abiding_solve", fit)
    monkeypatch.setattr(development, "select_fit_output_weighted_columns", choose)
    writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert calls == ["select", "solve", "select", "solve"]
    report = development.fold_loaded(tensor_data(), policy())
    assert report["accepted"] is report["acceptance_enabled"] is report["candidate_written"] is False
    def no_tensors(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for v in value.values(): no_tensors(v)
        elif isinstance(value, list):
            for v in value: no_tensors(v)
    no_tensors(report)


def test_both_stream_source_replay_before_any_selection(tmp_path, monkeypatch):
    data = tensor_data(); data["forwards"][-1]["txt"]["y"][0, -1, 0] += 10
    monkeypatch.setattr(development, "select_fit_output_weighted_columns", lambda *args: pytest.fail("source refusal reached selection"))
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(data, policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


@pytest.mark.parametrize("bad_call,metric,value", [(3, "rel_l2", .31), (5, "rel_l2", .31),
                                                 (3, "cos", .94), (5, "cos", .94)])
def test_pooled_and_each_heldout_gates_survive_reload(tmp_path, monkeypatch, bad_call, metric, value):
    score = development.score
    candidate_scores = []
    def poison(actual, target):
        values = score(actual, target); candidate_scores.append(1)
        # Only the separately invoked persisted-scoring phase is poisoned.
        if len(candidate_scores) == bad_call: values[metric] = value
        return values
    rescore = writer._rescore_persisted
    def altered(*args, **kwargs):
        with monkeypatch.context() as patcher:
            patcher.setattr(development, "score", poison)
            return rescore(*args, **kwargs)
    monkeypatch.setattr(writer, "_rescore_persisted", altered)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_corrupted_written_tensor_refuses_final_marker(tmp_path, monkeypatch):
    write = original.write_tensors
    def corrupt(path, tensors):
        record = write(path, tensors)
        if Path(path).name == "folded_source_block_and_time.safetensors":
            with Path(path).open("ab") as handle: handle.write(b"corrupt")
        return record
    monkeypatch.setattr(original, "write_tensors", corrupt)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_serializer_failure_preserves_partial_without_success(tmp_path, monkeypatch):
    write = original.write_tensors
    def fail(path, tensors):
        if Path(path).name == "selection.safetensors": raise OSError("injected serializer failure")
        return write(path, tensors)
    monkeypatch.setattr(original, "write_tensors", fail)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert (tmp_path / "candidate/folded_source_block_and_time.safetensors").exists()
    assert_no_acceptance(tmp_path / "candidate")


def test_serialized_refit_shape_or_dtype_change_refused(tmp_path, monkeypatch):
    write = original.write_tensors
    def alter(path, tensors):
        if Path(path).name == "folded_source_block_and_time.safetensors":
            tensors = dict(tensors)
            key = "transformer_blocks.0.img_mlp.net.2.bias"
            tensors[key] = tensors[key].float()
        return write(path, tensors)
    monkeypatch.setattr(original, "write_tensors", alter)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_frozen_attention_change_refused_after_valid_file_hashes(tmp_path, monkeypatch):
    write = original.write_tensors
    def alter(path, tensors):
        if Path(path).name == "folded_source_block_and_time.safetensors":
            tensors = dict(tensors); key = "transformer_blocks.0.attn.to_q.weight"
            tensors[key] = (tensors[key].float() + .1).to(tensors[key].dtype)
        return write(path, tensors)
    monkeypatch.setattr(original, "write_tensors", alter)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_refit_finite_but_bf16_overflow_refused(tmp_path, monkeypatch):
    solve = development.law_abiding_solve
    def overflow(*args):
        fitted, receipt = solve(*args); return torch.full_like(fitted, 1e100), receipt
    monkeypatch.setattr(development, "law_abiding_solve", overflow)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_stored_update_veto_refused(tmp_path, monkeypatch):
    solve = development.law_abiding_solve
    def changed(*args):
        fitted, receipt = solve(*args); return args[2] * 3, receipt
    monkeypatch.setattr(development, "law_abiding_solve", changed)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_source_revalidation_failure_before_publication(tmp_path):
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 3: raise ValueError("source pin changed after reload")
    with pytest.raises(original.FoldRejected):
        writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate", revalidate=guard)
    assert_no_acceptance(tmp_path / "candidate")


def test_payload_fsync_failure_precedes_success_marker(tmp_path, monkeypatch):
    fsync = writer.os.fsync
    calls = []
    def fail(fd):
        calls.append(1)
        if len(calls) == 2: raise OSError("injected payload fsync failure")
        return fsync(fd)
    monkeypatch.setattr(writer.os, "fsync", fail)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")


def test_exclusive_publication_conflict_leaves_no_author_success(tmp_path, monkeypatch):
    def fail(*args): raise FileExistsError("injected publication conflict")
    monkeypatch.setattr(writer.os, "link", fail)
    with pytest.raises(original.FoldRejected): writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    assert_no_acceptance(tmp_path / "candidate")
    refusal = json.loads((tmp_path / "candidate.rejected.json").read_text())
    assert refusal["staged_receipt_present"] is True
    assert refusal["staged_receipt_authoritative"] is False
    assert refusal["canonical_receipt_present"] is False
    assert refusal["canonical_receipt_published_by_this_writer"] is False
    assert refusal["candidate_written"] is refusal["selection_tensors_written"] is True


def test_fixture_cannot_claim_nonfixture_acceptance(tmp_path):
    data = tensor_data(); data["fixture"] = False
    with pytest.raises(ValueError, match="synthetic fixture"):
        writer._convert_fixture_for_tests(data, policy(), tmp_path / "candidate")


def test_existing_output_refuses_before_kernel(tmp_path, monkeypatch):
    target = tmp_path / "candidate"; target.mkdir()
    monkeypatch.setattr(development, "_fit_loaded_candidate", lambda *args, **kwargs: pytest.fail("existing output reached fit"))
    with pytest.raises(ValueError): writer._convert_fixture_for_tests(tensor_data(), policy(), target)


def test_nonfinite_receipt_cannot_publish_marker(tmp_path):
    with pytest.raises(ValueError): writer._publish_final(tmp_path / "fold_receipt.json", {"accepted": True, "bad": float("nan")})
    assert not (tmp_path / "fold_receipt.json").exists()


def test_writer_requires_new_recipe_source_pin_even_with_stage_metadata(tmp_path):
    from test_c16_qimg_fresh_stages import Fixture
    fixture = Fixture(tmp_path)
    ctx = writer.gate.load(**fixture.args)
    path = Path(original.__file__).parent / "qimg_fold_policy_broad_sep07.json"
    with pytest.raises(ValueError, match="writer/kernel/selector"):
        writer._code_and_policy(ctx, fixture.args, path, original.file_sha(path))


def test_production_has_no_caller_selected_captures_or_fixture_flag():
    import inspect
    signature = inspect.signature(writer.convert_fresh)
    assert "captures" not in signature.parameters and "fixture" not in signature.parameters


def test_source_and_authority_overlap_refused_before_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(writer.gate, "load", lambda **kwargs: pytest.fail("unsafe destination reached authority read"))
    with pytest.raises(ValueError, match="overlaps"):
        writer.convert_fresh(run_receipt_path=tmp_path / "evidence/run.json", run_receipt_sha="0"*64,
            policy_path=tmp_path / "code/policy.json", policy_sha="0"*64, out=tmp_path / "code/child",
            code_root=tmp_path / "code", evidence_root=tmp_path / "evidence")


@pytest.mark.parametrize("name", [".fold_receipt.publish_pending.json", "provisional_fold_receipt.json", "renamed.json"])
def test_consumer_rejects_noncanonical_name_even_with_accepted_body(tmp_path, monkeypatch, name):
    # Deliberately malformed synthetic metadata, not a candidate acceptance.
    path = tmp_path / name; path.write_text('{"accepted":true}')
    monkeypatch.setattr(writer.gate.meta, "strict_json", lambda *args: pytest.fail("noncanonical input reached receipt read"))
    with pytest.raises(ValueError, match="canonical regular"):
        writer.load_accepted_component(receipt_path=path, receipt_sha="0"*64,
            run_receipt_path="unused", run_receipt_sha="0"*64, policy_path="unused", policy_sha="0"*64)


def test_consumer_rejects_canonical_fixture_before_authority_reads(tmp_path, monkeypatch):
    writer._convert_fixture_for_tests(tensor_data(), policy(), tmp_path / "candidate")
    path = tmp_path / "candidate/fold_receipt.json"
    monkeypatch.setattr(writer.gate, "load", lambda **kwargs: pytest.fail("fixture reached real authority read"))
    with pytest.raises(ValueError, match="nonfixture component"):
        writer.load_accepted_component(receipt_path=path, receipt_sha=original.file_sha(path),
            run_receipt_path="unused", run_receipt_sha="0"*64, policy_path="unused", policy_sha="0"*64)


def test_consumer_rejects_symlink_marker_before_metadata(tmp_path, monkeypatch):
    target = tmp_path / "target.json"; target.write_text('{}')
    path = tmp_path / "fold_receipt.json"; path.symlink_to(target)
    monkeypatch.setattr(writer.gate.meta, "strict_json", lambda *args: pytest.fail("symlink reached receipt read"))
    with pytest.raises(ValueError, match="canonical regular"):
        writer._published_header(path, "0"*64)


@pytest.mark.parametrize("key,value", [("fixture", True), ("child_accepted", True), ("native_block_accepted", True),
                                      ("fit_or_selector_retry", True), ("persisted_numeric_gates_passed", False),
                                      ("acceptance_scope", "installed child")])
def test_consumer_rejects_malformed_acceptance_header(tmp_path, key, value):
    # Header-only invalid fixture: no tensors, capture, authority or acceptance.
    body = {"schema": original.FOLD_SCHEMA, "status": "measured_candidate", "recipe_id": development.RECIPE_ID, "acceptance_scope": writer.SCOPE,
            "serialization_validation": writer.SERIALIZATION_COMPLETE,
            **{k: True for k in ("accepted", "acceptance_enabled", "numerical_policy_passed", "persisted_numeric_gates_passed",
                                "persisted_tensors_rescored_without_refit", "candidate_written", "selection_tensors_written")},
            **{k: False for k in ("fixture", "fit_or_selector_retry", "native_block_accepted", "media_installed",
                                 "child_accepted", "capability_accepted", "full_joint_block_replay_performed")}}
    body[key] = value
    path = tmp_path / "fold_receipt.json"; path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="nonfixture component"): writer._published_header(path, original.file_sha(path))
