"""Current owner policy on real serialization of tiny synthetic CPU tensors.

No genuine capture, production admission, model capability or GPU proof claimed.
"""
import copy
import json
import sys
from pathlib import Path
import pytest
import torch
import c16_qimg_fold as fold
import c16_qimg_development_refold as development
import c16_qimg_fresh_fold as fresh
import c16_qimg_joint_replay as joint
from test_c16_qimg_development_refold import tensor_data, policy, bounded_threads
from test_c16_qimg_fold import bundle


def read_fixture(tmp_path, data, result):
    path = tmp_path / "artifact/fold_receipt.json"
    stored, identity = joint.load_candidate(path, fold.file_sha(path), data)
    assert identity["output_sha256"] == result["output"]["file_sha256"]
    assert result["accepted"] is result["acceptance_enabled"] is result["capability_accepted"] is False
    assert result["artifact_admitted"] is False  # synthetic is never real admission
    assert result["structural_artifact_valid"] is result["candidate_written"] is True
    assert result["owner_policy"] == fold.current_owner_policy()
    return stored


def test_actual_finite_poor_fidelity_emits_and_loads_without_quality_success(tmp_path):
    data = tensor_data()
    for item in data["forwards"]:
        if item["split"] == "holdout":
            for stream in fold.STREAMS: item[stream]["y"] = (item[stream]["y"].float()+50).bfloat16()
    result = fresh._convert_fixture_for_tests(data, policy(), tmp_path/"artifact")
    stored = read_fixture(tmp_path, data, result)
    assert result["numerical_policy_passed"] is result["persisted_numeric_gates_passed"] is False
    assert result["source_replay_diagnostic_passed"] is False
    assert all(s["heldout_scores"]["rel_l2"] > .30 for s in result["streams"].values())
    assert all(bool(torch.isfinite(v.float()).all()) for v in stored.values())


def test_full_keep_low_sample_and_fourfold_shortfall_are_real_diagnostics(tmp_path):
    data = tensor_data()
    for item in data["forwards"]:
        if item["split"] == "fit":
            for stream in fold.STREAMS:
                item[stream] = {"x": item[stream]["x"][:, :1].clone(), "y": item[stream]["y"][:, :1].clone(),
                                "valid": torch.ones(1, dtype=torch.bool)}
    for stream in fold.STREAMS: data["row_counts"][stream]["fit"] = 2
    p = policy(keep=data["inner"])
    result = fresh._convert_fixture_for_tests(data, p, tmp_path/"artifact")
    stored = read_fixture(tmp_path, data, result)
    assert result["all_numeric_diagnostic_targets_met"] is False
    for stream in fold.STREAMS:
        values = result["streams"][stream]
        assert values["keep"] == data["inner"]
        assert values["component_compression_x"] == 1.0
        assert values["fourfold_target_met"] is False
        assert values["selector_diagnostics"]["rank_shortfall_fallback_used"] is True
        assert values["selected_channel_indices"] == list(range(data["inner"]))
        assert result["row_preflight"][stream]["fit_conditioning_target_met"] is False
        assert stored[f"transformer_blocks.0.{stream}_mlp.net.2.weight"].shape == (data["width"], data["inner"])


def test_constant_rank_zero_and_undefined_condition_diagnostic_publish(tmp_path, monkeypatch):
    data = tensor_data()
    for item in data["forwards"]:
        for stream in fold.STREAMS:
            item[stream]["x"] = torch.zeros_like(item[stream]["x"])
            _, item[stream]["y"] = fold.mlp_forward(item[stream]["x"], fold.mlp_weights(data["frozen"],0,stream,"cpu"))
    solver = development.law_abiding_solve
    def undefined_condition(*args):
        fitted, notes = solver(*args)
        return fitted, {**notes, "condition_number": float("inf")}
    monkeypatch.setattr(development, "law_abiding_solve", undefined_condition)
    result = fresh._convert_fixture_for_tests(data, policy(), tmp_path/"artifact")
    read_fixture(tmp_path, data, result)
    for stream in fold.STREAMS:
        assert result["streams"][stream]["selector_diagnostics"]["rank_shortfall_fallback_used"] is True
        assert result["streams"][stream]["solve"]["condition_number"] is None
        assert result["streams"][stream]["solve"]["undefined_diagnostics"]
    assert result["diagnostic_measurements_complete"] is result["all_numeric_diagnostic_targets_met"] is False
    json.dumps(result, allow_nan=False)


def test_fallback_uses_no_heldout_rows_or_targets(tmp_path):
    data = tensor_data()
    for item in data["forwards"]:
        if item["split"] == "fit":
            for stream in fold.STREAMS:
                item[stream]["x"] = torch.zeros_like(item[stream]["x"])
                _, item[stream]["y"] = fold.mlp_forward(item[stream]["x"], fold.mlp_weights(data["frozen"],0,stream,"cpu"))
    changed = copy.deepcopy(data)
    for item in changed["forwards"]:
        if item["split"] == "holdout":
            for stream in fold.STREAMS:
                item[stream]["x"] = (item[stream]["x"].float()*2+5).bfloat16()
                item[stream]["y"] = (item[stream]["y"].float()-12).bfloat16()
    a, sa, ra = development._fit_loaded_candidate(data, policy())
    b, sb, rb = development._fit_loaded_candidate(changed, policy())
    assert all(joint.same_bytes(v, b[k]) for k,v in a.items())
    assert all(joint.same_bytes(v, sb[k]) for k,v in sa.items())
    assert all(ra["streams"][s]["selector_diagnostics"]["rank_shortfall_fallback_used"] for s in fold.STREAMS)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_actual_nonfinite_input_is_not_a_diagnostic_fallback(tmp_path, bad):
    data = tensor_data(); data["forwards"][0]["img"]["x"][0,0,0] = bad
    with pytest.raises(fold.FoldRejected): fresh._convert_fixture_for_tests(data, policy(), tmp_path/"artifact")
    assert not (tmp_path/"artifact/fold_receipt.json").exists()
    assert json.loads((tmp_path/"artifact.rejected.json").read_text())["structural_artifact_valid"] is False


@pytest.mark.parametrize("message", ["non-finite FIT feature", "parent readout channel count does not match FIT features", "synthetic unexpected error"])
def test_rank_fallback_does_not_swallow_other_selector_errors(monkeypatch, message):
    def fail(*args): raise ValueError(message)
    monkeypatch.setattr(development, "select_fit_output_weighted_columns", fail)
    with pytest.raises(ValueError, match=message): development.select_current_recipe(torch.zeros(4,3),torch.zeros(2,3,dtype=torch.bfloat16),2)


@pytest.mark.parametrize("keep", [True, 0, -1, 33])
def test_invalid_geometry_remains_a_hard_failure(keep):
    with pytest.raises(ValueError): fold.row_preflight(tensor_data(), policy(keep=keep))


def test_missing_authentic_capture_file_is_still_rejected(tmp_path):
    (tmp_path/"capture").mkdir()
    path, receipt, p, _ = bundle(tmp_path/"capture")
    (path.parent/receipt["captures"][0]["file"]).unlink()
    with pytest.raises(ValueError): fold.convert([path],p,fold.file_sha(p),tmp_path/"artifact",fixture=True)
    assert not (tmp_path/"artifact").exists()


def test_legacy_numerical_rejection_is_not_relabelled_by_new_reader(tmp_path):
    data=tensor_data()
    result=fresh._convert_fixture_for_tests(data,policy(),tmp_path/"artifact")
    path=tmp_path/"artifact/fold_receipt.json"
    legacy=json.loads(path.read_text()); legacy["schema"]=fold.FOLD_SCHEMA; legacy["numerical_policy_passed"]=False
    archive=path.with_name("historical_rejected_receipt.json"); archive.write_text(json.dumps(legacy))
    before=archive.read_bytes()
    with pytest.raises(ValueError): joint.load_candidate(archive,fold.file_sha(archive),data)
    assert archive.read_bytes()==before


def test_new_header_allows_false_finite_diagnostics_without_quality_claim(tmp_path):
    # Header validator only; no fabricated production data or reader invocation.
    body={"schema":fold.OWNER_FOLD_SCHEMA,"status":"measured_candidate","owner_policy":fold.current_owner_policy(),
          "recipe_id":development.RECIPE_ID,"acceptance_scope":fresh.SCOPE,"serialization_validation":fresh.SERIALIZATION_COMPLETE,
          "numerical_policy_passed":False,"persisted_numeric_gates_passed":False,
          **{k:True for k in ("artifact_admitted","structural_artifact_valid","persisted_tensors_rescored_without_refit","candidate_written","selection_tensors_written")},
          **{k:False for k in ("accepted","acceptance_enabled","fixture","fit_or_selector_retry","native_block_accepted","media_installed","child_accepted","capability_accepted","full_joint_block_replay_performed")}}
    path=tmp_path/"fold_receipt.json"; path.write_text(json.dumps(body))
    assert fresh._published_header(path,fold.file_sha(path))[1]==body


def test_current_fresh_writer_does_not_silently_reinterpret_legacy_recipe():
    root=Path(fold.__file__).resolve().parents[4]
    policy_path=Path(fold.__file__).parent/"qimg_fold_policy_broad_sep07.json"
    modules=[fresh,development,sys.modules["mediafold.experimental_fit_selector"]]
    paths=[Path(m.__file__).resolve() for m in modules]+[policy_path.resolve()]
    pins=[{"path":str(p.relative_to(root)),"sha256":fold.file_sha(p),"bytes":p.stat().st_size} for p in paths]
    body=json.loads(policy_path.read_text())
    ctx={"recipe":{"source_tool_pins":pins,"streams":body["streams"],
        "rules":{"selector_recipe_id":development.HISTORICAL_RECIPE_ID}}}
    with pytest.raises(ValueError,match="explicit versioned fresh contract"):
        fresh._code_and_policy(ctx,{"code_root":root},policy_path,fold.file_sha(policy_path))
