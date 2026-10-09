"""Synthetic CPU-only runner checks; no real captures, models or acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_development_refold as runner
import c16_qimg_fold as original
from test_c16_qimg_fold import bundle, dump


@pytest.fixture(autouse=True)
def bounded_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def policy(keep=3):
    return {"schema": original.POLICY_SCHEMA, "preregistered_before_scoring": True,
        "gate_scope": "pooled_and_each_heldout_forward",
        "streams": {s: {"keep": keep, "max_source_replay_rel_l2": .005,
            "max_heldout_rel_l2": .30, "min_heldout_cos": .95} for s in runner.STREAMS}}


def tensor_data(*, sparse=True):
    generator = torch.Generator().manual_seed(471)
    block, width, inner = 0, 4, 32
    frozen = {k: (torch.randn(shape, generator=generator) * .2).to(torch.float8_e4m3fn)
        for k, shape in original.frozen_shapes(block, width, inner, 2).items()}
    if sparse:
        for stream in runner.STREAMS:
            key = f"transformer_blocks.0.{stream}_mlp.net.2.weight"
            readout = torch.zeros(width, inner)
            readout[:, [3, 7, 12]] = torch.randn(width, 3, generator=generator) * .2
            frozen[key] = readout.to(torch.float8_e4m3fn)
    forwards, counts, splits = [], {s: {"fit": 0, "holdout": 0} for s in runner.STREAMS}, {}
    for i, split in enumerate(("fit", "fit", "holdout")):
        image = digest(f"synthetic image {i}")
        item = {"file": f"synthetic forward {i}", "image_sha256": image, "split": split,
            "file_sha256": digest(f"synthetic tensor envelope {i}"), "sigma": .3 + .2 * i}
        splits[image] = split
        for stream, rows in (("img", 12), ("txt", 8)):
            x = torch.randn(1, rows, width, generator=generator).bfloat16()
            _, y = original.mlp_forward(x, original.mlp_weights(frozen, block, stream, "cpu"))
            valid = torch.ones(rows, dtype=torch.bool)
            if stream == "txt": valid[-2:] = False
            item[stream] = {"x": x, "y": y, "valid": valid}
            counts[stream][split] += int(valid.sum())
        forwards.append(item)
    return {"fixture": True, "block": block, "width": width, "inner": inner,
        "frozen": frozen, "frozen_file_sha256": digest("synthetic frozen envelope, not a file"),
        "forwards": forwards, "row_counts": counts, "image_splits": splits, "capture_receipts": []}


def assert_nonaccepting(receipt):
    assert receipt["schema"] == runner.DEVELOPMENT_SCHEMA
    assert receipt["accepted"] is receipt["acceptance_enabled"] is False
    assert receipt["candidate_written"] is receipt["selection_tensors_written"] is False
    assert receipt["fresh_data_acceptance_validated"] is False


def test_native_bf16_both_streams_numeric_pass_is_still_nonaccepting(monkeypatch):
    monkeypatch.setattr(runner, "load_captures", lambda *a, **k: pytest.fail("tensor API read captures"))
    monkeypatch.setattr(runner, "load_policy", lambda *a, **k: pytest.fail("tensor API read policy"))
    result = runner.fold_loaded(tensor_data(), policy())
    assert_nonaccepting(result)
    assert result["status"] == "development_complete"
    assert result["development_numeric_gates_passed"] is True
    assert result["runtime"]["compute_dtype"] == "torch.bfloat16"
    assert "caller-supplied" in result["input_contract"]
    for stream in runner.STREAMS:
        measured = result["streams"][stream]
        assert measured["selected_channel_indices"] == [3, 7, 12]
        assert 4 * measured["child_parameters"] <= measured["parent_parameters"]
        assert measured["stored_update_scale"] <= 1
        assert measured["heldout_scores"]["rel_l2"] <= .30
        assert measured["heldout_scores"]["cos"] >= .95
    def no_tensors(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for v in value.values(): no_tensors(v)
        elif isinstance(value, (list, tuple)):
            for v in value: no_tensors(v)
    no_tensors(result)


def test_all_source_forwards_of_both_streams_replay_before_selection(monkeypatch):
    data = tensor_data()
    replay, selected = runner.mlp_forward, runner.select_fit_output_weighted_columns
    calls = []
    def observe_forward(*args):
        calls.append("forward")
        return replay(*args)
    def observe_selection(*args):
        assert calls[:6] == ["forward"] * 6
        calls.append("selection")
        return selected(*args)
    monkeypatch.setattr(runner, "mlp_forward", observe_forward)
    monkeypatch.setattr(runner, "select_fit_output_weighted_columns", observe_selection)
    runner.fold_loaded(data, policy())
    assert calls.index("selection") == 6
    assert original.mlp_forward is runner._source_mlp_forward  # No production global patch occurred.


def test_finite_invalid_text_row_replay_difference_is_diagnostic():
    data = tensor_data()
    data["forwards"][-1]["txt"]["y"][0, -1, 0] += 1
    result = runner.fold_loaded(data, policy())
    assert result["source_replay_diagnostic_passed"] is False
    assert result["source_replay"]["txt"][-1]["diagnostic_target_met"] is False
    assert_nonaccepting(result)


def test_fit_only_selection_and_actual_targets_plus_parent_bias(monkeypatch):
    data = tensor_data()
    select, solve = runner.select_fit_output_weighted_columns, runner.law_abiding_solve
    selections, targets, priors = [], [], []
    def inspect_select(features, native_w2, keep):
        stream = runner.STREAMS[len(selections)]
        expected = []
        for item in data["forwards"]:
            if item["split"] == "fit":
                hidden, _ = original.mlp_forward(item[stream]["x"], original.mlp_weights(data["frozen"], 0, stream, "cpu"))
                expected.append(hidden.squeeze(0)[item[stream]["valid"]])
        assert torch.equal(features, torch.cat(expected))
        assert native_w2.dtype == torch.bfloat16
        chosen = select(features, native_w2, keep)
        selections.append(chosen)
        return chosen
    def inspect_solve(design, target, prior):
        assert torch.equal(design[:, -1], torch.ones(len(design), dtype=torch.float64))
        targets.append(target.clone()); priors.append(prior.clone())
        return solve(design, target, prior)
    monkeypatch.setattr(runner, "select_fit_output_weighted_columns", inspect_select)
    monkeypatch.setattr(runner, "law_abiding_solve", inspect_solve)
    runner.fold_loaded(data, policy())
    for stream, chosen, target, prior in zip(runner.STREAMS, selections, targets, priors):
        expected = torch.cat([i[stream]["y"].squeeze(0)[i[stream]["valid"]] for i in data["forwards"] if i["split"] == "fit"]).double()
        assert torch.equal(target, expected)
        prefix = f"transformer_blocks.0.{stream}_mlp."
        assert torch.equal(prior[:, :-1], data["frozen"][prefix + "net.2.weight"].bfloat16().double()[:, chosen])
        assert torch.equal(prior[:, -1], data["frozen"][prefix + "net.2.bias"].bfloat16().double())
    assert original.law_abiding_solve is solve


def test_changed_heldout_never_changes_selection_or_refit():
    data = tensor_data()
    first = runner.fold_loaded(data, policy())
    changed = copy.deepcopy(data)
    for stream in runner.STREAMS:
        item = changed["forwards"][-1][stream]
        item["x"] = (item["x"].float() * 1.3 + .2).bfloat16()
        _, item["y"] = original.mlp_forward(item["x"], original.mlp_weights(changed["frozen"], 0, stream, "cpu"))
    second = runner.fold_loaded(changed, policy())
    for stream in runner.STREAMS:
        assert first["streams"][stream]["selected_channel_indices"] == second["streams"][stream]["selected_channel_indices"]
        assert first["streams"][stream]["ephemeral_refit_sha256"] == second["streams"][stream]["ephemeral_refit_sha256"]


def test_stored_update_does_not_trust_solver_notes_and_is_diagnostic(monkeypatch):
    monkeypatch.setattr(runner, "law_abiding_solve", lambda design, target, prior: (prior * 3, {"update_scale": 0}))
    result = runner.fold_loaded(tensor_data(), policy())
    assert all(v["stored_update_scale"] > 1 and v["stored_update_target_met"] is False for v in result["streams"].values())
    assert_nonaccepting(result)


@pytest.mark.parametrize("problem", ["fourfold", "fit_rows", "held_rows"])
def test_conditioning_and_fourfold_targets_are_diagnostic_except_empty_rows(monkeypatch, problem):
    data, p = tensor_data(), policy()
    if problem == "fourfold": p["streams"]["img"]["keep"] = 8
    elif problem == "fit_rows": data["row_counts"]["txt"]["fit"] = 1
    else: data["row_counts"]["img"]["holdout"] = 0
    if problem == "held_rows":
        monkeypatch.setattr(runner, "mlp_forward", lambda *a: pytest.fail("empty evidence reached replay"))
        with pytest.raises(runner.DevelopmentRejected): runner.fold_loaded(data, p)
    else:
        result = runner.fold_loaded(data, p)
        stream, field = ("img", "fourfold_target_met") if problem == "fourfold" else ("txt", "fit_conditioning_target_met")
        assert result["row_preflight"][stream][field] is False
        assert_nonaccepting(result)


@pytest.mark.parametrize("stream", runner.STREAMS)
@pytest.mark.parametrize("field,value", [("max_source_replay_rel_l2", .006), ("max_heldout_rel_l2", .31), ("min_heldout_cos", .94)])
def test_current_api_preserves_finite_reference_values_without_enforcement(monkeypatch, stream, field, value):
    p = policy(); p["streams"][stream][field] = value
    result = runner.fold_loaded(tensor_data(), p)
    assert result["policy"]["streams"][stream][field] == value
    assert result["owner_policy"]["finite_numeric_targets"] == "diagnostic_only"
    assert_nonaccepting(result)


def test_original_tensors_policy_and_production_globals_unchanged():
    data, p = tensor_data(), policy()
    before, p_before = copy.deepcopy(data), copy.deepcopy(p)
    globals_before = {name: getattr(original, name) for name in ("fold_loaded", "law_abiding_solve", "pivoted_columns_gram", "mlp_forward")}
    runner.fold_loaded(data, p)
    assert p == p_before
    for name, value in data["frozen"].items(): assert torch.equal(value.view(torch.uint8), before["frozen"][name].view(torch.uint8))
    for item, saved in zip(data["forwards"], before["forwards"]):
        for stream in runner.STREAMS:
            for key in ("x", "y", "valid"): assert torch.equal(item[stream][key], saved[stream][key])
    assert all(getattr(original, name) is value for name, value in globals_before.items())


def test_tensor_metadata_cannot_be_returned_as_diagnostics():
    data = tensor_data()
    data["capture_receipts"] = [{"unexpected_tensor": torch.ones(2)}]
    with pytest.raises(runner.DevelopmentRejected, match="never tensors") as exc:
        runner.fold_loaded(data, policy())
    assert exc.value.receipt["status"] == "development_refused_receipt_contract"
    json.dumps(exc.value.receipt, allow_nan=False)


@pytest.mark.parametrize("stage", ["source", "candidate"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_native_arithmetic_explicitly_refuses(stage, bad, monkeypatch):
    data = tensor_data()
    forward = runner._source_mlp_forward
    calls = []
    def inject(x, weights):
        hidden, prediction = forward(x, weights)
        calls.append(1)
        if stage == "source" or len(calls) > 6:
            prediction = prediction.clone(); prediction[0, 0, 0] = bad
        return hidden, prediction
    monkeypatch.setattr(runner, "_source_mlp_forward", inject)
    with pytest.raises(runner.DevelopmentRejected) as exc:
        runner.fold_loaded(data, policy())
    assert exc.value.receipt["status"] == "development_rejected_nonfinite_arithmetic"
    assert exc.value.receipt["nonfinite_arithmetic_refused"] is True
    json.dumps(exc.value.receipt, allow_nan=False)
    assert_nonaccepting(exc.value.receipt)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_undefined_derived_score_is_null_and_never_mislabeled_pass(bad, monkeypatch):
    monkeypatch.setattr(runner, "_source_score", lambda *a: {"rel_l2": bad, "cos": 1.0})
    result = runner.fold_loaded(tensor_data(), policy())
    assert result["source_replay_diagnostic_passed"] is False
    assert result["development_numeric_gates_passed"] is False
    assert result["diagnostic_measurements_complete"] is False
    assert result["source_replay"]["img"][0]["scores"]["rel_l2"] is None
    assert result["source_replay"]["img"][0]["scores"]["undefined_diagnostics"]
    json.dumps(result, allow_nan=False)
    assert_nonaccepting(result)


@pytest.mark.parametrize("scope,call_index", [("per_forward", 9), ("pooled", 11)])
@pytest.mark.parametrize("field,value", [("rel_l2", .31), ("cos", .94)])
def test_each_development_numeric_target_is_observed_without_refusal(scope, call_index, field, value, monkeypatch):
    source_score = runner._source_score
    calls = []
    def inject(*args):
        result = source_score(*args); calls.append(1)
        if len(calls) == call_index: result[field] = value
        return result
    monkeypatch.setattr(runner, "_source_score", inject)
    result = runner.fold_loaded(tensor_data(), policy())
    assert result["status"] == "development_complete"
    assert result["streams"]["img"]["development_numeric_gates_passed"] is False
    assert result["streams"]["txt"]["development_numeric_gates_passed"] is True
    if scope == "per_forward":
        assert result["streams"]["img"]["heldout_scores"]["rel_l2"] <= .30
        assert result["streams"]["img"]["heldout_scores"]["cos"] >= .95
    assert_nonaccepting(result)


def strict_bundle(tmp_path):
    path, receipt, policy_path, p = bundle(tmp_path)
    for entry in p["streams"].values():
        entry.update(max_source_replay_rel_l2=.005, max_heldout_rel_l2=.30, min_heldout_cos=.95)
    dump(policy_path, p)
    return path, receipt, policy_path, p


def test_original_verified_loader_validate_only_is_nonaccepting_and_json_only(tmp_path, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    report = tmp_path / "research" / "validate.json"
    monkeypatch.setattr(runner, "fold_loaded", lambda *a, **k: pytest.fail("validate-only launched refold"))
    result = runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, validate_only=True, fixture=True)
    stored = json.loads(report.read_text())
    assert stored == result and stored["status"] == "development_inputs_validated"
    assert_nonaccepting(stored)
    assert stored["source_replay_performed"] is stored["fold_launched"] is False
    assert stored["capture_receipts"][0]["sha256"] == original.file_sha(path)
    assert sorted(p.name for p in report.parent.iterdir()) == ["validate.json"]


def test_verified_fixture_refold_persists_complete_or_refused_diagnostics_only(tmp_path):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    before = {p: original.file_sha(p) for p in path.parent.iterdir() if p.is_file()}
    report = tmp_path / "research" / "fold.json"
    try:
        result = runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    except runner.DevelopmentRejected as exc:
        result = exc.receipt
    stored = json.loads(report.read_text())
    assert stored == result
    assert_nonaccepting(stored)
    assert stored["recipe_id"] == runner.RECIPE_ID
    assert sorted(p.name for p in report.parent.iterdir()) == ["fold.json"]
    assert all(original.file_sha(p) == sha for p, sha in before.items())
    assert stored["implementation_sha256"]["c16_qimg_development_refold.py"] == original.file_sha(Path(runner.__file__))
    assert stored["implementation_sha256"]["mediafold/experimental_fit_selector.py"] == original.file_sha(runner.MEDIA / "mediafold/experimental_fit_selector.py")


def test_verified_sparse_fixture_numeric_pass_persists_nonaccepting_json_only(tmp_path):
    # All files here are new tiny synthetic fixtures, not historical captures.
    path, body, policy_path, p = strict_bundle(tmp_path)
    # CPU safetensors can return file-backed tensors. Detach before this fixture-
    # only same-path rewrite so truncating a file cannot clobber its mapped input.
    frozen = {name: tensor.clone() for name, tensor in original.load_file(
        str(path.parent / body["frozen_source"]["file"])).items()}
    for stream in runner.STREAMS:
        key = f"transformer_blocks.0.{stream}_mlp.net.2.weight"
        w = torch.zeros_like(frozen[key], dtype=torch.float32)
        w[:, [3, 7]] = frozen[key].float()[:, [3, 7]]
        frozen[key] = w.to(torch.float8_e4m3fn)
        p["streams"][stream]["keep"] = 2
    body["frozen_source"] = original.write_tensors(path.parent / body["frozen_source"]["file"], frozen)
    for record in body["captures"]:
        state = {name: tensor.clone() for name, tensor in original.load_file(
            str(path.parent / record["file"])).items()}
        preserved = {name: original.tensor_sha(state[name]) for name in ("sigma", "img_mlp_in", "txt_mlp_in")}
        for stream in runner.STREAMS:
            _, state[stream + "_mlp_out"] = original.mlp_forward(state[stream + "_mlp_in"], original.mlp_weights(frozen, 0, stream, "cpu"))
        record.update(original.write_tensors(path.parent / record["file"], state))
        reloaded = original.load_file(str(path.parent / record["file"]))
        assert all(original.tensor_sha(reloaded[name]) == sha for name, sha in preserved.items())
    assert len({record["file_sha256"] for record in body["captures"]}) == len(body["captures"])
    dump(path, body); dump(policy_path, p)
    source_hashes = {f: original.file_sha(f) for f in path.parent.iterdir() if f.is_file()}
    report = tmp_path / "research" / "passed_but_nonaccepting.json"
    result = runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    assert result["status"] == "development_complete" and result["development_numeric_gates_passed"] is True
    assert json.loads(report.read_text()) == result
    assert_nonaccepting(result)
    assert sorted(f.name for f in report.parent.iterdir()) == [report.name]
    assert all(original.file_sha(f) == sha for f, sha in source_hashes.items())


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_undefined_derived_diagnostic_is_durably_recorded_as_null(tmp_path, bad, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    monkeypatch.setattr(runner, "_source_score", lambda *a: {"rel_l2": bad, "cos": 1.0})
    report = tmp_path / "research" / "undefined.json"
    expected = runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    result = json.loads(report.read_text(), parse_constant=lambda value: pytest.fail("non-finite JSON emitted"))
    assert result == expected
    assert result["status"] == "development_complete"
    assert result["diagnostic_measurements_complete"] is False
    assert result["development_numeric_gates_passed"] is False
    assert_nonaccepting(result)


def test_finite_fitted_overflow_on_bf16_storage_is_durable_refusal(tmp_path, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    monkeypatch.setattr(runner, "law_abiding_solve", lambda design, target, prior:
        (torch.full_like(prior, 1e300), {"update_scale": 0.0}))
    report = tmp_path / "research" / "bf16_overflow.json"
    with pytest.raises(runner.DevelopmentRejected) as exc:
        runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    result = json.loads(report.read_text(), parse_constant=lambda value: pytest.fail("nonfinite JSON emitted"))
    assert result == exc.value.receipt
    assert result["status"] == "development_rejected_nonfinite_bf16_storage"
    assert result["stored_update_scale"] is None
    assert result["nonfinite_arithmetic_refused"] is True
    assert_nonaccepting(result)


def test_undefined_solver_notes_do_not_prevent_finite_artifact_measurement(tmp_path, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    monkeypatch.setattr(runner, "law_abiding_solve", lambda design, target, prior:
        (prior.clone(), {"update_scale": 0.0, "ladder": [{"rel_l2": float("nan")}]}))
    report = tmp_path / "research" / "solver_undefined.json"
    runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    result = json.loads(report.read_text(), parse_constant=lambda value: pytest.fail("nonfinite JSON emitted"))
    assert result["status"] == "development_complete"
    assert result["diagnostic_measurements_complete"] is False
    assert result["streams"]["img"]["solve"]["ladder"][0]["rel_l2"] is None
    assert_nonaccepting(result)


@pytest.mark.parametrize("failure", ["policy_hash", "capture_hash", "invalid_reference", "missing_development"])
def test_durable_input_refusal(tmp_path, failure):
    path, body, policy_path, p = strict_bundle(tmp_path)
    sha = original.file_sha(policy_path)
    explicit = True
    if failure == "policy_hash": sha = "0" * 64
    elif failure == "capture_hash":
        body["captures"][0]["file_sha256"] = "0" * 64; dump(path, body)
    elif failure == "invalid_reference":
        p["streams"]["txt"]["min_heldout_cos"] = 1.1; dump(policy_path, p); sha = original.file_sha(policy_path)
    else: explicit = False
    report = tmp_path / "research" / "refused.json"
    with pytest.raises(runner.DevelopmentRejected) as exc:
        runner.run_development([path], policy_path, sha, report, development_only=explicit, fixture=True)
    assert json.loads(report.read_text()) == exc.value.receipt
    assert_nonaccepting(exc.value.receipt)
    assert sorted(p.name for p in report.parent.iterdir()) == ["refused.json"]


def test_existing_report_refuses_before_tensor_load(tmp_path, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    report = tmp_path / "historical.json"; report.write_text("preserve")
    monkeypatch.setattr(runner, "load_captures", lambda *a, **k: pytest.fail("existing report reached tensor loading"))
    with pytest.raises(ValueError, match="new file"):
        runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    assert report.read_text() == "preserve"


def test_report_source_overlap_refuses_before_loading(tmp_path, monkeypatch):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    report = path.parent / "new_report.json"
    monkeypatch.setattr(runner, "load_captures", lambda *a, **k: pytest.fail("source overlap reached loading"))
    with pytest.raises(ValueError, match="overlaps"):
        runner.run_development([path], policy_path, original.file_sha(policy_path), report, development_only=True, fixture=True)
    assert not report.exists()


def test_real_cli_refuses_fixture_and_stdout_follows_durable_receipt(tmp_path, capsys):
    path, _, policy_path, _ = strict_bundle(tmp_path)
    report = tmp_path / "research" / "cli.json"
    code = runner.main(["--development-only", "--captures", str(path), "--policy", str(policy_path),
        "--policy-sha256", original.file_sha(policy_path), "--report", str(report)])
    stdout = json.loads(capsys.readouterr().out)
    assert code == 2 and report.is_file()
    assert stdout["report_sha256"] == original.file_sha(report)
    assert_nonaccepting(stdout)
    stored = json.loads(report.read_text())
    assert "authentic capture" in stored["rejection_reason"]


def test_cli_has_no_accepted_output_or_fixture_override(capsys):
    with pytest.raises(SystemExit) as exc: runner.main(["--help"])
    help_text = capsys.readouterr().out
    assert exc.value.code == 0
    assert "--development-only" in help_text and "--report" in help_text
    assert "--out " not in help_text and "--fixture" not in help_text
