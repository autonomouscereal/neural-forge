"""Explicit synthetic capture-contract/numerical tests, not model capability."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file, load_file

sys.path.insert(0, str(Path(__file__).parent))
import c17_fit_native_charts as tool


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": tool.host.digest(path)}


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(2)
    evidence = {"schema": "neural-forge-omni-evidence-v1", "rows": [
        {"batch_index": 10 + i, "ids": [2 + i, 2, 3, 4], "split": "held" if i == 3 else "fit"} for i in range(4)]}
    e = write_json(tmp_path / "evidence.json", evidence)
    boundary = {"schema": "neural-forge-k3-boundary-map-v2", "sequence_length": 4,
        "evidence_manifest_file_sha256": e["sha256"], "rows": [
            {"row_index": i, "file": "k3_input_ids", "source": "evidence", "split": row["split"],
             "qwen_tokens_real": 4, "k3_tokens_real": 4, "pairs": [[j, j] for j in range(4)]}
            for i, row in enumerate(evidence["rows"])]}
    boundary["boundary_map_sha256"] = tool.host.canonical_sha(boundary)
    b = write_json(tmp_path / "boundary.json", boundary)
    vectors = tool.original_vectors(tool.host.evidence_rows(evidence, boundary, e["sha256"]))
    g = torch.Generator().manual_seed(117)
    host_input = torch.randn(16, 4, generator=g)
    k3_write = torch.randn(16, 6, generator=g)
    c_out, c_in = torch.randn(4, 6, generator=g), torch.randn(6, 4, generator=g)
    k3_input, host_write = host_input @ c_out, k3_write @ c_in
    imported = write_json(tmp_path / "synthetic-import.json", {"schema": "chimera_omni.import.qwen35.v1", "fixture": True,
        "output_gate_waiver": None, "checkpoint": {"index_sha256": "1" * 64, "config_sha256": "2" * 64,
                                                   "directory_shard_files_match_index": True}})
    acceptance = {"version": tool.k3.native.ACCEPTANCE_VERSION, "constants": tool.k3.native.ACCEPTANCE_CONSTANTS, "verdict": "accepted"}
    context = {"synthetic": True, "corpus": "3" * 64}
    accepted = write_json(tmp_path / "synthetic-accepted.json", {"schema": tool.k3.native.RECEIPT_SCHEMA,
        "fixture": True, "resume_context": context, "layers": {"1": {"acceptance": acceptance, "ffn": {"sha256": "4" * 64}}}})
    def capture(family, layer, x, y, extra):
        folder = tmp_path / family
        folder.mkdir()
        path = folder / "boundaries.safetensors"
        schema = tool.host.SCHEMA if family == "host" else tool.k3.SCHEMA
        save_file({**vectors, "ffn_input": x, "ffn_output": y}, str(path), metadata={"schema": schema, "layer": str(layer), "fixture": "true"})
        rec = {"schema": schema, "fixture": True, "gradients_or_training": False,
            "evidence_sha256": e["sha256"], "boundary_sha256": b["sha256"], "code_sha256": {"synthetic-fixture": "a" * 64},
            "layers": [{"layer": layer, "path": path.name, "sha256": tool.host.digest(path), "rows": 16,
                "boundaries": {k: {"shape": list(t.shape), "dtype": str(t.dtype)} for k, t in (("ffn_input", x), ("ffn_output", y))}}], **extra}
        return write_json(folder / "capture_receipt.json", rec)
    hc = capture("host", 0, host_input, host_write, {"consumed_tensor_hashes_verified": True,
        "checkpoint_stat_stable": True, "host_receipt": imported, "index_sha256": "1" * 64, "config_sha256": "2" * 64})
    kc = capture("k3", 1, k3_input, k3_write, {"stream": "folded", "accepted_prefix_replay_exact": True,
        "teacher_reference_verified": True, "teacher_forward_replayed": False, "parent_experts_loaded": False,
        "new_fit": False, "input_stat_stable": True, "accepted_receipt": accepted,
        "accepted_resume_context": context, "installed_ffn_sha256": {"1": "4" * 64}})
    plan = {"schema": tool.PLAN_SCHEMA, "fixture": True, "threads": 2, "chunk": 3, "seed": 13,
        "max_working_bytes": 1024 * 1024, "evidence": e, "boundary_map": b, "host_layer": 0,
        "host_capture": hc, "host_import_receipt": imported,
        "k3_captures": [{"layer": 1, "capture": kc, "accepted_receipt": accepted}]}
    return tmp_path, plan, vectors, (host_input, k3_input, k3_write, host_write), (c_out, c_in)


def invoke(inputs, plan=None, suffix="candidate"):
    root, original, *_ = inputs
    plan = original if plan is None else plan
    pin = write_json(root / f"{suffix}-plan.json", plan)
    out = root / suffix
    return tool.run(Path(pin["path"]), pin["sha256"], out, synthetic_fixture_root=root), out


def test_candidate_fits_actual_input_and_write_directions(inputs):
    record, out = invoke(inputs)
    weights = load_file(str(out / "L00" / "ridge.safetensors"))
    expected_out, expected_in = inputs[-1]
    torch.testing.assert_close(weights["C_out"], expected_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(weights["C_in"], expected_in, rtol=1e-3, atol=1e-3)
    assert not weights["C_out_bias"].any() and not weights["C_in_bias"].any()
    assert (out / "L00" / "candidate_layer.json").is_file()
    assert not (out / "L00" / "layer.json").exists()
    assert record["fixture"] and record["production_install_eligible"] is False
    assert record["acceptance"]["exact_installed_computation_gate"] == "not_run"
    assert record["arms"]["ridge"]["C_out"]["refit_regime"]["design_row_identity_scope"] == "shared_host_pair"
    assert record["arms"]["ridge"]["C_in"]["refit_regime"]["design_row_identity_scope"] == "source_layer_and_host_pair"
    maps = record["arms"]["ridge"]["native_boundary_provenance"]["maps"]
    assert maps["C_in"]["domain"][0]["boundary"] == "ffn_output"
    assert maps["C_out"]["domain"][0]["boundary"] == "ffn_input"
    with pytest.raises(SystemExit, match="native chart boundary provenance|legacy charts"):
        tool.installer.load_chart(str(out), 0, 6, 4, False)


def test_held_values_cannot_change_selected_lambdas_or_weights(inputs):
    _, _, vectors, data, _ = inputs
    x, y = data[:2]
    first, notes = tool.fit_direction([(x, y)], vectors, seed=9, chunk=3)
    changed_x, changed_y = x.clone(), y.clone()
    held = vectors["held"] == 1
    changed_x[held] *= 100
    changed_y[held] *= -17
    second, other = tool.fit_direction([(changed_x, changed_y)], vectors, seed=9, chunk=3)
    assert torch.equal(first, second)
    assert notes["lambda_ladder"] == other["lambda_ladder"]
    assert notes["ridge_rel_selected"] == other["ridge_rel_selected"]
    assert notes["stored_f32_held"] != other["stored_f32_held"]
    assert set(notes["inner_fit_groups"]).isdisjoint(notes["inner_check_groups"])
    assert 3 not in notes["inner_fit_groups"] + notes["inner_check_groups"]


def test_chunked_gram_matches_reviewed_ridge_and_scores(inputs):
    _, _, vectors, data, _ = inputs
    x, y = data[:2]
    fit = vectors["held"] == 0
    a, b = tool.gram([(x, y)], fit, 3)
    actual, lam = tool.solve(a, b, 1e-4)
    expected = tool.legacy.linalg.fit_ridge_linear(x[fit], y[fit], ridge_rel=1e-4, fit_bias=False, holdout_fraction=0)
    torch.testing.assert_close(actual.T, expected.weight, rtol=1e-9, atol=1e-10)
    assert lam == pytest.approx(expected.ridge_lambda, rel=1e-12)
    measured = tool.scores([(x, y)], ~fit, actual, 3)
    reference = tool.legacy.linalg.write_scores(x[~fit].double() @ actual, y[~fit].double())
    for key in measured:
        assert measured[key] == pytest.approx(reference[key], rel=1e-10, abs=1e-12)


def test_pool_reuses_same_group_split_without_held_leakage(inputs):
    _, _, vectors, data, _ = inputs
    x, y = data[:2]
    pooled, notes = tool.fit_direction([(x, y), (x, y)], vectors, seed=2, chunk=4)
    single, one = tool.fit_direction([(x, y)], vectors, seed=2, chunk=4)
    torch.testing.assert_close(pooled, single, rtol=1e-5, atol=1e-6)
    assert notes["fit_rows"] == 2 * one["fit_rows"]
    assert notes["inner_fit_groups"] == one["inner_fit_groups"]


def test_pooled_host_observations_do_not_inflate_design_row_rank_bound(inputs):
    _, _, vectors, data, _ = inputs
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(16, 14, generator=generator)
    y = data[1]
    pairs = [(x, y), (x, y)]
    actual, notes = tool.fit_direction(pairs, vectors, seed=7, chunk=3, shared_host_design=True)
    # 12 original fit host rows are repeated twice: 24 observations cannot
    # establish 14 independent columns. Eight rows remain in inner fit.
    assert notes["fit_rows"] == 24 and notes["design_width"] == 14
    for name, rows in (("refit_regime", 12), ("inner_fit_regime", 8)):
        regime = notes[name]
        assert regime["original_host_pair_rows"] == rows
        assert regime["pooled_observations"] == rows * 2
        assert regime["design_row_identity_upper_bound"] == regime["rank_upper_bound"] == rows
        assert regime["underdetermined"] is True
        assert regime["underdetermined_by_row_count"] is True
        assert regime["numerical_rank"] is None and regime["rank_measured"] is False
    assert notes["underdetermined"] is True
    # Changing the diagnostic interpretation never changes the solve or scores.
    reference, other = tool.fit_direction(pairs, vectors, seed=7, chunk=3)
    assert torch.equal(actual, reference)
    for key in ("lambda_ladder", "ridge_rel_selected", "ridge_lambda", "fp64_fit", "fp64_held",
                "stored_f32_fit", "stored_f32_held", "zero_held"):
        assert notes[key] == other[key]


def test_inner_fit_regime_keeps_its_own_row_shortage(inputs):
    _, _, vectors, data, _ = inputs
    x = torch.randn(16, 10, generator=torch.Generator().manual_seed(37))
    y = data[1]
    _, notes = tool.fit_direction([(x, y), (x, y)], vectors, seed=7, chunk=3, shared_host_design=True)
    assert notes["refit_regime"]["original_host_pair_rows"] == 12
    assert notes["refit_regime"]["underdetermined"] is None
    assert notes["inner_fit_regime"]["original_host_pair_rows"] == 8
    assert notes["inner_fit_regime"]["pooled_observations"] == 16
    assert notes["inner_fit_regime"]["rank_upper_bound"] == 8
    assert notes["inner_fit_regime"]["underdetermined"] is True


def test_enough_row_identities_do_not_claim_full_rank(inputs):
    _, _, vectors, data, _ = inputs
    # All design rows are equal: plenty of identities, actual rank only one.
    x, y = torch.ones_like(data[0]), data[1]
    _, notes = tool.fit_direction([(x, y), (x, y)], vectors, seed=7, chunk=3, shared_host_design=True)
    assert int(torch.linalg.matrix_rank(x.double())) == 1
    assert notes["underdetermined"] is None
    for name in ("refit_regime", "inner_fit_regime"):
        regime = notes[name]
        assert regime["underdetermined_by_row_count"] is False
        assert regime["underdetermined"] is None
        assert regime["full_column_rank_verified"] is False
        assert regime["rank_measured"] is False and regime["numerical_rank"] is None


def test_source_layer_design_rows_remain_only_an_upper_bound(inputs):
    _, _, vectors, data, _ = inputs
    x, y = data[2:]
    mask = vectors["held"] == 0
    regime = tool.design_regime([(x, y), (x.clone(), y)], vectors, mask, shared_host_design=False)
    assert regime["original_host_pair_rows"] == 12
    assert regime["pooled_observations"] == regime["design_row_identity_upper_bound"] == 24
    assert regime["design_row_identity_scope"] == "source_layer_and_host_pair"
    assert regime["underdetermined"] is None and regime["full_column_rank_verified"] is False
    with pytest.raises(tool.Refusal, match="same captured tensor"):
        tool.design_regime([(x, y), (x.clone(), y)], vectors, mask, shared_host_design=True)


@pytest.mark.parametrize("fault", ["held_order", "position", "missing_write", "teacher_stream", "source_gate",
    "host_hash", "host_waiver", "host_preflight", "ffn_identity", "capture_sha", "boundary_sha", "budget", "fixture"])
def test_invalid_source_and_pairing_refuses(inputs, fault):
    root, original, *_ = inputs
    plan = copy.deepcopy(original)
    if fault in ("held_order", "position", "missing_write", "teacher_stream", "source_gate"):
        pin = plan["k3_captures"][0]["capture"]
        rec = json.loads(Path(pin["path"]).read_text())
        if fault in ("teacher_stream", "source_gate"):
            rec["stream" if fault == "teacher_stream" else "accepted_prefix_replay_exact"] = "teacher" if fault == "teacher_stream" else False
        else:
            path = root / "k3" / "boundaries.safetensors"
            payload = load_file(str(path))
            if fault == "held_order":
                payload["held"] = payload["held"].flip(0)
            elif fault == "position":
                payload["k3_position"][0] = 2
            else:
                payload.pop("ffn_output")
            save_file(payload, str(path), metadata={"schema": tool.k3.SCHEMA, "layer": "1", "fixture": "true"})
            rec["layers"][0]["sha256"] = tool.host.digest(path)
        plan["k3_captures"][0]["capture"] = write_json(Path(pin["path"]), rec)
    elif fault in ("host_hash", "host_waiver", "host_preflight"):
        pin = plan["host_import_receipt"]
        imported = json.loads(Path(pin["path"]).read_text())
        if fault == "host_hash":
            imported["checkpoint"]["index_sha256"] = "b" * 64
        elif fault == "host_waiver":
            imported["output_gate_waiver"] = {"waived": True}
        else:
            imported["schema"] = "qf.s3-read-only-preflight.v1"
        plan["host_import_receipt"] = write_json(Path(pin["path"]), imported)
        capture = json.loads(Path(plan["host_capture"]["path"]).read_text())
        capture["host_receipt"] = plan["host_import_receipt"]
        plan["host_capture"] = write_json(Path(plan["host_capture"]["path"]), capture)
    elif fault == "ffn_identity":
        pin = plan["k3_captures"][0]["accepted_receipt"]
        accepted = json.loads(Path(pin["path"]).read_text())
        accepted["layers"]["1"]["ffn"]["sha256"] = "b" * 64
        plan["k3_captures"][0]["accepted_receipt"] = write_json(Path(pin["path"]), accepted)
    elif fault == "capture_sha":
        plan["host_capture"]["sha256"] = "0" * 64
    elif fault == "boundary_sha":
        plan["boundary_map"]["sha256"] = "0" * 64
    elif fault == "budget":
        plan["max_working_bytes"] = 1
    elif fault == "fixture":
        plan["fixture"] = False
    with pytest.raises(tool.Refusal):
        invoke(inputs, plan)
    assert not (root / "candidate" / "L00" / "candidate_layer.json").exists()


def test_production_cannot_accept_synthetic_plan(inputs):
    root, plan, *_ = inputs
    pin = write_json(root / "production.json", plan)
    with pytest.raises(tool.Refusal, match="fixture"):
        tool.run(Path(pin["path"]), pin["sha256"], root / "prod")


def test_factorization_never_increases_lambda_or_falls_back():
    with pytest.raises(tool.Refusal, match="factorization failed"):
        tool.solve(torch.tensor([[1., 3.], [3., 1.]], dtype=torch.float64), torch.eye(2, dtype=torch.float64), 1e-6)
    with pytest.raises(tool.Refusal, match="no finite design energy"):
        tool.solve(torch.zeros(2, 2), torch.eye(2), 1e-6)


def test_one_original_fit_group_is_not_enough(inputs):
    _, _, vectors, data, _ = inputs
    vectors = {k: v.clone() for k, v in vectors.items()}
    vectors["evidence_row_index"][vectors["held"] == 0] = 0
    with pytest.raises(tool.Refusal, match="two ORIGINAL fit"):
        tool.fit_direction([data[:2]], vectors)


@pytest.mark.parametrize("fault", [None, "prefix", "missing_layer", "not_exact"])
def test_selected_window_layer_has_accepted_prefix_evidence(inputs, fault):
    root, plan, *_ = inputs
    plan = copy.deepcopy(plan)
    item = plan["k3_captures"][0]
    accepted = json.loads(Path(item["accepted_receipt"]["path"]).read_text())
    accepted["layers"]["1"]["prefix_out_sha256"] = {"folded": "8" * 64}
    item["accepted_receipt"] = write_json(Path(item["accepted_receipt"]["path"]), accepted)
    captured = json.loads(Path(item["capture"]["path"]).read_text())
    captured["accepted_receipt"] = item["accepted_receipt"]
    captured["final_window_verification"] = {"all_folded_prefixes_exact": fault != "not_exact",
                                            "layers": [] if fault == "missing_layer" else [1]}
    captured["layers"][0]["accepted_folded_prefix_exact"] = True
    captured["layers"][0]["folded_prefix_sha256"] = ("0" if fault == "prefix" else "8") * 64
    item["capture"] = write_json(Path(item["capture"]["path"]), captured)
    if fault is None:
        record, _ = invoke(inputs, plan)
        assert record["production_install_eligible"] is False
    else:
        with pytest.raises(tool.Refusal, match="window-prefix"):
            invoke(inputs, plan)


def test_no_output_acceptance_when_source_changes_during_fit(inputs, monkeypatch):
    root, _, *_ = inputs
    original = tool.fit_direction
    changed = False
    def mutate(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            path = root / "candidate-plan.json"
            path.write_text(path.read_text() + " ", encoding="utf-8")
            changed = True
        return result
    monkeypatch.setattr(tool, "fit_direction", mutate)
    with pytest.raises(tool.Refusal, match="changed during"):
        invoke(inputs)
    assert not (root / "candidate" / "L00" / "candidate_layer.json").exists()
