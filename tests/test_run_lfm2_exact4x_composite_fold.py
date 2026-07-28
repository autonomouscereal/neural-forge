from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments import run_lfm2_exact4x_composite_fold as runner  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_lfm2_exact4x_composite_fold.py"
LAUNCH = (
    ROOT / "experiments" / "remote_launch_lfm2_exact4x_fp32_directional_20260724.sh"
)


def test_failure_exception_chain_receipt_preserves_bounded_causes() -> None:
    leaf = ValueError("leaf")
    middle = KeyError("middle")
    middle.__cause__ = leaf
    root = RuntimeError("root")
    root.__cause__ = middle

    receipt = runner._exception_chain_receipt(root)

    assert receipt["chain_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "chain_sha256"}
    )
    assert receipt["entry_count"] == 3
    assert receipt["truncated"] is False
    assert receipt["cycle_detected"] is False
    assert receipt["traceback_source_lines_serialized"] is False
    assert receipt["traceback_locals_serialized"] is False
    assert [entry["relationship"] for entry in receipt["entries"]] == [
        "root",
        "cause",
        "cause",
    ]
    assert [entry["exception_type"] for entry in receipt["entries"]] == [
        "RuntimeError",
        "KeyError",
        "ValueError",
    ]


def test_failure_exception_chain_receipt_bounds_messages_and_cycles() -> None:
    root = RuntimeError("x" * (runner.FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS + 1))
    root.__cause__ = root

    receipt = runner._exception_chain_receipt(root)

    assert receipt["entry_count"] == 1
    assert receipt["cycle_detected"] is True
    assert receipt["truncated"] is False
    assert receipt["entries"][0]["message_truncated"] is True
    assert len(receipt["entries"][0]["message"]) == (
        runner.FAILURE_EXCEPTION_MESSAGE_MAXIMUM_CHARACTERS
    )


def test_failure_exception_chain_receipt_retains_only_frame_locations() -> None:
    try:
        raise ValueError("located")
    except ValueError as exc:
        receipt = runner._exception_chain_receipt(exc)

    entry = receipt["entries"][0]
    assert entry["traceback_frame_count"] == 1
    assert entry["traceback_frames_truncated"] is False
    assert entry["traceback_source_lines_serialized"] is False
    assert entry["traceback_locals_serialized"] is False
    assert entry["traceback_frames"][0]["source"].endswith(
        "test_run_lfm2_exact4x_composite_fold.py"
    )
    assert entry["traceback_frames"][0]["function"] == (
        "test_failure_exception_chain_receipt_retains_only_frame_locations"
    )


def _sealed_pre_auth_receipt() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "synthetic-semantic-layer-stack-v1",
        "status": "sealed-before-runner-authentication",
        "nested": {"layer": 0},
    }
    return {**body, "result_sha256": canonical_sha256(body)}


@pytest.mark.parametrize("fail", (False, True), ids=("success", "failure"))
def test_runner_persists_pre_auth_receipt_on_success_and_failure(
    tmp_path: Path,
    fail: bool,
) -> None:
    receipt = _sealed_pre_auth_receipt()
    original = RuntimeError("original-fold-failure")

    def stage_body(sink: dict[str, object]) -> None:
        sink.update(receipt)
        if fail:
            raise original

    if fail:
        with pytest.raises(RuntimeError) as captured:
            with runner._semantic_layer_stack_pre_auth_capture(tmp_path) as sink:
                stage_body(sink)
        assert captured.value is original
    else:
        with runner._semantic_layer_stack_pre_auth_capture(tmp_path) as sink:
            stage_body(sink)

    persisted = json.loads(
        (tmp_path / "semantic-layer-stack-pre-auth.json").read_text(encoding="ascii")
    )
    assert persisted == receipt
    body = dict(persisted)
    result_sha256 = body.pop("result_sha256")
    assert result_sha256 == canonical_sha256(body)


def test_runner_pre_auth_persistence_failure_does_not_mask_fold_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "semantic-layer-stack-pre-auth.json"
    path.write_text("already-owned\n", encoding="ascii")
    original = RuntimeError("original-fold-failure")

    with pytest.raises(RuntimeError) as captured:
        with runner._semantic_layer_stack_pre_auth_capture(tmp_path) as sink:
            sink.update(_sealed_pre_auth_receipt())
            raise original

    assert captured.value is original
    assert path.read_text(encoding="ascii") == "already-owned\n"
    assert any(
        "Could not persist semantic-layer-stack-pre-auth.json" in note
        for note in getattr(original, "__notes__", ())
    )


def test_runner_persists_postfold_reclosure_seeds_without_model_mutation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    archive_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_lfm2_postfold_reclosure_seed_archive"
    ]
    assert len(archive_calls) == 1
    keywords = {item.arg: item.value for item in archive_calls[0].keywords}
    assert set(keywords) == {
        "mapped_parent_tied_table",
        "producer_encoder",
        "deferred_final_w2_seed",
        "foundation_interface_solve_receipt",
        "semantic_layer_stack_receipt",
        "deferred_final_w2_installation_receipt",
    }
    mapped = keywords["mapped_parent_tied_table"]
    assert isinstance(mapped, ast.Name) and mapped.id == "contextual_parent_rows"
    seed = keywords["deferred_final_w2_seed"]
    assert isinstance(seed, ast.Name) and seed.id == "deferred_final_w2_seed"
    assert source.index("deferred_final_w2_seed = (") > source.index(
        "compile_install_lfm2_semantic_layer_stack_stage("
    )
    assert source.index("deferred_final_w2_seed = (") < source.index(
        "install_lfm2_deferred_final_w2_prior("
    )
    assert source.index("write_lfm2_postfold_reclosure_seed_archive(") > source.index(
        "install_lfm2_deferred_final_w2_prior("
    )
    assert 'run_root / "postfold-reclosure-seeds"' in source
    assert 'run_root / "postfold-reclosure-seed-publication.json"' in source
    assert '"child_model_state_unchanged": True' in source
    assert '"postfold_reclosure_seed_archive_manifest_sha256"' in source
    assert '"postfold_reclosure_seed_archive_model_state_unchanged": True' in source


def _diagnostic_stack(*, dtype: str = "float32", level: str = "requested"):
    representability = {
        "selected_level": level,
        "selected_ray_multiplier": 1.0,
        "requested_analytic_ray_multiplier": 1.0,
        "target_dtype_ulp_escalation_applied": False,
        "requested_analytic_dense_delta_l2_norm": 2.0,
        "installed_target_dtype_dense_delta_l2_norm": 1.98,
        "installed_to_requested_analytic_direction_cosine": 0.999,
        "installed_to_requested_analytic_delta_l2_ratio": 0.99,
        "installed_to_requested_analytic_projection_scale": 0.989,
        "selected_changed_coordinate_count": 12,
        "parameter_diagnostics": [{"target_dtype": dtype}],
    }
    compute = {"compute_dtype": "float64"}
    dense = {
        "report_sha256": "d" * 64,
        "compute_execution_receipt": compute,
        "target_dtype_representability": representability,
    }
    linearized = {
        "report_sha256": "l" * 64,
        "weighted_residual_squared_before": 10.0,
        "weighted_residual_squared_at_requested_analytic_ray": 7.0,
        "weighted_residual_squared_at_installed_delta_projected_ray": 7.1,
        "requested_analytic_functional_effect_weighted_residual_reduction": 3.0,
        "installed_delta_projected_ray_functional_effect_weighted_residual_"
        "reduction": 2.9,
    }
    closure = {
        "report_sha256": "c" * 64,
        "dense_transaction_receipt": dense,
        "linearized_residual_diagnostics": linearized,
    }
    return {
        "result_sha256": "s" * 64,
        "global_residual_closure_receipt": closure,
    }


def test_fp32_directional_receipt_binds_direction_and_function_without_gate() -> None:
    receipt = runner._body_fold_realization_receipt(
        _diagnostic_stack(),
        child_dtype="float32",
        requested_representability_level="requested",
    )

    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert receipt["child_materialization_dtype"] == "float32"
    assert receipt["dense_absorption_analytic_compute_dtype"] == "float64"
    assert receipt["causal_head_analytic_compute_dtype"] == "float32"
    assert receipt["materialized_ray_multiplier"] == 1.0
    assert receipt["requested_unit_ray_preserved_without_ulp_escalation"] is True
    assert receipt["target_dtype_ulp_escalation_applied"] is False
    assert receipt["installed_to_requested_direction_cosine"] == 0.999
    assert (
        receipt["requested_analytic_functional_effect_weighted_residual_reduction"]
        == 3.0
    )
    assert (
        receipt["installed_projected_ray_functional_effect_weighted_residual_reduction"]
        == 2.9
    )
    assert receipt["direction_and_function_diagnostics_are_advisory_only"] is True
    assert (
        receipt["diagnostic_numbers_have_accept_reject_delete_stop_authority"] is False
    )
    assert receipt["float32_storage_is_not_a_quantization_claim"] is True
    assert all(value == 0 for value in receipt["operation_audit"].values())


def test_fp32_directional_receipt_rejects_first_effectful_execution_policy() -> None:
    stack = _diagnostic_stack(level="first_effectful")
    with pytest.raises(
        runner.Lfm2Exact4xCompositeFoldError,
        match="requested unit ray",
    ):
        runner._body_fold_realization_receipt(
            stack,
            child_dtype="float32",
            requested_representability_level="first_effectful",
        )


def test_integrated_runner_parameterizes_same_child_checkpoint_and_eval_dtype() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    actions = {value.dest: value for value in runner._parser()._actions}

    assert actions["child_dtype"].choices == ("bfloat16", "float32")
    assert runner.EXPECTED_CHILD_UNIQUE_PARAMETERS == 57_422_440
    assert runner.EXPECTED_CHILD_TOPOLOGY == {
        "vocab_size": 4_096,
        "hidden_size": 776,
        "intermediate_size": 700,
        "num_hidden_layers": 14,
        "num_attention_heads": 10,
        "num_key_value_heads": 5,
        "head_dim": 64,
        "tie_word_embeddings": False,
    }
    assert "dtype=child_torch_dtype" in source
    assert "dtype=child_dtype_name" in source
    assert 'run_root / "body-fold-realization.json"' in source
    assert "compiled_attention_candidate_sink=structural_attention_candidates" in (
        source
    )
    assert 'run_root / "structural-attention-candidates"' in source
    assert "structural_attention_candidate_archive_manifest_sha256" in source
    assert '"parameters_quantized": 0' in source
    assert '"parent_trace_recaptures": 0' in source
    assert '"diagnostic_numbers_have_accept_reject_delete_stop_authority": False' in (
        source
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "review_frontier"
        for node in ast.walk(tree)
    )


def test_exact4_parser_defaults_to_body_global_coordinate_completion() -> None:
    actions = {value.dest: value for value in runner._parser()._actions}

    assert actions["short_conv_construction"].default == (
        "body-global-coordinate-complete-structural-v1"
    )
    assert actions["coordinate_completion_recipe"].default == (
        "bridge-procrustes-isometric"
    )
    assert actions["coordinate_completion_recipe"].choices == (
        "bridge-procrustes-isometric",
    )
    assert actions["normalized_input_correction_rho"].default == 0.25


def test_exact4_runner_preaggregates_parent_geometry_once_and_releases_inventory() -> (
    None
):
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_inventory_helpers = [
        alias
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "neural_forge.hf_lfm2_weight_complete_basis"
        for alias in node.names
        if alias.name == "inventory_lfm2_weight_complete_operators"
    ]
    inventory_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "inventory_lfm2_weight_complete_operators"
    ]
    semantic_stage_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compile_install_lfm2_semantic_layer_stack_stage"
    ]
    geometry_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compile_role_balanced_operator_geometry"
    ]

    assert len(imported_inventory_helpers) == 1
    assert len(inventory_calls) == 1
    assert len(geometry_calls) == 1
    assert len(semantic_stage_calls) == 1
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is inventory_calls[0]
    )
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    inventory_name = assignment.targets[0].id
    geometry_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is geometry_calls[0]
    )
    assert len(geometry_assignment.targets) == 1
    assert isinstance(geometry_assignment.targets[0], ast.Name)
    geometry_name = geometry_assignment.targets[0].id
    geometry_keywords = {
        keyword.arg: keyword.value for keyword in geometry_calls[0].keywords
    }
    assert isinstance(geometry_calls[0].args[0], ast.Attribute)
    assert geometry_calls[0].args[0].attr == "operators"
    assert isinstance(geometry_calls[0].args[0].value, ast.Name)
    assert geometry_calls[0].args[0].value.id == inventory_name
    assert isinstance(geometry_keywords["compute_device"], ast.Name)
    assert geometry_keywords["compute_device"].id == "fold_execution_profile"

    inventory_deletes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        and any(
            isinstance(target, ast.Name) and target.id == inventory_name
            for target in node.targets
        )
    ]
    assert len(inventory_deletes) == 1
    assert (
        inventory_calls[0].lineno
        < geometry_calls[0].lineno
        < inventory_deletes[0].lineno
        < semantic_stage_calls[0].lineno
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id == inventory_name
        and isinstance(node.ctx, ast.Load)
        and node.lineno > inventory_deletes[0].lineno
        for node in ast.walk(tree)
    )
    stage_keywords = {
        keyword.arg: keyword.value for keyword in semantic_stage_calls[0].keywords
    }
    assert "coordinate_completion_parent_operators" not in stage_keywords
    forwarded_geometry = stage_keywords["coordinate_completion_parent_geometry"]
    assert isinstance(forwarded_geometry, ast.Name)
    assert forwarded_geometry.id == geometry_name
    forwarded_profile = stage_keywords[
        "coordinate_completion_least_squares_execution_profile"
    ]
    assert isinstance(forwarded_profile, ast.Name)
    assert forwarded_profile.id == "analytic_execution_profile"
    forwarded_recipe = stage_keywords["coordinate_completion_recipe"]
    assert isinstance(forwarded_recipe, ast.Name)
    assert forwarded_recipe.id == "coordinate_completion_recipe"
    forwarded_rho = stage_keywords["normalized_input_correction_rho"]
    assert isinstance(forwarded_rho, ast.Name)
    assert forwarded_rho.id == "normalized_input_correction_rho"
    assert source.index('run_root / "body-global-parent-geometry.json"') < (
        source.index("compile_install_lfm2_semantic_layer_stack_stage(")
    )
    assert '"parent_operator_inventory_count": 1' in source
    assert '"parent_geometry_compile_count": 1' in source
    assert '"parent_operator_reconstruction_after_geometry_compile": False' in source
    assert '"body_global_parent_geometry_sha256"' in source
    assert '"fold_execution_profile_sha256"' in source
    assert '"analytic_execution_profile_sha256"' in source


def test_exact4_runner_resolves_two_production_cuda_profiles_without_fallback() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    profile_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_analytic_execution_profile"
    ]

    assert len(profile_calls) == 2
    assert {
        call.args[0].id
        for call in profile_calls
        if call.args and isinstance(call.args[0], ast.Name)
    } == {"fold_compute_device", "analytic_solve_device"}
    for call in profile_calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert isinstance(keywords["require_cuda"], ast.Constant)
        assert keywords["require_cuda"].value is True
        assert isinstance(keywords["allow_cpu_fallback"], ast.Constant)
        assert keywords["allow_cpu_fallback"].value is False


def test_exact4_production_proof_rejects_legacy_short_conv_constructions() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    rejection_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "LFM2_SEMANTIC_SHORT_CONV_COORDINATE_COMPLETE_CONSTRUCTION"
        in ast.dump(node.test)
        and "coordinate_completion_recipe" in ast.dump(node.test)
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
    ]

    assert len(rejection_guards) == 1
    assert "exact-4x body-global proof requires the explicit coordinate-" in source
    assert "complete short-conv construction and bridge-procrustes-isometric" in (
        source
    )


def test_child_dtype_resolution_is_explicit_and_does_not_change_parameter_count() -> (
    None
):
    assert runner._child_torch_dtype("float32") is torch.float32
    assert runner._child_torch_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(
        runner.Lfm2Exact4xCompositeFoldError,
        match="child_dtype",
    ):
        runner._child_torch_dtype("float16")


def test_exact_fp32_launch_reuses_pinned_a0005_evidence_without_recapture() -> None:
    source = LAUNCH.read_text(encoding="utf-8")

    assert "--child-dtype float32" in source
    assert "--target-dtype-representability-level requested" in source
    assert "--global-residual-basis-rank 128" in source
    assert "--expected-parent-unique-parameters 229693184" in source
    assert "5c1a452cbe6bb78cb0c346bc9ca5d784200aad0b8848847e559cd62fe4852384" in (
        source
    )
    assert "76ff3cbc946ea52b7bc30225877ed344e5b0212307756b8b80ebe78fc2aa7a95" in (
        source
    )
    assert "dd9efbe218b4e6e20c3fd60e56d1d5b72165ca61bcd46794e15b0aded3b9361b" in (
        source
    )
    assert "4fbe0a9750446926c2a63f70d31078af2786971048d4c7f9d45882c4fdeb8dc9" in (
        source
    )
    assert "compile" not in source.lower().replace("compiler-cache", "")
    assert "build_lfm2_cached_causal_evidence.py" not in source
    assert "capture_lfm2" not in source
    assert (
        '--normalized-input-request-capture-directory "${RUN_DIR}/'
        'normalized-input-request-capsules"'
    ) in source


def test_live_child_full_prefix_replay_is_an_explicit_off_by_default_option() -> None:
    parser = runner._parser()
    action = next(
        value
        for value in parser._actions
        if value.dest == "live_child_prefix_recapture_mode"
    )

    assert action.default == runner.LFM2_LIVE_CHILD_PREFIX_RECAPTURE_DISABLED
    assert tuple(action.choices) == runner.LFM2_LIVE_CHILD_PREFIX_RECAPTURE_MODES


def test_output_interface_fit_uses_post_body_public_full_prefix_recapture() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    deferred_install = source.index("install_lfm2_deferred_final_w2_prior(")
    child_recapture = source.index("collect_lfm2_child_causal_residual_evidence(")
    causal_head_compile = source.index("compile_lfm2_causal_residual_token_bridge(")
    assert deferred_install < child_recapture < causal_head_compile
    assert 'child_evidence_receipt["full_prefix_use_cache_false"]' in source
    assert '"output_interface_recap_parent_forward_calls": 0' in source
    assert '"output_interface_recap_parent_trace_recaptures": 0' in source
    assert '"output_interface_recap_parent_cache_regenerations": 0' in source
    assert '"output_interface_recap_training_steps": 0' in source
