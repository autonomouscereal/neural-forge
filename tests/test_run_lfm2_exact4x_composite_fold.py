from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments import run_lfm2_exact4x_composite_fold as runner  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_bridge_only_control import (  # noqa: E402
    _tensor_sha256 as _bridge_tensor_sha256,
)


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
        "body-global-coordinate-complete-fhvr-mf-cap10k-sweeps1-full-history-"
        "full-coverage-v1"
    )
    assert actions["global_residual_closure_version"].default == "auto"
    assert actions["coordinate_completion_recipe"].default == (
        runner.LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE
    )
    assert actions["coordinate_completion_recipe"].choices == (
        "bridge-procrustes-isometric",
        runner.LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
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
    forwarded_balanced = stage_keywords[
        "coordinate_completion_circuit_balanced_evidence"
    ]
    assert isinstance(forwarded_balanced, ast.Name)
    assert forwarded_balanced.id == "circuit_balanced_coordinate_evidence"
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
        and "uses_body_global_coordinate_completion" in ast.dump(node.test)
        and "coordinate_completion_recipe" in ast.dump(node.test)
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
    ]

    assert len(rejection_guards) == 1
    combined = runner.resolve_lfm2_semantic_short_conv_construction(
        actions := runner._parser().get_default("short_conv_construction")
    )
    assert actions == combined.construction_id
    assert combined.uses_body_global_coordinate_completion is True
    assert combined.uses_fhvr_full_coverage is True
    assert "requires an authenticated short-conv" in source
    assert "construction with coordinate-completion capability" in source


def _moving_stack_receipt() -> dict[str, object]:
    body: dict[str, object] = {
        "global_residual_closure_version": "moving_gauge_v3",
        "global_residual_requested_closure_version": "auto",
        "global_residual_selected_closure_version": "moving_gauge_v3",
        "global_residual_closure_selection_authenticated": True,
        "global_residual_moving_gauge_v3": True,
        "global_residual_deferred_physical_finalizer_required": True,
        "moving_gauge_deferred_finalization_plan_sha256": "p" * 64,
        "global_residual_closure_receipt_sha256": "c" * 64,
    }
    return {**body, "result_sha256": canonical_sha256(body)}


def test_closure_selection_binds_auto_request_to_moving_engine() -> None:
    stack = _moving_stack_receipt()

    receipt = runner._global_residual_closure_selection_receipt(
        stack,
        requested_closure_version="auto",
    )

    assert receipt["requested_closure_version"] == "auto"
    assert receipt["selected_closure_version"] == "moving_gauge_v3"
    assert receipt["moving_gauge_selected"] is True
    assert receipt["post_w2_physical_finalizer_required"] is True
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_moving_body_realization_binds_effectful_post_w2_finalization() -> None:
    stack = _moving_stack_receipt()
    selection = runner._global_residual_closure_selection_receipt(
        stack,
        requested_closure_version="auto",
    )
    finalization_body = {
        "semantic_layer_stack_result_sha256": stack["result_sha256"],
        "deferred_finalization_plan_sha256": "p" * 64,
        "final_w2_was_installed_before_moving_publication": True,
        "final_w2_changed_by_moving_finalizer": False,
        "actual_final_recapture_includes_corrected_final_w2": True,
        "model_state_after_sha256": "i" * 64,
        "changed_state_keys": ["model.layers.0.self_attn.o_proj.weight"],
        "actual_final_physical_recapture_sha256": "r" * 64,
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "parent_model_forward_calls": 0,
        "parent_trace_recaptures": 0,
    }
    finalization = {
        **finalization_body,
        "result_sha256": canonical_sha256(finalization_body),
    }

    receipt = runner._moving_gauge_body_fold_realization_receipt(
        stack,
        selection,
        finalization,
        child_dtype="float32",
        requested_representability_level="requested",
        dynamic_model_state_after_sha256="m" * 64,
        lfm2_identity_after_sha256="i" * 64,
    )

    assert receipt["selected_closure_version"] == "moving_gauge_v3"
    assert receipt["physical_target_dtype_effect_observed"] is True
    assert receipt["causal_head_must_be_solved_after_final_physical_body"] is True
    assert receipt["moving_gauge_post_w2_finalization_sha256"] == (
        finalization["result_sha256"]
    )
    assert receipt["final_child_body_dynamic_model_state_sha256"] == "m" * 64
    assert receipt["final_child_body_lfm2_identity_sha256"] == "i" * 64
    assert (
        receipt["final_child_body_dynamic_model_state_sha256"]
        != receipt["final_child_body_lfm2_identity_sha256"]
    )


def test_balanced_recipe_preserves_full_structural_table_under_causal_residual() -> (
    None
):
    source = RUNNER.read_text(encoding="utf-8")

    assert (
        "LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL"
        in source
    )
    assert '"fully_unsupported_output_rows_equal_structural_baseline"' in source
    assert '"fully_unsupported_output_row_logit_drift_max_abs"' in source
    assert '"adaptive_background_retained_every_nonzero_candidate"' in source
    assert '"vocabulary_row_coverage_complete"' in source
    assert '"installed_input_table_sha256"' in source
    assert "_causal_structural_install_foundation_lineage_valid" in source
    assert '"causal-output-bridge-pre-install.json"' in source
    assert '"residual-only-over-full-structural-foundation"' in source


def _sealed_result(body: dict[str, object]) -> dict[str, object]:
    return {**body, "result_sha256": canonical_sha256(body)}


def _causal_structural_lineage_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    str,
    str,
]:
    analytic_foundation = torch.tensor(
        [[0.123456789, -0.987654321], [1.234567891, -1.111111119]],
        dtype=torch.float64,
    )
    materialized_foundation = analytic_foundation.to(dtype=torch.float32)
    analytic_sha256 = _bridge_tensor_sha256(analytic_foundation)
    materialized_sha256 = _bridge_tensor_sha256(materialized_foundation)
    foundation_report_sha256 = "f" * 64
    structural = _sealed_result(
        {
            "schema": runner.LFM2_INDEPENDENT_STRUCTURAL_INTERFACE_INSTALL_SCHEMA,
            "foundation_active_interface_report_sha256": (
                foundation_report_sha256
            ),
            "installed_input_table_sha256": materialized_sha256,
            "installed_output_baseline_sha256": materialized_sha256,
            "input_output_values_equal_at_structural_install": True,
        }
    )
    coordinate = _sealed_result(
        {
            "schema": runner.LFM2_CIRCUIT_BALANCED_COORDINATE_EVIDENCE_SCHEMA,
            "recipe": runner.LFM2_CIRCUIT_BALANCED_COORDINATE_RECIPE,
            "foundation_interface": {
                "report_sha256": foundation_report_sha256,
                "structural_target_table_sha256": analytic_sha256,
                "structural_target_table_preserved": True,
                "target_vocabulary_row_count": 2,
            },
            "token_bridge": {
                "full_target_vocabulary_row_coverage": True,
                "full_target_vocabulary_row_count": 2,
            },
            "causal_output_contract": {
                "full_structural_table_is_foundation": True,
                "partial_causal_trie_may_supply_residual_only": True,
                "partial_causal_trie_may_replace_structural_table": False,
                "unsupported_logit_policy_required": (
                    runner.LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
                ),
                "fully_unsupported_rows_must_equal_structural_baseline": True,
            },
        }
    )
    causal = {
        "unsupported_logit_policy": (
            runner.LFM2_UNSUPPORTED_LOGIT_POLICY_ADAPTIVE_STRONGEST_SURVIVING_BACKGROUND_SEAL
        ),
        "structural_output_table_sha256": materialized_sha256,
        "target_row_count": 2,
    }
    return causal, structural, coordinate, analytic_sha256, materialized_sha256


def test_causal_foundation_lineage_accepts_authenticated_dtype_materialization() -> (
    None
):
    causal, structural, coordinate, analytic_sha256, materialized_sha256 = (
        _causal_structural_lineage_fixture()
    )

    assert analytic_sha256 != materialized_sha256
    assert coordinate["foundation_interface"]["structural_target_table_sha256"] == (
        analytic_sha256
    )
    assert causal["structural_output_table_sha256"] == materialized_sha256
    assert runner._causal_structural_install_foundation_lineage_valid(
        causal,
        structural,
        coordinate,
        target_vocabulary_size=2,
    )


def test_causal_foundation_lineage_rejects_analytic_hash_as_live_baseline() -> None:
    causal, structural, coordinate, analytic_sha256, _materialized_sha256 = (
        _causal_structural_lineage_fixture()
    )
    causal["structural_output_table_sha256"] = analytic_sha256

    assert not runner._causal_structural_install_foundation_lineage_valid(
        causal,
        structural,
        coordinate,
        target_vocabulary_size=2,
    )


def test_causal_foundation_lineage_rejects_resealed_report_or_vocab_tamper() -> None:
    causal, structural, coordinate, _analytic_sha256, _materialized_sha256 = (
        _causal_structural_lineage_fixture()
    )
    structural_body = dict(structural)
    structural_body.pop("result_sha256")
    structural_body["foundation_active_interface_report_sha256"] = "a" * 64
    mismatched_structural = _sealed_result(structural_body)
    assert not runner._causal_structural_install_foundation_lineage_valid(
        causal,
        mismatched_structural,
        coordinate,
        target_vocabulary_size=2,
    )

    coordinate_body = dict(coordinate)
    coordinate_body.pop("result_sha256")
    coordinate_body["foundation_interface"] = {
        **coordinate_body["foundation_interface"],
        "target_vocabulary_row_count": 3,
    }
    mismatched_coordinate = _sealed_result(coordinate_body)
    assert not runner._causal_structural_install_foundation_lineage_valid(
        causal,
        structural,
        mismatched_coordinate,
        target_vocabulary_size=2,
    )


@pytest.mark.parametrize(
    "tamper",
    (
        "analytic-foundation-digest",
        "causal-policy-contract",
        "full-vocabulary-coverage",
        "materialized-output-baseline",
    ),
)
def test_causal_foundation_lineage_rejects_resealed_contract_tamper(
    tamper: str,
) -> None:
    causal, structural, coordinate, _analytic_sha256, _materialized_sha256 = (
        _causal_structural_lineage_fixture()
    )
    if tamper == "materialized-output-baseline":
        structural_body = dict(structural)
        structural_body.pop("result_sha256")
        structural_body["installed_output_baseline_sha256"] = "b" * 64
        structural = _sealed_result(structural_body)
    else:
        coordinate_body = dict(coordinate)
        coordinate_body.pop("result_sha256")
        if tamper == "analytic-foundation-digest":
            coordinate_body["foundation_interface"] = {
                **coordinate_body["foundation_interface"],
                "structural_target_table_sha256": "not-a-digest",
            }
        elif tamper == "causal-policy-contract":
            coordinate_body["causal_output_contract"] = {
                **coordinate_body["causal_output_contract"],
                "unsupported_logit_policy_required": "measure-drift",
            }
        else:
            coordinate_body["token_bridge"] = {
                **coordinate_body["token_bridge"],
                "full_target_vocabulary_row_coverage": False,
            }
        coordinate = _sealed_result(coordinate_body)

    assert not runner._causal_structural_install_foundation_lineage_valid(
        causal,
        structural,
        coordinate,
        target_vocabulary_size=2,
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
    moving_finalizer = source.index(
        "finalize_lfm2_moving_gauge_after_deferred_w2(",
        deferred_install,
    )
    deferred_receipt_write = source.index(
        'run_root / "deferred-final-w2-install.json"',
        deferred_install,
    )
    body_fingerprint = source.index(
        "body_state = fingerprint_model_state(child)",
        deferred_install,
    )
    child_recapture = source.index("collect_lfm2_child_causal_residual_evidence(")
    causal_head_compile = source.index("compile_lfm2_causal_residual_token_bridge(")
    assert (
        deferred_install
        < moving_finalizer
        < deferred_receipt_write
        < body_fingerprint
        < child_recapture
        < causal_head_compile
    )
    assert 'run_root / "moving-gauge-post-w2-finalization.json"' in source
    assert '"moving_gauge_post_w2_finalization_sha256"' in source
    assert 'child_evidence_receipt["full_prefix_use_cache_false"]' in source
    assert '"output_interface_recap_parent_forward_calls": 0' in source
    assert '"output_interface_recap_parent_trace_recaptures": 0' in source
    assert '"output_interface_recap_parent_cache_regenerations": 0' in source
    assert '"output_interface_recap_training_steps": 0' in source


def test_deferred_production_retains_full_body_before_causal_compilation() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    body_fingerprint = source.index("body_state = fingerprint_model_state(child)")
    capsule_publish = source.index(
        "publish_lfm2_pre_causal_body_resume_capsule("
    )
    child_recapture = source.index("collect_lfm2_child_causal_residual_evidence(")

    assert body_fingerprint < capsule_publish < child_recapture
    assert "Lfm2PreCausalBodyResumeBindings(" in source
    assert "serialized_inference_state_inventory(child)" in source
    assert "body_runtime_state_sha256=fingerprint_model_runtime_state(child)" in source
    assert 'output_directory=run_root / "pre-causal-body-resume"' in source
    assert 'run_root / "pre-causal-body-resume-reference.json"' in source
    assert '"causal_output_head_installed": False' in source
    assert '"pre_causal_body_resume_reference"' in source
    assert "pre_causal_capsule_verification_reload_materializations = 1" in source
    assert (
        '"pre_causal_capsule_verification_reload_materializations"'
        in source
    )
    assert '"complete_child_verification_reload_materializations"' in source
