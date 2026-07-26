from __future__ import annotations

import ast
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


def test_exact4_runner_inventories_parent_geometry_once_and_forwards_it() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
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

    assert len(imported_inventory_helpers) == 1
    assert len(inventory_calls) == 1
    assert len(semantic_stage_calls) == 1
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is inventory_calls[0]
    )
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    inventory_name = assignment.targets[0].id
    stage_keywords = {
        keyword.arg: keyword.value for keyword in semantic_stage_calls[0].keywords
    }
    forwarded_operators = stage_keywords["coordinate_completion_parent_operators"]
    assert isinstance(forwarded_operators, ast.Attribute)
    assert forwarded_operators.attr == "operators"
    assert isinstance(forwarded_operators.value, ast.Name)
    assert forwarded_operators.value.id == inventory_name
    forwarded_recipe = stage_keywords["coordinate_completion_recipe"]
    assert isinstance(forwarded_recipe, ast.Name)
    assert forwarded_recipe.id == "coordinate_completion_recipe"


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
    assert "capture" not in source.lower()
