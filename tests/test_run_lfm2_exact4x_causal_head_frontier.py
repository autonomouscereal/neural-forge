from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from experiments import run_lfm2_exact4x_causal_head_frontier as frontier


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run_lfm2_exact4x_causal_head_frontier.py"
INTEGRATED = ROOT / "experiments" / "run_lfm2_exact4x_composite_fold.py"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assignment_literal(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Missing assignment {name!r}.")


def _call_nodes(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == name
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        )
    ]


def test_resume_runner_has_six_factor_isolated_all_retained_recipes() -> None:
    source = _source(RUNNER)
    tree = ast.parse(source)
    recipes = _assignment_literal(tree, "CAUSAL_HEAD_RECIPES")

    assert isinstance(recipes, tuple)
    assert len(recipes) == 6
    assert len({recipe["candidate_id"] for recipe in recipes}) == 6
    assert {recipe["residual_scale_ppm"] for recipe in recipes} == {
        0,
        500_000,
        1_000_000,
    }
    assert {recipe["ridge_fraction"] for recipe in recipes} == {
        1e-3,
        1e-2,
        1e-1,
    }
    assert {recipe["row_norm_calibration_ppm"] for recipe in recipes} == {
        250_000,
        1_000_000,
    }
    assert all(recipe["protect_specials"] is True for recipe in recipes)
    assert '"all_candidates_materialized": True' in source
    assert '"diagnostic_numbers_have_decision_authority": False' in source
    assert '"automatic_behavior_acceptance_or_rejection_performed": False' in source


def test_resume_runner_replays_parent_and_child_evidence_once_without_refold() -> None:
    source = _source(RUNNER)
    tree = ast.parse(source)

    assert len(_call_nodes(tree, "open_lfm2_circuit_trace_cache")) == 1
    assert len(_call_nodes(tree, "collect_lfm2_terminal_cache_evidence")) == 1
    assert len(_call_nodes(tree, "compile_lfm2_causal_trie_evidence")) == 1
    assert len(_call_nodes(tree, "collect_lfm2_child_causal_residual_evidence")) == 1
    assert len(_call_nodes(tree, "compile_lfm2_causal_residual_token_bridge")) == 1
    assert len(_call_nodes(tree, "install_lfm2_independent_causal_output_head")) == 1
    assert len(_call_nodes(tree, "review_frontier")) == 1

    forbidden = (
        "load_lfm2_direct_local_model",
        "DynamicTargetSynthesizer",
        "fit_lfm2_semantic_foundation_interface",
        "compile_install_lfm2_semantic_layer_stack_stage",
        "migrate_lfm2_cpcf_selective_evidence_provider",
        "compiler_cache_directory",
        "run_lfm2_residual_attention_fold",
        ".backward(",
        "torch.optim",
        "autograd.grad",
    )
    assert all(value not in source for value in forbidden)
    assert '"parent_model_loads": 0' in source
    assert '"compiler_cache_opens": 0' in source
    assert '"body_refolds": 0' in source
    assert '"semantic_cache_authenticated_sessions": 1' in source
    assert '"child_causal_replays": 1' in source


def test_integrated_runner_persists_body_boundary_before_causal_replay() -> None:
    source = _source(INTEGRATED)

    receipt_write = source.index('run_root / "structural-body-receipt.json"')
    causal_replay = source.index("collect_lfm2_child_causal_residual_evidence(")
    assert receipt_write < causal_replay
    assert '"causal_output_head_installed": False' in source
    assert '"standalone_fresh_reload_verified": True' in source
    assert '"checkpoint_inventory_sha256"' in source
    assert '"body_model_state_sha256"' in source
    assert '"body_lfm2_identity_sha256"' in source
    assert '"causal_body_inventory_sha256"' in source
    assert '"checkpoint_inventory": [' in source
    assert '"checkpoint_inventory_sha256"' in source
    assert '"lfm2_identity_sha256"' in source
    assert '"lfm2_config_sha256"' in source


def test_source_checkpoint_inventory_pin_is_required_and_rechecked_before_loads(
    tmp_path: Path,
) -> None:
    action = next(
        value
        for value in frontier._parser()._actions
        if value.dest == "expected_source_checkpoint_inventory_sha256"
    )
    assert action.required is True

    checkpoint = tmp_path / "structural-body-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"saved-body")
    generation_config = checkpoint / "generation_config.json"
    generation_config.write_text('{"max_new_tokens": 256}', encoding="utf-8")
    inventory = frontier._artifact_inventory(checkpoint)
    expected = frontier.canonical_sha256(inventory)
    assert (
        frontier._authenticate_source_checkpoint_inventory(
            checkpoint,
            expected_sha256=expected,
        )
        == inventory
    )
    generation_config.write_text('{"max_new_tokens": 64}', encoding="utf-8")
    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="externally pinned inventory",
    ):
        frontier._authenticate_source_checkpoint_inventory(
            checkpoint,
            expected_sha256=expected,
        )

    source = _source(RUNNER)
    initial_authentication = source.index(
        "checkpoint_inventory = _authenticate_source_checkpoint_inventory("
    )
    initial_load = source.index("replay_model = checkpoint_io.load(base_checkpoint)")
    candidate_authentication = source.index(
        "_authenticate_source_checkpoint_inventory(",
        initial_load + 1,
    )
    candidate_load = source.index("candidate = checkpoint_io.load(base_checkpoint)")
    assert initial_authentication < initial_load
    assert initial_load < candidate_authentication < candidate_load


def test_source_stack_to_w2_transition_tamper_fails_closed() -> None:
    structural = {
        "operation_audit": dict(frontier.STRUCTURAL_ZERO_OPERATION_AUDIT),
    }
    stack = {
        "result_sha256": "a" * 64,
        "model_state_after_sha256": "b" * 64,
        "final_layer_mlp_w2_deferred_to_fused_terminal": True,
        "final_layer_mlp_w2_install_count": 0,
        "operation_audit": dict(frontier.STACK_ZERO_OPERATION_AUDIT),
    }
    deferred = {
        "model_state_before_sha256": "b" * 64,
        "model_state_after_sha256": "c" * 64,
        "live_w2_sha256_before": "d" * 64,
        "live_w2_sha256_after": "e" * 64,
        "prior_tensor_sha256": "e" * 64,
        "final_w2_parameter_name": "model.layers.13.feed_forward.down_proj.weight",
        "changed_state_keys": [
            "model.layers.13.feed_forward.down_proj.weight",
        ],
        "effectful_installation": True,
        **dict(frontier.DEFERRED_W2_ZERO_OPERATION_AUDIT),
    }
    authenticated = frontier._authenticate_source_fold_receipts(
        structural_receipt=structural,
        semantic_layer_stack_receipt=stack,
        deferred_w2_receipt=deferred,
    )
    assert authenticated["deferred_final_w2"] == (
        frontier.DEFERRED_W2_ZERO_OPERATION_AUDIT
    )

    tampered = copy.deepcopy(deferred)
    tampered["changed_state_keys"] = ["model.embed_tokens.weight"]
    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="stack-to-final-W2 transition",
    ):
        frontier._authenticate_source_fold_receipts(
            structural_receipt=structural,
            semantic_layer_stack_receipt=stack,
            deferred_w2_receipt=tampered,
        )

    tampered_stack = copy.deepcopy(stack)
    tampered_stack["operation_audit"]["training_steps"] = 1
    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="zero-operation",
    ):
        frontier._authenticate_source_fold_receipts(
            structural_receipt=structural,
            semantic_layer_stack_receipt=tampered_stack,
            deferred_w2_receipt=deferred,
        )
