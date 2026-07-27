from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from experiments import run_lfm2_exact4x_causal_head_frontier as frontier  # noqa: E402
from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402


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
    sealed = [
        recipe
        for recipe in recipes
        if recipe["unsupported_logit_policy"] == "sealed-background-nullspace"
    ]
    measured = [
        recipe
        for recipe in recipes
        if recipe["unsupported_logit_policy"] == "measure-drift"
    ]
    assert len(sealed) == 1
    assert len(measured) == 5
    assert sealed[0]["candidate_id"] == (
        "exact4x-causal-a1000-r1e2-norm250-sealed-background"
    )
    assert sealed[0]["ridge_fraction"] == 1e-2
    assert sealed[0]["residual_scale_ppm"] == 1_000_000
    assert sealed[0]["row_norm_calibration_ppm"] == 250_000
    assert "exact4x-causal-a1000-r1e2-norm250" not in {
        recipe["candidate_id"] for recipe in recipes
    }
    assert '"all_candidates_materialized": True' in source
    assert '"all_six_policy_arms_received_full_raw_decode": True' in source
    assert '"diagnostic_numbers_have_decision_authority": False' in source
    assert '"automatic_behavior_acceptance_or_rejection_performed": False' in source
    assert '"unsupported_logit_policy_used_as_behavioral_gate": False' in source


def test_resume_runner_replays_parent_and_child_evidence_once_without_refold() -> None:
    source = _source(RUNNER)
    tree = ast.parse(source)

    assert len(_call_nodes(tree, "open_lfm2_circuit_trace_cache")) == 1
    assert len(_call_nodes(tree, "collect_lfm2_terminal_cache_evidence")) == 1
    assert len(_call_nodes(tree, "compile_lfm2_causal_trie_evidence")) == 1
    assert len(_call_nodes(tree, "collect_lfm2_child_causal_residual_evidence")) == 1
    causal_compile_calls = _call_nodes(
        tree,
        "compile_lfm2_causal_residual_token_bridge",
    )
    assert len(causal_compile_calls) == 1
    policy_keywords = [
        keyword
        for keyword in causal_compile_calls[0].keywords
        if keyword.arg == "unsupported_logit_policy"
    ]
    assert len(policy_keywords) == 1
    assert isinstance(policy_keywords[0].value, ast.Name)
    assert policy_keywords[0].value.id == "unsupported_logit_policy"
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
    assert '"causal_head_analytic_solves": len(CAUSAL_HEAD_RECIPES)' in source
    assert '"diagnostic_metrics_used_for_materialization": False' in source
    assert '"manual_behavior_score_used_for_materialization": False' in source
    assert '"structural_integrity_not_behavioral_score_gate": True' in source


def test_deferred_w2_receipt_identity_is_not_reinterpreted_as_checkpoint_state() -> (
    None
):
    source = _source(RUNNER)

    assert "_authenticate_loaded_deferred_final_w2(" in source
    assert (
        'deferred_w2_receipt.get("model_state_after_sha256") != body_state'
        not in source
    )
    assert (
        'deferred_w2_receipt.get("model_state_after_sha256") != body_identity'
        not in source
    )


class _DeferredW2Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.feed_forward = torch.nn.Module()
        layer.feed_forward.w2 = torch.nn.Linear(3, 4, bias=False)
        self.model.layers = torch.nn.ModuleList([layer])


def _deferred_w2_receipt(model: _DeferredW2Model) -> dict[str, object]:
    parameter = model.model.layers[-1].feed_forward.w2.weight
    return {
        "final_w2_parameter_name": "model.layers.0.feed_forward.w2.weight",
        "live_w2_sha256_after": (
            frontier.lfm2_cpcf_terminal_action_tensor_sha256(parameter)
        ),
        "model_state_after_sha256": "f" * 64,
    }


def test_fresh_deferred_w2_authentication_uses_name_and_tensor_hash_only() -> None:
    model = _DeferredW2Model()
    receipt = _deferred_w2_receipt(model)

    proof = frontier._authenticate_loaded_deferred_final_w2(model, receipt)

    assert proof["fresh_checkpoint_final_w2_sha256"] == receipt["live_w2_sha256_after"]
    assert proof["same_terminal_action_tensor_hash_domain"] is True
    assert proof["deferred_receipt_whole_model_identity_reinterpreted"] is False


def test_fresh_deferred_w2_authentication_rejects_bad_parameter_name() -> None:
    model = _DeferredW2Model()
    receipt = _deferred_w2_receipt(model)
    receipt["final_w2_parameter_name"] = "model.layers.0.feed_forward.w1.weight"

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="receipt name",
    ):
        frontier._authenticate_loaded_deferred_final_w2(model, receipt)


def test_fresh_deferred_w2_authentication_rejects_bad_hash_or_bytes() -> None:
    model = _DeferredW2Model()
    receipt = _deferred_w2_receipt(model)
    receipt["live_w2_sha256_after"] = "0" * 64
    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="bytes differ",
    ):
        frontier._authenticate_loaded_deferred_final_w2(model, receipt)

    receipt = _deferred_w2_receipt(model)
    with torch.no_grad():
        model.model.layers[-1].feed_forward.w2.weight[0, 0] += 1.0
    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="bytes differ",
    ):
        frontier._authenticate_loaded_deferred_final_w2(model, receipt)


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


def _write_structural_body_dtype_receipt(
    source_root: Path,
    *,
    checkpoint_dtype: object,
    parameter_dtype: object,
) -> str:
    body = {
        "schema": frontier.STRUCTURAL_BODY_RECEIPT_SCHEMA,
        "checkpoint_reload_dtype": checkpoint_dtype,
        "child_parameter_dtype": parameter_dtype,
    }
    receipt_sha256 = canonical_sha256(body)
    (source_root / "structural-body-receipt.json").write_text(
        json.dumps({**body, "receipt_sha256": receipt_sha256}) + "\n",
        encoding="ascii",
    )
    return receipt_sha256


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_source_checkpoint_dtype_comes_from_pinned_sealed_receipt(
    tmp_path: Path,
    dtype: str,
) -> None:
    receipt_sha256 = _write_structural_body_dtype_receipt(
        tmp_path,
        checkpoint_dtype=dtype,
        parameter_dtype=dtype,
    )

    assert (
        frontier._resolve_source_checkpoint_dtype_name(
            tmp_path,
            expected_receipt_sha256=receipt_sha256,
        )
        == dtype
    )


def test_source_checkpoint_dtype_legacy_fallback_requires_receipt_absence(
    tmp_path: Path,
) -> None:
    assert (
        frontier._resolve_source_checkpoint_dtype_name(
            tmp_path,
            expected_receipt_sha256=None,
        )
        == "bfloat16"
    )

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="missing",
    ):
        frontier._resolve_source_checkpoint_dtype_name(
            tmp_path,
            expected_receipt_sha256="1" * 64,
        )


@pytest.mark.parametrize(
    ("checkpoint_dtype", "parameter_dtype", "match"),
    [
        ("float32", "bfloat16", "disagree"),
        ("float16", "float16", "unsupported"),
        (None, "float32", "lacks"),
        ("float32", None, "lacks"),
    ],
)
def test_source_checkpoint_dtype_receipt_mismatch_fails_closed(
    tmp_path: Path,
    checkpoint_dtype: object,
    parameter_dtype: object,
    match: str,
) -> None:
    receipt_sha256 = _write_structural_body_dtype_receipt(
        tmp_path,
        checkpoint_dtype=checkpoint_dtype,
        parameter_dtype=parameter_dtype,
    )

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match=match,
    ):
        frontier._resolve_source_checkpoint_dtype_name(
            tmp_path,
            expected_receipt_sha256=receipt_sha256,
        )


def test_source_checkpoint_dtype_receipt_must_be_externally_pinned(
    tmp_path: Path,
) -> None:
    _write_structural_body_dtype_receipt(
        tmp_path,
        checkpoint_dtype="float32",
        parameter_dtype="float32",
    )

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="not externally pinned",
    ):
        frontier._resolve_source_checkpoint_dtype_name(
            tmp_path,
            expected_receipt_sha256=None,
        )


def test_source_checkpoint_dtype_is_reused_for_io_materialization_and_review() -> None:
    tree = ast.parse(_source(RUNNER))
    checkpoint_io_call = _call_nodes(tree, "Lfm2DirectLocalCheckpointIO")
    review_call = _call_nodes(tree, "review_frontier")
    assert len(checkpoint_io_call) == 1
    assert len(review_call) == 1
    io_dtype = next(
        keyword.value
        for keyword in checkpoint_io_call[0].keywords
        if keyword.arg == "dtype"
    )
    review_dtype = next(
        keyword.value for keyword in review_call[0].keywords if keyword.arg == "dtype"
    )
    assert isinstance(io_dtype, ast.Name)
    assert io_dtype.id == "source_checkpoint_torch_dtype"
    assert isinstance(review_dtype, ast.Name)
    assert review_dtype.id == "source_checkpoint_dtype_name"
    source = _source(RUNNER)
    assert '"materialization_dtype": source_checkpoint_dtype_name' in source


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


def _exact_legacy_source_basis_kwargs() -> dict[str, object]:
    table_sha256 = "1" * 64
    return {
        "expected_checkpoint_inventory_sha256": "2" * 64,
        "checkpoint_inventory_sha256": "2" * 64,
        "body_state_sha256": "3" * 64,
        "body_identity_sha256": "4" * 64,
        "body_inventory_sha256": "5" * 64,
        "input_table_sha256": table_sha256,
        "output_table_sha256": table_sha256,
        "target_topology": dict(frontier.EXPECTED_CHILD_TOPOLOGY),
        "target_unique_parameters": frontier.EXPECTED_CHILD_UNIQUE_PARAMETERS,
        "structural_receipt": {
            "result_sha256": "6" * 64,
            "parameter_inventory_sha256": "7" * 64,
            "target_topology_sha256": "8" * 64,
            "installed_input_table_sha256": table_sha256,
            "installed_output_baseline_sha256": table_sha256,
            "exact_target_unique_parameters": (
                frontier.EXPECTED_CHILD_UNIQUE_PARAMETERS
            ),
            "input_and_output_parameters_distinct": True,
            "input_and_output_storage_distinct": True,
            "input_output_values_equal_at_structural_install": True,
            "parameter_topology_preserved": True,
        },
        "semantic_layer_stack_receipt": {"result_sha256": "9" * 64},
        "deferred_w2_receipt": {
            "result_sha256": "a" * 64,
            "final_w2_parameter_name": ("model.layers.13.feed_forward.w2.weight"),
            "live_w2_sha256_after": "b" * 64,
        },
        "deferred_w2_authentication": {
            "final_w2_parameter_name": "model.layers.13.feed_forward.w2.weight",
            "live_w2_sha256_after": "b" * 64,
            "fresh_checkpoint_final_w2_sha256": "b" * 64,
            "same_terminal_action_tensor_hash_domain": True,
            "deferred_receipt_whole_model_identity_reinterpreted": False,
        },
        "optional_structural_body_receipt": None,
    }


def test_legacy_source_basis_binds_complete_alternative_authentication() -> None:
    basis = frontier._build_source_body_authentication_basis(
        **_exact_legacy_source_basis_kwargs()
    )

    assert basis["authentication_mode"] == (
        "legacy-external-checkpoint-and-sealed-fold-chain"
    )
    assert basis["structural_body_receipt_sha256"] is None
    assert basis["legacy_receipt_absence_covered_by_complete_basis"] is True
    assert frontier._verify_source_body_authentication_basis(basis) == basis


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("checkpoint", "externally pinned checkpoint"),
        ("topology", "exact target topology"),
        ("interface", "incomplete or invalid"),
        ("stable_w2", "incomplete or invalid"),
    ],
)
def test_legacy_source_basis_adversarial_mismatch_fails_closed(
    mutation: str,
    match: str,
) -> None:
    values = _exact_legacy_source_basis_kwargs()
    if mutation == "checkpoint":
        values["checkpoint_inventory_sha256"] = "f" * 64
    elif mutation == "topology":
        topology = dict(values["target_topology"])
        topology["num_hidden_layers"] = 13
        values["target_topology"] = topology
    elif mutation == "interface":
        values["output_table_sha256"] = "f" * 64
    else:
        stable_w2 = dict(values["deferred_w2_authentication"])
        stable_w2["fresh_checkpoint_final_w2_sha256"] = "f" * 64
        values["deferred_w2_authentication"] = stable_w2

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match=match,
    ):
        frontier._build_source_body_authentication_basis(**values)


class _TreatedConfig:
    vocab_size = 8
    hidden_size = 4
    intermediate_size = 3
    num_hidden_layers = 1
    num_attention_heads = 2
    num_key_value_heads = 1
    head_dim = 2
    tie_word_embeddings = False
    layer_types = ("full_attention",)
    model_type = "lfm2"

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "tie_word_embeddings",
                "layer_types",
                "model_type",
            )
        }


class _TreatedAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.k_proj = torch.nn.Linear(4, 2, bias=False)
        self.v_proj = torch.nn.Linear(4, 2, bias=False)
        self.out_proj = torch.nn.Linear(4, 4, bias=False)


class _TreatedLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.is_attention_layer = True
        self.self_attn = _TreatedAttention()


class _TreatedBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([_TreatedLayer()])


class _TreatedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _TreatedConfig()
        self.model = _TreatedBackbone()
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.model.embed_tokens.weight)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def _test_source_body_authentication_basis(
    baseline: _TreatedModel,
    baseline_inventory: list[dict[str, object]],
    *,
    receipt_sha256: str | None,
) -> dict[str, object]:
    inventory_sha256 = canonical_sha256(baseline_inventory)
    table_sha256 = "2" * 64
    body = {
        "schema": frontier.SOURCE_BODY_AUTHENTICATION_BASIS_SCHEMA,
        "status": "fresh-source-body-authenticated",
        "authentication_mode": (
            "legacy-external-checkpoint-and-sealed-fold-chain"
            if receipt_sha256 is None
            else "sealed-structural-body-receipt"
        ),
        "structural_body_receipt_sha256": receipt_sha256,
        "legacy_receipt_absence_covered_by_complete_basis": receipt_sha256 is None,
        "externally_pinned_checkpoint_inventory_sha256": inventory_sha256,
        "observed_checkpoint_inventory_sha256": inventory_sha256,
        "body_model_state_sha256": frontier.fingerprint_model_state(baseline),
        "body_lfm2_identity_sha256": (
            frontier.fingerprint_lfm2_parent(baseline).identity_sha256
        ),
        "causal_body_inventory_sha256": (
            frontier.fingerprint_lfm2_causal_body_inventory(baseline)
        ),
        "target_topology": frontier._target_topology(baseline),
        "target_unique_parameters": frontier._unique_parameter_count(baseline),
        "structural_interface_authentication": {
            "structural_interface_install_sha256": "3" * 64,
            "parameter_inventory_sha256": "4" * 64,
            "target_topology_sha256": "5" * 64,
            "installed_input_table_sha256": table_sha256,
            "installed_output_baseline_sha256": table_sha256,
            "live_input_table_sha256": table_sha256,
            "live_output_table_sha256": table_sha256,
            "exact_target_unique_parameters": frontier._unique_parameter_count(
                baseline
            ),
            "input_and_output_parameters_distinct": True,
            "input_and_output_storage_distinct": True,
            "input_output_values_equal_at_structural_install": True,
            "parameter_topology_preserved": True,
        },
        "sealed_fold_receipt_chain": {
            "structural_interface_install_sha256": "3" * 64,
            "semantic_layer_stack_sha256": "6" * 64,
            "deferred_final_w2_install_sha256": "7" * 64,
        },
        "stable_deferred_final_w2_authentication": {
            "final_w2_parameter_name": "model.layers.0.feed_forward.w2.weight",
            "live_w2_sha256_after": "8" * 64,
            "fresh_checkpoint_final_w2_sha256": "8" * 64,
            "same_terminal_action_tensor_hash_domain": True,
            "deferred_receipt_whole_model_identity_reinterpreted": False,
        },
    }
    return {**body, "basis_sha256": canonical_sha256(body)}


def _write_treated_bundle(
    tmp_path: Path,
    *,
    mutate_role: str = "q_proj",
    audit: dict[str, object] | None = None,
    receipt_sha256: str | None = "1" * 64,
) -> tuple[
    _TreatedModel,
    _TreatedModel,
    Path,
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    torch.manual_seed(17)
    baseline = _TreatedModel()
    treated = copy.deepcopy(baseline)
    with torch.no_grad():
        getattr(treated.model.layers[0].self_attn, mutate_role).weight[0, 0] += 1.0
    baseline_checkpoint = tmp_path / "baseline"
    treated_root = tmp_path / "treated"
    treated_checkpoint = treated_root / "checkpoint"
    baseline_checkpoint.mkdir()
    treated_checkpoint.mkdir(parents=True)
    (baseline_checkpoint / "config.json").write_text("{}\n", encoding="ascii")
    (treated_checkpoint / "config.json").write_text("{}\n", encoding="ascii")
    (baseline_checkpoint / "model.safetensors").write_bytes(b"baseline")
    (treated_checkpoint / "model.safetensors").write_bytes(b"treated")
    baseline_inventory = frontier._artifact_inventory(baseline_checkpoint)
    treated_inventory = frontier._artifact_inventory(treated_checkpoint)
    baseline_entries = dict(frontier.fingerprint_model_state_entries(baseline))
    treated_entries = dict(frontier.fingerprint_model_state_entries(treated))
    changed = sorted(
        name
        for name in baseline_entries
        if baseline_entries[name] != treated_entries[name]
    )
    baseline_fingerprint = frontier.fingerprint_lfm2_parent(baseline)
    treated_fingerprint = frontier.fingerprint_lfm2_parent(treated)
    source_basis = _test_source_body_authentication_basis(
        baseline,
        baseline_inventory,
        receipt_sha256=receipt_sha256,
    )
    body = {
        "schema": frontier.QK_TREATED_BODY_PROVENANCE_SCHEMA,
        "status": "saved-reloaded-qk-only-treated-body",
        "source_body_authentication_basis": source_basis,
        "source_body_authentication_basis_sha256": source_basis["basis_sha256"],
        "source_structural_body_receipt_sha256": receipt_sha256,
        "source_checkpoint_inventory_sha256": canonical_sha256(baseline_inventory),
        "source_body_model_state_sha256": frontier.fingerprint_model_state(baseline),
        "source_body_lfm2_identity_sha256": baseline_fingerprint.identity_sha256,
        "source_causal_body_inventory_sha256": (
            frontier.fingerprint_lfm2_causal_body_inventory(baseline)
        ),
        "source_deferred_final_w2_authentication": source_basis[
            "stable_deferred_final_w2_authentication"
        ],
        "treated_checkpoint_inventory_sha256": canonical_sha256(treated_inventory),
        "treated_body_model_state_sha256": frontier.fingerprint_model_state(treated),
        "treated_body_lfm2_identity_sha256": treated_fingerprint.identity_sha256,
        "treated_causal_body_inventory_sha256": (
            frontier.fingerprint_lfm2_causal_body_inventory(treated)
        ),
        "changed_checkpoint_relative_paths": ["model.safetensors"],
        "changed_state_keys": changed,
        "causal_output_head_installed": False,
        "standalone_fresh_reload_verified": True,
        "operation_audit": (
            dict(frontier.QK_TREATED_BODY_ZERO_OPERATION_AUDIT)
            if audit is None
            else audit
        ),
    }
    provenance = {**body, "receipt_sha256": canonical_sha256(body)}
    (treated_root / "treated-body-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return (
        baseline,
        treated,
        treated_root,
        baseline_inventory,
        provenance,
        source_basis,
    )


@pytest.mark.parametrize("receipt_sha256", ["1" * 64, None])
def test_qk_treated_body_authentication_accepts_only_qk_delta(
    tmp_path: Path,
    receipt_sha256: str | None,
) -> None:
    baseline, treated, treated_root, baseline_inventory, provenance, source_basis = (
        _write_treated_bundle(tmp_path, receipt_sha256=receipt_sha256)
    )
    checkpoint_io = type("_IO", (), {"load": lambda self, _path: treated})()

    reloaded, authentication, inventory, verified = (
        frontier._authenticate_qk_treated_body(
            treated_body_directory=treated_root,
            expected_provenance_sha256=provenance["receipt_sha256"],
            checkpoint_io=checkpoint_io,
            baseline_model=baseline,
            baseline_checkpoint_inventory=baseline_inventory,
            baseline_checkpoint_inventory_sha256=canonical_sha256(baseline_inventory),
            baseline_state_sha256=frontier.fingerprint_model_state(baseline),
            baseline_identity_sha256=(
                frontier.fingerprint_lfm2_parent(baseline).identity_sha256
            ),
            baseline_body_inventory_sha256=(
                frontier.fingerprint_lfm2_causal_body_inventory(baseline)
            ),
            source_body_authentication_basis=source_basis,
        )
    )

    assert reloaded is treated
    assert authentication["qk_only_transition_authenticated"] is True
    assert authentication["changed_state_keys"] == [
        "model.layers.0.self_attn.q_proj.weight"
    ]
    assert (
        canonical_sha256(inventory) == verified["treated_checkpoint_inventory_sha256"]
    )


@pytest.mark.parametrize("mutate_role", ["v_proj", "out_proj"])
def test_qk_treated_body_authentication_rejects_non_qk_delta(
    tmp_path: Path,
    mutate_role: str,
) -> None:
    baseline, treated, treated_root, baseline_inventory, provenance, source_basis = (
        _write_treated_bundle(tmp_path, mutate_role=mutate_role)
    )
    checkpoint_io = type("_IO", (), {"load": lambda self, _path: treated})()

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="Q/K-only transition",
    ):
        frontier._authenticate_qk_treated_body(
            treated_body_directory=treated_root,
            expected_provenance_sha256=provenance["receipt_sha256"],
            checkpoint_io=checkpoint_io,
            baseline_model=baseline,
            baseline_checkpoint_inventory=baseline_inventory,
            baseline_checkpoint_inventory_sha256=canonical_sha256(baseline_inventory),
            baseline_state_sha256=frontier.fingerprint_model_state(baseline),
            baseline_identity_sha256=(
                frontier.fingerprint_lfm2_parent(baseline).identity_sha256
            ),
            baseline_body_inventory_sha256=(
                frontier.fingerprint_lfm2_causal_body_inventory(baseline)
            ),
            source_body_authentication_basis=source_basis,
        )


def test_qk_treated_body_rejects_resealed_source_basis_substitution(
    tmp_path: Path,
) -> None:
    baseline, treated, treated_root, baseline_inventory, provenance, source_basis = (
        _write_treated_bundle(tmp_path, receipt_sha256=None)
    )
    tampered = copy.deepcopy(provenance)
    tampered_basis = dict(tampered["source_body_authentication_basis"])
    tampered_basis_body = dict(tampered_basis)
    tampered_basis_body.pop("basis_sha256")
    tampered_chain = dict(tampered_basis_body["sealed_fold_receipt_chain"])
    tampered_chain["semantic_layer_stack_sha256"] = "f" * 64
    tampered_basis_body["sealed_fold_receipt_chain"] = tampered_chain
    tampered_basis = {
        **tampered_basis_body,
        "basis_sha256": canonical_sha256(tampered_basis_body),
    }
    tampered["source_body_authentication_basis"] = tampered_basis
    tampered["source_body_authentication_basis_sha256"] = tampered_basis["basis_sha256"]
    tampered_body = dict(tampered)
    tampered_body.pop("receipt_sha256")
    tampered = {
        **tampered_body,
        "receipt_sha256": canonical_sha256(tampered_body),
    }
    (treated_root / "treated-body-provenance.json").write_text(
        json.dumps(tampered, sort_keys=True) + "\n",
        encoding="ascii",
    )
    checkpoint_io = type("_IO", (), {"load": lambda self, _path: treated})()

    with pytest.raises(
        frontier.Lfm2Exact4xCausalHeadFrontierError,
        match="Q/K-only transition",
    ):
        frontier._authenticate_qk_treated_body(
            treated_body_directory=treated_root,
            expected_provenance_sha256=tampered["receipt_sha256"],
            checkpoint_io=checkpoint_io,
            baseline_model=baseline,
            baseline_checkpoint_inventory=baseline_inventory,
            baseline_checkpoint_inventory_sha256=canonical_sha256(baseline_inventory),
            baseline_state_sha256=frontier.fingerprint_model_state(baseline),
            baseline_identity_sha256=(
                frontier.fingerprint_lfm2_parent(baseline).identity_sha256
            ),
            baseline_body_inventory_sha256=(
                frontier.fingerprint_lfm2_causal_body_inventory(baseline)
            ),
            source_body_authentication_basis=source_basis,
        )
