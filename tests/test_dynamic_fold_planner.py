from __future__ import annotations

from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_diagnostics import DiagnosticFoldStrategy
from neural_forge.dynamic_fold_planner import (
    DynamicFoldPlanner,
    DynamicFoldPlanningError,
    EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY,
    canonical_resource_fingerprints,
    external_resource_bindings_sha256,
    validate_fold_action_evidence_channels,
    validate_fold_method_contract,
)
from neural_forge.dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    FoldMethodRegistry,
    LedgerRecord,
    RecordStatus,
    default_method_registry,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _parent_states():
    return {"parent": canonical_sha256({"parent_state": "fixed"})}


def _llama(
    *,
    intermediate: int = 24,
    hidden_act: str = "silu",
    rope_theta: float = 10_000.0,
    tie_word_embeddings: bool = True,
    vocab_size: int = 64,
):
    return transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            intermediate_size=intermediate,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=tie_word_embeddings,
        )
    )


def _lfm2(*, intermediate: int, layer_type: str):
    return transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=intermediate,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            layer_types=[layer_type],
            conv_L_cache=3,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )


def _registry_replacing(method_id: str, **changes) -> FoldMethodRegistry:
    base = default_method_registry()
    replacement = replace(base.get(method_id), **changes)
    return FoldMethodRegistry(
        tuple(
            replacement if method.method_id == method_id else method
            for method in base.methods
        )
    )


def test_same_topology_plan_automatically_covers_every_target_parameter() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    assert plan.ready
    assert not plan.gaps
    assert {
        path for action in plan.actions for path in action.target_parameter_paths
    } == (set(child.parameter_paths))
    assert all(action.method_id == "copy_exact" for action in plan.actions)
    assert plan.as_dict()["known_closed_methods_filtered"] is True
    assert plan.as_dict()["complete_target_parameter_coverage"] is True


def test_changed_mlp_width_creates_an_explicit_operator_gap() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(intermediate=24),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(intermediate=20),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    assert not plan.ready
    assert any(
        gap.required_capability == "cross_family_closed_loop_layer_operator"
        for gap in plan.gaps
    )


def test_concrete_child_must_equal_requested_exact_budget() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    with pytest.raises(DynamicFoldPlanningError, match="exact budget"):
        DynamicFoldPlanner().plan(
            (parent,),
            child,
            child_state_fingerprint=canonical_sha256({"child": "blank"}),
            evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
            parent_state_fingerprints=_parent_states(),
            exact_target_unique_parameters=child.unique_parameter_count - 1,
        )


def test_ledger_closure_marks_repeated_action_not_ready() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    state = canonical_sha256({"child": "blank"})
    evidence = canonical_sha256({"cache": "parent-once"})
    initial = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=state,
        evidence_fingerprint=evidence,
        parent_state_fingerprints=_parent_states(),
    )
    blocked = initial.actions[0]
    record = LedgerRecord(
        signature=blocked.signature,
        closure_level=ClosureLevel.EXACT,
        status=RecordStatus.REJECTED,
        reason_class="behavior",
        reason="Already failed this exact child state.",
    )
    planner = DynamicFoldPlanner(ledger=ExperimentLedger((record,)))
    repeated = planner.plan(
        (parent,),
        child,
        child_state_fingerprint=state,
        evidence_fingerprint=evidence,
        parent_state_fingerprints=_parent_states(),
    )

    assert repeated.ready is False
    assert repeated.actions[0].ledger_decision.allowed is False


def test_unknown_tokenizer_identity_fails_closed() -> None:
    parent = inspect_model_circuits(_llama(), model_name="parent")
    child = inspect_model_circuits(_llama(), model_name="child")

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    assert plan.ready is False
    assert any(gap.required_capability == "tokenizer_fingerprint" for gap in plan.gaps)


def test_same_shapes_with_different_activation_are_not_exact_copy() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(hidden_act="silu"),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(hidden_act="gelu"),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    assert plan.ready is False
    assert any(
        gap.required_capability == "cross_family_closed_loop_layer_operator"
        for gap in plan.gaps
    )


def test_parameterless_position_semantics_are_owned_and_compared() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(rope_theta=10_000.0),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(rope_theta=500_000.0),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    assert plan.ready is False
    assert any("input.position" in gap.target_node_ids for gap in plan.gaps)


def test_tied_embedding_and_head_share_one_storage_owning_action() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )

    token_action = next(
        action
        for action in plan.actions
        if "input.token_embedding" in action.target_node_ids
    )
    assert "output.lm_head" in token_action.target_node_ids
    all_groups = [
        group for action in plan.actions for group in action.target_storage_groups
    ]
    assert len(all_groups) == len(set(all_groups))


def test_cross_tokenizer_untied_interface_is_one_complete_bridge_action() -> None:
    parent_tokenizer = canonical_sha256({"tokenizer": "parent"})
    child_tokenizer = canonical_sha256({"tokenizer": "child"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=parent_tokenizer,
    )
    child = inspect_model_circuits(
        _llama(tie_word_embeddings=False, vocab_size=32),
        model_name="child",
        tokenizer_fingerprint=child_tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "untied-blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
        exact_target_unique_parameters=child.unique_parameter_count,
    )

    bridge_actions = tuple(
        action for action in plan.actions if action.method_id == "token_interface_bridge"
    )
    assert plan.ready
    assert len(bridge_actions) == 1
    bridge = bridge_actions[0]
    assert set(bridge.target_node_ids) == {
        "input.token_embedding",
        "output.lm_head",
    }
    assert len(bridge.sources) == 1
    assert set(bridge.sources[0].node_ids) == {
        "input.token_embedding",
        "output.lm_head",
    }
    assert len(bridge.target_storage_groups) == 2


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"source_kinds": ("attention",)}, "source kind contract"),
        ({"target_kinds": ("attention",)}, "target kind contract"),
        ({"coupled_roles": ("q_k_v_o",)}, "coupled role"),
    ),
)
def test_planner_enforces_registered_method_bundle_contracts(
    changes,
    message,
) -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    with pytest.raises(DynamicFoldPlanningError, match=message):
        DynamicFoldPlanner(methods=_registry_replacing("copy_exact", **changes)).plan(
            (parent,),
            child,
            child_state_fingerprint=canonical_sha256({"child": "blank"}),
            evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
            parent_state_fingerprints=_parent_states(),
        )


@pytest.mark.parametrize("layer_type", ("full_attention", "conv"))
def test_lfm2_method_contract_accepts_attention_and_short_conv_layers(
    layer_type: str,
) -> None:
    tokenizer = canonical_sha256({"tokenizer": "lfm2"})
    parent_model = _lfm2(intermediate=6, layer_type=layer_type)
    child_model = _lfm2(intermediate=5, layer_type=layer_type)
    parent = inspect_model_circuits(
        parent_model,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        child_model,
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256(
            {"child": layer_type, "intermediate": 5}
        ),
        evidence_fingerprint=canonical_sha256({"cache": "lfm2"}),
        parent_state_fingerprints=_parent_states(),
        exact_target_unique_parameters=child.unique_parameter_count,
    )

    assert plan.ready
    layer_action = next(
        action
        for action in plan.actions
        if action.method_id == "lfm2_closed_loop_layer"
    )
    assert (
        layer_action.evidence_requirements
        == default_method_registry().get("lfm2_closed_loop_layer").evidence_channels
    )


def test_explicit_node_attribute_can_prove_a_coupled_role() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    method = replace(
        default_method_registry().get("coupled_gqa_medoid_endpoint"),
        coupled_roles=("grouped_query_attention",),
    )

    validate_fold_method_contract(
        method,
        source_groups=((parent, (parent.node("layer.000.attention"),)),),
        target_graph=child,
        target_nodes=(child.node("layer.000.attention"),),
    )


def test_evidence_channel_validation_hook_is_exact_by_default() -> None:
    tokenizer = canonical_sha256({"tokenizer": "same"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    plan = DynamicFoldPlanner().plan(
        (parent,),
        child,
        child_state_fingerprint=canonical_sha256({"child": "blank"}),
        evidence_fingerprint=canonical_sha256({"cache": "parent-once"}),
        parent_state_fingerprints=_parent_states(),
    )
    action = plan.actions[0]

    assert validate_fold_action_evidence_channels(
        action,
        ("structure",),
    ) == ("structure",)
    with pytest.raises(DynamicFoldPlanningError, match="missing"):
        validate_fold_action_evidence_channels(action, ())
    with pytest.raises(DynamicFoldPlanningError, match="unexpected"):
        validate_fold_action_evidence_channels(
            action,
            ("structure", "diagnostic"),
        )
    assert validate_fold_action_evidence_channels(
        action,
        ("structure", "diagnostic"),
        allow_additional=True,
    ) == ("diagnostic", "structure")


def test_external_resource_fingerprint_helpers_are_canonical_and_strict() -> None:
    resources = {
        "second_bundle": "2" * 64,
        "first_bundle": "1" * 64,
    }

    assert canonical_resource_fingerprints(resources) == (
        ("first_bundle", "1" * 64),
        ("second_bundle", "2" * 64),
    )
    assert external_resource_bindings_sha256(resources) == canonical_sha256(
        dict(sorted(resources.items()))
    )
    assert canonical_resource_fingerprints(None) == ()
    assert external_resource_bindings_sha256({}) is None

    invalid = (
        {"bundle": "A" * 64},
        {"bundle": "0" * 63},
        {" ": "0" * 64},
        (("bundle", "0" * 64), (" bundle ", "1" * 64)),
    )
    for value in invalid:
        with pytest.raises(DynamicFoldPlanningError):
            canonical_resource_fingerprints(value)


def test_external_resources_bind_every_action_identity_without_legacy_drift() -> None:
    tokenizer = canonical_sha256({"tokenizer": "resource-binding"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )
    plan_arguments = {
        "child_state_fingerprint": canonical_sha256({"child": "blank"}),
        "evidence_fingerprint": canonical_sha256({"cache": "resource-test"}),
        "parent_state_fingerprints": _parent_states(),
    }

    def plan(resources=None):
        return DynamicFoldPlanner(resource_fingerprints=resources).plan(
            (parent,),
            child,
            **plan_arguments,
        )

    legacy = DynamicFoldPlanner().plan((parent,), child, **plan_arguments)
    assert plan(None).as_dict() == legacy.as_dict()
    assert plan({}).as_dict() == legacy.as_dict()

    resources = {
        "z_resource": "2" * 64,
        "a_resource": "1" * 64,
    }
    reversed_resources = dict(reversed(tuple(resources.items())))
    bound = plan(resources)
    reordered = plan(reversed_resources)
    changed = plan({**resources, "a_resource": "3" * 64})
    expected_binding = external_resource_bindings_sha256(resources)

    assert bound.as_dict() == reordered.as_dict()
    assert bound.fingerprint != legacy.fingerprint
    assert bound.fingerprint != changed.fingerprint
    assert len(bound.actions) == len(legacy.actions) == len(changed.actions)
    for legacy_action, bound_action, changed_action in zip(
        legacy.actions,
        bound.actions,
        changed.actions,
        strict=True,
    ):
        assert EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY not in dict(
            legacy_action.configuration
        )
        assert (
            dict(bound_action.configuration)[
                EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY
            ]
            == expected_binding
        )
        assert bound_action.configuration == bound_action.signature.configuration
        assert (
            bound_action.signature.configuration_sha256
            != legacy_action.signature.configuration_sha256
        )
        assert (
            bound_action.signature.configuration_sha256
            != changed_action.signature.configuration_sha256
        )
        assert (
            bound_action.signature.candidate_sha256
            != legacy_action.signature.candidate_sha256
        )
        assert (
            bound_action.signature.candidate_sha256
            != changed_action.signature.candidate_sha256
        )
        assert bound_action.action_id != legacy_action.action_id
        assert bound_action.action_id != changed_action.action_id
        assert bound_action.action_sha256 != legacy_action.action_sha256
        assert bound_action.action_sha256 != changed_action.action_sha256


def test_external_resource_configuration_key_is_reserved_and_diagnostic_safe() -> None:
    digest = "a" * 64
    strategy = DiagnosticFoldStrategy(
        method_id="test_fold",
        method_version=1,
        family_id="test_fold_family",
        method_spec_sha256=digest,
        operator_type="test.Operator",
        operator_source_sha256="b" * 64,
        operator_prepare_implementation_fingerprint="c" * 64,
        configuration=((EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY, digest),),
    )
    assert dict(strategy.configuration) == {
        EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY: digest
    }

    tokenizer = canonical_sha256({"tokenizer": "reserved-resource-key"})
    parent = inspect_model_circuits(
        _llama(),
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    child = inspect_model_circuits(
        _llama(),
        model_name="child",
        tokenizer_fingerprint=tokenizer,
    )

    class CollidingPlanner(DynamicFoldPlanner):
        def _action(self, **kwargs):
            configuration = dict(kwargs["configuration"])
            configuration[EXTERNAL_RESOURCE_BINDINGS_CONFIGURATION_KEY] = digest
            kwargs["configuration"] = configuration
            return super()._action(**kwargs)

    with pytest.raises(DynamicFoldPlanningError, match="is reserved"):
        CollidingPlanner(resource_fingerprints={"bundle": digest}).plan(
            (parent,),
            child,
            child_state_fingerprint=canonical_sha256({"child": "blank"}),
            evidence_fingerprint=canonical_sha256({"cache": "reserved-key"}),
            parent_state_fingerprints=_parent_states(),
        )
