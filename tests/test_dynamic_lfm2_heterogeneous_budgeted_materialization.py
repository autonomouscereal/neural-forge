from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from neural_forge.dynamic_fold_planner import DynamicFoldPlanner
from neural_forge.dynamic_lfm2_adapter import (
    DynamicLfm2AdapterError,
    Lfm2DenseBudgetedSelectedTarget,
    Lfm2TargetSynthesisAdapter,
    validate_lfm2_dense_budgeted_child,
)
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_parameter_frontier import (
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
    Lfm2DenseTargetTopologyCatalog,
    Lfm2HeterogeneousLayoutSelectionInput,
    component_allocation_topology_fingerprint,
)
from neural_forge.hf_lfm2_dynamic_blank_materializer import (
    extract_lfm2_dynamic_physical_layout_from_config,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)
from neural_forge.hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    seal_strict_tokenizer_bridge_artifact,
)


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _BridgeTokenizer:
    def __init__(self, surfaces: list[str]) -> None:
        self.surfaces = list(surfaces)
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = None
        self.all_special_ids = [0, 1, 2]

    def __len__(self) -> int:
        return len(self.surfaces)

    def get_vocab(self) -> dict[str, int]:
        return {
            f"raw-{index}-{surface}": index
            for index, surface in enumerate(self.surfaces)
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"raw-{token_id}-{self.surfaces[token_id]}"

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self.surfaces[token_id] for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self.surfaces.index(text)]


def _strict_bridge(tmp_path: Path):
    fixed = ["<pad>", "<bos>", "<eos>"]
    body = [f"<token-{index:02d}>" for index in range(3, 64)]
    donor = _BridgeTokenizer([*fixed, *body])
    target = _BridgeTokenizer([*fixed, *body[:14]])
    root = tmp_path / "heterogeneous-target-tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return seal_strict_tokenizer_bridge_artifact(
        donor,
        target,
        root,
        donor_row_capacity=64,
        target_row_capacity=17,
    )


def _parent():
    torch.manual_seed(9201)
    model = transformers.Lfm2ForCausalLM(
        transformers.Lfm2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=[
                "conv",
                "full_attention",
                "conv",
                "full_attention",
            ],
            conv_L_cache=3,
            conv_bias=False,
            block_auto_adjust_ff_dim=False,
            tie_word_embeddings=True,
        )
    )
    model.eval()
    return model


def _heterogeneous_case(*, tokenizer_bridge=None):
    parent = _parent()
    tokenizer = (
        _digest("heterogeneous-adapter-tokenizer")
        if tokenizer_bridge is None
        else tokenizer_bridge.donor_tokenizer_semantics_sha256
    )
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
    )
    target_vocab = (
        64
        if tokenizer_bridge is None
        else tokenizer_bridge.target_vocabulary_size
    )
    per_layer = {
        0: (18, 12, None),
        1: (20, None, (2, 1, 4)),
        2: (22, 14, None),
        # The second attention intentionally expands Q*D=32 beyond target
        # H=16.  The dynamic projection is still no larger than its parent
        # Q*D=32 and is the same geometry required by the production witness.
        3: (24, None, (4, 2, 8)),
    }
    options = []
    for discovered in discover_lfm2_source_layers(graph):
        intermediate, internal, attention = per_layer[discovered.ordinal]
        if discovered.layer_type == "conv":
            assert internal is not None
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=discovered.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    internal_width=internal,
                    selection_evidence_sha256=_digest(
                        f"layer-{discovered.ordinal}-evidence"
                    ),
                    structural_risk=0.2 + discovered.ordinal / 100,
                )
            )
        else:
            assert attention is not None
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=discovered.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    query_heads=attention[0],
                    key_value_heads=attention[1],
                    head_dim=attention[2],
                    selection_evidence_sha256=_digest(
                        f"layer-{discovered.ordinal}-evidence"
                    ),
                    structural_risk=0.1 + discovered.ordinal / 100,
                )
            )
    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=16,
        vocab_size=target_vocab,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=(
            _digest("same-vocab-evidence")
            if tokenizer_bridge is None
            else tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
        runtime_evidence_sha256=_digest("runtime-evidence"),
        token_bridge_evidence_sha256=(
            _digest("same-token-interface")
            if tokenizer_bridge is None
            else tokenizer_bridge.artifact_sha256
        ),
    )
    # Untied interface, nine norms, four ragged MLPs, two ragged ShortConv
    # operators, and two different attention geometries.
    exact = 9_910 - 2 * 16 * (64 - target_vocab)
    catalog = Lfm2DenseTargetTopologyCatalog(
        tokenizer_bridge_artifact=tokenizer_bridge,
        target_token_interface_mode="untied",
        heterogeneous_layout_selection_input=(
            Lfm2HeterogeneousLayoutSelectionInput(
                target_pins=pins,
                layer_options=tuple(options),
            )
        ),
    )
    frontier = catalog.enumerate_frontier(
        graph,
        requested_unique_parameters=exact,
        target_vocabulary_size=target_vocab,
        match_policy="exact",
        maximum_candidates=1,
    )
    candidate = frontier.candidates[0]
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=None,
        evidence_fingerprint=_digest("cached-parent-evidence"),
        device="cpu",
        dtype=torch.float32,
        tokenizer_bridge_artifact=tokenizer_bridge,
    )
    request = candidate.to_target_request(
        child_name="heterogeneous-exact-blank",
        tokenizer_fingerprint=(
            tokenizer
            if tokenizer_bridge is None
            else tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
        adapter_id=adapter.adapter_id,
        blank_seed=9202,
    )
    return parent, graph, tokenizer, adapter, candidate, request


def test_selects_materializes_and_inspects_exact_ragged_dense_child() -> None:
    parent, graph, tokenizer, adapter, candidate, request = _heterogeneous_case()

    materialized = adapter.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    )

    selected = materialized.adapter_payload
    assert isinstance(selected, Lfm2DenseBudgetedSelectedTarget)
    assert selected.heterogeneous_layout is True
    assert selected.exact_target_unique_parameters == 9_910
    assert materialized.graph.unique_parameter_count == 9_910
    assert materialized.graph.logical_parameter_count == 9_910
    assert candidate.unique_parameters == 9_910
    descriptor = extract_lfm2_dynamic_physical_layout_from_config(
        materialized.model.config
    )
    assert descriptor == selected.heterogeneous_layout_descriptor
    assert [layer.mlp_intermediate_size for layer in descriptor.layers] == [
        18,
        20,
        22,
        24,
    ]
    assert [
        layer.short_conv_internal_width
        for layer in descriptor.layers
        if layer.layer_type == "conv"
    ] == [12, 14]
    assert [
        (layer.query_heads, layer.key_value_heads, layer.head_dim)
        for layer in descriptor.layers
        if layer.layer_type == "full_attention"
    ] == [(2, 1, 4), (4, 2, 8)]
    parameters = tuple(
        materialized.model.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    )
    assert all(
        parameter.dtype == torch.float32
        and parameter.layout == torch.strided
        and not parameter.is_sparse
        and not parameter.requires_grad
        for _name, parameter in parameters
    )
    assert len({id(parameter) for _name, parameter in parameters}) == len(parameters)
    assert len(
        {parameter.untyped_storage().data_ptr() for _name, parameter in parameters}
    ) == len(parameters)
    validate_lfm2_dense_budgeted_child(
        selected,
        materialized.model,
        materialized.graph,
    )
    assert materialized.graph.tokenizer_fingerprint == tokenizer
    assert materialized.model.config._name_or_path == ""
    config = materialized.model.config
    assert config._neural_forge_dense_topology_catalog == selected.catalog_id
    assert config._neural_forge_dense_ordered_source_spans == "0;1;2;3"
    assert config._neural_forge_dense_target_layer_types == (
        "conv,full_attention,conv,full_attention"
    )
    assert config._neural_forge_tokenizer_bridge_required is False
    metadata = dict(materialized.graph.metadata)
    assert metadata["fold_method_id"] == (
        "lfm2_dense_budget_residual_attention_model"
    )
    assert metadata["dense_topology_catalog"] == selected.catalog_id
    assert metadata["ordered_source_spans"] == "0;1;2;3"
    assert metadata["target_layer_types"] == (
        "conv,full_attention,conv,full_attention"
    )
    assert metadata["tokenizer_bridge_required"] is False

    plan = DynamicFoldPlanner().plan(
        (graph,),
        materialized.graph,
        child_state_fingerprint=str(
            materialized.report["blank_state_fingerprint"]
        ),
        evidence_fingerprint=_digest("sealed-parent-cache"),
        parent_state_fingerprints={
            graph.model_name: selected.source_state_fingerprint
        },
        exact_target_unique_parameters=selected.exact_target_unique_parameters,
    )
    assert plan.ready
    assert [action.method_id for action in plan.actions] == [
        "lfm2_dense_budget_residual_attention_model"
    ]
    assert set(plan.actions[0].target_node_ids) == {
        node.node_id for node in materialized.graph.nodes
    }


def test_bridged_ragged_child_preserves_the_complete_model_fold_contract(
    tmp_path: Path,
) -> None:
    bridge = _strict_bridge(tmp_path)
    parent, graph, _tokenizer, adapter, _candidate, request = (
        _heterogeneous_case(tokenizer_bridge=bridge)
    )

    materialized = adapter.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    )

    selected = materialized.adapter_payload
    assert isinstance(selected, Lfm2DenseBudgetedSelectedTarget)
    assert selected.heterogeneous_layout is True
    assert selected.tokenizer_bridge_required is True
    expected_config = {
        TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY: bridge.artifact_sha256,
        TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY: bridge.profile_fingerprint,
        PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.donor_tokenizer_semantics_sha256
        ),
        CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY: (
            bridge.target_tokenizer_semantics_sha256
        ),
    }
    assert all(
        getattr(materialized.model.config, name) == value
        for name, value in expected_config.items()
    )
    metadata = dict(materialized.graph.metadata)
    assert metadata["fold_method_id"] == (
        "lfm2_dense_budget_residual_attention_model"
    )
    assert metadata["tokenizer_bridge_required"] is True
    assert metadata["tokenizer_bridge_artifact_sha256"] == (
        bridge.artifact_sha256
    )
    assert metadata["tokenizer_transport_profile_sha256"] == (
        bridge.profile_fingerprint
    )
    assert metadata["parent_tokenizer_semantics_sha256"] == (
        bridge.donor_tokenizer_semantics_sha256
    )
    assert metadata["child_tokenizer_semantics_sha256"] == (
        bridge.target_tokenizer_semantics_sha256
    )
    validate_lfm2_dense_budgeted_child(
        selected,
        materialized.model,
        materialized.graph,
    )

    plan = DynamicFoldPlanner().plan(
        (graph,),
        materialized.graph,
        child_state_fingerprint=str(
            materialized.report["blank_state_fingerprint"]
        ),
        evidence_fingerprint=_digest("bridged-sealed-parent-cache"),
        parent_state_fingerprints={
            graph.model_name: selected.source_state_fingerprint
        },
        exact_target_unique_parameters=selected.exact_target_unique_parameters,
    )
    assert plan.ready
    assert [action.method_id for action in plan.actions] == [
        "lfm2_dense_budget_residual_attention_model"
    ]
    action_configuration = dict(plan.actions[0].configuration)
    assert action_configuration["tokenizer_bridge_required"] is True
    assert action_configuration["tokenizer_bridge_artifact_sha256"] == (
        bridge.artifact_sha256
    )
    assert set(plan.actions[0].target_node_ids) == {
        node.node_id for node in materialized.graph.nodes
    }


def test_heterogeneous_request_and_materialized_config_tampering_fail_closed() -> None:
    parent, graph, _tokenizer, adapter, _candidate, request = (
        _heterogeneous_case()
    )
    receipt = request.component_allocation_receipt
    assert receipt is not None
    configuration = dict(request.configuration)
    configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY] = (
        "0" * 64
    )
    changed_topology_sha256 = component_allocation_topology_fingerprint(
        configuration
    )
    changed_receipt = replace(
        receipt,
        topology_configuration_sha256=changed_topology_sha256,
    )
    configuration[COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY] = (
        changed_topology_sha256
    )
    configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = (
        changed_receipt.fingerprint
    )
    tampered_request = replace(
        request,
        configuration=tuple(configuration.items()),
        component_allocation_receipt=changed_receipt,
    )
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="artifacts failed authentication",
    ):
        adapter.materialize(
            tampered_request,
            {"parent": parent},
            {"parent": graph},
            None,
        )

    materialized = adapter.materialize(
        request,
        {"parent": parent},
        {"parent": graph},
        None,
    )
    selected = materialized.adapter_payload
    raw = copy.deepcopy(
        materialized.model.config.neural_forge_dynamic_physical_layout
    )
    raw["layers"][0]["short_conv_internal_width"] += 1
    materialized.model.config.neural_forge_dynamic_physical_layout = raw
    with pytest.raises(
        DynamicLfm2AdapterError,
        match="no authenticated physical layout",
    ):
        validate_lfm2_dense_budgeted_child(
            selected,
            materialized.model,
            materialized.graph,
        )


def test_uniform_catalog_regression_remains_on_native_materialization_path() -> None:
    parent = _parent()
    tokenizer = _digest("uniform-adapter-tokenizer")
    graph = inspect_model_circuits(
        parent,
        model_name="uniform-parent",
        tokenizer_fingerprint=tokenizer,
    )
    catalog = Lfm2DenseTargetTopologyCatalog()
    frontier = catalog.enumerate_frontier(
        graph,
        requested_unique_parameters=graph.unique_parameter_count // 2,
        target_vocabulary_size=None,
        match_policy="at_or_below",
        maximum_candidates=1,
    )
    candidate = frontier.candidates[0]
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=None,
        evidence_fingerprint=_digest("uniform-cache"),
        device="cpu",
        dtype=torch.float32,
    )
    request = candidate.to_target_request(
        child_name="uniform-regression-blank",
        tokenizer_fingerprint=tokenizer,
        adapter_id=adapter.adapter_id,
    )
    materialized = adapter.materialize(
        request,
        {"uniform-parent": parent},
        {"uniform-parent": graph},
        None,
    )
    selected = materialized.adapter_payload
    assert isinstance(selected, Lfm2DenseBudgetedSelectedTarget)
    assert selected.heterogeneous_layout is False
    assert not hasattr(
        materialized.model.config,
        "neural_forge_dynamic_physical_layout",
    )
    assert materialized.graph.unique_parameter_count == candidate.unique_parameters
