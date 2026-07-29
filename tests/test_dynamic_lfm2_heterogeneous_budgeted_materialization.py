from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest

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


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


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


def _heterogeneous_case():
    parent = _parent()
    tokenizer = _digest("heterogeneous-adapter-tokenizer")
    graph = inspect_model_circuits(
        parent,
        model_name="parent",
        tokenizer_fingerprint=tokenizer,
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
        vocab_size=64,
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
        target_vocab_evidence_sha256=_digest("same-vocab-evidence"),
        runtime_evidence_sha256=_digest("runtime-evidence"),
        token_bridge_evidence_sha256=_digest("same-token-interface"),
    )
    # Untied interface, nine norms, four ragged MLPs, two ragged ShortConv
    # operators, and two different attention geometries.
    exact = 9_910
    catalog = Lfm2DenseTargetTopologyCatalog(
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
        target_vocabulary_size=64,
        match_policy="exact",
        maximum_candidates=1,
    )
    candidate = frontier.candidates[0]
    adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=None,
        evidence_fingerprint=_digest("cached-parent-evidence"),
        device="cpu",
        dtype=torch.float32,
    )
    request = candidate.to_target_request(
        child_name="heterogeneous-exact-blank",
        tokenizer_fingerprint=tokenizer,
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
