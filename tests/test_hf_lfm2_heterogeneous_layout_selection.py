from __future__ import annotations

import hashlib
import json

import pytest

from neural_forge.dynamic_circuit_ir import (
    CircuitKind,
    CircuitNode,
    ModelCircuitGraph,
    ParameterSpec,
)
from neural_forge.dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    build_counted_child_state_plan,
    verify_counted_child_state,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (
    derive_lfm2_automatic_mlp_width_policy,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    Lfm2AuthenticatedLayerPhysicalOption,
    Lfm2HeterogeneousAllocationDecisionCore,
    Lfm2HeterogeneousLayoutSelectionError,
    Lfm2HeterogeneousLayoutSelectionReceipt,
    Lfm2HeterogeneousLayoutUnattainable,
    Lfm2HeterogeneousTargetPins,
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
    select_lfm2_heterogeneous_physical_layout,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _parameter(
    path: str,
    role: str,
    shape: tuple[int, ...],
) -> ParameterSpec:
    return ParameterSpec(
        path=path,
        role=role,
        shape=shape,
        dtype="torch.float32",
        storage_group=f"storage.{path}",
        trainable=False,
    )


def _node(
    *,
    node_id: str,
    kind: CircuitKind,
    order: int,
    path: str,
    layer_index: int | None,
    parameters: tuple[ParameterSpec, ...] = (),
    attributes: dict[str, object] | None = None,
) -> CircuitNode:
    return CircuitNode(
        node_id=node_id,
        kind=kind,
        order=order,
        path=path,
        layer_index=layer_index,
        inputs=(),
        outputs=(),
        parameters=parameters,
        attributes=attributes or {},
    )


def _source_graph(
    schedule: tuple[str, ...],
    *,
    prefix: str = "arbitrary_renamed_stack",
    hidden: int = 1024,
    vocab: int = 65_536,
    intermediate: int = 1024,
    conv_width: int = 1024,
    query_heads: int = 16,
    key_value_heads: int = 8,
    head_dim: int = 64,
) -> ModelCircuitGraph:
    nodes = []
    order = 0
    embed_path = f"{prefix}.boundary.lexicon"
    nodes.append(
        _node(
            node_id=f"{prefix}.incoming_symbols",
            kind=CircuitKind.TOKEN_EMBEDDING,
            order=order,
            path=embed_path,
            layer_index=None,
            parameters=(
                _parameter(
                    f"{embed_path}.matrix",
                    "embedding.weight",
                    (vocab, hidden),
                ),
            ),
            attributes={"vocabulary_size": vocab, "hidden_size": hidden},
        )
    )
    order += 1
    for ordinal, layer_type in enumerate(schedule):
        # Deliberately non-contiguous numeric IDs and non-HF paths.  Selection
        # policy must use inspected causal groups, not these values or names.
        graph_index = 101 + 7 * ordinal
        stem = f"{prefix}.cell_{ordinal}.renamed"
        norm_a_path = f"{stem}.pre_operator_scale"
        nodes.append(
            _node(
                node_id=f"{prefix}.unit_{ordinal}.first_norm",
                kind=CircuitKind.NORM,
                order=order,
                path=norm_a_path,
                layer_index=graph_index,
                parameters=(
                    _parameter(
                        f"{norm_a_path}.gain",
                        "operator_norm.weight",
                        (hidden,),
                    ),
                ),
            )
        )
        order += 1
        operator_path = f"{stem}.temporal_operator"
        if layer_type == "conv":
            operator_parameters = (
                _parameter(
                    f"{operator_path}.ingress",
                    "operator.in_proj.weight",
                    (3 * conv_width, hidden),
                ),
                _parameter(
                    f"{operator_path}.memory",
                    "operator.fir.weight",
                    (conv_width, 1, 3),
                ),
                _parameter(
                    f"{operator_path}.egress",
                    "operator.out_proj.weight",
                    (hidden, conv_width),
                ),
            )
            operator_kind = CircuitKind.SHORT_CONV
            attributes = {"fir_length": 3, "operator_type": "short_conv"}
        else:
            query_width = query_heads * head_dim
            key_value_width = key_value_heads * head_dim
            operator_parameters = (
                _parameter(
                    f"{operator_path}.ask",
                    "operator.q_proj.weight",
                    (query_width, hidden),
                ),
                _parameter(
                    f"{operator_path}.index",
                    "operator.k_proj.weight",
                    (key_value_width, hidden),
                ),
                _parameter(
                    f"{operator_path}.payload",
                    "operator.v_proj.weight",
                    (key_value_width, hidden),
                ),
                _parameter(
                    f"{operator_path}.write",
                    "operator.out_proj.weight",
                    (hidden, query_width),
                ),
                _parameter(
                    f"{operator_path}.ask_norm",
                    "operator.q_norm.weight",
                    (head_dim,),
                ),
                _parameter(
                    f"{operator_path}.index_norm",
                    "operator.k_norm.weight",
                    (head_dim,),
                ),
            )
            operator_kind = CircuitKind.ATTENTION
            attributes = {
                "query_heads": query_heads,
                "key_value_heads": key_value_heads,
                "head_dim": head_dim,
                "operator_type": "full_attention",
            }
        nodes.append(
            _node(
                node_id=f"{prefix}.unit_{ordinal}.mixer",
                kind=operator_kind,
                order=order,
                path=operator_path,
                layer_index=graph_index,
                parameters=operator_parameters,
                attributes=attributes,
            )
        )
        order += 1
        norm_b_path = f"{stem}.pre_nonlinear_scale"
        nodes.append(
            _node(
                node_id=f"{prefix}.unit_{ordinal}.second_norm",
                kind=CircuitKind.NORM,
                order=order,
                path=norm_b_path,
                layer_index=graph_index,
                parameters=(
                    _parameter(
                        f"{norm_b_path}.gain",
                        "mlp_norm.weight",
                        (hidden,),
                    ),
                ),
            )
        )
        order += 1
        mlp_path = f"{stem}.nonlinear_writer"
        nodes.append(
            _node(
                node_id=f"{prefix}.unit_{ordinal}.swiglu",
                kind=CircuitKind.GATED_MLP,
                order=order,
                path=mlp_path,
                layer_index=graph_index,
                parameters=(
                    _parameter(
                        f"{mlp_path}.producer_a",
                        "mlp.w1.weight",
                        (intermediate, hidden),
                    ),
                    _parameter(
                        f"{mlp_path}.producer_b",
                        "mlp.w3.weight",
                        (intermediate, hidden),
                    ),
                    _parameter(
                        f"{mlp_path}.writer",
                        "mlp.w2.weight",
                        (hidden, intermediate),
                    ),
                ),
                attributes={"intermediate_size": intermediate},
            )
        )
        order += 1
    final_path = f"{prefix}.boundary.final_scale"
    nodes.append(
        _node(
            node_id=f"{prefix}.final_norm",
            kind=CircuitKind.NORM,
            order=order,
            path=final_path,
            layer_index=None,
            parameters=(
                _parameter(
                    f"{final_path}.gain",
                    "final_norm.weight",
                    (hidden,),
                ),
            ),
        )
    )
    order += 1
    head_path = f"{prefix}.boundary.readout"
    nodes.append(
        _node(
            node_id=f"{prefix}.outgoing_symbols",
            kind=CircuitKind.OUTPUT_HEAD,
            order=order,
            path=head_path,
            layer_index=None,
            parameters=(
                _parameter(
                    f"{head_path}.matrix",
                    "output_head.weight",
                    (vocab, hidden),
                ),
            ),
            attributes={"vocabulary_size": vocab},
        )
    )
    logical = sum(node.logical_parameter_count for node in nodes)
    return ModelCircuitGraph(
        model_name=f"{prefix}.model",
        family="lfm2",
        model_class="RenamedSyntheticLfm2",
        nodes=tuple(reversed(nodes)),
        edges=(),
        interfaces=(),
        logical_parameter_count=logical,
        unique_parameter_count=logical,
        unique_parameter_bytes=4 * logical,
        config_fingerprint=_digest(f"{prefix}-config"),
        tokenizer_fingerprint=_digest(f"{prefix}-tokenizer"),
        metadata={"hidden_size": hidden, "layer_count": len(schedule)},
    )


def _pins(*, hidden: int = 784, vocab: int = 4096):
    return build_lfm2_heterogeneous_target_pins(
        hidden_size=hidden,
        vocab_size=vocab,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=128_000,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=7,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=_digest(f"target-vocab-{vocab}"),
        runtime_evidence_sha256=_digest("runtime-semantics"),
        token_bridge_evidence_sha256=_digest(f"token-bridge-{vocab}"),
    )


_SCHEDULE = (
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
)


def _grid_options(
    graph: ModelCircuitGraph,
    *,
    high_conv_ordinals: frozenset[int] = frozenset({0, 13}),
    high_attention_ordinals: frozenset[int] = frozenset({2, 6, 10}),
    high_mlp_ordinals: frozenset[int] = frozenset({4, 9}),
):
    options = []
    for layer in discover_lfm2_source_layers(graph):
        for intermediate in (566, 567):
            mlp_mismatch = int(
                (intermediate == 567) != (layer.ordinal in high_mlp_ordinals)
            )
            if layer.layer_type == "conv":
                for internal in (704, 1024):
                    operator_mismatch = int(
                        (internal == 1024) != (layer.ordinal in high_conv_ordinals)
                    )
                    label = f"layer-{layer.ordinal}-conv-C{internal}-I{intermediate}"
                    options.append(
                        build_lfm2_short_conv_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            internal_width=internal,
                            selection_evidence_sha256=_digest(label),
                            structural_risk=10 * operator_mismatch + mlp_mismatch,
                        )
                    )
            else:
                for query, key_value in ((12, 6), (16, 8)):
                    operator_mismatch = int(
                        ((query, key_value) == (16, 8))
                        != (layer.ordinal in high_attention_ordinals)
                    )
                    label = (
                        f"layer-{layer.ordinal}-attention-Q{query}-KV{key_value}"
                        f"-I{intermediate}"
                    )
                    options.append(
                        build_lfm2_attention_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            query_heads=query,
                            key_value_heads=key_value,
                            head_dim=64,
                            selection_evidence_sha256=_digest(label),
                            structural_risk=10 * operator_mismatch + mlp_mismatch,
                        )
                    )
    return tuple(options)


def test_exact_dense_writer_witness_selects_evidence_preferred_2_3_2_sites() -> None:
    graph = _source_graph(_SCHEDULE)
    result = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=_grid_options(graph),
        target_pins=_pins(),
        exact_target_unique_parameters=57_420_144,
    )
    descriptor = result.descriptor
    high_conv = {
        layer.layer_index
        for layer in descriptor.layers
        if layer.layer_type == "conv" and layer.short_conv_internal_width == 1024
    }
    high_attention = {
        layer.layer_index
        for layer in descriptor.layers
        if layer.layer_type == "full_attention"
        and (layer.query_heads, layer.key_value_heads) == (16, 8)
    }
    high_mlp = {
        layer.layer_index
        for layer in descriptor.layers
        if layer.mlp_intermediate_size == 567
    }

    assert descriptor.accounting.unique_parameter_elements == 57_420_144
    assert descriptor.accounting.short_conv_factor_writer_parameter_elements == 0
    assert high_conv == {0, 13}
    assert high_attention == {2, 6, 10}
    assert high_mlp == {4, 9}
    assert result.receipt.allocation_decision.source_graph_fingerprint == (
        graph.fingerprint
    )
    assert result.receipt.allocation_decision.target_vocab_evidence_sha256 == (
        _pins().target_vocab_evidence_sha256
    )
    assert result.receipt.allocation_decision.runtime_evidence_sha256 == (
        _pins().runtime_evidence_sha256
    )
    assert result.receipt.allocation_decision.token_bridge_evidence_sha256 == (
        _pins().token_bridge_evidence_sha256
    )
    assert result.receipt.descriptor_sha256 == descriptor.descriptor_sha256
    assert descriptor.allocation_evidence_sha256 == (
        result.receipt.allocation_decision.allocation_decision_sha256
    )
    assert (
        result.receipt.allocation_decision.as_dict()["selection_policy"][
            "risk_is_behavioral_hard_gate"
        ]
        is False
    )


def test_exact4x_counted_body_is_representable_by_supplied_factor_writer_option() -> (
    None
):
    """The r31/I765 coordinates are a witness fixture, never selection policy."""

    graph = _source_graph(_SCHEDULE)
    mlp_widths = {
        ordinal: (327 if ordinal in {2, 4} else 765 if ordinal == 5 else 320)
        for ordinal in range(len(_SCHEDULE))
    }
    options = []
    for layer in discover_lfm2_source_layers(graph):
        intermediate = mlp_widths[layer.ordinal]
        if layer.layer_type == "conv":
            internal = 1_024 if layer.ordinal in {0, 13} else 704
            writer_rank = 31 if layer.ordinal == 13 else None
            label = (
                f"exact4x-witness-L{layer.ordinal}-conv-C{internal}"
                f"-I{intermediate}-R{writer_rank}"
            )
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=layer.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    internal_width=internal,
                    writer_rank=writer_rank,
                    selection_evidence_sha256=_digest(label),
                )
            )
        else:
            label = f"exact4x-witness-L{layer.ordinal}-attention-I{intermediate}"
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=layer.source_layer_sha256,
                    mlp_intermediate_size=intermediate,
                    query_heads=16,
                    key_value_heads=8,
                    head_dim=64,
                    selection_evidence_sha256=_digest(label),
                )
            )

    dense_target = 53_552_368
    result = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=tuple(options),
        target_pins=_pins(hidden=816, vocab=4_096),
        exact_target_unique_parameters=dense_target,
    )
    descriptor = result.descriptor

    assert descriptor.accounting.unique_parameter_elements == dense_target
    assert descriptor.layers[13].short_conv_writer_rank == 31
    assert descriptor.layers[5].mlp_intermediate_size == 765
    assert descriptor.accounting.short_conv_factor_writer_parameter_elements == (
        31 * (816 + 1_024)
    )

    route_elements = 1_933_888
    allocations = tuple(
        ContinuousChildStateAllocation(
            state_id=state_id,
            parameter_elements=route_elements,
            tensor_sha256=_digest(f"{state_id}-tensor"),
            artifact_sha256=_digest(f"{state_id}-artifact"),
        )
        for state_id in ("C_in", "C_out")
    )
    counted = build_counted_child_state_plan(
        parent_unique_parameters=229_693_184,
        target_total_parameter_elements=57_420_144,
        continuous_allocations=allocations,
    )
    authenticated = verify_counted_child_state(
        counted,
        physical_dense_parameter_elements=descriptor.accounting.unique_parameter_elements,
        observed_allocations=allocations,
    )

    assert counted.dense_model_parameter_elements == dense_target
    assert counted.continuous_parameter_elements == 2 * route_elements
    assert authenticated["target_total_parameter_elements"] == 57_420_144


def test_exact4x_counted_body_is_selected_from_automatic_factor_lattice() -> None:
    """The production body count is reached without a pinned rank or layer."""

    graph = _source_graph(_SCHEDULE, intermediate=2_560)
    hidden = 816
    dense_target = 53_552_368
    policy = derive_lfm2_automatic_mlp_width_policy(
        layer_types=_SCHEDULE,
        source_widths=(2_560,) * len(_SCHEDULE),
        short_conv_internal_widths=(704, 1_024),
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        target_hidden_size=hidden,
        target_vocab_size=4_096,
        exact_target_unique_parameters=dense_target,
        conv_kernel_size=3,
        discover_exact_short_conv_writer_ranks=True,
    )
    discovered = discover_lfm2_source_layers(graph)
    factor_source = min(
        (
            layer.source_layer_sha256
            for layer in discovered
            if layer.layer_type == "conv"
        )
    )
    factor_ordinal = next(
        layer.ordinal
        for layer in discovered
        if layer.source_layer_sha256 == factor_source
    )
    options = []
    for layer, widths in zip(discovered, policy.widths_by_layer, strict=True):
        for intermediate in widths:
            if layer.layer_type == "conv":
                for internal in (704, 1_024):
                    ranks = (None,)
                    if layer.source_layer_sha256 == factor_source:
                        ranks += policy.short_conv_writer_ranks
                    for writer_rank in ranks:
                        label = (
                            f"automatic-exact4x-{layer.source_layer_sha256}-"
                            f"C{internal}-I{intermediate}-R{writer_rank}"
                        )
                        options.append(
                            build_lfm2_short_conv_layer_option(
                                source_layer_sha256=layer.source_layer_sha256,
                                mlp_intermediate_size=intermediate,
                                internal_width=internal,
                                writer_rank=writer_rank,
                                selection_evidence_sha256=_digest(label),
                            )
                        )
            else:
                for query, key_value in ((12, 6), (16, 8)):
                    label = (
                        f"automatic-exact4x-{layer.source_layer_sha256}-"
                        f"Q{query}-KV{key_value}-I{intermediate}"
                    )
                    options.append(
                        build_lfm2_attention_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            query_heads=query,
                            key_value_heads=key_value,
                            head_dim=64,
                            selection_evidence_sha256=_digest(label),
                        )
                    )

    result = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=tuple(options),
        target_pins=_pins(hidden=hidden, vocab=4_096),
        exact_target_unique_parameters=dense_target,
        max_states=20_000,
    )
    factor_layers = tuple(
        layer
        for layer in result.descriptor.layers
        if layer.short_conv_writer_rank is not None
    )

    assert result.descriptor.accounting.unique_parameter_elements == dense_target
    assert len(factor_layers) == 1
    assert factor_layers[0].layer_index == factor_ordinal
    assert factor_layers[0].short_conv_writer_rank in (
        policy.short_conv_writer_ranks
    )
    assert result.budget_selection.topology_selection.peak_retained_states < 20_000


def test_real_h784_broad_mlp_bank_has_bounded_exact_solver_frontier() -> None:
    graph = _source_graph(_SCHEDULE, intermediate=2_560)
    policy = derive_lfm2_automatic_mlp_width_policy(
        layer_types=_SCHEDULE,
        source_widths=(2_560,) * len(_SCHEDULE),
        short_conv_internal_widths=(704, 1_024),
        attention_head_options=((12, 6, 64), (16, 8, 64)),
        target_hidden_size=784,
        target_vocab_size=4_096,
        exact_target_unique_parameters=57_420_144,
        conv_kernel_size=3,
    )
    options = []
    for layer, widths in zip(
        discover_lfm2_source_layers(graph),
        policy.widths_by_layer,
        strict=True,
    ):
        for intermediate in widths:
            if layer.layer_type == "conv":
                for internal in (704, 1_024):
                    label = f"stress-{layer.ordinal}-conv-C{internal}-I{intermediate}"
                    options.append(
                        build_lfm2_short_conv_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            internal_width=internal,
                            selection_evidence_sha256=_digest(label),
                            structural_risk=(
                                (2_560 - intermediate) / 2_560
                                + (1_024 - internal) / 100_000
                            ),
                        )
                    )
            else:
                for query, key_value in ((12, 6), (16, 8)):
                    label = (
                        f"stress-{layer.ordinal}-attention-Q{query}-"
                        f"KV{key_value}-I{intermediate}"
                    )
                    options.append(
                        build_lfm2_attention_layer_option(
                            source_layer_sha256=layer.source_layer_sha256,
                            mlp_intermediate_size=intermediate,
                            query_heads=query,
                            key_value_heads=key_value,
                            head_dim=64,
                            selection_evidence_sha256=_digest(label),
                            structural_risk=(
                                (2_560 - intermediate) / 2_560 + (16 - query) / 100_000
                            ),
                        )
                    )

    assert 0 < len(options) <= 780
    result = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=tuple(options),
        target_pins=_pins(),
        exact_target_unique_parameters=57_420_144,
        max_states=20_000,
    )
    exact_search = result.budget_selection.topology_selection

    assert result.descriptor.accounting.unique_parameter_elements == 57_420_144
    assert [layer.mlp_intermediate_size for layer in result.descriptor.layers] == [
        621
    ] * len(_SCHEDULE)
    assert exact_search.peak_retained_states < 20_000
    assert exact_search.states_considered < 6_000_000


def test_option_permutation_is_deterministic_and_alternate_risk_moves_sites() -> None:
    graph = _source_graph(_SCHEDULE)
    options = _grid_options(graph)
    forward = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=options,
        target_pins=_pins(),
        exact_target_unique_parameters=57_420_144,
    )
    reversed_input = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=tuple(reversed(options)),
        target_pins=_pins().serialize(),
        exact_target_unique_parameters=57_420_144,
    )
    assert reversed_input.descriptor.serialize() == forward.descriptor.serialize()
    assert reversed_input.receipt.serialize() == forward.receipt.serialize()

    alternate = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=_grid_options(
            graph,
            high_conv_ordinals=frozenset({1, 11}),
            high_attention_ordinals=frozenset({4, 8, 12}),
            high_mlp_ordinals=frozenset({3, 7}),
        ),
        target_pins=_pins(),
        exact_target_unique_parameters=57_420_144,
    )
    assert alternate.descriptor.descriptor_sha256 != (
        forward.descriptor.descriptor_sha256
    )
    assert {
        layer.layer_index
        for layer in alternate.descriptor.layers
        if layer.layer_type == "conv" and layer.short_conv_internal_width == 1024
    } == {1, 11}
    assert {
        layer.layer_index
        for layer in alternate.descriptor.layers
        if layer.layer_type == "full_attention" and layer.query_heads == 16
    } == {4, 8, 12}
    assert {
        layer.layer_index
        for layer in alternate.descriptor.layers
        if layer.mlp_intermediate_size == 567
    } == {3, 7}


def test_arbitrary_schedule_renamed_paths_and_rank_writer_are_representable() -> None:
    schedule = ("full_attention", "conv", "conv")
    graph = _source_graph(
        schedule,
        prefix="nothing_matches_hf_module_names",
        hidden=64,
        vocab=256,
        intermediate=48,
        conv_width=64,
        query_heads=8,
        key_value_heads=4,
        head_dim=8,
    )
    discovered = discover_lfm2_source_layers(graph)
    assert tuple(layer.graph_layer_index for layer in discovered) == (101, 108, 115)
    assert tuple(layer.layer_type for layer in discovered) == schedule
    options = (
        build_lfm2_attention_layer_option(
            source_layer_sha256=discovered[0].source_layer_sha256,
            mlp_intermediate_size=20,
            query_heads=4,
            key_value_heads=2,
            head_dim=8,
            selection_evidence_sha256=_digest("renamed-attention"),
        ),
        build_lfm2_short_conv_layer_option(
            source_layer_sha256=discovered[1].source_layer_sha256,
            mlp_intermediate_size=20,
            internal_width=24,
            writer_rank=7,
            selection_evidence_sha256=_digest("renamed-rank-writer"),
            structural_risk=1e100,
        ),
        build_lfm2_short_conv_layer_option(
            source_layer_sha256=discovered[2].source_layer_sha256,
            mlp_intermediate_size=20,
            internal_width=24,
            selection_evidence_sha256=_digest("renamed-dense-writer"),
            structural_risk=1e100,
        ),
    )
    pins = _pins(hidden=32, vocab=64)
    # interface+final norm 4,128; layer norms 192; MLP 5,760;
    # attention 3,088; rank conv 2,768; dense conv 3,144.
    target = 19_080
    result = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=options,
        target_pins=pins,
        exact_target_unique_parameters=target,
    )

    assert result.descriptor.accounting.unique_parameter_elements == target
    assert result.descriptor.layers[1].short_conv_writer_rank == 7
    assert result.budget_selection.total_predicted_risk > 1e99
    assert (
        result.receipt.allocation_decision.as_dict()["selection_policy"][
            "risk_is_behavioral_hard_gate"
        ]
        is False
    )


def test_tamper_incomplete_duplicate_wrong_kind_and_unattainable_fail_closed() -> None:
    graph = _source_graph(_SCHEDULE)
    options = _grid_options(graph)
    tampered = options[0].as_dict()
    tampered["mlp_intermediate_size"] += 1
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="seal"):
        Lfm2AuthenticatedLayerPhysicalOption.restore(tampered)

    pins_payload = _pins().as_dict()
    pins_payload["runtime_semantics"]["norm_eps"] = 2e-5
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="seal"):
        Lfm2HeterogeneousTargetPins.restore(pins_payload)

    first_source = options[0].source_layer_sha256
    incomplete = tuple(
        option for option in options if option.source_layer_sha256 != first_source
    )
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="incomplete"):
        select_lfm2_heterogeneous_physical_layout(
            source_graph=graph,
            layer_options=incomplete,
            target_pins=_pins(),
            exact_target_unique_parameters=57_420_144,
        )
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="duplicate"):
        select_lfm2_heterogeneous_physical_layout(
            source_graph=graph,
            layer_options=options + (options[0],),
            target_pins=_pins(),
            exact_target_unique_parameters=57_420_144,
        )

    conv_layer = next(
        layer
        for layer in discover_lfm2_source_layers(graph)
        if layer.layer_type == "conv"
    )
    wrong_kind = build_lfm2_attention_layer_option(
        source_layer_sha256=conv_layer.source_layer_sha256,
        mlp_intermediate_size=566,
        query_heads=12,
        key_value_heads=6,
        head_dim=64,
        selection_evidence_sha256=_digest("wrong-kind"),
    )
    without_conv = tuple(
        option
        for option in options
        if option.source_layer_sha256 != conv_layer.source_layer_sha256
    )
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="wrong"):
        select_lfm2_heterogeneous_physical_layout(
            source_graph=graph,
            layer_options=without_conv + (wrong_kind,),
            target_pins=_pins(),
            exact_target_unique_parameters=57_420_144,
        )
    with pytest.raises(Lfm2HeterogeneousLayoutUnattainable, match="exactly"):
        select_lfm2_heterogeneous_physical_layout(
            source_graph=graph,
            layer_options=options,
            target_pins=_pins(),
            exact_target_unique_parameters=57_420_145,
        )

    valid = select_lfm2_heterogeneous_physical_layout(
        source_graph=graph,
        layer_options=options,
        target_pins=_pins(),
        exact_target_unique_parameters=57_420_144,
    )
    assert (
        Lfm2HeterogeneousAllocationDecisionCore.restore(
            valid.receipt.allocation_decision.as_dict()
        )
        == valid.receipt.allocation_decision
    )
    assert (
        Lfm2HeterogeneousLayoutSelectionReceipt.restore(valid.receipt.serialize())
        == valid.receipt
    )
    receipt_payload = json.loads(valid.receipt.serialize())
    receipt_payload["descriptor_sha256"] = "0" * 64
    with pytest.raises(Lfm2HeterogeneousLayoutSelectionError, match="seal"):
        Lfm2HeterogeneousLayoutSelectionReceipt.restore(receipt_payload)
