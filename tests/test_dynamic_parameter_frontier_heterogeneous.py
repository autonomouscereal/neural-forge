from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_parameter_frontier import (
    CAP_HETEROGENEOUS_LAYER_LAYOUT,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    DynamicParameterFrontierError,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_KIND,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
    LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    Lfm2DenseTargetTopologyCatalog,
    Lfm2HeterogeneousLayoutSelectionInput,
    select_best_executable_dense_candidate,
    synthesize_dense_parameter_frontier,
)
from neural_forge.hf_lfm2_dynamic_physical_layout import (
    build_lfm2_heterogeneous_physical_layout,
    restore_lfm2_heterogeneous_physical_layout,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA,
    Lfm2HeterogeneousLayoutSelectionReceipt,
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)
from test_dynamic_parameter_frontier import _lfm2_graph


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _rebuild_descriptor(descriptor, **changes):
    values = {
        "hidden_size": descriptor.hidden_size,
        "vocab_size": descriptor.vocab_size,
        "conv_kernel_size": descriptor.conv_kernel_size,
        "rope_theta": descriptor.rope_theta,
        "norm_eps": descriptor.norm_eps,
        "max_position_embeddings": descriptor.max_position_embeddings,
        "use_cache": descriptor.use_cache,
        "attention_implementation": descriptor.attention_implementation,
        "bos_token_id": descriptor.bos_token_id,
        "eos_token_id": descriptor.eos_token_id,
        "pad_token_id": descriptor.pad_token_id,
        "block_auto_adjust_ff_dim": descriptor.block_auto_adjust_ff_dim,
        "layers": descriptor.layers,
        "source_topology_sha256": descriptor.source_topology_sha256,
        "allocation_evidence_sha256": descriptor.allocation_evidence_sha256,
        "token_bridge_evidence_sha256": (
            descriptor.token_bridge_evidence_sha256
        ),
        "expected_unique_parameter_count": (
            descriptor.accounting.unique_parameter_elements
        ),
    }
    values.update(changes)
    return build_lfm2_heterogeneous_physical_layout(**values)


def _receipt_for_descriptor(selection_receipt, descriptor):
    decision = selection_receipt.allocation_decision
    body = {
        "schema": LFM2_HETEROGENEOUS_SELECTION_RECEIPT_SCHEMA,
        "allocation_decision": decision.as_dict(),
        "descriptor_sha256": descriptor.descriptor_sha256,
        "exact_unique_parameter_count": decision.exact_target_unique_parameters,
        "allocation_precedes_descriptor": True,
    }
    return Lfm2HeterogeneousLayoutSelectionReceipt(
        allocation_decision=decision,
        descriptor_sha256=descriptor.descriptor_sha256,
        receipt_sha256=canonical_sha256(body),
    )


def _configuration_with_artifacts(candidate, descriptor, selection_receipt):
    configuration = dict(candidate.configuration)
    configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY] = (
        descriptor.serialize().decode("ascii")
    )
    configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY] = (
        descriptor.descriptor_sha256
    )
    configuration[LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY] = (
        selection_receipt.serialize().decode("ascii")
    )
    configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY
    ] = selection_receipt.receipt_sha256
    return tuple(configuration.items())


def _selection_input(*, factor_writer_rank: int | None = None):
    graph = _lfm2_graph(
        vocab=64,
        hidden=32,
        intermediate=64,
        query_heads=4,
        key_value_heads=2,
        head_dim=8,
        layer_types=("conv", "full_attention"),
    )
    options = []
    for layer in discover_lfm2_source_layers(graph):
        if layer.layer_type == "conv":
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=layer.source_layer_sha256,
                    mlp_intermediate_size=20,
                    internal_width=12,
                    writer_rank=factor_writer_rank,
                    selection_evidence_sha256=_digest("conv-evidence"),
                    structural_risk=0.2,
                )
            )
        else:
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=layer.source_layer_sha256,
                    mlp_intermediate_size=22,
                    query_heads=2,
                    key_value_heads=1,
                    head_dim=4,
                    selection_evidence_sha256=_digest("attention-evidence"),
                    structural_risk=0.1,
                )
            )
    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=16,
        vocab_size=64,
        conv_kernel_size=3,
        rope_theta=1_000_000.0,
        norm_eps=1e-5,
        max_position_embeddings=4096,
        use_cache=True,
        attention_implementation="sdpa",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=_digest("target-vocab"),
        runtime_evidence_sha256=_digest("runtime"),
        token_bridge_evidence_sha256=_digest("same-vocab-bridge-evidence"),
    )
    dense_writer = factor_writer_rank is None
    # Untied interface + five norms + two MLPs + one attention + one ShortConv.
    short_conv_writer = 16 * 12 if dense_writer else factor_writer_rank * (16 + 12)
    exact = (
        2 * 64 * 16
        + 5 * 16
        + 3 * 16 * (20 + 22)
        + (2 * 16 * 4 * (2 + 1) + 2 * 4)
        + 3 * 16 * 12
        + 3 * 12
        + short_conv_writer
    )
    return (
        graph,
        Lfm2HeterogeneousLayoutSelectionInput(
            target_pins=pins,
            layer_options=tuple(options),
        ),
        exact,
    )


def test_existing_catalog_returns_authenticated_exact_heterogeneous_candidate() -> None:
    graph, selection_input, exact = _selection_input()
    result = synthesize_dense_parameter_frontier(
        graph,
        DenseParameterBudgetRequest(
            target_family="lfm2",
            target_modalities=("text",),
            target_unique_parameters=exact,
            match_policy="exact",
        ),
        catalogs=DenseTargetTopologyCatalogRegistry(
            (
                Lfm2DenseTargetTopologyCatalog(
                    target_token_interface_mode="untied",
                    heterogeneous_layout_selection_input=selection_input,
                ),
            )
        ),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    selected = select_best_executable_dense_candidate(result)
    assert selected.selected_candidate_rank == 0
    assert candidate.unique_parameters == exact
    assert candidate.logical_parameters == exact
    assert candidate.native_ties == ()
    assert candidate.native_tied_parameter_count == 0
    assert candidate.ordered_source_spans == ((0,), (1,))
    assert candidate.compiler_executable is True
    assert CAP_HETEROGENEOUS_LAYER_LAYOUT in (candidate.required_compiler_capabilities)
    assert candidate.factorized is False

    configuration = dict(candidate.configuration)
    descriptor = restore_lfm2_heterogeneous_physical_layout(
        configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY],
        expected_descriptor_sha256=configuration[
            LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
        ],
    )
    selection_receipt = Lfm2HeterogeneousLayoutSelectionReceipt.restore(
        configuration[LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY]
    )
    assert (
        selection_receipt.receipt_sha256
        == configuration[LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY]
    )
    assert descriptor.descriptor_sha256 == selection_receipt.descriptor_sha256
    assert descriptor.source_topology_sha256 == graph.fingerprint
    assert descriptor.accounting.unique_parameter_elements == exact
    assert descriptor.accounting.parameter_bytes_fp32 == 4 * exact
    assert all(layer.short_conv_writer_rank is None for layer in descriptor.layers)

    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    assert receipt.physical_layout_kind == LFM2_HETEROGENEOUS_LAYOUT_KIND
    assert receipt.physical_layout_descriptor_sha256 == descriptor.descriptor_sha256
    assert receipt.physical_layout_selection_receipt_sha256 == (
        selection_receipt.receipt_sha256
    )
    assert receipt.physical_layout_source_topology_sha256 == graph.fingerprint
    serialized_receipt = receipt.as_dict()
    assert serialized_receipt["shared_hidden_intermediate_and_head_widths"] is False
    assert serialized_receipt["heterogeneous_per_layer_widths_claimed"] is True
    assert receipt.allocation_selection.exact_unique_parameters == exact
    target_request = candidate.to_target_request(child_name="heterogeneous-blank")
    assert target_request.configuration == candidate.configuration
    assert target_request.component_allocation_receipt == receipt


def test_heterogeneous_candidate_tamper_and_factor_writer_authentication() -> None:
    graph, selection_input, exact = _selection_input()
    frontier = Lfm2DenseTargetTopologyCatalog(
        target_token_interface_mode="untied",
        heterogeneous_layout_selection_input=selection_input,
    ).enumerate_frontier(
        graph,
        requested_unique_parameters=exact,
        target_vocabulary_size=64,
        match_policy="exact",
        maximum_candidates=1,
    )
    candidate = frontier.candidates[0]
    tampered = dict(candidate.configuration)
    tampered[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY] = "0" * 64
    with pytest.raises(
        DynamicParameterFrontierError,
        match="failed authentication",
    ):
        replace(candidate, configuration=tuple(tampered.items()))

    receipt = candidate.component_allocation_receipt
    assert receipt is not None
    changed_receipt = replace(
        receipt,
        physical_layout_selection_receipt_sha256="0" * 64,
    )
    changed_configuration = dict(candidate.configuration)
    changed_configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = (
        changed_receipt.fingerprint
    )
    with pytest.raises(
        DynamicParameterFrontierError,
        match="bindings differ",
    ):
        replace(
            candidate,
            configuration=tuple(changed_configuration.items()),
            component_allocation_receipt=changed_receipt,
        )

    allocation = receipt.allocation_selection
    topology = allocation.topology_selection
    allocation_tampers = (
        replace(
            allocation,
            problem_fingerprint=_digest("different-budget-problem"),
        ),
        replace(
            allocation,
            topology_selection=replace(
                topology,
                problem_fingerprint=_digest("different-topology-problem"),
            ),
        ),
        replace(
            allocation,
            topology_selection=replace(
                topology,
                selected_option_ids=tuple(reversed(topology.selected_option_ids)),
            ),
        ),
        replace(
            allocation,
            topology_selection=replace(
                topology,
                total_predicted_risk=topology.total_predicted_risk + 0.25,
            ),
        ),
    )
    for allocation_tamper in allocation_tampers:
        changed_receipt = replace(
            receipt,
            allocation_selection=allocation_tamper,
        )
        changed_configuration = dict(candidate.configuration)
        changed_configuration[COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY] = (
            changed_receipt.fingerprint
        )
        with pytest.raises(
            DynamicParameterFrontierError,
            match="bindings differ",
        ):
            replace(
                candidate,
                configuration=tuple(changed_configuration.items()),
                component_allocation_receipt=changed_receipt,
            )

    configuration = dict(candidate.configuration)
    descriptor = restore_lfm2_heterogeneous_physical_layout(
        configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY],
        expected_descriptor_sha256=configuration[
            LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
        ],
    )
    selection_receipt = Lfm2HeterogeneousLayoutSelectionReceipt.restore(
        configuration[
            LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY
        ]
    )
    changed_layers = (
        replace(
            descriptor.layers[0],
            selection_evidence_sha256=_digest("different-layer-option"),
        ),
        *descriptor.layers[1:],
    )
    artifact_tampers = (
        _rebuild_descriptor(
            descriptor,
            token_bridge_evidence_sha256=_digest("different-token-bridge"),
        ),
        _rebuild_descriptor(descriptor, layers=changed_layers),
        _rebuild_descriptor(descriptor, norm_eps=descriptor.norm_eps * 2),
    )
    for artifact_tamper in artifact_tampers:
        changed_selection_receipt = _receipt_for_descriptor(
            selection_receipt,
            artifact_tamper,
        )
        with pytest.raises(
            DynamicParameterFrontierError,
            match="seals differ",
        ):
            replace(
                candidate,
                configuration=_configuration_with_artifacts(
                    candidate,
                    artifact_tamper,
                    changed_selection_receipt,
                ),
            )

    factor_graph, factor_input, factor_exact = _selection_input(factor_writer_rank=4)
    factor_frontier = Lfm2DenseTargetTopologyCatalog(
        target_token_interface_mode="untied",
        heterogeneous_layout_selection_input=factor_input,
    ).enumerate_frontier(
        factor_graph,
        requested_unique_parameters=factor_exact,
        target_vocabulary_size=64,
        match_policy="exact",
        maximum_candidates=1,
    )
    factor_candidate = factor_frontier.candidates[0]
    factor_configuration = dict(factor_candidate.configuration)
    factor_descriptor = restore_lfm2_heterogeneous_physical_layout(
        factor_configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY],
        expected_descriptor_sha256=factor_configuration[
            LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
        ],
    )

    assert factor_candidate.unique_parameters == factor_exact
    assert factor_descriptor.layers[0].short_conv_writer_rank == 4
    assert (
        factor_descriptor.accounting.short_conv_factor_writer_parameter_elements
        == 4 * (16 + 12)
    )

    unsealed_factor = factor_descriptor.as_dict()
    unsealed_factor["layers"][0]["short_conv_writer_rank"] = 5
    factor_configuration[LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY] = (
        canonical_json_bytes(unsealed_factor).decode("ascii")
    )
    with pytest.raises(
        DynamicParameterFrontierError,
        match="failed authentication",
    ):
        replace(
            factor_candidate,
            configuration=tuple(factor_configuration.items()),
        )


def test_legacy_uniform_catalog_output_remains_byte_identical() -> None:
    graph = _lfm2_graph()
    frontier = Lfm2DenseTargetTopologyCatalog().enumerate_frontier(
        graph,
        requested_unique_parameters=graph.unique_parameter_count // 4,
        target_vocabulary_size=None,
        match_policy="at_or_below",
        maximum_candidates=3,
    )
    payload = {
        "candidates": [candidate.as_dict() for candidate in frontier.candidates],
        "diagnostics": frontier.diagnostics,
    }
    # Captured from the parent branch immediately before heterogeneous support.
    assert hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == (
        "931dea7d0e1216d561da7cd483811fcad4602b2a43faae0d9ddd2cc201a66d4e"
    )
    receipt = frontier.candidates[0].component_allocation_receipt
    assert receipt is not None
    assert not {
        "physical_layout_kind",
        "physical_layout_descriptor_sha256",
        "physical_layout_selection_receipt_sha256",
        "physical_layout_source_topology_sha256",
    }.intersection(receipt.as_dict())
