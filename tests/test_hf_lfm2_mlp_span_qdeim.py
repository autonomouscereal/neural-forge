from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_mlp_qdeim import (  # noqa: E402
    Lfm2MlpQdeimError,
    Lfm2MlpQdeimSpanSource,
    _deterministic_source_covering_pivoted_qr_indices,
    assemble_lfm2_mlp_qdeim_span_sources,
    compile_lfm2_mlp_span_coupled_nonlinear_qdeim,
    lfm2_mlp_qdeim_source_state_sha256,
    lfm2_mlp_qdeim_span_functional_evidence_sha256,
    lfm2_mlp_qdeim_target_template_sha256,
)
from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldSpan  # noqa: E402
from neural_forge.hf_lfm2_structural_residual_attention_fold import (  # noqa: E402
    lfm2_authenticated_tensor_sha256,
)


def _span_rows(source_layer_count: int) -> tuple[torch.Tensor, ...]:
    latent = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    if source_layer_count == 2:
        latent = latent[:, :2]
        return (
            torch.stack((latent[:, 0], latent[:, 1]), dim=1),
            torch.stack(
                (latent[:, 0] + latent[:, 1], latent[:, 0] - latent[:, 1]),
                dim=1,
            ),
        )
    if source_layer_count == 3:
        return (
            torch.stack((latent[:, 0], latent[:, 1]), dim=1),
            torch.stack(
                (latent[:, 1] + latent[:, 2], latent[:, 2]),
                dim=1,
            ),
            torch.stack(
                (latent[:, 0] + latent[:, 2], latent[:, 0] - latent[:, 1]),
                dim=1,
            ),
        )
    raise AssertionError("The synthetic fixture supports two or three layers.")


def test_covering_seed_finds_a_basis_that_greedy_mandatory_qr_strands() -> None:
    """Exact 103x3 audit counterexample for spend-spare-pivots-first QR."""

    basis = torch.zeros((103, 3), dtype=torch.float64)
    basis[0] = torch.tensor((0.0, 1.0, 0.0), dtype=torch.float64)
    basis[1] = torch.tensor((0.99**0.5, 0.0, 0.0), dtype=torch.float64)
    basis[2:102, 2] = 0.1
    basis[102] = torch.tensor((0.1, 0.0, 0.0), dtype=torch.float64)
    groups = (tuple(range(102)), (102,))

    selected, residual_norms = _deterministic_source_covering_pivoted_qr_indices(
        basis,
        groups,
    )

    # B.T @ B is identity to FP64 precision, so this is a valid POD basis.
    torch.testing.assert_close(
        basis.T @ basis,
        torch.eye(3, dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )
    assert selected == (102, 0, 2)
    assert 102 in selected and any(index < 102 for index in selected)
    sampled = basis[list(selected)]
    assert int(torch.linalg.matrix_rank(sampled).item()) == 3
    assert abs(float(torch.linalg.det(sampled).item())) == pytest.approx(0.01)
    assert bool((residual_norms > 0.0).all().item())


def _stability_audit_basis() -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    machine_epsilon = torch.finfo(torch.float64).eps
    weak_residual = 1.35 * machine_epsilon
    critical = torch.tensor(
        [
            [0.5, 0.0, 0.0],
            [0.0, 0.49, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.4],
            [0.5, 0.0, weak_residual],
        ],
        dtype=torch.float64,
    )
    remainder = torch.eye(3, dtype=torch.float64) - critical.T @ critical
    eigenvalues, eigenvectors = torch.linalg.eigh(remainder)
    assert float(eigenvalues.min().item()) >= -1e-15
    rows_per_mode = 300
    fill = torch.cat(
        [
            (
                (eigenvalues[index].clamp_min(0.0) / rows_per_mode).sqrt()
                * eigenvectors[:, index].reshape(1, 3).repeat(rows_per_mode, 1)
            )
            for index in range(3)
        ],
        dim=0,
    )
    group_size = 902
    basis = torch.cat(
        (
            torch.cat((critical[:2], fill), dim=0),
            torch.cat(
                (
                    critical[2:4],
                    torch.zeros((group_size - 2, 3), dtype=torch.float64),
                ),
                dim=0,
            ),
            torch.cat(
                (
                    critical[4:5],
                    torch.zeros((group_size - 1, 3), dtype=torch.float64),
                ),
                dim=0,
            ),
        ),
        dim=0,
    )
    # Remove only round-off drift so this is an exact numerical POD basis.
    gram_values, gram_vectors = torch.linalg.eigh(basis.T @ basis)
    basis = basis @ (gram_vectors @ torch.diag(gram_values.rsqrt()) @ gram_vectors.T)
    groups = (
        tuple(range(0, group_size)),
        tuple(range(group_size, 2 * group_size)),
        tuple(range(2 * group_size, 3 * group_size)),
    )
    return basis.contiguous(), groups


def test_covering_search_skips_weak_seed_that_fails_final_svd_stability() -> None:
    basis, groups = _stability_audit_basis()
    machine_epsilon = torch.finfo(torch.float64).eps
    weak_selection = (0, 902, 1804)
    weak_singular = torch.linalg.svdvals(basis[list(weak_selection)])
    weak_tolerance = 3 * machine_epsilon * float(weak_singular.max().item())

    selected, residual_norms = _deterministic_source_covering_pivoted_qr_indices(
        basis,
        groups,
    )

    assert float(weak_singular.min().item()) == pytest.approx(2.1196e-16, rel=5e-4)
    assert float(weak_singular.min().item()) <= weak_tolerance
    assert selected == (1, 903, 1804)
    selected_singular = torch.linalg.svdvals(basis[list(selected)])
    torch.testing.assert_close(
        selected_singular,
        torch.tensor((0.5, 0.49, 0.4), dtype=torch.float64),
        atol=1e-14,
        rtol=1e-14,
    )
    selected_tolerance = 3 * machine_epsilon * float(selected_singular.max().item())
    assert float(selected_singular.min().item()) > selected_tolerance
    assert float(residual_norms.min().item()) > selected_tolerance


def test_stability_audit_rows_compile_without_the_prior_false_rejection() -> None:
    basis, groups = _stability_audit_basis()
    singular = torch.diag(torch.tensor((3.0, 2.0, 1.0), dtype=torch.float64))
    union_features = singular @ basis.T
    span = Lfm2ModelFoldSpan((0, 1, 2), target_layer_index=0)
    sources = []
    for layer_index, group in enumerate(groups):
        rows = union_features[:, list(group)].contiguous()
        intermediate = int(rows.shape[1])
        gate = torch.zeros((intermediate, 2), dtype=torch.float64)
        up = torch.zeros_like(gate)
        down = torch.zeros((2, intermediate), dtype=torch.float64)
        decoder = torch.eye(2, dtype=torch.float64)
        chart = torch.eye(2, dtype=torch.float64)
        sources.append(
            Lfm2MlpQdeimSpanSource(
                source_layer_index=layer_index,
                source_gate_weight=gate,
                source_up_weight=up,
                source_down_weight=down,
                expected_source_state_sha256=lfm2_mlp_qdeim_source_state_sha256(
                    gate,
                    up,
                    down,
                ),
                cached_parent_post_swiglu_rows=rows,
                expected_cached_parent_post_swiglu_sha256=(
                    lfm2_authenticated_tensor_sha256(rows)
                ),
                cache_receipt_sha256=f"{layer_index + 11:064x}",
                normalized_input_decoder=decoder,
                expected_normalized_input_decoder_sha256=(
                    lfm2_authenticated_tensor_sha256(decoder)
                ),
                residual_output_chart=chart,
                expected_residual_output_chart_sha256=(
                    lfm2_authenticated_tensor_sha256(chart)
                ),
            )
        )
    sources = tuple(sources)
    target_gate = torch.zeros((3, 2), dtype=torch.bfloat16)
    target_up = torch.zeros_like(target_gate)
    target_down = torch.zeros((2, 3), dtype=torch.bfloat16)

    result = compile_lfm2_mlp_span_coupled_nonlinear_qdeim(
        span=span,
        sources=sources,
        expected_functional_evidence_sha256=(
            lfm2_mlp_qdeim_span_functional_evidence_sha256(
                span=span,
                sources=sources,
            )
        ),
        target_gate_weight_template=target_gate,
        target_up_weight_template=target_up,
        target_down_weight_template=target_down,
        expected_target_template_sha256=lfm2_mlp_qdeim_target_template_sha256(
            target_gate,
            target_up,
            target_down,
        ),
    )

    assert result.selected_source_neurons == ((0, 1), (1, 1), (2, 0))
    sampled = result.functional_pod_basis[
        [1, groups[1][1], groups[2][0]],
        :,
    ]
    sampled_singular = torch.linalg.svdvals(sampled)
    tolerance = (
        3 * torch.finfo(torch.float64).eps * float(sampled_singular.max().item())
    )
    assert float(sampled_singular.min().item()) > tolerance
    assert not result.report["source_covering_seed"][
        "weak_above_epsilon_seed_can_terminate_search"
    ]


def _source(layer_index: int, rows: torch.Tensor) -> Lfm2MlpQdeimSpanSource:
    offset = float(layer_index + 1)
    gate = torch.tensor(
        [
            [1.0 + offset, -0.5, 0.25 * offset],
            [-0.75, 0.5 + offset, 1.25],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    up = torch.tensor(
        [
            [0.25, 1.0 + offset, -0.5],
            [1.5 + offset, -0.25, 0.75],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    down = torch.tensor(
        [
            [1.0 + offset, -0.5],
            [0.75, 1.25 + offset],
            [-1.0, 0.5 + 0.25 * offset],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    decoder = torch.tensor(
        [
            [1.0, 0.125 * offset, 0.0],
            [0.0, 1.0, 0.25 * offset],
        ],
        dtype=torch.float64,
    )
    chart = torch.tensor(
        [
            [1.0, 0.0],
            [0.125 * offset, 1.0],
            [0.0, 0.25 * offset],
        ],
        dtype=torch.float64,
    )
    return Lfm2MlpQdeimSpanSource(
        source_layer_index=layer_index,
        source_gate_weight=gate,
        source_up_weight=up,
        source_down_weight=down,
        expected_source_state_sha256=lfm2_mlp_qdeim_source_state_sha256(
            gate,
            up,
            down,
        ),
        cached_parent_post_swiglu_rows=rows,
        expected_cached_parent_post_swiglu_sha256=(
            lfm2_authenticated_tensor_sha256(rows)
        ),
        cache_receipt_sha256=f"{layer_index + 1:064x}",
        normalized_input_decoder=decoder,
        expected_normalized_input_decoder_sha256=(
            lfm2_authenticated_tensor_sha256(decoder)
        ),
        residual_output_chart=chart,
        expected_residual_output_chart_sha256=(lfm2_authenticated_tensor_sha256(chart)),
    )


def _fixture(source_layer_count: int) -> dict[str, Any]:
    target_intermediate = source_layer_count
    span = Lfm2ModelFoldSpan(
        tuple(range(4, 4 + source_layer_count)),
        target_layer_index=2,
    )
    sources = tuple(
        _source(layer_index, rows)
        for layer_index, rows in zip(
            span.source_layer_indices,
            _span_rows(source_layer_count),
            strict=True,
        )
    )
    gate_template = torch.zeros(
        (target_intermediate, 2),
        dtype=torch.bfloat16,
    )
    up_template = torch.zeros_like(gate_template)
    down_template = torch.zeros(
        (2, target_intermediate),
        dtype=torch.bfloat16,
    )
    weights = tuple(1.0 + 0.125 * index for index in range(6))
    return {
        "span": span,
        "sources": sources,
        "gate_template": gate_template,
        "up_template": up_template,
        "down_template": down_template,
        "row_weights": weights,
    }


def _compile(fixture: dict[str, Any], **overrides: Any):
    arguments = {
        "span": fixture["span"],
        "sources": fixture["sources"],
        "expected_functional_evidence_sha256": (
            lfm2_mlp_qdeim_span_functional_evidence_sha256(
                span=fixture["span"],
                sources=fixture["sources"],
                row_sample_weights=fixture["row_weights"],
            )
        ),
        "target_gate_weight_template": fixture["gate_template"],
        "target_up_weight_template": fixture["up_template"],
        "target_down_weight_template": fixture["down_template"],
        "expected_target_template_sha256": (
            lfm2_mlp_qdeim_target_template_sha256(
                fixture["gate_template"],
                fixture["up_template"],
                fixture["down_template"],
            )
        ),
        "row_sample_weights": fixture["row_weights"],
    }
    arguments.update(overrides)
    return compile_lfm2_mlp_span_coupled_nonlinear_qdeim(**arguments)


def _assert_exact_span_construction(
    fixture: dict[str, Any],
    result: Any,
) -> None:
    span = fixture["span"]
    sources = fixture["sources"]
    union_rows = torch.cat(
        [source.cached_parent_post_swiglu_rows for source in sources],
        dim=1,
    )
    offsets: dict[int, int] = {}
    cursor = 0
    for source in sources:
        offsets[source.source_layer_index] = cursor
        cursor += int(source.cached_parent_post_swiglu_rows.shape[1])
    selected_global = [
        offsets[layer_index] + neuron_index
        for layer_index, neuron_index in result.selected_source_neurons
    ]
    reconstructed = union_rows[:, selected_global] @ result.interpolation_reconstruction
    torch.testing.assert_close(reconstructed, union_rows, atol=1e-12, rtol=1e-12)

    expected_gate = torch.stack(
        [
            source.source_gate_weight[neuron_index].detach()
            @ source.normalized_input_decoder.T
            for layer_index, neuron_index in result.selected_source_neurons
            for source in sources
            if source.source_layer_index == layer_index
        ]
    ).to(torch.bfloat16)
    expected_up = torch.stack(
        [
            source.source_up_weight[neuron_index].detach()
            @ source.normalized_input_decoder.T
            for layer_index, neuron_index in result.selected_source_neurons
            for source in sources
            if source.source_layer_index == layer_index
        ]
    ).to(torch.bfloat16)
    assert torch.equal(result.gate_weight, expected_gate)
    assert torch.equal(result.up_weight, expected_up)

    writer_parts = []
    cursor = 0
    for source in sources:
        width = int(source.source_down_weight.shape[1])
        reconstruction_slice = result.interpolation_reconstruction[
            :,
            cursor : cursor + width,
        ]
        writer_parts.append(
            source.residual_output_chart.T
            @ source.source_down_weight.detach()
            @ reconstruction_slice.T
        )
        cursor += width
    assert all(
        float(torch.linalg.vector_norm(part).item()) > 0.0 for part in writer_parts
    )
    expected_down = torch.stack(writer_parts).sum(dim=0).to(torch.bfloat16)
    assert torch.equal(result.down_weight, expected_down)
    assert set(
        result.selected_source_neurons[index][0]
        for index in range(len(result.selected_source_neurons))
    ) == set(span.source_layer_indices)
    assert all(
        value["complete_source_w2_consumed"]
        and value["writer_contribution_frobenius_norm"] > 0.0
        for value in result.report["writer_transport"]["source_contributions"]
    )


@pytest.mark.parametrize("source_layer_count", (2, 3))
def test_span_coupled_qdeim_exactly_uses_every_layer_and_is_deterministic(
    source_layer_count: int,
) -> None:
    fixture = _fixture(source_layer_count)

    first = _compile(fixture)
    second = _compile(fixture)

    _assert_exact_span_construction(fixture, first)
    assert first.selected_source_neurons == second.selected_source_neurons
    assert torch.equal(first.gate_weight, second.gate_weight)
    assert torch.equal(first.up_weight, second.up_weight)
    assert torch.equal(first.down_weight, second.down_weight)
    assert torch.equal(
        first.interpolation_reconstruction,
        second.interpolation_reconstruction,
    )
    assert first.report["result_sha256"] == second.report["result_sha256"]
    assert first.report["span"] == fixture["span"].as_dict()
    assert first.report["all_source_layers_consumed"]
    assert first.report["all_source_w1_w3_w2_tensors_match_supplied_expected_sha256"]
    assert first.report["source_covering_seed"]["selected_count"] == (
        source_layer_count
    )
    assert first.report["cache_receipt_binding"] == {
        "supplied_receipt_digests_bound_into_functional_evidence": True,
        "upstream_receipt_authentication_required": True,
        "upstream_receipt_authentication_performed_by_this_compiler": False,
    }
    assert first.report["all_source_layers_retain_real_selected_producer"]
    assert first.report["semantic_layer_stack_adapter_seam"] == {
        "span_owner": "Lfm2ModelFoldSpan",
        "ordered_source_payload": "Lfm2MlpQdeimSpanSource",
        "assembly_helper": "assemble_lfm2_mlp_qdeim_span_sources",
        "typed": True,
        "production_wired": False,
        "planner_duplicated": False,
        "parent_recapture_required": False,
        "upstream_cache_receipt_authentication_required": True,
    }


def test_span_adapter_seam_requires_exact_existing_span_coverage() -> None:
    fixture = _fixture(3)
    span = fixture["span"]
    sources_by_layer = {
        source.source_layer_index: source for source in reversed(fixture["sources"])
    }

    ordered = assemble_lfm2_mlp_qdeim_span_sources(
        span=span,
        sources_by_layer=sources_by_layer,
    )

    assert tuple(value.source_layer_index for value in ordered) == (
        span.source_layer_indices
    )
    sources_by_layer.pop(span.source_layer_indices[-1])
    with pytest.raises(Lfm2MlpQdeimError, match="exactly cover"):
        assemble_lfm2_mlp_qdeim_span_sources(
            span=span,
            sources_by_layer=sources_by_layer,
        )


def test_span_qdeim_authentication_fails_closed_on_rows_decoder_and_span() -> None:
    fixture = _fixture(2)
    expected_evidence = lfm2_mlp_qdeim_span_functional_evidence_sha256(
        span=fixture["span"],
        sources=fixture["sources"],
        row_sample_weights=fixture["row_weights"],
    )
    fixture["sources"][0].cached_parent_post_swiglu_rows[0, 0] += 0.5
    with pytest.raises(Lfm2MlpQdeimError, match="functional evidence changed"):
        _compile(
            fixture,
            expected_functional_evidence_sha256=expected_evidence,
        )

    fixture = _fixture(2)
    source = fixture["sources"][1]
    source.normalized_input_decoder[0, 0] += 0.25
    with pytest.raises(Lfm2MlpQdeimError, match="decoder changed"):
        _compile(fixture)

    fixture = _fixture(2)
    with torch.no_grad():
        fixture["sources"][0].source_down_weight[0, 0] += 0.25
    with pytest.raises(Lfm2MlpQdeimError, match="W1/W3/W2 state changed"):
        _compile(fixture)

    fixture = _fixture(2)
    wrong_span = Lfm2ModelFoldSpan((3, 4), target_layer_index=2)
    with pytest.raises(Lfm2MlpQdeimError, match="exactly cover"):
        _compile(fixture, span=wrong_span)


@pytest.mark.parametrize("component", ("source_state", "decoder", "chart"))
def test_external_functional_root_rejects_paired_tensor_hash_substitution(
    component: str,
) -> None:
    fixture = _fixture(2)
    old_root = lfm2_mlp_qdeim_span_functional_evidence_sha256(
        span=fixture["span"],
        sources=fixture["sources"],
        row_sample_weights=fixture["row_weights"],
    )
    source = fixture["sources"][0]
    if component == "source_state":
        with torch.no_grad():
            source.source_gate_weight[0, 0] += 0.5
        source = replace(
            source,
            expected_source_state_sha256=lfm2_mlp_qdeim_source_state_sha256(
                source.source_gate_weight,
                source.source_up_weight,
                source.source_down_weight,
            ),
        )
    elif component == "decoder":
        source.normalized_input_decoder[0, 0] += 0.5
        source = replace(
            source,
            expected_normalized_input_decoder_sha256=(
                lfm2_authenticated_tensor_sha256(source.normalized_input_decoder)
            ),
        )
    else:
        source.residual_output_chart[0, 0] += 0.5
        source = replace(
            source,
            expected_residual_output_chart_sha256=(
                lfm2_authenticated_tensor_sha256(source.residual_output_chart)
            ),
        )
    fixture["sources"] = (source, fixture["sources"][1])

    with pytest.raises(Lfm2MlpQdeimError, match="functional evidence changed"):
        _compile(
            fixture,
            expected_functional_evidence_sha256=old_root,
        )


def test_span_qdeim_outputs_are_dense_ordinary_zero_gradient_child_tensors() -> None:
    fixture = _fixture(3)
    before = tuple(
        lfm2_mlp_qdeim_source_state_sha256(
            source.source_gate_weight,
            source.source_up_weight,
            source.source_down_weight,
        )
        for source in fixture["sources"]
    )

    result = _compile(fixture)

    after = tuple(
        lfm2_mlp_qdeim_source_state_sha256(
            source.source_gate_weight,
            source.source_up_weight,
            source.source_down_weight,
        )
        for source in fixture["sources"]
    )
    assert after == before
    assert all(
        parameter.grad is None
        for source in fixture["sources"]
        for parameter in (
            source.source_gate_weight,
            source.source_up_weight,
            source.source_down_weight,
        )
    )
    state = result.state_dict()
    assert set(state) == {"w1.weight", "w3.weight", "w2.weight"}
    assert all(
        tensor.layout == torch.strided
        and tensor.device.type == "cpu"
        and tensor.dtype == torch.bfloat16
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and tensor.grad_fn is None
        for tensor in state.values()
    )
    assert len({tensor.data_ptr() for tensor in state.values()}) == 3
    assert result.report["dense_ordinary_child_tensors"]
    assert result.report["parameter_sharing_used"] is False
    assert result.report["source_parameters_pruned"] == 0
    assert result.report["parameters_quantized"] == 0
    assert result.report["parameters_sparsified"] == 0
    assert result.report["autograd_grad_calls"] == 0
    assert result.report["backward_calls"] == 0
    assert result.report["optimizer_steps"] == 0
    assert result.report["training_steps"] == 0


def test_zero_source_writer_is_reported_as_a_diagnostic_not_a_false_claim() -> None:
    fixture = _fixture(2)
    first = fixture["sources"][0]
    with torch.no_grad():
        first.source_down_weight.zero_()
    first = replace(
        first,
        expected_source_state_sha256=lfm2_mlp_qdeim_source_state_sha256(
            first.source_gate_weight,
            first.source_up_weight,
            first.source_down_weight,
        ),
    )
    fixture["sources"] = (first, fixture["sources"][1])

    result = _compile(fixture)

    contributions = result.report["writer_transport"]["source_contributions"]
    assert contributions[0]["writer_contribution_frobenius_norm"] == 0.0
    assert (
        "all_source_writer_contributions_nonzero"
        not in result.report["writer_transport"]
    )


def test_span_qdeim_rejects_target_too_narrow_to_retain_every_source() -> None:
    fixture = _fixture(3)
    fixture["gate_template"] = torch.zeros((2, 2), dtype=torch.bfloat16)
    fixture["up_template"] = torch.zeros_like(fixture["gate_template"])
    fixture["down_template"] = torch.zeros((2, 2), dtype=torch.bfloat16)

    with pytest.raises(Lfm2MlpQdeimError, match="at least one intermediate"):
        _compile(fixture)
