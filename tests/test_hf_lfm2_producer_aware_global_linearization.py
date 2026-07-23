from __future__ import annotations

import math
import sys
from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_global_linearization import (  # noqa: E402
    Lfm2AttentionParameters,
    Lfm2MlpParameters,
    Lfm2ShortConvParameters,
    Lfm2StackPrimalBatch,
    Lfm2TerminalNormParameters,
    Lfm2WholeStackPrimalSnapshot,
    capture_lfm2_attention_primal,
    capture_lfm2_macrocell_linearization,
    capture_lfm2_mlp_primal,
    capture_lfm2_short_conv_primal,
    capture_lfm2_terminal_norm_primal,
)
from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualWriterSite,
    Lfm2GlobalResidualWriterTopology,
    fingerprint_lfm2_global_residual_tensor,
    seal_lfm2_global_residual_boundary,
    seal_lfm2_global_residual_linearization_receipt,
)
from neural_forge import hf_lfm2_producer_aware_global_linearization as producer_module  # noqa: E402
from neural_forge.hf_lfm2_producer_aware_global_linearization import (  # noqa: E402
    Lfm2ProducerAwareAuthenticationError,
    Lfm2ProducerAwareLinearizationError,
    Lfm2ProducerAwareWholeStackLinearizationKernel,
    build_lfm2_producer_aware_writer_topology,
    capture_lfm2_producer_aware_layer_seed,
    install_lfm2_producer_aware_dense_transaction,
    lfm2_producer_aware_attention_segment_jvp,
    lfm2_producer_aware_attention_segment_vjp,
    lfm2_producer_aware_mlp_segment_jvp,
    lfm2_producer_aware_mlp_segment_vjp,
    lfm2_producer_aware_short_conv_segment_jvp,
    lfm2_producer_aware_short_conv_segment_vjp,
    lfm2_producer_aware_whole_stack_norm_evidence,
    materialize_lfm2_producer_aware_stack_transaction,
)


DTYPE = torch.float64


def _randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, dtype=DTYPE) * scale


def _inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum()


def _attention_parameters(
    *,
    batch: int,
    sequence: int,
    hidden: int,
) -> Lfm2AttentionParameters:
    angles = _randn(batch, sequence, 1)
    cosine = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sine = torch.cat((angles.sin(), angles.sin()), dim=-1)
    return Lfm2AttentionParameters(
        operator_norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        operator_norm_epsilon=1e-3,
        query_weight=_randn(4, hidden, scale=0.15),
        key_weight=_randn(2, hidden, scale=0.15),
        value_weight=_randn(2, hidden, scale=0.15),
        output_weight=_randn(hidden, 4, scale=0.15),
        query_norm_gain=torch.rand(2, dtype=DTYPE) + 0.5,
        key_norm_gain=torch.rand(2, dtype=DTYPE) + 0.5,
        query_key_norm_epsilon=2e-3,
        rope_cosine=cosine,
        rope_sine=sine,
        causal_mask=torch.tril(
            torch.ones((sequence, sequence), dtype=torch.bool)
        ).reshape(1, 1, sequence, sequence),
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )


def _conv_parameters(
    *,
    batch: int,
    sequence: int,
    hidden: int,
) -> Lfm2ShortConvParameters:
    return Lfm2ShortConvParameters(
        operator_norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        operator_norm_epsilon=1e-3,
        input_weight=_randn(3 * hidden, hidden, scale=0.15),
        stored_fir_kernel=_randn(hidden, 1, 3, scale=0.15),
        output_weight=_randn(hidden, hidden, scale=0.15),
        input_bias=_randn(3 * hidden, scale=0.02),
        fir_bias=_randn(hidden, scale=0.02),
        output_bias=_randn(hidden, scale=0.02),
        padding_mask=torch.ones((batch, sequence), dtype=DTYPE),
    )


def _mlp_parameters(*, hidden: int, intermediate: int) -> Lfm2MlpParameters:
    return Lfm2MlpParameters(
        norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        norm_epsilon=1e-3,
        gate_weight=_randn(intermediate, hidden, scale=0.15),
        up_weight=_randn(intermediate, hidden, scale=0.15),
        down_weight=_randn(hidden, intermediate, scale=0.15),
    )


def test_attention_input_transport_and_output_writer_finite_difference_adjoint() -> (
    None
):
    torch.manual_seed(201)
    batch, sequence, hidden, rank = 2, 4, 6, 3
    residual = _randn(batch, sequence, hidden)
    incoming = _randn(batch, sequence, hidden)
    parameters = _attention_parameters(
        batch=batch,
        sequence=sequence,
        hidden=hidden,
    )
    snapshot = capture_lfm2_attention_primal(residual, parameters)
    basis = _randn(hidden, rank)
    transport = _randn(rank, hidden)
    output = _randn(rank, parameters.writer_input_width)
    cotangent = _randn(batch, sequence, hidden)
    step = 1e-6

    tangent = lfm2_producer_aware_attention_segment_jvp(
        snapshot,
        incoming,
        transport,
        output,
        basis,
    )
    transpose = lfm2_producer_aware_attention_segment_vjp(
        snapshot,
        cotangent,
        basis,
    )
    transport_matrix = basis @ transport

    def replay(signed_step: float):
        perturbed = Lfm2AttentionParameters(
            operator_norm_gain=parameters.operator_norm_gain,
            operator_norm_epsilon=parameters.operator_norm_epsilon,
            query_weight=parameters.query_weight
            + signed_step * (parameters.query_weight @ transport_matrix),
            key_weight=parameters.key_weight
            + signed_step * (parameters.key_weight @ transport_matrix),
            value_weight=parameters.value_weight
            + signed_step * (parameters.value_weight @ transport_matrix),
            output_weight=parameters.output_weight + signed_step * (basis @ output),
            query_norm_gain=parameters.query_norm_gain,
            key_norm_gain=parameters.key_norm_gain,
            query_key_norm_epsilon=parameters.query_key_norm_epsilon,
            rope_cosine=parameters.rope_cosine,
            rope_sine=parameters.rope_sine,
            causal_mask=parameters.causal_mask,
            query_heads=parameters.query_heads,
            key_value_heads=parameters.key_value_heads,
            head_dim=parameters.head_dim,
            scaling=parameters.scaling,
        )
        return capture_lfm2_attention_primal(
            residual + signed_step * incoming,
            perturbed,
        ).output

    finite_difference = (replay(step) - replay(-step)) / (2.0 * step)
    torch.testing.assert_close(tangent.output, finite_difference, rtol=8e-8, atol=8e-9)
    torch.testing.assert_close(
        _inner(tangent.output, cotangent),
        _inner(incoming, transpose.incoming_residual)
        + _inner(transport, transpose.input_transport_coefficients)
        + _inner(output, transpose.output_writer_coefficients),
        rtol=3e-12,
        atol=3e-12,
    )


def test_short_conv_input_transport_and_output_writer_finite_difference_adjoint() -> (
    None
):
    torch.manual_seed(202)
    batch, sequence, hidden, rank = 2, 4, 6, 3
    residual = _randn(batch, sequence, hidden)
    incoming = _randn(batch, sequence, hidden)
    parameters = _conv_parameters(batch=batch, sequence=sequence, hidden=hidden)
    snapshot = capture_lfm2_short_conv_primal(residual, parameters)
    basis = _randn(hidden, rank)
    transport = _randn(rank, hidden)
    output = _randn(rank, hidden)
    cotangent = _randn(batch, sequence, hidden)
    step = 1e-6

    tangent = lfm2_producer_aware_short_conv_segment_jvp(
        snapshot,
        incoming,
        transport,
        output,
        basis,
    )
    transpose = lfm2_producer_aware_short_conv_segment_vjp(
        snapshot,
        cotangent,
        basis,
    )
    transport_matrix = basis @ transport

    def replay(signed_step: float):
        perturbed = Lfm2ShortConvParameters(
            operator_norm_gain=parameters.operator_norm_gain,
            operator_norm_epsilon=parameters.operator_norm_epsilon,
            input_weight=parameters.input_weight
            + signed_step * (parameters.input_weight @ transport_matrix),
            stored_fir_kernel=parameters.stored_fir_kernel,
            output_weight=parameters.output_weight + signed_step * (basis @ output),
            input_bias=parameters.input_bias,
            fir_bias=parameters.fir_bias,
            output_bias=parameters.output_bias,
            padding_mask=parameters.padding_mask,
        )
        return capture_lfm2_short_conv_primal(
            residual + signed_step * incoming,
            perturbed,
        ).output

    finite_difference = (replay(step) - replay(-step)) / (2.0 * step)
    torch.testing.assert_close(tangent.output, finite_difference, rtol=8e-8, atol=8e-9)
    torch.testing.assert_close(
        _inner(tangent.output, cotangent),
        _inner(incoming, transpose.incoming_residual)
        + _inner(transport, transpose.input_transport_coefficients)
        + _inner(output, transpose.output_writer_coefficients),
        rtol=3e-12,
        atol=3e-12,
    )


def test_mlp_input_transport_and_down_writer_finite_difference_adjoint() -> None:
    torch.manual_seed(203)
    batch, sequence, hidden, intermediate, rank = 2, 4, 6, 7, 3
    residual = _randn(batch, sequence, hidden)
    incoming = _randn(batch, sequence, hidden)
    parameters = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    snapshot = capture_lfm2_mlp_primal(residual, parameters)
    basis = _randn(hidden, rank)
    transport = _randn(rank, hidden)
    output = _randn(rank, intermediate)
    cotangent = _randn(batch, sequence, hidden)
    step = 1e-6

    tangent = lfm2_producer_aware_mlp_segment_jvp(
        snapshot,
        incoming,
        transport,
        output,
        basis,
    )
    transpose = lfm2_producer_aware_mlp_segment_vjp(snapshot, cotangent, basis)
    transport_matrix = basis @ transport

    def replay(signed_step: float):
        perturbed = Lfm2MlpParameters(
            norm_gain=parameters.norm_gain,
            norm_epsilon=parameters.norm_epsilon,
            gate_weight=parameters.gate_weight
            + signed_step * (parameters.gate_weight @ transport_matrix),
            up_weight=parameters.up_weight
            + signed_step * (parameters.up_weight @ transport_matrix),
            down_weight=parameters.down_weight + signed_step * (basis @ output),
        )
        return capture_lfm2_mlp_primal(
            residual + signed_step * incoming,
            perturbed,
        ).output

    finite_difference = (replay(step) - replay(-step)) / (2.0 * step)
    torch.testing.assert_close(tangent.output, finite_difference, rtol=8e-8, atol=8e-9)
    torch.testing.assert_close(
        _inner(tangent.output, cotangent),
        _inner(incoming, transpose.incoming_residual)
        + _inner(transport, transpose.input_transport_coefficients)
        + _inner(output, transpose.output_writer_coefficients),
        rtol=3e-12,
        atol=3e-12,
    )


def _build_v2_stack_fixture(layer_count: int = 14):
    torch.manual_seed(204)
    sequence, hidden, intermediate, rank = 3, 4, 5, 4
    basis = _randn(hidden, rank, scale=0.12)
    residual = _randn(1, sequence, hidden, scale=0.12)
    initial_residual = residual.clone()
    macrocells = []
    cast_offsets = []
    for layer_index in range(layer_count):
        operator = (
            _attention_parameters(batch=1, sequence=sequence, hidden=hidden)
            if layer_index % 2 == 0
            else _conv_parameters(batch=1, sequence=sequence, hidden=hidden)
        )
        mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
        macrocell = capture_lfm2_macrocell_linearization(residual, operator, mlp)
        macrocells.append(macrocell)
        rounded = macrocell.mlp.output.to(torch.bfloat16).to(DTYPE)
        cast_offsets.append(rounded - macrocell.mlp.output)
        residual = rounded
    terminal_parameters = Lfm2TerminalNormParameters(
        gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        epsilon=1e-3,
    )
    terminal = capture_lfm2_terminal_norm_primal(residual, terminal_parameters)
    batch = Lfm2StackPrimalBatch(
        macrocells=tuple(macrocells),
        selected_target_positions=torch.tensor([0, 2]),
        canonical_boundary_row_indices=torch.tensor([0, 1]),
        terminal=terminal,
    )
    snapshot = Lfm2WholeStackPrimalSnapshot((batch,))
    coefficients = []
    sites = []
    for layer_index, macrocell in enumerate(macrocells):
        operator_transport_kind = (
            "attention_input_transport"
            if macrocell.operator_kind == "full_attention"
            else "short_conv_input_transport"
        )
        operator_output_kind = (
            "attention_output"
            if macrocell.operator_kind == "full_attention"
            else "short_conv_output"
        )
        widths = (hidden, macrocell.input_widths[0], hidden, intermediate)
        kinds = (
            operator_transport_kind,
            operator_output_kind,
            "mlp_input_transport",
            "mlp_down",
        )
        causal_orders = (
            6 * layer_index,
            6 * layer_index + 1,
            6 * layer_index + 3,
            6 * layer_index + 4,
        )
        for local_index, (width, kind, causal_order) in enumerate(
            zip(widths, kinds, causal_orders, strict=True)
        ):
            coefficients.append(_randn(rank, width, scale=0.045))
            sites.append(
                Lfm2GlobalResidualWriterSite(
                    site_id=f"layer.{layer_index}.site.{local_index}.{kind}",
                    layer_index=layer_index,
                    causal_order=causal_order,
                    writer_kind=kind,
                    input_width=width,
                )
            )
    topology = Lfm2GlobalResidualWriterTopology(
        hidden_size=hidden,
        head_dimension=2,
        sites=tuple(sites),
    )
    derived_topology = build_lfm2_producer_aware_writer_topology(
        snapshot,
        head_dimension=2,
        site_ids=tuple(site.site_id for site in sites),
    )
    assert derived_topology.topology_sha256 == topology.topology_sha256
    full_rank_topology = build_lfm2_producer_aware_writer_topology(
        snapshot,
        head_dimension=2,
        site_ids=tuple(site.site_id for site in sites),
        requested_basis_rank=hidden,
    )
    assert full_rank_topology.requested_basis_rank == hidden
    assert full_rank_topology.effective_basis_rank == hidden
    assert full_rank_topology.topology_basis_rank_formula == (
        "explicit_requested_basis_rank"
    )
    assert full_rank_topology.topology_sha256 != topology.topology_sha256
    boundaries = []
    for layer_index in range(layer_count):
        for boundary_offset, kind in enumerate(
            ("post_operator_residual", "post_layer_residual")
        ):
            defect = torch.zeros((2, hidden), dtype=DTYPE)
            mass = torch.ones(2, dtype=DTYPE)
            boundaries.append(
                seal_lfm2_global_residual_boundary(
                    boundary_id=f"layer.{layer_index}.{kind}",
                    layer_index=layer_index,
                    causal_order=6 * layer_index + 2 + 3 * boundary_offset,
                    boundary_kind=kind,
                    defect=defect,
                    defect_sha256=fingerprint_lfm2_global_residual_tensor(defect),
                    row_mass=mass,
                    row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass),
                )
            )
    terminal_defect = torch.zeros((2, hidden), dtype=DTYPE)
    terminal_mass = torch.ones(2, dtype=DTYPE)
    boundaries.append(
        seal_lfm2_global_residual_boundary(
            boundary_id="terminal.final_norm",
            layer_index=layer_count - 1,
            causal_order=6 * layer_count,
            boundary_kind="terminal_observable",
            defect=terminal_defect,
            defect_sha256=fingerprint_lfm2_global_residual_tensor(terminal_defect),
            row_mass=terminal_mass,
            row_mass_sha256=fingerprint_lfm2_global_residual_tensor(terminal_mass),
        )
    )
    boundary_bundle = Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))
    conditioning, certified = lfm2_producer_aware_whole_stack_norm_evidence(
        snapshot,
        basis,
    )
    maximum_coordinate_weight = max(
        float(boundary.row_mass.max().item()) for boundary in boundary_bundle.boundaries
    ) / (len(boundary_bundle.boundaries) * hidden)
    receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        boundary_bundle,
        kernel_id="toy.lfm2.producer_aware.v2a",
        source_sha256={"stack_primal": snapshot.authentication_sha256},
        analytic_math_components=(
            "coupled_normalized_input_transport",
            "attention_qkv_absorption",
            "short_conv_bcx_absorption",
            "swiglu_gate_up_absorption",
            "residual_output_writers",
            "manual_exact_transpose",
        ),
        operator_squared_norm_upper_bound=(certified.operator_squared_norm_upper_bound),
        block_squared_norm_upper_bounds=(certified.block_squared_norm_upper_bounds),
        operator_squared_conditioning_scale=(
            conditioning.operator_squared_scale * maximum_coordinate_weight
        ),
        block_squared_conditioning_scales=tuple(
            value * maximum_coordinate_weight
            for value in conditioning.block_squared_scales
        ),
        conditioning_scale_method=conditioning.method,
    )
    kernel = Lfm2ProducerAwareWholeStackLinearizationKernel(
        topology,
        boundary_bundle,
        snapshot,
        linearization_receipt=receipt,
    )
    return {
        "basis": basis,
        "cast_offsets": tuple(cast_offsets),
        "coefficients": tuple(coefficients),
        "initial_residual": initial_residual,
        "kernel": kernel,
        "macrocells": tuple(macrocells),
        "snapshot": snapshot,
        "terminal_parameters": terminal_parameters,
    }


def _replay_perturbed_v2_stack(fixture, signed_step: float):
    residual = fixture["initial_residual"]
    boundaries = []
    basis = fixture["basis"]
    coefficients = fixture["coefficients"]
    for layer_index, baseline in enumerate(fixture["macrocells"]):
        site_offset = 4 * layer_index
        operator_transport = basis @ coefficients[site_offset]
        mlp_transport = basis @ coefficients[site_offset + 2]
        p = baseline.operator.parameters
        if baseline.operator_kind == "full_attention":
            operator = Lfm2AttentionParameters(
                operator_norm_gain=p.operator_norm_gain,
                operator_norm_epsilon=p.operator_norm_epsilon,
                query_weight=p.query_weight
                + signed_step * (p.query_weight @ operator_transport),
                key_weight=p.key_weight
                + signed_step * (p.key_weight @ operator_transport),
                value_weight=p.value_weight
                + signed_step * (p.value_weight @ operator_transport),
                output_weight=p.output_weight
                + signed_step * (basis @ coefficients[site_offset + 1]),
                query_norm_gain=p.query_norm_gain,
                key_norm_gain=p.key_norm_gain,
                query_key_norm_epsilon=p.query_key_norm_epsilon,
                rope_cosine=p.rope_cosine,
                rope_sine=p.rope_sine,
                causal_mask=p.causal_mask,
                query_heads=p.query_heads,
                key_value_heads=p.key_value_heads,
                head_dim=p.head_dim,
                scaling=p.scaling,
            )
        else:
            operator = Lfm2ShortConvParameters(
                operator_norm_gain=p.operator_norm_gain,
                operator_norm_epsilon=p.operator_norm_epsilon,
                input_weight=p.input_weight
                + signed_step * (p.input_weight @ operator_transport),
                stored_fir_kernel=p.stored_fir_kernel,
                output_weight=p.output_weight
                + signed_step * (basis @ coefficients[site_offset + 1]),
                input_bias=p.input_bias,
                fir_bias=p.fir_bias,
                output_bias=p.output_bias,
                padding_mask=p.padding_mask,
            )
        mp = baseline.mlp.parameters
        mlp = Lfm2MlpParameters(
            norm_gain=mp.norm_gain,
            norm_epsilon=mp.norm_epsilon,
            gate_weight=mp.gate_weight + signed_step * (mp.gate_weight @ mlp_transport),
            up_weight=mp.up_weight + signed_step * (mp.up_weight @ mlp_transport),
            down_weight=mp.down_weight
            + signed_step * (basis @ coefficients[site_offset + 3]),
        )
        macrocell = capture_lfm2_macrocell_linearization(residual, operator, mlp)
        boundaries.extend((macrocell.operator.output, macrocell.mlp.output))
        residual = macrocell.mlp.output + fixture["cast_offsets"][layer_index]
    terminal = capture_lfm2_terminal_norm_primal(
        residual,
        fixture["terminal_parameters"],
    )
    boundaries.append(terminal.output)
    selected = torch.tensor([0, 2])
    return tuple(
        value.reshape(-1, int(value.shape[-1])).index_select(0, selected)
        for value in boundaries
    )


def test_producer_aware_whole_stack_finite_difference_adjoint_and_causality() -> None:
    fixture = _build_v2_stack_fixture()
    kernel = fixture["kernel"]
    coefficients = fixture["coefficients"]
    basis = fixture["basis"]
    step = 1e-6
    effects = kernel.run_jvp_bundle(
        coefficients,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    plus = _replay_perturbed_v2_stack(fixture, step)
    minus = _replay_perturbed_v2_stack(fixture, -step)
    for observed, plus_value, minus_value in zip(
        effects,
        plus,
        minus,
        strict=True,
    ):
        torch.testing.assert_close(
            observed,
            (plus_value - minus_value) / (2.0 * step),
            rtol=3e-6,
            atol=3e-8,
        )
    conditioning, certified = lfm2_producer_aware_whole_stack_norm_evidence(
        fixture["snapshot"],
        basis,
    )
    coefficient_energy = sum(
        float(value.square().sum().item()) for value in coefficients
    )
    effect_energy = sum(float(value.square().sum().item()) for value in effects)
    assert effect_energy <= (
        certified.operator_squared_norm_upper_bound * coefficient_energy
    )
    assert conditioning.operator_squared_scale == pytest.approx(
        sum(conditioning.block_squared_scales)
    )
    cotangents = tuple(_randn(*value.shape) for value in effects)
    transpose = kernel.run_vjp_bundle(
        cotangents,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    torch.testing.assert_close(
        sum(
            _inner(effect, cotangent)
            for effect, cotangent in zip(effects, cotangents, strict=True)
        ),
        sum(
            _inner(coefficient, coefficient_bar)
            for coefficient, coefficient_bar in zip(
                coefficients,
                transpose,
                strict=True,
            )
        ),
        rtol=3e-11,
        atol=3e-11,
    )
    causal_coefficients = tuple(torch.zeros_like(value) for value in coefficients)
    active_site = 4 * 8 + 2
    causal_coefficients = tuple(
        _randn(*value.shape, scale=0.05) if index == active_site else value
        for index, value in enumerate(causal_coefficients)
    )
    causal_effects = kernel.run_jvp_bundle(
        causal_coefficients,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    assert all(
        int(torch.count_nonzero(value).item()) == 0 for value in causal_effects[:17]
    )
    assert float(torch.linalg.vector_norm(causal_effects[17]).item()) > 0.0
    assert kernel.supports_bundle_linearization
    assert kernel.verify_authentication(force_raw=True)


def test_producer_aware_norm_evidence_saturates_huge_finite_products(
    monkeypatch,
) -> None:
    fixture = _build_v2_stack_fixture(layer_count=2)
    huge = 1e200
    monkeypatch.setattr(
        producer_module,
        "_macrocell_norm_factors",
        lambda _macrocell, _basis: (huge, huge, huge, huge),
    )
    monkeypatch.setattr(
        producer_module,
        "_producer_input_direct_norm_factors",
        lambda _macrocell, _basis: (huge, huge),
    )
    monkeypatch.setattr(
        producer_module,
        "_terminal_transfer_norm_upper",
        lambda _terminal: huge,
    )

    conditioning, certified = lfm2_producer_aware_whole_stack_norm_evidence(
        fixture["snapshot"],
        fixture["basis"],
    )

    maximum = producer_module._MAX_FINITE_NORM_EVIDENCE
    assert certified.finite_float_overflow_observed is True
    assert certified.operator_log_squared_norm_evidence is not None
    assert (
        certified.operator_log_squared_norm_evidence
        > producer_module._LOG_MAX_FINITE_NORM_EVIDENCE
    )
    assert all(
        value is not None
        and value > producer_module._LOG_MAX_FINITE_NORM_EVIDENCE
        for value in certified.block_log_squared_norm_evidence
    )
    assert conditioning.operator_squared_scale == maximum
    assert certified.operator_squared_norm_upper_bound == maximum
    assert conditioning.block_squared_scales == (maximum,) * 8
    assert certified.block_squared_norm_upper_bounds == (maximum,) * 8
    assert all(
        math.isfinite(value)
        for value in (
            conditioning.operator_squared_scale,
            *conditioning.block_squared_scales,
            certified.operator_squared_norm_upper_bound,
            *certified.block_squared_norm_upper_bounds,
        )
    )


def test_producer_aware_norm_evidence_tracks_exact_log_boundary_saturation(
    monkeypatch,
) -> None:
    fixture = _build_v2_stack_fixture(layer_count=2)
    first, second = fixture["snapshot"].batches[0].macrocells
    just_over_square_limit = math.sqrt(sys.float_info.max) * math.nextafter(
        1.0,
        math.inf,
    )

    def boundary_factors(macrocell, _basis):
        if macrocell is first:
            return (1.0, 1.0, 0.0, 1.0)
        assert macrocell is second
        return (just_over_square_limit, 0.0, 0.0, 0.0)

    monkeypatch.setattr(
        producer_module,
        "_macrocell_norm_factors",
        boundary_factors,
    )
    monkeypatch.setattr(
        producer_module,
        "_producer_input_direct_norm_factors",
        lambda _macrocell, _basis: (0.0, 0.0),
    )
    monkeypatch.setattr(
        producer_module,
        "_terminal_transfer_norm_upper",
        lambda _terminal: 0.0,
    )

    conditioning, certified = lfm2_producer_aware_whole_stack_norm_evidence(
        fixture["snapshot"],
        fixture["basis"],
    )

    assert certified.block_log_squared_norm_evidence[3] == math.log(
        sys.float_info.max
    )
    assert certified.operator_log_squared_norm_evidence == math.log(
        sys.float_info.max
    )
    assert certified.block_squared_norm_upper_bounds[3] == sys.float_info.max
    assert certified.operator_squared_norm_upper_bound == sys.float_info.max
    assert certified.finite_float_overflow_observed is True
    assert conditioning.operator_squared_scale == 1.0


@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_producer_aware_norm_evidence_still_rejects_nonfinite_factors(
    monkeypatch,
    nonfinite: float,
) -> None:
    fixture = _build_v2_stack_fixture(layer_count=1)
    monkeypatch.setattr(
        producer_module,
        "_macrocell_norm_factors",
        lambda _macrocell, _basis: (1.0, 1.0, 1.0, 1.0),
    )
    monkeypatch.setattr(
        producer_module,
        "_producer_input_direct_norm_factors",
        lambda _macrocell, _basis: (nonfinite, 1.0),
    )

    with pytest.raises(
        Lfm2ProducerAwareLinearizationError,
        match="non-finite or negative factor",
    ):
        lfm2_producer_aware_whole_stack_norm_evidence(
            fixture["snapshot"],
            fixture["basis"],
        )


def test_every_producer_aware_site_is_block_lower_triangular() -> None:
    fixture = _build_v2_stack_fixture()
    kernel = fixture["kernel"]
    zeros = tuple(torch.zeros_like(value) for value in fixture["coefficients"])
    for site_index, site in enumerate(kernel.topology.sites):
        coefficients = tuple(
            torch.ones_like(value) if index == site_index else value
            for index, value in enumerate(zeros)
        )
        effects = kernel.run_jvp_bundle(
            coefficients,
            fixture["basis"],
            maximum_workspace_bytes=64 * 1024 * 1024,
        )
        for boundary, effect in zip(
            kernel.boundaries.boundaries,
            effects,
            strict=True,
        ):
            if boundary.causal_order < site.causal_order:
                assert int(torch.count_nonzero(effect).item()) == 0


@pytest.mark.parametrize("layer_count", (2, 5))
def test_producer_aware_stack_geometry_derives_from_layer_count(
    layer_count: int,
) -> None:
    fixture = _build_v2_stack_fixture(layer_count)
    kernel = fixture["kernel"]
    assert fixture["snapshot"].layer_count == layer_count
    assert len(kernel.topology.sites) == 4 * layer_count
    assert len(kernel.boundaries.boundaries) == 2 * layer_count + 1
    assert kernel.boundaries.boundaries[-1].layer_index == layer_count - 1
    effects = kernel.run_jvp_bundle(
        fixture["coefficients"],
        fixture["basis"],
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    assert len(effects) == 2 * layer_count + 1
    transaction = materialize_lfm2_producer_aware_stack_transaction(
        _v2_layer_seed_evidence(fixture),
        fixture["basis"],
        fixture["coefficients"],
    )
    assert transaction.verify_authentication()
    assert max(transaction.layer_indices) == layer_count - 1


def _v2_layer_seed_evidence(fixture):
    layers = []
    for layer_index, macrocell in enumerate(fixture["macrocells"]):
        roles = (
            (
                "query_weight",
                "key_weight",
                "value_weight",
                "operator_output_weight",
                "mlp_gate_weight",
                "mlp_up_weight",
                "mlp_down_weight",
            )
            if macrocell.operator_kind == "full_attention"
            else (
                "operator_input_weight",
                "operator_output_weight",
                "mlp_gate_weight",
                "mlp_up_weight",
                "mlp_down_weight",
            )
        )
        layers.append(
            capture_lfm2_producer_aware_layer_seed(
                layer_index,
                macrocell,
                parameter_ids={role: f"layer.{layer_index}.{role}" for role in roles},
            )
        )
    return tuple(layers)


def test_dense_absorption_is_coupled_independent_and_atomic() -> None:
    fixture = _build_v2_stack_fixture()
    layers = _v2_layer_seed_evidence(fixture)
    transaction = materialize_lfm2_producer_aware_stack_transaction(
        layers,
        fixture["basis"],
        fixture["coefficients"],
    )
    assert transaction.verify_authentication()
    corrected = dict(
        zip(transaction.parameter_ids, transaction.corrected_weights, strict=True)
    )
    attention = layers[0]
    attention_transport = fixture["basis"] @ fixture["coefficients"][0]
    for role in ("query_weight", "key_weight", "value_weight"):
        torch.testing.assert_close(
            corrected[attention.parameter_ids[role]],
            attention.seed_weights[role]
            + attention.seed_weights[role] @ attention_transport,
        )
    conv = layers[1]
    conv_transport = fixture["basis"] @ fixture["coefficients"][4]
    torch.testing.assert_close(
        corrected[conv.parameter_ids["operator_input_weight"]],
        conv.seed_weights["operator_input_weight"]
        + conv.seed_weights["operator_input_weight"] @ conv_transport,
    )
    mlp_transport = fixture["basis"] @ fixture["coefficients"][2]
    for role in ("mlp_gate_weight", "mlp_up_weight"):
        torch.testing.assert_close(
            corrected[attention.parameter_ids[role]],
            attention.seed_weights[role] + attention.seed_weights[role] @ mlp_transport,
        )
    assert len(
        {
            int(value.untyped_storage().data_ptr())
            for value in transaction.corrected_weights
        }
    ) == len(transaction.corrected_weights)
    live = {
        parameter_id: torch.nn.Parameter(seed.clone(), requires_grad=False)
        for parameter_id, seed in zip(
            transaction.parameter_ids,
            transaction.seed_weights,
            strict=True,
        )
    }
    receipt = install_lfm2_producer_aware_dense_transaction(transaction, live)
    assert receipt.verify_authentication()
    for parameter_id, expected in corrected.items():
        torch.testing.assert_close(live[parameter_id], expected)


def test_dense_absorption_cpu_execution_receipt_and_outputs_are_authenticated() -> None:
    fixture = _build_v2_stack_fixture()
    transaction = materialize_lfm2_producer_aware_stack_transaction(
        _v2_layer_seed_evidence(fixture),
        fixture["basis"],
        fixture["coefficients"],
        compute_device="cpu",
    )

    receipt = transaction.report["compute_execution_receipt"]
    assert transaction.verify_authentication()
    assert receipt["requested_device"] == "cpu"
    assert receipt["resolved_device"] == "cpu"
    assert receipt["backend"] == "cpu"
    assert receipt["compute_dtype"] == "float64"
    assert receipt["cuda_identity"] is None
    assert receipt["silent_device_fallback_performed"] is False
    assert receipt["archive_devices"] == {
        "layer_seed_weights": "cpu",
        "output_basis": "cpu",
        "coefficient_blocks": "cpu",
    }
    assert receipt["output_devices"] == {
        "transaction_seed_weights": "cpu",
        "dense_corrected_weights": "cpu",
    }
    assert (
        transaction.report["compute_execution_receipt_sha256"]
        == receipt["report_sha256"]
    )
    assert all(value.device.type == "cpu" for value in transaction.seed_weights)
    assert all(value.device.type == "cpu" for value in transaction.corrected_weights)
    for counter in (
        "autograd_grad_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
    ):
        assert receipt[counter] == 0


@pytest.mark.parametrize(
    "compute_device",
    (None, "", "CPU", "cuda", "cuda:", "cuda:00", "cuda:-1", "cuda:\uff10", "mps:0"),
)
def test_dense_absorption_rejects_noncanonical_compute_device(compute_device) -> None:
    fixture = _build_v2_stack_fixture()
    with pytest.raises(Lfm2ProducerAwareLinearizationError, match="compute_device"):
        materialize_lfm2_producer_aware_stack_transaction(
            _v2_layer_seed_evidence(fixture),
            fixture["basis"],
            fixture["coefficients"],
            compute_device=compute_device,
        )


def test_dense_absorption_rejects_unavailable_cuda_without_fallback() -> None:
    fixture = _build_v2_stack_fixture()
    unavailable_index = int(torch.cuda.device_count())
    with pytest.raises(Lfm2ProducerAwareLinearizationError, match="unavailable"):
        materialize_lfm2_producer_aware_stack_transaction(
            _v2_layer_seed_evidence(fixture),
            fixture["basis"],
            fixture["coefficients"],
            compute_device=f"cuda:{unavailable_index}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dense_absorption_cuda_matches_cpu_and_returns_cpu_tensors() -> None:
    fixture = _build_v2_stack_fixture()
    layers = _v2_layer_seed_evidence(fixture)
    cpu_transaction = materialize_lfm2_producer_aware_stack_transaction(
        layers,
        fixture["basis"],
        fixture["coefficients"],
        compute_device="cpu",
    )
    cuda_transaction = materialize_lfm2_producer_aware_stack_transaction(
        layers,
        fixture["basis"],
        fixture["coefficients"],
        compute_device="cuda:0",
    )

    assert cuda_transaction.verify_authentication()
    receipt = cuda_transaction.report["compute_execution_receipt"]
    assert receipt["requested_device"] == "cuda:0"
    assert receipt["resolved_device"] == "cuda:0"
    assert receipt["backend"] == "cuda"
    assert receipt["gpu_is_ephemeral_dense_algebra_workspace"] is True
    assert receipt["cuda_identity"]["device_index"] == 0
    assert receipt["cuda_identity"]["device_name"]
    assert all(value.device.type == "cpu" for value in cuda_transaction.seed_weights)
    assert all(
        value.device.type == "cpu" for value in cuda_transaction.corrected_weights
    )
    for cpu_value, cuda_value in zip(
        cpu_transaction.corrected_weights,
        cuda_transaction.corrected_weights,
        strict=True,
    ):
        torch.testing.assert_close(cpu_value, cuda_value, rtol=1e-10, atol=1e-12)


def test_dense_absorption_rejects_resealed_inconsistent_compute_receipt() -> None:
    fixture = _build_v2_stack_fixture()
    transaction = materialize_lfm2_producer_aware_stack_transaction(
        _v2_layer_seed_evidence(fixture),
        fixture["basis"],
        fixture["coefficients"],
    )
    receipt_body = dict(transaction.report["compute_execution_receipt"])
    receipt_body.pop("report_sha256")
    receipt_body["requested_device"] = "cuda:0"
    tampered_receipt = producer_module._seal_report(receipt_body)
    transaction_body = dict(transaction.report)
    transaction_body.pop("report_sha256")
    transaction_body["compute_execution_receipt"] = dict(tampered_receipt)
    transaction_body["compute_execution_receipt_sha256"] = tampered_receipt[
        "report_sha256"
    ]

    with pytest.raises(
        Lfm2ProducerAwareAuthenticationError,
        match="does not authenticate",
    ):
        replace(
            transaction,
            report=producer_module._seal_report(transaction_body),
        )


def test_dense_absorption_transaction_rolls_back_partial_failure(monkeypatch) -> None:
    fixture = _build_v2_stack_fixture()
    transaction = materialize_lfm2_producer_aware_stack_transaction(
        _v2_layer_seed_evidence(fixture),
        fixture["basis"],
        fixture["coefficients"],
    )
    live = {
        parameter_id: torch.nn.Parameter(seed.clone(), requires_grad=False)
        for parameter_id, seed in zip(
            transaction.parameter_ids,
            transaction.seed_weights,
            strict=True,
        )
    }
    before = {name: value.detach().clone() for name, value in live.items()}
    original_copy = producer_module._copy_live_parameter
    calls = 0

    def fail_second_copy(parameter, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected publication failure")
        original_copy(parameter, value)

    monkeypatch.setattr(producer_module, "_copy_live_parameter", fail_second_copy)
    with pytest.raises(
        Lfm2ProducerAwareLinearizationError,
        match="every changed parameter was restored",
    ):
        install_lfm2_producer_aware_dense_transaction(transaction, live)
    for parameter_id, expected in before.items():
        torch.testing.assert_close(live[parameter_id], expected)
