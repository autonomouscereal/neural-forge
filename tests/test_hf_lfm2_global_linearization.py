from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_global_linearization import (  # noqa: E402
    Lfm2AttentionParameters,
    Lfm2GlobalLinearizationError,
    Lfm2MlpParameters,
    Lfm2MacrocellLinearizationKernel,
    Lfm2ShortConvParameters,
    Lfm2StackPrimalBatch,
    Lfm2TerminalNormParameters,
    Lfm2WholeStackLinearizationKernel,
    Lfm2WholeStackPrimalSnapshot,
    capture_lfm2_macrocell_linearization,
    capture_lfm2_mlp_primal,
    capture_lfm2_terminal_norm_primal,
    lfm2_continuous_fp64_rmsnorm,
    lfm2_continuous_fp64_rmsnorm_jvp,
    lfm2_continuous_fp64_rmsnorm_vjp,
    lfm2_mlp_segment_jvp,
    lfm2_mlp_segment_vjp,
)
from neural_forge.hf_lfm2_global_residual_closure import (  # noqa: E402
    Lfm2GlobalResidualBoundaryBundle,
    Lfm2GlobalResidualClosureError,
    Lfm2GlobalResidualWriterSite,
    Lfm2GlobalResidualWriterTopology,
    fingerprint_lfm2_global_residual_tensor,
    seal_lfm2_global_residual_boundary,
    seal_lfm2_global_residual_linearization_receipt,
)


DTYPE = torch.float64


def _randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, dtype=DTYPE) * scale


def _attention_parameters(
    *,
    batch: int = 2,
    sequence: int = 4,
    hidden: int = 6,
    query_heads: int = 3,
    key_value_heads: int = 1,
    head_dim: int = 2,
) -> Lfm2AttentionParameters:
    angles = _randn(batch, sequence, head_dim // 2)
    cosine_half = angles.cos()
    sine_half = angles.sin()
    cosine = torch.cat((cosine_half, cosine_half), dim=-1)
    sine = torch.cat((sine_half, sine_half), dim=-1)
    mask = torch.tril(torch.ones((sequence, sequence), dtype=torch.bool)).reshape(
        1, 1, sequence, sequence
    )
    return Lfm2AttentionParameters(
        operator_norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        operator_norm_epsilon=1e-3,
        query_weight=_randn(query_heads * head_dim, hidden, scale=0.2),
        key_weight=_randn(key_value_heads * head_dim, hidden, scale=0.2),
        value_weight=_randn(key_value_heads * head_dim, hidden, scale=0.2),
        output_weight=_randn(hidden, query_heads * head_dim, scale=0.2),
        query_norm_gain=torch.rand(head_dim, dtype=DTYPE) + 0.5,
        key_norm_gain=torch.rand(head_dim, dtype=DTYPE) + 0.5,
        query_key_norm_epsilon=2e-3,
        rope_cosine=cosine,
        rope_sine=sine,
        causal_mask=mask,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )


def _conv_parameters(
    *,
    batch: int = 2,
    sequence: int = 4,
    hidden: int = 6,
    fir_length: int = 3,
) -> Lfm2ShortConvParameters:
    padding = torch.ones((batch, sequence), dtype=DTYPE)
    padding[1, -1] = 0.0
    return Lfm2ShortConvParameters(
        operator_norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        operator_norm_epsilon=1e-3,
        input_weight=_randn(3 * hidden, hidden, scale=0.2),
        stored_fir_kernel=_randn(hidden, 1, fir_length, scale=0.2),
        output_weight=_randn(hidden, hidden, scale=0.2),
        input_bias=_randn(3 * hidden, scale=0.05),
        fir_bias=_randn(hidden, scale=0.05),
        output_bias=_randn(hidden, scale=0.05),
        padding_mask=padding,
    )


def _mlp_parameters(
    *,
    hidden: int = 6,
    intermediate: int = 7,
) -> Lfm2MlpParameters:
    return Lfm2MlpParameters(
        norm_gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        norm_epsilon=1e-3,
        gate_weight=_randn(intermediate, hidden, scale=0.2),
        up_weight=_randn(intermediate, hidden, scale=0.2),
        down_weight=_randn(hidden, intermediate, scale=0.2),
    )


def _inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum()


def _assert_detached_fp64(*values: torch.Tensor) -> None:
    for value in values:
        assert value.dtype == DTYPE
        assert not value.requires_grad
        assert value.grad_fn is None
        assert torch.isfinite(value).all()


def test_continuous_fp64_rmsnorm_direction_and_adjoint() -> None:
    torch.manual_seed(10)
    rows = _randn(2, 3, 5)
    direction = _randn(2, 3, 5)
    cotangent = _randn(2, 3, 5)
    gain = torch.rand(5, dtype=DTYPE) + 0.25
    epsilon = 3e-3
    step = 1e-6

    tangent = lfm2_continuous_fp64_rmsnorm_jvp(
        rows,
        direction,
        gain,
        epsilon,
    )
    finite_difference = (
        lfm2_continuous_fp64_rmsnorm(
            rows + step * direction,
            gain,
            epsilon,
        )
        - lfm2_continuous_fp64_rmsnorm(
            rows - step * direction,
            gain,
            epsilon,
        )
    ) / (2.0 * step)
    transpose = lfm2_continuous_fp64_rmsnorm_vjp(
        rows,
        cotangent,
        gain,
        epsilon,
    )

    torch.testing.assert_close(tangent, finite_difference, rtol=2e-8, atol=2e-9)
    torch.testing.assert_close(
        _inner(tangent, cotangent),
        _inner(direction, transpose),
        rtol=2e-12,
        atol=2e-12,
    )


def test_mlp_segment_joint_writer_and_residual_linearization() -> None:
    torch.manual_seed(11)
    batch, sequence, hidden, intermediate, rank = 2, 3, 5, 7, 3
    residual = _randn(batch, sequence, hidden)
    parameters = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    snapshot = capture_lfm2_mlp_primal(residual, parameters)
    incoming = _randn(batch, sequence, hidden)
    basis = _randn(hidden, rank)
    coefficients = _randn(rank, intermediate)
    cotangent = _randn(batch, sequence, hidden)
    step = 1e-6

    tangent = lfm2_mlp_segment_jvp(
        snapshot,
        incoming,
        coefficients,
        basis,
    )
    transpose = lfm2_mlp_segment_vjp(snapshot, cotangent, basis)
    plus_parameters = Lfm2MlpParameters(
        norm_gain=parameters.norm_gain,
        norm_epsilon=parameters.norm_epsilon,
        gate_weight=parameters.gate_weight,
        up_weight=parameters.up_weight,
        down_weight=parameters.down_weight + step * (basis @ coefficients),
    )
    minus_parameters = Lfm2MlpParameters(
        norm_gain=parameters.norm_gain,
        norm_epsilon=parameters.norm_epsilon,
        gate_weight=parameters.gate_weight,
        up_weight=parameters.up_weight,
        down_weight=parameters.down_weight - step * (basis @ coefficients),
    )
    plus = capture_lfm2_mlp_primal(
        residual + step * incoming,
        plus_parameters,
    )
    minus = capture_lfm2_mlp_primal(
        residual - step * incoming,
        minus_parameters,
    )
    finite_difference = (plus.output - minus.output) / (2.0 * step)

    torch.testing.assert_close(
        tangent.output,
        finite_difference,
        rtol=3e-8,
        atol=3e-9,
    )
    torch.testing.assert_close(
        _inner(tangent.output, cotangent),
        _inner(incoming, transpose.incoming_residual)
        + _inner(coefficients, transpose.writer_coefficients),
        rtol=2e-12,
        atol=2e-12,
    )


def _perturbed_attention_macrocell(
    *,
    residual: torch.Tensor,
    incoming: torch.Tensor,
    attention: Lfm2AttentionParameters,
    mlp: Lfm2MlpParameters,
    basis: torch.Tensor,
    coefficient_blocks: tuple[torch.Tensor, torch.Tensor],
    signed_step: float,
):
    perturbed_attention = Lfm2AttentionParameters(
        operator_norm_gain=attention.operator_norm_gain,
        operator_norm_epsilon=attention.operator_norm_epsilon,
        query_weight=attention.query_weight,
        key_weight=attention.key_weight,
        value_weight=attention.value_weight,
        output_weight=(
            attention.output_weight + signed_step * (basis @ coefficient_blocks[0])
        ),
        query_norm_gain=attention.query_norm_gain,
        key_norm_gain=attention.key_norm_gain,
        query_key_norm_epsilon=attention.query_key_norm_epsilon,
        rope_cosine=attention.rope_cosine,
        rope_sine=attention.rope_sine,
        causal_mask=attention.causal_mask,
        query_heads=attention.query_heads,
        key_value_heads=attention.key_value_heads,
        head_dim=attention.head_dim,
        scaling=attention.scaling,
    )
    perturbed_mlp = Lfm2MlpParameters(
        norm_gain=mlp.norm_gain,
        norm_epsilon=mlp.norm_epsilon,
        gate_weight=mlp.gate_weight,
        up_weight=mlp.up_weight,
        down_weight=mlp.down_weight + signed_step * (basis @ coefficient_blocks[1]),
    )
    return capture_lfm2_macrocell_linearization(
        residual + signed_step * incoming,
        perturbed_attention,
        perturbed_mlp,
    )


def test_attention_macrocell_finite_difference_and_adjoint() -> None:
    torch.manual_seed(12)
    batch, sequence, hidden, intermediate, rank = 2, 4, 6, 7, 3
    residual = _randn(batch, sequence, hidden)
    attention = _attention_parameters(
        batch=batch,
        sequence=sequence,
        hidden=hidden,
    )
    mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    linearization = capture_lfm2_macrocell_linearization(
        residual,
        attention,
        mlp,
    )
    incoming = _randn(batch, sequence, hidden)
    basis = _randn(hidden, rank)
    coefficients = (
        _randn(rank, attention.writer_input_width),
        _randn(rank, mlp.writer_input_width),
    )
    boundary_cotangents = (
        _randn(batch, sequence, hidden),
        _randn(batch, sequence, hidden),
    )
    step = 1e-6

    tangent = linearization.coupled_jvp(incoming, coefficients, basis)
    transpose = linearization.coupled_vjp(boundary_cotangents, basis)
    plus = _perturbed_attention_macrocell(
        residual=residual,
        incoming=incoming,
        attention=attention,
        mlp=mlp,
        basis=basis,
        coefficient_blocks=coefficients,
        signed_step=step,
    )
    minus = _perturbed_attention_macrocell(
        residual=residual,
        incoming=incoming,
        attention=attention,
        mlp=mlp,
        basis=basis,
        coefficient_blocks=coefficients,
        signed_step=-step,
    )
    finite_differences = (
        (plus.operator.output - minus.operator.output) / (2.0 * step),
        (plus.mlp.output - minus.mlp.output) / (2.0 * step),
    )

    for observed, expected in zip(
        tangent.boundaries,
        finite_differences,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=4e-8, atol=4e-9)
    left = sum(
        _inner(value, cotangent)
        for value, cotangent in zip(
            tangent.boundaries,
            boundary_cotangents,
            strict=True,
        )
    )
    right = _inner(incoming, transpose.incoming_residual) + sum(
        _inner(value, cotangent)
        for value, cotangent in zip(
            coefficients,
            transpose.coefficient_blocks,
            strict=True,
        )
    )
    torch.testing.assert_close(left, right, rtol=3e-12, atol=3e-12)
    assert linearization.operator_kind == "full_attention"
    assert linearization.boundary_order == (
        "post_operator_residual",
        "post_layer_residual",
    )


def _perturbed_conv_macrocell(
    *,
    residual: torch.Tensor,
    incoming: torch.Tensor,
    conv: Lfm2ShortConvParameters,
    mlp: Lfm2MlpParameters,
    basis: torch.Tensor,
    coefficient_blocks: tuple[torch.Tensor, torch.Tensor],
    signed_step: float,
):
    perturbed_conv = Lfm2ShortConvParameters(
        operator_norm_gain=conv.operator_norm_gain,
        operator_norm_epsilon=conv.operator_norm_epsilon,
        input_weight=conv.input_weight,
        stored_fir_kernel=conv.stored_fir_kernel,
        output_weight=(
            conv.output_weight + signed_step * (basis @ coefficient_blocks[0])
        ),
        input_bias=conv.input_bias,
        fir_bias=conv.fir_bias,
        output_bias=conv.output_bias,
        padding_mask=conv.padding_mask,
    )
    perturbed_mlp = Lfm2MlpParameters(
        norm_gain=mlp.norm_gain,
        norm_epsilon=mlp.norm_epsilon,
        gate_weight=mlp.gate_weight,
        up_weight=mlp.up_weight,
        down_weight=mlp.down_weight + signed_step * (basis @ coefficient_blocks[1]),
    )
    return capture_lfm2_macrocell_linearization(
        residual + signed_step * incoming,
        perturbed_conv,
        perturbed_mlp,
    )


def test_short_conv_macrocell_finite_difference_and_adjoint() -> None:
    torch.manual_seed(13)
    batch, sequence, hidden, intermediate, rank = 2, 4, 6, 7, 3
    residual = _randn(batch, sequence, hidden)
    conv = _conv_parameters(batch=batch, sequence=sequence, hidden=hidden)
    mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    linearization = capture_lfm2_macrocell_linearization(residual, conv, mlp)
    incoming = _randn(batch, sequence, hidden)
    basis = _randn(hidden, rank)
    coefficients = (
        _randn(rank, conv.writer_input_width),
        _randn(rank, mlp.writer_input_width),
    )
    boundary_cotangents = (
        _randn(batch, sequence, hidden),
        _randn(batch, sequence, hidden),
    )
    step = 1e-6

    tangent = linearization.coupled_jvp(incoming, coefficients, basis)
    transpose = linearization.coupled_vjp(boundary_cotangents, basis)
    plus = _perturbed_conv_macrocell(
        residual=residual,
        incoming=incoming,
        conv=conv,
        mlp=mlp,
        basis=basis,
        coefficient_blocks=coefficients,
        signed_step=step,
    )
    minus = _perturbed_conv_macrocell(
        residual=residual,
        incoming=incoming,
        conv=conv,
        mlp=mlp,
        basis=basis,
        coefficient_blocks=coefficients,
        signed_step=-step,
    )
    finite_differences = (
        (plus.operator.output - minus.operator.output) / (2.0 * step),
        (plus.mlp.output - minus.mlp.output) / (2.0 * step),
    )

    for observed, expected in zip(
        tangent.boundaries,
        finite_differences,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=4e-8, atol=4e-9)
    left = sum(
        _inner(value, cotangent)
        for value, cotangent in zip(
            tangent.boundaries,
            boundary_cotangents,
            strict=True,
        )
    )
    right = _inner(incoming, transpose.incoming_residual) + sum(
        _inner(value, cotangent)
        for value, cotangent in zip(
            coefficients,
            transpose.coefficient_blocks,
            strict=True,
        )
    )
    torch.testing.assert_close(left, right, rtol=3e-12, atol=3e-12)
    assert linearization.operator_kind == "short_conv"


@pytest.mark.parametrize("operator_kind", ["attention", "short_conv"])
def test_writer_only_callback_contract_and_norm_bounds(operator_kind: str) -> None:
    torch.manual_seed(14)
    batch, sequence, hidden, intermediate, rank = 2, 3, 6, 7, 3
    residual = _randn(batch, sequence, hidden)
    operator = (
        _attention_parameters(
            batch=batch,
            sequence=sequence,
            hidden=hidden,
        )
        if operator_kind == "attention"
        else _conv_parameters(
            batch=batch,
            sequence=sequence,
            hidden=hidden,
        )
    )
    mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    linearization = capture_lfm2_macrocell_linearization(
        residual,
        operator,
        mlp,
    )
    basis = _randn(hidden, rank)
    coefficients = tuple(_randn(rank, width) for width in linearization.input_widths)
    cotangents = (
        _randn(batch, sequence, hidden),
        _randn(batch, sequence, hidden),
    )

    effects = linearization.jvp(coefficients, basis)
    transpose = linearization.vjp(cotangents, basis)
    left = sum(
        _inner(value, cotangent)
        for value, cotangent in zip(effects, cotangents, strict=True)
    )
    right = sum(
        _inner(value, cotangent)
        for value, cotangent in zip(coefficients, transpose, strict=True)
    )
    torch.testing.assert_close(left, right, rtol=3e-12, atol=3e-12)
    bounds = linearization.squared_norm_upper_bounds(basis)
    assert bounds.operator_squared_norm_upper_bound == pytest.approx(
        sum(bounds.block_squared_norm_upper_bounds)
    )
    assert all(
        math.isfinite(value) and value > 0.0
        for value in bounds.block_squared_norm_upper_bounds
    )
    assert bounds.site_order == linearization.site_order
    assert bounds.boundary_order == linearization.boundary_order
    _assert_detached_fp64(*effects, *transpose)


def test_primal_snapshots_are_independently_owned_and_detached() -> None:
    torch.manual_seed(15)
    residual = _randn(1, 3, 4)
    original = residual.clone()
    attention = _attention_parameters(
        batch=1,
        sequence=3,
        hidden=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )
    mlp = _mlp_parameters(hidden=4, intermediate=5)
    linearization = capture_lfm2_macrocell_linearization(
        residual,
        attention,
        mlp,
    )
    residual.add_(100.0)

    torch.testing.assert_close(linearization.operator.residual, original)
    _assert_detached_fp64(
        linearization.operator.residual,
        linearization.operator.writer_input,
        linearization.operator.output,
        linearization.mlp.writer_input,
        linearization.mlp.output,
    )
    assert linearization.state_storage_device == "cpu"
    assert linearization.verify_authentication()


def test_primal_state_mutation_fails_authentication_before_callback() -> None:
    torch.manual_seed(16)
    residual = _randn(1, 3, 4)
    attention = _attention_parameters(
        batch=1,
        sequence=3,
        hidden=4,
        query_heads=2,
        key_value_heads=1,
        head_dim=2,
    )
    mlp = _mlp_parameters(hidden=4, intermediate=5)
    linearization = capture_lfm2_macrocell_linearization(
        residual,
        attention,
        mlp,
    )
    basis = _randn(4, 2)
    coefficients = tuple(
        torch.zeros((2, width), dtype=DTYPE) for width in linearization.input_widths
    )
    linearization.operator.writer_input[0, 0, 0] += 1.0

    assert not linearization.verify_authentication()
    with pytest.raises(
        Lfm2GlobalLinearizationError,
        match="changed after authentication",
    ):
        linearization.jvp(coefficients, basis)


def _kernel_contract(linearization, basis: torch.Tensor):
    writer_kind = (
        "attention_output"
        if linearization.operator_kind == "full_attention"
        else "short_conv_output"
    )
    topology = Lfm2GlobalResidualWriterTopology(
        hidden_size=int(linearization.operator.residual.shape[-1]),
        head_dimension=int(basis.shape[1]) // 2,
        sites=(
            Lfm2GlobalResidualWriterSite(
                site_id="layer.0.operator",
                layer_index=0,
                causal_order=0,
                writer_kind=writer_kind,
                input_width=linearization.input_widths[0],
            ),
            Lfm2GlobalResidualWriterSite(
                site_id="layer.0.mlp",
                layer_index=0,
                causal_order=1,
                writer_kind="mlp_down",
                input_width=linearization.input_widths[1],
            ),
        ),
    )
    boundaries = []
    for index, (kind, primal) in enumerate(
        zip(
            linearization.boundary_order,
            (linearization.operator.output, linearization.mlp.output),
            strict=True,
        )
    ):
        defect = torch.zeros_like(primal)
        mass = torch.ones(primal.shape[:-1], dtype=DTYPE)
        boundaries.append(
            seal_lfm2_global_residual_boundary(
                boundary_id=f"layer.0.{kind}",
                layer_index=0,
                causal_order=index,
                boundary_kind=kind,
                defect=defect,
                defect_sha256=fingerprint_lfm2_global_residual_tensor(defect),
                row_mass=mass,
                row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass),
            )
        )
    bundle = Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))
    bounds = linearization.squared_norm_upper_bounds(basis)
    receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        bundle,
        kernel_id="toy.lfm2.macrocell.fp64.v1",
        source_sha256={"macrocell_primal": linearization.authentication_sha256},
        analytic_math_components=(
            "continuous_fp64_rmsnorm",
            "causal_operator_and_swiglu_jvp",
            "manual_exact_transpose",
        ),
        operator_squared_norm_upper_bound=(bounds.operator_squared_norm_upper_bound),
        block_squared_norm_upper_bounds=(bounds.block_squared_norm_upper_bounds),
    )
    return topology, bundle, receipt


@pytest.mark.parametrize("operator_kind", ["attention", "short_conv"])
def test_authenticated_tensor_only_kernel_streams_exact_chunks(
    operator_kind: str,
) -> None:
    torch.manual_seed(17)
    batch, sequence, hidden, intermediate, rank = 2, 4, 6, 7, 4
    residual = _randn(batch, sequence, hidden)
    operator = (
        _attention_parameters(
            batch=batch,
            sequence=sequence,
            hidden=hidden,
        )
        if operator_kind == "attention"
        else _conv_parameters(
            batch=batch,
            sequence=sequence,
            hidden=hidden,
        )
    )
    mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
    linearization = capture_lfm2_macrocell_linearization(
        residual,
        operator,
        mlp,
    )
    basis = _randn(hidden, rank)
    coefficients = tuple(_randn(rank, width) for width in linearization.input_widths)
    boundary_cotangents = (
        _randn(batch, sequence, hidden),
        _randn(batch, sequence, hidden),
    )
    topology, boundaries, receipt = _kernel_contract(linearization, basis)
    kernel = Lfm2MacrocellLinearizationKernel(
        topology,
        boundaries,
        linearization,
        linearization_receipt=receipt,
    )

    expected_effects = linearization.jvp(coefficients, basis)
    observed_effects = []
    row_count = batch * sequence
    for boundary_index in range(2):
        chunks = []
        for row_start in range(0, row_count, 3):
            row_stop = min(row_count, row_start + 3)
            chunks.append(
                kernel.run_jvp_chunk(
                    coefficients,
                    basis,
                    boundary_index=boundary_index,
                    row_start=row_start,
                    row_stop=row_stop,
                )
            )
        observed_effects.append(
            torch.cat(chunks, dim=0).reshape(batch, sequence, hidden)
        )
    for observed, expected in zip(
        observed_effects,
        expected_effects,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)

    expected_transpose = linearization.vjp(boundary_cotangents, basis)
    observed_transpose = tuple(torch.zeros_like(value) for value in coefficients)
    for boundary_index, cotangent in enumerate(boundary_cotangents):
        flat = cotangent.reshape(row_count, hidden)
        for row_start in range(0, row_count, 3):
            row_stop = min(row_count, row_start + 3)
            contributions = kernel.run_vjp_chunk(
                flat[row_start:row_stop],
                basis,
                boundary_index=boundary_index,
                row_start=row_start,
                row_stop=row_stop,
            )
            observed_transpose = tuple(
                total + contribution
                for total, contribution in zip(
                    observed_transpose,
                    contributions,
                    strict=True,
                )
            )
    for observed, expected in zip(
        observed_transpose,
        expected_transpose,
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)
    assert kernel.state_storage_device == "cpu"
    assert kernel.verify_authentication(force_raw=True)
    assert kernel.authentication_sha256 == kernel.recompute_authentication_sha256()
    assert kernel.transpose_contract_sha256
    with pytest.raises(Lfm2GlobalResidualClosureError, match="immutable"):
        kernel.extra_state = object()


def _stack_layer_blueprints(*, hidden: int, intermediate: int):
    layers = []
    for layer_index in range(14):
        mlp = _mlp_parameters(hidden=hidden, intermediate=intermediate)
        if layer_index % 2 == 0:
            layers.append(
                {
                    "kind": "attention",
                    "operator_norm_gain": torch.rand(hidden, dtype=DTYPE) + 0.5,
                    "query_weight": _randn(4, hidden, scale=0.04),
                    "key_weight": _randn(2, hidden, scale=0.04),
                    "value_weight": _randn(2, hidden, scale=0.04),
                    "output_weight": _randn(hidden, 4, scale=0.04),
                    "query_norm_gain": torch.rand(2, dtype=DTYPE) + 0.5,
                    "key_norm_gain": torch.rand(2, dtype=DTYPE) + 0.5,
                    "mlp": mlp,
                }
            )
        else:
            layers.append(
                {
                    "kind": "short_conv",
                    "operator_norm_gain": torch.rand(hidden, dtype=DTYPE) + 0.5,
                    "input_weight": _randn(3 * hidden, hidden, scale=0.04),
                    "stored_fir_kernel": _randn(hidden, 1, 2, scale=0.04),
                    "output_weight": _randn(hidden, hidden, scale=0.04),
                    "input_bias": _randn(3 * hidden, scale=0.01),
                    "fir_bias": _randn(hidden, scale=0.01),
                    "output_bias": _randn(hidden, scale=0.01),
                    "mlp": mlp,
                }
            )
    return tuple(layers)


def _stack_operator_parameters(
    blueprint,
    *,
    batch_index: int,
    sequence: int,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    signed_step: float,
):
    output_weight = blueprint["output_weight"] + signed_step * (basis @ coefficients)
    if blueprint["kind"] == "attention":
        angles = torch.arange(sequence, dtype=DTYPE).reshape(1, sequence, 1) * (
            0.031 + 0.001 * batch_index
        )
        cosine_half = angles.cos()
        sine_half = angles.sin()
        mask = torch.tril(torch.ones((sequence, sequence), dtype=torch.bool)).reshape(
            1,
            1,
            sequence,
            sequence,
        )
        return Lfm2AttentionParameters(
            operator_norm_gain=blueprint["operator_norm_gain"],
            operator_norm_epsilon=1e-3,
            query_weight=blueprint["query_weight"],
            key_weight=blueprint["key_weight"],
            value_weight=blueprint["value_weight"],
            output_weight=output_weight,
            query_norm_gain=blueprint["query_norm_gain"],
            key_norm_gain=blueprint["key_norm_gain"],
            query_key_norm_epsilon=2e-3,
            rope_cosine=torch.cat((cosine_half, cosine_half), dim=-1),
            rope_sine=torch.cat((sine_half, sine_half), dim=-1),
            causal_mask=mask,
            query_heads=2,
            key_value_heads=1,
            head_dim=2,
        )
    padding = torch.ones((1, sequence), dtype=DTYPE)
    if batch_index == 1:
        padding[0, -1] = 0.0
    return Lfm2ShortConvParameters(
        operator_norm_gain=blueprint["operator_norm_gain"],
        operator_norm_epsilon=1e-3,
        input_weight=blueprint["input_weight"],
        stored_fir_kernel=blueprint["stored_fir_kernel"],
        output_weight=output_weight,
        input_bias=blueprint["input_bias"],
        fir_bias=blueprint["fir_bias"],
        output_bias=blueprint["output_bias"],
        padding_mask=padding,
    )


def _stack_mlp_parameters(
    blueprint,
    *,
    basis: torch.Tensor,
    coefficients: torch.Tensor,
    signed_step: float,
) -> Lfm2MlpParameters:
    mlp = blueprint["mlp"]
    return Lfm2MlpParameters(
        norm_gain=mlp.norm_gain,
        norm_epsilon=mlp.norm_epsilon,
        gate_weight=mlp.gate_weight,
        up_weight=mlp.up_weight,
        down_weight=mlp.down_weight + signed_step * (basis @ coefficients),
    )


def _build_stack_batch(
    blueprints,
    *,
    batch_index: int,
    initial_residual: torch.Tensor,
    selected_target_positions: torch.Tensor,
    canonical_boundary_row_indices: torch.Tensor,
    terminal_parameters: Lfm2TerminalNormParameters,
    basis: torch.Tensor,
    coefficients: tuple[torch.Tensor, ...],
    signed_step: float = 0.0,
    fixed_cast_offsets: tuple[torch.Tensor, ...] | None = None,
):
    residual = initial_residual
    macrocells = []
    captured_offsets = []
    sequence = int(initial_residual.shape[1])
    for layer_index, blueprint in enumerate(blueprints):
        operator = _stack_operator_parameters(
            blueprint,
            batch_index=batch_index,
            sequence=sequence,
            basis=basis,
            coefficients=coefficients[2 * layer_index],
            signed_step=signed_step,
        )
        mlp = _stack_mlp_parameters(
            blueprint,
            basis=basis,
            coefficients=coefficients[2 * layer_index + 1],
            signed_step=signed_step,
        )
        macrocell = capture_lfm2_macrocell_linearization(residual, operator, mlp)
        macrocells.append(macrocell)
        if fixed_cast_offsets is None:
            rounded = macrocell.mlp.output.to(torch.bfloat16).to(DTYPE)
            offset = rounded - macrocell.mlp.output
        else:
            offset = fixed_cast_offsets[layer_index]
        captured_offsets.append(offset)
        residual = macrocell.mlp.output + offset
    terminal = capture_lfm2_terminal_norm_primal(residual, terminal_parameters)
    return (
        Lfm2StackPrimalBatch(
            macrocells=tuple(macrocells),
            selected_target_positions=selected_target_positions,
            canonical_boundary_row_indices=canonical_boundary_row_indices,
            terminal=terminal,
        ),
        tuple(captured_offsets),
    )


def _gather_stack_boundaries(
    snapshot: Lfm2WholeStackPrimalSnapshot,
) -> tuple[torch.Tensor, ...]:
    row_count = snapshot.row_count
    hidden = snapshot.hidden_size
    values = [torch.zeros((row_count, hidden), dtype=DTYPE) for _ in range(29)]
    for batch in snapshot.batches:
        selected = batch.selected_target_positions
        canonical = batch.canonical_boundary_row_indices
        for layer_index, macrocell in enumerate(batch.macrocells):
            for boundary_offset, primal in enumerate(
                (macrocell.operator.output, macrocell.mlp.output)
            ):
                values[2 * layer_index + boundary_offset].index_copy_(
                    0,
                    canonical,
                    primal.reshape(-1, hidden).index_select(0, selected),
                )
        values[-1].index_copy_(
            0,
            canonical,
            batch.terminal.output.reshape(-1, hidden).index_select(0, selected),
        )
    return tuple(value.contiguous() for value in values)


def _whole_stack_fixture():
    torch.manual_seed(18)
    hidden, intermediate, rank = 4, 5, 4
    blueprints = _stack_layer_blueprints(
        hidden=hidden,
        intermediate=intermediate,
    )
    basis = _randn(hidden, rank, scale=0.15)
    coefficients = tuple(
        _randn(
            rank,
            hidden if index % 2 == 0 else intermediate,
            scale=0.08,
        )
        for index in range(28)
    )
    terminal_parameters = Lfm2TerminalNormParameters(
        gain=torch.rand(hidden, dtype=DTYPE) + 0.5,
        epsilon=1e-3,
    )
    batch_specs = (
        (
            _randn(1, 7, hidden, scale=0.15),
            torch.tensor([0, 2, 4, 6]),
            torch.tensor([0, 2, 4, 6]),
        ),
        (
            _randn(1, 5, hidden, scale=0.15),
            torch.tensor([0, 2, 4]),
            torch.tensor([1, 3, 5]),
        ),
    )
    batches = []
    offsets = []
    for batch_index, (initial, selected, canonical) in enumerate(batch_specs):
        batch, batch_offsets = _build_stack_batch(
            blueprints,
            batch_index=batch_index,
            initial_residual=initial,
            selected_target_positions=selected,
            canonical_boundary_row_indices=canonical,
            terminal_parameters=terminal_parameters,
            basis=basis,
            coefficients=coefficients,
        )
        batches.append(batch)
        offsets.append(batch_offsets)
    snapshot = Lfm2WholeStackPrimalSnapshot(tuple(batches))
    sites = []
    for layer_index, blueprint in enumerate(blueprints):
        sites.extend(
            (
                Lfm2GlobalResidualWriterSite(
                    site_id=f"layer.{layer_index}.operator",
                    layer_index=layer_index,
                    causal_order=2 * layer_index,
                    writer_kind=(
                        "attention_output"
                        if blueprint["kind"] == "attention"
                        else "short_conv_output"
                    ),
                    input_width=hidden,
                ),
                Lfm2GlobalResidualWriterSite(
                    site_id=f"layer.{layer_index}.mlp",
                    layer_index=layer_index,
                    causal_order=2 * layer_index + 1,
                    writer_kind="mlp_down",
                    input_width=intermediate,
                ),
            )
        )
    topology = Lfm2GlobalResidualWriterTopology(
        hidden_size=hidden,
        head_dimension=2,
        sites=tuple(sites),
    )
    boundaries = []
    for boundary_index in range(29):
        terminal = boundary_index == 28
        layer_index = 13 if terminal else boundary_index // 2
        kind = (
            "terminal_observable"
            if terminal
            else (
                "post_operator_residual"
                if boundary_index % 2 == 0
                else "post_layer_residual"
            )
        )
        defect = torch.zeros((snapshot.row_count, hidden), dtype=DTYPE)
        mass = torch.ones(snapshot.row_count, dtype=DTYPE)
        boundaries.append(
            seal_lfm2_global_residual_boundary(
                boundary_id=f"boundary.{boundary_index}.{kind}",
                layer_index=layer_index,
                causal_order=boundary_index,
                boundary_kind=kind,
                defect=defect,
                defect_sha256=fingerprint_lfm2_global_residual_tensor(defect),
                row_mass=mass,
                row_mass_sha256=fingerprint_lfm2_global_residual_tensor(mass),
            )
        )
    bundle = Lfm2GlobalResidualBoundaryBundle(tuple(boundaries))
    bounds = snapshot.squared_norm_upper_bounds(basis)
    receipt = seal_lfm2_global_residual_linearization_receipt(
        topology,
        bundle,
        kernel_id="toy.lfm2.whole_stack.fp64.v1",
        source_sha256={"stack_primal": snapshot.authentication_sha256},
        analytic_math_components=(
            "continuous_fp64_14_layer_stack",
            "runtime_cast_straight_through_identity",
            "token_bridge_gather_scatter",
            "manual_exact_transpose",
        ),
        operator_squared_norm_upper_bound=bounds.operator_squared_norm_upper_bound,
        block_squared_norm_upper_bounds=bounds.block_squared_norm_upper_bounds,
    )
    kernel = Lfm2WholeStackLinearizationKernel(
        topology,
        bundle,
        snapshot,
        linearization_receipt=receipt,
    )
    return {
        "basis": basis,
        "batch_specs": batch_specs,
        "blueprints": blueprints,
        "coefficients": coefficients,
        "kernel": kernel,
        "offsets": tuple(offsets),
        "snapshot": snapshot,
        "terminal_parameters": terminal_parameters,
    }


def _perturbed_whole_stack(fixture, signed_step: float):
    batches = []
    for batch_index, (initial, selected, canonical) in enumerate(
        fixture["batch_specs"]
    ):
        batch, _ = _build_stack_batch(
            fixture["blueprints"],
            batch_index=batch_index,
            initial_residual=initial,
            selected_target_positions=selected,
            canonical_boundary_row_indices=canonical,
            terminal_parameters=fixture["terminal_parameters"],
            basis=fixture["basis"],
            coefficients=fixture["coefficients"],
            signed_step=signed_step,
            fixed_cast_offsets=fixture["offsets"][batch_index],
        )
        batches.append(batch)
    return Lfm2WholeStackPrimalSnapshot(tuple(batches))


def test_whole_stack_bundle_matches_finite_difference_and_exact_transpose() -> None:
    fixture = _whole_stack_fixture()
    kernel = fixture["kernel"]
    basis = fixture["basis"]
    coefficients = fixture["coefficients"]
    snapshot = fixture["snapshot"]
    step = 2e-6

    assert snapshot.batches[0].full_target_row_count == 7
    assert snapshot.batches[0].row_count == 4
    assert any(
        not torch.equal(
            batch.macrocells[layer_index - 1].mlp.output,
            batch.macrocells[layer_index].operator.residual,
        )
        for batch in snapshot.batches
        for layer_index in range(1, 14)
    )
    effects = kernel.run_jvp_bundle(
        coefficients,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    plus = _gather_stack_boundaries(_perturbed_whole_stack(fixture, step))
    minus = _gather_stack_boundaries(_perturbed_whole_stack(fixture, -step))
    finite_differences = tuple(
        (plus_value - minus_value) / (2.0 * step)
        for plus_value, minus_value in zip(plus, minus, strict=True)
    )
    for observed, expected in zip(effects, finite_differences, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2e-6, atol=2e-8)
    conditioning = snapshot.conditioning_scales(basis)
    bounds = snapshot.squared_norm_upper_bounds(basis)
    coefficient_energy = sum(
        float(value.square().sum().item()) for value in coefficients
    )
    effect_energy = sum(float(value.square().sum().item()) for value in effects)
    assert (
        effect_energy <= bounds.operator_squared_norm_upper_bound * coefficient_energy
    )
    assert bounds.operator_squared_norm_upper_bound == pytest.approx(
        sum(bounds.block_squared_norm_upper_bounds)
    )
    assert conditioning.operator_squared_scale == pytest.approx(
        sum(conditioning.block_squared_scales)
    )
    assert (
        conditioning.operator_squared_scale <= bounds.operator_squared_norm_upper_bound
    )

    cotangents = tuple(
        _randn(snapshot.row_count, snapshot.hidden_size) for _ in range(29)
    )
    transpose = kernel.run_vjp_bundle(
        cotangents,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    left = sum(
        _inner(effect, cotangent)
        for effect, cotangent in zip(effects, cotangents, strict=True)
    )
    right = sum(
        _inner(coefficient, coefficient_cotangent)
        for coefficient, coefficient_cotangent in zip(
            coefficients,
            transpose,
            strict=True,
        )
    )
    torch.testing.assert_close(left, right, rtol=2e-11, atol=2e-11)

    terminal_cotangent = _randn(snapshot.row_count, snapshot.hidden_size)
    terminal_only = tuple(
        terminal_cotangent if index == 28 else torch.zeros_like(terminal_cotangent)
        for index in range(29)
    )
    terminal_transpose = kernel.run_vjp_bundle(
        terminal_only,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    torch.testing.assert_close(
        _inner(effects[-1], terminal_cotangent),
        sum(
            _inner(coefficient, coefficient_cotangent)
            for coefficient, coefficient_cotangent in zip(
                coefficients,
                terminal_transpose,
                strict=True,
            )
        ),
        rtol=2e-11,
        atol=2e-11,
    )
    assert kernel.supports_bundle_linearization
    assert kernel.state_storage_device == "cpu"
    assert kernel.verify_authentication(force_raw=True)
    _assert_detached_fp64(*effects, *transpose)


def test_whole_stack_causality_and_scaled_transpose() -> None:
    fixture = _whole_stack_fixture()
    kernel = fixture["kernel"]
    basis = fixture["basis"]
    coefficients = tuple(torch.zeros_like(value) for value in fixture["coefficients"])
    coefficients = tuple(
        _randn(*value.shape, scale=0.08) if index == 15 else value
        for index, value in enumerate(coefficients)
    )
    effects = kernel.run_jvp_bundle(
        coefficients,
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    assert all(int(torch.count_nonzero(value).item()) == 0 for value in effects[:15])
    assert float(torch.linalg.vector_norm(effects[15]).item()) > 0.0

    row_count = fixture["snapshot"].row_count
    hidden = fixture["snapshot"].hidden_size
    cotangents = tuple(_randn(row_count, hidden) for _ in range(29))
    scales = tuple(torch.rand(row_count, dtype=DTYPE) + 0.25 for _ in range(29))
    scaled = kernel.run_vjp_bundle(
        cotangents,
        basis,
        boundary_row_scales_cpu=scales,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    explicit = kernel.run_vjp_bundle(
        tuple(
            cotangent * scale.unsqueeze(-1)
            for cotangent, scale in zip(cotangents, scales, strict=True)
        ),
        basis,
        maximum_workspace_bytes=64 * 1024 * 1024,
    )
    for observed, expected in zip(scaled, explicit, strict=True):
        torch.testing.assert_close(observed, expected, rtol=2e-12, atol=2e-12)
