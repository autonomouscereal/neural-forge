"""Stage 1 of the attention <-> ShortConv operator-KIND translation.

The claims under test are the ones the design document is willing to make at toy
scale, and no others:

* a kernel-``L`` depthwise FIR with frozen gates **is** a banded linear
  attention, exactly, including the boundary truncation and the additive
  constant;
* with live gates the bilinear expansion leaves a quadratic remainder that grows
  with gate dispersion and is measured rather than fitted;
* the compiled softmax GQA/RoPE branch is produced by the *existing* attention
  kernel fed synthesized evidence, and everything it loses — band leak, the
  unrepresentable FIR prefix, the additive constant — is measured;
* a cached attention delta splits row-exactly into band and far mass, the band
  is fitted by bounded deterministic variable projection, and the far mass is
  routed or the conversion fails closed;
* every receipt is self-sealed and two builds are byte-identical.
"""

from __future__ import annotations

from typing import Any

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_causal_gqa_attention_fold import (  # noqa: E402
    HFCausalGQATopology,
    apply_rope,
    causal_attention_probabilities,
    compile_hf_causal_gqa_attention_fold,
)
from neural_forge.hf_causal_kind_translation import (  # noqa: E402
    ATTENTION_TO_SHORT_CONV_SCHEMA,
    KIND_TRANSLATION_REMAINDER_SCHEMA,
    SHORT_CONV_TO_ATTENTION_SCHEMA,
    AttentionToShortConvPolicy,
    HFCausalKindTranslationError,
    ShortConvBranchParameters,
    ShortConvToAttentionPolicy,
    attention_band_decomposition,
    attention_evidence_from_projections,
    carrier_endpoint_target_with_remainder,
    compile_attention_to_short_conv_translation,
    compile_short_conv_to_attention_translation,
    evaluate_carrier_routing_checkpoint,
    linearize_short_conv_branch,
    seal_kind_translation_remainder,
)
from neural_forge.hf_lfm2_canonical import (  # noqa: E402
    Lfm2ShortConvParameters,
    lfm2_short_conv_reference,
)


BATCH = 8
SEQUENCE = 16
HIDDEN = 8
CHANNELS = 8
FIR_LENGTH = 3


def _rows(seed: int = 11, *, noise: float = 0.15) -> Any:
    """Operator-norm rows with a dominant DC direction.

    Direction B compiles constant pre-RoPE Q/K out of the child rows, so the
    band it achieves is a function of how much of the residual stream is a
    shared direction.  A real RMSNorm output has exactly this shape; the noise
    scale is the knob that makes the risk visible.
    """

    generator = torch.Generator().manual_seed(seed)
    direct = (
        torch.randn(HIDDEN, generator=generator, dtype=torch.float64) * 1.5
    ).reshape(1, 1, -1)
    return direct + noise * torch.randn(
        BATCH, SEQUENCE, HIDDEN, generator=generator, dtype=torch.float64
    )


def _short_conv(
    *,
    seed: int = 5,
    gate_scale: float = 0.0,
    bias: bool = True,
    channels: int = CHANNELS,
) -> ShortConvBranchParameters:
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int) -> Any:
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    gate_b = gate_scale * normal(channels, HIDDEN) / 4.0
    gate_c = gate_scale * normal(channels, HIDDEN) / 4.0
    stream_x = normal(channels, HIDDEN) / 3.0
    biases: dict[str, Any] = {}
    if bias:
        biases = {
            "input_projection_bias": torch.cat(
                (
                    torch.ones(channels, dtype=torch.float64),
                    torch.ones(channels, dtype=torch.float64),
                    torch.zeros(channels, dtype=torch.float64),
                )
            ),
            "conv_bias": torch.zeros(channels, dtype=torch.float64),
            "output_projection_bias": torch.zeros(HIDDEN, dtype=torch.float64),
        }
    return ShortConvBranchParameters(
        input_projection_weight=torch.cat((gate_b, gate_c, stream_x), dim=0),
        conv_weight=normal(channels, 1, FIR_LENGTH),
        output_projection_weight=normal(HIDDEN, channels) / 2.0,
        **biases,
    )


def _partitions() -> tuple[Any, Any, Any, Any]:
    valid = torch.ones(BATCH, SEQUENCE, dtype=torch.bool)
    fit = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    selection = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    validation = torch.zeros(BATCH, SEQUENCE, dtype=torch.bool)
    for batch in range(BATCH):
        for position in range(SEQUENCE):
            bucket = (batch * SEQUENCE + position) % 4
            if bucket == 3:
                validation[batch, position] = True
            elif bucket == 2:
                selection[batch, position] = True
            else:
                fit[batch, position] = True
    return valid, fit, selection, validation


def _positions() -> Any:
    return torch.arange(SEQUENCE).reshape(1, -1).expand(BATCH, -1).contiguous()


def _attention_mask() -> Any:
    return torch.ones(BATCH, SEQUENCE, dtype=torch.bool)


def _topology(*, query_heads: int = 4, key_value_heads: int = 4) -> HFCausalGQATopology:
    return HFCausalGQATopology(
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=8,
        rope_theta=10_000.0,
        attention_scaling=1.0 / (8.0**0.5),
    )


def _translate_conv(
    branch: ShortConvBranchParameters,
    *,
    rows: Any | None = None,
    topology: HFCausalGQATopology | None = None,
    policy: ShortConvToAttentionPolicy | None = None,
    carrier_layer_index: int | None = None,
    **overrides: Any,
) -> Any:
    valid, fit, selection, validation = _partitions()
    arguments: dict[str, Any] = {
        "short_conv_parameters": branch,
        "target_topology": topology or _topology(),
        "attention_mask": _attention_mask(),
        "position_ids": _positions(),
        "valid_token_mask": valid,
        "fit_token_mask": fit,
        "selection_token_mask": selection,
        "validation_token_mask": validation,
        "carrier_layer_index": carrier_layer_index,
        "policy": policy or ShortConvToAttentionPolicy(),
    }
    arguments.update(overrides)
    return compile_short_conv_to_attention_translation(
        _rows() if rows is None else rows, **arguments
    )


def _relative_distance_attention(
    rows: Any,
    *,
    lags: tuple[int, ...],
    sharpness: float = 4.0,
    content_coupling: float = 0.0,
    seed: int = 3,
) -> dict[str, Any]:
    """A cached attention layer whose head ``h`` concentrates on ``lags[h]``.

    Scores are ``-sharpness * (t - s - lag)^2 + content_coupling * <n_t, n_s>``.
    The quadratic part is realized exactly by three feature coordinates, so the
    probabilities come out of ``causal_attention_probabilities`` itself rather
    than being hand-written.  ``content_coupling`` turns the band from
    lag-separable into a genuinely non-separable content interaction, which is
    the adversarial leg of the design's stage-1 plan.
    """

    generator = torch.Generator().manual_seed(seed)
    heads = len(lags)
    head_dim = 4
    positions = torch.arange(SEQUENCE, dtype=torch.float64)
    coupling = float(content_coupling) ** 0.5
    query_features = []
    key_features = []
    for lag in lags:
        shifted = positions + float(lag)
        query = torch.stack(
            (
                positions.square(),
                positions,
                torch.ones(SEQUENCE, dtype=torch.float64),
            ),
            dim=-1,
        )
        key = torch.stack(
            (
                torch.full((SEQUENCE,), -sharpness, dtype=torch.float64),
                2.0 * sharpness * shifted,
                -sharpness * shifted.square(),
            ),
            dim=-1,
        )
        query = query.reshape(1, SEQUENCE, 3).expand(BATCH, -1, -1)
        key = key.reshape(1, SEQUENCE, 3).expand(BATCH, -1, -1)
        if coupling > 0.0:
            query = torch.cat((query, coupling * rows), dim=-1)
            key = torch.cat((key, coupling * rows), dim=-1)
        query_features.append(query)
        key_features.append(key)
    query = torch.stack(query_features, dim=1).contiguous()
    key = torch.stack(key_features, dim=1).contiguous()
    _scores, probabilities = causal_attention_probabilities(
        query,
        key,
        attention_mask=_attention_mask(),
        scaling=1.0,
    )
    value_weight = (
        torch.randn(heads * head_dim, HIDDEN, generator=generator, dtype=torch.float64)
        / 3.0
    )
    output_weight = (
        torch.randn(HIDDEN, heads * head_dim, generator=generator, dtype=torch.float64)
        / 3.0
    )
    values = rows @ value_weight.T
    head_values = (
        values.reshape(BATCH, SEQUENCE, heads, head_dim)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    contexts = torch.matmul(probabilities, head_values)
    delta = (
        contexts.permute(0, 2, 1, 3).reshape(BATCH, SEQUENCE, heads * head_dim)
        @ output_weight.T
    )
    return {
        "attention_probabilities": probabilities,
        "attention_value_sequences": values,
        "attention_output_weight": output_weight,
        "query_heads": heads,
        "key_value_heads": heads,
        "head_dim": head_dim,
        "delta": delta,
    }


def _translate_attention(
    evidence: dict[str, Any],
    *,
    rows: Any,
    carrier_layer_index: int | None = None,
    policy: AttentionToShortConvPolicy | None = None,
    **overrides: Any,
) -> Any:
    valid, fit, selection, validation = _partitions()
    arguments: dict[str, Any] = {
        "attention_probabilities": evidence["attention_probabilities"],
        "attention_value_sequences": evidence["attention_value_sequences"],
        "attention_output_weight": evidence["attention_output_weight"],
        "query_heads": evidence["query_heads"],
        "key_value_heads": evidence["key_value_heads"],
        "head_dim": evidence["head_dim"],
        "target_residual_delta_sequences": evidence["delta"],
        "valid_token_mask": valid,
        "fit_token_mask": fit,
        "selection_token_mask": selection,
        "validation_token_mask": validation,
        "carrier_layer_index": carrier_layer_index,
        "policy": policy
        or AttentionToShortConvPolicy(fir_length=FIR_LENGTH, channel_width=HIDDEN),
    }
    arguments.update(overrides)
    return compile_attention_to_short_conv_translation(rows, **arguments)


def _self_sealed(report: Any) -> bool:
    body = {key: value for key, value in dict(report).items() if key != "report_sha256"}
    return canonical_sha256(body) == report["report_sha256"]


# ---------------------------------------------------------------------------
# ShortConv semantics and the banded-linear-attention identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_short_conv_forward_reproduces_the_canonical_reference(bias: bool) -> None:
    branch = _short_conv(gate_scale=1.0, bias=bias)
    rows = _rows(noise=1.0)
    reference = Lfm2ShortConvParameters(
        in_proj_weight=branch.input_projection_weight,
        in_proj_bias=branch.input_projection_bias,
        conv_weight=branch.conv_weight,
        conv_bias=branch.conv_bias,
        out_proj_weight=branch.output_projection_weight,
        out_proj_bias=branch.output_projection_bias,
    )
    expected = lfm2_short_conv_reference(reference, rows)
    assert torch.equal(branch.forward(rows), expected)


def test_gate_collapse_makes_the_banded_identity_exact() -> None:
    branch = _short_conv(gate_scale=0.0)
    rows = _rows()
    _valid, fit, _selection, _validation = _partitions()
    linearization = linearize_short_conv_branch(branch, rows, row_mask=fit)
    delta = branch.forward(rows)
    scale = float(delta.abs().max().item())
    assert scale > 1.0
    # The identity is Delta_t = sum_l n_{t-l} M_l^T + n_t M_self^T + c_t; under
    # gate collapse M_self is exactly zero and the remainder is exactly zero.
    assert float(linearization.self_gate_map.abs().max().item()) == 0.0
    assert float((linearization.banded_delta - delta).abs().max().item()) < 1e-12 * scale
    assert linearization.diagnostics["gate_collapse_measure"] == 0.0
    assert linearization.diagnostics["quadratic_remainder_relative_energy"] < 1e-28
    assert len(linearization.lag_maps) == FIR_LENGTH
    assert linearization.diagnostics["boundary_constant_rows"] == FIR_LENGTH - 1


def test_gate_collapse_identity_holds_for_heterogeneous_channels() -> None:
    branch = _short_conv(gate_scale=0.0, channels=5)
    rows = _rows()
    _valid, fit, _selection, _validation = _partitions()
    linearization = linearize_short_conv_branch(branch, rows, row_mask=fit)
    delta = branch.forward(rows)
    assert branch.channel_width == 5
    assert float((linearization.banded_delta - delta).abs().max().item()) < 1e-12 * float(
        delta.abs().max().item()
    )


def test_gate_linearization_remainder_grows_with_gate_variance() -> None:
    rows = _rows()
    _valid, fit, _selection, _validation = _partitions()
    dispersions: list[float] = []
    energies: list[float] = []
    for scale in (0.0, 0.25, 0.5, 1.0, 2.0):
        linearization = linearize_short_conv_branch(
            _short_conv(gate_scale=scale), rows, row_mask=fit
        )
        dispersions.append(linearization.diagnostics["gate_collapse_measure"])
        energies.append(
            float(linearization.quadratic_remainder[fit].norm().item())
        )
    assert dispersions[0] == 0.0
    assert energies[0] < 1e-12
    assert dispersions == sorted(dispersions)
    assert energies == sorted(energies)
    assert energies[-1] > 1.0


# ---------------------------------------------------------------------------
# Direction B: ShortConv -> attention
# ---------------------------------------------------------------------------


def test_collapsed_gate_translation_compiles_a_banded_attention() -> None:
    translation = _translate_conv(_short_conv(gate_scale=0.0))
    report = translation.report
    assert report["schema"] == SHORT_CONV_TO_ATTENTION_SCHEMA
    assert report["reused_attention_kernel"] == "compile_hf_causal_gqa_attention_fold"
    assert _self_sealed(report)
    assert report["gate_collapse_detected"] is True
    assert report["self_gate_stream_included"] is False
    assert report["stream_count"] == FIR_LENGTH
    # The exact part is exact: the frozen-gate/linear-FIR baseline reproduces the
    # branch to numerical precision, so every reported endpoint error belongs to
    # the softmax compilation, not to the operator algebra.
    assert report["baseline_frozen_gate_linear_fir_validation_relative_rmse"] < 1e-12
    for stream in report["streams"]:
        assert stream["stream_rank_truncation_relative_error"] < 1e-12
        assert stream["margin_logits"] > 0.0
        assert stream["series_construction_max_absolute_error"] < 1e-6
        assert stream["normalized_margin"] > 0.0
    assert report["minimum_validation_assigned_lag_probability_mass"] > 0.8
    assert report["validation_endpoint_relative_rmse"] < 0.05
    assert report["validation_endpoint_relative_rmse_bf16"] < 0.05
    assert report["validation_endpoint_relative_rmse_bf16"] < (
        report["baseline_zero_validation_relative_rmse"]
    )
    assert report["bf16_versus_fp64_endpoint_relative_delta"] < 0.05
    # The FIR prefix is dropped from the solve, counted, and scored separately.
    assert report["unrepresentable_prefix_positions"] == FIR_LENGTH - 1
    assert report["fit_rows_excluded_as_unrepresentable_prefix"] > 0
    assert report["fit_rows"] + report["fit_rows_excluded_as_unrepresentable_prefix"] == (
        report["supplied_fit_rows"]
    )
    assert report["total_representable_value_rank"] == 32
    assert report["parent_forward_calls"] == 0
    assert report["optimizer_steps"] == 0


def test_live_gate_translation_measures_the_quadratic_remainder() -> None:
    translation = _translate_conv(_short_conv(gate_scale=1.0), carrier_layer_index=6)
    report = translation.report
    assert report["gate_collapse_detected"] is False
    assert report["self_gate_stream_included"] is True
    assert report["stream_count"] == FIR_LENGTH + 1
    quadratic = report["quadratic_remainder_relative_energy"]
    assert 0.0 < quadratic < 0.01
    # rho_2 is measured, not fitted: the frozen-gate baseline is exactly the
    # square root of the quadratic energy ratio.
    assert report["baseline_frozen_gate_linear_fir_validation_relative_rmse"] > 0.0
    assert report["validation_endpoint_relative_rmse"] < (
        report["baseline_zero_validation_relative_rmse"]
    )
    bundle = seal_kind_translation_remainder(
        direction="short_conv_to_attention",
        source_layer_index=report["source_layer_index"],
        carrier_layer_index=report["carrier_layer_index"],
        remainder=translation.remainder,
        source_report_sha256=report["report_sha256"],
        diagnostics={
            "quadratic_remainder_relative_energy": quadratic,
            "gate_collapse_measure": report["linearization"]["gate_collapse_measure"],
        },
    )
    assert bundle.report["schema"] == KIND_TRANSLATION_REMAINDER_SCHEMA
    assert bundle.report["remainder_shape"] == [BATCH, SEQUENCE, HIDDEN]
    assert bundle.report["source_report_sha256"] == report["report_sha256"]
    assert _self_sealed(bundle.report)


def test_translation_fails_closed_without_a_key_value_head_per_stream() -> None:
    with pytest.raises(HFCausalKindTranslationError, match="key/value heads"):
        _translate_conv(
            _short_conv(gate_scale=0.0),
            topology=_topology(query_heads=4, key_value_heads=2),
        )


def test_translation_fails_closed_on_unrouted_gate_nonlinearity() -> None:
    with pytest.raises(HFCausalKindTranslationError, match="quadratic mass"):
        _translate_conv(_short_conv(gate_scale=1.0), carrier_layer_index=None)


def test_translation_fails_closed_when_the_band_is_not_placed() -> None:
    with pytest.raises(HFCausalKindTranslationError, match="assigned lag"):
        _translate_conv(
            _short_conv(gate_scale=0.0),
            rows=_rows(noise=3.0),
            policy=ShortConvToAttentionPolicy(
                minimum_assigned_lag_probability_mass=0.99
            ),
        )


def test_translation_fails_closed_below_the_no_branch_baseline() -> None:
    # Isotropic rows leave no DC direction for the constant Q/K preimage to ride,
    # the band dissolves, and the compiled endpoint is worse than writing nothing
    # at all.  This is the v3 lesson as a gate: the mass floor is lowered here so
    # the endpoint gate is the one under test.
    with pytest.raises(HFCausalKindTranslationError, match="no-branch baseline"):
        _translate_conv(
            _short_conv(gate_scale=0.0),
            rows=_rows(noise=3.0),
            policy=ShortConvToAttentionPolicy(
                minimum_assigned_lag_probability_mass=0.0
            ),
        )


def test_translation_fails_closed_on_partition_overlap() -> None:
    valid, fit, _selection, validation = _partitions()
    with pytest.raises(HFCausalKindTranslationError, match="overlap"):
        _translate_conv(
            _short_conv(gate_scale=0.0),
            valid_token_mask=valid,
            fit_token_mask=fit | validation,
            validation_token_mask=validation,
            selection_token_mask=None,
        )


def test_translation_fails_closed_on_non_finite_evidence() -> None:
    rows = _rows()
    rows[0, 0, 0] = float("nan")
    with pytest.raises(HFCausalKindTranslationError, match="non-finite"):
        _translate_conv(_short_conv(gate_scale=0.0), rows=rows)


# ---------------------------------------------------------------------------
# Direction A: attention -> ShortConv
# ---------------------------------------------------------------------------


def test_band_decomposition_is_row_exact() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 1))
    _valid, fit, _selection, _validation = _partitions()
    decomposition = attention_band_decomposition(
        attention_probabilities=evidence["attention_probabilities"],
        attention_value_sequences=evidence["attention_value_sequences"],
        attention_output_weight=evidence["attention_output_weight"],
        query_heads=evidence["query_heads"],
        key_value_heads=evidence["key_value_heads"],
        head_dim=evidence["head_dim"],
        fir_length=FIR_LENGTH,
        row_mask=fit,
    )
    reconstruction = decomposition.band_delta + decomposition.far_delta
    scale = float(evidence["delta"].abs().max().item())
    assert float((reconstruction - evidence["delta"]).abs().max().item()) < 1e-12 * scale
    diagnostics = decomposition.diagnostics
    assert diagnostics["rho_band"] > 0.95
    assert diagnostics["rho_far"] < 1e-3
    assert diagnostics["band_mass_mean"] > 0.95
    assert 0.0 <= diagnostics["band_mass_minimum"] <= diagnostics["band_mass_maximum"]
    assert diagnostics["band_mass_maximum"] <= 1.0 + 1e-12
    deciles = diagnostics["band_mass_deciles"]
    assert len(deciles) == 11
    assert deciles == sorted(deciles)
    assert deciles[0] == diagnostics["band_mass_minimum"]
    assert deciles[-1] == diagnostics["band_mass_maximum"]


def test_attention_to_short_conv_fits_a_band_concentrated_layer() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 1))
    translation = _translate_attention(evidence, rows=rows)
    report = translation.report
    assert report["schema"] == ATTENTION_TO_SHORT_CONV_SCHEMA
    assert _self_sealed(report)
    assert report["band_decomposition_reconstruction_relative_rmse"] < 1e-9
    assert report["decomposition"]["rho_far"] < 1e-3
    assert report["fit_band_relative_rmse"] < 0.2
    assert report["validation_band_relative_rmse"] < 0.15
    assert report["validation_band_relative_rmse_bf16"] < 0.15
    assert report["band_factorization_residual_relative_energy"] < 0.05
    assert report["remainder_validation_relative_energy"] < 0.02
    assert report["bf16_versus_fp64_endpoint_relative_delta"] < 0.05
    # The conv reaches the class limit: the best *unconstrained* lag-separable
    # linear operator on the same rows does no better, so what is left is the
    # softmax normalization coupling, not a solver failure.
    assert report["fit_band_relative_rmse"] < 1.15 * (
        report["band_lag_separable_linear_floor_fit_relative_rmse"]
    )
    assert report["validation_band_relative_rmse"] <= (
        report["band_lag_separable_linear_floor_validation_relative_rmse"]
    )
    assert report["transported_value_prior"][
        "value_projection_recovery_relative_rmse"
    ] < 1e-4
    # Bounded deterministic variable projection: fixed rounds, guarded monotone.
    solve = report["solve"]
    assert solve["initialization"] in {
        "value_transport",
        "lag_blocked",
        "mean_lag_profile",
    }
    assert solve["fit_objective_monotone_nonincreasing"] is True
    trace = [entry["fit_objective"] for entry in solve["objective_trace"]]
    assert len(trace) == 1 + 5 * solve["rounds"]
    assert trace[-1] < trace[0] * 0.05
    assert len(report["solve_candidates"]) == 3 * 2 * 2 * 2
    assert report["solve_candidates"][report["selected_candidate_index"]][
        "selection_band_relative_rmse"
    ] == min(
        candidate["selection_band_relative_rmse"]
        for candidate in report["solve_candidates"]
    )
    assert report["worst_regularized_condition"] <= max(
        AttentionToShortConvPolicy().condition_caps
    ) * (1.0 + 1e-9)
    assert report["initial_lag_profile"][0] > report["initial_lag_profile"][2]
    assert report["parent_forward_calls"] == 0
    assert report["optimizer_steps"] == 0
    # The compiled branch is a materializable LFM2 ShortConv.
    parameters = translation.parameters
    assert parameters.channel_width == HIDDEN
    assert parameters.fir_length == FIR_LENGTH
    assert tuple(parameters.input_projection_weight.shape) == (3 * HIDDEN, HIDDEN)
    assert tuple(parameters.conv_weight.shape) == (HIDDEN, 1, FIR_LENGTH)
    reference = Lfm2ShortConvParameters(
        in_proj_weight=parameters.input_projection_weight,
        in_proj_bias=parameters.input_projection_bias,
        conv_weight=parameters.conv_weight,
        conv_bias=parameters.conv_bias,
        out_proj_weight=parameters.output_projection_weight,
        out_proj_bias=parameters.output_projection_bias,
    )
    assert torch.equal(parameters.forward(rows), lfm2_short_conv_reference(reference, rows))


def test_attention_to_short_conv_measures_a_non_separable_band() -> None:
    rows = _rows(noise=1.0)
    separable = _translate_attention(
        _relative_distance_attention(rows, lags=(0, 1)), rows=rows
    )
    coupled = _translate_attention(
        _relative_distance_attention(rows, lags=(0, 1), content_coupling=0.35),
        rows=rows,
    )
    # Content coupling keeps the mass in the band but makes the in-band weights
    # depend jointly on the query and key rows, which the per-channel rank-one
    # factorization provably cannot represent.
    assert coupled.report["decomposition"]["rho_band"] > 0.9
    assert coupled.report["band_factorization_residual_relative_energy"] > (
        separable.report["band_factorization_residual_relative_energy"]
    )


def test_attention_to_short_conv_fails_closed_on_unrouted_far_mass() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 9))
    with pytest.raises(HFCausalKindTranslationError, match="rho_far"):
        _translate_attention(evidence, rows=rows, carrier_layer_index=None)


def test_attention_to_short_conv_routes_far_mass_to_a_carrier() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 9))
    translation = _translate_attention(evidence, rows=rows, carrier_layer_index=5)
    report = translation.report
    assert report["decomposition"]["rho_far"] > 0.1
    assert report["remainder_validation_relative_energy"] > 0.1
    bundle = seal_kind_translation_remainder(
        direction="attention_to_short_conv",
        source_layer_index=report["source_layer_index"],
        carrier_layer_index=report["carrier_layer_index"],
        remainder=translation.remainder,
        source_report_sha256=report["report_sha256"],
        evidence_cache_key="toy-cache",
        carrier_action_id="toy-carrier-action",
        diagnostics={
            "rho_band": report["decomposition"]["rho_band"],
            "rho_far": report["decomposition"]["rho_far"],
        },
    )
    assert bundle.carrier_layer_index == 5
    assert bundle.report["remainder_sha256"] == report["remainder_sha256"]
    assert _self_sealed(bundle.report)
    carrier_delta = torch.randn(BATCH, SEQUENCE, HIDDEN, dtype=torch.float64)
    routed = carrier_endpoint_target_with_remainder(carrier_delta, (bundle,))
    assert torch.allclose(routed, carrier_delta + translation.remainder)


def test_attention_to_short_conv_fails_closed_on_inconsistent_evidence() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 1))
    evidence["delta"] = evidence["delta"] * 1.5
    with pytest.raises(HFCausalKindTranslationError, match="reconstruct"):
        _translate_attention(evidence, rows=rows)


def test_attention_to_short_conv_fails_closed_on_partition_overlap() -> None:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 1))
    _valid, fit, selection, _validation = _partitions()
    with pytest.raises(HFCausalKindTranslationError, match="overlap"):
        _translate_attention(evidence, rows=rows, validation_token_mask=fit | selection)


# ---------------------------------------------------------------------------
# Routing rule and shared evidence helpers
# ---------------------------------------------------------------------------


def test_carrier_routing_checkpoint_rejects_a_worse_checkpoint() -> None:
    _valid, _fit, _selection, validation = _partitions()
    target = torch.randn(BATCH, SEQUENCE, HIDDEN, dtype=torch.float64)
    metrics = evaluate_carrier_routing_checkpoint(
        checkpoint_target=target,
        routing_on_prediction=target * 0.99,
        routing_off_prediction=target * 0.5,
        validation_token_mask=validation,
    )
    assert metrics["routing_improves_checkpoint"] is True
    assert metrics["routing_on_checkpoint_relative_rmse"] < (
        metrics["routing_off_checkpoint_relative_rmse"]
    )
    assert _self_sealed(metrics)
    with pytest.raises(HFCausalKindTranslationError, match="fails closed"):
        evaluate_carrier_routing_checkpoint(
            checkpoint_target=target,
            routing_on_prediction=torch.zeros_like(target),
            routing_off_prediction=target * 0.99,
            validation_token_mask=validation,
        )


def test_remainder_routing_requires_a_downstream_carrier() -> None:
    remainder = torch.zeros(BATCH, SEQUENCE, HIDDEN, dtype=torch.float64)
    with pytest.raises(HFCausalKindTranslationError, match="requires a carrier"):
        seal_kind_translation_remainder(
            direction="attention_to_short_conv",
            source_layer_index=1,
            carrier_layer_index=None,
            remainder=remainder,
            source_report_sha256="0" * 64,
        )
    with pytest.raises(HFCausalKindTranslationError, match="downstream"):
        seal_kind_translation_remainder(
            direction="attention_to_short_conv",
            source_layer_index=4,
            carrier_layer_index=4,
            remainder=remainder,
            source_report_sha256="0" * 64,
        )


def _carrier_attention(rows: Any, *, seed: int = 41) -> dict[str, Any]:
    """A surviving attention layer, recompiled by the branch kernel unchanged."""

    generator = torch.Generator().manual_seed(seed)
    topology = _topology(query_heads=2, key_value_heads=2)
    width = topology.query_heads * topology.head_dim

    def projection() -> Any:
        return torch.randn(width, HIDDEN, generator=generator, dtype=torch.float64) / 3.0

    query_weight, key_weight, value_weight = projection(), projection(), projection()
    output_weight = (
        torch.randn(HIDDEN, width, generator=generator, dtype=torch.float64) / 3.0
    )
    query, key, value = (
        rows @ query_weight.T,
        rows @ key_weight.T,
        rows @ value_weight.T,
    )

    def heads(value_rows: Any) -> Any:
        return (
            value_rows.reshape(
                BATCH, SEQUENCE, topology.key_value_heads, topology.head_dim
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    _scores, probabilities = causal_attention_probabilities(
        apply_rope(heads(query), _positions(), theta=topology.rope_theta),
        apply_rope(heads(key), _positions(), theta=topology.rope_theta),
        attention_mask=_attention_mask(),
        scaling=topology.attention_scaling,
    )
    delta = (
        torch.matmul(probabilities, heads(value))
        .permute(0, 2, 1, 3)
        .reshape(BATCH, SEQUENCE, width)
        @ output_weight.T
    )
    return {
        "topology": topology,
        "query": query,
        "key": key,
        "value": value,
        "delta": delta,
    }


def _compile_carrier(carrier: dict[str, Any], rows: Any, target: Any) -> Any:
    valid, fit, _selection, validation = _partitions()
    fold = compile_hf_causal_gqa_attention_fold(
        rows,
        source_query_sequences=carrier["query"],
        source_key_sequences=carrier["key"],
        source_value_sequences=carrier["value"],
        target_residual_delta_sequences=target,
        attention_mask=_attention_mask(),
        position_ids=_positions(),
        valid_token_mask=valid,
        fit_token_mask=fit,
        validation_token_mask=validation,
        source_topology=carrier["topology"],
        target_topology=carrier["topology"],
    )
    prediction, _scores, _probabilities = fold.replay(
        rows, attention_mask=_attention_mask(), position_ids=_positions()
    )
    return prediction


def _routing_checkpoint(
    *, lags: tuple[int, ...], content_coupling: float, fail_closed: bool
) -> Any:
    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(
        rows, lags=lags, content_coupling=content_coupling
    )
    converted = _translate_attention(evidence, rows=rows, carrier_layer_index=5)
    bundle = seal_kind_translation_remainder(
        direction="attention_to_short_conv",
        source_layer_index=converted.report["source_layer_index"],
        carrier_layer_index=converted.report["carrier_layer_index"],
        remainder=converted.remainder,
        source_report_sha256=converted.report["report_sha256"],
    )
    carrier = _carrier_attention(rows)
    routed_target = carrier_endpoint_target_with_remainder(carrier["delta"], (bundle,))
    conv_prediction = converted.parameters.forward(rows)
    _valid, _fit, _selection, validation = _partitions()
    return evaluate_carrier_routing_checkpoint(
        # The checkpoint quantity is the cumulative residual the downstream
        # stack consumes across both layers, in parent coordinates.
        checkpoint_target=evidence["delta"] + carrier["delta"],
        routing_on_prediction=conv_prediction
        + _compile_carrier(carrier, rows, routed_target),
        routing_off_prediction=conv_prediction
        + _compile_carrier(carrier, rows, carrier["delta"]),
        validation_token_mask=validation,
        fail_closed=fail_closed,
    )


def test_routing_a_remainder_into_a_real_carrier_is_scored_at_the_checkpoint() -> None:
    metrics = _routing_checkpoint(
        lags=(0, 4), content_coupling=0.35, fail_closed=True
    )
    assert metrics["routing_improves_checkpoint"] is True
    assert metrics["routing_on_checkpoint_relative_rmse"] < (
        metrics["routing_off_checkpoint_relative_rmse"]
    )
    assert metrics["checkpoint_validation_rows"] > 0
    assert _self_sealed(metrics)


def test_a_carrier_that_cannot_carry_the_remainder_fails_the_conversion_closed() -> None:
    # The design names this risk explicitly: the carrier is fitted from its own
    # state, and nothing guarantees that state predicts another layer's far
    # mass.  The honest outcome is a refused conversion, not a silent loss.
    observed = _routing_checkpoint(lags=(0, 9), content_coupling=0.0, fail_closed=False)
    assert observed["routing_improves_checkpoint"] is False
    with pytest.raises(HFCausalKindTranslationError, match="fails closed"):
        _routing_checkpoint(lags=(0, 9), content_coupling=0.0, fail_closed=True)


def test_attention_evidence_from_projections_replays_kernel_probabilities() -> None:
    topology = _topology(query_heads=4, key_value_heads=2)
    generator = torch.Generator().manual_seed(19)
    shapes = (
        topology.query_heads * topology.head_dim,
        topology.key_value_heads * topology.head_dim,
    )
    query = torch.randn(BATCH, SEQUENCE, shapes[0], generator=generator, dtype=torch.float64)
    key = torch.randn(BATCH, SEQUENCE, shapes[1], generator=generator, dtype=torch.float64)
    value = torch.randn(BATCH, SEQUENCE, shapes[1], generator=generator, dtype=torch.float64)
    probabilities, values = attention_evidence_from_projections(
        query_projection_sequences=query,
        key_projection_sequences=key,
        value_projection_sequences=value,
        topology=topology,
        attention_mask=_attention_mask(),
        position_ids=_positions(),
    )
    assert tuple(probabilities.shape) == (
        BATCH,
        topology.query_heads,
        SEQUENCE,
        SEQUENCE,
    )
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(BATCH, topology.query_heads, SEQUENCE, dtype=torch.float64),
    )
    # Causality: nothing above the diagonal.
    upper = torch.triu(torch.ones(SEQUENCE, SEQUENCE, dtype=torch.bool), diagonal=1)
    assert float(probabilities[:, :, upper].abs().max().item()) == 0.0
    assert torch.equal(values, value)


def test_translations_are_byte_identical_across_builds() -> None:
    first = _translate_conv(_short_conv(gate_scale=1.0), carrier_layer_index=6)
    second = _translate_conv(_short_conv(gate_scale=1.0), carrier_layer_index=6)
    assert first.report["report_sha256"] == second.report["report_sha256"]
    assert first.report["attention_kernel_report_sha256"] == (
        second.report["attention_kernel_report_sha256"]
    )
    for name in ("query_weight", "key_weight", "value_weight", "output_weight"):
        assert torch.equal(getattr(first.fold, name), getattr(second.fold, name))
    assert torch.equal(first.remainder, second.remainder)

    rows = _rows(noise=1.0)
    evidence = _relative_distance_attention(rows, lags=(0, 1))
    third = _translate_attention(evidence, rows=rows)
    fourth = _translate_attention(evidence, rows=rows)
    assert third.report["report_sha256"] == fourth.report["report_sha256"]
    assert third.report["short_conv"] == fourth.report["short_conv"]
    assert torch.equal(
        third.parameters.input_projection_weight,
        fourth.parameters.input_projection_weight,
    )
    assert torch.equal(third.remainder, fourth.remainder)
