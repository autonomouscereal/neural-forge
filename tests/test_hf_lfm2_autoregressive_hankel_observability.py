from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_lfm2_autoregressive_hankel_observability import (  # noqa: E402
    Lfm2AutoregressiveHankelObservabilityError,
    build_lfm2_autoregressive_hankel_observability,
)
from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    accumulate_lfm2_ahb_trajectory,
)


def _future_predictive_case(source_digest: str, *, nuisance_scale: float = 1.0):
    nuisance = nuisance_scale * torch.tensor(
        [10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0]
    )
    signal = torch.tensor(
        [0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.3, 0.3, 0.0]
    )
    rows = torch.stack((nuisance, signal), dim=1)
    boundaries = torch.stack((rows, rows * torch.tensor([1.0, 2.0])), dim=0)
    behavior = torch.zeros_like(rows)
    behavior[1:, 1] = signal[:-1]
    return accumulate_lfm2_ahb_trajectory(
        boundaries,
        behavior,
        future_lags=(0, 1),
        source_receipt_sha256=source_digest,
    )


def test_future_causal_geometry_recovers_low_variance_predictive_direction() -> None:
    statistics = _future_predictive_case("a" * 64)
    static_covariance = statistics.state_second_moment[0, 0]
    static_values, static_vectors = torch.linalg.eigh(static_covariance)
    static_top = static_vectors[:, int(torch.argmax(static_values).item())]
    assert abs(float(static_top[0].item())) > 0.999

    result = build_lfm2_autoregressive_hankel_observability(
        (statistics,),
        boundary_names=(
            "layer.000.residual.pre",
            "layer.000.residual.post_layer",
        ),
    )

    values, vectors = torch.linalg.eigh(result.gram)
    future_top = vectors[:, int(torch.argmax(values).item())]
    assert abs(float(future_top[1].item())) > 0.999
    assert abs(float(future_top[0].item())) < 1e-8
    assert torch.allclose(result.gram, result.gram.T, rtol=0.0, atol=0.0)
    assert float(torch.trace(result.gram).item()) > 0.0
    assert result.evidence.verify() is True
    assert result.report["configuration"]["lag_zero_used_for_selection"] is False
    assert result.report["empirical_on_trajectory_not_action_complete_bisimulation"]
    assert result.report["training_steps"] == 0
    assert result.report["model_forward_calls"] == 0

    rescaled = build_lfm2_autoregressive_hankel_observability(
        (_future_predictive_case("f" * 64, nuisance_scale=1_000.0),),
        boundary_names=(
            "layer.000.residual.pre",
            "layer.000.residual.post_layer",
        ),
    )
    scaled_values, scaled_vectors = torch.linalg.eigh(rescaled.gram)
    scaled_top = scaled_vectors[:, int(torch.argmax(scaled_values).item())]
    assert abs(float(scaled_top[1].item())) > 0.999
    assert abs(float(scaled_top[0].item())) < 1e-8


def test_pooling_centers_each_trajectory_and_authenticates_all_sources() -> None:
    first = _future_predictive_case("b" * 64)
    second = _future_predictive_case("c" * 64)
    result = build_lfm2_autoregressive_hankel_observability(
        (first, second),
        boundary_names=("input", "post_layer"),
    )

    assert result.report["source_trajectory_count"] == 2
    assert result.report["configuration"]["pooled_pair_centering"] is True
    assert all(
        boundary["lags"][0]["pooled_centered_degrees"] == 15
        for boundary in result.report["boundaries"]
    )
    provenance = result.evidence.provenance
    assert len(provenance["source_statistics_receipt_sha256s"]) == 2
    assert len(provenance["source_tensor_payload_sha256s"]) == 2


def test_duplicate_or_mutated_statistics_fail_closed() -> None:
    statistics = _future_predictive_case("d" * 64)
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityError,
        match="must not repeat",
    ):
        build_lfm2_autoregressive_hankel_observability(
            (statistics, statistics),
            boundary_names=("input", "post_layer"),
        )

    with torch.inference_mode():
        statistics.state_sum[0, 0, 0] += 1.0
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityError,
        match="changed after sealing",
    ):
        build_lfm2_autoregressive_hankel_observability(
            (statistics,),
            boundary_names=("input", "post_layer"),
        )


def test_unbound_statistics_and_mismatched_boundaries_are_rejected() -> None:
    rows = torch.arange(24, dtype=torch.float64).reshape(2, 6, 2)
    behavior = torch.arange(12, dtype=torch.float64).reshape(6, 2)
    unbound = accumulate_lfm2_ahb_trajectory(
        rows,
        behavior,
        future_lags=(0, 1),
    )
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityError,
        match="not bound",
    ):
        build_lfm2_autoregressive_hankel_observability(
            (unbound,),
            boundary_names=("input", "post_layer"),
        )

    bound = _future_predictive_case("e" * 64)
    with pytest.raises(
        Lfm2AutoregressiveHankelObservabilityError,
        match="uniquely name every",
    ):
        build_lfm2_autoregressive_hankel_observability(
            (bound,),
            boundary_names=("only-one",),
        )
