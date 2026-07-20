from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.hf_cpcf_boundary_chart import (  # noqa: E402
    fit_cpcf_boundary_chart,
)
from neural_forge.hf_cpcf_sequential_boundary_transport import (  # noqa: E402
    CpcfSequentialBoundaryTransportError,
    fit_cpcf_sequential_downstream_chart_transport,
)


def _charts(seed: int = 20260720):
    generator = torch.Generator().manual_seed(seed)
    source_width = 7
    target_width = 3
    fit = torch.randn(
        72,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    validation = torch.randn(
        24,
        source_width,
        generator=generator,
        dtype=torch.float64,
    )
    operator_future = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    layer_future = torch.randn(
        source_width,
        target_width,
        generator=generator,
        dtype=torch.float64,
    )
    upstream = fit_cpcf_boundary_chart(
        "layer.002.residual.post_operator",
        fit,
        fit @ operator_future,
        target_width=target_width,
        validation_parent_states=validation,
        validation_future_coordinates=validation @ operator_future,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )
    downstream = fit_cpcf_boundary_chart(
        "layer.002.residual.post_layer",
        fit,
        fit @ layer_future,
        target_width=target_width,
        validation_parent_states=validation,
        validation_future_coordinates=validation @ layer_future,
        state_reconstruction_weight=0.0,
        minimum_predictive_energy_fraction=0.99,
        minimum_validation_block_retention=0.99,
    )
    return generator, upstream, downstream


def test_sequential_transport_removes_downstream_gauge_error_after_replay():
    generator, upstream, downstream = _charts()
    fit_parent = torch.randn(
        54,
        downstream.source_width,
        generator=generator,
        dtype=torch.float64,
    )
    raw = torch.randn(
        downstream.target_width,
        downstream.target_width,
        generator=generator,
        dtype=torch.float64,
    )
    expected_rotation, _ = torch.linalg.qr(raw)
    actual_post_operator = downstream.encode(fit_parent) @ expected_rotation
    selection_parent = torch.randn(
        19,
        downstream.source_width,
        generator=generator,
        dtype=torch.float64,
    )
    selection_actual = downstream.encode(selection_parent) @ expected_rotation
    fit_weights = torch.linspace(0.4, 1.8, fit_parent.shape[0])
    selection_weights = torch.linspace(0.7, 1.3, selection_parent.shape[0])
    upstream_payload_before = (
        upstream.mapping.encoder.clone(),
        upstream.mapping.decoder.clone(),
        {
            name: value.clone()
            for name, value in upstream.mapping.feature_predictors.items()
        },
    )

    result = fit_cpcf_sequential_downstream_chart_transport(
        upstream,
        downstream,
        fit_parent_downstream_states=fit_parent,
        fit_actual_child_upstream_rows=actual_post_operator,
        fit_provenance="h776/layer.002/operator-replay/fit",
        fit_weights=fit_weights,
        selection_parent_downstream_states=selection_parent,
        selection_actual_child_upstream_rows=selection_actual,
        selection_weights=selection_weights,
        selection_provenance=(
            "h776/layer.002/operator-replay/selection"
        ),
    )

    assert result.upstream_chart is upstream
    assert result.source_downstream_chart is downstream
    assert result.downstream_chart is not downstream
    assert result.report["fit_required_write_rms_before"] > 0.1
    assert result.report["fit_required_write_rms_after"] < 1e-12
    selection = result.report["selection_evaluation"]
    assert selection["required_write_rms_before"] > 0.1
    assert selection["required_write_rms_after"] < 1e-12
    assert selection["used_by_solver"] is False
    assert selection["read_after_transport_freeze"] is True
    assert result.report["validation_rows_read"] == 0
    assert result.report["one_closed_form_svd"] is True
    assert result.report["training_steps"] == 0
    assert result.report["optimizer_steps"] == 0
    assert result.report["backward_calls"] == 0
    assert result.report["parameters_pruned"] == 0
    assert result.report["parameters_quantized"] == 0
    assert result.report["parent_sidecar_required"] is False
    assert result.report["automatic_retry_or_retuning_performed"] is False

    torch.testing.assert_close(
        upstream.mapping.encoder,
        upstream_payload_before[0],
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        upstream.mapping.decoder,
        upstream_payload_before[1],
        atol=0.0,
        rtol=0.0,
    )
    for name, value in upstream_payload_before[2].items():
        torch.testing.assert_close(
            upstream.mapping.feature_predictors[name],
            value,
            atol=0.0,
            rtol=0.0,
        )

    probe = torch.randn(
        11,
        downstream.source_width,
        generator=generator,
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        result.downstream_chart.mapping.decode(
            result.downstream_chart.mapping.encode(probe)
        ),
        downstream.mapping.decode(downstream.mapping.encode(probe)),
        atol=1e-12,
        rtol=1e-12,
    )
    for name in downstream.mapping.feature_predictors:
        torch.testing.assert_close(
            result.downstream_chart.mapping.predict_feature(name, probe),
            downstream.mapping.predict_feature(name, probe),
            atol=1e-12,
            rtol=1e-12,
        )


def test_sequential_transport_is_deterministic_and_fit_only():
    generator, upstream, downstream = _charts(seed=20260721)
    fit_parent = torch.randn(
        41,
        downstream.source_width,
        generator=generator,
        dtype=torch.float64,
    )
    raw = torch.randn(
        downstream.target_width,
        downstream.target_width,
        generator=generator,
        dtype=torch.float64,
    )
    rotation, _ = torch.linalg.qr(raw)
    actual = downstream.encode(fit_parent) @ rotation

    first = fit_cpcf_sequential_downstream_chart_transport(
        upstream,
        downstream,
        fit_parent_downstream_states=fit_parent,
        fit_actual_child_upstream_rows=actual,
        fit_provenance="deterministic-fit-20260721",
    )
    second = fit_cpcf_sequential_downstream_chart_transport(
        upstream,
        downstream,
        fit_parent_downstream_states=fit_parent.clone(),
        fit_actual_child_upstream_rows=actual.clone(),
        fit_provenance="deterministic-fit-20260721",
    )

    torch.testing.assert_close(
        first.downstream_transport.rotation,
        second.downstream_transport.rotation,
        atol=0.0,
        rtol=0.0,
    )
    assert first.report["report_sha256"] == second.report["report_sha256"]
    assert first.report["selection_evaluation"] is None
    assert first.report["selection_read_after_transport_freeze"] is None
    assert first.report["validation_argument_supported"] is False


def test_sequential_transport_rejects_leaky_or_incomplete_evidence():
    generator, upstream, downstream = _charts(seed=20260722)
    parent = torch.randn(
        22,
        downstream.source_width,
        generator=generator,
        dtype=torch.float64,
    )
    actual = downstream.encode(parent)

    with pytest.raises(
        CpcfSequentialBoundaryTransportError,
        match="together",
    ):
        fit_cpcf_sequential_downstream_chart_transport(
            upstream,
            downstream,
            fit_parent_downstream_states=parent,
            fit_actual_child_upstream_rows=actual,
            fit_provenance="fit",
            selection_parent_downstream_states=parent.clone(),
        )

    with pytest.raises(
        CpcfSequentialBoundaryTransportError,
        match="share tensor storage",
    ):
        fit_cpcf_sequential_downstream_chart_transport(
            upstream,
            downstream,
            fit_parent_downstream_states=parent,
            fit_actual_child_upstream_rows=actual,
            fit_provenance="fit",
            selection_parent_downstream_states=parent,
            selection_actual_child_upstream_rows=actual.clone(),
            selection_weights=torch.ones(parent.shape[0]),
            selection_provenance="selection",
        )

    with pytest.raises(
        CpcfSequentialBoundaryTransportError,
        match="must differ",
    ):
        fit_cpcf_sequential_downstream_chart_transport(
            upstream,
            downstream,
            fit_parent_downstream_states=parent,
            fit_actual_child_upstream_rows=actual,
            fit_provenance="same",
            selection_parent_downstream_states=parent.clone(),
            selection_actual_child_upstream_rows=actual.clone(),
            selection_weights=torch.ones(parent.shape[0]),
            selection_provenance="same",
        )

    with pytest.raises(
        CpcfSequentialBoundaryTransportError,
        match="positive",
    ):
        fit_cpcf_sequential_downstream_chart_transport(
            upstream,
            downstream,
            fit_parent_downstream_states=parent,
            fit_actual_child_upstream_rows=actual,
            fit_provenance="fit",
            fit_weights=torch.zeros(parent.shape[0]),
        )
