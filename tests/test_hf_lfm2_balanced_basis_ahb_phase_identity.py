from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_balanced_circuit_basis import (  # noqa: E402
    PrecomputedDownstreamObservability,
    _tensor_sha256,
)
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    CircuitKind,
    canonical_sha256,
)
import neural_forge.hf_lfm2_balanced_basis_phase_cache as phase_cache  # noqa: E402
from neural_forge.hf_lfm2_autoregressive_hankel_observability import (  # noqa: E402
    LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROVENANCE_SCHEMA,
    Lfm2AutoregressiveHankelObservabilityResult,
    build_lfm2_autoregressive_hankel_observability,
)
from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    accumulate_lfm2_ahb_trajectory,
)


def _ahb_evidence(scale: float) -> PrecomputedDownstreamObservability:
    gram = torch.diag(
        torch.tensor([scale, scale + 1.0], dtype=torch.float64)
    ).contiguous()
    gram_sha256 = _tensor_sha256(gram)
    provenance_body = {
        "schema": LFM2_AUTOREGRESSIVE_HANKEL_OBSERVABILITY_PROVENANCE_SCHEMA,
        "method": "test-future-causal-geometry",
        "source_statistics_receipt_sha256s": ["a" * 64],
        "source_batch_receipt_sha256s": ["b" * 64],
        "source_tensor_payload_sha256s": ["c" * 64],
        "boundary_names": ["layer.000.residual.pre"],
        "future_lags": [0, 1],
        "configuration": {
            "lag_zero_used_for_selection": False,
            "cross_trajectory_pairs_created": False,
        },
        "boundary_gram_sha256s": [gram_sha256],
        "global_gram_sha256": gram_sha256,
        "fit_only_selection_evidence": True,
        "parent_forward_calls": 0,
        "child_forward_calls": 0,
        "trace_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "autograd_used": False,
    }
    provenance = {
        **provenance_body,
        "provenance_sha256": canonical_sha256(provenance_body),
    }
    return PrecomputedDownstreamObservability(
        evidence_id=f"test-ahb-{scale}",
        node_id="output.lm_head",
        circuit_kind=CircuitKind.OUTPUT_HEAD,
        role="autoregressive_hankel_future_causal_geometry",
        gram=gram,
        provenance=provenance,
        interface_ids=("text.logits.output",),
    )


def _ahb_result() -> Lfm2AutoregressiveHankelObservabilityResult:
    nuisance = torch.tensor([10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0])
    signal = torch.tensor([0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.3, 0.3, 0.0])
    rows = torch.stack((nuisance, signal), dim=1)
    boundaries = torch.stack(
        (rows, rows * torch.tensor([1.0, 2.0])),
        dim=0,
    )
    behavior = torch.zeros_like(rows)
    behavior[1:, 1] = signal[:-1]
    statistics = accumulate_lfm2_ahb_trajectory(
        boundaries,
        behavior,
        future_lags=(0, 1),
        source_receipt_sha256="e" * 64,
    )
    return build_lfm2_autoregressive_hankel_observability(
        (statistics,),
        boundary_names=(
            "layer.000.residual.pre",
            "layer.000.residual.post_layer",
        ),
    )


def _request(
    evidence: (
        Lfm2AutoregressiveHankelObservabilityResult
        | PrecomputedDownstreamObservability
        | None
    ),
) -> phase_cache.Lfm2BalancedBasisPhaseBuildRequest:
    return phase_cache.Lfm2BalancedBasisPhaseBuildRequest(
        adapter_source_sha256="d" * 64,
        autoregressive_hankel_observability=evidence,
    )


def test_ahb_evidence_is_exact_phase_identity_forwarded_and_reauthenticated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_evidence = _ahb_evidence(1.0)
    second_evidence = _ahb_evidence(2.0)
    first = _request(first_evidence)
    second = _request(second_evidence)

    absent_config = _request(None).as_basis_config()
    assert "autoregressive_hankel_observability" not in absent_config

    first_config = first.as_basis_config()
    second_config = second.as_basis_config()
    first_binding = first_config["autoregressive_hankel_observability"]
    first_document = first_evidence.as_dict()
    assert first_binding == {
        "schema": "neural-forge-lfm2-balanced-basis-ahb-binding-v1",
        "input_kind": "sealed-precomputed-evidence",
        "evidence_sha256": first_document["evidence_sha256"],
        "provenance_sha256": first_document["provenance"]["provenance_sha256"],
        "gram_sha256": first_document["gram_sha256"],
        "result_report_sha256": None,
    }
    assert canonical_sha256(first_config) != canonical_sha256(second_config)
    assert (
        first_binding["evidence_sha256"]
        != second_config["autoregressive_hankel_observability"]["evidence_sha256"]
    )
    assert (
        first_binding["provenance_sha256"]
        != second_config["autoregressive_hankel_observability"]["provenance_sha256"]
    )
    assert (
        first_binding["gram_sha256"]
        != second_config["autoregressive_hankel_observability"]["gram_sha256"]
    )

    result_evidence = _ahb_result()
    result_request = _request(result_evidence)
    result_binding = result_request.as_basis_config()[
        "autoregressive_hankel_observability"
    ]
    assert result_binding["input_kind"] == "sealed-result"
    assert (
        result_binding["result_report_sha256"]
        == result_evidence.report["report_sha256"]
    )

    marker = object()
    forwarded: dict[str, object] = {}

    def cold_builder(*_args, **kwargs):
        forwarded.update(kwargs)
        return marker

    monkeypatch.setattr(
        phase_cache,
        "build_lfm2_circuit_balanced_basis",
        cold_builder,
    )
    assert (
        result_request.build(
            object(),
            object(),
            1,
            circuit_graph=object(),
            selective_session=None,
        )
        is marker
    )
    assert forwarded["autoregressive_hankel_observability"] is result_evidence

    with torch.no_grad():
        result_evidence.gram[0, 0].add_(0.25)
    with pytest.raises(
        phase_cache.Lfm2BalancedBasisPhaseCacheError,
        match="failed reauthentication",
    ):
        result_request.as_basis_config()
