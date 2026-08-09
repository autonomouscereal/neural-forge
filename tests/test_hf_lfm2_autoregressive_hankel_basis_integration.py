from __future__ import annotations

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from neural_forge.dynamic_balanced_circuit_basis import (  # noqa: E402
    PrecomputedDownstreamObservability,
)
from neural_forge.dynamic_circuit_ir import (  # noqa: E402
    CircuitKind,
    canonical_sha256,
)
from neural_forge.hf_lfm2_autoregressive_hankel_observability import (  # noqa: E402
    build_lfm2_autoregressive_hankel_observability,
)
from neural_forge.hf_lfm2_autoregressive_hankel_statistics import (  # noqa: E402
    accumulate_lfm2_ahb_trajectory,
)
import neural_forge.hf_lfm2_balanced_circuit_basis as balanced  # noqa: E402


def _ahb_result():
    generator = torch.Generator().manual_seed(887)
    row_count = 24
    source_width = 8
    input_rows = torch.randn(
        row_count, source_width, generator=generator, dtype=torch.float64
    )
    transition = torch.diag(
        torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 1.2, 1.4, 1.6, 1.8],
            dtype=torch.float64,
        )
    )
    boundaries = torch.stack(
        (
            input_rows,
            input_rows @ transition,
            input_rows @ transition @ transition,
        )
    )
    behavior = torch.zeros_like(input_rows)
    behavior[1:] = boundaries[2, :-1]
    statistics = accumulate_lfm2_ahb_trajectory(
        boundaries,
        behavior,
        future_lags=(0, 1, 2),
        source_receipt_sha256="a" * 64,
    )
    return build_lfm2_autoregressive_hankel_observability(
        (statistics,),
        boundary_names=(
            "layer.000.residual.pre",
            "layer.000.residual.post_layer",
            "layer.001.residual.post_layer",
        ),
    )


class _FakeFingerprint:
    identity_sha256 = "1" * 64

    @staticmethod
    def as_dict() -> dict[str, object]:
        return {
            "schema": "test-parent-fingerprint-v1",
            "identity_sha256": _FakeFingerprint.identity_sha256,
        }


class _FakeInventory:
    source_width = 8
    operators: tuple[object, ...] = ()
    manifest = {"inventory_sha256": "2" * 64}

    @staticmethod
    def as_dict() -> dict[str, object]:
        return {
            "schema": "test-inventory-v1",
            "source_width": 8,
            "inventory_sha256": "2" * 64,
        }


class _FakeGraph:
    family = "lfm2"
    fingerprint = "3" * 64


class _FakeBasisResult:
    def __init__(
        self,
        precomputed: tuple[PrecomputedDownstreamObservability, ...],
    ) -> None:
        self.basis = torch.eye(8, dtype=torch.float64)[:, :4].contiguous()
        self.balanced_encoder = self.basis.T.contiguous()
        self.balanced_decoder = self.basis
        identity = torch.eye(8, dtype=torch.float64)
        self.full_weight_gram = identity
        self.controllability_gram = identity
        self.observability_gram = identity + sum(
            (value.gram for value in precomputed),
            start=torch.zeros_like(identity),
        )
        self.heldout_gram = identity
        self.source_width = 8
        self.target_width = 4
        self.map_sha256 = "4" * 64

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "test-balanced-basis-result-v1",
            "hashes": {"basis_sha256": self.map_sha256},
        }


class _FakeCache:
    cache_key = "5" * 64
    corpus_fingerprint = "6" * 64
    parent_fingerprint = _FakeFingerprint.identity_sha256
    batch_count = 4

    def verify(self):
        return self


def _records():
    return tuple(
        SimpleNamespace(
            index=index,
            record_id=f"record-{index}",
            phase_count=3,
            decode_phase_count=2,
            location_count=18,
        )
        for index in range(4)
    )


def _install_outer_builder_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[PrecomputedDownstreamObservability, ...]]:
    records = _records()
    captured: list[tuple[PrecomputedDownstreamObservability, ...]] = []
    monkeypatch.setattr(balanced, "BalancedCircuitBasisResult", _FakeBasisResult)
    monkeypatch.setattr(balanced, "Lfm2WeightCompleteInventory", _FakeInventory)
    monkeypatch.setattr(balanced, "ModelCircuitGraph", _FakeGraph)
    monkeypatch.setattr(
        balanced,
        "fingerprint_lfm2_parent",
        lambda _parent: _FakeFingerprint(),
    )
    monkeypatch.setattr(
        balanced,
        "inventory_lfm2_weight_complete_operators",
        lambda _parent: _FakeInventory(),
    )
    monkeypatch.setattr(
        balanced,
        "_load_cache_records",
        lambda *_args, **_kwargs: records,
    )
    monkeypatch.setattr(
        balanced,
        "_split_records",
        lambda *_args, **_kwargs: (
            records[:3],
            records[3:],
            {
                "schema": "test-whole-record-split-v1",
                "fit_record_count": 3,
                "heldout_record_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        balanced,
        "_snapshot_blocks",
        lambda *_args, **_kwargs: tuple(object() for _ in range(18)),
    )
    monkeypatch.setattr(
        balanced,
        "_build_observability",
        lambda *_args, **_kwargs: (
            SimpleNamespace(circuit_kind=CircuitKind.OUTPUT_HEAD.value),
        ),
    )

    def fake_generic_builder(*_args, **kwargs):
        lanes = kwargs["precomputed_downstream_observability"]
        captured.append(lanes)
        return _FakeBasisResult(lanes)

    monkeypatch.setattr(
        balanced,
        "build_balanced_circuit_basis",
        fake_generic_builder,
    )
    return captured


def test_ahb_evidence_reaches_generic_builder_and_result_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _install_outer_builder_fixture(monkeypatch)
    parent = torch.nn.Linear(1, 1)
    parent.config = SimpleNamespace(num_hidden_layers=2, vocab_size=31)
    ahb = _ahb_result()

    baseline = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        _FakeCache(),
        4,
        circuit_graph=_FakeGraph(),
    )
    integrated = balanced.build_lfm2_circuit_balanced_basis(
        parent,
        _FakeCache(),
        4,
        circuit_graph=_FakeGraph(),
        autoregressive_hankel_observability=ahb,
    )

    assert captured == [(), (ahb.evidence,)]
    assert not torch.equal(
        baseline.observability_gram,
        integrated.observability_gram,
    )
    report = integrated.as_dict()
    integration = report["autoregressive_hankel_observability"]
    integration_body = dict(integration)
    claimed_sha256 = integration_body.pop("report_sha256")
    assert claimed_sha256 == canonical_sha256(integration_body)
    assert integration["input_kind"] == "sealed-result"
    assert integration["evidence"] == ahb.evidence.as_dict()
    assert integration["source_result"] == ahb.as_dict()
    assert integration["fit_only_selection_evidence"] is True
    assert integration["precomputed_gram_consumed_directly"] is True
    assert integration["parent_forward_calls"] == 0
    assert integration["child_forward_calls"] == 0
    assert integration["training_steps"] == 0
    assert integration["backward_calls"] == 0
    assert integration["optimizer_steps"] == 0
    assert integration["autograd_used"] is False
    assert report["hashes"]["autoregressive_hankel_observability_sha256"] == (
        ahb.evidence.evidence_sha256
    )
    assert "plus-autoregressive-hankel-observability" in report["method"]
    assert "autoregressive_hankel_observability" not in baseline.as_dict()


def test_ahb_joins_streaming_lane_and_fails_closed() -> None:
    ahb = _ahb_result()
    provenance_body = {"schema": "test-streaming-margin-provenance-v1"}
    margin = PrecomputedDownstreamObservability(
        evidence_id="interface.text.logits.vocabulary_margin.test",
        node_id="output.lm_head",
        circuit_kind=CircuitKind.OUTPUT_HEAD,
        role="vocabulary_margin_geometry",
        gram=torch.eye(8, dtype=torch.float64),
        provenance={
            **provenance_body,
            "provenance_sha256": canonical_sha256(provenance_body),
        },
        interface_ids=("text.logits.output",),
    )
    evidence, report = balanced._validated_autoregressive_hankel_observability(
        ahb.evidence,
        source_width=8,
        layer_count=2,
    )
    assert evidence is ahb.evidence
    assert report is not None
    assert report["input_kind"] == "sealed-precomputed-evidence"
    assert report["source_result"] is None
    assert balanced._combined_precomputed_observability(
        streaming_vocabulary_margin=margin,
        autoregressive_hankel=evidence,
    ) == (margin, ahb.evidence)

    with pytest.raises(
        balanced.Lfm2CircuitBalancedBasisError,
        match="model-wide",
    ):
        balanced._validated_autoregressive_hankel_observability(
            ahb.evidence,
            source_width=8,
            layer_count=3,
        )

    with torch.inference_mode():
        ahb.evidence.gram[0, 0] += 1.0
    with pytest.raises(
        balanced.Lfm2CircuitBalancedBasisError,
        match="reauthentication",
    ):
        balanced._validated_autoregressive_hankel_observability(
            ahb,
            source_width=8,
            layer_count=2,
        )


def test_ahb_parameter_rejects_untyped_values_before_cache_access() -> None:
    with pytest.raises(
        balanced.Lfm2CircuitBalancedBasisError,
        match="must be one sealed LFM2 AHB result",
    ):
        balanced.build_lfm2_circuit_balanced_basis(
            torch.nn.Linear(1, 1),
            object(),
            1,
            autoregressive_hankel_observability=object(),
        )
