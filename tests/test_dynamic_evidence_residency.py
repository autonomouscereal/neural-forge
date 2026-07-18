from __future__ import annotations

from dataclasses import dataclass

import pytest

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_evidence_residency import (
    DynamicEvidenceLayerAuthentication,
    DynamicEvidenceLayerCatalog,
    DynamicEvidenceLayerReference,
    DynamicEvidenceResidencyError,
    DynamicEvidenceResidencyState,
    DynamicEvidenceResidencyWindow,
)


SOURCE_SHA256 = canonical_sha256({"source": "sealed-cache"})
PROVIDER_SHA256 = canonical_sha256({"provider": "selective-layer-v1"})
MAXIMUM_RESIDENT_BYTES = 32


@dataclass
class _Payload:
    layer_index: int
    value: bytearray


def _payload_sha256(payload: _Payload) -> str:
    return canonical_sha256(
        {
            "layer_index": payload.layer_index,
            "value_hex": bytes(payload.value).hex(),
        }
    )


class _Provider:
    source_sha256 = SOURCE_SHA256
    provider_sha256 = PROVIDER_SHA256

    def __init__(self, count: int) -> None:
        self._values = {
            index: bytearray(f"layer-{index}".encode()) for index in range(count)
        }
        self.loaded: list[int] = []
        self.authenticated: list[int] = []
        self.released: list[int] = []
        self.released_payloads: list[_Payload] = []
        self.authentication_overrides: dict[
            int,
            DynamicEvidenceLayerAuthentication,
        ] = {}

    def catalog(self) -> DynamicEvidenceLayerCatalog:
        return DynamicEvidenceLayerCatalog.create(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            evidence_sha256=[
                _payload_sha256(_Payload(index, bytearray(value)))
                for index, value in self._values.items()
            ],
            logical_bytes=[len(value) for value in self._values.values()],
        )

    def load_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
    ) -> _Payload:
        self.loaded.append(reference.layer_index)
        return _Payload(
            reference.layer_index,
            bytearray(self._values[reference.layer_index]),
        )

    def authenticate_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: _Payload,
    ) -> DynamicEvidenceLayerAuthentication:
        self.authenticated.append(reference.layer_index)
        override = self.authentication_overrides.get(reference.layer_index)
        if override is not None:
            return override
        return DynamicEvidenceLayerAuthentication.create(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            layer_index=payload.layer_index,
            reference_sha256=reference.reference_sha256,
            evidence_sha256=_payload_sha256(payload),
            logical_bytes=len(payload.value),
        )

    def release_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: _Payload,
    ) -> None:
        self.released.append(reference.layer_index)
        payload.value[:] = b"\x00" * len(payload.value)
        self.released_payloads.append(payload)


def test_current_plus_next_window_is_bounded_and_canonical() -> None:
    provider = _Provider(4)
    window = DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )

    first = window.start()
    assert first.layer_index == 0
    assert window.wait_for_prefetch() == 1
    assert window.occupied_layer_indices == (0, 1)

    second = window.advance(completed_layer_index=0)
    assert second is not None and second.layer_index == 1
    assert bytes(first.payload.value) == b"\x00" * len(b"layer-0")
    assert window.wait_for_prefetch() == 2
    assert window.occupied_layer_indices == (1, 2)

    third = window.advance(completed_layer_index=1)
    assert third is not None and third.layer_index == 2
    assert window.wait_for_prefetch() == 3
    fourth = window.advance(completed_layer_index=2)
    assert fourth is not None and fourth.layer_index == 3
    assert window.wait_for_prefetch() is None
    assert window.advance(completed_layer_index=3) is None

    assert window.state is DynamicEvidenceResidencyState.COMPLETE
    assert window.peak_occupied_layers == 2
    assert window.peak_resident_bytes == 2 * len(b"layer-0")
    assert provider.loaded == [0, 1, 2, 3]
    assert provider.released == [0, 1, 2, 3]
    snapshot = dict(window.snapshot())
    assert snapshot["requested_layer_indices"] == [0, 1, 2, 3]
    assert snapshot["promoted_layer_indices"] == [0, 1, 2, 3]
    assert snapshot["released_layer_indices"] == [0, 1, 2, 3]
    assert snapshot["parent_forward_calls"] == 0
    assert snapshot["training_steps"] == 0
    assert snapshot["maximum_resident_bytes"] == MAXIMUM_RESIDENT_BYTES
    assert snapshot["peak_resident_bytes"] == 2 * len(b"layer-0")
    assert snapshot["standalone_orchestration_only"] is True
    assert snapshot["compiler_fingerprint_binding"].startswith("integration_required")
    assert snapshot["resume_schema_binding"].startswith("integration_required")
    window.close()
    assert window.state is DynamicEvidenceResidencyState.CLOSED


def test_resume_starts_at_exact_layer_and_only_prefetches_successor() -> None:
    provider = _Provider(5)
    with DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    ) as window:
        current = window.start(start_layer_index=2)
        assert current.layer_index == 2
        assert window.wait_for_prefetch() == 3
        assert provider.loaded == [2, 3]
        assert window.occupied_layer_indices == (2, 3)

    assert provider.released == [3, 2]
    assert all(not any(payload.value) for payload in provider.released_payloads)


def test_stale_current_payload_fails_closed_and_releases_window() -> None:
    provider = _Provider(3)
    window = DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )
    current = window.start()
    assert window.wait_for_prefetch() == 1
    current.payload.value[0] ^= 1

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="does not match|changed",
    ):
        window.advance(completed_layer_index=0)

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert window.occupied_layer_indices == ()
    assert sorted(provider.released) == [0, 1]
    assert all(not any(payload.value) for payload in provider.released_payloads)
    window.close()


def test_mismatched_prefetch_authentication_fails_closed() -> None:
    provider = _Provider(3)
    catalog = provider.catalog()
    next_reference = catalog.references[1]
    provider.authentication_overrides[1] = DynamicEvidenceLayerAuthentication.create(
        source_sha256=provider.source_sha256,
        provider_sha256=provider.provider_sha256,
        layer_index=1,
        reference_sha256=next_reference.reference_sha256,
        evidence_sha256=canonical_sha256({"wrong": "payload"}),
        logical_bytes=next_reference.logical_bytes,
    )
    window = DynamicEvidenceResidencyWindow(
        catalog,
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )
    window.start()

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="Canonical prefetch failed",
    ):
        window.advance(completed_layer_index=0)

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert sorted(provider.released) == [0, 1]
    window.close()


def test_initial_source_mismatch_releases_payload_and_never_activates() -> None:
    provider = _Provider(2)
    catalog = provider.catalog()
    first_reference = catalog.references[0]
    provider.authentication_overrides[0] = DynamicEvidenceLayerAuthentication.create(
        source_sha256=canonical_sha256({"wrong": "source"}),
        provider_sha256=provider.provider_sha256,
        layer_index=0,
        reference_sha256=first_reference.reference_sha256,
        evidence_sha256=first_reference.evidence_sha256,
        logical_bytes=first_reference.logical_bytes,
    )
    window = DynamicEvidenceResidencyWindow(
        catalog,
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="does not match",
    ):
        window.start()

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert provider.loaded == [0]
    assert provider.released == [0]
    assert window.occupied_layer_indices == ()
    window.close()


def test_catalog_rejects_noncanonical_or_cross_provider_references() -> None:
    catalog = _Provider(3).catalog()

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="contiguous and canonically ordered",
    ):
        DynamicEvidenceLayerCatalog(
            source_sha256=catalog.source_sha256,
            provider_sha256=catalog.provider_sha256,
            references=tuple(reversed(catalog.references)),
            catalog_sha256=catalog.catalog_sha256,
        )

    changed = DynamicEvidenceLayerReference.create(
        source_sha256=catalog.source_sha256,
        provider_sha256=canonical_sha256({"provider": "other"}),
        layer_index=1,
        evidence_sha256=catalog.references[1].evidence_sha256,
        logical_bytes=catalog.references[1].logical_bytes,
    )
    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="changed source or provider",
    ):
        DynamicEvidenceLayerCatalog(
            source_sha256=catalog.source_sha256,
            provider_sha256=catalog.provider_sha256,
            references=(
                catalog.references[0],
                changed,
                catalog.references[2],
            ),
            catalog_sha256=catalog.catalog_sha256,
        )


def test_close_releases_current_and_completed_prefetch() -> None:
    provider = _Provider(3)
    window = DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )
    current = window.start(start_layer_index=1)
    assert window.wait_for_prefetch() == 2

    window.close()

    assert window.state is DynamicEvidenceResidencyState.CLOSED
    assert sorted(provider.released) == [1, 2]
    assert bytes(current.payload.value) == b"\x00" * len(b"layer-1")
    assert all(not any(payload.value) for payload in provider.released_payloads)


def test_advance_requires_exact_current_layer() -> None:
    provider = _Provider(2)
    with DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    ) as window:
        window.start()
        with pytest.raises(
            DynamicEvidenceResidencyError,
            match="cannot skip or repeat",
        ):
            window.advance(completed_layer_index=1)


def test_byte_budget_fails_before_loading_or_prefetching_pair() -> None:
    provider = _Provider(3)
    catalog = provider.catalog()
    pair_bytes = sum(reference.logical_bytes for reference in catalog.references[:2])
    window = DynamicEvidenceResidencyWindow(
        catalog,
        provider,
        maximum_resident_bytes=pair_bytes - 1,
    )

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="above maximum_resident_bytes",
    ):
        window.start()

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert provider.loaded == []
    assert window.peak_resident_bytes == 0
    window.close()


def test_future_pair_budget_fails_before_promotion_or_prefetch() -> None:
    provider = _Provider(3)
    provider._values = {
        0: bytearray(b"a"),
        1: bytearray(b"b"),
        2: bytearray(b"0123456789"),
    }
    window = DynamicEvidenceResidencyWindow(
        provider.catalog(),
        provider,
        maximum_resident_bytes=10,
    )
    window.start()
    assert window.wait_for_prefetch() == 1

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="above maximum_resident_bytes",
    ):
        window.advance(completed_layer_index=0)

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert provider.loaded == [0, 1]
    assert sorted(provider.released) == [0, 1]
    assert dict(window.snapshot())["promoted_layer_indices"] == [0]
    window.close()


def test_mismatched_authenticated_logical_bytes_fail_closed() -> None:
    provider = _Provider(2)
    catalog = provider.catalog()
    first_reference = catalog.references[0]
    provider.authentication_overrides[0] = DynamicEvidenceLayerAuthentication.create(
        source_sha256=provider.source_sha256,
        provider_sha256=provider.provider_sha256,
        layer_index=0,
        reference_sha256=first_reference.reference_sha256,
        evidence_sha256=first_reference.evidence_sha256,
        logical_bytes=first_reference.logical_bytes + 1,
    )
    window = DynamicEvidenceResidencyWindow(
        catalog,
        provider,
        maximum_resident_bytes=MAXIMUM_RESIDENT_BYTES,
    )

    with pytest.raises(
        DynamicEvidenceResidencyError,
        match="does not match",
    ):
        window.start()

    assert window.state is DynamicEvidenceResidencyState.FAILED
    assert provider.loaded == [0]
    assert provider.released == [0]
    window.close()
