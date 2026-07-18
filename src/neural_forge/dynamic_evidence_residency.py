"""Bounded, authenticated evidence residency for dynamic folding.

This module owns orchestration only.  It deliberately does not reinterpret,
rehash, or alter a compiler's evidence mathematics.  A registered provider
loads one layer payload and authenticates it against an already-sealed layer
reference.  The window keeps only the current layer and its canonical
successor reserved, prefetches through one worker, and releases evidence in
strict layer order.  References also seal logical payload bytes, and callers
must supply an explicit maximum for the current-plus-next reservation.

The provider is an explicit trust boundary.  It must derive the reported
``evidence_sha256`` from the loaded payload using the owning compiler/cache
contract.  The window verifies every source, provider, layer, reference, and
evidence binding, then reauthenticates both current and prefetched payloads at
the promotion boundary.  It never invents a replacement model-fold
fingerprint or resume schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_circuit_ir import canonical_sha256


EVIDENCE_LAYER_REFERENCE_SCHEMA = "neural-forge-dynamic-evidence-layer-reference-v1"
EVIDENCE_LAYER_CATALOG_SCHEMA = "neural-forge-dynamic-evidence-layer-catalog-v1"
EVIDENCE_LAYER_AUTHENTICATION_SCHEMA = (
    "neural-forge-dynamic-evidence-layer-authentication-v1"
)
EVIDENCE_RESIDENCY_WINDOW_SCHEMA = "neural-forge-dynamic-evidence-residency-window-v1"


class DynamicEvidenceResidencyError(RuntimeError):
    """A catalog, provider, authentication, ordering, or release failed."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicEvidenceResidencyError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _layer_index(value: Any, name: str = "layer_index") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicEvidenceResidencyError(f"{name} must be a nonnegative integer.")
    return value


def _positive_bytes(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicEvidenceResidencyError(
            f"{name} must be a positive integer byte count."
        )
    return value


@dataclass(frozen=True, slots=True)
class DynamicEvidenceLayerReference:
    """One lightweight, tensor-free layer evidence identity."""

    source_sha256: str
    provider_sha256: str
    layer_index: int
    evidence_sha256: str
    logical_bytes: int
    reference_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_sha256: str,
        provider_sha256: str,
        layer_index: int,
        evidence_sha256: str,
        logical_bytes: int,
    ) -> DynamicEvidenceLayerReference:
        body = {
            "schema": EVIDENCE_LAYER_REFERENCE_SCHEMA,
            "source_sha256": _digest(source_sha256, "source_sha256"),
            "provider_sha256": _digest(provider_sha256, "provider_sha256"),
            "layer_index": _layer_index(layer_index),
            "evidence_sha256": _digest(
                evidence_sha256,
                "evidence_sha256",
            ),
            "logical_bytes": _positive_bytes(
                logical_bytes,
                "logical_bytes",
            ),
        }
        return cls(
            source_sha256=body["source_sha256"],
            provider_sha256=body["provider_sha256"],
            layer_index=body["layer_index"],
            evidence_sha256=body["evidence_sha256"],
            logical_bytes=body["logical_bytes"],
            reference_sha256=canonical_sha256(body),
        )

    def __post_init__(self) -> None:
        body = {
            "schema": EVIDENCE_LAYER_REFERENCE_SCHEMA,
            "source_sha256": _digest(
                self.source_sha256,
                "source_sha256",
            ),
            "provider_sha256": _digest(
                self.provider_sha256,
                "provider_sha256",
            ),
            "layer_index": _layer_index(self.layer_index),
            "evidence_sha256": _digest(
                self.evidence_sha256,
                "evidence_sha256",
            ),
            "logical_bytes": _positive_bytes(
                self.logical_bytes,
                "logical_bytes",
            ),
        }
        if _digest(
            self.reference_sha256,
            "reference_sha256",
        ) != canonical_sha256(body):
            raise DynamicEvidenceResidencyError(
                "Layer evidence reference seal is invalid."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_LAYER_REFERENCE_SCHEMA,
            "source_sha256": self.source_sha256,
            "provider_sha256": self.provider_sha256,
            "layer_index": self.layer_index,
            "evidence_sha256": self.evidence_sha256,
            "logical_bytes": self.logical_bytes,
            "reference_sha256": self.reference_sha256,
        }


@dataclass(frozen=True, slots=True)
class DynamicEvidenceLayerCatalog:
    """Canonical contiguous references for one source/provider pair."""

    source_sha256: str
    provider_sha256: str
    references: tuple[DynamicEvidenceLayerReference, ...]
    catalog_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_sha256: str,
        provider_sha256: str,
        evidence_sha256: Sequence[str],
        logical_bytes: Sequence[int],
    ) -> DynamicEvidenceLayerCatalog:
        source = _digest(source_sha256, "source_sha256")
        provider = _digest(provider_sha256, "provider_sha256")
        evidence = tuple(evidence_sha256)
        byte_counts = tuple(logical_bytes)
        if len(evidence) != len(byte_counts):
            raise DynamicEvidenceResidencyError(
                "Evidence hashes and logical byte counts must align exactly."
            )
        references = tuple(
            DynamicEvidenceLayerReference.create(
                source_sha256=source,
                provider_sha256=provider,
                layer_index=index,
                evidence_sha256=value,
                logical_bytes=byte_counts[index],
            )
            for index, value in enumerate(evidence)
        )
        if not references:
            raise DynamicEvidenceResidencyError("An evidence catalog cannot be empty.")
        body = {
            "schema": EVIDENCE_LAYER_CATALOG_SCHEMA,
            "source_sha256": source,
            "provider_sha256": provider,
            "layer_count": len(references),
            "total_logical_bytes": sum(
                reference.logical_bytes for reference in references
            ),
            "ordered_reference_sha256": [
                reference.reference_sha256 for reference in references
            ],
        }
        return cls(
            source_sha256=source,
            provider_sha256=provider,
            references=references,
            catalog_sha256=canonical_sha256(body),
        )

    def __post_init__(self) -> None:
        source = _digest(self.source_sha256, "source_sha256")
        provider = _digest(self.provider_sha256, "provider_sha256")
        references = tuple(self.references)
        if not references or any(
            not isinstance(reference, DynamicEvidenceLayerReference)
            for reference in references
        ):
            raise DynamicEvidenceResidencyError(
                "Catalog references must contain typed layer references."
            )
        if tuple(reference.layer_index for reference in references) != tuple(
            range(len(references))
        ):
            raise DynamicEvidenceResidencyError(
                "Catalog references must be contiguous and canonically ordered."
            )
        if any(
            reference.source_sha256 != source or reference.provider_sha256 != provider
            for reference in references
        ):
            raise DynamicEvidenceResidencyError(
                "Catalog references changed source or provider identity."
            )
        body = {
            "schema": EVIDENCE_LAYER_CATALOG_SCHEMA,
            "source_sha256": source,
            "provider_sha256": provider,
            "layer_count": len(references),
            "total_logical_bytes": sum(
                reference.logical_bytes for reference in references
            ),
            "ordered_reference_sha256": [
                reference.reference_sha256 for reference in references
            ],
        }
        if _digest(
            self.catalog_sha256,
            "catalog_sha256",
        ) != canonical_sha256(body):
            raise DynamicEvidenceResidencyError(
                "Layer evidence catalog seal is invalid."
            )
        object.__setattr__(self, "references", references)

    @property
    def layer_count(self) -> int:
        return len(self.references)

    @property
    def total_logical_bytes(self) -> int:
        return sum(reference.logical_bytes for reference in self.references)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_LAYER_CATALOG_SCHEMA,
            "source_sha256": self.source_sha256,
            "provider_sha256": self.provider_sha256,
            "layer_count": self.layer_count,
            "total_logical_bytes": self.total_logical_bytes,
            "ordered_reference_sha256": [
                reference.reference_sha256 for reference in self.references
            ],
            "catalog_sha256": self.catalog_sha256,
        }


@dataclass(frozen=True, slots=True)
class DynamicEvidenceLayerAuthentication:
    """Provider receipt for the bytes currently backing one layer payload."""

    source_sha256: str
    provider_sha256: str
    layer_index: int
    reference_sha256: str
    evidence_sha256: str
    logical_bytes: int
    authentication_sha256: str

    @classmethod
    def create(
        cls,
        *,
        source_sha256: str,
        provider_sha256: str,
        layer_index: int,
        reference_sha256: str,
        evidence_sha256: str,
        logical_bytes: int,
    ) -> DynamicEvidenceLayerAuthentication:
        body = {
            "schema": EVIDENCE_LAYER_AUTHENTICATION_SCHEMA,
            "source_sha256": _digest(source_sha256, "source_sha256"),
            "provider_sha256": _digest(
                provider_sha256,
                "provider_sha256",
            ),
            "layer_index": _layer_index(layer_index),
            "reference_sha256": _digest(
                reference_sha256,
                "reference_sha256",
            ),
            "evidence_sha256": _digest(
                evidence_sha256,
                "evidence_sha256",
            ),
            "logical_bytes": _positive_bytes(
                logical_bytes,
                "logical_bytes",
            ),
            "content_authenticated": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }
        return cls(
            source_sha256=body["source_sha256"],
            provider_sha256=body["provider_sha256"],
            layer_index=body["layer_index"],
            reference_sha256=body["reference_sha256"],
            evidence_sha256=body["evidence_sha256"],
            logical_bytes=body["logical_bytes"],
            authentication_sha256=canonical_sha256(body),
        )

    def __post_init__(self) -> None:
        body = {
            "schema": EVIDENCE_LAYER_AUTHENTICATION_SCHEMA,
            "source_sha256": _digest(
                self.source_sha256,
                "source_sha256",
            ),
            "provider_sha256": _digest(
                self.provider_sha256,
                "provider_sha256",
            ),
            "layer_index": _layer_index(self.layer_index),
            "reference_sha256": _digest(
                self.reference_sha256,
                "reference_sha256",
            ),
            "evidence_sha256": _digest(
                self.evidence_sha256,
                "evidence_sha256",
            ),
            "logical_bytes": _positive_bytes(
                self.logical_bytes,
                "logical_bytes",
            ),
            "content_authenticated": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
        }
        if _digest(
            self.authentication_sha256,
            "authentication_sha256",
        ) != canonical_sha256(body):
            raise DynamicEvidenceResidencyError(
                "Layer evidence authentication receipt is invalid."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": EVIDENCE_LAYER_AUTHENTICATION_SCHEMA,
            "source_sha256": self.source_sha256,
            "provider_sha256": self.provider_sha256,
            "layer_index": self.layer_index,
            "reference_sha256": self.reference_sha256,
            "evidence_sha256": self.evidence_sha256,
            "logical_bytes": self.logical_bytes,
            "content_authenticated": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "authentication_sha256": self.authentication_sha256,
        }


class DynamicEvidenceLayerProvider(Protocol):
    """Trust boundary implemented by an architecture/cache adapter."""

    source_sha256: str
    provider_sha256: str

    def load_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
    ) -> Any:
        """Load only the requested layer without a parent-model rerun."""

    def authenticate_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Any,
    ) -> DynamicEvidenceLayerAuthentication:
        """Content-authenticate a loaded or already-resident payload."""

    def release_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Any,
    ) -> None:
        """Release provider resources and zeroize mutable staging if safe."""


@dataclass(frozen=True, slots=True)
class DynamicResidentEvidenceLayer:
    """One authenticated payload owned by a residency window."""

    reference: DynamicEvidenceLayerReference
    authentication: DynamicEvidenceLayerAuthentication
    payload: Any

    def __post_init__(self) -> None:
        if not isinstance(
            self.reference,
            DynamicEvidenceLayerReference,
        ) or not isinstance(
            self.authentication,
            DynamicEvidenceLayerAuthentication,
        ):
            raise DynamicEvidenceResidencyError(
                "Resident evidence requires typed reference and authentication."
            )
        _assert_authentication_matches(
            self.reference,
            self.authentication,
        )

    @property
    def layer_index(self) -> int:
        return self.reference.layer_index

    @property
    def logical_bytes(self) -> int:
        return self.reference.logical_bytes


def _assert_authentication_matches(
    reference: DynamicEvidenceLayerReference,
    authentication: DynamicEvidenceLayerAuthentication,
) -> None:
    if (
        authentication.source_sha256 != reference.source_sha256
        or authentication.provider_sha256 != reference.provider_sha256
        or authentication.layer_index != reference.layer_index
        or authentication.reference_sha256 != reference.reference_sha256
        or authentication.evidence_sha256 != reference.evidence_sha256
        or authentication.logical_bytes != reference.logical_bytes
    ):
        raise DynamicEvidenceResidencyError(
            f"Authenticated evidence for layer {reference.layer_index} "
            "does not match its sealed reference."
        )


class DynamicEvidenceResidencyState(str, Enum):
    NEW = "new"
    ACTIVE = "active"
    COMPLETE = "complete"
    FAILED = "failed"
    CLOSED = "closed"


class DynamicEvidenceResidencyWindow:
    """Keep exactly the current and canonical next evidence layers warm."""

    def __init__(
        self,
        catalog: DynamicEvidenceLayerCatalog,
        provider: DynamicEvidenceLayerProvider,
        *,
        maximum_resident_bytes: int,
    ) -> None:
        if not isinstance(catalog, DynamicEvidenceLayerCatalog):
            raise TypeError("catalog must be DynamicEvidenceLayerCatalog.")
        for method_name in (
            "load_layer_payload",
            "authenticate_layer_payload",
            "release_layer_payload",
        ):
            if not callable(getattr(provider, method_name, None)):
                raise DynamicEvidenceResidencyError(
                    f"Evidence provider lacks callable {method_name}."
                )
        if (
            _digest(
                getattr(provider, "source_sha256", None),
                "provider source_sha256",
            )
            != catalog.source_sha256
            or _digest(
                getattr(provider, "provider_sha256", None),
                "provider provider_sha256",
            )
            != catalog.provider_sha256
        ):
            raise DynamicEvidenceResidencyError(
                "Evidence provider identity differs from the sealed catalog."
            )
        self._catalog = catalog
        self._provider = provider
        self._maximum_resident_bytes = _positive_bytes(
            maximum_resident_bytes,
            "maximum_resident_bytes",
        )
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="neural-forge-evidence-prefetch",
        )
        self._state = DynamicEvidenceResidencyState.NEW
        self._current: DynamicResidentEvidenceLayer | None = None
        self._prefetch: (
            tuple[
                DynamicEvidenceLayerReference,
                Future[DynamicResidentEvidenceLayer],
            ]
            | None
        ) = None
        self._start_layer_index: int | None = None
        self._requested_layer_indices: list[int] = []
        self._promoted_layer_indices: list[int] = []
        self._released_layer_indices: list[int] = []
        self._peak_occupied_layers = 0
        self._peak_resident_bytes = 0
        self._failure: BaseException | None = None

    @property
    def state(self) -> DynamicEvidenceResidencyState:
        return self._state

    @property
    def current(self) -> DynamicResidentEvidenceLayer:
        if (
            self._state is not DynamicEvidenceResidencyState.ACTIVE
            or self._current is None
        ):
            raise DynamicEvidenceResidencyError(
                "The residency window has no active current layer."
            )
        return self._current

    @property
    def occupied_layer_indices(self) -> tuple[int, ...]:
        result = []
        if self._current is not None:
            result.append(self._current.layer_index)
        if self._prefetch is not None:
            result.append(self._prefetch[0].layer_index)
        return tuple(result)

    @property
    def peak_occupied_layers(self) -> int:
        return self._peak_occupied_layers

    @property
    def resident_bytes(self) -> int:
        result = 0
        if self._current is not None:
            result += self._current.logical_bytes
        if self._prefetch is not None:
            result += self._prefetch[0].logical_bytes
        return result

    @property
    def peak_resident_bytes(self) -> int:
        return self._peak_resident_bytes

    def _record_occupancy(self) -> None:
        occupancy = len(self.occupied_layer_indices)
        resident_bytes = self.resident_bytes
        self._peak_occupied_layers = max(
            self._peak_occupied_layers,
            occupancy,
        )
        self._peak_resident_bytes = max(
            self._peak_resident_bytes,
            resident_bytes,
        )
        if occupancy > 2:
            raise DynamicEvidenceResidencyError(
                "Evidence residency exceeded current-plus-next capacity."
            )
        if resident_bytes > self._maximum_resident_bytes:
            raise DynamicEvidenceResidencyError(
                "Evidence residency exceeded maximum_resident_bytes."
            )

    def _assert_pair_capacity(
        self,
        current: DynamicEvidenceLayerReference,
        successor: DynamicEvidenceLayerReference | None,
    ) -> None:
        required = current.logical_bytes + (
            0 if successor is None else successor.logical_bytes
        )
        if required > self._maximum_resident_bytes:
            pair = (
                str(current.layer_index)
                if successor is None
                else f"{current.layer_index}+{successor.layer_index}"
            )
            raise DynamicEvidenceResidencyError(
                f"Evidence layer window {pair} requires {required} logical "
                f"bytes, above maximum_resident_bytes="
                f"{self._maximum_resident_bytes}."
            )

    def _authenticate(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Any,
    ) -> DynamicEvidenceLayerAuthentication:
        try:
            authentication = self._provider.authenticate_layer_payload(
                reference,
                payload,
            )
        except BaseException as exc:
            raise DynamicEvidenceResidencyError(
                f"Evidence authentication failed for layer {reference.layer_index}."
            ) from exc
        if not isinstance(
            authentication,
            DynamicEvidenceLayerAuthentication,
        ):
            raise DynamicEvidenceResidencyError(
                "Evidence provider returned an untyped authentication receipt."
            )
        _assert_authentication_matches(reference, authentication)
        return authentication

    def _release_raw(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Any,
    ) -> None:
        try:
            self._provider.release_layer_payload(reference, payload)
        except BaseException as exc:
            raise DynamicEvidenceResidencyError(
                f"Evidence release failed for layer {reference.layer_index}."
            ) from exc
        self._released_layer_indices.append(reference.layer_index)

    def _load_authenticated(
        self,
        reference: DynamicEvidenceLayerReference,
    ) -> DynamicResidentEvidenceLayer:
        payload: Any = None
        loaded = False
        try:
            payload = self._provider.load_layer_payload(reference)
            loaded = True
            authentication = self._authenticate(reference, payload)
            return DynamicResidentEvidenceLayer(
                reference=reference,
                authentication=authentication,
                payload=payload,
            )
        except BaseException:
            if loaded:
                self._release_raw(reference, payload)
            raise

    def _schedule_prefetch(self, layer_index: int) -> None:
        if self._prefetch is not None:
            raise DynamicEvidenceResidencyError(
                "A canonical next-layer prefetch is already reserved."
            )
        reference = self._catalog.references[layer_index]
        if self._current is None:
            raise DynamicEvidenceResidencyError(
                "A current layer is required before successor prefetch."
            )
        self._assert_pair_capacity(
            self._current.reference,
            reference,
        )
        self._requested_layer_indices.append(layer_index)
        self._prefetch = (
            reference,
            self._executor.submit(
                self._load_authenticated,
                reference,
            ),
        )
        self._record_occupancy()

    def start(
        self,
        *,
        start_layer_index: int = 0,
    ) -> DynamicResidentEvidenceLayer:
        """Authenticate the resume layer and prefetch only its successor."""

        if self._state is not DynamicEvidenceResidencyState.NEW:
            raise DynamicEvidenceResidencyError(
                "A residency window can be started exactly once."
            )
        index = _layer_index(
            start_layer_index,
            "start_layer_index",
        )
        if index >= self._catalog.layer_count:
            raise DynamicEvidenceResidencyError(
                "start_layer_index is outside the evidence catalog."
            )
        self._start_layer_index = index
        reference = self._catalog.references[index]
        successor = (
            None
            if index + 1 >= self._catalog.layer_count
            else self._catalog.references[index + 1]
        )
        try:
            self._assert_pair_capacity(reference, successor)
        except BaseException as exc:
            self._failure = exc
            self._state = DynamicEvidenceResidencyState.FAILED
            raise
        self._requested_layer_indices.append(index)
        try:
            current = self._load_authenticated(reference)
        except BaseException as exc:
            self._failure = exc
            self._state = DynamicEvidenceResidencyState.FAILED
            raise
        self._current = current
        self._promoted_layer_indices.append(index)
        self._state = DynamicEvidenceResidencyState.ACTIVE
        self._record_occupancy()
        if index + 1 < self._catalog.layer_count:
            self._schedule_prefetch(index + 1)
        return current

    def wait_for_prefetch(self) -> int | None:
        """Wait for the reserved successor without promoting or reordering it."""

        if self._state is not DynamicEvidenceResidencyState.ACTIVE:
            raise DynamicEvidenceResidencyError(
                "Prefetch waiting requires an active residency window."
            )
        if self._prefetch is None:
            return None
        reference, future = self._prefetch
        try:
            loaded = future.result()
        except BaseException as exc:
            raise DynamicEvidenceResidencyError(
                f"Canonical prefetch failed for layer {reference.layer_index}."
            ) from exc
        if loaded.reference != reference:
            raise DynamicEvidenceResidencyError(
                "Prefetched evidence changed its reserved reference."
            )
        return reference.layer_index

    def verify_current(self) -> DynamicEvidenceLayerAuthentication:
        """Reauthenticate current bytes without changing residency order."""

        current = self.current
        authentication = self._authenticate(
            current.reference,
            current.payload,
        )
        if (
            authentication.authentication_sha256
            != current.authentication.authentication_sha256
        ):
            raise DynamicEvidenceResidencyError(
                f"Resident evidence for layer {current.layer_index} changed "
                "after its authenticated load."
            )
        return authentication

    def _release_resident(
        self,
        resident: DynamicResidentEvidenceLayer,
    ) -> None:
        self._release_raw(
            resident.reference,
            resident.payload,
        )

    def _consume_prefetch(self) -> DynamicResidentEvidenceLayer:
        if self._prefetch is None:
            raise DynamicEvidenceResidencyError(
                "The current layer has no canonical successor."
            )
        reference, future = self._prefetch
        self._prefetch = None
        try:
            resident = future.result()
        except BaseException as exc:
            raise DynamicEvidenceResidencyError(
                f"Canonical prefetch failed for layer {reference.layer_index}."
            ) from exc
        if resident.reference != reference:
            self._release_resident(resident)
            raise DynamicEvidenceResidencyError(
                "Prefetched evidence changed its reserved reference."
            )
        return resident

    def _fail_active(
        self,
        error: BaseException,
        *,
        prefetched: DynamicResidentEvidenceLayer | None = None,
    ) -> None:
        cleanup_errors = []
        current, self._current = self._current, None
        if current is not None:
            try:
                self._release_resident(current)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if prefetched is not None:
            try:
                self._release_resident(prefetched)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if self._prefetch is not None:
            _reference, future = self._prefetch
            self._prefetch = None
            if not future.cancel():
                try:
                    queued = future.result()
                except BaseException as exc:
                    cleanup_errors.append(exc)
                else:
                    try:
                        self._release_resident(queued)
                    except BaseException as exc:
                        cleanup_errors.append(exc)
        self._failure = error
        self._state = DynamicEvidenceResidencyState.FAILED
        for cleanup_error in cleanup_errors:
            error.add_note(
                "Evidence cleanup also failed: "
                f"{type(cleanup_error).__qualname__}: {cleanup_error}"
            )

    def advance(
        self,
        *,
        completed_layer_index: int,
    ) -> DynamicResidentEvidenceLayer | None:
        """Release a verified current layer and promote its exact successor."""

        current = self.current
        completed = _layer_index(
            completed_layer_index,
            "completed_layer_index",
        )
        if completed != current.layer_index:
            raise DynamicEvidenceResidencyError(
                "Residency advance cannot skip or repeat a layer."
            )
        prefetched: DynamicResidentEvidenceLayer | None = None
        try:
            self.verify_current()
            if self._prefetch is None:
                self._current = None
                self._release_resident(current)
                self._state = DynamicEvidenceResidencyState.COMPLETE
                return None
            prefetched = self._consume_prefetch()
            next_authentication = self._authenticate(
                prefetched.reference,
                prefetched.payload,
            )
            if (
                next_authentication.authentication_sha256
                != prefetched.authentication.authentication_sha256
                or prefetched.layer_index != current.layer_index + 1
            ):
                raise DynamicEvidenceResidencyError(
                    "Prefetched evidence changed before canonical promotion."
                )
            next_index = prefetched.layer_index + 1
            following_reference = (
                None
                if next_index >= self._catalog.layer_count
                else self._catalog.references[next_index]
            )
            self._assert_pair_capacity(
                prefetched.reference,
                following_reference,
            )
            self._current = None
            self._release_resident(current)
            promoted = prefetched
            self._current = promoted
            self._promoted_layer_indices.append(promoted.layer_index)
            prefetched = None
            if next_index < self._catalog.layer_count:
                self._schedule_prefetch(next_index)
            self._record_occupancy()
            return promoted
        except BaseException as exc:
            if self._state is DynamicEvidenceResidencyState.ACTIVE:
                self._fail_active(exc, prefetched=prefetched)
            raise

    def snapshot(self) -> Mapping[str, Any]:
        """Return deterministic orchestration state, never a fold fingerprint."""

        body = {
            "schema": EVIDENCE_RESIDENCY_WINDOW_SCHEMA,
            "catalog_sha256": self._catalog.catalog_sha256,
            "source_sha256": self._catalog.source_sha256,
            "provider_sha256": self._catalog.provider_sha256,
            "state": self._state.value,
            "start_layer_index": self._start_layer_index,
            "current_layer_index": (
                None if self._current is None else self._current.layer_index
            ),
            "occupied_layer_indices": list(self.occupied_layer_indices),
            "max_resident_layers": 2,
            "maximum_resident_bytes": self._maximum_resident_bytes,
            "resident_bytes": self.resident_bytes,
            "prefetch_distance_layers": 1,
            "peak_occupied_layers": self._peak_occupied_layers,
            "peak_resident_bytes": self._peak_resident_bytes,
            "requested_layer_indices": list(self._requested_layer_indices),
            "promoted_layer_indices": list(self._promoted_layer_indices),
            "released_layer_indices": list(self._released_layer_indices),
            "canonical_layer_order_required": True,
            "parent_forward_calls": 0,
            "training_steps": 0,
            "compiler_fingerprint_binding": (
                "integration_required_not_asserted_by_residency_provider"
            ),
            "resume_schema_binding": (
                "integration_required_not_asserted_by_residency_provider"
            ),
            "standalone_orchestration_only": True,
        }
        return MappingProxyType(
            {
                **body,
                "snapshot_sha256": canonical_sha256(body),
            }
        )

    def close(self) -> None:
        """Release current/prefetched payloads and stop the prefetch worker."""

        if self._state is DynamicEvidenceResidencyState.CLOSED:
            return
        errors: list[BaseException] = []
        current, self._current = self._current, None
        prefetch, self._prefetch = self._prefetch, None
        if prefetch is not None:
            _reference, future = prefetch
            if not future.cancel():
                try:
                    prefetched = future.result()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    try:
                        self._release_resident(prefetched)
                    except BaseException as exc:
                        errors.append(exc)
        if current is not None:
            try:
                self._release_resident(current)
            except BaseException as exc:
                errors.append(exc)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._state = DynamicEvidenceResidencyState.CLOSED
        if errors:
            error = DynamicEvidenceResidencyError(
                "Evidence residency close encountered cleanup failures."
            )
            for cleanup_error in errors:
                error.add_note(f"{type(cleanup_error).__qualname__}: {cleanup_error}")
            raise error

    def __enter__(self) -> DynamicEvidenceResidencyWindow:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


__all__ = [
    "DynamicEvidenceLayerAuthentication",
    "DynamicEvidenceLayerCatalog",
    "DynamicEvidenceLayerProvider",
    "DynamicEvidenceLayerReference",
    "DynamicEvidenceResidencyError",
    "DynamicEvidenceResidencyState",
    "DynamicEvidenceResidencyWindow",
    "DynamicResidentEvidenceLayer",
    "EVIDENCE_LAYER_AUTHENTICATION_SCHEMA",
    "EVIDENCE_LAYER_CATALOG_SCHEMA",
    "EVIDENCE_LAYER_REFERENCE_SCHEMA",
    "EVIDENCE_RESIDENCY_WINDOW_SCHEMA",
]
