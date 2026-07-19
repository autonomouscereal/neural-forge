"""Authenticated, bounded LFM2 CPCF evidence loaded one source layer at a time.

This module is the architecture-specific provider behind
``DynamicEvidenceResidencyWindow``.  It consumes only an already committed
local :class:`~neural_forge.hf_lfm2_circuit_trace_cache.Lfm2CircuitTraceCache`.
The cache session authenticates every trace file once, then this provider
loads only the tensor names for one source layer across canonical
batch/phase order.  Existing CPCF functional checks and evidence hashes are
preserved exactly.

The first bounded pass creates a tensor-free catalog.  It intentionally does
not alter the current CPCF model compiler, transaction fingerprint, resume
schema, or live fold.  Compiler integration requires a separately reviewed
fingerprint/schema transition.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_evidence_catalog_artifact import (
    DynamicEvidenceCatalogArtifact,
    DynamicEvidenceCatalogArtifactStore,
    DynamicEvidenceCatalogBinding,
)
from .dynamic_evidence_residency import (
    DynamicEvidenceLayerAuthentication,
    DynamicEvidenceLayerCatalog,
    DynamicEvidenceLayerReference,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectedBatchPayload,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from . import hf_lfm2_cpcf_installability as installability


LFM2_CPCF_SELECTIVE_PROVIDER_IMPLEMENTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-provider-implementation-v1"
)
LFM2_CPCF_SELECTIVE_SOURCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-source-v1"
)
LFM2_CPCF_SELECTIVE_LAYER_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-layer-payload-v1"
)
LFM2_CPCF_SELECTIVE_FINAL_NORM_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-final-norm-v1"
)
LFM2_CPCF_SELECTIVE_PHASE_REFERENCE_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-phase-reference-v1"
)
LFM2_CPCF_SELECTIVE_CATALOG_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-catalog-v1"
)
LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_CODEC_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-catalog-artifact-codec-v1"
)
LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_PAYLOAD_SCHEMA = (
    "neural-forge-lfm2-cpcf-selective-catalog-artifact-payload-v1"
)
LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_KIND = (
    "lfm2-cpcf-selective-evidence"
)


class Lfm2CpcfSelectiveEvidenceError(RuntimeError):
    """Selective evidence authentication, ordering, or release failed."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfSelectiveEvidenceError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfSelectiveEvidenceError(
            f"{name} must be a nonnegative integer."
        )
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2CpcfSelectiveEvidenceError(
            f"Cannot fingerprint provider source {path}."
        ) from exc
    return digest.hexdigest()


def lfm2_cpcf_selective_provider_implementation() -> Mapping[str, Any]:
    """Content-address the provider and the two evidence contracts it calls."""

    import neural_forge.dynamic_evidence_residency as residency
    import neural_forge.hf_lfm2_circuit_trace_cache as trace_cache

    body = {
        "schema": LFM2_CPCF_SELECTIVE_PROVIDER_IMPLEMENTATION_SCHEMA,
        "provider_source_sha256": _file_sha256(Path(__file__).resolve()),
        "trace_cache_source_sha256": _file_sha256(
            Path(trace_cache.__file__).resolve()
        ),
        "installability_source_sha256": _file_sha256(
            Path(installability.__file__).resolve()
        ),
        "residency_source_sha256": _file_sha256(
            Path(residency.__file__).resolve()
        ),
        "parent_forward_calls": 0,
        "training_steps": 0,
        "arithmetic_conversion": False,
    }
    return MappingProxyType(
        {
            **body,
            "implementation_sha256": canonical_sha256(body),
        }
    )


def _logical_tensor_bytes(values: Mapping[str, Any]) -> int:
    result = sum(
        int(value.numel() * value.element_size()) for value in values.values()
    )
    return _positive_int(result, "logical tensor bytes")


def _field_inventory_sha256(values: Mapping[str, Any]) -> str:
    return canonical_sha256(installability._field_inventory(values))


def _layer_names(
    topology: Sequence[Mapping[str, Any]],
    layer_index: int,
) -> tuple[str, ...]:
    contract = topology[layer_index]["contract"]
    operator_kind = str(contract["kind"])
    return (
        *installability._COMMON_FIELDS,
        *(
            installability._ATTENTION_FIELDS
            if operator_kind == "full_attention"
            else installability._SHORT_CONV_FIELDS
        ),
    )


@dataclass(frozen=True, slots=True)
class _PhasePlan:
    phase_index: int
    batch_index: int
    input_sha256: str
    phase: str
    diagnostics: Mapping[str, Any]


def _phase_reference_body(
    *,
    phase_index: int,
    cache_key: str,
    batch_index: int,
    input_sha256: str,
    phase: str,
    layer_operator_kinds: Sequence[str],
    layer_evidence_sha256: Sequence[str],
    layer_field_inventory_sha256: Sequence[str],
    layer_logical_bytes: Sequence[int],
    final_norm_evidence_sha256: str,
    final_norm_field_inventory_sha256: str,
    final_norm_logical_bytes: int,
    existing_phase_evidence_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SELECTIVE_PHASE_REFERENCE_SCHEMA,
        "phase_index": phase_index,
        "cache_key": cache_key,
        "batch_index": batch_index,
        "input_sha256": input_sha256,
        "phase": phase,
        "layer_operator_kinds": list(layer_operator_kinds),
        "layer_evidence_sha256": list(layer_evidence_sha256),
        "layer_field_inventory_sha256": list(
            layer_field_inventory_sha256
        ),
        "layer_logical_bytes": list(layer_logical_bytes),
        "final_norm_evidence_sha256": final_norm_evidence_sha256,
        "final_norm_field_inventory_sha256": (
            final_norm_field_inventory_sha256
        ),
        "final_norm_logical_bytes": final_norm_logical_bytes,
        "existing_phase_evidence_sha256": existing_phase_evidence_sha256,
        "parent_side_compiler_inputs_ready": True,
        "tensor_payloads_resident": False,
    }


@dataclass(frozen=True, slots=True)
class Lfm2CpcfSelectivePhaseReference:
    """One tensor-free phase identity preserving existing CPCF evidence seals."""

    phase_index: int
    cache_key: str
    batch_index: int
    input_sha256: str
    phase: str
    layer_operator_kinds: tuple[str, ...]
    layer_evidence_sha256: tuple[str, ...]
    layer_field_inventory_sha256: tuple[str, ...]
    layer_logical_bytes: tuple[int, ...]
    final_norm_evidence_sha256: str
    final_norm_field_inventory_sha256: str
    final_norm_logical_bytes: int
    existing_phase_evidence_sha256: str
    reference_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.phase_index, "phase_index")
        _digest(self.cache_key, "cache_key")
        _nonnegative_int(self.batch_index, "batch_index")
        _digest(self.input_sha256, "input_sha256")
        if not isinstance(self.phase, str) or not self.phase:
            raise Lfm2CpcfSelectiveEvidenceError("phase must be nonempty.")
        count = len(self.layer_operator_kinds)
        if (
            count == 0
            or len(self.layer_evidence_sha256) != count
            or len(self.layer_field_inventory_sha256) != count
            or len(self.layer_logical_bytes) != count
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Phase layer reference columns must align exactly."
            )
        if any(
            value not in {"full_attention", "short_conv"}
            for value in self.layer_operator_kinds
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Phase reference has a noncanonical operator kind."
            )
        for value in (
            *self.layer_evidence_sha256,
            *self.layer_field_inventory_sha256,
            self.final_norm_evidence_sha256,
            self.final_norm_field_inventory_sha256,
            self.existing_phase_evidence_sha256,
        ):
            _digest(value, "phase evidence digest")
        for value in self.layer_logical_bytes:
            _positive_int(value, "layer_logical_bytes")
        _positive_int(
            self.final_norm_logical_bytes,
            "final_norm_logical_bytes",
        )
        body = _phase_reference_body(
            phase_index=self.phase_index,
            cache_key=self.cache_key,
            batch_index=self.batch_index,
            input_sha256=self.input_sha256,
            phase=self.phase,
            layer_operator_kinds=self.layer_operator_kinds,
            layer_evidence_sha256=self.layer_evidence_sha256,
            layer_field_inventory_sha256=(
                self.layer_field_inventory_sha256
            ),
            layer_logical_bytes=self.layer_logical_bytes,
            final_norm_evidence_sha256=self.final_norm_evidence_sha256,
            final_norm_field_inventory_sha256=(
                self.final_norm_field_inventory_sha256
            ),
            final_norm_logical_bytes=self.final_norm_logical_bytes,
            existing_phase_evidence_sha256=(
                self.existing_phase_evidence_sha256
            ),
        )
        if _digest(
            self.reference_sha256,
            "reference_sha256",
        ) != canonical_sha256(body):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective phase reference seal is invalid."
            )

    def as_dict(self) -> dict[str, Any]:
        body = _phase_reference_body(
            phase_index=self.phase_index,
            cache_key=self.cache_key,
            batch_index=self.batch_index,
            input_sha256=self.input_sha256,
            phase=self.phase,
            layer_operator_kinds=self.layer_operator_kinds,
            layer_evidence_sha256=self.layer_evidence_sha256,
            layer_field_inventory_sha256=(
                self.layer_field_inventory_sha256
            ),
            layer_logical_bytes=self.layer_logical_bytes,
            final_norm_evidence_sha256=self.final_norm_evidence_sha256,
            final_norm_field_inventory_sha256=(
                self.final_norm_field_inventory_sha256
            ),
            final_norm_logical_bytes=self.final_norm_logical_bytes,
            existing_phase_evidence_sha256=(
                self.existing_phase_evidence_sha256
            ),
        )
        return {**body, "reference_sha256": self.reference_sha256}


@dataclass(frozen=True, slots=True)
class _LayerPhasePayload:
    phase_reference_sha256: str
    evidence: installability.Lfm2CpcfInstallableLayerEvidence


class Lfm2CpcfSelectiveLayerPayload:
    """All canonical phases for one source layer, and no other layer."""

    __slots__ = (
        "evidence_sha256",
        "layer_index",
        "logical_bytes",
        "operator_kind",
        "_phases",
        "_released",
    )

    def __init__(
        self,
        *,
        layer_index: int,
        operator_kind: str,
        phases: Sequence[_LayerPhasePayload],
        evidence_sha256: str,
        logical_bytes: int,
    ) -> None:
        self.layer_index = layer_index
        self.operator_kind = operator_kind
        self.evidence_sha256 = evidence_sha256
        self.logical_bytes = logical_bytes
        self._phases: tuple[_LayerPhasePayload, ...] | None = tuple(phases)
        self._released = False

    @property
    def phases(self) -> tuple[_LayerPhasePayload, ...]:
        if self._phases is None:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective layer payload was released."
            )
        return self._phases

    @property
    def released(self) -> bool:
        return self._released

    def _release(self) -> None:
        self._phases = None
        self._released = True


@dataclass(frozen=True, slots=True)
class _FinalNormPhasePayload:
    phase_reference_sha256: str
    batch_index: int
    input_sha256: str
    phase: str
    tensors: Mapping[str, Any]
    evidence_sha256: str
    logical_bytes: int


class Lfm2CpcfSelectiveFinalNormPayload:
    """Separately bounded final-normalization tensors for all phases."""

    __slots__ = (
        "evidence_sha256",
        "logical_bytes",
        "_phases",
        "_released",
    )

    def __init__(
        self,
        *,
        phases: Sequence[_FinalNormPhasePayload],
        evidence_sha256: str,
        logical_bytes: int,
    ) -> None:
        self.evidence_sha256 = evidence_sha256
        self.logical_bytes = logical_bytes
        self._phases: tuple[_FinalNormPhasePayload, ...] | None = tuple(phases)
        self._released = False

    @property
    def phases(self) -> tuple[_FinalNormPhasePayload, ...]:
        if self._phases is None:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective final-norm payload was released."
            )
        return self._phases

    @property
    def released(self) -> bool:
        return self._released

    def _release(self) -> None:
        self._phases = None
        self._released = True


@dataclass(frozen=True, slots=True)
class Lfm2CpcfSelectiveEvidenceCatalog:
    """Tensor-free phase catalog plus the generic dynamic layer catalog."""

    cache_key: str
    source_sha256: str
    provider_sha256: str
    selective_session_sha256: str
    layer_topology_sha256: str
    capture_selection_sha256: str
    source_width: int
    phase_references: tuple[Lfm2CpcfSelectivePhaseReference, ...]
    dynamic_catalog: DynamicEvidenceLayerCatalog
    final_norm_evidence_sha256: str
    final_norm_logical_bytes: int
    catalog_sha256: str

    @classmethod
    def create(
        cls,
        *,
        cache_key: str,
        source_sha256: str,
        provider_sha256: str,
        selective_session_sha256: str,
        layer_topology_sha256: str,
        capture_selection_sha256: str,
        source_width: int,
        phase_references: Sequence[Lfm2CpcfSelectivePhaseReference],
        dynamic_catalog: DynamicEvidenceLayerCatalog,
        final_norm_evidence_sha256: str,
        final_norm_logical_bytes: int,
    ) -> Lfm2CpcfSelectiveEvidenceCatalog:
        references = tuple(phase_references)
        body = _catalog_body(
            cache_key=cache_key,
            source_sha256=source_sha256,
            provider_sha256=provider_sha256,
            selective_session_sha256=selective_session_sha256,
            layer_topology_sha256=layer_topology_sha256,
            capture_selection_sha256=capture_selection_sha256,
            source_width=source_width,
            phase_references=references,
            dynamic_catalog=dynamic_catalog,
            final_norm_evidence_sha256=final_norm_evidence_sha256,
            final_norm_logical_bytes=final_norm_logical_bytes,
        )
        return cls(
            cache_key=cache_key,
            source_sha256=source_sha256,
            provider_sha256=provider_sha256,
            selective_session_sha256=selective_session_sha256,
            layer_topology_sha256=layer_topology_sha256,
            capture_selection_sha256=capture_selection_sha256,
            source_width=source_width,
            phase_references=references,
            dynamic_catalog=dynamic_catalog,
            final_norm_evidence_sha256=final_norm_evidence_sha256,
            final_norm_logical_bytes=final_norm_logical_bytes,
            catalog_sha256=canonical_sha256(body),
        )

    def __post_init__(self) -> None:
        for value, name in (
            (self.cache_key, "cache_key"),
            (self.source_sha256, "source_sha256"),
            (self.provider_sha256, "provider_sha256"),
            (
                self.selective_session_sha256,
                "selective_session_sha256",
            ),
            (self.layer_topology_sha256, "layer_topology_sha256"),
            (
                self.capture_selection_sha256,
                "capture_selection_sha256",
            ),
            (
                self.final_norm_evidence_sha256,
                "final_norm_evidence_sha256",
            ),
        ):
            _digest(value, name)
        _positive_int(self.source_width, "source_width")
        _positive_int(
            self.final_norm_logical_bytes,
            "final_norm_logical_bytes",
        )
        if not self.phase_references or tuple(
            reference.phase_index for reference in self.phase_references
        ) != tuple(range(len(self.phase_references))):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Phase references must be nonempty and canonically ordered."
            )
        if (
            self.dynamic_catalog.source_sha256 != self.source_sha256
            or self.dynamic_catalog.provider_sha256 != self.provider_sha256
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Dynamic catalog identity differs from the LFM2 catalog."
            )
        body = self._body()
        if _digest(
            self.catalog_sha256,
            "catalog_sha256",
        ) != canonical_sha256(body):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective LFM2 catalog seal is invalid."
            )

    @property
    def layer_count(self) -> int:
        return self.dynamic_catalog.layer_count

    @property
    def phase_count(self) -> int:
        return len(self.phase_references)

    def _body(self) -> dict[str, Any]:
        return _catalog_body(
            cache_key=self.cache_key,
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            selective_session_sha256=self.selective_session_sha256,
            layer_topology_sha256=self.layer_topology_sha256,
            capture_selection_sha256=self.capture_selection_sha256,
            source_width=self.source_width,
            phase_references=self.phase_references,
            dynamic_catalog=self.dynamic_catalog,
            final_norm_evidence_sha256=self.final_norm_evidence_sha256,
            final_norm_logical_bytes=self.final_norm_logical_bytes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "phase_references": [
                reference.as_dict() for reference in self.phase_references
            ],
            "catalog_sha256": self.catalog_sha256,
        }


def _catalog_body(
    *,
    cache_key: str,
    source_sha256: str,
    provider_sha256: str,
    selective_session_sha256: str,
    layer_topology_sha256: str,
    capture_selection_sha256: str,
    source_width: int,
    phase_references: Sequence[Lfm2CpcfSelectivePhaseReference],
    dynamic_catalog: DynamicEvidenceLayerCatalog,
    final_norm_evidence_sha256: str,
    final_norm_logical_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SELECTIVE_CATALOG_SCHEMA,
        "cache_key": cache_key,
        "source_sha256": source_sha256,
        "provider_sha256": provider_sha256,
        "selective_session_sha256": selective_session_sha256,
        "layer_topology_sha256": layer_topology_sha256,
        "capture_selection_sha256": capture_selection_sha256,
        "source_width": source_width,
        "layer_count": dynamic_catalog.layer_count,
        "phase_count": len(phase_references),
        "ordered_phase_reference_sha256": [
            reference.reference_sha256
            for reference in phase_references
        ],
        "dynamic_catalog_sha256": dynamic_catalog.catalog_sha256,
        "final_norm_evidence_sha256": final_norm_evidence_sha256,
        "final_norm_logical_bytes": final_norm_logical_bytes,
        "tensor_payloads_resident": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }


@dataclass(frozen=True, slots=True)
class _MetadataCacheView:
    cache_key: str
    batch_count: int
    manifest: Mapping[str, Any]

    def verify(self) -> _MetadataCacheView:
        return self


def _layer_aggregate_body(
    *,
    source_sha256: str,
    provider_sha256: str,
    layer_index: int,
    operator_kind: str,
    phase_rows: Sequence[Mapping[str, Any]],
    logical_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SELECTIVE_LAYER_PAYLOAD_SCHEMA,
        "source_sha256": source_sha256,
        "provider_sha256": provider_sha256,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "phase_count": len(phase_rows),
        "ordered_phases": [dict(row) for row in phase_rows],
        "logical_bytes": logical_bytes,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "arithmetic_conversion": False,
    }


def _final_norm_phase_body(
    *,
    cache_key: str,
    batch_index: int,
    input_sha256: str,
    phase: str,
    inventory: Mapping[str, Any],
    logical_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SELECTIVE_FINAL_NORM_SCHEMA,
        "cache_key": cache_key,
        "batch_index": batch_index,
        "input_sha256": input_sha256,
        "phase": phase,
        "field_inventory": dict(inventory),
        "logical_bytes": logical_bytes,
        "final_norm_to_lm_head_identity_verified": True,
    }


def _final_norm_aggregate_body(
    *,
    source_sha256: str,
    provider_sha256: str,
    phase_rows: Sequence[Mapping[str, Any]],
    logical_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": LFM2_CPCF_SELECTIVE_FINAL_NORM_SCHEMA,
        "scope": "all-canonical-phases",
        "source_sha256": source_sha256,
        "provider_sha256": provider_sha256,
        "phase_count": len(phase_rows),
        "ordered_phases": [dict(row) for row in phase_rows],
        "logical_bytes": logical_bytes,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "arithmetic_conversion": False,
    }


class Lfm2CpcfSelectiveEvidenceProvider:
    """Dynamic residency provider over one immutable selective cache session."""

    def __init__(
        self,
        *,
        session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
        adapter: installability.Lfm2CpcfInstallabilityAdapter,
        phase_plans: Sequence[_PhasePlan],
        catalog: Lfm2CpcfSelectiveEvidenceCatalog | None,
        source_sha256: str,
        provider_sha256: str,
        catalog_artifact: DynamicEvidenceCatalogArtifact | None = None,
    ) -> None:
        self._session = session
        self._adapter = adapter
        self._phase_plans = tuple(phase_plans)
        self._catalog = catalog
        self._catalog_artifact = catalog_artifact
        self.source_sha256 = _digest(source_sha256, "source_sha256")
        self.provider_sha256 = _digest(
            provider_sha256,
            "provider_sha256",
        )
        self._closed = False

    @property
    def catalog(self) -> Lfm2CpcfSelectiveEvidenceCatalog:
        if self._catalog is None:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective provider catalog is not finalized."
            )
        return self._catalog

    @property
    def dynamic_catalog(self) -> DynamicEvidenceLayerCatalog:
        return self.catalog.dynamic_catalog

    @property
    def selective_session_evidence(self) -> Mapping[str, Any]:
        return MappingProxyType(self._session.as_dict())

    @property
    def catalog_artifact(
        self,
    ) -> DynamicEvidenceCatalogArtifact | None:
        return self._catalog_artifact

    @property
    def catalog_artifact_receipt(self) -> Mapping[str, Any] | None:
        if self._catalog_artifact is None:
            return None
        return self._catalog_artifact.as_dict()

    def _assert_open(self) -> None:
        if self._closed:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective evidence provider is closed."
            )

    def _plans_by_batch(self) -> tuple[tuple[_PhasePlan, ...], ...]:
        result: list[list[_PhasePlan]] = [
            [] for _ in range(self._session.batch_count)
        ]
        for plan in self._phase_plans:
            result[plan.batch_index].append(plan)
        return tuple(tuple(values) for values in result)

    def _load_selected_batch(
        self,
        *,
        batch_index: int,
        names: Sequence[str],
    ) -> Lfm2CircuitTraceSelectedBatchPayload:
        return self._session.load_selected_tensors(
            batch_index,
            tuple(sorted(names)),
        )

    def _build_layer_payload(
        self,
        layer_index: int,
    ) -> Lfm2CpcfSelectiveLayerPayload:
        self._assert_open()
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index >= len(self._adapter.topology)
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "layer_index is outside the LFM2 topology."
            )
        names = _layer_names(self._adapter.topology, layer_index)
        operator_kind = str(
            self._adapter.topology[layer_index]["contract"]["kind"]
        )
        phases: list[_LayerPhasePayload] = []
        rows: list[dict[str, Any]] = []
        logical_bytes = 0
        for batch_index, plans in enumerate(self._plans_by_batch()):
            full_names = tuple(
                f"{plan.phase}.layer.{layer_index:03d}.{name}"
                for plan in plans
                for name in names
            )
            selected = self._load_selected_batch(
                batch_index=batch_index,
                names=full_names,
            )
            try:
                for plan in plans:
                    evidence = self._adapter._layer_evidence(
                        selected.tensors,
                        plan.diagnostics,
                        batch_index=plan.batch_index,
                        phase=plan.phase,
                        layer_index=layer_index,
                    )
                    phase_bytes = _logical_tensor_bytes(evidence.tensors)
                    inventory_sha256 = _field_inventory_sha256(
                        evidence.tensors
                    )
                    phase_reference_sha256 = (
                        ""
                        if self._catalog is None
                        else self._catalog.phase_references[
                            plan.phase_index
                        ].reference_sha256
                    )
                    phases.append(
                        _LayerPhasePayload(
                            phase_reference_sha256=(
                                phase_reference_sha256
                            ),
                            evidence=evidence,
                        )
                    )
                    rows.append(
                        {
                            "phase_index": plan.phase_index,
                            "batch_index": plan.batch_index,
                            "input_sha256": plan.input_sha256,
                            "phase": plan.phase,
                            "phase_reference_sha256": (
                                phase_reference_sha256
                            ),
                            "layer_evidence_sha256": (
                                evidence.evidence_sha256
                            ),
                            "field_inventory_sha256": inventory_sha256,
                            "logical_bytes": phase_bytes,
                        }
                    )
                    logical_bytes += phase_bytes
            finally:
                self._session.release_selected_tensors(selected)
        body = _layer_aggregate_body(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            layer_index=layer_index,
            operator_kind=operator_kind,
            phase_rows=rows,
            logical_bytes=logical_bytes,
        )
        return Lfm2CpcfSelectiveLayerPayload(
            layer_index=layer_index,
            operator_kind=operator_kind,
            phases=phases,
            evidence_sha256=canonical_sha256(body),
            logical_bytes=logical_bytes,
        )

    def _layer_rows(
        self,
        payload: Lfm2CpcfSelectiveLayerPayload,
    ) -> tuple[dict[str, Any], ...]:
        if len(payload.phases) != len(self._phase_plans):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective layer payload phase count changed."
            )
        rows = []
        logical_bytes = 0
        for plan, phase_payload in zip(
            self._phase_plans,
            payload.phases,
            strict=True,
        ):
            evidence = phase_payload.evidence
            if (
                evidence.layer_index != payload.layer_index
                or evidence.operator_kind != payload.operator_kind
            ):
                raise Lfm2CpcfSelectiveEvidenceError(
                    "Selective layer payload identity changed."
                )
            inventory = installability._field_inventory(evidence.tensors)
            body = {
                "schema": installability.LFM2_CPCF_INSTALLABLE_LAYER_SCHEMA,
                "cache_key": self._adapter.cache_key,
                "batch_index": plan.batch_index,
                "phase": plan.phase,
                "layer_index": payload.layer_index,
                "operator_kind": payload.operator_kind,
                "source_width": self._adapter.source_width,
                "field_inventory": inventory,
                "functional_identities_verified": True,
            }
            existing_sha256 = installability._sha(body)
            phase_bytes = _logical_tensor_bytes(evidence.tensors)
            rows.append(
                {
                    "phase_index": plan.phase_index,
                    "batch_index": plan.batch_index,
                    "input_sha256": plan.input_sha256,
                    "phase": plan.phase,
                    "phase_reference_sha256": (
                        phase_payload.phase_reference_sha256
                    ),
                    "layer_evidence_sha256": existing_sha256,
                    "field_inventory_sha256": canonical_sha256(inventory),
                    "logical_bytes": phase_bytes,
                }
            )
            logical_bytes += phase_bytes
        if logical_bytes != payload.logical_bytes:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective layer payload logical bytes changed."
            )
        return tuple(rows)

    def load_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
    ) -> Lfm2CpcfSelectiveLayerPayload:
        self._assert_open()
        if not isinstance(reference, DynamicEvidenceLayerReference):
            raise TypeError(
                "reference must be DynamicEvidenceLayerReference."
            )
        expected = self.dynamic_catalog.references[
            reference.layer_index
        ]
        if reference != expected:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Requested layer reference differs from the sealed catalog."
            )
        payload = self._build_layer_payload(reference.layer_index)
        try:
            self._assert_payload_matches(reference, payload)
        except BaseException:
            payload._release()
            raise
        return payload

    def _assert_payload_matches(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Lfm2CpcfSelectiveLayerPayload,
    ) -> tuple[dict[str, Any], ...]:
        rows = self._layer_rows(payload)
        for plan, row in zip(self._phase_plans, rows, strict=True):
            phase_reference = self.catalog.phase_references[
                plan.phase_index
            ]
            layer_index = payload.layer_index
            if (
                row["phase_reference_sha256"]
                != phase_reference.reference_sha256
                or row["layer_evidence_sha256"]
                != phase_reference.layer_evidence_sha256[layer_index]
                or row["field_inventory_sha256"]
                != phase_reference.layer_field_inventory_sha256[layer_index]
                or row["logical_bytes"]
                != phase_reference.layer_logical_bytes[layer_index]
                or payload.operator_kind
                != phase_reference.layer_operator_kinds[layer_index]
            ):
                raise Lfm2CpcfSelectiveEvidenceError(
                    "Selective layer payload differs from its tensor-free "
                    "phase reference."
                )
        body = _layer_aggregate_body(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            layer_index=payload.layer_index,
            operator_kind=payload.operator_kind,
            phase_rows=rows,
            logical_bytes=payload.logical_bytes,
        )
        evidence_sha256 = canonical_sha256(body)
        if (
            evidence_sha256 != payload.evidence_sha256
            or evidence_sha256 != reference.evidence_sha256
            or payload.logical_bytes != reference.logical_bytes
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective layer payload does not match its dynamic reference."
            )
        return rows

    def authenticate_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Lfm2CpcfSelectiveLayerPayload,
    ) -> DynamicEvidenceLayerAuthentication:
        self._assert_open()
        if not isinstance(payload, Lfm2CpcfSelectiveLayerPayload):
            raise TypeError(
                "payload must be Lfm2CpcfSelectiveLayerPayload."
            )
        self._assert_payload_matches(reference, payload)
        return DynamicEvidenceLayerAuthentication.create(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            layer_index=payload.layer_index,
            reference_sha256=reference.reference_sha256,
            evidence_sha256=payload.evidence_sha256,
            logical_bytes=payload.logical_bytes,
        )

    def release_layer_payload(
        self,
        reference: DynamicEvidenceLayerReference,
        payload: Lfm2CpcfSelectiveLayerPayload,
    ) -> None:
        if not isinstance(reference, DynamicEvidenceLayerReference):
            raise TypeError(
                "reference must be DynamicEvidenceLayerReference."
            )
        if not isinstance(payload, Lfm2CpcfSelectiveLayerPayload):
            raise TypeError(
                "payload must be Lfm2CpcfSelectiveLayerPayload."
            )
        if payload.layer_index != reference.layer_index:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Cannot release a payload through another layer reference."
            )
        payload._release()

    def _build_final_norm_payload(
        self,
        *,
        expected_last_shapes: Sequence[Sequence[int]],
    ) -> Lfm2CpcfSelectiveFinalNormPayload:
        phases: list[_FinalNormPhasePayload] = []
        rows: list[dict[str, Any]] = []
        logical_bytes = 0
        for batch_index, plans in enumerate(self._plans_by_batch()):
            full_names = tuple(
                f"{plan.phase}.{name}"
                for plan in plans
                for name in installability._OUTPUT_FIELDS
            )
            selected = self._load_selected_batch(
                batch_index=batch_index,
                names=full_names,
            )
            try:
                for plan in plans:
                    output_diagnostics = plan.diagnostics.get("output")
                    if (
                        not isinstance(output_diagnostics, Mapping)
                        or output_diagnostics.get(
                            "final_norm_boundary_captured"
                        )
                        is not True
                        or output_diagnostics.get(
                            "lm_head_input_exactly_verified"
                        )
                        is not True
                    ):
                        raise Lfm2CpcfSelectiveEvidenceError(
                            "Final-normalization diagnostics are not "
                            "authenticated."
                        )
                    output = installability._copy_fields(
                        selected.tensors,
                        f"{plan.phase}.",
                        installability._OUTPUT_FIELDS,
                        batch_index=plan.batch_index,
                        phase=plan.phase,
                    )
                    normalized = {
                        name: installability._tensor(
                            output[name],
                            name,
                            dimensions=3,
                        )
                        for name in installability._OUTPUT_FIELDS
                    }
                    expected_shape = tuple(
                        int(value)
                        for value in expected_last_shapes[
                            plan.phase_index
                        ]
                    )
                    if any(
                        tuple(value.shape) != expected_shape
                        for value in normalized.values()
                    ):
                        raise Lfm2CpcfSelectiveEvidenceError(
                            "Final-normalization boundary changed shape."
                        )
                    installability._close(
                        normalized["output.final_norm_output"],
                        normalized["output.lm_head_input"],
                        "final RMSNorm output to LM-head input",
                        exact=True,
                    )
                    inventory = installability._field_inventory(output)
                    phase_bytes = _logical_tensor_bytes(output)
                    phase_body = _final_norm_phase_body(
                        cache_key=self._adapter.cache_key,
                        batch_index=plan.batch_index,
                        input_sha256=plan.input_sha256,
                        phase=plan.phase,
                        inventory=inventory,
                        logical_bytes=phase_bytes,
                    )
                    evidence_sha256 = canonical_sha256(phase_body)
                    phase_reference_sha256 = (
                        ""
                        if self._catalog is None
                        else self._catalog.phase_references[
                            plan.phase_index
                        ].reference_sha256
                    )
                    phases.append(
                        _FinalNormPhasePayload(
                            phase_reference_sha256=(
                                phase_reference_sha256
                            ),
                            batch_index=plan.batch_index,
                            input_sha256=plan.input_sha256,
                            phase=plan.phase,
                            tensors=output,
                            evidence_sha256=evidence_sha256,
                            logical_bytes=phase_bytes,
                        )
                    )
                    rows.append(
                        {
                            "phase_index": plan.phase_index,
                            "batch_index": plan.batch_index,
                            "input_sha256": plan.input_sha256,
                            "phase": plan.phase,
                            "phase_reference_sha256": (
                                phase_reference_sha256
                            ),
                            "evidence_sha256": evidence_sha256,
                            "field_inventory_sha256": canonical_sha256(
                                inventory
                            ),
                            "logical_bytes": phase_bytes,
                        }
                    )
                    logical_bytes += phase_bytes
            finally:
                self._session.release_selected_tensors(selected)
        aggregate = _final_norm_aggregate_body(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            phase_rows=rows,
            logical_bytes=logical_bytes,
        )
        return Lfm2CpcfSelectiveFinalNormPayload(
            phases=phases,
            evidence_sha256=canonical_sha256(aggregate),
            logical_bytes=logical_bytes,
        )

    def _final_norm_rows(
        self,
        payload: Lfm2CpcfSelectiveFinalNormPayload,
    ) -> tuple[dict[str, Any], ...]:
        if len(payload.phases) != len(self._phase_plans):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Final-norm payload phase count changed."
            )
        rows = []
        logical_bytes = 0
        for plan, phase_payload in zip(
            self._phase_plans,
            payload.phases,
            strict=True,
        ):
            if (
                phase_payload.batch_index != plan.batch_index
                or phase_payload.input_sha256 != plan.input_sha256
                or phase_payload.phase != plan.phase
            ):
                raise Lfm2CpcfSelectiveEvidenceError(
                    "Final-norm payload canonical order changed."
                )
            inventory = installability._field_inventory(
                phase_payload.tensors
            )
            phase_bytes = _logical_tensor_bytes(phase_payload.tensors)
            phase_body = _final_norm_phase_body(
                cache_key=self._adapter.cache_key,
                batch_index=plan.batch_index,
                input_sha256=plan.input_sha256,
                phase=plan.phase,
                inventory=inventory,
                logical_bytes=phase_bytes,
            )
            rows.append(
                {
                    "phase_index": plan.phase_index,
                    "batch_index": plan.batch_index,
                    "input_sha256": plan.input_sha256,
                    "phase": plan.phase,
                    "phase_reference_sha256": (
                        phase_payload.phase_reference_sha256
                    ),
                    "evidence_sha256": canonical_sha256(phase_body),
                    "field_inventory_sha256": canonical_sha256(inventory),
                    "logical_bytes": phase_bytes,
                }
            )
            logical_bytes += phase_bytes
        if logical_bytes != payload.logical_bytes:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Final-norm payload logical bytes changed."
            )
        return tuple(rows)

    def load_final_norm_payload(
        self,
    ) -> Lfm2CpcfSelectiveFinalNormPayload:
        self._assert_open()
        # Shape values are authenticated again from the last-layer record
        # descriptors. They are metadata-only and never load last-layer data.
        last_layer = self.catalog.layer_count - 1
        last_name = (
            f"layer.{last_layer:03d}.residual.post_layer"
        )
        shapes: list[Sequence[int]] = []
        for plan in self._phase_plans:
            record = self._session.batch_references[
                plan.batch_index
            ].record
            descriptor = record["tensors"][
                f"{plan.phase}.{last_name}"
            ]
            shapes.append(tuple(descriptor["shape"]))
        payload = self._build_final_norm_payload(
            expected_last_shapes=shapes
        )
        try:
            self.authenticate_final_norm_payload(payload)
        except BaseException:
            payload._release()
            raise
        return payload

    def authenticate_final_norm_payload(
        self,
        payload: Lfm2CpcfSelectiveFinalNormPayload,
    ) -> str:
        self._assert_open()
        if not isinstance(payload, Lfm2CpcfSelectiveFinalNormPayload):
            raise TypeError(
                "payload must be Lfm2CpcfSelectiveFinalNormPayload."
            )
        rows = self._final_norm_rows(payload)
        for plan, row in zip(self._phase_plans, rows, strict=True):
            reference = self.catalog.phase_references[plan.phase_index]
            if (
                row["phase_reference_sha256"]
                != reference.reference_sha256
                or row["evidence_sha256"]
                != reference.final_norm_evidence_sha256
                or row["field_inventory_sha256"]
                != reference.final_norm_field_inventory_sha256
                or row["logical_bytes"]
                != reference.final_norm_logical_bytes
            ):
                raise Lfm2CpcfSelectiveEvidenceError(
                    "Final-norm payload differs from its phase reference."
                )
        aggregate = _final_norm_aggregate_body(
            source_sha256=self.source_sha256,
            provider_sha256=self.provider_sha256,
            phase_rows=rows,
            logical_bytes=payload.logical_bytes,
        )
        evidence_sha256 = canonical_sha256(aggregate)
        if (
            evidence_sha256 != payload.evidence_sha256
            or evidence_sha256
            != self.catalog.final_norm_evidence_sha256
            or payload.logical_bytes
            != self.catalog.final_norm_logical_bytes
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Final-norm payload does not match the catalog aggregate."
            )
        return evidence_sha256

    def release_final_norm_payload(
        self,
        payload: Lfm2CpcfSelectiveFinalNormPayload,
    ) -> None:
        if not isinstance(payload, Lfm2CpcfSelectiveFinalNormPayload):
            raise TypeError(
                "payload must be Lfm2CpcfSelectiveFinalNormPayload."
            )
        payload._release()

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        if self._catalog_artifact is not None:
            try:
                self._catalog_artifact.verify()
            except BaseException as exc:
                failures.append(exc)
        try:
            self._session.close()
        except BaseException as exc:
            failures.append(exc)
        self._closed = True
        if failures:
            failure = failures[0]
            for extra in failures[1:]:
                failure.add_note(
                    "Selective provider close also failed: "
                    f"{type(extra).__qualname__}: {extra}"
                )
            raise failure

    def __enter__(self) -> Lfm2CpcfSelectiveEvidenceProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


def _phase_plans(
    session: Lfm2CircuitTraceSelectiveAuthenticatedSession,
    adapter: installability.Lfm2CpcfInstallabilityAdapter,
) -> tuple[_PhasePlan, ...]:
    result = []
    for batch_reference in session.batch_references:
        batch_index = batch_reference.index
        if (
            batch_reference.input_sha256
            != adapter.ordered_input_sha256[batch_index]
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                f"Cache batch {batch_index} input identity changed."
            )
        record = batch_reference.record
        phases = installability._phase_names(record, batch_index)
        diagnostics = adapter._validate_capture_record(
            record,
            batch_index=batch_index,
            phases=phases,
        )
        for phase in phases:
            phase_diagnostics = diagnostics.get(phase)
            if not isinstance(phase_diagnostics, Mapping):
                raise Lfm2CpcfSelectiveEvidenceError(
                    f"Cache batch {batch_index} phase {phase!r} has no "
                    "diagnostics."
                )
            result.append(
                _PhasePlan(
                    phase_index=len(result),
                    batch_index=batch_index,
                    input_sha256=batch_reference.input_sha256,
                    phase=phase,
                    diagnostics=MappingProxyType(
                        dict(phase_diagnostics)
                    ),
                )
            )
    if not result:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective evidence cache has no canonical phases."
        )
    return tuple(result)


@dataclass(slots=True)
class Lfm2CpcfAuthenticatedSelectiveSource:
    """One authenticated trace session prepared for cold or warm catalogs."""

    session: Lfm2CircuitTraceSelectiveAuthenticatedSession
    adapter: installability.Lfm2CpcfInstallabilityAdapter
    phase_plans: tuple[_PhasePlan, ...]
    implementation: Mapping[str, Any]
    source_sha256: str
    provider_sha256: str
    _claimed: bool = False
    _closed: bool = False

    @property
    def layer_count(self) -> int:
        return len(self.adapter.topology)

    @property
    def phase_count(self) -> int:
        return len(self.phase_plans)

    def _claim_session(self) -> None:
        if self._closed or self._claimed:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Authenticated selective source session is unavailable."
            )
        self._claimed = True

    def close(self) -> None:
        if self._closed:
            return
        if not self._claimed:
            self.session.close()
        self._closed = True

    def __enter__(self) -> Lfm2CpcfAuthenticatedSelectiveSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


def authenticate_lfm2_cpcf_selective_source(
    cache: Lfm2CircuitTraceCache,
) -> Lfm2CpcfAuthenticatedSelectiveSource:
    """Hash and pin one local trace cache, without building layer catalogs."""

    if not isinstance(cache, Lfm2CircuitTraceCache):
        raise TypeError("cache must be Lfm2CircuitTraceCache.")
    session = cache.selective_authenticated_session()
    try:
        metadata_view = _MetadataCacheView(
            cache_key=session.cache_key,
            batch_count=session.batch_count,
            manifest=MappingProxyType(session.manifest),
        )
        adapter = installability.Lfm2CpcfInstallabilityAdapter(
            metadata_view
        )
        plans = _phase_plans(session, adapter)
        implementation = lfm2_cpcf_selective_provider_implementation()
        provider_sha256 = str(implementation["implementation_sha256"])
        session_evidence = session.as_dict()
        source_body = {
            "schema": LFM2_CPCF_SELECTIVE_SOURCE_SCHEMA,
            "cache_key": session.cache_key,
            "manifest_sha256": session_evidence["manifest_sha256"],
            "completion_sha256": session_evidence["completion_sha256"],
            "selective_session_sha256": session.evidence_sha256,
            "layer_topology_sha256": adapter.layer_topology_sha256,
            "capture_selection_sha256": (
                adapter.capture_selection_sha256
            ),
        }
        return Lfm2CpcfAuthenticatedSelectiveSource(
            session=session,
            adapter=adapter,
            phase_plans=plans,
            implementation=implementation,
            source_sha256=canonical_sha256(source_body),
            provider_sha256=provider_sha256,
        )
    except BaseException:
        session.close()
        raise


def _validate_catalog_against_source(
    source: Lfm2CpcfAuthenticatedSelectiveSource,
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
) -> None:
    if not isinstance(source, Lfm2CpcfAuthenticatedSelectiveSource):
        raise TypeError(
            "source must be Lfm2CpcfAuthenticatedSelectiveSource."
        )
    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise TypeError(
            "catalog must be Lfm2CpcfSelectiveEvidenceCatalog."
        )
    adapter = source.adapter
    session = source.session
    plans = source.phase_plans
    expected_operator_kinds = tuple(
        str(layer["contract"]["kind"]) for layer in adapter.topology
    )
    if (
        catalog.cache_key != session.cache_key
        or catalog.source_sha256 != source.source_sha256
        or catalog.provider_sha256 != source.provider_sha256
        or catalog.selective_session_sha256 != session.evidence_sha256
        or catalog.layer_topology_sha256
        != adapter.layer_topology_sha256
        or catalog.capture_selection_sha256
        != adapter.capture_selection_sha256
        or catalog.source_width != adapter.source_width
        or catalog.layer_count != len(adapter.topology)
        or catalog.phase_count != len(plans)
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective catalog identity differs from its authenticated source."
        )
    for plan, reference in zip(
        plans,
        catalog.phase_references,
        strict=True,
    ):
        if (
            reference.phase_index != plan.phase_index
            or reference.cache_key != adapter.cache_key
            or reference.batch_index != plan.batch_index
            or reference.input_sha256 != plan.input_sha256
            or reference.phase != plan.phase
            or reference.layer_operator_kinds
            != expected_operator_kinds
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Selective catalog phase order differs from its source."
            )
    for layer_index, dynamic_reference in enumerate(
        catalog.dynamic_catalog.references
    ):
        phase_rows = [
            {
                "phase_index": plan.phase_index,
                "batch_index": plan.batch_index,
                "input_sha256": plan.input_sha256,
                "phase": plan.phase,
                "phase_reference_sha256": catalog.phase_references[
                    plan.phase_index
                ].reference_sha256,
                "layer_evidence_sha256": catalog.phase_references[
                    plan.phase_index
                ].layer_evidence_sha256[layer_index],
                "field_inventory_sha256": catalog.phase_references[
                    plan.phase_index
                ].layer_field_inventory_sha256[layer_index],
                "logical_bytes": catalog.phase_references[
                    plan.phase_index
                ].layer_logical_bytes[layer_index],
            }
            for plan in plans
        ]
        logical_bytes = sum(
            int(row["logical_bytes"]) for row in phase_rows
        )
        aggregate = _layer_aggregate_body(
            source_sha256=source.source_sha256,
            provider_sha256=source.provider_sha256,
            layer_index=layer_index,
            operator_kind=expected_operator_kinds[layer_index],
            phase_rows=phase_rows,
            logical_bytes=logical_bytes,
        )
        if (
            dynamic_reference.evidence_sha256
            != canonical_sha256(aggregate)
            or dynamic_reference.logical_bytes != logical_bytes
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Dynamic layer catalog does not reconstruct from phase seals."
            )
    final_rows = [
        {
            "phase_index": plan.phase_index,
            "batch_index": plan.batch_index,
            "input_sha256": plan.input_sha256,
            "phase": plan.phase,
            "phase_reference_sha256": catalog.phase_references[
                plan.phase_index
            ].reference_sha256,
            "evidence_sha256": catalog.phase_references[
                plan.phase_index
            ].final_norm_evidence_sha256,
            "field_inventory_sha256": catalog.phase_references[
                plan.phase_index
            ].final_norm_field_inventory_sha256,
            "logical_bytes": catalog.phase_references[
                plan.phase_index
            ].final_norm_logical_bytes,
        }
        for plan in plans
    ]
    final_bytes = sum(int(row["logical_bytes"]) for row in final_rows)
    final_aggregate = _final_norm_aggregate_body(
        source_sha256=source.source_sha256,
        provider_sha256=source.provider_sha256,
        phase_rows=final_rows,
        logical_bytes=final_bytes,
    )
    if (
        catalog.final_norm_evidence_sha256
        != canonical_sha256(final_aggregate)
        or catalog.final_norm_logical_bytes != final_bytes
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Final-norm catalog does not reconstruct from phase seals."
        )
    session.verify()


def build_lfm2_cpcf_selective_provider_from_authenticated_catalog(
    source: Lfm2CpcfAuthenticatedSelectiveSource,
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    *,
    catalog_artifact: DynamicEvidenceCatalogArtifact | None = None,
) -> Lfm2CpcfSelectiveEvidenceProvider:
    """Construct the live provider from a typed, source-bound catalog."""

    _validate_catalog_against_source(source, catalog)
    if catalog_artifact is not None:
        if not isinstance(
            catalog_artifact,
            DynamicEvidenceCatalogArtifact,
        ):
            raise TypeError(
                "catalog_artifact must be DynamicEvidenceCatalogArtifact."
            )
        expected_binding = lfm2_cpcf_selective_catalog_artifact_binding(
            source
        )
        if catalog_artifact.binding != expected_binding:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Catalog artifact binding differs from the source."
            )
        catalog_artifact.verify()
        if dict(catalog_artifact.payload) != (
            lfm2_cpcf_selective_catalog_to_artifact_payload(catalog)
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Catalog artifact payload differs from the typed catalog."
            )
    source._claim_session()
    return Lfm2CpcfSelectiveEvidenceProvider(
        session=source.session,
        adapter=source.adapter,
        phase_plans=source.phase_plans,
        catalog=catalog,
        source_sha256=source.source_sha256,
        provider_sha256=source.provider_sha256,
        catalog_artifact=catalog_artifact,
    )


def _build_catalog_from_authenticated_source(
    source: Lfm2CpcfAuthenticatedSelectiveSource,
) -> Lfm2CpcfSelectiveEvidenceProvider:
    """Run the original cold tensor sweep using an existing auth session."""

    session = source.session
    adapter = source.adapter
    plans = source.phase_plans
    source_sha256 = source.source_sha256
    provider_sha256 = source.provider_sha256
    builder = Lfm2CpcfSelectiveEvidenceProvider(
        session=session,
        adapter=adapter,
        phase_plans=plans,
        catalog=None,
        source_sha256=source_sha256,
        provider_sha256=provider_sha256,
    )
    phase_layer_sha = [
        ["" for _ in adapter.topology] for _ in plans
    ]
    phase_layer_inventory_sha = [
        ["" for _ in adapter.topology] for _ in plans
    ]
    phase_layer_bytes = [
        [0 for _ in adapter.topology] for _ in plans
    ]
    phase_operator_kinds = [
        ["" for _ in adapter.topology] for _ in plans
    ]
    last_layer_shapes: list[Sequence[int] | None] = [
        None for _ in plans
    ]
    for layer_index in range(len(adapter.topology)):
        payload = builder._build_layer_payload(layer_index)
        try:
            rows = builder._layer_rows(payload)
            for plan, row, phase_payload in zip(
                plans,
                rows,
                payload.phases,
                strict=True,
            ):
                phase_layer_sha[plan.phase_index][layer_index] = str(
                    row["layer_evidence_sha256"]
                )
                phase_layer_inventory_sha[
                    plan.phase_index
                ][layer_index] = str(
                    row["field_inventory_sha256"]
                )
                phase_layer_bytes[
                    plan.phase_index
                ][layer_index] = int(row["logical_bytes"])
                phase_operator_kinds[
                    plan.phase_index
                ][layer_index] = payload.operator_kind
                if layer_index == len(adapter.topology) - 1:
                    last_layer_shapes[plan.phase_index] = tuple(
                        phase_payload.evidence.tensors[
                            "residual.post_layer"
                        ].shape
                    )
        finally:
            payload._release()
    if any(shape is None for shape in last_layer_shapes):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Last-layer shapes are incomplete."
        )
    final_payload = builder._build_final_norm_payload(
        expected_last_shapes=[
            tuple(shape or ()) for shape in last_layer_shapes
        ]
    )
    try:
        final_rows = builder._final_norm_rows(final_payload)
        final_aggregate_bytes = final_payload.logical_bytes
        phase_references = []
        for plan, final_row, final_phase_payload in zip(
            plans,
            final_rows,
            final_payload.phases,
            strict=True,
        ):
            final_inventory = installability._field_inventory(
                final_phase_payload.tensors
            )
            existing_phase_body = {
                "schema": (
                    installability.LFM2_CPCF_INSTALLABLE_PHASE_SCHEMA
                ),
                "cache_key": adapter.cache_key,
                "batch_index": plan.batch_index,
                "input_sha256": plan.input_sha256,
                "phase": plan.phase,
                "layer_evidence_sha256": list(
                    phase_layer_sha[plan.phase_index]
                ),
                "final_norm_field_inventory": final_inventory,
                "parent_side_compiler_inputs_ready": True,
            }
            existing_phase_sha256 = installability._sha(
                existing_phase_body
            )
            body = _phase_reference_body(
                phase_index=plan.phase_index,
                cache_key=adapter.cache_key,
                batch_index=plan.batch_index,
                input_sha256=plan.input_sha256,
                phase=plan.phase,
                layer_operator_kinds=phase_operator_kinds[
                    plan.phase_index
                ],
                layer_evidence_sha256=phase_layer_sha[
                    plan.phase_index
                ],
                layer_field_inventory_sha256=(
                    phase_layer_inventory_sha[plan.phase_index]
                ),
                layer_logical_bytes=phase_layer_bytes[
                    plan.phase_index
                ],
                final_norm_evidence_sha256=str(
                    final_row["evidence_sha256"]
                ),
                final_norm_field_inventory_sha256=str(
                    final_row["field_inventory_sha256"]
                ),
                final_norm_logical_bytes=int(
                    final_row["logical_bytes"]
                ),
                existing_phase_evidence_sha256=(
                    existing_phase_sha256
                ),
            )
            phase_references.append(
                Lfm2CpcfSelectivePhaseReference(
                    phase_index=plan.phase_index,
                    cache_key=adapter.cache_key,
                    batch_index=plan.batch_index,
                    input_sha256=plan.input_sha256,
                    phase=plan.phase,
                    layer_operator_kinds=tuple(
                        phase_operator_kinds[plan.phase_index]
                    ),
                    layer_evidence_sha256=tuple(
                        phase_layer_sha[plan.phase_index]
                    ),
                    layer_field_inventory_sha256=tuple(
                        phase_layer_inventory_sha[plan.phase_index]
                    ),
                    layer_logical_bytes=tuple(
                        phase_layer_bytes[plan.phase_index]
                    ),
                    final_norm_evidence_sha256=str(
                        final_row["evidence_sha256"]
                    ),
                    final_norm_field_inventory_sha256=str(
                        final_row["field_inventory_sha256"]
                    ),
                    final_norm_logical_bytes=int(
                        final_row["logical_bytes"]
                    ),
                    existing_phase_evidence_sha256=(
                        existing_phase_sha256
                    ),
                    reference_sha256=canonical_sha256(body),
                )
            )
    finally:
        final_payload._release()
    rebuilt_layer_sha: list[str] = []
    rebuilt_layer_bytes: list[int] = []
    for layer_index in range(len(adapter.topology)):
        rows = [
            {
                "phase_index": plan.phase_index,
                "batch_index": plan.batch_index,
                "input_sha256": plan.input_sha256,
                "phase": plan.phase,
                "phase_reference_sha256": phase_references[
                    plan.phase_index
                ].reference_sha256,
                "layer_evidence_sha256": phase_layer_sha[
                    plan.phase_index
                ][layer_index],
                "field_inventory_sha256": (
                    phase_layer_inventory_sha[plan.phase_index][
                        layer_index
                    ]
                ),
                "logical_bytes": phase_layer_bytes[
                    plan.phase_index
                ][layer_index],
            }
            for plan in plans
        ]
        logical_bytes = sum(
            phase_layer_bytes[plan.phase_index][layer_index]
            for plan in plans
        )
        body = _layer_aggregate_body(
            source_sha256=source_sha256,
            provider_sha256=provider_sha256,
            layer_index=layer_index,
            operator_kind=phase_operator_kinds[0][layer_index],
            phase_rows=rows,
            logical_bytes=logical_bytes,
        )
        rebuilt_layer_sha.append(canonical_sha256(body))
        rebuilt_layer_bytes.append(logical_bytes)
    dynamic_catalog = DynamicEvidenceLayerCatalog.create(
        source_sha256=source_sha256,
        provider_sha256=provider_sha256,
        evidence_sha256=rebuilt_layer_sha,
        logical_bytes=rebuilt_layer_bytes,
    )
    final_rows = [
        {
            **dict(row),
            "phase_reference_sha256": phase_references[
                int(row["phase_index"])
            ].reference_sha256,
        }
        for row in final_rows
    ]
    final_body = _final_norm_aggregate_body(
        source_sha256=source_sha256,
        provider_sha256=provider_sha256,
        phase_rows=final_rows,
        logical_bytes=final_aggregate_bytes,
    )
    catalog = Lfm2CpcfSelectiveEvidenceCatalog.create(
        cache_key=adapter.cache_key,
        source_sha256=source_sha256,
        provider_sha256=provider_sha256,
        selective_session_sha256=session.evidence_sha256,
        layer_topology_sha256=adapter.layer_topology_sha256,
        capture_selection_sha256=adapter.capture_selection_sha256,
        source_width=adapter.source_width,
        phase_references=tuple(phase_references),
        dynamic_catalog=dynamic_catalog,
        final_norm_evidence_sha256=canonical_sha256(final_body),
        final_norm_logical_bytes=final_aggregate_bytes,
    )
    _validate_catalog_against_source(source, catalog)
    builder._catalog = catalog
    source._claim_session()
    return builder


def lfm2_cpcf_selective_catalog_artifact_binding(
    source: Lfm2CpcfAuthenticatedSelectiveSource,
) -> DynamicEvidenceCatalogBinding:
    """Bind reusable metadata to exact authenticated inputs and codec."""

    if not isinstance(source, Lfm2CpcfAuthenticatedSelectiveSource):
        raise TypeError(
            "source must be Lfm2CpcfAuthenticatedSelectiveSource."
        )
    codec_body = {
        "schema": LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_CODEC_SCHEMA,
        "provider_sha256": source.provider_sha256,
        "catalog_schema": LFM2_CPCF_SELECTIVE_CATALOG_SCHEMA,
        "phase_reference_schema": (
            LFM2_CPCF_SELECTIVE_PHASE_REFERENCE_SCHEMA
        ),
        "payload_schema": (
            LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_PAYLOAD_SCHEMA
        ),
        "typed_reconstruction_required": True,
    }
    bindings = {
        "schema": LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_CODEC_SCHEMA,
        "cache_key": source.session.cache_key,
        "selective_session": source.session.as_dict(),
        "layer_topology_sha256": source.adapter.layer_topology_sha256,
        "capture_selection_sha256": (
            source.adapter.capture_selection_sha256
        ),
        "source_width": source.adapter.source_width,
        "layer_count": source.layer_count,
        "phase_count": source.phase_count,
        "ordered_phases": [
            {
                "phase_index": plan.phase_index,
                "batch_index": plan.batch_index,
                "input_sha256": plan.input_sha256,
                "phase": plan.phase,
            }
            for plan in source.phase_plans
        ],
        "ordered_layer_operator_kinds": [
            str(layer["contract"]["kind"])
            for layer in source.adapter.topology
        ],
    }
    return DynamicEvidenceCatalogBinding(
        catalog_kind=LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_KIND,
        source_sha256=source.source_sha256,
        provider_sha256=source.provider_sha256,
        authenticated_input_sha256=source.session.evidence_sha256,
        codec_sha256=canonical_sha256(codec_body),
        bindings=bindings,
    )


def lfm2_cpcf_selective_catalog_to_artifact_payload(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
) -> dict[str, Any]:
    """Encode all typed references needed for lossless reconstruction."""

    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise TypeError(
            "catalog must be Lfm2CpcfSelectiveEvidenceCatalog."
        )
    return {
        "schema": LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_PAYLOAD_SCHEMA,
        "catalog": catalog.as_dict(),
        "dynamic_references": [
            reference.as_dict()
            for reference in catalog.dynamic_catalog.references
        ],
        "tensor_payloads_resident": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
    }


def _phase_reference_from_artifact(
    value: Mapping[str, Any],
) -> Lfm2CpcfSelectivePhaseReference:
    expected = {
        "schema",
        "phase_index",
        "cache_key",
        "batch_index",
        "input_sha256",
        "phase",
        "layer_operator_kinds",
        "layer_evidence_sha256",
        "layer_field_inventory_sha256",
        "layer_logical_bytes",
        "final_norm_evidence_sha256",
        "final_norm_field_inventory_sha256",
        "final_norm_logical_bytes",
        "existing_phase_evidence_sha256",
        "parent_side_compiler_inputs_ready",
        "tensor_payloads_resident",
        "reference_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Artifact phase reference fields are not exact."
        )
    if (
        value["schema"] != LFM2_CPCF_SELECTIVE_PHASE_REFERENCE_SCHEMA
        or value["parent_side_compiler_inputs_ready"] is not True
        or value["tensor_payloads_resident"] is not False
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Artifact phase reference invariants are invalid."
        )
    for name in (
        "layer_operator_kinds",
        "layer_evidence_sha256",
        "layer_field_inventory_sha256",
        "layer_logical_bytes",
    ):
        if not isinstance(value[name], list):
            raise Lfm2CpcfSelectiveEvidenceError(
                f"Artifact phase reference {name} must be a list."
            )
    result = Lfm2CpcfSelectivePhaseReference(
        phase_index=value["phase_index"],
        cache_key=value["cache_key"],
        batch_index=value["batch_index"],
        input_sha256=value["input_sha256"],
        phase=value["phase"],
        layer_operator_kinds=tuple(value["layer_operator_kinds"]),
        layer_evidence_sha256=tuple(value["layer_evidence_sha256"]),
        layer_field_inventory_sha256=tuple(
            value["layer_field_inventory_sha256"]
        ),
        layer_logical_bytes=tuple(value["layer_logical_bytes"]),
        final_norm_evidence_sha256=value[
            "final_norm_evidence_sha256"
        ],
        final_norm_field_inventory_sha256=value[
            "final_norm_field_inventory_sha256"
        ],
        final_norm_logical_bytes=value["final_norm_logical_bytes"],
        existing_phase_evidence_sha256=value[
            "existing_phase_evidence_sha256"
        ],
        reference_sha256=value["reference_sha256"],
    )
    if result.as_dict() != dict(value):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Artifact phase reference changed during reconstruction."
        )
    return result


def _dynamic_catalog_from_artifact(
    values: Any,
    *,
    source_sha256: str,
    provider_sha256: str,
    claimed_catalog_sha256: str,
) -> DynamicEvidenceLayerCatalog:
    if not isinstance(values, list) or not values:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Artifact dynamic references must be a nonempty list."
        )
    references = []
    expected = {
        "schema",
        "source_sha256",
        "provider_sha256",
        "layer_index",
        "evidence_sha256",
        "logical_bytes",
        "reference_sha256",
    }
    for value in values:
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Artifact dynamic reference fields are not exact."
            )
        reference = DynamicEvidenceLayerReference(
            source_sha256=value["source_sha256"],
            provider_sha256=value["provider_sha256"],
            layer_index=value["layer_index"],
            evidence_sha256=value["evidence_sha256"],
            logical_bytes=value["logical_bytes"],
            reference_sha256=value["reference_sha256"],
        )
        if reference.as_dict() != dict(value):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Artifact dynamic reference changed during reconstruction."
            )
        references.append(reference)
    rebuilt = DynamicEvidenceLayerCatalog.create(
        source_sha256=source_sha256,
        provider_sha256=provider_sha256,
        evidence_sha256=[
            reference.evidence_sha256 for reference in references
        ],
        logical_bytes=[
            reference.logical_bytes for reference in references
        ],
    )
    if (
        rebuilt.references != tuple(references)
        or rebuilt.catalog_sha256 != claimed_catalog_sha256
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Artifact dynamic catalog seal is invalid."
        )
    return rebuilt


def lfm2_cpcf_selective_catalog_from_artifact_payload(
    payload: Mapping[str, Any],
) -> Lfm2CpcfSelectiveEvidenceCatalog:
    """Decode canonical JSON into fully validated typed catalog objects."""

    expected = {
        "schema",
        "catalog",
        "dynamic_references",
        "tensor_payloads_resident",
        "parent_forward_calls",
        "training_steps",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact payload fields are not exact."
        )
    if (
        payload["schema"]
        != LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_PAYLOAD_SCHEMA
        or payload["tensor_payloads_resident"] is not False
        or payload["parent_forward_calls"] != 0
        or payload["training_steps"] != 0
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact payload invariants are invalid."
        )
    catalog_value = payload["catalog"]
    if not isinstance(catalog_value, Mapping):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact catalog must be a JSON object."
        )
    expected_catalog_fields = {
        "schema",
        "cache_key",
        "source_sha256",
        "provider_sha256",
        "selective_session_sha256",
        "layer_topology_sha256",
        "capture_selection_sha256",
        "source_width",
        "layer_count",
        "phase_count",
        "ordered_phase_reference_sha256",
        "dynamic_catalog_sha256",
        "final_norm_evidence_sha256",
        "final_norm_logical_bytes",
        "tensor_payloads_resident",
        "parent_forward_calls",
        "training_steps",
        "phase_references",
        "catalog_sha256",
    }
    if set(catalog_value) != expected_catalog_fields:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact catalog fields are not exact."
        )
    if (
        catalog_value["schema"] != LFM2_CPCF_SELECTIVE_CATALOG_SCHEMA
        or catalog_value["tensor_payloads_resident"] is not False
        or catalog_value["parent_forward_calls"] != 0
        or catalog_value["training_steps"] != 0
        or not isinstance(
            catalog_value["ordered_phase_reference_sha256"],
            list,
        )
    ):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact catalog invariants are invalid."
        )
    phase_values = catalog_value.get("phase_references")
    if not isinstance(phase_values, list) or not phase_values:
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective artifact phase references are invalid."
        )
    phase_references = tuple(
        _phase_reference_from_artifact(value)
        for value in phase_values
    )
    dynamic_catalog = _dynamic_catalog_from_artifact(
        payload["dynamic_references"],
        source_sha256=catalog_value["source_sha256"],
        provider_sha256=catalog_value["provider_sha256"],
        claimed_catalog_sha256=catalog_value[
            "dynamic_catalog_sha256"
        ],
    )
    result = Lfm2CpcfSelectiveEvidenceCatalog.create(
        cache_key=catalog_value["cache_key"],
        source_sha256=catalog_value["source_sha256"],
        provider_sha256=catalog_value["provider_sha256"],
        selective_session_sha256=catalog_value[
            "selective_session_sha256"
        ],
        layer_topology_sha256=catalog_value[
            "layer_topology_sha256"
        ],
        capture_selection_sha256=catalog_value[
            "capture_selection_sha256"
        ],
        source_width=catalog_value["source_width"],
        phase_references=phase_references,
        dynamic_catalog=dynamic_catalog,
        final_norm_evidence_sha256=catalog_value[
            "final_norm_evidence_sha256"
        ],
        final_norm_logical_bytes=catalog_value[
            "final_norm_logical_bytes"
        ],
    )
    if result.as_dict() != dict(catalog_value):
        raise Lfm2CpcfSelectiveEvidenceError(
            "Selective catalog changed during typed reconstruction."
        )
    return result


def open_or_build_lfm2_cpcf_selective_evidence_provider(
    cache: Lfm2CircuitTraceCache,
    *,
    artifact_store: (
        DynamicEvidenceCatalogArtifactStore | str | Path
    ),
) -> Lfm2CpcfSelectiveEvidenceProvider:
    """Use a verified catalog hit or run and publish the original cold pass."""

    store = (
        artifact_store
        if isinstance(
            artifact_store,
            DynamicEvidenceCatalogArtifactStore,
        )
        else DynamicEvidenceCatalogArtifactStore(artifact_store)
    )
    source = authenticate_lfm2_cpcf_selective_source(cache)
    try:
        binding = lfm2_cpcf_selective_catalog_artifact_binding(source)
        artifact = store.load(binding)
        if artifact is not None:
            catalog = lfm2_cpcf_selective_catalog_from_artifact_payload(
                artifact.payload
            )
            return (
                build_lfm2_cpcf_selective_provider_from_authenticated_catalog(
                    source,
                    catalog,
                    catalog_artifact=artifact,
                )
            )
        provider = _build_catalog_from_authenticated_source(source)
        try:
            artifact = store.publish(
                binding,
                lfm2_cpcf_selective_catalog_to_artifact_payload(
                    provider.catalog
                ),
            )
            reloaded_catalog = (
                lfm2_cpcf_selective_catalog_from_artifact_payload(
                    artifact.payload
                )
            )
            _validate_catalog_against_source(source, reloaded_catalog)
            if reloaded_catalog.as_dict() != provider.catalog.as_dict():
                raise Lfm2CpcfSelectiveEvidenceError(
                    "Published catalog differs from the cold catalog."
                )
            provider._catalog = reloaded_catalog
            provider._catalog_artifact = artifact
            return provider
        except BaseException:
            provider.close()
            raise
    except BaseException:
        source.close()
        raise


def open_lfm2_cpcf_selective_evidence_provider_from_catalog(
    cache: Lfm2CircuitTraceCache,
    *,
    artifact_store: (
        DynamicEvidenceCatalogArtifactStore | str | Path
    ),
    expected_binding_sha256: str,
    expected_manifest_sha256: str,
    expected_catalog_sha256: str,
) -> Lfm2CpcfSelectiveEvidenceProvider:
    """Require one exact warm artifact hit; never run the cold catalog pass."""

    for value, name in (
        (expected_binding_sha256, "expected_binding_sha256"),
        (expected_manifest_sha256, "expected_manifest_sha256"),
        (expected_catalog_sha256, "expected_catalog_sha256"),
    ):
        _digest(value, name)
    store = (
        artifact_store
        if isinstance(
            artifact_store,
            DynamicEvidenceCatalogArtifactStore,
        )
        else DynamicEvidenceCatalogArtifactStore(artifact_store)
    )
    source = authenticate_lfm2_cpcf_selective_source(cache)
    try:
        binding = lfm2_cpcf_selective_catalog_artifact_binding(source)
        if binding.binding_sha256 != expected_binding_sha256:
            raise Lfm2CpcfSelectiveEvidenceError(
                "Authenticated source differs from the required catalog "
                "binding."
            )
        artifact = store.load(binding)
        if artifact is None:
            raise Lfm2CpcfSelectiveEvidenceError(
                "The required authenticated catalog is not prepared; "
                "production folding cannot run the cold catalog builder."
            )
        if (
            artifact.disposition != "loaded"
            or artifact.manifest_sha256 != expected_manifest_sha256
            or artifact.catalog_sha256 != expected_catalog_sha256
        ):
            raise Lfm2CpcfSelectiveEvidenceError(
                "Warm catalog hit differs from its required manifest or "
                "catalog identity."
            )
        catalog = lfm2_cpcf_selective_catalog_from_artifact_payload(
            artifact.payload
        )
        return build_lfm2_cpcf_selective_provider_from_authenticated_catalog(
            source,
            catalog,
            catalog_artifact=artifact,
        )
    except BaseException:
        source.close()
        raise


def build_lfm2_cpcf_selective_evidence_provider(
    cache: Lfm2CircuitTraceCache,
) -> Lfm2CpcfSelectiveEvidenceProvider:
    """Build a tensor-free catalog and reusable selective layer provider."""

    source = authenticate_lfm2_cpcf_selective_source(cache)
    try:
        return _build_catalog_from_authenticated_source(source)
    except BaseException:
        source.close()
        raise


__all__ = [
    "LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_CODEC_SCHEMA",
    "LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_KIND",
    "LFM2_CPCF_SELECTIVE_CATALOG_ARTIFACT_PAYLOAD_SCHEMA",
    "LFM2_CPCF_SELECTIVE_CATALOG_SCHEMA",
    "LFM2_CPCF_SELECTIVE_FINAL_NORM_SCHEMA",
    "LFM2_CPCF_SELECTIVE_LAYER_PAYLOAD_SCHEMA",
    "LFM2_CPCF_SELECTIVE_PHASE_REFERENCE_SCHEMA",
    "LFM2_CPCF_SELECTIVE_PROVIDER_IMPLEMENTATION_SCHEMA",
    "LFM2_CPCF_SELECTIVE_SOURCE_SCHEMA",
    "Lfm2CpcfAuthenticatedSelectiveSource",
    "Lfm2CpcfSelectiveEvidenceCatalog",
    "Lfm2CpcfSelectiveEvidenceError",
    "Lfm2CpcfSelectiveEvidenceProvider",
    "Lfm2CpcfSelectiveFinalNormPayload",
    "Lfm2CpcfSelectiveLayerPayload",
    "Lfm2CpcfSelectivePhaseReference",
    "authenticate_lfm2_cpcf_selective_source",
    "build_lfm2_cpcf_selective_evidence_provider",
    "build_lfm2_cpcf_selective_provider_from_authenticated_catalog",
    "lfm2_cpcf_selective_catalog_artifact_binding",
    "lfm2_cpcf_selective_catalog_from_artifact_payload",
    "lfm2_cpcf_selective_catalog_to_artifact_payload",
    "lfm2_cpcf_selective_provider_implementation",
    "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
    "open_or_build_lfm2_cpcf_selective_evidence_provider",
]
