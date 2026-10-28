"""Canonical adapters for independently implemented LFM2 matrix kernels.

Technique modules contain fold math only.  This registry authenticates and
dispatches those providers without granting them receipt, evaluation, or
compaction authority.  Parent-shell isolation and blank-child promotion stay
the responsibility of :mod:`neural_forge.hf_lfm2_matrix_cell` executors.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_circuit_ir import canonical_sha256

PROVIDER_SCHEMA = "neural-forge-lfm2-matrix-kernel-provider-v1"

_FACTORIES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "T4": (
            "neural_forge.matrix_technique_lowrank_writers",
            "matrix_technique_registry_entry",
        ),
        "T5": (
            "neural_forge.matrix_technique_body_routes",
            "matrix_technique_registry_entry",
        ),
        "T7": (
            "neural_forge.matrix_technique_relation_matching",
            "matrix_technique_registry_entries",
        ),
        "T8": (
            "neural_forge.matrix_technique_relation_matching",
            "matrix_technique_registry_entries",
        ),
    }
)

_COMPONENT_IDS = tuple(f"C{index:02d}" for index in range(1, 16))
_TECHNIQUE_IDS = tuple(f"T{index}" for index in range(1, 10))
_COMPONENT_SITES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "C01": ("GLOBAL",),
        "C02": ("GLOBAL",),
        "C03": ("L00", "L05", "L13"),
        "C04": ("L00", "L05", "L13"),
        "C05": ("L00", "L05", "L13"),
        "C06": ("L02", "L08", "L12"),
        "C07": ("L02", "L08", "L12"),
        "C08": ("L02", "L08", "L12"),
        "C09": ("L02", "L08", "L12"),
        "C10": ("L02", "L08"),
        "C11": ("L00", "L05", "L13"),
        "C12": ("L00", "L05", "L13"),
        "C13": ("L00", "L05", "L13"),
        "C14": ("L00", "FINAL"),
        "C15": ("GLOBAL",),
    }
)
_ALL_COMPONENTS = frozenset(_COMPONENT_IDS)
_DESIGNED_SUPPORT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "T1": _ALL_COMPONENTS,
        "T2": _ALL_COMPONENTS - {"C04", "C10", "C14"},
        "T3": _ALL_COMPONENTS - {"C14"},
        "T4": _ALL_COMPONENTS - {"C14"},
        "T5": _ALL_COMPONENTS - {"C14"},
        "T6": frozenset({"C04", "C06", "C07", "C08", "C09", "C10"}),
        "T7": frozenset({"C06", "C07", "C08", "C09", "C10"}),
        "T8": _ALL_COMPONENTS,
        "T9": _ALL_COMPONENTS - {"C14"},
    }
)
_CANONICAL_KERNEL_SUPPORT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "T4": frozenset({"C05", "C09", "C13"}),
        "T5": frozenset(
            {"C03", "C05", "C06", "C07", "C08", "C09", "C11", "C12", "C13"}
        ),
        "T7": frozenset({"C06", "C07"}),
        "T8": frozenset(
            {"C03", "C05", "C06", "C07", "C08", "C09", "C11", "C12", "C13"}
        ),
    }
)
_LEGACY_OR_PROVEN_SUPPORT: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "T1": _ALL_COMPONENTS,
        "T5": frozenset({"C01", "C02"}),
    }
)


class Lfm2ComponentMethodMatrixError(RuntimeError):
    """A kernel provider or invocation violated the canonical boundary."""


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise Lfm2ComponentMethodMatrixError(
        f"Provider description contains non-JSON {type(value).__name__}."
    )


def _module_sha256(module: Any) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise Lfm2ComponentMethodMatrixError("Kernel module has no source file.")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MatrixKernelProvider:
    technique_id: str
    technique_name: str
    module_name: str
    components: tuple[str, ...]
    levels: tuple[str, ...]
    module_sha256: str
    description_sha256: str
    fit: Callable[[Mapping[str, Any]], Mapping[str, Any]]

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_SCHEMA,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "module_name": self.module_name,
            "components": list(self.components),
            "levels": list(self.levels),
            "module_sha256": self.module_sha256,
            "description_sha256": self.description_sha256,
            "kernel_only_no_compaction_claim_authority": True,
        }

    def physical_component_ids(
        self, measurement_component_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return the minimum provider-owned relation, not target geometry."""

        requested = tuple(measurement_component_ids)
        if self.technique_id == "T7" and set(requested).issubset({"C06", "C07"}):
            return ("C06", "C07")
        return requested

    def invoke(
        self,
        *,
        measurement_component_ids: tuple[str, ...],
        level_id: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        component = "+".join(measurement_component_ids)
        if component not in self.components:
            raise Lfm2ComponentMethodMatrixError(
                f"{self.technique_id} does not support component {component}."
            )
        if level_id not in self.levels:
            raise Lfm2ComponentMethodMatrixError(
                f"{self.technique_id} does not support level {level_id}."
            )
        result = self.fit(payload)
        if not isinstance(result, Mapping):
            raise Lfm2ComponentMethodMatrixError(
                "Kernel provider did not return a mapping."
            )
        if result.get("technique_id") != self.technique_id:
            raise Lfm2ComponentMethodMatrixError(
                "Kernel result technique identity is inconsistent."
            )
        if not isinstance(result.get("technique_record"), Mapping):
            raise Lfm2ComponentMethodMatrixError(
                "Kernel result has no authenticated technique record."
            )
        return MappingProxyType(dict(result))


def _entries_for(technique_id: str) -> tuple[Any, ...]:
    try:
        module_name, factory_name = _FACTORIES[technique_id]
    except KeyError as exc:
        raise Lfm2ComponentMethodMatrixError(
            f"No kernel provider is registered for {technique_id}."
        ) from exc
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    value = factory()
    return tuple(value) if isinstance(value, tuple) else (value,)


def load_matrix_kernel_provider(technique_id: str) -> MatrixKernelProvider:
    entries = _entries_for(technique_id)
    entry = next(
        (value for value in entries if value.technique_id == technique_id),
        None,
    )
    if entry is None:
        raise Lfm2ComponentMethodMatrixError(
            f"Provider factory omitted {technique_id}."
        )
    description = _plain(entry.describe())
    module = importlib.import_module(entry.module)
    provider = MatrixKernelProvider(
        technique_id=str(entry.technique_id),
        technique_name=str(entry.technique),
        module_name=str(entry.module),
        components=tuple(str(value) for value in entry.components),
        levels=tuple(str(value) for value in entry.levels),
        module_sha256=_module_sha256(module),
        description_sha256=canonical_sha256(description),
        fit=entry.fit,
    )
    if description.get("technique_id") != provider.technique_id:
        raise Lfm2ComponentMethodMatrixError(
            "Provider description disagrees with its entry."
        )
    return provider


def discover_matrix_kernel_providers() -> Mapping[str, MatrixKernelProvider]:
    return MappingProxyType(
        {
            technique_id: load_matrix_kernel_provider(technique_id)
            for technique_id in sorted(_FACTORIES)
        }
    )


def _physical_bundle(component_id: str, technique_id: str) -> tuple[str, ...]:
    if technique_id == "T7" and component_id in {"C06", "C07"}:
        return ("C06", "C07")
    if technique_id == "T6":
        if component_id == "C04":
            return ("C03", "C04", "C05")
        return ("C06", "C07", "C08", "C09")
    return (component_id,)


def matrix_campaign_pair_coverage() -> tuple[Mapping[str, Any], ...]:
    """Account for every designed technique/component pair, including gaps.

    A row is a measurement scope.  ``physical_component_ids`` may name a
    coupled relation that the technique mathematically requires; this keeps a
    Q-only measurement distinct from silently pretending that Q/K are
    independent.  Missing implementations are explicit campaign gaps rather
    than omitted cells.
    """

    rows: list[Mapping[str, Any]] = []
    for technique_id in _TECHNIQUE_IDS:
        designed = _DESIGNED_SUPPORT[technique_id]
        canonical = _CANONICAL_KERNEL_SUPPORT.get(technique_id, frozenset())
        legacy = _LEGACY_OR_PROVEN_SUPPORT.get(technique_id, frozenset())
        for component_id in _COMPONENT_IDS:
            applicable = component_id in designed
            if not applicable:
                state = "not_applicable"
            elif component_id in canonical:
                state = "canonical_kernel_available"
            elif component_id in legacy:
                state = "legacy_or_previously_proven_kernel"
            else:
                state = "kernel_missing"
            rows.append(
                MappingProxyType(
                    {
                        "technique_id": technique_id,
                        "measurement_component_ids": [component_id],
                        "physical_component_ids": list(
                            _physical_bundle(component_id, technique_id)
                        ),
                        "designed_applicable": applicable,
                        "implementation_state": state,
                        "sites": list(_COMPONENT_SITES[component_id]),
                        "levels": ["NA"]
                        if component_id == "C14"
                        else ["M16", "M8", "M4", "X4"],
                        "individual_measurement_scope": True,
                    }
                )
            )
    return tuple(rows)


def full_component_matrix_cells() -> tuple[Mapping[str, Any], ...]:
    """Enumerate every applicable individual measurement cell deterministically."""

    cells: list[Mapping[str, Any]] = []
    for pair in matrix_campaign_pair_coverage():
        if pair["designed_applicable"] is not True:
            continue
        component_id = str(pair["measurement_component_ids"][0])
        technique_id = str(pair["technique_id"])
        for level_id in pair["levels"]:
            for site_id in pair["sites"]:
                cells.append(
                    MappingProxyType(
                        {
                            "cell_id": (
                                f"{component_id}.{technique_id}.{level_id}.{site_id}"
                            ),
                            "technique_id": technique_id,
                            "measurement_component_ids": [component_id],
                            "physical_component_ids": list(
                                pair["physical_component_ids"]
                            ),
                            "level_id": level_id,
                            "site_id": site_id,
                            "implementation_state": pair["implementation_state"],
                            "evidence_class": "isolation_only",
                            "numeric_metrics_are_diagnostic_only": True,
                            "manual_intelligence_review_required": True,
                        }
                    )
                )
    return tuple(cells)


__all__ = [
    "PROVIDER_SCHEMA",
    "Lfm2ComponentMethodMatrixError",
    "MatrixKernelProvider",
    "discover_matrix_kernel_providers",
    "full_component_matrix_cells",
    "load_matrix_kernel_provider",
    "matrix_campaign_pair_coverage",
]
