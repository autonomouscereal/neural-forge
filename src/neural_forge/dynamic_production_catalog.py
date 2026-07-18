"""Assembled, fail-closed production catalog for automatic dynamic folding.

The lower-level dynamic modules deliberately expose separate registries for
target synthesis, fold-stack construction, and candidate search.  This module
binds those extension points once and presents one neutral production facade:
parents + objective + destination.  It does not claim that an unregistered
family, modality, source family, or target-size range is supported.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .dynamic_circuit_ir import JsonScalar, canonical_sha256
from .dynamic_fold_artifact import (
    DynamicFoldArtifactCertificate,
    DynamicFoldArtifactError,
)
from .dynamic_fold_compiler import (
    DynamicFoldCompilerError,
    validate_disjoint_parent_models,
)
from .dynamic_fold_executor import run_fold_only_callback
from .dynamic_fold_registry import (
    DurableExperimentLedger,
    ExperimentLedger,
    FoldMethodRegistry,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_model_inspector import InspectorRegistry, default_inspector_registry
from .dynamic_search import (
    AutomaticDynamicFoldSearchSystem,
    DynamicFoldPortfolioRegistry,
    DynamicFoldSearchCandidate,
    DynamicFoldSearchObjective,
    DynamicFoldSearchResult,
    FailureDisposition,
)
from .dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from .dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisResult,
)


SCHEMA = "neural-forge-production-dynamic-fold-catalog-v1"
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_RESERVED_METADATA = frozenset({"capability_id", "method_id"})
_BROAD_FAILURE_TYPES = frozenset({BaseException, Exception, RuntimeError, OSError})
_EXPECTED_CANDIDATE_COMPILER_FAILURES = frozenset(
    {
        (
            "Automatic planning found unsupported target circuits or a "
            "previously closed exact attempt; inspect plan.gaps and action "
            "ledger decisions."
        ),
        (
            "Dynamic folding did not produce a behavior-accepted complete "
            "compilation; no artifact was published."
        ),
    }
)
_EXPECTED_CANDIDATE_ARTIFACT_PREFIXES = (
    "Freshly reloaded child failed its behavioral gate:",
)


class ProductionDynamicFoldCatalogError(RuntimeError):
    """A production bundle, capability, candidate, or request is unsafe."""


class ProductionCandidateRejected(RuntimeError):
    """A resource binding can raise this for one safely rejected search arm."""


class ProductionInfrastructureFailure(RuntimeError):
    """A resource binding can raise this when search must stop and retry later."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionDynamicFoldCatalogError(f"{name} must be a nonempty string.")
    return value.strip()


def _safe_id(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if _SAFE_ID.fullmatch(result) is None or ".." in result:
        raise ProductionDynamicFoldCatalogError(
            f"{name} must be a safe lowercase identifier."
        )
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionDynamicFoldCatalogError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionDynamicFoldCatalogError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionDynamicFoldCatalogError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionDynamicFoldCatalogError(f"{name} must be finite.")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionDynamicFoldCatalogError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _names(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise ProductionDynamicFoldCatalogError(f"{name} must be a tuple or list.")
    result = tuple(sorted({_text(value, name).lower() for value in values}))
    if not result:
        raise ProductionDynamicFoldCatalogError(f"{name} must not be empty.")
    return result


def _scalar(value: Any, name: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise ProductionDynamicFoldCatalogError(f"{name} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ProductionDynamicFoldCatalogError(f"{name} must be finite.")
    return value


def _items(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
    name: str,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    source = value.items() if isinstance(value, Mapping) else value
    result = tuple(
        sorted(
            (
                _text(str(key), f"{name} key"),
                _scalar(item, f"{name} {key}"),
            )
            for key, item in source
        )
    )
    if len({key for key, _item in result}) != len(result):
        raise ProductionDynamicFoldCatalogError(f"{name} keys must be unique.")
    return result


def _exception_types(
    values: Any,
    name: str,
) -> tuple[type[Exception], ...]:
    if not isinstance(values, (tuple, list)):
        raise ProductionDynamicFoldCatalogError(f"{name} must be a tuple or list.")
    result: list[type[Exception]] = []
    for value in values:
        if (
            not isinstance(value, type)
            or not issubclass(value, Exception)
            or value in _BROAD_FAILURE_TYPES
        ):
            raise ProductionDynamicFoldCatalogError(
                f"{name} contains an unsafe broad or non-exception type."
            )
        if value not in result:
            result.append(value)
    return tuple(result)


def _adapter_id(adapter: Any, name: str) -> str:
    return _text(getattr(adapter, "adapter_id", None), name)


def _adapter_descriptor(adapter: Any, *, kind: str) -> dict[str, Any]:
    adapter_type = type(adapter)
    version = getattr(adapter, "version", None)
    priority = getattr(adapter, "priority", None)
    if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
        raise ProductionDynamicFoldCatalogError(
            f"{kind} version must be a positive integer."
        )
    if isinstance(priority, bool) or not isinstance(priority, int):
        raise ProductionDynamicFoldCatalogError(f"{kind} priority must be an integer.")
    resource_fingerprint = getattr(adapter, "resource_fingerprint", None)
    if resource_fingerprint is None:
        resource_fingerprint = getattr(adapter, "fingerprint", None)
    if not (
        isinstance(resource_fingerprint, str)
        and len(resource_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in resource_fingerprint)
    ):
        resource_fingerprint = None
    result = {
        "adapter_id": _adapter_id(adapter, f"{kind} adapter_id"),
        "version": version,
        "priority": priority,
        "implementation": (f"{adapter_type.__module__}.{adapter_type.__qualname__}"),
        "target_families": list(
            _names(
                getattr(adapter, "target_families", None),
                f"{kind} target_families",
            )
        ),
        "target_modalities": list(
            _names(
                getattr(adapter, "target_modalities", None),
                f"{kind} target_modalities",
            )
        ),
        "resource_fingerprint": resource_fingerprint,
    }
    if kind == "stack":
        result["method_ids"] = list(
            _names(
                getattr(adapter, "method_ids", None),
                "stack method_ids",
            )
        )
    return result


def _validate_adapter_capability(
    adapter: Any,
    capability: "ProductionFoldCapability",
    *,
    kind: str,
) -> None:
    families = set(
        _names(
            getattr(adapter, "target_families", None),
            f"{kind} target_families",
        )
    )
    modalities = set(
        _names(
            getattr(adapter, "target_modalities", None),
            f"{kind} target_modalities",
        )
    )
    if capability.target_family not in families and "auto" not in families:
        raise ProductionDynamicFoldCatalogError(
            f"{kind} does not declare capability family {capability.target_family!r}."
        )
    if not set(capability.target_modalities).issubset(modalities):
        raise ProductionDynamicFoldCatalogError(
            f"{kind} does not declare every capability modality."
        )


@dataclass(frozen=True, slots=True)
class ProductionFoldCapability:
    """One explicit supported source/target envelope, never a universal claim."""

    capability_id: str
    source_families: tuple[str, ...]
    target_family: str
    target_modalities: tuple[str, ...]
    minimum_target_unique_parameters: int
    maximum_target_unique_parameters: int
    minimum_parent_models: int = 1
    maximum_parent_models: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capability_id",
            _safe_id(self.capability_id, "capability_id"),
        )
        object.__setattr__(
            self,
            "source_families",
            _names(self.source_families, "source_families"),
        )
        object.__setattr__(
            self,
            "target_family",
            _text(self.target_family, "target_family").lower(),
        )
        object.__setattr__(
            self,
            "target_modalities",
            _names(self.target_modalities, "target_modalities"),
        )
        object.__setattr__(
            self,
            "minimum_target_unique_parameters",
            _positive_int(
                self.minimum_target_unique_parameters,
                "minimum_target_unique_parameters",
            ),
        )
        object.__setattr__(
            self,
            "maximum_target_unique_parameters",
            _positive_int(
                self.maximum_target_unique_parameters,
                "maximum_target_unique_parameters",
            ),
        )
        object.__setattr__(
            self,
            "minimum_parent_models",
            _positive_int(
                self.minimum_parent_models,
                "minimum_parent_models",
            ),
        )
        object.__setattr__(
            self,
            "maximum_parent_models",
            _positive_int(
                self.maximum_parent_models,
                "maximum_parent_models",
            ),
        )
        if (
            self.maximum_target_unique_parameters
            < self.minimum_target_unique_parameters
        ):
            raise ProductionDynamicFoldCatalogError(
                "Capability maximum target size is smaller than its minimum."
            )
        if self.maximum_parent_models < self.minimum_parent_models:
            raise ProductionDynamicFoldCatalogError(
                "Capability maximum parent count is smaller than its minimum."
            )

    def supports(self, request: DynamicTargetRequest) -> bool:
        return (
            isinstance(request, DynamicTargetRequest)
            and request.target_family == self.target_family
            and request.target_modalities == self.target_modalities
            and self.minimum_target_unique_parameters
            <= request.exact_unique_parameters
            <= self.maximum_target_unique_parameters
        )

    def supports_parent_count(self, parent_count: int) -> bool:
        return (
            not isinstance(parent_count, bool)
            and isinstance(parent_count, int)
            and self.minimum_parent_models
            <= parent_count
            <= self.maximum_parent_models
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "capability",
            "capability_id": self.capability_id,
            "source_families": list(self.source_families),
            "target_family": self.target_family,
            "target_modalities": list(self.target_modalities),
            "minimum_target_unique_parameters": (self.minimum_target_unique_parameters),
            "maximum_target_unique_parameters": (self.maximum_target_unique_parameters),
            "minimum_parent_models": self.minimum_parent_models,
            "maximum_parent_models": self.maximum_parent_models,
        }


@dataclass(frozen=True, slots=True)
class CertificateScoreTerm:
    """One signed numeric metric from the fresh-reload certificate decision."""

    metric_name: str
    weight: float = 1.0
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metric_name",
            _text(self.metric_name, "certificate metric_name"),
        )
        object.__setattr__(
            self,
            "weight",
            _finite_float(self.weight, "certificate metric weight"),
        )
        if not isinstance(self.required, bool):
            raise ProductionDynamicFoldCatalogError(
                "Certificate score term required must be boolean."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "weight": self.weight,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class ProductionFoldCandidateSpec:
    """Declarative exact-budget arm bound to cached evidence and one method."""

    candidate_id: str
    method_id: str
    evidence_fingerprint: str
    target_adapter_id: str | None = None
    priority: int = 0
    blank_seed: int | None = None
    configuration: tuple[tuple[str, JsonScalar], ...] = ()
    metadata: tuple[tuple[str, JsonScalar], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _safe_id(self.candidate_id, "candidate_id"),
        )
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "evidence_fingerprint",
            _digest(self.evidence_fingerprint, "evidence_fingerprint"),
        )
        if self.target_adapter_id is not None:
            object.__setattr__(
                self,
                "target_adapter_id",
                _text(self.target_adapter_id, "target_adapter_id"),
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ProductionDynamicFoldCatalogError(
                "Candidate priority must be an integer."
            )
        if self.blank_seed is not None:
            object.__setattr__(
                self,
                "blank_seed",
                _nonnegative_int(self.blank_seed, "blank_seed"),
            )
        object.__setattr__(
            self,
            "configuration",
            _items(self.configuration, "candidate configuration"),
        )
        metadata = _items(self.metadata, "candidate metadata")
        if _RESERVED_METADATA.intersection(dict(metadata)):
            raise ProductionDynamicFoldCatalogError(
                "Candidate metadata uses a catalog-reserved key."
            )
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "method_id": self.method_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "target_adapter_id": self.target_adapter_id,
            "priority": self.priority,
            "blank_seed": self.blank_seed,
            "configuration": dict(self.configuration),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ProductionFoldBundle:
    """Resource-bound target, stack, and portfolio registration unit."""

    capability: ProductionFoldCapability
    target_adapters: tuple[Any, ...]
    stack_adapters: tuple[Any, ...]
    candidate_specs: tuple[ProductionFoldCandidateSpec, ...]
    maximum_attempts: int
    portfolio_id: str | None = None
    version: int = 1
    priority: int = 100
    score_terms: tuple[CertificateScoreTerm, ...] = ()
    candidate_rejection_exceptions: tuple[type[Exception], ...] = (
        ProductionCandidateRejected,
    )
    infrastructure_exceptions: tuple[type[Exception], ...] = (
        ProductionInfrastructureFailure,
        MemoryError,
        TimeoutError,
        ConnectionError,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.capability, ProductionFoldCapability):
            raise ProductionDynamicFoldCatalogError(
                "Bundle capability must be ProductionFoldCapability."
            )
        targets = tuple(self.target_adapters)
        stacks = tuple(self.stack_adapters)
        specs = tuple(self.candidate_specs)
        if not targets or not stacks or not specs:
            raise ProductionDynamicFoldCatalogError(
                "A bundle needs target, stack, and candidate registrations."
            )
        target_ids = tuple(_adapter_id(value, "target adapter_id") for value in targets)
        stack_ids = tuple(_adapter_id(value, "stack adapter_id") for value in stacks)
        if len(set(target_ids)) != len(target_ids):
            raise ProductionDynamicFoldCatalogError("Bundle target adapter IDs repeat.")
        if len(set(stack_ids)) != len(stack_ids):
            raise ProductionDynamicFoldCatalogError("Bundle stack adapter IDs repeat.")
        for adapter in targets:
            _validate_adapter_capability(
                adapter,
                self.capability,
                kind="target adapter",
            )
        for adapter in stacks:
            _validate_adapter_capability(
                adapter,
                self.capability,
                kind="stack adapter",
            )
        if any(not isinstance(value, ProductionFoldCandidateSpec) for value in specs):
            raise ProductionDynamicFoldCatalogError(
                "Bundle candidates must be ProductionFoldCandidateSpec objects."
            )
        if len({value.candidate_id for value in specs}) != len(specs):
            raise ProductionDynamicFoldCatalogError("Bundle candidate IDs repeat.")
        for spec in specs:
            if spec.target_adapter_id is None and len(target_ids) != 1:
                raise ProductionDynamicFoldCatalogError(
                    "Candidates in a multi-target bundle must pin target_adapter_id."
                )
            selected_id = spec.target_adapter_id or target_ids[0]
            if selected_id not in target_ids:
                raise ProductionDynamicFoldCatalogError(
                    f"Candidate {spec.candidate_id!r} references an unregistered "
                    "target adapter."
                )
            if not any(
                spec.method_id
                in _names(
                    getattr(adapter, "method_ids", None),
                    "stack method_ids",
                )
                for adapter in stacks
            ):
                raise ProductionDynamicFoldCatalogError(
                    f"Candidate {spec.candidate_id!r} references a fold method "
                    "not declared by any registered stack adapter."
                )
        object.__setattr__(self, "target_adapters", targets)
        object.__setattr__(self, "stack_adapters", stacks)
        object.__setattr__(self, "candidate_specs", specs)
        object.__setattr__(
            self,
            "maximum_attempts",
            _positive_int(self.maximum_attempts, "maximum_attempts"),
        )
        default_portfolio = f"neural_forge.production.{self.capability.capability_id}"
        object.__setattr__(
            self,
            "portfolio_id",
            _safe_id(self.portfolio_id or default_portfolio, "portfolio_id"),
        )
        object.__setattr__(self, "version", _positive_int(self.version, "version"))
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ProductionDynamicFoldCatalogError(
                "Bundle priority must be an integer."
            )
        terms = tuple(self.score_terms)
        if any(not isinstance(value, CertificateScoreTerm) for value in terms):
            raise ProductionDynamicFoldCatalogError(
                "score_terms must contain CertificateScoreTerm objects."
            )
        if len({value.metric_name for value in terms}) != len(terms):
            raise ProductionDynamicFoldCatalogError(
                "Certificate score metric names repeat."
            )
        object.__setattr__(self, "score_terms", terms)
        rejected = _exception_types(
            self.candidate_rejection_exceptions,
            "candidate_rejection_exceptions",
        )
        infrastructure = _exception_types(
            self.infrastructure_exceptions,
            "infrastructure_exceptions",
        )
        for rejected_type in rejected:
            for infrastructure_type in infrastructure:
                if issubclass(rejected_type, infrastructure_type) or issubclass(
                    infrastructure_type,
                    rejected_type,
                ):
                    raise ProductionDynamicFoldCatalogError(
                        "Candidate and infrastructure exception classes overlap."
                    )
        object.__setattr__(self, "candidate_rejection_exceptions", rejected)
        object.__setattr__(self, "infrastructure_exceptions", infrastructure)

    @property
    def target_adapter_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                _adapter_id(value, "target adapter_id")
                for value in self.target_adapters
            )
        )

    @property
    def stack_adapter_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                _adapter_id(value, "stack adapter_id") for value in self.stack_adapters
            )
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "bundle",
            "capability": self.capability.as_dict(),
            "target_adapter_ids": list(self.target_adapter_ids),
            "stack_adapter_ids": list(self.stack_adapter_ids),
            "target_adapters": [
                _adapter_descriptor(value, kind="target")
                for value in self.target_adapters
            ],
            "stack_adapters": [
                _adapter_descriptor(value, kind="stack")
                for value in self.stack_adapters
            ],
            "candidate_specs": [value.as_dict() for value in self.candidate_specs],
            "maximum_attempts": self.maximum_attempts,
            "portfolio_id": self.portfolio_id,
            "version": self.version,
            "priority": self.priority,
            "score_terms": [value.as_dict() for value in self.score_terms],
            "candidate_rejection_exceptions": [
                f"{value.__module__}.{value.__qualname__}"
                for value in self.candidate_rejection_exceptions
            ],
            "infrastructure_exceptions": [
                f"{value.__module__}.{value.__qualname__}"
                for value in self.infrastructure_exceptions
            ],
        }


class RegisteredExactTargetPortfolio:
    """Deterministic exact-size search portfolio generated from one bundle."""

    def __init__(self, bundle: ProductionFoldBundle) -> None:
        if not isinstance(bundle, ProductionFoldBundle):
            raise ProductionDynamicFoldCatalogError(
                "Portfolio bundle must be ProductionFoldBundle."
            )
        self.bundle = bundle
        self.portfolio_id = str(bundle.portfolio_id)
        self.version = bundle.version
        self.priority = bundle.priority
        self.target_families = (bundle.capability.target_family,)
        self.target_modalities = bundle.capability.target_modalities

    def supports(self, objective: DynamicFoldSearchObjective) -> bool:
        if not isinstance(objective, DynamicFoldSearchObjective):
            return False
        request = objective.target
        return (
            self.bundle.capability.supports(request)
            and (
                request.adapter_id is None
                or request.adapter_id in self.bundle.target_adapter_ids
            )
            and (
                request.fold_method_id is None
                or any(
                    spec.method_id == request.fold_method_id
                    for spec in self.bundle.candidate_specs
                )
            )
        )

    def _target_for_spec(
        self,
        request: DynamicTargetRequest,
        spec: ProductionFoldCandidateSpec,
    ) -> DynamicTargetRequest:
        target_adapter_id = spec.target_adapter_id
        if target_adapter_id is None:
            target_adapter_id = self.bundle.target_adapter_ids[0]
        if request.adapter_id is not None:
            if request.adapter_id != target_adapter_id:
                raise ProductionDynamicFoldCatalogError(
                    f"Candidate {spec.candidate_id!r} conflicts with the "
                    "objective's pinned target adapter."
                )
            target_adapter_id = request.adapter_id
        configuration = dict(request.configuration)
        configuration.update(dict(spec.configuration))
        return replace(
            request,
            child_name=f"{request.child_name}--{spec.candidate_id}",
            adapter_id=target_adapter_id,
            fold_method_id=spec.method_id,
            blank_seed=(
                request.blank_seed if spec.blank_seed is None else spec.blank_seed
            ),
            configuration=tuple(configuration.items()),
        )

    def candidates(
        self,
        objective: DynamicFoldSearchObjective,
    ) -> Sequence[DynamicFoldSearchCandidate]:
        if not self.supports(objective):
            raise ProductionDynamicFoldCatalogError(
                "Portfolio cannot generate candidates outside its capability."
            )
        specs = [
            value
            for value in self.bundle.candidate_specs
            if objective.target.adapter_id is None
            or (value.target_adapter_id or self.bundle.target_adapter_ids[0])
            == objective.target.adapter_id
        ]
        specs = [
            value
            for value in specs
            if objective.target.fold_method_id is None
            or value.method_id == objective.target.fold_method_id
        ]
        specs.sort(key=lambda value: (-value.priority, value.candidate_id))
        specs = specs[: self.bundle.maximum_attempts]
        if not specs:
            raise ProductionDynamicFoldCatalogError(
                "No registered candidate uses the objective's pinned adapter."
            )
        return tuple(
            DynamicFoldSearchCandidate(
                candidate_id=spec.candidate_id,
                target=self._target_for_spec(objective.target, spec),
                evidence_fingerprint=spec.evidence_fingerprint,
                priority=spec.priority,
                metadata=(
                    ("capability_id", self.bundle.capability.capability_id),
                    ("method_id", spec.method_id),
                    *spec.metadata,
                ),
            )
            for spec in specs
        )

    def classify_failure(
        self,
        _candidate: DynamicFoldSearchCandidate,
        error: Exception,
    ) -> FailureDisposition:
        if isinstance(error, self.bundle.candidate_rejection_exceptions):
            return FailureDisposition.CANDIDATE_REJECTED
        if type(error) is DynamicFoldCompilerError and str(error) in (
            _EXPECTED_CANDIDATE_COMPILER_FAILURES
        ):
            return FailureDisposition.CANDIDATE_REJECTED
        if type(error) is DynamicFoldArtifactError and str(error).startswith(
            _EXPECTED_CANDIDATE_ARTIFACT_PREFIXES
        ):
            return FailureDisposition.CANDIDATE_REJECTED
        if isinstance(error, self.bundle.infrastructure_exceptions):
            return FailureDisposition.INFRASTRUCTURE
        return FailureDisposition.FATAL

    def score(
        self,
        _candidate: DynamicFoldSearchCandidate,
        result: Any,
    ) -> float:
        artifact = getattr(getattr(result, "published", None), "artifact", None)
        if not isinstance(artifact, DynamicFoldArtifactCertificate):
            raise ProductionDynamicFoldCatalogError(
                "Default scoring requires a real publication certificate."
            )
        if artifact.certificate_sha256 != canonical_sha256(artifact.body()):
            raise ProductionDynamicFoldCatalogError(
                "Default scoring refuses an unauthenticated certificate."
            )
        decision = artifact.fresh_reload_decision
        if not decision.accepted:
            raise ProductionDynamicFoldCatalogError(
                "Default scoring refuses a rejected fresh-reload decision."
            )
        metrics = dict(decision.metrics)
        score = 1.0
        for term in self.bundle.score_terms:
            if term.metric_name not in metrics:
                if term.required:
                    raise ProductionDynamicFoldCatalogError(
                        "Fresh-reload certificate lacks required score metric "
                        f"{term.metric_name!r}."
                    )
                continue
            metric = metrics[term.metric_name]
            if isinstance(metric, bool):
                numeric = float(metric)
            else:
                numeric = _finite_float(
                    metric,
                    f"certificate metric {term.metric_name}",
                )
            score += term.weight * numeric
        if not math.isfinite(score):
            raise ProductionDynamicFoldCatalogError(
                "Certificate-based candidate score is not finite."
            )
        return score


class _CapabilityBoundStackAdapter:
    """Prevent a stack from leaking into another range with the same family."""

    def __init__(self, bundle: ProductionFoldBundle, adapter: Any) -> None:
        self._bundle = bundle
        self._adapter = adapter
        identity = canonical_sha256(
            {
                "capability": bundle.capability.as_dict(),
                "stack_adapter": _adapter_descriptor(
                    adapter,
                    kind="stack",
                ),
            }
        )[:16]
        self.adapter_id = (
            f"neural_forge.production_bound.{bundle.capability.capability_id}."
            f"{identity}"
        )
        self.version = int(getattr(adapter, "version"))
        self.priority = int(getattr(adapter, "priority"))
        self.target_families = (bundle.capability.target_family,)
        self.target_modalities = bundle.capability.target_modalities
        self.method_ids = _names(
            getattr(adapter, "method_ids", None),
            "stack method_ids",
        )

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        return (
            isinstance(target, DynamicTargetSynthesisResult)
            and self._bundle.capability.supports(target.request)
            and target.adapter_id in self._bundle.target_adapter_ids
            and (
                target.request.fold_method_id is None
                or target.request.fold_method_id in self.method_ids
            )
            and bool(self._adapter.supports(target))
        )

    def build(self, target: DynamicTargetSynthesisResult) -> Any:
        if not self.supports(target):
            raise ProductionDynamicFoldCatalogError(
                "Capability-bound stack received a target outside its bundle."
            )
        return self._adapter.build(target)


@dataclass(slots=True)
class ProductionDynamicFoldFacade:
    """Neutral runtime entrypoint backed by one assembled search system."""

    search_system: AutomaticDynamicFoldSearchSystem
    inspectors: InspectorRegistry
    bundles: tuple[ProductionFoldBundle, ...]

    @property
    def capabilities(self) -> tuple[ProductionFoldCapability, ...]:
        return tuple(bundle.capability for bundle in self.bundles)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": SCHEMA,
                "kind": "assembled_facade",
                "bundles": [bundle.as_dict() for bundle in self.bundles],
            }
        )

    def _select_bundle(
        self,
        objective: DynamicFoldSearchObjective,
    ) -> ProductionFoldBundle:
        if not isinstance(objective, DynamicFoldSearchObjective):
            raise ProductionDynamicFoldCatalogError(
                "objective must be DynamicFoldSearchObjective."
            )
        matches = [
            bundle
            for bundle in self.bundles
            if bundle.capability.supports(objective.target)
        ]
        if len(matches) != 1:
            raise ProductionDynamicFoldCatalogError(
                "No single registered production capability supports the exact "
                "target family, modalities, and parameter range."
            )
        bundle = matches[0]
        if (
            objective.portfolio_id is not None
            and objective.portfolio_id != bundle.portfolio_id
        ):
            raise ProductionDynamicFoldCatalogError(
                "The requested portfolio is not the registered portfolio for "
                "this exact capability."
            )
        if (
            objective.target.adapter_id is not None
            and objective.target.adapter_id not in bundle.target_adapter_ids
        ):
            raise ProductionDynamicFoldCatalogError(
                "The objective pins a target adapter outside the capability."
            )
        return bundle

    def run(
        self,
        *,
        parent_models: Mapping[str, Any],
        objective: DynamicFoldSearchObjective,
        destination_root: str | Path,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    ) -> DynamicFoldSearchResult:
        bundle = self._select_bundle(objective)
        parents = dict(parent_models)
        if not parents:
            raise ProductionDynamicFoldCatalogError(
                "Production folding requires at least one parent model."
            )
        if not bundle.capability.supports_parent_count(len(parents)):
            raise ProductionDynamicFoldCatalogError(
                "Parent-model count is outside the registered production "
                "capability envelope."
            )
        try:
            validate_disjoint_parent_models(parents)
        except DynamicFoldCompilerError as exc:
            raise ProductionDynamicFoldCatalogError(
                "Production parent set is aliased or cannot be authenticated."
            ) from exc
        tokenizer_fingerprints = dict(parent_tokenizer_fingerprints or {})
        if set(tokenizer_fingerprints) - set(parents):
            raise ProductionDynamicFoldCatalogError(
                "Tokenizer fingerprints reference an undeclared parent."
            )
        observed_families, _preflight_guard = run_fold_only_callback(
            lambda: tuple(
                sorted(
                    {
                        self.inspectors.inspect(
                            model,
                            model_name=_text(name, "parent name"),
                            tokenizer_fingerprint=(tokenizer_fingerprints.get(name)),
                        ).family
                        for name, model in parents.items()
                    }
                )
            ),
            phase="production capability inspection",
            protected_models=tuple(parents.values()),
            freeze_parameters=False,
        )
        unsupported = set(observed_families) - set(bundle.capability.source_families)
        if unsupported:
            raise ProductionDynamicFoldCatalogError(
                "Inspected parent families are outside the registered source "
                "capability: " + ", ".join(sorted(unsupported))
            )
        return self.search_system.run(
            parent_models=parents,
            objective=objective,
            destination_root=destination_root,
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        )


class ProductionDynamicFoldCatalog:
    """Register resource-bound bundles once, then seal into one facade."""

    def __init__(
        self,
        bundles: Sequence[ProductionFoldBundle] = (),
        *,
        inspectors: InspectorRegistry | None = None,
        methods: FoldMethodRegistry | None = None,
        ledger: ExperimentLedger | None = None,
        ledger_path: str | Path | None = None,
        allow_ephemeral_ledger: bool = False,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
        require_positive_compaction: bool = True,
    ) -> None:
        self._bundles: list[ProductionFoldBundle] = []
        self._inspectors = inspectors or default_inspector_registry()
        if not isinstance(self._inspectors, InspectorRegistry):
            raise ProductionDynamicFoldCatalogError(
                "inspectors must be InspectorRegistry."
            )
        self._methods = methods
        if not isinstance(allow_ephemeral_ledger, bool):
            raise ProductionDynamicFoldCatalogError(
                "allow_ephemeral_ledger must be boolean."
            )
        if ledger is not None and ledger_path is not None:
            raise ProductionDynamicFoldCatalogError(
                "Provide either ledger or ledger_path, not both."
            )
        if ledger_path is not None:
            self._ledger = DurableExperimentLedger(ledger_path)
        elif ledger is not None:
            self._ledger = ledger
        elif allow_ephemeral_ledger:
            self._ledger = ExperimentLedger()
        else:
            raise ProductionDynamicFoldCatalogError(
                "Production catalogs require a durable ledger_path or an "
                "explicit caller-owned ledger. Ephemeral ledgers are test-only."
            )
        if not isinstance(self._ledger, ExperimentLedger):
            raise ProductionDynamicFoldCatalogError(
                "ledger must be ExperimentLedger or None."
            )
        self._snapshot_config = snapshot_config
        if not isinstance(require_positive_compaction, bool):
            raise ProductionDynamicFoldCatalogError(
                "require_positive_compaction must be boolean."
            )
        self._require_positive_compaction = require_positive_compaction
        self._facade: ProductionDynamicFoldFacade | None = None
        for bundle in bundles:
            self.register(bundle)

    @property
    def sealed(self) -> bool:
        return self._facade is not None

    @property
    def bundles(self) -> tuple[ProductionFoldBundle, ...]:
        return tuple(
            sorted(
                self._bundles,
                key=lambda value: value.capability.capability_id,
            )
        )

    def register(self, bundle: ProductionFoldBundle) -> None:
        if self.sealed:
            raise ProductionDynamicFoldCatalogError(
                "The production catalog is sealed; adapters register only once."
            )
        if not isinstance(bundle, ProductionFoldBundle):
            raise ProductionDynamicFoldCatalogError(
                "Catalog entries must be ProductionFoldBundle objects."
            )
        for existing in self._bundles:
            if (
                existing.capability.capability_id == bundle.capability.capability_id
                or existing.portfolio_id == bundle.portfolio_id
            ):
                raise ProductionDynamicFoldCatalogError(
                    "Capability and portfolio IDs must be globally unique."
                )
            same_contract = (
                existing.capability.target_family == bundle.capability.target_family
                and existing.capability.target_modalities
                == bundle.capability.target_modalities
            )
            ranges_overlap = not (
                existing.capability.maximum_target_unique_parameters
                < bundle.capability.minimum_target_unique_parameters
                or bundle.capability.maximum_target_unique_parameters
                < existing.capability.minimum_target_unique_parameters
            )
            if same_contract and ranges_overlap:
                raise ProductionDynamicFoldCatalogError(
                    "Production capability target ranges overlap ambiguously."
                )
        self._bundles.append(bundle)

    @staticmethod
    def _unique_adapters(
        bundles: Sequence[ProductionFoldBundle],
        attribute: str,
    ) -> tuple[Any, ...]:
        result: dict[str, Any] = {}
        for bundle in bundles:
            for adapter in getattr(bundle, attribute):
                adapter_id = _adapter_id(adapter, f"{attribute} adapter_id")
                existing = result.get(adapter_id)
                if existing is not None and existing is not adapter:
                    raise ProductionDynamicFoldCatalogError(
                        f"Adapter ID {adapter_id!r} is bound to multiple objects."
                    )
                result[adapter_id] = adapter
        return tuple(result[key] for key in sorted(result))

    def assemble(self) -> ProductionDynamicFoldFacade:
        if self._facade is not None:
            return self._facade
        bundles = self.bundles
        if not bundles:
            raise ProductionDynamicFoldCatalogError(
                "Cannot assemble an empty production catalog."
            )
        target_adapters = self._unique_adapters(bundles, "target_adapters")
        stack_adapters = tuple(
            _CapabilityBoundStackAdapter(bundle, adapter)
            for bundle in bundles
            for adapter in bundle.stack_adapters
        )
        folding_system = AutomaticDynamicFoldingSystem(
            target_adapters=DynamicTargetAdapterRegistry(target_adapters),
            stack_adapters=DynamicFoldStackAdapterRegistry(stack_adapters),
            inspectors=self._inspectors,
            methods=self._methods,
            ledger=self._ledger,
            snapshot_config=self._snapshot_config,
            require_positive_compaction=self._require_positive_compaction,
        )
        search_system = AutomaticDynamicFoldSearchSystem(
            folding_system=folding_system,
            portfolios=DynamicFoldPortfolioRegistry(
                tuple(RegisteredExactTargetPortfolio(value) for value in bundles)
            ),
        )
        self._facade = ProductionDynamicFoldFacade(
            search_system=search_system,
            inspectors=self._inspectors,
            bundles=bundles,
        )
        return self._facade


__all__ = [
    "CertificateScoreTerm",
    "ProductionCandidateRejected",
    "ProductionDynamicFoldCatalog",
    "ProductionDynamicFoldCatalogError",
    "ProductionDynamicFoldFacade",
    "ProductionFoldBundle",
    "ProductionFoldCandidateSpec",
    "ProductionFoldCapability",
    "ProductionInfrastructureFailure",
    "RegisteredExactTargetPortfolio",
    "SCHEMA",
]
