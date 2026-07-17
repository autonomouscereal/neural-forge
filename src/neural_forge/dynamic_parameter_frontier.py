"""Catalog-driven dense blank-target topology frontiers.

The folding compiler already knows how to authenticate an exact concrete
blank target, but callers previously had to know a representable parameter
count and construct its topology themselves.  This module fills that upstream
gap without touching model weights:

* a request names a target family/modalities and either a unique learned
  parameter budget or a compression ratio;
* a registered family catalog reads an inspected :class:`ModelCircuitGraph`;
* the catalog emits legal dense topology/configuration candidates;
* common code ranks candidates and proves exact logical/unique accounting.

Only real dense topology reduction is represented here.  Quantization,
pruning, sparse storage, packing, factorization, codebooks, and non-native
parameter sharing are rejected.  Architecture-native ties (for example a
tied token embedding and LM head) are explicit and counted once in the unique
learned-parameter total and twice in the logical parameter total.

The first concrete catalog covers LFM2 hybrid text decoders.  New families or
modalities extend the registry with another catalog; they do not add manual
layer surgery to this module.

All ``*_retention`` values below describe retained topology geometry only.
They are allocation objectives for choosing what the folding compiler can
attempt, never evidence of retained intelligence or behavioral capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Mapping, Protocol, Sequence

from .dynamic_circuit_ir import (
    CircuitKind,
    JsonScalar,
    ModelCircuitGraph,
    canonical_sha256,
)
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    strict_tokenizer_bridge_configuration,
)


SCHEMA = "neural-forge-dense-parameter-frontier-v1"
LFM2_CATALOG_ID = "neural-forge.catalog.lfm2-dense-topology-v1"
_POLICIES = frozenset({"exact", "at_or_below", "nearest"})

CAP_DENSE_BLANK_MATERIALIZATION = "dense_blank_materialization"
CAP_RESIDUAL_WIDTH_PROJECTION = "residual_width_projection"
CAP_FFN_WIDTH_PROJECTION = "ffn_width_projection"
CAP_ATTENTION_TOPOLOGY_PROJECTION = "attention_topology_projection"
CAP_ORDERED_DEPTH_COMPOSITION = "ordered_depth_composition"
CAP_HYBRID_OPERATOR_COMPOSITION = "hybrid_operator_composition"
CAP_TOKENIZER_BRIDGE = "tokenizer_bridge"

# These are the transformations exercised by the current full-depth LFM2
# compiler path.  Depth/operator composition and tokenizer translation are
# deliberately absent: topology search may describe such targets, but the
# automatic selector must not claim it can execute them.
LFM2_CURRENT_COMPILER_CAPABILITIES = (
    CAP_ATTENTION_TOPOLOGY_PROJECTION,
    CAP_DENSE_BLANK_MATERIALIZATION,
    CAP_FFN_WIDTH_PROJECTION,
    CAP_RESIDUAL_WIDTH_PROJECTION,
)

_PARETO_AXES = (
    "budget_utilization",
    "residual_width_retention",
    "depth_retention",
    "attention_role_retention",
    "query_channel_retention",
    "head_dim_retention",
    "gqa_ratio_retention",
    "ffn_width_retention",
    "vocabulary_budget_headroom",
    "compiler_executable",
)


class DynamicParameterFrontierError(ValueError):
    """A frontier request, catalog, or candidate is invalid."""


class DynamicParameterFrontierUnattainable(DynamicParameterFrontierError):
    """No legal dense topology satisfies the requested budget policy."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)
        self.report = self.diagnostics


class DynamicParameterFrontierCapabilityError(
    DynamicParameterFrontierUnattainable
):
    """No returned dense topology is executable by declared capabilities."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicParameterFrontierError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicParameterFrontierError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicParameterFrontierError(
            f"{name} must be a non-negative integer."
        )
    return value


def _modalities(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted({_text(value, "modality").lower() for value in values}))
    if not result:
        raise DynamicParameterFrontierError(
            "At least one target modality is required."
        )
    return result


def _configuration(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
) -> tuple[tuple[str, JsonScalar], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    result: list[tuple[str, JsonScalar]] = []
    for key, value in items:
        normalized_key = _text(str(key), "configuration key")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise DynamicParameterFrontierError(
                f"configuration.{normalized_key} must be a JSON scalar."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise DynamicParameterFrontierError(
                f"configuration.{normalized_key} must be finite."
            )
        result.append((normalized_key, value))
    result.sort(key=lambda item: item[0])
    if len({key for key, _value in result}) != len(result):
        raise DynamicParameterFrontierError(
            "Target configuration keys must be unique."
        )
    return tuple(result)


def _capabilities(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(sorted({_text(value, name) for value in values}))
    return normalized


def _retention_profile(
    values: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: list[tuple[str, int]] = []
    for key, value in items:
        axis = _text(str(key), "retention-profile axis")
        ppm = _nonnegative_int(value, f"retention_profile_ppm.{axis}")
        if ppm > 1_000_000:
            raise DynamicParameterFrontierError(
                f"retention_profile_ppm.{axis} cannot exceed 1,000,000."
            )
        normalized.append((axis, ppm))
    normalized.sort()
    if len({key for key, _value in normalized}) != len(normalized):
        raise DynamicParameterFrontierError(
            "Retention-profile axes must be unique."
        )
    return tuple(normalized)


def _ratio(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise DynamicParameterFrontierError(
            "compression_ratio must be a finite number greater than one."
        )
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise DynamicParameterFrontierError(
            "compression_ratio must be a finite number greater than one."
        ) from exc
    if result <= 1:
        raise DynamicParameterFrontierError(
            "compression_ratio must be greater than one."
        )
    return result


@dataclass(frozen=True, slots=True)
class DenseParameterBudgetRequest:
    """One family-neutral blank-target budget request.

    Exactly one of ``target_unique_parameters`` and ``compression_ratio`` is
    required.  The parameter budget always refers to unique dense learned
    elements after architecture-native ties are counted once.
    """

    target_family: str
    target_modalities: tuple[str, ...]
    target_unique_parameters: int | None = None
    compression_ratio: str | int | float | None = None
    target_vocabulary_size: int | None = None
    match_policy: str = "at_or_below"
    maximum_candidates: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_family",
            _text(self.target_family, "target_family").lower(),
        )
        object.__setattr__(
            self,
            "target_modalities",
            _modalities(self.target_modalities),
        )
        supplied = (
            self.target_unique_parameters is not None,
            self.compression_ratio is not None,
        )
        if sum(supplied) != 1:
            raise DynamicParameterFrontierError(
                "Supply exactly one target_unique_parameters or compression_ratio."
            )
        if self.target_unique_parameters is not None:
            object.__setattr__(
                self,
                "target_unique_parameters",
                _positive_int(
                    self.target_unique_parameters,
                    "target_unique_parameters",
                ),
            )
        if self.compression_ratio is not None:
            normalized = _ratio(self.compression_ratio)
            object.__setattr__(
                self,
                "compression_ratio",
                (
                    str(normalized.numerator)
                    if normalized.denominator == 1
                    else f"{normalized.numerator}/{normalized.denominator}"
                ),
            )
        if self.target_vocabulary_size is not None:
            object.__setattr__(
                self,
                "target_vocabulary_size",
                _positive_int(
                    self.target_vocabulary_size,
                    "target_vocabulary_size",
                ),
            )
        policy = _text(self.match_policy, "match_policy").lower()
        if policy not in _POLICIES:
            raise DynamicParameterFrontierError(
                "match_policy must be exact, at_or_below, or nearest."
            )
        object.__setattr__(self, "match_policy", policy)
        object.__setattr__(
            self,
            "maximum_candidates",
            _positive_int(self.maximum_candidates, "maximum_candidates"),
        )

    def resolve_unique_parameter_budget(self, source_unique_parameters: int) -> int:
        source = _positive_int(
            source_unique_parameters,
            "source_unique_parameters",
        )
        if self.target_unique_parameters is not None:
            budget = self.target_unique_parameters
        else:
            ratio = _ratio(self.compression_ratio)
            budget = source * ratio.denominator // ratio.numerator
        if budget < 1:
            raise DynamicParameterFrontierUnattainable(
                "The requested compression ratio resolves below one learned "
                "parameter.",
                {
                    "source_unique_parameters": source,
                    "requested_compression_ratio": self.compression_ratio,
                    "resolved_target_unique_parameters": budget,
                    "reason": "ratio_below_one_parameter",
                },
            )
        if budget >= source:
            raise DynamicParameterFrontierError(
                "A dense compression target must be smaller than the inspected "
                "parent."
            )
        return budget

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        ratio = (
            None
            if self.compression_ratio is None
            else _ratio(self.compression_ratio)
        )
        return {
            "schema": SCHEMA,
            "kind": "budget_request",
            "target_family": self.target_family,
            "target_modalities": list(self.target_modalities),
            "target_unique_parameters": self.target_unique_parameters,
            "compression_ratio": (
                None
                if ratio is None
                else {
                    "numerator": ratio.numerator,
                    "denominator": ratio.denominator,
                }
            ),
            "target_vocabulary_size": self.target_vocabulary_size,
            "match_policy": self.match_policy,
            "maximum_candidates": self.maximum_candidates,
        }


@dataclass(frozen=True, slots=True)
class DenseTargetTopologyCandidate:
    """One fully counted dense blank-target topology/configuration."""

    candidate_id: str
    catalog_id: str
    family: str
    modalities: tuple[str, ...]
    requested_unique_parameters: int
    source_unique_parameters: int
    unique_parameters: int
    logical_parameters: int
    native_tied_parameter_count: int
    retention_score_ppm: int
    configuration: tuple[tuple[str, JsonScalar], ...]
    ordered_source_spans: tuple[tuple[int, ...], ...]
    native_ties: tuple[tuple[str, str], ...] = ()
    source_vocabulary_size: int | None = None
    target_vocabulary_size: int | None = None
    vocabulary_unique_parameters: int | None = None
    tokenizer_bridge_required: bool = False
    retention_profile_ppm: tuple[tuple[str, int], ...] = ()
    required_compiler_capabilities: tuple[str, ...] = ()
    declared_compiler_capabilities: tuple[str, ...] = ()
    missing_compiler_capabilities: tuple[str, ...] = ()
    compiler_executable: bool = False
    pareto_support_profiles: tuple[str, ...] = ()
    pareto_dominance_verified: bool = False
    dense_parameters: bool = True
    quantized: bool = False
    pruned: bool = False
    sparse_storage: bool = False
    packed: bool = False
    factorized: bool = False
    codebook_backed: bool = False
    non_native_weight_sharing: bool = False

    def __post_init__(self) -> None:
        for name in ("candidate_id", "catalog_id", "family"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "modalities", _modalities(self.modalities))
        for name in (
            "requested_unique_parameters",
            "source_unique_parameters",
            "unique_parameters",
            "logical_parameters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "native_tied_parameter_count",
            _nonnegative_int(
                self.native_tied_parameter_count,
                "native_tied_parameter_count",
            ),
        )
        score = _nonnegative_int(self.retention_score_ppm, "retention_score_ppm")
        if score > 1_000_000:
            raise DynamicParameterFrontierError(
                "retention_score_ppm cannot exceed 1,000,000."
            )
        object.__setattr__(self, "retention_score_ppm", score)
        object.__setattr__(
            self,
            "configuration",
            _configuration(self.configuration),
        )
        spans = tuple(tuple(span) for span in self.ordered_source_spans)
        if not spans or any(not span for span in spans):
            raise DynamicParameterFrontierError(
                "ordered_source_spans must contain one nonempty span per target "
                "layer."
            )
        flattened = tuple(index for span in spans for index in span)
        if (
            any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in flattened
            )
            or flattened != tuple(range(len(flattened)))
        ):
            raise DynamicParameterFrontierError(
                "ordered_source_spans must cover every source layer exactly once "
                "in causal order."
            )
        object.__setattr__(self, "ordered_source_spans", spans)
        ties = tuple(sorted(tuple(pair) for pair in self.native_ties))
        if any(
            len(pair) != 2
            or not all(isinstance(value, str) and value.strip() for value in pair)
            for pair in ties
        ):
            raise DynamicParameterFrontierError(
                "native_ties must contain pairs of parameter-role names."
            )
        object.__setattr__(self, "native_ties", ties)
        if not isinstance(self.dense_parameters, bool) or not self.dense_parameters:
            raise DynamicParameterFrontierError(
                "Frontier candidates must use dense learned parameters."
            )
        forbidden = {
            name: getattr(self, name)
            for name in (
                "quantized",
                "pruned",
                "sparse_storage",
                "packed",
                "factorized",
                "codebook_backed",
                "non_native_weight_sharing",
            )
        }
        if any(not isinstance(value, bool) for value in forbidden.values()):
            raise DynamicParameterFrontierError(
                "Compression-mechanism audit flags must be boolean."
            )
        enabled = sorted(name for name, value in forbidden.items() if value)
        if enabled:
            raise DynamicParameterFrontierError(
                "Dense topology frontier rejects parameter-count cheats: "
                + ", ".join(enabled)
                + "."
            )
        if self.unique_parameters >= self.source_unique_parameters:
            raise DynamicParameterFrontierError(
                "A frontier candidate must remove unique learned parameters."
            )
        if self.logical_parameters < self.unique_parameters:
            raise DynamicParameterFrontierError(
                "logical_parameters cannot be below unique_parameters."
            )
        if (
            self.logical_parameters - self.unique_parameters
            != self.native_tied_parameter_count
        ):
            raise DynamicParameterFrontierError(
                "Logical/unique accounting does not equal the declared native tie."
            )
        if bool(self.native_ties) != bool(self.native_tied_parameter_count):
            raise DynamicParameterFrontierError(
                "Native tie roles and native tied parameter count disagree."
            )
        vocabulary_values = (
            self.source_vocabulary_size,
            self.target_vocabulary_size,
            self.vocabulary_unique_parameters,
        )
        if any(value is not None for value in vocabulary_values):
            if any(value is None for value in vocabulary_values):
                raise DynamicParameterFrontierError(
                    "Vocabulary accounting fields must be supplied together."
                )
            for name in (
                "source_vocabulary_size",
                "target_vocabulary_size",
                "vocabulary_unique_parameters",
            ):
                object.__setattr__(
                    self,
                    name,
                    _positive_int(getattr(self, name), name),
                )
        if not isinstance(self.tokenizer_bridge_required, bool):
            raise DynamicParameterFrontierError(
                "tokenizer_bridge_required must be boolean."
            )
        if (
            self.source_vocabulary_size is not None
            and self.tokenizer_bridge_required
            != (self.source_vocabulary_size != self.target_vocabulary_size)
        ):
            raise DynamicParameterFrontierError(
                "Tokenizer-bridge declaration disagrees with source/target "
                "vocabulary sizes."
            )
        profile = _retention_profile(self.retention_profile_ppm)
        object.__setattr__(self, "retention_profile_ppm", profile)
        required = _capabilities(
            self.required_compiler_capabilities,
            "required compiler capability",
        )
        declared = _capabilities(
            self.declared_compiler_capabilities,
            "declared compiler capability",
        )
        missing = _capabilities(
            self.missing_compiler_capabilities,
            "missing compiler capability",
        )
        object.__setattr__(self, "required_compiler_capabilities", required)
        object.__setattr__(self, "declared_compiler_capabilities", declared)
        object.__setattr__(self, "missing_compiler_capabilities", missing)
        if not isinstance(self.compiler_executable, bool):
            raise DynamicParameterFrontierError(
                "compiler_executable must be boolean."
            )
        has_capability_provenance = bool(
            required
            or declared
            or missing
            or profile
            or self.pareto_support_profiles
            or self.pareto_dominance_verified
        )
        if has_capability_provenance:
            expected_missing = tuple(
                sorted(set(required).difference(declared))
            )
            if missing != expected_missing:
                raise DynamicParameterFrontierError(
                    "Missing compiler capabilities must exactly equal required "
                    "minus declared capabilities."
                )
            if self.compiler_executable != (not missing):
                raise DynamicParameterFrontierError(
                    "compiler_executable disagrees with the audited capability "
                    "difference."
                )
        support = _capabilities(
            self.pareto_support_profiles,
            "Pareto support profile",
        )
        object.__setattr__(self, "pareto_support_profiles", support)
        if not isinstance(self.pareto_dominance_verified, bool):
            raise DynamicParameterFrontierError(
                "pareto_dominance_verified must be boolean."
            )
        if self.pareto_dominance_verified:
            axes = tuple(key for key, _value in profile)
            if axes != tuple(sorted(_PARETO_AXES)):
                raise DynamicParameterFrontierError(
                    "A Pareto-verified candidate must carry every audited "
                    "retention/capability axis exactly once."
                )
            if not support:
                raise DynamicParameterFrontierError(
                    "A Pareto-verified candidate needs at least one supporting "
                    "scalarization profile."
                )

    @property
    def budget_delta(self) -> int:
        return self.unique_parameters - self.requested_unique_parameters

    @property
    def at_or_below_budget(self) -> bool:
        return self.budget_delta <= 0

    @property
    def exact_budget_match(self) -> bool:
        return self.budget_delta == 0

    @property
    def compression_ratio(self) -> float:
        return self.source_unique_parameters / self.unique_parameters

    @property
    def retention_profile(self) -> dict[str, int]:
        return dict(self.retention_profile_ppm)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def to_target_request(
        self,
        *,
        child_name: str,
        tokenizer_fingerprint: str | None = None,
        adapter_id: str | None = None,
        fold_method_id: str | None = None,
        blank_seed: int = 0,
    ) -> Any:
        """Create the existing exact blank-target request for this candidate."""

        from .dynamic_target_synthesis import DynamicTargetRequest

        return DynamicTargetRequest(
            child_name=child_name,
            target_family=self.family,
            target_modalities=self.modalities,
            exact_unique_parameters=self.unique_parameters,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=adapter_id,
            fold_method_id=fold_method_id,
            blank_seed=blank_seed,
            configuration=self.configuration,
        )

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "dense_target_topology_candidate",
            "candidate_id": self.candidate_id,
            "catalog_id": self.catalog_id,
            "family": self.family,
            "modalities": list(self.modalities),
            "requested_unique_parameters": self.requested_unique_parameters,
            "source_unique_parameters": self.source_unique_parameters,
            "unique_parameters": self.unique_parameters,
            "logical_parameters": self.logical_parameters,
            "native_tied_parameter_count": self.native_tied_parameter_count,
            "budget_delta": self.budget_delta,
            "at_or_below_budget": self.at_or_below_budget,
            "exact_budget_match": self.exact_budget_match,
            "compression_ratio": self.compression_ratio,
            "retention_score_ppm": self.retention_score_ppm,
            "configuration": dict(self.configuration),
            "ordered_source_spans": [
                list(span) for span in self.ordered_source_spans
            ],
            "native_ties": [list(pair) for pair in self.native_ties],
            "source_vocabulary_size": self.source_vocabulary_size,
            "target_vocabulary_size": self.target_vocabulary_size,
            "vocabulary_unique_parameters": self.vocabulary_unique_parameters,
            "vocabulary_budget_share": (
                None
                if self.vocabulary_unique_parameters is None
                else self.vocabulary_unique_parameters
                / self.requested_unique_parameters
            ),
            "tokenizer_bridge_required": self.tokenizer_bridge_required,
            "tokenizer_mapping_implemented": False,
            "retention_profile_ppm": dict(self.retention_profile_ppm),
            "retention_profile_semantics": (
                "structural_topology_allocation_only"
            ),
            "behavioral_capability_retention_claimed": False,
            "intelligence_retention_inferred_from_parameter_arithmetic": False,
            "required_compiler_capabilities": list(
                self.required_compiler_capabilities
            ),
            "declared_compiler_capabilities": list(
                self.declared_compiler_capabilities
            ),
            "missing_compiler_capabilities": list(
                self.missing_compiler_capabilities
            ),
            "compiler_executable": self.compiler_executable,
            "pareto_support_profiles": list(self.pareto_support_profiles),
            "pareto_dominance_verified": self.pareto_dominance_verified,
            "blank_target_configuration": True,
            "model_weights_created_or_loaded": False,
            "materialization_requires_compiler_zero_blank": True,
            "dense_parameters": self.dense_parameters,
            "quantized": self.quantized,
            "pruned": self.pruned,
            "sparse_storage": self.sparse_storage,
            "packed": self.packed,
            "factorized": self.factorized,
            "codebook_backed": self.codebook_backed,
            "non_native_weight_sharing": self.non_native_weight_sharing,
            "parameter_count_reduction_is_topological": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class CatalogFrontier:
    candidates: tuple[DenseTargetTopologyCandidate, ...]
    diagnostics: Mapping[str, Any]


class DenseTargetTopologyCatalog(Protocol):
    catalog_id: str
    version: int
    priority: int
    target_families: tuple[str, ...]
    target_modalities: tuple[str, ...]

    def supports(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        target_family: str,
        target_modalities: tuple[str, ...],
    ) -> bool: ...

    def enumerate_frontier(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        requested_unique_parameters: int,
        target_vocabulary_size: int | None,
        match_policy: str,
        maximum_candidates: int,
    ) -> CatalogFrontier: ...


class DenseTargetTopologyCatalogRegistry:
    """Deterministically select one family/modality topology catalog."""

    def __init__(
        self,
        catalogs: Sequence[DenseTargetTopologyCatalog] = (),
    ) -> None:
        self._catalogs: dict[str, DenseTargetTopologyCatalog] = {}
        for catalog in catalogs:
            self.register(catalog)

    def register(self, catalog: DenseTargetTopologyCatalog) -> None:
        catalog_id = _text(getattr(catalog, "catalog_id", None), "catalog_id")
        version = getattr(catalog, "version", None)
        priority = getattr(catalog, "priority", None)
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise DynamicParameterFrontierError(
                "Topology catalog version must be positive."
            )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicParameterFrontierError(
                "Topology catalog priority must be an integer."
            )
        if not callable(getattr(catalog, "supports", None)) or not callable(
            getattr(catalog, "enumerate_frontier", None)
        ):
            raise DynamicParameterFrontierError(
                "Topology catalogs must implement supports() and "
                "enumerate_frontier()."
            )
        if catalog_id in self._catalogs:
            raise DynamicParameterFrontierError(
                f"Topology catalog {catalog_id!r} is already registered."
            )
        self._catalogs[catalog_id] = catalog

    @property
    def catalog_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._catalogs))

    def select(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        target_family: str,
        target_modalities: tuple[str, ...],
    ) -> DenseTargetTopologyCatalog:
        candidates = [
            value
            for value in self._catalogs.values()
            if value.supports(
                parent_graph,
                target_family=target_family,
                target_modalities=target_modalities,
            )
        ]
        if not candidates:
            raise DynamicParameterFrontierError(
                "No registered dense topology catalog supports the inspected "
                "parent, target family, and exact modality set."
            )
        candidates.sort(
            key=lambda value: (-int(value.priority), str(value.catalog_id))
        )
        if (
            len(candidates) > 1
            and int(candidates[0].priority) == int(candidates[1].priority)
        ):
            raise DynamicParameterFrontierError(
                "Multiple dense topology catalogs have equal highest priority."
            )
        return candidates[0]


@dataclass(frozen=True, slots=True)
class DenseParameterFrontierResult:
    request: DenseParameterBudgetRequest
    parent_graph_fingerprint: str
    source_unique_parameters: int
    requested_unique_parameters: int
    catalog_id: str
    catalog_version: int
    candidates: tuple[DenseTargetTopologyCandidate, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise DynamicParameterFrontierError(
                "A successful frontier result needs at least one candidate."
            )
        if any(
            candidate.requested_unique_parameters
            != self.requested_unique_parameters
            or candidate.source_unique_parameters != self.source_unique_parameters
            or candidate.family != self.request.target_family
            or candidate.modalities != self.request.target_modalities
            for candidate in self.candidates
        ):
            raise DynamicParameterFrontierError(
                "A frontier candidate is not bound to its request/source."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "dense_parameter_frontier",
            "request": self.request.as_dict(),
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "source_unique_parameters": self.source_unique_parameters,
            "requested_unique_parameters": self.requested_unique_parameters,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "diagnostics": dict(self.diagnostics),
            "ranked": True,
            "weights_or_downloads_used": False,
            "native_ties_counted_once": True,
            "quantization_or_pruning_used": False,
            "packing_or_factorization_used": False,
            "non_native_weight_sharing_used": False,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class DenseExecutableCandidateSelection:
    """Audited selection of one currently executable supported-Pareto point."""

    frontier_fingerprint: str
    selected_candidate_rank: int
    selected_candidate_fingerprint: str
    available_compiler_capabilities: tuple[str, ...]
    selection_policy: str
    executable_candidate_count: int
    rejected_candidates: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frontier_fingerprint",
            _text(self.frontier_fingerprint, "frontier_fingerprint"),
        )
        object.__setattr__(
            self,
            "selected_candidate_fingerprint",
            _text(
                self.selected_candidate_fingerprint,
                "selected_candidate_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "available_compiler_capabilities",
            _capabilities(
                self.available_compiler_capabilities,
                "available compiler capability",
            ),
        )
        object.__setattr__(
            self,
            "selection_policy",
            _text(self.selection_policy, "selection_policy"),
        )
        object.__setattr__(
            self,
            "selected_candidate_rank",
            _nonnegative_int(
                self.selected_candidate_rank,
                "selected_candidate_rank",
            ),
        )
        object.__setattr__(
            self,
            "executable_candidate_count",
            _positive_int(
                self.executable_candidate_count,
                "executable_candidate_count",
            ),
        )
        rejected = tuple(
            sorted(
                (
                    _text(fingerprint, "rejected candidate fingerprint"),
                    _capabilities(missing, "missing compiler capability"),
                )
                for fingerprint, missing in self.rejected_candidates
            )
        )
        object.__setattr__(self, "rejected_candidates", rejected)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "dense_executable_candidate_selection",
            "frontier_fingerprint": self.frontier_fingerprint,
            "selected_candidate_rank": self.selected_candidate_rank,
            "selected_candidate_fingerprint": (
                self.selected_candidate_fingerprint
            ),
            "available_compiler_capabilities": list(
                self.available_compiler_capabilities
            ),
            "selection_policy": self.selection_policy,
            "selection_reason": (
                "all_required_compiler_transforms_are_declared_then_choose_"
                "the_best_circuit_balanced_supported_pareto_allocation"
            ),
            "exact_parameter_distance_is_not_the_primary_selection_rule": True,
            "executable_candidate_count": self.executable_candidate_count,
            "rejected_candidates": [
                {
                    "candidate_fingerprint": fingerprint,
                    "missing_compiler_capabilities": list(missing),
                }
                for fingerprint, missing in self.rejected_candidates
            ],
            "fail_closed": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def _executable_pareto_selection_rank(
    candidate: DenseTargetTopologyCandidate,
) -> tuple[Any, ...]:
    profile = candidate.retention_profile
    # Capability is a hard eligibility gate.  Among executable candidates,
    # favor a balanced allocation rather than whichever topology happens to
    # land one parameter closer to the requested count.
    weights = {
        "budget_utilization": 6,
        "residual_width_retention": 3,
        "depth_retention": 3,
        "attention_role_retention": 2,
        "query_channel_retention": 2,
        "head_dim_retention": 2,
        "gqa_ratio_retention": 2,
        "ffn_width_retention": 2,
        "vocabulary_budget_headroom": 1,
        "compiler_executable": 1,
    }
    balanced_score = sum(
        weights[axis] * profile.get(axis, 0) for axis in _PARETO_AXES
    )
    return (
        -balanced_score,
        abs(candidate.budget_delta),
        -candidate.retention_score_ppm,
        -candidate.unique_parameters,
        tuple((key, str(value)) for key, value in candidate.configuration),
    )


def select_best_executable_dense_candidate(
    frontier: DenseParameterFrontierResult,
    *,
    available_compiler_capabilities: Sequence[str] | None = None,
) -> DenseExecutableCandidateSelection:
    """Select the best supported-Pareto topology the compiler can execute.

    With no override, the catalog's capability declaration is used only when
    every returned candidate carries the same declaration.  Missing or
    inconsistent provenance fails closed; it never silently falls back to
    candidate rank zero.
    """

    if not isinstance(frontier, DenseParameterFrontierResult):
        raise DynamicParameterFrontierError(
            "frontier must be DenseParameterFrontierResult."
        )
    if available_compiler_capabilities is None:
        declarations = {
            candidate.declared_compiler_capabilities
            for candidate in frontier.candidates
        }
        if len(declarations) != 1:
            raise DynamicParameterFrontierCapabilityError(
                "Frontier candidates do not share one audited compiler "
                "capability declaration.",
                {
                    "frontier_fingerprint": frontier.fingerprint,
                    "reason": "inconsistent_capability_declarations",
                    "declaration_count": len(declarations),
                },
            )
        available = next(iter(declarations))
    else:
        available = _capabilities(
            available_compiler_capabilities,
            "available compiler capability",
        )
    available_set = set(available)
    executable: list[tuple[int, DenseTargetTopologyCandidate]] = []
    rejected: list[tuple[str, tuple[str, ...]]] = []
    for rank, candidate in enumerate(frontier.candidates):
        missing = tuple(
            sorted(
                set(candidate.required_compiler_capabilities).difference(
                    available_set
                )
            )
        )
        if not candidate.pareto_dominance_verified:
            missing = (*missing, "pareto_provenance")
        if missing:
            rejected.append((candidate.fingerprint, tuple(sorted(set(missing)))))
            continue
        executable.append((rank, candidate))
    if not executable:
        raise DynamicParameterFrontierCapabilityError(
            "No supported-Pareto dense target is executable by the declared "
            "compiler capabilities.",
            {
                "frontier_fingerprint": frontier.fingerprint,
                "reason": "no_executable_pareto_candidate",
                "available_compiler_capabilities": list(available),
                "rejected_candidates": [
                    {
                        "candidate_fingerprint": fingerprint,
                        "missing_compiler_capabilities": list(missing),
                    }
                    for fingerprint, missing in rejected
                ],
            },
        )
    selected_rank, selected = min(
        executable,
        key=lambda item: _executable_pareto_selection_rank(item[1]),
    )
    return DenseExecutableCandidateSelection(
        frontier_fingerprint=frontier.fingerprint,
        selected_candidate_rank=selected_rank,
        selected_candidate_fingerprint=selected.fingerprint,
        available_compiler_capabilities=available,
        selection_policy="supported_pareto_balanced_executable_v1",
        executable_candidate_count=len(executable),
        rejected_candidates=tuple(rejected),
    )


class DenseParameterFrontierSynthesizer:
    """Resolve a catalog and produce a validated deterministic frontier."""

    def __init__(self, catalogs: DenseTargetTopologyCatalogRegistry) -> None:
        if not isinstance(catalogs, DenseTargetTopologyCatalogRegistry):
            raise DynamicParameterFrontierError(
                "catalogs must be DenseTargetTopologyCatalogRegistry."
            )
        self.catalogs = catalogs

    def synthesize(
        self,
        parent_graph: ModelCircuitGraph,
        request: DenseParameterBudgetRequest,
    ) -> DenseParameterFrontierResult:
        if not isinstance(parent_graph, ModelCircuitGraph):
            raise DynamicParameterFrontierError(
                "parent_graph must be an inspected ModelCircuitGraph."
            )
        if not isinstance(request, DenseParameterBudgetRequest):
            raise DynamicParameterFrontierError(
                "request must be DenseParameterBudgetRequest."
            )
        budget = request.resolve_unique_parameter_budget(
            parent_graph.unique_parameter_count
        )
        catalog = self.catalogs.select(
            parent_graph,
            target_family=request.target_family,
            target_modalities=request.target_modalities,
        )
        frontier = catalog.enumerate_frontier(
            parent_graph,
            requested_unique_parameters=budget,
            target_vocabulary_size=request.target_vocabulary_size,
            match_policy=request.match_policy,
            maximum_candidates=request.maximum_candidates,
        )
        if not isinstance(frontier, CatalogFrontier):
            raise DynamicParameterFrontierError(
                "Topology catalog returned an invalid frontier object."
            )
        candidates = tuple(frontier.candidates)
        if not candidates:
            diagnostics = {
                **dict(frontier.diagnostics),
                "request": request.as_dict(),
                "parent_graph_fingerprint": parent_graph.fingerprint,
            }
            raise DynamicParameterFrontierUnattainable(
                "No legal dense target topology satisfies the requested budget "
                "policy.",
                diagnostics,
            )
        if len(candidates) > request.maximum_candidates:
            raise DynamicParameterFrontierError(
                "Topology catalog exceeded maximum_candidates."
            )
        for candidate in candidates:
            if not isinstance(candidate, DenseTargetTopologyCandidate):
                raise DynamicParameterFrontierError(
                    "Topology catalog returned a non-candidate value."
                )
            if (
                candidate.catalog_id != str(catalog.catalog_id)
                or candidate.family != request.target_family
                or candidate.modalities != request.target_modalities
                or candidate.requested_unique_parameters != budget
                or candidate.source_unique_parameters
                != parent_graph.unique_parameter_count
            ):
                raise DynamicParameterFrontierError(
                    "Topology catalog returned a candidate outside its bound "
                    "request."
                )
            if request.match_policy == "exact" and not candidate.exact_budget_match:
                raise DynamicParameterFrontierError(
                    "Exact policy returned a non-exact candidate."
                )
            if (
                request.match_policy == "at_or_below"
                and not candidate.at_or_below_budget
            ):
                raise DynamicParameterFrontierError(
                    "At-or-below policy returned an over-budget candidate."
                )
        return DenseParameterFrontierResult(
            request=request,
            parent_graph_fingerprint=parent_graph.fingerprint,
            source_unique_parameters=parent_graph.unique_parameter_count,
            requested_unique_parameters=budget,
            catalog_id=str(catalog.catalog_id),
            catalog_version=int(catalog.version),
            candidates=candidates,
            diagnostics=dict(frontier.diagnostics),
        )


@dataclass(frozen=True, slots=True)
class _Lfm2Geometry:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    fir_length: int
    layer_types: tuple[str, ...]
    tied_token_interface: bool
    source_unique_parameters: int
    source_logical_parameters: int


def _attributes(value: Any) -> dict[str, JsonScalar]:
    return dict(getattr(value, "attributes", ()))


def _single_node(
    graph: ModelCircuitGraph,
    kind: CircuitKind | str,
    *,
    layer_index: int | None = None,
) -> Any:
    values = tuple(
        value
        for value in graph.nodes_of_kind(kind)
        if value.layer_index == layer_index
    )
    if len(values) != 1:
        normalized = kind.value if isinstance(kind, CircuitKind) else str(kind)
        raise DynamicParameterFrontierError(
            f"LFM2 graph requires exactly one {normalized!r} node"
            + (
                "."
                if layer_index is None
                else f" in layer {layer_index}."
            )
        )
    return values[0]


def _lfm2_counts(
    geometry: _Lfm2Geometry,
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: Sequence[str],
    vocab_size: int | None = None,
) -> tuple[int, int]:
    hidden = _positive_int(hidden_size, "hidden_size")
    intermediate = _positive_int(intermediate_size, "intermediate_size")
    query = _positive_int(query_heads, "query_heads")
    key_value = _positive_int(key_value_heads, "key_value_heads")
    dimension = _positive_int(head_dim, "head_dim")
    if query % key_value:
        raise DynamicParameterFrontierError(
            "LFM2 query heads must be divisible by key/value heads."
        )
    vocabulary = (
        geometry.vocab_size
        if vocab_size is None
        else _positive_int(vocab_size, "vocab_size")
    )
    token_matrix = vocabulary * hidden
    token_unique = token_matrix if geometry.tied_token_interface else 2 * token_matrix
    token_logical = 2 * token_matrix
    body = hidden  # final embedding RMSNorm
    for layer_type in layer_types:
        if layer_type == "conv":
            body += (
                4 * hidden * hidden
                + geometry.fir_length * hidden
                + 2 * hidden
            )
        elif layer_type == "full_attention":
            body += (
                2 * hidden * dimension * (query + key_value)
                + 2 * dimension
                + 2 * hidden
            )
        else:
            raise DynamicParameterFrontierError(
                "LFM2 layer types must be conv or full_attention."
            )
        body += 3 * hidden * intermediate
    return token_unique + body, token_logical + body


def _inspect_lfm2_geometry(graph: ModelCircuitGraph) -> _Lfm2Geometry:
    if graph.family != "lfm2":
        raise DynamicParameterFrontierError(
            "The LFM2 dense catalog requires an inspected LFM2 graph."
        )
    modalities = tuple(sorted({value.modality for value in graph.interfaces}))
    if modalities != ("text",):
        raise DynamicParameterFrontierError(
            "The LFM2 dense catalog supports exactly the text modality."
        )
    metadata = dict(graph.metadata)
    hidden = _positive_int(metadata.get("hidden_size"), "graph hidden_size")
    layer_indices = graph.layer_indices
    if layer_indices != tuple(range(len(layer_indices))):
        raise DynamicParameterFrontierError(
            "LFM2 source layers must be contiguous and zero-based."
        )
    embedding = _single_node(graph, CircuitKind.TOKEN_EMBEDDING)
    head = _single_node(graph, CircuitKind.OUTPUT_HEAD)
    final_norm = _single_node(graph, CircuitKind.NORM, layer_index=None)
    embedding_attributes = _attributes(embedding)
    head_attributes = _attributes(head)
    vocab = _positive_int(
        embedding_attributes.get("vocabulary_size"),
        "LFM2 vocabulary_size",
    )
    if (
        embedding_attributes.get("hidden_size") != hidden
        or head_attributes.get("vocabulary_size") != vocab
        or embedding.logical_parameter_count != vocab * hidden
        or head.logical_parameter_count != vocab * hidden
        or final_norm.logical_parameter_count != hidden
    ):
        raise DynamicParameterFrontierError(
            "LFM2 token interface/final norm does not match dense catalog shapes."
        )
    tied = bool(head_attributes.get("tied_to_embedding"))
    if len(embedding.parameters) != 1 or len(head.parameters) != 1:
        raise DynamicParameterFrontierError(
            "LFM2 dense catalog requires one embedding and one head matrix."
        )
    observed_tied = (
        embedding.parameters[0].storage_group
        == head.parameters[0].storage_group
    )
    if tied != observed_tied:
        raise DynamicParameterFrontierError(
            "LFM2 token-interface tie metadata and inspected storage disagree."
        )
    repeated_storage = {
        group
        for group, count in _storage_group_counts(graph).items()
        if count > 1
    }
    permitted_repeated = (
        {embedding.parameters[0].storage_group} if tied else set()
    )
    if repeated_storage != permitted_repeated:
        raise DynamicParameterFrontierError(
            "LFM2 dense catalog rejects non-native source parameter sharing."
        )

    layer_types: list[str] = []
    intermediate_sizes: set[int] = set()
    query_heads: set[int] = set()
    key_value_heads: set[int] = set()
    head_dims: set[int] = set()
    fir_lengths: set[int] = set()
    for index in layer_indices:
        mlp = _single_node(graph, CircuitKind.GATED_MLP, layer_index=index)
        intermediate = _positive_int(
            _attributes(mlp).get("intermediate_size"),
            "LFM2 intermediate_size",
        )
        intermediate_sizes.add(intermediate)
        if mlp.logical_parameter_count != 3 * hidden * intermediate:
            raise DynamicParameterFrontierError(
                "LFM2 MLP does not match the dense SwiGLU catalog formula."
            )
        attention = graph.nodes_of_kind(CircuitKind.ATTENTION, layer_index=index)
        convolution = graph.nodes_of_kind(CircuitKind.SHORT_CONV, layer_index=index)
        if bool(attention) == bool(convolution):
            raise DynamicParameterFrontierError(
                "Every LFM2 layer needs exactly one attention or short-conv "
                "operator."
            )
        if attention:
            operator = attention[0]
            attributes = _attributes(operator)
            query = _positive_int(
                attributes.get("query_heads"),
                "LFM2 query_heads",
            )
            key_value = _positive_int(
                attributes.get("key_value_heads"),
                "LFM2 key_value_heads",
            )
            dimension = _positive_int(
                attributes.get("head_dim"),
                "LFM2 head_dim",
            )
            if query % key_value:
                raise DynamicParameterFrontierError(
                    "Inspected LFM2 query heads are not divisible by KV heads."
                )
            query_heads.add(query)
            key_value_heads.add(key_value)
            head_dims.add(dimension)
            expected = (
                2 * hidden * dimension * (query + key_value)
                + 2 * dimension
            )
            if operator.logical_parameter_count != expected:
                raise DynamicParameterFrontierError(
                    "LFM2 attention operator does not match the dense catalog "
                    "formula."
                )
            layer_types.append("full_attention")
        else:
            operator = convolution[0]
            fir = _positive_int(
                _attributes(operator).get("fir_length"),
                "LFM2 fir_length",
            )
            fir_lengths.add(fir)
            expected = 4 * hidden * hidden + fir * hidden
            if operator.logical_parameter_count != expected:
                raise DynamicParameterFrontierError(
                    "LFM2 short-conv operator does not match the dense catalog "
                    "formula."
                )
            layer_types.append("conv")
    if (
        len(intermediate_sizes) != 1
        or len(query_heads) != 1
        or len(key_value_heads) != 1
        or len(head_dims) != 1
        or len(fir_lengths) != 1
    ):
        raise DynamicParameterFrontierError(
            "The current LFM2 dense catalog requires uniform FFN, attention, "
            "and short-conv geometry."
        )
    geometry = _Lfm2Geometry(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=next(iter(intermediate_sizes)),
        query_heads=next(iter(query_heads)),
        key_value_heads=next(iter(key_value_heads)),
        head_dim=next(iter(head_dims)),
        fir_length=next(iter(fir_lengths)),
        layer_types=tuple(layer_types),
        tied_token_interface=tied,
        source_unique_parameters=graph.unique_parameter_count,
        source_logical_parameters=graph.logical_parameter_count,
    )
    expected_unique, expected_logical = _lfm2_counts(
        geometry,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        query_heads=geometry.query_heads,
        key_value_heads=geometry.key_value_heads,
        head_dim=geometry.head_dim,
        layer_types=geometry.layer_types,
    )
    if (
        expected_unique != graph.unique_parameter_count
        or expected_logical != graph.logical_parameter_count
    ):
        raise DynamicParameterFrontierError(
            "LFM2 source graph does not reproduce under exact dense catalog "
            "parameter accounting."
        )
    return geometry


def _storage_group_counts(graph: ModelCircuitGraph) -> dict[str, int]:
    result: dict[str, int] = {}
    for node in graph.nodes:
        for parameter in node.parameters:
            result[parameter.storage_group] = (
                result.get(parameter.storage_group, 0) + 1
            )
    return result


def _balanced_spans(source_depth: int, target_depth: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            range(
                index * source_depth // target_depth,
                (index + 1) * source_depth // target_depth,
            )
        )
        for index in range(target_depth)
    )


def _lfm2_layouts(
    source_types: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], tuple[tuple[int, ...], ...]], ...]:
    """Enumerate one deterministic legal layout per attainable attention count."""

    source_depth = len(source_types)
    source_has_attention = "full_attention" in source_types
    layouts: list[
        tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]
    ] = []
    for target_depth in range(1, source_depth + 1):
        spans = _balanced_spans(source_depth, target_depth)
        available = tuple(
            frozenset(source_types[index] for index in span) for span in spans
        )
        forced_attention = tuple(
            index
            for index, values in enumerate(available)
            if values == frozenset({"full_attention"})
        )
        flexible = tuple(
            index
            for index, values in enumerate(available)
            if values == frozenset({"conv", "full_attention"})
        )
        minimum_attention = len(forced_attention)
        maximum_attention = minimum_attention + len(flexible)
        for attention_count in range(minimum_attention, maximum_attention + 1):
            if source_has_attention and attention_count == 0:
                continue
            chosen_flexible = set(
                flexible[: attention_count - minimum_attention]
            )
            types = tuple(
                "full_attention"
                if index in forced_attention or index in chosen_flexible
                else "conv"
                for index in range(target_depth)
            )
            layouts.append((types, spans))
    return tuple(layouts)


def _lfm2_head_frontier(
    geometry: _Lfm2Geometry,
    hidden_size: int,
    minimum_head_dim: int,
) -> tuple[tuple[int, int, int], ...]:
    """Return one best legal Q/KV pair for every distinct dense head cost."""

    dimensions = []
    value = 2
    while value < geometry.head_dim:
        dimensions.append(value)
        value *= 2
    dimensions.append(geometry.head_dim)
    best: dict[int, tuple[int, int, int, int, int, int, int]] = {}
    for head_dim in sorted(set(dimensions)):
        if (
            head_dim < minimum_head_dim
            or head_dim > hidden_size
            or head_dim % 2
        ):
            continue
        maximum_query = min(
            geometry.query_heads,
            hidden_size // head_dim,
        )
        for query_heads in range(1, maximum_query + 1):
            for key_value_heads in range(
                1,
                min(geometry.key_value_heads, query_heads) + 1,
            ):
                if query_heads % key_value_heads:
                    continue
                cost_key = head_dim * (
                    hidden_size * (query_heads + key_value_heads) + 1
                )
                ratio_error = abs(
                    query_heads * geometry.key_value_heads
                    - key_value_heads * geometry.query_heads
                )
                quality = (
                    ratio_error,
                    -query_heads * head_dim,
                    -key_value_heads * head_dim,
                    -head_dim,
                )
                current = best.get(cost_key)
                encoded = (
                    *quality,
                    query_heads,
                    key_value_heads,
                    head_dim,
                )
                if current is None or encoded[:4] < current[:4]:
                    best[cost_key] = encoded
    return tuple(
        (value[4], value[5], value[6])
        for _key, value in sorted(best.items())
    )


def _retention_score_ppm(
    geometry: _Lfm2Geometry,
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: Sequence[str],
) -> int:
    """Summarize topology geometry; never estimate behavioral retention."""

    source_attention = geometry.layer_types.count("full_attention")
    target_attention = tuple(layer_types).count("full_attention")
    terms = (
        len(tuple(layer_types)) * 1_000_000 // len(geometry.layer_types),
        hidden_size * 1_000_000 // geometry.hidden_size,
        intermediate_size * 1_000_000 // geometry.intermediate_size,
        query_heads * head_dim * 1_000_000
        // (geometry.query_heads * geometry.head_dim),
        key_value_heads * head_dim * 1_000_000
        // (geometry.key_value_heads * geometry.head_dim),
        target_attention * 1_000_000 // source_attention,
    )
    return min(1_000_000, sum(terms) // len(terms))


def _bounded_ratio_ppm(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise DynamicParameterFrontierError(
            "Pareto-retention denominator must be positive."
        )
    return min(1_000_000, max(0, numerator * 1_000_000 // denominator))


def _lfm2_required_compiler_capabilities(
    geometry: _Lfm2Geometry,
    *,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: Sequence[str],
    target_vocabulary_size: int,
) -> tuple[str, ...]:
    required = {CAP_DENSE_BLANK_MATERIALIZATION}
    if hidden_size != geometry.hidden_size:
        required.add(CAP_RESIDUAL_WIDTH_PROJECTION)
    if intermediate_size != geometry.intermediate_size:
        required.add(CAP_FFN_WIDTH_PROJECTION)
    if (
        query_heads != geometry.query_heads
        or key_value_heads != geometry.key_value_heads
        or head_dim != geometry.head_dim
    ):
        required.add(CAP_ATTENTION_TOPOLOGY_PROJECTION)
    target_types = tuple(layer_types)
    if len(target_types) != len(geometry.layer_types):
        required.add(CAP_ORDERED_DEPTH_COMPOSITION)
    if target_types != geometry.layer_types:
        required.add(CAP_HYBRID_OPERATOR_COMPOSITION)
    if target_vocabulary_size != geometry.vocab_size:
        required.add(CAP_TOKENIZER_BRIDGE)
    return tuple(sorted(required))


def _lfm2_pareto_profile(
    geometry: _Lfm2Geometry,
    *,
    requested_unique_parameters: int,
    unique_parameters: int,
    vocabulary_unique_parameters: int,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    layer_types: Sequence[str],
    compiler_executable: bool,
) -> tuple[tuple[str, int], ...]:
    target_types = tuple(layer_types)
    source_attention = geometry.layer_types.count("full_attention")
    target_attention = target_types.count("full_attention")
    source_gqa_numerator = geometry.query_heads
    source_gqa_denominator = geometry.key_value_heads
    target_gqa_numerator = query_heads
    target_gqa_denominator = key_value_heads
    gqa_cross_a = target_gqa_numerator * source_gqa_denominator
    gqa_cross_b = source_gqa_numerator * target_gqa_denominator
    budget_min = min(unique_parameters, requested_unique_parameters)
    budget_max = max(unique_parameters, requested_unique_parameters)
    vocabulary_pressure = _bounded_ratio_ppm(
        vocabulary_unique_parameters,
        requested_unique_parameters,
    )
    return _retention_profile(
        {
            "budget_utilization": _bounded_ratio_ppm(
                budget_min,
                budget_max,
            ),
            "residual_width_retention": _bounded_ratio_ppm(
                hidden_size,
                geometry.hidden_size,
            ),
            "depth_retention": _bounded_ratio_ppm(
                len(target_types),
                len(geometry.layer_types),
            ),
            "attention_role_retention": _bounded_ratio_ppm(
                target_attention,
                source_attention,
            ),
            "query_channel_retention": _bounded_ratio_ppm(
                query_heads * head_dim,
                geometry.query_heads * geometry.head_dim,
            ),
            "head_dim_retention": _bounded_ratio_ppm(
                head_dim,
                geometry.head_dim,
            ),
            "gqa_ratio_retention": _bounded_ratio_ppm(
                min(gqa_cross_a, gqa_cross_b),
                max(gqa_cross_a, gqa_cross_b),
            ),
            "ffn_width_retention": _bounded_ratio_ppm(
                intermediate_size,
                geometry.intermediate_size,
            ),
            "vocabulary_budget_headroom": 1_000_000 - vocabulary_pressure,
            "compiler_executable": 1_000_000 if compiler_executable else 0,
        }
    )


# Every scalarization gives every objective a strictly positive weight.  A
# winner is therefore guaranteed not to be dominated in the enumerated legal
# search space.  The profiles expose supported trade-offs without retaining
# millions of arbitrary near-exact configurations in memory.
_PARETO_SCALARIZATIONS: tuple[
    tuple[str, tuple[tuple[str, int], ...]], ...
] = (
    (
        "balanced",
        (
            ("budget_utilization", 8),
            ("residual_width_retention", 3),
            ("depth_retention", 3),
            ("attention_role_retention", 2),
            ("query_channel_retention", 2),
            ("head_dim_retention", 2),
            ("gqa_ratio_retention", 2),
            ("ffn_width_retention", 2),
            ("vocabulary_budget_headroom", 1),
            ("compiler_executable", 4),
        ),
    ),
    *tuple(
        (
            f"{axis}_first",
            tuple(
                (
                    candidate_axis,
                    (
                        10_000_000
                        if (
                            candidate_axis == axis
                            and axis == "budget_utilization"
                        )
                        else 32
                        if candidate_axis == axis
                        else 1
                    ),
                )
                for candidate_axis in _PARETO_AXES
            ),
        )
        for axis in _PARETO_AXES
    ),
)


def _pareto_scalarization_rank(
    candidate: DenseTargetTopologyCandidate,
    *,
    profile_weights: Sequence[tuple[str, int]],
    match_policy: str,
) -> tuple[Any, ...]:
    retention = candidate.retention_profile
    score = sum(
        weight * retention[axis] for axis, weight in profile_weights
    )
    return (
        -score,
        _candidate_rank(candidate, match_policy),
    )


def _candidate_dominates(
    left: DenseTargetTopologyCandidate,
    right: DenseTargetTopologyCandidate,
) -> bool:
    left_profile = left.retention_profile
    right_profile = right.retention_profile
    return all(
        left_profile[axis] >= right_profile[axis] for axis in _PARETO_AXES
    ) and any(
        left_profile[axis] > right_profile[axis] for axis in _PARETO_AXES
    )


def _candidate_rank(
    candidate: DenseTargetTopologyCandidate,
    policy: str,
) -> tuple[Any, ...]:
    if policy == "exact":
        distance = 0
        side = 0
    elif policy == "at_or_below":
        distance = -candidate.budget_delta
        side = 0
    else:
        distance = abs(candidate.budget_delta)
        side = 0 if candidate.at_or_below_budget else 1
    configuration = tuple(
        (key, str(value)) for key, value in candidate.configuration
    )
    return (
        distance,
        side,
        -candidate.retention_score_ppm,
        -candidate.unique_parameters,
        configuration,
    )


@dataclass(frozen=True, slots=True)
class Lfm2DenseTargetTopologyCatalog:
    """Pure-IR dense topology catalog for LFM2 hybrid text decoders."""

    catalog_id: str = LFM2_CATALOG_ID
    version: int = 1
    priority: int = 100
    target_families: tuple[str, ...] = ("lfm2",)
    target_modalities: tuple[str, ...] = ("text",)
    minimum_hidden_size: int = 32
    hidden_size_multiple: int = 8
    minimum_head_dim: int = 8
    compiler_capabilities: tuple[str, ...] = (
        LFM2_CURRENT_COMPILER_CAPABILITIES
    )
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None

    def __post_init__(self) -> None:
        _positive_int(self.minimum_hidden_size, "minimum_hidden_size")
        _positive_int(self.hidden_size_multiple, "hidden_size_multiple")
        _positive_int(self.minimum_head_dim, "minimum_head_dim")
        if self.minimum_hidden_size % self.hidden_size_multiple:
            raise DynamicParameterFrontierError(
                "minimum_hidden_size must be a hidden_size_multiple."
            )
        capabilities = _capabilities(
            self.compiler_capabilities,
            "LFM2 compiler capability",
        )
        bridge = self.tokenizer_bridge_artifact
        if bridge is None:
            if CAP_TOKENIZER_BRIDGE in capabilities:
                raise DynamicParameterFrontierError(
                    "CAP_TOKENIZER_BRIDGE requires an explicit verified strict "
                    "tokenizer bridge artifact."
                )
        else:
            if not isinstance(bridge, StrictTokenizerBridgeArtifact):
                raise DynamicParameterFrontierError(
                    "tokenizer_bridge_artifact must be a "
                    "StrictTokenizerBridgeArtifact."
                )
            try:
                bridge.verify_files()
            except ValueError as exc:
                raise DynamicParameterFrontierError(
                    "tokenizer_bridge_artifact failed verification."
                ) from exc
            capabilities = tuple(
                sorted({*capabilities, CAP_TOKENIZER_BRIDGE})
            )
        object.__setattr__(self, "compiler_capabilities", capabilities)

    def supports(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        target_family: str,
        target_modalities: tuple[str, ...],
    ) -> bool:
        return (
            isinstance(parent_graph, ModelCircuitGraph)
            and parent_graph.family == "lfm2"
            and target_family == "lfm2"
            and target_modalities == ("text",)
        )

    def enumerate_frontier(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        requested_unique_parameters: int,
        target_vocabulary_size: int | None,
        match_policy: str,
        maximum_candidates: int,
    ) -> CatalogFrontier:
        geometry = _inspect_lfm2_geometry(parent_graph)
        requested = _positive_int(
            requested_unique_parameters,
            "requested_unique_parameters",
        )
        maximum = _positive_int(maximum_candidates, "maximum_candidates")
        target_vocab = (
            geometry.vocab_size
            if target_vocabulary_size is None
            else _positive_int(
                target_vocabulary_size,
                "target_vocabulary_size",
            )
        )
        tokenizer_bridge_configuration: dict[str, str] = {}
        if target_vocab != geometry.vocab_size:
            bridge = self.tokenizer_bridge_artifact
            if bridge is not None:
                try:
                    bridge.verify_files()
                except ValueError as exc:
                    raise DynamicParameterFrontierError(
                        "Tokenizer bridge artifact changed after catalog "
                        "registration."
                    ) from exc
                profile = bridge.transport_profile
                if (
                    parent_graph.tokenizer_fingerprint
                    != bridge.donor_tokenizer_semantics_sha256
                    or profile.donor_row_capacity != geometry.vocab_size
                    or profile.target_row_capacity != target_vocab
                ):
                    raise DynamicParameterFrontierError(
                        "Tokenizer bridge artifact does not match the inspected "
                        "parent semantics or requested vocabulary capacities."
                    )
                tokenizer_bridge_configuration = (
                    strict_tokenizer_bridge_configuration(bridge)
                )
        if match_policy not in _POLICIES:
            raise DynamicParameterFrontierError("Unknown budget match policy.")
        layouts = _lfm2_layouts(geometry.layer_types)
        minimum_hidden = min(self.minimum_hidden_size, geometry.hidden_size)
        hidden_sizes = tuple(
            range(
                minimum_hidden,
                geometry.hidden_size + 1,
                self.hidden_size_multiple,
            )
        )
        if hidden_sizes[-1] != geometry.hidden_size:
            hidden_sizes = (*hidden_sizes, geometry.hidden_size)

        profile_winners: dict[
            str,
            tuple[tuple[Any, ...], DenseTargetTopologyCandidate],
        ] = {}
        considered = 0
        nearest_below: int | None = None
        nearest_above: int | None = None
        minimum_representable: int | None = None
        maximum_representable: int | None = None
        exact_representable_seen = False
        for hidden in hidden_sizes:
            heads = _lfm2_head_frontier(
                geometry,
                hidden,
                self.minimum_head_dim,
            )
            for layer_types, spans in layouts:
                depth = len(layer_types)
                for query_heads, key_value_heads, head_dim in heads:
                    token_unique = (
                        target_vocab * hidden
                        if geometry.tied_token_interface
                        else 2 * target_vocab * hidden
                    )
                    fixed = token_unique + hidden
                    for layer_type in layer_types:
                        if layer_type == "conv":
                            fixed += (
                                4 * hidden * hidden
                                + geometry.fir_length * hidden
                                + 2 * hidden
                            )
                        else:
                            fixed += (
                                2
                                * hidden
                                * head_dim
                                * (query_heads + key_value_heads)
                                + 2 * head_dim
                                + 2 * hidden
                            )
                    per_intermediate = 3 * hidden * depth
                    minimum_intermediate = min(
                        geometry.intermediate_size,
                        hidden,
                    )
                    floor_width = (requested - fixed) // per_intermediate
                    widths = {
                        minimum_intermediate,
                        geometry.intermediate_size,
                        max(
                            minimum_intermediate,
                            min(geometry.intermediate_size, floor_width),
                        ),
                        max(
                            minimum_intermediate,
                            min(geometry.intermediate_size, floor_width + 1),
                        ),
                    }
                    for intermediate in sorted(widths):
                        unique, logical = _lfm2_counts(
                            geometry,
                            hidden_size=hidden,
                            intermediate_size=intermediate,
                            query_heads=query_heads,
                            key_value_heads=key_value_heads,
                            head_dim=head_dim,
                            layer_types=layer_types,
                            vocab_size=target_vocab,
                        )
                        if unique >= geometry.source_unique_parameters:
                            continue
                        considered += 1
                        minimum_representable = (
                            unique
                            if minimum_representable is None
                            else min(minimum_representable, unique)
                        )
                        maximum_representable = (
                            unique
                            if maximum_representable is None
                            else max(maximum_representable, unique)
                        )
                        if unique <= requested and (
                            nearest_below is None or unique > nearest_below
                        ):
                            nearest_below = unique
                        if unique >= requested and (
                            nearest_above is None or unique < nearest_above
                        ):
                            nearest_above = unique
                        if unique == requested:
                            exact_representable_seen = True
                        eligible = (
                            unique == requested
                            if match_policy == "exact"
                            else (
                                unique <= requested
                                if match_policy == "at_or_below"
                                else True
                            )
                        )
                        if not eligible:
                            continue
                        retention_score = _retention_score_ppm(
                            geometry,
                            hidden_size=hidden,
                            intermediate_size=intermediate,
                            query_heads=query_heads,
                            key_value_heads=key_value_heads,
                            head_dim=head_dim,
                            layer_types=layer_types,
                        )
                        if match_policy == "exact":
                            distance = 0
                            side = 0
                        elif match_policy == "at_or_below":
                            distance = requested - unique
                            side = 0
                        else:
                            distance = abs(unique - requested)
                            side = 0 if unique <= requested else 1
                        rough_rank = (
                            distance,
                            side,
                            -retention_score,
                            -unique,
                            -hidden,
                            -depth,
                            -intermediate,
                            -query_heads,
                            -key_value_heads,
                            -head_dim,
                            layer_types,
                            spans,
                        )
                        tied_count = (
                            target_vocab * hidden
                            if geometry.tied_token_interface
                            else 0
                        )
                        vocabulary_unique = (
                            target_vocab
                            * hidden
                            * (
                                1
                                if geometry.tied_token_interface
                                else 2
                            )
                        )
                        required_capabilities = (
                            _lfm2_required_compiler_capabilities(
                                geometry,
                                hidden_size=hidden,
                                intermediate_size=intermediate,
                                query_heads=query_heads,
                                key_value_heads=key_value_heads,
                                head_dim=head_dim,
                                layer_types=layer_types,
                                target_vocabulary_size=target_vocab,
                            )
                        )
                        missing_capabilities = tuple(
                            sorted(
                                set(required_capabilities).difference(
                                    self.compiler_capabilities
                                )
                            )
                        )
                        compiler_executable = not missing_capabilities
                        pareto_profile = _lfm2_pareto_profile(
                            geometry,
                            requested_unique_parameters=requested,
                            unique_parameters=unique,
                            vocabulary_unique_parameters=vocabulary_unique,
                            hidden_size=hidden,
                            intermediate_size=intermediate,
                            query_heads=query_heads,
                            key_value_heads=key_value_heads,
                            head_dim=head_dim,
                            layer_types=layer_types,
                            compiler_executable=compiler_executable,
                        )
                        profile_values = dict(pareto_profile)
                        winning_profiles: list[
                            tuple[str, tuple[Any, ...]]
                        ] = []
                        for profile_name, weights in _PARETO_SCALARIZATIONS:
                            score = sum(
                                weight * profile_values[axis]
                                for axis, weight in weights
                            )
                            supported_rank = (-score, rough_rank)
                            previous = profile_winners.get(profile_name)
                            if (
                                previous is None
                                or supported_rank < previous[0]
                            ):
                                winning_profiles.append(
                                    (profile_name, supported_rank)
                                )
                        if not winning_profiles:
                            continue
                        span_string = ";".join(
                            ",".join(str(index) for index in span)
                            for span in spans
                        )
                        layer_string = ",".join(layer_types)
                        configuration = _configuration(
                            {
                                "model_type": "lfm2",
                                "vocab_size": target_vocab,
                                "hidden_size": hidden,
                                "intermediate_size": intermediate,
                                "num_hidden_layers": depth,
                                "num_attention_heads": query_heads,
                                "num_key_value_heads": key_value_heads,
                                "head_dim": head_dim,
                                "layer_types": layer_string,
                                "conv_L_cache": geometry.fir_length,
                                "block_auto_adjust_ff_dim": False,
                                "tie_word_embeddings": (
                                    geometry.tied_token_interface
                                ),
                                "_neural_forge_ordered_source_spans": (
                                    span_string
                                ),
                                "_neural_forge_target_layer_types": layer_string,
                                "_neural_forge_dense_topology_catalog": (
                                    self.catalog_id
                                ),
                                "_neural_forge_tokenizer_bridge_required": (
                                    target_vocab != geometry.vocab_size
                                ),
                                **tokenizer_bridge_configuration,
                            }
                        )
                        identity = canonical_sha256(
                            {
                                "catalog_id": self.catalog_id,
                                "configuration": dict(configuration),
                                "unique_parameters": unique,
                                "logical_parameters": logical,
                            }
                        )
                        candidate = DenseTargetTopologyCandidate(
                            candidate_id=f"lfm2-dense-{identity[:16]}",
                            catalog_id=self.catalog_id,
                            family="lfm2",
                            modalities=("text",),
                            requested_unique_parameters=requested,
                            source_unique_parameters=(
                                geometry.source_unique_parameters
                            ),
                            unique_parameters=unique,
                            logical_parameters=logical,
                            native_tied_parameter_count=tied_count,
                            retention_score_ppm=retention_score,
                            configuration=configuration,
                            ordered_source_spans=spans,
                            native_ties=(
                                (
                                    "input.token_embedding.weight",
                                    "output.lm_head.weight",
                                ),
                            )
                            if geometry.tied_token_interface
                            else (),
                            source_vocabulary_size=geometry.vocab_size,
                            target_vocabulary_size=target_vocab,
                            vocabulary_unique_parameters=(
                                target_vocab
                                * hidden
                                * (
                                    1
                                    if geometry.tied_token_interface
                                    else 2
                                )
                            ),
                            tokenizer_bridge_required=(
                                target_vocab != geometry.vocab_size
                            ),
                            retention_profile_ppm=pareto_profile,
                            required_compiler_capabilities=(
                                required_capabilities
                            ),
                            declared_compiler_capabilities=(
                                self.compiler_capabilities
                            ),
                            missing_compiler_capabilities=(
                                missing_capabilities
                            ),
                            compiler_executable=compiler_executable,
                        )
                        for profile_name, supported_rank in winning_profiles:
                            profile_winners[profile_name] = (
                                supported_rank,
                                candidate,
                            )

        support_by_candidate: dict[str, set[str]] = {}
        candidate_by_id: dict[str, DenseTargetTopologyCandidate] = {}
        for profile_name, (_rank, candidate) in profile_winners.items():
            candidate_by_id[candidate.candidate_id] = candidate
            support_by_candidate.setdefault(candidate.candidate_id, set()).add(
                profile_name
            )
        supported: list[DenseTargetTopologyCandidate] = []
        for candidate_id, candidate in candidate_by_id.items():
            supported.append(
                replace(
                    candidate,
                    pareto_support_profiles=tuple(
                        sorted(support_by_candidate[candidate_id])
                    ),
                    pareto_dominance_verified=True,
                )
            )
        for index, candidate in enumerate(supported):
            if any(
                _candidate_dominates(other, candidate)
                for other_index, other in enumerate(supported)
                if other_index != index
            ):
                raise DynamicParameterFrontierError(
                    "A supported scalarization winner was Pareto dominated; "
                    "the catalog objective audit is inconsistent."
                )

        # Preserve capability selection and then expose distinct objective
        # trade-offs.  This is deliberately not an exact-count-only sort.
        profile_order = (
            *(
                (
                    "budget_utilization_first",
                    "compiler_executable_first",
                )
                if match_policy == "nearest"
                else (
                    "compiler_executable_first",
                    "budget_utilization_first",
                )
            ),
            "balanced",
            "residual_width_retention_first",
            "depth_retention_first",
            "attention_role_retention_first",
            "query_channel_retention_first",
            "head_dim_retention_first",
            "gqa_ratio_retention_first",
            "ffn_width_retention_first",
            "vocabulary_budget_headroom_first",
        )
        supported_by_id = {
            candidate.candidate_id: candidate for candidate in supported
        }
        deduplicated: list[DenseTargetTopologyCandidate] = []
        seen_ids: set[str] = set()
        for profile_name in profile_order:
            winner = profile_winners.get(profile_name)
            if winner is None:
                continue
            candidate_id = winner[1].candidate_id
            if candidate_id in seen_ids:
                continue
            seen_ids.add(candidate_id)
            deduplicated.append(supported_by_id[candidate_id])
            if len(deduplicated) == maximum:
                break
        if len(deduplicated) < maximum:
            for candidate in sorted(
                supported,
                key=lambda value: _candidate_rank(value, match_policy),
            ):
                if candidate.candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate.candidate_id)
                deduplicated.append(candidate)
                if len(deduplicated) == maximum:
                    break
        dense_token_floor = (
            target_vocab
            * minimum_hidden
            * (1 if geometry.tied_token_interface else 2)
        )
        token_floor_share = dense_token_floor / requested
        if dense_token_floor > requested:
            vocabulary_pressure = "exceeds_budget"
        elif token_floor_share >= 0.85:
            vocabulary_pressure = "critical"
        elif token_floor_share >= 0.5:
            vocabulary_pressure = "dominant"
        else:
            vocabulary_pressure = "bounded"
        diagnostics = {
            "catalog_id": self.catalog_id,
            "catalog_version": self.version,
            "source_family": parent_graph.family,
            "source_graph_fingerprint": parent_graph.fingerprint,
            "source_unique_parameters": geometry.source_unique_parameters,
            "source_logical_parameters": geometry.source_logical_parameters,
            "requested_unique_parameters": requested,
            "source_vocabulary_size": geometry.vocab_size,
            "target_vocabulary_size": target_vocab,
            "dense_token_interface_parameter_floor": dense_token_floor,
            "dense_token_interface_floor_budget_share": token_floor_share,
            "vocabulary_budget_pressure": vocabulary_pressure,
            "tokenizer_bridge_required": (
                target_vocab != geometry.vocab_size
            ),
            "tokenizer_bridge_required_for_requested_budget": (
                dense_token_floor > requested
            ),
            "tokenizer_mapping_implemented": bool(
                tokenizer_bridge_configuration
            ),
            "tokenizer_bridge_artifact_sha256": (
                None
                if not tokenizer_bridge_configuration
                else tokenizer_bridge_configuration[
                    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
                ]
            ),
            "match_policy": match_policy,
            "topologies_considered": considered,
            "minimum_representable_unique_parameters": minimum_representable,
            "maximum_representable_unique_parameters": maximum_representable,
            "nearest_representable_below": nearest_below,
            "nearest_representable_above": nearest_above,
            "exact_budget_representable": exact_representable_seen,
            "returned_candidates": len(deduplicated),
            "supported_pareto_candidates_before_limit": len(supported),
            "pareto_scalarization_profiles": [
                name for name, _weights in _PARETO_SCALARIZATIONS
            ],
            "pareto_objective_axes": list(_PARETO_AXES),
            "pareto_method": (
                "exact_supported_points_from_strictly_positive_"
                "scalarizations_over_all_eligible_enumerated_topologies"
            ),
            "exact_full_pareto_surface_enumerated": False,
            "returned_candidates_pairwise_nondominated": True,
            "declared_compiler_capabilities": list(
                self.compiler_capabilities
            ),
            "compiler_capability_selection_fail_closed": True,
            "executable_candidate_count": sum(
                candidate.compiler_executable for candidate in deduplicated
            ),
            "source_geometry": {
                "vocab_size": geometry.vocab_size,
                "hidden_size": geometry.hidden_size,
                "intermediate_size": geometry.intermediate_size,
                "query_heads": geometry.query_heads,
                "key_value_heads": geometry.key_value_heads,
                "head_dim": geometry.head_dim,
                "fir_length": geometry.fir_length,
                "layer_types": list(geometry.layer_types),
                "tied_token_interface": geometry.tied_token_interface,
            },
            "searched_axes": [
                "hidden_size",
                "ordered_depth_and_operator_mix",
                "query_heads",
                "key_value_heads",
                "head_dim",
                "intermediate_size",
            ],
            "selection_is_not_exact_count_only": True,
            "dense_topology_only": True,
            "minimum_hidden_size": minimum_hidden,
            "minimum_head_dim": self.minimum_head_dim,
            "minimum_ffn_expansion_ratio": 1.0,
            "intelligence_retention_proven": False,
            "structural_profile_is_behavioral_capability_evidence": False,
            "behavioral_capability_retention_claimed": False,
            "selection_evidence_scope": (
                "parameter_accounting_topology_geometry_and_declared_"
                "compiler_transform_availability_only"
            ),
            "weights_or_downloads_used": False,
            "quantization_pruning_packing_factorization_used": False,
            "non_native_weight_sharing_used": False,
        }
        return CatalogFrontier(
            candidates=tuple(deduplicated),
            diagnostics=diagnostics,
        )


def default_dense_topology_catalog_registry(
) -> DenseTargetTopologyCatalogRegistry:
    return DenseTargetTopologyCatalogRegistry((Lfm2DenseTargetTopologyCatalog(),))


def synthesize_dense_parameter_frontier(
    parent_graph: ModelCircuitGraph,
    request: DenseParameterBudgetRequest,
    *,
    catalogs: DenseTargetTopologyCatalogRegistry | None = None,
) -> DenseParameterFrontierResult:
    """Convenience entrypoint for one budget or compression-ratio request."""

    return DenseParameterFrontierSynthesizer(
        catalogs or default_dense_topology_catalog_registry()
    ).synthesize(parent_graph, request)


def synthesize_dense_compression_ratio_frontier(
    parent_graph: ModelCircuitGraph,
    *,
    target_family: str,
    target_modalities: Sequence[str],
    compression_ratios: Sequence[str | int | float] = (2, 4, 10, 100),
    match_policy: str = "at_or_below",
    maximum_candidates_per_ratio: int = 4,
    catalogs: DenseTargetTopologyCatalogRegistry | None = None,
) -> tuple[DenseParameterFrontierResult, ...]:
    """Produce deterministic ranked targets for several orders of compaction."""

    ratios = tuple(compression_ratios)
    if not ratios:
        raise DynamicParameterFrontierError(
            "compression_ratios must contain at least one ratio."
        )
    synthesizer = DenseParameterFrontierSynthesizer(
        catalogs or default_dense_topology_catalog_registry()
    )
    return tuple(
        synthesizer.synthesize(
            parent_graph,
            DenseParameterBudgetRequest(
                target_family=target_family,
                target_modalities=tuple(target_modalities),
                compression_ratio=ratio,
                match_policy=match_policy,
                maximum_candidates=maximum_candidates_per_ratio,
            ),
        )
        for ratio in ratios
    )


__all__ = [
    "CAP_ATTENTION_TOPOLOGY_PROJECTION",
    "CAP_DENSE_BLANK_MATERIALIZATION",
    "CAP_FFN_WIDTH_PROJECTION",
    "CAP_HYBRID_OPERATOR_COMPOSITION",
    "CAP_ORDERED_DEPTH_COMPOSITION",
    "CAP_RESIDUAL_WIDTH_PROJECTION",
    "CAP_TOKENIZER_BRIDGE",
    "CatalogFrontier",
    "DenseExecutableCandidateSelection",
    "DenseParameterBudgetRequest",
    "DenseParameterFrontierResult",
    "DenseParameterFrontierSynthesizer",
    "DenseTargetTopologyCandidate",
    "DenseTargetTopologyCatalog",
    "DenseTargetTopologyCatalogRegistry",
    "DynamicParameterFrontierError",
    "DynamicParameterFrontierCapabilityError",
    "DynamicParameterFrontierUnattainable",
    "LFM2_CATALOG_ID",
    "LFM2_CURRENT_COMPILER_CAPABILITIES",
    "Lfm2DenseTargetTopologyCatalog",
    "SCHEMA",
    "default_dense_topology_catalog_registry",
    "select_best_executable_dense_candidate",
    "synthesize_dense_compression_ratio_frontier",
    "synthesize_dense_parameter_frontier",
]
