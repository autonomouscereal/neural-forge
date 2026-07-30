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

The production catalog covers LFM2 hybrid text decoders, and the first
cross-family seam catalogs dense, untied Hugging Face Llama causal decoders.
New families or modalities extend the registry with another catalog; they do
not add manual layer surgery to this module.

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
from .dynamic_fold_budget import (
    ArchitectureDiscoveredBudgetProblem,
    ArchitectureDiscoveredBudgetSelection,
    COMPONENT_RETENTION_RISK_POLICY,
    ComponentProtectionRule,
    DiscoveredComponentChoiceGroup,
    DiscoveredComponentTopologyOption,
    RetainedParameterAllocation,
    SEMANTIC_ATTENTION_QUERY_KEY,
    SEMANTIC_ATTENTION_WRITER,
    SEMANTIC_INTERFACE,
    SEMANTIC_MLP_PRODUCER,
    SEMANTIC_MLP_WRITER,
    SEMANTIC_NORMALIZATION,
    SEMANTIC_OTHER,
    SEMANTIC_STATE_MIXER_INTERNAL,
    _effective_component_protection_rules,
    _normalized_component_retention_risk,
    classify_component_parameter_semantic,
    residual_attention_component_protection_rules,
    solve_architecture_discovered_budget,
)
from .hf_attention_fold import (
    AttentionFoldError,
    validate_gqa_attention_fold_geometry,
)
from .hf_lfm2_dynamic_physical_layout import (
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    restore_lfm2_heterogeneous_physical_layout,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    Lfm2AuthenticatedLayerPhysicalOption,
    Lfm2HeterogeneousLayoutSelection,
    Lfm2HeterogeneousLayoutSelectionError,
    Lfm2HeterogeneousLayoutSelectionReceipt,
    Lfm2HeterogeneousLayoutUnattainable,
    Lfm2HeterogeneousTargetPins,
    build_lfm2_heterogeneous_target_pins,
    select_lfm2_heterogeneous_physical_layout,
)
from .hf_tokenizer_bridge import (
    StrictTokenizerBridgeArtifact,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
    strict_tokenizer_bridge_configuration,
)


SCHEMA = "neural-forge-dense-parameter-frontier-v1"
LFM2_CATALOG_ID = "neural-forge.catalog.lfm2-dense-topology-v1"
HF_LLAMA_CATALOG_ID = (
    "neural-forge.catalog.hf-llama-dense-causal-v1"
)
HF_LLAMA_TARGET_SCHEMA = "neural-forge-hf-llama-dense-target-v1"
HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY = (
    "_neural_forge_hf_llama_target_schema"
)
HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY = (
    "_neural_forge_parameter_alias_policy"
)
HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE = "none"
HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY = (
    "_neural_forge_ordered_source_spans"
)
HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY = "_neural_forge_source_family"
HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY = (
    "_neural_forge_source_layer_count"
)
DENSE_TOPOLOGY_CATALOG_CONFIG_KEY = (
    "_neural_forge_dense_topology_catalog"
)
LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY = (
    "_neural_forge_dense_target_token_interface_mode"
)
LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE = "preserve"
LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED = "untied"
_LFM2_DENSE_TOKEN_INTERFACE_MODES = frozenset(
    {
        LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE,
        LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED,
    }
)
LFM2_COMPONENT_POLICY_PROFILE = (
    "component_protection_policy_full_eligible_stream"
)
_POLICIES = frozenset({"exact", "at_or_below", "nearest"})

CAP_DENSE_BLANK_MATERIALIZATION = "dense_blank_materialization"
CAP_RESIDUAL_WIDTH_PROJECTION = "residual_width_projection"
CAP_FFN_WIDTH_PROJECTION = "ffn_width_projection"
CAP_ATTENTION_TOPOLOGY_PROJECTION = "attention_topology_projection"
CAP_ORDERED_DEPTH_COMPOSITION = "ordered_depth_composition"
CAP_HYBRID_OPERATOR_COMPOSITION = "hybrid_operator_composition"
CAP_TOKENIZER_BRIDGE = "tokenizer_bridge"
CAP_HETEROGENEOUS_LAYER_LAYOUT = "heterogeneous_layer_layout"
CAP_CAUSAL_DECODER_FAMILY_TRANSLATION = (
    "causal_decoder_family_translation"
)

LFM2_HETEROGENEOUS_LAYOUT_KIND = "lfm2_heterogeneous_dense_fp32_v1"
LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY = (
    "_neural_forge_lfm2_heterogeneous_layout_descriptor_json"
)
LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY = (
    "_neural_forge_lfm2_heterogeneous_layout_descriptor_sha256"
)
LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY = (
    "_neural_forge_lfm2_heterogeneous_layout_selection_receipt_json"
)
LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY = (
    "_neural_forge_lfm2_heterogeneous_layout_selection_receipt_sha256"
)
_LFM2_HETEROGENEOUS_LAYOUT_CONFIGURATION_KEYS = frozenset(
    {
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY,
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY,
    }
)

COMPONENT_ALLOCATION_POLICY = (
    "architecture_discovered_whole_graph_global_topology_v2_semantic_balance"
)
COMPONENT_ALLOCATION_POLICY_CONFIG_KEY = (
    "_neural_forge_component_allocation_policy"
)
COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY = (
    "_neural_forge_component_allocation_receipt_sha256"
)
COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY = (
    "_neural_forge_component_allocation_topology_sha256"
)
_COMPONENT_ALLOCATION_CONFIGURATION_KEYS = frozenset(
    {
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    }
)

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


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicParameterFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
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


def component_allocation_core_configuration(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
) -> tuple[tuple[str, JsonScalar], ...]:
    """Remove only compiler-owned allocation seals from a target config."""

    configuration = _configuration(values)
    return tuple(
        (key, value)
        for key, value in configuration
        if key not in _COMPONENT_ALLOCATION_CONFIGURATION_KEYS
    )


def component_allocation_topology_fingerprint(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
) -> str:
    """Fingerprint the materializable topology without its receipt seals."""

    return canonical_sha256(
        dict(component_allocation_core_configuration(values))
    )


def _restore_lfm2_heterogeneous_candidate_artifacts(
    values: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]],
) -> tuple[
    Lfm2HeterogeneousPhysicalLayoutDescriptor,
    Lfm2HeterogeneousLayoutSelectionReceipt,
] | None:
    configuration = dict(_configuration(values))
    present = _LFM2_HETEROGENEOUS_LAYOUT_CONFIGURATION_KEYS.intersection(
        configuration
    )
    if not present:
        return None
    if present != _LFM2_HETEROGENEOUS_LAYOUT_CONFIGURATION_KEYS:
        raise DynamicParameterFrontierError(
            "LFM2 heterogeneous layout configuration markers are incomplete."
        )
    descriptor_json = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY
    ]
    descriptor_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY
    ]
    selection_json = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY
    ]
    selection_sha256 = configuration[
        LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY
    ]
    if not all(
        isinstance(value, str)
        for value in (
            descriptor_json,
            descriptor_sha256,
            selection_json,
            selection_sha256,
        )
    ):
        raise DynamicParameterFrontierError(
            "LFM2 heterogeneous layout markers must be ASCII JSON/digest "
            "strings."
        )
    try:
        descriptor = restore_lfm2_heterogeneous_physical_layout(
            descriptor_json,
            expected_descriptor_sha256=_sha256_digest(
                descriptor_sha256,
                "heterogeneous descriptor SHA-256",
            ),
        )
        receipt = Lfm2HeterogeneousLayoutSelectionReceipt.restore(
            selection_json
        )
        decision = receipt.allocation_decision
        restored_pins = build_lfm2_heterogeneous_target_pins(
            hidden_size=descriptor.hidden_size,
            vocab_size=descriptor.vocab_size,
            conv_kernel_size=descriptor.conv_kernel_size,
            rope_theta=descriptor.rope_theta,
            norm_eps=descriptor.norm_eps,
            max_position_embeddings=descriptor.max_position_embeddings,
            use_cache=descriptor.use_cache,
            attention_implementation=descriptor.attention_implementation,
            bos_token_id=descriptor.bos_token_id,
            eos_token_id=descriptor.eos_token_id,
            pad_token_id=descriptor.pad_token_id,
            block_auto_adjust_ff_dim=(
                descriptor.block_auto_adjust_ff_dim
            ),
            target_vocab_evidence_sha256=(
                decision.target_vocab_evidence_sha256
            ),
            runtime_evidence_sha256=decision.runtime_evidence_sha256,
            token_bridge_evidence_sha256=(
                decision.token_bridge_evidence_sha256
            ),
        )
    except (ValueError, TypeError) as exc:
        raise DynamicParameterFrontierError(
            "LFM2 heterogeneous candidate artifacts failed authentication."
        ) from exc
    if (
        receipt.receipt_sha256
        != _sha256_digest(
            selection_sha256,
            "heterogeneous selection receipt SHA-256",
        )
        or receipt.descriptor_sha256 != descriptor.descriptor_sha256
        or descriptor.source_topology_sha256
        != decision.source_graph_fingerprint
        or descriptor.allocation_evidence_sha256
        != decision.allocation_decision_sha256
        or descriptor.accounting.unique_parameter_elements
        != decision.exact_target_unique_parameters
        or descriptor.token_bridge_evidence_sha256
        != decision.token_bridge_evidence_sha256
        or tuple(
            layer.selection_evidence_sha256 for layer in descriptor.layers
        )
        != decision.selected_layer_option_sha256
        or restored_pins.pins_sha256 != decision.target_pins_sha256
    ):
        raise DynamicParameterFrontierError(
            "LFM2 heterogeneous descriptor/selection/source/count seals differ."
        )
    return descriptor, receipt


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
class Lfm2HeterogeneousLayoutSelectionInput:
    """Authenticated physical alternatives supplied to the existing catalog.

    This is deliberately an input seam, not another topology catalog.  Every
    option and every target runtime/vocabulary pin is restored and seal-checked
    before the existing architecture-discovered selector sees a request.
    """

    target_pins: Lfm2HeterogeneousTargetPins
    layer_options: tuple[Lfm2AuthenticatedLayerPhysicalOption, ...]
    max_states: int = 100_000

    def __post_init__(self) -> None:
        try:
            pins = Lfm2HeterogeneousTargetPins.restore(self.target_pins)
            options = tuple(
                Lfm2AuthenticatedLayerPhysicalOption.restore(value)
                for value in self.layer_options
            )
        except Lfm2HeterogeneousLayoutSelectionError as exc:
            raise DynamicParameterFrontierError(
                "Heterogeneous LFM2 selection input failed authentication."
            ) from exc
        if not options:
            raise DynamicParameterFrontierError(
                "Heterogeneous LFM2 selection input needs physical options."
            )
        object.__setattr__(self, "target_pins", pins)
        object.__setattr__(self, "layer_options", options)
        object.__setattr__(
            self,
            "max_states",
            _positive_int(self.max_states, "heterogeneous max_states"),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema": "neural-forge-lfm2-heterogeneous-catalog-input-v1",
                "target_pins": self.target_pins.as_dict(),
                "layer_options": [value.as_dict() for value in self.layer_options],
                "max_states": self.max_states,
            }
        )

    def select(
        self,
        source_graph: ModelCircuitGraph,
        *,
        exact_target_unique_parameters: int,
    ) -> Lfm2HeterogeneousLayoutSelection:
        return select_lfm2_heterogeneous_physical_layout(
            source_graph=source_graph,
            layer_options=self.layer_options,
            target_pins=self.target_pins,
            exact_target_unique_parameters=exact_target_unique_parameters,
            max_states=self.max_states,
        )


@dataclass(frozen=True, slots=True)
class DenseComponentAllocationReceipt:
    """Role-aware allocation proof for one materializable global topology.

    The current LFM2 runtime exposes shared model widths, so one receipt binds
    one legal whole-graph configuration.  It never claims heterogeneous
    per-layer widths.  The embedded generic allocator selection supplies the
    explicit semantic-role retention and exact-count audit.
    """

    policy: str
    topology_option_id: str
    topology_configuration_sha256: str
    source_graph_fingerprint: str
    source_unique_parameters: int
    source_logical_parameters: int
    target_unique_parameters: int
    target_logical_parameters: int
    allocation_selection: ArchitectureDiscoveredBudgetSelection
    physical_layout_kind: str | None = None
    physical_layout_descriptor_sha256: str | None = None
    physical_layout_selection_receipt_sha256: str | None = None
    physical_layout_source_topology_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _text(self.policy, "policy"))
        if self.policy != COMPONENT_ALLOCATION_POLICY:
            raise DynamicParameterFrontierError(
                "Unknown component-allocation receipt policy."
            )
        object.__setattr__(
            self,
            "topology_option_id",
            _text(self.topology_option_id, "topology_option_id"),
        )
        for name in (
            "topology_configuration_sha256",
            "source_graph_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name),
            )
        for name in (
            "source_unique_parameters",
            "source_logical_parameters",
            "target_unique_parameters",
            "target_logical_parameters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        selection = self.allocation_selection
        if not isinstance(selection, ArchitectureDiscoveredBudgetSelection):
            raise DynamicParameterFrontierError(
                "allocation_selection must be an "
                "ArchitectureDiscoveredBudgetSelection."
            )
        if (
            selection.source_graph_fingerprint
            != self.source_graph_fingerprint
            or selection.exact_unique_parameters
            != self.target_unique_parameters
            or self.source_logical_parameters < self.source_unique_parameters
            or self.target_logical_parameters < self.target_unique_parameters
            or self.target_unique_parameters >= self.source_unique_parameters
        ):
            raise DynamicParameterFrontierError(
                "Component-allocation receipt is not bound to one exact "
                "whole-graph topology/source."
            )
        physical_values = (
            self.physical_layout_kind,
            self.physical_layout_descriptor_sha256,
            self.physical_layout_selection_receipt_sha256,
            self.physical_layout_source_topology_sha256,
        )
        heterogeneous = any(value is not None for value in physical_values)
        if not heterogeneous:
            if selection.selected_option_ids != (self.topology_option_id,):
                raise DynamicParameterFrontierError(
                    "A shared-width component receipt must select its one "
                    "whole-graph topology option."
                )
        else:
            if any(value is None for value in physical_values):
                raise DynamicParameterFrontierError(
                    "A heterogeneous physical-layout receipt requires every "
                    "layout/source/selection seal."
                )
            if self.physical_layout_kind != LFM2_HETEROGENEOUS_LAYOUT_KIND:
                raise DynamicParameterFrontierError(
                    "Unknown heterogeneous physical-layout receipt kind."
                )
            for name in (
                "physical_layout_descriptor_sha256",
                "physical_layout_selection_receipt_sha256",
                "physical_layout_source_topology_sha256",
            ):
                object.__setattr__(
                    self,
                    name,
                    _sha256_digest(getattr(self, name), name),
                )
            if (
                self.physical_layout_source_topology_sha256
                != self.source_graph_fingerprint
                or self.target_logical_parameters
                != self.target_unique_parameters
                or not selection.selected_option_ids
            ):
                raise DynamicParameterFrontierError(
                    "Heterogeneous physical-layout receipt is not bound to "
                    "its untied exact source/selection/count."
                )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    @property
    def semantic_retention(self) -> dict[str, int]:
        return {
            value.scope_id: value.retention_ppm
            for value in self.allocation_selection.semantic_retention
        }

    @property
    def total_predicted_risk(self) -> float:
        return self.allocation_selection.total_predicted_risk

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "dense_component_allocation_receipt",
            "policy": self.policy,
            "topology_option_id": self.topology_option_id,
            "topology_configuration_sha256": (
                self.topology_configuration_sha256
            ),
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "source_unique_parameters": self.source_unique_parameters,
            "source_logical_parameters": self.source_logical_parameters,
            "target_unique_parameters": self.target_unique_parameters,
            "target_logical_parameters": self.target_logical_parameters,
            "allocation_selection": self.allocation_selection.as_dict(),
            "semantic_role_retention_ppm": self.semantic_retention,
            "total_predicted_risk": self.total_predicted_risk,
            "whole_graph_option": True,
            "materializable_global_topology": True,
            "shared_hidden_intermediate_and_head_widths": (
                self.physical_layout_kind is None
            ),
            "heterogeneous_per_layer_widths_claimed": (
                self.physical_layout_kind is not None
            ),
            "exact_parameter_budget_met": True,
            "behavioral_retention_claimed": False,
        }
        if self.physical_layout_kind is not None:
            value.update(
                {
                    "physical_layout_kind": self.physical_layout_kind,
                    "physical_layout_descriptor_sha256": (
                        self.physical_layout_descriptor_sha256
                    ),
                    "physical_layout_selection_receipt_sha256": (
                        self.physical_layout_selection_receipt_sha256
                    ),
                    "physical_layout_source_topology_sha256": (
                        self.physical_layout_source_topology_sha256
                    ),
                }
            )
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class HfLlamaDenseTopology:
    """One legal dense, bias-free, independently stored Llama decoder.

    The topology describes learned elements only.  It deliberately has no
    representation for tied embeddings, parameter aliases, quantization,
    pruning, sparse storage, or factorized weights.  The catalog uses this
    same value object for enumeration and exact-count authentication so the
    public arithmetic cannot drift from emitted blank-target configurations.
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    parameter_alias_policy: str = HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        if self.num_attention_heads * self.head_dim != self.hidden_size:
            raise DynamicParameterFrontierError(
                "Dense Llama requires num_attention_heads * head_dim == "
                "hidden_size."
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise DynamicParameterFrontierError(
                "Dense Llama requires num_key_value_heads to divide "
                "num_attention_heads."
            )
        if self.head_dim % 2:
            raise DynamicParameterFrontierError(
                "Dense Llama requires an even RoPE head_dim."
            )
        if self.tie_word_embeddings is not False:
            raise DynamicParameterFrontierError(
                "Dense Llama blank targets require independently stored, "
                "untied embedding and LM-head matrices."
            )
        if self.attention_bias is not False or self.mlp_bias is not False:
            raise DynamicParameterFrontierError(
                "The dense Llama topology catalog supports only bias-free "
                "attention and SwiGLU projections."
            )
        alias_policy = _text(
            self.parameter_alias_policy,
            "parameter_alias_policy",
        ).lower()
        if alias_policy != HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE:
            raise DynamicParameterFrontierError(
                "Dense Llama blank targets reject shared or aliased "
                "parameters."
            )
        object.__setattr__(self, "parameter_alias_policy", alias_policy)

    @property
    def token_interface_parameters(self) -> int:
        return 2 * self.vocab_size * self.hidden_size

    @property
    def attention_parameters_per_layer(self) -> int:
        return (
            2
            * self.hidden_size
            * self.head_dim
            * (self.num_attention_heads + self.num_key_value_heads)
        )

    @property
    def mlp_parameters_per_layer(self) -> int:
        return 3 * self.hidden_size * self.intermediate_size

    @property
    def normalization_parameters(self) -> int:
        return (2 * self.num_hidden_layers + 1) * self.hidden_size

    @property
    def unique_parameters(self) -> int:
        return (
            self.token_interface_parameters
            + self.normalization_parameters
            + self.num_hidden_layers
            * (
                self.attention_parameters_per_layer
                + self.mlp_parameters_per_layer
            )
        )

    @property
    def logical_parameters(self) -> int:
        return self.unique_parameters

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": HF_LLAMA_TARGET_SCHEMA,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
            "mlp_bias": self.mlp_bias,
            "parameter_alias_policy": self.parameter_alias_policy,
            "unique_parameters": self.unique_parameters,
            "logical_parameters": self.logical_parameters,
            "parameter_aliases": 0,
            "dense_parameters": True,
        }


def _parse_hf_llama_ordered_source_spans(
    value: Any,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, str) or not value:
        raise DynamicParameterFrontierError(
            "Dense Llama target requires encoded ordered source spans."
        )
    spans: list[tuple[int, ...]] = []
    for encoded_span in value.split(";"):
        if not encoded_span:
            raise DynamicParameterFrontierError(
                "Dense Llama ordered source spans cannot be empty."
            )
        indices: list[int] = []
        for encoded_index in encoded_span.split(","):
            if (
                not encoded_index.isascii()
                or not encoded_index.isdecimal()
                or str(int(encoded_index)) != encoded_index
            ):
                raise DynamicParameterFrontierError(
                    "Dense Llama ordered source spans must use canonical "
                    "non-negative decimal indices."
                )
            indices.append(int(encoded_index))
        spans.append(tuple(indices))
    return tuple(spans)


def _validate_hf_llama_candidate_configuration(
    candidate: DenseTargetTopologyCandidate,
) -> None:
    configuration = dict(candidate.configuration)
    schema = configuration.get(HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY)
    if schema is None:
        return
    if schema != HF_LLAMA_TARGET_SCHEMA:
        raise DynamicParameterFrontierError(
            "Dense Llama target schema marker is unknown."
        )
    if (
        candidate.family != "llama"
        or configuration.get(DENSE_TOPOLOGY_CATALOG_CONFIG_KEY)
        != candidate.catalog_id
    ):
        raise DynamicParameterFrontierError(
            "Dense Llama target family/catalog provenance disagrees."
        )
    encoded_spans = _parse_hf_llama_ordered_source_spans(
        configuration.get(HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY)
    )
    source_layer_count = _positive_int(
        configuration.get(HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY),
        "Dense Llama source layer count",
    )
    target_layer_count = _positive_int(
        configuration.get("num_hidden_layers"),
        "Dense Llama num_hidden_layers",
    )
    flattened = tuple(index for span in candidate.ordered_source_spans for index in span)
    if (
        encoded_spans != candidate.ordered_source_spans
        or len(candidate.ordered_source_spans) != target_layer_count
        or len(flattened) != source_layer_count
        or flattened != tuple(range(source_layer_count))
    ):
        raise DynamicParameterFrontierError(
            "Dense Llama typed/configured spans or source/target layer counts "
            "disagree."
        )
    topology = HfLlamaDenseTopology(
        vocab_size=configuration.get("vocab_size"),
        hidden_size=configuration.get("hidden_size"),
        intermediate_size=configuration.get("intermediate_size"),
        num_hidden_layers=target_layer_count,
        num_attention_heads=configuration.get("num_attention_heads"),
        num_key_value_heads=configuration.get("num_key_value_heads"),
        head_dim=configuration.get("head_dim"),
        tie_word_embeddings=configuration.get("tie_word_embeddings"),
        attention_bias=configuration.get("attention_bias"),
        mlp_bias=configuration.get("mlp_bias"),
        parameter_alias_policy=configuration.get(
            HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY
        ),
    )
    if (
        candidate.native_tied_parameter_count
        or candidate.native_ties
        or candidate.unique_parameters != topology.unique_parameters
        or candidate.logical_parameters != topology.logical_parameters
        or candidate.target_vocabulary_size != topology.vocab_size
        or candidate.vocabulary_unique_parameters
        != topology.token_interface_parameters
    ):
        raise DynamicParameterFrontierError(
            "Dense Llama candidate topology, exact count, vocabulary, or "
            "untied storage contract disagrees with its configuration."
        )
    bridge_seals = TOKENIZER_BRIDGE_CONFIG_KEYS.intersection(configuration)
    if candidate.tokenizer_bridge_required:
        if bridge_seals and bridge_seals != TOKENIZER_BRIDGE_CONFIG_KEYS:
            raise DynamicParameterFrontierError(
                "Dense Llama tokenizer bridge configuration is incomplete."
            )
        declares_bridge = CAP_TOKENIZER_BRIDGE in (
            candidate.declared_compiler_capabilities
        )
        if (
            declares_bridge or candidate.compiler_executable
        ) and bridge_seals != TOKENIZER_BRIDGE_CONFIG_KEYS:
            raise DynamicParameterFrontierError(
                "Dense Llama changed-vocabulary execution requires complete "
                "authenticated tokenizer bridge bindings."
            )
        for key in bridge_seals:
            _sha256_digest(configuration[key], f"Dense Llama {key}")
    elif bridge_seals:
        raise DynamicParameterFrontierError(
            "Dense Llama same-vocabulary target contains an unexpected "
            "tokenizer bridge binding."
        )


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
    component_allocation_receipt: DenseComponentAllocationReceipt | None = None
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
        _validate_hf_llama_candidate_configuration(self)
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
        receipt = self.component_allocation_receipt
        configuration_values = dict(self.configuration)
        heterogeneous_artifacts = (
            _restore_lfm2_heterogeneous_candidate_artifacts(
                self.configuration
            )
        )
        allocation_seals = _COMPONENT_ALLOCATION_CONFIGURATION_KEYS.intersection(
            configuration_values
        )
        if receipt is None:
            if allocation_seals or heterogeneous_artifacts is not None:
                raise DynamicParameterFrontierError(
                    "A topology candidate contains authenticated allocation/"
                    "physical-layout artifacts without their typed receipt."
                )
        else:
            if not isinstance(receipt, DenseComponentAllocationReceipt):
                raise DynamicParameterFrontierError(
                    "component_allocation_receipt has the wrong type."
                )
            expected_seals = {
                COMPONENT_ALLOCATION_POLICY_CONFIG_KEY: receipt.policy,
                COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY: receipt.fingerprint,
                COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY: (
                    receipt.topology_configuration_sha256
                ),
            }
            if (
                allocation_seals != _COMPONENT_ALLOCATION_CONFIGURATION_KEYS
                or any(
                    configuration_values.get(key) != value
                    for key, value in expected_seals.items()
                )
                or receipt.topology_configuration_sha256
                != component_allocation_topology_fingerprint(self.configuration)
                or receipt.source_unique_parameters
                != self.source_unique_parameters
                or receipt.target_unique_parameters != self.unique_parameters
                or receipt.target_logical_parameters != self.logical_parameters
            ):
                raise DynamicParameterFrontierError(
                    "Component-allocation receipt/seals differ from the "
                    "candidate topology or exact parameter counts."
                )
            if heterogeneous_artifacts is None:
                if receipt.physical_layout_kind is not None:
                    raise DynamicParameterFrontierError(
                        "A heterogeneous component receipt has no physical "
                        "layout artifacts."
                    )
            else:
                descriptor, selection_receipt = heterogeneous_artifacts
                decision = selection_receipt.allocation_decision
                allocation = receipt.allocation_selection
                selected_layer_option_sha256 = tuple(
                    option_id.rsplit("/", 1)[-1]
                    for option_id in allocation.selected_option_ids
                    if "/option/" in option_id
                )
                if (
                    receipt.physical_layout_kind
                    != LFM2_HETEROGENEOUS_LAYOUT_KIND
                    or receipt.physical_layout_descriptor_sha256
                    != descriptor.descriptor_sha256
                    or receipt.physical_layout_selection_receipt_sha256
                    != selection_receipt.receipt_sha256
                    or receipt.physical_layout_source_topology_sha256
                    != descriptor.source_topology_sha256
                    or descriptor.source_topology_sha256
                    != receipt.source_graph_fingerprint
                    or descriptor.accounting.unique_parameter_elements
                    != self.unique_parameters
                    or descriptor.accounting.logical_parameter_elements
                    != self.logical_parameters
                    or self.native_tied_parameter_count
                    or self.native_ties
                    or self.target_vocabulary_size != descriptor.vocab_size
                    or self.vocabulary_unique_parameters
                    != 2 * descriptor.vocab_size * descriptor.hidden_size
                    or allocation.problem_fingerprint
                    != decision.budget_problem_fingerprint
                    or allocation.fingerprint
                    != decision.budget_selection_fingerprint
                    or len(allocation.selected_option_ids)
                    != len(decision.selected_layer_option_sha256) + 1
                    or not allocation.selected_option_ids[0].endswith("/fixed")
                    or selected_layer_option_sha256
                    != decision.selected_layer_option_sha256
                    or allocation.topology_selection.problem_fingerprint
                    != decision.exact_topology_problem_fingerprint
                    or allocation.topology_selection.fingerprint
                    != decision.exact_topology_selection_fingerprint
                    or allocation.exact_unique_parameters
                    != decision.exact_target_unique_parameters
                    or not math.isclose(
                        allocation.total_predicted_risk,
                        decision.total_predicted_risk,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise DynamicParameterFrontierError(
                        "Heterogeneous candidate/receipt/descriptor bindings "
                        "differ."
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
            component_allocation_receipt=self.component_allocation_receipt,
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
            "component_allocation_receipt": (
                None
                if self.component_allocation_receipt is None
                else self.component_allocation_receipt.as_dict()
            ),
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
        if self.selection_policy == (
            "lfm2_untied_preserve_frontier_rebalance_v1"
        ):
            selection_reason = (
                "select_the_authenticated_tied_balanced_topology_then_"
                "reserve_a_dense_untied_output_inside_the_same_budget"
            )
        elif self.selection_policy == (
            "architecture_discovered_component_risk_executable_v1"
        ):
            selection_reason = (
                "choose_the_lowest_discovered_component_risk_among_"
                "materializable_global_topologies"
            )
        else:
            selection_reason = (
                "all_required_compiler_transforms_are_declared_then_choose_"
                "the_best_circuit_balanced_supported_pareto_allocation"
            )
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
            "selection_reason": selection_reason,
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
    *,
    prefer_component_risk: bool,
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
    legacy_rank = (
        -balanced_score,
        abs(candidate.budget_delta),
        -candidate.retention_score_ppm,
        -candidate.unique_parameters,
        tuple(
            (key, str(value))
            for key, value in component_allocation_core_configuration(
                candidate.configuration
            )
        ),
    )
    if not prefer_component_risk:
        # Component-risk planning can add a new full-stream winner that was
        # absent from the historical supported-Pareto frontier.  Keep it as a
        # fail-safe fallback, but do not let that policy-only candidate replace
        # a generic Pareto candidate in balanced legacy selection.
        generic_pareto_support = any(
            profile_name != LFM2_COMPONENT_POLICY_PROFILE
            for profile_name in candidate.pareto_support_profiles
        )
        return (0 if generic_pareto_support else 1, *legacy_rank)
    receipt = candidate.component_allocation_receipt
    if receipt is None:
        return (1, *legacy_rank)
    semantic = receipt.semantic_retention
    return (
        0,
        receipt.total_predicted_risk,
        -semantic.get(SEMANTIC_STATE_MIXER_INTERNAL, 0),
        -semantic.get(SEMANTIC_ATTENTION_QUERY_KEY, 0),
        *legacy_rank,
    )


def select_best_executable_dense_candidate(
    frontier: DenseParameterFrontierResult,
    *,
    available_compiler_capabilities: Sequence[str] | None = None,
    prefer_component_risk: bool = True,
) -> DenseExecutableCandidateSelection:
    """Select the best supported-Pareto topology the compiler can execute.

    With no override, the catalog's capability declaration is used only when
    every returned candidate carries the same declaration.  Missing or
    inconsistent provenance fails closed; it never silently falls back to
    candidate rank zero.  ``prefer_component_risk=False`` preserves the
    historical generic-Pareto balanced selection while retaining newer
    component-only candidates as fallbacks.
    """

    if not isinstance(frontier, DenseParameterFrontierResult):
        raise DynamicParameterFrontierError(
            "frontier must be DenseParameterFrontierResult."
        )
    if not isinstance(prefer_component_risk, bool):
        raise DynamicParameterFrontierError(
            "prefer_component_risk must be boolean."
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
        key=lambda item: _executable_pareto_selection_rank(
            item[1],
            prefer_component_risk=prefer_component_risk,
        ),
    )
    selection_policy = (
        "architecture_discovered_component_risk_executable_v1"
        if (
            prefer_component_risk
            and selected.component_allocation_receipt is not None
        )
        else "supported_pareto_balanced_executable_v1"
    )
    return DenseExecutableCandidateSelection(
        frontier_fingerprint=frontier.fingerprint,
        selected_candidate_rank=selected_rank,
        selected_candidate_fingerprint=selected.fingerprint,
        available_compiler_capabilities=available,
        selection_policy=selection_policy,
        executable_candidate_count=len(executable),
        rejected_candidates=tuple(rejected),
    )


def select_lfm2_untied_rebalance_candidate(
    frontier: DenseParameterFrontierResult,
) -> DenseExecutableCandidateSelection:
    """Select the unique catalog-derived untied companion to the tied lane.

    The topology catalog authenticates and full-search Pareto-checks this
    candidate while enumerating the frontier.  This helper does no reranking:
    it names that explicit allocation contract so callers can pass its stable
    rank to :func:`plan_budgeted_dynamic_target` and materialize it through the
    existing target synthesizer.
    """

    if not isinstance(frontier, DenseParameterFrontierResult):
        raise DynamicParameterFrontierError(
            "frontier must be DenseParameterFrontierResult."
        )
    matches = tuple(
        (rank, candidate)
        for rank, candidate in enumerate(frontier.candidates)
        if _LFM2_UNTIED_REBALANCE_PROFILE
        in candidate.pareto_support_profiles
    )
    if len(matches) != 1:
        raise DynamicParameterFrontierCapabilityError(
            "Frontier does not contain exactly one authenticated LFM2 untied "
            "rebalance candidate.",
            {
                "frontier_fingerprint": frontier.fingerprint,
                "reason": "missing_or_ambiguous_untied_rebalance",
                "matching_candidate_count": len(matches),
            },
        )
    rank, candidate = matches[0]
    configuration = dict(candidate.configuration)
    if (
        frontier.request.target_family != "lfm2"
        or configuration.get(LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY)
        != LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
        or configuration.get("tie_word_embeddings") is not False
        or candidate.native_tied_parameter_count != 0
        or candidate.native_ties
        or candidate.logical_parameters != candidate.unique_parameters
        or not candidate.at_or_below_budget
        or not candidate.compiler_executable
        or candidate.missing_compiler_capabilities
        or not candidate.pareto_dominance_verified
    ):
        raise DynamicParameterFrontierCapabilityError(
            "Authenticated LFM2 untied rebalance candidate is not a dense "
            "executable at-or-below target.",
            {
                "frontier_fingerprint": frontier.fingerprint,
                "candidate_fingerprint": candidate.fingerprint,
                "reason": "untied_rebalance_contract_failed",
            },
        )
    return DenseExecutableCandidateSelection(
        frontier_fingerprint=frontier.fingerprint,
        selected_candidate_rank=rank,
        selected_candidate_fingerprint=candidate.fingerprint,
        available_compiler_capabilities=(
            candidate.declared_compiler_capabilities
        ),
        selection_policy=(
            "lfm2_untied_preserve_frontier_rebalance_v1"
        ),
        executable_candidate_count=1,
        rejected_candidates=(),
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
            receipt = candidate.component_allocation_receipt
            if (
                receipt is not None
                and receipt.source_graph_fingerprint != parent_graph.fingerprint
            ):
                raise DynamicParameterFrontierError(
                    "Topology catalog returned a component-allocation receipt "
                    "bound to a different inspected graph."
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
    tied_token_interface: bool | None = None,
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
    target_tied = (
        geometry.tied_token_interface
        if tied_token_interface is None
        else tied_token_interface
    )
    if not isinstance(target_tied, bool):
        raise DynamicParameterFrontierError(
            "tied_token_interface must be boolean when supplied."
        )
    token_matrix = vocabulary * hidden
    token_unique = token_matrix if target_tied else 2 * token_matrix
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


@dataclass(frozen=True, slots=True)
class _DenseComponentPolicyScore:
    total_predicted_risk: float
    semantic_retention_ppm: tuple[tuple[str, int], ...]

    @property
    def semantic_retention(self) -> dict[str, int]:
        return dict(self.semantic_retention_ppm)


class _DenseComponentPolicyScorer:
    """Exact lightweight form of the one-option component allocator.

    A dense catalog candidate is already one complete global topology, so the
    generic exact allocator has no combinatorial choice left to solve.  This
    prepared scorer reproduces its retention floors and predicted-risk
    arithmetic without constructing a typed option/problem/receipt for every
    topology in the (large) native stream.  Typed receipts remain mandatory
    for the small returned frontier and are cross-checked against this score.
    """

    __slots__ = (
        "_bottleneck_risk_weight",
        "_entries",
        "_entries_by_semantic",
        "_fast_semantic_only",
        "_internal_source_parameters",
        "_loss_scope_specs",
        "_rules",
        "_source_by_semantic",
        "_total_loss_scope_weight",
    )

    def __init__(
        self,
        graph: ModelCircuitGraph,
        *,
        protection_rules: tuple[ComponentProtectionRule, ...],
        bottleneck_risk_weight: float,
    ) -> None:
        rules = (
            protection_rules
            if protection_rules
            else residual_attention_component_protection_rules()
        )
        entries: list[tuple[int, str, str, int]] = []
        entries_by_semantic: dict[
            str,
            list[tuple[int, str, int]],
        ] = {}
        ordinal = 0
        for node in graph.nodes:
            by_role: dict[str, int] = {}
            for parameter in node.parameters:
                by_role[parameter.role] = (
                    by_role.get(parameter.role, 0) + parameter.numel
                )
            for role, count in by_role.items():
                semantic = classify_component_parameter_semantic(
                    node.kind,
                    role,
                )
                entries.append((ordinal, node.kind, semantic, count))
                entries_by_semantic.setdefault(semantic, []).append(
                    (ordinal, node.kind, count)
                )
                ordinal += 1
        source_by_semantic = {
            semantic: sum(count for _order, _kind, count in values)
            for semantic, values in entries_by_semantic.items()
        }
        if sum(source_by_semantic.values()) != graph.logical_parameter_count:
            raise DynamicParameterFrontierError(
                "Prepared component-policy source accounting does not "
                "reproduce the inspected graph logical parameter count."
            )
        self._rules = _effective_component_protection_rules(rules)
        self._entries = tuple(entries)
        self._entries_by_semantic = {
            semantic: tuple(values)
            for semantic, values in entries_by_semantic.items()
        }
        self._source_by_semantic = source_by_semantic
        self._bottleneck_risk_weight = bottleneck_risk_weight
        self._fast_semantic_only = all(
            not rule.circuit_kinds for rule in self._rules
        )
        semantics = tuple(sorted(source_by_semantic))
        loss_scope_specs: list[
            tuple[ComponentProtectionRule | None, str, float, int]
        ] = [
            (None, semantic, 1.0, source_by_semantic[semantic])
            for semantic in semantics
            if source_by_semantic[semantic]
        ]
        for rule in self._rules:
            if not rule.loss_weight:
                continue
            for semantic in semantics:
                source = sum(
                    count
                    for _ordinal, kind, role_semantic, count in self._entries
                    if role_semantic == semantic
                    and rule.matches(kind, role_semantic)
                )
                if source:
                    loss_scope_specs.append(
                        (rule, semantic, rule.loss_weight, source)
                    )
        self._loss_scope_specs = tuple(loss_scope_specs)
        self._total_loss_scope_weight = math.fsum(
            weight for _rule, _semantic, weight, _source in loss_scope_specs
        )
        self._internal_source_parameters = tuple(
            (semantic, source_by_semantic[semantic])
            for semantic in (
                SEMANTIC_STATE_MIXER_INTERNAL,
                SEMANTIC_ATTENTION_QUERY_KEY,
                SEMANTIC_ATTENTION_WRITER,
                SEMANTIC_MLP_PRODUCER,
                SEMANTIC_MLP_WRITER,
                SEMANTIC_NORMALIZATION,
            )
            if semantic in source_by_semantic
        )

    def _normalize_targets(
        self,
        target_by_semantic: Mapping[str, int],
    ) -> dict[str, int]:
        targets = dict(target_by_semantic)
        for primary, fallback in (
            (SEMANTIC_ATTENTION_QUERY_KEY, SEMANTIC_ATTENTION_WRITER),
            (SEMANTIC_ATTENTION_WRITER, SEMANTIC_ATTENTION_QUERY_KEY),
            (SEMANTIC_MLP_WRITER, SEMANTIC_MLP_PRODUCER),
            (SEMANTIC_MLP_PRODUCER, SEMANTIC_MLP_WRITER),
        ):
            if (
                primary not in self._source_by_semantic
                and fallback in self._source_by_semantic
            ):
                targets[fallback] = (
                    targets.get(fallback, 0) + targets.get(primary, 0)
                )
                targets[primary] = 0
        unsupported = {
            semantic: count
            for semantic, count in targets.items()
            if count and semantic not in self._source_by_semantic
        }
        if unsupported:
            raise DynamicParameterFrontierError(
                "The inspected graph lacks parameter-role semantics required "
                f"by the legal target: {sorted(unsupported)}."
            )
        for semantic, source in self._source_by_semantic.items():
            target = _nonnegative_int(
                targets.get(semantic, 0),
                f"target capacity for {semantic}",
            )
            if target > source:
                raise DynamicParameterFrontierError(
                    "A legal global topology claims more retained component-"
                    f"role capacity than its source for {semantic!r}."
                )
            targets[semantic] = target
        return targets

    def _semantic_only_risk(
        self,
        targets: Mapping[str, int],
    ) -> float | None:
        for rule in self._rules:
            semantics = tuple(
                semantic
                for semantic in self._source_by_semantic
                if rule.matches("", semantic)
            )
            source = sum(
                self._source_by_semantic[semantic]
                for semantic in semantics
            )
            retained = sum(targets[semantic] for semantic in semantics)
            if (
                source
                and _bounded_ratio_ppm(retained, source)
                < rule.minimum_retention_ppm
            ):
                return None
        return _normalized_component_retention_risk(
            structural_risk=0.0,
            weighted_loss_scopes=tuple(
                (weight, source, source, targets[semantic])
                for _rule, semantic, weight, source in (
                    self._loss_scope_specs
                )
            ),
            total_loss_scope_weight=self._total_loss_scope_weight,
            internal_retention_scopes=tuple(
                (source, source, targets[semantic])
                for semantic, source in self._internal_source_parameters
            ),
            internal_semantic_count=len(
                self._internal_source_parameters
            ),
            bottleneck_risk_weight=self._bottleneck_risk_weight,
        )

    def _kind_sensitive_risk(
        self,
        targets: Mapping[str, int],
    ) -> float | None:
        retained_by_ordinal: dict[int, int] = {}
        for semantic, entries in self._entries_by_semantic.items():
            source = self._source_by_semantic[semantic]
            target = targets[semantic]
            floors: list[tuple[int, int, int]] = []
            floor_total = 0
            for ordinal, _kind, count in entries:
                numerator = count * target
                value = numerator // source
                floors.append((numerator % source, ordinal, value))
                floor_total += value
            extra = target - floor_total
            for index, (_remainder, ordinal, value) in enumerate(
                sorted(floors, key=lambda item: (-item[0], item[1]))
            ):
                retained_by_ordinal[ordinal] = value + (
                    1 if index < extra else 0
                )
        for rule in self._rules:
            source = 0
            retained = 0
            for ordinal, kind, semantic, count in self._entries:
                if rule.matches(kind, semantic):
                    source += count
                    retained += retained_by_ordinal[ordinal]
            if (
                source
                and _bounded_ratio_ppm(retained, source)
                < rule.minimum_retention_ppm
            ):
                return None
        weighted_loss_scopes = []
        for rule, scope_semantic, weight, global_source in (
            self._loss_scope_specs
        ):
            values = tuple(
                (ordinal, count)
                for ordinal, kind, semantic, count in self._entries
                if semantic == scope_semantic
                and (rule is None or rule.matches(kind, semantic))
            )
            weighted_loss_scopes.append(
                (
                    weight,
                    global_source,
                    sum(count for _ordinal, count in values),
                    sum(
                        retained_by_ordinal[ordinal]
                        for ordinal, _count in values
                    ),
                )
            )
        internal_retention_scopes = []
        for semantic, global_source in self._internal_source_parameters:
            values = tuple(
                (ordinal, count)
                for ordinal, _kind, role_semantic, count in self._entries
                if role_semantic == semantic
            )
            internal_retention_scopes.append(
                (
                    global_source,
                    sum(count for _ordinal, count in values),
                    sum(
                        retained_by_ordinal[ordinal]
                        for ordinal, _count in values
                    ),
                )
            )
        return _normalized_component_retention_risk(
            structural_risk=0.0,
            weighted_loss_scopes=weighted_loss_scopes,
            total_loss_scope_weight=self._total_loss_scope_weight,
            internal_retention_scopes=internal_retention_scopes,
            internal_semantic_count=len(
                self._internal_source_parameters
            ),
            bottleneck_risk_weight=self._bottleneck_risk_weight,
        )

    def score(
        self,
        target_by_semantic: Mapping[str, int],
    ) -> _DenseComponentPolicyScore | None:
        targets = self._normalize_targets(target_by_semantic)
        if self._fast_semantic_only:
            risk = self._semantic_only_risk(targets)
        else:
            risk = self._kind_sensitive_risk(targets)
        if risk is None:
            return None
        return _DenseComponentPolicyScore(
            total_predicted_risk=risk,
            semantic_retention_ppm=tuple(
                (
                    semantic,
                    _bounded_ratio_ppm(
                        targets[semantic],
                        self._source_by_semantic[semantic],
                    ),
                )
                for semantic in sorted(self._source_by_semantic)
            ),
        )


def _lfm2_target_semantic_capacities_from_geometry(
    geometry: _Lfm2Geometry,
    *,
    logical_parameters: int,
    hidden_size: int,
    intermediate_size: int,
    query_heads: int,
    key_value_heads: int,
    head_dim: int,
    vocabulary_size: int,
    layer_types: Sequence[str],
) -> dict[str, int]:
    hidden = _positive_int(hidden_size, "hidden_size")
    intermediate = _positive_int(intermediate_size, "intermediate_size")
    query = _positive_int(query_heads, "num_attention_heads")
    key_value = _positive_int(key_value_heads, "num_key_value_heads")
    dimension = _positive_int(head_dim, "head_dim")
    vocabulary = _positive_int(vocabulary_size, "vocab_size")
    target_types = tuple(layer_types)
    attention_layers = target_types.count("full_attention")
    convolution_layers = target_types.count("conv")
    depth = len(target_types)
    capacities = {
        SEMANTIC_INTERFACE: 2 * vocabulary * hidden,
        SEMANTIC_NORMALIZATION: (2 * depth + 1) * hidden,
        SEMANTIC_STATE_MIXER_INTERNAL: convolution_layers
        * (4 * hidden * hidden + geometry.fir_length * hidden),
        SEMANTIC_ATTENTION_QUERY_KEY: attention_layers
        * (
            hidden * dimension * (query + key_value)
            + 2 * dimension
        ),
        SEMANTIC_ATTENTION_WRITER: attention_layers
        * hidden
        * dimension
        * (query + key_value),
        SEMANTIC_MLP_PRODUCER: depth * 2 * hidden * intermediate,
        SEMANTIC_MLP_WRITER: depth * hidden * intermediate,
        SEMANTIC_OTHER: 0,
    }
    if sum(capacities.values()) != logical_parameters:
        raise DynamicParameterFrontierError(
            "Component-role capacity accounting does not reproduce the legal "
            "LFM2 target logical parameter count."
        )
    return capacities


def _lfm2_target_semantic_capacities(
    geometry: _Lfm2Geometry,
    candidate: DenseTargetTopologyCandidate,
) -> dict[str, int]:
    """Exact logical target capacity by family-neutral component semantic."""

    heterogeneous = _restore_lfm2_heterogeneous_candidate_artifacts(
        candidate.configuration
    )
    if heterogeneous is not None:
        descriptor, _selection_receipt = heterogeneous
        hidden = descriptor.hidden_size
        mlp_producer = 0
        mlp_writer = 0
        attention_query_key = 0
        attention_writer = 0
        state_mixer = 0
        for layer in descriptor.layers:
            intermediate = layer.mlp_intermediate_size
            mlp_producer += 2 * hidden * intermediate
            mlp_writer += hidden * intermediate
            if layer.layer_type == "full_attention":
                assert layer.query_heads is not None
                assert layer.key_value_heads is not None
                assert layer.head_dim is not None
                role_width = layer.query_heads + layer.key_value_heads
                attention_query_key += (
                    hidden * layer.head_dim * role_width
                    + 2 * layer.head_dim
                )
                attention_writer += hidden * layer.head_dim * role_width
            else:
                assert layer.short_conv_internal_width is not None
                internal = layer.short_conv_internal_width
                writer = (
                    hidden * internal
                    if layer.short_conv_writer_rank is None
                    else layer.short_conv_writer_rank * (hidden + internal)
                )
                state_mixer += (
                    3 * hidden * internal
                    + descriptor.conv_kernel_size * internal
                    + writer
                )
        capacities = {
            SEMANTIC_INTERFACE: 2 * descriptor.vocab_size * hidden,
            SEMANTIC_NORMALIZATION: (
                (2 * descriptor.layer_count + 1) * hidden
            ),
            SEMANTIC_STATE_MIXER_INTERNAL: state_mixer,
            SEMANTIC_ATTENTION_QUERY_KEY: attention_query_key,
            SEMANTIC_ATTENTION_WRITER: attention_writer,
            SEMANTIC_MLP_PRODUCER: mlp_producer,
            SEMANTIC_MLP_WRITER: mlp_writer,
            SEMANTIC_OTHER: 0,
        }
        if sum(capacities.values()) != candidate.logical_parameters:
            raise DynamicParameterFrontierError(
                "Heterogeneous component-role capacities do not reproduce "
                "the exact target count."
            )
        return capacities

    configuration = dict(
        component_allocation_core_configuration(candidate.configuration)
    )
    hidden = _positive_int(configuration.get("hidden_size"), "hidden_size")
    intermediate = _positive_int(
        configuration.get("intermediate_size"),
        "intermediate_size",
    )
    query = _positive_int(
        configuration.get("num_attention_heads"),
        "num_attention_heads",
    )
    key_value = _positive_int(
        configuration.get("num_key_value_heads"),
        "num_key_value_heads",
    )
    head_dim = _positive_int(configuration.get("head_dim"), "head_dim")
    vocabulary = _positive_int(configuration.get("vocab_size"), "vocab_size")
    layer_types_value = configuration.get("layer_types")
    if not isinstance(layer_types_value, str):
        raise DynamicParameterFrontierError(
            "LFM2 component allocation requires encoded layer_types."
        )
    return _lfm2_target_semantic_capacities_from_geometry(
        geometry,
        logical_parameters=candidate.logical_parameters,
        hidden_size=hidden,
        intermediate_size=intermediate,
        query_heads=query,
        key_value_heads=key_value,
        head_dim=head_dim,
        vocabulary_size=vocabulary,
        layer_types=tuple(layer_types_value.split(",")),
    )


def _whole_graph_retained_allocations(
    graph: ModelCircuitGraph,
    target_by_semantic: Mapping[str, int],
) -> tuple[RetainedParameterAllocation, ...]:
    """Distribute exact role capacity without consulting paths/layer IDs."""

    grouped: dict[str, list[tuple[int, str, str, int]]] = {}
    ordinal = 0
    for node in graph.nodes:
        by_role: dict[str, int] = {}
        role_order: list[str] = []
        for parameter in node.parameters:
            if parameter.role not in by_role:
                by_role[parameter.role] = 0
                role_order.append(parameter.role)
            by_role[parameter.role] += parameter.numel
        for role in role_order:
            semantic = classify_component_parameter_semantic(node.kind, role)
            grouped.setdefault(semantic, []).append(
                (ordinal, node.node_id, role, by_role[role])
            )
            ordinal += 1

    normalized_targets = dict(target_by_semantic)
    for primary, fallback in (
        (SEMANTIC_ATTENTION_QUERY_KEY, SEMANTIC_ATTENTION_WRITER),
        (SEMANTIC_ATTENTION_WRITER, SEMANTIC_ATTENTION_QUERY_KEY),
        (SEMANTIC_MLP_WRITER, SEMANTIC_MLP_PRODUCER),
        (SEMANTIC_MLP_PRODUCER, SEMANTIC_MLP_WRITER),
    ):
        if primary not in grouped and fallback in grouped:
            normalized_targets[fallback] = (
                normalized_targets.get(fallback, 0)
                + normalized_targets.get(primary, 0)
            )
            normalized_targets[primary] = 0
    unsupported = {
        semantic: count
        for semantic, count in normalized_targets.items()
        if count and semantic not in grouped
    }
    if unsupported:
        raise DynamicParameterFrontierError(
            "The inspected graph lacks parameter-role semantics required by "
            f"the legal target: {sorted(unsupported)}."
        )

    retained: dict[tuple[str, str], int] = {}
    for semantic, entries in grouped.items():
        source = sum(count for _order, _node, _role, count in entries)
        target = _nonnegative_int(
            normalized_targets.get(semantic, 0),
            f"target capacity for {semantic}",
        )
        if target > source:
            raise DynamicParameterFrontierError(
                "A legal global topology claims more retained component-role "
                f"capacity than its source for {semantic!r}."
            )
        floors: list[tuple[int, int, str, str, int]] = []
        floor_total = 0
        for order, node_id, role, count in entries:
            numerator = count * target
            value = 0 if source == 0 else numerator // source
            remainder = 0 if source == 0 else numerator % source
            floors.append((remainder, order, node_id, role, value))
            floor_total += value
        extra = target - floor_total
        for index, (_remainder, _order, node_id, role, value) in enumerate(
            sorted(floors, key=lambda item: (-item[0], item[1]))
        ):
            retained[(node_id, role)] = value + (1 if index < extra else 0)
    return tuple(
        RetainedParameterAllocation(
            node_id=node.node_id,
            parameter_role=role,
            retained_parameters=retained.get((node.node_id, role), 0),
        )
        for node in graph.nodes
        for role in dict.fromkeys(parameter.role for parameter in node.parameters)
    )


def _bind_dense_component_allocation_receipt(
    graph: ModelCircuitGraph,
    candidate: DenseTargetTopologyCandidate,
    *,
    target_by_semantic: Mapping[str, int],
    protection_rules: tuple[ComponentProtectionRule, ...],
    bottleneck_risk_weight: float,
) -> DenseTargetTopologyCandidate:
    """Bind one dense global topology to the existing exact allocator proof."""

    if candidate.component_allocation_receipt is not None:
        raise DynamicParameterFrontierError(
            "Dense candidate already has a component-allocation receipt."
        )
    core_configuration = component_allocation_core_configuration(
        candidate.configuration
    )
    topology_sha256 = component_allocation_topology_fingerprint(
        core_configuration
    )
    option_id = f"global-topology.{candidate.candidate_id}"
    parameterized_nodes = tuple(
        node.node_id for node in graph.nodes if node.parameters
    )
    retained = _whole_graph_retained_allocations(
        graph,
        target_by_semantic,
    )
    bytes_per_unique_parameter = (
        graph.unique_parameter_bytes // graph.unique_parameter_count
        if graph.unique_parameter_count
        and graph.unique_parameter_bytes % graph.unique_parameter_count == 0
        else 0
    )
    option = DiscoveredComponentTopologyOption(
        option_id=option_id,
        group_id="materializable-global-topology",
        source_node_ids=parameterized_nodes,
        local_unique_parameters=candidate.unique_parameters,
        local_serialized_bytes=(
            candidate.unique_parameters * bytes_per_unique_parameter
        ),
        retained_parameters=retained,
        structural_risk=0.0,
        metadata=(
            ("family", candidate.family),
            ("global_shared_dimensions", True),
            ("heterogeneous_per_layer_dimensions", False),
            ("topology_configuration_sha256", topology_sha256),
        ),
    )
    problem = ArchitectureDiscoveredBudgetProblem(
        source_graph=graph,
        exact_target_unique_parameters=candidate.unique_parameters,
        groups=(
            DiscoveredComponentChoiceGroup(
                "materializable-global-topology",
                (option,),
            ),
        ),
        protection_rules=protection_rules,
        bottleneck_risk_weight=bottleneck_risk_weight,
    )
    allocation = solve_architecture_discovered_budget(problem)
    receipt = DenseComponentAllocationReceipt(
        policy=COMPONENT_ALLOCATION_POLICY,
        topology_option_id=option_id,
        topology_configuration_sha256=topology_sha256,
        source_graph_fingerprint=graph.fingerprint,
        source_unique_parameters=graph.unique_parameter_count,
        source_logical_parameters=graph.logical_parameter_count,
        target_unique_parameters=candidate.unique_parameters,
        target_logical_parameters=candidate.logical_parameters,
        allocation_selection=allocation,
    )
    configuration = {
        **dict(core_configuration),
        COMPONENT_ALLOCATION_POLICY_CONFIG_KEY: receipt.policy,
        COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY: receipt.fingerprint,
        COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY: topology_sha256,
    }
    return replace(
        candidate,
        configuration=_configuration(configuration),
        component_allocation_receipt=receipt,
    )


def _bind_lfm2_component_allocation_receipt(
    graph: ModelCircuitGraph,
    geometry: _Lfm2Geometry,
    candidate: DenseTargetTopologyCandidate,
    *,
    protection_rules: tuple[ComponentProtectionRule, ...],
    bottleneck_risk_weight: float,
) -> DenseTargetTopologyCandidate:
    """Run the shared exact allocator for one real global LFM2 topology."""

    return _bind_dense_component_allocation_receipt(
        graph,
        candidate,
        target_by_semantic=_lfm2_target_semantic_capacities(
            geometry,
            candidate,
        ),
        protection_rules=protection_rules,
        bottleneck_risk_weight=bottleneck_risk_weight,
    )


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
                try:
                    validate_gqa_attention_fold_geometry(
                        source_query_heads=geometry.query_heads,
                        source_key_value_heads=geometry.key_value_heads,
                        source_head_dim=geometry.head_dim,
                        target_query_heads=query_heads,
                        target_key_value_heads=key_value_heads,
                        target_head_dim=head_dim,
                    )
                except AttentionFoldError:
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

_LFM2_UNTIED_REBALANCE_PROFILE = "untied_preserve_frontier_rebalance"


def _lfm2_rebalance_tied_candidate_for_untied_output(
    *,
    requested_unique_parameters: int,
    base_tied_unique_parameters: int,
    target_vocabulary_size: int,
    hidden_size: int,
    layer_count: int,
    current_intermediate_size: int,
) -> tuple[int, int, int]:
    """Reserve one dense output matrix and absorb it through FFN width.

    This is the catalog-level form of the interface-frontier arithmetic: each
    LFM2 intermediate column costs exactly three dense hidden-size columns per
    layer, so ceiling division gives the first integral width at or below the
    same total-size ceiling.  It changes no residual, attention, vocabulary,
    depth, or operator axis.
    """

    requested = _positive_int(
        requested_unique_parameters,
        "requested_unique_parameters",
    )
    base = _positive_int(
        base_tied_unique_parameters,
        "base_tied_unique_parameters",
    )
    vocabulary = _positive_int(
        target_vocabulary_size,
        "target_vocabulary_size",
    )
    hidden = _positive_int(hidden_size, "hidden_size")
    depth = _positive_int(layer_count, "layer_count")
    intermediate = _positive_int(
        current_intermediate_size,
        "current_intermediate_size",
    )
    added_output = vocabulary * hidden
    unrebalanced = base + added_output
    overage = max(0, unrebalanced - requested)
    parameters_per_step = 3 * hidden * depth
    reduction = (
        0
        if overage == 0
        else (overage + parameters_per_step - 1) // parameters_per_step
    )
    target_intermediate = intermediate - reduction
    if target_intermediate < 1:
        raise DynamicParameterFrontierUnattainable(
            "The selected tied topology cannot absorb an untied dense output "
            "inside the requested budget.",
            {
                "requested_unique_parameters": requested,
                "base_tied_unique_parameters": base,
                "added_dense_output_parameters": added_output,
                "parameters_per_intermediate_step": parameters_per_step,
                "required_intermediate_reduction": reduction,
                "current_intermediate_size": intermediate,
                "reason": "untied_output_exhausts_ffn_width",
            },
        )
    target_unique = unrebalanced - reduction * parameters_per_step
    if target_unique > requested:
        raise DynamicParameterFrontierError(
            "Untied dense-output FFN rebalance remained above its ceiling."
        )
    return target_intermediate, target_unique, requested - target_unique


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
        (key, str(value))
        for key, value in component_allocation_core_configuration(
            candidate.configuration
        )
    )
    return (
        distance,
        side,
        -candidate.retention_score_ppm,
        -candidate.unique_parameters,
        configuration,
    )


@dataclass(frozen=True, slots=True)
class _HfDenseCausalSourceGeometry:
    family: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    layer_count: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    tied_token_interface: bool
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int


def _one_graph_node(
    graph: ModelCircuitGraph,
    kind: CircuitKind | str,
    *,
    layer_index: int | None,
    label: str,
) -> Any:
    values = tuple(
        node
        for node in graph.nodes_of_kind(kind)
        if node.layer_index == layer_index
    )
    if len(values) != 1:
        raise DynamicParameterFrontierError(
            f"Dense causal source requires exactly one {label}; got "
            f"{len(values)}."
        )
    return values[0]


def _compact_parameter_role(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if character.isascii() and character.isalnum()
    )


def _require_dense_operator_parameter(
    node: Any,
    *,
    role: str,
    shape: tuple[int, ...],
    layer_index: int,
) -> None:
    compact_role = _compact_parameter_role(role)
    matches = tuple(
        parameter
        for parameter in node.parameters
        if _compact_parameter_role(parameter.role) == compact_role
    )
    if len(matches) != 1 or matches[0].shape != shape:
        observed = tuple(
            (parameter.role, parameter.shape) for parameter in matches
        )
        raise DynamicParameterFrontierError(
            f"Dense causal source layer {layer_index} requires exactly one "
            f"{role} with shape {shape}; observed {observed!r}."
        )


def _inspect_hf_dense_causal_source(
    graph: ModelCircuitGraph,
) -> _HfDenseCausalSourceGeometry:
    """Authenticate the family-neutral IR facts needed by the Llama catalog."""

    modalities = tuple(sorted({value.modality for value in graph.interfaces}))
    if modalities != ("text",):
        raise DynamicParameterFrontierError(
            "The dense Hugging Face/Llama catalog supports exactly the text "
            "modality."
        )
    layer_indices = graph.layer_indices
    if not layer_indices or layer_indices != tuple(range(len(layer_indices))):
        raise DynamicParameterFrontierError(
            "Dense causal source layers must be contiguous and zero-based."
        )
    metadata = dict(graph.metadata)
    for marker in (
        "quantized",
        "pruned",
        "sparse_storage",
        "factorized",
        "codebook_backed",
        "non_native_weight_sharing",
    ):
        if metadata.get(marker) is True:
            raise DynamicParameterFrontierError(
                "Dense causal source rejects compressed or shared parameter "
                f"storage ({marker})."
            )

    hidden = _positive_int(metadata.get("hidden_size"), "graph hidden_size")
    embedding = _one_graph_node(
        graph,
        CircuitKind.TOKEN_EMBEDDING,
        layer_index=None,
        label="token embedding",
    )
    head = _one_graph_node(
        graph,
        CircuitKind.OUTPUT_HEAD,
        layer_index=None,
        label="output head",
    )
    final_norm = _one_graph_node(
        graph,
        CircuitKind.NORM,
        layer_index=None,
        label="final normalization",
    )
    embedding_attributes = _attributes(embedding)
    head_attributes = _attributes(head)
    vocab = _positive_int(
        embedding_attributes.get("vocabulary_size"),
        "source vocabulary_size",
    )
    if (
        embedding_attributes.get("hidden_size") != hidden
        or head_attributes.get("vocabulary_size") != vocab
        or embedding.logical_parameter_count != vocab * hidden
        or head.logical_parameter_count != vocab * hidden
        or len(embedding.parameters) != 1
        or len(head.parameters) != 1
    ):
        raise DynamicParameterFrontierError(
            "Dense causal source token interfaces do not match one full "
            "embedding and one full LM-head matrix."
        )
    configured_tie = head_attributes.get("tied_to_embedding")
    if not isinstance(configured_tie, bool):
        raise DynamicParameterFrontierError(
            "Dense causal source must explicitly declare embedding/head tying."
        )
    observed_tie = (
        embedding.parameters[0].storage_group
        == head.parameters[0].storage_group
    )
    if configured_tie != observed_tie:
        raise DynamicParameterFrontierError(
            "Dense causal source configured and observed token-interface ties "
            "disagree."
        )
    allowed_tied_group = (
        embedding.parameters[0].storage_group if observed_tie else None
    )
    storage_counts = _storage_group_counts(graph)
    unexpected_aliases = tuple(
        sorted(
            group
            for group, count in storage_counts.items()
            if count > 1 and group != allowed_tied_group
        )
    )
    if unexpected_aliases:
        raise DynamicParameterFrontierError(
            "Dense causal source rejects non-native parameter sharing or "
            f"aliases: {unexpected_aliases[:8]!r}."
        )
    expected_alias_delta = vocab * hidden if observed_tie else 0
    if (
        graph.logical_parameter_count - graph.unique_parameter_count
        != expected_alias_delta
    ):
        raise DynamicParameterFrontierError(
            "Dense causal source logical/unique counts do not reproduce its "
            "declared native token-interface tie."
        )

    attention_geometries: set[tuple[int, int, int]] = set()
    intermediate_sizes: set[int] = set()
    for layer_index in layer_indices:
        attention = _one_graph_node(
            graph,
            CircuitKind.ATTENTION,
            layer_index=layer_index,
            label=f"attention operator in layer {layer_index}",
        )
        attention_attributes = _attributes(attention)
        query = _positive_int(
            attention_attributes.get("query_heads"),
            f"layer {layer_index} query_heads",
        )
        key_value = _positive_int(
            attention_attributes.get("key_value_heads"),
            f"layer {layer_index} key_value_heads",
        )
        dimension = _positive_int(
            attention_attributes.get("head_dim"),
            f"layer {layer_index} head_dim",
        )
        if query % key_value:
            raise DynamicParameterFrontierError(
                "Dense causal source has invalid grouped-query attention."
            )
        if attention_attributes.get("projection_style") != "split_qkv":
            raise DynamicParameterFrontierError(
                "Dense causal source requires split q/k/v/o attention "
                f"projections in layer {layer_index}."
            )
        query_width = query * dimension
        key_value_width = key_value * dimension
        for role, shape in (
            ("attention.q_proj.weight", (query_width, hidden)),
            ("attention.k_proj.weight", (key_value_width, hidden)),
            ("attention.v_proj.weight", (key_value_width, hidden)),
            ("attention.o_proj.weight", (hidden, query_width)),
        ):
            _require_dense_operator_parameter(
                attention,
                role=role,
                shape=shape,
                layer_index=layer_index,
            )
        attention_geometries.add((query, key_value, dimension))
        mlps = tuple(
            node
            for kind in (CircuitKind.GATED_MLP, CircuitKind.DENSE_MLP)
            for node in graph.nodes_of_kind(kind)
            if node.layer_index == layer_index
        )
        if len(mlps) != 1:
            raise DynamicParameterFrontierError(
                "Dense causal source requires exactly one dense MLP in each "
                f"layer; layer {layer_index} has {len(mlps)}."
            )
        mlp = mlps[0]
        if mlp.kind != CircuitKind.GATED_MLP.value:
            raise DynamicParameterFrontierError(
                "Dense Llama translation requires one gated SwiGLU-style MLP "
                f"in layer {layer_index}."
            )
        layer_intermediate = _positive_int(
            _attributes(mlp).get("intermediate_size"),
            f"layer {layer_index} intermediate_size",
        )
        for role, shape in (
            ("mlp.gate_proj.weight", (layer_intermediate, hidden)),
            ("mlp.up_proj.weight", (layer_intermediate, hidden)),
            ("mlp.down_proj.weight", (hidden, layer_intermediate)),
        ):
            _require_dense_operator_parameter(
                mlp,
                role=role,
                shape=shape,
                layer_index=layer_index,
            )
        intermediate_sizes.add(layer_intermediate)
        layer_norms = tuple(
            node
            for node in graph.nodes_of_kind(CircuitKind.NORM)
            if node.layer_index == layer_index
        )
        if len(layer_norms) < 2:
            raise DynamicParameterFrontierError(
                "Dense causal source requires pre-attention and pre-MLP "
                f"normalization in layer {layer_index}."
            )
    if len(attention_geometries) != 1 or len(intermediate_sizes) != 1:
        raise DynamicParameterFrontierError(
            "The first generic Llama catalog seam requires uniform source "
            "attention and MLP geometry."
        )
    query, key_value, head_dim = next(iter(attention_geometries))
    intermediate = next(iter(intermediate_sizes))

    final_norm_attributes = _attributes(final_norm)
    epsilon_value = final_norm_attributes.get("epsilon", 1e-5)
    if (
        isinstance(epsilon_value, bool)
        or not isinstance(epsilon_value, (int, float))
        or not math.isfinite(float(epsilon_value))
        or float(epsilon_value) <= 0.0
    ):
        raise DynamicParameterFrontierError(
            "Dense causal source normalization epsilon must be finite and "
            "positive."
        )
    position_nodes = tuple(graph.nodes_of_kind(CircuitKind.POSITION_ENCODING))
    position_attributes = (
        _attributes(position_nodes[0]) if len(position_nodes) == 1 else {}
    )
    rope_theta_value = position_attributes.get("rope_theta", 10_000.0)
    if (
        isinstance(rope_theta_value, bool)
        or not isinstance(rope_theta_value, (int, float))
        or not math.isfinite(float(rope_theta_value))
        or float(rope_theta_value) <= 0.0
    ):
        rope_theta_value = 10_000.0
    max_positions_value = position_attributes.get("max_positions", 2048)
    if (
        isinstance(max_positions_value, bool)
        or not isinstance(max_positions_value, int)
        or max_positions_value <= 0
    ):
        max_positions_value = 2048
    return _HfDenseCausalSourceGeometry(
        family=graph.family,
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=intermediate,
        layer_count=len(layer_indices),
        query_heads=query,
        key_value_heads=key_value,
        head_dim=head_dim,
        tied_token_interface=observed_tie,
        rms_norm_eps=float(epsilon_value),
        rope_theta=float(rope_theta_value),
        max_position_embeddings=max_positions_value,
    )


def _hf_llama_target_semantic_capacities(
    topology: HfLlamaDenseTopology,
) -> dict[str, int]:
    attention_half = (
        topology.hidden_size
        * topology.head_dim
        * (topology.num_attention_heads + topology.num_key_value_heads)
    )
    capacities = {
        SEMANTIC_INTERFACE: topology.token_interface_parameters,
        SEMANTIC_NORMALIZATION: topology.normalization_parameters,
        SEMANTIC_STATE_MIXER_INTERNAL: 0,
        SEMANTIC_ATTENTION_QUERY_KEY: (
            topology.num_hidden_layers * attention_half
        ),
        SEMANTIC_ATTENTION_WRITER: (
            topology.num_hidden_layers * attention_half
        ),
        SEMANTIC_MLP_PRODUCER: (
            topology.num_hidden_layers
            * 2
            * topology.hidden_size
            * topology.intermediate_size
        ),
        SEMANTIC_MLP_WRITER: (
            topology.num_hidden_layers
            * topology.hidden_size
            * topology.intermediate_size
        ),
        SEMANTIC_OTHER: 0,
    }
    if sum(capacities.values()) != topology.logical_parameters:
        raise DynamicParameterFrontierError(
            "Dense Llama semantic capacities do not reproduce its exact "
            "parameter count."
        )
    return capacities


def _hf_llama_required_capabilities(
    source: _HfDenseCausalSourceGeometry,
    topology: HfLlamaDenseTopology,
) -> tuple[str, ...]:
    required = {CAP_DENSE_BLANK_MATERIALIZATION}
    if source.family != "llama":
        required.add(CAP_CAUSAL_DECODER_FAMILY_TRANSLATION)
    if topology.hidden_size != source.hidden_size:
        required.add(CAP_RESIDUAL_WIDTH_PROJECTION)
    if topology.intermediate_size != source.intermediate_size:
        required.add(CAP_FFN_WIDTH_PROJECTION)
    if (
        topology.num_attention_heads != source.query_heads
        or topology.num_key_value_heads != source.key_value_heads
        or topology.head_dim != source.head_dim
    ):
        required.add(CAP_ATTENTION_TOPOLOGY_PROJECTION)
    if topology.num_hidden_layers != source.layer_count:
        required.add(CAP_ORDERED_DEPTH_COMPOSITION)
    if topology.vocab_size != source.vocab_size:
        required.add(CAP_TOKENIZER_BRIDGE)
    return tuple(sorted(required))


def _hf_llama_pareto_profile(
    source: _HfDenseCausalSourceGeometry,
    topology: HfLlamaDenseTopology,
    *,
    requested_unique_parameters: int,
    compiler_executable: bool,
) -> tuple[tuple[str, int], ...]:
    gqa_target = topology.num_attention_heads
    gqa_target_denominator = topology.num_key_value_heads
    gqa_source = source.query_heads
    gqa_source_denominator = source.key_value_heads
    gqa_cross_a = gqa_target * gqa_source_denominator
    gqa_cross_b = gqa_source * gqa_target_denominator
    budget_min = min(topology.unique_parameters, requested_unique_parameters)
    budget_max = max(topology.unique_parameters, requested_unique_parameters)
    vocabulary_pressure = _bounded_ratio_ppm(
        topology.token_interface_parameters,
        requested_unique_parameters,
    )
    return _retention_profile(
        {
            "budget_utilization": _bounded_ratio_ppm(
                budget_min,
                budget_max,
            ),
            "residual_width_retention": _bounded_ratio_ppm(
                topology.hidden_size,
                source.hidden_size,
            ),
            "depth_retention": _bounded_ratio_ppm(
                topology.num_hidden_layers,
                source.layer_count,
            ),
            "attention_role_retention": _bounded_ratio_ppm(
                topology.num_hidden_layers,
                source.layer_count,
            ),
            "query_channel_retention": _bounded_ratio_ppm(
                topology.num_attention_heads * topology.head_dim,
                source.query_heads * source.head_dim,
            ),
            "head_dim_retention": _bounded_ratio_ppm(
                topology.head_dim,
                source.head_dim,
            ),
            "gqa_ratio_retention": _bounded_ratio_ppm(
                min(gqa_cross_a, gqa_cross_b),
                max(gqa_cross_a, gqa_cross_b),
            ),
            "ffn_width_retention": _bounded_ratio_ppm(
                topology.intermediate_size,
                source.intermediate_size,
            ),
            "vocabulary_budget_headroom": 1_000_000 - vocabulary_pressure,
            "compiler_executable": 1_000_000 if compiler_executable else 0,
        }
    )


def _hf_llama_topology_rank(
    topology: HfLlamaDenseTopology,
    *,
    requested_unique_parameters: int,
    match_policy: str,
    retention_score_ppm: int,
) -> tuple[Any, ...]:
    delta = topology.unique_parameters - requested_unique_parameters
    if match_policy == "exact":
        distance = 0
        side = 0
    elif match_policy == "at_or_below":
        distance = -delta
        side = 0
    else:
        distance = abs(delta)
        side = 0 if delta <= 0 else 1
    return (
        distance,
        side,
        -retention_score_ppm,
        -topology.unique_parameters,
        topology.hidden_size,
        topology.intermediate_size,
        topology.num_hidden_layers,
        topology.num_attention_heads,
        topology.num_key_value_heads,
        topology.head_dim,
    )


@dataclass(frozen=True, slots=True)
class _HfLlamaEnumeratedTopology:
    topology: HfLlamaDenseTopology
    spans: tuple[tuple[int, ...], ...]
    required_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    compiler_executable: bool
    retention_profile: tuple[tuple[str, int], ...]
    retention_score_ppm: int

    @property
    def key(self) -> tuple[int, ...]:
        value = self.topology
        return (
            value.vocab_size,
            value.hidden_size,
            value.intermediate_size,
            value.num_hidden_layers,
            value.num_attention_heads,
            value.num_key_value_heads,
            value.head_dim,
        )


@dataclass(frozen=True, slots=True)
class HfLlamaDenseTargetTopologyCatalog:
    """Dynamic dense topology catalog for blank Hugging Face Llama children."""

    catalog_id: str = HF_LLAMA_CATALOG_ID
    version: int = 1
    priority: int = 100
    target_families: tuple[str, ...] = ("llama",)
    target_modalities: tuple[str, ...] = ("text",)
    minimum_hidden_size: int = 32
    hidden_size_multiple: int = 8
    minimum_intermediate_size: int = 32
    intermediate_size_multiple: int = 8
    minimum_head_dim: int = 8
    head_dim_multiple: int = 8
    compiler_capabilities: tuple[str, ...] = ()
    tokenizer_bridge_artifact: StrictTokenizerBridgeArtifact | None = None
    component_protection_rules: tuple[ComponentProtectionRule, ...] = ()
    component_bottleneck_risk_weight: float = 8.0
    tie_word_embeddings: bool = False
    parameter_alias_policy: str = HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE

    def __post_init__(self) -> None:
        for name in (
            "minimum_hidden_size",
            "hidden_size_multiple",
            "minimum_intermediate_size",
            "intermediate_size_multiple",
            "minimum_head_dim",
            "head_dim_multiple",
        ):
            _positive_int(getattr(self, name), name)
        if self.minimum_hidden_size % self.hidden_size_multiple:
            raise DynamicParameterFrontierError(
                "minimum_hidden_size must be a hidden_size_multiple."
            )
        if self.minimum_intermediate_size % self.intermediate_size_multiple:
            raise DynamicParameterFrontierError(
                "minimum_intermediate_size must be an "
                "intermediate_size_multiple."
            )
        if (
            self.minimum_head_dim % self.head_dim_multiple
            or self.minimum_head_dim % 2
            or self.head_dim_multiple % 2
        ):
            raise DynamicParameterFrontierError(
                "Dense Llama head dimensions must use even aligned values."
            )
        if self.tie_word_embeddings is not False:
            raise DynamicParameterFrontierError(
                "The Hugging Face/Llama catalog rejects tied target "
                "embeddings."
            )
        alias_policy = _text(
            self.parameter_alias_policy,
            "parameter_alias_policy",
        ).lower()
        if alias_policy != HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE:
            raise DynamicParameterFrontierError(
                "The Hugging Face/Llama catalog rejects shared or aliased "
                "target parameters."
            )
        object.__setattr__(self, "parameter_alias_policy", alias_policy)
        capabilities = _capabilities(
            self.compiler_capabilities,
            "Hugging Face/Llama compiler capability",
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
        rules = tuple(self.component_protection_rules)
        if any(not isinstance(value, ComponentProtectionRule) for value in rules):
            raise DynamicParameterFrontierError(
                "component_protection_rules must contain "
                "ComponentProtectionRule objects."
            )
        if len({value.rule_id for value in rules}) != len(rules):
            raise DynamicParameterFrontierError(
                "component_protection_rules IDs must be unique."
            )
        object.__setattr__(self, "component_protection_rules", rules)
        bottleneck_weight = self.component_bottleneck_risk_weight
        if (
            isinstance(bottleneck_weight, bool)
            or not isinstance(bottleneck_weight, (int, float))
            or not math.isfinite(float(bottleneck_weight))
            or bottleneck_weight < 0
        ):
            raise DynamicParameterFrontierError(
                "component_bottleneck_risk_weight must be finite and "
                "non-negative."
            )
        object.__setattr__(
            self,
            "component_bottleneck_risk_weight",
            float(bottleneck_weight),
        )

    def supports(
        self,
        parent_graph: ModelCircuitGraph,
        *,
        target_family: str,
        target_modalities: tuple[str, ...],
    ) -> bool:
        return (
            isinstance(parent_graph, ModelCircuitGraph)
            and target_family == "llama"
            and target_modalities == ("text",)
            and bool(parent_graph.layer_indices)
            and {value.modality for value in parent_graph.interfaces} == {"text"}
        )

    def _candidate_intermediate_sizes(
        self,
        *,
        requested_unique_parameters: int,
        base_parameters: int,
        coefficient: int,
        maximum_intermediate_size: int,
        match_policy: str,
    ) -> tuple[int, ...]:
        remaining = requested_unique_parameters - base_parameters
        floor_value = remaining // coefficient
        floor_value -= floor_value % self.intermediate_size_multiple
        maximum_aligned = maximum_intermediate_size - (
            maximum_intermediate_size % self.intermediate_size_multiple
        )
        if maximum_aligned < self.minimum_intermediate_size:
            return ()
        values: set[int] = set()
        if match_policy == "exact":
            if (
                remaining > 0
                and remaining % coefficient == 0
                and floor_value * coefficient == remaining
            ):
                values.add(floor_value)
        elif match_policy == "at_or_below":
            if floor_value >= self.minimum_intermediate_size:
                values.add(min(floor_value, maximum_aligned))
        else:
            for value in (
                floor_value,
                floor_value + self.intermediate_size_multiple,
            ):
                values.add(
                    min(
                        max(value, self.minimum_intermediate_size),
                        maximum_aligned,
                    )
                )
        return tuple(
            sorted(
                value
                for value in values
                if self.minimum_intermediate_size
                <= value
                <= maximum_aligned
                and value % self.intermediate_size_multiple == 0
            )
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
        source = _inspect_hf_dense_causal_source(parent_graph)
        requested = _positive_int(
            requested_unique_parameters,
            "requested_unique_parameters",
        )
        maximum = _positive_int(maximum_candidates, "maximum_candidates")
        if match_policy not in _POLICIES:
            raise DynamicParameterFrontierError(
                "match_policy must be exact, at_or_below, or nearest."
            )
        target_vocab = (
            source.vocab_size
            if target_vocabulary_size is None
            else _positive_int(target_vocabulary_size, "target_vocabulary_size")
        )
        tokenizer_bridge_configuration: dict[str, str] = {}
        if target_vocab != source.vocab_size:
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
                    or profile.donor_row_capacity != source.vocab_size
                    or profile.target_row_capacity != target_vocab
                ):
                    raise DynamicParameterFrontierError(
                        "Tokenizer bridge artifact does not match the inspected "
                        "parent semantics or requested vocabulary capacities."
                    )
                tokenizer_bridge_configuration = (
                    strict_tokenizer_bridge_configuration(bridge)
                )
        scorer = _DenseComponentPolicyScorer(
            parent_graph,
            protection_rules=self.component_protection_rules,
            bottleneck_risk_weight=self.component_bottleneck_risk_weight,
        )
        scalar_winners: dict[
            str,
            tuple[int, tuple[Any, ...], _HfLlamaEnumeratedTopology],
        ] = {}
        legal_topologies = 0
        policy_eligible_topologies = 0
        allocation_eligible_topologies = 0
        for hidden in range(
            self.minimum_hidden_size,
            source.hidden_size + 1,
            self.hidden_size_multiple,
        ):
            for head_dim in range(
                self.minimum_head_dim,
                min(source.head_dim, hidden) + 1,
                self.head_dim_multiple,
            ):
                if hidden % head_dim:
                    continue
                query_heads = hidden // head_dim
                if query_heads > source.query_heads:
                    continue
                for key_value_heads in range(
                    1,
                    min(source.key_value_heads, query_heads) + 1,
                ):
                    if query_heads % key_value_heads:
                        continue
                    for layer_count in range(1, source.layer_count + 1):
                        base = (
                            2 * target_vocab * hidden
                            + (2 * layer_count + 1) * hidden
                            + layer_count
                            * 2
                            * hidden
                            * head_dim
                            * (query_heads + key_value_heads)
                        )
                        coefficient = 3 * layer_count * hidden
                        for intermediate in self._candidate_intermediate_sizes(
                            requested_unique_parameters=requested,
                            base_parameters=base,
                            coefficient=coefficient,
                            maximum_intermediate_size=source.intermediate_size,
                            match_policy=match_policy,
                        ):
                            topology = HfLlamaDenseTopology(
                                vocab_size=target_vocab,
                                hidden_size=hidden,
                                intermediate_size=intermediate,
                                num_hidden_layers=layer_count,
                                num_attention_heads=query_heads,
                                num_key_value_heads=key_value_heads,
                                head_dim=head_dim,
                            )
                            legal_topologies += 1
                            if topology.unique_parameters >= (
                                parent_graph.unique_parameter_count
                            ):
                                continue
                            delta = topology.unique_parameters - requested
                            if (
                                match_policy == "exact"
                                and delta != 0
                                or match_policy == "at_or_below"
                                and delta > 0
                            ):
                                continue
                            policy_eligible_topologies += 1
                            target_capacities = (
                                _hf_llama_target_semantic_capacities(topology)
                            )
                            try:
                                allocation_score = scorer.score(
                                    target_capacities
                                )
                            except DynamicParameterFrontierError:
                                continue
                            if allocation_score is None:
                                continue
                            allocation_eligible_topologies += 1
                            required = _hf_llama_required_capabilities(
                                source,
                                topology,
                            )
                            missing = tuple(
                                sorted(
                                    set(required).difference(
                                        self.compiler_capabilities
                                    )
                                )
                            )
                            executable = not missing
                            profile = _hf_llama_pareto_profile(
                                source,
                                topology,
                                requested_unique_parameters=requested,
                                compiler_executable=executable,
                            )
                            retention = sum(
                                value for _axis, value in profile
                            ) // len(profile)
                            enumerated = _HfLlamaEnumeratedTopology(
                                topology=topology,
                                spans=_balanced_spans(
                                    source.layer_count,
                                    layer_count,
                                ),
                                required_capabilities=required,
                                missing_capabilities=missing,
                                compiler_executable=executable,
                                retention_profile=profile,
                                retention_score_ppm=retention,
                            )
                            rank = _hf_llama_topology_rank(
                                topology,
                                requested_unique_parameters=requested,
                                match_policy=match_policy,
                                retention_score_ppm=retention,
                            )
                            profile_values = dict(profile)
                            for profile_name, weights in _PARETO_SCALARIZATIONS:
                                scalar_score = sum(
                                    weight * profile_values[axis]
                                    for axis, weight in weights
                                )
                                current = scalar_winners.get(profile_name)
                                if current is None or (
                                    -scalar_score,
                                    rank,
                                    enumerated.key,
                                ) < (
                                    -current[0],
                                    current[1],
                                    current[2].key,
                                ):
                                    scalar_winners[profile_name] = (
                                        scalar_score,
                                        rank,
                                        enumerated,
                                    )

        support_by_topology: dict[tuple[int, ...], set[str]] = {}
        enumerated_by_topology: dict[
            tuple[int, ...],
            _HfLlamaEnumeratedTopology,
        ] = {}
        for profile_name, (_score, _rank, enumerated) in scalar_winners.items():
            support_by_topology.setdefault(enumerated.key, set()).add(
                profile_name
            )
            enumerated_by_topology[enumerated.key] = enumerated
        ranked = sorted(
            enumerated_by_topology.values(),
            key=lambda value: _hf_llama_topology_rank(
                value.topology,
                requested_unique_parameters=requested,
                match_policy=match_policy,
                retention_score_ppm=value.retention_score_ppm,
            ),
        )[:maximum]
        candidates: list[DenseTargetTopologyCandidate] = []
        for enumerated in ranked:
            topology = enumerated.topology
            span_string = ";".join(
                ",".join(str(index) for index in span)
                for span in enumerated.spans
            )
            configuration = {
                "model_type": "llama",
                "vocab_size": topology.vocab_size,
                "hidden_size": topology.hidden_size,
                "intermediate_size": topology.intermediate_size,
                "num_hidden_layers": topology.num_hidden_layers,
                "num_attention_heads": topology.num_attention_heads,
                "num_key_value_heads": topology.num_key_value_heads,
                "head_dim": topology.head_dim,
                "hidden_act": "silu",
                "tie_word_embeddings": False,
                "attention_bias": False,
                "mlp_bias": False,
                "attention_dropout": 0.0,
                "rms_norm_eps": source.rms_norm_eps,
                "rope_theta": source.rope_theta,
                "max_position_embeddings": source.max_position_embeddings,
                "use_cache": True,
                DENSE_TOPOLOGY_CATALOG_CONFIG_KEY: self.catalog_id,
                HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY: HF_LLAMA_TARGET_SCHEMA,
                HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY: (
                    HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE
                ),
                HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY: span_string,
                HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY: source.family,
                HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY: (
                    source.layer_count
                ),
                **tokenizer_bridge_configuration,
            }
            candidate = DenseTargetTopologyCandidate(
                candidate_id=(
                    "hf-llama:"
                    f"h{topology.hidden_size}:"
                    f"i{topology.intermediate_size}:"
                    f"l{topology.num_hidden_layers}:"
                    f"q{topology.num_attention_heads}:"
                    f"kv{topology.num_key_value_heads}:"
                    f"d{topology.head_dim}:v{topology.vocab_size}"
                ),
                catalog_id=self.catalog_id,
                family="llama",
                modalities=("text",),
                requested_unique_parameters=requested,
                source_unique_parameters=parent_graph.unique_parameter_count,
                unique_parameters=topology.unique_parameters,
                logical_parameters=topology.logical_parameters,
                native_tied_parameter_count=0,
                retention_score_ppm=enumerated.retention_score_ppm,
                configuration=_configuration(configuration),
                ordered_source_spans=enumerated.spans,
                native_ties=(),
                source_vocabulary_size=source.vocab_size,
                target_vocabulary_size=topology.vocab_size,
                vocabulary_unique_parameters=(
                    topology.token_interface_parameters
                ),
                tokenizer_bridge_required=(
                    topology.vocab_size != source.vocab_size
                ),
                retention_profile_ppm=enumerated.retention_profile,
                required_compiler_capabilities=(
                    enumerated.required_capabilities
                ),
                declared_compiler_capabilities=self.compiler_capabilities,
                missing_compiler_capabilities=(
                    enumerated.missing_capabilities
                ),
                compiler_executable=enumerated.compiler_executable,
                pareto_support_profiles=tuple(
                    sorted(support_by_topology[enumerated.key])
                ),
                pareto_dominance_verified=True,
            )
            candidates.append(
                _bind_dense_component_allocation_receipt(
                    parent_graph,
                    candidate,
                    target_by_semantic=(
                        _hf_llama_target_semantic_capacities(topology)
                    ),
                    protection_rules=self.component_protection_rules,
                    bottleneck_risk_weight=(
                        self.component_bottleneck_risk_weight
                    ),
                )
            )
        return CatalogFrontier(
            candidates=tuple(candidates),
            diagnostics={
                "catalog_id": self.catalog_id,
                "catalog_version": self.version,
                "source_family": source.family,
                "source_graph_fingerprint": parent_graph.fingerprint,
                "requested_unique_parameters": requested,
                "target_vocabulary_size": target_vocab,
                "match_policy": match_policy,
                "legal_topologies_considered": legal_topologies,
                "policy_eligible_topologies": policy_eligible_topologies,
                "allocation_eligible_topologies": (
                    allocation_eligible_topologies
                ),
                "supported_pareto_topologies": len(
                    enumerated_by_topology
                ),
                "returned_candidates": len(candidates),
                "target_token_interface_tied": False,
                "target_parameter_alias_policy": (
                    HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE
                ),
                "tokenizer_bridge_required": (
                    target_vocab != source.vocab_size
                ),
                "tokenizer_mapping_implemented": bool(
                    tokenizer_bridge_configuration
                ),
                "dense_parameter_count_reduction_only": True,
                "quantization_used": False,
                "pruning_used": False,
                "sparse_storage_used": False,
                "parameter_sharing_used": False,
                "behavioral_retention_claimed": False,
            },
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
    component_protection_rules: tuple[ComponentProtectionRule, ...] = ()
    component_bottleneck_risk_weight: float = 8.0
    target_token_interface_mode: str = (
        LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
    )
    heterogeneous_layout_selection_input: (
        Lfm2HeterogeneousLayoutSelectionInput | None
    ) = None

    def __post_init__(self) -> None:
        _positive_int(self.minimum_hidden_size, "minimum_hidden_size")
        _positive_int(self.hidden_size_multiple, "hidden_size_multiple")
        _positive_int(self.minimum_head_dim, "minimum_head_dim")
        token_interface_mode = _text(
            self.target_token_interface_mode,
            "target_token_interface_mode",
        ).lower()
        if token_interface_mode not in _LFM2_DENSE_TOKEN_INTERFACE_MODES:
            raise DynamicParameterFrontierError(
                "target_token_interface_mode must be preserve or untied."
            )
        object.__setattr__(
            self,
            "target_token_interface_mode",
            token_interface_mode,
        )
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
        rules = tuple(self.component_protection_rules)
        if any(not isinstance(value, ComponentProtectionRule) for value in rules):
            raise DynamicParameterFrontierError(
                "component_protection_rules must contain "
                "ComponentProtectionRule objects."
            )
        if len({value.rule_id for value in rules}) != len(rules):
            raise DynamicParameterFrontierError(
                "component_protection_rules IDs must be unique."
            )
        object.__setattr__(self, "component_protection_rules", rules)
        bottleneck_weight = self.component_bottleneck_risk_weight
        if (
            isinstance(bottleneck_weight, bool)
            or not isinstance(bottleneck_weight, (int, float))
            or not math.isfinite(float(bottleneck_weight))
            or bottleneck_weight < 0
        ):
            raise DynamicParameterFrontierError(
                "component_bottleneck_risk_weight must be finite and "
                "non-negative."
            )
        object.__setattr__(
            self,
            "component_bottleneck_risk_weight",
            float(bottleneck_weight),
        )
        heterogeneous_input = self.heterogeneous_layout_selection_input
        if heterogeneous_input is not None:
            if not isinstance(
                heterogeneous_input,
                Lfm2HeterogeneousLayoutSelectionInput,
            ):
                raise DynamicParameterFrontierError(
                    "heterogeneous_layout_selection_input must be an "
                    "authenticated Lfm2HeterogeneousLayoutSelectionInput."
                )
            if (
                self.target_token_interface_mode
                != LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
            ):
                raise DynamicParameterFrontierError(
                    "Heterogeneous LFM2 physical layouts require the untied "
                    "target token-interface mode."
                )
            object.__setattr__(
                self,
                "compiler_capabilities",
                tuple(
                    sorted(
                        {
                            *self.compiler_capabilities,
                            CAP_HETEROGENEOUS_LAYER_LAYOUT,
                        }
                    )
                ),
            )

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

    def _enumerate_heterogeneous_frontier(
        self,
        parent_graph: ModelCircuitGraph,
        geometry: _Lfm2Geometry,
        *,
        requested: int,
        target_vocab: int,
        match_policy: str,
        tokenizer_bridge_configuration: Mapping[str, str],
    ) -> CatalogFrontier:
        selection_input = self.heterogeneous_layout_selection_input
        if selection_input is None:  # pragma: no cover - private call contract
            raise DynamicParameterFrontierError(
                "Heterogeneous frontier requested without selection input."
            )
        pins = selection_input.target_pins
        if pins.vocab_size != target_vocab:
            raise DynamicParameterFrontierError(
                "Heterogeneous target pins do not match the requested target "
                "vocabulary size."
            )
        if tokenizer_bridge_configuration:
            expected_bridge = tokenizer_bridge_configuration[
                TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
            ]
            if pins.token_bridge_evidence_sha256 != expected_bridge:
                raise DynamicParameterFrontierError(
                    "Heterogeneous target pins do not bind the verified strict "
                    "tokenizer bridge artifact."
                )
        try:
            selection = selection_input.select(
                parent_graph,
                exact_target_unique_parameters=requested,
            )
        except Lfm2HeterogeneousLayoutUnattainable as exc:
            raise DynamicParameterFrontierUnattainable(
                "No supplied heterogeneous LFM2 physical layout exactly "
                "matches the requested dense parameter count.",
                {
                    "catalog_id": self.catalog_id,
                    "source_graph_fingerprint": parent_graph.fingerprint,
                    "requested_unique_parameters": requested,
                    "match_policy": match_policy,
                    "heterogeneous_selection_input_sha256": (
                        selection_input.fingerprint
                    ),
                    "exact_budget_representable": False,
                    "reason": "heterogeneous_exact_selection_unattainable",
                },
            ) from exc
        except Lfm2HeterogeneousLayoutSelectionError as exc:
            raise DynamicParameterFrontierError(
                "Heterogeneous LFM2 physical selection failed authentication."
            ) from exc

        descriptor = selection.descriptor
        selection_receipt = selection.receipt
        if (
            descriptor.source_topology_sha256 != parent_graph.fingerprint
            or descriptor.vocab_size != target_vocab
            or descriptor.layer_count != len(geometry.layer_types)
            or tuple(layer.layer_type for layer in descriptor.layers)
            != geometry.layer_types
            or descriptor.accounting.unique_parameter_elements != requested
            or descriptor.accounting.logical_parameter_elements != requested
        ):
            raise DynamicParameterFrontierError(
                "Selected heterogeneous descriptor differs from the requested "
                "whole-depth source/target/count contract."
            )
        attention_layers = tuple(
            layer
            for layer in descriptor.layers
            if layer.layer_type == "full_attention"
        )
        if attention_layers:
            scaffold_attention = attention_layers[0]
            assert scaffold_attention.query_heads is not None
            assert scaffold_attention.key_value_heads is not None
            assert scaffold_attention.head_dim is not None
            scaffold_query = scaffold_attention.query_heads
            scaffold_key_value = scaffold_attention.key_value_heads
            scaffold_head_dim = scaffold_attention.head_dim
            minimum_query_channels = min(
                int(layer.query_heads) * int(layer.head_dim)
                for layer in attention_layers
            )
            minimum_head_dim = min(
                int(layer.head_dim) for layer in attention_layers
            )
            gqa_ratios = tuple(
                (
                    int(layer.query_heads),
                    int(layer.key_value_heads),
                )
                for layer in attention_layers
            )
            source_ratio_numerator = geometry.query_heads
            source_ratio_denominator = geometry.key_value_heads
            gqa_retention = min(
                _bounded_ratio_ppm(
                    min(
                        query * source_ratio_denominator,
                        source_ratio_numerator * key_value,
                    ),
                    max(
                        query * source_ratio_denominator,
                        source_ratio_numerator * key_value,
                    ),
                )
                for query, key_value in gqa_ratios
            )
        else:
            scaffold_query = 1
            scaffold_key_value = 1
            scaffold_head_dim = 2
            minimum_query_channels = geometry.query_heads * geometry.head_dim
            minimum_head_dim = geometry.head_dim
            gqa_retention = 1_000_000
        layer_types = tuple(layer.layer_type for layer in descriptor.layers)
        layer_string = ",".join(layer_types)
        spans = tuple((index,) for index in range(descriptor.layer_count))
        span_string = ";".join(str(index) for index in range(descriptor.layer_count))
        descriptor_json = descriptor.serialize().decode("ascii")
        selection_json = selection_receipt.serialize().decode("ascii")
        scaffold_intermediate = descriptor.layers[0].mlp_intermediate_size
        configuration_values: dict[str, JsonScalar] = {
            "model_type": "lfm2",
            "vocab_size": descriptor.vocab_size,
            "hidden_size": descriptor.hidden_size,
            "intermediate_size": scaffold_intermediate,
            "num_hidden_layers": descriptor.layer_count,
            "num_attention_heads": scaffold_query,
            "num_key_value_heads": scaffold_key_value,
            "head_dim": scaffold_head_dim,
            "layer_types": layer_string,
            "conv_L_cache": descriptor.conv_kernel_size,
            "block_auto_adjust_ff_dim": False,
            "tie_word_embeddings": False,
            "_neural_forge_ordered_source_spans": span_string,
            "_neural_forge_target_layer_types": layer_string,
            "_neural_forge_dense_topology_catalog": self.catalog_id,
            "_neural_forge_tokenizer_bridge_required": (
                target_vocab != geometry.vocab_size
            ),
            LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY: (
                LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
            ),
            LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY: descriptor_json,
            LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY: (
                descriptor.descriptor_sha256
            ),
            LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY: (
                selection_json
            ),
            LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY: (
                selection_receipt.receipt_sha256
            ),
            **tokenizer_bridge_configuration,
        }
        core_configuration = _configuration(configuration_values)
        topology_sha256 = component_allocation_topology_fingerprint(
            core_configuration
        )
        topology_option_id = (
            "heterogeneous-layout." + descriptor.descriptor_sha256
        )
        allocation_receipt = DenseComponentAllocationReceipt(
            policy=COMPONENT_ALLOCATION_POLICY,
            topology_option_id=topology_option_id,
            topology_configuration_sha256=topology_sha256,
            source_graph_fingerprint=parent_graph.fingerprint,
            source_unique_parameters=geometry.source_unique_parameters,
            source_logical_parameters=geometry.source_logical_parameters,
            target_unique_parameters=requested,
            target_logical_parameters=requested,
            allocation_selection=selection.budget_selection,
            physical_layout_kind=LFM2_HETEROGENEOUS_LAYOUT_KIND,
            physical_layout_descriptor_sha256=descriptor.descriptor_sha256,
            physical_layout_selection_receipt_sha256=(
                selection_receipt.receipt_sha256
            ),
            physical_layout_source_topology_sha256=parent_graph.fingerprint,
        )
        configuration = _configuration(
            {
                **dict(core_configuration),
                COMPONENT_ALLOCATION_POLICY_CONFIG_KEY: (
                    allocation_receipt.policy
                ),
                COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY: (
                    allocation_receipt.fingerprint
                ),
                COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY: topology_sha256,
            }
        )

        required_capabilities = {
            CAP_DENSE_BLANK_MATERIALIZATION,
            CAP_HETEROGENEOUS_LAYER_LAYOUT,
        }
        if descriptor.hidden_size != geometry.hidden_size:
            required_capabilities.add(CAP_RESIDUAL_WIDTH_PROJECTION)
        if any(
            layer.mlp_intermediate_size != geometry.intermediate_size
            for layer in descriptor.layers
        ):
            required_capabilities.add(CAP_FFN_WIDTH_PROJECTION)
        if any(
            int(layer.query_heads) != geometry.query_heads
            or int(layer.key_value_heads) != geometry.key_value_heads
            or int(layer.head_dim) != geometry.head_dim
            for layer in attention_layers
        ):
            required_capabilities.add(CAP_ATTENTION_TOPOLOGY_PROJECTION)
        if target_vocab != geometry.vocab_size:
            required_capabilities.add(CAP_TOKENIZER_BRIDGE)
        required = tuple(sorted(required_capabilities))
        missing = tuple(
            sorted(set(required).difference(self.compiler_capabilities))
        )
        compiler_executable = not missing
        vocabulary_parameters = 2 * target_vocab * descriptor.hidden_size
        vocabulary_pressure = _bounded_ratio_ppm(
            vocabulary_parameters,
            requested,
        )
        average_intermediate = sum(
            layer.mlp_intermediate_size for layer in descriptor.layers
        ) // descriptor.layer_count
        retention_profile = _retention_profile(
            {
                "budget_utilization": 1_000_000,
                "residual_width_retention": _bounded_ratio_ppm(
                    descriptor.hidden_size,
                    geometry.hidden_size,
                ),
                "depth_retention": 1_000_000,
                "attention_role_retention": 1_000_000,
                "query_channel_retention": _bounded_ratio_ppm(
                    minimum_query_channels,
                    geometry.query_heads * geometry.head_dim,
                ),
                "head_dim_retention": _bounded_ratio_ppm(
                    minimum_head_dim,
                    geometry.head_dim,
                ),
                "gqa_ratio_retention": gqa_retention,
                "ffn_width_retention": _bounded_ratio_ppm(
                    average_intermediate,
                    geometry.intermediate_size,
                ),
                "vocabulary_budget_headroom": 1_000_000
                - vocabulary_pressure,
                "compiler_executable": (
                    1_000_000 if compiler_executable else 0
                ),
            }
        )
        structural_terms = tuple(
            value
            for key, value in retention_profile
            if key not in {"budget_utilization", "compiler_executable"}
        )
        retention_score = sum(structural_terms) // len(structural_terms)
        identity = canonical_sha256(
            {
                "catalog_id": self.catalog_id,
                "configuration": dict(configuration),
                "unique_parameters": requested,
                "logical_parameters": requested,
            }
        )
        candidate = DenseTargetTopologyCandidate(
            candidate_id=f"lfm2-dense-{identity[:16]}",
            catalog_id=self.catalog_id,
            family="lfm2",
            modalities=("text",),
            requested_unique_parameters=requested,
            source_unique_parameters=geometry.source_unique_parameters,
            unique_parameters=requested,
            logical_parameters=requested,
            native_tied_parameter_count=0,
            retention_score_ppm=retention_score,
            configuration=configuration,
            ordered_source_spans=spans,
            source_vocabulary_size=geometry.vocab_size,
            target_vocabulary_size=target_vocab,
            vocabulary_unique_parameters=vocabulary_parameters,
            tokenizer_bridge_required=target_vocab != geometry.vocab_size,
            retention_profile_ppm=retention_profile,
            required_compiler_capabilities=required,
            declared_compiler_capabilities=self.compiler_capabilities,
            missing_compiler_capabilities=missing,
            compiler_executable=compiler_executable,
            pareto_support_profiles=("heterogeneous_exact_component_risk",),
            pareto_dominance_verified=True,
            component_allocation_receipt=allocation_receipt,
        )
        _lfm2_target_semantic_capacities(geometry, candidate)
        diagnostics = {
            "catalog_id": self.catalog_id,
            "catalog_version": self.version,
            "source_family": parent_graph.family,
            "source_graph_fingerprint": parent_graph.fingerprint,
            "source_unique_parameters": geometry.source_unique_parameters,
            "source_logical_parameters": geometry.source_logical_parameters,
            "requested_unique_parameters": requested,
            "target_vocabulary_size": target_vocab,
            "match_policy": match_policy,
            "returned_candidates": 1,
            "topologies_considered": len(selection_input.layer_options),
            "exact_budget_representable": True,
            "minimum_representable_unique_parameters": requested,
            "maximum_representable_unique_parameters": requested,
            "nearest_representable_below": requested,
            "nearest_representable_above": requested,
            "declared_compiler_capabilities": list(
                self.compiler_capabilities
            ),
            "executable_candidate_count": int(compiler_executable),
            "heterogeneous_layout_selection_used": True,
            "heterogeneous_selection_input_sha256": selection_input.fingerprint,
            "heterogeneous_descriptor_sha256": descriptor.descriptor_sha256,
            "heterogeneous_selection_receipt_sha256": (
                selection_receipt.receipt_sha256
            ),
            "component_allocation_heterogeneous_per_layer_widths_claimed": True,
            "whole_causal_depth_preserved": True,
            "independent_untied_vocab_interface": True,
            "dense_fp32_physical_layout": True,
            "factor_writer_used": False,
            "weights_or_downloads_used": False,
            "training_used": False,
            "quantization_pruning_packing_factorization_used": False,
            "non_native_weight_sharing_used": False,
            "behavioral_capability_retention_claimed": False,
        }
        return CatalogFrontier(candidates=(candidate,), diagnostics=diagnostics)

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
        target_tied_token_interface = (
            geometry.tied_token_interface
            if self.target_token_interface_mode
            == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
            else False
        )
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
        if self.heterogeneous_layout_selection_input is not None:
            return self._enumerate_heterogeneous_frontier(
                parent_graph,
                geometry,
                requested=requested,
                target_vocab=target_vocab,
                match_policy=match_policy,
                tokenizer_bridge_configuration=(
                    tokenizer_bridge_configuration
                ),
            )
        untied_rebalance_key: tuple[Any, ...] | None = None
        untied_rebalance_source: DenseTargetTopologyCandidate | None = None
        untied_rebalance_expected_unique: int | None = None
        untied_rebalance_spare: int | None = None
        untied_rebalance_profile: tuple[tuple[str, int], ...] | None = None
        untied_rebalance_dominated = False
        if (
            self.target_token_interface_mode
            == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
            and geometry.tied_token_interface
            and match_policy != "exact"
        ):
            preserve_frontier = replace(
                self,
                target_token_interface_mode=(
                    LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
                ),
            ).enumerate_frontier(
                parent_graph,
                requested_unique_parameters=requested,
                target_vocabulary_size=target_vocab,
                match_policy=match_policy,
                maximum_candidates=max(
                    maximum,
                    len(_PARETO_SCALARIZATIONS),
                ),
            )
            balanced_sources = tuple(
                candidate
                for candidate in preserve_frontier.candidates
                if "balanced" in candidate.pareto_support_profiles
            )
            executable_balanced_sources = tuple(
                candidate
                for candidate in balanced_sources
                if candidate.compiler_executable
            )
            if executable_balanced_sources:
                untied_rebalance_source = executable_balanced_sources[0]
            elif balanced_sources:
                untied_rebalance_source = balanced_sources[0]
            if untied_rebalance_source is not None:
                source_configuration = dict(
                    untied_rebalance_source.configuration
                )
                if source_configuration.get("tie_word_embeddings") is not True:
                    raise DynamicParameterFrontierError(
                        "Untied rebalance source is not a tied catalog target."
                    )
                target_intermediate, target_unique, spare = (
                    _lfm2_rebalance_tied_candidate_for_untied_output(
                        requested_unique_parameters=requested,
                        base_tied_unique_parameters=(
                            untied_rebalance_source.unique_parameters
                        ),
                        target_vocabulary_size=target_vocab,
                        hidden_size=int(source_configuration["hidden_size"]),
                        layer_count=int(
                            source_configuration["num_hidden_layers"]
                        ),
                        current_intermediate_size=int(
                            source_configuration["intermediate_size"]
                        ),
                    )
                )
                untied_rebalance_key = (
                    int(source_configuration["hidden_size"]),
                    target_intermediate,
                    int(source_configuration["num_attention_heads"]),
                    int(source_configuration["num_key_value_heads"]),
                    int(source_configuration["head_dim"]),
                    tuple(
                        str(source_configuration["layer_types"]).split(",")
                    ),
                    untied_rebalance_source.ordered_source_spans,
                )
                untied_rebalance_expected_unique = target_unique
                untied_rebalance_spare = spare
                rebalance_required_capabilities = (
                    _lfm2_required_compiler_capabilities(
                        geometry,
                        hidden_size=int(source_configuration["hidden_size"]),
                        intermediate_size=target_intermediate,
                        query_heads=int(
                            source_configuration["num_attention_heads"]
                        ),
                        key_value_heads=int(
                            source_configuration["num_key_value_heads"]
                        ),
                        head_dim=int(source_configuration["head_dim"]),
                        layer_types=untied_rebalance_key[5],
                        target_vocabulary_size=target_vocab,
                    )
                )
                untied_rebalance_profile = _lfm2_pareto_profile(
                    geometry,
                    requested_unique_parameters=requested,
                    unique_parameters=target_unique,
                    vocabulary_unique_parameters=(
                        2
                        * target_vocab
                        * int(source_configuration["hidden_size"])
                    ),
                    hidden_size=int(source_configuration["hidden_size"]),
                    intermediate_size=target_intermediate,
                    query_heads=int(
                        source_configuration["num_attention_heads"]
                    ),
                    key_value_heads=int(
                        source_configuration["num_key_value_heads"]
                    ),
                    head_dim=int(source_configuration["head_dim"]),
                    layer_types=untied_rebalance_key[5],
                    compiler_executable=not set(
                        rebalance_required_capabilities
                    ).difference(self.compiler_capabilities),
                )
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
        component_policy_scorer = _DenseComponentPolicyScorer(
            parent_graph,
            protection_rules=self.component_protection_rules,
            bottleneck_risk_weight=self.component_bottleneck_risk_weight,
        )
        component_policy_full_stream_enabled = True
        balanced_component_weights = dict(_PARETO_SCALARIZATIONS[0][1])
        component_policy_winner: tuple[
            tuple[Any, ...],
            DenseTargetTopologyCandidate,
        ] | None = None
        component_policy_scored = 0
        component_policy_floor_rejections = 0
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
                        if target_tied_token_interface
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
                    minimum_intermediate = (
                        min(geometry.intermediate_size, hidden)
                        if self.target_token_interface_mode
                        == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
                        else 1
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
                            tied_token_interface=(
                                target_tied_token_interface
                            ),
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
                            if target_tied_token_interface
                            else 0
                        )
                        vocabulary_unique = (
                            target_vocab
                            * hidden
                            * (
                                1
                                if target_tied_token_interface
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
                        topology_key = (
                            hidden,
                            intermediate,
                            query_heads,
                            key_value_heads,
                            head_dim,
                            tuple(layer_types),
                            tuple(spans),
                        )
                        profile_values = dict(pareto_profile)
                        component_supported_rank: tuple[Any, ...] = ()
                        component_winning = False
                        if component_policy_full_stream_enabled:
                            component_policy_scored += 1
                            component_policy_score = (
                                component_policy_scorer.score(
                                    _lfm2_target_semantic_capacities_from_geometry(
                                        geometry,
                                        logical_parameters=logical,
                                        hidden_size=hidden,
                                        intermediate_size=intermediate,
                                        query_heads=query_heads,
                                        key_value_heads=key_value_heads,
                                        head_dim=head_dim,
                                        vocabulary_size=target_vocab,
                                        layer_types=layer_types,
                                    ),
                                )
                            )
                            if component_policy_score is None:
                                component_policy_floor_rejections += 1
                                continue
                            component_semantic = (
                                component_policy_score.semantic_retention
                            )
                            balanced_score = sum(
                                balanced_component_weights[axis]
                                * profile_values[axis]
                                for axis in _PARETO_AXES
                            )
                            component_supported_rank = (
                                component_policy_score.total_predicted_risk,
                                -component_semantic.get(
                                    SEMANTIC_STATE_MIXER_INTERNAL,
                                    0,
                                ),
                                -component_semantic.get(
                                    SEMANTIC_ATTENTION_QUERY_KEY,
                                    0,
                                ),
                                -balanced_score,
                                abs(unique - requested),
                                -retention_score,
                                -unique,
                                rough_rank,
                            )
                            component_winning = (
                                compiler_executable
                                and (
                                    component_policy_winner is None
                                    or component_supported_rank
                                    < component_policy_winner[0]
                                )
                            )
                        if (
                            untied_rebalance_profile is not None
                            and topology_key != untied_rebalance_key
                        ):
                            rebalance_values = dict(
                                untied_rebalance_profile
                            )
                            if all(
                                profile_values[axis]
                                >= rebalance_values[axis]
                                for axis in _PARETO_AXES
                            ) and any(
                                profile_values[axis]
                                > rebalance_values[axis]
                                for axis in _PARETO_AXES
                            ):
                                untied_rebalance_dominated = True
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
                        if topology_key == untied_rebalance_key:
                            if unique != untied_rebalance_expected_unique:
                                raise DynamicParameterFrontierError(
                                    "Untied rebalance topology did not reproduce "
                                    "its derived exact parameter count."
                                )
                            winning_profiles.append(
                                (
                                    _LFM2_UNTIED_REBALANCE_PROFILE,
                                    ((0,), rough_rank),
                                )
                            )
                        if not winning_profiles and not component_winning:
                            continue
                        span_string = ";".join(
                            ",".join(str(index) for index in span)
                            for span in spans
                        )
                        layer_string = ",".join(layer_types)
                        configuration_values: dict[str, JsonScalar] = {
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
                                    target_tied_token_interface
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
                        if (
                            self.target_token_interface_mode
                            == LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
                        ):
                            configuration_values[
                                LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY
                            ] = LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
                        configuration = _configuration(configuration_values)
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
                            if target_tied_token_interface
                            else (),
                            source_vocabulary_size=geometry.vocab_size,
                            target_vocabulary_size=target_vocab,
                            vocabulary_unique_parameters=(
                                target_vocab
                                * hidden
                                * (
                                    1
                                    if target_tied_token_interface
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
                        if component_winning:
                            component_policy_winner = (
                                component_supported_rank,
                                candidate,
                            )

        if untied_rebalance_key is not None:
            if _LFM2_UNTIED_REBALANCE_PROFILE not in profile_winners:
                raise DynamicParameterFrontierError(
                    "Derived untied preserve-frontier rebalance was not "
                    "enumerated by the target catalog."
                )
            if untied_rebalance_dominated:
                raise DynamicParameterFrontierError(
                    "Derived untied preserve-frontier rebalance is Pareto "
                    "dominated in the complete target search space."
                )
        if component_policy_winner is not None:
            profile_winners[LFM2_COMPONENT_POLICY_PROFILE] = (
                component_policy_winner
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
            generic_support = tuple(
                profile
                for profile in candidate.pareto_support_profiles
                if profile != LFM2_COMPONENT_POLICY_PROFILE
            )
            if generic_support and any(
                _candidate_dominates(other, candidate)
                for other_index, other in enumerate(supported)
                if other_index != index
            ):
                raise DynamicParameterFrontierError(
                    "A generic scalarization winner was Pareto dominated; "
                    "the catalog objective audit is inconsistent."
                )
        component_policy_generic_axis_dominated = False
        if component_policy_winner is not None:
            component_candidate_id = component_policy_winner[1].candidate_id
            component_candidate = next(
                candidate
                for candidate in supported
                if candidate.candidate_id == component_candidate_id
            )
            component_policy_generic_axis_dominated = any(
                _candidate_dominates(other, component_candidate)
                for other in supported
                if other.candidate_id != component_candidate_id
            )

        # Preserve capability selection and then expose distinct objective
        # trade-offs.  This is deliberately not an exact-count-only sort.
        profile_order = (
            *(
                (
                    "budget_utilization_first",
                    LFM2_COMPONENT_POLICY_PROFILE,
                    "compiler_executable_first",
                )
                if match_policy == "nearest"
                else (
                    LFM2_COMPONENT_POLICY_PROFILE,
                    "compiler_executable_first",
                    "budget_utilization_first",
                )
            ),
            "balanced",
            *(
                (_LFM2_UNTIED_REBALANCE_PROFILE,)
                if untied_rebalance_key is not None
                else ()
            ),
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
        bound_candidates: list[DenseTargetTopologyCandidate] = []
        for candidate in deduplicated:
            expected_component_score = component_policy_scorer.score(
                _lfm2_target_semantic_capacities(geometry, candidate),
            )
            if expected_component_score is None:
                raise DynamicParameterFrontierError(
                    "A returned component-policy candidate violates its "
                    "configured retention floor."
                )
            bound = _bind_lfm2_component_allocation_receipt(
                parent_graph,
                geometry,
                candidate,
                protection_rules=self.component_protection_rules,
                bottleneck_risk_weight=(
                    self.component_bottleneck_risk_weight
                ),
            )
            receipt = bound.component_allocation_receipt
            assert receipt is not None
            if (
                not math.isclose(
                    receipt.total_predicted_risk,
                    expected_component_score.total_predicted_risk,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or receipt.semantic_retention
                != expected_component_score.semantic_retention
            ):
                raise DynamicParameterFrontierError(
                    "Lightweight full-stream component score differs from "
                    "the final typed allocator receipt."
                )
            bound_candidates.append(bound)
        deduplicated = bound_candidates
        dense_token_floor = (
            target_vocab
            * minimum_hidden
            * (1 if target_tied_token_interface else 2)
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
                (
                    "exact_supported_points_from_strictly_positive_"
                    "scalarizations_over_all_eligible_enumerated_topologies"
                )
                if untied_rebalance_key is None
                else (
                    "strictly_positive_scalarization_winners_plus_"
                    "full_search_pareto_verified_untied_rebalance"
                )
            ),
            "exact_full_pareto_surface_enumerated": False,
            "returned_candidates_pairwise_nondominated": True,
            "returned_candidates_pairwise_nondominated_objectives": (
                "generic_pareto_axes_plus_component_predicted_risk"
            ),
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
            "component_allocation_policy": COMPONENT_ALLOCATION_POLICY,
            "component_allocation_receipts_embedded": True,
            "component_allocation_candidate_count": len(deduplicated),
            "component_policy_support_profile": (
                LFM2_COMPONENT_POLICY_PROFILE
            ),
            "component_policy_extends_generic_pareto_objectives": True,
            "component_policy_winner_generic_axis_dominated": (
                component_policy_generic_axis_dominated
            ),
            "component_policy_full_eligible_stream_scored": (
                component_policy_full_stream_enabled
            ),
            "component_policy_implicit_and_explicit_default_equivalent": True,
            "component_policy_topologies_scored_before_candidate_limit": (
                component_policy_scored
            ),
            "component_policy_topologies_meeting_retention_floors": (
                component_policy_scored
                - component_policy_floor_rejections
            ),
            "component_policy_retention_floor_rejections": (
                component_policy_floor_rejections
            ),
            "component_policy_executable_winner_preserved": (
                component_policy_winner is not None
            ),
            "component_policy_typed_receipts_built_only_after_limit": True,
            "component_policy_lightweight_scores_typed_receipt_verified": (
                True
            ),
            "component_bottleneck_risk_weight": (
                self.component_bottleneck_risk_weight
            ),
            "component_retention_risk_policy": (
                COMPONENT_RETENTION_RISK_POLICY
            ),
            "component_policy_reference_topology_prior_used": False,
            "component_allocation_uses_materializable_global_topologies": True,
            "component_allocation_heterogeneous_per_layer_widths_claimed": False,
            "dense_topology_only": True,
            "minimum_hidden_size": minimum_hidden,
            "minimum_head_dim": self.minimum_head_dim,
            "minimum_ffn_expansion_ratio": (
                1.0
                if self.target_token_interface_mode
                == LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
                else 1 / geometry.hidden_size
            ),
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
        if (
            self.target_token_interface_mode
            != LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE
        ):
            diagnostics["target_token_interface_mode"] = (
                self.target_token_interface_mode
            )
            diagnostics["target_token_interface_tied"] = (
                target_tied_token_interface
            )
            diagnostics["source_token_interface_tied"] = (
                geometry.tied_token_interface
            )
        if untied_rebalance_key is not None:
            assert untied_rebalance_source is not None
            diagnostics["untied_rebalance_profile"] = (
                _LFM2_UNTIED_REBALANCE_PROFILE
            )
            diagnostics["untied_rebalance_source_candidate_fingerprint"] = (
                untied_rebalance_source.fingerprint
            )
            diagnostics["untied_rebalance_source_unique_parameters"] = (
                untied_rebalance_source.unique_parameters
            )
            diagnostics["untied_rebalance_target_unique_parameters"] = (
                untied_rebalance_expected_unique
            )
            diagnostics["untied_rebalance_parameter_ceiling_spare"] = (
                untied_rebalance_spare
            )
            diagnostics["untied_rebalance_full_search_nondominated"] = True
            diagnostics["untied_rebalance_changed_only_ffn_width"] = True
        return CatalogFrontier(
            candidates=tuple(deduplicated),
            diagnostics=diagnostics,
        )


def default_dense_topology_catalog_registry(
) -> DenseTargetTopologyCatalogRegistry:
    return DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(),
            HfLlamaDenseTargetTopologyCatalog(),
        )
    )


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
    "CAP_CAUSAL_DECODER_FAMILY_TRANSLATION",
    "CAP_DENSE_BLANK_MATERIALIZATION",
    "CAP_FFN_WIDTH_PROJECTION",
    "CAP_HETEROGENEOUS_LAYER_LAYOUT",
    "CAP_HYBRID_OPERATOR_COMPOSITION",
    "CAP_ORDERED_DEPTH_COMPOSITION",
    "CAP_RESIDUAL_WIDTH_PROJECTION",
    "CAP_TOKENIZER_BRIDGE",
    "COMPONENT_ALLOCATION_POLICY",
    "COMPONENT_ALLOCATION_POLICY_CONFIG_KEY",
    "COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY",
    "COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY",
    "CatalogFrontier",
    "DenseComponentAllocationReceipt",
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
    "DENSE_TOPOLOGY_CATALOG_CONFIG_KEY",
    "HF_LLAMA_CATALOG_ID",
    "HF_LLAMA_ORDERED_SOURCE_SPANS_CONFIG_KEY",
    "HF_LLAMA_PARAMETER_ALIAS_POLICY_CONFIG_KEY",
    "HF_LLAMA_PARAMETER_ALIAS_POLICY_NONE",
    "HF_LLAMA_SOURCE_FAMILY_CONFIG_KEY",
    "HF_LLAMA_SOURCE_LAYER_COUNT_CONFIG_KEY",
    "HF_LLAMA_TARGET_SCHEMA",
    "HF_LLAMA_TARGET_SCHEMA_CONFIG_KEY",
    "HfLlamaDenseTargetTopologyCatalog",
    "HfLlamaDenseTopology",
    "LFM2_CATALOG_ID",
    "LFM2_COMPONENT_POLICY_PROFILE",
    "LFM2_CURRENT_COMPILER_CAPABILITIES",
    "LFM2_DENSE_TOKEN_INTERFACE_MODE_CONFIG_KEY",
    "LFM2_DENSE_TOKEN_INTERFACE_MODE_PRESERVE",
    "LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED",
    "LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_CONFIG_KEY",
    "LFM2_HETEROGENEOUS_LAYOUT_DESCRIPTOR_SHA256_CONFIG_KEY",
    "LFM2_HETEROGENEOUS_LAYOUT_KIND",
    "LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_CONFIG_KEY",
    "LFM2_HETEROGENEOUS_LAYOUT_SELECTION_RECEIPT_SHA256_CONFIG_KEY",
    "Lfm2DenseTargetTopologyCatalog",
    "Lfm2HeterogeneousLayoutSelectionInput",
    "SCHEMA",
    "default_dense_topology_catalog_registry",
    "component_allocation_core_configuration",
    "component_allocation_topology_fingerprint",
    "select_best_executable_dense_candidate",
    "select_lfm2_untied_rebalance_candidate",
    "synthesize_dense_compression_ratio_frontier",
    "synthesize_dense_parameter_frontier",
]
