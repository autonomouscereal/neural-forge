"""Cache-only global exact-topology frontier for production LFM2 folding.

The existing heterogeneous selector chooses MLP and operator capacity at one
residual width.  This module lifts that selector over every catalog-legal
residual width without changing the downstream blank-target or fold path.

Every feasible point is a complete, full-depth, dense child with the same
exact learned-parameter count.  Width is scored from a role-balanced cached
functional spectrum.  Per-layer geometry is scored from the cached nonlinear
capacity bank.  Parameter count is an equality constraint, never a quality
score.  No model is executed and no parameter is trained, pruned, quantized,
shared, or sparsified here.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .dynamic_parameter_frontier import Lfm2HeterogeneousLayoutSelectionInput
from .hf_lfm2_cached_functional_width import (
    Lfm2CachedFunctionalWidthEvidence,
    build_lfm2_cached_functional_width_evidence,
    enumerate_lfm2_global_hidden_sizes,
)
from .hf_lfm2_cached_heterogeneous_capacity import (
    ATTENTION_GEOMETRY_ENABLED_POLICY,
    AUTOMATIC_MLP_OPTION_POLICY,
    EXPLICIT_MLP_OPTION_POLICY,
    Lfm2CachedHeterogeneousCapacityError,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedLayerCapacityEvidence,
    Lfm2MlpWidthOptionPolicy,
    build_lfm2_cached_heterogeneous_capacity_option_bank,
    derive_lfm2_automatic_mlp_width_policy,
)
from .hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
)
from .hf_lfm2_heterogeneous_layout_selection import (
    Lfm2HeterogeneousLayoutSelection,
    Lfm2HeterogeneousLayoutUnattainable,
    build_lfm2_attention_layer_option,
    build_lfm2_heterogeneous_target_pins,
    build_lfm2_short_conv_layer_option,
    discover_lfm2_source_layers,
)


GLOBAL_TOPOLOGY_RUNTIME_SCHEMA = "neural-forge-lfm2-global-runtime-pins-v1"
GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA = "neural-forge-lfm2-global-topology-candidate-v1"
GLOBAL_TOPOLOGY_REJECTION_SCHEMA = "neural-forge-lfm2-global-topology-rejection-v1"
GLOBAL_TOPOLOGY_FRONTIER_SCHEMA = "neural-forge-lfm2-global-topology-frontier-v1"
GLOBAL_TOPOLOGY_SCORE_METHOD = (
    "role-balanced-functional-loss-plus-mean-cached-component-risk-v1"
)
RETARGETED_CAPACITY_EVIDENCE_SCHEMA = (
    "neural-forge-lfm2-retargeted-capacity-evidence-v1"
)


class Lfm2CachedGlobalTopologyFrontierError(RuntimeError):
    """The cache cannot authenticate a complete global exact frontier."""


class Lfm2CachedGlobalTopologyFrontierUnattainable(
    Lfm2CachedGlobalTopologyFrontierError
):
    """No legal residual width can materialize the requested exact child."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a positive integer."
        )
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite_nonnegative(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be finite and non-negative."
        )
    return float(value)


def _positive_float(value: Any, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result <= 0.0:
        raise Lfm2CachedGlobalTopologyFrontierError(f"{name} must be positive.")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CachedGlobalTopologyFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2CachedGlobalTopologyFrontierError(f"{name} must be nonempty text.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTargetRuntimePins:
    """Runtime and token-interface pins shared by every residual width."""

    vocab_size: int
    conv_kernel_size: int
    rope_theta: float
    norm_eps: float
    max_position_embeddings: int
    use_cache: bool
    attention_implementation: str
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    block_auto_adjust_ff_dim: bool
    target_vocab_evidence_sha256: str
    token_bridge_evidence_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.vocab_size, "vocab_size")
        _positive_int(self.conv_kernel_size, "conv_kernel_size")
        _positive_float(self.rope_theta, "rope_theta")
        _positive_float(self.norm_eps, "norm_eps")
        _positive_int(self.max_position_embeddings, "max_position_embeddings")
        if type(self.use_cache) is not bool:
            raise Lfm2CachedGlobalTopologyFrontierError("use_cache must be boolean.")
        _text(self.attention_implementation, "attention_implementation")
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            _nonnegative_int(getattr(self, name), name)
        if type(self.block_auto_adjust_ff_dim) is not bool:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "block_auto_adjust_ff_dim must be boolean."
            )
        _digest(
            self.target_vocab_evidence_sha256,
            "target_vocab_evidence_sha256",
        )
        _digest(
            self.token_bridge_evidence_sha256,
            "token_bridge_evidence_sha256",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_RUNTIME_SCHEMA,
            "vocab_size": self.vocab_size,
            "conv_kernel_size": self.conv_kernel_size,
            "rope_theta": self.rope_theta,
            "norm_eps": self.norm_eps,
            "max_position_embeddings": self.max_position_embeddings,
            "use_cache": self.use_cache,
            "attention_implementation": self.attention_implementation,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "block_auto_adjust_ff_dim": self.block_auto_adjust_ff_dim,
            "target_vocab_evidence_sha256": (self.target_vocab_evidence_sha256),
            "token_bridge_evidence_sha256": (self.token_bridge_evidence_sha256),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTopologyCandidateReport:
    """One feasible complete exact child on the global frontier."""

    hidden_size: int
    exact_unique_parameters: int
    target_layer_count: int
    role_balanced_functional_retention: float
    role_balanced_functional_loss: float
    mean_selected_component_risk: float
    combined_predicted_risk: float
    functional_score_sha256: str
    mlp_width_lattice_sha256: str
    capacity_option_bank_sha256: str
    selection_input_sha256: str
    descriptor_sha256: str
    layout_selection_receipt_sha256: str
    candidate_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.hidden_size, "hidden_size")
        _positive_int(self.exact_unique_parameters, "exact_unique_parameters")
        _positive_int(self.target_layer_count, "target_layer_count")
        retained = _finite_nonnegative(
            self.role_balanced_functional_retention,
            "role_balanced_functional_retention",
        )
        loss = _finite_nonnegative(
            self.role_balanced_functional_loss,
            "role_balanced_functional_loss",
        )
        if (
            retained > 1.0 + 1e-12
            or loss > 1.0 + 1e-12
            or not math.isclose(
                retained + loss,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Functional retention and loss must be complementary fractions."
            )
        component = _finite_nonnegative(
            self.mean_selected_component_risk,
            "mean_selected_component_risk",
        )
        combined = _finite_nonnegative(
            self.combined_predicted_risk,
            "combined_predicted_risk",
        )
        if not math.isclose(combined, loss + component, rel_tol=0.0, abs_tol=1e-10):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Combined frontier risk differs from its cached evidence terms."
            )
        for name in (
            "functional_score_sha256",
            "mlp_width_lattice_sha256",
            "capacity_option_bank_sha256",
            "selection_input_sha256",
            "descriptor_sha256",
            "layout_selection_receipt_sha256",
        ):
            _digest(getattr(self, name), name)
        if canonical_sha256(self._body()) != _digest(
            self.candidate_sha256,
            "candidate_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology candidate seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_CANDIDATE_SCHEMA,
            "hidden_size": self.hidden_size,
            "exact_unique_parameters": self.exact_unique_parameters,
            "target_layer_count": self.target_layer_count,
            "role_balanced_functional_retention": (
                self.role_balanced_functional_retention
            ),
            "role_balanced_functional_loss": (self.role_balanced_functional_loss),
            "mean_selected_component_risk": self.mean_selected_component_risk,
            "combined_predicted_risk": self.combined_predicted_risk,
            "score_method": GLOBAL_TOPOLOGY_SCORE_METHOD,
            "functional_score_sha256": self.functional_score_sha256,
            "mlp_width_lattice_sha256": self.mlp_width_lattice_sha256,
            "capacity_option_bank_sha256": self.capacity_option_bank_sha256,
            "selection_input_sha256": self.selection_input_sha256,
            "descriptor_sha256": self.descriptor_sha256,
            "layout_selection_receipt_sha256": (self.layout_selection_receipt_sha256),
            "parameter_count_used_as_quality_score": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "candidate_sha256": self.candidate_sha256}

    @classmethod
    def build(
        cls,
        *,
        hidden_size: int,
        exact_unique_parameters: int,
        target_layer_count: int,
        functional_retention: float,
        functional_loss: float,
        mean_component_risk: float,
        functional_score_sha256: str,
        mlp_width_lattice_sha256: str,
        capacity_option_bank_sha256: str,
        selection_input_sha256: str,
        descriptor_sha256: str,
        layout_selection_receipt_sha256: str,
    ) -> Lfm2GlobalTopologyCandidateReport:
        values = {
            "hidden_size": hidden_size,
            "exact_unique_parameters": exact_unique_parameters,
            "target_layer_count": target_layer_count,
            "role_balanced_functional_retention": functional_retention,
            "role_balanced_functional_loss": functional_loss,
            "mean_selected_component_risk": mean_component_risk,
            "combined_predicted_risk": functional_loss + mean_component_risk,
            "functional_score_sha256": functional_score_sha256,
            "mlp_width_lattice_sha256": mlp_width_lattice_sha256,
            "capacity_option_bank_sha256": capacity_option_bank_sha256,
            "selection_input_sha256": selection_input_sha256,
            "descriptor_sha256": descriptor_sha256,
            "layout_selection_receipt_sha256": (layout_selection_receipt_sha256),
        }
        provisional = object.__new__(cls)
        for name, value in values.items():
            object.__setattr__(provisional, name, value)
        object.__setattr__(provisional, "candidate_sha256", "0" * 64)
        return cls(**values, candidate_sha256=canonical_sha256(provisional._body()))


@dataclass(frozen=True, slots=True)
class Lfm2GlobalTopologyRejectedWidth:
    hidden_size: int
    reason_class: str
    reason: str
    rejection_sha256: str

    def __post_init__(self) -> None:
        _positive_int(self.hidden_size, "hidden_size")
        _text(self.reason_class, "reason_class")
        _text(self.reason, "reason")
        if canonical_sha256(self._body()) != _digest(
            self.rejection_sha256,
            "rejection_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology rejection seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_REJECTION_SCHEMA,
            "hidden_size": self.hidden_size,
            "reason_class": self.reason_class,
            "reason": self.reason,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "rejection_sha256": self.rejection_sha256}

    @classmethod
    def build(
        cls,
        hidden_size: int,
        reason_class: str,
        reason: str,
    ) -> Lfm2GlobalTopologyRejectedWidth:
        body = {
            "schema": GLOBAL_TOPOLOGY_REJECTION_SCHEMA,
            "hidden_size": hidden_size,
            "reason_class": reason_class,
            "reason": reason,
        }
        return cls(
            hidden_size=hidden_size,
            reason_class=reason_class,
            reason=reason,
            rejection_sha256=canonical_sha256(body),
        )


def select_lfm2_global_topology_report(
    candidates: Sequence[Lfm2GlobalTopologyCandidateReport],
) -> Lfm2GlobalTopologyCandidateReport:
    """Choose by cached function and component risk, never width or count."""

    values = tuple(candidates)
    if not values or any(
        type(value) is not Lfm2GlobalTopologyCandidateReport for value in values
    ):
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No authenticated exact global-topology candidate is available."
        )
    counts = {value.exact_unique_parameters for value in values}
    if len(counts) != 1:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "Global candidates must share one exact learned-parameter count."
        )
    return min(
        values,
        key=lambda value: (
            value.combined_predicted_risk,
            value.role_balanced_functional_loss,
            value.mean_selected_component_risk,
            value.candidate_sha256,
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CachedGlobalTopologyFrontier:
    """Selected production input plus the authenticated global-width audit."""

    source_graph_fingerprint: str
    parent_cache_key: str
    exact_target_unique_parameters: int
    source_layer_count: int
    candidate_hidden_sizes: tuple[int, ...]
    base_union_capacity_option_bank_sha256: str
    functional_width_evidence: Lfm2CachedFunctionalWidthEvidence
    candidates: tuple[Lfm2GlobalTopologyCandidateReport, ...]
    rejected_widths: tuple[Lfm2GlobalTopologyRejectedWidth, ...]
    selected_candidate_sha256: str
    selected_input: Lfm2HeterogeneousLayoutSelectionInput
    selected_capacity_option_bank: Lfm2CachedHeterogeneousCapacityOptionBank
    selected_layout: Lfm2HeterogeneousLayoutSelection
    frontier_sha256: str

    def __post_init__(self) -> None:
        _digest(self.source_graph_fingerprint, "source_graph_fingerprint")
        _digest(self.parent_cache_key, "parent_cache_key")
        target = _positive_int(
            self.exact_target_unique_parameters,
            "exact_target_unique_parameters",
        )
        source_layer_count = _positive_int(
            self.source_layer_count,
            "source_layer_count",
        )
        widths = tuple(self.candidate_hidden_sizes)
        if not widths or widths != tuple(sorted(set(widths))):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global candidate widths must be nonempty and canonical."
            )
        _digest(
            self.base_union_capacity_option_bank_sha256,
            "base_union_capacity_option_bank_sha256",
        )
        candidates = tuple(self.candidates)
        rejected = tuple(self.rejected_widths)
        if (
            tuple(value.hidden_size for value in candidates)
            != tuple(sorted(value.hidden_size for value in candidates))
            or tuple(value.hidden_size for value in rejected)
            != tuple(sorted(value.hidden_size for value in rejected))
            or {value.hidden_size for value in candidates}
            & {value.hidden_size for value in rejected}
            or {value.hidden_size for value in candidates}
            | {value.hidden_size for value in rejected}
            != set(widths)
            or any(value.exact_unique_parameters != target for value in candidates)
            or any(
                value.target_layer_count != source_layer_count for value in candidates
            )
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global frontier coverage or exact accounting is inconsistent."
            )
        selected = select_lfm2_global_topology_report(candidates)
        if selected.candidate_sha256 != _digest(
            self.selected_candidate_sha256,
            "selected_candidate_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Selected global candidate is not the cached-risk optimum."
            )
        if (
            self.selected_input.fingerprint != selected.selection_input_sha256
            or self.selected_capacity_option_bank.evidence_sha256
            != selected.capacity_option_bank_sha256
            or self.selected_capacity_option_bank.attention_geometry_policy
            != ATTENTION_GEOMETRY_ENABLED_POLICY
            or self.selected_layout.descriptor.descriptor_sha256
            != selected.descriptor_sha256
            or self.selected_layout.receipt.receipt_sha256
            != selected.layout_selection_receipt_sha256
            or self.selected_layout.descriptor.accounting.unique_parameter_elements
            != target
            or self.selected_input.target_pins.hidden_size != selected.hidden_size
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Selected production artifacts differ from the frontier report."
            )
        if canonical_sha256(self._body()) != _digest(
            self.frontier_sha256,
            "frontier_sha256",
        ):
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Global topology frontier seal is invalid."
            )

    def _body(self) -> dict[str, Any]:
        return {
            "schema": GLOBAL_TOPOLOGY_FRONTIER_SCHEMA,
            "source_graph_fingerprint": self.source_graph_fingerprint,
            "parent_cache_key": self.parent_cache_key,
            "exact_target_unique_parameters": self.exact_target_unique_parameters,
            "source_layer_count": self.source_layer_count,
            "candidate_hidden_sizes": list(self.candidate_hidden_sizes),
            "base_union_capacity_option_bank_sha256": (
                self.base_union_capacity_option_bank_sha256
            ),
            "functional_width_evidence_sha256": (
                self.functional_width_evidence.evidence_sha256
            ),
            "candidates": [value.as_dict() for value in self.candidates],
            "rejected_widths": [value.as_dict() for value in self.rejected_widths],
            "selected_candidate_sha256": self.selected_candidate_sha256,
            "selected_input_sha256": self.selected_input.fingerprint,
            "selected_capacity_option_bank_sha256": (
                self.selected_capacity_option_bank.evidence_sha256
            ),
            "attention_geometry_policy": (
                self.selected_capacity_option_bank.attention_geometry_policy
            ),
            "selected_descriptor_sha256": (
                self.selected_layout.descriptor.descriptor_sha256
            ),
            "selected_layout_selection_receipt_sha256": (
                self.selected_layout.receipt.receipt_sha256
            ),
            "selection_contract": {
                "complete_full_depth_child_per_feasible_width": True,
                "same_exact_parameter_count_per_feasible_width": True,
                "parameter_count_used_as_quality_score": False,
                "cached_parent_function_only": True,
                "cached_attention_geometry_allocation": True,
                "parent_model_calls": 0,
                "child_model_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "sparse_parameters": 0,
                "shared_parameters": 0,
            },
        }

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                **self._body(),
                "functional_width_evidence": dict(
                    self.functional_width_evidence.as_dict()
                ),
                "frontier_sha256": self.frontier_sha256,
            }
        )


def _source_layer_contract(
    source_graph: ModelCircuitGraph,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    discovered = discover_lfm2_source_layers(source_graph)
    nodes = {value.node_id: value for value in source_graph.nodes}
    layer_types = tuple(value.layer_type for value in discovered)
    widths = []
    for layer in discovered:
        node = nodes[layer.mlp_node_id]
        attributes = dict(node.attributes)
        width = attributes.get("intermediate_size")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            producer_widths = {
                parameter.shape[0]
                for parameter in node.parameters
                if parameter.role in {"mlp.w1.weight", "mlp.w3.weight"}
                and len(parameter.shape) == 2
            }
            writer_widths = {
                parameter.shape[1]
                for parameter in node.parameters
                if parameter.role == "mlp.w2.weight" and len(parameter.shape) == 2
            }
            inferred = producer_widths & writer_widths
            if len(inferred) != 1:
                raise Lfm2CachedGlobalTopologyFrontierError(
                    "Cannot discover one source SwiGLU width for layer "
                    f"{layer.ordinal}."
                )
            width = inferred.pop()
        widths.append(width)
    if len(set(widths)) != 1:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The current LFM2 runtime requires one configured source MLP width; "
            "the inspected graph exposes heterogeneous source widths."
        )
    return layer_types, tuple(widths)


def _retarget_capacity_option_bank(
    base: Lfm2CachedHeterogeneousCapacityOptionBank,
    policy: Lfm2MlpWidthOptionPolicy,
) -> Lfm2CachedHeterogeneousCapacityOptionBank:
    """Filter one union cache scan into an authenticated per-H option bank."""

    if base.mlp_option_policy != EXPLICIT_MLP_OPTION_POLICY:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The global frontier base bank must be an explicit union scan."
        )
    if base.attention_geometry_policy != ATTENTION_GEOMETRY_ENABLED_POLICY:
        raise Lfm2CachedGlobalTopologyFrontierError(
            "The residual-attention frontier requires an authenticated cached "
            "attention-geometry option bank."
        )
    if policy.policy_id != AUTOMATIC_MLP_OPTION_POLICY or len(
        policy.widths_by_layer
    ) != len(base.layer_evidence):
        raise Lfm2CachedGlobalTopologyFrontierError(
            "Automatic MLP policy differs from the union capacity bank layers."
        )
    layer_evidence = []
    allowed_by_source: dict[str, frozenset[int]] = {}
    evidence_by_source: dict[str, str] = {}
    for evidence, widths in zip(
        base.layer_evidence,
        policy.widths_by_layer,
        strict=True,
    ):
        risk_by_width = {
            value.intermediate_size: value for value in evidence.mlp_capacity_risks
        }
        missing = tuple(width for width in widths if width not in risk_by_width)
        if missing:
            raise Lfm2CachedGlobalTopologyFrontierError(
                "Union nonlinear scan lacks automatic widths "
                f"{missing} for layer {evidence.layer_index}."
            )
        risks = tuple(risk_by_width[width] for width in widths)
        writer_risks = tuple(
            value
            for value in evidence.short_conv_writer_risks
            if value.writer_rank in policy.short_conv_writer_ranks
        )
        evidence_body = {
            "schema": RETARGETED_CAPACITY_EVIDENCE_SCHEMA,
            "base_capacity_option_bank_sha256": base.evidence_sha256,
            "base_layer_evidence_sha256": evidence.selection_evidence_sha256,
            "automatic_mlp_option_policy": policy.policy_id,
            "attention_geometry_policy": base.attention_geometry_policy,
            "exact_budget_lattice_sha256": policy.exact_budget_lattice_sha256,
            "layer_index": evidence.layer_index,
            "source_layer_sha256": evidence.source_layer_sha256,
            "mlp_capacity_risks": [value.as_dict() for value in risks],
            "attention_geometry_risks": [
                value.as_dict() for value in evidence.attention_geometry_risks
            ],
            "short_conv_writer_risks": [
                value.as_dict() for value in writer_risks
            ],
        }
        retargeted = Lfm2CachedLayerCapacityEvidence(
            layer_index=evidence.layer_index,
            layer_type=evidence.layer_type,
            source_layer_sha256=evidence.source_layer_sha256,
            operator_delta_energy_fraction=(evidence.operator_delta_energy_fraction),
            mlp_delta_energy_fraction=evidence.mlp_delta_energy_fraction,
            sampled_scalar_count=evidence.sampled_scalar_count,
            nonlinear_sampled_row_count=evidence.nonlinear_sampled_row_count,
            source_mlp_intermediate_size=evidence.source_mlp_intermediate_size,
            mlp_capacity_risks=risks,
            attention_geometry_risks=evidence.attention_geometry_risks,
            short_conv_writer_risks=writer_risks,
            selection_evidence_sha256=canonical_sha256(evidence_body),
        )
        layer_evidence.append(retargeted)
        allowed_by_source[evidence.source_layer_sha256] = frozenset(widths)
        evidence_by_source[evidence.source_layer_sha256] = (
            retargeted.selection_evidence_sha256
        )

    options = []
    for option in base.layer_options:
        if (
            option.mlp_intermediate_size
            not in allowed_by_source[option.source_layer_sha256]
        ):
            continue
        if (
            option.layer_type == "conv"
            and option.short_conv_writer_rank is not None
            and option.short_conv_writer_rank
            not in policy.short_conv_writer_ranks
        ):
            continue
        option_evidence = canonical_sha256(
            {
                "schema": "neural-forge-lfm2-retargeted-capacity-option-v1",
                "base_option_sha256": option.option_sha256,
                "base_option_evidence_sha256": (option.selection_evidence_sha256),
                "retargeted_layer_evidence_sha256": evidence_by_source[
                    option.source_layer_sha256
                ],
                "automatic_mlp_option_policy": policy.policy_id,
                "attention_geometry_policy": base.attention_geometry_policy,
                "exact_budget_lattice_sha256": (policy.exact_budget_lattice_sha256),
            }
        )
        if option.layer_type == "conv":
            assert option.short_conv_internal_width is not None
            options.append(
                build_lfm2_short_conv_layer_option(
                    source_layer_sha256=option.source_layer_sha256,
                    mlp_intermediate_size=option.mlp_intermediate_size,
                    internal_width=option.short_conv_internal_width,
                    writer_rank=option.short_conv_writer_rank,
                    selection_evidence_sha256=option_evidence,
                    structural_risk=option.structural_risk,
                )
            )
        else:
            assert option.query_heads is not None
            assert option.key_value_heads is not None
            assert option.head_dim is not None
            options.append(
                build_lfm2_attention_layer_option(
                    source_layer_sha256=option.source_layer_sha256,
                    mlp_intermediate_size=option.mlp_intermediate_size,
                    query_heads=option.query_heads,
                    key_value_heads=option.key_value_heads,
                    head_dim=option.head_dim,
                    selection_evidence_sha256=option_evidence,
                    structural_risk=option.structural_risk,
                )
            )
    provisional = object.__new__(Lfm2CachedHeterogeneousCapacityOptionBank)
    values = {
        "source_graph_fingerprint": base.source_graph_fingerprint,
        "parent_cache_key": base.parent_cache_key,
        "mlp_option_policy": policy.policy_id,
        "attention_geometry_policy": base.attention_geometry_policy,
        "exact_budget_lattice_sha256": policy.exact_budget_lattice_sha256,
        "layer_evidence": tuple(layer_evidence),
        "layer_options": tuple(options),
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "evidence_sha256", "0" * 64)
    return Lfm2CachedHeterogeneousCapacityOptionBank(
        **values,
        evidence_sha256=canonical_sha256(provisional._body()),
    )


def _target_pins_for_width(
    *,
    hidden_size: int,
    runtime: Lfm2GlobalTargetRuntimePins,
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    base_bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    selected_bank: Lfm2CachedHeterogeneousCapacityOptionBank,
    functional_evidence: Lfm2CachedFunctionalWidthEvidence,
    policy: Lfm2MlpWidthOptionPolicy,
    exact_target_unique_parameters: int,
) -> Any:
    runtime_body = {
        "schema": "neural-forge-lfm2-global-topology-runtime-evidence-v1",
        "runtime_pins_sha256": runtime.fingerprint,
        "source_graph_fingerprint": source_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "base_union_capacity_option_bank_sha256": base_bank.evidence_sha256,
        "selected_capacity_option_bank_sha256": selected_bank.evidence_sha256,
        "functional_width_evidence_sha256": functional_evidence.evidence_sha256,
        "functional_width_score_sha256": (
            functional_evidence.score_for(hidden_size).score_sha256
        ),
        "mlp_width_lattice_sha256": policy.exact_budget_lattice_sha256,
        "attention_geometry_policy": selected_bank.attention_geometry_policy,
        "hidden_size": hidden_size,
        "exact_target_unique_parameters": exact_target_unique_parameters,
        "cached_parent_traces_only": True,
        "parent_model_calls": 0,
        "child_model_calls": 0,
        "training_steps": 0,
    }
    return build_lfm2_heterogeneous_target_pins(
        hidden_size=hidden_size,
        vocab_size=runtime.vocab_size,
        conv_kernel_size=runtime.conv_kernel_size,
        rope_theta=runtime.rope_theta,
        norm_eps=runtime.norm_eps,
        max_position_embeddings=runtime.max_position_embeddings,
        use_cache=runtime.use_cache,
        attention_implementation=runtime.attention_implementation,
        bos_token_id=runtime.bos_token_id,
        eos_token_id=runtime.eos_token_id,
        pad_token_id=runtime.pad_token_id,
        block_auto_adjust_ff_dim=runtime.block_auto_adjust_ff_dim,
        target_vocab_evidence_sha256=runtime.target_vocab_evidence_sha256,
        runtime_evidence_sha256=canonical_sha256(runtime_body),
        token_bridge_evidence_sha256=runtime.token_bridge_evidence_sha256,
    )


def _policy_is_unattainable(error: Lfm2CachedHeterogeneousCapacityError) -> bool:
    message = str(error)
    return message.startswith(
        "The exact whole-child budget has no dense compact MLP/operator "
    ) or message.startswith(
        "The exact whole-child budget has no dense-or-single-factor compact "
        "MLP/operator "
    )


def build_lfm2_cached_global_topology_frontier(
    source_graph: ModelCircuitGraph,
    parent_cache: Lfm2CircuitTraceCache,
    *,
    runtime: Lfm2GlobalTargetRuntimePins,
    exact_target_unique_parameters: int,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    minimum_hidden_size: int = 32,
    hidden_size_multiple: int = 8,
    short_conv_internal_widths: Sequence[int] = (704, 1024),
    discover_exact_short_conv_writer_ranks: bool = False,
    attention_head_options: Sequence[tuple[int, int, int]] = (
        (12, 6, 64),
        (16, 8, 64),
    ),
    maximum_functional_rows_per_role: int = 2048,
    max_states: int = 100_000,
) -> Lfm2CachedGlobalTopologyFrontier:
    """Build and choose the complete exact child across all legal widths."""

    if not isinstance(source_graph, ModelCircuitGraph) or source_graph.family != "lfm2":
        raise TypeError("source_graph must be an inspected LFM2 ModelCircuitGraph.")
    if not isinstance(parent_cache, Lfm2CircuitTraceCache):
        raise TypeError("parent_cache must be Lfm2CircuitTraceCache.")
    if type(runtime) is not Lfm2GlobalTargetRuntimePins:
        raise TypeError("runtime must be Lfm2GlobalTargetRuntimePins.")
    if selective_session is not None and not isinstance(
        selective_session,
        Lfm2CircuitTraceSelectiveAuthenticatedSession,
    ):
        raise TypeError(
            "selective_session must be an authenticated selective session or None."
        )
    target = _positive_int(
        exact_target_unique_parameters,
        "exact_target_unique_parameters",
    )
    state_ceiling = _positive_int(max_states, "max_states")
    widths = enumerate_lfm2_global_hidden_sizes(
        source_graph,
        exact_target_unique_parameters=target,
        target_vocabulary_size=runtime.vocab_size,
        minimum_hidden_size=minimum_hidden_size,
        hidden_size_multiple=hidden_size_multiple,
    )
    layer_types, source_widths = _source_layer_contract(source_graph)
    policies: dict[int, Lfm2MlpWidthOptionPolicy] = {}
    rejected = []
    for hidden_size in widths:
        try:
            policies[hidden_size] = derive_lfm2_automatic_mlp_width_policy(
                layer_types=layer_types,
                source_widths=source_widths,
                short_conv_internal_widths=short_conv_internal_widths,
                attention_head_options=attention_head_options,
                target_hidden_size=hidden_size,
                target_vocab_size=runtime.vocab_size,
                exact_target_unique_parameters=target,
                conv_kernel_size=runtime.conv_kernel_size,
                discover_exact_short_conv_writer_ranks=(
                    discover_exact_short_conv_writer_ranks
                ),
            )
        except Lfm2CachedHeterogeneousCapacityError as error:
            if not _policy_is_unattainable(error):
                raise
            rejected.append(
                Lfm2GlobalTopologyRejectedWidth.build(
                    hidden_size,
                    "no_exact_integer_lattice",
                    str(error),
                )
            )
    if not policies:
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No legal residual width has an exact dense MLP/operator lattice."
        )
    union_widths = tuple(
        sorted(
            {
                width
                for policy in policies.values()
                for row in policy.widths_by_layer
                for width in row
            }
        )
    )
    union_writer_ranks = tuple(
        sorted(
            {
                rank
                for policy in policies.values()
                for rank in policy.short_conv_writer_ranks
            }
        )
    )
    session_context = (
        parent_cache.selective_authenticated_session()
        if selective_session is None
        else nullcontext(selective_session)
    )
    with session_context as session:
        session.verify()
        base_bank = build_lfm2_cached_heterogeneous_capacity_option_bank(
            source_graph,
            parent_cache,
            selective_session=session,
            short_conv_internal_widths=short_conv_internal_widths,
            short_conv_writer_ranks=union_writer_ranks,
            attention_head_options=attention_head_options,
            mlp_intermediate_sizes=union_widths,
            target_hidden_size=max(widths),
            conv_kernel_size=runtime.conv_kernel_size,
            attention_geometry_risk_enabled=True,
        )
        functional = build_lfm2_cached_functional_width_evidence(
            source_graph,
            parent_cache,
            widths,
            selective_session=session,
            maximum_rows_per_role=maximum_functional_rows_per_role,
        )
        session.verify()

    reports = []
    artifacts: dict[
        str,
        tuple[
            Lfm2HeterogeneousLayoutSelectionInput,
            Lfm2CachedHeterogeneousCapacityOptionBank,
            Lfm2HeterogeneousLayoutSelection,
        ],
    ] = {}
    for hidden_size, policy in sorted(policies.items()):
        bank = _retarget_capacity_option_bank(base_bank, policy)
        pins = _target_pins_for_width(
            hidden_size=hidden_size,
            runtime=runtime,
            source_graph=source_graph,
            parent_cache=parent_cache,
            base_bank=base_bank,
            selected_bank=bank,
            functional_evidence=functional,
            policy=policy,
            exact_target_unique_parameters=target,
        )
        selection_input = Lfm2HeterogeneousLayoutSelectionInput(
            target_pins=pins,
            layer_options=bank.layer_options,
            max_states=state_ceiling,
        )
        try:
            selection = selection_input.select(
                source_graph,
                exact_target_unique_parameters=target,
            )
        except Lfm2HeterogeneousLayoutUnattainable as error:
            rejected.append(
                Lfm2GlobalTopologyRejectedWidth.build(
                    hidden_size,
                    "exact_selector_unattainable",
                    str(error),
                )
            )
            continue
        score = functional.score_for(hidden_size)
        mean_component_risk = selection.budget_selection.total_predicted_risk / len(
            layer_types
        )
        report = Lfm2GlobalTopologyCandidateReport.build(
            hidden_size=hidden_size,
            exact_unique_parameters=target,
            target_layer_count=len(selection.descriptor.layers),
            functional_retention=score.role_balanced_functional_retention,
            functional_loss=score.role_balanced_functional_loss,
            mean_component_risk=mean_component_risk,
            functional_score_sha256=score.score_sha256,
            mlp_width_lattice_sha256=policy.exact_budget_lattice_sha256,
            capacity_option_bank_sha256=bank.evidence_sha256,
            selection_input_sha256=selection_input.fingerprint,
            descriptor_sha256=selection.descriptor.descriptor_sha256,
            layout_selection_receipt_sha256=selection.receipt.receipt_sha256,
        )
        reports.append(report)
        artifacts[report.candidate_sha256] = (selection_input, bank, selection)
    if not reports:
        raise Lfm2CachedGlobalTopologyFrontierUnattainable(
            "No legal residual width produced a complete exact physical child."
        )
    reports.sort(key=lambda value: value.hidden_size)
    rejected.sort(key=lambda value: value.hidden_size)
    selected = select_lfm2_global_topology_report(reports)
    selected_input, selected_bank, selected_layout = artifacts[
        selected.candidate_sha256
    ]
    values = {
        "source_graph_fingerprint": source_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "exact_target_unique_parameters": target,
        "source_layer_count": len(layer_types),
        "candidate_hidden_sizes": widths,
        "base_union_capacity_option_bank_sha256": base_bank.evidence_sha256,
        "functional_width_evidence": functional,
        "candidates": tuple(reports),
        "rejected_widths": tuple(rejected),
        "selected_candidate_sha256": selected.candidate_sha256,
        "selected_input": selected_input,
        "selected_capacity_option_bank": selected_bank,
        "selected_layout": selected_layout,
    }
    provisional = object.__new__(Lfm2CachedGlobalTopologyFrontier)
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "frontier_sha256", "0" * 64)
    return Lfm2CachedGlobalTopologyFrontier(
        **values,
        frontier_sha256=canonical_sha256(provisional._body()),
    )


__all__ = [
    "GLOBAL_TOPOLOGY_SCORE_METHOD",
    "Lfm2CachedGlobalTopologyFrontier",
    "Lfm2CachedGlobalTopologyFrontierError",
    "Lfm2CachedGlobalTopologyFrontierUnattainable",
    "Lfm2GlobalTargetRuntimePins",
    "Lfm2GlobalTopologyCandidateReport",
    "Lfm2GlobalTopologyRejectedWidth",
    "build_lfm2_cached_global_topology_frontier",
    "select_lfm2_global_topology_report",
]
