"""Automatic target-family selection and exact child materialization.

The folding compiler must not require a human to construct a blank child.
This module defines the family-neutral request, adapter registry, independent
post-materialization verification, and auditable result used by the
end-to-end compiler. Family adapters own only topology/config construction;
they do not own folding mathematics or behavioral promotion.
"""

from __future__ import annotations

import functools
import hashlib
import math
import re
import sys
import types
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_circuit_ir import JsonScalar, ModelCircuitGraph, canonical_sha256
from .dynamic_fold_executor import fingerprint_model_state
from .dynamic_model_inspector import InspectorRegistry
from .dynamic_parameter_frontier import (
    CAP_TOKENIZER_BRIDGE,
    COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
    DenseComponentAllocationReceipt,
    DenseParameterBudgetRequest,
    DenseParameterFrontierResult,
    DynamicParameterFrontierCapabilityError,
    DenseTargetTopologyCatalogRegistry,
    DenseTargetTopologyCandidate,
    component_allocation_topology_fingerprint,
    select_best_executable_dense_candidate,
    synthesize_dense_parameter_frontier,
)
from .hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    TOKENIZER_BRIDGE_CONFIG_KEYS,
)


SCHEMA = "neural-forge-dynamic-target-synthesis-v2"
BLANK_PROVENANCE_SCHEMA = "neural-forge-compiler-blank-provenance-v1"
COMPILER_BLANK_METHOD = "compiler-in-place-zero-unique-parameters-v1"
INFERENCE_STATE_INVENTORY_SCHEMA = "neural-forge-inference-tensor-state-v1"
_REACHABLE_STATE_MAX_DEPTH = 64
_REACHABLE_STATE_MAX_OBJECTS = 250_000
_REACHABLE_STATE_MAX_CONTAINER_ITEMS = 1_000_000
_MODULE_REGISTRATION_ATTRIBUTES = frozenset({"_parameters", "_buffers", "_modules"})


class DynamicTargetSynthesisError(RuntimeError):
    """No adapter could create and independently prove the requested target."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicTargetSynthesisError(f"{name} must be a nonempty string.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicTargetSynthesisError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DynamicTargetSynthesisError(f"{name} must be a non-negative integer.")
    return value


def _scalar(value: Any, name: str) -> JsonScalar:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise DynamicTargetSynthesisError(f"{name} must be a JSON scalar.")
    if isinstance(value, float) and not math.isfinite(value):
        raise DynamicTargetSynthesisError(f"{name} must be finite.")
    return value


def _configuration(
    value: Mapping[str, JsonScalar] | Sequence[tuple[str, JsonScalar]] | None,
) -> tuple[tuple[str, JsonScalar], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    result = tuple(
        sorted(
            (
                _text(str(key), "configuration key"),
                _scalar(item, f"configuration {key}"),
            )
            for key, item in items
        )
    )
    if len({key for key, _item in result}) != len(result):
        raise DynamicTargetSynthesisError("Configuration keys must be unique.")
    return result


def _sha256_digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicTargetSynthesisError(f"{name} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True, slots=True)
class DynamicTargetRequest:
    """A model-family-neutral exact target request."""

    child_name: str
    target_family: str
    target_modalities: tuple[str, ...]
    exact_unique_parameters: int
    maximum_parameter_bytes: int | None = None
    tokenizer_fingerprint: str | None = None
    adapter_id: str | None = None
    fold_method_id: str | None = None
    blank_seed: int = 0
    configuration: tuple[tuple[str, JsonScalar], ...] = ()
    component_allocation_receipt: DenseComponentAllocationReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "child_name", _text(self.child_name, "child_name"))
        object.__setattr__(
            self,
            "target_family",
            _text(self.target_family, "target_family").lower(),
        )
        modalities = tuple(
            sorted(
                {
                    _text(value, "target modality").lower()
                    for value in self.target_modalities
                }
            )
        )
        if not modalities:
            raise DynamicTargetSynthesisError(
                "At least one target modality is required."
            )
        object.__setattr__(self, "target_modalities", modalities)
        object.__setattr__(
            self,
            "exact_unique_parameters",
            _positive_int(
                self.exact_unique_parameters,
                "exact_unique_parameters",
            ),
        )
        if self.maximum_parameter_bytes is not None:
            object.__setattr__(
                self,
                "maximum_parameter_bytes",
                _positive_int(
                    self.maximum_parameter_bytes,
                    "maximum_parameter_bytes",
                ),
            )
        if self.tokenizer_fingerprint is not None:
            value = self.tokenizer_fingerprint
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DynamicTargetSynthesisError(
                    "tokenizer_fingerprint must be a lowercase SHA-256 digest."
                )
        if self.adapter_id is not None:
            object.__setattr__(
                self,
                "adapter_id",
                _text(self.adapter_id, "adapter_id"),
            )
        if self.fold_method_id is not None:
            object.__setattr__(
                self,
                "fold_method_id",
                _text(self.fold_method_id, "fold_method_id"),
            )
        object.__setattr__(
            self,
            "blank_seed",
            _nonnegative_int(self.blank_seed, "blank_seed"),
        )
        configuration = _configuration(self.configuration)
        object.__setattr__(self, "configuration", configuration)
        configuration_values = dict(configuration)
        allocation_keys = {
            COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
            COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
            COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
        }
        present_allocation_keys = allocation_keys.intersection(configuration_values)
        receipt = self.component_allocation_receipt
        if receipt is None:
            if present_allocation_keys:
                raise DynamicTargetSynthesisError(
                    "Component-allocation configuration seals require their "
                    "authenticated DenseComponentAllocationReceipt binding."
                )
        elif not isinstance(receipt, DenseComponentAllocationReceipt):
            raise DynamicTargetSynthesisError(
                "component_allocation_receipt must be a "
                "DenseComponentAllocationReceipt."
            )
        elif (
            present_allocation_keys != allocation_keys
            or configuration_values.get(COMPONENT_ALLOCATION_POLICY_CONFIG_KEY)
            != receipt.policy
            or configuration_values.get(COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY)
            != receipt.fingerprint
            or configuration_values.get(COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY)
            != receipt.topology_configuration_sha256
            or component_allocation_topology_fingerprint(configuration)
            != receipt.topology_configuration_sha256
            or receipt.target_unique_parameters != self.exact_unique_parameters
        ):
            raise DynamicTargetSynthesisError(
                "Authenticated component-allocation receipt differs from the "
                "request seals, materializable topology, or exact budget."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "target_request",
            "child_name": self.child_name,
            "target_family": self.target_family,
            "target_modalities": list(self.target_modalities),
            "exact_unique_parameters": self.exact_unique_parameters,
            "maximum_parameter_bytes": self.maximum_parameter_bytes,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "adapter_id": self.adapter_id,
            "fold_method_id": self.fold_method_id,
            "blank_seed": self.blank_seed,
            "configuration": dict(self.configuration),
            "component_allocation_receipt": (
                None
                if self.component_allocation_receipt is None
                else self.component_allocation_receipt.as_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class PlannedBlankTargetGraphCertificate:
    """Weight-free graph/config contract for a ranked topology candidate."""

    family: str
    modalities: tuple[str, ...]
    logical_parameter_count: int
    unique_parameter_count: int
    native_tied_parameter_count: int
    layer_count: int
    configuration: tuple[tuple[str, JsonScalar], ...]
    ordered_source_spans: tuple[tuple[int, ...], ...]
    candidate_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "family",
            _text(self.family, "planned target family").lower(),
        )
        modalities = tuple(
            sorted(
                {
                    _text(value, "planned target modality").lower()
                    for value in self.modalities
                }
            )
        )
        if not modalities:
            raise DynamicTargetSynthesisError(
                "A planned blank target graph needs a modality."
            )
        object.__setattr__(self, "modalities", modalities)
        for name in (
            "logical_parameter_count",
            "unique_parameter_count",
            "layer_count",
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
        if (
            self.logical_parameter_count - self.unique_parameter_count
            != self.native_tied_parameter_count
        ):
            raise DynamicTargetSynthesisError(
                "Planned graph logical/unique counts disagree with native ties."
            )
        object.__setattr__(
            self,
            "configuration",
            _configuration(self.configuration),
        )
        spans = tuple(tuple(span) for span in self.ordered_source_spans)
        if len(spans) != self.layer_count:
            raise DynamicTargetSynthesisError(
                "Planned graph needs exactly one source span per target layer."
            )
        flattened = tuple(index for span in spans for index in span)
        if (
            any(not span for span in spans)
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in flattened
            )
            or flattened != tuple(range(len(flattened)))
        ):
            raise DynamicTargetSynthesisError(
                "Planned graph source spans must cover the source once in order."
            )
        object.__setattr__(self, "ordered_source_spans", spans)
        object.__setattr__(
            self,
            "candidate_fingerprint",
            _sha256_digest(
                self.candidate_fingerprint,
                "candidate_fingerprint",
            ),
        )
        configured_depth = dict(self.configuration).get("num_hidden_layers")
        if configured_depth is not None and configured_depth != self.layer_count:
            raise DynamicTargetSynthesisError(
                "Planned graph layer count differs from target configuration."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "planned_blank_target_graph",
            "family": self.family,
            "modalities": list(self.modalities),
            "logical_parameter_count": self.logical_parameter_count,
            "unique_parameter_count": self.unique_parameter_count,
            "native_tied_parameter_count": self.native_tied_parameter_count,
            "layer_count": self.layer_count,
            "configuration": dict(self.configuration),
            "ordered_source_spans": [
                list(span) for span in self.ordered_source_spans
            ],
            "candidate_fingerprint": self.candidate_fingerprint,
            "topology_only": True,
            "model_or_weights_materialized": False,
            "blank_parameters_required": True,
            "independent_concrete_graph_validation_required": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class DynamicTargetTopologyCertificate:
    """Selected-frontier provenance bound to an existing target request."""

    parent_graph_fingerprint: str
    frontier_fingerprint: str
    selected_candidate_rank: int
    selected_candidate_fingerprint: str
    selection_policy: str
    target_request: DynamicTargetRequest
    planned_graph: PlannedBlankTargetGraphCertificate
    component_allocation_receipt: DenseComponentAllocationReceipt | None = None

    def __post_init__(self) -> None:
        for name in (
            "parent_graph_fingerprint",
            "frontier_fingerprint",
            "selected_candidate_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _sha256_digest(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "selected_candidate_rank",
            _nonnegative_int(
                self.selected_candidate_rank,
                "selected_candidate_rank",
            ),
        )
        if self.selection_policy not in {
            "explicit_rank_from_deterministic_frontier",
            "supported_pareto_balanced_executable_v1",
            "architecture_discovered_component_risk_executable_v1",
        }:
            raise DynamicTargetSynthesisError(
                "Unknown topology-candidate selection policy."
            )
        if not isinstance(self.target_request, DynamicTargetRequest):
            raise DynamicTargetSynthesisError(
                "target_request must be DynamicTargetRequest."
            )
        if not isinstance(
            self.planned_graph,
            PlannedBlankTargetGraphCertificate,
        ):
            raise DynamicTargetSynthesisError(
                "planned_graph must be PlannedBlankTargetGraphCertificate."
            )
        if (
            self.selected_candidate_fingerprint
            != self.planned_graph.candidate_fingerprint
            or self.target_request.target_family != self.planned_graph.family
            or self.target_request.target_modalities
            != self.planned_graph.modalities
            or self.target_request.exact_unique_parameters
            != self.planned_graph.unique_parameter_count
            or self.target_request.configuration
            != self.planned_graph.configuration
        ):
            raise DynamicTargetSynthesisError(
                "Target topology certificate fields are not bound to one "
                "candidate/request."
            )
        receipt = self.component_allocation_receipt
        configuration = dict(self.target_request.configuration)
        if receipt is None:
            if self.target_request.component_allocation_receipt is not None or any(
                key in configuration
                for key in (
                    COMPONENT_ALLOCATION_POLICY_CONFIG_KEY,
                    COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY,
                    COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY,
                )
            ):
                raise DynamicTargetSynthesisError(
                    "Target certificate lacks the component-allocation receipt "
                    "named by its request."
                )
        elif (
            not isinstance(receipt, DenseComponentAllocationReceipt)
            or receipt.source_graph_fingerprint
            != self.parent_graph_fingerprint
            or receipt.target_unique_parameters
            != self.planned_graph.unique_parameter_count
            or receipt.target_logical_parameters
            != self.planned_graph.logical_parameter_count
            or configuration.get(COMPONENT_ALLOCATION_POLICY_CONFIG_KEY)
            != receipt.policy
            or configuration.get(COMPONENT_ALLOCATION_RECEIPT_CONFIG_KEY)
            != receipt.fingerprint
            or configuration.get(COMPONENT_ALLOCATION_TOPOLOGY_CONFIG_KEY)
            != receipt.topology_configuration_sha256
            or self.target_request.component_allocation_receipt is None
            or self.target_request.component_allocation_receipt.fingerprint
            != receipt.fingerprint
        ):
            raise DynamicTargetSynthesisError(
                "Target certificate component-allocation receipt is not bound "
                "to its source/request/topology."
            )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def verify_materialized_target(
        self,
        target: "DynamicTargetSynthesisResult",
    ) -> str:
        """Bind a later concrete zero-blank result back to this preflight."""

        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicTargetSynthesisError(
                "Materialized target verification needs a synthesis result."
            )
        profile = target.resource_profile
        parent_graphs = dict(target.parent_graph_fingerprints)
        source_parent = dict(target.adapter_report).get("source_parent")
        receipt = self.component_allocation_receipt
        allocation_report_matches = receipt is None or (
            dict(target.adapter_report).get(
                "component_allocation_receipt_sha256"
            )
            == receipt.fingerprint
            and dict(target.adapter_report).get(
                "component_allocation_topology_sha256"
            )
            == receipt.topology_configuration_sha256
            and dict(target.adapter_report).get(
                "component_allocation_verified_by_target_adapter"
            )
            is True
        )
        if (
            target.request.fingerprint != self.target_request.fingerprint
            or target.graph.family != self.planned_graph.family
            or tuple(sorted(profile.modalities)) != self.planned_graph.modalities
            or target.graph.logical_parameter_count
            != self.planned_graph.logical_parameter_count
            or target.graph.unique_parameter_count
            != self.planned_graph.unique_parameter_count
            or len(target.graph.layer_indices) != self.planned_graph.layer_count
            or self.parent_graph_fingerprint not in parent_graphs.values()
            or (
                source_parent is not None
                and parent_graphs.get(str(source_parent))
                != self.parent_graph_fingerprint
            )
            or not allocation_report_matches
        ):
            raise DynamicTargetSynthesisError(
                "Concrete blank target or selected source parent differs from "
                "its planned graph/config certificate."
            )
        verify_compiler_authenticated_target(target)
        return canonical_sha256(
            {
                "topology_certificate": self.fingerprint,
                "materialized_target": target.fingerprint,
                "compiler_topology": target.topology_fingerprint,
            }
        )

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "target_topology_certificate",
            "parent_graph_fingerprint": self.parent_graph_fingerprint,
            "frontier_fingerprint": self.frontier_fingerprint,
            "selected_candidate_rank": self.selected_candidate_rank,
            "selected_candidate_fingerprint": (
                self.selected_candidate_fingerprint
            ),
            "selection_policy": self.selection_policy,
            "target_request": self.target_request.as_dict(),
            "planned_graph": self.planned_graph.as_dict(),
            "component_allocation_receipt": (
                None
                if self.component_allocation_receipt is None
                else self.component_allocation_receipt.as_dict()
            ),
            "weights_or_downloads_used": False,
            "ready_for_existing_target_synthesizer": True,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class BudgetedDynamicTargetPlan:
    """Both requested frontier evidence and the selected target certificate."""

    frontier: DenseParameterFrontierResult
    target_certificate: DynamicTargetTopologyCertificate

    def __post_init__(self) -> None:
        if not isinstance(self.frontier, DenseParameterFrontierResult):
            raise DynamicTargetSynthesisError(
                "frontier must be DenseParameterFrontierResult."
            )
        if not isinstance(
            self.target_certificate,
            DynamicTargetTopologyCertificate,
        ):
            raise DynamicTargetSynthesisError(
                "target_certificate must be DynamicTargetTopologyCertificate."
            )
        rank = self.target_certificate.selected_candidate_rank
        if rank >= len(self.frontier.candidates):
            raise DynamicTargetSynthesisError(
                "Selected topology rank is outside the frontier."
            )
        candidate = self.frontier.candidates[rank]
        planned = self.target_certificate.planned_graph
        if (
            self.frontier.fingerprint
            != self.target_certificate.frontier_fingerprint
            or candidate.fingerprint
            != self.target_certificate.selected_candidate_fingerprint
            or candidate.family != planned.family
            or candidate.modalities != planned.modalities
            or candidate.logical_parameters
            != planned.logical_parameter_count
            or candidate.unique_parameters
            != planned.unique_parameter_count
            or candidate.native_tied_parameter_count
            != planned.native_tied_parameter_count
            or candidate.configuration != planned.configuration
            or candidate.ordered_source_spans
            != planned.ordered_source_spans
            or (
                None
                if candidate.component_allocation_receipt is None
                else candidate.component_allocation_receipt.fingerprint
            )
            != (
                None
                if self.target_certificate.component_allocation_receipt is None
                else self.target_certificate.component_allocation_receipt.fingerprint
            )
        ):
            raise DynamicTargetSynthesisError(
                "Target certificate is not bound to the complete selected "
                "frontier candidate/rank."
            )

    @property
    def target_request(self) -> DynamicTargetRequest:
        return self.target_certificate.target_request

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict(include_fingerprint=False))

    def as_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "kind": "budgeted_dynamic_target_plan",
            "frontier": self.frontier.as_dict(),
            "target_certificate": self.target_certificate.as_dict(),
            "selected_rank_provenance_preserved": True,
            "weights_or_downloads_used": False,
        }
        if include_fingerprint:
            value["fingerprint"] = canonical_sha256(value)
        return value


def plan_budgeted_dynamic_target(
    parent_graph: ModelCircuitGraph,
    budget_request: DenseParameterBudgetRequest,
    *,
    child_name: str,
    selected_candidate_rank: int | None = None,
    prefer_component_risk: bool = True,
    tokenizer_fingerprint: str | None = None,
    adapter_id: str | None = None,
    fold_method_id: str | None = None,
    blank_seed: int = 0,
    catalogs: DenseTargetTopologyCatalogRegistry | None = None,
) -> BudgetedDynamicTargetPlan:
    """Plan one exact concrete-target request without loading any model."""

    if not isinstance(parent_graph, ModelCircuitGraph):
        raise DynamicTargetSynthesisError(
            "parent_graph must be an inspected ModelCircuitGraph."
        )
    if not isinstance(budget_request, DenseParameterBudgetRequest):
        raise DynamicTargetSynthesisError(
            "budget_request must be DenseParameterBudgetRequest."
        )
    if not isinstance(prefer_component_risk, bool):
        raise DynamicTargetSynthesisError(
            "prefer_component_risk must be boolean."
        )
    frontier = synthesize_dense_parameter_frontier(
        parent_graph,
        budget_request,
        catalogs=catalogs,
    )
    if selected_candidate_rank is None:
        try:
            executable_selection = select_best_executable_dense_candidate(
                frontier,
                prefer_component_risk=prefer_component_risk,
            )
        except DynamicParameterFrontierCapabilityError as exc:
            if frontier.candidates and all(
                candidate.tokenizer_bridge_required
                for candidate in frontier.candidates
            ):
                raise DynamicTargetSynthesisError(
                    "Selected topology requires a tokenizer bridge, but "
                    "tokenizer mapping is not implemented by the dense target "
                    "planner."
                ) from exc
            raise DynamicTargetSynthesisError(
                "No supported-Pareto dense topology is executable by the "
                "currently registered compiler capabilities."
            ) from exc
        rank = executable_selection.selected_candidate_rank
        selection_policy = executable_selection.selection_policy
    else:
        rank = _nonnegative_int(
            selected_candidate_rank,
            "selected_candidate_rank",
        )
        selection_policy = "explicit_rank_from_deterministic_frontier"
    if rank >= len(frontier.candidates):
        raise DynamicTargetSynthesisError(
            "selected_candidate_rank is outside the ranked frontier."
        )
    candidate: DenseTargetTopologyCandidate = frontier.candidates[rank]
    if not candidate.at_or_below_budget:
        raise DynamicTargetSynthesisError(
            "Automatic target synthesis refuses an over-budget nearest "
            "diagnostic candidate."
        )
    selected_tokenizer_fingerprint = tokenizer_fingerprint
    if candidate.tokenizer_bridge_required:
        configuration = dict(candidate.configuration)
        if (
            not candidate.compiler_executable
            or CAP_TOKENIZER_BRIDGE
            not in candidate.declared_compiler_capabilities
            or CAP_TOKENIZER_BRIDGE
            in candidate.missing_compiler_capabilities
            or not TOKENIZER_BRIDGE_CONFIG_KEYS.issubset(configuration)
        ):
            raise DynamicTargetSynthesisError(
                "Selected topology requires a tokenizer bridge, but no "
                "verified bridge artifact capability is bound to the planner."
            )
        parent_semantics = configuration[
            PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY
        ]
        child_semantics = configuration[
            CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY
        ]
        if (
            parent_graph.tokenizer_fingerprint != parent_semantics
            or not isinstance(child_semantics, str)
        ):
            raise DynamicTargetSynthesisError(
                "Selected tokenizer bridge differs from the inspected parent "
                "or lacks child semantics."
            )
        if (
            selected_tokenizer_fingerprint is not None
            and selected_tokenizer_fingerprint != child_semantics
        ):
            raise DynamicTargetSynthesisError(
                "Requested child tokenizer differs from the selected strict "
                "tokenizer bridge."
            )
        selected_tokenizer_fingerprint = child_semantics
    target_request = candidate.to_target_request(
        child_name=child_name,
        tokenizer_fingerprint=selected_tokenizer_fingerprint,
        adapter_id=adapter_id,
        fold_method_id=fold_method_id,
        blank_seed=blank_seed,
    )
    planned_graph = PlannedBlankTargetGraphCertificate(
        family=candidate.family,
        modalities=candidate.modalities,
        logical_parameter_count=candidate.logical_parameters,
        unique_parameter_count=candidate.unique_parameters,
        native_tied_parameter_count=candidate.native_tied_parameter_count,
        layer_count=len(candidate.ordered_source_spans),
        configuration=candidate.configuration,
        ordered_source_spans=candidate.ordered_source_spans,
        candidate_fingerprint=candidate.fingerprint,
    )
    certificate = DynamicTargetTopologyCertificate(
        parent_graph_fingerprint=parent_graph.fingerprint,
        frontier_fingerprint=frontier.fingerprint,
        selected_candidate_rank=rank,
        selected_candidate_fingerprint=candidate.fingerprint,
        selection_policy=selection_policy,
        target_request=target_request,
        planned_graph=planned_graph,
        component_allocation_receipt=candidate.component_allocation_receipt,
    )
    return BudgetedDynamicTargetPlan(
        frontier=frontier,
        target_certificate=certificate,
    )


@dataclass(frozen=True, slots=True)
class DynamicTargetResourceProfile:
    unique_parameters: int
    parameter_bytes: int
    family: str
    modalities: tuple[str, ...]
    layer_count: int
    serialized_state_elements: int = 0
    serialized_state_bytes: int = 0
    persistent_buffer_elements: int = 0
    persistent_buffer_bytes: int = 0
    runtime_state_elements: int = 0
    runtime_state_bytes: int = 0
    runtime_buffer_elements: int = 0
    runtime_buffer_bytes: int = 0
    nonpersistent_buffer_elements: int = 0
    nonpersistent_buffer_bytes: int = 0
    runtime_named_buffer_count: int = 0
    nonpersistent_named_buffer_count: int = 0
    runtime_unique_buffer_view_count: int = 0
    runtime_tensor_state_fingerprint: str = "0" * 64

    @classmethod
    def from_graph(
        cls,
        graph: ModelCircuitGraph,
        model: Any | None = None,
    ) -> "DynamicTargetResourceProfile":
        inventory = (
            {
                "serialized_state_elements": graph.unique_parameter_count,
                "serialized_state_bytes": graph.unique_parameter_bytes,
                "persistent_buffer_elements": 0,
                "persistent_buffer_bytes": 0,
                "runtime_state_elements": graph.unique_parameter_count,
                "runtime_state_bytes": graph.unique_parameter_bytes,
                "runtime_buffer_elements": 0,
                "runtime_buffer_bytes": 0,
                "nonpersistent_buffer_elements": 0,
                "nonpersistent_buffer_bytes": 0,
                "runtime_named_buffer_count": 0,
                "nonpersistent_named_buffer_count": 0,
                "runtime_unique_buffer_view_count": 0,
                "runtime_tensor_state_fingerprint": canonical_sha256(
                    {
                        "schema": INFERENCE_STATE_INVENTORY_SCHEMA,
                        "kind": "graph_only_inventory",
                        "graph_fingerprint": graph.fingerprint,
                    }
                ),
            }
            if model is None
            else serialized_inference_state_inventory(model)
        )
        return cls(
            unique_parameters=graph.unique_parameter_count,
            parameter_bytes=graph.unique_parameter_bytes,
            family=graph.family,
            modalities=tuple(sorted({value.modality for value in graph.interfaces})),
            layer_count=len(graph.layer_indices),
            **inventory,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "unique_parameters": self.unique_parameters,
            "parameter_bytes": self.parameter_bytes,
            "family": self.family,
            "modalities": list(self.modalities),
            "layer_count": self.layer_count,
            "serialized_state_elements": self.serialized_state_elements,
            "serialized_state_bytes": self.serialized_state_bytes,
            "persistent_buffer_elements": self.persistent_buffer_elements,
            "persistent_buffer_bytes": self.persistent_buffer_bytes,
            "runtime_state_elements": self.runtime_state_elements,
            "runtime_state_bytes": self.runtime_state_bytes,
            "runtime_buffer_elements": self.runtime_buffer_elements,
            "runtime_buffer_bytes": self.runtime_buffer_bytes,
            "nonpersistent_buffer_elements": self.nonpersistent_buffer_elements,
            "nonpersistent_buffer_bytes": self.nonpersistent_buffer_bytes,
            "runtime_named_buffer_count": self.runtime_named_buffer_count,
            "nonpersistent_named_buffer_count": (self.nonpersistent_named_buffer_count),
            "runtime_unique_buffer_view_count": (self.runtime_unique_buffer_view_count),
            "runtime_tensor_state_fingerprint": (self.runtime_tensor_state_fingerprint),
        }

    def inference_state_inventory(self) -> dict[str, int | str]:
        """Return only the compiler-authenticated inference-state inventory."""

        return {
            "serialized_state_elements": self.serialized_state_elements,
            "serialized_state_bytes": self.serialized_state_bytes,
            "persistent_buffer_elements": self.persistent_buffer_elements,
            "persistent_buffer_bytes": self.persistent_buffer_bytes,
            "runtime_state_elements": self.runtime_state_elements,
            "runtime_state_bytes": self.runtime_state_bytes,
            "runtime_buffer_elements": self.runtime_buffer_elements,
            "runtime_buffer_bytes": self.runtime_buffer_bytes,
            "nonpersistent_buffer_elements": self.nonpersistent_buffer_elements,
            "nonpersistent_buffer_bytes": self.nonpersistent_buffer_bytes,
            "runtime_named_buffer_count": self.runtime_named_buffer_count,
            "nonpersistent_named_buffer_count": (self.nonpersistent_named_buffer_count),
            "runtime_unique_buffer_view_count": (self.runtime_unique_buffer_view_count),
            "runtime_tensor_state_fingerprint": (self.runtime_tensor_state_fingerprint),
        }


@dataclass(frozen=True, slots=True)
class StructuralBufferPolicy:
    """Versioned adapter/family declaration of structural nonzero buffers."""

    policy_id: str
    version: int
    adapter_id: str
    adapter_version: int
    entries: tuple[tuple[str, str], ...] = ()
    source: str = "adapter-declared"
    compiler_reviewed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "adapter_id", _text(self.adapter_id, "adapter_id"))
        object.__setattr__(self, "source", _text(self.source, "policy source"))
        if not isinstance(self.compiler_reviewed, bool):
            raise DynamicTargetSynthesisError("compiler_reviewed must be a boolean.")
        object.__setattr__(
            self,
            "version",
            _positive_int(self.version, "structural buffer policy version"),
        )
        object.__setattr__(
            self,
            "adapter_version",
            _positive_int(self.adapter_version, "adapter_version"),
        )
        normalized = tuple(
            sorted(
                (
                    _text(path, "structural buffer path"),
                    _text(role, f"structural buffer role for {path}"),
                )
                for path, role in self.entries
            )
        )
        if len({path for path, _role in normalized}) != len(normalized):
            raise DynamicTargetSynthesisError(
                "Structural buffer policy paths must be unique."
            )
        object.__setattr__(self, "entries", normalized)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-structural-buffer-policy-v1",
            "policy_id": self.policy_id,
            "version": self.version,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "source": self.source,
            "compiler_reviewed": self.compiler_reviewed,
            "entries": [{"path": path, "role": role} for path, role in self.entries],
        }


@dataclass(frozen=True, slots=True)
class StructuralBufferAudit:
    """Compiler observation binding one policy to live registered buffers."""

    policy_digest: str
    nonzero_buffer_count: int
    nonzero_buffer_elements: int
    nonzero_buffer_bytes: int
    nonzero_buffer_paths: tuple[str, ...]
    registered_buffer_manifest_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_digest",
            _sha256_digest(self.policy_digest, "policy_digest"),
        )
        object.__setattr__(
            self,
            "registered_buffer_manifest_fingerprint",
            _sha256_digest(
                self.registered_buffer_manifest_fingerprint,
                "registered_buffer_manifest_fingerprint",
            ),
        )
        for field_name in (
            "nonzero_buffer_count",
            "nonzero_buffer_elements",
            "nonzero_buffer_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        paths = tuple(
            sorted(
                _text(path, "nonzero buffer path") for path in self.nonzero_buffer_paths
            )
        )
        if len(set(paths)) != len(paths) or len(paths) != self.nonzero_buffer_count:
            raise DynamicTargetSynthesisError(
                "Structural buffer audit paths must be unique and match its count."
            )
        object.__setattr__(self, "nonzero_buffer_paths", paths)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "neural-forge-structural-buffer-audit-v1",
            "policy_digest": self.policy_digest,
            "nonzero_buffer_count": self.nonzero_buffer_count,
            "nonzero_buffer_elements": self.nonzero_buffer_elements,
            "nonzero_buffer_bytes": self.nonzero_buffer_bytes,
            "nonzero_buffer_paths": list(self.nonzero_buffer_paths),
            "registered_buffer_manifest_fingerprint": (
                self.registered_buffer_manifest_fingerprint
            ),
        }


@dataclass(frozen=True, slots=True)
class CompilerBlankProvenance:
    """Canonical evidence for compiler-owned in-place blank construction."""

    method: str
    compiler_created_blank: bool
    parent_child_alias_free: bool
    zeroed_under_no_grad: bool
    dense_materialized_parameters: bool
    all_parameters_zero: bool
    parameter_gradients_absent: bool
    runtime_buffers_preserved: bool
    unique_parameter_object_count: int
    unique_parameter_storage_count: int
    unique_parameter_element_count: int
    unique_parameter_bytes: int
    buffer_tensor_count: int
    parameter_layout_fingerprint: str
    buffer_layout_fingerprint: str
    buffer_state_fingerprint_before: str
    buffer_state_fingerprint_after: str
    blank_state_fingerprint: str
    structural_buffer_policy_digest: str | None = None
    structural_buffer_audit_digest: str | None = None
    structural_buffer_policy_verified: bool = False
    nonzero_structural_buffer_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _text(self.method, "blank method"))
        boolean_fields = (
            "compiler_created_blank",
            "parent_child_alias_free",
            "zeroed_under_no_grad",
            "dense_materialized_parameters",
            "all_parameters_zero",
            "parameter_gradients_absent",
            "runtime_buffers_preserved",
            "structural_buffer_policy_verified",
        )
        for field_name in boolean_fields:
            if not isinstance(getattr(self, field_name), bool):
                raise DynamicTargetSynthesisError(f"{field_name} must be a boolean.")
        count_fields = (
            "unique_parameter_object_count",
            "unique_parameter_storage_count",
            "unique_parameter_element_count",
            "unique_parameter_bytes",
            "buffer_tensor_count",
            "nonzero_structural_buffer_count",
        )
        for field_name in count_fields:
            object.__setattr__(
                self,
                field_name,
                _nonnegative_int(getattr(self, field_name), field_name),
            )
        digest_fields = (
            "parameter_layout_fingerprint",
            "buffer_layout_fingerprint",
            "buffer_state_fingerprint_before",
            "buffer_state_fingerprint_after",
            "blank_state_fingerprint",
        )
        for field_name in digest_fields:
            object.__setattr__(
                self,
                field_name,
                _sha256_digest(getattr(self, field_name), field_name),
            )
        for field_name in (
            "structural_buffer_policy_digest",
            "structural_buffer_audit_digest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _sha256_digest(value, field_name),
                )
        if self.compiler_created_blank:
            required = (
                self.parent_child_alias_free,
                self.zeroed_under_no_grad,
                self.dense_materialized_parameters,
                self.all_parameters_zero,
                self.parameter_gradients_absent,
                self.runtime_buffers_preserved,
            )
            if not all(required):
                raise DynamicTargetSynthesisError(
                    "Compiler-created blank provenance cannot contain an "
                    "unverified safety property."
                )
            if (
                self.unique_parameter_object_count <= 0
                or self.unique_parameter_storage_count <= 0
                or self.unique_parameter_element_count <= 0
                or self.unique_parameter_bytes <= 0
            ):
                raise DynamicTargetSynthesisError(
                    "A compiler-created blank must contain concrete learned parameters."
                )
            if (
                self.buffer_state_fingerprint_before
                != self.buffer_state_fingerprint_after
            ):
                raise DynamicTargetSynthesisError(
                    "Runtime buffers changed during compiler blank creation."
                )
            if self.nonzero_structural_buffer_count > 0 and (
                not self.structural_buffer_policy_verified
                or self.structural_buffer_policy_digest is None
                or self.structural_buffer_audit_digest is None
            ):
                raise DynamicTargetSynthesisError(
                    "Nonzero compiler-blank buffers require a verified structural "
                    "buffer policy and audit."
                )

    @classmethod
    def unverified_external(
        cls,
        blank_state_fingerprint: str,
    ) -> "CompilerBlankProvenance":
        """Represent a directly constructed legacy/test result without overclaiming."""

        marker = canonical_sha256(
            {
                "schema": BLANK_PROVENANCE_SCHEMA,
                "kind": "unverified_external_target",
                "blank_state_fingerprint": _sha256_digest(
                    blank_state_fingerprint,
                    "blank_state_fingerprint",
                ),
            }
        )
        return cls(
            method="unverified-external-target",
            compiler_created_blank=False,
            parent_child_alias_free=False,
            zeroed_under_no_grad=False,
            dense_materialized_parameters=False,
            all_parameters_zero=False,
            parameter_gradients_absent=False,
            runtime_buffers_preserved=False,
            unique_parameter_object_count=0,
            unique_parameter_storage_count=0,
            unique_parameter_element_count=0,
            unique_parameter_bytes=0,
            buffer_tensor_count=0,
            parameter_layout_fingerprint=marker,
            buffer_layout_fingerprint=marker,
            buffer_state_fingerprint_before=marker,
            buffer_state_fingerprint_after=marker,
            blank_state_fingerprint=blank_state_fingerprint,
            structural_buffer_policy_digest=None,
            structural_buffer_audit_digest=None,
            structural_buffer_policy_verified=False,
            nonzero_structural_buffer_count=0,
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BLANK_PROVENANCE_SCHEMA,
            "method": self.method,
            "compiler_created_blank": self.compiler_created_blank,
            "parent_child_alias_free": self.parent_child_alias_free,
            "zeroed_under_no_grad": self.zeroed_under_no_grad,
            "dense_materialized_parameters": self.dense_materialized_parameters,
            "all_parameters_zero": self.all_parameters_zero,
            "parameter_gradients_absent": self.parameter_gradients_absent,
            "runtime_buffers_preserved": self.runtime_buffers_preserved,
            "unique_parameter_object_count": self.unique_parameter_object_count,
            "unique_parameter_storage_count": self.unique_parameter_storage_count,
            "unique_parameter_element_count": self.unique_parameter_element_count,
            "unique_parameter_bytes": self.unique_parameter_bytes,
            "buffer_tensor_count": self.buffer_tensor_count,
            "parameter_layout_fingerprint": self.parameter_layout_fingerprint,
            "buffer_layout_fingerprint": self.buffer_layout_fingerprint,
            "buffer_state_fingerprint_before": self.buffer_state_fingerprint_before,
            "buffer_state_fingerprint_after": self.buffer_state_fingerprint_after,
            "blank_state_fingerprint": self.blank_state_fingerprint,
            "structural_buffer_policy_digest": self.structural_buffer_policy_digest,
            "structural_buffer_audit_digest": self.structural_buffer_audit_digest,
            "structural_buffer_policy_verified": (
                self.structural_buffer_policy_verified
            ),
            "nonzero_structural_buffer_count": self.nonzero_structural_buffer_count,
            "adapter_report_trusted_for_blank": False,
        }


@dataclass(frozen=True, slots=True)
class AdapterMaterialization:
    """Untrusted adapter output before independent inspection."""

    model: Any
    graph: ModelCircuitGraph
    topology_fingerprint: str
    report: Mapping[str, JsonScalar]
    adapter_payload: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ModelCircuitGraph):
            raise DynamicTargetSynthesisError(
                "Adapter materialization graph is invalid."
            )
        if (
            not isinstance(self.topology_fingerprint, str)
            or len(self.topology_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.topology_fingerprint
            )
        ):
            raise DynamicTargetSynthesisError(
                "topology_fingerprint must be a lowercase SHA-256 digest."
            )
        object.__setattr__(
            self,
            "report",
            MappingProxyType(
                {
                    _text(str(key), "adapter report key"): _scalar(
                        value,
                        f"adapter report {key}",
                    )
                    for key, value in self.report.items()
                }
            ),
        )


def _named_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise DynamicTargetSynthesisError(
            "A materialized target and every parent must expose named_modules()."
        )
    try:
        return tuple(method(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        return tuple(method())


def _named_tensors(model: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise DynamicTargetSynthesisError(
            f"A materialized target and every parent must expose {method_name}()."
        )
    try:
        values = tuple(method(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        values = tuple(method())
    for name, _value in values:
        if not isinstance(name, str) or not name:
            raise DynamicTargetSynthesisError(
                f"{method_name}() returned an invalid state path."
            )
    return values


def _dense_storage_identity(value: Any, *, label: str) -> tuple[Any, ...]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for compiler-owned blank construction."
        ) from exc
    if not isinstance(value, torch.Tensor):
        raise DynamicTargetSynthesisError(f"{label} is not a tensor.")
    if value.device.type == "meta" or value.layout != torch.strided:
        raise DynamicTargetSynthesisError(
            f"{label} must be a materialized dense tensor."
        )
    try:
        storage = value.untyped_storage()
        pointer = int(storage.data_ptr())
        storage_bytes = int(storage.nbytes())
        storage_handle = int(getattr(storage, "_cdata", 0))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise DynamicTargetSynthesisError(
            f"Could not authenticate storage for {label}."
        ) from exc
    device = (str(value.device.type), value.device.index)
    if pointer:
        return ("storage", *device, pointer, storage_bytes)
    if storage_handle:
        return ("storage_handle", *device, storage_handle, storage_bytes)
    return ("storage_object", *device, id(storage), storage_bytes)


def _parameter_view_identity(value: Any, *, label: str) -> tuple[Any, ...]:
    return (
        *_dense_storage_identity(value, label=label),
        str(value.dtype),
        int(value.storage_offset()),
        tuple(int(size) for size in value.shape),
        tuple(int(stride) for stride in value.stride()),
    )


def _stable_container_key(value: Any) -> str | None:
    """Return a process-independent path token for unordered/container keys."""

    if value is None:
        return "none"
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value!r}"
    if isinstance(value, complex):
        return f"complex:{value!r}"
    if isinstance(value, str):
        return f"str:{value!r}"
    if isinstance(value, bytes):
        return f"bytes:{value.hex()}"
    if isinstance(value, type):
        return f"type:{value.__module__}.{value.__qualname__}"
    if isinstance(value, tuple):
        nested = tuple(_stable_container_key(item) for item in value)
        if any(item is None for item in nested):
            return None
        return "tuple:(" + ",".join(item for item in nested if item is not None) + ")"
    if isinstance(value, frozenset):
        nested = tuple(_stable_container_key(item) for item in value)
        if any(item is None for item in nested):
            return None
        return "frozenset:{" + ",".join(sorted(item for item in nested if item)) + "}"
    return None


def _raw_reachable_attributes(value: Any) -> tuple[tuple[str, Any], ...]:
    """Read instance dictionaries and concrete slots without invoking properties."""

    attributes: dict[str, Any] = {}
    try:
        instance_values = vars(value)
    except TypeError:
        instance_values = {}
    for name, item in instance_values.items():
        if isinstance(name, str):
            attributes[name] = item
    for owner_type in type(value).__mro__:
        slots = owner_type.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or name in attributes:
                continue
            try:
                attributes[name] = object.__getattribute__(value, name)
            except AttributeError:
                continue
            except Exception as exc:
                raise DynamicTargetSynthesisError(
                    "Reachable-state audit could not inspect slot "
                    f"{type(value).__module__}.{type(value).__qualname__}.{name}."
                ) from exc
    return tuple(sorted(attributes.items()))


def validate_registered_inference_state(
    model: Any,
    *,
    forbidden_models: Mapping[str, Any] | None = None,
) -> None:
    """Fail closed when inference-reachable Torch state evades registration.

    The traversal is compiler-bounded, cycle-safe, and deterministic. Every
    registered ``nn.Module`` is audited through its non-registration instance
    attributes. Mappings, sequences, sets, function captures, partials, and
    ordinary Python objects are followed without invoking user properties.
    Any reachable Tensor, Parameter, or Module must already occur in the
    model's canonical named registration surfaces.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for reachable inference-state auditing."
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise DynamicTargetSynthesisError(
            "Reachable inference-state auditing requires an nn.Module."
        )

    named_modules = tuple(sorted(_named_modules(model), key=lambda item: item[0]))
    named_parameters = _named_tensors(model, "named_parameters")
    named_buffers = _named_tensors(model, "named_buffers")
    registered_module_ids = {id(module) for _name, module in named_modules}
    if id(model) not in registered_module_ids:
        raise DynamicTargetSynthesisError(
            "named_modules() omitted the audited root model."
        )
    registered_tensor_ids = {
        id(value) for _name, value in (*named_parameters, *named_buffers)
    }

    forbidden_module_objects: dict[int, str] = {}
    forbidden_tensor_objects: dict[int, str] = {}
    forbidden_tensor_storages: dict[tuple[Any, ...], str] = {}
    for parent_name, parent in sorted((forbidden_models or {}).items()):
        for path, module in _named_modules(parent):
            forbidden_module_objects.setdefault(
                id(module),
                f"parent {parent_name!r} module {path or '<root>'!r}",
            )
        for kind, method_name in (
            ("Parameter", "named_parameters"),
            ("buffer", "named_buffers"),
        ):
            for path, value in _named_tensors(parent, method_name):
                description = f"parent {parent_name!r} {kind} {path!r}"
                forbidden_tensor_objects.setdefault(id(value), description)
                forbidden_tensor_storages.setdefault(
                    _dense_storage_identity(value, label=description),
                    description,
                )

    seen: set[int] = set()
    object_count = 0
    container_item_count = 0

    def claim(value: Any, path: str, depth: int) -> bool:
        nonlocal object_count
        if depth > _REACHABLE_STATE_MAX_DEPTH:
            raise DynamicTargetSynthesisError(
                f"Reachable inference-state audit exceeded its maximum depth at {path}."
            )
        identity = id(value)
        if identity in seen:
            return False
        seen.add(identity)
        object_count += 1
        if object_count > _REACHABLE_STATE_MAX_OBJECTS:
            raise DynamicTargetSynthesisError(
                f"Reachable inference-state audit exceeded its object bound at {path}."
            )
        return True

    def add_items(count: int, path: str) -> None:
        nonlocal container_item_count
        container_item_count += count
        if container_item_count > _REACHABLE_STATE_MAX_CONTAINER_ITEMS:
            raise DynamicTargetSynthesisError(
                "Reachable inference-state audit exceeded its container-item "
                f"bound at {path}."
            )

    def walk(value: Any, path: str, depth: int) -> None:
        if isinstance(value, torch.nn.Module):
            if id(value) in registered_module_ids:
                return
            parent = forbidden_module_objects.get(id(value))
            alias = "" if parent is None else f" and aliases {parent}"
            raise DynamicTargetSynthesisError(
                f"Unregistered nn.Module reachable at {path}{alias}."
            )
        if isinstance(value, torch.Tensor):
            if id(value) in registered_tensor_ids:
                return
            kind = "Parameter" if isinstance(value, torch.nn.Parameter) else "Tensor"
            parent = forbidden_tensor_objects.get(id(value))
            if parent is None:
                storage = _dense_storage_identity(
                    value,
                    label=f"unregistered {kind} at {path}",
                )
                parent = forbidden_tensor_storages.get(storage)
            alias = "" if parent is None else f" and aliases {parent} storage"
            raise DynamicTargetSynthesisError(
                f"Unregistered {kind} reachable at {path}{alias}."
            )
        if isinstance(
            value,
            (
                str,
                bytes,
                bytearray,
                memoryview,
                int,
                float,
                complex,
                bool,
                type(None),
                range,
                slice,
                type,
                types.CodeType,
                types.ModuleType,
                types.BuiltinFunctionType,
                types.BuiltinMethodType,
                weakref.ReferenceType,
                weakref.ProxyType,
                weakref.CallableProxyType,
            ),
        ):
            return
        if not claim(value, path, depth):
            return

        if isinstance(value, Mapping):
            add_items(len(value), path)
            keyed: list[tuple[str, Any]] = []
            for key, item in value.items():
                token = _stable_container_key(key)
                if token is None:
                    raise DynamicTargetSynthesisError(
                        f"Reachable-state mapping has a nondeterministic key at {path}."
                    )
                keyed.append((token, item))
            if len({token for token, _item in keyed}) != len(keyed):
                raise DynamicTargetSynthesisError(
                    "Reachable-state mapping has ambiguous deterministic keys at "
                    f"{path}."
                )
            for token, item in sorted(keyed, key=lambda entry: entry[0]):
                walk(item, f"{path}[{token}]", depth + 1)
            return
        if isinstance(value, Sequence):
            add_items(len(value), path)
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]", depth + 1)
            return
        if isinstance(value, (set, frozenset)):
            add_items(len(value), path)
            keyed = []
            for item in value:
                token = _stable_container_key(item)
                if token is None:
                    raise DynamicTargetSynthesisError(
                        f"Reachable-state set has a nondeterministic member at {path}."
                    )
                keyed.append((token, item))
            if len({token for token, _item in keyed}) != len(keyed):
                raise DynamicTargetSynthesisError(
                    "Reachable-state set has ambiguous deterministic members at "
                    f"{path}."
                )
            for token, item in sorted(keyed, key=lambda entry: entry[0]):
                walk(item, f"{path}[{token}]", depth + 1)
            return
        if isinstance(value, functools.partial):
            walk(value.func, f"{path}.func", depth + 1)
            walk(value.args, f"{path}.args", depth + 1)
            walk(value.keywords or {}, f"{path}.keywords", depth + 1)
            return
        if isinstance(value, types.MethodType):
            walk(value.__self__, f"{path}.__self__", depth + 1)
            walk(value.__func__, f"{path}.__func__", depth + 1)
            return
        if isinstance(value, types.FunctionType):
            walk(value.__defaults__ or (), f"{path}.__defaults__", depth + 1)
            walk(value.__kwdefaults__ or {}, f"{path}.__kwdefaults__", depth + 1)
            closure_values: list[Any] = []
            for cell in value.__closure__ or ():
                try:
                    closure_values.append(cell.cell_contents)
                except ValueError:
                    continue
            walk(tuple(closure_values), f"{path}.__closure__", depth + 1)
            walk(vars(value), f"{path}.__dict__", depth + 1)
            return

        attributes = _raw_reachable_attributes(value)
        add_items(len(attributes), path)
        for name, item in attributes:
            walk(item, f"{path}.{name}", depth + 1)

    unique_modules: dict[int, tuple[str, Any]] = {}
    for path, module in named_modules:
        unique_modules.setdefault(id(module), (path, module))
    for module_path, module in sorted(
        unique_modules.values(), key=lambda item: item[0]
    ):
        base_path = f"module[{module_path or '<root>'!r}]"
        attributes = tuple(
            (name, value)
            for name, value in _raw_reachable_attributes(module)
            if name not in _MODULE_REGISTRATION_ATTRIBUTES
        )
        add_items(len(attributes), base_path)
        for name, value in attributes:
            walk(value, f"{base_path}.{name}", 1)


def _rotary_structural_buffer_entries(model: Any) -> tuple[tuple[str, str], ...]:
    entries = []
    for path, _value in _named_tensors(model, "named_buffers"):
        segments = path.lower().split(".")
        if segments[-1] in {"inv_freq", "original_inv_freq"} and any(
            "rotary" in segment for segment in segments[:-1]
        ):
            entries.append((path, "rotary_frequency_geometry"))
    return tuple(sorted(entries))


def _has_exact_loaded_type(
    value: Any,
    *,
    module_name: str,
    class_name: str,
) -> bool:
    module = sys.modules.get(module_name)
    expected = None if module is None else getattr(module, class_name, None)
    return expected is not None and type(value) is expected


def _standard_transformers_llama_buffer_policy(
    *,
    adapter_id: str,
    adapter_version: int,
    model: Any,
) -> StructuralBufferPolicy | None:
    """Authenticate stock Transformers Llama rotary state from its config."""

    module_name = "transformers.models.llama.modeling_llama"
    if not _has_exact_loaded_type(
        model,
        module_name=module_name,
        class_name="LlamaForCausalLM",
    ):
        return None
    decoder = getattr(model, "model", None)
    rotary = getattr(decoder, "rotary_emb", None)
    if not _has_exact_loaded_type(
        rotary,
        module_name=module_name,
        class_name="LlamaRotaryEmbedding",
    ):
        return None
    observed = dict(_named_tensors(model, "named_buffers"))
    expected_paths = {
        "model.rotary_emb.inv_freq": "inv_freq",
        "model.rotary_emb.original_inv_freq": "original_inv_freq",
    }
    if not set(expected_paths).issubset(observed):
        return None
    try:
        import torch

        reference = type(rotary)(
            model.config,
            device=observed["model.rotary_emb.inv_freq"].device,
        )
        reference_buffers = dict(_named_tensors(reference, "named_buffers"))
        with torch.no_grad():
            for path, short_name in expected_paths.items():
                value = observed[path].detach()
                expected = (
                    reference_buffers[short_name]
                    .detach()
                    .to(
                        device=value.device,
                        dtype=value.dtype,
                    )
                )
                if not torch.equal(value, expected):
                    return None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None
    return StructuralBufferPolicy(
        policy_id="compiler.transformers.llama.rotary-geometry",
        version=1,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        entries=tuple(
            (path, "config_derived_rotary_frequency_geometry")
            for path in sorted(expected_paths)
        ),
        source=(
            "compiler_concrete_model_registry:"
            "transformers.models.llama.modeling_llama."
            "LlamaForCausalLM/LlamaRotaryEmbedding"
        ),
        compiler_reviewed=True,
    )


def _qwen3_codebook_runtime_buffer_policy(
    *,
    adapter_id: str,
    adapter_version: int,
    model: Any,
    graph_family: str,
) -> StructuralBufferPolicy | None:
    """Authenticate concrete codebook topology rather than an adapter claim."""

    if graph_family != "qwen3_codebook" or not _has_exact_loaded_type(
        model,
        module_name="transformers.models.qwen3.modeling_qwen3",
        class_name="Qwen3ForCausalLM",
    ):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency boundary
        return None

    attention = getattr(model, "attention_codebook", None)
    attention_registry = getattr(attention, "registry", None)
    neurons = getattr(model, "neuron_codebook", None)
    neuron_registry = getattr(neurons, "registry", None)
    report = getattr(model, "neural_forge_codebook_report", None)
    exact_runtime_types = (
        _has_exact_loaded_type(
            attention,
            module_name="neural_forge.hf_attention_codebook",
            class_name="GlobalAttentionCodebook",
        ),
        _has_exact_loaded_type(
            attention_registry,
            module_name="neural_forge.hf_attention_codebook",
            class_name="AttentionCodebookRegistry",
        ),
        _has_exact_loaded_type(
            neurons,
            module_name="neural_forge.hf_neuron_codebook",
            class_name="GlobalNeuronCodebook",
        ),
        _has_exact_loaded_type(
            neuron_registry,
            module_name="neural_forge.hf_neuron_codebook",
            class_name="NeuronCodebookRegistry",
        ),
        _has_exact_loaded_type(
            report,
            module_name="neural_forge.hf_qwen_codebook_model",
            class_name="Qwen3CodebookParameterReport",
        ),
    )
    if not all(exact_runtime_types):
        return None

    modules = _named_modules(model)
    required_modules = {
        "attention_codebook": attention,
        "attention_codebook.registry": attention_registry,
        "neuron_codebook": neurons,
        "neuron_codebook.registry": neuron_registry,
    }
    if any(
        not any(path == expected_path and value is expected for path, value in modules)
        for expected_path, expected in required_modules.items()
    ):
        return None
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) <= 0:
        return None
    logical_layers = len(layers)
    if (
        int(getattr(report, "logical_layers", -1)) != logical_layers
        or int(getattr(attention_registry, "logical_layers", -1)) != logical_layers
        or int(getattr(neuron_registry, "logical_layers", -1)) != logical_layers
        or len(attention) != logical_layers
        or len(neurons) != logical_layers
        or int(getattr(getattr(model, "config", None), "num_hidden_layers", -1))
        != logical_layers
        or getattr(getattr(model, "config", None), "model_type", None) != "qwen3"
    ):
        return None
    for index, layer in enumerate(layers):
        layer_attention = getattr(layer, "self_attn", None)
        expected_attention = attention[index]
        expected_neuron = neurons[index]
        if getattr(layer, "mlp", None) is not expected_neuron:
            return None
        if any(
            getattr(layer_attention, role, None)
            is not getattr(expected_attention, role)
            for role in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            return None

    buffers = dict(_named_tensors(model, "named_buffers"))
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }

    def cpu_integer(path: str) -> Any | None:
        value = buffers.get(path)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype not in integer_dtypes
            or value.device.type == "meta"
            or value.layout != torch.strided
            or value.numel() <= 0
        ):
            return None
        return value.detach().to(device="cpu", dtype=torch.int64)

    parent_rows = cpu_integer("parent_token_rows")
    target_vocabulary = int(getattr(report, "target_vocab_size", -1))
    source_vocabulary = int(getattr(report, "source_vocab_size", -1))
    if (
        parent_rows is None
        or parent_rows.ndim != 1
        or int(parent_rows.numel()) != target_vocabulary
        or target_vocabulary
        != int(getattr(getattr(model, "config", None), "vocab_size", -1))
        or source_vocabulary <= 0
        or int(parent_rows.min()) < 0
        or int(parent_rows.max()) >= source_vocabulary
        or int(torch.unique(parent_rows).numel()) != target_vocabulary
    ):
        return None

    attention_prefix = "attention_codebook.registry."
    attention_assignments = cpu_integer(attention_prefix + "assignments")
    attention_sources = cpu_integer(attention_prefix + "prototype_source_indices")
    attention_permutations = cpu_integer(attention_prefix + "group_permutations")
    attention_schema = cpu_integer(attention_prefix + "_registry_schema_version")
    attention_layout = cpu_integer(attention_prefix + "_registry_layout")
    prototype_q = getattr(attention_registry, "prototype_q_weight", None)
    if (
        attention_assignments is None
        or attention_sources is None
        or attention_permutations is None
        or attention_schema is None
        or attention_layout is None
        or not isinstance(prototype_q, torch.nn.Parameter)
        or prototype_q.ndim != 4
    ):
        return None
    groups = int(prototype_q.shape[0])
    attention_prototypes = int(prototype_q.shape[1])
    if (
        groups <= 0
        or attention_prototypes <= 0
        or tuple(attention_assignments.shape) != (logical_layers, groups)
        or tuple(attention_sources.shape) != (groups, attention_prototypes)
        or tuple(attention_permutations.shape) != (logical_layers, groups)
        or attention_schema.numel() != 1
        or int(attention_schema.item()) != 1
        or attention_layout.numel() != 1
        or int(attention_layout.item()) != 0
        or int(attention_assignments.min()) < 0
        or int(attention_assignments.max()) >= attention_prototypes
        or int(attention_sources.min()) < 0
        or int(attention_sources.max()) >= logical_layers * groups
        or int(torch.unique(attention_sources).numel()) != groups * attention_prototypes
    ):
        return None
    expected_groups = list(range(groups))
    if any(
        sorted(map(int, row)) != expected_groups
        for row in attention_permutations.tolist()
    ):
        return None
    for group_index in range(groups):
        for source_index in attention_sources[group_index]:
            layer_index, physical_group = divmod(int(source_index), groups)
            if int(attention_permutations[layer_index, group_index]) != physical_group:
                return None

    neuron_prefix = "neuron_codebook.registry."
    neuron_assignments = cpu_integer(neuron_prefix + "assignments")
    neuron_sources = cpu_integer(neuron_prefix + "prototype_source_indices")
    activation_code = cpu_integer(neuron_prefix + "_activation_code")
    prototype_gate = getattr(neuron_registry, "prototype_gate_weight", None)
    if (
        neuron_assignments is None
        or neuron_sources is None
        or activation_code is None
        or not isinstance(prototype_gate, torch.nn.Parameter)
        or prototype_gate.ndim != 2
    ):
        return None
    neuron_prototypes = int(prototype_gate.shape[0])
    intermediate = int(getattr(report, "intermediate_size", -1))
    activation_codes = {
        "identity": 0,
        "silu": 1,
        "gelu": 2,
        "gelu_tanh": 3,
        "relu": 4,
    }
    activation = getattr(neuron_registry, "activation", None)
    if (
        neuron_prototypes <= 0
        or intermediate <= 0
        or tuple(neuron_assignments.shape) != (logical_layers, intermediate)
        or tuple(neuron_sources.shape) != (neuron_prototypes,)
        or activation_code.numel() != 1
        or activation not in activation_codes
        or int(activation_code.item()) != activation_codes[activation]
        or int(neuron_assignments.min()) < 0
        or int(neuron_assignments.max()) >= neuron_prototypes
        or int(neuron_sources.min()) < 0
        or int(neuron_sources.max()) >= logical_layers * intermediate
        or int(torch.unique(neuron_sources).numel()) != neuron_prototypes
    ):
        return None

    if (
        not all(
            bool(getattr(report, field, False))
            for field in (
                "full_logical_depth",
                "exact_parent_token_rows",
                "tied_input_output_embeddings",
                "parameterless_layer_views",
                "single_registry_owner_per_codebook",
            )
        )
        or int(getattr(report, "parent_runtime_reference_count", -1)) != 0
    ):
        return None
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        return None

    rotary = getattr(backbone, "rotary_emb", None)
    if not _has_exact_loaded_type(
        rotary,
        module_name="transformers.models.qwen3.modeling_qwen3",
        class_name="Qwen3RotaryEmbedding",
    ):
        return None
    rotary_paths = {
        "model.rotary_emb.inv_freq": "inv_freq",
        "model.rotary_emb.original_inv_freq": "original_inv_freq",
    }
    if not set(rotary_paths).issubset(buffers):
        return None
    try:
        reference_rotary = type(rotary)(
            model.config,
            device=buffers["model.rotary_emb.inv_freq"].device,
        )
        reference_buffers = dict(_named_tensors(reference_rotary, "named_buffers"))
        with torch.no_grad():
            for path, short_name in rotary_paths.items():
                value = buffers[path].detach()
                expected = (
                    reference_buffers[short_name]
                    .detach()
                    .to(
                        device=value.device,
                        dtype=value.dtype,
                    )
                )
                if not torch.equal(value, expected):
                    return None
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return None

    entries = [
        (path, "config_derived_rotary_frequency_geometry") for path in rotary_paths
    ]
    entries.append(("parent_token_rows", "validated_parent_vocabulary_row_map"))
    entries.extend(
        (
            attention_prefix + name,
            role,
        )
        for name, role in (
            ("assignments", "validated_attention_prototype_assignments"),
            (
                "prototype_source_indices",
                "validated_attention_prototype_provenance",
            ),
            ("group_permutations", "validated_attention_group_permutations"),
            ("_registry_schema_version", "attention_registry_schema_marker"),
            ("_registry_layout", "attention_registry_layout_marker"),
        )
    )
    entries.extend(
        (
            neuron_prefix + name,
            role,
        )
        for name, role in (
            ("assignments", "validated_neuron_prototype_assignments"),
            (
                "prototype_source_indices",
                "validated_neuron_prototype_provenance",
            ),
            ("_activation_code", "validated_neuron_activation_marker"),
        )
    )
    return StructuralBufferPolicy(
        policy_id="compiler.neural_forge.qwen3.codebook-runtime-topology",
        version=1,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        entries=tuple(entries),
        source=(
            "compiler_concrete_runtime_registry:"
            "Qwen3ForCausalLM/GlobalAttentionCodebook/"
            "AttentionCodebookRegistry/GlobalNeuronCodebook/"
            "NeuronCodebookRegistry"
        ),
        compiler_reviewed=True,
    )


def _builtin_structural_buffer_policy(
    *,
    adapter: Any,
    model: Any,
    graph_family: str,
) -> StructuralBufferPolicy | None:
    """Return compiler-reviewed policies for the currently registered lanes."""

    adapter_id = str(adapter.adapter_id)
    adapter_version = int(adapter.version)
    concrete_model_policy = _standard_transformers_llama_buffer_policy(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        model=model,
    )
    if concrete_model_policy is not None:
        return concrete_model_policy
    codebook_runtime_policy = _qwen3_codebook_runtime_buffer_policy(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        model=model,
        graph_family=graph_family,
    )
    if codebook_runtime_policy is not None:
        return codebook_runtime_policy
    key = (adapter_id, adapter_version)
    reviewed_implementations = {
        ("neural_forge.lfm2.progressive_exact", 1): (
            "neural_forge.dynamic_lfm2_adapter",
            "Lfm2TargetSynthesisAdapter",
        ),
        ("qwen3-compensated-ragged-target", 1): (
            "neural_forge.dynamic_qwen_compensated_adapter",
            "Qwen3CompensatedTargetSynthesisAdapter",
        ),
        ("neural_forge.neo.native_vision", 1): (
            "neural_forge.dynamic_neo_adapter",
            "NeoNativeVisionTargetSynthesisAdapter",
        ),
    }
    implementation = reviewed_implementations.get(key)
    if implementation is None or not _has_exact_loaded_type(
        adapter,
        module_name=implementation[0],
        class_name=implementation[1],
    ):
        return None
    source = "compiler_builtin_registry:" + ".".join(implementation)
    rotary_only = {
        ("neural_forge.lfm2.progressive_exact", 1),
        ("qwen3-compensated-ragged-target", 1),
    }
    if key in rotary_only:
        return StructuralBufferPolicy(
            policy_id=f"compiler.{adapter_id}.rotary-geometry",
            version=1,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            entries=_rotary_structural_buffer_entries(model),
            source=source,
            compiler_reviewed=True,
        )
    if key == ("neural_forge.neo.native_vision", 1):
        entries: list[tuple[str, str]] = []
        vision_pattern = re.compile(
            r"^vision_model\.embeddings\.(?:cos_cached_[xy]|sin_cached_[xy])$"
        )
        decoder_pattern = re.compile(
            r"^language_model\.model\.layers\.\d+\.self_attn\."
            r"(?:rotary_emb|rotary_emb_hw)\."
            r"(?:inv_freq|original_inv_freq)$"
        )
        for path, _value in _named_tensors(model, "named_buffers"):
            if vision_pattern.fullmatch(path):
                entries.append((path, "vision_rotary_cos_sin_geometry"))
            elif decoder_pattern.fullmatch(path):
                entries.append((path, "decoder_rotary_frequency_geometry"))
        return StructuralBufferPolicy(
            policy_id="compiler.neural_forge.neo.native_vision.rotary-geometry",
            version=1,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            entries=tuple(sorted(set(entries))),
            source=source,
            compiler_reviewed=True,
        )
    return None


def validate_structural_runtime_buffers(
    model: Any,
    policy: StructuralBufferPolicy | None = None,
    *,
    expected_adapter_id: str | None = None,
    expected_adapter_version: int | None = None,
) -> StructuralBufferAudit:
    """Require explicit policy coverage for every nonzero registered buffer."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for structural-buffer validation."
        ) from exc
    buffers = _named_tensors(model, "named_buffers")
    _unique_tensor_view_inventory(buffers, label="registered-buffer")
    if policy is None:
        policy = StructuralBufferPolicy(
            policy_id="compiler.no-nonzero-runtime-buffers",
            version=1,
            adapter_id=expected_adapter_id or "unbound-runtime",
            adapter_version=expected_adapter_version or 1,
            entries=(),
            source="compiler_empty_policy",
            compiler_reviewed=True,
        )
    if not isinstance(policy, StructuralBufferPolicy):
        raise DynamicTargetSynthesisError(
            "structural_buffer_policy() must return StructuralBufferPolicy."
        )
    if expected_adapter_id is not None and policy.adapter_id != expected_adapter_id:
        raise DynamicTargetSynthesisError(
            "Structural buffer policy is bound to another target adapter."
        )
    if (
        expected_adapter_version is not None
        and policy.adapter_version != expected_adapter_version
    ):
        raise DynamicTargetSynthesisError(
            "Structural buffer policy is bound to another adapter version."
        )
    registered_paths = {path for path, _value in buffers}
    policy_entries = dict(policy.entries)
    unknown = set(policy_entries) - registered_paths
    if unknown:
        raise DynamicTargetSynthesisError(
            "Structural buffer policy references unregistered paths: "
            + ", ".join(sorted(unknown))
        )
    nonzero: list[tuple[str, Any]] = []
    for path, value in buffers:
        _dense_storage_identity(value, label=f"runtime buffer {path!r}")
        if int(torch.count_nonzero(value.detach()).item()) != 0:
            nonzero.append((path, value))
    uncovered = {path for path, _value in nonzero} - set(policy_entries)
    if uncovered:
        raise DynamicTargetSynthesisError(
            "Nonzero runtime buffers lack an explicit structural policy: "
            + ", ".join(sorted(uncovered))
        )
    if nonzero and not policy.compiler_reviewed:
        raise DynamicTargetSynthesisError(
            "Nonzero runtime buffers require a compiler-reviewed structural "
            f"policy; policy {policy.policy_id!r} came from {policy.source!r}."
        )
    manifest = {
        "schema": "neural-forge-registered-buffer-manifest-v1",
        "policy_digest": policy.digest,
        "buffers": [
            {
                "path": path,
                "role": policy_entries.get(path),
                "shape": [int(size) for size in value.shape],
                "stride": [int(stride) for stride in value.stride()],
                "storage_offset": int(value.storage_offset()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "nonzero": path in {name for name, _item in nonzero},
            }
            for path, value in sorted(buffers)
        ],
        "buffer_state_fingerprint": _tensor_content_fingerprint(
            buffers,
            domain="structural_buffer_policy",
        ),
    }
    return StructuralBufferAudit(
        policy_digest=policy.digest,
        nonzero_buffer_count=len(nonzero),
        nonzero_buffer_elements=sum(int(value.numel()) for _path, value in nonzero),
        nonzero_buffer_bytes=sum(
            int(value.numel()) * int(value.element_size()) for _path, value in nonzero
        ),
        nonzero_buffer_paths=tuple(path for path, _value in nonzero),
        registered_buffer_manifest_fingerprint=canonical_sha256(manifest),
    )


def _resolve_structural_buffer_policy(
    *,
    adapter: Any,
    request: DynamicTargetRequest,
    materialized: AdapterMaterialization,
) -> StructuralBufferPolicy:
    policy = _builtin_structural_buffer_policy(
        adapter=adapter,
        model=materialized.model,
        graph_family=materialized.graph.family,
    )
    if policy is not None:
        return policy
    factory = getattr(adapter, "structural_buffer_policy", None)
    if callable(factory):
        declared = factory(request, materialized)
        if declared is not None and not isinstance(declared, StructuralBufferPolicy):
            raise DynamicTargetSynthesisError(
                "structural_buffer_policy() returned an invalid policy."
            )
        if declared is not None:
            raise DynamicTargetSynthesisError(
                "Adapter-declared structural buffer policies are untrusted; "
                "the adapter implementation and policy must be added to the "
                "compiler-reviewed built-in registry."
            )
    return StructuralBufferPolicy(
        policy_id="compiler.no-nonzero-runtime-buffers",
        version=1,
        adapter_id=str(adapter.adapter_id),
        adapter_version=int(adapter.version),
        entries=(),
        source="compiler_empty_policy",
        compiler_reviewed=True,
    )


def _unique_tensor_view_inventory(
    values: Sequence[tuple[str, Any]],
    *,
    label: str,
) -> dict[tuple[Any, ...], tuple[int, int]]:
    unique: dict[tuple[Any, ...], tuple[int, int]] = {}
    backing_views: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for name, value in values:
        backing = _dense_storage_identity(value, label=f"{label} {name!r}")
        identity = _parameter_view_identity(value, label=f"{label} {name!r}")
        previous_view = backing_views.get(backing)
        if previous_view is not None and previous_view != identity:
            raise DynamicTargetSynthesisError(
                f"Distinct {label} tensor views share one backing storage; "
                f"offending path {name!r}."
            )
        backing_views[backing] = identity
        count_and_size = (int(value.numel()), int(value.element_size()))
        previous = unique.get(identity)
        if previous is not None and previous != count_and_size:
            raise DynamicTargetSynthesisError(
                f"Aliased {label} views have inconsistent sizes."
            )
        unique[identity] = count_and_size
    return unique


def _inventory_totals(
    unique: Mapping[tuple[Any, ...], tuple[int, int]],
) -> tuple[int, int]:
    return (
        sum(count for count, _size in unique.values()),
        sum(count * size for count, size in unique.values()),
    )


def _runtime_tensor_manifest(
    *,
    parameters: Sequence[tuple[str, Any]],
    buffers: Sequence[tuple[str, Any]],
    persistent_buffer_paths: frozenset[str],
    serialized_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Describe every runtime tensor path without binding process-local pointers."""

    object_groups: dict[int, int] = {}
    storage_groups: dict[tuple[Any, ...], int] = {}
    records: list[dict[str, Any]] = []
    typed_values = [(name, "parameter", True, value) for name, value in parameters] + [
        (name, "buffer", name in persistent_buffer_paths, value)
        for name, value in buffers
    ]
    for name, kind, persistent, value in sorted(
        typed_values,
        key=lambda item: (item[0], item[1]),
    ):
        storage = _dense_storage_identity(
            value,
            label=f"runtime {kind} {name!r}",
        )
        object_group = object_groups.setdefault(id(value), len(object_groups))
        storage_group = storage_groups.setdefault(storage, len(storage_groups))
        records.append(
            {
                "name": name,
                "kind": kind,
                "persistent": persistent,
                "shape": [int(size) for size in value.shape],
                "stride": [int(stride) for stride in value.stride()],
                "storage_offset": int(value.storage_offset()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "object_group": object_group,
                "storage_group": storage_group,
            }
        )
    return {
        "schema": INFERENCE_STATE_INVENTORY_SCHEMA,
        "serialized_paths": list(serialized_paths),
        "runtime_tensors": records,
    }


def serialized_inference_state_inventory(model: Any) -> dict[str, int | str]:
    """Inventory persistent serialization and complete runtime tensor state.

    ``state_dict`` omits buffers registered with ``persistent=False``.  Those
    tensors still affect inference and can carry an arbitrarily large payload,
    so the runtime totals and structural fingerprint deliberately enumerate
    every value returned by ``named_buffers(remove_duplicate=False)``.
    """

    validate_registered_inference_state(model)

    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise DynamicTargetSynthesisError(
            "Serialized-state accounting requires state_dict()."
        )
    try:
        state = state_dict(keep_vars=True)
    except TypeError:
        state = state_dict()
    if not isinstance(state, Mapping) or not state:
        raise DynamicTargetSynthesisError(
            "Serialized-state accounting requires a nonempty state mapping."
        )
    parameters = _named_tensors(model, "named_parameters")
    buffers = _named_tensors(model, "named_buffers")
    parameter_paths = {name for name, _value in parameters}
    serialized_values: list[tuple[str, Any]] = []
    persistent_buffer_values: list[tuple[str, Any]] = []
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name:
            raise DynamicTargetSynthesisError(
                "Serialized inference state contains an invalid path."
            )
        serialized_values.append((name, value))
        if name not in parameter_paths:
            persistent_buffer_values.append((name, value))
    persistent_buffer_paths = frozenset(
        name for name, _value in persistent_buffer_values
    )
    nonpersistent_buffer_values = [
        (name, value) for name, value in buffers if name not in persistent_buffer_paths
    ]
    unique_state = _unique_tensor_view_inventory(
        serialized_values,
        label="serialized-state",
    )
    unique_persistent_buffers = _unique_tensor_view_inventory(
        persistent_buffer_values,
        label="persistent-buffer",
    )
    unique_runtime_buffers = _unique_tensor_view_inventory(
        buffers,
        label="runtime-buffer",
    )
    unique_nonpersistent_buffers = _unique_tensor_view_inventory(
        nonpersistent_buffer_values,
        label="nonpersistent-buffer",
    )
    unique_runtime_state = _unique_tensor_view_inventory(
        (*parameters, *buffers),
        label="runtime-state",
    )
    serialized_elements, serialized_bytes = _inventory_totals(unique_state)
    persistent_elements, persistent_bytes = _inventory_totals(unique_persistent_buffers)
    runtime_elements, runtime_bytes = _inventory_totals(unique_runtime_state)
    runtime_buffer_elements, runtime_buffer_bytes = _inventory_totals(
        unique_runtime_buffers
    )
    nonpersistent_elements, nonpersistent_bytes = _inventory_totals(
        unique_nonpersistent_buffers
    )
    manifest = _runtime_tensor_manifest(
        parameters=parameters,
        buffers=buffers,
        persistent_buffer_paths=persistent_buffer_paths,
        serialized_paths=tuple(sorted(state)),
    )
    return {
        "serialized_state_elements": serialized_elements,
        "serialized_state_bytes": serialized_bytes,
        "persistent_buffer_elements": persistent_elements,
        "persistent_buffer_bytes": persistent_bytes,
        "runtime_state_elements": runtime_elements,
        "runtime_state_bytes": runtime_bytes,
        "runtime_buffer_elements": runtime_buffer_elements,
        "runtime_buffer_bytes": runtime_buffer_bytes,
        "nonpersistent_buffer_elements": nonpersistent_elements,
        "nonpersistent_buffer_bytes": nonpersistent_bytes,
        "runtime_named_buffer_count": len(buffers),
        "nonpersistent_named_buffer_count": len(nonpersistent_buffer_values),
        "runtime_unique_buffer_view_count": len(unique_runtime_buffers),
        "runtime_tensor_state_fingerprint": canonical_sha256(manifest),
    }


def _layout_fingerprint(
    values: tuple[tuple[str, Any], ...],
    *,
    domain: str,
) -> str:
    object_groups: dict[int, int] = {}
    storage_groups: dict[tuple[Any, ...], int] = {}
    records: list[dict[str, Any]] = []
    for name, value in sorted(values):
        storage = _dense_storage_identity(value, label=f"{domain} {name!r}")
        object_group = object_groups.setdefault(id(value), len(object_groups))
        storage_group = storage_groups.setdefault(storage, len(storage_groups))
        records.append(
            {
                "name": name,
                "shape": [int(size) for size in value.shape],
                "stride": [int(stride) for stride in value.stride()],
                "storage_offset": int(value.storage_offset()),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "requires_grad": bool(getattr(value, "requires_grad", False)),
                "object_group": object_group,
                "storage_group": storage_group,
            }
        )
    return canonical_sha256(
        {
            "schema": BLANK_PROVENANCE_SCHEMA,
            "kind": "state_layout",
            "domain": domain,
            "values": records,
        }
    )


def _tensor_content_fingerprint(
    values: tuple[tuple[str, Any], ...],
    *,
    domain: str,
) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for compiler-owned blank construction."
        ) from exc
    digest = hashlib.sha256()

    def update(payload: bytes | memoryview) -> None:
        length = payload.nbytes if isinstance(payload, memoryview) else len(payload)
        digest.update(int(length).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    update(BLANK_PROVENANCE_SCHEMA.encode("utf-8"))
    update(domain.encode("utf-8"))
    for name, value in sorted(values):
        _dense_storage_identity(value, label=f"{domain} {name!r}")
        tensor = value.detach().to(device="cpu").contiguous()
        update(name.encode("utf-8"))
        update(str(tuple(int(size) for size in tensor.shape)).encode("ascii"))
        update(str(tensor.dtype).encode("ascii"))
        array = tensor.reshape(-1).view(torch.uint8).numpy()
        update(memoryview(array))
    return digest.hexdigest()


def _reject_parent_child_aliases(
    child_model: Any,
    parent_models: Mapping[str, Any],
) -> None:
    """Reject every object or backing-storage bridge into a parent model."""

    parent_model_objects = {
        id(model): f"parent model {name!r}"
        for name, model in sorted(parent_models.items())
    }
    if id(child_model) in parent_model_objects:
        raise DynamicTargetSynthesisError(
            "Materialized child aliases a parent model object: "
            f"{parent_model_objects[id(child_model)]}."
        )

    parent_module_objects: dict[int, str] = {}
    parent_tensor_objects: dict[int, str] = {}
    parent_storages: dict[tuple[Any, ...], str] = {}
    for parent_name, parent in sorted(parent_models.items()):
        for path, module in _named_modules(parent):
            parent_module_objects.setdefault(
                id(module),
                f"parent {parent_name!r} module {path or '<root>'!r}",
            )
        for kind, method_name in (
            ("Parameter", "named_parameters"),
            ("buffer", "named_buffers"),
        ):
            for path, value in _named_tensors(parent, method_name):
                description = f"parent {parent_name!r} {kind} {path!r}"
                parent_tensor_objects.setdefault(id(value), description)
                parent_storages.setdefault(
                    _dense_storage_identity(value, label=description),
                    description,
                )

    for path, module in _named_modules(child_model):
        parent_description = parent_module_objects.get(id(module))
        if parent_description is not None:
            raise DynamicTargetSynthesisError(
                "Materialized child aliases a parent module object at "
                f"{path or '<root>'!r}: {parent_description}."
            )
    for kind, method_name in (
        ("Parameter", "named_parameters"),
        ("buffer", "named_buffers"),
    ):
        for path, value in _named_tensors(child_model, method_name):
            parent_description = parent_tensor_objects.get(id(value))
            if parent_description is not None:
                raise DynamicTargetSynthesisError(
                    f"Materialized child {kind} {path!r} aliases a parent "
                    f"tensor object: {parent_description}."
                )
            storage = _dense_storage_identity(
                value,
                label=f"child {kind} {path!r}",
            )
            parent_description = parent_storages.get(storage)
            if parent_description is not None:
                raise DynamicTargetSynthesisError(
                    f"Materialized child {kind} {path!r} aliases parent "
                    f"tensor storage: {parent_description}."
                )


@dataclass(frozen=True, slots=True)
class _CompilerBlankPreparation:
    unique_parameter_object_count: int
    unique_parameter_storage_count: int
    unique_parameter_element_count: int
    unique_parameter_bytes: int
    buffer_tensor_count: int
    parameter_layout_fingerprint: str
    buffer_layout_fingerprint: str
    buffer_state_fingerprint_before: str
    buffer_state_fingerprint_after: str

    def provenance(
        self,
        blank_state_fingerprint: str,
        structural_buffer_policy: StructuralBufferPolicy,
        structural_buffer_audit: StructuralBufferAudit,
    ) -> CompilerBlankProvenance:
        if structural_buffer_audit.policy_digest != structural_buffer_policy.digest:
            raise DynamicTargetSynthesisError(
                "Structural buffer audit is not bound to its policy."
            )
        return CompilerBlankProvenance(
            method=COMPILER_BLANK_METHOD,
            compiler_created_blank=True,
            parent_child_alias_free=True,
            zeroed_under_no_grad=True,
            dense_materialized_parameters=True,
            all_parameters_zero=True,
            parameter_gradients_absent=True,
            runtime_buffers_preserved=True,
            unique_parameter_object_count=self.unique_parameter_object_count,
            unique_parameter_storage_count=self.unique_parameter_storage_count,
            unique_parameter_element_count=self.unique_parameter_element_count,
            unique_parameter_bytes=self.unique_parameter_bytes,
            buffer_tensor_count=self.buffer_tensor_count,
            parameter_layout_fingerprint=self.parameter_layout_fingerprint,
            buffer_layout_fingerprint=self.buffer_layout_fingerprint,
            buffer_state_fingerprint_before=self.buffer_state_fingerprint_before,
            buffer_state_fingerprint_after=self.buffer_state_fingerprint_after,
            blank_state_fingerprint=blank_state_fingerprint,
            structural_buffer_policy_digest=(structural_buffer_audit.policy_digest),
            structural_buffer_audit_digest=structural_buffer_audit.digest,
            structural_buffer_policy_verified=(
                structural_buffer_policy.compiler_reviewed
            ),
            nonzero_structural_buffer_count=(
                structural_buffer_audit.nonzero_buffer_count
            ),
        )


def _create_compiler_owned_blank(model: Any) -> _CompilerBlankPreparation:
    """Zero learned parameters in place without duplicating the model or buffers."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for compiler-owned blank construction."
        ) from exc
    parameters_before = _named_tensors(model, "named_parameters")
    buffers_before = _named_tensors(model, "named_buffers")
    if not parameters_before:
        raise DynamicTargetSynthesisError(
            "A materialized target must contain learned Parameters."
        )
    _unique_tensor_view_inventory(
        (*parameters_before, *buffers_before),
        label="child-runtime-state",
    )
    unique_parameters: dict[int, Any] = {}
    unique_storages: set[tuple[Any, ...]] = set()
    unique_views: dict[tuple[Any, ...], tuple[int, int]] = {}
    for name, parameter in parameters_before:
        if not isinstance(parameter, torch.nn.Parameter):
            raise DynamicTargetSynthesisError(
                f"Materialized target state {name!r} is not a Parameter."
            )
        _dense_storage_identity(parameter, label=f"child Parameter {name!r}")
        unique_parameters.setdefault(id(parameter), parameter)
        unique_storages.add(
            _dense_storage_identity(parameter, label=f"child Parameter {name!r}")
        )
        view = _parameter_view_identity(
            parameter,
            label=f"child Parameter {name!r}",
        )
        count_and_size = (int(parameter.numel()), int(parameter.element_size()))
        previous = unique_views.get(view)
        if previous is not None and previous != count_and_size:
            raise DynamicTargetSynthesisError(
                "Aliased child Parameter views have inconsistent sizes."
            )
        unique_views[view] = count_and_size

    parameter_layout = _layout_fingerprint(
        parameters_before,
        domain="parameters",
    )
    buffer_layout = _layout_fingerprint(buffers_before, domain="buffers")
    buffer_state_before = _tensor_content_fingerprint(
        buffers_before,
        domain="runtime_buffers",
    )
    with torch.no_grad():
        if torch.is_grad_enabled():  # pragma: no cover - invariant of torch.no_grad
            raise DynamicTargetSynthesisError(
                "Compiler blank construction unexpectedly enabled autograd."
            )
        for parameter in unique_parameters.values():
            parameter.zero_()
            parameter.grad = None

    parameters_after = _named_tensors(model, "named_parameters")
    buffers_after = _named_tensors(model, "named_buffers")
    if tuple((name, id(value)) for name, value in parameters_before) != tuple(
        (name, id(value)) for name, value in parameters_after
    ):
        raise DynamicTargetSynthesisError(
            "Materialized target Parameters changed identity during blank creation."
        )
    if tuple((name, id(value)) for name, value in buffers_before) != tuple(
        (name, id(value)) for name, value in buffers_after
    ):
        raise DynamicTargetSynthesisError(
            "Runtime buffers changed identity during blank creation."
        )
    if _layout_fingerprint(parameters_after, domain="parameters") != parameter_layout:
        raise DynamicTargetSynthesisError(
            "Parameter layout changed during compiler blank creation."
        )
    if _layout_fingerprint(buffers_after, domain="buffers") != buffer_layout:
        raise DynamicTargetSynthesisError(
            "Runtime-buffer layout changed during compiler blank creation."
        )
    for name, parameter in parameters_after:
        _dense_storage_identity(parameter, label=f"child Parameter {name!r}")
        if parameter.grad_fn is not None or parameter.grad is not None:
            raise DynamicTargetSynthesisError(
                f"Child Parameter {name!r} retains autograd state."
            )
        if int(torch.count_nonzero(parameter.detach()).item()) != 0:
            raise DynamicTargetSynthesisError(
                f"Child Parameter {name!r} was not blanked to exact zero."
            )
    buffer_state_after = _tensor_content_fingerprint(
        buffers_after,
        domain="runtime_buffers",
    )
    if buffer_state_after != buffer_state_before:
        raise DynamicTargetSynthesisError(
            "Runtime-buffer contents changed during compiler blank creation."
        )
    return _CompilerBlankPreparation(
        unique_parameter_object_count=len(unique_parameters),
        unique_parameter_storage_count=len(unique_storages),
        unique_parameter_element_count=sum(value[0] for value in unique_views.values()),
        unique_parameter_bytes=sum(
            count * size for count, size in unique_views.values()
        ),
        buffer_tensor_count=len({id(value) for _name, value in buffers_before}),
        parameter_layout_fingerprint=parameter_layout,
        buffer_layout_fingerprint=buffer_layout,
        buffer_state_fingerprint_before=buffer_state_before,
        buffer_state_fingerprint_after=buffer_state_after,
    )


def _verify_compiler_owned_blank(
    model: Any,
    preparation: _CompilerBlankPreparation,
) -> None:
    """Reauthenticate the blank after independent model inspection."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicTargetSynthesisError(
            "PyTorch is required for compiler-owned blank construction."
        ) from exc
    parameters = _named_tensors(model, "named_parameters")
    buffers = _named_tensors(model, "named_buffers")
    if _layout_fingerprint(parameters, domain="parameters") != (
        preparation.parameter_layout_fingerprint
    ):
        raise DynamicTargetSynthesisError(
            "Independent inspection changed the compiler-created Parameter layout."
        )
    if _layout_fingerprint(buffers, domain="buffers") != (
        preparation.buffer_layout_fingerprint
    ):
        raise DynamicTargetSynthesisError(
            "Independent inspection changed the runtime-buffer layout."
        )
    if _tensor_content_fingerprint(buffers, domain="runtime_buffers") != (
        preparation.buffer_state_fingerprint_after
    ):
        raise DynamicTargetSynthesisError(
            "Independent inspection changed runtime-buffer contents."
        )

    unique_objects: set[int] = set()
    unique_storages: set[tuple[Any, ...]] = set()
    unique_views: dict[tuple[Any, ...], tuple[int, int]] = {}
    for name, parameter in parameters:
        if not isinstance(parameter, torch.nn.Parameter):
            raise DynamicTargetSynthesisError(
                f"Materialized target state {name!r} is not a Parameter."
            )
        unique_objects.add(id(parameter))
        unique_storages.add(
            _dense_storage_identity(parameter, label=f"child Parameter {name!r}")
        )
        view = _parameter_view_identity(
            parameter,
            label=f"child Parameter {name!r}",
        )
        unique_views[view] = (
            int(parameter.numel()),
            int(parameter.element_size()),
        )
        if parameter.grad_fn is not None or parameter.grad is not None:
            raise DynamicTargetSynthesisError(
                f"Child Parameter {name!r} retains autograd state."
            )
        if int(torch.count_nonzero(parameter.detach()).item()) != 0:
            raise DynamicTargetSynthesisError(
                f"Child Parameter {name!r} is no longer exact zero."
            )
    observed = (
        len(unique_objects),
        len(unique_storages),
        sum(value[0] for value in unique_views.values()),
        sum(count * size for count, size in unique_views.values()),
        len({id(value) for _name, value in buffers}),
    )
    expected = (
        preparation.unique_parameter_object_count,
        preparation.unique_parameter_storage_count,
        preparation.unique_parameter_element_count,
        preparation.unique_parameter_bytes,
        preparation.buffer_tensor_count,
    )
    if observed != expected:
        raise DynamicTargetSynthesisError(
            "Independent inspection changed the compiler-created blank inventory."
        )


class DynamicTargetAdapter(Protocol):
    adapter_id: str
    version: int
    priority: int
    target_families: tuple[str, ...]
    target_modalities: tuple[str, ...]

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool: ...

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, ModelCircuitGraph],
        inspectors: InspectorRegistry,
    ) -> AdapterMaterialization: ...


@dataclass(frozen=True, slots=True)
class DynamicTargetSynthesisResult:
    request: DynamicTargetRequest
    adapter_id: str
    adapter_version: int
    model: Any
    graph: ModelCircuitGraph
    resource_profile: DynamicTargetResourceProfile
    topology_fingerprint: str
    adapter_report: Mapping[str, JsonScalar]
    parent_graph_fingerprints: tuple[tuple[str, str], ...]
    blank_state_fingerprint: str
    adapter_payload: Any = None
    blank_provenance: CompilerBlankProvenance | None = None
    topology_provenance_verified: bool = False
    structural_buffer_policy: StructuralBufferPolicy | None = None
    structural_buffer_audit: StructuralBufferAudit | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "blank_state_fingerprint",
            _sha256_digest(
                self.blank_state_fingerprint,
                "blank_state_fingerprint",
            ),
        )
        provenance = self.blank_provenance
        if provenance is None:
            provenance = CompilerBlankProvenance.unverified_external(
                self.blank_state_fingerprint
            )
            object.__setattr__(self, "blank_provenance", provenance)
        if not isinstance(provenance, CompilerBlankProvenance):
            raise DynamicTargetSynthesisError(
                "blank_provenance must be CompilerBlankProvenance."
            )
        if provenance.blank_state_fingerprint != self.blank_state_fingerprint:
            raise DynamicTargetSynthesisError(
                "Blank provenance is bound to a different model-state fingerprint."
            )
        if not isinstance(self.topology_provenance_verified, bool):
            raise DynamicTargetSynthesisError(
                "topology_provenance_verified must be boolean."
            )
        if self.structural_buffer_policy is not None and not isinstance(
            self.structural_buffer_policy,
            StructuralBufferPolicy,
        ):
            raise DynamicTargetSynthesisError(
                "structural_buffer_policy must be StructuralBufferPolicy or None."
            )
        if self.structural_buffer_audit is not None and not isinstance(
            self.structural_buffer_audit,
            StructuralBufferAudit,
        ):
            raise DynamicTargetSynthesisError(
                "structural_buffer_audit must be StructuralBufferAudit or None."
            )
        if self.structural_buffer_audit is not None:
            if self.structural_buffer_policy is None or (
                self.structural_buffer_audit.policy_digest
                != self.structural_buffer_policy.digest
            ):
                raise DynamicTargetSynthesisError(
                    "Structural buffer audit is not bound to its policy."
                )
            if (
                self.structural_buffer_audit.nonzero_buffer_count > 0
                and not self.structural_buffer_policy.compiler_reviewed
            ):
                raise DynamicTargetSynthesisError(
                    "Nonzero target buffers require a compiler-reviewed policy."
                )
        if provenance.compiler_created_blank:
            if (
                self.structural_buffer_policy is None
                or self.structural_buffer_audit is None
            ):
                raise DynamicTargetSynthesisError(
                    "Compiler-created blank results require their structural "
                    "buffer policy and audit."
                )
            if (
                provenance.structural_buffer_policy_digest
                != self.structural_buffer_policy.digest
                or provenance.structural_buffer_audit_digest
                != self.structural_buffer_audit.digest
                or provenance.nonzero_structural_buffer_count
                != self.structural_buffer_audit.nonzero_buffer_count
                or provenance.structural_buffer_policy_verified
                != self.structural_buffer_policy.compiler_reviewed
            ):
                raise DynamicTargetSynthesisError(
                    "Blank provenance is not bound to the structural buffer proof."
                )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    @property
    def blank_provenance_digest(self) -> str:
        assert self.blank_provenance is not None
        return self.blank_provenance.digest

    def as_dict(self) -> dict[str, Any]:
        assert self.blank_provenance is not None
        return {
            "schema": SCHEMA,
            "kind": "materialized_target",
            "request": self.request.as_dict(),
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "graph": self.graph.as_dict(),
            "resource_profile": self.resource_profile.as_dict(),
            "topology_fingerprint": self.topology_fingerprint,
            "adapter_report": dict(self.adapter_report),
            "parent_graph_fingerprints": dict(self.parent_graph_fingerprints),
            "blank_state_fingerprint": self.blank_state_fingerprint,
            "blank_provenance": self.blank_provenance.as_dict(),
            "blank_provenance_digest": self.blank_provenance_digest,
            "structural_buffer_policy": (
                None
                if self.structural_buffer_policy is None
                else self.structural_buffer_policy.as_dict()
            ),
            "structural_buffer_policy_digest": (
                None
                if self.structural_buffer_policy is None
                else self.structural_buffer_policy.digest
            ),
            "structural_buffer_audit": (
                None
                if self.structural_buffer_audit is None
                else self.structural_buffer_audit.as_dict()
            ),
            "structural_buffer_audit_digest": (
                None
                if self.structural_buffer_audit is None
                else self.structural_buffer_audit.digest
            ),
            "compiler_created_blank_verified": (
                self.blank_provenance.compiler_created_blank
            ),
            "independently_reinspected": True,
            "exact_budget_match": True,
            "pre_fold_state_independently_fingerprinted": True,
            "adapter_blank_state_trusted": False,
            "topology_fingerprint_compiler_derived": (
                self.topology_provenance_verified
            ),
            "adapter_topology_fingerprint_trusted": False,
        }


def compiler_authenticated_topology_fingerprint(
    *,
    request: DynamicTargetRequest,
    adapter_id: str,
    adapter_version: int,
    graph: ModelCircuitGraph,
    blank_provenance: CompilerBlankProvenance,
) -> str:
    """Derive the topology seal exclusively from compiler-observed evidence."""

    if not isinstance(request, DynamicTargetRequest):
        raise DynamicTargetSynthesisError(
            "Authenticated topology requires DynamicTargetRequest."
        )
    if not isinstance(graph, ModelCircuitGraph):
        raise DynamicTargetSynthesisError(
            "Authenticated topology requires ModelCircuitGraph."
        )
    if not isinstance(blank_provenance, CompilerBlankProvenance):
        raise DynamicTargetSynthesisError(
            "Authenticated topology requires compiler blank provenance."
        )
    return canonical_sha256(
        {
            "schema": SCHEMA,
            "kind": "compiler_authenticated_target_topology",
            "request_fingerprint": request.fingerprint,
            "adapter_id": _text(adapter_id, "adapter_id"),
            "adapter_version": _positive_int(adapter_version, "adapter_version"),
            "graph_fingerprint": graph.fingerprint,
            "parameter_layout_fingerprint": (
                blank_provenance.parameter_layout_fingerprint
            ),
            "buffer_layout_fingerprint": blank_provenance.buffer_layout_fingerprint,
            "buffer_state_fingerprint": (
                blank_provenance.buffer_state_fingerprint_after
            ),
        }
    )


def verify_compiler_authenticated_target(
    target: DynamicTargetSynthesisResult,
) -> str:
    """Recompute and verify a synthesized target's compiler-owned topology seal."""

    if not isinstance(target, DynamicTargetSynthesisResult):
        raise DynamicTargetSynthesisError(
            "Compiler target verification requires DynamicTargetSynthesisResult."
        )
    if not target.topology_provenance_verified:
        raise DynamicTargetSynthesisError(
            "Target topology lacks compiler-authenticated provenance."
        )
    assert target.blank_provenance is not None
    expected = compiler_authenticated_topology_fingerprint(
        request=target.request,
        adapter_id=target.adapter_id,
        adapter_version=target.adapter_version,
        graph=target.graph,
        blank_provenance=target.blank_provenance,
    )
    if target.topology_fingerprint != expected:
        raise DynamicTargetSynthesisError(
            "Target topology fingerprint differs from compiler-observed evidence."
        )
    return expected


class DynamicTargetAdapterRegistry:
    """Select one topology adapter without caller-side architecture surgery."""

    def __init__(
        self,
        adapters: Sequence[DynamicTargetAdapter] = (),
    ) -> None:
        self._adapters: dict[str, DynamicTargetAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DynamicTargetAdapter) -> None:
        adapter_id = _text(
            getattr(adapter, "adapter_id", None),
            "adapter_id",
        )
        version = getattr(adapter, "version", None)
        priority = getattr(adapter, "priority", None)
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise DynamicTargetSynthesisError(
                "Target adapter version must be positive."
            )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicTargetSynthesisError(
                "Target adapter priority must be an integer."
            )
        if not callable(getattr(adapter, "supports", None)) or not callable(
            getattr(adapter, "materialize", None)
        ):
            raise DynamicTargetSynthesisError(
                "Target adapter must implement supports() and materialize()."
            )
        if adapter_id in self._adapters:
            raise DynamicTargetSynthesisError(
                f"Target adapter {adapter_id!r} is already registered."
            )
        self._adapters[adapter_id] = adapter

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def select(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> DynamicTargetAdapter:
        if request.adapter_id is not None:
            try:
                adapter = self._adapters[request.adapter_id]
            except KeyError as exc:
                raise DynamicTargetSynthesisError(
                    f"Unknown requested target adapter {request.adapter_id!r}."
                ) from exc
            if not adapter.supports(request, parent_graphs):
                raise DynamicTargetSynthesisError(
                    f"Target adapter {request.adapter_id!r} does not support "
                    "the requested family/modality/source combination."
                )
            return adapter
        candidates = [
            adapter
            for adapter in self._adapters.values()
            if adapter.supports(request, parent_graphs)
        ]
        if not candidates:
            raise DynamicTargetSynthesisError(
                "No registered target adapter supports this exact request."
            )
        candidates.sort(key=lambda value: (-int(value.priority), str(value.adapter_id)))
        return candidates[0]


class DynamicTargetSynthesizer:
    """Resolve an adapter, create a blank, and independently authenticate it."""

    def __init__(
        self,
        *,
        inspectors: InspectorRegistry,
        adapters: DynamicTargetAdapterRegistry,
    ) -> None:
        self.inspectors = inspectors
        self.adapters = adapters

    def synthesize(
        self,
        *,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    ) -> DynamicTargetSynthesisResult:
        parents = dict(parent_models)
        if not parents:
            raise DynamicTargetSynthesisError(
                "Target synthesis requires at least one loaded parent."
            )
        tokenizer_fingerprints = dict(parent_tokenizer_fingerprints or {})
        if set(tokenizer_fingerprints) - set(parents):
            raise DynamicTargetSynthesisError(
                "Parent tokenizer fingerprints reference unknown models."
            )
        parent_graphs = {
            name: self.inspectors.inspect(
                model,
                model_name=name,
                tokenizer_fingerprint=tokenizer_fingerprints.get(name),
            )
            for name, model in sorted(parents.items())
        }
        allocation_receipt = request.component_allocation_receipt
        if allocation_receipt is not None:
            matching_sources = tuple(
                graph
                for graph in parent_graphs.values()
                if graph.fingerprint == allocation_receipt.source_graph_fingerprint
            )
            if not matching_sources or any(
                graph.unique_parameter_count
                != allocation_receipt.source_unique_parameters
                or graph.logical_parameter_count
                != allocation_receipt.source_logical_parameters
                for graph in matching_sources
            ):
                raise DynamicTargetSynthesisError(
                    "Authenticated component-allocation receipt differs from "
                    "the loaded source graph or its exact counts."
                )
        adapter = self.adapters.select(request, parent_graphs)
        materialized = adapter.materialize(
            request,
            MappingProxyType(parents),
            MappingProxyType(parent_graphs),
            self.inspectors,
        )
        validate_registered_inference_state(
            materialized.model,
            forbidden_models=parents,
        )
        _reject_parent_child_aliases(materialized.model, parents)
        structural_buffer_policy = _resolve_structural_buffer_policy(
            adapter=adapter,
            request=request,
            materialized=materialized,
        )
        structural_buffer_audit = validate_structural_runtime_buffers(
            materialized.model,
            structural_buffer_policy,
            expected_adapter_id=str(adapter.adapter_id),
            expected_adapter_version=int(adapter.version),
        )
        blank_preparation = _create_compiler_owned_blank(materialized.model)
        independently_inspected = self.inspectors.inspect(
            materialized.model,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        _verify_compiler_owned_blank(materialized.model, blank_preparation)
        if independently_inspected.fingerprint != materialized.graph.fingerprint:
            raise DynamicTargetSynthesisError(
                "Adapter-reported graph differs from independent inspection."
            )
        profile = DynamicTargetResourceProfile.from_graph(
            independently_inspected,
            materialized.model,
        )
        if profile.unique_parameters != request.exact_unique_parameters:
            raise DynamicTargetSynthesisError(
                "Materialized target misses the exact unique-parameter budget."
            )
        if allocation_receipt is not None and (
            profile.unique_parameters != allocation_receipt.target_unique_parameters
            or independently_inspected.logical_parameter_count
            != allocation_receipt.target_logical_parameters
        ):
            raise DynamicTargetSynthesisError(
                "Materialized target differs from the authenticated "
                "component-allocation receipt counts."
            )
        if (
            request.maximum_parameter_bytes is not None
            and profile.parameter_bytes > request.maximum_parameter_bytes
        ):
            raise DynamicTargetSynthesisError(
                "Materialized target exceeds the parameter-byte budget."
            )
        if (
            request.target_family not in {"any", "auto"}
            and profile.family != request.target_family
        ):
            raise DynamicTargetSynthesisError(
                "Materialized target family differs from the request."
            )
        if set(profile.modalities) != set(request.target_modalities):
            raise DynamicTargetSynthesisError(
                "Materialized target modalities differ from the request."
            )
        blank_state_fingerprint = fingerprint_model_state(materialized.model)
        blank_provenance = blank_preparation.provenance(
            blank_state_fingerprint,
            structural_buffer_policy,
            structural_buffer_audit,
        )
        certified_adapter_report = dict(materialized.report)
        if allocation_receipt is not None and (
            certified_adapter_report.get("component_allocation_receipt_sha256")
            != allocation_receipt.fingerprint
            or certified_adapter_report.get("component_allocation_topology_sha256")
            != allocation_receipt.topology_configuration_sha256
            or certified_adapter_report.get(
                "component_allocation_verified_by_target_adapter"
            )
            is not True
        ):
            raise DynamicTargetSynthesisError(
                "Target adapter did not authenticate the bound "
                "component-allocation receipt."
            )
        certified_adapter_report["adapter_reported_topology_fingerprint"] = (
            materialized.topology_fingerprint
        )
        if "blank_state_fingerprint" in certified_adapter_report:
            # This field is retained for compatibility, but its value is replaced
            # with the compiler-authenticated post-zeroing state. The adapter's
            # original blank-state claim is never propagated or trusted.
            certified_adapter_report["blank_state_fingerprint"] = (
                blank_state_fingerprint
            )
        topology_fingerprint = compiler_authenticated_topology_fingerprint(
            request=request,
            adapter_id=str(adapter.adapter_id),
            adapter_version=int(adapter.version),
            graph=independently_inspected,
            blank_provenance=blank_provenance,
        )
        return DynamicTargetSynthesisResult(
            request=request,
            adapter_id=str(adapter.adapter_id),
            adapter_version=int(adapter.version),
            model=materialized.model,
            graph=independently_inspected,
            resource_profile=profile,
            topology_fingerprint=topology_fingerprint,
            adapter_report=MappingProxyType(certified_adapter_report),
            parent_graph_fingerprints=tuple(
                (name, graph.fingerprint)
                for name, graph in sorted(parent_graphs.items())
            ),
            blank_state_fingerprint=blank_state_fingerprint,
            adapter_payload=materialized.adapter_payload,
            blank_provenance=blank_provenance,
            topology_provenance_verified=True,
            structural_buffer_policy=structural_buffer_policy,
            structural_buffer_audit=structural_buffer_audit,
        )

    def synthesize_planned(
        self,
        *,
        plan: BudgetedDynamicTargetPlan,
        parent_models: Mapping[str, Any],
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    ) -> DynamicTargetSynthesisResult:
        """Materialize through the existing path and verify the budget preflight."""

        if not isinstance(plan, BudgetedDynamicTargetPlan):
            raise DynamicTargetSynthesisError(
                "plan must be BudgetedDynamicTargetPlan."
            )
        result = self.synthesize(
            request=plan.target_request,
            parent_models=parent_models,
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        )
        plan.target_certificate.verify_materialized_target(result)
        return result


__all__ = [
    "AdapterMaterialization",
    "BLANK_PROVENANCE_SCHEMA",
    "COMPILER_BLANK_METHOD",
    "CompilerBlankProvenance",
    "BudgetedDynamicTargetPlan",
    "compiler_authenticated_topology_fingerprint",
    "DynamicTargetTopologyCertificate",
    "DynamicTargetAdapter",
    "DynamicTargetAdapterRegistry",
    "DynamicTargetRequest",
    "DynamicTargetResourceProfile",
    "DynamicTargetSynthesisError",
    "DynamicTargetSynthesisResult",
    "DynamicTargetSynthesizer",
    "PlannedBlankTargetGraphCertificate",
    "plan_budgeted_dynamic_target",
    "SCHEMA",
    "StructuralBufferAudit",
    "StructuralBufferPolicy",
    "serialized_inference_state_inventory",
    "validate_registered_inference_state",
    "validate_structural_runtime_buffers",
    "verify_compiler_authenticated_target",
]
