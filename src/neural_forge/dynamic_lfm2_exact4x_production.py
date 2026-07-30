"""Sealed one-request entrypoint for the current exact-4x LFM2 backend.

The call site submits one family-neutral dense parameter budget.  A configured
runtime owns parent loading, topology planning, blank synthesis, semantic
folding, publication, fresh reload, and decoded manual-review evidence.  The
runtime contains no component or layer dispatch; those details stay behind the
registered target catalog and semantic resource resolver.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_counted_child_state import (
    ContinuousChildStateAllocation,
    CountedChildStatePlan,
    counted_child_state_plan_from_dict,
    verify_counted_child_state,
)
from .dynamic_lfm2_adapter import (
    Lfm2TargetSynthesisAdapter,
)
from .dynamic_lfm2_exact4x_semantic_backend import (
    SCHEMA as LFM2_EXACT4X_BACKEND_SCHEMA,
    Lfm2Exact4xSemanticBackendConfig,
    Lfm2Exact4xSemanticResourceResolver,
)
from .dynamic_fold_budget import (
    ComponentProtectionRule,
    residual_attention_component_protection_rules,
)
from .dynamic_model_inspector import InspectorRegistry, default_inspector_registry
from .dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED,
    Lfm2DenseTargetTopologyCatalog,
)
from .dynamic_semantic_fold import ResolvingSemanticWholeModelFoldStackAdapter
from .dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from .dynamic_target_synthesis import DynamicTargetAdapterRegistry
from .hf_lfm2_dense_budget_model_fold import DENSE_BUDGET_MODEL_METHOD
from .hf_lfm2_direct_local import (
    load_lfm2_direct_local_model,
    load_lfm2_direct_local_tokenizer,
)
from .hf_tokenizer_bridge import load_strict_tokenizer_bridge_artifact
from .lfm2_native_exact4x_target import (
    LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE,
)


SCHEMA = "neural-forge-lfm2-exact4x-production-request-v1"
COMPONENT_PROTECTION_POLICY_SCHEMA = (
    "neural-forge-dense-component-protection-policy-v1"
)


class Lfm2Exact4xProductionError(RuntimeError):
    """A sealed request or configured production runtime is invalid."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Lfm2Exact4xProductionError(f"{name} must be nonempty.")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2Exact4xProductionError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2Exact4xProductionError(f"{name} must be non-negative.")
    return value


def _finite_nonnegative_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise Lfm2Exact4xProductionError(
            f"{name} must be a finite non-negative number."
        )
    return float(value)


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2Exact4xProductionError(f"{name} must be a lowercase SHA-256.")
    return value


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, str) or "://" in value:
        raise Lfm2Exact4xProductionError(f"{name} must be an absolute local path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise Lfm2Exact4xProductionError(f"{name} must be an absolute local path.")
    return path.absolute()


@dataclass(frozen=True, slots=True)
class NeutralFullModelFoldRequest:
    """One complete-child objective with no component-level instructions."""

    request_id: str
    parent_name: str
    child_name: str
    budget: DenseParameterBudgetRequest
    destination: Path
    blank_seed: int = 0
    counted_child_state_plan: CountedChildStatePlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _text(self.request_id, "request_id"))
        object.__setattr__(self, "parent_name", _text(self.parent_name, "parent_name"))
        object.__setattr__(self, "child_name", _text(self.child_name, "child_name"))
        if not isinstance(self.budget, DenseParameterBudgetRequest):
            raise Lfm2Exact4xProductionError(
                "budget must be DenseParameterBudgetRequest."
            )
        destination = Path(self.destination).expanduser().absolute()
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self,
            "blank_seed",
            _nonnegative_int(self.blank_seed, "blank_seed"),
        )
        plan = self.counted_child_state_plan
        if plan is not None:
            if not isinstance(plan, CountedChildStatePlan):
                raise Lfm2Exact4xProductionError(
                    "counted_child_state_plan has the wrong type."
                )
            if (
                self.budget.target_unique_parameters
                != plan.target_total_parameter_elements
                or self.budget.compression_ratio is not None
            ):
                raise Lfm2Exact4xProductionError(
                    "The neutral fold budget must equal the counted child-state "
                    "total."
                )

    @property
    def dense_planning_budget(self) -> DenseParameterBudgetRequest:
        """Physical dense budget after reserving all auxiliary child state."""

        if self.counted_child_state_plan is None:
            return self.budget
        return self.counted_child_state_plan.dense_budget(self.budget)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        value = {
            "request_id": self.request_id,
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "target_family": self.budget.target_family,
            "target_modalities": list(self.budget.target_modalities),
            "target_unique_parameters": self.budget.target_unique_parameters,
            "compression_ratio": self.budget.compression_ratio,
            "target_vocabulary_size": self.budget.target_vocabulary_size,
            "match_policy": self.budget.match_policy,
            "maximum_candidates": self.budget.maximum_candidates,
            "blank_seed": self.blank_seed,
            "destination": str(self.destination),
        }
        if self.counted_child_state_plan is not None:
            value["counted_child_state_plan"] = (
                self.counted_child_state_plan.as_dict()
            )
        return value


@dataclass(frozen=True, slots=True)
class RegisteredParentSource:
    """A configure-once parent source behind the neutral request boundary."""

    name: str
    tokenizer_fingerprint: str
    loader: Callable[[], Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "parent source name"))
        object.__setattr__(
            self,
            "tokenizer_fingerprint",
            _digest(self.tokenizer_fingerprint, "parent tokenizer fingerprint"),
        )
        if not callable(self.loader):
            raise Lfm2Exact4xProductionError("parent loader must be callable.")


class _AutomaticSystem(Protocol):
    inspectors: InspectorRegistry | None

    def plan_budgeted_target(self, **kwargs: Any) -> Any: ...

    def run(self, **kwargs: Any) -> Any: ...


class CountedChildStateObserver(Protocol):
    """Authenticate packaged continuous state from the published child.

    The observer is configured with the family/runtime integration.  It must
    inspect the published standalone artifact (and, in production, its fresh
    reload), rather than echoing the planning request.
    """

    def __call__(
        self,
        result: Any,
        plan: CountedChildStatePlan,
    ) -> Sequence[ContinuousChildStateAllocation]: ...


@dataclass(frozen=True, slots=True)
class FullModelFoldExecution:
    """Preflight target certificate plus the published complete child."""

    request: NeutralFullModelFoldRequest
    target_plan: Any
    result: Any
    materialized_target_binding_sha256: str
    component_protection_policy_sha256: str | None = None
    counted_child_state_verification: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "materialized_target_binding_sha256",
            _digest(
                self.materialized_target_binding_sha256,
                "materialized_target_binding_sha256",
            ),
        )
        if self.component_protection_policy_sha256 is not None:
            object.__setattr__(
                self,
                "component_protection_policy_sha256",
                _digest(
                    self.component_protection_policy_sha256,
                    "component_protection_policy_sha256",
                ),
            )
        counted_plan = self.request.counted_child_state_plan
        verification = self.counted_child_state_verification
        if counted_plan is None and verification is not None:
            raise Lfm2Exact4xProductionError(
                "An uncounted child cannot carry counted-state verification."
            )
        if counted_plan is not None:
            if not isinstance(verification, Mapping):
                raise Lfm2Exact4xProductionError(
                    "A counted child is incomplete until packaged continuous "
                    "state is independently verified."
                )
            verification = dict(verification)
            if (
                verification.get("plan_fingerprint") != counted_plan.fingerprint
                or verification.get("physical_dense_parameter_elements")
                != counted_plan.dense_model_parameter_elements
                or verification.get("continuous_parameter_elements")
                != counted_plan.continuous_parameter_elements
                or verification.get("target_total_parameter_elements")
                != counted_plan.target_total_parameter_elements
                or verification.get("status")
                != "standalone-counted-child-state-exact"
            ):
                raise Lfm2Exact4xProductionError(
                    "Counted child-state verification differs from its sealed plan."
                )
            body = dict(verification)
            observed_sha256 = body.pop("verification_sha256", None)
            if (
                not isinstance(observed_sha256, str)
                or canonical_sha256(body) != observed_sha256
            ):
                raise Lfm2Exact4xProductionError(
                    "Counted child-state verification seal is invalid."
                )
            object.__setattr__(
                self,
                "counted_child_state_verification",
                verification,
            )

    def summary(self) -> dict[str, Any]:
        artifact = self.result.published.artifact
        certificate = self.target_plan.target_certificate
        component_receipt = certificate.component_allocation_receipt
        counted_plan = self.request.counted_child_state_plan
        exact_total = (
            artifact.exact_unique_parameters
            if counted_plan is None
            else counted_plan.target_total_parameter_elements
        )
        verification = self.counted_child_state_verification
        return {
            "schema": SCHEMA,
            "status": "published-complete-child-ready-for-manual-review",
            "request_fingerprint": self.request.fingerprint,
            "target_plan_fingerprint": self.target_plan.fingerprint,
            "target_request_fingerprint": self.target_plan.target_request.fingerprint,
            "target_topology_certificate_sha256": certificate.fingerprint,
            "target_selection_policy": certificate.selection_policy,
            "component_allocation_receipt_sha256": (
                None if component_receipt is None else component_receipt.fingerprint
            ),
            "component_protection_policy_sha256": (
                self.component_protection_policy_sha256
            ),
            "materialized_target_binding_sha256": (
                self.materialized_target_binding_sha256
            ),
            "complete": self.result.complete,
            "artifact_path": artifact.path,
            "artifact_certificate_sha256": artifact.certificate_sha256,
            "exact_unique_parameters": exact_total,
            "physical_dense_unique_parameters": artifact.exact_unique_parameters,
            "continuous_parameter_elements": (
                0
                if counted_plan is None
                else counted_plan.continuous_parameter_elements
            ),
            "counted_child_state_plan_fingerprint": (
                None if counted_plan is None else counted_plan.fingerprint
            ),
            "counted_child_state_verification_sha256": (
                None
                if verification is None
                else verification["verification_sha256"]
            ),
            "fresh_reload_accepted": artifact.fresh_reload_decision.accepted,
            "automatic_intelligence_verdict_performed": False,
            "manual_decoded_review_required": True,
        }


class ConfiguredFullModelFoldEntrypoint:
    """Generic orchestration over configure-once adapters and resources."""

    def __init__(
        self,
        *,
        system: _AutomaticSystem,
        parent_sources: Sequence[RegisteredParentSource],
        budget_parent_name: str,
        child_tokenizer_fingerprint: str,
        evidence_fingerprint: str,
        component_protection_policy_sha256: str | None = None,
        counted_child_state_observer: CountedChildStateObserver | None = None,
        target_adapter_id: str | None = None,
        fold_method_id: str | None = None,
    ) -> None:
        if not callable(getattr(system, "plan_budgeted_target", None)) or not callable(
            getattr(system, "run", None)
        ):
            raise Lfm2Exact4xProductionError(
                "system must expose plan_budgeted_target() and run()."
            )
        sources = tuple(parent_sources)
        if not sources or any(
            not isinstance(value, RegisteredParentSource) for value in sources
        ):
            raise Lfm2Exact4xProductionError(
                "parent_sources must contain registered sources."
            )
        names = tuple(value.name for value in sources)
        if len(set(names)) != len(names):
            raise Lfm2Exact4xProductionError("parent source names must be unique.")
        selected_parent = _text(budget_parent_name, "budget_parent_name")
        if selected_parent not in names:
            raise Lfm2Exact4xProductionError(
                "budget_parent_name is not a registered parent source."
            )
        self.system = system
        self.parent_sources = sources
        self.budget_parent_name = selected_parent
        self.child_tokenizer_fingerprint = _digest(
            child_tokenizer_fingerprint,
            "child tokenizer fingerprint",
        )
        self.evidence_fingerprint = _digest(
            evidence_fingerprint,
            "evidence fingerprint",
        )
        self.component_protection_policy_sha256 = (
            None
            if component_protection_policy_sha256 is None
            else _digest(
                component_protection_policy_sha256,
                "component_protection_policy_sha256",
            )
        )
        if counted_child_state_observer is not None and not callable(
            counted_child_state_observer
        ):
            raise Lfm2Exact4xProductionError(
                "counted_child_state_observer must be callable or None."
            )
        self.counted_child_state_observer = counted_child_state_observer
        self.target_adapter_id = (
            None
            if target_adapter_id is None
            else _text(target_adapter_id, "target_adapter_id")
        )
        self.fold_method_id = (
            None
            if fold_method_id is None
            else _text(fold_method_id, "fold_method_id")
        )

    def run(self, request: NeutralFullModelFoldRequest) -> FullModelFoldExecution:
        """Plan, synthesize, fold, publish, reload, and expose decoded output."""

        if not isinstance(request, NeutralFullModelFoldRequest):
            raise Lfm2Exact4xProductionError(
                "run requires one NeutralFullModelFoldRequest."
            )
        if request.parent_name != self.budget_parent_name:
            raise Lfm2Exact4xProductionError(
                "Request parent differs from the configured budget parent."
            )
        parent_models = {value.name: value.loader() for value in self.parent_sources}
        tokenizer_fingerprints = {
            value.name: value.tokenizer_fingerprint for value in self.parent_sources
        }
        inspectors = self.system.inspectors or default_inspector_registry()
        parent_graph = inspectors.inspect(
            parent_models[self.budget_parent_name],
            model_name=self.budget_parent_name,
            tokenizer_fingerprint=tokenizer_fingerprints[self.budget_parent_name],
        )
        plan = self.system.plan_budgeted_target(
            parent_graph=parent_graph,
            budget_request=request.dense_planning_budget,
            child_name=request.child_name,
            tokenizer_fingerprint=self.child_tokenizer_fingerprint,
            adapter_id=self.target_adapter_id,
            fold_method_id=self.fold_method_id,
            blank_seed=request.blank_seed,
        )
        result = self.system.run(
            parent_models=parent_models,
            target_request=plan.target_request,
            evidence_fingerprint=self.evidence_fingerprint,
            destination=request.destination,
            parent_tokenizer_fingerprints=tokenizer_fingerprints,
        )
        certificate = getattr(plan, "target_certificate", None)
        verify_materialized_target = getattr(
            certificate,
            "verify_materialized_target",
            None,
        )
        materialized_target = getattr(result, "target", None)
        if not callable(verify_materialized_target) or materialized_target is None:
            raise Lfm2Exact4xProductionError(
                "Automatic folding did not return a verifiable planned target."
            )
        materialized_binding = _digest(
            verify_materialized_target(materialized_target),
            "materialized target certificate binding",
        )
        counted_verification = None
        counted_plan = request.counted_child_state_plan
        if counted_plan is not None:
            if self.counted_child_state_observer is None:
                raise Lfm2Exact4xProductionError(
                    "A counted fold requires an observer that authenticates the "
                    "packaged continuous state after publication and fresh reload."
                )
            artifact = getattr(getattr(result, "published", None), "artifact", None)
            physical = getattr(artifact, "exact_unique_parameters", None)
            if isinstance(physical, bool) or not isinstance(physical, int):
                raise Lfm2Exact4xProductionError(
                    "A counted fold did not publish an authenticated dense count."
                )
            observed = tuple(self.counted_child_state_observer(result, counted_plan))
            if not observed or any(
                not isinstance(value, ContinuousChildStateAllocation)
                for value in observed
            ):
                raise Lfm2Exact4xProductionError(
                    "The counted-state observer did not return typed packaged "
                    "allocations."
                )
            counted_verification = verify_counted_child_state(
                counted_plan,
                physical_dense_parameter_elements=physical,
                observed_allocations=observed,
            )
        return FullModelFoldExecution(
            request=request,
            target_plan=plan,
            result=result,
            materialized_target_binding_sha256=materialized_binding,
            component_protection_policy_sha256=(
                self.component_protection_policy_sha256
            ),
            counted_child_state_verification=counted_verification,
        )


@dataclass(frozen=True, slots=True)
class DenseComponentProtectionPolicy:
    """Dimension-agnostic allocation policy over discovered component roles."""

    rules: tuple[ComponentProtectionRule, ...] = field(
        default_factory=residual_attention_component_protection_rules
    )
    bottleneck_risk_weight: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple) or any(
            not isinstance(value, ComponentProtectionRule)
            for value in self.rules
        ):
            raise Lfm2Exact4xProductionError(
                "component protection rules must be a typed tuple of "
                "ComponentProtectionRule objects."
            )
        if not self.rules:
            raise Lfm2Exact4xProductionError(
                "component protection policy must contain at least one rule."
            )
        if len({value.rule_id for value in self.rules}) != len(self.rules):
            raise Lfm2Exact4xProductionError(
                "component protection rule IDs must be unique."
            )
        object.__setattr__(
            self,
            "bottleneck_risk_weight",
            _finite_nonnegative_float(
                self.bottleneck_risk_weight,
                "component bottleneck risk weight",
            ),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPONENT_PROTECTION_POLICY_SCHEMA,
            "rules": [value.as_dict() for value in self.rules],
            "bottleneck_risk_weight": self.bottleneck_risk_weight,
        }


@dataclass(frozen=True, slots=True)
class Lfm2Exact4xProductionRegistration:
    request: NeutralFullModelFoldRequest
    backend: Lfm2Exact4xSemanticBackendConfig
    configuration_sha256: str
    component_protection_policy: DenseComponentProtectionPolicy = field(
        default_factory=DenseComponentProtectionPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.component_protection_policy,
            DenseComponentProtectionPolicy,
        ):
            raise Lfm2Exact4xProductionError(
                "component_protection_policy has the wrong type."
            )

    @property
    def component_protection_policy_sha256(self) -> str:
        return self.component_protection_policy.fingerprint


_BACKEND_REQUIRED_FIELDS = frozenset(
    {
        "parent_model_directory",
        "donor_tokenizer_directory",
        "target_tokenizer_directory",
        "bridge_directory",
        "expected_bridge_artifact_sha256",
        "compiler_cache_directory",
        "expected_compiler_cache_key",
        "semantic_cache_directory",
        "expected_semantic_cache_key",
        "trajectory_corpus",
        "selective_catalog_store_directory",
        "selective_catalog_receipt_json",
        "expected_selective_catalog_receipt_sha256",
        "expected_parent_identity_sha256",
        "expected_parent_unique_parameters",
        "prompts_json",
        "expected_prompts_file_sha256",
        "parent_envelope",
        "expected_parent_envelope_sha256",
        "expected_parent_envelope_file_sha256",
        "body_recipe",
        "expected_body_recipe_sha256",
        "short_conv_construction",
        "model_device",
        "fold_compute_device",
        "analytic_solve_device",
        "global_residual_backend",
        "global_residual_basis_rank",
        "correction_scale",
        "target_dtype_representability_level",
        "child_dtype",
        "maximum_prompt_tokens",
        "maximum_new_tokens",
        "compile_artifact_root",
    }
)
_BACKEND_OPTIONAL_FIELDS = frozenset(
    {
        "expected_parent_graph_fingerprint",
        "source_parent_name",
        "coordinate_completion_recipe",
        "global_residual_closure_version",
        "normalized_input_correction_rho",
        "normalized_input_request_capture_directory",
        "live_child_prefix_recapture_mode",
        "priority",
    }
)
_BACKEND_DERIVED_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "fingerprint",
        "network_allowed",
        "transitional_repository_kernel_runtime_dependency",
        "fully_packaged_production_kernel",
    }
)
_BACKEND_PATH_FIELDS = frozenset(
    {
        "parent_model_directory",
        "donor_tokenizer_directory",
        "target_tokenizer_directory",
        "bridge_directory",
        "compiler_cache_directory",
        "semantic_cache_directory",
        "trajectory_corpus",
        "selective_catalog_store_directory",
        "selective_catalog_receipt_json",
        "prompts_json",
        "parent_envelope",
        "compile_artifact_root",
        "normalized_input_request_capture_directory",
    }
)


def _backend_from_payload(
    payload: Mapping[str, Any],
) -> Lfm2Exact4xSemanticBackendConfig:
    values = dict(payload)
    allowed = _BACKEND_REQUIRED_FIELDS | _BACKEND_OPTIONAL_FIELDS | _BACKEND_DERIVED_FIELDS
    if set(values) - allowed or _BACKEND_REQUIRED_FIELDS - set(values):
        raise Lfm2Exact4xProductionError(
            "backend fields differ from the exact-4x backend contract."
        )
    if "schema" in values and values["schema"] != LFM2_EXACT4X_BACKEND_SCHEMA:
        raise Lfm2Exact4xProductionError("backend schema is invalid.")
    if "kind" in values and values["kind"] != "exact4x_semantic_backend_config":
        raise Lfm2Exact4xProductionError("backend kind is invalid.")
    if values.get("network_allowed", False) is not False:
        raise Lfm2Exact4xProductionError("network access cannot be enabled.")
    if values.get("transitional_repository_kernel_runtime_dependency", True) is not True:
        raise Lfm2Exact4xProductionError(
            "backend packaging dependency declaration is invalid."
        )
    if values.get("fully_packaged_production_kernel", False) is not False:
        raise Lfm2Exact4xProductionError(
            "backend cannot claim the transitional kernel is fully packaged."
        )
    expected_fingerprint = values.get("fingerprint")
    kwargs = {
        key: value
        for key, value in values.items()
        if key in _BACKEND_REQUIRED_FIELDS | _BACKEND_OPTIONAL_FIELDS
    }
    for name in _BACKEND_PATH_FIELDS.intersection(kwargs):
        value = kwargs[name]
        if value is not None:
            kwargs[name] = _absolute_path(value, f"backend {name}")
    backend = Lfm2Exact4xSemanticBackendConfig(**kwargs)
    if expected_fingerprint is not None and (
        _digest(expected_fingerprint, "backend fingerprint") != backend.fingerprint
    ):
        raise Lfm2Exact4xProductionError("backend fingerprint changed.")
    return backend


def _request_from_payload(payload: Mapping[str, Any]) -> NeutralFullModelFoldRequest:
    values = dict(payload)
    required = {
        "request_id",
        "parent_name",
        "child_name",
        "target_family",
        "target_modalities",
        "target_unique_parameters",
        "compression_ratio",
        "target_vocabulary_size",
        "match_policy",
        "maximum_candidates",
        "blank_seed",
        "destination",
    }
    optional = {"counted_child_state_plan"}
    if set(values) - (required | optional) or required - set(values):
        raise Lfm2Exact4xProductionError(
            "request fields differ from the neutral full-model contract."
        )
    modalities = values["target_modalities"]
    if not isinstance(modalities, list):
        raise Lfm2Exact4xProductionError("target_modalities must be a JSON list.")
    if values["compression_ratio"] is not None:
        raise Lfm2Exact4xProductionError(
            "The sealed exact-target request requires compression_ratio=null."
        )
    budget = DenseParameterBudgetRequest(
        target_family=_text(values["target_family"], "target_family"),
        target_modalities=tuple(modalities),
        target_unique_parameters=_positive_int(
            values["target_unique_parameters"],
            "target_unique_parameters",
        ),
        target_vocabulary_size=_positive_int(
            values["target_vocabulary_size"],
            "target_vocabulary_size",
        ),
        match_policy=_text(values["match_policy"], "match_policy"),
        maximum_candidates=_positive_int(
            values["maximum_candidates"],
            "maximum_candidates",
        ),
    )
    counted_plan_value = values.get("counted_child_state_plan")
    try:
        counted_plan = (
            None
            if counted_plan_value is None
            else counted_child_state_plan_from_dict(counted_plan_value)
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xProductionError(
            "counted_child_state_plan is invalid."
        ) from exc
    return NeutralFullModelFoldRequest(
        request_id=values["request_id"],
        parent_name=values["parent_name"],
        child_name=values["child_name"],
        budget=budget,
        destination=_absolute_path(values["destination"], "destination"),
        blank_seed=values["blank_seed"],
        counted_child_state_plan=counted_plan,
    )


def _component_protection_rule_from_payload(
    payload: Mapping[str, Any],
    *,
    index: int,
) -> ComponentProtectionRule:
    values = dict(payload)
    required = {
        "rule_id",
        "circuit_kinds",
        "parameter_semantics",
        "loss_weight",
        "minimum_retention_ppm",
    }
    if set(values) != required:
        raise Lfm2Exact4xProductionError(
            f"component protection rule {index} fields are invalid."
        )
    circuit_kinds = values["circuit_kinds"]
    parameter_semantics = values["parameter_semantics"]
    if not isinstance(circuit_kinds, list) or not isinstance(
        parameter_semantics,
        list,
    ):
        raise Lfm2Exact4xProductionError(
            f"component protection rule {index} selectors must be JSON lists."
        )
    minimum_retention_ppm = _nonnegative_int(
        values["minimum_retention_ppm"],
        f"component protection rule {index} minimum_retention_ppm",
    )
    if minimum_retention_ppm > 1_000_000:
        raise Lfm2Exact4xProductionError(
            f"component protection rule {index} minimum_retention_ppm "
            "cannot exceed 1000000."
        )
    try:
        return ComponentProtectionRule(
            rule_id=_text(
                values["rule_id"],
                f"component protection rule {index} rule_id",
            ),
            circuit_kinds=tuple(circuit_kinds),
            parameter_semantics=tuple(parameter_semantics),
            loss_weight=_finite_nonnegative_float(
                values["loss_weight"],
                f"component protection rule {index} loss_weight",
            ),
            minimum_retention_ppm=minimum_retention_ppm,
        )
    except (TypeError, ValueError) as exc:
        raise Lfm2Exact4xProductionError(
            f"component protection rule {index} is invalid."
        ) from exc


def _component_protection_policy_from_payload(
    payload: Mapping[str, Any],
) -> DenseComponentProtectionPolicy:
    values = dict(payload)
    required = {"schema", "rules", "bottleneck_risk_weight"}
    if set(values) != required:
        raise Lfm2Exact4xProductionError(
            "component protection policy fields are invalid."
        )
    if values["schema"] != COMPONENT_PROTECTION_POLICY_SCHEMA:
        raise Lfm2Exact4xProductionError(
            "component protection policy schema is invalid."
        )
    rule_payloads = values["rules"]
    if not isinstance(rule_payloads, list) or not rule_payloads:
        raise Lfm2Exact4xProductionError(
            "component protection policy rules must be a nonempty JSON list."
        )
    if any(not isinstance(value, Mapping) for value in rule_payloads):
        raise Lfm2Exact4xProductionError(
            "component protection policy rules must be JSON objects."
        )
    return DenseComponentProtectionPolicy(
        rules=tuple(
            _component_protection_rule_from_payload(value, index=index)
            for index, value in enumerate(rule_payloads)
        ),
        bottleneck_risk_weight=_finite_nonnegative_float(
            values["bottleneck_risk_weight"],
            "component bottleneck risk weight",
        ),
    )


def seal_lfm2_exact4x_production_document(
    body: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copy sealed against configuration drift."""

    value = dict(body)
    required = {"schema", "request", "backend"}
    allowed = {*required, "component_protection_policy"}
    if (
        not required.issubset(value)
        or set(value) - allowed
        or value.get("schema") != SCHEMA
    ):
        raise Lfm2Exact4xProductionError("production request body is invalid.")
    if "component_protection_policy" in value:
        policy_payload = value["component_protection_policy"]
        if not isinstance(policy_payload, Mapping):
            raise Lfm2Exact4xProductionError(
                "component_protection_policy must be a JSON object."
            )
        value["component_protection_policy"] = (
            _component_protection_policy_from_payload(policy_payload).as_dict()
        )
    return {**value, "configuration_sha256": canonical_sha256(value)}


def load_lfm2_exact4x_production_registration(
    path: str | Path,
) -> Lfm2Exact4xProductionRegistration:
    """Load one immutable local JSON request; no network fallback exists."""

    source = Path(path).expanduser().absolute()
    if not source.is_file() or source.is_symlink():
        raise Lfm2Exact4xProductionError("config must be one real local JSON file.")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2Exact4xProductionError("could not read production config.") from exc
    if not isinstance(payload, Mapping):
        raise Lfm2Exact4xProductionError("production config must be a JSON object.")
    values = dict(payload)
    required = {"schema", "request", "backend", "configuration_sha256"}
    allowed = {*required, "component_protection_policy"}
    if not required.issubset(values) or set(values) - allowed:
        raise Lfm2Exact4xProductionError("production config fields are invalid.")
    seal = _digest(values.pop("configuration_sha256"), "configuration_sha256")
    if values.get("schema") != SCHEMA or canonical_sha256(values) != seal:
        raise Lfm2Exact4xProductionError("production config seal does not match.")
    request_payload = values["request"]
    backend_payload = values["backend"]
    if not isinstance(request_payload, Mapping) or not isinstance(
        backend_payload, Mapping
    ):
        raise Lfm2Exact4xProductionError("request and backend must be JSON objects.")
    request = _request_from_payload(request_payload)
    backend = _backend_from_payload(backend_payload)
    if "component_protection_policy" not in values:
        component_protection_policy = DenseComponentProtectionPolicy()
    elif isinstance(values["component_protection_policy"], Mapping):
        component_protection_policy = (
            _component_protection_policy_from_payload(
                values["component_protection_policy"]
            )
        )
    else:
        raise Lfm2Exact4xProductionError(
            "component_protection_policy must be a JSON object."
        )
    if backend.source_parent_name not in {None, request.parent_name}:
        raise Lfm2Exact4xProductionError(
            "backend source parent differs from the neutral request."
        )
    return Lfm2Exact4xProductionRegistration(
        request=request,
        backend=backend,
        configuration_sha256=seal,
        component_protection_policy=component_protection_policy,
    )


def assemble_lfm2_exact4x_production_entrypoint(
    registration: Lfm2Exact4xProductionRegistration,
) -> ConfiguredFullModelFoldEntrypoint:
    """Register the current real backend once behind the neutral entrypoint."""

    if not isinstance(registration, Lfm2Exact4xProductionRegistration):
        raise Lfm2Exact4xProductionError("registration has the wrong type.")
    request = registration.request
    component_protection_policy = registration.component_protection_policy
    budget = request.budget
    backend = registration.backend.verify()
    quarter_parameter_ceiling = backend.expected_parent_unique_parameters // 4
    if (
        budget.target_family != "lfm2"
        or budget.target_modalities != ("text",)
        or budget.target_unique_parameters is None
        or budget.target_unique_parameters > quarter_parameter_ceiling
        or budget.target_vocabulary_size is None
        or budget.match_policy not in {"exact", "at_or_below"}
    ):
        raise Lfm2Exact4xProductionError(
            "This registered backend requires a dense LFM2/text target at or "
            "below one quarter of its authenticated parent parameter count, "
            "using exact or at_or_below matching."
        )
    donor_tokenizer = load_lfm2_direct_local_tokenizer(
        backend.donor_tokenizer_directory
    )
    target_tokenizer = load_lfm2_direct_local_tokenizer(
        backend.target_tokenizer_directory
    )
    bridge = load_strict_tokenizer_bridge_artifact(
        backend.bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=backend.expected_bridge_artifact_sha256,
    )
    if (
        budget.target_vocabulary_size
        != bridge.transport_profile.target_row_capacity
        or budget.target_vocabulary_size
        != LFM2_NATIVE_EXACT4X_VOCABULARY_SIZE
    ):
        raise Lfm2Exact4xProductionError(
            "Requested target vocabulary, authenticated token bridge, and native "
            "semantic-kernel class must all use the same 4096-row interface."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production dependency
        raise Lfm2Exact4xProductionError("PyTorch is required for LFM2 folding.") from exc
    child_dtype = getattr(torch, backend.child_dtype)
    target_adapter = Lfm2TargetSynthesisAdapter(
        # Dense-budget requests carry their compiler-selected topology in the
        # authenticated catalog receipt; there is no manually pinned fallback.
        terminal_topology=None,
        evidence_fingerprint=backend.expected_semantic_cache_key,
        device=torch.device(backend.model_device),
        dtype=child_dtype,
        tokenizer_bridge_artifact=bridge,
    )
    topology_catalogs = DenseTargetTopologyCatalogRegistry(
        (
            Lfm2DenseTargetTopologyCatalog(
                tokenizer_bridge_artifact=bridge,
                target_token_interface_mode=(
                    LFM2_DENSE_TOKEN_INTERFACE_MODE_UNTIED
                ),
                component_protection_rules=(
                    component_protection_policy.rules
                ),
                component_bottleneck_risk_weight=(
                    component_protection_policy.bottleneck_risk_weight
                ),
            ),
        )
    )
    resolver = Lfm2Exact4xSemanticResourceResolver((backend,))
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry(
            (ResolvingSemanticWholeModelFoldStackAdapter(resolver),)
        ),
        inspectors=default_inspector_registry(),
        topology_catalogs=topology_catalogs,
    )

    def load_parent() -> Any:
        # The sealed caches and parent identity were captured from the canonical
        # CPU/BF16 source.  Child precision and placement are independent target
        # choices and must never mutate that authenticated parent contract.
        return load_lfm2_direct_local_model(
            backend.parent_model_directory,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )

    return ConfiguredFullModelFoldEntrypoint(
        system=system,
        parent_sources=(
            RegisteredParentSource(
                name=request.parent_name,
                tokenizer_fingerprint=bridge.donor_tokenizer_semantics_sha256,
                loader=load_parent,
            ),
        ),
        budget_parent_name=request.parent_name,
        child_tokenizer_fingerprint=bridge.target_tokenizer_semantics_sha256,
        evidence_fingerprint=backend.expected_semantic_cache_key,
        component_protection_policy_sha256=(
            component_protection_policy.fingerprint
        ),
        target_adapter_id=target_adapter.adapter_id,
        fold_method_id=DENSE_BUDGET_MODEL_METHOD,
    )


def run_lfm2_exact4x_production_config(
    path: str | Path,
) -> FullModelFoldExecution:
    registration = load_lfm2_exact4x_production_registration(path)
    entrypoint = assemble_lfm2_exact4x_production_entrypoint(registration)
    return entrypoint.run(registration.request)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neural-forge-lfm2-exact4x",
        description=(
            "Run one sealed complete-model exact-4x fold using only local assets."
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    execution = run_lfm2_exact4x_production_config(args.config)
    print(json.dumps(execution.summary(), allow_nan=False, sort_keys=True))
    return 0


__all__ = [
    "COMPONENT_PROTECTION_POLICY_SCHEMA",
    "ConfiguredFullModelFoldEntrypoint",
    "DenseComponentProtectionPolicy",
    "FullModelFoldExecution",
    "Lfm2Exact4xProductionError",
    "Lfm2Exact4xProductionRegistration",
    "NeutralFullModelFoldRequest",
    "RegisteredParentSource",
    "SCHEMA",
    "assemble_lfm2_exact4x_production_entrypoint",
    "build_parser",
    "load_lfm2_exact4x_production_registration",
    "main",
    "run_lfm2_exact4x_production_config",
    "seal_lfm2_exact4x_production_document",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
