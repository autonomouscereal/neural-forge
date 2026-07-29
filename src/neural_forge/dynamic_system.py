"""Persistent automatic catalog for target and folding-stack adapters.

The lower-level compiler accepts a target-adapter registry and a stack factory.
Those are useful extension boundaries, but a production folding service should
configure them once and then accept model/size/modality requests without
per-run family dispatch.  This module provides that persistent catalog.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dynamic_fold_compiler import (
    DynamicFoldSystemStack,
    SynthesizedPublishedDynamicFoldCompilation,
    run_dynamic_folding_system,
)
from .dynamic_fold_registry import ExperimentLedger, FoldMethodRegistry
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_model_inspector import InspectorRegistry
from .dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
)
from .dynamic_circuit_ir import ModelCircuitGraph
from .dynamic_target_synthesis import (
    BudgetedDynamicTargetPlan,
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisResult,
    plan_budgeted_dynamic_target,
)


class DynamicFoldingSystemError(RuntimeError):
    """The automatic catalog is invalid or cannot satisfy a request."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicFoldingSystemError(f"{name} must be a nonempty string.")
    return value.strip()


def _names(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise DynamicFoldingSystemError(f"{name} must be a tuple or list.")
    result = tuple(sorted({_text(value, name).lower() for value in values}))
    if not result:
        raise DynamicFoldingSystemError(f"{name} must not be empty.")
    return result


class DynamicFoldStackAdapter(Protocol):
    """One reusable family/modality implementation behind the neutral system."""

    adapter_id: str
    version: int
    priority: int
    target_families: tuple[str, ...]
    target_modalities: tuple[str, ...]
    method_ids: tuple[str, ...]

    def supports(self, target: DynamicTargetSynthesisResult) -> bool: ...

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack: ...


class DynamicFoldStackAdapterRegistry:
    """Deterministically choose a complete folding stack after synthesis."""

    def __init__(
        self,
        adapters: Sequence[DynamicFoldStackAdapter] = (),
    ) -> None:
        self._adapters: dict[str, DynamicFoldStackAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: DynamicFoldStackAdapter) -> None:
        adapter_id = _text(
            getattr(adapter, "adapter_id", None),
            "stack adapter_id",
        )
        version = getattr(adapter, "version", None)
        priority = getattr(adapter, "priority", None)
        if isinstance(version, bool) or not isinstance(version, int) or version <= 0:
            raise DynamicFoldingSystemError(
                "Stack adapter version must be a positive integer."
            )
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicFoldingSystemError(
                "Stack adapter priority must be an integer."
            )
        _names(
            getattr(adapter, "target_families", None),
            "stack target_families",
        )
        _names(
            getattr(adapter, "target_modalities", None),
            "stack target_modalities",
        )
        _names(
            getattr(adapter, "method_ids", None),
            "stack method_ids",
        )
        if not callable(getattr(adapter, "supports", None)) or not callable(
            getattr(adapter, "build", None)
        ):
            raise DynamicFoldingSystemError(
                "Stack adapters must implement supports() and build()."
            )
        if adapter_id in self._adapters:
            raise DynamicFoldingSystemError(
                f"Stack adapter {adapter_id!r} is already registered."
            )
        self._adapters[adapter_id] = adapter

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def select(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldStackAdapter:
        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicFoldingSystemError(
                "Stack selection requires a synthesized target result."
            )
        family = target.resource_profile.family.lower()
        modalities = set(target.resource_profile.modalities)
        candidates = []
        for adapter in self._adapters.values():
            declared_families = set(
                _names(adapter.target_families, "stack target_families")
            )
            declared_modalities = set(
                _names(adapter.target_modalities, "stack target_modalities")
            )
            if family not in declared_families and "auto" not in declared_families:
                continue
            if not modalities.issubset(declared_modalities):
                continue
            requested_method = target.request.fold_method_id
            if requested_method is not None and requested_method not in _names(
                adapter.method_ids, "stack method_ids"
            ):
                continue
            if adapter.supports(target):
                candidates.append(adapter)
        if not candidates:
            raise DynamicFoldingSystemError(
                "No registered folding-stack adapter supports the "
                f"synthesized {family}/{sorted(modalities)} target."
            )
        candidates.sort(key=lambda value: (-int(value.priority), str(value.adapter_id)))
        return candidates[0]

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        adapter = self.select(target)
        stack = adapter.build(target)
        if not isinstance(stack, DynamicFoldSystemStack):
            raise DynamicFoldingSystemError(
                f"Stack adapter {adapter.adapter_id!r} did not return "
                "DynamicFoldSystemStack."
            )
        return stack


@dataclass(slots=True)
class AutomaticDynamicFoldingSystem:
    """Configure folding implementations once, then submit neutral requests."""

    target_adapters: DynamicTargetAdapterRegistry
    stack_adapters: DynamicFoldStackAdapterRegistry
    inspectors: InspectorRegistry | None = None
    topology_catalogs: DenseTargetTopologyCatalogRegistry | None = None
    methods: FoldMethodRegistry | None = None
    ledger: ExperimentLedger | None = None
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    require_positive_compaction: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.target_adapters,
            DynamicTargetAdapterRegistry,
        ):
            raise DynamicFoldingSystemError(
                "target_adapters must be DynamicTargetAdapterRegistry."
            )
        if not isinstance(
            self.stack_adapters,
            DynamicFoldStackAdapterRegistry,
        ):
            raise DynamicFoldingSystemError(
                "stack_adapters must be DynamicFoldStackAdapterRegistry."
            )
        if not isinstance(self.require_positive_compaction, bool):
            raise DynamicFoldingSystemError(
                "require_positive_compaction must be boolean."
            )
        if self.topology_catalogs is not None and not isinstance(
            self.topology_catalogs,
            DenseTargetTopologyCatalogRegistry,
        ):
            raise DynamicFoldingSystemError(
                "topology_catalogs must be "
                "DenseTargetTopologyCatalogRegistry or None."
            )
        if self.ledger is None:
            self.ledger = ExperimentLedger()
        elif not isinstance(self.ledger, ExperimentLedger):
            raise DynamicFoldingSystemError("ledger must be ExperimentLedger or None.")

    def plan_budgeted_target(
        self,
        *,
        parent_graph: ModelCircuitGraph,
        budget_request: DenseParameterBudgetRequest,
        child_name: str,
        selected_candidate_rank: int | None = None,
        prefer_component_risk: bool = True,
        tokenizer_fingerprint: str | None = None,
        adapter_id: str | None = None,
        fold_method_id: str | None = None,
        blank_seed: int = 0,
    ) -> BudgetedDynamicTargetPlan:
        """Select and certify an exact target request without model creation."""

        return plan_budgeted_dynamic_target(
            parent_graph,
            budget_request,
            child_name=child_name,
            selected_candidate_rank=selected_candidate_rank,
            prefer_component_risk=prefer_component_risk,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=adapter_id,
            fold_method_id=fold_method_id,
            blank_seed=blank_seed,
            catalogs=self.topology_catalogs,
        )

    def run(
        self,
        *,
        parent_models: Mapping[str, Any],
        target_request: DynamicTargetRequest,
        evidence_fingerprint: str,
        destination: str | Path,
        parent_tokenizer_fingerprints: (Mapping[str, str | None] | None) = None,
    ) -> SynthesizedPublishedDynamicFoldCompilation:
        """Synthesize, select, fold, gate, publish, reload, and certify."""

        return run_dynamic_folding_system(
            parent_models=parent_models,
            target_request=target_request,
            target_adapters=self.target_adapters,
            stack_factory=self.stack_adapters.build,
            evidence_fingerprint=evidence_fingerprint,
            destination=destination,
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
            inspectors=self.inspectors,
            methods=self.methods,
            ledger=self.ledger,
            snapshot_config=self.snapshot_config,
            require_positive_compaction=self.require_positive_compaction,
        )


__all__ = [
    "AutomaticDynamicFoldingSystem",
    "DynamicFoldStackAdapter",
    "DynamicFoldStackAdapterRegistry",
    "DynamicFoldingSystemError",
]
