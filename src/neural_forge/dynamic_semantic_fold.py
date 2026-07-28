"""Family-neutral whole-model semantic folding transaction.

This module is the production seam between a family-specific analytic fold
compiler and :class:`~neural_forge.dynamic_system.AutomaticDynamicFoldingSystem`.
The compiler is called once for one planner-owned action which owns the entire
blank child's parameter surface.  Per-component information is retained as
diagnostic receipts inside that one transaction; it cannot turn component
experiments into independent partial-model objectives.

The seam deliberately does not prescribe attention, convolution, MLP, token,
or modality math.  A backend receives authenticated cached evidence and the
already-synthesized blank, and returns a sealed whole-model composition
receipt.  Runtime guards in the generic executor still forbid training,
pruning, and quantization and observe live parent execution.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .dynamic_behavior_gate import (
    BehaviorCallableIdentity,
    identify_behavior_callable,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldAction,
    FoldDecision,
    FoldEvaluationContext,
    FoldEvidence,
    FoldExecutionContext,
    FoldOperatorRegistry,
    fingerprint_model_state,
)
from .dynamic_fold_planner import DynamicFoldPlan
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_target_synthesis import DynamicTargetSynthesisResult


SCHEMA = "neural-forge-semantic-whole-model-fold-v1"


class DynamicSemanticFoldError(RuntimeError):
    """A semantic backend or its whole-model transaction is invalid."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DynamicSemanticFoldError(f"{name} must be a nonempty string.")
    return value.strip()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicSemanticFoldError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicSemanticFoldError(f"{name} must be a positive integer.")
    return value


def _canonical_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DynamicSemanticFoldError(f"{name} must be a mapping.")
    try:
        payload = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise DynamicSemanticFoldError(f"{name} must be canonical JSON.") from exc
    if not isinstance(payload, dict):
        raise DynamicSemanticFoldError(f"{name} must encode a JSON object.")
    return MappingProxyType(payload)


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(sorted({_text(value, name) for value in values}))
    if not result:
        raise DynamicSemanticFoldError(f"{name} must not be empty.")
    return result


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        values = tuple(
            model.named_parameters(recurse=True, remove_duplicate=False)
        )
    except TypeError:  # pragma: no cover - old Torch compatibility
        values = tuple(model.named_parameters(recurse=True))
    if not values or len({name for name, _value in values}) != len(values):
        raise DynamicSemanticFoldError(
            "The semantic target must expose uniquely named parameters."
        )
    return tuple(sorted(values, key=lambda item: item[0]))


def _named_buffers(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        values = tuple(model.named_buffers(recurse=True, remove_duplicate=False))
    except TypeError:  # pragma: no cover - old Torch compatibility
        values = tuple(model.named_buffers(recurse=True))
    return tuple(sorted(values, key=lambda item: item[0]))


def _capture_state_values(model: Any) -> tuple[tuple[Any, Any], ...]:
    """Capture unique tensor objects on CPU for exact transaction rollback."""

    seen: set[int] = set()
    result = []
    for _name, tensor in _named_parameters(model) + _named_buffers(model):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        result.append(
            (
                tensor,
                tensor.detach().to(device="cpu").contiguous().clone(),
            )
        )
    return tuple(result)


def _restore_state_values(values: Sequence[tuple[Any, Any]]) -> None:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise DynamicSemanticFoldError(
            "PyTorch is required to restore a semantic fold transaction."
        ) from exc
    with torch.no_grad():
        for destination, source in values:
            if tuple(destination.shape) != tuple(source.shape):
                raise DynamicSemanticFoldError(
                    "Semantic transaction rollback found changed tensor topology."
                )
            destination.copy_(
                source.to(device=destination.device, dtype=destination.dtype)
            )


def _explicit_resource_seal(value: Any, name: str) -> tuple[str, str]:
    verifier = getattr(value, "verify", None)
    if not callable(verifier):
        raise DynamicSemanticFoldError(f"{name} must expose verify().")
    verifier()
    for attribute in ("cache_key", "fingerprint", "evidence_fingerprint"):
        seal = getattr(value, attribute, None)
        if seal is not None:
            return attribute, _digest(seal, f"{name} {attribute}")
    raise DynamicSemanticFoldError(
        f"{name} must expose cache_key, fingerprint, or evidence_fingerprint."
    )


@dataclass(frozen=True, slots=True)
class SemanticWholeModelTargetBinding:
    """Declarative target identity shared by every family/modality backend."""

    binding_id: str
    target_family: str
    target_modalities: tuple[str, ...]
    target_adapter_id: str
    target_adapter_version: int
    method_id: str
    exact_target_unique_parameters: int
    parent_graph_fingerprints: tuple[tuple[str, str], ...]
    child_tokenizer_fingerprint: str | None = None
    target_configuration_sha256: str | None = None
    topology_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_id", _text(self.binding_id, "binding_id"))
        object.__setattr__(
            self,
            "target_family",
            _text(self.target_family, "target_family").lower(),
        )
        object.__setattr__(
            self,
            "target_modalities",
            tuple(
                sorted(
                    {
                        _text(value, "target modality").lower()
                        for value in self.target_modalities
                    }
                )
            ),
        )
        if not self.target_modalities:
            raise DynamicSemanticFoldError(
                "Semantic target binding requires at least one modality."
            )
        object.__setattr__(
            self,
            "target_adapter_id",
            _text(self.target_adapter_id, "target_adapter_id"),
        )
        object.__setattr__(
            self,
            "target_adapter_version",
            _positive_int(self.target_adapter_version, "target_adapter_version"),
        )
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "exact_target_unique_parameters",
            _positive_int(
                self.exact_target_unique_parameters,
                "exact_target_unique_parameters",
            ),
        )
        parents = tuple(
            sorted(
                (
                    _text(parent, "parent name"),
                    _digest(fingerprint, f"parent {parent} graph fingerprint"),
                )
                for parent, fingerprint in self.parent_graph_fingerprints
            )
        )
        if not parents or len({name for name, _digest_value in parents}) != len(
            parents
        ):
            raise DynamicSemanticFoldError(
                "Semantic target binding needs unique parent graph names."
            )
        object.__setattr__(self, "parent_graph_fingerprints", parents)
        for field_name in (
            "child_tokenizer_fingerprint",
            "target_configuration_sha256",
            "topology_fingerprint",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _digest(value, field_name))

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_whole_model_target_binding",
            "binding_id": self.binding_id,
            "target_family": self.target_family,
            "target_modalities": list(self.target_modalities),
            "target_adapter_id": self.target_adapter_id,
            "target_adapter_version": self.target_adapter_version,
            "method_id": self.method_id,
            "exact_target_unique_parameters": (
                self.exact_target_unique_parameters
            ),
            "parent_graph_fingerprints": dict(self.parent_graph_fingerprints),
            "child_tokenizer_fingerprint": self.child_tokenizer_fingerprint,
            "target_configuration_sha256": self.target_configuration_sha256,
            "topology_fingerprint": self.topology_fingerprint,
        }

    def matches(self, target: DynamicTargetSynthesisResult) -> bool:
        if not isinstance(target, DynamicTargetSynthesisResult):
            return False
        request = target.request
        return (
            target.adapter_id == self.target_adapter_id
            and target.adapter_version == self.target_adapter_version
            and target.resource_profile.family == self.target_family
            and target.graph.family == self.target_family
            and target.resource_profile.modalities == self.target_modalities
            and request.target_modalities == self.target_modalities
            and request.exact_unique_parameters
            == self.exact_target_unique_parameters
            and target.resource_profile.unique_parameters
            == self.exact_target_unique_parameters
            and target.graph.unique_parameter_count
            == self.exact_target_unique_parameters
            and request.fold_method_id in {None, self.method_id}
            and dict(target.parent_graph_fingerprints)
            == dict(self.parent_graph_fingerprints)
            and (
                self.child_tokenizer_fingerprint is None
                or (
                    request.tokenizer_fingerprint
                    == self.child_tokenizer_fingerprint
                    and target.graph.tokenizer_fingerprint
                    == self.child_tokenizer_fingerprint
                )
            )
            and (
                self.target_configuration_sha256 is None
                or canonical_sha256(dict(request.configuration))
                == self.target_configuration_sha256
            )
            and (
                self.topology_fingerprint is None
                or target.topology_fingerprint == self.topology_fingerprint
            )
            and target.topology_provenance_verified
            and target.blank_provenance is not None
            and target.blank_provenance.compiler_created_blank
        )


@dataclass(frozen=True, slots=True)
class SemanticDiagnosticInputReceipt:
    """One prior component experiment supplied as advisory compiler evidence."""

    component_id: str
    receipt: Mapping[str, Any]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "diagnostic component_id"),
        )
        receipt = _canonical_mapping(self.receipt, "diagnostic receipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", canonical_sha256(dict(receipt)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "receipt": dict(self.receipt),
            "receipt_sha256": self.receipt_sha256,
            "advisory_only": True,
            "acceptance_authority": False,
        }


class SemanticWholeModelFoldEvidence:
    """Sealed cache root plus component diagnostics for one analytic compile."""

    __slots__ = (
        "root_fingerprint",
        "source",
        "component_diagnostics",
        "_source_seal_name",
        "_source_seal",
        "_fingerprint",
    )

    def __init__(
        self,
        *,
        root_fingerprint: str,
        source: Any,
        component_diagnostics: Sequence[SemanticDiagnosticInputReceipt] = (),
    ) -> None:
        self.root_fingerprint = _digest(
            root_fingerprint,
            "semantic evidence root_fingerprint",
        )
        self.source = source
        self.component_diagnostics = tuple(component_diagnostics)
        if any(
            not isinstance(value, SemanticDiagnosticInputReceipt)
            for value in self.component_diagnostics
        ):
            raise DynamicSemanticFoldError(
                "component_diagnostics must contain typed advisory receipts."
            )
        component_ids = [value.component_id for value in self.component_diagnostics]
        if len(set(component_ids)) != len(component_ids):
            raise DynamicSemanticFoldError(
                "Diagnostic component receipt identifiers must be unique."
            )
        self._source_seal_name, self._source_seal = _explicit_resource_seal(
            source,
            "semantic evidence source",
        )
        self._fingerprint = canonical_sha256(self._body())
        self.verify()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_whole_model_fold_evidence",
            "root_fingerprint": self.root_fingerprint,
            "source_type": (
                f"{type(self.source).__module__}.{type(self.source).__qualname__}"
            ),
            "source_seal_name": self._source_seal_name,
            "source_seal": self._source_seal,
            "component_diagnostics": [
                value.as_dict() for value in self.component_diagnostics
            ],
            "cached_parent_evidence_only": True,
            "component_diagnostics_are_advisory_only": True,
        }

    def verify(self) -> "SemanticWholeModelFoldEvidence":
        seal_name, seal = _explicit_resource_seal(
            self.source,
            "semantic evidence source",
        )
        if (
            seal_name != self._source_seal_name
            or seal != self._source_seal
            or canonical_sha256(self._body()) != self._fingerprint
        ):
            raise DynamicSemanticFoldError(
                "Semantic evidence source or component receipts changed."
            )
        return self


@dataclass(frozen=True, slots=True)
class SemanticComponentFoldReceipt:
    """One component result inside a complete-model semantic transaction."""

    component_id: str
    covered_parameter_paths: tuple[str, ...]
    receipt: Mapping[str, Any]
    receipt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id",
            _text(self.component_id, "component_id"),
        )
        paths = _names(self.covered_parameter_paths, "covered parameter path")
        object.__setattr__(self, "covered_parameter_paths", paths)
        receipt = _canonical_mapping(self.receipt, "component fold receipt")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "receipt_sha256", canonical_sha256(dict(receipt)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "covered_parameter_paths": list(self.covered_parameter_paths),
            "receipt": dict(self.receipt),
            "receipt_sha256": self.receipt_sha256,
            "diagnostic_metrics_are_advisory_only": True,
            "independent_partial_model_goal": False,
        }


_ZERO_AUDIT_FIELDS = (
    "training_steps",
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "pruning_operations",
    "quantization_operations",
    "parent_forward_calls",
    "parent_trace_recaptures",
)


@dataclass(frozen=True, slots=True)
class SemanticWholeModelCompileResult:
    """Normalized result returned by a family-specific semantic compiler."""

    covered_target_parameter_paths: tuple[str, ...]
    component_receipts: tuple[SemanticComponentFoldReceipt, ...]
    whole_model_receipt: Mapping[str, Any]
    operation_audit: Mapping[str, Any]
    result_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        covered = _names(
            self.covered_target_parameter_paths,
            "covered target parameter path",
        )
        object.__setattr__(self, "covered_target_parameter_paths", covered)
        components = tuple(self.component_receipts)
        if not components or any(
            not isinstance(value, SemanticComponentFoldReceipt)
            for value in components
        ):
            raise DynamicSemanticFoldError(
                "A semantic whole-model result needs typed component receipts."
            )
        if len({value.component_id for value in components}) != len(components):
            raise DynamicSemanticFoldError(
                "Semantic component receipt identifiers must be unique."
            )
        component_coverage = {
            path for value in components for path in value.covered_parameter_paths
        }
        if component_coverage != set(covered):
            raise DynamicSemanticFoldError(
                "Component receipts do not compose the declared whole target."
            )
        object.__setattr__(self, "component_receipts", components)
        receipt = _canonical_mapping(
            self.whole_model_receipt,
            "whole-model semantic receipt",
        )
        audit = _canonical_mapping(self.operation_audit, "operation audit")
        if any(audit.get(name) != 0 for name in _ZERO_AUDIT_FIELDS):
            raise DynamicSemanticFoldError(
                "Semantic whole-model compile audit reports a forbidden operation."
            )
        required_flags = {
            "cached_parent_evidence_only": True,
            "complete_model_compaction": True,
            "component_diagnostics_have_acceptance_authority": False,
            "numeric_diagnostics_have_acceptance_authority": False,
        }
        if any(audit.get(name) is not value for name, value in required_flags.items()):
            raise DynamicSemanticFoldError(
                "Semantic operation audit does not preserve whole-model, "
                "cache-only, advisory-diagnostic semantics."
            )
        object.__setattr__(self, "whole_model_receipt", receipt)
        object.__setattr__(self, "operation_audit", audit)
        object.__setattr__(self, "result_sha256", canonical_sha256(self._body()))

    def _body(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_whole_model_compile_result",
            "covered_target_parameter_paths": list(
                self.covered_target_parameter_paths
            ),
            "component_receipts": [
                value.as_dict() for value in self.component_receipts
            ],
            "whole_model_receipt": dict(self.whole_model_receipt),
            "operation_audit": dict(self.operation_audit),
            "component_diagnostics_are_transaction_receipts": True,
            "complete_child_is_the_only_compaction_product": True,
        }

    def as_dict(self) -> dict[str, Any]:
        return {**self._body(), "result_sha256": self.result_sha256}


@dataclass(frozen=True, slots=True)
class SemanticWholeModelCompileRequest:
    """Family-neutral ABI passed to one registered semantic compiler."""

    target: DynamicTargetSynthesisResult
    action: FoldAction
    context: FoldExecutionContext
    evidence: SemanticWholeModelFoldEvidence


class SemanticWholeModelCompiler(Protocol):
    compiler_id: str
    version: int
    resource_fingerprint: str

    def compile(
        self,
        request: SemanticWholeModelCompileRequest,
    ) -> SemanticWholeModelCompileResult: ...


class SemanticWholeModelResourceResolver(Protocol):
    """Configure-once backend factory backed by sealed local catalogs.

    A resolver may select a target-specific cache bundle and compiler from the
    synthesized topology.  It is the family-neutral route from a persistent
    automatic catalog to exact-bound resources; it must not capture or run a
    parent model.
    """

    resolver_id: str
    version: int
    priority: int
    target_families: tuple[str, ...]
    target_modalities: tuple[str, ...]
    method_ids: tuple[str, ...]
    resource_fingerprint: str

    def supports(self, target: DynamicTargetSynthesisResult) -> bool: ...

    def resolve(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> "SemanticWholeModelFoldResources": ...


@dataclass(frozen=True, slots=True)
class SemanticWholeModelFoldResources:
    """One registered backend, evidence root, and publication boundary."""

    binding: SemanticWholeModelTargetBinding
    evidence: SemanticWholeModelFoldEvidence
    compiler: SemanticWholeModelCompiler
    final_evaluator: Any
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Any
    saver_id: str
    loader_id: str
    stack_adapter_id: str
    stack_adapter_version: int = 1
    priority: int = 400
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    publication_recorder: Callable[[Any], None] | None = None
    compiler_identity: BehaviorCallableIdentity = field(init=False)
    resource_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, SemanticWholeModelTargetBinding):
            raise DynamicSemanticFoldError(
                "binding must be SemanticWholeModelTargetBinding."
            )
        if not isinstance(self.evidence, SemanticWholeModelFoldEvidence):
            raise DynamicSemanticFoldError(
                "evidence must be SemanticWholeModelFoldEvidence."
            )
        self.evidence.verify()
        compiler_callback = getattr(self.compiler, "compile", None)
        if not callable(compiler_callback):
            raise DynamicSemanticFoldError(
                "semantic compiler must expose compile()."
            )
        compiler_id = _text(
            getattr(self.compiler, "compiler_id", None),
            "compiler_id",
        )
        compiler_version = _positive_int(
            getattr(self.compiler, "version", None),
            "compiler version",
        )
        _digest(
            getattr(self.compiler, "resource_fingerprint", None),
            "compiler resource_fingerprint",
        )
        identity = identify_behavior_callable(
            compiler_callback,
            implementation_id=compiler_id,
            version=compiler_version,
        )
        if not identity.registered:
            raise DynamicSemanticFoldError(
                "Semantic compiler implementation must be explicitly registered."
            )
        object.__setattr__(self, "compiler_identity", identity)
        for name in ("model_saver", "model_loader"):
            if not callable(getattr(self, name)):
                raise DynamicSemanticFoldError(f"{name} must be callable.")
        if self.publication_recorder is not None and not callable(
            self.publication_recorder
        ):
            raise DynamicSemanticFoldError(
                "publication_recorder must be callable or None."
            )
        object.__setattr__(self, "saver_id", _text(self.saver_id, "saver_id"))
        object.__setattr__(self, "loader_id", _text(self.loader_id, "loader_id"))
        object.__setattr__(
            self,
            "stack_adapter_id",
            _text(self.stack_adapter_id, "stack_adapter_id"),
        )
        object.__setattr__(
            self,
            "stack_adapter_version",
            _positive_int(self.stack_adapter_version, "stack_adapter_version"),
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise DynamicSemanticFoldError("priority must be an integer.")
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicSemanticFoldError(
                "snapshot_config must be DynamicFoldSnapshotConfig or None."
            )
        final_binding = resolve_behavior_gate_binding(self.final_evaluator)
        reload_binding = resolve_behavior_gate_binding(self.fresh_reload_evaluator)
        if final_binding.manifest_fingerprint != reload_binding.manifest_fingerprint:
            raise DynamicSemanticFoldError(
                "Semantic final and fresh-reload behavior gates differ."
            )
        resource_body = {
            "schema": SCHEMA,
            "kind": "semantic_whole_model_fold_resources",
            "target_binding_sha256": self.binding.fingerprint,
            "evidence_fingerprint": self.evidence.fingerprint,
            "evidence_root_fingerprint": self.evidence.root_fingerprint,
            "compiler_identity": identity.as_dict(),
            "behavior_gate_manifest_fingerprint": (
                final_binding.manifest_fingerprint
            ),
            "saver_id": self.saver_id,
            "loader_id": self.loader_id,
            "stack_adapter_id": self.stack_adapter_id,
            "stack_adapter_version": self.stack_adapter_version,
            "priority": self.priority,
            "snapshot_config": (
                None
                if self.snapshot_config is None
                else self.snapshot_config.as_dict()
            ),
            "family_neutral_operator_abi": True,
            "one_whole_model_transaction": True,
        }
        object.__setattr__(
            self,
            "resource_fingerprint",
            canonical_sha256(resource_body),
        )

    def verify(self) -> "SemanticWholeModelFoldResources":
        self.evidence.verify()
        observed = identify_behavior_callable(
            self.compiler.compile,
            implementation_id=self.compiler_identity.implementation_id,
            version=self.compiler_identity.version,
            code_fingerprint=self.compiler_identity.code_fingerprint,
            resource_fingerprint=self.compiler_identity.resource_fingerprint,
        )
        if observed.fingerprint != self.compiler_identity.fingerprint:
            raise DynamicSemanticFoldError(
                "Semantic compiler implementation changed after registration."
            )
        return self


class SemanticWholeModelEvidenceProvider:
    """Serve one sealed cache bundle without parent trace regeneration."""

    def __init__(self, evidence: SemanticWholeModelFoldEvidence) -> None:
        if not isinstance(evidence, SemanticWholeModelFoldEvidence):
            raise DynamicSemanticFoldError(
                "evidence must be SemanticWholeModelFoldEvidence."
            )
        self.evidence = evidence.verify()
        self.prepare_count = 0
        self.refresh_count = 0
        self.restore_count = 0

    @property
    def base_fingerprint(self) -> str:
        return self.evidence.root_fingerprint

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        self.evidence.verify()
        self.prepare_count += 1
        return FoldEvidence(
            self.evidence.fingerprint,
            payload=self.evidence,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
            root_fingerprint=self.evidence.root_fingerprint,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.evidence.verify()
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del action_ids
        if context.child_state_fingerprint != _digest(
            state_fingerprint,
            "restored child state fingerprint",
        ):
            raise DynamicSemanticFoldError(
                "Semantic evidence restore observed an unrestored child."
            )
        self.evidence.verify()
        self.restore_count += 1


class SemanticWholeModelTransaction:
    """Rollback-capable transaction for one complete semantic child install."""

    def __init__(
        self,
        *,
        method_id: str,
        context: FoldExecutionContext,
        incumbent_state_fingerprint: str,
        candidate_state_fingerprint: str,
        snapshot: Sequence[tuple[Any, Any]],
        result: SemanticWholeModelCompileResult,
        resource_fingerprint: str,
    ) -> None:
        self.method_id = _text(method_id, "transaction method_id")
        self.context = context
        self.incumbent_state_fingerprint = _digest(
            incumbent_state_fingerprint,
            "incumbent_state_fingerprint",
        )
        self.candidate_state_fingerprint = _digest(
            candidate_state_fingerprint,
            "candidate_state_fingerprint",
        )
        self.snapshot = tuple(snapshot)
        self.result = result
        self.resource_fingerprint = _digest(
            resource_fingerprint,
            "transaction resource_fingerprint",
        )
        self.active = True
        self.verified_noop = False

    def _report(self, *, action: str, accepted: bool) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "semantic_whole_model_transaction",
            "method_id": self.method_id,
            "action": action,
            "accepted": accepted,
            "incumbent_state_fingerprint": self.incumbent_state_fingerprint,
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "final_state_fingerprint": self.context.child_state_fingerprint,
            "semantic_compile_result": self.result.as_dict(),
            "semantic_compile_result_sha256": self.result.result_sha256,
            "resource_fingerprint": self.resource_fingerprint,
            "complete_model_transaction": True,
            "component_receipts_are_diagnostics_inside_transaction": True,
        }

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicSemanticFoldError(
                "Cannot commit an inactive semantic transaction."
            )
        if self.context.child_state_fingerprint != self.candidate_state_fingerprint:
            raise DynamicSemanticFoldError(
                "Semantic candidate changed before transaction commit."
            )
        self.active = False
        return self._report(action="commit", accepted=True)

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicSemanticFoldError(
                "Cannot roll back an inactive semantic transaction."
            )
        _restore_state_values(self.snapshot)
        if self.context.child_state_fingerprint != self.incumbent_state_fingerprint:
            raise DynamicSemanticFoldError(
                "Semantic transaction rollback did not restore the blank child."
            )
        self.active = False
        return self._report(action="rollback", accepted=False)


class SemanticWholeModelFoldOperator:
    """Invoke one registered backend for one complete planner-owned action."""

    def __init__(
        self,
        target: DynamicTargetSynthesisResult,
        resources: SemanticWholeModelFoldResources,
    ) -> None:
        if not isinstance(target, DynamicTargetSynthesisResult):
            raise DynamicSemanticFoldError(
                "Semantic operator target must be DynamicTargetSynthesisResult."
            )
        if not isinstance(resources, SemanticWholeModelFoldResources):
            raise DynamicSemanticFoldError(
                "resources must be SemanticWholeModelFoldResources."
            )
        resources.verify()
        if not resources.binding.matches(target):
            raise DynamicSemanticFoldError(
                "Semantic resources are bound to a different synthesized target."
            )
        self.target = target
        self.resources = resources
        self.method_id = resources.binding.method_id
        self.resource_fingerprint = resources.resource_fingerprint

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> SemanticWholeModelTransaction:
        resources = self.resources.verify()
        if (
            action.method_id != self.method_id
            or action.compression_delta <= 0
            or evidence.payload is not resources.evidence
            or evidence.root_fingerprint != resources.evidence.root_fingerprint
            or evidence.fingerprint != resources.evidence.fingerprint
        ):
            raise DynamicSemanticFoldError(
                "Semantic operator received a different method, evidence, or "
                "non-compressive action."
            )
        expected_parents = dict(resources.binding.parent_graph_fingerprints)
        observed_parents = {
            name: graph.fingerprint for name, graph in context.parent_graphs.items()
        }
        target_paths = {name for name, _value in _named_parameters(context.child_model)}
        if (
            observed_parents != expected_parents
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
            or set(action.target_parameter_paths) != target_paths
            or set(action.target_parameter_paths)
            != set(context.child_graph.parameter_paths)
        ):
            raise DynamicSemanticFoldError(
                "Semantic action does not own the complete authenticated parent "
                "and target surfaces."
            )
        incumbent = context.child_state_fingerprint
        if (
            incumbent != self.target.blank_state_fingerprint
            or not resources.binding.matches(self.target)
        ):
            raise DynamicSemanticFoldError(
                "Semantic whole-model compilation did not start from its blank."
            )
        snapshot = _capture_state_values(context.child_model)
        request = SemanticWholeModelCompileRequest(
            target=self.target,
            action=action,
            context=context,
            evidence=resources.evidence,
        )
        try:
            result = resources.compiler.compile(request)
            if not isinstance(result, SemanticWholeModelCompileResult):
                raise DynamicSemanticFoldError(
                    "Semantic compiler must return SemanticWholeModelCompileResult."
                )
            if set(result.covered_target_parameter_paths) != target_paths:
                raise DynamicSemanticFoldError(
                    "Semantic compiler result does not cover every target parameter."
                )
            candidate = fingerprint_model_state(context.child_model)
            if candidate == incumbent:
                raise DynamicSemanticFoldError(
                    "Semantic whole-model compiler left the blank unchanged."
                )
            resources.evidence.verify()
        except BaseException:
            _restore_state_values(snapshot)
            raise
        return SemanticWholeModelTransaction(
            method_id=self.method_id,
            context=context,
            incumbent_state_fingerprint=incumbent,
            candidate_state_fingerprint=candidate,
            snapshot=snapshot,
            result=result,
            resource_fingerprint=resources.resource_fingerprint,
        )


class SemanticWholeModelAdmissionEvaluator:
    """Admit structural completion; decoded behavior remains the final gate."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        transaction = context.transaction
        if not isinstance(transaction, SemanticWholeModelTransaction):
            return FoldDecision.reject(
                "Semantic structural admission received another transaction type."
            )
        result = transaction.result
        expected_paths = set(context.action.target_parameter_paths)
        if (
            set(result.covered_target_parameter_paths) != expected_paths
            or context.candidate_state_fingerprint
            != transaction.candidate_state_fingerprint
            or context.candidate_state_fingerprint
            == context.incumbent_state_fingerprint
        ):
            return FoldDecision.reject(
                "Semantic transaction did not install one complete nonblank child."
            )
        return FoldDecision.accept(
            "Complete semantic child installed; decoded whole-output review is "
            "the final intelligence authority.",
            {
                "covered_target_parameter_count": len(expected_paths),
                "component_receipt_count": len(result.component_receipts),
                "complete_model_transaction": True,
                "numeric_diagnostics_used_as_gate": False,
                "parent_forward_passes": 0,
                "parent_trace_recaptures": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
            },
        )


class SemanticWholeModelFoldInitializer:
    """Prove one action owns a compiler-created blank and all target paths."""

    def __init__(
        self,
        target: DynamicTargetSynthesisResult,
        resources: SemanticWholeModelFoldResources,
    ) -> None:
        self.target = target
        self.resources = resources

    def initialize(
        self,
        plan: DynamicFoldPlan,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        resources = self.resources.verify()
        binding = resources.binding
        before = context.child_state_fingerprint
        if (
            not binding.matches(self.target)
            or before != self.target.blank_state_fingerprint
            or plan.child_state_fingerprint != before
            or plan.evidence_fingerprint != resources.evidence.root_fingerprint
            or plan.exact_target_unique_parameters
            != binding.exact_target_unique_parameters
            or len(plan.actions) != 1
        ):
            raise DynamicSemanticFoldError(
                "Semantic initializer target, blank, evidence, budget, or action "
                "count changed."
            )
        action = plan.actions[0]
        parameter_paths = {name for name, _value in _named_parameters(context.child_model)}
        if (
            action.method_id != binding.method_id
            or not action.signature.zero_gradient
            or action.compression_delta <= 0
            or action.dependencies
            or set(action.target_node_ids)
            != {node.node_id for node in context.child_graph.nodes}
            or set(action.target_parameter_paths) != parameter_paths
        ):
            raise DynamicSemanticFoldError(
                "Semantic target must plan as one positive-compaction, "
                "zero-gradient, complete-model action."
            )
        if context.child_state_fingerprint != before:
            raise DynamicSemanticFoldError(
                "Semantic whole-model initializer mutated the blank child."
            )
        return {
            "kind": "semantic_whole_model_fold_only",
            "child_state_fingerprint": before,
            "target_binding_sha256": binding.fingerprint,
            "resource_fingerprint": resources.resource_fingerprint,
            "action_count": 1,
            "complete_target_coverage": True,
            "compiler_created_blank_verified": True,
            "component_diagnostics_are_advisory_receipts": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
        }


class BoundSemanticWholeModelFoldStackAdapter:
    """Exact-resource stack registration used by the automatic facade.

    This adapter is intentionally bound to one target/evidence/compiler
    identity.  Use :class:`ResolvingSemanticWholeModelFoldStackAdapter` for a
    configure-once catalog which resolves many target requests dynamically.
    """

    def __init__(self, resources: SemanticWholeModelFoldResources) -> None:
        if not isinstance(resources, SemanticWholeModelFoldResources):
            raise DynamicSemanticFoldError(
                "resources must be SemanticWholeModelFoldResources."
            )
        self.resources = resources.verify()
        binding = resources.binding
        self.adapter_id = resources.stack_adapter_id
        self.version = resources.stack_adapter_version
        self.priority = resources.priority
        self.target_families = (binding.target_family,)
        self.target_modalities = binding.target_modalities
        self.method_ids = (binding.method_id,)

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            self.resources.verify()
            return self.resources.binding.matches(target)
        except Exception:
            return False

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        if not self.supports(target):
            raise DynamicSemanticFoldError(
                "Semantic resources do not support the synthesized target."
            )
        resources = self.resources.verify()
        behavior = resolve_behavior_gate_binding(resources.final_evaluator)
        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry(
                (SemanticWholeModelFoldOperator(target, resources),)
            ),
            evidence_provider=SemanticWholeModelEvidenceProvider(
                resources.evidence
            ),
            evaluator=SemanticWholeModelAdmissionEvaluator(),
            final_evaluator=resources.final_evaluator,
            model_saver=resources.model_saver,
            model_loader=resources.model_loader,
            fresh_reload_evaluator=resources.fresh_reload_evaluator,
            saver_id=resources.saver_id,
            loader_id=resources.loader_id,
            behavior_gate_id=behavior.gate_id,
            behavior_gate_version=behavior.gate_version,
            behavior_baseline_fingerprint=behavior.baseline_fingerprint,
            behavior_gate_manifest_fingerprint=behavior.manifest_fingerprint,
            behavior_gate_manifest=behavior.as_dict(),
            initializer=SemanticWholeModelFoldInitializer(target, resources),
            snapshot_config=resources.snapshot_config,
            evidence_fingerprint=resources.evidence.root_fingerprint,
            publication_recorder=resources.publication_recorder,
        )


class ResolvingSemanticWholeModelFoldStackAdapter:
    """Persistent family-neutral stack backed by a target resource resolver."""

    def __init__(self, resolver: SemanticWholeModelResourceResolver) -> None:
        resolve = getattr(resolver, "resolve", None)
        supports = getattr(resolver, "supports", None)
        if not callable(resolve) or not callable(supports):
            raise DynamicSemanticFoldError(
                "Semantic resource resolver must expose supports() and resolve()."
            )
        self.resolver = resolver
        self.adapter_id = _text(
            getattr(resolver, "resolver_id", None),
            "resolver_id",
        )
        self.version = _positive_int(
            getattr(resolver, "version", None),
            "resolver version",
        )
        priority = getattr(resolver, "priority", None)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise DynamicSemanticFoldError("resolver priority must be an integer.")
        self.priority = priority
        self.target_families = tuple(
            value.lower()
            for value in _names(
                getattr(resolver, "target_families", ()),
                "resolver target family",
            )
        )
        self.target_modalities = tuple(
            value.lower()
            for value in _names(
                getattr(resolver, "target_modalities", ()),
                "resolver target modality",
            )
        )
        self.method_ids = _names(
            getattr(resolver, "method_ids", ()),
            "resolver method_id",
        )
        _digest(
            getattr(resolver, "resource_fingerprint", None),
            "resolver resource_fingerprint",
        )
        self._resolve_identity = identify_behavior_callable(
            resolve,
            implementation_id=self.adapter_id,
            version=self.version,
        )
        self._supported_resources: dict[
            str,
            SemanticWholeModelFoldResources,
        ] = {}
        self._lock = threading.RLock()

    def _resolve(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> SemanticWholeModelFoldResources:
        identify_behavior_callable(
            self.resolver.resolve,
            implementation_id=self._resolve_identity.implementation_id,
            version=self._resolve_identity.version,
            code_fingerprint=self._resolve_identity.code_fingerprint,
            resource_fingerprint=self._resolve_identity.resource_fingerprint,
        )
        resources = self.resolver.resolve(target)
        if not isinstance(resources, SemanticWholeModelFoldResources):
            raise DynamicSemanticFoldError(
                "Semantic resolver did not return typed fold resources."
            )
        resources.verify()
        binding = resources.binding
        if (
            not binding.matches(target)
            or binding.target_family not in self.target_families
            or not set(binding.target_modalities).issubset(self.target_modalities)
            or binding.method_id not in self.method_ids
        ):
            raise DynamicSemanticFoldError(
                "Resolved semantic resources differ from the synthesized target "
                "or resolver capability declaration."
            )
        return resources

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        try:
            if not self.resolver.supports(target):
                return False
            resources = self._resolve(target)
        except Exception:
            return False
        with self._lock:
            # Keep the exact verified object through stack construction.  A
            # real resolver may have opened and authenticated a very large
            # selective cache; resolving a second time would waste that work
            # and could create two nominally equivalent cache sessions.
            self._supported_resources[target.fingerprint] = resources
        return True

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        with self._lock:
            resources = self._supported_resources.pop(
                target.fingerprint,
                None,
            )
        if resources is None:
            resources = self._resolve(target)
        else:
            resources.verify()
        return BoundSemanticWholeModelFoldStackAdapter(resources).build(target)


__all__ = [
    "SCHEMA",
    "DynamicSemanticFoldError",
    "SemanticComponentFoldReceipt",
    "SemanticDiagnosticInputReceipt",
    "SemanticWholeModelAdmissionEvaluator",
    "SemanticWholeModelCompileRequest",
    "SemanticWholeModelCompileResult",
    "SemanticWholeModelCompiler",
    "SemanticWholeModelEvidenceProvider",
    "SemanticWholeModelFoldEvidence",
    "SemanticWholeModelFoldInitializer",
    "SemanticWholeModelFoldOperator",
    "SemanticWholeModelFoldResources",
    "SemanticWholeModelResourceResolver",
    "SemanticWholeModelTargetBinding",
    "SemanticWholeModelTransaction",
    "BoundSemanticWholeModelFoldStackAdapter",
    "ResolvingSemanticWholeModelFoldStackAdapter",
]
