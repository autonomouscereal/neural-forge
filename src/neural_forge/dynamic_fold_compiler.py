"""Public model-agnostic folding compiler facade."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .dynamic_circuit_ir import ModelCircuitGraph
from .dynamic_behavior_gate import (
    DynamicBehaviorGateError,
    observe_model_execution,
    resolve_behavior_gate_binding,
)
from .dynamic_fold_artifact import (
    DynamicFoldArtifactCertificate,
    DynamicFoldArtifactError,
    publish_dynamic_fold_artifact,
    remove_authenticated_dynamic_fold_artifact,
)
from .dynamic_fold_executor import (
    DynamicFoldExecutor,
    DynamicFoldRun,
    FoldEvaluator,
    FoldEvidenceProvider,
    FoldExecutionContext,
    FoldInitializer,
    FoldOnlyGuardReport,
    FoldOperatorRegistry,
    fingerprint_model_runtime_state,
    fingerprint_model_state,
    run_fold_only_callback,
)
from .dynamic_fold_planner import (
    DynamicFoldPlan,
    DynamicFoldPlanner,
    DynamicFoldPlanningError,
    canonical_resource_fingerprints,
)
from .dynamic_fold_registry import (
    ClosureLevel,
    ExperimentLedger,
    FoldMethodRegistry,
    LedgerRecord,
    RecordStatus,
    default_method_registry,
)
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_model_inspector import (
    InspectorRegistry,
    default_inspector_registry,
)
from .dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
    DynamicTargetSynthesisResult,
    DynamicTargetSynthesizer,
    serialized_inference_state_inventory,
)


class DynamicFoldCompilerError(RuntimeError):
    pass


@contextmanager
def _observe_parent_model_execution(parent_models: Mapping[str, Any]):
    """Observe every parent module throughout one compiler callback phase."""

    counters: dict[str, Any] = {
        "module_forward_calls": 0,
        "root_forward_calls": 0,
        "observers": [],
    }
    with ExitStack() as stack:
        for _name, model in sorted(parent_models.items()):
            counters["observers"].append(
                stack.enter_context(observe_model_execution(model))
            )
        yield counters
    counters["module_forward_calls"] = sum(
        int(value["module_forward_calls"]) for value in counters["observers"]
    )
    counters["root_forward_calls"] = sum(
        int(value["root_forward_calls"]) for value in counters["observers"]
    )


def _parent_forward_counts(value: Mapping[str, Any]) -> dict[str, int]:
    return {
        "module_forward_calls": int(value["module_forward_calls"]),
        "root_forward_calls": int(value["root_forward_calls"]),
    }


def _sum_parent_forward_counts(
    values: tuple[Mapping[str, Any], ...],
) -> dict[str, int]:
    return {
        name: sum(int(value[name]) for value in values)
        for name in ("module_forward_calls", "root_forward_calls")
    }


def _require_zero_parent_forwards(
    value: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    counts = _parent_forward_counts(value)
    if counts["module_forward_calls"] != 0 or counts["root_forward_calls"] != 0:
        raise DynamicFoldCompilerError(
            f"{phase} executed a live parent model; cached evidence is required."
        )


def _parameter_storage_identity(parameter: Any) -> tuple[Any, ...]:
    """Return a conservative learned-storage identity for cross-model aliases."""

    try:
        storage = parameter.untyped_storage()
        pointer = int(storage.data_ptr())
        if pointer:
            return (
                "storage",
                str(getattr(parameter, "device", "")),
                pointer,
                int(storage.nbytes()),
            )
        return ("storage_object", id(storage))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return ("parameter_object", id(parameter))


def validate_disjoint_parent_models(
    parent_models: Mapping[str, Any],
) -> int | None:
    """Reject aliases and return the distinct learned-parameter count.

    ``None`` means a lightweight test double does not expose ``parameters()``;
    production positive-compaction paths require a concrete count.
    """

    parents = dict(parent_models)
    if not parents:
        raise DynamicFoldCompilerError("At least one loaded parent model is required.")
    identities: dict[int, str] = {}
    for name, model in sorted(parents.items()):
        if id(model) in identities:
            raise DynamicFoldCompilerError(
                "Parent model mappings must not alias the same model object: "
                f"{identities[id(model)]!r} and {name!r}."
            )
        identities[id(model)] = name

    complete = True
    total = 0
    seen_module_objects: dict[int, str] = {}
    seen_state_objects: dict[int, str] = {}
    seen_state_storages: dict[tuple[Any, ...], str] = {}
    for name, model in sorted(parents.items()):
        parameters = getattr(model, "parameters", None)
        if not callable(parameters):
            complete = False
            continue
        modules = getattr(model, "modules", None)
        if callable(modules):
            for module in modules():
                previous_module = seen_module_objects.get(id(module))
                if previous_module is not None and previous_module != name:
                    raise DynamicFoldCompilerError(
                        "Declared parent models share a module object: "
                        f"{previous_module!r} and {name!r}."
                    )
                seen_module_objects[id(module)] = name
        local_objects: set[int] = set()
        local_storage_counts: dict[tuple[Any, ...], int] = {}
        for parameter in parameters():
            object_identity = id(parameter)
            storage_identity = _parameter_storage_identity(parameter)
            if object_identity in local_objects:
                continue
            local_objects.add(object_identity)
            previous_object = seen_state_objects.get(object_identity)
            previous_storage = seen_state_storages.get(storage_identity)
            if previous_object is not None:
                raise DynamicFoldCompilerError(
                    "Declared parent models share a Parameter object: "
                    f"{previous_object!r} and {name!r}."
                )
            if previous_storage is not None and previous_storage != name:
                raise DynamicFoldCompilerError(
                    "Declared parent models share learned parameter storage: "
                    f"{previous_storage!r} and {name!r}."
                )
            seen_state_objects[object_identity] = name
            seen_state_storages[storage_identity] = name
            numel = getattr(parameter, "numel", None)
            if not callable(numel):
                complete = False
            else:
                value = numel()
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise DynamicFoldCompilerError(
                        "A parent parameter reports an invalid element count."
                    )
                local_storage_counts[storage_identity] = max(
                    value,
                    local_storage_counts.get(storage_identity, 0),
                )
        total += sum(local_storage_counts.values())
        buffers = getattr(model, "buffers", None)
        if callable(buffers):
            for buffer in buffers():
                object_identity = id(buffer)
                storage_identity = _parameter_storage_identity(buffer)
                previous_object = seen_state_objects.get(object_identity)
                previous_storage = seen_state_storages.get(storage_identity)
                if previous_object is not None and previous_object != name:
                    raise DynamicFoldCompilerError(
                        "Declared parent models share a buffer object: "
                        f"{previous_object!r} and {name!r}."
                    )
                if previous_storage is not None and previous_storage != name:
                    raise DynamicFoldCompilerError(
                        "Declared parent models share inference-state storage: "
                        f"{previous_storage!r} and {name!r}."
                    )
                seen_state_objects[object_identity] = name
                seen_state_storages[storage_identity] = name
    return total if complete else None


def _sum_guard_reports(
    reports: tuple[FoldOnlyGuardReport, ...],
) -> dict[str, int]:
    return {
        "optimizer_constructions": sum(
            value.optimizer_constructions for value in reports
        ),
        "optimizer_steps": sum(value.optimizer_steps for value in reports),
        "backward_calls": sum(value.backward_calls for value in reports),
        "autograd_grad_calls": sum(value.autograd_grad_calls for value in reports),
    }


def _staged_ledger_extension(
    destination: ExperimentLedger,
    staged: ExperimentLedger,
    base_records: tuple[Any, ...],
) -> tuple[LedgerRecord, ...]:
    """Return a private extension only if its base is still current."""

    base_hashes = tuple(record.record_sha256 for record in base_records)
    destination_hashes = tuple(record.record_sha256 for record in destination.records)
    staged_hashes = tuple(record.record_sha256 for record in staged.records)
    if destination_hashes != base_hashes or staged_hashes[: len(base_hashes)] != (
        base_hashes
    ):
        raise DynamicFoldCompilerError(
            "Cannot commit the staged execution ledger because its base "
            "hash chain changed."
        )
    return tuple(staged.records[len(base_hashes) :])


def _merge_staged_ledger(
    destination: ExperimentLedger,
    staged: ExperimentLedger,
    base_records: tuple[Any, ...],
) -> None:
    """Atomically commit a private execution-ledger extension."""

    destination.append_batch(
        _staged_ledger_extension(destination, staged, base_records)
    )


def _rollback_published_artifact(
    *,
    ledger: ExperimentLedger,
    provisional_record_sha256s: tuple[str, ...],
    artifact: DynamicFoldArtifactCertificate,
    reason_class: str,
    reason: str,
) -> None:
    """Block records, remove the artifact, then finalize supersession.

    The rollback-pending record is the crash boundary: once it is durable, an
    accepted record is no longer effective even if deletion succeeds and the
    final supersession append subsequently fails.
    """

    try:
        pending = ledger.prepare_supersession(
            provisional_record_sha256s,
            reason_class=f"{reason_class}_pending",
            reason=(f"Rollback is pending authenticated artifact removal. {reason}"),
        )
    except BaseException as exc:
        raise DynamicFoldCompilerError(
            "Could not durably mark provisional records rollback-pending; "
            "the certified artifact was retained."
        ) from exc
    artifact_path = Path(artifact.path)
    try:
        removed = remove_authenticated_dynamic_fold_artifact(
            artifact_path,
            expected_parent=artifact_path.parent,
            expected_name=artifact_path.name,
        )
    except BaseException as exc:
        raise DynamicFoldCompilerError(
            "Rollback was durably marked pending, but the certified artifact "
            "could not be safely removed."
        ) from exc
    if removed != artifact:
        raise DynamicFoldCompilerError(
            "Removed artifact certificate differed from the published result."
        )
    try:
        ledger.supersede_records(
            tuple(record.record_sha256 for record in pending),
            reason_class=reason_class,
            reason=reason,
        )
    except BaseException as exc:
        raise DynamicFoldCompilerError(
            "The certified artifact was removed; its provisional records "
            "remain safely blocked in supersession-pending state because "
            "finalization failed."
        ) from exc


@dataclass(frozen=True, slots=True)
class DynamicFoldCompilation:
    parent_graphs: tuple[ModelCircuitGraph, ...]
    child_graph: ModelCircuitGraph
    plan: DynamicFoldPlan
    run: DynamicFoldRun

    @property
    def complete(self) -> bool:
        return self.run.complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_graphs": [value.as_dict() for value in self.parent_graphs],
            "child_graph": self.child_graph.as_dict(),
            "plan": self.plan.as_dict(),
            "run": self.run.as_dict(),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class PublishedDynamicFoldCompilation:
    """One accepted compilation plus its atomic fresh-reload certificate."""

    compilation: DynamicFoldCompilation
    artifact: DynamicFoldArtifactCertificate

    @property
    def complete(self) -> bool:
        return self.compilation.complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "compilation": self.compilation.as_dict(),
            "artifact": self.artifact.as_dict(),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class SynthesizedPublishedDynamicFoldCompilation:
    """Target synthesis plus fold, semantic gate, and artifact certificate."""

    target: DynamicTargetSynthesisResult
    published: PublishedDynamicFoldCompilation

    @property
    def complete(self) -> bool:
        return self.published.complete

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.as_dict(),
            "published": self.published.as_dict(),
            "complete": self.complete,
        }


@dataclass(frozen=True, slots=True)
class DynamicFoldSystemStack:
    """All family/modality callbacks selected after target synthesis."""

    operators: FoldOperatorRegistry
    evidence_provider: FoldEvidenceProvider
    evaluator: FoldEvaluator
    final_evaluator: FoldEvaluator
    model_saver: Callable[[Any, Path], None]
    model_loader: Callable[[Path], Any]
    fresh_reload_evaluator: Callable[[Any], Any]
    saver_id: str
    loader_id: str
    behavior_gate_id: str
    behavior_gate_version: int
    behavior_baseline_fingerprint: str
    behavior_gate_manifest_fingerprint: str | None = None
    behavior_gate_manifest: Mapping[str, Any] | None = None
    initializer: FoldInitializer | None = None
    snapshot_config: DynamicFoldSnapshotConfig | None = None
    evidence_fingerprint: str | None = None
    resource_fingerprints: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
    publication_recorder: (
        Callable[[SynthesizedPublishedDynamicFoldCompilation], None] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.operators, FoldOperatorRegistry):
            raise DynamicFoldCompilerError(
                "System stack operators must be FoldOperatorRegistry."
            )
        for name in (
            "model_saver",
            "model_loader",
            "fresh_reload_evaluator",
        ):
            if not callable(getattr(self, name)):
                raise DynamicFoldCompilerError(f"System stack {name} must be callable.")
        for name in ("saver_id", "loader_id", "behavior_gate_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DynamicFoldCompilerError(f"System stack {name} must be nonempty.")
        if (
            isinstance(self.behavior_gate_version, bool)
            or not isinstance(self.behavior_gate_version, int)
            or self.behavior_gate_version <= 0
        ):
            raise DynamicFoldCompilerError(
                "System stack behavior_gate_version must be positive."
            )
        if (
            not isinstance(self.behavior_baseline_fingerprint, str)
            or len(self.behavior_baseline_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.behavior_baseline_fingerprint
            )
        ):
            raise DynamicFoldCompilerError(
                "System stack behavior_baseline_fingerprint must be a "
                "lowercase SHA-256 digest."
            )
        try:
            final_binding = resolve_behavior_gate_binding(self.final_evaluator)
            reload_binding = resolve_behavior_gate_binding(self.fresh_reload_evaluator)
        except DynamicBehaviorGateError as exc:
            raise DynamicFoldCompilerError(
                "Final and fresh-reload evaluators must expose a canonical "
                "behavior-gate manifest."
            ) from exc
        if final_binding.manifest_fingerprint != reload_binding.manifest_fingerprint:
            raise DynamicFoldCompilerError(
                "Final and fresh-reload evaluators use different behavior gates."
            )
        declared_manifest_fingerprint = (
            final_binding.manifest_fingerprint
            if self.behavior_gate_manifest_fingerprint is None
            else self.behavior_gate_manifest_fingerprint
        )
        declared_manifest = (
            final_binding.as_dict()
            if self.behavior_gate_manifest is None
            else dict(self.behavior_gate_manifest)
        )
        if (
            self.behavior_gate_id != final_binding.gate_id
            or self.behavior_gate_version != final_binding.gate_version
            or self.behavior_baseline_fingerprint != final_binding.baseline_fingerprint
            or declared_manifest_fingerprint != final_binding.manifest_fingerprint
            or declared_manifest != final_binding.as_dict()
        ):
            raise DynamicFoldCompilerError(
                "System stack behavior metadata differs from its evaluator manifest."
            )
        object.__setattr__(
            self,
            "behavior_gate_manifest_fingerprint",
            final_binding.manifest_fingerprint,
        )
        object.__setattr__(
            self,
            "behavior_gate_manifest",
            final_binding.manifest,
        )
        if self.snapshot_config is not None and not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicFoldCompilerError(
                "System stack snapshot_config must be "
                "DynamicFoldSnapshotConfig or None."
            )
        if self.evidence_fingerprint is not None:
            value = self.evidence_fingerprint
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise DynamicFoldCompilerError(
                    "System stack evidence_fingerprint must be a lowercase "
                    "SHA-256 digest or None."
                )
        try:
            resource_fingerprints = canonical_resource_fingerprints(
                self.resource_fingerprints
            )
        except DynamicFoldPlanningError as exc:
            raise DynamicFoldCompilerError(
                "System stack resource_fingerprints are invalid."
            ) from exc
        object.__setattr__(
            self,
            "resource_fingerprints",
            resource_fingerprints,
        )
        if self.publication_recorder is not None and not callable(
            self.publication_recorder
        ):
            raise DynamicFoldCompilerError(
                "System stack publication_recorder must be callable or None."
            )


class DynamicFoldingCompiler:
    """Inspect, plan, execute, and audit a loaded-model folding request."""

    def __init__(
        self,
        *,
        inspectors: InspectorRegistry | None = None,
        methods: FoldMethodRegistry | None = None,
        operators: FoldOperatorRegistry,
        ledger: ExperimentLedger | None = None,
        snapshot_config: DynamicFoldSnapshotConfig | None = None,
    ) -> None:
        self.inspectors = inspectors or default_inspector_registry()
        self.methods = methods or default_method_registry()
        self.operators = operators
        self.ledger = ExperimentLedger() if ledger is None else ledger
        self.snapshot_config = (
            DynamicFoldSnapshotConfig.memory()
            if snapshot_config is None
            else snapshot_config
        )
        if not isinstance(
            self.snapshot_config,
            DynamicFoldSnapshotConfig,
        ):
            raise DynamicFoldCompilerError(
                "snapshot_config must be DynamicFoldSnapshotConfig."
            )

    def compile(
        self,
        *,
        parent_models: Mapping[str, Any],
        child_model: Any,
        child_name: str,
        evidence_provider: FoldEvidenceProvider,
        evaluator: FoldEvaluator,
        evidence_fingerprint: str,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
        child_tokenizer_fingerprint: str | None = None,
        exact_target_unique_parameters: int | None = None,
        initializer: FoldInitializer | None = None,
        final_evaluator: FoldEvaluator | None = None,
        stop_on_rejection: bool = True,
        resource_fingerprints: Mapping[str, str] | None = None,
    ) -> DynamicFoldCompilation:
        parents = dict(parent_models)
        validate_disjoint_parent_models(parents)
        if final_evaluator is None or not callable(
            getattr(final_evaluator, "evaluate", None)
        ):
            raise DynamicFoldCompilerError(
                "A final whole-output behavioral evaluator is mandatory."
            )
        tokenizer_fingerprints = dict(parent_tokenizer_fingerprints or {})
        unknown = set(tokenizer_fingerprints) - set(parents)
        if unknown:
            raise DynamicFoldCompilerError(
                "Tokenizer fingerprints reference unknown parents: "
                + ", ".join(sorted(unknown))
            )
        parent_graphs = tuple(
            self.inspectors.inspect(
                model,
                model_name=name,
                tokenizer_fingerprint=tokenizer_fingerprints.get(name),
            )
            for name, model in sorted(parents.items())
        )
        parent_state_fingerprints = {
            name: fingerprint_model_state(model)
            for name, model in sorted(parents.items())
        }
        child_graph = self.inspectors.inspect(
            child_model,
            model_name=child_name,
            tokenizer_fingerprint=child_tokenizer_fingerprint,
        )
        child_state = fingerprint_model_state(child_model)
        planner = DynamicFoldPlanner(
            methods=self.methods,
            ledger=self.ledger,
            resource_fingerprints=resource_fingerprints,
        )
        plan = planner.plan(
            parent_graphs,
            child_graph,
            child_state_fingerprint=child_state,
            evidence_fingerprint=evidence_fingerprint,
            parent_state_fingerprints=parent_state_fingerprints,
            exact_target_unique_parameters=exact_target_unique_parameters,
        )
        if not plan.ready:
            gap_details = [gap.as_dict() for gap in plan.gaps]
            blocked_actions = [
                {
                    "action_id": action.action_id,
                    "method_id": action.method_id,
                    "reason": action.ledger_decision.reason,
                }
                for action in plan.actions
                if not action.ledger_decision.allowed
            ]
            raise DynamicFoldCompilerError(
                "Automatic planning found unsupported target circuits or a "
                "previously closed exact attempt; "
                f"gaps={gap_details}; blocked_actions={blocked_actions}."
            )
        context = FoldExecutionContext(
            parent_models=parents,
            child_model=child_model,
            parent_graphs={value.model_name: value for value in parent_graphs},
            child_graph=child_graph,
        )
        run = DynamicFoldExecutor(
            operators=self.operators,
            evidence_provider=evidence_provider,
            evaluator=evaluator,
            ledger=self.ledger,
            initializer=initializer,
            final_evaluator=final_evaluator,
            stop_on_rejection=stop_on_rejection,
            snapshot_config=self.snapshot_config,
        ).execute(plan, context)
        return DynamicFoldCompilation(
            parent_graphs=parent_graphs,
            child_graph=child_graph,
            plan=plan,
            run=run,
        )

    def compile_and_publish(
        self,
        *,
        parent_models: Mapping[str, Any],
        child_model: Any,
        child_name: str,
        evidence_provider: FoldEvidenceProvider,
        evaluator: FoldEvaluator,
        final_evaluator: FoldEvaluator,
        evidence_fingerprint: str,
        destination: str | os.PathLike[str],
        model_saver: Callable[[Any, Path], None],
        model_loader: Callable[[Path], Any],
        fresh_reload_evaluator: Callable[[Any], Any],
        saver_id: str,
        loader_id: str,
        behavior_gate_id: str | None = None,
        behavior_gate_version: int | None = None,
        behavior_baseline_fingerprint: str | None = None,
        behavior_gate_manifest_fingerprint: str | None = None,
        behavior_gate_manifest: Mapping[str, Any] | None = None,
        dynamic_target: DynamicTargetSynthesisResult | None = None,
        expected_state_inventory: Mapping[str, int | str] | None = None,
        preparation_guard_applied: bool = False,
        preparation_guard_counts: Mapping[str, int] | None = None,
        preparation_parent_forward_counts: Mapping[str, int] | None = None,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
        child_tokenizer_fingerprint: str | None = None,
        exact_target_unique_parameters: int | None = None,
        initializer: FoldInitializer | None = None,
        stop_on_rejection: bool = True,
        resource_fingerprints: Mapping[str, str] | None = None,
    ) -> PublishedDynamicFoldCompilation:
        """Run the compiler and publish only a freshly authenticated child."""

        base_records = self.ledger.records
        staged_ledger = ExperimentLedger(base_records)
        staged_compiler = DynamicFoldingCompiler(
            inspectors=self.inspectors,
            methods=self.methods,
            operators=self.operators,
            ledger=staged_ledger,
            snapshot_config=self.snapshot_config,
        )
        compilation = staged_compiler.compile(
            parent_models=parent_models,
            child_model=child_model,
            child_name=child_name,
            evidence_provider=evidence_provider,
            evaluator=evaluator,
            final_evaluator=final_evaluator,
            evidence_fingerprint=evidence_fingerprint,
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
            child_tokenizer_fingerprint=child_tokenizer_fingerprint,
            exact_target_unique_parameters=exact_target_unique_parameters,
            initializer=initializer,
            stop_on_rejection=stop_on_rejection,
            resource_fingerprints=resource_fingerprints,
        )
        if not compilation.complete:
            if self.ledger.records == base_records:
                _merge_staged_ledger(
                    self.ledger,
                    staged_ledger,
                    base_records,
                )
            raise DynamicFoldCompilerError(
                "Dynamic folding did not produce a behavior-accepted complete "
                "compilation; no artifact was published."
            )
        if (
            compilation.run.observed_parent_module_forward_calls != 0
            or compilation.run.observed_parent_root_forward_calls != 0
        ):
            raise DynamicFoldCompilerError(
                "Dynamic fold execution called a live parent model; production "
                "folding must use cached evidence only."
            )
        if (
            compilation.run.observed_final_child_root_forward_calls <= 0
            or compilation.run.observed_final_child_inference_entrypoint_calls <= 0
        ):
            raise DynamicFoldCompilerError(
                "The in-memory final behavior gate accepted without executing "
                "the child model root required by its declared inference "
                "entrypoint."
            )
        try:
            artifact = publish_dynamic_fold_artifact(
                compilation=compilation,
                child_model=child_model,
                destination=destination,
                model_saver=model_saver,
                model_loader=model_loader,
                fresh_reload_evaluator=fresh_reload_evaluator,
                saver_id=saver_id,
                loader_id=loader_id,
                behavior_gate_id=behavior_gate_id,
                behavior_gate_version=behavior_gate_version,
                behavior_baseline_fingerprint=(behavior_baseline_fingerprint),
                behavior_gate_manifest_fingerprint=(behavior_gate_manifest_fingerprint),
                behavior_gate_manifest=behavior_gate_manifest,
                dynamic_target=dynamic_target,
                expected_state_inventory=expected_state_inventory,
                parent_models=parent_models,
                preparation_guard_applied=preparation_guard_applied,
                preparation_guard_counts=preparation_guard_counts,
                preparation_parent_forward_counts=(
                    preparation_parent_forward_counts
                ),
                inspectors=self.inspectors,
                tokenizer_fingerprint=child_tokenizer_fingerprint,
                snapshot_config=self.snapshot_config,
            )
        except DynamicFoldArtifactError as exc:
            if str(exc).startswith(
                "Freshly reloaded child failed its behavioral gate:"
            ):
                staged_extension = _staged_ledger_extension(
                    self.ledger, staged_ledger, base_records
                )
                accepted_records = tuple(
                    record
                    for record in staged_extension
                    if record.status == RecordStatus.ACCEPTED.value
                )
                closures = []
                previous = (
                    staged_extension[-1].record_sha256
                    if staged_extension
                    else (None if not base_records else base_records[-1].record_sha256)
                )
                for record in accepted_records:
                    closure = LedgerRecord(
                        signature=record.signature,
                        closure_level=ClosureLevel.EXACT,
                        status=RecordStatus.REJECTED,
                        reason_class="fresh_reload_behavior_gate_reject",
                        reason=str(exc),
                        metrics=(),
                        previous_record_sha256=previous,
                    )
                    closures.append(closure)
                    previous = closure.record_sha256
                self.ledger.append_batch((*staged_extension, *closures))
            raise
        try:
            _merge_staged_ledger(
                self.ledger,
                staged_ledger,
                base_records,
            )
        except BaseException as ledger_error:
            staged_hashes = {
                record.record_sha256
                for record in staged_ledger.records[len(base_records) :]
                if record.status
                in {
                    RecordStatus.ACCEPTED.value,
                    RecordStatus.REJECTED.value,
                }
            }
            partial_hashes = tuple(
                record.record_sha256
                for record in self.ledger.records
                if record.record_sha256 in staged_hashes
            )
            try:
                _rollback_published_artifact(
                    ledger=self.ledger,
                    provisional_record_sha256s=partial_hashes,
                    artifact=artifact,
                    reason_class="publication_ledger_commit_rollback",
                    reason=(
                        "The certified artifact was removed because its staged "
                        "ledger transaction could not be committed atomically."
                    ),
                )
            except BaseException as rollback_error:
                raise DynamicFoldCompilerError(
                    "Publication ledger commit failed and rollback did not "
                    "fully finalize."
                ) from rollback_error
            raise DynamicFoldCompilerError(
                "Publication ledger commit failed; the certified artifact was "
                "removed without leaving active provisional records."
            ) from ledger_error
        return PublishedDynamicFoldCompilation(
            compilation=compilation,
            artifact=artifact,
        )

    def synthesize_compile_and_publish(
        self,
        *,
        parent_models: Mapping[str, Any],
        target_request: DynamicTargetRequest,
        target_adapters: DynamicTargetAdapterRegistry,
        evidence_provider: FoldEvidenceProvider,
        evaluator: FoldEvaluator,
        final_evaluator: FoldEvaluator,
        evidence_fingerprint: str,
        destination: str | os.PathLike[str],
        model_saver: Callable[[Any, Path], None],
        model_loader: Callable[[Path], Any],
        fresh_reload_evaluator: Callable[[Any], Any],
        saver_id: str,
        loader_id: str,
        behavior_gate_id: str | None = None,
        behavior_gate_version: int | None = None,
        behavior_baseline_fingerprint: str | None = None,
        behavior_gate_manifest_fingerprint: str | None = None,
        behavior_gate_manifest: Mapping[str, Any] | None = None,
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
        initializer: FoldInitializer | None = None,
        stop_on_rejection: bool = True,
        resource_fingerprints: Mapping[str, str] | None = None,
    ) -> SynthesizedPublishedDynamicFoldCompilation:
        """Primary automatic path: request a target, never hand-build a child."""

        parents = dict(parent_models)
        validate_disjoint_parent_models(parents)
        with _observe_parent_model_execution(parents) as target_parent_forwards:
            target, target_guard = run_fold_only_callback(
                lambda: DynamicTargetSynthesizer(
                    inspectors=self.inspectors,
                    adapters=target_adapters,
                ).synthesize(
                    request=target_request,
                    parent_models=parents,
                    parent_tokenizer_fingerprints=(parent_tokenizer_fingerprints),
                ),
                phase="dynamic target synthesis",
                protected_models=tuple(parents.values()),
                freeze_parameters=False,
            )
        _require_zero_parent_forwards(
            target_parent_forwards,
            phase="Dynamic target synthesis",
        )
        published = self.compile_and_publish(
            parent_models=parents,
            child_model=target.model,
            child_name=target_request.child_name,
            evidence_provider=evidence_provider,
            evaluator=evaluator,
            final_evaluator=final_evaluator,
            evidence_fingerprint=evidence_fingerprint,
            destination=destination,
            model_saver=model_saver,
            model_loader=model_loader,
            fresh_reload_evaluator=fresh_reload_evaluator,
            saver_id=saver_id,
            loader_id=loader_id,
            behavior_gate_id=behavior_gate_id,
            behavior_gate_version=behavior_gate_version,
            behavior_baseline_fingerprint=(behavior_baseline_fingerprint),
            behavior_gate_manifest_fingerprint=(behavior_gate_manifest_fingerprint),
            behavior_gate_manifest=behavior_gate_manifest,
            dynamic_target=target,
            expected_state_inventory=target.resource_profile.inference_state_inventory(),
            preparation_guard_applied=True,
            preparation_guard_counts=_sum_guard_reports((target_guard,)),
            preparation_parent_forward_counts=_parent_forward_counts(
                target_parent_forwards
            ),
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
            child_tokenizer_fingerprint=target_request.tokenizer_fingerprint,
            exact_target_unique_parameters=(target_request.exact_unique_parameters),
            initializer=initializer,
            stop_on_rejection=stop_on_rejection,
            resource_fingerprints=resource_fingerprints,
        )
        return SynthesizedPublishedDynamicFoldCompilation(
            target=target,
            published=published,
        )

    def synthesize_fold_and_publish(
        self,
        *,
        parent_models: Mapping[str, Any],
        target_request: DynamicTargetRequest,
        target_adapters: DynamicTargetAdapterRegistry,
        stack_factory: Callable[
            [DynamicTargetSynthesisResult],
            DynamicFoldSystemStack,
        ],
        evidence_fingerprint: str,
        destination: str | os.PathLike[str],
        parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
        stop_on_rejection: bool = True,
        require_positive_compaction: bool = True,
    ) -> SynthesizedPublishedDynamicFoldCompilation:
        """Resolve target and fold stack dynamically, then publish atomically."""

        if not callable(stack_factory):
            raise DynamicFoldCompilerError("stack_factory must be callable.")
        if not isinstance(require_positive_compaction, bool):
            raise DynamicFoldCompilerError(
                "require_positive_compaction must be boolean."
            )
        parents = dict(parent_models)
        source_unique_parameters = validate_disjoint_parent_models(parents)
        with _observe_parent_model_execution(parents) as target_parent_forwards:
            target, target_guard = run_fold_only_callback(
                lambda: DynamicTargetSynthesizer(
                    inspectors=self.inspectors,
                    adapters=target_adapters,
                ).synthesize(
                    request=target_request,
                    parent_models=parents,
                    parent_tokenizer_fingerprints=(parent_tokenizer_fingerprints),
                ),
                phase="dynamic target synthesis",
                protected_models=tuple(parents.values()),
                freeze_parameters=False,
            )
        _require_zero_parent_forwards(
            target_parent_forwards,
            phase="Dynamic target synthesis",
        )
        if require_positive_compaction:
            if source_unique_parameters is None or source_unique_parameters <= 0:
                raise DynamicFoldCompilerError(
                    "Automatic production folding could not prove the distinct "
                    "learned-parameter count of its parent set."
                )
            if target.resource_profile.unique_parameters >= source_unique_parameters:
                raise DynamicFoldCompilerError(
                    "Automatic production folding requires positive learned-"
                    "parameter compaction relative to the declared parent set."
                )
            try:
                source_runtime_state_bytes = sum(
                    int(
                        serialized_inference_state_inventory(model)[
                            "runtime_state_bytes"
                        ]
                    )
                    for model in parents.values()
                )
            except Exception as exc:
                raise DynamicFoldCompilerError(
                    "Automatic production folding could not authenticate the "
                    "complete runtime tensor-state size of its parent set."
                ) from exc
            if (
                target.resource_profile.runtime_state_bytes
                >= source_runtime_state_bytes
            ):
                raise DynamicFoldCompilerError(
                    "Automatic production folding requires positive runtime "
                    "tensor-state compaction, including nonpersistent buffers, "
                    "relative to the declared parent set."
                )
        with _observe_parent_model_execution(parents) as stack_parent_forwards:
            stack, stack_guard = run_fold_only_callback(
                lambda: stack_factory(target),
                phase="dynamic fold-stack construction",
                protected_models=(*parents.values(), target.model),
                freeze_parameters=False,
            )
        _require_zero_parent_forwards(
            stack_parent_forwards,
            phase="Dynamic fold-stack construction",
        )
        if not isinstance(stack, DynamicFoldSystemStack):
            raise DynamicFoldCompilerError(
                "stack_factory must return DynamicFoldSystemStack."
            )
        if (
            stack.evidence_fingerprint is not None
            and stack.evidence_fingerprint != evidence_fingerprint
        ):
            raise DynamicFoldCompilerError(
                "System stack cached evidence fingerprint differs from the "
                "requested run binding."
            )
        runner = DynamicFoldingCompiler(
            inspectors=self.inspectors,
            methods=self.methods,
            operators=stack.operators,
            ledger=self.ledger,
            snapshot_config=(
                self.snapshot_config
                if stack.snapshot_config is None
                else stack.snapshot_config
            ),
        )
        published = runner.compile_and_publish(
            parent_models=parents,
            child_model=target.model,
            child_name=target_request.child_name,
            evidence_provider=stack.evidence_provider,
            evaluator=stack.evaluator,
            final_evaluator=stack.final_evaluator,
            evidence_fingerprint=evidence_fingerprint,
            destination=destination,
            model_saver=stack.model_saver,
            model_loader=stack.model_loader,
            fresh_reload_evaluator=stack.fresh_reload_evaluator,
            saver_id=stack.saver_id,
            loader_id=stack.loader_id,
            behavior_gate_id=stack.behavior_gate_id,
            behavior_gate_version=stack.behavior_gate_version,
            behavior_baseline_fingerprint=(stack.behavior_baseline_fingerprint),
            behavior_gate_manifest_fingerprint=(
                stack.behavior_gate_manifest_fingerprint
            ),
            behavior_gate_manifest=stack.behavior_gate_manifest,
            dynamic_target=target,
            expected_state_inventory=target.resource_profile.inference_state_inventory(),
            preparation_guard_applied=True,
            preparation_guard_counts=_sum_guard_reports((target_guard, stack_guard)),
            preparation_parent_forward_counts=_sum_parent_forward_counts(
                (target_parent_forwards, stack_parent_forwards)
            ),
            parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
            child_tokenizer_fingerprint=(target_request.tokenizer_fingerprint),
            exact_target_unique_parameters=(target_request.exact_unique_parameters),
            initializer=stack.initializer,
            stop_on_rejection=stop_on_rejection,
            resource_fingerprints=dict(stack.resource_fingerprints),
        )
        result = SynthesizedPublishedDynamicFoldCompilation(
            target=target,
            published=published,
        )
        if stack.publication_recorder is not None:
            try:
                child_state_before = fingerprint_model_state(target.model)
                child_runtime_before = fingerprint_model_runtime_state(target.model)
                _recorder_result, recorder_guard = run_fold_only_callback(
                    lambda: stack.publication_recorder(result),
                    phase="dynamic publication recorder",
                    protected_models=(*parents.values(), target.model),
                    freeze_parameters=False,
                )
                del _recorder_result
                if not recorder_guard.clean:
                    raise DynamicFoldCompilerError(
                        "System stack publication recorder attempted a training API."
                    )
                if (
                    fingerprint_model_state(target.model) != child_state_before
                    or fingerprint_model_runtime_state(target.model)
                    != child_runtime_before
                ):
                    raise DynamicFoldCompilerError(
                        "System stack publication recorder mutated the folded child."
                    )
            except BaseException as recorder_error:
                try:
                    _rollback_published_artifact(
                        ledger=self.ledger,
                        provisional_record_sha256s=tuple(
                            attempt.ledger_record_sha256
                            for attempt in published.compilation.run.attempts
                            if attempt.accepted
                        ),
                        artifact=published.artifact,
                        reason_class="publication_recorder_rollback",
                        reason=(
                            "The post-publication recorder failed after artifact "
                            "creation; the provisional execution was rolled back."
                        ),
                    )
                except BaseException as rollback_error:
                    raise DynamicFoldCompilerError(
                        "Publication recorder failed and its artifact lifecycle "
                        "rollback did not fully finalize."
                    ) from rollback_error
                raise recorder_error
        return result


def compile_loaded_models(
    *,
    parent_models: Mapping[str, Any],
    child_model: Any,
    child_name: str,
    operators: FoldOperatorRegistry,
    evidence_provider: FoldEvidenceProvider,
    evaluator: FoldEvaluator,
    evidence_fingerprint: str,
    parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    child_tokenizer_fingerprint: str | None = None,
    exact_target_unique_parameters: int | None = None,
    initializer: FoldInitializer | None = None,
    final_evaluator: FoldEvaluator | None = None,
    inspectors: InspectorRegistry | None = None,
    methods: FoldMethodRegistry | None = None,
    ledger: ExperimentLedger | None = None,
    snapshot_config: DynamicFoldSnapshotConfig | None = None,
    resource_fingerprints: Mapping[str, str] | None = None,
) -> DynamicFoldCompilation:
    return DynamicFoldingCompiler(
        inspectors=inspectors,
        methods=methods,
        operators=operators,
        ledger=ledger,
        snapshot_config=snapshot_config,
    ).compile(
        parent_models=parent_models,
        child_model=child_model,
        child_name=child_name,
        evidence_provider=evidence_provider,
        evaluator=evaluator,
        evidence_fingerprint=evidence_fingerprint,
        parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        child_tokenizer_fingerprint=child_tokenizer_fingerprint,
        exact_target_unique_parameters=exact_target_unique_parameters,
        initializer=initializer,
        final_evaluator=final_evaluator,
        resource_fingerprints=resource_fingerprints,
    )


def compile_and_publish_loaded_models(
    *,
    parent_models: Mapping[str, Any],
    child_model: Any,
    child_name: str,
    operators: FoldOperatorRegistry,
    evidence_provider: FoldEvidenceProvider,
    evaluator: FoldEvaluator,
    final_evaluator: FoldEvaluator,
    evidence_fingerprint: str,
    destination: str | os.PathLike[str],
    model_saver: Callable[[Any, Path], None],
    model_loader: Callable[[Path], Any],
    fresh_reload_evaluator: Callable[[Any], Any],
    saver_id: str,
    loader_id: str,
    parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    child_tokenizer_fingerprint: str | None = None,
    exact_target_unique_parameters: int | None = None,
    initializer: FoldInitializer | None = None,
    inspectors: InspectorRegistry | None = None,
    methods: FoldMethodRegistry | None = None,
    ledger: ExperimentLedger | None = None,
    snapshot_config: DynamicFoldSnapshotConfig | None = None,
    resource_fingerprints: Mapping[str, str] | None = None,
) -> PublishedDynamicFoldCompilation:
    """Primary end-to-end API: compile, gate, save, reload, and certify."""

    return DynamicFoldingCompiler(
        inspectors=inspectors,
        methods=methods,
        operators=operators,
        ledger=ledger,
        snapshot_config=snapshot_config,
    ).compile_and_publish(
        parent_models=parent_models,
        child_model=child_model,
        child_name=child_name,
        evidence_provider=evidence_provider,
        evaluator=evaluator,
        final_evaluator=final_evaluator,
        evidence_fingerprint=evidence_fingerprint,
        destination=destination,
        model_saver=model_saver,
        model_loader=model_loader,
        fresh_reload_evaluator=fresh_reload_evaluator,
        saver_id=saver_id,
        loader_id=loader_id,
        parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        child_tokenizer_fingerprint=child_tokenizer_fingerprint,
        exact_target_unique_parameters=exact_target_unique_parameters,
        initializer=initializer,
        resource_fingerprints=resource_fingerprints,
    )


def synthesize_compile_and_publish_models(
    *,
    parent_models: Mapping[str, Any],
    target_request: DynamicTargetRequest,
    target_adapters: DynamicTargetAdapterRegistry,
    operators: FoldOperatorRegistry,
    evidence_provider: FoldEvidenceProvider,
    evaluator: FoldEvaluator,
    final_evaluator: FoldEvaluator,
    evidence_fingerprint: str,
    destination: str | os.PathLike[str],
    model_saver: Callable[[Any, Path], None],
    model_loader: Callable[[Path], Any],
    fresh_reload_evaluator: Callable[[Any], Any],
    saver_id: str,
    loader_id: str,
    parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    initializer: FoldInitializer | None = None,
    inspectors: InspectorRegistry | None = None,
    methods: FoldMethodRegistry | None = None,
    ledger: ExperimentLedger | None = None,
    snapshot_config: DynamicFoldSnapshotConfig | None = None,
    resource_fingerprints: Mapping[str, str] | None = None,
) -> SynthesizedPublishedDynamicFoldCompilation:
    """Fully automatic target-synthesis and certified folding entrypoint."""

    return DynamicFoldingCompiler(
        inspectors=inspectors,
        methods=methods,
        operators=operators,
        ledger=ledger,
        snapshot_config=snapshot_config,
    ).synthesize_compile_and_publish(
        parent_models=parent_models,
        target_request=target_request,
        target_adapters=target_adapters,
        evidence_provider=evidence_provider,
        evaluator=evaluator,
        final_evaluator=final_evaluator,
        evidence_fingerprint=evidence_fingerprint,
        destination=destination,
        model_saver=model_saver,
        model_loader=model_loader,
        fresh_reload_evaluator=fresh_reload_evaluator,
        saver_id=saver_id,
        loader_id=loader_id,
        parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        initializer=initializer,
        resource_fingerprints=resource_fingerprints,
    )


def run_dynamic_folding_system(
    *,
    parent_models: Mapping[str, Any],
    target_request: DynamicTargetRequest,
    target_adapters: DynamicTargetAdapterRegistry,
    stack_factory: Callable[
        [DynamicTargetSynthesisResult],
        DynamicFoldSystemStack,
    ],
    evidence_fingerprint: str,
    destination: str | os.PathLike[str],
    parent_tokenizer_fingerprints: Mapping[str, str | None] | None = None,
    inspectors: InspectorRegistry | None = None,
    methods: FoldMethodRegistry | None = None,
    ledger: ExperimentLedger | None = None,
    snapshot_config: DynamicFoldSnapshotConfig | None = None,
    require_positive_compaction: bool = True,
) -> SynthesizedPublishedDynamicFoldCompilation:
    """Top-level dynamic system API with no caller-materialized child."""

    return DynamicFoldingCompiler(
        inspectors=inspectors,
        methods=methods,
        operators=FoldOperatorRegistry(),
        ledger=ledger,
        snapshot_config=snapshot_config,
    ).synthesize_fold_and_publish(
        parent_models=parent_models,
        target_request=target_request,
        target_adapters=target_adapters,
        stack_factory=stack_factory,
        evidence_fingerprint=evidence_fingerprint,
        destination=destination,
        parent_tokenizer_fingerprints=parent_tokenizer_fingerprints,
        require_positive_compaction=require_positive_compaction,
    )


__all__ = [
    "DynamicFoldCompilation",
    "DynamicFoldCompilerError",
    "DynamicFoldSystemStack",
    "DynamicFoldingCompiler",
    "PublishedDynamicFoldCompilation",
    "SynthesizedPublishedDynamicFoldCompilation",
    "compile_and_publish_loaded_models",
    "compile_loaded_models",
    "run_dynamic_folding_system",
    "synthesize_compile_and_publish_models",
]
