"""Dynamic-system adapter for authenticated Qwen3 compensated-fold replay."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

from .dynamic_behavior_gate import (
    CachedWholeOutputBehaviorGate,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import ModelCircuitGraph, canonical_sha256
from .dynamic_fold_compiler import DynamicFoldSystemStack
from .dynamic_fold_executor import (
    FoldDecision,
    FoldEvidence,
    FoldEvaluationContext,
    FoldExecutionContext,
    FoldOperatorRegistry,
    fingerprint_model_state,
)
from .dynamic_fold_planner import FoldAction
from .dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from .dynamic_model_inspector import InspectorRegistry
from .dynamic_target_synthesis import (
    AdapterMaterialization,
    DynamicTargetRequest,
    DynamicTargetSynthesisError,
    DynamicTargetSynthesisResult,
)
from .hf_qwen_compensated_runtime import (
    create_blank_compensated_qwen3,
    load_compensated_qwen3,
    qwen3_layers,
    save_compensated_qwen3,
    validate_compensated_qwen3_runtime,
)


METHOD_ID = "qwen3_compensated_transaction_replay"


class DynamicQwen3CompensatedError(RuntimeError):
    """The authenticated replay contract or dynamic adapter is invalid."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DynamicQwen3CompensatedError(f"{name} must be a positive integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DynamicQwen3CompensatedError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


@dataclass(frozen=True, slots=True)
class CompensatedReplayStep:
    """One accepted current-width neuron removal and multi-anchor redirect."""

    source_width: int
    target_width: int
    dropped: int
    anchors: tuple[int, ...]
    coefficients: tuple[float, ...]
    fold_sha256: str
    compute_indices_sha256: str

    def __post_init__(self) -> None:
        source = _positive_int(self.source_width, "source_width")
        target = _positive_int(self.target_width, "target_width")
        if target != source - 1:
            raise DynamicQwen3CompensatedError(
                "Each authenticated replay step must remove exactly one neuron."
            )
        if (
            isinstance(self.dropped, bool)
            or not isinstance(self.dropped, int)
            or self.dropped < 0
            or self.dropped >= source
        ):
            raise DynamicQwen3CompensatedError("Replay dropped index is out of range.")
        anchors = tuple(int(value) for value in self.anchors)
        coefficients = tuple(float(value) for value in self.coefficients)
        if (
            not anchors
            or len(anchors) != len(coefficients)
            or len(set(anchors)) != len(anchors)
            or self.dropped in anchors
            or any(value < 0 or value >= source for value in anchors)
            or any(not math.isfinite(value) for value in coefficients)
        ):
            raise DynamicQwen3CompensatedError(
                "Replay anchors/coefficients are invalid."
            )
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(
            self,
            "fold_sha256",
            _digest(self.fold_sha256, "fold_sha256"),
        )
        object.__setattr__(
            self,
            "compute_indices_sha256",
            _digest(
                self.compute_indices_sha256,
                "compute_indices_sha256",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "target_width": self.target_width,
            "dropped": self.dropped,
            "anchors": list(self.anchors),
            "coefficients": list(self.coefficients),
            "fold_sha256": self.fold_sha256,
            "compute_indices_sha256": self.compute_indices_sha256,
        }


@dataclass(frozen=True, slots=True)
class Qwen3CompensatedReplayEvidence:
    """Immutable evidence needed to synthesize and execute one replay."""

    layer_index: int
    steps: tuple[CompensatedReplayStep, ...]
    transaction_records: tuple[Mapping[str, Any], ...]
    transaction_report_sha256: str
    cached_parent_sha256: str
    cached_candidate_sha256: str
    manual_packet_sha256: str
    expected_parent_state_sha256: str
    expected_parent_state_tensor_count: int
    expected_parent_parameters: int
    expected_target_parameters: int
    evidence_fingerprint: str
    max_transient_compute_bytes: int = 2 * 1024**3

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise DynamicQwen3CompensatedError(
                "layer_index must be a non-negative integer."
            )
        steps = tuple(self.steps)
        if not steps or any(
            not isinstance(value, CompensatedReplayStep) for value in steps
        ):
            raise DynamicQwen3CompensatedError(
                "Replay evidence requires accepted merge steps."
            )
        for previous, current in zip(steps, steps[1:]):
            if previous.target_width != current.source_width:
                raise DynamicQwen3CompensatedError(
                    "Replay step widths do not form one sequential chain."
                )
        object.__setattr__(self, "steps", steps)
        records = tuple(
            MappingProxyType(dict(value)) for value in self.transaction_records
        )
        if len(records) != len(steps):
            raise DynamicQwen3CompensatedError(
                "Transaction records and replay steps differ in length."
            )
        object.__setattr__(self, "transaction_records", records)
        for name in (
            "transaction_report_sha256",
            "cached_parent_sha256",
            "cached_candidate_sha256",
            "manual_packet_sha256",
            "expected_parent_state_sha256",
            "evidence_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        for name in (
            "expected_parent_state_tensor_count",
            "expected_parent_parameters",
            "expected_target_parameters",
            "max_transient_compute_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        if self.expected_target_parameters >= self.expected_parent_parameters:
            raise DynamicQwen3CompensatedError(
                "Replay target must have fewer parameters than its parent."
            )

    @property
    def source_width(self) -> int:
        return self.steps[0].source_width

    @property
    def target_width(self) -> int:
        return self.steps[-1].target_width

    @property
    def compute_indices(self) -> tuple[int, ...]:
        indices = tuple(range(self.source_width))
        for step in self.steps:
            indices = tuple(
                value for index, value in enumerate(indices) if index != step.dropped
            )
        return indices

    @property
    def removed_neurons(self) -> int:
        return self.source_width - self.target_width

    @classmethod
    def from_transaction_report(
        cls,
        report: Mapping[str, Any],
        *,
        transaction_report_sha256: str,
        cached_parent_sha256: str,
        cached_candidate_sha256: str,
        manual_packet_sha256: str,
        expected_parent_state_sha256: str,
        expected_parent_state_tensor_count: int,
        evidence_fingerprint: str,
        max_transient_compute_bytes: int = 2 * 1024**3,
    ) -> "Qwen3CompensatedReplayEvidence":
        accepted = report.get("accepted_transactions")
        initial = report.get("initial")
        final = report.get("final")
        if (
            not isinstance(accepted, list)
            or not accepted
            or not isinstance(initial, Mapping)
            or not isinstance(final, Mapping)
            or report.get("accepted_count") != len(accepted)
            or report.get("stop_reason") != "target-reached"
            or report.get("zero_optimizer_steps") != 0
            or report.get("backward_called") is not False
            or report.get("recovery_examples") != 0
        ):
            raise DynamicQwen3CompensatedError(
                "Transaction report is not an accepted zero-training replay source."
            )
        records = []
        steps = []
        layer_index: int | None = None
        for raw in accepted:
            if not isinstance(raw, Mapping):
                raise DynamicQwen3CompensatedError(
                    "Accepted transaction row is malformed."
                )
            proposal = raw.get("proposal")
            compute_shape = raw.get("compute_shape")
            merges = proposal.get("merges") if isinstance(proposal, Mapping) else None
            if (
                not isinstance(proposal, Mapping)
                or not isinstance(compute_shape, Mapping)
                or not isinstance(merges, list)
                or len(merges) != 1
                or not isinstance(merges[0], Mapping)
                or raw.get("decision") != "accepted-committed"
                or raw.get("zero_optimizer_steps") != 0
                or raw.get("backward_called") is not False
            ):
                raise DynamicQwen3CompensatedError(
                    "Accepted transaction does not satisfy the replay contract."
                )
            merge = merges[0]
            current_layer = int(proposal.get("layer_index", -1))
            if layer_index is None:
                layer_index = current_layer
            if current_layer != layer_index:
                raise DynamicQwen3CompensatedError(
                    "Accepted transactions span multiple layers."
                )
            step = CompensatedReplayStep(
                source_width=int(proposal["source_width"]),
                target_width=int(proposal["target_width"]),
                dropped=int(merge["dropped"]),
                anchors=tuple(int(value) for value in merge["anchors"]),
                coefficients=tuple(float(value) for value in merge["coefficients"]),
                fold_sha256=str(proposal["fold_sha256"]),
                compute_indices_sha256=str(compute_shape["compute_indices_sha256"]),
            )
            before = raw.get("before")
            after = raw.get("after")
            if (
                not isinstance(before, Mapping)
                or not isinstance(after, Mapping)
                or int(after.get("unique_parameter_count", -1))
                >= int(before.get("unique_parameter_count", -1))
                or int(proposal.get("parameter_savings", -1))
                != int(before["unique_parameter_count"])
                - int(after["unique_parameter_count"])
            ):
                raise DynamicQwen3CompensatedError(
                    "Accepted transaction parameter accounting is invalid."
                )
            steps.append(step)
            records.append(dict(raw))
        if layer_index is None:
            raise AssertionError("Accepted transaction layer vanished.")
        parent_parameters = int(initial.get("unique_parameter_count", -1))
        target_parameters = int(final.get("unique_parameter_count", -1))
        if (
            parent_parameters <= target_parameters
            or report.get("target", {}).get("reached") is not True
            or int(report["target"]["unique_parameter_count"]) != target_parameters
        ):
            raise DynamicQwen3CompensatedError(
                "Transaction report final target accounting is invalid."
            )
        return cls(
            layer_index=layer_index,
            steps=tuple(steps),
            transaction_records=tuple(records),
            transaction_report_sha256=transaction_report_sha256,
            cached_parent_sha256=cached_parent_sha256,
            cached_candidate_sha256=cached_candidate_sha256,
            manual_packet_sha256=manual_packet_sha256,
            expected_parent_state_sha256=expected_parent_state_sha256,
            expected_parent_state_tensor_count=expected_parent_state_tensor_count,
            expected_parent_parameters=parent_parameters,
            expected_target_parameters=target_parameters,
            evidence_fingerprint=evidence_fingerprint,
            max_transient_compute_bytes=max_transient_compute_bytes,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "steps": [value.as_dict() for value in self.steps],
            "transaction_report_sha256": self.transaction_report_sha256,
            "cached_parent_sha256": self.cached_parent_sha256,
            "cached_candidate_sha256": self.cached_candidate_sha256,
            "manual_packet_sha256": self.manual_packet_sha256,
            "expected_parent_state_sha256": self.expected_parent_state_sha256,
            "expected_parent_state_tensor_count": (
                self.expected_parent_state_tensor_count
            ),
            "expected_parent_parameters": self.expected_parent_parameters,
            "expected_target_parameters": self.expected_target_parameters,
            "evidence_fingerprint": self.evidence_fingerprint,
            "compute_indices_count": len(self.compute_indices),
            "compute_indices_sha256": canonical_sha256(list(self.compute_indices)),
            "max_transient_compute_bytes": self.max_transient_compute_bytes,
        }


class Qwen3CompensatedTargetSynthesisAdapter:
    """Create the exact ragged blank selected by authenticated evidence."""

    adapter_id = "qwen3-compensated-ragged-target"
    version = 1
    priority = 260
    target_families = ("qwen3",)
    target_modalities = ("text",)

    def __init__(self, evidence: Qwen3CompensatedReplayEvidence) -> None:
        if not isinstance(evidence, Qwen3CompensatedReplayEvidence):
            raise DynamicQwen3CompensatedError(
                "Target adapter requires Qwen3CompensatedReplayEvidence."
            )
        self.evidence = evidence

    def supports(
        self,
        request: DynamicTargetRequest,
        parent_graphs: Mapping[str, ModelCircuitGraph],
    ) -> bool:
        return (
            request.target_family in {"qwen3", "auto", "any"}
            and request.target_modalities == ("text",)
            and request.exact_unique_parameters
            == self.evidence.expected_target_parameters
            and len(parent_graphs) == 1
            and next(iter(parent_graphs.values())).family == "qwen3"
        )

    def materialize(
        self,
        request: DynamicTargetRequest,
        parent_models: Mapping[str, Any],
        parent_graphs: Mapping[str, ModelCircuitGraph],
        inspectors: InspectorRegistry,
    ) -> AdapterMaterialization:
        if not self.supports(request, parent_graphs):
            raise DynamicTargetSynthesisError(
                "Qwen3 compensated evidence does not support this exact target."
            )
        parent = next(iter(parent_models.values()))
        actual_parent = sum(int(value.numel()) for value in parent.parameters())
        if actual_parent != self.evidence.expected_parent_parameters:
            raise DynamicTargetSynthesisError(
                "Loaded Qwen3 parent misses the evidence-bound parameter count."
            )
        layers = qwen3_layers(parent)
        if (
            self.evidence.layer_index >= len(layers)
            or int(layers[self.evidence.layer_index].mlp.gate_proj.weight.shape[0])
            != self.evidence.source_width
        ):
            raise DynamicTargetSynthesisError(
                "Loaded Qwen3 parent misses the evidence-bound source width."
            )
        child = create_blank_compensated_qwen3(
            parent,
            layer_index=self.evidence.layer_index,
            target_width=self.evidence.target_width,
            compute_indices=self.evidence.compute_indices,
            max_transient_compute_bytes=(self.evidence.max_transient_compute_bytes),
        )
        graph = inspectors.inspect(
            child,
            model_name=request.child_name,
            tokenizer_fingerprint=request.tokenizer_fingerprint,
        )
        if graph.unique_parameter_count != self.evidence.expected_target_parameters:
            raise DynamicTargetSynthesisError(
                "Synthesized compensated blank misses the exact target budget."
            )
        topology = canonical_sha256(
            {
                "graph": graph.fingerprint,
                "layer_index": self.evidence.layer_index,
                "source_width": self.evidence.source_width,
                "target_width": self.evidence.target_width,
                "compute_indices": list(self.evidence.compute_indices),
            }
        )
        return AdapterMaterialization(
            model=child,
            graph=graph,
            topology_fingerprint=topology,
            report={
                "layer_index": self.evidence.layer_index,
                "source_width": self.evidence.source_width,
                "target_width": self.evidence.target_width,
                "removed_neurons": self.evidence.removed_neurons,
                "exact_target_parameters": (self.evidence.expected_target_parameters),
                "zero_initialized_blank": True,
                "zero_optimizer_steps": 0,
                "backward_called": False,
            },
            adapter_payload=self.evidence,
        )


class Qwen3CompensatedEvidenceProvider:
    def __init__(self, evidence: Qwen3CompensatedReplayEvidence) -> None:
        self.evidence = evidence
        self.refresh_count = 0
        self.restore_count = 0

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
    ) -> FoldEvidence:
        del context
        if action.method_id != METHOD_ID:
            raise DynamicQwen3CompensatedError(
                "Compensated evidence was requested for another method."
            )
        return FoldEvidence(
            self.evidence.evidence_fingerprint,
            self.evidence,
            channels=action.evidence_requirements,
            parent_forward_count=0,
            child_forward_count=0,
        )

    def refresh(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        transaction_report: Mapping[str, Any],
    ) -> None:
        del action, context, transaction_report
        self.refresh_count += 1

    def restore(
        self,
        context: FoldExecutionContext,
        state_fingerprint: str,
        action_ids: Sequence[str],
    ) -> None:
        del context, state_fingerprint, action_ids
        self.restore_count += 1


class VerifiedZeroBlankInitializer:
    def initialize(
        self,
        plan: Any,
        context: FoldExecutionContext,
    ) -> Mapping[str, Any]:
        del plan
        import torch

        if any(
            bool(torch.count_nonzero(value.detach()).item())
            for value in context.child_model.parameters()
        ):
            raise DynamicQwen3CompensatedError(
                "Compensated target is not the authenticated zero blank."
            )
        return {
            "kind": "verified_zero_compensated_blank",
            "complete_target_coverage": False,
            "child_state_fingerprint": context.child_state_fingerprint,
        }


class Qwen3CompensatedReplayTransaction:
    method_id = METHOD_ID

    def __init__(
        self,
        child_model: Any,
        *,
        candidate_state_fingerprint: str,
        evidence: Qwen3CompensatedReplayEvidence,
        changed_parameter_count: int,
    ) -> None:
        self.child_model = child_model
        self.candidate_state_fingerprint = candidate_state_fingerprint
        self.evidence = evidence
        self.changed_parameter_count = changed_parameter_count
        self.active = True
        self.verified_noop = False

    def commit(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicQwen3CompensatedError(
                "Compensated transaction is no longer active."
            )
        self.active = False
        return {
            "kind": "authenticated_compensated_transaction_replay",
            "transaction_report_sha256": (self.evidence.transaction_report_sha256),
            "accepted_step_count": len(self.evidence.steps),
            "changed_parameter_count": self.changed_parameter_count,
            "target_parameter_count": self.evidence.expected_target_parameters,
            "compute_indices_sha256": canonical_sha256(
                list(self.evidence.compute_indices)
            ),
            "zero_optimizer_steps": 0,
            "backward_called": False,
        }

    def rollback(self) -> Mapping[str, Any]:
        if not self.active:
            raise DynamicQwen3CompensatedError(
                "Compensated transaction is no longer active."
            )
        import torch

        with torch.no_grad():
            seen: set[int] = set()
            for parameter in self.child_model.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameter.zero_()
        self.active = False
        return {
            "kind": "zero_blank_restore",
            "rollback_exact": True,
            "zero_optimizer_steps": 0,
            "backward_called": False,
        }


class Qwen3CompensatedReplayOperator:
    method_id = METHOD_ID

    @staticmethod
    def _replay_folded_mlp(
        parent_mlp: Any,
        evidence: Qwen3CompensatedReplayEvidence,
    ) -> tuple[Any, Any, Any]:
        import torch

        gate = parent_mlp.gate_proj.weight.detach()
        up = parent_mlp.up_proj.weight.detach()
        down = parent_mlp.down_proj.weight.detach().clone()
        current_indices = tuple(range(evidence.source_width))
        for step in evidence.steps:
            if (
                tuple(gate.shape)[0] != step.source_width
                or tuple(up.shape)[0] != step.source_width
                or tuple(down.shape)[1] != step.source_width
            ):
                raise DynamicQwen3CompensatedError(
                    "Live Qwen3 MLP width differs during replay."
                )
            source_column = down[:, step.dropped].clone()
            for anchor, coefficient in zip(
                step.anchors,
                step.coefficients,
                strict=True,
            ):
                down[:, anchor] = down[:, anchor] + coefficient * source_column
            kept = tuple(
                index for index in range(step.source_width) if index != step.dropped
            )
            selector = torch.tensor(
                kept,
                dtype=torch.long,
                device=gate.device,
            )
            gate = gate.index_select(0, selector).contiguous()
            up = up.index_select(0, selector).contiguous()
            down = down.index_select(
                1,
                selector.to(device=down.device),
            ).contiguous()
            current_indices = tuple(current_indices[index] for index in kept)
            if current_indices != evidence.compute_indices[: len(current_indices)]:
                # Intermediate indices need not be a prefix of the final set.
                # The final exact comparison below is authoritative.
                pass
        if current_indices != evidence.compute_indices:
            raise DynamicQwen3CompensatedError(
                "Replayed source-coordinate lineage differs from evidence."
            )
        return gate, up, down

    def prepare(
        self,
        action: FoldAction,
        context: FoldExecutionContext,
        evidence: FoldEvidence,
    ) -> Qwen3CompensatedReplayTransaction:
        import torch

        if action.method_id != self.method_id or not isinstance(
            evidence.payload,
            Qwen3CompensatedReplayEvidence,
        ):
            raise DynamicQwen3CompensatedError(
                "Compensated operator received incompatible action/evidence."
            )
        replay = evidence.payload
        if len(context.parent_models) != 1:
            raise DynamicQwen3CompensatedError(
                "Compensated replay requires exactly one Qwen3 parent."
            )
        parent = next(iter(context.parent_models.values()))
        child = context.child_model
        validate_compensated_qwen3_runtime(child)
        parent_parameters = dict(
            parent.named_parameters(recurse=True, remove_duplicate=False)
        )
        child_parameters = dict(
            child.named_parameters(recurse=True, remove_duplicate=False)
        )
        if set(parent_parameters) != set(child_parameters):
            raise DynamicQwen3CompensatedError(
                "Parent/child Qwen3 parameter paths differ."
            )
        prefix = f"model.layers.{replay.layer_index}.mlp."
        gate, up, down = self._replay_folded_mlp(
            qwen3_layers(parent)[replay.layer_index].mlp,
            replay,
        )
        folded = {
            f"{prefix}gate_proj.weight": gate,
            f"{prefix}up_proj.weight": up,
            f"{prefix}down_proj.weight": down,
        }
        with torch.no_grad():
            for name, target in child_parameters.items():
                source = folded.get(name, parent_parameters[name])
                if tuple(source.shape) != tuple(target.shape):
                    raise DynamicQwen3CompensatedError(
                        f"Replay tensor shape differs for {name}."
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        actual = sum(int(value.numel()) for value in child.parameters())
        if actual != replay.expected_target_parameters:
            raise DynamicQwen3CompensatedError(
                "Replayed child misses the exact parameter target."
            )
        candidate = fingerprint_model_state(child)
        return Qwen3CompensatedReplayTransaction(
            child,
            candidate_state_fingerprint=candidate,
            evidence=replay,
            changed_parameter_count=len(child_parameters),
        )


class CompensatedReplayIntegrityEvaluator:
    """Accept an installed replay only after its structural postconditions."""

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        replay = context.evidence.payload
        if not isinstance(replay, Qwen3CompensatedReplayEvidence):
            return FoldDecision.reject("Replay evidence type changed.")
        try:
            report = validate_compensated_qwen3_runtime(
                context.execution_context.child_model
            )
        except Exception as exc:
            return FoldDecision.reject(
                f"Compensated runtime validation failed: {type(exc).__name__}."
            )
        actual = int(report["parameter_count"])
        if actual != replay.expected_target_parameters:
            return FoldDecision.reject(
                "Compensated runtime misses the exact target.",
                {"actual_parameters": actual},
            )
        return FoldDecision.accept(
            "Authenticated compensated replay installed structurally.",
            {
                "actual_parameters": actual,
                "accepted_step_count": len(replay.steps),
                "parent_forward_passes": 0,
                "optimizer_steps": 0,
            },
        )


class Qwen3CompensatedFoldStackAdapter:
    """Build the complete generic-system stack for compensated Qwen3."""

    adapter_id = "qwen3-compensated-replay-stack"
    version = 1
    priority = 260
    target_families = ("qwen3",)
    target_modalities = ("text",)
    method_ids = (METHOD_ID,)

    def __init__(
        self,
        *,
        evidence: Qwen3CompensatedReplayEvidence,
        behavior_gate: CachedWholeOutputBehaviorGate,
        tokenizer_source: str | Path | None,
        reload_device: str,
        snapshot_config: DynamicFoldSnapshotConfig,
        evidence_binding: Mapping[str, Any],
        before_reload: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(evidence, Qwen3CompensatedReplayEvidence):
            raise DynamicQwen3CompensatedError(
                "Stack adapter requires compensated replay evidence."
            )
        if not isinstance(behavior_gate, CachedWholeOutputBehaviorGate):
            raise DynamicQwen3CompensatedError(
                "Stack adapter requires CachedWholeOutputBehaviorGate."
            )
        if not isinstance(snapshot_config, DynamicFoldSnapshotConfig):
            raise DynamicQwen3CompensatedError(
                "Stack adapter requires DynamicFoldSnapshotConfig."
            )
        self.evidence = evidence
        self.behavior_gate = behavior_gate
        self.tokenizer_source = tokenizer_source
        self.reload_device = str(reload_device)
        self.snapshot_config = snapshot_config
        self.evidence_binding = dict(evidence_binding)
        if before_reload is not None and not callable(before_reload):
            raise DynamicQwen3CompensatedError(
                "before_reload must be callable or None."
            )
        self.before_reload = before_reload
        self._before_reload_called = False

    def supports(self, target: DynamicTargetSynthesisResult) -> bool:
        return (
            target.resource_profile.family == "qwen3"
            and target.resource_profile.modalities == ("text",)
            and target.request.exact_unique_parameters
            == self.evidence.expected_target_parameters
            and target.adapter_payload is self.evidence
        )

    def build(
        self,
        target: DynamicTargetSynthesisResult,
    ) -> DynamicFoldSystemStack:
        if not self.supports(target):
            raise DynamicQwen3CompensatedError(
                "Compensated stack does not support the synthesized target."
            )

        def saver(model: Any, path: Path) -> None:
            save_compensated_qwen3(
                model,
                path,
                tokenizer_source=self.tokenizer_source,
                transaction_records=self.evidence.transaction_records,
                evidence_binding=self.evidence_binding,
            )

        def loader(path: Path) -> Any:
            source = Path(path).expanduser()
            required = [
                "config.json",
                "fold_manifest.json",
                "model.safetensors",
            ]
            if self.tokenizer_source is not None:
                required.extend(("tokenizer.json", "tokenizer_config.json"))
            if (
                not source.is_dir()
                or source.is_symlink()
                or any(
                    not (source / filename).is_file()
                    or (source / filename).is_symlink()
                    for filename in required
                )
            ):
                raise DynamicQwen3CompensatedError(
                    "Compensated loader requires a complete regular local "
                    "artifact before the one-shot parent release."
                )
            if self.before_reload is not None and not self._before_reload_called:
                self.before_reload()
                self._before_reload_called = True
            return load_compensated_qwen3(
                path,
                device=self.reload_device,
            )

        return DynamicFoldSystemStack(
            operators=FoldOperatorRegistry((Qwen3CompensatedReplayOperator(),)),
            evidence_provider=Qwen3CompensatedEvidenceProvider(self.evidence),
            evaluator=CompensatedReplayIntegrityEvaluator(),
            final_evaluator=self.behavior_gate,
            model_saver=saver,
            model_loader=loader,
            fresh_reload_evaluator=self.behavior_gate,
            saver_id="qwen3-compensated-safe-safetensors-v1",
            loader_id="qwen3-compensated-concrete-local-v1",
            behavior_gate_id=resolve_behavior_gate_binding(self.behavior_gate).gate_id,
            behavior_gate_version=resolve_behavior_gate_binding(
                self.behavior_gate
            ).gate_version,
            behavior_baseline_fingerprint=resolve_behavior_gate_binding(
                self.behavior_gate
            ).baseline_fingerprint,
            initializer=VerifiedZeroBlankInitializer(),
            snapshot_config=self.snapshot_config,
            evidence_fingerprint=self.evidence.evidence_fingerprint,
        )


__all__ = [
    "METHOD_ID",
    "CompensatedReplayIntegrityEvaluator",
    "CompensatedReplayStep",
    "DynamicQwen3CompensatedError",
    "Qwen3CompensatedEvidenceProvider",
    "Qwen3CompensatedFoldStackAdapter",
    "Qwen3CompensatedReplayEvidence",
    "Qwen3CompensatedReplayOperator",
    "Qwen3CompensatedReplayTransaction",
    "Qwen3CompensatedTargetSynthesisAdapter",
    "VerifiedZeroBlankInitializer",
]
