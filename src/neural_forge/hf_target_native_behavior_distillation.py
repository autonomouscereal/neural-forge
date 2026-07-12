"""Output-only distillation for an exact target-native compiler transaction.

The trainer in this module is deliberately blind to model internals.  Its only
teacher signal is the parent's causal output distribution and greedy action on
two closed-loop lanes: prefixes produced by the parent and prefixes produced
on-policy by the child.  Candidate selection likewise uses only action and
free-running token behavior.  Weights, hidden states, activations, attention
maps, and cache values are neither requested nor accepted as targets.

Temporary QKVO and Gate/Up/Down factors are selected on a disjoint calibration
domain with a step-zero rollback candidate.  The sealed domain is inaccessible
until a preregistered output-behavior gate passes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Optimizer

from neural_forge.hf_evaluation_provenance import token_sequence_sha256
from neural_forge.hf_functional_decode_trace import (
    FunctionalDecodeStep,
    FunctionalDecodeTrace,
    trace_mapping,
)
from neural_forge.hf_progressive_target_bridge import (
    ExactTargetCompilerTransaction,
)
from neural_forge.hf_residual_attention_functional import (
    ResidualAttentionFunctionalReport,
    score_residual_attention_function,
)


TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA = (
    "neural-forge-target-native-behavior-distillation-v1"
)
BEHAVIOR_DOMAIN_SCHEMA = "neural-forge-behavior-domain-v1"
BEHAVIOR_FACTOR_PACK_SCHEMA = "neural-forge-target-native-factor-pack-v1"
BEHAVIOR_CHECKPOINT_SCHEMA = "neural-forge-behavior-checkpoint-v1"

BehaviorEvaluationSplit = Literal[
    "calibration-selection",
    "calibration-exploratory",
    "heldout-evaluation",
]
BehaviorSelectionPolicy = Literal[
    "strict-nonregression",
    "rollout-first-acquisition",
]


class TargetNativeBehaviorDistillationError(RuntimeError):
    """An output-only training, provenance, or selection contract failed."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TargetNativeBehaviorDistillationError(
            "Behavior evidence is not canonical JSON."
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise TargetNativeBehaviorDistillationError(f"{name} must be a SHA-256.")
    return result


def _index(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise TargetNativeBehaviorDistillationError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be an integer."
        ) from error
    if (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise TargetNativeBehaviorDistillationError(f"{name} must be {qualifier}.")
    return result


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be finite."
        ) from error
    if not math.isfinite(result):
        raise TargetNativeBehaviorDistillationError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be at least {minimum}."
        )
    if maximum is not None and result > maximum:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be at most {maximum}."
        )
    return result


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TargetNativeBehaviorDistillationError(
                f"{name} must contain integer token IDs."
            )
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be a non-empty token sequence."
        )
    result: list[int] = []
    for index, item in enumerate(value):
        token_id = _index(item, f"{name}[{index}]")
        result.append(token_id)
    if not result:
        raise TargetNativeBehaviorDistillationError(
            f"{name} must be a non-empty token sequence."
        )
    return tuple(result)


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BehaviorPrompt:
    """One pre-tokenized prompt with a content-derived immutable identity."""

    token_ids: tuple[int, ...]
    prompt_id: str = field(init=False)
    token_ids_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        tokens = _token_ids(self.token_ids, "token_ids")
        digest = token_sequence_sha256(tokens)
        object.__setattr__(self, "token_ids", tokens)
        object.__setattr__(self, "token_ids_sha256", digest)
        object.__setattr__(self, "prompt_id", f"prompt-{digest}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": len(self.token_ids),
        }


@dataclass(frozen=True, slots=True)
class BehaviorDomainSuite:
    """A bounded, hash-bound prompt domain used in exactly one split role."""

    split_name: str
    prompts: tuple[BehaviorPrompt, ...]
    artifact_sha256: str
    domain_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.split_name).strip().lower()
        if not name:
            raise TargetNativeBehaviorDistillationError("split_name cannot be blank.")
        prompts = tuple(self.prompts)
        if not prompts or any(
            not isinstance(value, BehaviorPrompt) for value in prompts
        ):
            raise TargetNativeBehaviorDistillationError(
                "prompts must be a non-empty BehaviorPrompt sequence."
            )
        ids = [value.prompt_id for value in prompts]
        if len(set(ids)) != len(ids):
            raise TargetNativeBehaviorDistillationError(
                "A behavior domain cannot contain duplicate prompts."
            )
        artifact = _sha256(self.artifact_sha256, "artifact_sha256")
        payload = {
            "schema": BEHAVIOR_DOMAIN_SCHEMA,
            "split_name": name,
            "artifact_sha256": artifact,
            "prompts": [value.as_dict() for value in prompts],
        }
        object.__setattr__(self, "split_name", name)
        object.__setattr__(self, "prompts", prompts)
        object.__setattr__(self, "artifact_sha256", artifact)
        object.__setattr__(self, "domain_sha256", _canonical_sha256(payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BEHAVIOR_DOMAIN_SCHEMA,
            "split_name": self.split_name,
            "artifact_sha256": self.artifact_sha256,
            "prompt_count": len(self.prompts),
            "prompts": [value.as_dict() for value in self.prompts],
            "domain_sha256": self.domain_sha256,
            "raw_prompt_text_retained": False,
        }


def load_behavior_domain_suite(
    split_name: str,
    payload: Mapping[str, Any],
    *,
    artifact_sha256: str,
) -> BehaviorDomainSuite:
    """Load one pre-tokenized domain and verify any supplied prompt bindings."""

    if not isinstance(payload, Mapping):
        raise TargetNativeBehaviorDistillationError("payload must be a mapping.")
    raw_prompts = payload.get("prompts")
    if isinstance(raw_prompts, (str, bytes)) or not isinstance(raw_prompts, Sequence):
        raise TargetNativeBehaviorDistillationError(
            "payload.prompts must be a non-empty sequence."
        )
    prompts: list[BehaviorPrompt] = []
    for index, raw in enumerate(raw_prompts):
        supplied_id: Any = None
        supplied_hash: Any = None
        if isinstance(raw, Mapping):
            tokens = raw.get("token_ids", raw.get("input_ids"))
            supplied_id = raw.get("prompt_id")
            supplied_hash = raw.get("token_ids_sha256")
        else:
            tokens = raw
        prompt = BehaviorPrompt(_token_ids(tokens, f"prompts[{index}].token_ids"))
        if supplied_id is not None and str(supplied_id) != prompt.prompt_id:
            raise TargetNativeBehaviorDistillationError(
                f"prompts[{index}].prompt_id does not match its token content."
            )
        if supplied_hash is not None and str(supplied_hash) != prompt.token_ids_sha256:
            raise TargetNativeBehaviorDistillationError(
                f"prompts[{index}].token_ids_sha256 does not match its content."
            )
        prompts.append(prompt)
    declared_artifact = payload.get("artifact_sha256")
    if declared_artifact is not None and _sha256(
        declared_artifact, "payload.artifact_sha256"
    ) != _sha256(artifact_sha256, "artifact_sha256"):
        raise TargetNativeBehaviorDistillationError(
            "payload artifact binding disagrees with artifact_sha256."
        )
    return BehaviorDomainSuite(
        split_name=str(split_name),
        prompts=tuple(prompts),
        artifact_sha256=artifact_sha256,
    )


@dataclass(frozen=True, slots=True)
class BehaviorDistillationSchedule:
    """Preregistered finite work and checkpoint bounds."""

    steps: int
    learning_rate: float
    checkpoint_steps: tuple[int, ...] = ()
    fit_horizon: int = 4
    evaluation_horizon: int = 8
    fit_prompts_per_step: int = 1
    temperature: float = 1.0
    action_loss_weight: float = 1.0
    distribution_loss_weight: float = 1.0
    top_k: int = 5
    gradient_clip_norm: float = 1.0
    max_prompt_tokens: int = 512
    max_train_prefixes: int = 100_000
    max_evaluation_prefixes: int = 1_000_000
    max_checkpoints: int = 16
    selection_policy: BehaviorSelectionPolicy = "strict-nonregression"

    def __post_init__(self) -> None:
        policy = str(self.selection_policy).strip().lower()
        if policy not in {"strict-nonregression", "rollout-first-acquisition"}:
            raise TargetNativeBehaviorDistillationError(
                "selection_policy must be strict-nonregression or "
                "rollout-first-acquisition."
            )
        object.__setattr__(self, "selection_policy", policy)
        integer_fields = (
            "steps",
            "fit_horizon",
            "evaluation_horizon",
            "fit_prompts_per_step",
            "top_k",
            "max_prompt_tokens",
            "max_train_prefixes",
            "max_evaluation_prefixes",
            "max_checkpoints",
        )
        for name in integer_fields:
            object.__setattr__(
                self, name, _index(getattr(self, name), name, positive=True)
            )
        if self.steps > 10_000:
            raise TargetNativeBehaviorDistillationError("steps exceeds 10,000.")
        if self.fit_horizon > 128 or self.evaluation_horizon > 128:
            raise TargetNativeBehaviorDistillationError("horizon exceeds 128 tokens.")
        for name in (
            "learning_rate",
            "temperature",
            "action_loss_weight",
            "distribution_loss_weight",
            "gradient_clip_norm",
        ):
            minimum = 0.0 if name.endswith("loss_weight") else 1e-12
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=minimum),
            )
        if self.action_loss_weight + self.distribution_loss_weight <= 0.0:
            raise TargetNativeBehaviorDistillationError(
                "At least one output loss weight must be positive."
            )
        steps = tuple(
            _index(value, "checkpoint_step") for value in self.checkpoint_steps
        )
        if any(value > self.steps for value in steps):
            raise TargetNativeBehaviorDistillationError(
                "checkpoint_steps cannot exceed steps."
            )
        object.__setattr__(self, "checkpoint_steps", tuple(sorted(set(steps))))
        if len(self.normalized_checkpoint_steps) > self.max_checkpoints:
            raise TargetNativeBehaviorDistillationError(
                "Checkpoint count exceeds max_checkpoints."
            )

    @property
    def normalized_checkpoint_steps(self) -> tuple[int, ...]:
        return tuple(sorted({0, self.steps, *self.checkpoint_steps}))

    def as_dict(self) -> dict[str, Any]:
        return {
            field_name: (
                list(self.normalized_checkpoint_steps)
                if field_name == "checkpoint_steps"
                else getattr(self, field_name)
            )
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateBehaviorThresholds:
    """Preregistered calibration gate that alone can unlock sealed evidence."""

    minimum_action_accuracy: float = 0.0
    minimum_exact_rollout_fraction: float = 0.0
    minimum_token_match_fraction: float = 0.0
    minimum_lcp_fraction: float = 0.0
    minimum_action_count: int = 1
    minimum_rollout_count: int = 1

    def __post_init__(self) -> None:
        for name in (
            "minimum_action_accuracy",
            "minimum_exact_rollout_fraction",
            "minimum_token_match_fraction",
            "minimum_lcp_fraction",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=0.0, maximum=1.0),
            )
        for name in ("minimum_action_count", "minimum_rollout_count"):
            object.__setattr__(
                self,
                name,
                _index(getattr(self, name), name, positive=True),
            )

    def passes(self, metrics: CausalBehaviorMetrics) -> bool:
        if not isinstance(metrics, CausalBehaviorMetrics):
            raise TargetNativeBehaviorDistillationError(
                "metrics must be CausalBehaviorMetrics."
            )
        return (
            metrics.action_count >= self.minimum_action_count
            and metrics.rollout_count >= self.minimum_rollout_count
            and metrics.action_accuracy >= self.minimum_action_accuracy
            and metrics.exact_rollout_fraction >= self.minimum_exact_rollout_fraction
            and metrics.token_match_fraction >= self.minimum_token_match_fraction
            and metrics.lcp_fraction >= self.minimum_lcp_fraction
        )

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CausalBehaviorMetrics:
    """Action and free-rollout metrics derived from output traces only."""

    split: BehaviorEvaluationSplit
    action_count: int
    action_agreement_count: int
    action_accuracy: float
    parent_action_mean_logprob: float
    parent_action_topk_fraction: float
    parent_action_mean_rank: float
    parent_action_worst_rank: int
    rollout_count: int
    exact_rollout_count: int
    exact_rollout_fraction: float
    rollout_token_count: int
    token_match_count: int
    token_match_fraction: float
    lcp_token_count: int
    lcp_fraction: float
    repeat_fraction_excess: float
    adjacent_repeat_fraction_excess: float
    longest_run_excess: int
    report_sha256: str

    @classmethod
    def from_report(
        cls, report: ResidualAttentionFunctionalReport
    ) -> CausalBehaviorMetrics:
        forced = report.forced
        greedy = report.greedy
        return cls(
            split=report.split,
            action_count=forced.action_count,
            action_agreement_count=forced.candidate_parent_top1_agreement_count,
            action_accuracy=forced.candidate_parent_top1_agreement_rate,
            parent_action_mean_logprob=forced.candidate_parent_action_mean_logprob,
            parent_action_topk_fraction=forced.candidate_parent_action_topk_rate,
            parent_action_mean_rank=forced.candidate_parent_action_mean_rank,
            parent_action_worst_rank=forced.candidate_parent_action_worst_rank,
            rollout_count=greedy.sequence_count,
            exact_rollout_count=greedy.exact_rollout_match_count,
            exact_rollout_fraction=greedy.exact_rollout_match_rate,
            rollout_token_count=greedy.parent_token_count,
            token_match_count=greedy.positional_match_count,
            token_match_fraction=greedy.exact_token_retention_rate,
            lcp_token_count=greedy.longest_common_prefix_tokens,
            lcp_fraction=greedy.longest_common_prefix_rate,
            repeat_fraction_excess=greedy.candidate_repeat_fraction_excess,
            adjacent_repeat_fraction_excess=(
                greedy.candidate_adjacent_repeat_fraction_excess
            ),
            longest_run_excess=greedy.candidate_longest_run_excess,
            report_sha256=report.report_sha256,
        )

    @property
    def pareto_objectives(self) -> tuple[float, ...]:
        return (
            self.action_accuracy,
            self.parent_action_mean_logprob,
            self.parent_action_topk_fraction,
            -self.parent_action_mean_rank,
            -float(self.parent_action_worst_rank),
            self.token_match_fraction,
            self.lcp_fraction,
            self.exact_rollout_fraction,
            -self.repeat_fraction_excess,
            -self.adjacent_repeat_fraction_excess,
            -float(self.longest_run_excess),
        )

    @property
    def selection_key(self) -> tuple[float, ...]:
        return tuple(-value for value in self.pareto_objectives)

    @property
    def rollout_first_acquisition_key(self) -> tuple[int, ...]:
        """Ascending key for acquiring behavior in an initially blank stage."""

        return (
            -self.exact_rollout_count,
            -self.token_match_count,
            -self.lcp_token_count,
            -self.action_agreement_count,
        )

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class TargetNativeFactorState:
    """A CPU clone of only the temporary target-native factor tensors."""

    names: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    tensor_sha256s: tuple[str, ...]
    factor_state_sha256: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema": BEHAVIOR_FACTOR_PACK_SCHEMA,
            "factor_count": len(self.names),
            "factors": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "tensor_sha256": digest,
                }
                for name, tensor, digest in zip(
                    self.names, self.tensors, self.tensor_sha256s, strict=True
                )
            ],
            "factor_state_sha256": self.factor_state_sha256,
            "parent_state_included": False,
            "native_child_state_included": False,
            "optimizer_state_included": False,
        }

    def as_payload(self) -> dict[str, Any]:
        return {
            **self.as_manifest(),
            "factor_tensors": {
                name: tensor.detach().cpu().clone()
                for name, tensor in zip(self.names, self.tensors, strict=True)
            },
        }


def _factor_parameters(
    transaction: ExactTargetCompilerTransaction,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise TargetNativeBehaviorDistillationError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    entries = tuple(
        (f"qkvo.{name}", value)
        for name, value in transaction.qkvo_handle.adapter_registry.named_parameters(
            recurse=False
        )
    ) + tuple(
        (f"gud.{name}", value)
        for name, value in transaction.swiglu_handle.registry.named_parameters(
            recurse=False
        )
    )
    expected_names = {
        *(
            f"qkvo.{role}_{side}"
            for role in ("q", "k", "v", "o")
            for side in ("left", "right")
        ),
        *(
            f"gud.{role}_{side}"
            for role in ("gate", "up", "down")
            for side in ("left", "right")
        ),
    }
    if {name for name, _value in entries} != expected_names:
        raise TargetNativeBehaviorDistillationError(
            "Compiler transaction does not expose one complete QKVO/GUD factor set."
        )
    trainable = transaction.output_trainable_parameters()
    if {id(value) for _name, value in entries} != {id(value) for value in trainable}:
        raise TargetNativeBehaviorDistillationError(
            "Named factors disagree with the compiler training scope."
        )
    return tuple(sorted(entries, key=lambda item: item[0]))


def capture_target_native_factor_state(
    transaction: ExactTargetCompilerTransaction,
) -> TargetNativeFactorState:
    entries = _factor_parameters(transaction)
    tensors = tuple(
        value.detach().cpu().contiguous().clone() for _name, value in entries
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise TargetNativeBehaviorDistillationError(
            "Cannot capture non-finite target-native factors."
        )
    names = tuple(name for name, _value in entries)
    hashes = tuple(_tensor_sha256(value) for value in tensors)
    payload = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "tensor_sha256": digest,
        }
        for name, value, digest in zip(names, tensors, hashes, strict=True)
    ]
    return TargetNativeFactorState(
        names=names,
        tensors=tensors,
        tensor_sha256s=hashes,
        factor_state_sha256=_canonical_sha256(payload),
    )


def restore_target_native_factor_state(
    transaction: ExactTargetCompilerTransaction,
    state: TargetNativeFactorState,
) -> None:
    if not isinstance(state, TargetNativeFactorState):
        raise TargetNativeBehaviorDistillationError(
            "state must be TargetNativeFactorState."
        )
    entries = _factor_parameters(transaction)
    if tuple(name for name, _value in entries) != state.names:
        raise TargetNativeBehaviorDistillationError(
            "Factor-state names do not match the live transaction."
        )
    for (name, parameter), value, digest in zip(
        entries, state.tensors, state.tensor_sha256s, strict=True
    ):
        if (
            tuple(parameter.shape) != tuple(value.shape)
            or parameter.dtype != value.dtype
            or _tensor_sha256(value) != digest
        ):
            raise TargetNativeBehaviorDistillationError(
                f"Factor state {name!r} has invalid shape, dtype, or digest."
            )
    with torch.no_grad():
        for (_name, parameter), value in zip(entries, state.tensors, strict=True):
            parameter.copy_(value.to(device=parameter.device))
    transaction.assert_live()
    if (
        capture_target_native_factor_state(transaction).factor_state_sha256
        != state.factor_state_sha256
    ):
        raise TargetNativeBehaviorDistillationError(
            "Factor-state restore did not verify."
        )


@dataclass(frozen=True, slots=True)
class GradientActivationAudit:
    factor_names: tuple[str, ...]
    finite_gradient_factor_names: tuple[str, ...]
    nonzero_gradient_factor_names: tuple[str, ...]
    backward_observation_count: int

    @property
    def complete(self) -> bool:
        expected = set(self.factor_names)
        return (
            set(self.finite_gradient_factor_names) == expected
            and set(self.nonzero_gradient_factor_names) == expected
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "factor_names": list(self.factor_names),
            "finite_gradient_factor_names": list(self.finite_gradient_factor_names),
            "nonzero_gradient_factor_names": list(self.nonzero_gradient_factor_names),
            "backward_observation_count": self.backward_observation_count,
            "every_qkvo_gud_factor_gradient_activated": self.complete,
        }


@dataclass(frozen=True, slots=True)
class BehaviorTrainingAudit:
    optimizer_steps: int
    optimizer_class: str
    parent_greedy_prefix_count: int
    child_on_policy_prefix_count: int
    parent_output_forward_count: int
    child_output_forward_count: int
    gradient_activation: GradientActivationAudit

    def as_dict(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "optimizer_class": self.optimizer_class,
            "parent_greedy_prefix_count": self.parent_greedy_prefix_count,
            "child_on_policy_prefix_count": self.child_on_policy_prefix_count,
            "parent_output_forward_count": self.parent_output_forward_count,
            "child_output_forward_count": self.child_output_forward_count,
            "gradient_activation": self.gradient_activation.as_dict(),
            "teacher_signal": "parent-causal-output-logits-and-actions",
            "parent_greedy_prefix_training": True,
            "child_greedy_on_policy_prefix_training": True,
            "weight_targets_used": False,
            "activation_targets_used": False,
            "hidden_state_targets_used": False,
            "attention_map_targets_used": False,
            "cache_value_targets_used": False,
        }


@dataclass(frozen=True, slots=True)
class BehaviorCheckpoint:
    step: int
    metrics: CausalBehaviorMetrics
    factor_state_sha256: str
    pareto_safe_vs_step0: bool
    pareto_frontier: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BEHAVIOR_CHECKPOINT_SCHEMA,
            "step": self.step,
            "metrics": self.metrics.as_dict(),
            "factor_state_sha256": self.factor_state_sha256,
            "pareto_safe_vs_step0": self.pareto_safe_vs_step0,
            "pareto_frontier": self.pareto_frontier,
            "strict_selection_key": list(self.metrics.selection_key),
            "rollout_first_acquisition_key": list(
                self.metrics.rollout_first_acquisition_key
            ),
        }


@dataclass(frozen=True, slots=True)
class BehaviorDistillationResult:
    seed: int
    selection_policy: BehaviorSelectionPolicy
    selected_step: int
    checkpoints: tuple[BehaviorCheckpoint, ...]
    pareto_frontier_steps: tuple[int, ...]
    fit_metrics: CausalBehaviorMetrics
    candidate_metrics: CausalBehaviorMetrics
    sealed_metrics: CausalBehaviorMetrics | None
    candidate_thresholds: CandidateBehaviorThresholds
    candidate_gate_passed: bool
    sealed_evaluation_unlocked: bool
    training_audit: BehaviorTrainingAudit
    factor_pack: TargetNativeFactorState
    domain_binding_sha256: str
    schedule_sha256: str
    evidence_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_sha256",
            _canonical_sha256(self.evidence_payload()),
        )

    @property
    def calibration_metrics(self) -> CausalBehaviorMetrics:
        return self.candidate_metrics

    def factor_pack_payload(self) -> dict[str, Any]:
        return self.factor_pack.as_payload()

    def evidence_payload(self) -> dict[str, Any]:
        """Return the exact canonical JSON preimage of ``evidence_sha256``."""

        return {
            "schema": TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA,
            "seed": self.seed,
            "selection_policy": self.selection_policy,
            "selected_step": self.selected_step,
            "checkpoints": [value.as_dict() for value in self.checkpoints],
            "pareto_frontier_steps": list(self.pareto_frontier_steps),
            "fit_metrics": self.fit_metrics.as_dict(),
            "candidate_metrics": self.candidate_metrics.as_dict(),
            "sealed_metrics": (
                None if self.sealed_metrics is None else self.sealed_metrics.as_dict()
            ),
            "candidate_thresholds": self.candidate_thresholds.as_dict(),
            "candidate_gate_passed": self.candidate_gate_passed,
            "sealed_evaluation_unlocked": self.sealed_evaluation_unlocked,
            "training_audit": self.training_audit.as_dict(),
            "factor_pack_manifest": self.factor_pack.as_manifest(),
            "domain_binding_sha256": self.domain_binding_sha256,
            "schedule_sha256": self.schedule_sha256,
            "selection_uses_internal_numeric_targets": False,
            "internal_numeric_matching_used": False,
        }

    def as_dict(self) -> dict[str, Any]:
        evidence = self.evidence_payload()
        return {
            **evidence,
            "step0_retained": bool(self.checkpoints and self.checkpoints[0].step == 0),
            "evidence_sha256": self.evidence_sha256,
            "evidence_payload": evidence,
            "selection_inputs": "calibration-output-actions-and-free-rollouts-only",
            "selection_policy_semantics": (
                "every causal-output diagnostic must weakly preserve step0 and at "
                "least one must improve"
                if self.selection_policy == "strict-nonregression"
                else "lexicographically maximize exact rollouts, positional token "
                "matches, longest-common-prefix tokens, then action agreements"
            ),
            "sealed_used_for_selection": False,
            "internal_numeric_matching_used": False,
            "optimizer_state_retained": False,
            "parent_runtime_reference_retained": False,
        }


def _model_device(model: nn.Module) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        module = embeddings()
        weight = getattr(module, "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    for value in (*model.parameters(), *model.buffers()):
        return value.device
    raise TargetNativeBehaviorDistillationError("Model has no parameter device.")


def _forward_kwargs(model: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return values
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return values
    return {
        name: value for name, value in values.items() if name in signature.parameters
    }


def _last_causal_logits(model: nn.Module, prefix: Sequence[int]) -> Tensor:
    device = _model_device(model)
    ids = torch.tensor([list(prefix)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    values = {
        "input_ids": ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "return_dict": True,
        "output_attentions": False,
        "output_hidden_states": False,
        "logits_to_keep": 1,
    }
    kwargs = _forward_kwargs(model, values)
    output = model(**kwargs) if "input_ids" in kwargs else model(ids)
    logits: Any
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or int(logits.shape[0]) != 1
        or int(logits.shape[1]) <= 0
        or int(logits.shape[2]) <= 1
        or not logits.dtype.is_floating_point
    ):
        raise TargetNativeBehaviorDistillationError(
            "Causal model output must contain [1,sequence,vocab] floating logits."
        )
    action = logits[:, -1, :]
    if not bool(torch.isfinite(action.detach()).all()):
        raise TargetNativeBehaviorDistillationError(
            "Causal output logits are non-finite."
        )
    return action


def _greedy_path(
    model: nn.Module,
    prompt: Sequence[int],
    horizon: int,
) -> tuple[tuple[int, ...], tuple[Tensor, ...]]:
    prefix = tuple(prompt)
    selected: list[int] = []
    rows: list[Tensor] = []
    with torch.no_grad():
        for _step in range(horizon):
            logits = (
                _last_causal_logits(model, prefix).detach().cpu().contiguous().clone()
            )
            token = int(torch.argmax(logits, dim=-1).item())
            rows.append(logits)
            selected.append(token)
            prefix = (*prefix, token)
    return tuple(selected), tuple(rows)


def _forced_path(
    model: nn.Module,
    prompt: Sequence[int],
    selected: Sequence[int],
) -> tuple[Tensor, ...]:
    prefix = tuple(prompt)
    rows: list[Tensor] = []
    with torch.no_grad():
        for token in selected:
            rows.append(
                _last_causal_logits(model, prefix).detach().cpu().contiguous().clone()
            )
            prefix = (*prefix, int(token))
    return tuple(rows)


def _functional_trace(
    *,
    prompt: BehaviorPrompt,
    lane: Literal["forced", "greedy"],
    selected: Sequence[int],
    rows: Sequence[Tensor],
) -> FunctionalDecodeTrace:
    if len(selected) != len(rows) or not rows:
        raise TargetNativeBehaviorDistillationError(
            "A functional trace needs one output row per selected action."
        )
    steps: list[FunctionalDecodeStep] = []
    for index, (token, row) in enumerate(zip(selected, rows, strict=True)):
        greedy = int(torch.argmax(row, dim=-1).item())
        steps.append(
            FunctionalDecodeStep(
                step_index=index,
                prefix_length=len(prompt.token_ids) + index,
                selected_token_id=int(token),
                greedy_token_id=greedy,
                cache_length=len(prompt.token_ids) + index,
                action_logits=row.detach().cpu().contiguous().clone(),
                action_logits_sha256=_tensor_sha256(row),
            )
        )
    selected_tuple = tuple(int(value) for value in selected)
    payload = {
        "sequence_id": prompt.prompt_id,
        "lane": lane,
        "prompt_token_ids_sha256": prompt.token_ids_sha256,
        "selected_token_ids": list(selected_tuple),
        "action_logits_sha256s": [value.action_logits_sha256 for value in steps],
    }
    return FunctionalDecodeTrace(
        sequence_id=prompt.prompt_id,
        lane=lane,
        prompt_token_ids=prompt.token_ids,
        consumed_token_ids=selected_tuple[:-1],
        selected_token_ids=selected_tuple,
        steps=tuple(steps),
        vocabulary_size=int(rows[0].shape[-1]),
        logit_bytes=sum(int(value.numel() * value.element_size()) for value in rows),
        model_class="output-only-causal-model",
        cache_class="full-prefix-functional-decode",
        trace_sha256=_canonical_sha256(payload),
    )


def evaluate_behavior_domain(
    parent_model: nn.Module,
    child_model: nn.Module,
    domain: BehaviorDomainSuite,
    *,
    horizon: int,
    top_k: int,
    split: BehaviorEvaluationSplit,
) -> CausalBehaviorMetrics:
    """Evaluate fixed-parent actions plus independent free-running rollouts."""

    if not isinstance(parent_model, nn.Module) or not isinstance(
        child_model, nn.Module
    ):
        raise TargetNativeBehaviorDistillationError(
            "parent_model and child_model must be torch modules."
        )
    if parent_model is child_model:
        raise TargetNativeBehaviorDistillationError(
            "Parent and child must be distinct model objects."
        )
    if not isinstance(domain, BehaviorDomainSuite):
        raise TargetNativeBehaviorDistillationError(
            "domain must be BehaviorDomainSuite."
        )
    expected_splits: dict[str, BehaviorEvaluationSplit] = {
        "fit": "calibration-exploratory",
        "calibration": "calibration-selection",
        "candidate": "calibration-selection",
        "sealed": "heldout-evaluation",
    }
    expected_split = expected_splits.get(domain.split_name)
    if expected_split is not None and split != expected_split:
        raise TargetNativeBehaviorDistillationError(
            f"The {domain.split_name} domain must use split={expected_split!r}."
        )
    horizon_value = _index(horizon, "horizon", positive=True)
    top_k_value = _index(top_k, "top_k", positive=True)
    parent_traces: list[FunctionalDecodeTrace] = []
    child_traces: list[FunctionalDecodeTrace] = []
    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    child_modes = tuple(
        (module, bool(module.training)) for module in child_model.modules()
    )
    try:
        parent_model.eval()
        child_model.eval()
        for prompt in domain.prompts:
            parent_tokens, parent_rows = _greedy_path(
                parent_model, prompt.token_ids, horizon_value
            )
            child_forced_rows = _forced_path(
                child_model, prompt.token_ids, parent_tokens
            )
            child_tokens, child_rows = _greedy_path(
                child_model, prompt.token_ids, horizon_value
            )
            parent_traces.extend(
                (
                    _functional_trace(
                        prompt=prompt,
                        lane="forced",
                        selected=parent_tokens,
                        rows=parent_rows,
                    ),
                    _functional_trace(
                        prompt=prompt,
                        lane="greedy",
                        selected=parent_tokens,
                        rows=parent_rows,
                    ),
                )
            )
            child_traces.extend(
                (
                    _functional_trace(
                        prompt=prompt,
                        lane="forced",
                        selected=parent_tokens,
                        rows=child_forced_rows,
                    ),
                    _functional_trace(
                        prompt=prompt,
                        lane="greedy",
                        selected=child_tokens,
                        rows=child_rows,
                    ),
                )
            )
    finally:
        for module, mode in parent_modes:
            module.training = mode
        for module, mode in child_modes:
            module.training = mode
    report = score_residual_attention_function(
        trace_mapping(parent_traces),
        trace_mapping(child_traces),
        candidate_id=f"target-native-{domain.domain_sha256}",
        split=split,
        top_k=top_k_value,
    )
    return CausalBehaviorMetrics.from_report(report)


def _normalize_domains(
    domains: Mapping[str, BehaviorDomainSuite] | Sequence[BehaviorDomainSuite],
) -> Mapping[str, BehaviorDomainSuite]:
    if isinstance(domains, Mapping):
        values = dict(domains)
    elif not isinstance(domains, (str, bytes)) and isinstance(domains, Sequence):
        if len(domains) != 3:
            raise TargetNativeBehaviorDistillationError(
                "Domain sequence must contain fit, calibration, and sealed suites."
            )
        values = dict(zip(("fit", "calibration", "sealed"), domains, strict=True))
    else:
        raise TargetNativeBehaviorDistillationError(
            "domains must be a mapping or ordered three-suite sequence."
        )
    if "candidate" in values and "calibration" not in values:
        values["calibration"] = values.pop("candidate")
    if set(values) != {"fit", "calibration", "sealed"}:
        raise TargetNativeBehaviorDistillationError(
            "domains must contain exactly fit, calibration, and sealed."
        )
    if any(not isinstance(value, BehaviorDomainSuite) for value in values.values()):
        raise TargetNativeBehaviorDistillationError(
            "Every domain value must be BehaviorDomainSuite."
        )
    for role, suite in values.items():
        allowed_names = {role, "candidate"} if role == "calibration" else {role}
        if suite.split_name not in allowed_names:
            raise TargetNativeBehaviorDistillationError(
                f"The {role} suite has split_name={suite.split_name!r}."
            )
    seen_ids: dict[str, str] = {}
    meaningful_prefixes: dict[str, str] = {}
    for role in ("fit", "calibration", "sealed"):
        for prompt in values[role].prompts:
            previous = seen_ids.get(prompt.prompt_id)
            if previous is not None:
                raise TargetNativeBehaviorDistillationError(
                    f"Prompt content overlaps {previous} and {role}."
                )
            seen_ids[prompt.prompt_id] = role
            for length in range(8, len(prompt.token_ids) + 1):
                prefix = token_sequence_sha256(prompt.token_ids[:length])
                previous = meaningful_prefixes.get(prefix)
                if previous is not None and previous != role:
                    raise TargetNativeBehaviorDistillationError(
                        f"A meaningful token prefix overlaps {previous} and {role}."
                    )
                meaningful_prefixes[prefix] = role
    return MappingProxyType(values)


def _dominates(left: CausalBehaviorMetrics, right: CausalBehaviorMetrics) -> bool:
    pairs = tuple(zip(left.pareto_objectives, right.pareto_objectives, strict=True))
    return all(a >= b for a, b in pairs) and any(a > b for a, b in pairs)


def _pareto_frontier(checkpoints: Sequence[BehaviorCheckpoint]) -> tuple[int, ...]:
    eligible = tuple(value for value in checkpoints if value.pareto_safe_vs_step0)
    return tuple(
        value.step
        for index, value in enumerate(eligible)
        if not any(
            other_index != index and _dominates(other.metrics, value.metrics)
            for other_index, other in enumerate(eligible)
        )
    )


def select_behavior_checkpoint(
    checkpoints: Sequence[BehaviorCheckpoint],
    *,
    selection_policy: BehaviorSelectionPolicy,
) -> BehaviorCheckpoint:
    """Select one calibration checkpoint under a preregistered output policy.

    ``strict-nonregression`` is appropriate when step zero already has useful
    behavior: a candidate must dominate it across every causal diagnostic.
    ``rollout-first-acquisition`` is appropriate for a blank stage: it ignores
    non-promoting diagnostics and lexicographically maximizes exact rollouts,
    positional token matches, LCP tokens, and then parent-action agreements.  In
    either policy, a metric tie retains the earlier checkpoint, including
    step zero.
    """

    values = tuple(checkpoints)
    if (
        not values
        or any(not isinstance(value, BehaviorCheckpoint) for value in values)
        or values[0].step != 0
        or tuple(value.step for value in values)
        != tuple(sorted({value.step for value in values}))
    ):
        raise TargetNativeBehaviorDistillationError(
            "checkpoints must be unique, ordered, and retain step zero first."
        )
    policy = str(selection_policy).strip().lower()
    if policy not in {"strict-nonregression", "rollout-first-acquisition"}:
        raise TargetNativeBehaviorDistillationError(
            "selection_policy must be strict-nonregression or "
            "rollout-first-acquisition."
        )
    baseline = values[0]
    selected = baseline
    for candidate in values[1:]:
        if policy == "strict-nonregression":
            if not _dominates(candidate.metrics, baseline.metrics):
                continue
            key = (candidate.metrics.selection_key, candidate.step)
            selected_key = (selected.metrics.selection_key, selected.step)
        else:
            key = (candidate.metrics.rollout_first_acquisition_key, candidate.step)
            selected_key = (
                selected.metrics.rollout_first_acquisition_key,
                selected.step,
            )
        if key < selected_key:
            selected = candidate
    return selected


def _training_loss(
    parent_logits: Tensor,
    child_logits: Tensor,
    schedule: BehaviorDistillationSchedule,
) -> Tensor:
    if tuple(parent_logits.shape) != tuple(child_logits.shape):
        raise TargetNativeBehaviorDistillationError(
            "Parent and child output vocabularies must match exactly."
        )
    teacher = parent_logits.detach().to(device=child_logits.device, dtype=torch.float32)
    student = child_logits.to(dtype=torch.float32)
    temperature = schedule.temperature
    distribution = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)
    action = F.cross_entropy(student, torch.argmax(teacher, dim=-1))
    return (
        schedule.distribution_loss_weight * distribution
        + schedule.action_loss_weight * action
    ) / (schedule.distribution_loss_weight + schedule.action_loss_weight)


def train_target_native_behavior_distillation(
    transaction: ExactTargetCompilerTransaction,
    parent_model: nn.Module,
    domains: Mapping[str, BehaviorDomainSuite] | Sequence[BehaviorDomainSuite],
    schedule: BehaviorDistillationSchedule,
    seed: int = 0,
    *,
    candidate_thresholds: CandidateBehaviorThresholds | None = None,
    optimizer_factory: Callable[[Iterable[nn.Parameter]], Optimizer] | None = None,
) -> BehaviorDistillationResult:
    """Train, Pareto-select, restore, and optionally unlock sealed evaluation."""

    if not isinstance(transaction, ExactTargetCompilerTransaction):
        raise TargetNativeBehaviorDistillationError(
            "transaction must be ExactTargetCompilerTransaction."
        )
    transaction.assert_live()
    if not isinstance(parent_model, nn.Module) or parent_model is transaction.model:
        raise TargetNativeBehaviorDistillationError(
            "parent_model must be a distinct torch module."
        )
    if not isinstance(schedule, BehaviorDistillationSchedule):
        raise TargetNativeBehaviorDistillationError(
            "schedule must be BehaviorDistillationSchedule."
        )
    threshold = candidate_thresholds or CandidateBehaviorThresholds()
    if not isinstance(threshold, CandidateBehaviorThresholds):
        raise TargetNativeBehaviorDistillationError(
            "candidate_thresholds must be CandidateBehaviorThresholds."
        )
    seed_value = _index(seed, "seed")
    suites = _normalize_domains(domains)
    fit = suites["fit"]
    calibration = suites["calibration"]
    sealed = suites["sealed"]
    if schedule.fit_prompts_per_step > len(fit.prompts):
        raise TargetNativeBehaviorDistillationError(
            "fit_prompts_per_step exceeds the fit domain."
        )
    if any(
        len(prompt.token_ids) > schedule.max_prompt_tokens
        for suite in suites.values()
        for prompt in suite.prompts
    ):
        raise TargetNativeBehaviorDistillationError(
            "A prompt exceeds max_prompt_tokens."
        )
    worst_train = (
        schedule.steps * schedule.fit_prompts_per_step * 2 * schedule.fit_horizon
    )
    if worst_train > schedule.max_train_prefixes:
        raise TargetNativeBehaviorDistillationError(
            "Preregistered training work exceeds max_train_prefixes."
        )
    worst_evaluation = (
        4
        * schedule.evaluation_horizon
        * (
            len(schedule.normalized_checkpoint_steps) * len(calibration.prompts)
            + len(fit.prompts)
            + len(sealed.prompts)
        )
    )
    if worst_evaluation > schedule.max_evaluation_prefixes:
        raise TargetNativeBehaviorDistillationError(
            "Preregistered evaluation work exceeds max_evaluation_prefixes."
        )

    domain_payload = {
        role: suites[role].as_dict() for role in ("fit", "calibration", "sealed")
    }
    domain_binding_sha256 = _canonical_sha256(domain_payload)
    schedule_sha256 = _canonical_sha256(schedule.as_dict())
    factor_entries = _factor_parameters(transaction)
    factor_names = tuple(name for name, _value in factor_entries)
    finite_gradients: set[str] = set()
    nonzero_gradients: set[str] = set()
    gradient_observations = 0
    optimizer_class = ""
    parent_prefixes = 0
    child_prefixes = 0
    parent_forwards = 0
    child_forwards = 0

    step0_state = capture_target_native_factor_state(transaction)
    step0_metrics = evaluate_behavior_domain(
        parent_model,
        transaction.model,
        calibration,
        horizon=schedule.evaluation_horizon,
        top_k=schedule.top_k,
        split="calibration-selection",
    )
    checkpoints: list[BehaviorCheckpoint] = [
        BehaviorCheckpoint(
            step=0,
            metrics=step0_metrics,
            factor_state_sha256=step0_state.factor_state_sha256,
            pareto_safe_vs_step0=True,
        )
    ]
    selected_step = 0
    selected_state = step0_state
    selected_metrics = step0_metrics

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed_value)
    order = torch.randperm(len(fit.prompts), generator=generator).tolist()
    if optimizer_factory is None:

        def default_optimizer_factory(
            parameters: Iterable[nn.Parameter],
        ) -> Optimizer:
            return torch.optim.AdamW(
                parameters, lr=schedule.learning_rate, weight_decay=0.0
            )

        optimizer_factory = default_optimizer_factory

    parent_modes = tuple(
        (module, bool(module.training)) for module in parent_model.modules()
    )
    try:
        parent_model.eval()
        with transaction.output_training_scope() as trainable:
            optimizer = optimizer_factory(trainable)
            if not isinstance(optimizer, Optimizer):
                raise TargetNativeBehaviorDistillationError(
                    "optimizer_factory must return torch.optim.Optimizer."
                )
            optimizer_class = (
                f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            )
            optimizer_parameters = tuple(
                value for group in optimizer.param_groups for value in group["params"]
            )
            if len(optimizer_parameters) != len(trainable) or {
                id(value) for value in optimizer_parameters
            } != {id(value) for value in trainable}:
                raise TargetNativeBehaviorDistillationError(
                    "Optimizer must contain exactly the temporary compiler factors."
                )
            if any(
                _finite_float(group.get("lr"), "optimizer learning rate", minimum=0.0)
                != schedule.learning_rate
                for group in optimizer.param_groups
            ):
                raise TargetNativeBehaviorDistillationError(
                    "Optimizer learning rates must match the preregistered schedule."
                )
            for step in range(1, schedule.steps + 1):
                optimizer.zero_grad(set_to_none=True)
                selected_prompts = tuple(
                    fit.prompts[
                        order[
                            ((step - 1) * schedule.fit_prompts_per_step + offset)
                            % len(order)
                        ]
                    ]
                    for offset in range(schedule.fit_prompts_per_step)
                )
                divisor = float(len(selected_prompts) * 2 * schedule.fit_horizon)
                for prompt in selected_prompts:
                    parent_tokens, _parent_rows = _greedy_path(
                        parent_model, prompt.token_ids, schedule.fit_horizon
                    )
                    child_tokens, _child_rows = _greedy_path(
                        transaction.model, prompt.token_ids, schedule.fit_horizon
                    )
                    parent_forwards += schedule.fit_horizon
                    child_forwards += schedule.fit_horizon
                    for lane_index, lane_tokens in enumerate(
                        (parent_tokens, child_tokens)
                    ):
                        prefix = prompt.token_ids
                        for token in lane_tokens:
                            with torch.no_grad():
                                parent_logits = _last_causal_logits(
                                    parent_model, prefix
                                )
                            child_logits = _last_causal_logits(
                                transaction.model, prefix
                            )
                            parent_forwards += 1
                            child_forwards += 1
                            (
                                _training_loss(parent_logits, child_logits, schedule)
                                / divisor
                            ).backward()
                            if lane_index == 0:
                                parent_prefixes += 1
                            else:
                                child_prefixes += 1
                            prefix = (*prefix, int(token))
                gradient_observations += 1
                for name, parameter in factor_entries:
                    gradient = parameter.grad
                    if gradient is not None and bool(torch.isfinite(gradient).all()):
                        finite_gradients.add(name)
                        if int(torch.count_nonzero(gradient).item()) > 0:
                            nonzero_gradients.add(name)
                torch.nn.utils.clip_grad_norm_(trainable, schedule.gradient_clip_norm)
                optimizer.step()
                if any(
                    not bool(torch.isfinite(value.detach()).all())
                    for value in trainable
                ):
                    raise TargetNativeBehaviorDistillationError(
                        "Optimizer produced non-finite compiler factors."
                    )
                if step in schedule.normalized_checkpoint_steps:
                    metrics = evaluate_behavior_domain(
                        parent_model,
                        transaction.model,
                        calibration,
                        horizon=schedule.evaluation_horizon,
                        top_k=schedule.top_k,
                        split="calibration-selection",
                    )
                    state = capture_target_native_factor_state(transaction)
                    safe = (
                        metrics.pareto_objectives == step0_metrics.pareto_objectives
                        or _dominates(metrics, step0_metrics)
                    )
                    checkpoints.append(
                        BehaviorCheckpoint(
                            step=step,
                            metrics=metrics,
                            factor_state_sha256=state.factor_state_sha256,
                            pareto_safe_vs_step0=safe,
                        )
                    )
                    policy_winner = select_behavior_checkpoint(
                        checkpoints,
                        selection_policy=schedule.selection_policy,
                    )
                    if policy_winner.step == step:
                        selected_step = step
                        selected_state = state
                        selected_metrics = metrics
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)
    except Exception:
        if transaction.is_live:
            restore_target_native_factor_state(transaction, step0_state)
        raise
    finally:
        for module, mode in parent_modes:
            module.training = mode

    gradient_audit = GradientActivationAudit(
        factor_names=factor_names,
        finite_gradient_factor_names=tuple(sorted(finite_gradients)),
        nonzero_gradient_factor_names=tuple(sorted(nonzero_gradients)),
        backward_observation_count=gradient_observations,
    )
    if not gradient_audit.complete:
        missing = sorted(set(factor_names) - nonzero_gradients)
        restore_target_native_factor_state(transaction, step0_state)
        raise TargetNativeBehaviorDistillationError(
            "Not every QKVO/GUD factor received a finite nonzero output-loss "
            f"gradient; inactive={missing}."
        )

    frontier_steps = _pareto_frontier(checkpoints)
    checkpoints = [
        replace(value, pareto_frontier=value.step in frontier_steps)
        for value in checkpoints
    ]
    try:
        restore_target_native_factor_state(transaction, selected_state)
        factor_pack = capture_target_native_factor_state(transaction)
        if factor_pack.factor_state_sha256 != selected_state.factor_state_sha256:
            raise TargetNativeBehaviorDistillationError(
                "Selected factor pack does not match the restored checkpoint."
            )
        fit_metrics = evaluate_behavior_domain(
            parent_model,
            transaction.model,
            fit,
            horizon=schedule.evaluation_horizon,
            top_k=schedule.top_k,
            split="calibration-exploratory",
        )
        candidate_passed = threshold.passes(selected_metrics)
        sealed_metrics = None
        if candidate_passed:
            sealed_metrics = evaluate_behavior_domain(
                parent_model,
                transaction.model,
                sealed,
                horizon=schedule.evaluation_horizon,
                top_k=schedule.top_k,
                split="heldout-evaluation",
            )
    except Exception:
        if transaction.is_live:
            restore_target_native_factor_state(transaction, step0_state)
        raise

    training_audit = BehaviorTrainingAudit(
        optimizer_steps=schedule.steps,
        optimizer_class=optimizer_class,
        parent_greedy_prefix_count=parent_prefixes,
        child_on_policy_prefix_count=child_prefixes,
        parent_output_forward_count=parent_forwards,
        child_output_forward_count=child_forwards,
        gradient_activation=gradient_audit,
    )
    return BehaviorDistillationResult(
        seed=seed_value,
        selection_policy=schedule.selection_policy,
        selected_step=selected_step,
        checkpoints=tuple(checkpoints),
        pareto_frontier_steps=frontier_steps,
        fit_metrics=fit_metrics,
        candidate_metrics=selected_metrics,
        sealed_metrics=sealed_metrics,
        candidate_thresholds=threshold,
        candidate_gate_passed=candidate_passed,
        sealed_evaluation_unlocked=candidate_passed,
        training_audit=training_audit,
        factor_pack=factor_pack,
        domain_binding_sha256=domain_binding_sha256,
        schedule_sha256=schedule_sha256,
    )


__all__ = [
    "BEHAVIOR_CHECKPOINT_SCHEMA",
    "BEHAVIOR_DOMAIN_SCHEMA",
    "BEHAVIOR_FACTOR_PACK_SCHEMA",
    "TARGET_NATIVE_BEHAVIOR_DISTILLATION_SCHEMA",
    "BehaviorCheckpoint",
    "BehaviorDistillationResult",
    "BehaviorDistillationSchedule",
    "BehaviorDomainSuite",
    "BehaviorPrompt",
    "BehaviorSelectionPolicy",
    "BehaviorTrainingAudit",
    "CandidateBehaviorThresholds",
    "CausalBehaviorMetrics",
    "GradientActivationAudit",
    "TargetNativeBehaviorDistillationError",
    "TargetNativeFactorState",
    "capture_target_native_factor_state",
    "evaluate_behavior_domain",
    "load_behavior_domain_suite",
    "restore_target_native_factor_state",
    "select_behavior_checkpoint",
    "train_target_native_behavior_distillation",
]
