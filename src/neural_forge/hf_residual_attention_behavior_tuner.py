"""Behavior-space tuning for the exact p35 residual-attention redirect.

The tentative compact registry is already the structural ``280 -> 279`` edit
``L19/G4 -> L21/G6``.  This module tunes only the two *existing* V/O
assignment scales at the removed executable site.  Its differentiable loss is
defined entirely on causal-model outputs: the fixed parent action, its child
margin, and (optionally) the parent's top-k output distribution.  Parent and
child weights, Q/K/V tensors, attention matrices, hidden activations,
residuals, and cache values are never loss or selection inputs.

Training is a transaction.  A two-scalar manual Adam update avoids optimizer
parameters or checkpoint state, every other model parameter is frozen, and
the full role-scale tensor is restored exactly.  Fit prompts are used only for
gradients.  The immutable role-scale installer validates physical-to-canonical
addressing and evaluates the incumbent and fitted candidates on a disjoint
calibration split.  The audited functional scorer selects on calibration
outputs only; heldout output is read only after that selection is locked and
is reported without a winner.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from neural_forge.hf_attention_codebook import AttentionCodebookRegistry
from neural_forge.hf_functional_decode_trace import (
    FunctionalDecodeLimits,
    FunctionalDecodeTrace,
    collect_functional_decode_trace,
    trace_mapping,
)
from neural_forge.hf_residual_attention_functional import (
    build_residual_attention_functional_comparison,
    rank_functional_reports,
    score_residual_attention_function,
)
from neural_forge.hf_residual_attention_role_scale import (
    AttentionRoleScales,
    install_attention_role_scales,
)
from neural_forge.hf_rollout_gate import hash_rollout_records


BEHAVIOR_VO_TUNER_SCHEMA = "neural-forge-p35-behavior-vo-tuner-v2"
EXACT_PARENT_PROTOTYPE_COUNT = 280
EXACT_CHILD_PROTOTYPE_COUNT = 279
EXACT_LOGICAL_LAYERS = 36
EXACT_KEY_VALUE_HEADS = 8
EXACT_SCALE_LAYER = 19
EXACT_SCALE_PHYSICAL_GROUP = 4
EXACT_REMOVED_SOURCE = 19 * 8 + 4
EXACT_SURVIVOR_SOURCE = 21 * 8 + 6
EXACT_AFFECTED_SITES = ((2, 1), (19, 4), (21, 6))
FULL_P35_TO_P34_REDIRECT_COUNT = 8
_V_INDEX = 2
_O_INDEX = 3


class ResidualAttentionBehaviorTuningError(RuntimeError):
    """The behavior-only tuning request or transaction is invalid."""


def _index(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ResidualAttentionBehaviorTuningError(f"{name} must be an integer.")
    try:
        result = operator.index(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionBehaviorTuningError(
            f"{name} must be an integer."
        ) from error
    if result < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ResidualAttentionBehaviorTuningError(
            f"{name} must be a {qualifier} integer."
        )
    return result


def _finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ResidualAttentionBehaviorTuningError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ResidualAttentionBehaviorTuningError(
            f"{name} must be finite."
        ) from error
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise ResidualAttentionBehaviorTuningError(
            f"{name} is outside its finite bound."
        )
    return result


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
        raise ResidualAttentionBehaviorTuningError(
            "Behavior-tuning provenance is not canonical JSON."
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: Tensor) -> str:
    detached = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(detached.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _one_prototype_parameter_accounting(
    registry: AttentionCodebookRegistry,
) -> dict[str, int]:
    prototype_count = int(registry.prototype_count)
    parameter_count = 0
    byte_count = 0
    for name in (
        "prototype_q_weight",
        "prototype_k_weight",
        "prototype_v_weight",
        "prototype_o_weight",
        "prototype_q_bias",
        "prototype_k_bias",
        "prototype_v_bias",
    ):
        value = getattr(registry, name, None)
        if value is None:
            continue
        if int(value.shape[0]) != prototype_count:
            raise ResidualAttentionBehaviorTuningError(
                "Prototype tensor does not share the global prototype axis."
            )
        per_prototype = int(value.numel()) // prototype_count
        parameter_count += per_prototype
        byte_count += per_prototype * int(value.element_size())
    return {
        "one_redirect_prototype_parameters_removed": parameter_count,
        "one_redirect_prototype_bytes_removed": byte_count,
    }


@dataclass(frozen=True, slots=True)
class BehaviorVOTuningControls:
    """Hard resource and objective bounds for one two-scalar search."""

    steps: int = 16
    learning_rate: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_abs_scale: float = 16.0
    gradient_clip: float = 10.0
    action_ce_weight: float = 1.0
    action_margin_weight: float = 0.25
    action_margin: float = 0.5
    topk_distill_weight: float = 0.1
    topk_distill_tokens: int = 8
    distill_temperature: float = 1.0
    functional_top_k: int = 5
    max_prompts_per_split: int = 16
    max_horizon: int = 32
    max_total_fit_actions: int = 512

    def __post_init__(self) -> None:
        integer_bounds = {
            "steps": (0, 256),
            "topk_distill_tokens": (1, 256),
            "functional_top_k": (1, 256),
            "max_prompts_per_split": (1, 64),
            "max_horizon": (1, 128),
            "max_total_fit_actions": (1, 4096),
        }
        for name, (minimum, maximum) in integer_bounds.items():
            normalized = _index(getattr(self, name), name, minimum=minimum)
            if normalized > maximum:
                raise ResidualAttentionBehaviorTuningError(
                    f"{name} exceeds its hard bound {maximum}."
                )
            object.__setattr__(self, name, normalized)
        finite_bounds = {
            "learning_rate": (0.0, 1.0),
            "beta1": (0.0, 1.0),
            "beta2": (0.0, 1.0),
            "adam_epsilon": (0.0, 1.0),
            "max_abs_scale": (0.0, 64.0),
            "gradient_clip": (0.0, 1_000.0),
            "action_ce_weight": (0.0, 100.0),
            "action_margin_weight": (0.0, 100.0),
            "action_margin": (0.0, 100.0),
            "topk_distill_weight": (0.0, 100.0),
            "distill_temperature": (0.01, 100.0),
        }
        for name, (minimum, maximum) in finite_bounds.items():
            normalized = _finite(
                getattr(self, name),
                name,
                minimum=minimum,
                maximum=maximum,
            )
            object.__setattr__(self, name, normalized)
        if self.learning_rate <= 0.0:
            raise ResidualAttentionBehaviorTuningError(
                "learning_rate must be positive."
            )
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ResidualAttentionBehaviorTuningError(
                "Adam beta values must be in [0, 1)."
            )
        if self.adam_epsilon <= 0.0 or self.max_abs_scale <= 0.0:
            raise ResidualAttentionBehaviorTuningError(
                "adam_epsilon and max_abs_scale must be positive."
            )
        if self.gradient_clip <= 0.0 or self.action_ce_weight <= 0.0:
            raise ResidualAttentionBehaviorTuningError(
                "gradient_clip and action_ce_weight must be positive."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class BehaviorVOTuningResult:
    """A selected-but-not-installed candidate and its complete audit report."""

    selected_scales: AttentionRoleScales
    selected_candidate_id: str
    report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


@dataclass(frozen=True, slots=True)
class _TrainingTarget:
    sequence_id: str
    prompt_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    parent_topk_token_ids: Tensor
    parent_topk_logprob: Tensor
    parent_action_logits_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeParameterState:
    name: str
    parameter: nn.Parameter
    requires_grad: bool
    gradient: Tensor | None
    gradient_version: int | None
    data_ptr: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    parameter_version: int


@dataclass(frozen=True, slots=True)
class _RuntimeBufferState:
    name: str
    buffer: Tensor
    data_ptr: int
    shape: tuple[int, ...]
    dtype: str
    device: str
    version: int


def _normalized_traces(
    values: Mapping[Any, FunctionalDecodeTrace],
    name: str,
) -> Mapping[tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace]:
    if not isinstance(values, Mapping) or not values:
        raise ResidualAttentionBehaviorTuningError(f"{name} cannot be empty.")
    traces = tuple(values.values())
    if any(not isinstance(value, FunctionalDecodeTrace) for value in traces):
        raise ResidualAttentionBehaviorTuningError(
            f"{name} must contain FunctionalDecodeTrace values."
        )
    try:
        return trace_mapping(traces)
    except Exception as error:
        raise ResidualAttentionBehaviorTuningError(
            f"{name} is not a collision-free functional trace mapping."
        ) from error


def _prompt_set(
    traces: Mapping[Any, FunctionalDecodeTrace],
    *,
    name: str,
    maximum: int,
) -> set[tuple[int, ...]]:
    prompts = {tuple(value.prompt_token_ids) for value in traces.values()}
    if not prompts or len(prompts) > maximum:
        raise ResidualAttentionBehaviorTuningError(
            f"{name} prompt count is empty or exceeds its bound."
        )
    return prompts


def _validate_split_structure(
    fit: Mapping[Any, FunctionalDecodeTrace],
    calibration: Mapping[Any, FunctionalDecodeTrace],
    heldout: Mapping[Any, FunctionalDecodeTrace],
    *,
    maximum: int,
) -> dict[str, Any]:
    fit_prompts = _prompt_set(fit, name="fit", maximum=maximum)
    calibration_prompts = _prompt_set(
        calibration, name="calibration", maximum=maximum
    )
    heldout_prompts = _prompt_set(heldout, name="heldout", maximum=maximum)
    fit_calibration_overlap = fit_prompts & calibration_prompts
    fit_heldout_overlap = fit_prompts & heldout_prompts
    calibration_heldout_overlap = calibration_prompts & heldout_prompts
    if fit_calibration_overlap or fit_heldout_overlap or calibration_heldout_overlap:
        raise ResidualAttentionBehaviorTuningError(
            "Fit, calibration, and heldout prompt token rows must be pairwise disjoint."
        )
    return {
        "fit_prompt_count": len(fit_prompts),
        "calibration_prompt_count": len(calibration_prompts),
        "heldout_prompt_count": len(heldout_prompts),
        "fit_calibration_exact_prompt_overlap_count": 0,
        "fit_heldout_exact_prompt_overlap_count": 0,
        "calibration_heldout_exact_prompt_overlap_count": 0,
        "all_three_token_row_sets_pairwise_disjoint": True,
    }


def _current_row(
    registry: AttentionCodebookRegistry,
    *,
    layer: int,
    canonical: int,
) -> AttentionRoleScales:
    scales = registry.role_scales
    if scales is None:
        raise ResidualAttentionBehaviorTuningError(
            "The tentative compact registry has no existing role scales."
        )
    row = scales[layer, canonical].detach().cpu()
    if tuple(row.shape) != (4,) or not bool(torch.isfinite(row).all()):
        raise ResidualAttentionBehaviorTuningError(
            "The target assignment role-scale row is malformed."
        )
    return AttentionRoleScales(*map(float, row.tolist()))


def _exact_lineage_and_probe(
    model: nn.Module,
    registry: AttentionCodebookRegistry,
    *,
    max_abs_scale: float,
) -> tuple[int, AttentionRoleScales, dict[str, Any]]:
    if not isinstance(model, nn.Module) or not isinstance(
        registry, AttentionCodebookRegistry
    ):
        raise ResidualAttentionBehaviorTuningError(
            "model and registry must be torch modules with an attention registry."
        )
    if not any(value is registry.role_scales for value in model.parameters()):
        raise ResidualAttentionBehaviorTuningError(
            "registry.role_scales must be an existing parameter of model."
        )
    if (
        registry.logical_layers != EXACT_LOGICAL_LAYERS
        or registry.key_value_heads != EXACT_KEY_VALUE_HEADS
        or registry.prototype_count != EXACT_CHILD_PROTOTYPE_COUNT
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Behavior tuning requires the exact tentative 36x8 p35 280 -> 279 registry."
        )
    assignments = registry.assignments.detach().cpu().to(torch.long)
    sources = registry.prototype_source_indices.detach().cpu().to(torch.long).reshape(-1)
    permutations = registry.group_permutations.detach().cpu().to(torch.long)
    if (
        tuple(assignments.shape) != (EXACT_LOGICAL_LAYERS, EXACT_KEY_VALUE_HEADS)
        or int(sources.numel()) != EXACT_CHILD_PROTOTYPE_COUNT
        or int(torch.unique(sources).numel()) != EXACT_CHILD_PROTOTYPE_COUNT
        or EXACT_REMOVED_SOURCE in set(map(int, sources.tolist()))
        or EXACT_SURVIVOR_SOURCE not in set(map(int, sources.tolist()))
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Tentative source lineage is not the exact one-prototype removal."
        )
    expected = torch.arange(EXACT_KEY_VALUE_HEADS, dtype=torch.long)
    if any(
        not torch.equal(torch.sort(row).values, expected) for row in permutations
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Every group permutation must remain a complete bijection."
        )
    target_matches = torch.nonzero(
        permutations[EXACT_SCALE_LAYER] == EXACT_SCALE_PHYSICAL_GROUP,
        as_tuple=False,
    ).reshape(-1)
    if int(target_matches.numel()) != 1:
        raise ResidualAttentionBehaviorTuningError(
            "Physical L19/G4 does not resolve to one canonical assignment."
        )
    canonical = int(target_matches.item())
    target_slot = int(assignments[EXACT_SCALE_LAYER, canonical].item())
    if not 0 <= target_slot < int(sources.numel()) or int(
        sources[target_slot].item()
    ) != EXACT_SURVIVOR_SOURCE:
        raise ResidualAttentionBehaviorTuningError(
            "Physical L19/G4 is not redirected to survivor L21/G6."
        )
    affected: list[tuple[int, int]] = []
    for layer in range(EXACT_LOGICAL_LAYERS):
        for canonical_group in range(EXACT_KEY_VALUE_HEADS):
            slot = int(assignments[layer, canonical_group].item())
            if not 0 <= slot < int(sources.numel()):
                raise ResidualAttentionBehaviorTuningError(
                    "Tentative assignment contains an invalid prototype slot."
                )
            if int(sources[slot].item()) == EXACT_SURVIVOR_SOURCE:
                affected.append((layer, int(permutations[layer, canonical_group])))
    if tuple(sorted(affected)) != EXACT_AFFECTED_SITES:
        raise ResidualAttentionBehaviorTuningError(
            "Survivor L21/G6 does not serve the exact expected affected sites."
        )
    incumbent = _current_row(
        registry,
        layer=EXACT_SCALE_LAYER,
        canonical=canonical,
    )
    validation_limit = max(max_abs_scale, *(abs(value) for value in incumbent.as_tuple()))
    probe = install_attention_role_scales(
        registry,
        layer_index=EXACT_SCALE_LAYER,
        physical_group=EXACT_SCALE_PHYSICAL_GROUP,
        scales=incumbent,
        max_abs_scale=validation_limit,
    )
    try:
        probe_report = probe.assert_live()
        if probe_report.canonical_group != canonical:
            raise ResidualAttentionBehaviorTuningError(
                "Immutable scale installer disagrees with canonical addressing."
            )
        payload = {
            "parent_prototype_count": EXACT_PARENT_PROTOTYPE_COUNT,
            "child_prototype_count": EXACT_CHILD_PROTOTYPE_COUNT,
            "prototype_count_removed": 1,
            "logical_layers": EXACT_LOGICAL_LAYERS,
            "key_value_heads": EXACT_KEY_VALUE_HEADS,
            "removed_source_flat_index": EXACT_REMOVED_SOURCE,
            "survivor_source_flat_index": EXACT_SURVIVOR_SOURCE,
            "affected_sites": [list(value) for value in affected],
            "exact_p35_l19g4_to_l21g6_verified": True,
            "baseline_registry_sha256": probe_report.registry_sha256_before,
            "baseline_non_scale_state_sha256": probe_report.non_scale_state_sha256,
            "baseline_parameter_identity_sha256": (
                probe_report.parameter_identity_sha256
            ),
        }
    finally:
        probe.rollback()
    return canonical, incumbent, payload


def _extract_training_targets(
    traces: Mapping[Any, FunctionalDecodeTrace],
    controls: BehaviorVOTuningControls,
) -> tuple[tuple[_TrainingTarget, ...], dict[str, Any]]:
    by_lane_and_prompt = {
        (value.lane, value.sequence_id, value.prompt_token_ids): value
        for value in traces.values()
    }
    forced = sorted(
        (value for value in traces.values() if value.lane == "forced"),
        key=lambda value: (value.sequence_id, value.prompt_token_ids),
    )
    greedy = sorted(
        (value for value in traces.values() if value.lane == "greedy"),
        key=lambda value: (value.sequence_id, value.prompt_token_ids),
    )
    forced_keys = {
        (value.sequence_id, value.prompt_token_ids) for value in forced
    }
    greedy_keys = {
        (value.sequence_id, value.prompt_token_ids) for value in greedy
    }
    if not forced or forced_keys != greedy_keys:
        raise ResidualAttentionBehaviorTuningError(
            "Fit traces require paired forced and real greedy trajectories."
        )
    targets: list[_TrainingTarget] = []
    target_payload: list[dict[str, Any]] = []
    action_count = 0
    for trace in forced:
        paired_greedy = by_lane_and_prompt[
            ("greedy", trace.sequence_id, trace.prompt_token_ids)
        ]
        horizon = len(trace.steps)
        if horizon <= 0 or horizon > controls.max_horizon:
            raise ResidualAttentionBehaviorTuningError(
                "A fit trajectory exceeds the tuning horizon bound."
            )
        actions = tuple(trace.selected_token_ids)
        if (
            len(actions) != horizon
            or actions != paired_greedy.selected_token_ids
            or len(paired_greedy.steps) != horizon
            or any(
                step.selected_token_id != step.greedy_token_id
                for step in trace.steps
            )
        ):
            raise ResidualAttentionBehaviorTuningError(
                "Fit forced tokens must all be parent top-1 actions."
            )
        rows = []
        hashes = []
        vocab_size: int | None = None
        for step in trace.steps:
            row = step.action_logits.detach().cpu().to(torch.float64).reshape(-1)
            if row.numel() <= 1 or not bool(torch.isfinite(row).all()):
                raise ResidualAttentionBehaviorTuningError(
                    "Parent action logits must be one finite output row."
                )
            if vocab_size is None:
                vocab_size = int(row.numel())
            elif int(row.numel()) != vocab_size:
                raise ResidualAttentionBehaviorTuningError(
                    "Parent output vocabulary changed within a trajectory."
                )
            if step.selected_token_id >= int(row.numel()):
                raise ResidualAttentionBehaviorTuningError(
                    "Parent action is outside its output vocabulary."
                )
            rows.append(row)
            hashes.append(step.action_logits_sha256)
        assert vocab_size is not None
        if (
            controls.topk_distill_tokens > vocab_size
            or controls.functional_top_k > vocab_size
        ):
            raise ResidualAttentionBehaviorTuningError(
                "A requested top-k exceeds the parent output vocabulary."
            )
        row_tensor = torch.stack(rows)
        top_values, top_indices = torch.topk(
            row_tensor,
            k=controls.topk_distill_tokens,
            dim=-1,
            largest=True,
            sorted=True,
        )
        parent_top_logprob = F.log_softmax(
            top_values / controls.distill_temperature,
            dim=-1,
        ).to(torch.float32)
        targets.append(
            _TrainingTarget(
                sequence_id=trace.sequence_id,
                prompt_token_ids=tuple(trace.prompt_token_ids),
                action_token_ids=actions,
                parent_topk_token_ids=top_indices.to(torch.long),
                parent_topk_logprob=parent_top_logprob,
                parent_action_logits_sha256s=tuple(hashes),
            )
        )
        action_count += horizon
        target_payload.append(
            {
                "sequence_id": trace.sequence_id,
                "prompt_token_ids": list(trace.prompt_token_ids),
                "action_token_ids": list(actions),
                "parent_action_logits_sha256s": hashes,
            }
        )
    if not targets or action_count > controls.max_total_fit_actions:
        raise ResidualAttentionBehaviorTuningError(
            "Fit action count is empty or exceeds its hard bound."
        )
    rollout_records = [
        {
            "token_ids": [*value.prompt_token_ids, *value.action_token_ids],
            "prompt_length": len(value.prompt_token_ids),
        }
        for value in targets
    ]
    return tuple(targets), {
        "prompt_count": len(targets),
        "parent_action_count": action_count,
        "target_sha256": _canonical_sha256(target_payload),
        "parent_action_rollout_sha256": hash_rollout_records(rollout_records),
        "all_fit_forced_drivers_are_parent_top1": True,
        "used_for_gradient_updates_only": True,
    }


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    embeddings = getattr(model, "get_input_embeddings", None)
    if callable(embeddings):
        weight = getattr(embeddings(), "weight", None)
        if isinstance(weight, Tensor):
            return weight.device
    return fallback


def _filtered_forward_kwargs(model: nn.Module, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return values
    if any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    ):
        return values
    return {name: value for name, value in values.items() if name in signature.parameters}


def _teacher_forced_logits(
    model: nn.Module,
    target: _TrainingTarget,
    *,
    device: torch.device,
) -> Tensor:
    consumed = (*target.prompt_token_ids, *target.action_token_ids[:-1])
    input_ids = torch.tensor((consumed,), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model(
        **_filtered_forward_kwargs(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
                "output_attentions": False,
                "output_hidden_states": False,
                "logits_to_keep": 0,
            },
        )
    )
    logits = output.get("logits") if isinstance(output, Mapping) else None
    if logits is None:
        logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, (tuple, list)) and output:
        logits = output[0]
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 3
        or tuple(logits.shape[:2]) != tuple(input_ids.shape)
        or not logits.dtype.is_floating_point
        or not bool(torch.isfinite(logits.detach()).all())
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Child must return finite full-sequence causal output logits."
        )
    positions = torch.arange(
        len(target.action_token_ids), dtype=torch.long, device=logits.device
    ) + (len(target.prompt_token_ids) - 1)
    return logits[0].index_select(0, positions)


def _target_behavior_sums(
    model: nn.Module,
    target: _TrainingTarget,
    controls: BehaviorVOTuningControls,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, int]:
    child_logits = _teacher_forced_logits(model, target, device=device).float()
    actions = torch.tensor(
        target.action_token_ids, dtype=torch.long, device=child_logits.device
    )
    if int(child_logits.shape[-1]) <= int(actions.max().item()):
        raise ResidualAttentionBehaviorTuningError(
            "Child output vocabulary does not contain every parent action."
        )
    parent_topk = target.parent_topk_token_ids.to(child_logits.device)
    if int(parent_topk.max().item()) >= int(child_logits.shape[-1]):
        raise ResidualAttentionBehaviorTuningError(
            "Parent and child output vocabulary widths differ."
        )
    row_count = int(actions.numel())
    ce = F.cross_entropy(child_logits, actions, reduction="sum")
    action_scores = child_logits.gather(1, actions[:, None]).squeeze(1)
    excluded = child_logits.scatter(
        1,
        actions[:, None],
        torch.full(
            (row_count, 1),
            torch.finfo(child_logits.dtype).min,
            dtype=child_logits.dtype,
            device=child_logits.device,
        ),
    )
    competitor = excluded.max(dim=-1).values
    margin = F.relu(
        controls.action_margin - (action_scores - competitor)
    ).sum()
    child_logprob = F.log_softmax(
        child_logits / controls.distill_temperature,
        dim=-1,
    )
    child_top_logprob = child_logprob.gather(1, parent_topk)
    parent_top_logprob = target.parent_topk_logprob.to(child_logits.device)
    parent_top_probability = parent_top_logprob.exp()
    distill = (
        parent_top_probability * (parent_top_logprob - child_top_logprob)
    ).sum() * (controls.distill_temperature**2)
    return ce, margin, distill, row_count


def _behavior_objective(
    model: nn.Module,
    targets: Sequence[_TrainingTarget],
    controls: BehaviorVOTuningControls,
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    ce_sum: Tensor | None = None
    margin_sum: Tensor | None = None
    distill_sum: Tensor | None = None
    action_count = 0
    for target in targets:
        ce, margin, distill, row_count = _target_behavior_sums(
            model, target, controls, device=device
        )
        ce_sum = ce if ce_sum is None else ce_sum + ce
        margin_sum = margin if margin_sum is None else margin_sum + margin
        distill_sum = distill if distill_sum is None else distill_sum + distill
        action_count += row_count
    if ce_sum is None or margin_sum is None or distill_sum is None or action_count <= 0:
        raise ResidualAttentionBehaviorTuningError(
            "Behavior objective received no parent actions."
        )
    ce_mean = ce_sum / action_count
    margin_mean = margin_sum / action_count
    distill_mean = distill_sum / action_count
    total = (
        controls.action_ce_weight * ce_mean
        + controls.action_margin_weight * margin_mean
        + controls.topk_distill_weight * distill_mean
    )
    if not bool(torch.isfinite(total.detach())):
        raise ResidualAttentionBehaviorTuningError(
            "Behavior-space objective became non-finite."
        )
    return total, {
        "total": float(total.detach().cpu()),
        "parent_action_ce": float(ce_mean.detach().cpu()),
        "parent_action_margin_hinge": float(margin_mean.detach().cpu()),
        "parent_topk_output_kl": float(distill_mean.detach().cpu()),
    }


def _behavior_gradient(
    model: nn.Module,
    targets: Sequence[_TrainingTarget],
    controls: BehaviorVOTuningControls,
    role_scales: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Accumulate one role-scale gradient without retaining multiple prompts."""

    action_count = sum(len(value.action_token_ids) for value in targets)
    if action_count <= 0:
        raise ResidualAttentionBehaviorTuningError(
            "Behavior gradient received no parent actions."
        )
    gradient_sum = torch.zeros_like(role_scales)
    ce_value = 0.0
    margin_value = 0.0
    distill_value = 0.0
    for target in targets:
        ce, margin, distill, _row_count = _target_behavior_sums(
            model, target, controls, device=device
        )
        target_loss = (
            controls.action_ce_weight * ce
            + controls.action_margin_weight * margin
            + controls.topk_distill_weight * distill
        ) / action_count
        target_gradient = torch.autograd.grad(
            target_loss,
            role_scales,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        gradient_sum.add_(target_gradient)
        ce_value += float(ce.detach().cpu())
        margin_value += float(margin.detach().cpu())
        distill_value += float(distill.detach().cpu())
    components = {
        "parent_action_ce": ce_value / action_count,
        "parent_action_margin_hinge": margin_value / action_count,
        "parent_topk_output_kl": distill_value / action_count,
    }
    components["total"] = (
        controls.action_ce_weight * components["parent_action_ce"]
        + controls.action_margin_weight
        * components["parent_action_margin_hinge"]
        + controls.topk_distill_weight * components["parent_topk_output_kl"]
    )
    if not bool(torch.isfinite(gradient_sum).all()) or not math.isfinite(
        components["total"]
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Behavior-space gradient became non-finite."
        )
    return gradient_sum, components


def _parameter_states(model: nn.Module) -> tuple[_RuntimeParameterState, ...]:
    states = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        states.append(
            _RuntimeParameterState(
                name=name,
                parameter=parameter,
                requires_grad=bool(parameter.requires_grad),
                gradient=gradient,
                gradient_version=(
                    None if gradient is None else int(getattr(gradient, "_version", -1))
                ),
                data_ptr=int(parameter.untyped_storage().data_ptr()),
                shape=tuple(parameter.shape),
                dtype=str(parameter.dtype),
                device=str(parameter.device),
                parameter_version=int(parameter._version),
            )
        )
    return tuple(states)


def _buffer_states(model: nn.Module) -> tuple[_RuntimeBufferState, ...]:
    return tuple(
        _RuntimeBufferState(
            name=name,
            buffer=value,
            data_ptr=int(value.untyped_storage().data_ptr()),
            shape=tuple(value.shape),
            dtype=str(value.dtype),
            device=str(value.device),
            version=int(value._version),
        )
        for name, value in model.named_buffers()
    )


def _restore_execution_state(
    parameter_states: Sequence[_RuntimeParameterState],
    module_modes: Sequence[tuple[nn.Module, bool]],
) -> None:
    for state in parameter_states:
        if bool(state.parameter.requires_grad) != state.requires_grad:
            state.parameter.requires_grad_(state.requires_grad)
        if state.parameter.grad is not state.gradient:
            state.parameter.grad = state.gradient
    for module, training in module_modes:
        module.training = training


def _assert_only_target_vo_differs(
    baseline: Tensor,
    current: Tensor,
    *,
    canonical: int,
) -> tuple[tuple[int, int, int], ...]:
    if tuple(baseline.shape) != tuple(current.shape):
        raise ResidualAttentionBehaviorTuningError(
            "Role-scale shape changed during behavior tuning."
        )
    changed = tuple(
        tuple(map(int, row))
        for row in torch.nonzero(
            baseline.detach().cpu() != current.detach().cpu(),
            as_tuple=False,
        ).tolist()
    )
    allowed = {
        (EXACT_SCALE_LAYER, canonical, _V_INDEX),
        (EXACT_SCALE_LAYER, canonical, _O_INDEX),
    }
    if not set(changed).issubset(allowed):
        raise ResidualAttentionBehaviorTuningError(
            "A parameter outside the target L19/G4 V/O scalars changed."
        )
    return changed


def _assert_runtime_restored(
    model: nn.Module,
    role_scales: Tensor,
    baseline_scales: Tensor,
    parameter_states: Sequence[_RuntimeParameterState],
    buffer_states: Sequence[_RuntimeBufferState],
) -> None:
    if not torch.equal(role_scales.detach(), baseline_scales):
        raise ResidualAttentionBehaviorTuningError(
            "Exact full role-scale rollback failed."
        )
    current_parameters = tuple(model.named_parameters())
    if len(current_parameters) != len(parameter_states):
        raise ResidualAttentionBehaviorTuningError(
            "Model parameter registration changed during behavior tuning."
        )
    for (name, parameter), state in zip(
        current_parameters, parameter_states, strict=True
    ):
        gradient = parameter.grad
        gradient_version = (
            None if gradient is None else int(getattr(gradient, "_version", -1))
        )
        if (
            name != state.name
            or parameter is not state.parameter
            or bool(parameter.requires_grad) != state.requires_grad
            or gradient is not state.gradient
            or gradient_version != state.gradient_version
            or int(parameter.untyped_storage().data_ptr()) != state.data_ptr
            or tuple(parameter.shape) != state.shape
            or str(parameter.dtype) != state.dtype
            or str(parameter.device) != state.device
            or (
                parameter is not role_scales
                and int(parameter._version) != state.parameter_version
            )
        ):
            raise ResidualAttentionBehaviorTuningError(
                f"Frozen parameter {name!r} did not retain exact runtime identity."
            )
    current_buffers = tuple(model.named_buffers())
    if len(current_buffers) != len(buffer_states):
        raise ResidualAttentionBehaviorTuningError(
            "Model buffer registration changed during behavior tuning."
        )
    for (name, buffer), state in zip(current_buffers, buffer_states, strict=True):
        if (
            name != state.name
            or buffer is not state.buffer
            or int(buffer.untyped_storage().data_ptr()) != state.data_ptr
            or tuple(buffer.shape) != state.shape
            or str(buffer.dtype) != state.dtype
            or str(buffer.device) != state.device
            or int(buffer._version) != state.version
        ):
            raise ResidualAttentionBehaviorTuningError(
                f"Frozen buffer {name!r} changed during behavior tuning."
            )


def _fit_two_scalars(
    model: nn.Module,
    registry: AttentionCodebookRegistry,
    targets: Sequence[_TrainingTarget],
    controls: BehaviorVOTuningControls,
    *,
    canonical: int,
    incumbent: AttentionRoleScales,
) -> tuple[AttentionRoleScales, list[dict[str, Any]], dict[str, Any]]:
    scales = registry.role_scales
    assert scales is not None
    baseline = scales.detach().clone()
    baseline_sha = _tensor_sha256(baseline)
    parameter_states = _parameter_states(model)
    buffer_states = _buffer_states(model)
    module_modes = tuple((module, bool(module.training)) for module in model.modules())
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    device = _model_device(model, scales.device)
    master = torch.tensor(
        (incumbent.v, incumbent.o), dtype=torch.float32, device=scales.device
    )
    first_moment = torch.zeros_like(master)
    second_moment = torch.zeros_like(master)
    history: list[dict[str, Any]] = []
    best_pair = master.detach().clone()
    best_objective = float("inf")
    try:
        model.eval()
        for state in parameter_states:
            state.parameter.requires_grad_(state.parameter is scales)
        with torch.no_grad():
            initial_loss, initial_components = _behavior_objective(
                model, targets, controls, device=device
            )
        del initial_loss
        initial_changed = _assert_only_target_vo_differs(
            baseline, scales, canonical=canonical
        )
        best_objective = initial_components["total"]
        history.append(
            {
                "step": 0,
                "v": float(
                    scales[EXACT_SCALE_LAYER, canonical, _V_INDEX].detach().cpu()
                ),
                "o": float(
                    scales[EXACT_SCALE_LAYER, canonical, _O_INDEX].detach().cpu()
                ),
                **initial_components,
                "changed_coordinates": [list(value) for value in initial_changed],
            }
        )
        for step in range(1, controls.steps + 1):
            gradient, _pre_update = _behavior_gradient(
                model,
                targets,
                controls,
                scales,
                device=device,
            )
            _assert_only_target_vo_differs(baseline, scales, canonical=canonical)
            pair_gradient = torch.stack(
                (
                    gradient[EXACT_SCALE_LAYER, canonical, _V_INDEX],
                    gradient[EXACT_SCALE_LAYER, canonical, _O_INDEX],
                )
            ).float()
            if not bool(torch.isfinite(pair_gradient).all()):
                raise ResidualAttentionBehaviorTuningError(
                    "Target V/O behavior gradient became non-finite."
                )
            gradient_norm = torch.linalg.vector_norm(pair_gradient)
            if float(gradient_norm) > controls.gradient_clip:
                pair_gradient = pair_gradient * (
                    controls.gradient_clip / float(gradient_norm)
                )
            first_moment.mul_(controls.beta1).add_(
                pair_gradient, alpha=1.0 - controls.beta1
            )
            second_moment.mul_(controls.beta2).addcmul_(
                pair_gradient, pair_gradient, value=1.0 - controls.beta2
            )
            corrected_first = first_moment / (1.0 - controls.beta1**step)
            corrected_second = second_moment / (1.0 - controls.beta2**step)
            master.addcdiv_(
                corrected_first,
                corrected_second.sqrt().add(controls.adam_epsilon),
                value=-controls.learning_rate,
            ).clamp_(-controls.max_abs_scale, controls.max_abs_scale)
            with torch.no_grad():
                scales[EXACT_SCALE_LAYER, canonical, _V_INDEX].copy_(
                    master[0].to(scales.dtype)
                )
                scales[EXACT_SCALE_LAYER, canonical, _O_INDEX].copy_(
                    master[1].to(scales.dtype)
                )
                _post_loss, components = _behavior_objective(
                    model, targets, controls, device=device
                )
            changed = _assert_only_target_vo_differs(
                baseline, scales, canonical=canonical
            )
            actual_pair = torch.stack(
                (
                    scales[EXACT_SCALE_LAYER, canonical, _V_INDEX],
                    scales[EXACT_SCALE_LAYER, canonical, _O_INDEX],
                )
            ).detach().float()
            history.append(
                {
                    "step": step,
                    "v": float(actual_pair[0].cpu()),
                    "o": float(actual_pair[1].cpu()),
                    **components,
                    "target_gradient_norm": float(gradient_norm.detach().cpu()),
                    "changed_coordinates": [list(value) for value in changed],
                }
            )
            if components["total"] < best_objective:
                best_objective = components["total"]
                best_pair = actual_pair.to(scales.device)
    finally:
        with torch.no_grad():
            scales.copy_(baseline)
        _restore_execution_state(parameter_states, module_modes)
    _assert_runtime_restored(
        model,
        scales,
        baseline,
        parameter_states,
        buffer_states,
    )
    if _tensor_sha256(scales) != baseline_sha:
        raise ResidualAttentionBehaviorTuningError(
            "Role-scale bytes differ after the differentiable transaction."
        )
    fitted = AttentionRoleScales(
        q=incumbent.q,
        k=incumbent.k,
        v=float(best_pair[0].cpu()),
        o=float(best_pair[1].cpu()),
    )
    return fitted, history, {
        "parameter_count_before": parameter_count,
        "parameter_count_after": sum(
            int(value.numel()) for value in model.parameters()
        ),
        "role_scales_sha256_before": baseline_sha,
        "role_scales_sha256_after": _tensor_sha256(scales),
        "full_role_scale_rollback_exact": True,
        "non_target_parameter_storage_and_version_identity_exact": True,
        "model_buffer_identity_exact": True,
        "requires_grad_and_existing_gradient_objects_restored": True,
    }


def _candidate_id(scales: AttentionRoleScales) -> str:
    return "behavior-vo-" + _canonical_sha256(
        {"v": scales.v, "o": scales.o}
    )[:20]


def _collect_candidate_traces(
    model: nn.Module,
    parent_traces: Mapping[Any, FunctionalDecodeTrace],
    *,
    limits: FunctionalDecodeLimits,
) -> Mapping[tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace]:
    result = []
    for parent in sorted(
        parent_traces.values(),
        key=lambda value: (value.lane, value.sequence_id, value.prompt_token_ids),
    ):
        kwargs: dict[str, Any] = {}
        if parent.lane == "forced":
            kwargs["forced_token_ids"] = parent.selected_token_ids
        result.append(
            collect_functional_decode_trace(
                model,
                parent.prompt_token_ids,
                horizon=len(parent.steps),
                sequence_id=parent.sequence_id,
                lane=parent.lane,
                limits=limits,
                **kwargs,
            )
        )
    return trace_mapping(result)


def _evaluate_candidates(
    model: nn.Module,
    registry: AttentionCodebookRegistry,
    parent_traces: Mapping[Any, FunctionalDecodeTrace],
    candidates: Mapping[str, AttentionRoleScales],
    *,
    limits: FunctionalDecodeLimits,
    max_abs_scale: float,
) -> tuple[
    dict[str, Mapping[tuple[str, str, tuple[int, ...]], FunctionalDecodeTrace]],
    dict[str, Any],
]:
    traces = {}
    installs = {}
    for candidate_id in sorted(candidates):
        scales = candidates[candidate_id]
        validation_limit = max(
            max_abs_scale, *(abs(value) for value in scales.as_tuple())
        )
        installation = install_attention_role_scales(
            registry,
            layer_index=EXACT_SCALE_LAYER,
            physical_group=EXACT_SCALE_PHYSICAL_GROUP,
            scales=scales,
            max_abs_scale=validation_limit,
        )
        try:
            report = installation.assert_live()
            if (
                report.installed_scales.q != scales.q
                or report.installed_scales.k != scales.k
            ):
                raise ResidualAttentionBehaviorTuningError(
                    "Candidate evaluation changed fixed Q/K scales."
                )
            traces[candidate_id] = _collect_candidate_traces(
                model, parent_traces, limits=limits
            )
            installation.assert_live()
            installs[candidate_id] = report.as_dict()
        finally:
            installation.rollback()
    return traces, installs


def tune_exact_p35_residual_attention_vo(
    model: nn.Module,
    registry: AttentionCodebookRegistry,
    parent_fit_traces: Mapping[Any, FunctionalDecodeTrace],
    parent_calibration_traces: Mapping[Any, FunctionalDecodeTrace],
    parent_heldout_traces: Mapping[Any, FunctionalDecodeTrace],
    *,
    controls: BehaviorVOTuningControls | None = None,
    trace_limits: FunctionalDecodeLimits | None = None,
) -> BehaviorVOTuningResult:
    """Tune and select V/O scales, then restore the tentative child exactly.

    The input ``registry`` must already be the exact tentative p35 compact
    registry. ``parent_fit_traces`` alone supply gradients,
    ``parent_calibration_traces`` alone choose between incumbent and fitted
    candidates, and ``parent_heldout_traces`` are evaluation-only after the
    winner lock. The selected scales are returned as data only; this function
    never commits them and never saves a checkpoint.
    """

    settings = controls or BehaviorVOTuningControls()
    if not isinstance(settings, BehaviorVOTuningControls):
        raise ResidualAttentionBehaviorTuningError(
            "controls must be BehaviorVOTuningControls."
        )
    limits = trace_limits or FunctionalDecodeLimits(
        max_horizon=settings.max_horizon
    )
    if not isinstance(limits, FunctionalDecodeLimits):
        raise ResidualAttentionBehaviorTuningError(
            "trace_limits must be FunctionalDecodeLimits."
        )
    fit = _normalized_traces(parent_fit_traces, "parent_fit_traces")
    calibration = _normalized_traces(
        parent_calibration_traces, "parent_calibration_traces"
    )
    heldout = _normalized_traces(parent_heldout_traces, "parent_heldout_traces")
    split_audit = _validate_split_structure(
        fit,
        calibration,
        heldout,
        maximum=settings.max_prompts_per_split,
    )
    canonical, incumbent, lineage = _exact_lineage_and_probe(
        model,
        registry,
        max_abs_scale=settings.max_abs_scale,
    )
    role_scales = registry.role_scales
    assert role_scales is not None
    baseline_scales = role_scales.detach().clone()
    baseline_parameter_count = sum(int(value.numel()) for value in model.parameters())
    targets, target_audit = _extract_training_targets(fit, settings)

    fitted, history, training_transaction = _fit_two_scalars(
        model,
        registry,
        targets,
        settings,
        canonical=canonical,
        incumbent=incumbent,
    )
    candidate_values = {incumbent, fitted}
    candidates = {_candidate_id(value): value for value in candidate_values}
    calibration_candidate_traces, calibration_installs = _evaluate_candidates(
        model,
        registry,
        calibration,
        candidates,
        limits=limits,
        max_abs_scale=settings.max_abs_scale,
    )
    calibration_reports = tuple(
        score_residual_attention_function(
            calibration,
            calibration_candidate_traces[candidate_id],
            candidate_id=candidate_id,
            split="calibration-selection",
            top_k=settings.functional_top_k,
        )
        for candidate_id in sorted(candidates)
    )
    calibration_ranking = rank_functional_reports(
        calibration_reports, purpose="selection"
    )
    selected_id = calibration_ranking.candidates[0].candidate_id
    selected_scales = candidates[selected_id]
    selection_lock = {
        "selected_candidate_id": selected_id,
        "selected_scales": selected_scales.as_dict(),
        "calibration_ranking_sha256": calibration_ranking.ranking_sha256,
        "candidate_scales_sha256": _canonical_sha256(
            {
                candidate_id: value.as_dict()
                for candidate_id, value in sorted(candidates.items())
            }
        ),
        "fit_outputs_read_by_calibration_selector": False,
        "heldout_output_read_before_lock": False,
    }
    selection_lock_sha256 = _canonical_sha256(selection_lock)

    # Only structural prompt disjointness was inspected above.  Heldout output
    # values are first validated and decoded after the calibration lock exists.
    parent_heldout_self = score_residual_attention_function(
        heldout,
        heldout,
        candidate_id="parent-heldout-self-check",
        split="heldout-evaluation",
        top_k=settings.functional_top_k,
    )
    heldout_candidate_traces, heldout_installs = _evaluate_candidates(
        model,
        registry,
        heldout,
        candidates,
        limits=limits,
        max_abs_scale=settings.max_abs_scale,
    )
    functional_comparison = build_residual_attention_functional_comparison(
        calibration,
        heldout,
        calibration_candidate_traces,
        heldout_candidate_traces,
        top_k=settings.functional_top_k,
    )
    if (
        functional_comparison["calibration_selection"]["winner_candidate_id"]
        != selected_id
    ):
        raise ResidualAttentionBehaviorTuningError(
            "Heldout evaluation changed the locked calibration selection."
        )

    if not torch.equal(role_scales.detach(), baseline_scales):
        raise ResidualAttentionBehaviorTuningError(
            "Candidate evaluation did not restore tentative role scales exactly."
        )
    final_probe = install_attention_role_scales(
        registry,
        layer_index=EXACT_SCALE_LAYER,
        physical_group=EXACT_SCALE_PHYSICAL_GROUP,
        scales=incumbent,
        max_abs_scale=max(
            settings.max_abs_scale,
            *(abs(value) for value in incumbent.as_tuple()),
        ),
    )
    try:
        final_report = final_probe.assert_live()
        if (
            final_report.registry_sha256_before
            != lineage["baseline_registry_sha256"]
            or final_report.non_scale_state_sha256
            != lineage["baseline_non_scale_state_sha256"]
            or final_report.parameter_identity_sha256
            != lineage["baseline_parameter_identity_sha256"]
        ):
            raise ResidualAttentionBehaviorTuningError(
                "Tentative registry identity differs after the complete search."
            )
    finally:
        final_probe.rollback()
    if not torch.equal(role_scales.detach(), baseline_scales):
        raise ResidualAttentionBehaviorTuningError(
            "Final immutable-probe rollback was not byte exact."
        )
    final_parameter_count = sum(int(value.numel()) for value in model.parameters())
    if final_parameter_count != baseline_parameter_count:
        raise ResidualAttentionBehaviorTuningError(
            "Behavior tuning changed the model parameter count."
        )
    prototype_accounting = _one_prototype_parameter_accounting(registry)
    role_scale_parameters = int(role_scales.numel())
    role_scale_bytes = role_scale_parameters * int(role_scales.element_size())

    candidate_payload = {
        candidate_id: {
            "scales": value.as_dict(),
            "tentative_incumbent": value == incumbent,
            "differentiable_fit_candidate": value == fitted,
            "calibration_installation": calibration_installs[candidate_id],
            "heldout_installation": heldout_installs[candidate_id],
            "calibration_functional": functional_comparison["variants"][
                candidate_id
            ]["calibration_selection"],
            "heldout_functional": functional_comparison["variants"][candidate_id][
                "heldout_evaluation"
            ],
        }
        for candidate_id, value in sorted(candidates.items())
    }
    report: dict[str, Any] = {
        "schema": BEHAVIOR_VO_TUNER_SCHEMA,
        "exact_fold_lineage": lineage,
        "scale_target": {
            "layer_index": EXACT_SCALE_LAYER,
            "physical_group": EXACT_SCALE_PHYSICAL_GROUP,
            "canonical_group": canonical,
            "roles_trainable": ["v", "o"],
            "q_k_frozen": True,
            "every_other_parameter_frozen": True,
            "uses_existing_role_scale_parameter": True,
        },
        "controls": settings.as_dict(),
        "prompt_split_audit": split_audit,
        "fit_targets": target_audit,
        "objective_contract": {
            "space": "causal-output/action-space-only",
            "formula": (
                "action_ce_weight*CE(parent_top1) + "
                "action_margin_weight*hinge(parent_top1_margin) + "
                "topk_distill_weight*KL(parent_topk_output||child_output)"
            ),
            "numeric_logits_are_functional_outputs_not_weight_targets": True,
            "fixed_parent_actions_follow_rollout_gate_semantics": True,
            "weight_similarity_used": False,
            "activation_similarity_used": False,
            "residual_similarity_used": False,
            "attention_similarity_used": False,
            "cache_value_similarity_used": False,
            "qkv_similarity_used": False,
        },
        "differentiable_training": {
            "history": history,
            "history_sha256": _canonical_sha256(history),
            "fitted_candidate_scales": fitted.as_dict(),
            "manual_two_scalar_adam": True,
            "optimizer_object_constructed": False,
            "optimizer_state_is_ephemeral_nonparameter_data": True,
            "optimizer_state_counted_as_child_parameters": False,
            "optimizer_state_saved": False,
            "training_split": "fit-only",
            "calibration_logits_or_actions_used_for_gradients": False,
            "heldout_logits_or_actions_used": False,
            **training_transaction,
        },
        "candidates": candidate_payload,
        "calibration_selection": {
            "selected_candidate_id": selected_id,
            "selected_scales": selected_scales.as_dict(),
            "ranking": calibration_ranking.as_dict(),
            "selection_lock": selection_lock,
            "selection_lock_sha256": selection_lock_sha256,
            "fit_used_for_selection_only_via_fitted_candidate": True,
            "fit_logits_or_actions_used_directly_by_selector": False,
            "heldout_used_for_fitting": False,
            "heldout_used_for_selection": False,
        },
        "heldout_evaluation": {
            **functional_comparison["heldout_evaluation"],
            "parent_self_report": parent_heldout_self.as_dict(),
            "started_after_selection_lock_sha256": selection_lock_sha256,
            "winner_named": False,
            "used_for_fitting": False,
            "used_for_selection": False,
        },
        "functional_comparison": functional_comparison,
        "accounting": {
            "parent_prototype_count": EXACT_PARENT_PROTOTYPE_COUNT,
            "tentative_child_prototype_count": EXACT_CHILD_PROTOTYPE_COUNT,
            "prototype_count_removed": 1,
            "this_report_covers_one_redirect_only": True,
            "full_p35_to_p34_requires_redirect_count": (
                FULL_P35_TO_P34_REDIRECT_COUNT
            ),
            **prototype_accounting,
            "tentative_registry_role_scale_parameters": role_scale_parameters,
            "tentative_registry_role_scale_bytes": role_scale_bytes,
            "net_attention_parameter_reduction_if_parent_had_no_role_scales": (
                prototype_accounting[
                    "one_redirect_prototype_parameters_removed"
                ]
                - role_scale_parameters
            ),
            "model_parameter_count_before": baseline_parameter_count,
            "model_parameter_count_after": final_parameter_count,
            "runtime_parameters_added": 0,
            "search_changes_deployable_parameter_count": False,
            "manual_adam_state_saved_or_counted": False,
        },
        "transaction": {
            "full_tentative_role_scales_restored_exactly": True,
            "tentative_registry_hash_restored_exactly": True,
            "non_scale_registry_hash_restored_exactly": True,
            "parameter_identity_restored_exactly": True,
            "selected_candidate_installed_after_return": False,
        },
        "checkpoint_saved": False,
        "fold_committed": False,
        "compaction_certified": False,
    }
    report["report_sha256"] = _canonical_sha256(report)
    return BehaviorVOTuningResult(
        selected_scales=selected_scales,
        selected_candidate_id=selected_id,
        report=report,
    )


__all__ = [
    "BEHAVIOR_VO_TUNER_SCHEMA",
    "EXACT_AFFECTED_SITES",
    "EXACT_CHILD_PROTOTYPE_COUNT",
    "EXACT_PARENT_PROTOTYPE_COUNT",
    "EXACT_REMOVED_SOURCE",
    "EXACT_SCALE_LAYER",
    "EXACT_SCALE_PHYSICAL_GROUP",
    "EXACT_SURVIVOR_SOURCE",
    "FULL_P35_TO_P34_REDIRECT_COUNT",
    "BehaviorVOTuningControls",
    "BehaviorVOTuningResult",
    "ResidualAttentionBehaviorTuningError",
    "tune_exact_p35_residual_attention_vo",
]
